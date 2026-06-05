#!/usr/bin/env python3
# coding: utf-8
"""
Generative (text-to-text) evaluation for Han2Han.

Evaluates classification and inference tasks by generating text labels
and comparing to ground truth. Uses exact match and fuzzy matching for metrics.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
from typing import Dict, List, Optional, Literal, Union
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import pearsonr, spearmanr
import argparse
import logging
import re

from transformers import AutoTokenizer
from datasets import Dataset
from modeling_han2han_flax import FlaxHan2Han
from base_callback import GenerationMixin
from task_prompts import sample_task_prompt
from logging_utils import log_from_main_process
from han2han_tools import transcribe

logger = logging.getLogger(__name__)


YNAT_LABELS = ['정치', '경제', '사회', '생활문화', '세계', 'IT과학', '스포츠']
NLI_LABELS = ['수반', '중립', '모순']


def year_to_temporal_label(year: int) -> str:
    """Convert year to temporal label based on granularity rules.

    - 20th century+: decade (e.g., "1920년대")
    - 15th-19th century: decade (e.g., "1750년대")
    - Pre-15th century: century (e.g., "14세기")
    """
    if year >= 1500:
        decade = (year // 10) * 10
        return f"{decade}년대"
    else:
        century = (year // 100) + 1
        return f"{century}세기"


def parse_temporal_prediction(pred: str) -> Optional[int]:
    """Parse temporal prediction to extract year/decade/century.

    Returns the midpoint year for matching purposes.
    """
    pred = pred.strip()

    # try to extract a 4-digit year directly
    year_match = re.search(r'(\d{4})', pred)
    if year_match:
        return int(year_match.group(1))

    # try decade format like "1920년대"
    decade_match = re.search(r'(\d{3,4})년대', pred)
    if decade_match:
        decade = int(decade_match.group(1))
        return decade + 5  # midpoint

    # try century format like "19세기" or "14세기"
    century_match = re.search(r'(\d{1,2})세기', pred)
    if century_match:
        century = int(century_match.group(1))
        return (century - 1) * 100 + 50  # midpoint

    return None


class GenerativeEvaluator(GenerationMixin):
    """Text-to-text evaluator using generation for classification tasks.

    Accepts pre-loaded datasets from dynamic_data_loader.py or any list of dicts
    with the expected fields for each task:
    - ynat: 'input_text', 'target_text' (or 'title', 'label')
    - nli: 'input_text', 'target_text' (or 'premise', 'hypothesis', 'label')
    - sts: 'input_text', 'target_text', 'rounded_score'
    - temporal: 'text', 'year'
    """

    def __init__(
        self,
        model: FlaxHan2Han,
        tokenizer: AutoTokenizer,
        task: Literal['ynat', 'nli', 'sts', 'temporal', 'instruction'],
        dataset: Union[Dataset, List[Dict]],
        batch_size: int = 16,
        max_input_length: int = 256,
        max_output_length: int = 16,
        mesh: Optional[jax.sharding.Mesh] = None,
        use_task_prompts: bool = True,
        temperature: float = 0.0,
        top_k: int = 1,
        collator: Optional[object] = None,
        **kwargs
    ):
        """
        Args:
            model: FlaxHan2Han model for generation
            tokenizer: Tokenizer for encoding/decoding
            task: Task type ('ynat', 'nli', 'sts', 'temporal', 'instruction')
            dataset: Pre-loaded dataset (list of dicts or HF Dataset).
            batch_size: Batch size for generation
            max_input_length: Max encoder input length
            max_output_length: Max decoder output length
            mesh: JAX mesh for SPMD generation (None for single-host)
            use_task_prompts: Whether to prepend task prompts
            temperature: Generation temperature (0.0 for greedy)
            top_k: Top-k sampling parameter
            collator: Optional eval collator. When a ChatSFTCollator is passed,
                encoder inputs are formatted with chat-template boundaries
                (`<|system|>...<|user|>...<|end_of_turn|>`) and the decoder is
                primed with `<|assistant|>` instead of the pretraining-style
                `<s>...</s><hangul>` envelope.
        """
        GenerationMixin.__init__(
            self,
            temperature=temperature,
            top_k=top_k,
            top_p=1.0,
            num_beams=1,
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
        )

        self.model = model
        self.tokenizer = tokenizer
        self.task = task
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_input_length = max_input_length
        self.max_length = max_output_length
        self.mesh = mesh
        self.use_task_prompts = use_task_prompts
        self.collator = collator

        # set up label mappings for classification tasks
        if task == 'ynat':
            self.labels = YNAT_LABELS
            self.idx_to_label = {i: l for i, l in enumerate(YNAT_LABELS)}
            self.label_to_idx = {l: i for i, l in self.idx_to_label.items()}
        elif task == 'nli':
            self.labels = NLI_LABELS
            self.idx_to_label = {i: l for i, l in enumerate(NLI_LABELS)}
            self.label_to_idx = {l: i for i, l in self.idx_to_label.items()}
        elif task in ['sts', 'temporal', 'instruction']:
            self.labels = None
            self.idx_to_label = None
            self.label_to_idx = None
        else:
            raise ValueError(f"Unsupported task: {task}")

        # get script tokens
        self.hangul_token_id = tokenizer.convert_tokens_to_ids('<hangul>')
        self.hanja_token_id = tokenizer.convert_tokens_to_ids('<hanja>')
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id

        # chat-template token ids for CoT parsing. None when tokenizer is the
        # legacy unified pretraining vocab without ChatML additions.
        self.assistant_token_id = self._resolve_optional_token_id('<|assistant|>')
        self.end_of_turn_token_id = self._resolve_optional_token_id('<|end_of_turn|>')

        # per-call CoT overrides. mutated by the callback wrapper before each
        # evaluate() call so the second (CoT) pass parameterizes a single
        # cached evaluator instead of rebuilding it.
        self.decoder_start_token_id_override: Optional[int] = None
        self.parse_cot_answer: bool = False
        self.parse_fallback_count: int = 0

    def _resolve_optional_token_id(self, token_str: str) -> Optional[int]:
        """Look up a token id; return None when missing or maps to unk."""
        tid = self.tokenizer.convert_tokens_to_ids(token_str)
        unk = getattr(self.tokenizer, 'unk_token_id', None)
        if tid is None or tid == unk:
            return None
        return tid

    def _apply_decoder_start_override(
        self, inputs: Dict[str, jnp.ndarray]
    ) -> Dict[str, jnp.ndarray]:
        """Rewrite decoder_input_ids in-batch when a CoT override is active.

        base_callback.generate reads decoder_input_ids[0, 0] as the actual
        decoder_start_token_id, so swapping the array is the cleanest way to
        force `<|think|>` priming without touching the collator.
        """
        if self.decoder_start_token_id_override is None:
            return inputs
        batch_size = inputs['input_ids'].shape[0]
        inputs['decoder_input_ids'] = jnp.full(
            (batch_size, 1),
            self.decoder_start_token_id_override,
            dtype=jnp.int32,
        )
        return inputs

    def _decode_cot_sequence(self, ids) -> Dict[str, object]:
        """Split a generated id sequence into rationale + answer at <|assistant|>.

        Returns a dict with str fields `rationale`, `answer`, `full`, plus a
        `parsed` bool. When `<|assistant|>` is absent the answer falls back to
        the full decoded string and `parsed` is False; the caller is expected
        to bump parse_fallback_count.
        """
        ids = np.array(ids)
        # truncate at first pad
        pad_positions = np.where(ids == self.pad_token_id)[0]
        if len(pad_positions) > 0:
            ids = ids[:pad_positions[0]]

        full_text = self.tokenizer.decode(ids, skip_special_tokens=True).strip()

        if self.assistant_token_id is None:
            return {
                'rationale': '',
                'answer': full_text,
                'full': full_text,
                'parsed': False,
            }

        assistant_positions = np.where(ids == self.assistant_token_id)[0]
        if len(assistant_positions) == 0:
            return {
                'rationale': '',
                'answer': full_text,
                'full': full_text,
                'parsed': False,
            }

        split = int(assistant_positions[0])
        rationale_ids = ids[:split]
        answer_ids = ids[split + 1:]

        if self.end_of_turn_token_id is not None:
            eot_positions = np.where(answer_ids == self.end_of_turn_token_id)[0]
            if len(eot_positions) > 0:
                answer_ids = answer_ids[:eot_positions[0]]

        eos_positions = np.where(answer_ids == self.eos_token_id)[0]
        if len(eos_positions) > 0:
            answer_ids = answer_ids[:eos_positions[0]]

        rationale = self.tokenizer.decode(
            rationale_ids, skip_special_tokens=True
        ).strip()
        answer = self.tokenizer.decode(
            answer_ids, skip_special_tokens=True
        ).strip()

        return {
            'rationale': rationale,
            'answer': answer,
            'full': full_text,
            'parsed': True,
        }

    def decode_predictions_cot(
        self, generated_ids: jnp.ndarray
    ) -> List[Dict[str, object]]:
        """Parse a batch of CoT generations; updates parse_fallback_count."""
        parsed = []
        for ids in np.array(generated_ids):
            entry = self._decode_cot_sequence(ids)
            if not entry['parsed']:
                self.parse_fallback_count += 1
            parsed.append(entry)
        return parsed

    def _decode_if_tokenized(self, example: Dict, field: str) -> Optional[str]:
        """Decode a tokenized field if present, otherwise return None."""
        if field in example:
            val = example[field]
            # if it's already a string, return it
            if isinstance(val, str):
                return val
            # if it's token ids (list or array), decode
            if hasattr(val, 'tolist'):
                val = val.tolist()
            if isinstance(val, list) and len(val) > 0:
                # filter out padding tokens (-100 for labels)
                val = [t for t in val if t >= 0 and t != self.pad_token_id]
                if val:
                    return self.tokenizer.decode(val, skip_special_tokens=True)
        return None

    def _has_pretokenized_inputs(self, examples: List[Dict]) -> bool:
        """Check if examples have pre-tokenized input_ids from collator."""
        if not examples:
            return False
        first = examples[0]
        if 'input_ids' not in first:
            return False
        val = first['input_ids']
        # check if it's an array/list of ints, not a string
        if hasattr(val, 'tolist') or (isinstance(val, list) and len(val) > 0):
            return True
        return False

    def _batch_pretokenized_inputs(
        self, examples: List[Dict]
    ) -> Dict[str, jnp.ndarray]:
        """Batch pre-tokenized inputs from collator, padding to max length in batch.

        Also extracts decoder_input_ids (truncated to first token only) for
        passing the correct decoder start token to generate().
        """
        input_ids_list = []
        attention_mask_list = []
        decoder_start_tokens = []

        for ex in examples:
            ids = ex['input_ids']
            if hasattr(ids, 'tolist'):
                ids = ids.tolist()
            # filter padding tokens for length calculation
            ids = [t for t in ids if t != self.pad_token_id]
            input_ids_list.append(ids)

            # use existing attention_mask if available, else create from ids
            if 'attention_mask' in ex:
                mask = ex['attention_mask']
                if hasattr(mask, 'tolist'):
                    mask = mask.tolist()
                # filter to match non-padded length
                mask = mask[:len(ids)]
                attention_mask_list.append(mask)
            else:
                attention_mask_list.append([1] * len(ids))

            # extract decoder start token from decoder_input_ids[0]
            if 'decoder_input_ids' in ex:
                dec_ids = ex['decoder_input_ids']
                if hasattr(dec_ids, 'tolist'):
                    dec_ids = dec_ids.tolist()
                decoder_start_tokens.append(dec_ids[0])
            else:
                # fallback to hangul token if no decoder_input_ids
                decoder_start_tokens.append(self.hangul_token_id)

        # pad to fixed max_input_length (like tokenize_batch) for JIT compatibility
        max_len = self.max_input_length

        padded_ids = []
        padded_masks = []
        for ids, mask in zip(input_ids_list, attention_mask_list):
            # truncate if needed
            ids = ids[:max_len]
            mask = mask[:max_len]
            # pad to max_len
            pad_len = max_len - len(ids)
            padded_ids.append(ids + [self.pad_token_id] * pad_len)
            padded_masks.append(mask + [0] * pad_len)

        result = {
            'input_ids': jnp.array(padded_ids, dtype=jnp.int32),
            'attention_mask': jnp.array(padded_masks, dtype=jnp.int32),
        }

        # add decoder_input_ids as single token per example (for generate())
        if decoder_start_tokens:
            result['decoder_input_ids'] = jnp.array(
                [[t] for t in decoder_start_tokens], dtype=jnp.int32
            )

        return result

    def format_input(self, example: Dict) -> str:
        """Format input with task prompt.

        Handles raw text fields, collator text fields, and tokenized input_ids.
        """
        # try to get text from various sources
        text = None

        # first try tokenized input_ids (from collator)
        text = self._decode_if_tokenized(example, 'input_ids')

        # then try text fields
        if not text:
            text = (example.get('original_text') or
                    example.get('input_text') or
                    example.get('title') or
                    example.get('text') or '')

        # for NLI, format premise/hypothesis if present
        if self.task == 'nli' and not text:
            premise = example.get('premise', '')
            hypothesis = example.get('hypothesis', '')
            if premise and hypothesis:
                text = f"전제: {premise}\n가설: {hypothesis}"

        # for STS, format sentence pairs if present
        if self.task == 'sts' and not text:
            s1 = example.get('sentence1', '')
            s2 = example.get('sentence2', '')
            if s1 and s2:
                text = f"sentence1: {s1} sentence2: {s2}"

        # for instruction tasks, use instruction as prompt + input as text
        if self.task == 'instruction' and not text:
            instruction = example.get('metadata', example.get('instruction', ''))
            input_text = example.get('original_text', example.get('input_text', ''))
            if instruction:
                return f"{instruction} {input_text}".strip() if input_text else instruction
            text = input_text

        # transcribe Hanja to Hangul for temporal eval so the model
        # can't predict the year from script distribution alone
        if self.task == 'temporal' and text:
            text = transcribe(text)

        # add task prompt if requested (and not already in decoded text)
        if self.use_task_prompts and text and self.task != 'instruction':
            task_type = {
                'ynat': 'classification',
                'nli': 'nli',
                'sts': 'sts',
                'temporal': 'temporal_classification'
            }.get(self.task, self.task)
            prompt, _ = sample_task_prompt(task_type, language='ko')
            if prompt and prompt not in text:
                return f"{prompt} {text}"

        return text or ''

    def _is_chat_collator(self) -> bool:
        """True iff a ChatSFTCollator was wired in for chat-template eval."""
        if self.collator is None:
            return False
        try:
            from chat_sft_collator import ChatSFTCollator
        except ImportError:
            return False
        return isinstance(self.collator, ChatSFTCollator)

    def _to_collator_example(self, example: Dict) -> Dict:
        """Map an eval example to the chat collator's supervised contract.

        Splits the task prompt (system) from the user content so the collator
        can build `<|system|>{prompt}<|user|>{text}<|end_of_turn|>` instead of
        cramming the prompt into the user turn.
        """
        # extract user text per task (matches format_input but without the prompt prefix)
        if self.task == 'instruction':
            instruction = example.get('metadata') or example.get('instruction') or ''
            user_text = example.get('original_text') or example.get('input_text') or ''
            metadata = instruction
        else:
            user_text = (
                example.get('original_text')
                or example.get('input_text')
                or example.get('title')
                or example.get('text')
                or ''
            )

            if self.task == 'nli' and not user_text:
                premise = example.get('premise', '')
                hypothesis = example.get('hypothesis', '')
                if premise and hypothesis:
                    user_text = f"전제: {premise}\n가설: {hypothesis}"

            if self.task == 'sts' and not user_text:
                s1 = example.get('sentence1', '')
                s2 = example.get('sentence2', '')
                if s1 and s2:
                    user_text = f"sentence1: {s1} sentence2: {s2}"

            if self.task == 'temporal' and user_text:
                user_text = transcribe(user_text)

            if self.use_task_prompts:
                task_type = {
                    'ynat': 'classification',
                    'nli': 'nli',
                    'sts': 'sts',
                    'temporal': 'temporal_classification',
                }.get(self.task, self.task)
                prompt, _ = sample_task_prompt(task_type, language='ko')
                metadata = prompt or ''
            else:
                metadata = example.get('metadata', '') or ''

        return {
            'metadata': metadata,
            'original_text': user_text or '',
            # placeholder; not consumed during generation, but the chat
            # collator requires a labels field to build a (truncated) target
            'labels': '',
            'language': example.get('language', 'korean'),
            'source': example.get('source', 'eval'),
        }

    def _chat_tokenize_batch(self, examples: List[Dict]) -> Dict[str, jnp.ndarray]:
        """Tokenize a batch via the chat collator's supervised handler.

        Bypasses task-routing in `__call__` -- we already know each example is
        a supervised classification/STS/temporal/instruction case, and we want
        the chat formatting regardless of `data_type`.
        """
        prepared = [self._to_collator_example(ex) for ex in examples]
        collated = self.collator._handle_supervised(prepared, padding=True)

        input_ids = np.array(collated['input_ids'])
        attention_mask = np.array(collated['attention_mask'])
        decoder_input_ids = np.array(collated['decoder_input_ids'])[:, :1]

        return {
            'input_ids': jnp.array(input_ids, dtype=jnp.int32),
            'attention_mask': jnp.array(attention_mask, dtype=jnp.int32),
            'decoder_input_ids': jnp.array(decoder_input_ids, dtype=jnp.int32),
        }

    def tokenize_batch(self, texts: List[str]) -> Dict[str, jnp.ndarray]:
        """Tokenize a batch of texts for the encoder."""
        if not texts:
            raise ValueError("tokenize_batch received empty texts list")

        # sanitize: replace None or empty strings with placeholder
        sanitized = [t if t else "[EMPTY]" for t in texts]

        encoded = self.tokenizer(
            sanitized,
            padding='max_length',
            truncation=True,
            max_length=self.max_input_length - 1,
            return_tensors='np',
            add_special_tokens=True,
        )

        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']

        # append script token (hangul for modern Korean text)
        batch_size = input_ids.shape[0]
        script_tokens = np.full((batch_size, 1), self.hangul_token_id)
        script_mask = np.ones((batch_size, 1), dtype=attention_mask.dtype)

        input_ids = np.concatenate([input_ids, script_tokens], axis=1)
        attention_mask = np.concatenate([attention_mask, script_mask], axis=1)

        return {
            'input_ids': jnp.array(input_ids),
            'attention_mask': jnp.array(attention_mask),
        }

    def decode_predictions(self, generated_ids: jnp.ndarray) -> List[str]:
        """Decode generated token IDs to text."""
        generated_ids = np.array(generated_ids)
        predictions = []

        for ids in generated_ids:
            # find first EOS or pad token
            eos_positions = np.where((ids == self.eos_token_id) | (ids == self.pad_token_id))[0]
            if len(eos_positions) > 0:
                ids = ids[:eos_positions[0]]

            # decode and clean
            text = self.tokenizer.decode(ids, skip_special_tokens=True)
            text = text.strip()
            predictions.append(text)

        return predictions

    def match_label(self, prediction: str) -> Optional[int]:
        """Match prediction to a label index using exact and fuzzy matching."""
        prediction = prediction.strip()

        # exact match
        if prediction in self.label_to_idx:
            return self.label_to_idx[prediction]

        # fuzzy match: check if prediction starts with or contains a label
        for label, idx in self.label_to_idx.items():
            if prediction.startswith(label) or label in prediction:
                return idx

        # no match
        return None

    def evaluate(self, split: str = 'validation', max_samples: Optional[int] = None) -> Dict:
        """Run evaluation on the dataset."""
        # use pre-loaded dataset if available, otherwise load from source
        dataset = self.dataset

        if max_samples and len(dataset) > max_samples:
            if isinstance(dataset, list):
                import random
                random.seed(42)
                dataset = random.sample(dataset, max_samples)
            else:
                dataset = dataset.select(range(max_samples))

        # ensure model is in eval mode
        self.model.eval()

        # reset per-call CoT bookkeeping; the callback wrapper sets the
        # override attrs before each evaluate() call.
        self.parse_fallback_count = 0

        if self.task in ['ynat', 'nli']:
            return self._evaluate_classification(dataset, split)
        elif self.task == 'sts':
            return self._evaluate_sts(dataset, split)
        elif self.task == 'temporal':
            return self._evaluate_temporal(dataset, split)
        elif self.task == 'instruction':
            return self._evaluate_instruction(dataset, split)
        else:
            raise ValueError(f"Unsupported task: {self.task}")

    def _get_label(self, example: Dict) -> str:
        """Get label from example, handling HF, collator text, and tokenized labels."""
        # try to decode tokenized labels first (from collator)
        decoded = self._decode_if_tokenized(example, 'labels')
        if decoded:
            return decoded

        # collator uses: target, target_text
        if 'target' in example and isinstance(example['target'], str):
            return example['target']
        if 'target_text' in example:
            return example['target_text']

        # HF uses: label (as int index)
        if 'label' in example:
            label = example['label']
            # HF returns int index, convert to text
            if isinstance(label, int) and self.idx_to_label:
                return self.idx_to_label[label]
            if isinstance(label, str):
                return label

        return ''

    def _evaluate_classification(self, dataset, split: str) -> Dict:
        """Evaluate classification tasks (YNAT, NLI)."""
        all_preds = []
        all_labels = []
        all_pred_texts = []
        unmatched = []

        num_examples = len(dataset)
        num_batches = (num_examples + self.batch_size - 1) // self.batch_size

        # check once if we have pre-tokenized data
        first_batch = dataset[:min(self.batch_size, num_examples)]
        if isinstance(first_batch, dict):
            first_batch = [{k: first_batch[k][i] for k in first_batch.keys()}
                           for i in range(len(first_batch[list(first_batch.keys())[0]]))]
        use_pretokenized = self._has_pretokenized_inputs(first_batch)

        for batch_idx in tqdm(range(num_batches), desc=f"Evaluating {self.task}"):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, num_examples)

            # handle both HF datasets and list of dicts
            batch_data = dataset[start_idx:end_idx]
            if isinstance(batch_data, dict):
                batch_size = end_idx - start_idx
                examples = [{k: batch_data[k][i] for k in batch_data.keys()}
                            for i in range(batch_size)]
            else:
                examples = batch_data

            # always format texts for error logging
            texts = [self.format_input(ex) for ex in examples]

            # use pre-tokenized inputs if available, else tokenize formatted texts
            if use_pretokenized:
                inputs = self._batch_pretokenized_inputs(examples)
            elif self._is_chat_collator():
                inputs = self._chat_tokenize_batch(examples)
            else:
                inputs = self.tokenize_batch(texts)

            inputs = self._apply_decoder_start_override(inputs)

            # debug: log first batch inputs
            if batch_idx == 0:
                log_from_main_process(logger, 'info', f"\n{'='*60}")
                log_from_main_process(logger, 'info', f"DEBUG: First batch model inputs for {self.task}")
                log_from_main_process(logger, 'info', f"{'='*60}")
                for i in range(min(3, len(examples))):
                    log_from_main_process(logger, 'info', f"\n--- Example {i} ---")
                    log_from_main_process(logger, 'info', f"Available fields: {list(examples[i].keys())}")
                    log_from_main_process(logger, 'info', f"Formatted text: {texts[i][:200]}...")
                    input_ids = np.array(inputs['input_ids'][i])
                    non_pad = input_ids[input_ids != self.pad_token_id]
                    decoded_input = self.tokenizer.decode(non_pad, skip_special_tokens=False)
                    log_from_main_process(logger, 'info', f"Decoded input_ids: {decoded_input[:200]}...")
                    log_from_main_process(logger, 'info', f"Expected label: {self._get_label(examples[i])}")
                    log_from_main_process(logger, 'info', f"input_ids shape: {inputs['input_ids'].shape}")
                log_from_main_process(logger, 'info', f"{'='*60}\n")

            rngs = nnx.Rngs(jax.random.PRNGKey(batch_idx))
            # pass decoder_input_ids (first token only) if available from collator
            decoder_input_ids = inputs.get('decoder_input_ids', None)
            generated_ids = self.generate(
                self.model,
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                decoder_input_ids=decoder_input_ids,
                rngs=rngs,
            )

            if self.parse_cot_answer:
                cot_entries = self.decode_predictions_cot(generated_ids)
                pred_texts = [e['answer'] for e in cot_entries]
            else:
                cot_entries = None
                pred_texts = self.decode_predictions(generated_ids)

            # debug: log first batch outputs
            if batch_idx == 0:
                log_from_main_process(logger, 'info', f"\n{'='*60}")
                log_from_main_process(logger, 'info', f"DEBUG: First batch model outputs for {self.task}")
                log_from_main_process(logger, 'info', f"{'='*60}")
                # show actual decoder start token used
                if decoder_input_ids is not None:
                    start_token_id = int(decoder_input_ids[0, 0])
                    start_token = self.tokenizer.decode([start_token_id])
                    log_from_main_process(logger, 'info', f"Decoder start token (from collator): {start_token} (id={start_token_id})")
                else:
                    log_from_main_process(logger, 'info', f"Decoder start token (fallback): {self.tokenizer.decode([self.hangul_token_id])}")
                for i in range(min(3, len(pred_texts))):
                    if cot_entries is not None:
                        entry = cot_entries[i]
                        log_from_main_process(logger, 'info',
                            f"Rationale [{i}]: '{entry['rationale']}'")
                        log_from_main_process(logger, 'info',
                            f"Answer [{i}]: '{entry['answer']}' (parsed={entry['parsed']})")
                    else:
                        log_from_main_process(logger, 'info', f"Generated [{i}]: '{pred_texts[i]}'")
                    log_from_main_process(logger, 'info', f"Expected [{i}]: '{self._get_label(examples[i])}'")
                    raw_ids = np.array(generated_ids[i])
                    log_from_main_process(logger, 'info', f"Raw token ids: {raw_ids[:20].tolist()}...")
                log_from_main_process(logger, 'info', f"{'='*60}\n")

            all_pred_texts.extend(pred_texts)

            for i, pred_text in enumerate(pred_texts):
                true_label_text = self._get_label(examples[i])
                # convert text label to index
                true_idx = self.label_to_idx.get(true_label_text, -1)
                all_labels.append(true_idx)

                pred_idx = self.match_label(pred_text)
                if pred_idx is not None:
                    all_preds.append(pred_idx)
                else:
                    all_preds.append(-1)
                    unmatched.append({
                        'text': texts[i][:100],
                        'prediction': pred_text,
                        'true_label': true_label_text,
                    })

        accuracy = accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        results = {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'total_examples': len(all_labels),
            'unmatched_count': len(unmatched),
            'unmatched_ratio': len(unmatched) / len(all_labels) if all_labels else 0,
        }

        if self.parse_cot_answer:
            results['parse_fallback'] = self.parse_fallback_count
            results['parse_fallback_ratio'] = (
                self.parse_fallback_count / len(all_labels) if all_labels else 0
            )

        log_from_main_process(logger, 'info', f"\n{'='*60}")
        log_from_main_process(logger, 'info', f"{self.task.upper()} Evaluation Results ({split})")
        log_from_main_process(logger, 'info', f"{'='*60}")
        log_from_main_process(logger, 'info', f"Accuracy: {accuracy:.4f}")
        log_from_main_process(logger, 'info', f"Macro F1: {macro_f1:.4f}")
        log_from_main_process(logger, 'info', f"Unmatched: {len(unmatched)} ({results['unmatched_ratio']*100:.1f}%)")
        if self.parse_cot_answer:
            log_from_main_process(logger, 'info',
                f"CoT parse fallbacks: {self.parse_fallback_count} "
                f"({results['parse_fallback_ratio']*100:.1f}%)")

        if unmatched[:3]:
            log_from_main_process(logger, 'info', f"\nSample unmatched:")
            for u in unmatched[:3]:
                log_from_main_process(logger, 'info', f"  '{u['prediction']}' vs '{u['true_label']}'")

        return results

    def _evaluate_sts(self, dataset: List[Dict], split: str) -> Dict:
        """Evaluate STS task with Pearson/Spearman correlation."""
        all_pred_scores = []
        all_true_scores = []
        parse_failures = 0

        num_examples = len(dataset)
        num_batches = (num_examples + self.batch_size - 1) // self.batch_size

        # check once if we have pre-tokenized data
        first_batch = dataset[:min(self.batch_size, num_examples)]
        if isinstance(first_batch, dict):
            first_batch = [{k: first_batch[k][i] for k in first_batch.keys()}
                           for i in range(len(first_batch[list(first_batch.keys())[0]]))]
        use_pretokenized = self._has_pretokenized_inputs(first_batch)

        for batch_idx in tqdm(range(num_batches), desc="Evaluating STS"):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, num_examples)
            examples = dataset[start_idx:end_idx]

            # always format texts for error logging
            texts = [self.format_input(ex) for ex in examples]

            # use pre-tokenized inputs if available
            if use_pretokenized:
                inputs = self._batch_pretokenized_inputs(examples)
            elif self._is_chat_collator():
                inputs = self._chat_tokenize_batch(examples)
            else:
                inputs = self.tokenize_batch(texts)

            inputs = self._apply_decoder_start_override(inputs)

            # debug: log first batch inputs
            if batch_idx == 0:
                log_from_main_process(logger, 'info', f"\n{'='*60}")
                log_from_main_process(logger, 'info', f"DEBUG: First batch model inputs for STS")
                log_from_main_process(logger, 'info', f"{'='*60}")
                for i in range(min(3, len(examples))):
                    log_from_main_process(logger, 'info', f"\n--- Example {i} ---")
                    log_from_main_process(logger, 'info', f"Available fields: {list(examples[i].keys())}")
                    log_from_main_process(logger, 'info', f"Formatted text: {texts[i][:200]}...")
                    input_ids = np.array(inputs['input_ids'][i])
                    non_pad = input_ids[input_ids != self.pad_token_id]
                    decoded_input = self.tokenizer.decode(non_pad, skip_special_tokens=False)
                    log_from_main_process(logger, 'info', f"Decoded input_ids: {decoded_input[:200]}...")
                    ex = examples[i]
                    # mirror the scoring cascade below: labels -> target_text -> original_score
                    label_text = self._decode_if_tokenized(ex, 'labels')
                    if not label_text:
                        label_text = self._decode_if_tokenized(ex, 'target_text')
                    if not label_text and 'original_score' in ex:
                        label_text = str(ex['original_score'])
                    log_from_main_process(logger, 'info', f"Expected score (decoded labels): {label_text}")
                    log_from_main_process(logger, 'info', f"input_ids shape: {inputs['input_ids'].shape}")
                log_from_main_process(logger, 'info', f"{'='*60}\n")

            rngs = nnx.Rngs(jax.random.PRNGKey(batch_idx))
            # pass decoder_input_ids (first token only) if available from collator
            decoder_input_ids = inputs.get('decoder_input_ids', None)
            generated_ids = self.generate(
                self.model,
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                decoder_input_ids=decoder_input_ids,
                rngs=rngs,
            )

            if self.parse_cot_answer:
                cot_entries = self.decode_predictions_cot(generated_ids)
                pred_texts = [e['answer'] for e in cot_entries]
            else:
                cot_entries = None
                pred_texts = self.decode_predictions(generated_ids)

            # debug: log first batch outputs
            if batch_idx == 0:
                log_from_main_process(logger, 'info', f"\n{'='*60}")
                log_from_main_process(logger, 'info', f"DEBUG: First batch model outputs for STS")
                log_from_main_process(logger, 'info', f"{'='*60}")
                # show actual decoder start token used
                if decoder_input_ids is not None:
                    start_token_id = int(decoder_input_ids[0, 0])
                    start_token = self.tokenizer.decode([start_token_id])
                    log_from_main_process(logger, 'info', f"Decoder start token (from collator): {start_token} (id={start_token_id})")
                else:
                    log_from_main_process(logger, 'info', f"Decoder start token (fallback): {self.tokenizer.decode([self.hangul_token_id])}")
                for i in range(min(3, len(pred_texts))):
                    if cot_entries is not None:
                        entry = cot_entries[i]
                        log_from_main_process(logger, 'info',
                            f"Rationale [{i}]: '{entry['rationale']}'")
                        log_from_main_process(logger, 'info',
                            f"Answer [{i}]: '{entry['answer']}' (parsed={entry['parsed']})")
                    else:
                        log_from_main_process(logger, 'info', f"Generated [{i}]: '{pred_texts[i]}'")
                    raw_ids = np.array(generated_ids[i])
                    log_from_main_process(logger, 'info', f"Raw token ids: {raw_ids[:20].tolist()}...")
                log_from_main_process(logger, 'info', f"{'='*60}\n")

            for i, pred_text in enumerate(pred_texts):
                # decode labels to get score (collator output is tokenized)
                ex = examples[i]
                label_text = self._decode_if_tokenized(ex, 'labels')
                if not label_text:
                    label_text = self._decode_if_tokenized(ex, 'target_text')
                if not label_text and 'original_score' in ex:
                    label_text = str(ex['original_score'])
                if not label_text:
                    raise ValueError(f"Failed to decode labels from example: {list(ex.keys())}")
                true_score = float(label_text)
                all_true_scores.append(true_score)

                # try to parse score from prediction
                try:
                    score_match = re.search(r'(\d+\.?\d*)', pred_text)
                    if score_match:
                        pred_score = float(score_match.group(1))
                        pred_score = max(0.0, min(5.0, pred_score))
                    else:
                        pred_score = 2.5
                        parse_failures += 1
                except:
                    pred_score = 2.5
                    parse_failures += 1

                all_pred_scores.append(pred_score)

        pearson_r, _ = pearsonr(all_true_scores, all_pred_scores)
        spearman_r, _ = spearmanr(all_true_scores, all_pred_scores)
        mae = np.mean(np.abs(np.array(all_true_scores) - np.array(all_pred_scores)))

        results = {
            'pearson': pearson_r,
            'spearman': spearman_r,
            'mae': mae,
            'total_examples': len(all_true_scores),
            'parse_failures': parse_failures,
        }

        if self.parse_cot_answer:
            results['parse_fallback'] = self.parse_fallback_count
            results['parse_fallback_ratio'] = (
                self.parse_fallback_count / len(all_true_scores)
                if all_true_scores else 0
            )

        log_from_main_process(logger, 'info', f"\n{'='*60}")
        log_from_main_process(logger, 'info', f"STS Evaluation Results ({split})")
        log_from_main_process(logger, 'info', f"{'='*60}")
        log_from_main_process(logger, 'info', f"Pearson r: {pearson_r:.4f}")
        log_from_main_process(logger, 'info', f"Spearman r: {spearman_r:.4f}")
        log_from_main_process(logger, 'info', f"MAE: {mae:.4f}")
        log_from_main_process(logger, 'info', f"Parse failures: {parse_failures} ({parse_failures/len(all_true_scores)*100:.1f}%)")

        return results

    def _evaluate_temporal(self, dataset: List[Dict], split: str) -> Dict:
        """Evaluate temporal classification with decade/century matching."""
        correct_exact = 0
        correct_decade = 0
        correct_century = 0
        total = 0
        year_errors = []
        parse_failures = 0
        true_decades = []
        pred_decades = []

        num_examples = len(dataset)
        num_batches = (num_examples + self.batch_size - 1) // self.batch_size

        # check once if we have pre-tokenized data
        first_batch = dataset[:min(self.batch_size, num_examples)]
        if isinstance(first_batch, dict):
            first_batch = [{k: first_batch[k][i] for k in first_batch.keys()}
                           for i in range(len(first_batch[list(first_batch.keys())[0]]))]
        use_pretokenized = self._has_pretokenized_inputs(first_batch)

        for batch_idx in tqdm(range(num_batches), desc="Evaluating Temporal"):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, num_examples)
            examples = dataset[start_idx:end_idx]

            if not examples:
                continue

            # always format texts for error logging
            texts = [self.format_input(ex) for ex in examples]

            # use pre-tokenized inputs if available
            if use_pretokenized:
                inputs = self._batch_pretokenized_inputs(examples)
            elif self._is_chat_collator():
                inputs = self._chat_tokenize_batch(examples)
            else:
                inputs = self.tokenize_batch(texts)

            inputs = self._apply_decoder_start_override(inputs)

            # debug: log first batch inputs
            if batch_idx == 0:
                log_from_main_process(logger, 'info', f"\n{'='*60}")
                log_from_main_process(logger, 'info', f"DEBUG: First batch model inputs for Temporal")
                log_from_main_process(logger, 'info', f"{'='*60}")
                for i in range(min(3, len(examples))):
                    log_from_main_process(logger, 'info', f"\n--- Example {i} ---")
                    log_from_main_process(logger, 'info', f"Available fields: {list(examples[i].keys())}")
                    log_from_main_process(logger, 'info', f"Formatted text: {texts[i][:200]}...")
                    input_ids = np.array(inputs['input_ids'][i])
                    non_pad = input_ids[input_ids != self.pad_token_id]
                    decoded_input = self.tokenizer.decode(non_pad, skip_special_tokens=False)
                    log_from_main_process(logger, 'info', f"Decoded input_ids: {decoded_input[:200]}...")
                    ex = examples[i]
                    expected_year = ex.get('year', self._decode_if_tokenized(ex, 'labels'))
                    log_from_main_process(logger, 'info', f"Expected year: {expected_year}")
                    log_from_main_process(logger, 'info', f"input_ids shape: {inputs['input_ids'].shape}")
                log_from_main_process(logger, 'info', f"{'='*60}\n")

            rngs = nnx.Rngs(jax.random.PRNGKey(batch_idx))
            # pass decoder_input_ids (first token only) if available from collator
            decoder_input_ids = inputs.get('decoder_input_ids', None)
            generated_ids = self.generate(
                self.model,
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                decoder_input_ids=decoder_input_ids,
                rngs=rngs,
            )

            if self.parse_cot_answer:
                cot_entries = self.decode_predictions_cot(generated_ids)
                pred_texts = [e['answer'] for e in cot_entries]
            else:
                cot_entries = None
                pred_texts = self.decode_predictions(generated_ids)

            # debug: log first batch outputs
            if batch_idx == 0:
                log_from_main_process(logger, 'info', f"\n{'='*60}")
                log_from_main_process(logger, 'info', f"DEBUG: First batch model outputs for Temporal")
                log_from_main_process(logger, 'info', f"{'='*60}")
                # show actual decoder start token used
                if decoder_input_ids is not None:
                    start_token_id = int(decoder_input_ids[0, 0])
                    start_token = self.tokenizer.decode([start_token_id])
                    log_from_main_process(logger, 'info', f"Decoder start token (from collator): {start_token} (id={start_token_id})")
                else:
                    log_from_main_process(logger, 'info', f"Decoder start token (fallback): {self.tokenizer.decode([self.hangul_token_id])}")
                for i in range(min(3, len(pred_texts))):
                    if cot_entries is not None:
                        entry = cot_entries[i]
                        log_from_main_process(logger, 'info',
                            f"Rationale [{i}]: '{entry['rationale']}'")
                        log_from_main_process(logger, 'info',
                            f"Answer [{i}]: '{entry['answer']}' (parsed={entry['parsed']})")
                    else:
                        log_from_main_process(logger, 'info', f"Generated [{i}]: '{pred_texts[i]}'")
                    # show raw token ids
                    raw_ids = np.array(generated_ids[i])
                    log_from_main_process(logger, 'info', f"Raw token ids: {raw_ids[:20].tolist()}...")
                log_from_main_process(logger, 'info', f"{'='*60}\n")

            for i, pred_text in enumerate(pred_texts):
                ex = examples[i]

                # get true year: prefer 'year' field (raw int), then try decoded labels
                true_year = None
                if 'year' in ex:
                    try:
                        true_year = int(ex['year'])
                    except (ValueError, TypeError):
                        pass
                if true_year is None:
                    label_text = self._decode_if_tokenized(ex, 'label')
                    if not label_text:
                        label_text = self._decode_if_tokenized(ex, 'labels')
                    if not label_text:
                        raise ValueError(f"Failed to decode labels from example: {list(ex.keys())}")
                    true_year = parse_temporal_prediction(label_text)
                    if true_year is None:
                        raise ValueError(f"Failed to parse year from label: '{label_text}'")

                pred_year = parse_temporal_prediction(pred_text)
                if pred_year is None:
                    parse_failures += 1
                    continue

                total += 1
                year_error = abs(true_year - pred_year)
                year_errors.append(year_error)

                # exact year match (within 5 years)
                if year_error <= 5:
                    correct_exact += 1

                # decade match
                if true_year // 10 == pred_year // 10:
                    correct_decade += 1

                # century match
                if true_year // 100 == pred_year // 100:
                    correct_century += 1

                true_decades.append(true_year // 10 * 10)
                pred_decades.append(pred_year // 10 * 10)

        mae = np.mean(year_errors) if year_errors else float('inf')

        # decade-bucketed F1 (consistent with temporal_classification.py)
        from sklearn.metrics import accuracy_score, f1_score
        if true_decades:
            decade_macro_f1 = f1_score(true_decades, pred_decades, average='macro', zero_division=0)
            decade_weighted_f1 = f1_score(true_decades, pred_decades, average='weighted', zero_division=0)
            decade_class_accuracy = accuracy_score(true_decades, pred_decades)
        else:
            decade_macro_f1 = 0.0
            decade_weighted_f1 = 0.0
            decade_class_accuracy = 0.0

        results = {
            'exact_accuracy': correct_exact / total if total else 0,
            'decade_accuracy': correct_decade / total if total else 0,
            'century_accuracy': correct_century / total if total else 0,
            'mae_years': mae,
            'macro_f1': decade_macro_f1,
            'weighted_f1': decade_weighted_f1,
            'total_examples': total,
            'parse_failures': parse_failures,
        }

        if self.parse_cot_answer:
            # denominator must be every example processed, not just temporal-
            # parseable ones; `total` excludes temporal parse_failures and can
            # push the ratio above 1.0 when CoT outputs uniformly skip the
            # <|assistant|> transition.
            seen = total + parse_failures
            results['parse_fallback'] = self.parse_fallback_count
            results['parse_fallback_ratio'] = (
                self.parse_fallback_count / seen if seen else 0
            )

        log_from_main_process(logger, 'info', f"\n{'='*60}")
        log_from_main_process(logger, 'info', f"Temporal Classification Results ({split})")
        log_from_main_process(logger, 'info', f"{'='*60}")
        log_from_main_process(logger, 'info', f"Exact (+-5yr): {results['exact_accuracy']:.4f}")
        log_from_main_process(logger, 'info', f"Decade match:  {results['decade_accuracy']:.4f}")
        log_from_main_process(logger, 'info', f"Century match: {results['century_accuracy']:.4f}")
        log_from_main_process(logger, 'info', f"Macro F1:      {decade_macro_f1:.4f}")
        log_from_main_process(logger, 'info', f"Weighted F1:   {decade_weighted_f1:.4f}")
        log_from_main_process(logger, 'info', f"MAE (years):   {mae:.1f}")
        log_from_main_process(logger, 'info', f"Parse failures: {parse_failures}")

        return results

    def _evaluate_instruction(self, dataset: List[Dict], split: str) -> Dict:
        """Evaluate instruction-following with BLEU against reference outputs."""
        try:
            import sacrebleu
        except ImportError:
            log_from_main_process(logger, 'warning',
                "sacrebleu not installed, instruction eval will only show samples")
            sacrebleu = None

        all_refs = []
        all_preds = []

        num_examples = len(dataset)
        num_batches = (num_examples + self.batch_size - 1) // self.batch_size

        first_batch = dataset[:min(self.batch_size, num_examples)]
        if isinstance(first_batch, dict):
            first_batch = [{k: first_batch[k][i] for k in first_batch.keys()}
                           for i in range(len(first_batch[list(first_batch.keys())[0]]))]
        use_pretokenized = self._has_pretokenized_inputs(first_batch)

        for batch_idx in tqdm(range(num_batches), desc="Evaluating Instruction"):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, num_examples)
            examples = dataset[start_idx:end_idx]

            if not examples:
                continue

            texts = [self.format_input(ex) for ex in examples]

            if use_pretokenized:
                inputs = self._batch_pretokenized_inputs(examples)
            elif self._is_chat_collator():
                inputs = self._chat_tokenize_batch(examples)
            else:
                inputs = self.tokenize_batch(texts)

            inputs = self._apply_decoder_start_override(inputs)

            if batch_idx == 0:
                log_from_main_process(logger, 'info', f"\n{'='*60}")
                log_from_main_process(logger, 'info', f"DEBUG: First batch model inputs for Instruction")
                log_from_main_process(logger, 'info', f"{'='*60}")
                for i in range(min(3, len(examples))):
                    log_from_main_process(logger, 'info', f"\n--- Example {i} ---")
                    input_ids = np.array(inputs['input_ids'][i])
                    non_pad = input_ids[input_ids != self.pad_token_id]
                    decoded_input = self.tokenizer.decode(non_pad, skip_special_tokens=False)
                    log_from_main_process(logger, 'info', f"Decoded input_ids: {decoded_input[:300]}...")
                    label_text = self._get_label(examples[i])
                    log_from_main_process(logger, 'info', f"Reference: {label_text[:200]}...")
                log_from_main_process(logger, 'info', f"{'='*60}\n")

            rngs = nnx.Rngs(jax.random.PRNGKey(batch_idx))
            decoder_input_ids = inputs.get('decoder_input_ids', None)
            generated_ids = self.generate(
                self.model,
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                decoder_input_ids=decoder_input_ids,
                rngs=rngs,
            )

            if self.parse_cot_answer:
                cot_entries = self.decode_predictions_cot(generated_ids)
                pred_texts = [e['answer'] for e in cot_entries]
            else:
                cot_entries = None
                pred_texts = self.decode_predictions(generated_ids)

            if batch_idx == 0:
                log_from_main_process(logger, 'info', f"\n{'='*60}")
                log_from_main_process(logger, 'info', f"DEBUG: First batch model outputs for Instruction")
                log_from_main_process(logger, 'info', f"{'='*60}")
                for i in range(min(3, len(pred_texts))):
                    if cot_entries is not None:
                        entry = cot_entries[i]
                        log_from_main_process(logger, 'info',
                            f"Rationale [{i}]: '{entry['rationale'][:200]}'")
                        log_from_main_process(logger, 'info',
                            f"Answer [{i}]: '{entry['answer'][:200]}' (parsed={entry['parsed']})")
                    else:
                        log_from_main_process(logger, 'info', f"Generated [{i}]: '{pred_texts[i][:200]}'")
                    ref = self._get_label(examples[i])
                    log_from_main_process(logger, 'info', f"Reference [{i}]: '{ref[:200]}'")
                log_from_main_process(logger, 'info', f"{'='*60}\n")

            for i, pred_text in enumerate(pred_texts):
                ref_text = self._get_label(examples[i])
                all_refs.append(ref_text)
                all_preds.append(pred_text)

        results = {
            'total_examples': len(all_preds),
        }

        if self.parse_cot_answer:
            results['parse_fallback'] = self.parse_fallback_count
            results['parse_fallback_ratio'] = (
                self.parse_fallback_count / len(all_preds) if all_preds else 0
            )

        if sacrebleu is not None and all_preds:
            bleu = sacrebleu.corpus_bleu(all_preds, [all_refs])
            results['bleu'] = bleu.score
            results['bleu_bp'] = bleu.bp

            # exact match rate
            exact = sum(1 for p, r in zip(all_preds, all_refs) if p.strip() == r.strip())
            results['exact_match'] = exact / len(all_preds)

        log_from_main_process(logger, 'info', f"\n{'='*60}")
        log_from_main_process(logger, 'info', f"Instruction-Following Results ({split})")
        log_from_main_process(logger, 'info', f"{'='*60}")
        if 'bleu' in results:
            log_from_main_process(logger, 'info', f"BLEU: {results['bleu']:.2f}")
            log_from_main_process(logger, 'info', f"Brevity penalty: {results['bleu_bp']:.4f}")
            log_from_main_process(logger, 'info', f"Exact match: {results['exact_match']:.4f}")
        log_from_main_process(logger, 'info', f"Total examples: {results['total_examples']}")

        return results


def main():
    parser = argparse.ArgumentParser(description="Generative evaluation for Han2Han")
    parser.add_argument("--model_path", type=str, required=True, help="Path to Han2Han checkpoint")
    parser.add_argument("--task", type=str, choices=['ynat', 'nli'], required=True)
    parser.add_argument("--split", type=str, default='validation')
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_input_length", type=int, default=256)
    parser.add_argument("--max_output_length", type=int, default=16)
    parser.add_argument("--no_task_prompts", action='store_true', help="Disable task prompts")
    args = parser.parse_args()

    # load model and tokenizer
    log_from_main_process(logger, 'info', f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = FlaxHan2Han.from_pretrained(args.model_path, trust_remote_code=True)

    # create evaluator
    evaluator = GenerativeEvaluator(
        model=model,
        tokenizer=tokenizer,
        task=args.task,
        batch_size=args.batch_size,
        max_input_length=args.max_input_length,
        max_output_length=args.max_output_length,
        use_task_prompts=not args.no_task_prompts,
    )

    # run evaluation
    results = evaluator.evaluate(split=args.split)

    return results


if __name__ == "__main__":
    main()
