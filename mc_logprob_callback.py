#!/usr/bin/env python3
# coding: utf-8
"""
Multiple-choice log-probability evaluation callback.

Scores pre-loaded MC benchmarks (KMMLU, CLIcK, HAE-RAE) by teacher-forcing
each candidate answer through the decoder and picking the highest log-prob
per UTF-8 byte (the lm-eval-harness `acc_norm` metric). Dividing the summed
log-prob by the candidate's UTF-8 byte length neutralizes tokenizer-induced
length bias across variable-length answer options. Encoder runs once per
question; decoder runs once per candidate.

Data is pre-downloaded and uploaded to GCS as parquet. The callback receives
normalized examples via eval_data dict, not from HuggingFace directly.
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from base_callback import BaseCallback
from logging_utils import log_from_main_process
from task_prompts import sample_task_prompt

logger = logging.getLogger(__name__)


class MCLogProbCallback(BaseCallback):
    """Evaluate Korean knowledge benchmarks via log-prob MC scoring.

    Encodes the question once, teacher-forces each candidate through the
    decoder, sums token-level log-probs, and picks the argmax.

    Expected eval_data format (per benchmark):
        {
            'encoder_input': str,      # question/context
            'candidates': List[str],   # answer options
            'correct_idx': int,        # 0-indexed correct answer
            'category': str,           # for per-category reporting
            'benchmark': str,          # 'kmmlu', 'click', 'haerae'
        }
    """

    def __init__(
        self,
        tokenizer,
        eval_data: Dict[str, List[Dict]],
        mesh: Optional[Any] = None,
        max_input_length: int = 512,
        max_decoder_length: int = 64,
        eval_collator: Optional[Any] = None,
        max_eval_samples: int = 200,
        use_task_prompts: bool = True,
        batch_size: int = 8,
        seed: int = 42,
        **kwargs,
    ):
        """
        Args:
            tokenizer: Han2Han tokenizer
            eval_data: dict mapping benchmark name -> list of normalized MC examples
                (pre-loaded from GCS parquet in training script)
            mesh: JAX mesh for SPMD
            max_input_length: max encoder sequence length
            max_decoder_length: max decoder sequence length for candidates
            eval_collator: training collator for _handle_multiple_choice tokenization
            max_eval_samples: max examples per benchmark (sampling done upstream)
            use_task_prompts: whether to prepend task prompts
            batch_size: examples per forward pass (encoder and decoder batching)
            seed: random seed for task prompt sampling
        """
        self.mc_eval_data = eval_data
        self.max_input_length = max_input_length
        self.max_decoder_length = max_decoder_length
        self.use_task_prompts = use_task_prompts

        BaseCallback.__init__(
            self,
            tokenizer=tokenizer,
            max_length=max_input_length,
            max_eval_samples=max_eval_samples,
            batch_size=batch_size,
            seed=seed,
            mesh=mesh,
            eval_collator=eval_collator,
            **kwargs,
        )

    def _initialize_metrics(self):
        return None

    def _prepare_evaluation_data(self):
        """Validate and store pre-loaded benchmark data."""
        self.eval_examples: Dict[str, List[Dict]] = {}

        if not self.mc_eval_data:
            log_from_main_process(logger, 'warning',
                "No MC eval data provided, mc_logprob callback will be a no-op")
            return

        required_fields = {'encoder_input', 'candidates', 'correct_idx', 'category', 'benchmark'}

        for benchmark, examples in self.mc_eval_data.items():
            if not examples:
                log_from_main_process(logger, 'warning',
                    f"Empty eval data for {benchmark}, skipping")
                continue

            first = examples[0]
            missing = required_fields - set(first.keys())
            if missing:
                raise ValueError(
                    f"MC eval data for '{benchmark}' missing required fields: {missing}. "
                    f"Available: {list(first.keys())}. "
                    f"Run prepare_mc_eval_data.py to generate properly formatted parquets."
                )

            sampled = self._sample_evaluation_data(examples, self.max_eval_samples)
            if isinstance(sampled, list):
                self.eval_examples[benchmark] = sampled
            else:
                self.eval_examples[benchmark] = list(sampled)

            log_from_main_process(logger, 'info',
                f"MC eval: {benchmark} = {len(self.eval_examples[benchmark])} examples")

    def _tokenize_mc_example(self, example: Dict) -> Dict:
        """Tokenize one MC example into encoder + per-candidate decoder arrays.

        Uses the collator's _handle_multiple_choice for each candidate to
        ensure formatting matches training exactly.
        """
        encoder_input = example['encoder_input']
        candidates = example['candidates']

        all_input_ids = []
        all_attention_masks = []
        all_decoder_input_ids = []
        all_decoder_attention_masks = []
        all_labels = []

        for candidate in candidates:
            mc_example = {
                'original_text': encoder_input,
                'labels': candidate,
                'metadata': '',
                'source': 'mc_eval',
                'task_type': 'multiple_choice',
            }

            if self.eval_collator is not None:
                batch = self.eval_collator(
                    mc_example,
                    padding=True,
                )
            else:
                batch = self._fallback_tokenize(mc_example)

            all_input_ids.append(batch['input_ids'])
            all_attention_masks.append(batch['attention_mask'])
            all_decoder_input_ids.append(batch['decoder_input_ids'])
            all_decoder_attention_masks.append(batch['decoder_attention_mask'])
            all_labels.append(batch['labels'])

        byte_lens = np.array(
            [max(1, len(c.encode('utf-8'))) for c in candidates],
            dtype=np.float32,
        )

        return {
            'input_ids': np.stack(all_input_ids),
            'attention_mask': np.stack(all_attention_masks),
            'decoder_input_ids': np.stack(all_decoder_input_ids),
            'decoder_attention_mask': np.stack(all_decoder_attention_masks),
            'labels': np.stack(all_labels),
            'byte_lens': byte_lens,
            'correct_idx': example['correct_idx'],
            'num_candidates': len(candidates),
        }

    def _fallback_tokenize(self, example: Dict) -> Dict:
        """Tokenize without collator (for testing / standalone use).

        Replicates _handle_supervised format:
        encoder: [prompt] [question] <hangul>
        decoder_input: <hangul> [candidate]
        labels: [candidate] <hangul>
        """
        tokenizer = self.tokenizer
        hangul_id = tokenizer.convert_tokens_to_ids('<hangul>')

        prompt_text = ""
        if self.use_task_prompts:
            prompt_text, _ = sample_task_prompt('multiple_choice')
            prompt_text = prompt_text + " "

        input_text = example['original_text']
        label_text = example['labels']

        prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids if prompt_text else []
        input_ids = tokenizer(input_text, add_special_tokens=False).input_ids
        label_ids = tokenizer(label_text, add_special_tokens=False).input_ids

        enc_max = self.max_input_length
        dec_max = self.max_decoder_length

        encoder_text_max = enc_max - 1
        if len(prompt_ids) + len(input_ids) > encoder_text_max:
            available = encoder_text_max - len(prompt_ids)
            input_ids = input_ids[:available]
        if len(label_ids) > dec_max - 1:
            label_ids = label_ids[:dec_max - 1]

        encoder_ids = prompt_ids + input_ids + [hangul_id]
        decoder_input_ids = [hangul_id] + label_ids
        labels = label_ids + [hangul_id]

        pad_id = tokenizer.pad_token_id
        encoder_ids = encoder_ids + [pad_id] * (enc_max - len(encoder_ids))
        decoder_input_ids = decoder_input_ids + [pad_id] * (dec_max - len(decoder_input_ids))
        labels = labels + [-100] * (dec_max - len(labels))

        attention_mask = [1 if x != pad_id else 0 for x in encoder_ids]
        decoder_attention_mask = [1 if x != pad_id else 0 for x in decoder_input_ids]

        return {
            'input_ids': np.array(encoder_ids, dtype=np.int32),
            'attention_mask': np.array(attention_mask, dtype=np.int32),
            'decoder_input_ids': np.array(decoder_input_ids, dtype=np.int32),
            'decoder_attention_mask': np.array(decoder_attention_mask, dtype=np.int32),
            'labels': np.array(labels, dtype=np.int32),
        }

    def _get_jitted_fns(self, model):
        """Build and cache JIT-compiled encode / decode+score functions."""
        if hasattr(self, '_jitted_encode'):
            return self._jitted_encode, self._jitted_decode_score

        tie = model.config.tie_input_output_embeddings

        @nnx.jit
        def _encode(model, input_ids, attention_mask):
            outputs = model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                deterministic=True,
            )
            return outputs.last_hidden_state

        @nnx.jit
        def _decode_score(model, decoder_input_ids, decoder_attention_mask,
                          encoder_hidden, encoder_attention_mask,
                          shifted_labels, label_mask):
            """Batched decode + per-token log-prob scoring.

            Returns per-example total log probs: shape (batch,).
            """
            outputs = model.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=encoder_hidden,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=True,
                deterministic=True,
            )
            decoder_hidden = outputs.last_hidden_state
            if tie:
                logits = decoder_hidden @ model.decoder.wte.embedding.value.T
            else:
                logits = model.lm_head(decoder_hidden)
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            token_log_probs = jnp.take_along_axis(
                log_probs, shifted_labels[:, :, None], axis=-1,
            ).squeeze(-1)
            return jnp.sum(token_log_probs * label_mask, axis=-1)

        self._jitted_encode = _encode
        self._jitted_decode_score = _decode_score
        return self._jitted_encode, self._jitted_decode_score

    def _score_examples(self, model, tokenized_examples: List[Dict]):
        """Score MC examples in batches, return predicted indices and full log-prob matrix.

        Encodes a batch of questions, then scores each candidate separately
        with a dedicated decoder pass. This keeps compiled shapes constant
        regardless of num_candidates, ensuring stable memory usage.

        Returns:
            (predictions, all_logprobs) where predictions is a list[int] of argmax
            indices and all_logprobs is a (n_examples, num_candidates) float32 array.
            Subclasses (e.g. temporal scoring) may use the full matrix to compute
            distributional metrics like MAE / expected-value.
        """
        from sharding_utils import shard_batch_to_devices

        encode_fn, decode_score_fn = self._get_jitted_fns(model)
        num_candidates = tokenized_examples[0]['num_candidates']
        batch_size = self.batch_size
        n_examples = len(tokenized_examples)

        # SPMD sharding requires every per-step batch to have a global size
        # divisible by the data axis. batch_size is chosen to satisfy that;
        # the tail batch is the only thing that can break it. Pad up to a
        # multiple of batch_size with duplicates of the last example and trim
        # the duplicate rows from all_logprobs before returning.
        pad_n = (-n_examples) % batch_size
        if pad_n > 0:
            tokenized_examples = list(tokenized_examples) + [tokenized_examples[-1]] * pad_n
        n_total = len(tokenized_examples)

        all_enc_ids = np.stack([ex['input_ids'][0] for ex in tokenized_examples])
        all_enc_mask = np.stack([ex['attention_mask'][0] for ex in tokenized_examples])

        log_from_main_process(logger, 'info',
            f"Batched MC scoring: {n_examples} examples x {num_candidates} candidates, "
            f"batch_size={batch_size} (padded to {n_total})")

        all_logprobs = np.zeros((n_total, num_candidates), dtype=np.float32)

        for i in range(0, n_total, batch_size):
            end = i + batch_size

            enc_batch = shard_batch_to_devices({
                'input_ids': all_enc_ids[i:end],
                'attention_mask': all_enc_mask[i:end],
            }, self.mesh)
            encoder_hidden = encode_fn(
                model, enc_batch['input_ids'], enc_batch['attention_mask'])

            for c in range(num_candidates):
                dec_ids = np.stack([
                    tokenized_examples[j]['decoder_input_ids'][c]
                    for j in range(i, end)
                ])
                dec_mask = np.stack([
                    tokenized_examples[j]['decoder_attention_mask'][c]
                    for j in range(i, end)
                ])
                labels = np.stack([
                    tokenized_examples[j]['labels'][c]
                    for j in range(i, end)
                ])
                label_mask = (labels != -100).astype(np.float32)
                shifted_labels = np.where(labels == -100, 0, labels)

                dec_batch = shard_batch_to_devices({
                    'decoder_input_ids': dec_ids,
                    'decoder_attention_mask': dec_mask,
                    'shifted_labels': shifted_labels,
                    'label_mask': label_mask,
                }, self.mesh)

                candidate_logprobs = decode_score_fn(
                    model,
                    dec_batch['decoder_input_ids'],
                    dec_batch['decoder_attention_mask'],
                    encoder_hidden,
                    enc_batch['attention_mask'],
                    dec_batch['shifted_labels'],
                    dec_batch['label_mask'],
                )

                local_shards = candidate_logprobs.addressable_shards
                batch_lp = np.concatenate(
                    [np.asarray(s.data) for s in local_shards], axis=0)
                all_logprobs[i:end, c] = batch_lp

        all_logprobs = all_logprobs[:n_examples]
        predictions = np.argmax(all_logprobs, axis=1).tolist()
        return predictions, all_logprobs

    def evaluate(
        self,
        model: Any,
        step: int,
        rngs: Optional[nnx.Rngs] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run MC log-prob evaluation on all benchmarks."""
        results = {'step': step}
        model.eval()

        all_accuracies = []

        for benchmark, examples in self.eval_examples.items():
            jax.clear_caches()

            log_from_main_process(logger, 'info',
                f"MC scoring {benchmark} ({len(examples)} examples)...")

            tokenized = []
            for ex in examples:
                tok = self._tokenize_mc_example(ex)
                tok['category'] = ex['category']
                tokenized.append(tok)

            if not tokenized:
                log_from_main_process(logger, 'warning',
                    f"No tokenized examples for {benchmark}, skipping")
                continue

            # group by num_candidates so mixed-arity benchmarks (e.g. CLIcK
            # with 4-way + 5-way questions) get every option scored.
            # _score_examples sizes its output array from the first example's
            # num_candidates, so running all arities together would silently
            # drop the extra options.
            arity_groups: Dict[int, List[int]] = defaultdict(list)
            for idx, ex in enumerate(tokenized):
                arity_groups[ex['num_candidates']].append(idx)

            predictions: List[int] = [0] * len(tokenized)
            for nc, indices in arity_groups.items():
                group = [tokenized[i] for i in indices]
                if self.mesh is not None:
                    with self.mesh:
                        _, group_logprobs = self._score_examples(model, group)
                else:
                    _, group_logprobs = self._score_examples(model, group)

                # byte-length normalization (lm-eval-harness acc_norm): divide
                # each candidate's summed log-prob by its UTF-8 byte length so
                # longer answers aren't penalized by raw sequence length.
                for local_i, global_i in enumerate(indices):
                    byte_lens = tokenized[global_i]['byte_lens']
                    normalized = group_logprobs[local_i] / byte_lens
                    predictions[global_i] = int(np.argmax(normalized))

            correct_by_category = defaultdict(int)
            total_by_category = defaultdict(int)
            total_correct = 0

            for pred, ex in zip(predictions, tokenized):
                category = ex['category']
                is_correct = pred == ex['correct_idx']
                total_by_category[category] += 1
                if is_correct:
                    correct_by_category[category] += 1
                    total_correct += 1

            overall_acc = total_correct / len(tokenized) if tokenized else 0.0
            results[f'{benchmark}/accuracy'] = overall_acc
            all_accuracies.append(overall_acc)

            log_from_main_process(logger, 'info',
                f"{benchmark} accuracy: {overall_acc:.4f} ({total_correct}/{len(tokenized)})")

            for category in sorted(total_by_category.keys()):
                cat_total = total_by_category[category]
                cat_correct = correct_by_category[category]
                cat_acc = cat_correct / cat_total if cat_total > 0 else 0.0
                results[f'{benchmark}/{category}/accuracy'] = cat_acc

                log_from_main_process(logger, 'info',
                    f"  {category}: {cat_acc:.4f} ({cat_correct}/{cat_total})")

        if all_accuracies:
            results['total_mc_accuracy'] = sum(all_accuracies) / len(all_accuracies)

        model.train()
        return results
