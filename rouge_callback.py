#!/usr/bin/env python3
# coding: utf-8
"""
ROUGE evaluation callback for JAX/NNX training scripts.

Implements ROUGE scoring for summarization quality evaluation with support for
ROUGE-1, ROUGE-2, ROUGE-L, and ROUGE-Lsum metrics.
"""

import logging
from typing import Dict, Any, Optional, List
import jax.numpy as jnp
from flax import nnx
import numpy as np

# rouge evaluation
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("rouge-score not available. install with: pip install rouge-score")

from base_callback import BaseCallback, GenerationMixin
from logging_utils import log_from_main_process, log_from_all_processes

logger = logging.getLogger(__name__)


class SimpleROUGE:
    """Simple ROUGE implementation fallback when rouge-score is not available."""

    @staticmethod
    def rouge_n(reference: str, candidate: str, n: int = 1) -> Dict[str, float]:
        """Compute ROUGE-N score."""
        from collections import Counter

        def get_ngrams(text, n):
            tokens = text.lower().split()
            ngrams = []
            for i in range(len(tokens) - n + 1):
                ngrams.append(tuple(tokens[i:i+n]))
            return Counter(ngrams)

        ref_ngrams = get_ngrams(reference, n)
        cand_ngrams = get_ngrams(candidate, n)

        if not ref_ngrams or not cand_ngrams:
            return {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0}

        # overlap
        overlap = sum((ref_ngrams & cand_ngrams).values())

        # precision and recall
        precision = overlap / sum(cand_ngrams.values()) if cand_ngrams else 0.0
        recall = overlap / sum(ref_ngrams.values()) if ref_ngrams else 0.0

        # f1 score
        if precision + recall > 0:
            fmeasure = 2 * precision * recall / (precision + recall)
        else:
            fmeasure = 0.0

        return {
            'precision': precision,
            'recall': recall,
            'fmeasure': fmeasure
        }

    @staticmethod
    def rouge_l(reference: str, candidate: str) -> Dict[str, float]:
        """Compute ROUGE-L (longest common subsequence) score."""
        def lcs_length(x, y):
            m, n = len(x), len(y)
            dp = [[0] * (n + 1) for _ in range(m + 1)]

            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if x[i-1] == y[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])

            return dp[m][n]

        ref_tokens = reference.lower().split()
        cand_tokens = candidate.lower().split()

        if not ref_tokens or not cand_tokens:
            return {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0}

        lcs_len = lcs_length(ref_tokens, cand_tokens)

        precision = lcs_len / len(cand_tokens) if cand_tokens else 0.0
        recall = lcs_len / len(ref_tokens) if ref_tokens else 0.0

        if precision + recall > 0:
            fmeasure = 2 * precision * recall / (precision + recall)
        else:
            fmeasure = 0.0

        return {
            'precision': precision,
            'recall': recall,
            'fmeasure': fmeasure
        }


class ROUGECallback(BaseCallback, GenerationMixin):
    """ROUGE evaluation callback for summarization quality assessment."""

    def __init__(
        self,
        tokenizer,
        max_length: int = 128,
        eval_data: Optional[Any] = None,
        eval_collator: Optional[Any] = None,
        max_eval_samples: Optional[int] = 100,
        seed: int = 42,
        batch_size: int = 8,  # smaller batch for generation
        # generation parameters
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        num_beams: int = 4,
        repetition_penalty: float = 1.2,
        no_repeat_ngram_size: int = 2,
        # rouge-specific parameters
        rouge_types: Optional[List[str]] = None,  # ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
        use_stemmer: bool = False,  # whether to use porter stemmer
        # decoder start token (e.g. '<ko>' for mBART or tokenizer.bos_token for regular models)
        decoder_start_token: Optional[str] = None,
        # optional transform to compute encoder input from target on the fly
        # (e.g. han2han_tools.transcribe for transcription tasks)
        input_transform: Optional[callable] = None,
        # when True, temporarily disable eval_collator.use_task_prompts during
        # eval so the collator does NOT re-sample task prompts (which can pick
        # a direction inconsistent with the ROUGE eval direction). Caller is
        # responsible for pre-attaching coherent metadata to eval examples.
        freeze_metadata: bool = False,
        **kwargs
    ):
        """
        Initialize ROUGE callback.

        Args:
            tokenizer: HuggingFace tokenizer
            max_length: Maximum generation length
            eval_data: Evaluation dataset with 'input' and 'target' columns
            eval_collator: Optional data collator
            max_eval_samples: Maximum samples to evaluate
            seed: Random seed
            batch_size: Batch size for generation
            temperature: Generation temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            num_beams: Number of beams for beam search
            repetition_penalty: Repetition penalty (1.0 = no penalty)
            no_repeat_ngram_size: Block repeated n-grams
            rouge_types: List of ROUGE types to compute (default: ['rouge1', 'rouge2', 'rougeL'])
            use_stemmer: Whether to use stemmer for rouge calculation
            decoder_start_token: Decoder start token (defaults to `<ko>` for multilingual, tokenizer.bos_token for regular)
            input_transform: Callable that computes encoder input from target text.
                When provided, examples with only 'target' (no 'input') will have
                their encoder input computed as input_transform(target). Used for
                transcription eval where hangul input is derived from the original text.
            **kwargs: Additional parameters
        """
        if eval_collator is None:
            raise ValueError("eval_collator is required due to tokenizer special token limitations. "
                           "Cannot prepend script tokens via strings - must use collator.")

        # initialize mixins first
        GenerationMixin.__init__(
            self,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            **kwargs
        )

        self.input_transform = input_transform
        self.freeze_metadata = freeze_metadata

        # rouge configuration
        if rouge_types is None:
            # default to most common ROUGE metrics
            self.rouge_types = ['rouge1', 'rouge2', 'rougeL']
        else:
            self.rouge_types = rouge_types

        super().__init__(
            tokenizer=tokenizer,
            max_length=max_length,
            eval_data=eval_data,
            eval_collator=eval_collator,
            max_eval_samples=max_eval_samples,
            seed=seed,
            batch_size=batch_size,
            **kwargs
        )

        self.use_stemmer = use_stemmer

        # decoder start token (defaults to <ko> for multilingual, BOS for regular models)
        if decoder_start_token is None:
            # default to mBART multilingual format
            self.decoder_start_token = "<ko>"
            if "<ko>" in tokenizer.get_vocab():
                self.decoder_start_token_id = tokenizer.convert_tokens_to_ids("<ko>")
            else:
                self.decoder_start_token_id = tokenizer.bos_token_id
        else:
            self.decoder_start_token = decoder_start_token
            # convert to ID if it's a special token
            if decoder_start_token in tokenizer.get_vocab():
                self.decoder_start_token_id = tokenizer.convert_tokens_to_ids(decoder_start_token)
            else:
                # try as BOS token
                self.decoder_start_token_id = tokenizer.bos_token_id

        # debug logging for decoder start token
        log_from_main_process(logger, 'info',
            f"ROUGE decoder start token set to: '{self.decoder_start_token}' (ID: {self.decoder_start_token_id})")

        # check rouge availability
        self.use_rouge_score = ROUGE_AVAILABLE
        if self.use_rouge_score:
            self.scorer = rouge_scorer.RougeScorer(self.rouge_types, use_stemmer=self.use_stemmer)
        else:
            log_from_main_process(logger, 'warning', "rouge-score not available, using simple ROUGE implementation")
            self.scorer = None

        log_from_main_process(logger, 'info', f"Initialized ROUGECallback with {len(self.eval_examples)} examples, "
                   f"rouge_types={self.rouge_types}, "
                   f"rouge-score={'enabled' if self.use_rouge_score else 'disabled'}")

    def _initialize_metrics(self) -> Optional[nnx.MultiMetric]:
        """Initialize NNX metrics for ROUGE evaluation."""
        self.num_examples = 0

        # create metrics for each ROUGE type
        metric_dict = {
            'generation_length': nnx.metrics.Average('values'),
        }

        # add metrics for each rouge type
        for rouge_type in self.rouge_types:
            metric_dict[f'{rouge_type}_precision'] = nnx.metrics.Average('values')
            metric_dict[f'{rouge_type}_recall'] = nnx.metrics.Average('values')
            metric_dict[f'{rouge_type}_fmeasure'] = nnx.metrics.Average('values')

        return nnx.MultiMetric(**metric_dict)

    def _prepare_evaluation_data(self) -> None:
        """Prepare evaluation examples from dataset in collator-compatible format."""
        self.eval_examples = []

        if self.eval_data is not None:
            # sample evaluation data
            sampled_data = self._sample_evaluation_data(self.eval_data, self.max_eval_samples)

            if hasattr(sampled_data, 'to_dicts'):
                # polars DataFrame
                examples = sampled_data.to_dicts()
            else:
                # assume it's already a list
                examples = sampled_data

            for example in examples:
                # ensure collator-compatible format
                if 'original_text' in example and 'sentences' in example:
                    self.eval_examples.append(example)
                elif 'input' in example and 'target' in example:
                    self.eval_examples.append({
                        'original_text': example['input'],
                        'metadata': example.get('metadata', 'n/a'),
                        'sentences': [example['input']],
                        'source': example.get('source', 'unknown'),
                        'target': example['target'],
                    })
                elif 'source_text' in example and 'target_text' in example:
                    self.eval_examples.append({
                        'original_text': example['source_text'],
                        'metadata': example.get('metadata', 'n/a'),
                        'sentences': [example['source_text']],
                        'source': example.get('subsource', example.get('source', 'unknown')),
                        'target': example['target_text'],
                    })
                elif self.input_transform is not None and 'target' in example:
                    # compute encoder input on the fly (e.g. transcription:
                    # hangul in encoder, original hanja as ROUGE reference)
                    target_text = example['target']
                    input_text = self.input_transform(target_text)
                    self.eval_examples.append({
                        'original_text': input_text,
                        'metadata': example.get('metadata', ''),
                        'sentences': [input_text],
                        'source': example.get('source', 'unknown'),
                        'target': target_text,
                        'data_type': 'transcription',
                    })
                else:
                    log_from_main_process(logger, 'warning', f"Example missing required fields: {list(example.keys())}")

        # add synthetic examples if no data provided
        if not self.eval_examples:
            self.eval_examples = self._get_synthetic_examples()

    def _get_synthetic_examples(self) -> List[Dict[str, str]]:
        """Generate synthetic evaluation examples in collator-compatible format."""
        return [
            {
                'original_text': '한국의 수도는 서울입니다. 서울은 대한민국의 정치, 경제, 문화의 중심지입니다.',
                'metadata': '연도: 2025 시기: 현대 유형: N/A',
                'sentences': ['한국의 수도는 서울입니다. 서울은 대한민국의 정치, 경제, 문화의 중심지입니다.'],
                'source': 'synthetic',
                'target': '한국의 수도 서울은 정치, 경제, 문화의 중심지입니다.'
            },
            {
                'original_text': '인공지능 기술의 발전으로 우리의 삶이 크게 변화하고 있습니다. 특히 자연어 처리 분야에서 큰 진전이 있었습니다.',
                'metadata': '연도: 2025 시기: 현대 유형: N/A',
                'sentences': ['인공지능 기술의 발전으로 우리의 삶이 크게 변화하고 있습니다. 특히 자연어 처리 분야에서 큰 진전이 있었습니다.'],
                'source': 'synthetic',
                'target': '인공지능, 특히 자연어 처리 기술이 발전하고 있습니다.'
            },
        ]

    def _compute_rouge_scores(self, reference: str, candidate: str|None) -> Dict[str, Dict[str, float]]:
        """Compute ROUGE scores for a reference-candidate pair."""
        if candidate is None or not candidate.strip():
            # return zeros for all metrics
            return {
                rouge_type: {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0}
                for rouge_type in self.rouge_types
            }

        if self.use_rouge_score:
            try:
                # use rouge_scorer for accurate scoring
                scores = self.scorer.score(reference, candidate)
                return {
                    rouge_type: {
                        'precision': scores[rouge_type].precision,
                        'recall': scores[rouge_type].recall,
                        'fmeasure': scores[rouge_type].fmeasure
                    }
                    for rouge_type in self.rouge_types
                }
            except Exception as e:
                log_from_main_process(logger, 'warning', f"rouge_scorer failed, falling back to simple ROUGE: {e}")
                return self._compute_simple_rouge(reference, candidate)
        else:
            return self._compute_simple_rouge(reference, candidate)

    def _compute_simple_rouge(self, reference: str, candidate: str) -> Dict[str, Dict[str, float]]:
        """Compute ROUGE using simple fallback implementation."""
        results = {}

        for rouge_type in self.rouge_types:
            if rouge_type == 'rouge1':
                results[rouge_type] = SimpleROUGE.rouge_n(reference, candidate, n=1)
            elif rouge_type == 'rouge2':
                results[rouge_type] = SimpleROUGE.rouge_n(reference, candidate, n=2)
            elif rouge_type in ['rougeL', 'rougeLsum']:
                results[rouge_type] = SimpleROUGE.rouge_l(reference, candidate)
            else:
                log_from_main_process(logger, 'warning', f"unsupported rouge type: {rouge_type}")
                results[rouge_type] = {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0}

        return results

    def _batch_generate(self, model, examples: List[Dict], rngs: Optional[nnx.Rngs] = None,
                        use_metadata: bool = False) -> List[tuple[str, str, str]]:
        """Generate text for a batch of examples using collator for input preparation.

        Returns:
            List of tuples (encoder_input_decoded, generated_text, generated_text_with_special_tokens)
        """
        batch_size = len(examples)

        # apply collator to get properly formatted inputs with script tokens
        cooldown_phase = not use_metadata  # cooldown = no metadata
        collated = self.eval_collator(examples, cooldown_phase=cooldown_phase)

        # extract properly tokenized inputs as full batch arrays
        input_ids = jnp.array(collated['input_ids'])
        attention_mask = jnp.array(collated['attention_mask'])

        # pad encoder inputs to TPU-friendly length (multiple of 128)
        current_len = input_ids.shape[1]
        target_len = self._pad_to_tpu_length(current_len)
        if target_len > current_len:
            pad_len = target_len - current_len
            pad_token_id = self.tokenizer.pad_token_id or 0
            input_ids = jnp.pad(input_ids, ((0, 0), (0, pad_len)), constant_values=pad_token_id)
            attention_mask = jnp.pad(attention_mask, ((0, 0), (0, pad_len)), constant_values=0)

        # use our specified decoder_start_token_id for entire batch
        if self.decoder_start_token_id is not None:
            decoder_start_ids = jnp.full((batch_size, 1), self.decoder_start_token_id, dtype=jnp.int32)
        else:
            decoder_start_ids = jnp.array(collated['decoder_input_ids'])[:, :1]

        # generate for entire batch at once
        generated_ids = self.generate(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_start_ids,
            max_length=self.max_length,
            rngs=rngs
        )

        # decode each result
        generated_results = []
        for i in range(batch_size):
            # decode the actual encoder input (what the model sees)
            encoder_input_decoded = self.tokenizer.decode(
                np.array(collated['input_ids'][i]), skip_special_tokens=False
            )

            if isinstance(generated_ids, list) and generated_ids:
                seq = generated_ids[i] if i < len(generated_ids) else []
            elif hasattr(generated_ids, '__getitem__'):
                seq = generated_ids[i]
            else:
                seq = []

            if len(seq) > 0:
                generated_text = self.tokenizer.decode(seq, skip_special_tokens=True)
                generated_text_special = self.tokenizer.decode(seq, skip_special_tokens=False)
            else:
                generated_text = None
                generated_text_special = None

            generated_results.append((encoder_input_decoded, generated_text, generated_text_special))

        return generated_results

    def evaluate(self, model, step: int, rngs: Optional[nnx.Rngs] = None, use_metadata: bool = False) -> Dict[str, Any]:
        """
        Evaluate model using ROUGE scoring.

        Args:
            model: JAX/NNX model to evaluate
            step: Current training step
            rngs: Random number generators
            use_metadata: Whether to prepend metadata to inputs

        Returns:
            Dictionary with ROUGE scores and generation examples
        """
        results = {
            'step': step,
            'rouge_scores': [],
            'examples': [],
            'generation_stats': {}
        }

        log_from_main_process(logger, 'info', f"Evaluating ROUGE on {len(self.eval_examples)} examples...")

        # if requested, temporarily disable use_task_prompts on the eval
        # collator so it doesn't overwrite our pre-attached metadata with a
        # randomly-sampled task prompt during the supervised handler path
        prev_use_task_prompts = None
        if self.freeze_metadata and hasattr(self.eval_collator, 'use_task_prompts'):
            prev_use_task_prompts = self.eval_collator.use_task_prompts
            self.eval_collator.use_task_prompts = False

        try:
            # evaluate in batches
            num_batches = (
                len(self.eval_examples) + self.batch_size - 1
            ) // self.batch_size
            for batch_idx, i in enumerate(
                range(0, len(self.eval_examples), self.batch_size)
            ):
                batch_examples = self.eval_examples[i:i + self.batch_size]
                batch_input_chars = [
                    len(ex.get('original_text', '')) for ex in batch_examples
                ]
                log_from_main_process(
                    logger, 'info',
                    f"ROUGE batch {batch_idx + 1}/{num_batches} "
                    f"(examples {i}:{i + len(batch_examples)}, "
                    f"input_chars={batch_input_chars})"
                )

                # generate for batch
                generated_results = self._batch_generate(model, batch_examples, rngs, use_metadata)

                # compute ROUGE scores
                for example, (encoder_input, generated_text, generated_text_special) in zip(batch_examples, generated_results):
                    reference = example['target']
                    rouge_scores = self._compute_rouge_scores(reference, generated_text)

                    # update metrics for each rouge type
                    for rouge_type in self.rouge_types:
                        metric_name_p = f'{rouge_type}_precision'
                        metric_name_r = f'{rouge_type}_recall'
                        metric_name_f = f'{rouge_type}_fmeasure'

                        getattr(self.metrics, metric_name_p).update(values=rouge_scores[rouge_type]['precision'])
                        getattr(self.metrics, metric_name_r).update(values=rouge_scores[rouge_type]['recall'])
                        getattr(self.metrics, metric_name_f).update(values=rouge_scores[rouge_type]['fmeasure'])

                    self.metrics.generation_length.update(values=len(generated_text.split()) if generated_text else 0)
                    self.num_examples += 1

                    results['rouge_scores'].append(rouge_scores)
                    results['examples'].append({
                        'input': encoder_input,
                        'reference': reference,
                        'generated': generated_text,
                        'rouge_scores': rouge_scores,
                        'source': example['source']
                    })

                    if len(results['examples']) <= 3:
                        log_from_main_process(logger, 'info', f"Example {len(results['examples'])}:")
                        log_from_main_process(logger, 'info', f"  Encoder input: {encoder_input.replace('<pad>','')}")
                        log_from_main_process(logger, 'info', f"  Reference: {reference}")
                        log_from_main_process(logger, 'info', f"  Generated: {generated_text}")
                        log_from_main_process(logger, 'info', f"  Generated (special): {generated_text_special}")
                        for rouge_type in self.rouge_types:
                            log_from_main_process(
                                logger, 'info',
                                f"  {rouge_type}: P={rouge_scores[rouge_type]['precision']:.4f}, "
                                f"R={rouge_scores[rouge_type]['recall']:.4f}, "
                                f"F1={rouge_scores[rouge_type]['fmeasure']:.4f}"
                            )
        finally:
            if prev_use_task_prompts is not None:
                self.eval_collator.use_task_prompts = prev_use_task_prompts

        # compute final statistics
        if results['rouge_scores']:
            stats = {}

            for rouge_type in self.rouge_types:
                precisions = [s[rouge_type]['precision'] for s in results['rouge_scores']]
                recalls = [s[rouge_type]['recall'] for s in results['rouge_scores']]
                fmeasures = [s[rouge_type]['fmeasure'] for s in results['rouge_scores']]

                stats[f'{rouge_type}_precision'] = {
                    'mean': np.mean(precisions),
                    'std': np.std(precisions),
                    'min': min(precisions),
                    'max': max(precisions)
                }
                stats[f'{rouge_type}_recall'] = {
                    'mean': np.mean(recalls),
                    'std': np.std(recalls),
                    'min': min(recalls),
                    'max': max(recalls)
                }
                stats[f'{rouge_type}_fmeasure'] = {
                    'mean': np.mean(fmeasures),
                    'std': np.std(fmeasures),
                    'min': min(fmeasures),
                    'max': max(fmeasures)
                }

            stats['num_examples'] = len(results['rouge_scores'])
            results['generation_stats'] = stats

            log_from_main_process(logger, 'info', f"ROUGE Evaluation Results:")
            for rouge_type in self.rouge_types:
                log_from_main_process(
                    logger, 'info',
                    f"  {rouge_type} F1: {stats[f'{rouge_type}_fmeasure']['mean']:.4f} "
                    f"(±{stats[f'{rouge_type}_fmeasure']['std']:.4f})"
                )
            log_from_main_process(logger, 'info', f"  Examples: {len(results['rouge_scores'])}")

            # compute per-subsource ROUGE statistics (fmeasure per rouge type)
            subsource_scores = {}
            for example in results['examples']:
                subsource = example.get('source', 'unknown')
                if subsource not in subsource_scores:
                    subsource_scores[subsource] = {rouge_type: [] for rouge_type in self.rouge_types}
                for rouge_type in self.rouge_types:
                    subsource_scores[subsource][rouge_type].append(
                        example['rouge_scores'][rouge_type]['fmeasure']
                    )

            if len(subsource_scores) > 1:
                results['subsource_stats'] = {}
                log_from_main_process(logger, 'info', f"  Per-subsource ROUGE F1:")
                for subsource, per_type_scores in sorted(subsource_scores.items()):
                    any_type = self.rouge_types[0]
                    num = len(per_type_scores[any_type])
                    sub_stats = {'num_examples': num}
                    parts = []
                    for rouge_type in self.rouge_types:
                        scores = per_type_scores[rouge_type]
                        avg = float(np.mean(scores)) if scores else 0.0
                        sub_stats[f'{rouge_type}_fmeasure'] = avg
                        parts.append(f"{rouge_type}={avg:.4f}")
                    results['subsource_stats'][subsource] = sub_stats
                    log_from_main_process(
                        logger, 'info',
                        f"    {subsource}: {', '.join(parts)} (n={num})"
                    )

        return results


if __name__ == "__main__":
    # test the callback
    from transformers import AutoTokenizer
    import polars as pl
    from flax import nnx
    from han2han_config import Han2HanConfig
    from modeling_han2han_flax import FlaxHan2Han

    print("Testing ROUGECallback...")

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("./final_tokenizer")
    vocab_size = len(tokenizer)

    # create test data
    test_data = pl.DataFrame([
        {
            'input': '한국의 수도는 서울입니다. 서울은 대한민국의 정치, 경제, 문화의 중심지입니다.',
            'target': '한국의 수도 서울은 정치, 경제, 문화의 중심지입니다.',
            'source': 'test',
            'metadata': '년도: 2025 시기: 현대: 유형: n/a'
        },
        {
            'input': '인공지능 기술의 발전으로 우리의 삶이 크게 변화하고 있습니다.',
            'target': '인공지능이 우리 삶을 변화시키고 있습니다.',
            'source': 'test',
            'metadata': '년도: 2025 시기: 현대: 유형: n/a'
        }
    ])

    # create minimal model for testing
    config = Han2HanConfig(
        vocab_size=vocab_size,
        d_model=256,
        encoder_nlayer=2,
        decoder_nlayer=2,
        pad_token_id=tokenizer.pad_token_id,
        decoder_start_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    print("Creating test model...")
    model = FlaxHan2Han(
        config=config,
        sharding=(None, None),
        rngs=nnx.Rngs(params=42, dropout=43),
        dtype=jnp.float32,
        gradient_checkpointing=False,
    )

    # create callback
    callback = ROUGECallback(
        tokenizer=tokenizer,
        eval_data=test_data,
        max_length=64,
        max_eval_samples=2,
        temperature=1.0,
        num_beams=2,  # smaller for testing
        rouge_types=['rouge1', 'rouge2', 'rougeL']
    )

    print("Testing ROUGE evaluation...")
    results = callback(model, step=100, rngs=nnx.Rngs(params=42))

    print("\nResults:")
    if 'generation_stats' in results:
        for rouge_type in callback.rouge_types:
            print(f"{rouge_type} F1: {results['generation_stats'][f'{rouge_type}_fmeasure']['mean']:.4f}")
    print(f"Number of examples: {len(results['examples'])}")
    print(f"Metrics: {callback.compute_metrics()}")

    print("\nTest complete!")

