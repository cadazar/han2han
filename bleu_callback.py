#!/usr/bin/env python3
# coding: utf-8
"""
BLEU evaluation callback for JAX/NNX training scripts.

Implements BLEU scoring for generation quality evaluation with support for
both translation and text generation tasks using sacrebleu.
"""

import logging
from typing import Dict, Any, Optional, List
import jax.numpy as jnp
from flax import nnx
import numpy as np

# bleu evaluation
try:
    import sacrebleu
    SACREBLEU_AVAILABLE = True
except ImportError:
    SACREBLEU_AVAILABLE = False
    print("sacrebleu not available, falling back to simple BLEU implementation")

from base_callback import BaseCallback, GenerationMixin
from logging_utils import log_from_main_process, log_from_all_processes

logger = logging.getLogger(__name__)


class SimpleBLEU:
    """Simple BLEU implementation fallback when sacrebleu is not available."""
    
    @staticmethod
    def sentence_bleu(reference: List[str], candidate: str, smooth=True) -> float:
        """Compute sentence-level BLEU score."""
        from collections import Counter
        import math
        
        def tokenize(text):
            return text.lower().split()
        
        ref_tokens = [tokenize(ref) for ref in reference]
        cand_tokens = tokenize(candidate)
        
        if not cand_tokens:
            return 0.0
            
        # compute precision for n-grams (n=1 to 4)
        precisions = []
        for n in range(1, 5):
            cand_ngrams = Counter()
            ref_ngrams = Counter()
            
            # candidate n-grams
            for i in range(len(cand_tokens) - n + 1):
                ngram = tuple(cand_tokens[i:i+n])
                cand_ngrams[ngram] += 1
            
            # reference n-grams (max count across all references)
            for ref in ref_tokens:
                ref_ngram_counts = Counter()
                for i in range(len(ref) - n + 1):
                    ngram = tuple(ref[i:i+n])
                    ref_ngram_counts[ngram] += 1
                
                for ngram, count in ref_ngram_counts.items():
                    ref_ngrams[ngram] = max(ref_ngrams.get(ngram, 0), count)
            
            # compute precision
            if cand_ngrams:
                matches = sum(min(cand_ngrams[ngram], ref_ngrams.get(ngram, 0)) 
                             for ngram in cand_ngrams)
                total = sum(cand_ngrams.values())
                
                if smooth and total == 0:
                    precision = 0.0
                elif smooth and matches == 0:
                    precision = 1.0 / (2 ** n)  # smoothing
                else:
                    precision = matches / total if total > 0 else 0.0
            else:
                precision = 0.0
                
            precisions.append(precision)
        
        # geometric mean of precisions
        if all(p > 0 for p in precisions):
            geo_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
        else:
            geo_mean = 0.0
        
        # brevity penalty
        best_ref_len = min(len(ref) for ref in ref_tokens) if ref_tokens else 1
        cand_len = len(cand_tokens)
        
        if cand_len >= best_ref_len:
            bp = 1.0
        else:
            bp = math.exp(1 - best_ref_len / cand_len) if cand_len > 0 else 0.0
        
        return bp * geo_mean


class BLEUCallback(BaseCallback, GenerationMixin):
    """BLEU evaluation callback for generation quality assessment."""

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
        # bleu-specific parameters
        bleu_tokenize: str = 'intl',  # sacrebleu tokenization method
        smooth_method: str = 'exp',   # smoothing for sentence-level BLEU
        use_effective_order: bool = True,
        # decoder start token (e.g. '<ko>' for mBART or tokenizer.bos_token for regular models)
        decoder_start_token: Optional[str] = None,
        # optional transform to compute encoder input from target on the fly
        # (e.g. han2han_tools.transcribe for transcription tasks)
        input_transform: Optional[callable] = None,
        # when True, temporarily disable eval_collator.use_task_prompts during
        # eval so the collator does NOT re-sample task prompts (which can pick
        # a direction inconsistent with the BLEU eval direction). Caller is
        # responsible for pre-attaching coherent metadata to eval examples.
        freeze_metadata: bool = False,
        **kwargs
    ):
        """
        Initialize BLEU callback.

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
            bleu_tokenize: Tokenization method for sacrebleu ('intl', 'zh', 'ko')
            smooth_method: Smoothing method for sentence BLEU
            use_effective_order: Use effective n-gram order
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

        # bleu configuration
        self.bleu_tokenize = bleu_tokenize
        self.smooth_method = smooth_method
        self.use_effective_order = use_effective_order

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
            f"BLEU decoder start token set to: '{self.decoder_start_token}' (ID: {self.decoder_start_token_id})")

        # check sacrebleu availability
        self.use_sacrebleu = SACREBLEU_AVAILABLE
        if not self.use_sacrebleu:
            log_from_main_process(logger, 'warning', "sacrebleu not available, using simple BLEU implementation")

        log_from_main_process(logger, 'info', f"Initialized BLEUCallback with {len(self.eval_examples)} examples, "
                   f"sacrebleu={'enabled' if self.use_sacrebleu else 'disabled'}")

    def _initialize_metrics(self) -> Optional[nnx.MultiMetric]:
        """Initialize NNX metrics for BLEU evaluation."""
        # Initialize simple counter since NNX doesn't have a count metric ㅋㅋㅋ
        self.num_examples = 0
        return nnx.MultiMetric(
            bleu_score=nnx.metrics.Average('values'),
            generation_length=nnx.metrics.Average('values'),
        )

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
                    # hangul in encoder, original hanja as BLEU reference)
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
                'original_text': '한국의 수도는 서울입니다.',
                'metadata': '연도: 2025 시기: 현대 유형: N/A',
                'sentences': ['한국의 수도는 서울입니다.'],
                'source': 'synthetic',
                'target': '한국의 수도는 서울입니다.'
            },
            {
                'original_text': '인공지능 기술의 발전으로 우리의 삶이 크게 변화하고 있습니다.',
                'metadata': '연도: 2025 시기: 현대 유형: N/A',
                'sentences': ['인공지능 기술의 발전으로 우리의 삶이 크게 변화하고 있습니다.'],
                'source': 'synthetic',
                'target': '인공지능 기술의 발전으로 우리의 삶이 크게 변화하고 있습니다.'
            },
            {
                'original_text': '한국의 전통 음식 중에서 김치가 가장 유명합니다.',
                'metadata': '연도: 2025 시기: 현대 유형: N/A',
                'sentences': ['한국의 전통 음식 중에서 김치가 가장 유명합니다.'],
                'source': 'synthetic',
                'target': '한국의 전통 음식 중에서 김치가 가장 유명합니다.'
            },
            {
                'original_text': '조선시대 선비들은 학문을 중시하고 수양에 힘썼습니다.',
                'metadata': '연도: 2025 시기: 현대 유형: N/A',
                'sentences': ['조선시대 선비들은 학문을 중시하고 수양에 힘썼습니다.'],
                'source': 'synthetic',
                'target': '조선시대 선비들은 학문을 중시하고 수양에 힘썼습니다.'
            }
        ]

    def _compute_bleu_score(self, reference: str, candidate: str|None) -> float:
        """Compute BLEU score for a reference-candidate pair."""
        if candidate is None:
            return 0.0

        if not candidate.strip():
            return 0.0

        if self.use_sacrebleu:
            try:
                # use sacrebleu for more accurate scoring
                bleu = sacrebleu.sentence_bleu(
                    candidate,
                    [reference],
                    tokenize=self.bleu_tokenize,
                    smooth_method=self.smooth_method,
                    use_effective_order=self.use_effective_order
                )
                return bleu.score / 100.0  # normalize to [0, 1]
            except Exception as e:
                log_from_main_process(logger, 'warning', f"sacrebleu failed, falling back to simple BLEU: {e}")
                return SimpleBLEU.sentence_bleu([reference], candidate)
        else:
            return SimpleBLEU.sentence_bleu([reference], candidate)

    def _batch_generate(self, model, examples: List[Dict], rngs: Optional[nnx.Rngs] = None,
                        use_metadata: bool = False) -> List[tuple[str, str]]:
        """Generate text for a batch of examples using collator for input preparation.

        Returns:
            List of tuples (generated_text, generated_text_with_special_tokens)
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
        Evaluate model using BLEU scoring.

        Args:
            model: JAX/NNX model to evaluate
            step: Current training step
            rngs: Random number generators
            use_metadata: Whether to prepend metadata to inputs

        Returns:
            Dictionary with BLEU scores and generation examples
        """
        results = {
            'step': step,
            'bleu_scores': [],
            'examples': [],
            'generation_stats': {}
        }

        log_from_main_process(logger, 'info', f"Evaluating BLEU on {len(self.eval_examples)} examples...")

        # if requested, temporarily disable use_task_prompts on the eval
        # collator so it doesn't overwrite our pre-attached metadata with a
        # randomly-sampled task prompt during the supervised handler path
        prev_use_task_prompts = None
        if self.freeze_metadata and hasattr(self.eval_collator, 'use_task_prompts'):
            prev_use_task_prompts = self.eval_collator.use_task_prompts
            self.eval_collator.use_task_prompts = False

        try:
            # evaluate in batches
            for i in range(0, len(self.eval_examples), self.batch_size):
                batch_examples = self.eval_examples[i:i + self.batch_size]

                # generate for batch
                generated_results = self._batch_generate(model, batch_examples, rngs, use_metadata)

                # compute BLEU scores
                for example, (encoder_input, generated_text, generated_text_special) in zip(batch_examples, generated_results):
                    reference = example['target']
                    bleu_score = self._compute_bleu_score(reference, generated_text)

                    self.metrics.bleu_score.update(values=bleu_score)
                    self.metrics.generation_length.update(values=len(generated_text.split()) if generated_text else 0)
                    self.num_examples += 1

                    results['bleu_scores'].append(bleu_score)
                    results['examples'].append({
                        'input': encoder_input,
                        'reference': reference,
                        'generated': generated_text,
                        'bleu_score': bleu_score,
                        'source': example['source']
                    })

                    if len(results['examples']) <= 3:
                        log_from_main_process(logger, 'info', f"Example {len(results['examples'])}:")
                        log_from_main_process(logger, 'info', f"  Encoder input: {encoder_input.replace('<pad>','')}")
                        log_from_main_process(logger, 'info', f"  Reference: {reference}")
                        log_from_main_process(logger, 'info', f"  Generated: {generated_text}")
                        log_from_main_process(logger, 'info', f"  Generated (special): {generated_text_special}")
                        log_from_main_process(logger, 'info', f"  BLEU: {bleu_score:.4f}")
        finally:
            if prev_use_task_prompts is not None:
                self.eval_collator.use_task_prompts = prev_use_task_prompts

        # compute final statistics
        if results['bleu_scores']:
            avg_bleu = np.mean(results['bleu_scores'])
            median_bleu = np.median(results['bleu_scores'])
            std_bleu = np.std(results['bleu_scores'])

            results['generation_stats'] = {
                'avg_bleu': avg_bleu,
                'median_bleu': median_bleu,
                'std_bleu': std_bleu,
                'min_bleu': min(results['bleu_scores']),
                'max_bleu': max(results['bleu_scores']),
                'num_examples': len(results['bleu_scores'])
            }

            log_from_main_process(logger, 'info', f"BLEU Evaluation Results:")
            log_from_main_process(logger, 'info', f"  Average BLEU: {avg_bleu:.4f}")
            log_from_main_process(logger, 'info', f"  Median BLEU: {median_bleu:.4f}")
            log_from_main_process(logger, 'info', f"  Std BLEU: {std_bleu:.4f}")
            log_from_main_process(logger, 'info', f"  Examples: {len(results['bleu_scores'])}")

            # compute per-subsource BLEU statistics
            subsource_scores = {}
            for example in results['examples']:
                subsource = example.get('source', 'unknown')
                if subsource not in subsource_scores:
                    subsource_scores[subsource] = []
                subsource_scores[subsource].append(example['bleu_score'])

            if len(subsource_scores) > 1:
                results['subsource_stats'] = {}
                log_from_main_process(logger, 'info', f"  Per-subsource BLEU:")
                for subsource, scores in sorted(subsource_scores.items()):
                    subsource_avg = np.mean(scores)
                    results['subsource_stats'][subsource] = {
                        'avg_bleu': subsource_avg,
                        'num_examples': len(scores),
                    }
                    log_from_main_process(
                        logger, 'info',
                        f"    {subsource}: {subsource_avg:.4f} (n={len(scores)})"
                    )

        return results


if __name__ == "__main__":
    # test the callback
    from transformers import AutoTokenizer
    import polars as pl
    from flax import nnx
    from han2han_config import Han2HanConfig
    from modeling_han2han_flax import FlaxHan2Han
    
    print("Testing BLEUCallback...")

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("./final_tokenizer")
    vocab_size = len(tokenizer)

    # create test data
    test_data = pl.DataFrame([
        {
            'input': '한국의 수도는',
            'target': '한국의 수도는 서울입니다.',
            'source': 'test',
            'metadata': '년도: 2025 시기: 현대: 유형: n/a'
        },
        {
            'input': '인공지능은',
            'target': '인공지능은 미래 기술의 핵심입니다.',
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
    callback = BLEUCallback(
        tokenizer=tokenizer,
        eval_data=test_data,
        max_length=64,
        max_eval_samples=2,
        temperature=1.0,
        num_beams=2  # smaller for testing
    )
    
    print("Testing BLEU evaluation...")
    results = callback(model, step=100, rngs=nnx.Rngs(params=42))
    
    print("\nResults:")
    print(f"Average BLEU: {results.get('avg_bleu', 'N/A')}")
    print(f"Number of examples: {len(results['examples'])}")
    print(f"Metrics: {callback.compute_metrics()}")
    
    print("\nTest complete!")