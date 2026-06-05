#!/usr/bin/env python3
# coding: utf-8
"""
Temporal year-prediction log-probability evaluation callback.

For each held-out document we score a fixed grid of decade candidates by
teacher-forcing each through the decoder and picking the highest log-prob.
Reports decade accuracy, adjacent-decade accuracy, year MAE, and softmax
expected-year MAE so the training run produces a learning curve for
temporal classification without any post-training fine-tuning.

The eval format mirrors `_collate_temporal_continuation` in
phase2_collator.py exactly:
    encoder: [text_ids] <mask> <script_token>
    decoder: <script_token> [YEAR_PREFIX] [year_str]
    labels:  [YEAR_PREFIX] [year_str] <script_token>

Eval-data shape (per benchmark, e.g. 'temporal_ko' / 'temporal_en'):
    {
        'encoder_input': str,         # full source text
        'candidates': List[str],      # decade-start year strings, e.g. "1920"
        'correct_idx': int,           # which candidate matches the true decade
        'category': str,              # e.g. "decade_1920"
        'benchmark': str,             # 'temporal_ko' / 'temporal_en'
        'true_year': int,             # for MAE computations
    }
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
from flax import nnx

from han2han_tools import has_hanja, transcribe
from logging_utils import log_from_main_process
from mc_logprob_callback import MCLogProbCallback
from task_prompts import sample_task_prompt

logger = logging.getLogger(__name__)


KO_SCRIPT_TOKEN_HANGUL = '<hangul>'
KO_SCRIPT_TOKEN_HANJA = '<hanja>'

# GMM Mosaic kernel requires the flattened-token "m" dimension to be a
# multiple of mt=512. m = local_tokens * top_k, so padding decoder seq to
# a multiple of 512 keeps m aligned for any batch size / top_k.
GMM_TILE = 512


class TemporalLogProbCallback(MCLogProbCallback):
    """Year-prediction log-prob evaluation over a fixed decade grid.

    Subclasses MCLogProbCallback to reuse `_get_jitted_fns` and `_score_examples`
    unchanged. Overrides only the per-example tokenization (which must mirror
    the temporal_continuation training format) and the metric aggregation
    (which extends accuracy with MAE / expected-year / adjacent-decade).
    """

    REQUIRED_FIELDS = {
        'encoder_input', 'candidates', 'correct_idx',
        'category', 'benchmark', 'true_year',
    }

    def _prepare_evaluation_data(self):
        """Validate and store pre-loaded benchmark data.

        Same shape as MC eval, with two extra required fields: 
        (drives the script token + YEAR_PREFIX) and 'true_year' (drives MAE).
        """
        self.eval_examples: Dict[str, List[Dict]] = {}

        if not self.mc_eval_data:
            log_from_main_process(logger, 'warning',
                "No temporal eval data provided, temporal_logprob callback will be a no-op")
            return

        for benchmark, examples in self.mc_eval_data.items():
            if not examples:
                log_from_main_process(logger, 'warning',
                    f"Empty temporal eval data for {benchmark}, skipping")
                continue

            first = examples[0]
            missing = self.REQUIRED_FIELDS - set(first.keys())
            if missing:
                raise ValueError(
                    f"Temporal eval data for '{benchmark}' missing required fields: {missing}. "
                    f"Available: {list(first.keys())}. "
                    f"Run prepare_temporal_eval_data.py to regenerate the parquet."
                )

            sampled = self._sample_evaluation_data(examples, self.max_eval_samples)
            self.eval_examples[benchmark] = (
                sampled if isinstance(sampled, list) else list(sampled)
            )

            log_from_main_process(logger, 'info',
                f"Temporal eval: {benchmark} = {len(self.eval_examples[benchmark])} examples")

    def _aligned_decoder_length(self) -> int:
        """Round max_decoder_length up to a multiple of GMM_TILE.

        MoE sort dispatch's ragged_dot requires flat_M (= local_tokens *
        top_k) to be divisible by mt=512. Padding the decoder sequence to a
        multiple of 512 makes this hold regardless of batch size and top_k.
        """
        d = max(GMM_TILE, int(self.max_decoder_length))
        return ((d + GMM_TILE - 1) // GMM_TILE) * GMM_TILE

    def _script_token_id(self, text: str) -> int:
        """Pick the encoder/decoder boundary script token for this script.

        Mirrors the logic in phase2_collator.py:491-513 so eval and
        training use the same script token per (script, text) pair.
        """
        tok = KO_SCRIPT_TOKEN_HANJA if has_hanja(text) else KO_SCRIPT_TOKEN_HANGUL
        return self.tokenizer.convert_tokens_to_ids(tok)

    def _should_add_prompt(self) -> bool:
        """True iff we should prepend a task prompt to the encoder this eval.

        Training adds metadata/prompts only when not in cooldown (see
        phase2_collator.py:518 and train_han2han.py:3149
        where `collator.cooldown_phase` is set). Eval must match so the model
        sees the same encoder format it was trained on at this point.
        """
        if not self.use_task_prompts:
            return False
        if self.eval_collator is None:
            return True
        return not getattr(self.eval_collator, 'cooldown_phase', False)

    def _tokenize_mc_example(self, example: Dict) -> Dict:
        """Build encoder + per-candidate decoder arrays for one temporal example.

        Encoder: optional [task_prompt] + [text_ids] + <mask> + <script_token>.
        Decoder: <script_token> + " {YEAR_PREFIX} {year_str}" per candidate.
        """
        text = example['encoder_input']
        candidates = example['candidates']

        # whole point of the temporal eval: measure temporal signal from
        # Hangul alone, not hanja-density shortcuts. Transcribe Korean text
        # to pure Hangul before encoding so mixed-script examples don't leak
        # date information via their hanja ratio.
        if has_hanja(text):
            text = transcribe(text)

        script_token_id = self._script_token_id(text)
        mask_id = self.tokenizer.mask_token_id
        pad_id = self.tokenizer.pad_token_id
        year_prefix = '연도:'

        # semantic fit for the eval task: "estimate when this was written".
        prompt_ids: List[int] = []
        if self._should_add_prompt():
            prompt_text, _ = sample_task_prompt(
                'temporal_classification'
            )
            prompt_ids = self.tokenizer(
                prompt_text + ' ', add_special_tokens=False,
            ).input_ids

        text_ids = self.tokenizer(text, add_special_tokens=False).input_ids

        # encoder: [prompt] [text] <mask> <script_token>; reserve 2 slots.
        enc_max = self.max_input_length
        text_budget = max(0, enc_max - 2 - len(prompt_ids))
        text_ids = text_ids[:text_budget]
        encoder_ids = prompt_ids + text_ids + [mask_id, script_token_id]
        attention_mask = [1] * len(encoder_ids)
        if len(encoder_ids) < enc_max:
            pad_n = enc_max - len(encoder_ids)
            encoder_ids = encoder_ids + [pad_id] * pad_n
            attention_mask = attention_mask + [0] * pad_n

        enc_arr = np.array(encoder_ids, dtype=np.int32)
        enc_mask_arr = np.array(attention_mask, dtype=np.int32)

        # decoder padded to multiple of GMM tile (512) so MoE flat_M stays aligned.
        all_decoder_input_ids = []
        all_decoder_attention_masks = []
        all_labels = []
        dec_max = self._aligned_decoder_length()

        for year_str in candidates:
            year_text = f" {year_prefix} {year_str}"
            year_ids = self.tokenizer(year_text, add_special_tokens=False).input_ids
            year_ids = year_ids[:dec_max - 1]

            decoder_input_ids = [script_token_id] + year_ids
            labels = year_ids + [script_token_id]
            dec_attention = [1] * len(decoder_input_ids)
            if len(decoder_input_ids) < dec_max:
                dpad = dec_max - len(decoder_input_ids)
                decoder_input_ids = decoder_input_ids + [pad_id] * dpad
                dec_attention = dec_attention + [0] * dpad
            if len(labels) < dec_max:
                lpad = dec_max - len(labels)
                labels = labels + [-100] * lpad

            all_decoder_input_ids.append(np.array(decoder_input_ids, dtype=np.int32))
            all_decoder_attention_masks.append(np.array(dec_attention, dtype=np.int32))
            all_labels.append(np.array(labels, dtype=np.int32))

        # _score_examples does ex['input_ids'][0] / ex['attention_mask'][0], so
        # the encoder arrays are stacked once per candidate (identical copies).
        n = len(candidates)
        return {
            'input_ids': np.stack([enc_arr] * n),
            'attention_mask': np.stack([enc_mask_arr] * n),
            'decoder_input_ids': np.stack(all_decoder_input_ids),
            'decoder_attention_mask': np.stack(all_decoder_attention_masks),
            'labels': np.stack(all_labels),
            'correct_idx': example['correct_idx'],
            'num_candidates': n,
        }

    @staticmethod
    def _decade_midpoints(candidates: List[str]) -> np.ndarray:
        """Decade-start year strings -> midpoints (e.g. "1920" -> 1925.0)."""
        return np.array([int(c) + 5 for c in candidates], dtype=np.float32)

    def evaluate(
        self,
        model: Any,
        step: int,
        rngs: Optional[nnx.Rngs] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run temporal log-prob evaluation on all benchmarks."""
        import jax

        results: Dict[str, Any] = {'step': step}
        model.eval()

        all_decade_accs: List[float] = []

        for benchmark, examples in self.eval_examples.items():
            jax.clear_caches()

            log_from_main_process(logger, 'info',
                f"Temporal scoring {benchmark} ({len(examples)} examples)...")

            tokenized = []
            true_years = []
            categories = []
            for ex in examples:
                tok = self._tokenize_mc_example(ex)
                tokenized.append(tok)
                true_years.append(int(ex['true_year']))
                categories.append(ex['category'])

            if not tokenized:
                log_from_main_process(logger, 'warning',
                    f"No tokenized examples for {benchmark}, skipping")
                continue

            if self.mesh is not None:
                with self.mesh:
                    predictions, all_logprobs = self._score_examples(model, tokenized)
            else:
                predictions, all_logprobs = self._score_examples(model, tokenized)

            # candidates are identical across examples in a benchmark (fixed grid),
            # so derive the decade midpoints once from any example.
            midpoints = self._decade_midpoints(examples[0]['candidates'])
            n_candidates = len(midpoints)

            true_years_arr = np.asarray(true_years, dtype=np.float32)
            preds_arr = np.asarray(predictions, dtype=np.int64)
            correct_idx_arr = np.asarray(
                [t['correct_idx'] for t in tokenized], dtype=np.int64,
            )

            # decade accuracy + adjacent-decade accuracy
            decade_correct = (preds_arr == correct_idx_arr)
            adjacent_correct = (np.abs(preds_arr - correct_idx_arr) <= 1)

            # year MAE from argmax decade midpoint
            pred_years = midpoints[preds_arr]
            year_mae = float(np.mean(np.abs(pred_years - true_years_arr)))

            # expected-year MAE from softmax over candidate log-probs
            log_probs = all_logprobs.astype(np.float64)
            log_probs = log_probs - log_probs.max(axis=1, keepdims=True)
            probs = np.exp(log_probs)
            probs = probs / probs.sum(axis=1, keepdims=True)
            expected_years = (probs * midpoints[None, :]).sum(axis=1)
            expected_year_mae = float(np.mean(np.abs(expected_years - true_years_arr)))

            decade_acc = float(decade_correct.mean())
            adjacent_acc = float(adjacent_correct.mean())

            results[f'{benchmark}/decade_accuracy'] = decade_acc
            results[f'{benchmark}/adjacent_decade_accuracy'] = adjacent_acc
            results[f'{benchmark}/year_mae'] = year_mae
            results[f'{benchmark}/expected_year_mae'] = expected_year_mae

            all_decade_accs.append(decade_acc)

            log_from_main_process(logger, 'info',
                f"{benchmark}: decade_acc={decade_acc:.4f} "
                f"adj_acc={adjacent_acc:.4f} mae={year_mae:.2f}yr "
                f"e_mae={expected_year_mae:.2f}yr "
                f"({n_candidates}-way over {len(tokenized)} examples)")

            # per-category breakdown (one row per decade bucket)
            cat_correct = defaultdict(int)
            cat_total = defaultdict(int)
            for is_correct, cat in zip(decade_correct.tolist(), categories):
                cat_total[cat] += 1
                if is_correct:
                    cat_correct[cat] += 1
            for cat in sorted(cat_total.keys()):
                t = cat_total[cat]
                c = cat_correct[cat]
                results[f'{benchmark}/{cat}/accuracy'] = (c / t) if t > 0 else 0.0

        if all_decade_accs:
            results['total_temporal_accuracy'] = (
                sum(all_decade_accs) / len(all_decade_accs)
            )

        model.train()
        return results
