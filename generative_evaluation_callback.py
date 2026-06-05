#!/usr/bin/env python3
# coding: utf-8
"""
Generative evaluation callback for training integration.

Wraps GenerativeEvaluator for use as a training callback, enabling
text-to-text evaluation of KLUE tasks during training using the collator's datasets.
"""

import jax
from flax import nnx
from typing import Dict, List, Optional, Any, Literal
import logging

from base_callback import BaseCallback, GenerationMixin
from generative_eval import (
    GenerativeEvaluator,
    YNAT_LABELS,
    NLI_LABELS,
)
from logging_utils import log_from_main_process, log_from_all_processes

logger = logging.getLogger(__name__)

# map task names to data_type values used in DataSourceConfig
TASK_TO_DATA_TYPE = {
    'ynat': 'topic_classification',
    'nli': 'nli',
    'sts': 'sts',
    'temporal': 'temporal_classification',
    'instruction': 'instruction_following',
}


class GenerativeEvaluationCallback(BaseCallback, GenerationMixin):
    """Training callback for generative evaluation of KLUE tasks.

    Evaluates ynat (topic classification), nli (natural language inference),
    sts (semantic textual similarity), and temporal classification during training.

    Uses datasets from the collator to ensure format matches training exactly.
    """

    def __init__(
        self,
        tokenizer,
        collator=None,
        val_datasets: Optional[Dict] = None,
        val_source_configs: Optional[Dict] = None,
        tasks: List[Literal['ynat', 'nli', 'sts', 'temporal', 'instruction']] = None,
        max_eval_samples: int = 100,
        batch_size: int = 16,
        max_input_length: int = 256,
        max_output_length: int = 16,
        mesh: Optional[jax.sharding.Mesh] = None,
        use_task_prompts: bool = True,
        **kwargs
    ):
        """
        Args:
            tokenizer: Han2Han tokenizer
            collator: MultilingualCollator or UnifiedCollator with loaded datasets
            val_datasets: Raw validation datasets dict (deterministic, identical on all hosts).
                When provided, bypasses collator iterator for eval data.
            val_source_configs: Source configs for val_datasets (for data_type matching)
            tasks: List of tasks to evaluate (default: ['ynat', 'nli'])
            max_eval_samples: Max samples per task (0 = use full dataset)
            batch_size: Generation batch size
            max_input_length: Max encoder input length
            max_output_length: Max decoder output length
            mesh: JAX mesh for SPMD generation
            use_task_prompts: Whether to prepend task prompts
        """
        self.tasks = tasks or ['ynat', 'nli']
        self.collator = collator
        self.val_datasets = val_datasets
        self.val_source_configs = val_source_configs
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.use_task_prompts = use_task_prompts

        BaseCallback.__init__(
            self,
            tokenizer=tokenizer,
            max_length=max_output_length,
            max_eval_samples=max_eval_samples,
            batch_size=batch_size,
            mesh=mesh,
            **kwargs
        )

        GenerationMixin.__init__(
            self,
            temperature=0.0,
            top_k=1,
            top_p=1.0,
            num_beams=4,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
        )

        self.evaluators: Dict[str, GenerativeEvaluator] = {}

    def _prepare_evaluation_data(self) -> None:
        """Load evaluation datasets for each task.

        Prefers raw val_datasets (deterministic, full coverage) over collator
        iterator (random, capped at max_eval_samples). Raw datasets ensure
        identical eval data on all hosts in multi-host training.
        """
        self.eval_datasets: Dict[str, List[Dict]] = {}

        if self.val_datasets is not None:
            self._prepare_from_raw_datasets()
        elif self.collator is not None:
            self._prepare_from_collator()
        else:
            log_from_main_process(logger, 'warning',
                "No val_datasets or collator provided, skipping generative eval")

    def _prepare_from_raw_datasets(self) -> None:
        """Load eval data directly from raw validation datasets.

        Deterministic and identical across all hosts.
        """
        source_configs = self.val_source_configs or {}

        for task in self.tasks:
            data_type = TASK_TO_DATA_TYPE.get(task)
            if data_type is None:
                log_from_main_process(logger, 'warning',
                    f"Unknown task {task}, skipping")
                continue

            matching_names = set(
                name for name, config in source_configs.items()
                if config.data_type == data_type
            )

            if not matching_names:
                log_from_main_process(logger, 'info',
                    f"No sources found for task {task} (data_type={data_type}), skipping")
                continue

            samples = []
            for name in matching_names:
                ds = self.val_datasets.get(name)
                if ds is None:
                    continue
                for i in range(len(ds)):
                    ex = ds[i]
                    ex['_data_source'] = name
                    samples.append(ex)

            if samples:
                self.eval_datasets[task] = samples
                log_from_main_process(logger, 'info',
                    f"Loaded {len(samples)} samples for {task} from raw val_datasets")
            else:
                log_from_main_process(logger, 'info',
                    f"No samples found for {task}, skipping")

    def _prepare_from_collator(self) -> None:
        """Legacy path: collect eval data from collator iterator.

        Non-deterministic across hosts -- each host may get different samples.
        """
        source_configs = getattr(self.collator, 'source_configs', {})
        eval_data = getattr(self.collator, 'eval_data', None)

        if eval_data is None:
            log_from_main_process(logger, 'warning',
                "Collator has no eval_data iterator, skipping generative eval")
            return

        for task in self.tasks:
            data_type = TASK_TO_DATA_TYPE.get(task)
            if data_type is None:
                log_from_main_process(logger, 'warning',
                    f"Unknown task {task}, skipping")
                continue

            matching_sources = set(
                name for name, config in source_configs.items()
                if config.data_type == data_type
            )

            if not matching_sources:
                log_from_main_process(logger, 'info',
                    f"No sources found for task {task} (data_type={data_type}), skipping")
                continue

            samples = []
            max_attempts = self.max_eval_samples * 10
            attempts = 0
            while len(samples) < self.max_eval_samples and attempts < max_attempts:
                try:
                    example = next(eval_data)
                    source = example.get('_data_source') or example.get('source', '')
                    if source in matching_sources:
                        samples.append(example)
                except StopIteration:
                    log_from_main_process(logger, 'debug',
                        f"eval_data exhausted after {attempts} attempts for {task}")
                    break
                attempts += 1

            if samples:
                self.eval_datasets[task] = samples
                log_from_main_process(logger, 'info',
                    f"Loaded {len(samples)} samples for {task} from collator")
            else:
                log_from_main_process(logger, 'info',
                    f"No samples found for {task}, skipping")

    def _create_evaluator(self, model, task: str) -> GenerativeEvaluator:
        """Create or get cached evaluator for a task."""
        if task not in self.evaluators:
            dataset = self.eval_datasets.get(task)
            max_out = self.max_output_length
            if task == 'instruction':
                max_out = max(max_out, 128)
            self.evaluators[task] = GenerativeEvaluator(
                model=model,
                tokenizer=self.tokenizer,
                task=task,
                dataset=dataset,
                batch_size=self.batch_size,
                max_input_length=self.max_input_length,
                max_output_length=max_out,
                mesh=self.mesh,
                use_task_prompts=self.use_task_prompts,
                collator=self.collator,
            )
        else:
            self.evaluators[task].model = model
        return self.evaluators[task]

    def _resolve_token_id(self, token_str: str) -> Optional[int]:
        """Look up a token id from the tokenizer; None on miss or unk-mapping."""
        if not token_str:
            return None
        tid = self.tokenizer.convert_tokens_to_ids(token_str)
        unk = getattr(self.tokenizer, 'unk_token_id', None)
        if tid is None or tid == unk:
            return None
        return tid

    def evaluate(
        self,
        model: Any,
        step: int,
        rngs: Optional[nnx.Rngs] = None,
        decoder_start_token: Optional[str] = None,
        max_length: Optional[int] = None,
        parse_cot_answer: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Run generative evaluation on all configured tasks.

        Args:
            model: FlaxHan2Han model
            step: Current training step
            rngs: Random number generators
            decoder_start_token: Optional override for the decoder priming
                token (e.g. "<|think|>" for a CoT pass). When None the
                per-batch decoder_input_ids from the collator are used.
            max_length: Optional generation-length override for this call,
                used by the CoT pass to give reasoning room to land.
            parse_cot_answer: When True, treat every generation as
                {rationale}<|assistant|>{answer}<|end_of_turn|> and score the
                answer span only; the unparsed fallback is tracked separately.

        Returns:
            Dictionary of evaluation results for all tasks
        """
        results = {'step': step}

        model.eval()

        # resolve override token id once per call. None when token absent
        # from this tokenizer (e.g. unified pretraining vocab); the override
        # then silently no-ops, which is the right behavior since the
        # callback should still run direct-mode evaluation.
        override_token_id: Optional[int] = None
        if decoder_start_token is not None:
            override_token_id = self._resolve_token_id(decoder_start_token)
            if override_token_id is None:
                log_from_main_process(logger, 'warning',
                    f"decoder_start_token={decoder_start_token!r} not found in "
                    f"tokenizer; CoT decoder priming will fall back to the "
                    f"collator-provided start token.")

        for task in self.tasks:
            if task not in self.eval_datasets:
                log_from_main_process(logger, 'warning',
                    f"No dataset loaded for task {task}, skipping")
                continue

            log_from_main_process(logger, 'info',
                f"Running generative evaluation for {task}"
                f"{' (CoT)' if parse_cot_answer else ''}...")

            try:
                evaluator = self._create_evaluator(model, task)

                # apply per-call CoT overrides to the cached evaluator. these
                # are reset below so the next non-CoT call is unaffected.
                prev_override = evaluator.decoder_start_token_id_override
                prev_parse = evaluator.parse_cot_answer
                prev_max_length = evaluator.max_length

                evaluator.decoder_start_token_id_override = override_token_id
                evaluator.parse_cot_answer = parse_cot_answer
                if max_length is not None:
                    evaluator.max_length = max_length

                # 0 = "use full dataset" (val_datasets path); legacy collator
                # path still treats 0 as 0 -> falls back to evaluator default
                max_samples = self.max_eval_samples if self.max_eval_samples else None
                try:
                    task_results = evaluator.evaluate(
                        split='validation',
                        max_samples=max_samples
                    )
                finally:
                    evaluator.decoder_start_token_id_override = prev_override
                    evaluator.parse_cot_answer = prev_parse
                    evaluator.max_length = prev_max_length

                for key, value in task_results.items():
                    results[f'{task}/{key}'] = value

            except Exception as e:
                log_from_all_processes(logger, 'error',
                    f"Error evaluating {task}: {e}")
                import traceback
                traceback.print_exc()

        model.train()

        return results


def create_generative_eval_callback(
    tokenizer,
    collator=None,
    val_datasets: Optional[Dict] = None,
    val_source_configs: Optional[Dict] = None,
    tasks: List[str] = None,
    max_eval_samples: int = 100,
    batch_size: int = 16,
    mesh: Optional[jax.sharding.Mesh] = None,
    **kwargs
) -> GenerativeEvaluationCallback:
    """Factory function to create GenerativeEvaluationCallback.

    Args:
        tokenizer: Han2Han tokenizer
        collator: Collator with loaded datasets (legacy path, non-deterministic across hosts)
        val_datasets: Raw validation datasets dict from get_local_sft_datasets().
            When provided, eval uses these directly (deterministic, full coverage).
        val_source_configs: Source configs for val_datasets (for data_type matching)
        tasks: List of tasks to evaluate
        max_eval_samples: Max samples per task (0 = use full dataset). Applies
            to both the val_datasets path and the legacy collator path.
        batch_size: Generation batch size
        mesh: JAX mesh for SPMD generation

    Returns:
        Configured GenerativeEvaluationCallback
    """
    return GenerativeEvaluationCallback(
        tokenizer=tokenizer,
        collator=collator,
        val_datasets=val_datasets,
        val_source_configs=val_source_configs,
        tasks=tasks or ['ynat', 'nli'],
        max_eval_samples=max_eval_samples,
        batch_size=batch_size,
        mesh=mesh,
        **kwargs
    )


if __name__ == '__main__':
    print("GenerativeEvaluationCallback module loaded successfully")
    print(f"Available tasks: ynat, nli, sts, temporal")
    print(f"Task to data_type mapping: {TASK_TO_DATA_TYPE}")
    print(f"YNAT labels: {YNAT_LABELS}")
    print(f"NLI labels: {NLI_LABELS}")

