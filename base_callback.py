#!/usr/bin/env python3
# coding: utf-8
"""
Base callback architecture for JAX/NNX training scripts.

Provides a clean, extensible base class for callbacks with common initialization
patterns and integration with NNX metrics for idiomatic JAX training.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import jax.numpy as jnp
import jax
from flax import nnx
from transformers import AutoTokenizer
from modeling_han2han_flax import FlaxHan2Han

from logging_utils import log_from_main_process

logger = logging.getLogger(__name__)


class BaseCallback(ABC):
    """Base class for training callbacks with common initialization patterns."""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_length: int = 128,
        eval_data: Optional[Any] = None,
        eval_collator: Optional[Any] = None,
        max_eval_samples: Optional[int] = None,
        seed: int = 42,
        batch_size: int = 32,
        mesh: Optional[Any] = None,
        **kwargs
    ):
        """
        Initialize base callback with common parameters.

        Args:
            tokenizer: HuggingFace tokenizer for text processing
            max_length: Maximum sequence length for evaluation
            eval_data: Optional evaluation dataset (polars DataFrame or similar)
            eval_collator: Optional data collator for batch processing
            max_eval_samples: Maximum number of samples to evaluate (None for all)
            seed: Random seed for reproducible evaluation
            batch_size: Batch size for evaluation
            mesh: Optional JAX mesh for SPMD multihost generation (None for single-host)
            **kwargs: Additional callback-specific parameters
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.eval_data = eval_data
        self.eval_collator = eval_collator
        self.max_eval_samples = max_eval_samples
        self.seed = seed
        self.batch_size = batch_size
        self.mesh = mesh

        # store additional parameters
        self.kwargs = kwargs

        # initialize NNX metrics if needed by subclass
        self.metrics = self._initialize_metrics()

        # prepare evaluation examples
        self._prepare_evaluation_data()

        log_from_main_process(logger, 'info', f"Initialized {self.__class__.__name__} with max_length={max_length}, "
                   f"max_eval_samples={max_eval_samples}, "
                   f"SPMD mode={'enabled' if mesh is not None else 'disabled'}")

    def _initialize_metrics(self) -> Optional[nnx.MultiMetric]:
        """
        Initialize NNX metrics for the callback.
        Override in subclasses to add specific metrics.
        
        Returns:
            MultiMetric object or None if no metrics needed
        """
        return None

    @abstractmethod
    def _prepare_evaluation_data(self) -> None:
        """
        Prepare evaluation data and examples.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def evaluate(self, model: Any, step: int, rngs: Optional[nnx.Rngs] = None, **kwargs) -> Dict[str, Any]:
        """
        Run evaluation and return results.
        Must be implemented by subclasses.

        Args:
            model: JAX/NNX model to evaluate
            step: Current training step
            rngs: Optional random number generators
            **kwargs: Additional evaluation parameters

        Returns:
            Dictionary containing evaluation results and metrics
        """
        pass

    def reset_metrics(self) -> None:
        """Reset all metrics to initial state."""
        if self.metrics is not None:
            self.metrics.reset()

    def update_metrics(self, **values) -> None:
        """Update metrics with new values."""
        if self.metrics is not None:
            self.metrics.update(**values)

    def compute_metrics(self) -> Dict[str, float]:
        """Compute final metric values."""
        if self.metrics is not None:
            return self.metrics.compute()
        return {}

    def _prepare_model_inputs(self, input_ids, attention_mask=None, rngs=None):
        """Helper to prepare model inputs with proper format."""
        model_inputs = {
            'input_ids': input_ids,
            'return_dict': True,
        }

        if attention_mask is not None:
            model_inputs['attention_mask'] = attention_mask

        if rngs is not None:
            model_inputs['rngs'] = rngs

        return model_inputs

    def _pad_to_tpu_length(self, length: int) -> int:
        """Pad length to multiple of 128 for TPU efficiency. Returns original length on non-TPU."""
        try:
            from jax.extend.backend import get_backend
            if get_backend().platform != 'tpu':
                return length
        except Exception:
            return length
        return ((length + 127) // 128) * 128

    def _sample_evaluation_data(self, data: Any, max_samples: Optional[int] = None) -> Any:
        """
        Sample evaluation data with stratified sampling if possible.

        Args:
            data: Input data (polars DataFrame, HuggingFace Dataset, or list)
            max_samples: Maximum number of samples to select

        Returns:
            Sampled data in the same format as input
        """
        if max_samples is None:
            return data

        if len(data) <= max_samples:
            return data

        # handle HuggingFace Dataset (has shuffle + select but sample() has different signature)
        try:
            from datasets import Dataset
            if isinstance(data, Dataset):
                # delete custom attributes that break DatasetInfo.copy() during shuffle
                if hasattr(data.info, 'source_type'):
                    delattr(data.info, 'source_type')
                if hasattr(data.info, 'data_type'):
                    delattr(data.info, 'data_type')
                return data.shuffle(seed=self.seed).select(range(max_samples))
        except ImportError:
            pass

        # handle polars DataFrame
        if hasattr(data, 'sample'):
            return data.sample(max_samples, seed=self.seed)

        # handle lists
        if isinstance(data, list):
            import random
            random.seed(self.seed)
            return random.sample(data, max_samples)

        return data

    def __call__(self, model: Any, step: int, rngs: Optional[nnx.Rngs] = None, **kwargs) -> Dict[str, Any]:
        """
        Main callback entry point.

        Args:
            model: JAX/NNX model to evaluate
            step: Current training step
            rngs: Optional random number generators
            **kwargs: Additional parameters passed to evaluate()

        Returns:
            Dictionary containing evaluation results
        """
        log_from_main_process(logger, 'info', f"Running {self.__class__.__name__} evaluation at step {step}")

        # set model to eval mode: sets deterministic 
        # attribute to True in all dropout modules
        model.eval()

        try:
            # disable packing for eval (cleaner per-doc metrics)
            # save all buffer states to restore after eval
            original_packing_state = getattr(self.eval_collator, 'enable_packing', False)
            saved_packed_cache = None
            saved_packed_index = None
            saved_example_buffer = None
            saved_sample_buffer = None
            saved_korean_buffers = None

            # save and override max_length settings for eval
            # collator may have max_length, max_encoder_length, max_decoder_length (phase 2)
            original_max_length = getattr(self.eval_collator, 'max_length', None)
            original_max_encoder_length = getattr(self.eval_collator, 'max_encoder_length', None)
            original_max_decoder_length = getattr(self.eval_collator, 'max_decoder_length', None)

            if hasattr(self, 'max_length'):
                eval_max = self.max_length
                if original_max_length is not None:
                    self.eval_collator.max_length = eval_max
                if original_max_encoder_length is not None:
                    self.eval_collator.max_encoder_length = eval_max
                if original_max_decoder_length is not None:
                    self.eval_collator.max_decoder_length = eval_max

            if hasattr(self.eval_collator, 'enable_packing'):
                self.eval_collator.enable_packing = False

                # save and clear packed collator's buffers
                if hasattr(self.eval_collator, 'packed_batch_cache'):
                    saved_packed_cache = self.eval_collator.packed_batch_cache.copy()
                    self.eval_collator.packed_batch_cache = []
                if hasattr(self.eval_collator, 'packed_batch_index'):
                    saved_packed_index = self.eval_collator.packed_batch_index
                    self.eval_collator.packed_batch_index = 0
                if hasattr(self.eval_collator, 'example_buffer'):
                    saved_example_buffer = self.eval_collator.example_buffer.copy()
                    self.eval_collator.example_buffer = []

                # save and clear parent collator's buffers
                if hasattr(self.eval_collator, 'sample_buffer'):
                    saved_sample_buffer = {src: buf.copy() for src, buf in self.eval_collator.sample_buffer.items()}
                    for source in self.eval_collator.sample_buffer:
                        self.eval_collator.sample_buffer[source] = []
                if hasattr(self.eval_collator, 'korean_sub_buffers'):
                    saved_korean_buffers = {src: buf.copy() for src, buf in self.eval_collator.korean_sub_buffers.items()}
                    for sub_source in self.eval_collator.korean_sub_buffers:
                        self.eval_collator.korean_sub_buffers[sub_source] = []

            # reset metrics before evaluation
            self.reset_metrics()

            # run evaluation
            results = self.evaluate(model, step, rngs, **kwargs)

            # add computed metrics to results
            if self.metrics is not None:
                metric_results = self.compute_metrics()
                results.update(metric_results)

            # restore all saved states
            if original_max_length is not None:
                self.eval_collator.max_length = original_max_length
            if original_max_encoder_length is not None:
                self.eval_collator.max_encoder_length = original_max_encoder_length
            if original_max_decoder_length is not None:
                self.eval_collator.max_decoder_length = original_max_decoder_length

            if hasattr(self.eval_collator, 'enable_packing'):
                self.eval_collator.enable_packing = original_packing_state
                if saved_packed_cache is not None:
                    self.eval_collator.packed_batch_cache = saved_packed_cache
                if saved_packed_index is not None:
                    self.eval_collator.packed_batch_index = saved_packed_index
                if saved_example_buffer is not None:
                    self.eval_collator.example_buffer = saved_example_buffer
                if saved_sample_buffer is not None:
                    self.eval_collator.sample_buffer = saved_sample_buffer
                if saved_korean_buffers is not None:
                    self.eval_collator.korean_sub_buffers = saved_korean_buffers

            return results

        except Exception as e:
            log_from_main_process(logger, 'error', f"Evaluation failed: {e}")
            import traceback
            log_from_main_process(logger, 'error', traceback.format_exc())
            # restore all saved states even on error
            if original_max_length is not None:
                self.eval_collator.max_length = original_max_length
            if original_max_encoder_length is not None:
                self.eval_collator.max_encoder_length = original_max_encoder_length
            if original_max_decoder_length is not None:
                self.eval_collator.max_decoder_length = original_max_decoder_length

            if hasattr(self.eval_collator, 'enable_packing'):
                self.eval_collator.enable_packing = original_packing_state
                if saved_packed_cache is not None:
                    self.eval_collator.packed_batch_cache = saved_packed_cache
                if saved_packed_index is not None:
                    self.eval_collator.packed_batch_index = saved_packed_index
                if saved_example_buffer is not None:
                    self.eval_collator.example_buffer = saved_example_buffer
                if saved_sample_buffer is not None:
                    self.eval_collator.sample_buffer = saved_sample_buffer
                if saved_korean_buffers is not None:
                    self.eval_collator.korean_sub_buffers = saved_korean_buffers
            return {}   # return empty results instead of crashing


class GenerationMixin:
    """Mixin for callbacks that need text generation capabilities."""

    def __init__(self, *args, **kwargs):
        # extract generation-specific parameters
        self.temperature = kwargs.pop('temperature', 1.0)
        self.top_k = kwargs.pop('top_k', 50)
        self.top_p = kwargs.pop('top_p', 0.95)
        self.num_beams = kwargs.pop('num_beams', 4)
        self.repetition_penalty = kwargs.pop('repetition_penalty', 1.2)
        self.no_repeat_ngram_size = kwargs.pop('no_repeat_ngram_size', 3)

    def generate(
        self,
        model: FlaxHan2Han,
        input_ids: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        decoder_input_ids: Optional[jnp.ndarray] = None,
        max_length: Optional[int] = None,
        rngs: Optional[nnx.Rngs] = None
    ) -> jnp.ndarray:
        """
        Generate text using model with SPMD-native multihost generation.

        Supports both single-host and multihost SPMD generation:
        - If self.mesh is None: single-host generation (CPU or single TPU)
        - If self.mesh is provided: SPMD multihost generation with data parallelism

        Args:
            model: Flax NNX model with generate method
            input_ids: Input token IDs (host-local array)
            attention_mask: Optional attention mask (host-local array)
            decoder_input_ids: Optional decoder input IDs (first token used as decoder_start_token_id)
            max_length: Maximum generation length (defaults to self.max_length)
            rngs: Optional random number generators

        Returns:
            Generated token IDs (host-local array)
        """
        from jax.sharding import NamedSharding, PartitionSpec as P
        from sharding_utils import get_data_partition_spec

        if max_length is None:
            max_length = self.max_length

        # extract decoder_start_token_id from decoder_input_ids if provided
        if decoder_input_ids is not None and len(decoder_input_ids.shape) > 0 and decoder_input_ids.shape[-1] > 0:
            decoder_start_token_id = int(decoder_input_ids[0, 0] if len(decoder_input_ids.shape) > 1 else decoder_input_ids[0])
            log_from_main_process(logger, 'debug', f"Using decoder_start_token_id from decoder_input_ids: {decoder_start_token_id} ({self.tokenizer.convert_ids_to_tokens(decoder_start_token_id)})")
        elif hasattr(self, 'decoder_start_token_id') and self.decoder_start_token_id is not None:
            decoder_start_token_id = self.decoder_start_token_id
            log_from_main_process(logger, 'debug', f"Using decoder_start_token_id from self: {decoder_start_token_id} ({self.tokenizer.convert_ids_to_tokens(decoder_start_token_id)})")
        else:
            # fallback to <hangul> for Han2Han models
            decoder_start_token_id = self.tokenizer.convert_tokens_to_ids("<hangul>")
            log_from_main_process(logger, 'debug', f"Using fallback decoder_start_token_id: {decoder_start_token_id} (<hangul>)")

        pad_token_id = self.tokenizer.pad_token_id

        # ensure model is in eval mode
        model.eval()

        # spmd multihost generation if mesh is provided
        if self.mesh is not None:
            log_from_main_process(logger, 'debug', "Using SPMD multihost generation")
            import numpy as np
            # convert jax arrays to numpy for make_array_from_process_local_data
            if hasattr(input_ids, 'device'):
                input_ids = np.asarray(input_ids)
            if attention_mask is not None and hasattr(attention_mask, 'device'):
                attention_mask = np.asarray(attention_mask)

            # get local device count for this host, not global mesh size
            local_device_count = jax.local_device_count()
            original_batch_size = input_ids.shape[0]

            # pad batch to a multiple of local_device_count for even sharding
            remainder = original_batch_size % local_device_count
            if remainder != 0:
                pad_size = local_device_count - remainder
                input_ids = np.pad(input_ids, ((0, pad_size), (0, 0)))
                if attention_mask is not None:
                    attention_mask = np.pad(attention_mask, ((0, pad_size), (0, 0)))

            # create global arrays from host-local inputs
            batch_spec = get_data_partition_spec()
            array_spec = P(batch_spec[0] if batch_spec else None, None)
            global_sharding = NamedSharding(self.mesh, array_spec)
            if jax.process_count() > 1:
                input_ids_global = jax.make_array_from_process_local_data(
                    global_sharding, input_ids
                )
                attention_mask_global = jax.make_array_from_process_local_data(
                    global_sharding, attention_mask
                ) if attention_mask is not None else None
            else:
                input_ids_global = jax.device_put(input_ids, global_sharding)
                attention_mask_global = jax.device_put(
                    attention_mask, global_sharding
                ) if attention_mask is not None else None

            with self.mesh:
                generated_ids_global = model.generate(
                    input_ids=input_ids_global,
                    attention_mask=attention_mask_global,
                    max_length=max_length,
                    temperature=self.temperature,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    num_beams=self.num_beams,
                    repetition_penalty=self.repetition_penalty,
                    no_repeat_ngram_size=self.no_repeat_ngram_size,
                    decoder_start_token_id=decoder_start_token_id,
                    pad_token_id=pad_token_id,
                    rngs=rngs,
                    local_batch_size=original_batch_size,
                )

            # gather results from addressable shards on this host
            local_shards = [jax.device_get(s.data) for s in generated_ids_global.addressable_shards]
            generated_ids = np.concatenate(local_shards, axis=0)

            # slice back to original batch size if we padded
            generated_ids = generated_ids[:original_batch_size]

        else:
            # single-host generation (original behavior)
            log_from_main_process(logger, 'debug', "Using single-host generation")
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                num_beams=self.num_beams,
                repetition_penalty=self.repetition_penalty,
                no_repeat_ngram_size=self.no_repeat_ngram_size,
                decoder_start_token_id=decoder_start_token_id,
                pad_token_id=pad_token_id,
                rngs=rngs,
                local_batch_size=input_ids.shape[0],
            )

        return generated_ids

        return results