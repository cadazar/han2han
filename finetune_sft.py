#!/usr/bin/env python3
# coding: utf-8
"""
T5-style SFT fine-tuning for Han2Han using unified collator.

This script reuses setup_data_pipeline from the pretraining script to ensure
consistent collator setup. It provides:
- Parquet data via get_local_sft_datasets() (local or GCS with --data_bucket)
- UnifiedCollator for task routing (via setup_data_pipeline)
- GenerativeEvaluationCallback for eval
- Token-based training loop (not epoch-based)
"""

import logging
import sys
import os
from dataclasses import dataclass
from functools import partial

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    stream=sys.stdout,
    force=True
)

from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.WARNING)

# === JAX AND FLAX ===
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false' # avoid jax preallocating all memory
if os.getenv('ENABLE_JAX_DISTRIBUTED'):
    try:
        # jax and flax imports
        import jax
        jax.distributed.initialize()
        print("JAX distributed training initialized")
    except RuntimeError as e:
        import jax
        if "already been initialized" not in str(e):
            print(f"JAX distributed initialization failed: {e}")
else:
    import jax

import jax.numpy as jnp
from flax import nnx
from optax import softmax_cross_entropy_with_integer_labels
import numpy as np
import argparse
import base64
import pickle
import time
import yaml
from typing import Dict, Any, Callable, Optional

from transformers import AutoTokenizer

from modeling_han2han_flax import Han2HanConfig, FlaxHan2Han
from checkpoint_utils import (
    save_checkpoint,
    restore_checkpoint,
    prepare_metadata,
    setup_checkpoint_manager,
    load_config_from_checkpoint,
)
from sharding_utils import (
    setup_mesh_and_sharding,
    get_data_layout,
    shard_batch_to_devices,
    derive_param_sharding,
)
from logging_utils import log_from_main_process
from dynamic_data_loader import get_local_sft_datasets
from optimizer import (
    create_optimizer,
    create_learning_rate_schedule,
    patch_to_opt_state_for_factored_adafactor,
)
from train_han2han import (
    setup_data_pipeline,
    clip_and_norm,
)
from token_based_schedule import compute_lr_for_logging
from generative_evaluation_callback import create_generative_eval_callback
from bleu_callback import BLEUCallback
from rouge_callback import ROUGECallback
from han2han_tools import transcribe
from sft_methods import (
    ScheduledSamplingConfig,
    ContrastiveLearningConfig,
    scheduled_sampling_step,
    compute_contrastive_loss,
)

logger = logging.getLogger(__name__)


@dataclass
class SFTArgs:
    """Args namespace for setup_data_pipeline with SFT-appropriate defaults.

    This provides all fields required by setup_data_pipeline while disabling
    pretraining-specific features like denoising corruption.
    """
    # basic config
    smoke_test: bool = False
    seed: int = 42
    batch_size: int = 32
    eval_batch_size: int = 8
    eval_split_ratio: float = 0.05

    # sequence lengths
    sequence_length: int = 256
    max_encoder_length: int = 256
    max_decoder_length: int = 32

    # collator buffer
    collator_buffer_size: int = 500

    # task prompts
    use_task_prompts: bool = True

    # packing (disabled for SFT)
    enable_packing: bool = False
    packing_efficiency_threshold: float = 0.8
    packed_buffer_size: int = 1

    # denoising ratios (all 0 for pure SFT)
    mode_ratios: str = "0.0,0.0,0.0"
    morpheme_denoising_ratio: float = 0.0
    sentinel_denoising_ratio: float = 0.0
    byte_reconstruction_ratio: float = 0.0
    temporal_continuation_ratio: float = 0.0
    han2han_transcription_ratio: float = 0.0

    # denoising params (minimal values, not used in SFT)
    infilling_ratio: str = "0.0"
    heavy_infilling_ratio: str = "0.0"
    poisson_lambda: str = "3.0;3.0"
    morpheme_lambda: str = "2.0"
    sentence_permutation: bool = False
    use_phase2_collator: bool = False


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_batches(dataset_iter, batch_size: int, max_encoder_length: int, max_decoder_length: int,
                   collator=None, enable_packing: bool = False):
    """Batch individual examples from collator into proper batches.

    The collator yields individual examples, this function collects them
    and stacks into batches with proper padding. When enable_packing is True,
    expects and propagates packing keys (segment_ids/position_ids on both
    encoder and decoder side) and calls collator.create_packed_attention_masks
    to materialize 2D segment-aware attention masks.

    Args:
        dataset_iter: Iterator over individual examples from collator
        batch_size: Number of examples per batch
        max_encoder_length: Max encoder sequence length for padding
        max_decoder_length: Max decoder sequence length for padding (ignored
            for decoder fields when enable_packing=True, since the packed
            collator emits decoder packs at max_encoder_length)
        collator: Collator instance (required when enable_packing=True so we
            can build segment-aware 2D attention masks)
        enable_packing: When True, include packing-only keys and materialize
            2D masks via collator.create_packed_attention_masks

    Yields:
        Batched dict with stacked numpy arrays
    """
    base_keys = {'input_ids', 'decoder_input_ids', 'labels', 'attention_mask', 'decoder_attention_mask'}
    if enable_packing:
        if collator is None:
            raise ValueError("create_batches: enable_packing=True requires a collator instance")
        packing_keys = {'segment_ids', 'decoder_segment_ids', 'position_ids', 'decoder_position_ids'}
        expected_keys = base_keys | packing_keys
        # packed collator emits both encoder and decoder sides at self.max_length
        # (= max_encoder_length), so pad every field to max_encoder_length
        encoder_side_keys = expected_keys
    else:
        expected_keys = base_keys
        encoder_side_keys = {'input_ids', 'attention_mask'}

    batch_examples = []

    for example in dataset_iter:
        # remove metadata fields
        example.pop('_data_source', None)
        example.pop('_training_mode', None)

        # pad/truncate each field
        for key, value in list(example.items()):
            if key not in expected_keys:
                continue
            if isinstance(value, list):
                arr = np.array(value, dtype=np.int32)
            else:
                arr = np.asarray(value, dtype=np.int32)

            target_len = max_encoder_length if key in encoder_side_keys else max_decoder_length
            pad_value = -100 if key == 'labels' else 0

            if len(arr) < target_len:
                arr = np.pad(arr, (0, target_len - len(arr)), constant_values=pad_value)
            else:
                arr = arr[:target_len]
            example[key] = arr

        # filter to expected keys
        example = {k: v for k, v in example.items() if k in expected_keys}
        batch_examples.append(example)

        if len(batch_examples) >= batch_size:
            batch = {key: np.stack([ex[key] for ex in batch_examples], axis=0)
                     for key in batch_examples[0].keys()}

            if enable_packing:
                # materializes 2D (batch, 1, seq, seq) attention/decoder_attention masks
                # from segment_ids/decoder_segment_ids
                batch = collator.create_packed_attention_masks(batch)
                # collator may return jax arrays for the masks; force numpy for
                # downstream shard_batch_to_devices compatibility
                for k, v in batch.items():
                    if hasattr(v, 'device'):
                        batch[k] = np.asarray(v)

            yield batch
            batch_examples = []


@partial(
    nnx.jit,
    donate_argnums=(14,),
    static_argnums=(3, 7, 9, 10, 11, 12),
)
def sft_train_step(model: FlaxHan2Han, optimizer: nnx.Optimizer, dropout_rngs: nnx.Rngs,
                   lr_scheduler: Optional[Callable], learning_rate: float, warmup_ratio: float,
                   max_tokens: int, is_constant_schedule: bool, tokens_seen: int,
                   ss_config: Optional[ScheduledSamplingConfig],
                   cl_config: ContrastiveLearningConfig,
                   ss_decisions_from_train_pass: bool,
                   ss_differentiable_two_pass: bool,
                   grad_clipnorm: jnp.ndarray,
                   model_inputs: Dict[str, np.ndarray | jnp.ndarray]):
    """SFT train_step with scheduled sampling and contrastive learning.

    Mirrors the pretraining train_step.

    grad_clipnorm follows the train_han2han.py contract: a jnp scalar
    where 1e30 effectively disables external clipping (for optimizers that clip
    internally, e.g. muon's in-chain AdamW arm) while still surfacing global
    grad norm for monitoring.

    Forward-pass economy under SS+CL: the encoder is run at most once per
    train step (gold path) and its outputs are reused for the sampled pass via
    `precomputed_encoder_last_hidden_state`. When ss_decisions_from_train_pass
    is True, the eval-mode gold forward is skipped entirely and SS percentile
    decisions ride on the train-mode gold logits, removing one full forward.

    ss_differentiable_two_pass controls whether gradient flows from the
    sampled pass back into the gold pass through the SS soft embeddings.
    Only meaningful when ss_decisions_from_train_pass and
    ss_config.use_soft_embeddings are both True (gumbel_softmax is
    differentiable in logits; the predicted_ids path is argmax so no gradient
    flows regardless). Default False applies jax.lax.stop_gradient on
    gold_logits before SS -- the conservative non-differentiable two-pass.
    Mihaylova & Martins 2019 report differentiable two-pass as flaky to
    tune, but preliminary in-house experiments did not see degradation,
    so it's a knob worth A/B testing.
    """
    labels = model_inputs.pop("labels")

    # decisions-from-eval path: run an eval-mode gold forward to compute SS
    # percentile decisions outside the gradient graph. decisions-from-train
    # path skips this entirely; SS happens inside loss_fn against train-mode
    # gold logits (with stop_gradient).
    do_outer_eval_gold = (ss_config is not None and not ss_decisions_from_train_pass)
    if do_outer_eval_gold:
        model.eval()
        # deterministic=True is redundant with model.eval() but makes the
        # no-dropout guarantee unambiguous at the call site.
        initial_outputs = model(**model_inputs, rngs=None, deterministic=True)
        output_logits = initial_outputs.logits

        decoder_input_ids = model_inputs.get('decoder_input_ids')
        decoder_embeddings = model.decoder.wte

        modified_inputs, outer_ce_stats = scheduled_sampling_step(
            decoder_input_ids,
            output_logits,
            labels,
            decoder_embeddings,
            ss_config,
            dropout_rngs,
        )

        if ss_config.use_soft_embeddings:
            ss_model_inputs = dict(model_inputs)
            ss_model_inputs['decoder_input_embeddings'] = modified_inputs
            ss_model_inputs['decoder_input_ids'] = None
        else:
            ss_model_inputs = dict(model_inputs)
            ss_model_inputs['decoder_input_ids'] = modified_inputs

        model.train()
    else:
        ss_model_inputs = model_inputs
        outer_ce_stats = {}

    def loss_fn(model_local: FlaxHan2Han, rngs_local: nnx.Rngs):
        if ss_config is None:
            outputs = model_local(**model_inputs, rngs=rngs_local)
            logits = outputs.logits
            gold_logits = logits
            sampled_logits = logits
            ce_stats_local = {}
        else:
            # gold forward serves two possible consumers: contrastive loss
            # (uses gold_logits with grad) and ss_decisions_from_train_pass
            # (uses stop_gradient(gold_logits) for SS percentiles). Skip it
            # only when neither applies.
            need_train_gold = cl_config.enabled or ss_decisions_from_train_pass
            if need_train_gold:
                gold_outputs = model_local(**model_inputs, rngs=rngs_local)
                gold_logits = gold_outputs.logits
                cached_enc = gold_outputs.encoder_last_hidden_state
            else:
                gold_logits = None
                cached_enc = None

            if ss_decisions_from_train_pass:
                # SS decisions ride on train-mode gold logits. stop_gradient
                # is applied by default to keep the two-pass setup
                # non-differentiable (MM2019's conservative recipe). Set
                # ss_differentiable_two_pass to let gradient flow from the
                # sampled MLE back into gold params via the soft embeddings.
                gold_for_ss = (
                    gold_logits if ss_differentiable_two_pass
                    else jax.lax.stop_gradient(gold_logits)
                )
                inner_decoder_input_ids = model_inputs.get('decoder_input_ids')
                inner_decoder_embeddings = model_local.decoder.wte
                inner_modified_inputs, ce_stats_local = scheduled_sampling_step(
                    inner_decoder_input_ids,
                    gold_for_ss,
                    labels,
                    inner_decoder_embeddings,
                    ss_config,
                    rngs_local,
                )
                if ss_config.use_soft_embeddings:
                    ss_model_inputs_local = dict(model_inputs)
                    ss_model_inputs_local['decoder_input_embeddings'] = inner_modified_inputs
                    ss_model_inputs_local['decoder_input_ids'] = None
                else:
                    ss_model_inputs_local = dict(model_inputs)
                    ss_model_inputs_local['decoder_input_ids'] = inner_modified_inputs
            else:
                ss_model_inputs_local = ss_model_inputs
                ce_stats_local = outer_ce_stats

            # encoder reuse: when train-gold ran, its encoder hidden states
            # are identical to whatever the sampled pass would compute (same
            # encoder inputs, both train-mode), so skip the encoder forward.
            if cached_enc is not None:
                sampled_outputs = model_local(
                    **ss_model_inputs_local,
                    precomputed_encoder_last_hidden_state=cached_enc,
                    rngs=rngs_local,
                )
            else:
                sampled_outputs = model_local(**ss_model_inputs_local, rngs=rngs_local)
            sampled_logits = sampled_outputs.logits

        alpha = model_local.config.label_smoothing
        safe_labels = jnp.where(labels == -100, 0, labels)

        nll = softmax_cross_entropy_with_integer_labels(sampled_logits, safe_labels)
        if alpha > 0:
            lse = jax.nn.logsumexp(sampled_logits, axis=-1)
            uniform_nll = lse - jnp.mean(sampled_logits, axis=-1)
            mle_loss_val = (1 - alpha) * nll + alpha * uniform_nll
        else:
            mle_loss_val = nll

        weight_mask = (labels != -100).astype(jnp.float32)
        weighted_mle_loss = mle_loss_val * weight_mask
        total_mle_loss = weighted_mle_loss.sum()

        pad_token_id = model_local.config.pad_token_id
        encoder_mask = (model_inputs['input_ids'] != pad_token_id).astype(jnp.float32)
        encoder_tokens = encoder_mask.sum()
        decoder_tokens = weight_mask.sum()
        valid_tokens = encoder_tokens + decoder_tokens
        normalized_count = jnp.maximum(decoder_tokens, 1e-8)
        mle_loss = total_mle_loss / normalized_count

        if cl_config.enabled and ss_config is not None:
            contrastive_loss, cl_metrics = compute_contrastive_loss(
                gold_logits, sampled_logits, labels, weight_mask, cl_config
            )
            task_loss = mle_loss + cl_config.weight * contrastive_loss
        else:
            contrastive_loss = jnp.array(0.0)
            cl_metrics = {}
            task_loss = mle_loss

        total_loss = task_loss

        diagnostics = {
            "logits_max": jnp.max(sampled_logits),
            "logits_min": jnp.min(sampled_logits),
            "logits_mean": jnp.mean(sampled_logits),
            "logits_std": jnp.std(sampled_logits),
            "mle_loss": mle_loss,
            "cl_loss": contrastive_loss,
            "total_loss": total_loss,
            "valid_token_count": valid_tokens,
            **ce_stats_local,
            **cl_metrics,
        }

        return total_loss, diagnostics

    grad_fn = nnx.value_and_grad(
        loss_fn, has_aux=True, argnums=nnx.DiffState(0, optimizer.wrt)
    )
    (loss, diagnostics), grad = grad_fn(model, dropout_rngs)

    is_nan = jnp.isnan(loss)
    is_finite = jnp.logical_not(is_nan)
    grad_is_finite = jax.tree.reduce(
        lambda acc, x: jnp.logical_and(acc, jnp.all(jnp.isfinite(x))),
        grad,
        initializer=True,
    )
    safe_update = jnp.logical_and(is_finite, grad_is_finite)
    grad = jax.tree.map(
        lambda g: jnp.where(safe_update, g, jnp.zeros_like(g)),
        grad,
    )

    clipped_grad, grad_norm = clip_and_norm(grad, grad_clipnorm)

    progress = jnp.asarray(tokens_seen, dtype=jnp.float32) / max_tokens
    optimizer.update(model, clipped_grad, progress=progress)

    if lr_scheduler is not None:
        warmup_tokens = max_tokens * jnp.maximum(warmup_ratio, 1e-8)
        warmup_lr = learning_rate * (tokens_seen / warmup_tokens)
        if is_constant_schedule:
            current_lr = jnp.where(tokens_seen < warmup_tokens, warmup_lr, learning_rate)
        else:
            decay_progress = (tokens_seen - warmup_tokens) / (max_tokens - warmup_tokens)
            cosine_factor = 0.5 * (1 + jnp.cos(jnp.pi * jnp.clip(decay_progress, 0, 1)))
            decay_lr = learning_rate * (cosine_factor * 0.85 + 0.15)
            current_lr = jnp.where(tokens_seen < warmup_tokens, warmup_lr, decay_lr)
    else:
        current_lr = grad_norm / 1000.0

    metrics = {
        "loss": loss,
        "grad_norm": grad_norm,
        "learning_rate": current_lr,
        "all_grads_finite": grad_is_finite,
        **diagnostics,
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="T5-style SFT fine-tuning for Han2Han")

    # config file (optional - can override with CLI args)
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")

    # model arguments (not required - can come from config file)
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to pretrained model checkpoint")
    parser.add_argument("--config_source", type=str, default=None,
                        help="Optional checkpoint to load model architecture "
                             "config from when --model_path's metadata lacks "
                             "it (e.g. an older SFT checkpoint written before "
                             "the config-into-metadata fix). Defaults to "
                             "--model_path.")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="Path to tokenizer")
    parser.add_argument("--output_dir", type=str, default="sft_output",
                        help="Output directory for checkpoints")

    # checkpoint averaging (applied before training begins)
    parser.add_argument("--average_last_n", type=int, default=0,
                        help="Average the last N checkpoints before SFT (0 = disabled)")
    parser.add_argument("--average_steps", type=int, nargs='+', default=None,
                        help="Average specific checkpoint steps before SFT")
    parser.add_argument("--average_weights", type=float, nargs='+', default=None,
                        help="Per-checkpoint weights for averaging (normalized to sum=1)")
    parser.add_argument("--lookahead_alpha", type=float, default=0.0,
                        help="Momentum lookahead step size (0 = disabled). "
                             "Extrapolates w' = w + alpha * mu from latest checkpoint's "
                             "AdamW first moments. Applied after averaging if both are used.")

    # preemption recovery: when an in-progress SFT checkpoint exists under
    # output_dir/latest/checkpoints/, training auto-resumes from it (mirrors
    # train_han2han.py). these flags override that behavior.
    parser.add_argument("--skip_restore", action="store_true",
                        help="Force a fresh SFT start from --model_path even "
                             "if an in-progress checkpoint exists under "
                             "output_dir/latest/checkpoints/.")
    parser.add_argument("--restore_step", type=int, default=None,
                        help="Explicit step to restore from output_dir/latest/. "
                             "Default (None) picks the latest finalized step.")

    # data arguments
    parser.add_argument("--data_dir", type=str, default="task_data",
                        help="Directory containing task data (local or relative to data_bucket)")
    parser.add_argument("--data_bucket", type=str, default=None,
                        help="GCS bucket URI (e.g., gs://my-bucket). When set, reads data from "
                             "GCS instead of local disk.")
    parser.add_argument("--force_reload", type=bool, default=False,
                        help="Bypass the SFT per-host parquet cache and re-slice from source "
                             "parquets. Mirrors the pretraining flag in train_han2han.py.")
    parser.add_argument("--sft_tasks", type=str, default="all",
                        help="Tasks to train on (comma-separated or 'all')")
    parser.add_argument("--max_encoder_length", type=int, default=256,
                        help="Maximum encoder sequence length")
    parser.add_argument("--max_decoder_length", type=int, default=32,
                        help="Maximum decoder sequence length (short for SFT labels)")

    # SFT data mixing strategy
    parser.add_argument("--sft_sampling_strategy", type=str, default="manual",
                        choices=["manual", "temperature", "capped"],
                        help="How to weight SFT sources. 'manual' uses the "
                             "per-source weight= in create_local_sft_sources (legacy). "
                             "'temperature' uses examples-proportional with T smoothing "
                             "(T=1 natural, T->inf uniform). 'capped' is FLAN-style "
                             "examples-proportional capped at K per task.")
    parser.add_argument("--sft_sampling_temperature", type=float, default=2.0,
                        help="Temperature T for sft_sampling_strategy='temperature'")
    parser.add_argument("--sft_sampling_cap", type=int, default=None,
                        help="Explicit per-task cap K for "
                             "sft_sampling_strategy='capped'. Mutually exclusive "
                             "with sft_sampling_cap_multiplier.")
    parser.add_argument("--sft_sampling_cap_multiplier", type=float, default=None,
                        help="If set with sft_sampling_strategy='capped', auto K = "
                             "multiplier * min(N_i) across loaded tasks, where N_i "
                             "is measured before per-host slicing.")

    # training arguments
    parser.add_argument("--max_tokens", type=int, default=100_000_000,
                        help="Total tokens to train on")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size per device")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Peak learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--lr_schedule", type=str, default="cosine",
                        choices=["cosine", "constant", "linear", "rsqrt"],
                        help="Learning rate schedule type")
    parser.add_argument("--min_lr_ratio", type=float, default=0.15,
                        help="Minimum LR as ratio of peak for cosine schedule")
    parser.add_argument("--constant_ratio", type=float, default=0.0,
                        help="Fraction of training at constant peak LR before decay")
    parser.add_argument("--lr_cooldown_ratio", type=float, default=0.0,
                        help="Fraction of training for final linear LR cooldown (rsqrt only)")
    parser.add_argument("--lr_cooldown_type", type=str, default="linear",
                       choices=["linear", "sqrt"],
                       help="Cooldown shape: linear or sqrt (1-sqrt(x), fast initial decay)")
    parser.add_argument("--clipnorm", type=float, default=1.0,
                        help="Global gradient clip norm")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--freeze_embeddings", type=str, default=None, nargs='?', const='all',
                        help="Freeze embedding matrices. 'all' or True freezes wte,wce,wje. "
                             "Comma-separated list (e.g. 'wte,wje') freezes specific ones.")
    parser.add_argument("--freeze_pretrained_params", action="store_true",
                        help="Freeze pretrained params, only train cross-attention and embeddings")
    parser.add_argument("--label_smoothing", type=float, default=0.2,
                        help="Label smoothing alpha for cross-entropy loss (default: 0.2)")

    # optimizer selection
    parser.add_argument("--optimizer", type=str, default="muon",
                        choices=["muon"],
                        help="Optimizer type")

    # adafactor-specific settings
    parser.add_argument("--adafactor_beta2_cap", type=float, default=0.999,
                        help="Cap for second momentum decay rate")
    parser.add_argument("--adafactor_constant_beta2", action="store_true",
                        help="Use constant beta2 instead of variable 1-t^(-0.8)")
    parser.add_argument("--adafactor_momentum", type=float, default=0.0,
                        help="EMA momentum for Adafactor (0 = disabled)")
    parser.add_argument("--adafactor_burnin_steps", type=int, default=0,
                        help="Override constant_ratio for burn-in steps")
    parser.add_argument("--use_param_block_rms", action="store_true", default=True,
                        help="Enable param block RMS scaling (default: on)")
    parser.add_argument("--no_param_block_rms", dest="use_param_block_rms", action="store_false",
                        help="Disable param block RMS scaling")

    # trust ratio and fromage
    parser.add_argument("--use_trust_ratio", type=str, default="false",
                        choices=["false", "norm", "rms"],
                        help="LARS/LAMB-style trust ratio scaling")
    parser.add_argument("--trust_ratio_min_norm", type=float, default=1e-6,
                        help="Minimum norm for trust ratio computation")
    parser.add_argument("--use_fromage_style", type=str, default="false",
                        choices=["false", "norm", "rms"],
                        help="Fromage-style Pythagorean growth counteraction")

    # weight decay settings (tiered, 6-group)
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="Weight decay for attention params")
    parser.add_argument("--mlp_weight_decay", type=float, default=0.0,
                        help="Weight decay for MLP kernels")
    parser.add_argument("--embedding_weight_decay", type=float, default=0.0,
                        help="Weight decay for embeddings (wte/wce/wje)")
    parser.add_argument("--lm_head_weight_decay", type=float, default=0.0,
                        help="Weight decay for lm_head (when untied from embeddings)")
    parser.add_argument("--norm_weight_decay", type=float, default=0.0,
                        help="Weight decay for RMSNorm/LayerNorm scales")
    parser.add_argument("--bias_weight_decay", type=float, default=0.0,
                        help="Weight decay for bias parameters")
    parser.add_argument("--proportional_weight_decay", action="store_true",
                        help="Scale WD proportionally with LR")
    parser.add_argument("--adaptive_wd", action="store_true",
                        help="Use adaptive weight decay (LARS/LAMB-inspired param RMS scaling)")
    parser.add_argument("--wd_base", type=float, default=0.1,
                        help="Base weight decay for adaptive WD")
    parser.add_argument("--wd_min_value", type=float, default=1e-6,
                        help="Minimum weight decay value")
    parser.add_argument("--wd_max_value", type=float, default=1.0,
                        help="Maximum weight decay value")
    parser.add_argument("--wd_target_rms", type=float, default=1.0,
                        help="Target RMS for adaptive WD scaling")
    parser.add_argument("--wd_scale_metric", type=str, default="rms",
                        choices=["rms", "norm", "mean_abs", "max_abs"],
                        help="Metric for computing param scale in adaptive WD")
    parser.add_argument("--wd_scale_mult", type=float, default=0.01,
                        help="WD multiplier for norm scales")
    parser.add_argument("--wd_bias_mult", type=float, default=0.001,
                        help="WD multiplier for biases")
    parser.add_argument("--wd_warmup_scales", action="store_true",
                        help="Ramp scale/bias WD multipliers during LR warmup")
    parser.add_argument("--wd_warmup_shape", type=str, default="sigmoid",
                        choices=["sigmoid", "linear"],
                        help="Shape of WD warmup curve")

    parser.add_argument("--model_dtype", type=str, default="bfloat16",
                        choices=["float32", "bfloat16"],
                        help="Model dtype for optimizer accumulators")
    parser.add_argument("--optimizer_state_dtype", type=str, default="bfloat16",
                        choices=["float32", "bfloat16"],
                        help="Optimizer state dtype")
    parser.add_argument("--tie_input_output_embeddings", type=bool, default=False,
                        help="Whether input/output embeddings are tied (affects lm_head WD)")

    # scheduled sampling arguments (for improved generation). the schedule
    # itself is implicit (confidence percentile thresholds) -- Liu et al.
    # 2021's adaptive variant, not Mihaylova & Martins 2019's fixed schedule,
    # so no min-teacher-forcing / anneal-portion knobs are exposed.
    parser.add_argument("--use_scheduled_sampling", action="store_true",
                        help="Enable scheduled sampling for generation quality")
    parser.add_argument("--ss_temperature", type=float, default=1.0,
                        help="Temperature for scheduled sampling")
    parser.add_argument("--ss_mixing_method", type=str, default="gumbel",
                        choices=["gumbel", "softmax"],
                        help="Mixing method for scheduled sampling")
    parser.add_argument("--ss_threshold_mode", type=str, default="percentile",
                        choices=["percentile", "fixed"],
                        help="How the high/low-confidence boundaries are set. "
                             "'percentile' (default) uses per-batch quartiles "
                             "of the valid CE distribution -- stable bucket "
                             "sizes (25/50/25), meaning of 'confident' drifts. "
                             "'fixed' uses static thresholds from "
                             "--ss_high_conf_threshold / --ss_low_conf_threshold "
                             "-- stable semantics, bucket sizes drift with "
                             "model quality. Defaults of those flags match "
                             "Liu et al. 2021's -log(0.95) / -log(0.9) nats.")
    parser.add_argument("--ss_high_conf_threshold", type=float, default=0.05129329438755058,
                        help="High-confidence CE threshold (nats) for "
                             "--ss_threshold_mode=fixed. Tokens with CE below "
                             "this get replaced with random tokens. Default is "
                             "-log(0.95).")
    parser.add_argument("--ss_low_conf_threshold", type=float, default=0.10536051565782628,
                        help="Low-confidence CE threshold (nats) for "
                             "--ss_threshold_mode=fixed. Tokens with CE above "
                             "this get teacher forcing. Default is -log(0.9).")
    parser.add_argument("--ss_decisions_from_train_pass", action="store_true",
                        help="Drive SS percentile decisions from the train-mode "
                             "gold logits instead of a separate eval-mode forward. "
                             "Removes one full encoder+decoder forward per step "
                             "but rides on dropout-noisy logits.")
    parser.add_argument("--ss_differentiable_two_pass", action="store_true",
                        help="Let gradient flow from sampled MLE back into gold "
                             "params through the SS soft embeddings (i.e. drop "
                             "the stop_gradient on gold_logits before SS). Only "
                             "meaningful with --ss_decisions_from_train_pass and "
                             "use_soft_embeddings=True. Default is the conservative "
                             "non-differentiable two-pass.")

    # contrastive learning arguments (He et al. 2024)
    parser.add_argument("--use_contrastive_learning", action="store_true",
                        help="Enable contrastive learning")
    parser.add_argument("--cl_weight", type=float, default=0.1,
                        help="Weight for contrastive loss")
    parser.add_argument("--cl_margin", type=float, default=0.01,
                        help="Margin for contrastive loss")
    parser.add_argument("--cl_temperature", type=float, default=1.0,
                        help="Temperature for contrastive learning")

    # evaluation arguments
    parser.add_argument("--eval_every_tokens", type=int, default=1_000_000,
                        help="Evaluate every N tokens")
    parser.add_argument("--save_every_tokens", type=int, default=5_000_000,
                        help="Save checkpoint every N tokens")
    parser.add_argument("--save_best_metric", type=str, default=None,
                        help="Only keep best checkpoint by this metric (e.g. 'eval_loss', 'avg_bleu')")
    parser.add_argument("--save_best_mode", type=str, default="min",
                        choices=["min", "max"],
                        help="Whether best_metric should be minimized or maximized")
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--eval_tasks", type=str, default="temporal,sts,nli,ynat",
                        help="Tasks to evaluate on")
    parser.add_argument("--max_eval_samples", type=int, default=200,
                        help="Max samples per generative-eval task (0 = use the full validation set)")
    parser.add_argument("--eval_with_cot", type=lambda x: str(x).lower() in ("1", "true", "yes"),
                        default=False,
                        help="When True, run a second generative-eval pass with the decoder "
                             "primed on <|think|> and max_length=256 so the model can emit a "
                             "rationale before its answer. Scored separately under eval/cot/* "
                             "for direct-vs-CoT ablation.")

    # dropout settings (T5.1.1: zero dropout pretraining, higher for fine-tuning)
    parser.add_argument("--attndrop", type=float, default=0.1,
                        help="Attention dropout rate")
    parser.add_argument("--resdrop", type=float, default=0.1,
                        help="Residual connection dropout rate")
    parser.add_argument("--embddrop", type=float, default=0.1,
                       help="Embedding layer dropout rate (default: 0.1)")
    parser.add_argument("--embedding_dropout_rate", type=float, default=0.1,
                        help="Embedding module dropout rate")
    parser.add_argument("--layerdrop", type=float, default=0.0,
                        help="Layer dropout rate (skip entire layers)")
    parser.add_argument("--cross_attn_pdrop", type=float, default=0.1,
                        help="Cross-attention dropout rate")

    # misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float32", "bfloat16"])
    parser.add_argument("--use_task_prompts", action="store_true",
                        help="Use pretraining-style task prompts")
    parser.add_argument("--collator", type=str, default="unified",
                        choices=["unified", "chat"],
                        help="Which SFT collator to use. 'unified' (default): "
                             "script-tag boundaries (existing pretraining-era "
                             "format). 'chat': ChatML-style boundaries "
                             "(<|system|>, <|user|>, <|assistant|>, "
                             "<|end_of_turn|>, <|think|>). For Han2Han tokenizers, "
                             "regenerate with <|think|> via "
                             "prepare_multilingual_tokenizer.py. For the retired "
                             "Han2Han tokenizer, chat tokens are auto-aliased "
                             "onto sentinel slots via map_chat_tokens().")

    # wandb
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="han2han-sft")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    # get argparse defaults before parsing
    defaults = {action.dest: action.default for action in parser._actions if action.dest != 'help'}

    cli_args = parser.parse_args()

    # inject cuda flags for GPU/TPU
    os.environ['XLA_FLAGS'] = (
        '--xla_gpu_triton_gemm_any=true '
        '--xla_gpu_enable_latency_hiding_scheduler=true '
    )
    jax.config.update('jax_compilation_cache_dir', "/tmp/jax_cache")
    jax.config.update('jax2tf_associative_scan_reductions', True)       # also apparently speedy
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update(
        "jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir"
    )

    # backfill pretraining-only args (beta1, max_tokens, etc.) with their
    # pretraining-script defaults BEFORE the YAML merge, so they (a) exist as real
    # attributes on cli_args (no AttributeError downstream) and (b) become
    # overridable from this script's YAML config like any SFT-native arg.
    from train_han2han import get_config as _get_pretraining_config
    _pretrain_parser = _get_pretraining_config(return_parser_only=True)
    _backfilled = []
    for _action in _pretrain_parser._actions:
        if _action.dest == 'help' or hasattr(cli_args, _action.dest):
            continue
        setattr(cli_args, _action.dest, _action.default)
        defaults[_action.dest] = _action.default
        _backfilled.append(_action.dest)
    if _backfilled:
        logger.info(
            f"Backfilled {len(_backfilled)} pretraining-only args with their defaults; "
            f"these are now overridable via YAML/CLI."
        )

    # SFT-specific default override for parallelism strategy. The pretraining
    # parser defaults to "hybrid"; SFT yamls historically assumed "data_parallel",
    # so flip the backfilled default before the YAML merge. Mutating both
    # cli_args and defaults preserves the "yaml overrides only when current value
    # matches default" invariant, so a yaml `parallelism_strategy: fsdp`
    # still takes effect.
    if "parallelism_strategy" in _backfilled:
        cli_args.parallelism_strategy = "data_parallel"
        defaults["parallelism_strategy"] = "data_parallel"

    # load config file and merge with CLI args (yaml overrides defaults, CLI overrides yaml)
    if cli_args.config:
        config = load_config(cli_args.config)
        if config:
            for key, value in config.items():
                if hasattr(cli_args, key):
                    current_value = getattr(cli_args, key)
                    if current_value == defaults.get(key):
                        setattr(cli_args, key, value)

            # type coercion driven by parser action types (covers both SFT and
            # pretraining args). Handles YAML strings like "1e-4" that should be
            # floats, without needing hardcoded field lists.
            _all_actions = {
                a.dest: a
                for a in list(parser._actions) + list(_pretrain_parser._actions)
                if a.dest != 'help'
            }
            def _coerce(v, t):
                if t is float and not isinstance(v, float):
                    return float(v)
                if t is int and not isinstance(v, int):
                    return int(float(v))
                return v

            for key in config:
                action = _all_actions.get(key)
                if action is None or action.type not in (int, float):
                    continue
                value = getattr(cli_args, key, None)
                if value is None:
                    continue
                if action.nargs in ('+', '*') or isinstance(action.nargs, int):
                    setattr(cli_args, key, [_coerce(v, action.type) for v in value])
                else:
                    setattr(cli_args, key, _coerce(value, action.type))

            logger.info(f"Loaded configuration from {cli_args.config}")

    # validate required args after config merge
    if not cli_args.model_path:
        parser.error("--model_path is required (via CLI or config file)")
    if not cli_args.tokenizer_path:
        parser.error("--tokenizer_path is required (via CLI or config file)")

    # set seeds
    np.random.seed(cli_args.seed)

    # set dtype
    dtype = jnp.bfloat16 if cli_args.dtype == "bfloat16" else jnp.float32

    # wandb
    if cli_args.use_wandb and jax.process_index() == 0:
        try:
            import wandb
            wandb.init(
                project=cli_args.wandb_project,
                name=cli_args.wandb_run_name or f"sft-{time.strftime('%Y%m%d-%H%M%S')}",
                config=vars(cli_args)
            )
        except ImportError:
            logger.warning("wandb not available, disabling")
            cli_args.use_wandb = False

    # load tokenizer
    logger.info(f"Loading tokenizer from {cli_args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(cli_args.tokenizer_path)

    # han2han_tokenizer (retired pretraining tokenizer) doesn't have chat tokens
    # baked into its SPM model; alias them onto the first 5 sentinel slots so
    # existing checkpoints can be reused for IT experiments
    if cli_args.collator == "chat" and hasattr(tokenizer, "map_chat_tokens"):
        tokenizer.map_chat_tokens()

    # load model config from checkpoint metadata. Older SFT checkpoints
    # (pre-config-write fix) lack 'config' in their metadata -- fall back
    # to --config_source if provided so a stage-N -> stage-(N+1) resume
    # can borrow the architecture metadata from the original pretrain
    # checkpoint while still loading SFT weights from --model_path.
    config_source = cli_args.config_source or cli_args.model_path
    if config_source != cli_args.model_path:
        logger.info(
            f"Loading model config from --config_source={config_source} "
            f"(weights still come from --model_path={cli_args.model_path})"
        )
    model_config_dict = load_config_from_checkpoint(config_source)
    model_config = Han2HanConfig.from_dict(model_config_dict)

    # override dropout/label smoothing settings for fine-tuning
    model_config.attndrop = cli_args.attndrop
    model_config.resdrop = cli_args.resdrop
    model_config.embedding_dropout_rate = cli_args.embedding_dropout_rate
    model_config.layerdrop = cli_args.layerdrop
    model_config.cross_attn_pdrop = cli_args.cross_attn_pdrop
    model_config.embd_pdrop = cli_args.embddrop
    model_config.label_smoothing = cli_args.label_smoothing
    logger.info(f"Fine-tuning dropout/label smoothing settings: attndrop={cli_args.attndrop}, "
                f"resdrop={cli_args.resdrop}, embedding={cli_args.embedding_dropout_rate}, "
                f"layerdrop={cli_args.layerdrop}, cross_attn={cli_args.cross_attn_pdrop}, "
                f"label smoothing={cli_args.label_smoothing}")

    # setup mesh and sharding. parallelism_strategy / mesh_axes / mesh_shape are
    # backfilled from the pretraining parser above and overridable via yaml;
    # SFT default is data_parallel (see backfill block).
    mesh, _, parallelism_config = setup_mesh_and_sharding(cli_args)
    n_devices = jax.device_count()
    devices_per_host = n_devices // jax.process_count()

    # calculate batch sizes
    data_layout = get_data_layout(cli_args.batch_size, devices_per_host)
    global_batch_size = data_layout['global_batch_size']

    log_from_main_process(logger, 'info',
        f"Training with batch size {global_batch_size} per host, "
        f"{n_devices} total devices")

    # load SFT datasets (local or GCS)
    data_source = cli_args.data_bucket or cli_args.data_dir
    logger.info(f"Loading SFT data from {data_source}")
    datasets, sampling_ratios, source_configs = get_local_sft_datasets(
        base_dir=cli_args.data_dir,
        sft_tasks=cli_args.sft_tasks,
        split="train",
        data_bucket=cli_args.data_bucket,
        host_idx=jax.process_index(),
        num_hosts=jax.process_count(),
        sampling_strategy=cli_args.sft_sampling_strategy,
        sampling_temperature=cli_args.sft_sampling_temperature,
        sampling_cap=cli_args.sft_sampling_cap,
        sampling_cap_multiplier=cli_args.sft_sampling_cap_multiplier,
        force_reload=cli_args.force_reload,
    )
    logger.info(f"Loaded {len(datasets)} training datasets: {list(datasets.keys())}")
    logger.info(f"SFT sampling strategy: {cli_args.sft_sampling_strategy}")
    for name, ratio in sampling_ratios.items():
        logger.info(f"  {name}: {ratio:.2%}")

    # load validation datasets for eval -- decoupled from --sft_tasks so a
    # staged-curriculum run can still track the target tasks even when they
    # aren't in the current stage's training mix (e.g. stage 1 trains on
    # instruction_following only, but we want STS/NLI/YNAT/temporal eval
    # signal throughout). Map the short names used in --eval_tasks to the
    # data_type / source-name filter that get_local_sft_datasets expects;
    # unknown entries fall through untouched (the loader's filter accepts
    # either form).
    _EVAL_TASK_TO_DATA_TYPE = {
        'temporal': 'temporal_classification',
        'ynat': 'topic_classification',
    }
    eval_filter_tokens = [
        _EVAL_TASK_TO_DATA_TYPE.get(t.strip(), t.strip())
        for t in cli_args.eval_tasks.split(',')
        if t.strip()
    ]
    eval_filter = ','.join(eval_filter_tokens) if eval_filter_tokens else cli_args.sft_tasks
    logger.info(
        f"Loading validation data from {data_source} "
        f"(eval_filter='{eval_filter}', derived from --eval_tasks='{cli_args.eval_tasks}')"
    )
    val_datasets, val_ratios, val_source_configs = get_local_sft_datasets(
        base_dir=cli_args.data_dir,
        sft_tasks=eval_filter,
        split="validation",
        data_bucket=cli_args.data_bucket,
        force_reload=cli_args.force_reload,
    )
    if val_datasets:
        logger.info(f"Loaded {len(val_datasets)} validation datasets: {list(val_datasets.keys())}")
    else:
        logger.warning("No validation datasets found, eval will use training data split")

    # create SFTArgs for setup_data_pipeline (data pipeline fields only).
    # packing fields are backfilled onto cli_args from the pretraining argparse
    # and overridable from YAML; forward them here so setup_data_pipeline sees
    # them instead of falling back to SFTArgs defaults.
    sft_args = SFTArgs(
        seed=cli_args.seed,
        batch_size=global_batch_size,
        eval_batch_size=cli_args.eval_batch_size,
        sequence_length=cli_args.max_encoder_length,
        max_encoder_length=cli_args.max_encoder_length,
        max_decoder_length=cli_args.max_decoder_length,
        use_task_prompts=cli_args.use_task_prompts,
        enable_packing=cli_args.enable_packing,
        packing_efficiency_threshold=cli_args.packing_efficiency_threshold,
        packed_buffer_size=cli_args.packed_buffer_size,
    )
    if cli_args.enable_packing and cli_args.max_decoder_length != cli_args.max_encoder_length:
        logger.warning(
            "Packing is enabled but max_decoder_length (%d) != max_encoder_length (%d). "
            "The packed collator pads decoder packs to max_length (= encoder length), so "
            "the smaller max_decoder_length will truncate packed decoder sequences and drop "
            "most documents per pack. Set them equal in the YAML.",
            cli_args.max_decoder_length, cli_args.max_encoder_length,
        )

    # create training collator using setup_data_pipeline
    logger.info("Creating training collator via setup_data_pipeline...")
    collator = setup_data_pipeline(
        sft_args,
        tokenizer,
        cooldown_phase=not sft_args.use_task_prompts,
        streaming_datasets=datasets,
        sampling_ratios=sampling_ratios,
        source_configs=source_configs,
    )

    # opt-in: swap to ChatSFTCollator for ChatML-style encoder/decoder formatting.
    # ChatSFTCollator inherits from UnifiedCollator so all dataset/streaming
    # state survives the swap; only the per-example __call__ path changes.
    if cli_args.collator == "chat":
        from chat_sft_collator import ChatSFTCollator
        from unified_collator import UnifiedCollator
        if not isinstance(collator, UnifiedCollator):
            raise RuntimeError(
                f"Cannot swap collator to ChatSFTCollator: setup_data_pipeline "
                f"returned {type(collator).__name__}, expected UnifiedCollator. "
                f"ChatSFTCollator requires the production-pipeline path "
                f"(source_configs provided)."
            )
        logger.info("Swapping in ChatSFTCollator (chat-template format)")
        collator.__class__ = ChatSFTCollator
        collator._init_chat_token_ids()

    # create evaluation collator using validation datasets
    logger.info("Creating evaluation collator...")
    if val_datasets:
        # use dedicated validation split
        # NOTE: max_length_override controls ENCODER length, not decoder
        # decoder length for generation is handled by GenerativeEvaluationCallback
        eval_collator = setup_data_pipeline(
            sft_args,
            tokenizer,
            for_eval=True,
            streaming_datasets=val_datasets,
            sampling_ratios=val_ratios,
            source_configs=val_source_configs,
            cooldown_phase=not sft_args.use_task_prompts,
        )
    else:
        # fallback to training data split if no validation data
        eval_collator = setup_data_pipeline(
            sft_args,
            tokenizer,
            for_eval=True,
            eval_data=collator.eval_data,
            cooldown_phase=not sft_args.use_task_prompts,
        )

    if cli_args.collator == "chat":
        from chat_sft_collator import ChatSFTCollator
        from unified_collator import UnifiedCollator
        if not isinstance(eval_collator, UnifiedCollator):
            raise RuntimeError(
                f"Cannot swap eval collator to ChatSFTCollator: setup_data_pipeline "
                f"returned {type(eval_collator).__name__}, expected UnifiedCollator. "
                f"This usually means source_configs is missing on the eval path. "
                f"Provide validation_dataset_configs in your config."
            )
        eval_collator.__class__ = ChatSFTCollator
        eval_collator._init_chat_token_ids()

    # initialize model
    model_rngs = nnx.Rngs(params=cli_args.seed, dropout=cli_args.seed + 1)
    model_param_sharding = derive_param_sharding(cli_args.parallelism_strategy, parallelism_config)
    log_from_main_process(logger, 'info', f"Using model parameter sharding: {model_param_sharding}")
    with mesh:
        model = FlaxHan2Han(
            model_config,
            model_rngs,
            dtype=dtype,
            gradient_checkpointing=cli_args.remat_policy != "none",
            sharding=model_param_sharding,
            char_buckets=np.ones((model_config.vocab_size, 128)),
            jamo_buckets=np.ones((model_config.vocab_size, 128)),
        )

    # SFT chat-format token override: pretraining checkpoints serialize
    # </s>/<hangul> as eos/decoder_start. Under the chat collator the model
    # learns <|end_of_turn|> as turn closer and is primed with <|assistant|>,
    # so rebind config so generate() (and any reloaded SFT ckpt) inherits the
    # right ids without every call site needing to pass them explicitly.
    if cli_args.collator == "chat":
        model.config.eos_token_id = collator.end_of_turn_id
        model.config.decoder_start_token_id = collator.assistant_id
        model.config.sft_eos_token_id = collator.end_of_turn_id
        model.config.sft_decoder_start_token_id_default = collator.assistant_id
        model.config.sft_decoder_start_token_id_thinking = collator.think_id
        logger.info(
            f"Overrode model.config token ids for chat SFT: "
            f"eos={collator.end_of_turn_id} (<|end_of_turn|>), "
            f"decoder_start={collator.assistant_id} (<|assistant|>)"
        )

    # calculate training steps
    tokens_per_step = global_batch_size * (cli_args.max_encoder_length + cli_args.max_decoder_length)
    total_steps = cli_args.max_tokens // tokens_per_step
    warmup_steps = int(total_steps * cli_args.warmup_ratio)

    logger.info(f"Total steps: {total_steps}, warmup: {warmup_steps}")
    logger.info(f"Tokens per step: {tokens_per_step:,}")

    # setup checkpoint managers. two layouts under cli_args.output_dir:
    #   {output_dir}/best/checkpoints/   -- max_to_keep=1, gated on save_best_metric
    #   {output_dir}/latest/checkpoints/ -- max_to_keep=2, written every save event
    # 'latest' is the preemption-recovery target. max_to_keep=2 (not 1) protects
    # against the case where preemption strikes mid-save and corrupts the
    # newest checkpoint; orbax's latest_step() returns the newest *finalized*
    # step, so the previous one is the automatic fallback.
    import orbax.checkpoint as ocp
    pretrained_ckpt_manager = ocp.CheckpointManager(
        cli_args.model_path,
        options=ocp.CheckpointManagerOptions(max_to_keep=1, read_only=True),
        item_names=('model', 'optimizer_wd', 'optimizer_no_wd', 'meta')
    )
    use_gcs = str(cli_args.output_dir).startswith("gs://")
    best_metric = getattr(cli_args, 'save_best_metric', None)
    best_fn = (lambda m: m[best_metric]) if best_metric else None
    best_mode = getattr(cli_args, 'save_best_mode', 'min')

    def _mk_finetune_mgr(subdir: str, max_keep: int, _best_fn):
        suffix = f"/{subdir}"
        return setup_checkpoint_manager(
            output_dir=(cli_args.output_dir + suffix) if not use_gcs else None,
            gcs_output_dir=(cli_args.output_dir + suffix) if use_gcs else None,
            max_to_keep=max_keep,
            single_optimizer=True,
            best_fn=_best_fn,
            best_mode=best_mode,
            save_interval_steps=1,
        )

    finetune_ckpt_manager_best = _mk_finetune_mgr("best", 1, best_fn) if best_metric else None
    finetune_ckpt_manager_latest = _mk_finetune_mgr("latest", 2, None)

    # resume decision. mirrors train_han2han.py:5564-5582. tpu_monitor
    # relaunches the same CLI on preemption recovery, so the default path must
    # auto-resume when an in-progress checkpoint exists under 'latest'.
    if cli_args.skip_restore:
        restore_step = None
    elif cli_args.restore_step is not None:
        avail = sorted(finetune_ckpt_manager_latest.all_steps() or [])
        if cli_args.restore_step not in avail:
            raise ValueError(
                f"--restore_step {cli_args.restore_step} not found under "
                f"{cli_args.output_dir}/latest/checkpoints/. Available steps: {avail}"
            )
        restore_step = cli_args.restore_step
    else:
        restore_step = finetune_ckpt_manager_latest.latest_step()
    resuming = restore_step is not None
    if resuming:
        log_from_main_process(logger, 'info',
            f"Resuming SFT from {cli_args.output_dir}/latest/ step {restore_step} "
            f"(skipping --model_path pretrained restore)")
    else:
        log_from_main_process(logger, 'info',
            "Fresh SFT start (no in-progress checkpoint under output_dir/latest/)")

    # restore pretrained model (with optional checkpoint averaging). skipped when
    # resuming -- the in-progress finetune checkpoint already has the weights.
    avg_n = getattr(cli_args, 'average_last_n', 0) or 0
    avg_steps = getattr(cli_args, 'average_steps', None)
    avg_weights = getattr(cli_args, 'average_weights', None)
    do_averaging = (avg_n > 0 or avg_steps is not None) and not resuming

    if resuming:
        # no-op: pretrained restore is gated below. defined to keep downstream
        # references (steps_to_avg / pretrained_step in the lookahead block) safe.
        pass
    elif do_averaging:
        all_steps = sorted(pretrained_ckpt_manager.all_steps())
        if not all_steps:
            raise ValueError(f"No checkpoint steps found at {cli_args.model_path}")

        if avg_steps is not None:
            steps_to_avg = sorted(avg_steps)
            missing = [s for s in steps_to_avg if s not in all_steps]
            if missing:
                raise ValueError(f"Checkpoint steps not found: {missing}")
        else:
            steps_to_avg = all_steps[-avg_n:]

        if avg_weights is not None:
            if len(avg_weights) != len(steps_to_avg):
                raise ValueError(
                    f"Got {len(avg_weights)} weights for {len(steps_to_avg)} checkpoints"
                )
            w = jnp.array(avg_weights) / sum(avg_weights)
        else:
            w = jnp.array([1.0 / len(steps_to_avg)] * len(steps_to_avg))

        weights_str = ', '.join(f'{float(x):.3f}' for x in w)
        logger.info(f"Averaging {len(steps_to_avg)} checkpoints: {steps_to_avg}")
        logger.info(f"Weights: [{weights_str}]")

        accumulated = None
        for i, step in enumerate(steps_to_avg):
            logger.info(f"Loading checkpoint step {step} ({i+1}/{len(steps_to_avg)})...")
            restore_checkpoint(
                pretrained_ckpt_manager,
                model,
                optimizers=None,
                step=step,
                mesh=mesh,
                use_abstract_restoration=True,
                model_only=True,
            )
            state = nnx.state(model, nnx.Param)
            wi = float(w[i])
            if accumulated is None:
                accumulated = jax.tree.map(
                    lambda x: x.astype(jnp.float32) * wi, state
                )
            else:
                accumulated = jax.tree.map(
                    lambda acc, x: acc + x.astype(jnp.float32) * wi,
                    accumulated, state,
                )

        param_state = nnx.state(model, nnx.Param)
        final_state = jax.tree.map(
            lambda acc, orig: acc.astype(orig.dtype),
            accumulated, param_state,
        )
        nnx.update(model, final_state)
        logger.info("Checkpoint averaging complete")
        del accumulated, final_state
    else:
        pretrained_step = pretrained_ckpt_manager.latest_step()
        if pretrained_step is None:
            raise ValueError(f"No checkpoint found at {cli_args.model_path}")

        logger.info(f"Restoring pretrained model from step {pretrained_step}")
        restore_checkpoint(
            pretrained_ckpt_manager,
            model,
            optimizers=None,
            step=pretrained_step,
            mesh=mesh,
            use_abstract_restoration=True,
            model_only=True
        )

    # optional momentum lookahead: w' = w + alpha * mu. skipped when resuming
    # -- the finetune checkpoint already carries its own optimizer state.
    lookahead_alpha = getattr(cli_args, 'lookahead_alpha', 0.0)
    if lookahead_alpha > 0 and not resuming:
        lookahead_step = max(steps_to_avg) if do_averaging else pretrained_step
        logger.info(
            f"Applying momentum lookahead (alpha={lookahead_alpha}) "
            f"from optimizer state at step {lookahead_step}"
        )
        try:
            raw_restored = pretrained_ckpt_manager.restore(
                lookahead_step,
                args=ocp.args.Composite(
                    **{
                        name: ocp.args.StandardRestore()
                        for name in pretrained_ckpt_manager.item_names
                        if 'optimizer' in name
                    }
                ),
            )

            mu_leaves = {}
            for opt_name in raw_restored:
                flat, _ = jax.tree_util.tree_flatten_with_path(raw_restored[opt_name])
                for path, leaf in flat:
                    path_parts = [str(p) for p in path]
                    path_str = '.'.join(path_parts)
                    if '.mu.' in path_str:
                        param_key = path_str.split('.mu.', 1)[1]
                        mu_leaves[param_key] = leaf

            if not mu_leaves:
                logger.warning("No first moments (mu) found in optimizer state, skipping lookahead")
            else:
                logger.info(f"Found {len(mu_leaves)} first moment entries")
                model_state = nnx.state(model, nnx.Param)
                model_flat, model_treedef = jax.tree_util.tree_flatten_with_path(model_state)

                updated_leaves = []
                applied = 0
                for path, param in model_flat:
                    param_key = '.'.join(str(p) for p in path)
                    if param_key in mu_leaves:
                        mu = mu_leaves[param_key]
                        if hasattr(mu, 'shape') and mu.shape == param.shape:
                            new_val = param.astype(jnp.float32) + lookahead_alpha * jnp.asarray(mu).astype(jnp.float32)
                            updated_leaves.append(new_val.astype(param.dtype))
                            applied += 1
                        else:
                            updated_leaves.append(param)
                    else:
                        updated_leaves.append(param)

                logger.info(f"Lookahead applied to {applied}/{len(model_flat)} parameters")
                updated_state = model_treedef.unflatten(updated_leaves)
                nnx.update(model, updated_state)
                del mu_leaves, updated_state, updated_leaves

        except Exception as e:
            logger.warning(f"Lookahead failed (no optimizer state?): {e}")
            logger.warning("Continuing without lookahead")

    # pretrained manager is read-only and unused after this point; on resume
    # we never touched it anyway, but free it consistently in both paths.
    del pretrained_ckpt_manager

    # create learning rate schedule and optimizer (reusing pretraining infrastructure)
    lr_schedule = create_learning_rate_schedule(cli_args)
    is_constant_schedule = cli_args.lr_schedule == "constant"

    # build wrt_filter for parameter freezing
    frozen_embedding_filter = None
    if cli_args.freeze_embeddings:
        emb_spec = cli_args.freeze_embeddings
        if emb_spec in ('all', 'true', 'True'):
            frozen_names = ['wte', 'wce', 'wje']
        else:
            frozen_names = [n.strip() for n in emb_spec.split(',')]
        frozen_embedding_filter = nnx.filterlib.Any(
            *[nnx.filterlib.PathContains(name) for name in frozen_names]
        )
        log_from_main_process(logger, 'info',
            f"Embedding freeze enabled - freezing: {frozen_names}")

    if cli_args.freeze_pretrained_params:
        trainable_filter = nnx.filterlib.Any(
            nnx.filterlib.PathContains('crossattention'),
            nnx.filterlib.PathContains('wte'),
            nnx.filterlib.PathContains('subword_proj'),
            nnx.filterlib.PathContains('ln_cross_attn'),
            nnx.filterlib.PathContains('lm_head')
        )
        if frozen_embedding_filter is not None:
            trainable_filter = nnx.filterlib.All(
                trainable_filter,
                nnx.filterlib.Not(frozen_embedding_filter)
            )
        wrt_filter = nnx.filterlib.All(nnx.Param, trainable_filter)
        log_from_main_process(logger, 'info',
            "Parameter freezing enabled - only training crossattention, subword_proj, ln_cross_attn, lm_head"
            + (" (minus frozen embeddings)" if frozen_embedding_filter else ", wte"))
    elif frozen_embedding_filter is not None:
        wrt_filter = nnx.filterlib.All(
            nnx.Param,
            nnx.filterlib.Not(frozen_embedding_filter)
        )
    else:
        wrt_filter = None

    # Muon / Adafactor inject_hyperparams stash lower-rank state that inherits
    # the parent param's 3D sharding spec; patch to_opt_state to suppress eager
    # sharding for rank-mismatched leaves.
    opt_name = str(cli_args.optimizer).lower()
    needs_rank_mismatch_patch = (
        'adafactor' in opt_name
        or opt_name in ('factored_adam', 'muon')
    )
    with mesh:
        if needs_rank_mismatch_patch:
            with patch_to_opt_state_for_factored_adafactor():
                optimizer = create_optimizer(cli_args, lr_schedule, model, wrt_filter=wrt_filter)
        else:
            optimizer = create_optimizer(cli_args, lr_schedule, model, wrt_filter=wrt_filter)

    # preemption-recovery restore. mirrors train_han2han.py:5584-5634
    # (model+optimizer+metadata) and :5895-5914 (collator rng + iter_state).
    # restored_metadata starts as an empty dict so the loop-counter section
    # below can read from it unconditionally.
    restored_metadata: Dict[str, Any] = {}
    topology_changed = False
    if resuming:
        log_from_main_process(logger, 'info',
            f"Restoring SFT model + optimizer from latest/ step {restore_step}")
        with mesh:
            _restored_step, restored_metadata = restore_checkpoint(
                finetune_ckpt_manager_latest,
                model,
                optimizers=optimizer,
                step=restore_step,
                mesh=mesh,
                use_abstract_restoration=True,
                model_only=False,
            )
        restored_metadata = restored_metadata or {}
        if jax.process_count() > 1:
            from jax.experimental import multihost_utils
            multihost_utils.sync_global_devices("sft_ckpt_restored")

        # topology mismatch -> keep weights/optimizer but drop iter state / rng,
        # since per-host sharding of the dataset has changed.
        ckpt_num_hosts = restored_metadata.get('num_hosts', None)
        topology_changed = ckpt_num_hosts not in (None, jax.process_count())
        if topology_changed:
            log_from_main_process(logger, 'warning',
                f"Host count changed ({ckpt_num_hosts} -> {jax.process_count()}); "
                f"discarding collator rng + iter_state, keeping weights/optimizer only.")
        else:
            restored_rng_b64 = restored_metadata.get('gen_rng_b64')
            restored_iter_state = restored_metadata.get('iter_state', {}) or {}
            if restored_rng_b64:
                try:
                    collator.rng = pickle.loads(base64.b64decode(restored_rng_b64))
                    log_from_main_process(logger, 'info', "Restored collator RNG state")
                except Exception as e:
                    log_from_main_process(logger, 'warning',
                        f"Could not restore collator RNG state: {e}")
            sps = restored_iter_state.get('samples_per_source', {}) or {}
            rcps = restored_iter_state.get('raw_consumed_per_source', {}) or {}
            if (sps or rcps) and hasattr(collator, 'advance_iterators'):
                try:
                    collator.advance_iterators(sps, raw_consumed_per_source=rcps or None)
                except Exception as e:
                    log_from_main_process(logger, 'warning',
                        f"Could not advance iterators from checkpoint: {e}")

    # create scheduled sampling and contrastive learning configs
    if cli_args.use_scheduled_sampling:
        ss_config = ScheduledSamplingConfig(
            use_soft_embeddings=True,
            temperature=cli_args.ss_temperature,
            mixing_method=cli_args.ss_mixing_method,
            threshold_mode=cli_args.ss_threshold_mode,
            high_conf_threshold=cli_args.ss_high_conf_threshold,
            low_conf_threshold=cli_args.ss_low_conf_threshold,
        )
        logger.info(f"Scheduled sampling enabled: {ss_config}")
    else:
        ss_config = None

    if cli_args.use_contrastive_learning:
        cl_config = ContrastiveLearningConfig(
            enabled=True,
            margin=cli_args.cl_margin,
            weight=cli_args.cl_weight,
            temperature=cli_args.cl_temperature,
        )
        logger.info(f"Contrastive learning enabled: {cl_config}")
    else:
        cl_config = ContrastiveLearningConfig(enabled=False)

    # setup generative evaluation callback (KLUE NLU tasks)
    # pass raw val_datasets for deterministic, full-coverage eval across all hosts
    eval_tasks = cli_args.eval_tasks.split(',')
    eval_callback = create_generative_eval_callback(
        tokenizer=tokenizer,
        collator=eval_collator,
        val_datasets=val_datasets if val_datasets else None,
        val_source_configs=val_source_configs if val_datasets else None,
        tasks=eval_tasks,
        max_eval_samples=cli_args.max_eval_samples,
        batch_size=cli_args.eval_batch_size,
        mesh=mesh,
        max_input_length=cli_args.max_encoder_length,
    )

    # setup BLEU callback for transcription/translation tasks. gate on the
    # actual loaded val datasets, not the --sft_tasks string: 'all' implicitly
    # includes transcription but wouldn't substring-match against 'transcription'.
    bleu_callback = None
    has_transcription = bool(val_datasets) and any(
        'transcription' in name for name in val_datasets
    )

    if has_transcription:
        import polars as pl
        bleu_val_frames = []
        for ds_name, ds in val_datasets.items():
            if 'transcription' in ds_name:
                bleu_val_frames.append(ds.to_polars())

        if bleu_val_frames:
            bleu_val_df = pl.concat(bleu_val_frames)

            # BLEU eval is hardcoded to Hangul -> Hanja (encoder gets the
            # transcribed Hangul via input_transform, target is the original
            # Hanja). Pre-attach a Hangul -> Hanja system prompt to every row
            # so the chat collator's <|system|> content matches the direction
            # being evaluated; without this, per-example metadata picked
            # at data-prep time can advertise the opposite direction.
            if cli_args.use_task_prompts:
                from task_prompts import sample_task_prompt
                # one seeded prompt for the whole eval slice; see the
                # parallel comment in the ROUGE setup below for why per-row
                # sampling was unsafe across SPMD hosts.
                prompt, _ = sample_task_prompt(
                    'transcription_hangul_to_hanja',
                    seed=cli_args.seed,
                )
                bleu_val_df = bleu_val_df.with_columns(
                    pl.lit(prompt).alias('metadata')
                )

            bleu_callback = BLEUCallback(
                tokenizer=tokenizer,
                max_length=min(cli_args.max_decoder_length, 512),
                max_eval_samples=30,
                batch_size=2,
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                repetition_penalty=1.0,
                no_repeat_ngram_size=3,
                num_beams=4,
                seed=cli_args.seed,
                eval_data=bleu_val_df,
                eval_collator=eval_collator,
                # chat collator primes the decoder with <|assistant|>; the
                # pretraining-style unified collator uses <hanja> for the
                # Hangul -> Hanja (+ Hangul mixed-script) direction
                decoder_start_token=(
                    "<|assistant|>" if cli_args.collator == "chat" else "<hanja>"
                ),
                bleu_tokenize='intl',
                input_transform=transcribe,
                # the eval collator's use_task_prompts (when on) re-samples a
                # random direction per example, which can advertise a
                # direction inconsistent with BLEU's hardcoded Hangul -> Hanja
                # eval. Freeze metadata so our pre-attached prompts above are
                # used as-is.
                freeze_metadata=cli_args.use_task_prompts,
                mesh=mesh,
            )
            logger.info(f"BLEU callback initialized for transcription ({len(bleu_val_df)} val examples)")

    # setup ROUGE callback for AIHub summarization. mirrors the BLEU gate:
    # detect by val_datasets, not the --sft_tasks string.
    rouge_callback = None
    has_summarization = bool(val_datasets) and any(
        'summarization' in name for name in val_datasets
    )

    if has_summarization:
        import polars as pl
        rouge_val_frames = []
        for ds_name, ds in val_datasets.items():
            if 'summarization' in ds_name:
                rouge_val_frames.append(ds.to_polars())

        if rouge_val_frames:
            rouge_val_df = pl.concat(rouge_val_frames)

            # AIHub rows have text/summary/source/category/doc_id/task. ROUGE's
            # _prepare_evaluation_data picks up the source_text/target_text
            # branch, so rename text->source_text and summary->target_text.
            # Pre-attach a sampled summarization system prompt as metadata so
            # the chat collator's <|system|> slot is well-formed; freeze it
            # during eval to prevent per-example re-sampling. Per-subsource
            # ROUGE groups by the AIHub domain ('source' column) automatically
            # because that branch falls back to 'source' when 'subsource' is
            # absent.
            rouge_val_df = rouge_val_df.rename(
                {'text': 'source_text', 'summary': 'target_text'}
            )
            # defensive char-length filter: AIHub papers/patents can blow
            # past max_encoder_length and the intermediate tokenizer call
            # builds a large Python list before the chat collator truncates.
            # Korean tokenization runs ~1-3 chars/token; bound at 3x the
            # encoder window to keep outliers out of the eval slice.
            char_budget = cli_args.max_encoder_length * 3
            n_before = len(rouge_val_df)
            rouge_val_df = rouge_val_df.filter(
                pl.col('source_text').str.len_chars() <= char_budget
            )
            n_dropped = n_before - len(rouge_val_df)
            if n_dropped > 0:
                logger.info(
                    f"ROUGE eval: dropped {n_dropped}/{n_before} rows whose "
                    f"document exceeded {char_budget} chars "
                    f"(approx > {cli_args.max_encoder_length} tokens)"
                )
            if cli_args.use_task_prompts:
                from task_prompts import sample_task_prompt
                # one seeded prompt for the whole eval slice. per-row
                # sampling uses the global random module which is NOT
                # seeded consistently across SPMD hosts, so each worker
                # gets different system text. The eval rows then diverge
                # across the mesh, generation lengths drift, and the JAX
                # coordinator times out -> silent kill on every worker.
                prompt, _ = sample_task_prompt(
                    'summarization',
                    seed=cli_args.seed,
                )
                rouge_val_df = rouge_val_df.with_columns(
                    pl.lit(prompt).alias('metadata')
                )

            rouge_callback = ROUGECallback(
                tokenizer=tokenizer,
                max_length=min(cli_args.max_decoder_length, 256),
                max_eval_samples=56,
                batch_size=2,
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                repetition_penalty=1.0,
                no_repeat_ngram_size=3,
                num_beams=4,
                seed=cli_args.seed,
                eval_data=rouge_val_df,
                eval_collator=eval_collator,
                # chat collator primes the decoder with <|assistant|> for
                # non-thinking generation; summarization does not toggle
                # reasoning. unified collator falls back to bos.
                decoder_start_token=(
                    "<|assistant|>" if cli_args.collator == "chat" else None
                ),
                rouge_types=['rouge1', 'rouge2', 'rougeL'],
                freeze_metadata=cli_args.use_task_prompts,
                mesh=mesh,
            )
            logger.info(
                f"ROUGE callback initialized for summarization "
                f"({len(rouge_val_df)} val examples)"
            )

    def _build_save_additional_data(loss_value: float) -> Dict[str, Any]:
        """Assemble the per-checkpoint metadata blob for prepare_metadata.

        Mirrors train_han2han.py:4860-4875: tokens_seen + global_step
        + num_hosts + collator rng/iter_state + the eval/save token watermarks.
        On the next launch, restore reads back from this so the run picks up at
        the same tokens_seen and the next eval/save thresholds don't trip
        spuriously at the resume point.
        """
        data: Dict[str, Any] = {
            'loss': loss_value,
            'config': model.config.__dict__,
            'num_hosts': jax.process_count(),
            'micro_step': micro_step,
            'last_eval_tokens': last_eval_tokens,
            'last_save_tokens': last_save_tokens,
        }
        try:
            gs = collator.get_generator_state()
            data['gen_rng_b64'] = base64.b64encode(
                pickle.dumps(gs['rng_object'])).decode('utf-8')
            it = dict(gs.get('iter_state', {}) or {})
            if isinstance(it.get('processed_buckets'), set):
                it['processed_buckets'] = sorted(it['processed_buckets'])
            data['iter_state'] = it
        except Exception as e:
            log_from_main_process(logger, 'warning',
                f"collator state dump failed: {e}")
        return data

    # training loop
    logger.info("Starting SFT training...")
    # micro_step counts loop iterations; global_step counts completed optimizer
    # updates. with gradient_accumulation_steps=k, global_step advances once
    # every k micro_steps so logging/eval/save cadences remain comparable
    # across runs with different accumulation settings.
    grad_accum = max(int(getattr(cli_args, 'gradient_accumulation_steps', 1)), 1)
    if resuming:
        # default last_eval_tokens / last_save_tokens to tokens_seen (not 0) so
        # the eval+save thresholds don't trip immediately on the first step
        # after resume. micro_step must be restored too -- dropout/layerdrop
        # rngs derive from cli_args.seed + micro_step (see below) and would
        # desynchronize across the resume boundary otherwise.
        tokens_seen = int(restored_metadata.get('tokens_seen', 0))
        global_step = int(restored_metadata.get('global_step', 0))
        micro_step = int(restored_metadata.get('micro_step', global_step * grad_accum))
        last_eval_tokens = int(restored_metadata.get('last_eval_tokens', tokens_seen))
        last_save_tokens = int(restored_metadata.get('last_save_tokens', tokens_seen))
        log_from_main_process(logger, 'info',
            f"Resumed counters: tokens_seen={tokens_seen:,}, global_step={global_step}, "
            f"micro_step={micro_step}, last_eval_tokens={last_eval_tokens:,}, "
            f"last_save_tokens={last_save_tokens:,}")
    else:
        micro_step = 0
        global_step = 0
        tokens_seen = 0
        last_eval_tokens = 0
        last_save_tokens = 0
    last_log_time = time.time()

    # create batched iterator from collator (collator yields individual examples)
    def get_batch_iter():
        return create_batches(
            collator.sampled_datasets,
            batch_size=global_batch_size,
            max_encoder_length=cli_args.max_encoder_length,
            max_decoder_length=cli_args.max_decoder_length,
            collator=collator,
            enable_packing=cli_args.enable_packing,
        )

    train_iter = get_batch_iter()

    # external global-norm clip gating mirrors train_han2han.py:4435.
    # muon / fromage / big_vision_adafactor / adafactor-with-param-block-rms
    # all handle clipping internally (either via in-chain optax.clip_by_global_norm
    # on the AdamW arm or via factored row/col block_rms), so the external
    # clip_and_norm gets a 1e30 no-op while still surfacing the global norm.
    opt_lower = str(cli_args.optimizer).lower()
    use_global_clip = (
        opt_lower == "adamw"
        or (opt_lower in ("adafactor", "factored_adam")
            and not getattr(cli_args, 'use_param_block_rms', True))
    )
    if use_global_clip:
        clipnorm_value = cli_args.clipnorm if cli_args.clipnorm and cli_args.clipnorm > 0 else 1e30
    else:
        clipnorm_value = 1e30
    grad_clipnorm = jnp.array(clipnorm_value, dtype=jnp.float32)
    log_from_main_process(logger, 'info',
        f"External grad clip: {'ON' if use_global_clip else 'OFF (internal/no-op)'} "
        f"(optimizer={opt_lower}, clipnorm_value={clipnorm_value})")

    cached_train_step = nnx.cached_partial(
        sft_train_step,
        model,
        optimizer,
    )

    while tokens_seen < cli_args.max_tokens:
        # seed dropout from micro_step so each accumulated micro-batch gets a
        # fresh mask even when global_step does not advance this iteration
        dropout_rngs = nnx.Rngs(dropout=jax.random.PRNGKey(cli_args.seed + micro_step),
                                layerdrop=jax.random.PRNGKey(cli_args.seed + micro_step + 1),
                                default=jax.random.PRNGKey(cli_args.seed + micro_step + 2))

        # get batch
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = get_batch_iter()
            batch = next(train_iter)

        # shard batch across devices
        batch = shard_batch_to_devices(batch, mesh)

        # train step (sft_train_step preserves SS+CL)
        with mesh:
            metrics = cached_train_step(
                dropout_rngs, lr_schedule, cli_args.learning_rate,
                cli_args.warmup_ratio, cli_args.max_tokens,
                is_constant_schedule, tokens_seen,
                ss_config, cl_config,
                cli_args.ss_decisions_from_train_pass,
                cli_args.ss_differentiable_two_pass,
                grad_clipnorm, batch
            )

        # update counters. global_step (optimizer step count) only advances
        # at the end of a grad-accum window; step_completed marks that boundary
        # so logging/eval/save fire once per optimizer update, not per micro.
        # valid_token_count is computed across the global batch inside the
        # train_step; do NOT fall back to a per-host batch['labels'].size,
        # which under-counts and previously also tripped a KeyError because
        # the train_step pops 'labels' off model_inputs.
        micro_step += 1
        num_tokens = int(metrics['valid_token_count'])
        tokens_seen += num_tokens
        loss = metrics['loss']
        step_completed = (micro_step % grad_accum == 0)
        if step_completed:
            global_step += 1

        # logging (every 10 global / optimizer steps)
        if step_completed and global_step % 10 == 0:
            now = time.time()
            step_time = (now - last_log_time) / 10.0
            last_log_time = now
            # imported train_step's hardcoded cosine LR-for-logging ignores
            # args.lr_schedule; recompute here so WandB reflects the actual schedule
            progress = tokens_seen / max(cli_args.max_tokens, 1)
            lr = compute_lr_for_logging(
                progress=progress,
                learning_rate=cli_args.learning_rate,
                warmup_ratio=cli_args.warmup_ratio,
                constant_ratio=cli_args.constant_ratio,
                min_lr_ratio=cli_args.min_lr_ratio,
                schedule_type=cli_args.lr_schedule,
                lr_cooldown_ratio=getattr(cli_args, 'lr_cooldown_ratio', 0.0),
                lr_cooldown_type=getattr(cli_args, 'lr_cooldown_type', 'linear'),
            )
            grad_norm = float(metrics.get('grad_norm', 0.0))

            log_from_main_process(logger, 'info',
                f"Step {global_step}: loss={float(loss):.4f}, lr={lr:.2e}, "
                f"grad_norm={grad_norm:.4f}, tokens={tokens_seen:,}/{cli_args.max_tokens:,}, "
                f"step_time={step_time:.2f}s")

            if cli_args.use_wandb and jax.process_index() == 0:
                import wandb
                wandb_payload = {
                    'train/loss': float(loss),
                    'train/mle_loss': float(metrics.get('mle_loss', loss)),
                    'train/cl_loss': float(metrics.get('cl_loss', 0.0)),
                    'train/gold_win_rate': float(metrics.get('gold_win_rate', 0.0)),
                    'train/avg_margin': float(metrics.get('avg_margin', 0.0)),
                    'train/learning_rate': lr,
                    'train/grad_norm': grad_norm,
                    'train/tokens_seen': tokens_seen,
                    'train/step': global_step,
                    'train/micro_step': micro_step,
                }
                # scheduled-sampling confidence stats; present only when ss_config is enabled
                for ss_key in ('ce_mean', 'ce_std', 'ce_min', 'ce_max', 'ce_median',
                               'ce_high_conf_threshold', 'ce_low_conf_threshold'):
                    if ss_key in metrics:
                        wandb_payload[f'train/ss/{ss_key}'] = float(metrics[ss_key])
                wandb.log(wandb_payload)

        # evaluation
        if tokens_seen - last_eval_tokens >= cli_args.eval_every_tokens:
            logger.info(f"Running evaluation at {tokens_seen:,} tokens...")
            eval_results = eval_callback.evaluate(model, global_step)

            log_from_main_process(logger, 'info', f"Eval results: {eval_results}")

            if cli_args.use_wandb and jax.process_index() == 0:
                import wandb
                for key, value in eval_results.items():
                    if isinstance(value, (int, float)):
                        wandb.log({f'eval/{key}': value})

            # second generative-eval pass with <|think|> priming. mutually
            # exclusive with the direct pass: rationale conditions the answer.
            cot_eval_results = {}
            if cli_args.eval_with_cot:
                logger.info("Running CoT-mode generative evaluation...")
                cot_eval_results = eval_callback.evaluate(
                    model,
                    global_step,
                    decoder_start_token="<|think|>",
                    max_length=256,
                    parse_cot_answer=True,
                )
                log_from_main_process(logger, 'info',
                    f"CoT eval results: {cot_eval_results}")
                if cli_args.use_wandb and jax.process_index() == 0:
                    import wandb
                    for key, value in cot_eval_results.items():
                        if isinstance(value, (int, float)):
                            wandb.log({f'eval/cot/{key}': value})

            # BLEU evaluation for transcription
            if bleu_callback is not None:
                logger.info("Running BLEU evaluation...")
                bleu_results = bleu_callback.evaluate(model, global_step,
                                                      use_metadata=cli_args.use_task_prompts)
                bleu_stats = bleu_results.get('generation_stats', {})

                if bleu_stats:
                    log_from_main_process(logger, 'info',
                        f"BLEU: avg={bleu_stats['avg_bleu']:.4f}, "
                        f"median={bleu_stats['median_bleu']:.4f}")

                if cli_args.use_wandb and jax.process_index() == 0:
                    import wandb
                    for key, value in bleu_stats.items():
                        if isinstance(value, (int, float)):
                            wandb.log({f'eval/bleu_{key}': value})

            # ROUGE evaluation for summarization
            rouge_scalar_stats = {}
            if rouge_callback is not None:
                logger.info("Running ROUGE evaluation...")
                rouge_results = rouge_callback.evaluate(
                    model, global_step, use_metadata=cli_args.use_task_prompts
                )
                rouge_stats = rouge_results.get('generation_stats', {})

                # ROUGE stats are nested dicts ({rouge_type}_{p|r|f} -> {mean,
                # std, min, max}). flatten to scalars for wandb and best-metric
                # tracking. also surface per-subsource fmeasures.
                for key, value in rouge_stats.items():
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, (int, float)):
                                rouge_scalar_stats[f'{key}_{sub_key}'] = sub_value
                    elif isinstance(value, (int, float)):
                        rouge_scalar_stats[key] = value

                for subsource, sub_stats in rouge_results.get('subsource_stats', {}).items():
                    for sub_key, sub_value in sub_stats.items():
                        if isinstance(sub_value, (int, float)):
                            rouge_scalar_stats[f'{subsource}_{sub_key}'] = sub_value

                if rouge_stats:
                    log_from_main_process(logger, 'info',
                        f"ROUGE: " + ", ".join(
                            f"{rt}_F1={rouge_stats[f'{rt}_fmeasure']['mean']:.4f}"
                            for rt in rouge_callback.rouge_types
                            if f'{rt}_fmeasure' in rouge_stats
                        )
                    )

                if cli_args.use_wandb and jax.process_index() == 0:
                    import wandb
                    for key, value in rouge_scalar_stats.items():
                        wandb.log({f'eval/rouge_{key}': value})

            # save best checkpoint right after eval
            if best_metric:
                all_metrics = {**eval_results}
                if bleu_callback is not None and bleu_stats:
                    all_metrics.update(bleu_stats)
                if rouge_callback is not None and rouge_scalar_stats:
                    all_metrics.update(rouge_scalar_stats)
                # surface CoT metrics under cot_* so they can be selected as
                # save_best_metric (e.g. cot_temporal/macro_f1) without colliding
                # with the direct-pass keys.
                for key, value in cot_eval_results.items():
                    if key == 'step':
                        continue
                    if isinstance(value, (int, float)):
                        all_metrics[f'cot_{key}'] = value
                logger.info(f"Saving best-metric checkpoint ({best_metric}={all_metrics.get(best_metric, 'N/A')})...")
                meta = prepare_metadata(
                    tokens_seen, global_step,
                    _build_save_additional_data(float(loss)),
                )
                save_checkpoint(
                    finetune_ckpt_manager_best,
                    model,
                    optimizers=optimizer,
                    step=global_step,
                    metadata=meta,
                    metrics=all_metrics,
                )

            last_eval_tokens = tokens_seen

        # save 'latest' checkpoint on interval. fires in both best-metric and
        # plain modes -- in best-metric mode this is the *only* thing keeping
        # latest/ fresh between best-improvement events, so the resume window
        # stays small. orbax's max_to_keep=2 on latest/ gives us a one-step
        # fallback if preemption corrupts the newest write.
        if tokens_seen - last_save_tokens >= cli_args.save_every_tokens:
            logger.info(f"Saving 'latest' checkpoint at {tokens_seen:,} tokens...")
            meta = prepare_metadata(
                tokens_seen,
                global_step,
                _build_save_additional_data(float(loss)),
            )
            save_checkpoint(
                finetune_ckpt_manager_latest,
                model,
                optimizers=optimizer,
                step=global_step,
                metadata=meta,
            )
            last_save_tokens = tokens_seen

    # final evaluation
    logger.info("Running final evaluation...")
    eval_results = eval_callback.evaluate(model, global_step)
    log_from_main_process(logger, 'info', f"Final eval results: {eval_results}")

    final_cot_eval_results = {}
    if cli_args.eval_with_cot:
        logger.info("Running final CoT-mode generative evaluation...")
        final_cot_eval_results = eval_callback.evaluate(
            model,
            global_step,
            decoder_start_token="<|think|>",
            max_length=256,
            parse_cot_answer=True,
        )
        log_from_main_process(logger, 'info',
            f"Final CoT eval results: {final_cot_eval_results}")
        if cli_args.use_wandb and jax.process_index() == 0:
            import wandb
            for key, value in final_cot_eval_results.items():
                if isinstance(value, (int, float)):
                    wandb.log({f'eval/cot/{key}': value})

    final_bleu_stats = {}
    if bleu_callback is not None:
        logger.info("Running final BLEU evaluation...")
        bleu_results = bleu_callback.evaluate(
            model, global_step, use_metadata=cli_args.use_task_prompts
        )
        final_bleu_stats = bleu_results.get('generation_stats', {})
        log_from_main_process(logger, 'info', f"Final BLEU: {final_bleu_stats}")

    final_rouge_scalar_stats = {}
    if rouge_callback is not None:
        logger.info("Running final ROUGE evaluation...")
        rouge_results = rouge_callback.evaluate(
            model, global_step, use_metadata=cli_args.use_task_prompts
        )
        final_rouge_stats = rouge_results.get('generation_stats', {})
        for key, value in final_rouge_stats.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float)):
                        final_rouge_scalar_stats[f'{key}_{sub_key}'] = sub_value
            elif isinstance(value, (int, float)):
                final_rouge_scalar_stats[key] = value
        log_from_main_process(logger, 'info', f"Final ROUGE: {final_rouge_scalar_stats}")

    # final save. always writes 'latest' so a relaunched job after natural
    # exit doesn't re-run the entire SFT; also writes 'best' when applicable
    # so the final eval gets a chance to win best-metric selection.
    logger.info("Training complete, saving final checkpoint...")
    meta = prepare_metadata(
        tokens_seen,
        global_step,
        _build_save_additional_data(float(loss)),
    )
    if best_metric:
        final_metrics = {**eval_results, **final_bleu_stats, **final_rouge_scalar_stats}
        for key, value in final_cot_eval_results.items():
            if key == 'step':
                continue
            if isinstance(value, (int, float)):
                final_metrics[f'cot_{key}'] = value
        save_checkpoint(
            finetune_ckpt_manager_best,
            model,
            optimizers=optimizer,
            step=global_step,
            metadata=meta,
            metrics=final_metrics,
        )
    save_checkpoint(
        finetune_ckpt_manager_latest,
        model,
        optimizers=optimizer,
        step=global_step,
        metadata=meta,
    )

    # block on async metadata/commit threads before process teardown, otherwise
    # orbax's background writers can hit "cannot schedule new futures after
    # shutdown" and the final checkpoint never finalizes.
    if finetune_ckpt_manager_best is not None:
        finetune_ckpt_manager_best.wait_until_finished()
    finetune_ckpt_manager_latest.wait_until_finished()

    if cli_args.use_wandb and jax.process_index() == 0:
        import wandb
        wandb.finish()

    logger.info("SFT fine-tuning complete!")


if __name__ == "__main__":
    main()
