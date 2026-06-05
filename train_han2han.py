#!/usr/bin/env python3
# coding: utf-8
"""
Han2Han Pre-training Script

Distributed training script for the Han2Han model with:
- JAX SPMD (...mostly)
- Muon optimizer with MultiSteps gradient accumulation
- MultilingualCollator with `datasets`
- mBART-style denoising corruption
- YAML configuration
- TPU v4-64 optimized (8 hosts, 8 devices each)
"""

import logging
import sys
import gc
import json
import random
import pickle
import base64
from functools import partial
from typing import Dict, Any
import argparse
import yaml

# logging setup before any other imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    stream=sys.stdout,
    force=True
)

# suppress absl logging
from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.WARNING)

# initialize distributed training IMMEDIATELY after JAX import to prevent backend init
# only initialize if we're in a distributed TPU environment
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# inject LIBTPU_INIT_ARGS per TPU topology (must be set before JAX init)
# TPU_ACCELERATOR_TYPE is set by GCE on all TPU VMs (e.g. "v4-64", "v5litepod-64")
_tpu_accel = os.getenv('TPU_ACCELERATOR_TYPE', '')
if _tpu_accel and 'LIBTPU_INIT_ARGS' not in os.environ:
    _common = '--xla_enable_async_all_gather=true'

    # extract remat_policy from YAML config before argparse (needed pre-JAX-init)
    _remat_policy_early = None
    _config_path_early = None
    for _i, _arg in enumerate(sys.argv):
        if _arg == '--config' and _i + 1 < len(sys.argv):
            _config_path_early = sys.argv[_i + 1]
        elif _arg.startswith('--config='):
            _config_path_early = _arg.split('=', 1)[1]
        if _arg == '--remat_policy' and _i + 1 < len(sys.argv):
            _remat_policy_early = sys.argv[_i + 1]
        elif _arg.startswith('--remat_policy='):
            _remat_policy_early = _arg.split('=', 1)[1]
    if _remat_policy_early is None and _config_path_early:
        try:
            with open(_config_path_early) as _f:
                _remat_policy_early = (yaml.safe_load(_f) or {}).get('remat_policy', 'full')
        except (FileNotFoundError, yaml.YAMLError):
            _remat_policy_early = 'full'
    _remat_policy_early = _remat_policy_early or 'full'
    _uses_host_offload = 'offloaded' in _remat_policy_early
    _host_offload_flags = (
        ' --xla_tpu_enable_all_experimental_scheduler_features=true'
        ' --xla_tpu_enable_scheduler_memory_pressure_tracking=true'
        ' --xla_tpu_host_transfer_overlap_limit=24'
        ' --xla_tpu_aggressive_opt_barrier_removal=ENABLED'
        ' --xla_lhs_prioritize_async_depth_over_stall=ENABLED'
        ' --xla_tpu_enable_ag_backward_pipelining=true'
        ' --xla_should_allow_loop_variant_parameter_in_chain=ENABLED'
        ' --xla_should_add_loop_invariant_op_in_chain=ENABLED'
        ' --xla_max_concurrent_host_send_recv=100'
        ' --xla_tpu_scheduler_percent_shared_memory_limit=100'
        ' --xla_latency_hiding_scheduler_rerun=2'
    )

    if _tpu_accel.startswith('v4'):
        os.environ['LIBTPU_INIT_ARGS'] = f'{_common}'
    elif 'v5 lite' in _tpu_accel or _tpu_accel.startswith('v5e'):
        _v5e_flags = (
            f'{_common} '
            '--xla_tpu_scoped_vmem_limit_kib=81920 '
            '--xla_tpu_enable_data_parallel_all_reduce_opt=true '
            '--xla_tpu_data_parallel_opt_different_sized_ops=true '
            '--xla_tpu_enable_async_collective_fusion=true '
            '--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true '
            '--xla_tpu_enable_async_collective_fusion_multiple_steps=true '
            '--xla_tpu_overlap_compute_collective_tc=true '
            '--xla_tpu_use_minor_sharding_for_major_trivial_input=true '
            '--xla_tpu_relayout_group_size_threshold_for_reduce_scatter=1 '
            '--xla_tpu_assign_all_reduce_scatter_layout=true'
        )
        if _uses_host_offload:
            _v5e_flags += _host_offload_flags
        os.environ['LIBTPU_INIT_ARGS'] = _v5e_flags
    elif _tpu_accel.startswith('v6e'):
        _v6e_flags = (
            f'{_common} '
            '--xla_tpu_scoped_vmem_limit_kib=98304 '
            '--xla_tpu_enable_async_collective_fusion=true '
            '--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true '
            '--xla_tpu_enable_async_collective_fusion_multiple_steps=true '
            '--xla_tpu_overlap_compute_collective_tc=true'
        )
        if _uses_host_offload:
            _v6e_flags += _host_offload_flags
        os.environ['LIBTPU_INIT_ARGS'] = _v6e_flags
    print(f"LIBTPU_INIT_ARGS set for {_tpu_accel}: {os.environ.get('LIBTPU_INIT_ARGS', '')}")

if os.getenv('ENABLE_JAX_DISTRIBUTED'):
    try:
        import jax
        jax.distributed.initialize()
        print("JAX distributed training initialized")
    except RuntimeError as e:
        import jax
        if "already been initialized" not in str(e):
            print(f"JAX distributed initialization failed: {e}")
else:
    import jax

from logging_utils import log_from_main_process, log_from_all_processes

from jax.experimental import multihost_utils
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec

from flax import nnx
_nnx_to_arrays = getattr(nnx, 'to_arrays', None)
if _nnx_to_arrays is None:
    _nnx_to_arrays = getattr(nnx, 'as_array_vars', None)
if _nnx_to_arrays is None:
    _nnx_to_arrays = nnx.vars_as
_nnx_iter_modules = getattr(nnx, 'iter_modules', None)
if _nnx_iter_modules is None:
    _nnx_iter_modules = lambda m: m.iter_modules()

import optax
import numpy as np

import tensorflow as tf
tf.config.set_visible_devices([], device_type='GPU')
tf.config.set_visible_devices([], device_type='TPU')
import tensorflow_datasets as tfds

# checkpointing
from checkpoint_utils import (
    save_checkpoint,
    restore_checkpoint,
    prepare_metadata,
    extract_tokens_from_metadata,
    setup_checkpoint_manager,
)
from orbax.checkpoint.checkpoint_manager import StepAlreadyExistsError

# huggingface
from transformers import AutoTokenizer

# data handling
from tqdm.auto import tqdm

# local modules
import register_han2han  # register Han2Han tokenizer with HuggingFace AutoClasses
from multilingual_collator import MultilingualCollator
from unified_collator import UnifiedCollator
from modeling_han2han_flax import Han2HanConfig, FlaxHan2Han
from bleu_callback import BLEUCallback
from generative_evaluation_callback import GenerativeEvaluationCallback
from mc_logprob_callback import MCLogProbCallback
from temporal_logprob_callback import TemporalLogProbCallback
from subword_features import compute_subword_tables
from token_based_schedule import (
    compute_lr_for_logging,
)
from task_prompts import TASK_PROMPTS
from sharding_utils import (
    setup_mesh_and_sharding,
    derive_param_sharding,
    shard_batch_to_devices,
)
from optimizer import (
    create_learning_rate_schedule,
    patch_to_opt_state_for_factored_adafactor,
    _estimate_effective_infilling,
    _create_gradaccum_schedule,
    create_optimizer,
    _muon_arm_label,
)

logger = logging.getLogger(__name__)


# wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    # only log from main process to prevent duplicate warnings
    if jax.process_index() == 0:
        print("wandb not available, logging will be limited to console")

logger = logging.getLogger(__name__)


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def type_parse(arg):
    if isinstance(arg, str):
        x = arg.lower()
        if x in {"true", "t", "yes", "1"}:
            return True
        elif x in {"false", "f", "no", "0"}:
            return False
        return str(arg)
    else:
        return str(arg)


def get_config(return_parser_only: bool = False, argv=None):
    """Parse command line arguments with YAML config support.

    When ``return_parser_only=True``, build the parser and return it without
    parsing argv, loading YAML, or running validation. Used by downstream
    scripts (e.g. ``finetune_sft.py``) to backfill pretraining-only defaults
    onto their own Namespace so functions like ``create_optimizer`` don't
    AttributeError on flags the SFT parser doesn't define.

    When ``argv`` is provided, parse that list instead of ``sys.argv``. Lets
    downstream scripts (e.g. ``evaluate_all_checkpoints.py``) split off their
    own eval-specific flags via ``parse_known_args`` and forward the remainder
    here for full YAML merge + numeric coercion + validation.
    """
    parser = argparse.ArgumentParser(
        description="Train Han2Han Model on TPU Pod (Distributed)"
    )

    # === CONFIG ===
    parser.add_argument("--config", type=str, default="han2han-ul2-base-cooldown-v5e-eu.yaml",
                       help="Path to YAML configuration file")
    parser.add_argument("--skip_restore", action="store_true",
                       help="Skip restoring from checkpoint (start fresh)")
    parser.add_argument("--restore_step", type=int, default=None,
                       help="Restore from a specific checkpoint step instead of latest")
    parser.add_argument("--pretrained_weights_path", type=str, default=None,
                       help="Path to pretrained model checkpoint to initialize weights from (model reincarnation)")
    parser.add_argument("--restore_optimizer", type=lambda x: x.lower()=='true',
                       default=False,
                       help="Restore optimizer state when loading from pretrained_weights_path (default: False)")
    parser.add_argument("--eval_clear_caches", type=lambda x: x.lower()=='true',
                        default=True, help="Before / after evaluation, clear caches to save memory (default: True)")

    # === INPUT/OUTPUT ===
    parser.add_argument("--tokenizer_path", type=str,
                       default="han2han_v2_tokenizer",
                       help="Path to Han2Han tokenizer directory")
    parser.add_argument("--output_dir", type=str, default="output_han2han",
                       help="Local output directory")
    parser.add_argument("--gcs_output_dir", type=str, default=None,
                       help="GCS path for syncing outputs")

    # === MODEL ARCHITECTURE ===
    parser.add_argument("--d_model", type=int, default=1536,
                       help="Model embedding dimension")
    parser.add_argument("--d_prime", type=int, default=None,
                       help="Attention projection dimension (defaults to d_model for MHA)")
    parser.add_argument("--d_ff", type=int, default=12288,
                       help="Feed forward network inner dimension")
    parser.add_argument("--num_heads", type=int, default=12,
                       help="Number of attention heads")
    parser.add_argument("--head_dim", type=int, default=None,
                       help="MHA per-head dimension. With d_prime, both must match. With num_heads alone, d_prime = num_heads * head_dim. (MHA only)")
    parser.add_argument("--num_kv_heads", type=int, default=None,
                       help="MHA self-attn KV heads for GQA/MQA. None = num_heads (full MHA). 1 = MQA. Must divide num_heads. (MHA only)")
    parser.add_argument("--cross_attn_num_heads", type=int, default=None,
                       help="MHA cross-attn Q heads. None = inherit --num_heads. Lets cross-attn use full d_model fidelity (cross_attn_num_heads * head_dim) while self-attn stays compressed. (MHA only)")
    parser.add_argument("--cross_attn_num_kv_heads", type=int, default=None,
                       help="MHA cross-attn KV heads. None = inherit --num_kv_heads. Must divide cross_attn_num_heads. (MHA only)")
    parser.add_argument("--use_qk_norm", action="store_true",
                       help="Per-head RMSNorm on Q and K (Gemma 3 / T5Gemma 2 style). (MHA only)")
    parser.add_argument("--query_pre_attn_scalar", type=float, default=None,
                       help="HF-Gemma 3 Q multiplier semantics: scaling = scalar ** -0.5. None = head_dim ** -0.5. (MHA only)")
    parser.add_argument("--encoder_layers", type=int, default=18,
                       help="Number of encoder layers")
    parser.add_argument("--decoder_layers", type=int, default=18,
                       help="Number of decoder layers")
    parser.add_argument("--attention_mechanism", type=str, default="mha",
                       choices=["mha"],
                       help="Attention mechanism type (default for all layers)")
    parser.add_argument("--encoder_attention_types", type=str, default=None,
                       help="Per-layer encoder attention types (comma-separated, e.g., 'mha,mha-sliding')")
    parser.add_argument("--decoder_attention_types", type=str, default=None,
                       help="Per-layer decoder self-attention types (comma-separated, e.g., 'mha,mha-sliding')")
    parser.add_argument("--decoder_cross_attention_types", type=str, default=None,
                       help="Per-layer decoder cross-attention types (comma-separated, e.g., 'mha,mha')")
    parser.add_argument("--sliding_window_size", type=int, default=256,
                       help="Window size for 'mha-sliding' attention layers (0 = full attention)")
    parser.add_argument("--rope_theta", type=float, default=10000.0,
                       help="RoPE base frequency theta (default 10000). Applies to all layers unless --rope_theta_sliding is set.")
    parser.add_argument("--rope_theta_sliding", type=float, default=None,
                       help="RoPE theta override for 'sliding'/'local' attention layers (default: inherit --rope_theta).")
    parser.add_argument("--no_apply_legacy_rope_quirk", action="store_true",
                       help="Opt out of the legacy scan_rope_theta auto-correction. "
                            "Pre-fix Flax `_identify_scan_groups` collapsed mha/mha-sliding into one "
                            "scan stack whose rope_theta was baked from position_specs[0], so legacy "
                            "configs with mixed attention types + rope_theta != rope_theta_sliding "
                            "had their rope_theta silently overwritten during training. "
                            "Han2HanConfig auto-detects this pattern and rewrites rope_theta to match. "
                            "Pass this flag for V2+ runs that genuinely want hybrid rope_theta now that "
                            "the Flax bug is fixed.")
    parser.add_argument("--use_sub_ln", action="store_true",
                       help="SubLN: RMSNorm before output projections in attention and FFN")
    parser.add_argument("--initializer_range", type=float, default=0.02,
                       help="Stddev for weight initialization (DeepSeek uses ~0.006 for large models)")
    parser.add_argument("--kernel_init_type", type=str, default="normal",
                       choices=["normal", "variance_scaling"],
                       help="Kernel init strategy: normal(stddev=initializer_range) or variance_scaling(scale, fan_in, truncated_normal)")
    parser.add_argument("--kernel_init_scale", type=float, default=0.1,
                       help="Scale for variance_scaling init (0.1=T5X, 1.0=lecun_normal). Only used when kernel_init_type=variance_scaling")
    parser.add_argument("--init_biases_normal", action="store_true",
                       help="Init biases as normal(stddev=initializer_range) instead of zeros (V1 behavior)")
    parser.add_argument("--decoder_norm_type", type=str, default="rmsnorm",
                       choices=["rmsnorm", "rmsnorm_bias", "layernorm"],
                       help="Decoder normalization type: rmsnorm, rmsnorm_bias (adds learnable bias), or layernorm (default: rmsnorm)")
    parser.add_argument("--encoder_norm_type", type=str, default="rmsnorm",
                       choices=["rmsnorm", "rmsnorm_bias", "layernorm"],
                       help="Encoder normalization type: rmsnorm, rmsnorm_bias (adds learnable bias), or layernorm (default: rmsnorm)")
    parser.add_argument("--layer_norm_epsilon", type=float, default=1e-5,
                        help="Value of epsilon in layer normalization modules (incl. RMSNorm)")

    # bias configuration (DeepSeek V3 style)
    parser.add_argument("--use_bias", type=lambda x: x.lower()=='true', default=True,
                       help="Global toggle for biases in all linear layers (default: True)")

    parser.add_argument("--tie_word_embeddings", type=lambda x: x.lower()=='true',
                        default=False,
                        help="Tie encoder.wte and decoder.wte")
    parser.add_argument("--tie_subtoken_embeddings",
                        default=True,
                        help="Tie sub-token (wje, wce) embeddings (only if also tying token embeddings)")
    parser.add_argument("--tie_input_output_embeddings", type=lambda x: x.lower()=='true',
                        default=False,
                        help="Tie decoder.wte and lm_head weights")
    parser.add_argument("--tie_encoder_decoder", type=lambda x: x.lower()=='true',
                        default=False,
                        help="Tie all encoder and decoder layer weights for maximum efficiency")

    # === TRAINING HYPERPARAMETERS ===
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Training batch size per process")
    parser.add_argument("--learning_rate", type=float, default=1e-2,
                       help="Peak learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.05,
                       help="Warmup ratio (e.g., 0.05 = first 5% of training)")
    parser.add_argument("--constant_ratio", type=float, default=0.0,
                       help="Constant LR ratio (e.g., 0.15 = 15% at peak LR)")
    parser.add_argument("--min_lr_ratio", type=float, default=0.15,
                       help="Minimum LR as ratio of peak for decay schedule (default 0.15)")
    parser.add_argument("--lr_schedule", type=str, default="cosine",
                       choices=["cosine", "constant", "linear", "rsqrt"],
                       help="Learning rate schedule type: cosine, linear, rsqrt (1/sqrt, ViT-22B), or constant")
    parser.add_argument("--lr_cooldown_ratio", type=float, default=0.0,
                       help="Fraction of training for final LR cooldown (e.g. 0.1 = last 10%%)")
    parser.add_argument("--lr_cooldown_type", type=str, default="linear",
                       choices=["linear", "sqrt"],
                       help="Cooldown shape: linear or sqrt (1-sqrt(x), fast initial decay)")
    parser.add_argument("--optimizer", type=str, default="muon",
                       choices=["muon"],
                       help="Optimizer type: muon (orthogonalized momentum via Newton-Schulz "
                            "for 2D non-embed/lm_head weights; AdamW for 1D + embed/lm_head).")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                       help="Weight decay rate for attention modules (default: 0.0)")
    parser.add_argument("--mlp_weight_decay", type=float, default=0.0,
                       help="Weight decay rate for MLP kernels (default: 0.01)")
    parser.add_argument("--embedding_weight_decay", type=float, default=0.0,
                       help="Weight decay rate for embeddings (default: 0.1, constrain embedding norms)")
    parser.add_argument("--lm_head_weight_decay", type=float, default=0.0,
                       help="Weight decay rate for LM head when untied from embeddings (default: 0.0). "
                            "Only used when tie_input_output_embeddings=False. ViT paper suggests 100x body WD.")
    parser.add_argument("--norm_weight_decay", type=float, default=0.0,
                       help="Weight decay rate for RMSNorm/LayerNorm scales (default: 0.0, but Adafactor may need ~0.1)")
    parser.add_argument("--bias_weight_decay", type=float, default=0.0,
                       help="Weight decay rate for biases (default: 0.0, typically left at zero)")
    parser.add_argument("--beta1", type=float, default=0.9,
                       help="First-moment decay (Adam b1) for Muon's AdamW arm on 1D params and embeddings/lm_head.")
    parser.add_argument("--beta2", type=float, default=0.99,
                       help="Second-moment decay (Adam b2) for Muon's AdamW arm on 1D params and embeddings/lm_head.")
    parser.add_argument("--adam_eps", type=float, default=1e-8,
                       help="Epsilon added inside Adam's second-moment denominator (m / sqrt(v) + eps) "
                            "to prevent divide-by-zero on rare-update params. Default 1e-8 is the Adam "
                            "paper / PyTorch / optax muon-contrib standard. OLMo-1B used 1e-5. "
                            "Important for sparse-update params (embeddings, lm_head) where v can "
                            "approach 0: smaller eps -> larger worst-case update on never-updated "
                            "rows. Note: if optax stores nu in bf16 (same dtype as params), values "
                            "below ~1e-3 round away under bf16 precision -- use >=1e-6 for safety.")
    parser.add_argument("--muon_beta", type=float, default=0.95,
                       help="Momentum decay for the Muon arm (orthogonalized momentum). Only used "
                            "with --optimizer muon. Adam arm (1D + embed/lm_head) uses --beta1/--beta2.")
    parser.add_argument("--muon_gamma", type=float, default=0.18,
                       help="DeepSeek V4 / Moonlight update rescaling factor. Replaces optax's "
                            "sqrt(max(1, out/red)) with sqrt(max(m, n)) * gamma so per-element "
                            "Muon update RMS matches AdamW at the same LR. Default 0.18 is the "
                            "DeepSeek V4 published value, calibrated so the empirical AdamW "
                            "per-element update magnitude matches Muon's at a shared LR. "
                            "Only used with --optimizer muon.")
    parser.add_argument("--muon_hybrid_ns", type=lambda x: x.lower() == 'true', default=False,
                       help="Use DeepSeek V4's 10-step hybrid Newton-Schulz schedule (8 fast steps "
                            "with (3.4445, -4.7750, 2.0315) + 2 stabilizing steps with "
                            "(2.0, -1.5, 0.5)) instead of optax's 5-step default. Only used with "
                            "--optimizer muon.")
    parser.add_argument("--muon_adam_wd_ratio", type=float, default=0.0,
                       help="Multiplier on weight_decay for the AdamW arm of --optimizer muon. "
                            "Adam arm covers 1D params (RMSNorm scales, biases) and embeddings/"
                            "lm_head. Default 0.0 means no WD on those leaves; set e.g. 1.0 to "
                            "apply the same WD as the Muon arm to embeddings/lm_head. By default "
                            "we apply this multiplier only to ND>=2 adam-arm leaves (embeddings, "
                            "lm_head, expert_weights) via a mask matching create_weight_decay_masks' "
                            "mask_global_decay; toggle with --muon_adam_wd_skip_1d.")
    parser.add_argument("--muon_adam_wd_skip_1d", type=lambda x: x.lower() == 'true', default=True,
                       help="If True (default), the Muon AdamW arm's weight decay (controlled by "
                            "--muon_adam_wd_ratio) skips 1D parameters (RMSNorm scales, biases, "
                            "alpha_logits) using the existing mask_global_decay. If False, the WD "
                            "ratio is applied uniformly to every adam-routed leaf, matching the "
                            "pre-2026-05 behavior. Only used with --optimizer muon.")
    parser.add_argument("--ns_variant", type=str, default="standard",
                       choices=["standard", "gram"],
                       help="Newton-Schulz inner algorithm for --optimizer muon. 'standard' "
                            "= optax's quintic NS (Keller Jordan). 'gram' = Gram Newton-Schulz "
                            "(Dao 2026): iterate on the n x n Gram matrix XX^T with Polar-Express "
                            "coefficients and a restart at iteration 2. On v5e, gram saves FLOPs "
                            "for non-square matrices but lacks the CUDA symmetric-GEMM kernel "
                            "speedup; precision is more sensitive than standard NS (see "
                            "--gram_ns_dtype). 'gram' ignores --muon_hybrid_ns.")
    parser.add_argument("--gram_ns_reset_iters", type=str, default="2",
                       help="Comma-separated 0-indexed iteration indices BEFORE which to restart "
                            "Gram NS (recompute R = X X^T from Q @ X). Only used with --ns_variant "
                            "gram. Paper default '2' for 5 Polar-Express steps; '2,4' adds a "
                            "second restart if more stability is needed.")
    parser.add_argument("--gram_chunk_size", type=int, default=0,
                       help="Chunk size for streaming the Gram NS leading batch axis through "
                            "jax.lax.map(..., batch_size=chunk_size). 0 (default) = full jax.vmap "
                            "(every batch element in parallel; needs ~batch * m^2 HBM for Gram "
                            "intermediates). A positive value streams `chunk_size` batch elements "
                            "at a time via inner vmap; peak HBM for intermediates scales like "
                            "Only used with --ns_variant gram.")
    parser.add_argument("--gram_ns_dtype", type=str, default="bf16",
                       choices=["bf16", "fp16", "fp32"],
                       help="Working dtype for Gram NS inner R, Q, Z matrices. v5e MXUs natively "
                            "support bf16; 'fp16' on v5e is typically lowered by XLA to bf16 or "
                            "fp32 (no native fp16 MXU). Use 'fp32' as a precision reference. Only "
                            "used with --ns_variant gram.")
    parser.add_argument("--max_tokens", type=int, default=200_000_000_000,
                       help="Maximum training tokens (e.g., 200B)")
    parser.add_argument("--sequence_length", type=int, default=1024,
                       help="Input sequence length")
    parser.add_argument("--clipnorm", type=float, default=1.0,
                       help="Global gradient clip norm applied to the AdamW arm only (muon arm is scale-invariant)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                       help="Gradient accumulation steps (target value when batch_warmup_tokens is set)")
    parser.add_argument("--save_on_preemption", action="store_true", default=True,
                       help="Save a checkpoint and exit at the JAX preemption sync point on SIGTERM (default: True)")
    parser.add_argument("--no_save_on_preemption", dest="save_on_preemption", action="store_false",
                       help="Disable preemption-triggered checkpointing")
    parser.add_argument("--use_scan_layers", action="store_true",
                       help="Use nnx.scan over homogeneous layer groups (reduces compilation time)")

    # === DROPOUT CONFIGURATION ===
    parser.add_argument("--embedding_dropout_rate", type=float, default=0.1,
                       help="Module-level CJK embedding dropout rate (default: 0.1)")
    parser.add_argument("--layerdrop", type=float, default=0.1,
                       help="Layer dropout rate for encoder/decoder layers (default: 0.1)")
    parser.add_argument("--cross_attn_pdrop", type=float, default=0.1,
                       help="Cross-attention dropout rate (default: 0.1)")
    parser.add_argument("--resdrop", type=float, default=0.1,
                       help="Residual/activation dropout rate (default: 0.1)")
    parser.add_argument("--attndrop", type=float, default=0.1,
                       help="Self-attention dropout rate (default: 0.1)")
    parser.add_argument("--embddrop", type=float, default=0.1,
                       help="Embedding layer dropout rate (default: 0.1)")
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                       help="Label smoothing alpha for cross-entropy loss (default: 0.1)")

    # === DATA PROCESSING ===
    parser.add_argument("--cooldown_ratio", type=float, default=0.2,
                       help="Start metadata/task prefix cooldown ratio (e.g., 0.2 = at 80% of training)")
    parser.add_argument("--infilling_ratio", type=float, default=0.3,
                       help="Span masking ratio for regular denoising")
    parser.add_argument("--poisson_lambda", type=float, default=3.5,
                       help="Mean span length for token-based masking (default: 3.5)")
    parser.add_argument("--morpheme_lambda", type=float, default=2.0,
                       help="Mean span length for morpheme-based masking (default: 2.0)")
    parser.add_argument("--sentence_permutation", type=lambda x: x.lower()=='true',
                       default=True,
                       help="Enable BART sentence permutation (only in BART and when --sentinel_denoising_ratio < 1.0)")
    parser.add_argument("--use_phase2_collator", type=lambda x: x.lower()=='true',
                       default=False,
                       help="Use Phase2MixedCollator for continuation training (denoising+continuation+nlp tasks)")
    parser.add_argument("--collator_buffer_size", type=int, default=10000,
                       help="Minimum efficiency threshold for packing (default 10_000)")
    parser.add_argument("--mode_ratios", type=str, default="0.4,0.4,0.2",
                       help="Comma-separated ratios for denoising,denoising_heavy,continuation (UL2-style, default: 0.4,0.4,0.2)")
    parser.add_argument("--morpheme_denoising_ratio", type=float, default=0.5,
                       help="Within regular denoising, ratio of morpheme-level vs token-level corruption (default: 0.5)")
    parser.add_argument("--sentinel_denoising_ratio", type=float, default=0.5,
                       help="Within token-level denoising, ratio of T5-style sentinel vs BART-style mask (default: 0.5)")
    parser.add_argument("--heavy_infilling_ratio", type=float, default=0.50,
                       help="Corruption ratio for heavy denoising / X-denoiser (default: 0.50)")
    parser.add_argument("--byte_reconstruction_ratio", type=float, default=0.0,
                       help="Within denoising_heavy, ratio to route byte-containing samples to byte reconstruction (default: 0.0 = disabled)")
    parser.add_argument("--temporal_continuation_ratio", type=float, default=0.0,
                       help="For samples with year metadata, ratio to route to temporal continuation task (default: 0.0 = disabled)")
    parser.add_argument("--han2han_transcription_ratio", type=float, default=0.0,
                       help="Ratio of Korean samples to which to apply Hanja<->Hangul transcription (default: 0.0 = disabled)")
    parser.add_argument("--enable_packing", type=lambda x: x.lower()=='true',
                       default=False,
                       help="Enable document packing for improved training efficiency (5x speedup!)")
    parser.add_argument("--packing_efficiency_threshold", type=float, default=0.8,
                       help="Minimum efficiency threshold for packing (default 0.8 = 80%)")
    parser.add_argument("--packed_buffer_size", type=float, default=128,
                       help="Minimum efficiency threshold for packing (default 128)")
    parser.add_argument("--training_mode", type=str, default='full',
                        help="Mode of training, used to determine per-host data budget (default: full)")
    parser.add_argument("--disable_budget_limit", type=bool, default=False,
                        help="If True, use full data slices without budget truncation (default: False)")
    parser.add_argument("--force_reload", type=bool, default=False,
                        help="If True, ignore cache and reload data from GCS (default: False)")
    parser.add_argument("--data_bucket", type=str, default=None,
                        help="GCS bucket for training data (e.g., gs://han2han-us-central1)")
    parser.add_argument("--data_source_type", type=str, default="han2han",
                        choices=["han2han"],
                        help="Data source type (Korean-only)")
    parser.add_argument("--sft_tasks", type=str, default="all",
                        help="Comma-separated SFT tasks to include, or 'all'/'none' (e.g., 'sts,nli,ynat')")
    parser.add_argument("--mc_eval_benchmarks", type=str, default="none",
                        help="Comma-separated MC eval benchmarks or 'none' (e.g., 'kmmlu,click,haerae')")
    parser.add_argument("--mc_eval_max_samples", type=int, default=200,
                        help="Max examples per MC benchmark for log-prob scoring")
    parser.add_argument("--temporal_eval_benchmarks", type=str, default="none",
                        help="Comma-separated temporal eval benchmarks or 'none' "
                             "(e.g., 'temporal_ko,temporal_en'). Reads parquets prepared by "
                             "prepare_temporal_eval_data.py from {data_bucket}/eval/temporal/")
    parser.add_argument("--disable_generative_eval", action="store_true",
                        help="Disable generative evaluation callback (runs first in callback order if mc_eval enabled)")
    parser.add_argument("--temporal_eval_max_samples", type=int, default=200,
                        help="Max examples per temporal benchmark for log-prob scoring")

    # === HAN2HAN SUBWORD FEATURES ===
    parser.add_argument("--jamo_subwords", type=lambda x: x.lower()=='true',
                        default=False,
                        help="Enable jamo n-gram subword features (Han2Han's core innovation)")
    parser.add_argument("--char_subwords", type=lambda x: x.lower()=='true',
                        default=False,
                        help="Enable character-level subword features")
    parser.add_argument("--subword_embed_dim", type=int, default=None,
                        help="Hidden dim of wje/wce subtoken embeddings (None = d_model // 2)")
    parser.add_argument("--ngram_sizes", type=str, default=None,
                        help="Comma-separated jamo n-gram sizes (e.g., '3,6,9')")
    parser.add_argument("--min_n", type=int, default=2,
                        help="Minimum jamo n-gram size (if ngram_sizes not specified)")
    parser.add_argument("--max_n", type=int, default=4,
                        help="Maximum jamo n-gram size (if ngram_sizes not specified)")
    parser.add_argument("--ffn_activation", type=str, default="swiglu",
                        choices=["swiglu", "geglu", "reglu2", "gelu", "gelu_new", "relu2"],
                        help="FFN activation function")
    parser.add_argument("--dense_ffn_activation", type=str, default=None,
                        choices=[None, "swiglu", "geglu", "reglu2", "gelu", "gelu_new", "relu2"],
                        help="Activation for the dense FFN path. None = follow ffn_activation.")
    parser.add_argument("--swiglu_clamp_limit", type=float, default=None,
                        help="Clamp the gate to (-inf, limit] and up-projection to [-limit, limit] "
                             "before the gated multiply (DeepSeek V4 / GPT-OSS stability fix). "
                             "DeepSeek V4 used 10.0 with SwiGLU; GPT-OSS used 7.0. None disables.")

    # === PERFORMANCE ===
    parser.add_argument("--model_dtype", type=str, default="bfloat16",
                       choices=["bfloat16", "float32", "float16"],
                       help="Activation/compute dtype for the forward pass")
    parser.add_argument("--param_dtype", type=str, default=None,
                       choices=["bfloat16", "float32", "float16"],
                       help="Storage dtype for latent weights (Flax param_dtype). "
                            "Defaults to --model_dtype. Set to float32 for stable "
                            "moment accumulation when --model_dtype=bfloat16.")
    parser.add_argument("--optimizer_state_dtype", type=str, default=None,
                       choices=["bfloat16", "float32", "float16"],
                       help="Storage dtype for optimizer first-moment / EMA accumulators "
                            "(mu_dtype / accumulator_dtype). Defaults to --model_dtype.")
    parser.add_argument("--gradient_checkpointing", type=lambda x: x.lower()=='true',
                       default=False,
                       help="Use gradient checkpointing (superseded by --remat_policy)")
    parser.add_argument("--remat_policy", type=str, default=None,
                       choices=["full", "none", "save_attn_weights",
                                "save_qkv_proj", "minimal",
                                "qkv_proj_offloaded", "attn_weights_offloaded",
                                "minimal_offloaded"],
                       help="Rematerialization policy. Supersedes --gradient_checkpointing. "
                            "full=recompute all (default when gradient_checkpointing=true), "
                            "none=save all, save_qkv_proj=keep QKV on device, "
                            "qkv_proj_offloaded=offload QKV to host, "
                            "minimal_offloaded=offload all projections to host")

    # === LOGGING & CHECKPOINTING ===
    parser.add_argument("--logging_steps", type=int, default=100,
                       help="Steps between logging")
    parser.add_argument("--log_level",
                       choices=['debug', 'info', 'warning', 'error', 'critical'],
                       default='info',
                       help="Python logging level applied to the root logger after argparse. "
                            "Use 'debug' to surface log_from_main_process(... 'debug' ...) lines "
                            "without editing source. Default 'info'.")
    parser.add_argument("--keep_checkpoints", type=int, default=5,
                       help="Number of checkpoints to keep")

    # === TOKEN-BASED EVALUATION & CHECKPOINTING ===
    parser.add_argument("--eval_ratio", type=float, default=0.1,
                       help="Evaluate every X ratio of total tokens (0.1 = every 10%% of training)")
    parser.add_argument("--save_ratio", type=float, default=0.1,
                       help="Save checkpoint every X ratio of total tokens (0.1 = every 10%% of training)")

    # === EVALUATION CONFIGURATION ===
    parser.add_argument("--eval_split_ratio", type=float, default=0.05,
                       help="Fraction of data to use for evaluation (0.05 = 5%)")
    parser.add_argument("--eval_batch_size", type=int, default=4,
                       help="Batch size for evaluation")
    parser.add_argument("--eval_max_batches", type=int, default=100,
                       help="Maximum number of batches to evaluate per eval step")

    # === WANDB ===
    parser.add_argument("--wandb_project", type=str, default="han2han",
                       help="WandB project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                       help="WandB entity")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                       help="WandB run name")
    parser.add_argument("--wandb_online", type=lambda x: x.lower()=='true',
                       default=True,
                       help="Run WandB in online mode")

    # === PARALLELISM CONFIGURATION ===
    parser.add_argument("--parallelism_strategy", type=str, default="hybrid",
                       choices=["data_parallel", "model_parallel", "fsdp", "hybrid", "custom"],
                       help="Parallelism strategy preset")
    parser.add_argument("--mesh_axes", type=str, nargs='+', default=["data", "model"],
                       help="Custom mesh axis names (for custom strategy)")
    parser.add_argument("--mesh_shape", type=int, nargs='+', default=None,
                       help="Custom mesh dimensions (for custom strategy)")

    # === SYSTEM ===
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--smoke_test", type=type_parse,
                       default=False,
                       help="Enable smoke test mode with dummy data (bypasses expensive streaming setup)")

    if return_parser_only:
        return parser

    # get argparse defaults before parsing
    defaults = {action.dest: action.default for action in parser._actions if action.dest != 'help'}

    args = parser.parse_args(argv)

    # apply log level early so YAML-merge / orphan-key / config-derivation
    # messages below also respect the requested verbosity. note that YAML
    # itself cannot override --log_level after this point; CLI is authoritative.
    _level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(_level)
    logger.setLevel(_level)

    # load YAML config if provided and merge with args
    if args.config and os.path.exists(args.config):
        yaml_config = load_yaml_config(args.config)

        if yaml_config is not None:
            # merge yaml config with args (yaml overrides defaults, CLI overrides yaml)
            orphan_keys = []
            for key, value in yaml_config.items():
                if hasattr(args, key):
                    current_value = getattr(args, key)
                    if current_value == defaults.get(key):
                        setattr(args, key, value)
                else:
                    orphan_keys.append(key)
            if orphan_keys:
                logger.warning(
                    "YAML config %s has %d key(s) with no matching CLI arg, "
                    "they will be IGNORED: %s",
                    args.config, len(orphan_keys), sorted(orphan_keys),
                )

            # ensure numeric fields are properly converted to float/int
            # (YAML parsers can sometimes interpret scientific notation as strings)
            numeric_float_fields = [
                'learning_rate', 'weight_decay', 'min_lr_ratio', 'constant_ratio',
                'warmup_ratio', 'clipnorm', 'cooldown_ratio', 'embedding_weight_decay',
                'packing_efficiency_threshold', 'norm_weight_decay', 'bias_weight_decay',
                'initializer_range', 'eval_split_ratio', 'save_ratio', 'eval_ratio', 
                'cross_attn_pdrop', 'resdrop', 'attndrop', 'embddrop',
                'embedding_dropout_rate', 'layerdrop', 'mlp_weight_decay', 'lm_head_weight_decay',
                'muon_beta', 'muon_gamma', 'muon_adam_wd_ratio', 'adam_eps',
                'layer_norm_epsilon', 'swiglu_clamp_limit',
            ]
            numeric_int_fields = [
                'd_model', 'd_prime', 'd_ff', 'encoder_layers', 'decoder_layers',
                'num_heads', 'sequence_length', 'batch_size', 'gradient_accumulation_steps',
                'max_tokens', 'logging_steps', 'eval_batch_size', 'eval_max_batches',
                'keep_checkpoints', 'seed', 'collator_buffer_size', 'packed_buffer_size',
                'batch_warmup_tokens', 'initial_grad_accum',
                'gram_chunk_size',
            ]

            for field in numeric_float_fields:
                if hasattr(args, field):
                    value = getattr(args, field)
                    if value is not None and not isinstance(value, (int, float)):
                        setattr(args, field, float(value))

            for field in numeric_int_fields:
                if hasattr(args, field):
                    value = getattr(args, field)
                    if value is not None and not isinstance(value, int):
                        # handle scientific notation or strings like "34000000000"
                        setattr(args, field, int(float(value)))

            log_from_main_process(logger, 'info', f"Loaded configuration from {args.config}")
        else:
            log_from_main_process(logger, 'warning', f"Could not load YAML config from {args.config}")

    # validate data_bucket is set (from CLI or YAML) unless running locally
    if not args.data_bucket and not args.smoke_test:
        parser.error("--data_bucket is required (via CLI or config file). Use --smoke_test for local runs.")

    # derive gcs_output_dir from data_bucket if not explicitly provided
    if args.gcs_output_dir is None and args.data_bucket:
        args.gcs_output_dir = f"{args.data_bucket.rstrip('/')}/{args.output_dir}"
        log_from_main_process(logger, 'info', f"Derived gcs_output_dir: {args.gcs_output_dir}")

    return args


def _parse_attention_types(value):
    """Parse attention types from CLI string or YAML list."""
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return value.split(',')
    raise ValueError(f"Invalid attention_types format: {type(value)}")


def _parse_int_list(value):
    """Parse comma-separated int list from CLI string or YAML list."""
    if value is None:
        return None
    if isinstance(value, list):
        return [int(x) for x in value]
    if isinstance(value, str):
        return [int(x) for x in value.split(',')]
    raise ValueError(f"Invalid int list format: {type(value)}")


def create_model_config(args) -> Han2HanConfig:
    """Create Han2Han model configuration."""
    if args.remat_policy is not None:
        remat_policy = args.remat_policy
    elif args.gradient_checkpointing:
        remat_policy = "full"
    else:
        remat_policy = "none"

    return Han2HanConfig(
        vocab_size=100_000,         # will be updated from tokenizer
        n_positions=args.sequence_length,
        d_model=args.d_model,
        d_prime=args.d_prime,
        d_ff=args.d_ff,
        encoder_nlayer=args.encoder_layers,
        decoder_nlayer=args.decoder_layers,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        num_kv_heads=args.num_kv_heads,
        cross_attn_num_heads=args.cross_attn_num_heads,
        cross_attn_num_kv_heads=args.cross_attn_num_kv_heads,
        use_qk_norm=args.use_qk_norm,
        query_pre_attn_scalar=args.query_pre_attn_scalar,
        attention_mechanism=args.attention_mechanism,
        encoder_attention_types=_parse_attention_types(args.encoder_attention_types),
        decoder_attention_types=_parse_attention_types(args.decoder_attention_types),
        decoder_cross_attention_types=_parse_attention_types(args.decoder_cross_attention_types),
        sliding_window_size=args.sliding_window_size,
        rope_theta=args.rope_theta,
        rope_theta_sliding=args.rope_theta_sliding,
        apply_legacy_rope_quirk=False if args.no_apply_legacy_rope_quirk else None,
        ffn_activation=args.ffn_activation,
        dense_ffn_activation=args.dense_ffn_activation,
        swiglu_clamp_limit=args.swiglu_clamp_limit,
        embedding_dropout_rate=args.embedding_dropout_rate,
        jamo_subwords=args.jamo_subwords,
        char_subwords=args.char_subwords,
        subword_embed_dim=args.subword_embed_dim,
        layer_pdrop=args.layerdrop,
        cross_attn_pdrop=args.cross_attn_pdrop,
        resid_pdrop=args.resdrop,
        attn_pdrop=args.attndrop,
        embd_pdrop=args.embddrop,
        layer_norm_epsilon=args.layer_norm_epsilon,
        tie_word_embeddings=args.tie_word_embeddings,
        tie_input_output_embeddings=args.tie_input_output_embeddings,
        tie_encoder_decoder=args.tie_encoder_decoder,
        tie_subtoken_embeddings=args.tie_subtoken_embeddings,
        decoder_start_token_id=0,   # will be updated from tokenizer
        pad_token_id=0,             # will be updated from tokenizer
        eos_token_id=1,             # will be updated from tokenizer
        bos_token_id=2,             # will be updated from tokenizer
        use_bart_training=True,
        use_sub_ln=args.use_sub_ln,
        initializer_range=args.initializer_range,
        kernel_init_type=args.kernel_init_type,
        kernel_init_scale=args.kernel_init_scale,
        init_biases_normal=args.init_biases_normal,
        decoder_norm_type=args.decoder_norm_type,
        encoder_norm_type=args.encoder_norm_type,
        use_bias=args.use_bias,
        seed=args.seed,
        label_smoothing=args.label_smoothing,
        use_scan_layers=args.use_scan_layers,
        remat_policy=remat_policy,
    )


def clip_and_norm(grads, clipnorm):
    """Clip gradients and compute norms.

    clipnorm is a jnp scalar. Pass a large value (e.g. 1e30) to effectively
    disable clipping while still computing the global norm.
    """
    global_norm = jnp.sqrt(
        sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grads))
    )
    scale = jnp.minimum(1.0, clipnorm / (global_norm + 1e-6))
    grads_scaled = jax.tree.map(lambda g: g * scale, grads)
    return grads_scaled, global_norm


# Source of truth for the muon partition classifier. Names containing
# any of these substrings, plus all <2D leaves, route to the AdamW arm; the
# rest (matmul weight matrices) route to the Muon arm. Consumed by
# create_optimizer's muon block and by per-arm gradient norm logging in
# train_step so the two stay in lockstep.
def _per_arm_grad_norms(grads):
    """Returns (muon_arm_grad_norm, adam_arm_grad_norm) over the raw gradient tree.

    Path-based classification runs at trace time; ``optax.global_norm`` does
    the sqrt(sum(sq)) on each side. Diagnostic only: matches the muon
    partition split regardless of which optimizer is actually in use, which
    is useful both for tuning the muon clipping topology and for surveying
    where gradient mass lives under non-muon optimizers.
    """
    muon_leaves, adam_leaves = [], []
    for path, g in jax.tree_util.tree_leaves_with_path(grads):
        (muon_leaves if _muon_arm_label(path, g) == 'muon' else adam_leaves).append(g)
    return optax.global_norm(muon_leaves), optax.global_norm(adam_leaves)


@partial(nnx.jit, donate_argnums=(5,))
def train_step(model: FlaxHan2Han, optimizer: nnx.Optimizer, dropout_rngs: nnx.Rngs,
               grad_clipnorm: jnp.ndarray, progress: float,
               model_inputs: Dict[str, np.ndarray | jnp.ndarray]):
    labels = model_inputs.pop("labels")

    def loss_fn(model_local: FlaxHan2Han, rngs_local: nnx.Rngs):
        """Calculate loss and diagnostics for the current model state."""
        model_local.train()  # enable dropout
        model_outputs = model_local(**model_inputs, rngs=rngs_local, deterministic=False)
        logits = model_outputs.logits

        alpha = model_local.config.label_smoothing

        safe_labels = jnp.where(labels == -100, 0, labels)

        # label-smoothed cross-entropy WITHOUT materializing [B, S, V] tensors.
        # mathematically equivalent to:
        #   one_hot = jax.nn.one_hot(safe_labels, V)
        #   smooth = (1-a)*one_hot + a/V
        #   loss = -sum(smooth * log_softmax(logits))
        #        = (1-a)*nll + a*(-mean(log_softmax))
        #        = (1-a)*nll + a*(logsumexp - mean(logits))
        # avoids 3x [B, S, 38400] f32 tensors (~3.5 GB on our v4-64 config)
        nll = optax.softmax_cross_entropy_with_integer_labels(logits, safe_labels)
        if alpha > 0:
            lse = jax.nn.logsumexp(logits, axis=-1)
            uniform_nll = lse - jnp.mean(logits, axis=-1)
            loss_val = (1 - alpha) * nll + alpha * uniform_nll
        else:
            loss_val = nll

        # create mask to ignore loss from special tokens and metadata prefixes (all -100)
        weight_mask = (labels != -100).astype(jnp.float32)
        # apply token mask
        weighted_loss = loss_val * weight_mask

        # calculate total loss and number of valid tokens for normalization
        total_loss = weighted_loss.sum()

        # count ALL non-padding tokens (encoder + decoder) for accurate LR schedule
        # encoder tokens: input_ids that aren't padding
        pad_token_id = model_local.config.pad_token_id
        encoder_mask = (model_inputs['input_ids'] != pad_token_id).astype(jnp.float32)
        encoder_tokens = encoder_mask.sum()

        # decoder tokens: labels that aren't -100 (ignore index)
        decoder_tokens = weight_mask.sum()

        # total tokens processed in this batch (for LR schedule)
        valid_tokens = encoder_tokens + decoder_tokens

        normalized_count = jnp.maximum(decoder_tokens, 1e-8)  # safe denominator (use decoder tokens only for loss normalization)

        # compute per-sample losses for task-specific tracking
        # loss_val shape: (batch, seq_len), weight_mask shape: (batch, seq_len)
        per_sample_loss = jnp.sum(loss_val * weight_mask, axis=1) / jnp.maximum(
            jnp.sum(weight_mask, axis=1), 1.0
        )  # (batch,)

        # compute per-document losses within packed sequences
        dec_seg_ids = model_inputs.get('decoder_segment_ids', None)
        if dec_seg_ids is not None:
            max_segs = 32    # static segment_sum buffer for per-document loss diagnostics only
            # segment_sum over axis=-1: index 0 is padding, docs are 1..max_segs
            seg_loss_sums = jax.vmap(
                lambda wl, si: jax.ops.segment_sum(wl, si, num_segments=max_segs + 1)
            )(weighted_loss, dec_seg_ids)  # (batch, max_segs+1)
            seg_counts = jax.vmap(
                lambda wm, si: jax.ops.segment_sum(wm, si, num_segments=max_segs + 1)
            )(weight_mask, dec_seg_ids)  # (batch, max_segs+1)
            # slice off segment 0 (padding) -> (batch, max_segs)
            seg_loss_sums = seg_loss_sums[:, 1:]
            seg_counts = seg_counts[:, 1:]
            per_doc_loss = seg_loss_sums / jnp.maximum(seg_counts, 1.0)
            per_doc_valid = seg_counts > 0
        else:
            per_doc_loss = per_sample_loss[:, None]
            per_doc_valid = jnp.ones_like(per_doc_loss, dtype=jnp.bool_)

        # collect diagnostic info
        diagnostics = {
            "logits_max": jnp.max(logits),
            "logits_min": jnp.min(logits),
            "logits_mean": jnp.mean(logits),
            "raw_loss_max": jnp.max(loss_val * weight_mask),
            "raw_loss_min": jnp.min(loss_val * weight_mask),
            "raw_loss_mean": jnp.mean(loss_val * weight_mask),
            "valid_token_count": valid_tokens,
            "per_sample_loss": per_sample_loss,
            "per_doc_loss": per_doc_loss,
            "per_doc_valid": per_doc_valid,
        }

        # return average loss and diagnostics
        avg_loss = total_loss / normalized_count
        return avg_loss, diagnostics

    # gradient calculation
    # differentiate wrt the specific parameters for optimizer
    grad_fn = nnx.value_and_grad(
        loss_fn, has_aux=True,
        argnums=nnx.DiffState(0, optimizer.wrt)
    )
    (loss, diagnostics), grad = grad_fn(model, dropout_rngs)

    # perform safety checks
    is_nan = jnp.isnan(loss)
    is_finite = jnp.logical_not(is_nan)

    grad_is_finite = jax.tree.reduce(
        lambda acc, x: jnp.logical_and(acc, jnp.all(jnp.isfinite(x))),
        grad,
        initializer=True
    )

    safe_update = jnp.logical_and(is_finite, grad_is_finite)

    grad = jax.tree.map(
        lambda g: jnp.where(safe_update, g, jnp.zeros_like(g)),
        grad
    )

    # per-arm gradient norms on the UNCLIPPED grad: diagnoses which partition
    # (muon vs adamw under --optimizer muon) drives global_norm growth
    # before the clip. Use these to decide whether to move the clip inside
    # the AdamW arm only.
    muon_arm_grad_norm, adam_arm_grad_norm = _per_arm_grad_norms(grad)

    # external clip disabled (grad_clipnorm=1e30); clip_and_norm just computes the global
    # grad norm here. real clipping is applied to the AdamW arm inside the optimizer.
    clipped_grad, grad_norm = clip_and_norm(grad, grad_clipnorm)

    # update optimizer with progress as extra_arg for progress-based schedule
    optimizer.update(model, clipped_grad, progress=progress)

    # consolidate metrics (learning_rate computed outside JIT for logging)
    metrics = {
        "loss": loss,
        "grad_norm": grad_norm,
        "muon_arm_grad_norm": muon_arm_grad_norm,
        "adam_arm_grad_norm": adam_arm_grad_norm,
        **diagnostics
    }
    return metrics


def run_evaluation(model: FlaxHan2Han, collator: MultilingualCollator,
                   mesh: Mesh, args: Any):
    """Run distributed evaluation on all processes and return aggregated metrics.

    Uses the collator's eval_data iterator directly with simple batching (no packing).
    """
    log_from_main_process(logger, 'info', f"Running evaluation on {args.eval_max_batches} batches...")

    # simple eval batching - no packing, no buffering, just pad and batch
    def create_eval_batches():
        batch_examples = []
        expected_keys = {'input_ids', 'decoder_input_ids', 'labels', 'attention_mask', 'decoder_attention_mask'}

        for example in collator.eval_data:
            # pop metadata fields
            example.pop('_data_source', None)
            example.pop('_source', None)
            example.pop('_source_name', None)

            # pad to sequence_length
            for key, value in example.items():
                if isinstance(value, list):
                    arr = np.array(value, dtype=np.int32)
                    if len(arr) < args.sequence_length:
                        pad_value = -100 if key == 'labels' else 0
                        arr = np.pad(arr, (0, args.sequence_length - len(arr)), constant_values=pad_value)
                    example[key] = arr[:args.sequence_length]

            # filter to expected keys only
            example = {k: v for k, v in example.items() if k in expected_keys}
            batch_examples.append(example)

            if len(batch_examples) >= args.eval_batch_size:
                batch = {key: np.stack([ex[key] for ex in batch_examples], axis=0, dtype=np.int32)
                        for key in batch_examples[0].keys()}
                yield batch
                batch_examples = []

    eval_batches = create_eval_batches()
    if not eval_batches:
        log_from_main_process(logger, 'warning', "No evaluation batches created, skipping evaluation")
        return {}

    # run evaluation on all processes (distributed)
    total_eval_metrics = []

    eval_pbar = tqdm(total=args.eval_max_batches, desc="Evaluating...", position=2, 
                     leave=False, disable=not (jax.process_index() == 0))
    for batch_idx, batch in enumerate(eval_batches):
        if batch_idx > args.eval_max_batches:
            break

        # remove non-array fields before sharding/eval (like training loop does)
        batch.pop('_batch_sources', None)
        batch.pop('_batch_training_modes', None)

        # shard batch across devices
        sharded_batch = shard_batch_to_devices(batch, mesh)

        # run evaluation step on all processes
        with mesh:
            eval_metrics = eval_step(model, sharded_batch)

        # collect metrics from device
        eval_metrics_cpu = jax.device_get(eval_metrics)
        total_eval_metrics.append(eval_metrics_cpu)

        eval_pbar.update(1)

    eval_pbar.close()

    # aggregate evaluation metrics across all batches
    if total_eval_metrics:
        # compute weighted averages based on token counts
        total_tokens = sum(m['eval/total_tokens'] for m in total_eval_metrics)
        total_loss = sum(m['eval/total_loss'] for m in total_eval_metrics)
        total_correct = sum(m['eval/total_correct'] for m in total_eval_metrics)

        if total_tokens > 0:
            aggregated_metrics = {
                'eval/loss': float(total_loss / total_tokens),
                'eval/accuracy': float(total_correct / total_tokens),
                'eval/perplexity': float(np.exp(total_loss / total_tokens)),
                'eval/total_tokens': float(total_tokens),
                'eval/batches': len(total_eval_metrics)
            }
        else:
            aggregated_metrics = {
                'eval/loss': 0.0,
                'eval/accuracy': 0.0, 
                'eval/perplexity': 1.0,
                'eval/total_tokens': 0.0,
                'eval/batches': 0
            }

        log_from_main_process(logger, 'info',
            f"Evaluation completed: loss={aggregated_metrics['eval/loss']:.4f}, "
            f"acc={aggregated_metrics['eval/accuracy']:.3f}, "
            f"ppl={aggregated_metrics['eval/perplexity']:.2f}, "
            f"tokens={aggregated_metrics['eval/total_tokens']:.0f}")

        return aggregated_metrics
    else:
        return {}


@nnx.jit
def eval_step(model, batch):
    """Distributed evaluation step that runs on all processes."""
    # make a copy to avoid mutating original batch
    eval_batch = {k: v for k, v in batch.items()}
    labels = eval_batch.pop("labels")

    model.eval()  # disable dropout
    outputs = model(**eval_batch, deterministic=True)
    logits = outputs.logits

    # compute loss - log_softmax for stability, tpu-efficient gather for precision
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    labels_expanded = jnp.expand_dims(labels, axis=-1)
    target_log_probs = jnp.take_along_axis(log_probs, labels_expanded, axis=-1).squeeze(-1)

    # no label smoothing during evaluation - use hard targets
    loss_val = -target_log_probs

    # mask out padding tokens and metadata  
    weight_mask = (labels != -100).astype(jnp.float32)
    weighted_loss = loss_val * weight_mask

    # compute accuracy
    predictions = jnp.argmax(logits, axis=-1)
    correct = (predictions == labels) * weight_mask

    # compute metrics
    total_loss = jnp.sum(weighted_loss)
    total_correct = jnp.sum(correct)
    total_tokens = jnp.sum(weight_mask)

    # avoid division by zero
    safe_tokens = jnp.maximum(total_tokens, 1.0)
    avg_loss = total_loss / safe_tokens
    accuracy = total_correct / safe_tokens

    # compute perplexity
    perplexity = jnp.exp(avg_loss)

    return {
        "eval/loss": avg_loss,
        "eval/accuracy": accuracy,
        "eval/perplexity": perplexity,
        "eval/total_tokens": total_tokens,
        "eval/total_loss": total_loss,
        "eval/total_correct": total_correct
    }



def get_streaming_datasets(args):
    """Get streaming datasets for multilingual training.

    Returns:
        tuple: (datasets, sampling_ratios, source_configs) or single dict for smoke test
    """

    # use test samples for local testing with all data sources
    if args.smoke_test == 'test_cases':
        from tests.test_unified_collator import load_test_datasets_for_training
        log_from_main_process(logger, 'info', "Loading local test samples for all data sources")
        return load_test_datasets_for_training(n_samples_per_source=100000)

    # use dummy datasets for smoke test mode
    elif isinstance(args.smoke_test, bool) and args.smoke_test:
        return None  # dummy collator handles its own data
    elif isinstance(args.smoke_test, str) and os.path.exists(str(args.smoke_test)):
        from datasets import Dataset
        import polars as pl
        if args.smoke_test.endswith('parquet'): reader = pl.read_parquet
        else: reader = pl.read_ipc
        df = reader(args.smoke_test).sample(3_000_000, shuffle=True, seed=args.seed)
        ds = Dataset.from_polars(df.with_columns(pl.Series(['han2han_curated']*len(df))))
        log_from_main_process(logger, 'info', f"Loaded smoke test data, total {df['sequence_length'].sum()} "
                                              f"tokens in one epoch")
        ds.info.source_type = "han2han_curated"
        ds.info.data_type = "denoising"

        # create source config for smoke test
        from dynamic_data_loader import DataSourceConfig
        smoke_config = DataSourceConfig(
            name='han2han_curated',
            gcs_pattern='',  # local smoke test
            weight=1.0,
            data_type='denoising',
            text_field='original_text',
            metadata_field='metadata',
        )

        datasets = {"han2han_curated": ds}
        # for korean datasets, collator expects hanja_heavy/light split in sampling_ratios
        sampling_ratios = {
            'korean_hanja_heavy': 0.2,  # 20% hanja-heavy samples
            'korean_hanja_light': 0.8,  # 80% hanja-light samples
        }
        source_configs = {"han2han_curated": smoke_config}
        eval_datasets = {}  # no pre-loaded eval for smoke test

        return datasets, sampling_ratios, source_configs, eval_datasets
    else:
        log_from_main_process(logger, 'info', "Creating streaming datasets from GCS")

        # import dataset loaders
        from dynamic_data_loader import (
            get_han2han_datasets,
            get_eval_sources_from_configs,
            DynamicDataLoader,
        )

        # determine host info for distributed loading
        if jax.process_count() > 1:
            host_idx = jax.process_index()
            num_hosts = jax.process_count()
        else:
            host_idx = 0
            num_hosts = 1

        # load Korean-only (Han2Han) data sources
        log_from_main_process(logger, 'info', "Using Han2Han (Korean-only) data sources")
        sft_tasks = args.sft_tasks
        datasets, sampling_ratios, source_configs = get_han2han_datasets(
            host_idx=host_idx,
            num_hosts=num_hosts,
            training_mode=args.training_mode,
            disable_budget_limit=args.disable_budget_limit,
            force_reload=args.force_reload,
            data_bucket=args.data_bucket,
            sft_tasks=sft_tasks,
        )

        log_from_main_process(logger, 'info', f"Created {len(datasets)} streaming datasets: {list(datasets.keys())}")
        formatted_ratios = {k: f"{v:.3f}" for k, v in sampling_ratios.items()}
        log_from_main_process(logger, 'info', f"Sampling ratios: {formatted_ratios}")

        # load eval datasets for sources with stratified splits
        eval_datasets = {}

        eval_sources = get_eval_sources_from_configs(list(source_configs.values()))
        if eval_sources:
            log_from_main_process(logger, 'info', f"Loading {len(eval_sources)} pre-defined eval datasets")
            eval_loader = DynamicDataLoader(
                disable_budget_limit=True,
                data_bucket=args.data_bucket,
            )
            eval_data = eval_loader.load_all_sources(
                sources=eval_sources,
                host_idx=host_idx,
                num_hosts=num_hosts,
                training_mode='debug',
                force_reload=False,
            )
            for source in eval_sources:
                if source.name in eval_data:
                    ds = eval_data[source.name]
                    ds.info.source_type = source.name.replace('_eval', '')
                    ds.info.data_type = source.data_type
                    eval_datasets[source.name] = ds
                    log_from_main_process(logger, 'info',
                        f"Loaded eval dataset {source.name}: {len(ds):,} examples")

        return datasets, sampling_ratios, source_configs, eval_datasets


def setup_data_pipeline(args, tokenizer, for_eval=False,
                        eval_data=None, eval_iterators=None,
                        cooldown_phase=True, streaming_datasets=None,
                        sampling_ratios=None, source_configs=None,
                        max_length_override=None):
    """Setup multilingual data pipeline with streaming datasets and train/eval splitting.

    Args:
        max_length_override: If provided, use this instead of args.sequence_length.
            Useful for eval collators that need shorter sequences (e.g., 128 for generation).
    """

    # use dummy collator for smoke test mode
    if isinstance(args.smoke_test, bool) and args.smoke_test:
        from multilingual_collator import create_dummy_collator
        log_from_main_process(logger, 'info', "Smoke test mode enabled: using dummy collator")
        collator = create_dummy_collator(args, tokenizer)
        if cooldown_phase:
            collator._instantiate_dsets(cooldown_phase=True)
        return collator

    # get streaming datasets if not provided
    # if for_eval=True and eval_data provided, reuse train data instead of reloading
    eval_datasets_loaded = {}
    if streaming_datasets is None and not (for_eval and eval_data is not None):
        result = get_streaming_datasets(args)
        # handle tuple return (production) vs dict return (smoke test)
        if isinstance(result, tuple):
            if len(result) == 4:
                streaming_datasets, sampling_ratios, source_configs, eval_datasets_loaded = result
            else:
                streaming_datasets, sampling_ratios, source_configs = result
                eval_datasets_loaded = {}
        else:
            streaming_datasets = result
            # fallback to default ratios for smoke test
            sampling_ratios = {
                'korean_hanja_heavy': 0.10,
                'korean_hanja_light': 0.45,
                'c4_en':              0.10,
                'mc4_zh':             0.15,
                'mc4_ja':             0.20,
            }
            source_configs = None
            eval_datasets_loaded = {}

    # use provided ratios or fallback
    if sampling_ratios is None:
        sampling_ratios = {
            'korean_hanja_heavy': 0.10,
            'korean_hanja_light': 0.45,
            'c4_en':              0.10,
            'mc4_zh':             0.15,
            'mc4_ja':             0.20,
        }

    # parse UL2-style multi-config denoising parameters BEFORE building collator_kwargs
    def parse_multi_config(value, field_name):
        """Parse a config value that could be single float or comma-separated string."""
        if isinstance(value, (int, float)):
            return [float(value)]
        if isinstance(value, str):
            return [float(x.strip()) for x in value.split(',')]
        raise ValueError(f"Invalid {field_name}: {value}")

    def parse_poisson_lambda(value):
        """Parse poisson_lambda which has 'r_configs;x_configs' format."""
        if isinstance(value, (int, float)):
            return [float(value)], [float(value)]
        if isinstance(value, str):
            if ';' in value:
                parts = value.split(';')
                r_lambdas = [float(x.strip()) for x in parts[0].split(',')]
                x_lambdas = [float(x.strip()) for x in parts[1].split(',')]
                return r_lambdas, x_lambdas
            else:
                lambdas = [float(x.strip()) for x in value.split(',')]
                return lambdas, lambdas
        raise ValueError(f"Invalid poisson_lambda: {value}")

    # parse all multi-config values
    r_lambdas, x_lambdas = parse_poisson_lambda(args.poisson_lambda)
    r_ratios = parse_multi_config(args.infilling_ratio, 'infilling_ratio')
    x_ratios = parse_multi_config(args.heavy_infilling_ratio, 'heavy_infilling_ratio')
    m_lambdas = parse_multi_config(args.morpheme_lambda, 'morpheme_lambda')

    # validate lengths match
    if len(r_lambdas) != len(r_ratios):
        raise ValueError(f"poisson_lambda field 1 ({len(r_lambdas)}) must match infilling_ratio ({len(r_ratios)})")
    if len(x_lambdas) != len(x_ratios):
        raise ValueError(f"poisson_lambda field 2 ({len(x_lambdas)}) must match heavy_infilling_ratio ({len(x_ratios)})")
    if len(m_lambdas) != len(r_ratios):
        raise ValueError(f"morpheme_lambda ({len(m_lambdas)}) must match infilling_ratio ({len(r_ratios)})")

    # combine into config tuples for Phase2 collator
    r_denoiser_configs = list(zip(r_lambdas, r_ratios))
    x_denoiser_configs = list(zip(x_lambdas, x_ratios))
    morpheme_denoiser_configs = list(zip(m_lambdas, r_ratios))

    # log UL2-style configs if multi-config detected
    if len(r_denoiser_configs) > 1:
        log_from_main_process(logger, 'info', f"UL2-style R-denoiser configs: {r_denoiser_configs}")
    if len(x_denoiser_configs) > 1:
        log_from_main_process(logger, 'info', f"UL2-style X-denoiser configs: {x_denoiser_configs}")
    if len(morpheme_denoiser_configs) > 1:
        log_from_main_process(logger, 'info', f"UL2-style morpheme configs: {morpheme_denoiser_configs}")

    # use max_encoder_length/max_decoder_length if available, fall back to sequence_length
    default_encoder_length = getattr(args, 'max_encoder_length', args.sequence_length)
    default_decoder_length = getattr(args, 'max_decoder_length', args.sequence_length)
    effective_max_length = max_length_override if max_length_override is not None else default_encoder_length
    collator_kwargs = {
        'tokenizer': tokenizer,
        'rng': np.random.default_rng(args.seed + 1000),
        'max_length': effective_max_length,
        'model_max_length': effective_max_length,
        'seed': args.seed,
        'batch_size': args.batch_size,
        'buffer_size': args.collator_buffer_size,
        'eval_batch_size': args.eval_batch_size,
        'eval_split_ratio': args.eval_split_ratio,
        'use_bucketing': False,
        'adaptive_batch_size': False,
        'infilling_ratio': r_ratios[0],  # base value for parent class compatibility
        'sentence_permutation': args.sentence_permutation,
        'poisson_lambda': r_lambdas[0],  # base value for parent class compatibility
        'morpheme_lambda': m_lambdas[0], # base value for parent class compatibility
        'use_morpheme_masking': 0.0,     # legacy
        'hangul_decoder': False,         # legacy
        'hangul_only': False,            # legacy
        'hanja_heavy_threshold': 0.18,
        'sampling_ratios': sampling_ratios,
        'han2han_transcription_ratio': args.han2han_transcription_ratio if args.han2han_transcription_ratio > 0 else None,
    }

    # use unified collator if source_configs provided (production pipeline)
    if source_configs is not None and not for_eval:
        log_from_main_process(logger, 'info', "Using UnifiedCollator for production pipeline")
        collator_kwargs['datasets'] = streaming_datasets
        collator_kwargs['source_configs'] = source_configs
        collator_kwargs['sampling_ratios'] = sampling_ratios
        collator_kwargs['eval_datasets'] = eval_datasets_loaded
        collator_kwargs['use_task_prompts'] = True

        # always pass enable_packing (don't rely on default which is True)
        collator_kwargs['enable_packing'] = args.enable_packing
        if args.enable_packing:
            collator_kwargs['packing_efficiency_threshold'] = args.packing_efficiency_threshold
            collator_kwargs['packed_buffer_size'] = args.packed_buffer_size
            log_from_main_process(logger, 'info', "Packing enabled for UnifiedCollator")
        else:
            log_from_main_process(logger, 'info', "Packing disabled for UnifiedCollator")

        # parse mode ratios (UL2-style: denoising, denoising_heavy, continuation)
        ratios_str = args.mode_ratios.split(',')
        if len(ratios_str) == 3:
            mode_ratios = {
                'denoising': float(ratios_str[0]),
                'denoising_heavy': float(ratios_str[1]),
                'continuation': float(ratios_str[2]),
            }
        elif len(ratios_str) == 2:
            mode_ratios = {
                'denoising': float(ratios_str[0]),
                'denoising_heavy': 0.0,
                'continuation': float(ratios_str[1]),
            }
        else:
            raise ValueError(f"mode_ratios must have 2 or 3 values, got: {args.mode_ratios}")
        collator_kwargs['mode_ratios'] = mode_ratios
        collator_kwargs['morpheme_denoising_ratio'] = args.morpheme_denoising_ratio
        collator_kwargs['sentinel_denoising_ratio'] = args.sentinel_denoising_ratio
        collator_kwargs['heavy_infilling_ratio'] = args.heavy_infilling_ratio
        collator_kwargs['byte_reconstruction_ratio'] = args.byte_reconstruction_ratio
        collator_kwargs['temporal_continuation_ratio'] = args.temporal_continuation_ratio
        collator_kwargs['max_encoder_length'] = default_encoder_length
        collator_kwargs['max_decoder_length'] = default_decoder_length
        collator_kwargs['max_length'] = default_encoder_length
        log_from_main_process(logger, 'info', f"UL2 mode ratios: {mode_ratios}")
        log_from_main_process(logger, 'info', f"Morpheme denoising ratio: {args.morpheme_denoising_ratio}")
        log_from_main_process(logger, 'info', f"Byte reconstruction ratio: {args.byte_reconstruction_ratio}")
        log_from_main_process(logger, 'info', f"Temporal continuation ratio: {args.temporal_continuation_ratio}")
        log_from_main_process(logger, 'info', f"Sentinel denoising ratio: {args.sentinel_denoising_ratio}")

        # add UL2-style denoiser configs
        collator_kwargs['r_denoiser_configs'] = r_denoiser_configs
        collator_kwargs['x_denoiser_configs'] = x_denoiser_configs
        collator_kwargs['morpheme_denoiser_configs'] = morpheme_denoiser_configs

        collator = UnifiedCollator(**collator_kwargs)

    # use Phase2MixedCollator if requested (legacy mode)
    elif args.use_phase2_collator and not for_eval:
        if args.enable_packing:
            from packed_multilingual_collator import PackedMultilingualCollator
            log_from_main_process(logger, 'info',
                                 "Using PackedMultilingualCollator with generator-level packing (100% efficiency!)")
        else:
            from phase2_collator import Phase2MixedCollator
            log_from_main_process(logger, 'info', "Using Phase2MixedCollator for continuation training")

        # parse mode ratios (UL2-style: denoising, denoising_heavy, continuation)
        ratios_str = args.mode_ratios.split(',')
        if len(ratios_str) == 3:
            mode_ratios = {
                'denoising': float(ratios_str[0]),
                'denoising_heavy': float(ratios_str[1]),
                'continuation': float(ratios_str[2]),
            }
        elif len(ratios_str) == 2:
            mode_ratios = {
                'denoising': float(ratios_str[0]),
                'denoising_heavy': 0.0,
                'continuation': float(ratios_str[1]),
            }
        else:
            raise ValueError(f"mode_ratios must have 2 or 3 values, got: {args.mode_ratios}")
        collator_kwargs['mode_ratios'] = mode_ratios
        collator_kwargs['morpheme_denoising_ratio'] = args.morpheme_denoising_ratio
        collator_kwargs['sentinel_denoising_ratio'] = args.sentinel_denoising_ratio
        collator_kwargs['heavy_infilling_ratio'] = args.heavy_infilling_ratio
        collator_kwargs['byte_reconstruction_ratio'] = args.byte_reconstruction_ratio
        collator_kwargs['temporal_continuation_ratio'] = args.temporal_continuation_ratio
        # pass UL2-style denoiser configs
        collator_kwargs['r_denoiser_configs'] = r_denoiser_configs
        collator_kwargs['x_denoiser_configs'] = x_denoiser_configs
        collator_kwargs['morpheme_denoiser_configs'] = morpheme_denoiser_configs
        # don't include datasets in kwargs - we'll set it after initialization
        collator_kwargs['max_encoder_length'] = default_encoder_length
        collator_kwargs['max_decoder_length'] = default_decoder_length
        collator_kwargs['max_length'] = default_encoder_length
        log_from_main_process(logger, 'info', f"UL2 mode ratios: {mode_ratios}")
        log_from_main_process(logger, 'info', f"Morpheme denoising ratio: {args.morpheme_denoising_ratio}")
        log_from_main_process(logger, 'info', f"Byte reconstruction ratio: {args.byte_reconstruction_ratio}")
        log_from_main_process(logger, 'info', f"Temporal continuation ratio: {args.temporal_continuation_ratio}")
        log_from_main_process(logger, 'info', f"Sentinel denoising ratio: {args.sentinel_denoising_ratio}")

        # choose collator class based on packing flag
        CollatorClass = PackedMultilingualCollator if args.enable_packing else Phase2MixedCollator

        # add packing-specific arguments if enabled
        if args.enable_packing:
            collator_kwargs['enable_packing'] = True
            collator_kwargs['packing_efficiency_threshold'] = args.packing_efficiency_threshold
            collator_kwargs['packed_buffer_size'] = args.packed_buffer_size  # larger buffer for better packing opportunities

        collator = CollatorClass(**collator_kwargs)

        # set datasets and instantiate with correct cooldown phase
        collator.datasets = streaming_datasets
        collator._instantiate_dsets(cooldown_phase=cooldown_phase)

    elif for_eval:
        # use same collator as training to ensure train/test consistency with task prompts
        if source_configs is not None:
            log_from_main_process(logger, 'info', "Using UnifiedCollator for eval (with task prompts, packing disabled)")
            collator_kwargs['datasets'] = streaming_datasets
            collator_kwargs['source_configs'] = source_configs
            collator_kwargs['sampling_ratios'] = sampling_ratios
            collator_kwargs['eval_datasets'] = eval_datasets_loaded
            collator_kwargs['use_task_prompts'] = not cooldown_phase  # match training behavior
            collator_kwargs['enable_packing'] = False  # disable packing for eval
            collator = UnifiedCollator(**collator_kwargs)

            # if we have datasets but no pre-existing eval_data, initialize iterators
            # this is the SFT eval collator path - needs its own iterators
            if streaming_datasets is not None and eval_data is None:
                log_from_main_process(logger, 'info', "Initializing SFT eval collator iterators (eval_split_ratio=0.99)")
                # NOTE: 0.99 because 1.0 breaks train_test_split (doesn't handle empty train set)
                collator.eval_split_ratio = 0.99
                collator._instantiate_dsets(cooldown_phase=cooldown_phase)
        else:
            log_from_main_process(logger, 'info', "Using MultilingualCollator for eval")
            collator = MultilingualCollator(**collator_kwargs)

        # only assign external eval_data/iterators if provided (legacy path for BLEU/generation callbacks)
        if eval_data is not None:
            collator.eval_data = eval_data
        if eval_iterators is not None:
            collator.eval_iterators = eval_iterators
        collator.cooldown_phase = cooldown_phase

    else:
        log_from_main_process(logger, 'info', "Using MultilingualCollator for denoising training")
        collator = MultilingualCollator(**collator_kwargs)

        # set datasets and instantiate with correct cooldown phase
        if streaming_datasets:
            collator.datasets = streaming_datasets
            collator.eval_datasets = eval_datasets_loaded
            collator._instantiate_dsets(cooldown_phase=cooldown_phase)

    return collator


def training_loop(
    model: FlaxHan2Han,
    collator: MultilingualCollator,
    optimizer: nnx.Optimizer,
    args: Any,
    mesh: Any,
    data_sharding: PartitionSpec,
    checkpoint_manager,
    tokens_seen: int,
    global_step: int,
    callbacks=None,
    resumed_accum_counter: int = 0,
    resumed_micro_step: int = None,
    lr_scheduler=None,
):
    """Main token-based training loop with JAX SPMD."""

    log_from_main_process(logger, 'info', "Starting token-based training loop")
    log_from_main_process(logger, 'info', f"Data sharding: {data_sharding}")

    # setup wandb
    if args.wandb_project and WANDB_AVAILABLE and jax.process_index() == 0:
        try:
            wandb_mode = "online" if args.wandb_online else "offline"
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name,
                config=vars(args),
                mode=wandb_mode
            )
            wandb.run.define_metric("tokens_seen")
            wandb.run.define_metric("*", step_metric="tokens_seen")
            log_from_main_process(logger, 'info', f"WandB initialized in {wandb_mode} mode")
        except Exception as e:
            log_from_main_process(logger, 'warning', f"WandB initialization failed: {e}")

    # token-based tracking - tokens_seen and global_step already set from checkpoint restoration
    train_metrics = []
    task_losses = {}  # track per-source losses for debugging data quality
    mode_losses = {}  # track per-training-mode losses (granular: denoising_bart, morpheme_sentinel, etc.)
    length_losses = {}  # per-document losses bucketed by content length
    length_counts = {}  # document counts per length bucket for the current logging window

    # content tokens start after special (0-19), sentinel (20-275), byte fallback (276-531)
    CONTENT_TOKEN_MIN_ID = 532

    LENGTH_BUCKETS = [
        (0, 1024, '0-1k'),
        (1024, 2048, '1k-2k'),
        (2048, 4096, '2k-4k'),
        (4096, 8192, '4k-8k'),
        (8192, 16384, '8k-16k'),
    ]

    def _length_bucket(n_content_tokens):
        for lo, hi, name in LENGTH_BUCKETS:
            if lo <= n_content_tokens < hi:
                return name
        return '16k+'

    # token-based evaluation and saving thresholds
    eval_token_interval = int(args.max_tokens * args.eval_ratio)
    save_token_interval = int(args.max_tokens * args.save_ratio)

    # calculate next milestones based on current position
    if tokens_seen > 0:
        # recalculate next token-based milestones after restore
        next_eval_tokens = ((tokens_seen // eval_token_interval) + 1) * eval_token_interval
        next_save_tokens = ((tokens_seen // save_token_interval) + 1) * save_token_interval
        log_from_all_processes(logger, 'info', f"Next eval at {next_eval_tokens:_} tokens, next save at {next_save_tokens:_} tokens")
    else:
        next_eval_tokens = eval_token_interval
        next_save_tokens = save_token_interval

    log_from_main_process(logger, 'info', f"=== TOKEN-BASED TRAINING ===")
    warmup_tokens = int(args.max_tokens * args.warmup_ratio)
    constant_tokens = int(args.max_tokens * args.constant_ratio)
    cooldown_tokens = int(args.max_tokens * (1 - args.cooldown_ratio))

    log_from_main_process(logger, 'info', f"Training for {args.max_tokens/1e9:.1f}B tokens")
    log_from_main_process(logger, 'info', f"Warmup: {warmup_tokens/1e9:.1f}B tokens ({args.warmup_ratio:.1%})")
    if args.constant_ratio > 0:
        log_from_main_process(logger, 'info', f"Constant LR: {constant_tokens/1e9:.1f}B tokens ({args.constant_ratio:.1%}) - T5-style")
    log_from_main_process(logger, 'info', f"Cooldown starts: {cooldown_tokens/1e9:.1f}B tokens ({1-args.cooldown_ratio:.1%})")
    log_from_main_process(logger, 'info', f"Eval every: {eval_token_interval/1e9:.1f}B tokens ({args.eval_ratio:.1%})")
    log_from_main_process(logger, 'info', f"Save every: {save_token_interval/1e9:.1f}B tokens ({args.save_ratio:.1%})")

    # track phase transitions
    cooldown_phase_active = False
    in_eval = False

    def create_batches(dataset, batch_size):
        """Manual batching from unbatched dataset to avoid datasets.batch() scalar issues."""
        batch_examples = []
        batch_sources = []  # track data sources for per-source loss reporting
        batch_training_modes = []  # track training modes for per-task loss reporting

        # determine expected keys based on packing
        if args.enable_packing and not in_eval:
            expected_keys = {'input_ids', 'decoder_input_ids', 'labels', 'attention_mask', 'decoder_attention_mask',
                           'segment_ids', 'decoder_segment_ids', 'position_ids', 'decoder_position_ids'}
        else:
            expected_keys = {'input_ids', 'decoder_input_ids', 'labels', 'attention_mask', 'decoder_attention_mask'}

        for example in dataset:
            # extract data source for per-source loss tracking
            data_source = example.pop('_data_source', 'unknown')
            batch_sources.append(data_source)

            # extract training mode for per-task loss tracking (set by collator with granular suffixes)
            training_mode = example.pop('_training_mode', None)
            if training_mode is None:
                # decode input for debugging
                input_ids = example.get('input_ids', [])
                if hasattr(input_ids, 'tolist'):
                    input_ids = input_ids.tolist()
                first_50 = input_ids[:50] if len(input_ids) >= 50 else input_ids
                decoded = collator.tokenizer.decode(first_50, skip_special_tokens=False)
                raise ValueError(
                    f"Missing '_training_mode' field in example from source '{data_source}'. "
                    f"Example keys: {list(example.keys())}. "
                    f"Decoded input preview: {decoded[:200]}..."
                )
            batch_training_modes.append(training_mode)

            # pad lists to sequence_length if needed
            for key, value in example.items():
                if isinstance(value, list):
                    arr = np.array(value, dtype=np.int32)
                    if len(arr) < args.sequence_length:
                        pad_value = -100 if key == 'labels' else 0
                        arr = np.pad(arr, (0, args.sequence_length - len(arr)), constant_values=pad_value)
                    example[key] = arr[:args.sequence_length]

            # filter to expected keys only
            example = {k: v for k, v in example.items() if k in expected_keys}

            # validate: skip examples missing required keys (rare packing pipeline leak)
            missing_keys = expected_keys - example.keys()
            if missing_keys:
                log_from_main_process(logger, 'error',
                    f"[BATCH VALIDATION] Dropping example missing keys {missing_keys} "
                    f"(source={data_source}, mode={training_mode}, "
                    f"got_keys={set(example.keys())})")
                batch_sources.pop()
                batch_training_modes.pop()
                continue

            # validate: skip examples with wrong dimensionality (e.g. 2D from _create_empty_batch)
            bad_shape = False
            for key, value in example.items():
                arr = np.asarray(value)
                if arr.ndim != 1:
                    log_from_main_process(logger, 'error',
                        f"[BATCH VALIDATION] Dropping example with {arr.ndim}D array for '{key}' "
                        f"(shape={arr.shape}, source={data_source}, mode={training_mode})")
                    bad_shape = True
                    break
            if bad_shape:
                batch_sources.pop()
                batch_training_modes.pop()
                continue

            batch_examples.append(example)

            if len(batch_examples) >= batch_size:
                # stack examples into batch
                try:
                    batch = {key: np.stack([ex[key] for ex in batch_examples], axis=0, dtype=np.int32)
                            for key in batch_examples[0].keys()}
                except (ValueError, KeyError) as e:
                    log_from_main_process(logger, 'error', f"Batch stacking failed: {type(e).__name__}: {e}")
                    all_keys = [set(ex.keys()) for ex in batch_examples]
                    key_sets = set(frozenset(ks) for ks in all_keys)
                    if len(key_sets) > 1:
                        log_from_main_process(logger, 'error', f"  Inconsistent key sets across batch: {key_sets}")
                    for key in batch_examples[0].keys():
                        shapes = []
                        for ex in batch_examples:
                            if key in ex:
                                shapes.append(np.asarray(ex[key]).shape)
                            else:
                                shapes.append(None)
                        unique_shapes = set(shapes)
                        if len(unique_shapes) > 1:
                            log_from_main_process(logger, 'error', f"  Key '{key}' has mismatched shapes: {unique_shapes}")
                            for i, shape in enumerate(shapes):
                                if shape != shapes[0]:
                                    src = batch_sources[i] if i < len(batch_sources) else 'unknown'
                                    mode = batch_training_modes[i] if i < len(batch_training_modes) else 'unknown'
                                    log_from_main_process(logger, 'error', f"    Example {i}: shape={shape}, source={src}, mode={mode}")
                    raise

                # create 2D attention masks from segment_ids if packing enabled
                processed_batch = collator.create_packed_attention_masks(batch)

                # convert JAX arrays to numpy for tf.data compatibility
                for key, value in processed_batch.items():
                    if hasattr(value, 'device'):
                        processed_batch[key] = np.asarray(value)

                # add source and training mode information as numpy string arrays
                processed_batch['_batch_sources'] = np.array(batch_sources, dtype=object)
                processed_batch['_batch_training_modes'] = np.array(batch_training_modes, dtype=object)

                yield processed_batch

                batch_examples = []
                batch_sources = []
                batch_training_modes = []

    # simple parameter count instead of tabulate (Flax 0.12 compatibility)
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param)))
    log_from_main_process(logger, 'info', f"Model initialized with {param_count/1e6:.1f}M parameters")

    # bake the model and optimizer into the training step
    # to prevent unnecessary Python graph traversals
    cached_train_step = nnx.cached_partial(
        train_step,
        model,
        optimizer
    )

    # should print True for bias-aware layers
    for name, module in _nnx_iter_modules(model):
        if hasattr(module, 'is_bias_aware'):
            log_from_main_process(logger, 'info', f"{name}: is_bias_aware={module.is_bias_aware}")

    # main training loop with manual batching from unbatched dataset
    log_from_main_process(logger, 'info', 
                          f"Starting training loop: "
                          f"gradient_accumulation_steps={args.gradient_accumulation_steps}, "
                          f"logging_steps={args.logging_steps}")

    def batch_generator():
        yield from create_batches(collator.sampled_datasets, args.batch_size)

    # build output signature dynamically based on config
    bs, sl = args.batch_size, args.sequence_length
    train_sig = {
        'input_ids': tf.TensorSpec((bs, sl), tf.int32),
        'decoder_input_ids': tf.TensorSpec((bs, sl), tf.int32),
        'labels': tf.TensorSpec((bs, sl), tf.int32),
        '_batch_sources': tf.TensorSpec((bs,), tf.string),
        '_batch_training_modes': tf.TensorSpec((bs,), tf.string),
    }

    if args.enable_packing and not in_eval:
        train_sig['attention_mask'] = tf.TensorSpec((bs, 1, sl, sl), tf.float32)
        train_sig['decoder_attention_mask'] = tf.TensorSpec((bs, 1, sl, sl), tf.float32)
        train_sig['segment_ids'] = tf.TensorSpec((bs, sl), tf.int32)
        train_sig['decoder_segment_ids'] = tf.TensorSpec((bs, sl), tf.int32)
        train_sig['position_ids'] = tf.TensorSpec((bs, sl), tf.int32)
        train_sig['decoder_position_ids'] = tf.TensorSpec((bs, sl), tf.int32)
    else:
        train_sig['attention_mask'] = tf.TensorSpec((bs, sl), tf.int32)
        train_sig['decoder_attention_mask'] = tf.TensorSpec((bs, sl), tf.int32)

    train_ds = tf.data.Dataset.from_generator(batch_generator, output_signature=train_sig)
    tf_options = tf.data.Options()
    tf_options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    train_ds = train_ds.with_options(tf_options).prefetch(tf.data.AUTOTUNE)
    train_ds = tfds.as_numpy(train_ds)

    multisteps_every_k = _create_gradaccum_schedule(args)

    pbar = tqdm(total=args.max_tokens, desc="Training", leave=False,
                initial=tokens_seen, disable=not jax.process_index()==0)
    # restore shadow counters from checkpoint when present (keeps the python loop in sync
    # with optax MultiSteps' inner mini_step after a mid-accumulation-window save), else
    # fall back to the step-boundary default
    micro_step = (resumed_micro_step if resumed_micro_step is not None
                  else global_step * args.gradient_accumulation_steps)
    accum_counter = resumed_accum_counter

    for batch in train_ds:
        if tokens_seen >= args.max_tokens:
            log_from_main_process(logger, 'info', f"Training complete! Reached {args.max_tokens:_} tokens")
            break

        # determine if we're in cooldown phase (no metadata)
        cooldown_phase = tokens_seen >= cooldown_tokens

        # transition to cooldown phase if needed (skip if cooldown disabled via ratio=1.0)
        if cooldown_phase and not cooldown_phase_active and args.cooldown_ratio < 1.0:
            log_from_all_processes(logger, 'info', f"Transitioning to cooldown phase at {tokens_seen/1e9:.2f}B tokens")
            # get fresh streaming datasets and recreate with cooldown=True
            result = get_streaming_datasets(args)
            if isinstance(result, tuple):
                fresh_datasets = result[0]
            else:
                fresh_datasets = result
            collator._instantiate_dsets(cooldown_phase=True, new_datasets=fresh_datasets)
            cooldown_phase_active = True

        # extract data sources and training modes BEFORE sharding (for per-task loss tracking)
        batch_sources = batch.pop('_batch_sources', None)
        batch_training_modes = batch.pop('_batch_training_modes', None)

        # convert tf.string bytes to python strings (tfds.as_numpy returns bytes)
        if batch_sources is not None:
            batch_sources = [s.decode('utf-8') if isinstance(s, bytes) else s for s in batch_sources]
        if batch_training_modes is not None:
            batch_training_modes = [s.decode('utf-8') if isinstance(s, bytes) else s for s in batch_training_modes]

        # count content tokens per document (excluding special/sentinel/byte)
        # with packing, each batch row contains multiple documents separated by segment_ids
        enc_ids = batch['input_ids']
        dec_ids = batch['decoder_input_ids']
        enc_content_mask = enc_ids >= CONTENT_TOKEN_MIN_ID
        dec_content_mask = dec_ids >= CONTENT_TOKEN_MIN_ID

        if 'segment_ids' in batch:
            # per-document lengths within packed sequences
            batch_doc_lengths = []
            enc_segs = batch['segment_ids']
            dec_segs = batch.get('decoder_segment_ids', enc_segs)
            for row in range(enc_ids.shape[0]):
                for seg_id in range(1, int(enc_segs[row].max()) + 1):
                    enc_count = int(np.sum(enc_content_mask[row] & (enc_segs[row] == seg_id)))
                    dec_count = int(np.sum(dec_content_mask[row] & (dec_segs[row] == seg_id)))
                    batch_doc_lengths.append(enc_count + dec_count)
            batch_doc_lengths = np.array(batch_doc_lengths) if batch_doc_lengths else np.array([0])
        else:
            # unpacked: one document per row
            enc_content = np.sum(enc_content_mask, axis=-1)
            dec_content = np.sum(dec_content_mask, axis=-1)
            batch_doc_lengths = enc_content + dec_content

        # compute packing segment stats before sharding (for MoSA buffer tuning)
        if 'segment_ids' in batch:
            seg_max = np.max(batch['segment_ids'], axis=-1)
            batch_avg_segments = float(np.mean(seg_max))
            batch_max_segments = int(np.max(seg_max))
        else:
            batch_avg_segments = batch_max_segments = None

        # shard batch to devices
        batch = shard_batch_to_devices(batch, mesh)

        # create dropout rngs for this step
        process_offset = jax.process_index() * 1000
        dropout_rngs = nnx.Rngs(dropout     = jax.random.PRNGKey(args.seed + global_step + process_offset),
                                layerdrop   = jax.random.PRNGKey(args.seed + global_step + process_offset))

        # compute progress outside JIT (Python float arithmetic)
        progress = float(tokens_seen) / float(args.max_tokens)

        # compute current LR outside JIT for logging
        current_lr = compute_lr_for_logging(
            progress, args.learning_rate, args.warmup_ratio,
            args.constant_ratio, args.min_lr_ratio, args.lr_schedule,
            args.lr_cooldown_ratio, args.lr_cooldown_type,
        )

        # training step with progress
        # external clip disabled (1e30); real clipping is applied to the AdamW arm inside
        # the optimizer. clip_and_norm still computes the global grad norm for logging.
        grad_clipnorm = jnp.array(1e30, dtype=jnp.float32)

        with mesh:
            metrics = cached_train_step(
                dropout_rngs,
                grad_clipnorm,
                progress,
                batch
            )

        # add learning_rate to metrics (computed outside JIT)
        metrics['learning_rate'] = current_lr

        # extract synchronized token count from training step
        batch_tokens = int(metrics['valid_token_count'])

        # update counters
        tokens_seen += batch_tokens
        micro_step += 1
        accum_counter += 1

        current_accum_k = multisteps_every_k

        # increment global_step when gradients are actually applied
        if current_accum_k > 1:
            if accum_counter >= current_accum_k:
                global_step += 1
                step_completed = True
                accum_counter = 0
            else:
                step_completed = False
        else:
            global_step += 1
            step_completed = True
            accum_counter = 0

        # always update progress bar for every iteration (not just when step completes)
        pbar.update(batch_tokens)  # batch_tokens already synchronized and converted to int
        progress_pct = tokens_seen / args.max_tokens * 100

        # calculate phase boundaries for display
        constant_start = warmup_tokens
        decay_start = warmup_tokens + int(args.max_tokens * args.constant_ratio)

        # determine LR phase first
        if tokens_seen < constant_start:
            lr_phase = "warmup"
        elif tokens_seen < decay_start:
            lr_phase = "constant"
        else:
            lr_phase = "decay"

        # show cooldown only if actually enabled (ratio < 1.0), otherwise show LR phase
        if cooldown_phase and args.cooldown_ratio < 1.0:
            phase = f"cooldown({lr_phase})"
        else:
            phase = lr_phase

        # always update basic progress info
        pbar.set_postfix({
            'step': global_step,
            'micro': micro_step,
            'tokens': f"{tokens_seen/1e9:.2f}B",
            'phase': phase,
            'progress': f"{progress_pct:.1f}%"
        })

        in_eval = True                  # just one simple switch for the entire next block
        # logging happens either:
        # 1. when a full step is completed AND it's a logging step, OR
        # 2. every N micro steps for more regular updates during gradient accumulation
        # accumulate per-sample losses on EVERY micro-step so rare modes
        # (like temporal_continuation at ~0.1% of samples) get tracked across
        # all batches between log events, not just the single should_log batch
        per_sample_loss_sharded = metrics.pop('per_sample_loss', None)
        if per_sample_loss_sharded is not None:
            local_shards = [jax.device_get(s.data) for s in per_sample_loss_sharded.addressable_shards]
            per_sample_losses = np.concatenate(local_shards, axis=0)

            if batch_sources is not None:
                for i, source in enumerate(batch_sources):
                    if i < len(per_sample_losses):
                        if source not in task_losses:
                            task_losses[source] = []
                        task_losses[source].append(float(per_sample_losses[i]))

            if batch_training_modes is not None:
                for i, mode in enumerate(batch_training_modes):
                    if i < len(per_sample_losses):
                        if mode not in mode_losses:
                            mode_losses[mode] = []
                        mode_losses[mode].append(float(per_sample_losses[i]))

        # accumulate per-document losses bucketed by content length
        per_doc_loss_sharded = metrics.pop('per_doc_loss', None)
        per_doc_valid_sharded = metrics.pop('per_doc_valid', None)
        if per_doc_loss_sharded is not None:
            doc_losses = np.concatenate(
                [jax.device_get(s.data) for s in per_doc_loss_sharded.addressable_shards], axis=0
            )
            doc_valid = np.concatenate(
                [jax.device_get(s.data) for s in per_doc_valid_sharded.addressable_shards], axis=0
            )
            # doc_losses/doc_valid: (local_batch, max_segs) -- match with batch_doc_lengths
            # batch_doc_lengths is flat (all docs across all rows); rebuild per-row mapping
            doc_idx = 0
            for row in range(doc_losses.shape[0]):
                for seg in range(doc_losses.shape[1]):
                    if not doc_valid[row, seg]:
                        continue
                    if doc_idx < len(batch_doc_lengths):
                        bucket = _length_bucket(int(batch_doc_lengths[doc_idx]))
                        if bucket not in length_losses:
                            length_losses[bucket] = []
                            length_counts[bucket] = 0
                        length_losses[bucket].append(float(doc_losses[row, seg]))
                        length_counts[bucket] += 1
                    doc_idx += 1

        should_log = (step_completed and global_step % args.logging_steps == 0) or \
                     (micro_step % args.logging_steps == 0 and micro_step > 0)

        if should_log:
            metrics_cpu: Dict = jax.device_get(metrics)
            train_metrics.append(metrics_cpu)

            if jax.process_index() == 0:
                # update pbar with detailed metrics when logging
                pbar.write(
                    f"[Step {global_step}] "
                    f"micro={micro_step} | "
                    f"tokens={tokens_seen/1e9:.2f}B | "
                    f"loss={metrics_cpu['loss']:.4f} | "
                    f"lr={metrics_cpu.get('learning_rate', args.learning_rate):.2e} | "
                    f"phase={phase} | "
                    f"progress={progress_pct:.1f}%"
                )

            # log token distribution statistics periodically
            if hasattr(collator, 'get_token_statistics'):
                token_stats = collator.get_token_statistics()
                if token_stats['total_samples'] > 0 and token_stats['total_samples'] % 10000 == 0:
                    log_from_main_process(logger, 'info', f"Token distribution after {token_stats['total_samples']} samples:")
                    for source in collator.sampling_ratios:
                        actual = token_stats['token_ratios'].get(source, 0)
                        target = collator.sampling_ratios[source]
                        log_from_main_process(logger, 'info', 
                            f"  {source}: {actual:.3f} (target: {target:.3f}), "
                            f"tokens: {token_stats['tokens_per_source'].get(source, 0):,}")

            if WANDB_AVAILABLE and wandb.run:
                log_dict = {
                    **{f"train/{k}": v for k, v in metrics_cpu.items()
                        if not any(s in k for s in ('token', 'alpha/',
                                                    'grad_norm_class/'))},
                    **{k: v for k, v in metrics_cpu.items() if k.startswith('alpha/')},
                    # per-leaf-class grad norms get their own panel; strip the
                    # 'grad_norm_class/' prefix so keys read as grad/expert, grad/router, etc.
                    **{f"grad/{k[len('grad_norm_class/'):]}": v
                        for k, v in metrics_cpu.items() if k.startswith('grad_norm_class/')},
                    "tokens_seen": int(tokens_seen),
                    "global_step": int(global_step),
                    "micro_step": micro_step,
                    "batch_tokens": batch_tokens,
                }

                # add per-source losses for data quality monitoring
                if task_losses:
                    for source, losses in task_losses.items():
                        if losses:
                            avg_task_loss = sum(losses) / len(losses)
                            log_dict[f"loss/by_source/{source}"] = avg_task_loss
                    # reset task losses after logging
                    task_losses.clear()

                # add per-mode losses for corruption type monitoring
                if mode_losses:
                    for mode, losses in mode_losses.items():
                        if losses:
                            avg_mode_loss = sum(losses) / len(losses)
                            log_dict[f"loss/by_task/{mode}"] = avg_mode_loss
                    mode_losses.clear()

                # add per-document losses and counts by length bucket
                if length_losses:
                    for bucket, losses in length_losses.items():
                        if losses:
                            log_dict[f"loss/by_length/{bucket}"] = sum(losses) / len(losses)
                            log_dict[f"docs/by_length/{bucket}"] = length_counts.get(bucket, 0)
                    length_losses.clear()
                    length_counts.clear()


                if batch_avg_segments is not None:
                    log_dict["packing/avg_segments"] = batch_avg_segments
                    log_dict["packing/max_segments"] = batch_max_segments

                # add token distribution metrics
                if hasattr(collator, 'get_token_statistics'):
                    token_stats = collator.get_token_statistics()
                    for source in collator.sampling_ratios:
                        log_dict[f"token_ratio/{source}"] = token_stats['token_ratios'].get(source, 0)
                        log_dict[f"token_count/{source}"] = token_stats['tokens_per_source'].get(source, 0)

                wandb.log(log_dict)


        # checkpoint saving: trigger on token-based intervals only
        should_save_tokens = tokens_seen >= next_save_tokens

        # jax preemption sync manager returns True at the agreed safe micro-step on all
        # hosts after a SIGTERM (5-min autocheckpoint window). micro_step is SPMD-identical
        # and increments every iteration, so the sync point is reached within ~1 step
        # regardless of accumulation window size. returns False when the distributed client
        # is absent (local single-host runs).
        preemption_save = (
            args.save_on_preemption
            and multihost_utils.reached_preemption_sync_point(micro_step)
        )

        if should_save_tokens or preemption_save:
            if should_save_tokens:
                next_save_tokens += save_token_interval
                save_reason = f"token milestone ({tokens_seen/1e9:.2f}B / {args.max_tokens/1e9:.1f}B tokens)"
            else:
                save_reason = f"preemption sync point (micro_step {micro_step}, {tokens_seen/1e9:.2f}B tokens)"

            log_from_main_process(logger, 'info', f"Saving checkpoint at {save_reason}")

            # build additional metadata
            additional_data = {
                'config': model.config.__dict__,
                'num_hosts': jax.process_count(),
                'accum_counter': int(accum_counter),
                'micro_step': int(micro_step),
            }

            # add collator RNG state for reproducible resumption
            try:
                generator_state = collator.get_generator_state()
                gen_rng_bytes = pickle.dumps(generator_state['rng_object'])
                additional_data['gen_rng_b64'] = base64.b64encode(gen_rng_bytes).decode('utf-8')
                iter_state = generator_state.get('iter_state', {})
                if isinstance(iter_state.get('processed_buckets'), set):
                    iter_state['processed_buckets'] = sorted(list(iter_state['processed_buckets']))
                additional_data['iter_state'] = iter_state
            except Exception as e:
                log_from_main_process(logger, 'warning', f"Could not save collator RNG state: {e}")

            if args.eval_clear_caches:
                jax.clear_caches()
                gc.collect()

            metadata = prepare_metadata(tokens_seen, global_step, additional_data=additional_data)
            try:
                save_checkpoint(
                    checkpoint_manager,
                    model,
                    optimizer,
                    metadata,
                    global_step,
                    force=True
                )
            except StepAlreadyExistsError:
                log_from_main_process(logger, 'warning',
                    f"Step {global_step} already exists (likely from a previous run), overwriting")
                checkpoint_manager.delete(global_step)
                save_checkpoint(
                    checkpoint_manager,
                    model,
                    optimizer,
                    metadata,
                    global_step,
                    force=True
                )
            checkpoint_manager.check_for_errors()

            if preemption_save:
                # async writer must finalize before the process exits, otherwise SIGKILL
                # truncates the write (orbax leaves an unfinalized *-tmp dir that
                # latest_step() ignores, so the prior checkpoint stays safe but this save
                # is lost).
                checkpoint_manager.wait_until_finished()
                log_from_all_processes(logger, 'info',
                    f"Saved preemption checkpoint at micro_step {micro_step} (step {global_step}, "
                    f"{tokens_seen:_} tokens); exiting for restart")
                if WANDB_AVAILABLE and wandb.run:
                    wandb.finish()
                checkpoint_manager.close()
                pbar.close()
                sys.exit(0)

        # evaluation: trigger on token-based intervals only
        should_eval_tokens = tokens_seen >= next_eval_tokens

        if should_eval_tokens:
            checkpoint_manager.wait_until_finished()
            next_eval_tokens += eval_token_interval
            eval_reason = f"token milestone ({tokens_seen/1e9:.2f}B / {args.max_tokens/1e9:.1f}B tokens)"

            log_from_main_process(logger, 'info', f"Evaluation at {eval_reason}")

            if args.eval_clear_caches:
                jax.clear_caches()
                gc.collect()
                log_from_main_process(logger, 'info', f"Cleared JAX caches before callbacks")

            model.eval()    # turn off all dropout/layerdrop, we do this inside the callbacks as well

            # run distributed evaluation on all processes (uses collator.eval_data directly)
            eval_metrics = run_evaluation(model, collator, mesh, args)

            # run callbacks with SPMD-native multihost generation
            # all hosts must participate in generation, but only host 0 prints results
            if callbacks:
                callback_results = {}
                eval_rngs = nnx.Rngs(dropout    = args.seed + global_step + 1000,
                                     layerdrop  = args.seed + global_step + 1001,
                                     params     = args.seed + global_step + 1002,)

                for callback_name, callback in callbacks.items():
                    try:
                        log_from_main_process(logger, 'info', f"Running {callback_name} evaluation...")
                        # use SPMD model directly - all hosts participate in generation
                        callback_result = callback(model, global_step, eval_rngs, use_metadata=not cooldown_phase)
                        callback_results[callback_name] = callback_result
                        log_from_main_process(logger, 'info', f"{callback_name} evaluation completed")
                    except Exception as e:
                        # callback errors should be visible from all processes for debugging
                        log_from_all_processes(logger, 'error', f"Error running {callback_name} callback: {e}")
                        import traceback
                        log_from_all_processes(logger, 'error', traceback.format_exc())

                # log evaluation metrics to wandb
                if WANDB_AVAILABLE and wandb.run and eval_metrics:
                    wandb.log({**eval_metrics, "tokens_seen": tokens_seen, "global_step": global_step})

                # log callback results to wandb
                if WANDB_AVAILABLE and wandb.run and callback_results:
                    for callback_name, callback_result in callback_results.items():
                        callback_metrics = {}
                        if isinstance(callback_result, dict):
                            # flatten callback results for wandb
                            for key, value in callback_result.items():
                                if isinstance(value, (int, float)):
                                    callback_metrics[f"{callback_name}/{key}"] = value
                                elif key == 'generation_stats' and isinstance(value, dict):
                                    for stat_key, stat_value in value.items():
                                        if isinstance(stat_value, (int, float)):
                                            callback_metrics[f"{callback_name}/{stat_key}"] = stat_value
                                elif key == 'subsource_stats' and isinstance(value, dict):
                                    for subsource, stats in value.items():
                                        if isinstance(stats, dict):
                                            subsource_bleu = stats.get('avg_bleu')
                                            if subsource_bleu is not None:
                                                safe_name = subsource.replace('/', '_')
                                                callback_metrics[f"{callback_name}/subsource/{safe_name}"] = subsource_bleu

                        try:        # also add computed metrics from the callback
                            computed_metrics = callbacks[callback_name].compute_metrics()
                            for metric_key, metric_value in computed_metrics.items():
                                if isinstance(metric_value, (int, float)):
                                    callback_metrics[f"{callback_name}/{metric_key}"] = metric_value
                        except (AttributeError, Exception):
                            pass    # callback might not have compute_metrics method

                        callback_metrics.update({"tokens_seen": tokens_seen, "global_step": global_step})

                        if callback_metrics:
                            wandb.log(callback_metrics)

                if args.eval_clear_caches:
                    jax.clear_caches()  # perhaps a bit of a temporary slowdown but better than OOM
                    gc.collect()        # tyvm

            log_from_main_process(logger, 'info', f"Finished up with evaluation, global_step: {global_step}")
            model.train()   # ensure model is in the correct state 

        in_eval = False

    # final checkpoint
    log_from_all_processes(logger, 'info', "Saving final checkpoint")

    # build final additional metadata
    final_additional_data = {
        'config': model.config.__dict__,
        'num_hosts': jax.process_count(),
    }

    # add collator RNG state
    try:
        generator_state = collator.get_generator_state()
        gen_rng_bytes = pickle.dumps(generator_state['rng_object'])
        final_additional_data['gen_rng_b64'] = base64.b64encode(gen_rng_bytes).decode('utf-8')
        iter_state = generator_state.get('iter_state', {})
        if isinstance(iter_state.get('processed_buckets'), set):
            iter_state['processed_buckets'] = sorted(list(iter_state['processed_buckets']))
        final_additional_data['iter_state'] = iter_state
    except Exception as e:
        log_from_main_process(logger, 'warning', f"Could not save collator RNG state: {e}")

    metadata = prepare_metadata(tokens_seen, global_step, additional_data=final_additional_data)
    try:
        if args.eval_clear_caches:
            jax.clear_caches()
            gc.collect()
        save_checkpoint(
            checkpoint_manager,
            model,
            optimizer,
            metadata,
            global_step,
            force=True
        )
    except StepAlreadyExistsError:
        log_from_main_process(logger, 'info', f"Checkpoint at step {global_step} already exists, skipping final save")

    checkpoint_manager.wait_until_finished()

    log_from_main_process(logger, 'info', f"Training completed after {tokens_seen/1e9:.2f}B tokens ({global_step} steps)")

    if WANDB_AVAILABLE and wandb.run:
        wandb.finish()

    checkpoint_manager.close()
    pbar.close()


def main():
    """Main training function."""
    args = get_config()

    # inject cuda flags for GPU/TPU
    os.environ['XLA_FLAGS'] = (
        '--xla_gpu_triton_gemm_any=true '
        '--xla_gpu_enable_latency_hiding_scheduler=true '
    )
    compilation_dir = f"{args.gcs_output_dir}/jax-cache" if args.data_bucket else "/tmp/jax_cache"
    jax.config.update('jax_compilation_cache_dir', compilation_dir)
    jax.config.update('jax2tf_associative_scan_reductions', True)       # also apparently speedy
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update(
        "jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir"
    )

    proc_id = jax.process_index()
    n_procs = jax.process_count()

    log_from_main_process(logger, 'info', "=" * 70)
    log_from_main_process(logger, 'info', f"Han2Han Training (Process {proc_id+1}/{n_procs})")
    log_from_main_process(logger, 'info', "=" * 70)
    log_from_main_process(logger, 'info', f"Devices per process: {jax.local_device_count()}, Total: {jax.device_count()}")
    log_from_main_process(logger, 'info', json.dumps(vars(args), indent=2, default=str))
    log_from_main_process(logger, 'info', "=" * 70)

    # set random seeds
    random.seed(args.seed + proc_id)
    np.random.seed(args.seed + proc_id)

    # setup mesh and sharding (also sets global mesh via sharding_utils)
    mesh, data_sharding, parallelism_config = setup_mesh_and_sharding(args)

    # load tokenizer and update config
    log_from_main_process(logger, 'info', f"Loading tokenizer from {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    original_vocab_size = len(tokenizer)

    # create model config
    # use original vocab size for checkpoint compatibility; resize after restoration
    model_config = create_model_config(args)
    model_config.vocab_size = original_vocab_size
    model_config.pad_token_id = tokenizer.pad_token_id
    model_config.eos_token_id = tokenizer.eos_token_id
    model_config.bos_token_id = tokenizer.bos_token_id
    model_config.decoder_start_token_id = tokenizer.eos_token_id

    # add Han2Han jamo/char subword features if requested
    cbu = None
    jbu = None
    if args.jamo_subwords or args.char_subwords:
        log_from_main_process(logger, 'info', "Computing Han2Han subword tables...")
        jbu, cbu, config_updates = compute_subword_tables(
            tokenizer,
            jamo_subwords=args.jamo_subwords,
            char_subwords=args.char_subwords,
            ngram_sizes=args.ngram_sizes if args.jamo_subwords else None,
            min_n=args.min_n,
            max_n=args.max_n,
        )
        if args.jamo_subwords and jbu is not None:
            model_config.jamo_vocab_size = config_updates.get("num_jamo_buckets")
            log_from_main_process(logger, 'info', f"Jamo subwords: {jbu.shape}, buckets={model_config.jamo_vocab_size}")
        if args.char_subwords and cbu is not None:
            model_config.char_vocab_size = config_updates.get("num_char_buckets")
            log_from_main_process(logger, 'info', f"Char subwords: {cbu.shape}, buckets={model_config.char_vocab_size}")

    # create model and optimizer within mesh context.
    # --model_dtype is activation/compute precision. --param_dtype is latent-weight
    # storage precision (Flax param_dtype); defaults to dtype. Splitting them keeps
    # bf16 matmuls while allowing f32 weight storage for stability.
    dtype = getattr(jnp, args.model_dtype)
    param_dtype = getattr(jnp, args.param_dtype) if args.param_dtype else dtype
    optimizer_state_dtype = (getattr(jnp, args.optimizer_state_dtype)
                             if args.optimizer_state_dtype else dtype)
    log_from_main_process(
        logger, 'info',
        f"Precision: activation={dtype.__name__}, param={param_dtype.__name__}, "
        f"optimizer_state={optimizer_state_dtype.__name__}"
    )

    num_hosts = jax.process_count()

    # estimate tokens per step based on args, used for initial token count / lr schedule
    effective_infilling = _estimate_effective_infilling(args)
    tokens_per_step = (
                args.batch_size * args.sequence_length
                * num_hosts * args.gradient_accumulation_steps
                * (1 + effective_infilling) * args.packing_efficiency_threshold
            )

    # create token-based learning rate schedule
    lr_scheduler = create_learning_rate_schedule(args)

    # create model and optimizer outside JIT to avoid tracer issues
    # initialize model
    model_rngs = nnx.Rngs(
        params=args.seed,
        dropout=args.seed + 1,
    )

    # derive model parameter sharding from strategy
    model_param_sharding = derive_param_sharding(args.parallelism_strategy, parallelism_config)
    log_from_main_process(logger, 'info', f"Using model parameter sharding: {model_param_sharding}")

    # Flax 0.12 (FLIP 4844) requires jax.set_mesh for eager variable sharding.
    # scoped to model init only - global set_mesh changes sharding propagation semantics
    with jax.set_mesh(mesh):
        @nnx.jit
        def get_model_and_opt(init_rngs):
            model = FlaxHan2Han(
                model_config,
                init_rngs,
                dtype=dtype,
                param_dtype=param_dtype,
                gradient_checkpointing=model_config.remat_policy != "none",
                sharding=model_param_sharding,
                jamo_buckets=jbu,
                char_buckets=cbu,
            )

            # patch to_opt_state to fix sharding annotations on muon's 1D factored stats.
            with patch_to_opt_state_for_factored_adafactor():
                optimizer = create_optimizer(args, lr_scheduler, model)

            return model, optimizer

        model, optimizer = get_model_and_opt(model_rngs)

    # one-shot subword embedding diagnostic: verify wje/wce are wired into forward.
    # a mismatch (embedding module built but lookup missing) is a silent dead-grad bug,
    # so we raise rather than log-and-continue.
    enc_lookups = getattr(model.encoder, 'subword_lookups', None)
    dec_lookups = getattr(model.decoder, 'subword_lookups', None)
    enc_has_jbu = (enc_lookups is not None) and ('jbu' in enc_lookups)
    enc_has_cbu = (enc_lookups is not None) and ('cbu' in enc_lookups)
    dec_has_jbu = (dec_lookups is not None) and ('jbu' in dec_lookups)
    dec_has_cbu = (dec_lookups is not None) and ('cbu' in dec_lookups)
    log_from_main_process(logger, 'info',
        f"[subword-diag] config.jamo_subwords={model.config.jamo_subwords}, "
        f"config.char_subwords={model.config.char_subwords}, ")
    log_from_main_process(logger, 'info',
        f"[subword-diag] enc.wje is not None: {model.encoder.wje is not None}, "
        f"enc.wce is not None: {model.encoder.wce is not None}, "
        f"dec.wje is not None: {model.decoder.wje is not None}, "
        f"dec.wce is not None: {model.decoder.wce is not None}")
    log_from_main_process(logger, 'info',
        f"[subword-diag] enc lookups -- jbu: {enc_has_jbu}, cbu: {enc_has_cbu} | "
        f"dec lookups -- jbu: {dec_has_jbu}, cbu: {dec_has_cbu}")
    if model.encoder.wje is not None:
        wje_param = model.encoder.wje.embedding[...]
        log_from_main_process(logger, 'info',
            f"[subword-diag] enc.wje shape={wje_param.shape}, dtype={wje_param.dtype}, "
            f"id(enc.wje.embedding)={id(model.encoder.wje.embedding)}, "
            f"id(dec.wje.embedding)={id(model.decoder.wje.embedding) if model.decoder.wje is not None else None} "
            f"(tied if equal)")

    mismatches = []
    if model.encoder.wje is not None and not enc_has_jbu:
        mismatches.append("encoder.wje exists but subword_lookups.jbu is missing")
    if model.encoder.wce is not None and not enc_has_cbu:
        mismatches.append("encoder.wce exists but subword_lookups.cbu is missing")
    if model.decoder.wje is not None and not dec_has_jbu:
        mismatches.append("decoder.wje exists but subword_lookups.jbu is missing")
    if model.decoder.wce is not None and not dec_has_cbu:
        mismatches.append("decoder.wce exists but subword_lookups.cbu is missing")
    if mismatches:
        raise RuntimeError(
            "Subword embedding/lookup mismatch -- embeddings would receive zero gradient:\n  - "
            + "\n  - ".join(mismatches)
        )

    # set up checkpoint manager using helper function
    ckpt_manager = setup_checkpoint_manager(
        output_dir=args.output_dir,
        gcs_output_dir=args.gcs_output_dir,
        max_to_keep=args.keep_checkpoints,
        single_optimizer=True   # only single optimizer from hereon out (10.07.2025 lol)
    )

    # initialize tokens_seen to one step's worth so the first update gets a
    # non-zero LR (avoids wasting the first step while polluting Adam's nu)
    tokens_seen = np.int64(tokens_per_step)
    global_step = 0

    # check for checkpoint first to avoid creating model unnecessarily
    if args.skip_restore:
        restore_step = None
    elif args.restore_step is not None:
        available_steps = sorted(ckpt_manager.all_steps())
        if args.restore_step not in available_steps:
            raise ValueError(
                f"Requested --restore_step={args.restore_step} not found. "
                f"Available checkpoints: {available_steps}")
        restore_step = args.restore_step
        newer_steps = [s for s in available_steps if s > restore_step]
        if newer_steps:
            log_from_main_process(logger, 'warning',
                f"Restoring from step {restore_step}, but newer checkpoints exist: {newer_steps}. "
                f"These will be overwritten as training progresses past those steps.")
    else:
        restore_step = ckpt_manager.latest_step()

    if restore_step is not None:
        log_from_all_processes(logger, 'info', f"Found checkpoint at step {restore_step}, restoring...")

        restored_step, metadata = restore_checkpoint(
            ckpt_manager,
            model,
            optimizer,
            step=restore_step,
            mesh=mesh,
            use_abstract_restoration=True
        )

        # extract tokens_seen from metadata
        tokens_seen = extract_tokens_from_metadata(metadata)
        global_step = metadata.get('global_step', 0)

        # restore the python-side accumulation shadow counters so a mid-window save resumes
        # in phase with optax MultiSteps' restored mini_step. backward-compatible defaults
        # for checkpoints written before these were persisted (step-boundary assumption).
        resumed_accum_counter = int(metadata.get('accum_counter', 0))
        resumed_micro_step = int(metadata.get('micro_step',
                                              global_step * args.gradient_accumulation_steps))

        log_from_all_processes(logger, 'info', f"Will resume from {tokens_seen:_} tokens (step {global_step:_})")

        # check for topology change (different num_hosts invalidates iterator state)
        ckpt_num_hosts = metadata.get('num_hosts', None)
        current_num_hosts = jax.process_count()
        topology_changed = (
            ckpt_num_hosts is not None and ckpt_num_hosts != current_num_hosts
        )

        if topology_changed:
            log_from_main_process(logger, 'warning',
                f"Topology changed: checkpoint saved with {ckpt_num_hosts} hosts, "
                f"now running with {current_num_hosts} hosts. "
                f"Discarding iterator state and collator RNG (data slices differ). "
                f"tokens_seen ({tokens_seen:_}) and global_step ({global_step:_}) are kept.")
            restored_rng_b64 = None
            restored_iter_state = {}
        elif ckpt_num_hosts is None:
            log_from_main_process(logger, 'warning',
                "Checkpoint has no num_hosts metadata (old format). "
                "Cannot verify topology match; iterator state may be stale.")
            restored_rng_b64 = metadata.get('gen_rng_b64', None)
            restored_iter_state = metadata.get('iter_state', {})
        else:
            restored_rng_b64 = metadata.get('gen_rng_b64', None)
            restored_iter_state = metadata.get('iter_state', {})

        # ensure all processes have completed restoration
        if jax.process_count() > 1:
            multihost_utils.sync_global_devices("checkpoint_restored")

        log_from_main_process(logger, 'info', f"Successfully restored checkpoint from step {restored_step}")

    else:
        restored_rng_b64 = None
        restored_iter_state = {}
        resumed_accum_counter = 0
        resumed_micro_step = None

        # no checkpoint - check for pretrained weights to initialize from
        if args.pretrained_weights_path:
            log_from_main_process(logger, 'info', f"Loading pretrained weights from {args.pretrained_weights_path}")
            if args.restore_optimizer:
                log_from_main_process(logger, 'info', "Model and optimizer will be restored")
            else:
                log_from_main_process(logger, 'info', "Will restore model alone, no optimizer")

            # load pretrained checkpoint
            # determine checkpoint path
            from etils.epath import Path
            if Path(args.pretrained_weights_path).is_dir():
                # directory provided - find latest checkpoint
                pretrained_ckpt_manager = setup_checkpoint_manager(
                    output_dir=args.pretrained_weights_path,
                    gcs_output_dir=args.pretrained_weights_path,
                    for_pretrained_restoration=True,
                    single_optimizer=True,
                    max_to_keep=1
                )
                pretrained_step = pretrained_ckpt_manager.latest_step()
                if pretrained_step is None:
                    log_from_main_process(logger, 'warning', f"No checkpoint found in {args.pretrained_weights_path}, starting from scratch")
                    pretrained_ckpt_manager.close()
                else:
                    log_from_main_process(logger, 'info', f"Found pretrained checkpoint at step {pretrained_step}")

                    restored_step, metadata = restore_checkpoint(
                        pretrained_ckpt_manager,
                        model,
                        optimizer if args.restore_optimizer else None,
                        step=pretrained_step,
                        mesh=mesh,
                        use_abstract_restoration=True,
                        model_only=not args.restore_optimizer,
                    )

                    log_from_main_process(logger, 'info', f"Loaded pretrained weights from step {restored_step}")
                    log_from_main_process(logger, 'info', f"Original training: {extract_tokens_from_metadata(metadata)/1e9:.2f}B tokens")

                    # re-establish weight tying after restoration
                    # nnx.update() breaks Python object references, so TiedLinear.embedding_ref
                    # may point to stale (pre-restoration) embeddings
                    model.tie_weights()
                    log_from_main_process(logger, 'info', "Re-established weight tying after pretrained restoration")

                    if args.restore_optimizer:
                        log_from_main_process(logger, 'info', "Restored optimizer state - continuing training with previous optimization state")
                    else:
                        log_from_main_process(logger, 'info', "Starting fresh training with reincarnated weights!")
                        with jax.set_mesh(mesh):
                            @nnx.jit
                            def get_opt(model):
                                with patch_to_opt_state_for_factored_adafactor():
                                    optimizer = create_optimizer(args, lr_scheduler, model)
                                return optimizer

                            optimizer = get_opt(model)
                        log_from_main_process(logger, 'info', "Recreated optimizer to bind to restored model variables")

                    pretrained_ckpt_manager.close()
            else:
                log_from_main_process(logger, 'warning', f"Pretrained weights path {args.pretrained_weights_path} is not a directory")

        elif args.skip_restore:
            log_from_main_process(logger, 'info', "Skipping checkpoint restore (--skip_restore)")
        else:
            log_from_main_process(logger, 'info', "No checkpoint found, starting from scratch")

    total_params = sum(x.size for x in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param)))
    log_from_main_process(logger, 'info', f"Model ready with {total_params/1e6:.1f}M parameters")

    # detailed parameter breakdown
    if proc_id == 0:
        param_breakdown = {}
        flat_params = jax.tree_util.tree_leaves_with_path(nnx.state(model, nnx.Param))

        for path, param in flat_params:
            # extract module name from path
            if path:
                module_name = str(path[0].key) if hasattr(path[0], 'key') else str(path[0])
            else:
                module_name = 'root'

            if module_name not in param_breakdown:
                param_breakdown[module_name] = 0
            param_breakdown[module_name] += param.size

        log_from_main_process(logger, 'info', "Parameter breakdown by module:")
        for module_name, size in sorted(param_breakdown.items(), key=lambda x: x[1], reverse=True)[:15]:
            log_from_main_process(logger, 'info', f"  {module_name}: {size/1e6:.1f}M params ({size/total_params*100:.1f}%)")

    # calculate cooldown phase before creating collator
    cooldown_tokens = int(args.max_tokens * (1 - args.cooldown_ratio))
    initial_cooldown_phase = tokens_seen >= cooldown_tokens

    if initial_cooldown_phase:
        log_from_main_process(logger, 'info',
            f"Starting in cooldown phase: {tokens_seen/1e9:.2f}B >= {cooldown_tokens/1e9:.2f}B tokens")
    else:
        log_from_main_process(logger, 'info',
            f"Starting in training phase: {tokens_seen/1e9:.2f}B < {cooldown_tokens/1e9:.2f}B tokens")

    # setup data pipeline with correct initial cooldown state
    collator = setup_data_pipeline(args, tokenizer, cooldown_phase=initial_cooldown_phase)

    # restore collator RNG state and iterator positions from checkpoint
    if restored_rng_b64 is not None:
        try:
            gen_rng_bytes = base64.b64decode(restored_rng_b64)
            restored_rng = pickle.loads(gen_rng_bytes)
            collator.rng = restored_rng
            log_from_main_process(logger, 'info', "Restored collator RNG state from checkpoint")
        except Exception as e:
            log_from_main_process(logger, 'warning', f"Could not restore collator RNG state: {e}")

    samples_per_source = restored_iter_state.get('samples_per_source', {})
    raw_consumed_per_source = restored_iter_state.get('raw_consumed_per_source', {})
    if (samples_per_source or raw_consumed_per_source) and hasattr(collator, 'advance_iterators'):
        try:
            collator.advance_iterators(
                samples_per_source,
                raw_consumed_per_source=raw_consumed_per_source or None,
            )
        except Exception as e:
            log_from_main_process(logger, 'warning',
                f"Could not advance iterators from checkpoint: {e}")

    # setup evaluation callbacks
    # only host 0 will print results, but all must participate in model.generate()
    callbacks = {}

    log_from_main_process(logger, 'info', "Setting up evaluation callbacks...")

    # load real Korean art criticism data for evaluation
    eval_data = None
    try:
        import polars as pl
        eval_parquet = f"{args.data_bucket}/casasia_sentences.parquet" if args.data_bucket else "casasia_sentences.parquet"
        eval_data = pl.read_parquet(eval_parquet)

        # replace metadata with denoising instruction (matching training prompts)
        denoising_prompt = random.choice(TASK_PROMPTS['denoising']['ko'])
        eval_data = eval_data.with_columns(
            pl.lit(denoising_prompt).alias('metadata')
        )

        log_from_main_process(logger, 'info', f"Loaded {len(eval_data)} Korean art criticism examples for evaluation")
        log_from_main_process(logger, 'info', f"Using denoising prompt: '{denoising_prompt}'")
    except Exception as e:
        log_from_main_process(logger, 'warning', f"Could not load casasia_sentences.parquet, using synthetic examples: {e}")

    eval_collator = setup_data_pipeline(
        args, tokenizer, for_eval=True,
        eval_data=collator.eval_data, max_length_override=128
    )

    if not args.disable_generative_eval:
        # preprocess Korean art criticism data for BLEU callback format
        bleu_eval_data = None
        if eval_data is not None:
            bleu_eval_data = []
            for row in eval_data.sample(n=30, shuffle=True, seed=args.seed).to_dicts():
                # create input/target pairs from original_text
                original = row['original_text']
                # split roughly in half for input/target
                words = original.split()
                if len(words) > 6:  # need enough words to split meaningfully
                    split_idx = len(words) // 2
                    input_text = ' '.join(words[:split_idx])
                    target_text = ' '.join(words[split_idx:])
                    bleu_eval_data.append({
                        'original_text': input_text,
                        'sentences': input_text,
                        'input': input_text,
                        'target': target_text,
                        'metadata': row.get('metadata', ''),
                        'source': 'casasia',
                    })

        # initialize BLEU callback with preprocessed Korean art criticism data
        bleu_callback = BLEUCallback(
            tokenizer=tokenizer,
            max_length=min(256, args.sequence_length),
            max_eval_samples=2,
            batch_size=2,
            temperature=0.8,
            top_p=0.95,
            top_k=0,
            repetition_penalty=1.2,
            no_repeat_ngram_size=2,
            num_beams=0,
            seed=args.seed,
            eval_data=bleu_eval_data,
            eval_collator=eval_collator,
            decoder_start_token="<s>",
            mesh=mesh
        )
        callbacks['bleu'] = bleu_callback
        log_from_main_process(logger, 'info', "BLEU callback initialized")

    # initialize generative evaluation callback for KLUE tasks (ynat, nli, sts, temporal)
    # skip entirely if sft_tasks is 'none' or --disable_generative_eval is set
    if not args.disable_generative_eval and args.sft_tasks and args.sft_tasks.lower().strip() != 'none':
        from dynamic_data_loader import get_sft_eval_datasets
        from generative_evaluation_callback import TASK_TO_DATA_TYPE

        sft_datasets, sft_ratios, sft_source_configs = get_sft_eval_datasets(
            sft_tasks=args.sft_tasks,
            data_bucket=args.data_bucket,
            host_idx=jax.process_index(),
            num_hosts=jax.process_count(),
        )

        if sft_datasets:
            # create SFT eval collator with task prompts enabled
            sft_eval_collator = setup_data_pipeline(
                args, tokenizer,
                for_eval=True,
                streaming_datasets=sft_datasets,
                sampling_ratios=sft_ratios,
                source_configs=sft_source_configs,
                cooldown_phase=False,
                max_length_override=256,
            )

            # derive tasks from available data_types
            data_type_to_task = {v: k for k, v in TASK_TO_DATA_TYPE.items()}
            available_tasks = []
            for cfg in sft_source_configs.values():
                task = data_type_to_task.get(cfg.data_type)
                if task and task not in available_tasks:
                    available_tasks.append(task)

            log_from_main_process(logger, 'info',
                f"SFT eval collator created with tasks: {available_tasks}")

            generative_eval_callback = GenerativeEvaluationCallback(
                tokenizer=tokenizer,
                collator=sft_eval_collator,
                tasks=available_tasks,
                max_eval_samples=50,
                batch_size=16,
                max_input_length=256,
                max_output_length=16,
                mesh=mesh,
                use_task_prompts=True,
            )
            callbacks['generative_eval'] = generative_eval_callback
            log_from_main_process(logger, 'info',
                f"Generative evaluation callback initialized for tasks: {available_tasks}")
        else:
            log_from_main_process(logger, 'info',
                "No SFT eval data available, skipping generative evaluation callback")
    else:
        log_from_main_process(logger, 'info',
            "SFT tasks disabled, skipping generative evaluation callback")

    # === MC log-prob evaluation callback ===
    if args.mc_eval_benchmarks and args.mc_eval_benchmarks.lower() != 'none':
        mc_benchmarks = [b.strip() for b in args.mc_eval_benchmarks.split(',')]

        from dynamic_data_loader import get_mc_eval_data
        mc_eval_data = get_mc_eval_data(
            benchmarks=mc_benchmarks,
            data_bucket=args.data_bucket,
            max_samples_per_benchmark=args.mc_eval_max_samples,
            seed=args.seed,
        )

        if mc_eval_data:
            mc_callback = MCLogProbCallback(
                tokenizer=tokenizer,
                eval_data=mc_eval_data,
                mesh=mesh,
                max_input_length=args.sequence_length,
                max_decoder_length=min(64, args.sequence_length),
                eval_collator=collator,
                max_eval_samples=args.mc_eval_max_samples,
                use_task_prompts=True,
            )
            callbacks['mc_eval'] = mc_callback
            log_from_main_process(logger, 'info',
                f"MC log-prob callback initialized for: {list(mc_eval_data.keys())}")
        else:
            log_from_main_process(logger, 'warning',
                "No MC eval data loaded, skipping mc_eval callback")

    # === Temporal log-prob evaluation callback ===
    if args.temporal_eval_benchmarks and args.temporal_eval_benchmarks.lower() != 'none':
        temporal_benchmarks = [b.strip() for b in args.temporal_eval_benchmarks.split(',')]

        from dynamic_data_loader import get_temporal_eval_data
        temporal_eval_data = get_temporal_eval_data(
            benchmarks=temporal_benchmarks,
            data_bucket=args.data_bucket,
            max_samples_per_benchmark=args.temporal_eval_max_samples,
            seed=args.seed,
        )

        if temporal_eval_data:
            temporal_callback = TemporalLogProbCallback(
                tokenizer=tokenizer,
                eval_data=temporal_eval_data,
                mesh=mesh,
                max_input_length=args.sequence_length,
                max_decoder_length=min(64, int(args.sequence_length//4)),
                eval_collator=collator,
                max_eval_samples=args.temporal_eval_max_samples,
                use_task_prompts=True,
            )
            callbacks['temporal_eval'] = temporal_callback
            log_from_main_process(logger, 'info',
                f"Temporal log-prob callback initialized for: {list(temporal_eval_data.keys())}")
        else:
            log_from_main_process(logger, 'warning',
                "No temporal eval data loaded, skipping temporal_eval callback")

    # run mc_eval / temporal_eval first (cheapest, no generation, crash-safe)
    if 'temporal_eval' in callbacks:
        callbacks = {'temporal_eval': callbacks.pop('temporal_eval'), **callbacks}
    if 'mc_eval' in callbacks:
        callbacks = {'mc_eval': callbacks.pop('mc_eval'), **callbacks}

    log_from_main_process(logger, 'info', f"Initialized {len(callbacks)} callbacks: {list(callbacks.keys())}")

    # start training
    training_loop(
        model=model,
        collator=collator,
        optimizer=optimizer,
        args=args,
        mesh=mesh,
        data_sharding=data_sharding,
        checkpoint_manager=ckpt_manager,
        tokens_seen=tokens_seen,
        global_step=global_step,
        callbacks=callbacks,
        resumed_accum_counter=resumed_accum_counter,
        resumed_micro_step=resumed_micro_step,
        lr_scheduler=lr_scheduler,
    )

    log_from_main_process(logger, 'info', "Training completed successfully")


if __name__ == "__main__":
    main()
