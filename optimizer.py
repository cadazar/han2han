"""Optimizer assembly for Han2Han: learning-rate schedule, weight-decay masks,
and create_optimizer (Muon + Gram Newton-Schulz only).

Extracted from the pre-training script so the pre-training, instruction-tuning,
and classifier fine-tuning entrypoints share one optimizer definition. The
low-level Muon transform lives in muon_gramNS.py; this module assembles it with
parameter partitioning, weight-decay grouping, and the LR/WD schedules.
"""

import logging
from contextlib import contextmanager
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from jax.sharding import PartitionSpec
from optax.schedules import inject_hyperparams

from logging_utils import log_from_main_process
from token_based_schedule import ProgressConstantSchedule, create_progress_schedule

logger = logging.getLogger(__name__)

# nnx version shim: the arrays-from-state accessor was renamed across flax
# 0.10-0.12 (to_arrays -> as_array_vars -> vars_as).
_nnx_to_arrays = getattr(nnx, 'to_arrays', None)
if _nnx_to_arrays is None:
    _nnx_to_arrays = getattr(nnx, 'as_array_vars', None)
if _nnx_to_arrays is None:
    _nnx_to_arrays = nnx.vars_as


def create_learning_rate_schedule(args, peak_lr=None):
    """Create progress-based learning rate schedule.

    Args:
        args: Training arguments
        peak_lr: Optional peak LR override. When None, uses ``args.learning_rate``.

    Returns:
        Schedule instance (ProgressCosineSchedule, ProgressLinearSchedule, or ProgressConstantSchedule)
    """
    lr = args.learning_rate if peak_lr is None else peak_lr
    if args.lr_schedule == "constant":
        return ProgressConstantSchedule(
            lr=lr,
            warmup_ratio=args.warmup_ratio,
            lr_cooldown_ratio=args.lr_cooldown_ratio,
            lr_cooldown_type=args.lr_cooldown_type,
            min_lr_ratio=args.min_lr_ratio,
        )
    else:
        return create_progress_schedule(
            learning_rate=lr,
            warmup_ratio=args.warmup_ratio,
            constant_ratio=args.constant_ratio,
            min_lr_ratio=args.min_lr_ratio,
            schedule_type=args.lr_schedule,
            lr_cooldown_ratio=args.lr_cooldown_ratio,
            lr_cooldown_type=args.lr_cooldown_type,
        )

def create_weight_decay_masks(model, scale_mult=0.01, bias_mult=0.001, separate_lm_head=False,
                              wrt_filter=None):
    """Create weight decay masks including tiered float mask for adaptive WD.

    Args:
        model: The model to create masks for
        scale_mult: WD multiplier for layer norm scales (default: 0.01)
        bias_mult: WD multiplier for biases (default: 0.001)
        separate_lm_head: If True, create separate mask for lm_head (for untied embeddings)
        wrt_filter: NNX filter for trainable params (default: nnx.Param for all params)

    Returns:
        tuple: (mask_norm, mask_bias, mask_mlp, mask_emb, mask_lm_head, mask_attention,
                mask_global_decay, mask_tiered, param_counts)
        where mask_tiered is a pytree of floats for adaptive WD
    """
    if wrt_filter is None:
        wrt_filter = nnx.Param
    param_state = nnx.state(model, wrt_filter)
    param_arrays = _nnx_to_arrays(nnx.pure(param_state))

    # patterns to exclude from all weight decay (binary mask)
    no_decay_patterns = ['alpha_logits']

    def should_decay(path):
        """Check if parameter should have any weight decay."""
        path_str = '/'.join(str(p.key) if hasattr(p, 'key') else str(p) for p in path)
        return not any(pattern in path_str for pattern in no_decay_patterns)

    def is_norm_scale(path, _):
        """Check if parameter is a RMSNorm/LayerNorm scale."""
        path_str = '/'.join(str(p.key) if hasattr(p, 'key') else str(p) for p in path)
        return 'scale' in path_str

    def is_bias(path, _):
        """Check if parameter is a bias."""
        path_str = '/'.join(str(p.key) if hasattr(p, 'key') else str(p) for p in path)
        return 'bias' in path_str

    def is_mlp_kernel(path, _):
        """Check if parameter is an MLP kernel."""
        path_str = '/'.join(str(p.key) if hasattr(p, 'key') else str(p) for p in path)
        has_mlp = 'mlp' in path_str
        has_kernel = 'kernel' in path_str
        has_wi_wo = any(x in path_str for x in ['/wi/', '/wo/', '/wi_0/', '/wi_1/'])
        is_mlp = has_mlp and has_kernel and has_wi_wo
        return is_mlp and should_decay(path)

    def is_lm_head(path, _):
        """Check if parameter is lm_head (separate from embeddings when untied)."""
        path_str = '/'.join(str(p.key) if hasattr(p, 'key') else str(p) for p in path)
        return 'lm_head' in path_str and should_decay(path)

    def is_embedding(path, _):
        """Check if parameter is an embedding (or lm_head when not separated)."""
        path_str = '/'.join(str(p.key) if hasattr(p, 'key') else str(p) for p in path)
        if separate_lm_head:
            is_emb = any(x in path_str for x in ['wte', 'wce', 'wje'])
        else:
            is_emb = any(x in path_str for x in ['wte', 'wce', 'wje', 'lm_head'])
        return is_emb and should_decay(path)

    def is_attention_decay(path, _):
        """Check if parameter is attention/other (everything else that should decay)."""
        path_str = '/'.join(str(p.key) if hasattr(p, 'key') else str(p) for p in path)
        has_mlp = 'mlp' in path_str
        has_kernel = 'kernel' in path_str
        has_wi_wo = any(x in path_str for x in ['/wi/', '/wo/', '/wi_0/', '/wi_1/'])
        is_mlp = has_mlp and has_kernel and has_wi_wo
        is_emb = any(x in path_str for x in ['wte', 'wce', 'wje'])
        is_head = 'lm_head' in path_str
        is_scale = 'scale' in path_str
        return (not is_mlp) and (not is_emb) and (not is_head) and (not is_scale) and should_decay(path)

    # create boolean masks for weight decay groups
    mask_norm = jax.tree_util.tree_map_with_path(is_norm_scale, param_arrays)
    mask_bias = jax.tree_util.tree_map_with_path(is_bias, param_arrays)
    mask_mlp = jax.tree_util.tree_map_with_path(is_mlp_kernel, param_arrays)
    mask_emb = jax.tree_util.tree_map_with_path(is_embedding, param_arrays)
    mask_lm_head = jax.tree_util.tree_map_with_path(is_lm_head, param_arrays) if separate_lm_head else None
    mask_attention = jax.tree_util.tree_map_with_path(is_attention_decay, param_arrays)

    # single global boolean mask (legacy, excludes scale/bias)
    def should_decay_fn(path, _):
        path_str = '/'.join(str(p.key) if hasattr(p, 'key') else str(p) for p in path)
        exclude = ['alpha_logits', 'scale', 'bias']
        return not any(pattern in path_str for pattern in exclude)
    mask_global_decay = jax.tree_util.tree_map_with_path(should_decay_fn, param_arrays)

    # tiered float mask for adaptive WD (explicit float32 to avoid x64 dtype issues)
    def get_tiered_multiplier(path, _):
        """Return float32 multiplier based on parameter type."""
        path_str = '/'.join(str(p.key) if hasattr(p, 'key') else str(p) for p in path)
        if 'alpha_logits' in path_str:
            return jnp.float32(0.0)
        elif 'scale' in path_str:
            return jnp.float32(scale_mult)
        elif 'bias' in path_str:
            return jnp.float32(bias_mult)
        return jnp.float32(1.0)
    mask_tiered = jax.tree_util.tree_map_with_path(get_tiered_multiplier, param_arrays)

    # collect params for logging
    norm_params = []
    bias_params = []
    mlp_params = []
    emb_params = []
    lm_head_params = []
    attention_params = []
    no_decay_params = []

    def collect_params(path, val):
        path_str = '/'.join(str(p.key) if hasattr(p, 'key') else str(p) for p in path)
        if is_norm_scale(path, val):
            norm_params.append(path_str)
        elif is_bias(path, val):
            bias_params.append(path_str)
        elif is_mlp_kernel(path, val):
            mlp_params.append(path_str)
        elif separate_lm_head and is_lm_head(path, val):
            lm_head_params.append(path_str)
        elif is_embedding(path, val):
            emb_params.append(path_str)
        elif is_attention_decay(path, val):
            attention_params.append(path_str)
        else:
            no_decay_params.append(path_str)
        return val

    jax.tree_util.tree_map_with_path(collect_params, param_arrays)

    param_counts = {
        'norm': len(norm_params),
        'bias': len(bias_params),
        'mlp': len(mlp_params),
        'emb': len(emb_params),
        'lm_head': len(lm_head_params),
        'attention': len(attention_params),
        'no_decay': len(no_decay_params)
    }

    return mask_norm, mask_bias, mask_mlp, mask_emb, mask_lm_head, mask_attention, mask_global_decay, mask_tiered, param_counts


@contextmanager
def patch_to_opt_state_for_factored_adafactor():
    """Monkey-patch to_opt_state to fix sharding for Adafactor's 1D factored stats.

    Adafactor creates factored statistics (v_row, v_col) that are 1D arrays. When params
    have 2D sharding annotations (e.g., ('data', None)), to_opt_state tries to apply those
    same annotations to the 1D arrays, causing a rank mismatch error.

    This patch detects when an array's rank doesn't match its sharding annotation length
    and sets eager_sharding=False for just those arrays. This preserves the original
    out_sharding metadata (important for weight tying identity) while disabling eager
    sharding so the 1D arrays get replicated. Full-sized arrays like ema keep
    eager_sharding=True and get properly sharded.
    """
    from flax.nnx.training import optimizer as opt_module
    from flax.nnx.variablelib import Variable
    from flax.nnx.training.optimizer import OptVariable, OptArray

    original_to_opt_state = opt_module.to_opt_state

    def patched_to_opt_state(tree):
        def _to_opt_state(x):
            if isinstance(x, Variable):
                value = x.get_value() if hasattr(x, 'get_value') else x.value
                orig_metadata = x.get_metadata() if hasattr(x, 'get_metadata') else {}
                metadata = dict(orig_metadata)

                out_sharding = (metadata.get('out_sharding')
                                or metadata.get('sharding_names')
                                or metadata.get('sharding'))
                if out_sharding is not None and hasattr(value, 'ndim'):
                    # out_sharding may be a tuple of logical names or a
                    # NamedSharding; normalize to a spec tuple for length check
                    if hasattr(out_sharding, 'spec'):
                        spec = out_sharding.spec
                    elif isinstance(out_sharding, PartitionSpec):
                        spec = out_sharding
                    else:
                        spec = out_sharding
                    if value.ndim < len(spec):
                        metadata['eager_sharding'] = False
                        metadata.pop('out_sharding', None)
                        metadata.pop('sharding', None)
                        metadata.pop('sharding_names', None)
                    elif hasattr(value, 'shape'):
                        n_devices = jax.device_count()
                        for dim, axis_name in enumerate(spec):
                            if axis_name is not None and dim < len(value.shape):
                                if value.shape[dim] < n_devices:
                                    metadata['eager_sharding'] = False
                                    break

                opt_state = OptVariable(value, **metadata)
            else:
                opt_state = OptArray(x)
            return opt_state

        tree = jax.tree.map(
            _to_opt_state,
            tree,
            is_leaf=lambda x: isinstance(x, Variable),
        )
        return tree

    opt_module.to_opt_state = patched_to_opt_state
    try:
        yield
    finally:
        opt_module.to_opt_state = original_to_opt_state


def _estimate_effective_infilling(args):
    """Estimate effective infilling ratio from corruption config."""
    if args.use_phase2_collator:
        r_ratios = [float(x) for x in str(args.infilling_ratio).split(',')]
        x_ratios = [float(x) for x in str(args.heavy_infilling_ratio).split(',')]
        avg_r = sum(r_ratios) / len(r_ratios)
        avg_x = sum(x_ratios) / len(x_ratios)
        mode_strs = args.mode_ratios.split(',')
        if len(mode_strs) == 3:
            return float(mode_strs[0]) * avg_r + float(mode_strs[1]) * avg_x
        return float(mode_strs[0]) * avg_r
    return float(str(args.infilling_ratio).split(',')[0])


def _create_gradaccum_schedule(args):
    """Return the constant gradient-accumulation multiplier for MultiSteps."""
    return args.gradient_accumulation_steps if args.gradient_accumulation_steps > 1 else 1


def create_optimizer(args, lr_schedule, model, wrt_filter=None, wd_schedule=None):
    """Create nnx.Optimizer instance with proper wrt filters for NNX 0.12+.

    Builds the Muon optimizer (orthogonalized momentum via Newton-Schulz for
    >=2D non-embedding weights, AdamW for 1D params and embeddings/lm_head) with
    grouped weight decay.

    Args:
        args: Training arguments namespace
        lr_schedule: Learning rate schedule object
        model: The FlaxHan2Han model instance
        wrt_filter: NNX filter for trainable params (default: nnx.Param for all params).
        wd_schedule: Optional pre-created WD schedule (for keeping a mutable reference).
            Use a custom filter for parameter freezing (e.g., SFT fine-tuning).
    """
    if wrt_filter is None:
        wrt_filter = nnx.Param
    # optimizer state dtype (mu / EMA accumulators); defaults to model_dtype for
    # backward compat. set --optimizer_state_dtype=float32 when --model_dtype=bfloat16
    # is insufficient for stable moment accumulation.
    opt_state_dtype_name = args.optimizer_state_dtype or args.model_dtype
    opt_state_dtype = getattr(jnp, opt_state_dtype_name)
    multisteps_every_k = _create_gradaccum_schedule(args)
    multisteps_display = str(multisteps_every_k)

    # create weight decay masks
    # separate lm_head mask when embeddings are untied and lm_head_weight_decay is set
    separate_lm_head = (not args.tie_input_output_embeddings) and args.lm_head_weight_decay > 0
    mask_norm, mask_bias, mask_mlp, mask_emb, mask_lm_head, mask_attention, mask_global_decay, mask_tiered, param_counts = create_weight_decay_masks(
        model, separate_lm_head=separate_lm_head, wrt_filter=wrt_filter
    )

    log_from_main_process(logger, 'info', f"Weight decay groups:")
    log_from_main_process(logger, 'info', f"  Norm scales ({args.norm_weight_decay}): {param_counts['norm']} params")
    log_from_main_process(logger, 'info', f"  Biases ({args.bias_weight_decay}): {param_counts['bias']} params")
    log_from_main_process(logger, 'info', f"  MLP kernels ({args.mlp_weight_decay}): {param_counts['mlp']} params")
    log_from_main_process(logger, 'info', f"  Embeddings ({args.embedding_weight_decay}): {param_counts['emb']} params")
    if separate_lm_head:
        log_from_main_process(logger, 'info', f"  LM head ({args.lm_head_weight_decay}): {param_counts['lm_head']} params")
    log_from_main_process(logger, 'info', f"  Attention/other ({args.weight_decay}): {param_counts['attention']} params")
    log_from_main_process(logger, 'info', f"  Not targeted: {param_counts['no_decay']} params")

    if args.optimizer == "muon":
        # Muon: NS-orthogonalized momentum for ND>=2 weight matrices; AdamW for
        # everything else. Routing matches DeepSeek V4 / Keller Jordan: vocab axis is
        # token-identity, not feature-channel, so wte/wce/wje/lm_head route to the AdamW arm.
        #
        # --ns_variant gram swaps the inner Newton-Schulz orthogonalize for Gram NS
        # (Dao 2026): iterate on R = X X^T with Polar-Express coeffs + restart at iter 2.
        # On v5e this saves NS FLOPs for non-square matrices but lacks the CUDA
        # symmetric-GEMM kernel speedup; numerical stability is more sensitive than
        # standard NS (see --gram_ns_dtype). gram ignores --muon_hybrid_ns.
        #
        # ND>2 handling: for an N-D weight tensor we treat the last two axes as
        # (reduction, output) and vmap over all preceding axes. This covers:
        # - 3D MoE expert weights (E, D, F): vmap over E
        # - 3D scanned dense layers (L, D, F): vmap over L
        # - 4D scanned MoE (L, E, D, F): vmap over (L, E)
        # Norm scales and biases that happen to be 2D (e.g. per-expert SubLN scales of
        # shape (E, F)) are name-matched out of Muon so they stay in the Adam arm.
        #
        # Update RMS scaling: optax applies sqrt(max(1, out/red)) which leaves Muon-arm
        # per-element update RMS at 1/sqrt(reduction_dim) -- ~30x smaller than AdamW at
        # the same LR, forcing the published Keller-Jordan recipe to use lr=0.02 for the
        # Muon arm + lr=3e-4 for AdamW (a ~67x ratio). DeepSeek V4 / Moonlight (Liu 2025)
        # solve this by rescaling each Muon update by sqrt(max(m, n)) * gamma instead;
        # they tune gamma=0.18 (our default) to match the empirical AdamW per-element
        # update RMS at a shared LR -- AdamW's m_hat/sqrt(v_hat) per-element RMS is ~0.18
        # in practice (correlated m and v), not 1.0 as the textbook approximation
        # suggests. With gamma=0.18, a single LR transfers between arms directly.
        # We get there by chaining a per-leaf multiplier of sqrt(reduction_dim) * gamma
        # after the optax factory:
        #     sqrt(max(m, n)) = sqrt(max(1, out/red)) * sqrt(reduction_dim)
        # so multiplying the factory output by sqrt(reduction_dim) * gamma converts
        # optax's scaling into DeepSeek's exactly.
        # MUON_ADAM_NAME_PATTERNS lives at module scope so train_step's per-arm
        # gradient norm logging uses the same classifier as the partitioner here.
        def muon_dim_nums_fn(params):
            def classify(path, p):
                if _muon_arm_label(path, p) == 'adam':
                    return None
                return optax.contrib.MuonDimensionNumbers(
                    reduction_axis=p.ndim - 2,
                    output_axis=p.ndim - 1,
                )
            return jax.tree_util.tree_map_with_path(classify, params)

        # NS coefficient + step resolution. See muon_gramNS.resolve_ns_coeffs:
        #   standard + hybrid_ns=False : 5-step Keller-Jordan
        #   standard + hybrid_ns=True  : 10-step DeepSeek hybrid (8 fast + 2 stab)
        #   gram     + (hybrid ignored): 5-step Polar-Express, restart at iter 2
        from muon_gramNS import resolve_ns_coeffs, scale_by_muon_with_variant
        ns_coeffs, ns_steps = resolve_ns_coeffs(args.ns_variant, args.muon_hybrid_ns)

        # parse --gram_ns_reset_iters comma-separated list -> tuple[int, ...]
        try:
            gram_reset_iters = tuple(
                int(s.strip()) for s in args.gram_ns_reset_iters.split(',') if s.strip()
            )
        except ValueError as e:
            raise ValueError(
                f"--gram_ns_reset_iters must be comma-separated ints, got "
                f"{args.gram_ns_reset_iters!r}: {e}"
            )
        gram_dtype_map = {
            'bf16': jnp.bfloat16,
            'fp16': jnp.float16,
            'fp32': jnp.float32,
        }
        gram_work_dtype = gram_dtype_map[args.gram_ns_dtype]

        # custom rescaling transform that lifts optax's sqrt(max(1, out/red)) to
        # DeepSeek's sqrt(max(m, n)) * gamma. We precompute per-leaf scalar scales
        # at init using the dim_nums tree -- shapes are static so this only happens
        # once.
        muon_gamma = float(args.muon_gamma)

        class _MuonRescaleState(NamedTuple):
            scales: Any

        def deepseek_muon_rescale():
            def init_fn(params):
                dim_nums = muon_dim_nums_fn(params)
                is_dim_or_none = lambda x: x is None or isinstance(x, optax.contrib.MuonDimensionNumbers)

                def compute_scale(p, dn):
                    if dn is None:
                        return jnp.asarray(1.0, dtype=jnp.float32)
                    red_axes = dn.reduction_axis
                    if isinstance(red_axes, int):
                        red_axes = (red_axes,)
                    red_dim = 1
                    for ax in red_axes:
                        red_dim *= p.shape[ax]
                    return jnp.asarray((red_dim ** 0.5) * muon_gamma, dtype=jnp.float32)

                scales = jax.tree.map(compute_scale, params, dim_nums, is_leaf=is_dim_or_none)
                return _MuonRescaleState(scales=scales)

            def update_fn(updates, state, params=None):
                new_updates = jax.tree.map(
                    lambda u, s: u * s.astype(u.dtype),
                    updates, state.scales,
                )
                return new_updates, state

            return optax.GradientTransformation(init_fn, update_fn)

        # the adam-arm weight decay mask. mask_global_decay is True for leaves
        # that should receive WD: it excludes 1D params (scale, bias, alpha_logits)
        # by name match. For muon-routed leaves the mask value is irrelevant
        # because they go through the muon transform's own add_decayed_weights
        # below; combine.partition will replace them with MaskedNode before
        # adamw sees them. When --muon_adam_wd_skip_1d is False, every adam-routed
        # leaf is decayed regardless of ndim (legacy behavior).
        if args.muon_adam_wd_skip_1d:
            adam_arm_wd_mask = mask_global_decay
        else:
            adam_arm_wd_mask = jax.tree_util.tree_map(lambda _: True, mask_global_decay)

        # Inline replacement for optax.contrib.muon (v0.2.6) that swaps the adam
        # branch's unmasked adamw for one with our own mask, and dispatches the
        # muon arm between {standard, gram} NS via muon_gramNS. Per-leaf
        # MaskedNode-wrapped dim_nums fn is unchanged.
        from optax.contrib._muon import _is_weight_dim_nums
        from optax.transforms import _masking as _opt_masking

        def _build_param_labels(muon_dim_nums_callable):
            def param_labels(params):
                dim_nums = muon_dim_nums_callable(params)
                populate_subtree_ = lambda dim_num, x: jax.tree.map(
                    lambda y: 'muon' if dim_num is not None else 'adam', x
                )
                return jax.tree.map(
                    populate_subtree_, dim_nums, params,
                    is_leaf=lambda x: x is None or _is_weight_dim_nums(x),
                )
            return param_labels

        def _build_normalized_dim_nums(muon_dim_nums_callable, param_labels_fn):
            def normalized(params):
                dim_nums = muon_dim_nums_callable(params)
                mask = jax.tree.map(lambda label: label == 'muon', param_labels_fn(params))
                is_leaf = lambda x: (x is None or _is_weight_dim_nums(x)
                                      or isinstance(x, _opt_masking.MaskedNode))
                populate_subtree_ = lambda dn, submask: jax.tree.map(
                    lambda m: dn if m else _opt_masking.MaskedNode(), submask
                )
                return jax.tree.map(populate_subtree_, dim_nums, mask, is_leaf=is_leaf)
            return normalized

        param_labels_fn = _build_param_labels(muon_dim_nums_fn)
        normalized_dim_nums_fn = _build_normalized_dim_nums(muon_dim_nums_fn, param_labels_fn)

        def muon_with_split_decay(learning_rate, weight_decay):
            adam_arm_wd = weight_decay * args.muon_adam_wd_ratio

            # NS prologue eps: 1e-7 matches Moonlight, NorMuon, and Dao Gram NS
            # references. Pre-2026-05-30 we hardcoded 1e-20 which was a no-op in
            # bf16 work_dtype (rounds to 0); with the new fp32-cast prologue in
            # muon_gramNS this no longer matters for underflow, but the value
            # still enters jnp.linalg.norm + eps in fp32 so 1e-7 is the correct
            # reference-matching floor.
            muon_arm_transform = scale_by_muon_with_variant(
                ns_coeffs=ns_coeffs,
                ns_steps=ns_steps,
                beta=args.muon_beta,
                eps=1e-7,
                mu_dtype=opt_state_dtype,
                nesterov=True,
                weight_dimension_numbers=normalized_dim_nums_fn,
                ns_variant=args.ns_variant,
                gram_reset_iters=gram_reset_iters,
                gram_work_dtype=gram_work_dtype,
                gram_chunk_size=args.gram_chunk_size,
            )

            # AdamW-arm-only gradient clipping. Muon's orthogonalization is
            # scale-invariant in input gradient magnitude (Ortho(cG) = Ortho(G)),
            # so per-step global-norm clipping has no effect on its update;
            # AdamW benefits from outlier protection on embed/lm_head/scale
            # gradients which in practice dominate global gradient mass.
            # Replaces the external train_step clip path, which was previously
            # disabled for --optimizer muon by use_global_clip's gating.
            adam_arm_transforms = []
            if args.clipnorm is not None and args.clipnorm > 0:
                adam_arm_transforms.append(optax.clip_by_global_norm(args.clipnorm))
            adam_arm_transforms.append(optax.scale_by_adam(
                b1=args.beta1,
                b2=args.beta2,
                eps=args.adam_eps,
                mu_dtype=opt_state_dtype,
                nesterov=True,
            ))

            # fine-grained WD for the adam arm. Mirrors adamw_with_groups. When
            # any per-group WD (--norm_weight_decay, --bias_weight_decay,
            # --embedding_weight_decay, --lm_head_weight_decay) is set, use
            # explicit masked add_decayed_weights for each group; this is the
            # only way RMSNorm scales and biases receive any WD under --optimizer
            # muon, since the legacy adam_arm_wd_mask (mask_global_decay)
            # excludes 'scale' and 'bias' by name. Falls back to uniform
            # adam_arm_wd via adam_arm_wd_mask when no per-group WDs are set,
            # preserving pre-2026-05 behavior. Per-group values are scaled by
            # muon_adam_wd_ratio so the existing knob still controls overall
            # adam-arm WD magnitude relative to the muon arm.
            ratio = args.muon_adam_wd_ratio
            use_fine_grained = (
                args.norm_weight_decay > 0
                or args.bias_weight_decay > 0
                or args.embedding_weight_decay > 0
                or (separate_lm_head and args.lm_head_weight_decay > 0)
            )
            if use_fine_grained:
                if args.norm_weight_decay > 0:
                    adam_arm_transforms.append(optax.masked(
                        optax.add_decayed_weights(args.norm_weight_decay * ratio), mask_norm))
                if args.bias_weight_decay > 0:
                    adam_arm_transforms.append(optax.masked(
                        optax.add_decayed_weights(args.bias_weight_decay * ratio), mask_bias))
                if args.embedding_weight_decay > 0:
                    adam_arm_transforms.append(optax.masked(
                        optax.add_decayed_weights(args.embedding_weight_decay * ratio), mask_emb))
                if separate_lm_head and args.lm_head_weight_decay > 0:
                    adam_arm_transforms.append(optax.masked(
                        optax.add_decayed_weights(args.lm_head_weight_decay * ratio), mask_lm_head))
            else:
                # gate on the concrete config arg, not the traced adam_arm_wd:
                # inject_hyperparams passes weight_decay as a tracer, so a
                # `> 0` test on the derived adam_arm_wd would raise
                # TracerBoolConversionError at init. this mirrors the legacy
                # single-global-mask path, which keyed solely off weight_decay,
                # and the muon arm above, which applies add_decayed_weights
                # unconditionally. the traced adam_arm_wd still sets the
                # magnitude; a zero schedule value just makes it a no-op.
                if args.weight_decay > 0 and args.muon_adam_wd_ratio > 0:
                    adam_arm_transforms.append(optax.masked(
                        optax.add_decayed_weights(adam_arm_wd), adam_arm_wd_mask))

            adam_arm_transforms.append(optax.scale_by_learning_rate(learning_rate))

            transforms_dict = {
                'muon': optax.chain(
                    muon_arm_transform,
                    optax.add_decayed_weights(weight_decay, mask=None),
                    optax.scale_by_learning_rate(learning_rate),
                ),
                'adam': optax.chain(*adam_arm_transforms),
            }

            chain_steps = [
                optax.partition(
                    transforms=transforms_dict,
                    param_labels=param_labels_fn,
                ),
                deepseek_muon_rescale(),
            ]
            return optax.chain(*chain_steps)

        if wd_schedule is None:
            wd_schedule = args.weight_decay

        base_opt = inject_hyperparams(muon_with_split_decay)(
            learning_rate=lr_schedule, weight_decay=wd_schedule,
        )

        # audit log: print the recipe + a few representative per-leaf scales so the
        # effective Muon-arm step magnitude is verifiable from training logs.
        param_state_for_log = nnx.state(model, wrt_filter)
        param_arrays_for_log = _nnx_to_arrays(nnx.pure(param_state_for_log))
        dim_nums_for_log = muon_dim_nums_fn(param_arrays_for_log)
        muon_examples = []
        adam_count = 0
        muon_count = 0
        seen_shapes = set()
        for path, p in jax.tree_util.tree_leaves_with_path(param_arrays_for_log):
            dn = dim_nums_for_log
            for part in path:
                dn = dn[part.key] if hasattr(part, 'key') else dn[part]
            if dn is None:
                adam_count += 1
                continue
            muon_count += 1
            red_axes = dn.reduction_axis if isinstance(dn.reduction_axis, tuple) else (dn.reduction_axis,)
            red_dim = 1
            for ax in red_axes:
                red_dim *= p.shape[ax]
            scale = (red_dim ** 0.5) * muon_gamma
            shape_key = (tuple(p.shape), tuple(red_axes))
            if shape_key not in seen_shapes and len(muon_examples) < 5:
                seen_shapes.add(shape_key)
                name = '/'.join(str(part.key) if hasattr(part, 'key') else str(part) for part in path)
                muon_examples.append((name, p.shape, red_axes, scale))

        # count adam-arm leaves that will actually receive WD via the mask, so the
        # audit log reflects the split rather than just printing the global ratio.
        # adafactor-routed leaves are excluded from this count via the label check.
        adam_wd_count = 0
        adam_nowd_count = 0
        for path, m_val in jax.tree_util.tree_leaves_with_path(adam_arm_wd_mask):
            label = param_labels_fn(param_arrays_for_log)
            for part in path:
                label = label[part.key] if hasattr(part, 'key') else label[part]
            if label == 'adam':
                if bool(m_val):
                    adam_wd_count += 1
                else:
                    adam_nowd_count += 1

        recipe_label = "Muon (DeepSeek V4 / Moonlight recipe)"
        rescale_note = (
            f"  Update rescaling: sqrt(max(m, n)) * gamma where gamma={muon_gamma:.4f} "
            f"(per-element update RMS ~ lr * gamma, matches AdamW arm at same LR)"
        )

        if args.ns_variant == "gram":
            ns_schedule_note = (
                f"  NS schedule: Gram NS (Dao 2026), Polar-Express 5 steps, "
                f"restart_iters={list(gram_reset_iters)}, work_dtype={args.gram_ns_dtype}"
            )
        else:
            ns_schedule_note = (
                f"  NS schedule: "
                + ("hybrid 10-step (8x fast + 2x stabilize)" if args.muon_hybrid_ns
                   else "standard 5-step (Keller-Jordan)")
            )

        adam_arm_clipnorm_str = (
            f"{args.clipnorm}"
            if args.clipnorm is not None and args.clipnorm > 0
            else "off"
        )
        log_from_main_process(logger, 'info',
            f"Using {recipe_label}: muon_beta={args.muon_beta}, "
            f"adam_b1={args.beta1}, adam_b2={args.beta2}, mu_dtype={opt_state_dtype_name}, "
            f"Muon-arm WD={args.weight_decay}, AdamW-arm WD={args.weight_decay * args.muon_adam_wd_ratio} "
            f"(ratio={args.muon_adam_wd_ratio}, skip_1d={args.muon_adam_wd_skip_1d}), "
            f"adam_arm_clipnorm={adam_arm_clipnorm_str} (muon-arm unclipped: scale-invariant by NS), "
            f"MultiSteps (k={multisteps_display})"
        )
        log_from_main_process(logger, 'info', ns_schedule_note)
        log_from_main_process(logger, 'info', rescale_note)
        partition_summary = (
            f"  Routed to Muon arm: {muon_count} leaves; "
            f"routed to AdamW arm: {adam_count} leaves "
            f"({adam_wd_count} decayed, {adam_nowd_count} skipped)"
        )
        log_from_main_process(logger, 'info', partition_summary)
        log_from_main_process(logger, 'info', f"  Example Muon-arm scales:")
        for name, shape, red_axes, scale in muon_examples:
            log_from_main_process(logger, 'info',
                f"    {name}: shape={tuple(shape)} red_axes={red_axes} -> scale={scale:.3f}"
            )
    else:
        raise ValueError(
            f"create_optimizer supports only --optimizer muon in this release; "
            f"got {args.optimizer!r}"
        )

    # wrap with MultiSteps for gradient accumulation
    final_opt = optax.MultiSteps(base_opt, every_k_schedule=multisteps_every_k)

    # create nnx.Optimizer instances with wrt attribute
    optimizer = nnx.Optimizer(model, final_opt, wrt=wrt_filter)

    return optimizer


MUON_ADAM_NAME_PATTERNS = (
    'wte', 'wce', 'wje', 'lm_head', 'classifier',
    'scale', 'bias', 'alpha_logits',
)


def _muon_arm_label(path, leaf):
    """Returns 'muon' or 'adam' for a single leaf under the muon partitioner.

    Mirrors the classifier in create_optimizer's muon block exactly: <2D
    params, embedding/lm_head leaves, and named scale/bias leaves go to the
    AdamW arm; weight matrices go to the Muon arm.
    """
    name = '/'.join(str(part.key) if hasattr(part, 'key') else str(part) for part in path)
    if leaf.ndim < 2:
        return 'adam'
    if any(k in name for k in MUON_ADAM_NAME_PATTERNS):
        return 'adam'
    return 'muon'


