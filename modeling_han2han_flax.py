#!/usr/bin/env python3
# coding: utf-8

from typing import Optional, Tuple, Union, Dict, Any, List, TYPE_CHECKING

import functools
import inspect
import logging
import os

if TYPE_CHECKING:
    from han2han_sampler import Han2HanSampler

import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from jax.extend.backend import get_backend

import numpy as np

# splash attention (TPU Pallas kernel)
_USE_SPLASH_ATTN = os.environ.get('TPU_USE_SPLASH_ATTN', '1').lower() not in ('0', 'false', 'no')


def _parse_splash_env(name):
    """Parse a splash override env var into None/'auto'/'on'/'off'.

    Empty or unset = None (no override, heuristic runs).
    """
    raw = os.environ.get(name)
    if raw is None or raw.strip() == '':
        return None
    val = raw.strip().lower()
    if val in ('on', '1', 'true', 'yes'):
        return 'on'
    if val in ('off', '0', 'false', 'no'):
        return 'off'
    if val == 'auto':
        return 'auto'
    raise ValueError(
        f"{name}={raw!r} invalid; expected one of 'on'/'off'/'auto' (or 1/0/true/false/yes/no)"
    )


_SPLASH_ENV_OVERRIDE = _parse_splash_env('HAN2HAN_SPLASH')
_CROSS_SPLASH_ENV_OVERRIDE = _parse_splash_env('HAN2HAN_CROSS_SPLASH')
if _SPLASH_ENV_OVERRIDE is not None:
    logging.info(f"HAN2HAN_SPLASH={_SPLASH_ENV_OVERRIDE} (overrides splash heuristic)")
if _CROSS_SPLASH_ENV_OVERRIDE is not None:
    logging.info(f"HAN2HAN_CROSS_SPLASH={_CROSS_SPLASH_ENV_OVERRIDE} (overrides cross-attn splash heuristic)")


_splash_attn_available = False
if not _USE_SPLASH_ATTN:
    logging.warning("TPU_USE_SPLASH_ATTN=0 - splash attention DISABLED, using standard dot product attention")
else:
    try:
        from tokamax._src.ops.experimental.tpu.splash_attention.splash_attention_mask import (
            CausalMask,
            LocalMask,
            FullMask,
        )
        from tokamax._src.ops.experimental.tpu.splash_attention.splash_attention_kernel import (
            make_splash_mha_single_device,
            make_splash_mqa_single_device,
            SplashConfig,
        )
        from tokamax._src.ops.experimental.tpu.splash_attention.base import (
            SegmentIds as SplashSegmentIds,
        )
        _splash_attn_available = True
    except ImportError as e:
        logging.warning(f"Splash attention import failed: {e}")
        _splash_attn_available = False

import flax
from flax import nnx
from flax.nnx import remat
from flax.nnx.nn.attention import dot_product_attention_weights

from transformers.utils.generic import ModelOutput
from han2han_config import Han2HanConfig
from normalization import RMSNorm
from rope import _apply_rotary_pos_emb, _apply_rotary_pos_emb_with_ids, _precompute_freqs_cis

from logging_utils import log_from_main_process
from sharding_utils import get_global_mesh, get_global_mesh_config, maybe_constrain_activation

import logging
logger = logging.getLogger(__name__)


def get_remat_policy(policy_name: str):
    """Return a jax checkpoint policy for graduated rematerialization.

    Policies control which intermediate tensors are saved vs recomputed
    during the backward pass. Tensors are identified by checkpoint_name
    annotations placed on Q/K/V projections, output projections, MLP
    projections, and attention weights throughout the model.

    Args:
        policy_name: One of:
            - "full": recompute everything (most memory-efficient, slowest)
            - "none": save everything (fastest, most memory)
            - "save_attn_weights": save attention weights only (best ratio:
              avoids O(seq^2 * heads) recompute, projections are cheap)
            - "save_qkv_proj": save Q/K/V projections on device
            - "minimal": save all projections + attn weights on device
            - "qkv_proj_offloaded": offload Q/K/V to host pinned memory
            - "attn_weights_offloaded": offload only attn weights to host
            - "minimal_offloaded": offload all projections to host pinned memory
            - "save_projs_offload_attn": save projections on device, offload
              attn weights to host (use when HBM has ~30% headroom; fastest
              practical policy when attn_weights are too large to fit)

    Returns:
        A jax checkpoint policy, or None for full/none.
    """
    if policy_name in ("full", "none"):
        return None
    elif policy_name == "save_attn_weights":
        return jax.checkpoint_policies.save_only_these_names(
            "attn_weights",
        )
    elif policy_name == "save_qkv_proj":
        return jax.checkpoint_policies.save_only_these_names(
            "query_proj", "value_proj", "key_proj",
        )
    elif policy_name == "minimal":
        return jax.checkpoint_policies.save_only_these_names(
            "query_proj", "value_proj", "key_proj",
            "out_proj", "mlpwi_0", "mlpwi_1", "mlpwo",
            "attn_weights",
        )
    elif policy_name == "qkv_proj_offloaded":
        return jax.checkpoint_policies.save_and_offload_only_these_names(
            names_which_can_be_saved=[],
            names_which_can_be_offloaded=["query_proj", "value_proj", "key_proj"],
            offload_src="device",
            offload_dst="pinned_host",
        )
    elif policy_name == "attn_weights_offloaded":
        return jax.checkpoint_policies.save_and_offload_only_these_names(
            names_which_can_be_saved=[],
            names_which_can_be_offloaded=["attn_weights"],
            offload_src="device",
            offload_dst="pinned_host",
        )
    elif policy_name == "minimal_offloaded":
        return jax.checkpoint_policies.save_and_offload_only_these_names(
            names_which_can_be_saved=[],
            names_which_can_be_offloaded=[
                "decoder_layer_input", "attn_weights",
                "query_proj", "value_proj", "key_proj",
                "out_proj", "mlpwi_0", "mlpwi_1", "mlpwo",
            ],
            offload_src="device",
            offload_dst="pinned_host",
        )
    elif policy_name == "save_projs_offload_attn":
        return jax.checkpoint_policies.save_and_offload_only_these_names(
            names_which_can_be_saved=[
                "query_proj", "key_proj", "value_proj", "out_proj",
            ],
            names_which_can_be_offloaded=["attn_weights"],
            offload_src="device",
            offload_dst="pinned_host",
        )
    else:
        raise ValueError(
            f"Unknown remat_policy: {policy_name!r}. "
            f"Valid options: full, none, save_attn_weights, save_qkv_proj, "
            f"minimal, qkv_proj_offloaded, attn_weights_offloaded, "
            f"minimal_offloaded, save_projs_offload_attn"
        )


class StaticLookup(nnx.Variable):
    """A non-trainable Variable for lookup tables."""
    pass


class SubwordLookups(nnx.Pytree):
    """
    Container for subword lookup tables that shouldn't be treated as trainable parameters.
    """
    def __init__(self, lookups: dict):
        if "jbu" in lookups:
            self.jbu = StaticLookup(value=lookups["jbu"])
        else:
            self.jbu = None
        if "cbu" in lookups:
            self.cbu = StaticLookup(value=lookups["cbu"])
        else:
            self.cbu = None

    def __contains__(self, key):
        """Support 'in' operator for checking if a lookup exists."""
        return hasattr(self, key) and getattr(self, key) is not None

    def __getitem__(self, key):
        """Allow dictionary-style access for backwards compatibility."""
        var = getattr(self, key, None)
        return var.value if var is not None else None


@flax.struct.dataclass
class FlaxSeq2SeqLMOutput(ModelOutput):
    logits: jnp.ndarray = None
    hidden_states: Optional[Tuple[jnp.ndarray]] = None
    sentence_embeddings: Optional[jnp.ndarray] = None
    past_key_values: Optional[Tuple[Tuple[jnp.ndarray]]] = None
    decoder_hidden_states: Optional[Tuple[jnp.ndarray]] = None
    decoder_attentions: Optional[Tuple[jnp.ndarray]] = None
    cross_attentions: Optional[Tuple[jnp.ndarray]] = None
    encoder_last_hidden_state: Optional[jnp.ndarray] = None
    encoder_hidden_states: Optional[Tuple[jnp.ndarray]] = None
    encoder_attentions: Optional[Tuple[jnp.ndarray]] = None


@flax.struct.dataclass
class FlaxSequenceClassifierOutput(ModelOutput):
    logits: jnp.ndarray = None
    hidden_states: Optional[jnp.ndarray] = None


@flax.struct.dataclass
class FlaxBaseModelOutputWithPastAndCrossAttentions(ModelOutput):
    last_hidden_state: jnp.ndarray = None
    past_key_values: Optional[Tuple[Tuple[jnp.ndarray]]] = None
    hidden_states: Optional[Tuple[jnp.ndarray]] = None
    attentions: Optional[Tuple[jnp.ndarray]] = None
    cross_attentions: Optional[Tuple[jnp.ndarray]] = None

    def __post_init__(self):
        # skip ModelOutput.__post_init__ which doesn't recognize JAX arrays
        # as tensors (is_tensor checks for torch only) and tries to setattr
        # on this frozen dataclass
        pass


# create generation state dataclass similar to gemma NNX's _SamplingState
@flax.struct.dataclass
class GenerationState:
    decoding_step: jnp.int32
    token_buffer: jnp.ndarray
    cache: Optional[Any]
    done: jnp.ndarray
    total_steps: int = flax.struct.field(pytree_node=False)
    encoder_hidden_states: jnp.ndarray = None
    encoder_attention_mask: jnp.ndarray = None
    eos_token_id: Optional[int] = flax.struct.field(pytree_node=False, default=None)
    pad_token_id: int = flax.struct.field(pytree_node=False, default=0)
    min_length: int = flax.struct.field(pytree_node=False, default=0)
    do_sample: bool = flax.struct.field(pytree_node=False, default=False)
    temperature: float = flax.struct.field(pytree_node=False, default=1.0)
    top_k: int = flax.struct.field(pytree_node=False, default=50)
    top_p: float = flax.struct.field(pytree_node=False, default=1.0)
    repetition_penalty: float = flax.struct.field(pytree_node=False, default=1.0)
    no_repeat_ngram_size: int = flax.struct.field(pytree_node=False, default=0)
    # track the initial prompt length for proper cache handling
    prompt_length: int = flax.struct.field(pytree_node=False, default=1)
    # token suppression for banning specific tokens (converted to tuple for pytree compatibility)
    suppress_tokens: Optional[tuple] = flax.struct.field(pytree_node=False, default=None)
    # for sampling, we'll use a seed that gets folded with step
    rng_seed: Optional[jax.Array] = None


# beam search helper dataclass (defined at class level for use in beam_search method)
@flax.struct.dataclass
class BeamSearchState:
    """State for beam search generation with pre-allocated buffers."""
    decoding_step: jnp.int32
    decoder_tokens: jnp.ndarray  # (beam_batch_size, buffer_size)
    decoder_attention_mask: jnp.ndarray  # (beam_batch_size, buffer_size)
    beam_scores: jnp.ndarray  # (beam_batch_size,)
    cache: Optional[Any]
    finished_mask: jnp.ndarray  # (batch_size, num_beams) - tracks finished beams
    finished_scores: jnp.ndarray  # (batch_size, num_beams)
    finished_sequences: jnp.ndarray  # (batch_size, num_beams, buffer_size)
    encoder_hidden_states: jnp.ndarray
    encoder_attention_mask: jnp.ndarray
    # static fields (not traced) - required fields without defaults
    batch_size: int = flax.struct.field(pytree_node=False)
    num_beams: int = flax.struct.field(pytree_node=False)
    max_length: int = flax.struct.field(pytree_node=False)
    min_length: int = flax.struct.field(pytree_node=False)
    # optional fields with defaults must come after required fields
    eos_token_id: Optional[int] = flax.struct.field(pytree_node=False, default=None)
    pad_token_id: int = flax.struct.field(pytree_node=False, default=0)
    length_penalty: float = flax.struct.field(pytree_node=False, default=1.0)
    temperature: float = flax.struct.field(pytree_node=False, default=1.0)
    top_k: int = flax.struct.field(pytree_node=False, default=50)
    top_p: float = flax.struct.field(pytree_node=False, default=1.0)
    repetition_penalty: float = flax.struct.field(pytree_node=False, default=1.0)
    no_repeat_ngram_size: int = flax.struct.field(pytree_node=False, default=0)
    early_stopping: bool = flax.struct.field(pytree_node=False, default=False)
    use_cache: bool = flax.struct.field(pytree_node=False, default=True)
    use_fixed_length_generation: bool = flax.struct.field(pytree_node=False, default=False)
    # track the initial prompt length for proper cache handling
    prompt_length: int = flax.struct.field(pytree_node=False, default=1)
    # token suppression for banning specific tokens (converted to tuple for pytree compatibility)
    suppress_tokens: Optional[tuple] = flax.struct.field(pytree_node=False, default=None)
    rng_seed: Optional[jax.Array] = None


def shift_tokens_right(input_ids: jnp.ndarray, pad_token_id: int, decoder_start_token_id: int) -> jnp.ndarray:
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = jnp.zeros_like(input_ids)
    shifted_input_ids = shifted_input_ids.at[:, 1:].set(input_ids[:, :-1])
    shifted_input_ids = shifted_input_ids.at[:, 0].set(decoder_start_token_id)

    shifted_input_ids = jnp.where(shifted_input_ids == -100, pad_token_id, shifted_input_ids)
    return shifted_input_ids


class LayerNorm(nnx.Module):
    """Standard LayerNorm with mean centering for maximum stability."""
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-8,
        rngs: nnx.Rngs = None,
        dtype: jnp.dtype = jnp.float32,
        sharding: tuple = ('model',),
        use_bias: bool = True,
        param_dtype: Optional[jnp.dtype] = None,
    ):
        self.eps = eps
        self.hidden_size = hidden_size
        if param_dtype is None:
            param_dtype = dtype
        key = rngs.params()
        self.scale = nnx.Param(nnx.with_partitioning(nnx.initializers.ones, ((sharding[0],) if sharding else None)
                                           )(key, (hidden_size,), param_dtype))
        self.bias = nnx.data(None)
        if use_bias:
            self.bias = nnx.Param(nnx.with_partitioning(nnx.initializers.zeros, ((sharding[0],) if sharding else None)
                                            )(key, (hidden_size,), param_dtype))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # mean/var in f32 for stability, normalize in x.dtype. scale/bias stored
        # at param_dtype (may differ from x.dtype) so we promote at use site.
        x_f32 = x.astype(jnp.float32)
        mean = jnp.mean(x_f32, axis=-1, keepdims=True)
        var = jnp.var(x_f32, axis=-1, keepdims=True)
        inv_std = (1.0 / jnp.sqrt(var + self.eps)).astype(x.dtype)
        mean = mean.astype(x.dtype)
        x_norm = (x - mean) * inv_std
        scale = self.scale.astype(x.dtype)
        if self.bias is not None:
            return x_norm * scale + self.bias.astype(x.dtype)
        return x_norm * scale


def create_decoder_norm(
    config: 'Han2HanConfig',
    hidden_size: int,
    rngs: nnx.Rngs,
    dtype: jnp.dtype = jnp.float32,
    sharding: tuple = ('model',),
    param_dtype: Optional[jnp.dtype] = None,
) -> nnx.Module:
    """Create the appropriate normalization layer for decoder based on config."""
    norm_type = getattr(config, 'decoder_norm_type', 'rmsnorm')
    if param_dtype is None:
        param_dtype = dtype
    if norm_type == 'layernorm':
        return LayerNorm(
            hidden_size=hidden_size, eps=config.layer_norm_epsilon,
            dtype=dtype, rngs=rngs, sharding=sharding, use_bias=config.use_bias,
            param_dtype=param_dtype,
        )
    elif norm_type == 'rmsnorm_bias':
        return RMSNorm(
            hidden_size=hidden_size, eps=config.layer_norm_epsilon,
            dtype=dtype, rngs=rngs, sharding=sharding, use_bias=True,
            param_dtype=param_dtype,
        )
    else:  # default: rmsnorm
        return RMSNorm(
            hidden_size=hidden_size, eps=config.layer_norm_epsilon,
            dtype=dtype, rngs=rngs, sharding=sharding,
            param_dtype=param_dtype,
        )


def create_encoder_norm(
    config: 'Han2HanConfig',
    hidden_size: int,
    rngs: nnx.Rngs,
    dtype: jnp.dtype = jnp.float32,
    sharding: tuple = ('model',),
    param_dtype: Optional[jnp.dtype] = None,
) -> nnx.Module:
    """Create the appropriate normalization layer for encoder based on config."""
    norm_type = getattr(config, 'encoder_norm_type', 'rmsnorm')
    if param_dtype is None:
        param_dtype = dtype
    if norm_type == 'layernorm':
        return LayerNorm(
            hidden_size=hidden_size, eps=config.layer_norm_epsilon,
            dtype=dtype, rngs=rngs, sharding=sharding, use_bias=config.use_bias,
            param_dtype=param_dtype,
        )
    elif norm_type == 'rmsnorm_bias':
        return RMSNorm(
            hidden_size=hidden_size, eps=config.layer_norm_epsilon,
            dtype=dtype, rngs=rngs, sharding=sharding, use_bias=True,
            param_dtype=param_dtype,
        )
    else:  # default: rmsnorm
        return RMSNorm(
            hidden_size=hidden_size, eps=config.layer_norm_epsilon,
            dtype=dtype, rngs=rngs, sharding=sharding,
            param_dtype=param_dtype,
        )


def _make_attn_dropout_mask(dropout_rng, dropout_rate, broadcast_dropout, qkv_ndim, attn_w_shape, dtype):
    keep_prob = 1.0 - dropout_rate
    if broadcast_dropout:
        # dropout is broadcast across the batch + head dimensions
        dropout_shape = tuple([1] * (qkv_ndim - 2)) + attn_w_shape[-2:]
        keep = jax.random.bernoulli(dropout_rng, keep_prob, dropout_shape)
    else:
        keep = jax.random.bernoulli(dropout_rng, keep_prob, attn_w_shape)
    return keep.astype(dtype) / jnp.asarray(keep_prob, dtype=dtype)


def _build_splash_mask(causal, window_size, q_len, kv_len, is_cross_attn):
    """Build a 2D splash Mask from attention configuration.

    Returns a single Mask object (Tokamax expects 2D, not per-head 3D).
    Cross-attention is never causal; sliding cross-attention is symmetric
    around the decoder query position when window_size > 0.
    """
    if is_cross_attn and window_size > 0:
        return LocalMask(
            shape=(q_len, kv_len), window_size=(window_size, window_size), offset=0)
    elif is_cross_attn:
        return FullMask(_shape=(q_len, kv_len))
    elif causal and window_size > 0:
        return CausalMask(shape=(q_len, kv_len)) & LocalMask(
            shape=(q_len, kv_len), window_size=(window_size, 0), offset=0)
    elif causal:
        return CausalMask(shape=(q_len, kv_len))
    elif window_size > 0:
        return LocalMask(
            shape=(q_len, kv_len), window_size=(window_size, window_size), offset=0)
    else:
        return FullMask(_shape=(q_len, kv_len))


def maybe_flash_attention(q, k, v, mask=None, q_segment_ids=None, kv_segment_ids=None,
                          window_size=0, is_cross_attention=False, **kwargs):
    backend = get_backend().platform

    causal = kwargs.pop('causal', False)
    dropout_rng = kwargs.pop('dropout_rng', None)
    dropout_rate = kwargs.pop('dropout_rate', 0.0)
    deterministic = kwargs.pop('deterministic', True)
    dtype = kwargs.pop('dtype', q.dtype)
    num_kv_heads = kwargs.pop('num_kv_heads', None)
    q_pre_scaled = kwargs.pop('q_pre_scaled', False)

    q_len = q.shape[1]
    kv_len = k.shape[1]
    num_heads = q.shape[2]
    head_dim = q.shape[-1]
    if num_kv_heads is None:
        num_kv_heads = k.shape[2]
    use_gqa = (num_kv_heads != num_heads)

    # window_size may be a traced value (from scan), so avoid Python conditionals.
    window_is_static = not isinstance(window_size, jax.core.Tracer)

    # resolution: HAN2HAN_CROSS_SPLASH (cross-attn only) > HAN2HAN_SPLASH > heuristic.
    splash_mode = _CROSS_SPLASH_ENV_OVERRIDE if is_cross_attention else None
    if splash_mode is None:
        splash_mode = _SPLASH_ENV_OVERRIDE
    if splash_mode is None:
        splash_mode = 'auto'

    if not window_is_static or splash_mode == 'off':
        use_splash = False
    else:
        # 'on' bypasses any seq-len heuristic; the 1024 floor on both axes is a
        # hard Pallas block-size requirement, not a heuristic, so it always
        # gates. 'auto' is currently equivalent (no further heuristic remains
        # after dropping the legacy q_len>=2048 v4-megacore threshold).
        use_splash = (
            backend == 'tpu'
            and _splash_attn_available
            and q.dtype == jnp.bfloat16
            and q_len >= 1024
            and kv_len >= 1024
        )

    if not use_splash:
        # build sliding window mask with traced-compatible ops
        # when window_size==0, effective_ws=kv_len (full attention, all-True mask)
        effective_ws = jnp.where(window_size > 0, window_size, kv_len)
        rows = jnp.arange(q_len)[:, None]
        cols = jnp.arange(kv_len)[None, :]
        diff = rows - cols
        if causal:
            sw_mask = (diff >= 0) & (diff < effective_ws)
        else:
            sw_mask = (diff > -effective_ws) & (diff < effective_ws)
        sw_mask = sw_mask[None, None, :, :]
        mask = mask * sw_mask if mask is not None else sw_mask

        if use_gqa or q_pre_scaled:
            # T5Gemma 1 einsum-with-reshape GQA path (modules.py:236-279). Q is
            # already pre-scaled when q_pre_scaled=True; otherwise apply default scale.
            if not q_pre_scaled:
                q = q * jnp.asarray(head_dim ** -0.5, dtype=q.dtype)

            if use_gqa:
                G = num_heads // num_kv_heads
                q_r = q.reshape(q.shape[0], q_len, num_kv_heads, G, head_dim)
                logits = jnp.einsum('BTKGH,BSKH->BTKGS', q_r, k)
                logits = logits.reshape(q.shape[0], q_len, num_heads, kv_len)
            else:
                logits = jnp.einsum('BTNH,BSNH->BTNS', q, k)

            # match dot_product_attention_weights layout: (B, H, q, kv)
            logits = logits.transpose(0, 2, 1, 3)

            if mask is not None:
                big_neg = jnp.finfo(logits.dtype).min
                logits = jnp.where(mask, logits, big_neg)

            attn_weights = jax.nn.softmax(logits, axis=-1).astype(dtype)

            if not deterministic and dropout_rate > 0.0 and dropout_rng is not None:
                attn_weights = attn_weights * _make_attn_dropout_mask(
                    dropout_rng, dropout_rate, True, q.ndim, attn_weights.shape, dtype
                )

            attn_weights = checkpoint_name(attn_weights, "attn_weights")

            if use_gqa:
                G = num_heads // num_kv_heads
                aw_r = attn_weights.reshape(attn_weights.shape[0], num_kv_heads, G, q_len, kv_len)
                encoded = jnp.einsum('BKGTS,BSKD->BTKGD', aw_r, v)
                encoded = encoded.reshape(encoded.shape[0], q_len, num_heads, head_dim)
            else:
                encoded = jnp.einsum('BNTS,BSND->BTND', attn_weights, v)
            return encoded

        attn_weights = dot_product_attention_weights(q, k, mask=mask,
                                                     is_causal=causal,
                                                     **kwargs)
        attn_weights = checkpoint_name(attn_weights, "attn_weights")
        return jnp.einsum('...hqk,...khd->...qhd', attn_weights, v)

    # --- splash attention path ---

    splash_mask = _build_splash_mask(
        causal, window_size, q_len, kv_len, is_cross_attention)
    # MQA constructor when num_kv_heads == 1; MHA constructor handles arbitrary
    # GQA automatically (asserts num_q_heads % num_kv_heads == 0; computes
    # q_heads_per_kv_head from input K shape).
    if num_kv_heads == 1:
        splash_kernel = make_splash_mqa_single_device(
            splash_mask, config=SplashConfig.get_default(),
            partial_mask_blocks_dtype=np.int32)
    else:
        splash_kernel = make_splash_mha_single_device(
            splash_mask, config=SplashConfig.get_default(),
            partial_mask_blocks_dtype=np.int32)

    # prepare segment_ids (int32 for Pallas)
    if q_segment_ids is not None:
        if kv_segment_ids is None:
            kv_segment_ids = q_segment_ids
        seg_q = q_segment_ids.astype(jnp.int32)
        seg_kv = kv_segment_ids.astype(jnp.int32)
        has_segments = True
    elif mask is not None:
        # synthesize segment_ids from attention mask for non-packed training
        if mask.ndim == 4:
            seg_q = (mask[:, 0, 0, :q_len] > 0).astype(jnp.int32)
        else:
            seg_q = (mask[:, :q_len] > 0).astype(jnp.int32)
        seg_kv = seg_q
        has_segments = True
    else:
        has_segments = False

    # splash expects (num_heads, seq_len, head_dim) per example; vmap over batch
    # input: (B, seq, heads, d) -> (B, heads, seq, d) for splash
    q_t = q.transpose(0, 2, 1, 3)
    k_t = k.transpose(0, 2, 1, 3)
    v_t = v.transpose(0, 2, 1, 3)

    # MQA constructor expects 2D K/V (no head axis); squeeze the kv-head dim away.
    if num_kv_heads == 1:
        k_t = k_t.squeeze(1)
        v_t = v_t.squeeze(1)

    # splash kernel does NOT scale internally; pre-scale Q if caller hasn't.
    if not q_pre_scaled:
        q_t = q_t * jnp.asarray(head_dim ** -0.5, dtype=q_t.dtype)

    def _splash_per_example(q_i, k_i, v_i, seg_q_i, seg_kv_i):
        seg = SplashSegmentIds(q=seg_q_i, kv=seg_kv_i) if has_segments else None
        return splash_kernel(q_i, k_i, v_i, segment_ids=seg)

    is_multi_device = jax.device_count() > 1

    if is_multi_device:
        from jax.sharding import PartitionSpec as P
        mesh = get_global_mesh()
        # MQA squeezes the kv-head axis, dropping K/V to 3D.
        kv_spec = P('data', None, None) if num_kv_heads == 1 else P('data', None, None, None)

        if has_segments:
            in_specs = (
                P('data', None, None, None),
                kv_spec,
                kv_spec,
                P('data', None),
                P('data', None),
            )

            def splash_sharded(q_shard, k_shard, v_shard, sq, skv):
                return jax.vmap(_splash_per_example)(
                    q_shard, k_shard, v_shard, sq, skv)

            attn_out = jax.shard_map(
                splash_sharded,
                mesh=mesh,
                in_specs=in_specs,
                out_specs=P('data', None, None, None),
                check_vma=False,
            )(q_t, k_t, v_t, seg_q, seg_kv)
        else:
            in_specs = (
                P('data', None, None, None),
                kv_spec,
                kv_spec,
            )

            def _splash_no_seg(q_i, k_i, v_i):
                return splash_kernel(q_i, k_i, v_i, segment_ids=None)

            def splash_sharded(q_shard, k_shard, v_shard):
                return jax.vmap(_splash_no_seg)(q_shard, k_shard, v_shard)

            attn_out = jax.shard_map(
                splash_sharded,
                mesh=mesh,
                in_specs=in_specs,
                out_specs=P('data', None, None, None),
                check_vma=False,
            )(q_t, k_t, v_t)
    else:
        if has_segments:
            attn_out = jax.vmap(_splash_per_example)(
                q_t, k_t, v_t, seg_q, seg_kv)
        else:
            attn_out = jax.vmap(
                lambda qi, ki, vi: splash_kernel(qi, ki, vi, segment_ids=None)
            )(q_t, k_t, v_t)

    # post-kernel dropout (splash doesn't support in-kernel dropout)
    if not deterministic and dropout_rate > 0.0:
        multiplier = _make_attn_dropout_mask(
            dropout_rng, dropout_rate, True, k.ndim, attn_out.shape, dtype)
        attn_out *= multiplier

    # (B, heads, seq, d) -> (B, seq, heads, d)
    return attn_out.transpose(0, 2, 1, 3)


class FlaxHan2HanAttention(nnx.Module):
    def __init__(
        self,
        config: Han2HanConfig,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
        is_cross_attention: bool = False,
        is_causal: bool = True,
        sharding: tuple = ('model',None),
        layer_idx: int = -1,
        is_encoder: bool = False,
        rope_theta: Optional[float] = None,
        param_dtype: Optional[jnp.dtype] = None,
    ):
        if param_dtype is None:
            param_dtype = dtype
        # helper to create attention projections
        def make_proj(in_features, out_features):
            return nnx.Linear(
                in_features=in_features,
                out_features=out_features,
                dtype=dtype,
                rngs=rngs,
                param_dtype=param_dtype,
                use_bias=config.use_bias,
                kernel_init=nnx.with_partitioning(
                    config.make_kernel_init(dtype=dtype),
                    sharding=sharding),
                bias_init=nnx.with_partitioning(
                    config.make_bias_init(),
                    sharding=((sharding[0],) if sharding else None)))

        assert config.num_heads is not None, "num_heads must be specified for MHA attention"
        if is_cross_attention:
            num_heads = config.cross_attn_num_heads if config.cross_attn_num_heads is not None else config.num_heads
            num_kv_heads = config.cross_attn_num_kv_heads
        else:
            num_heads = config.num_heads
            num_kv_heads = config.num_kv_heads
        head_dim = config.head_dim
        if num_kv_heads is None:
            num_kv_heads = num_heads
        if head_dim is None:
            head_dim = config.d_prime // config.num_heads
        q_proj_dim = num_heads * head_dim
        kv_proj_dim = num_kv_heads * head_dim

        # qkv projections: Q has full num_heads; K/V have num_kv_heads (GQA/MQA).
        self.query = make_proj(config.d_model, q_proj_dim)
        self.key = make_proj(config.d_model, kv_proj_dim)
        self.value = make_proj(config.d_model, kv_proj_dim)

        self.attn_sub_norm = nnx.data(None)
        if config.use_sub_ln:
            norm_sharding = (sharding[0],) if sharding else None
            self.attn_sub_norm = RMSNorm(q_proj_dim, rngs=rngs, dtype=dtype,
                                         sharding=norm_sharding, use_bias=False,
                                         param_dtype=param_dtype, eps=config.layer_norm_epsilon)

        # per-head RMSNorm on Q and K (Gemma 3 / T5Gemma 2 style); broadcasts over head axis
        self.q_norm = nnx.data(None)
        self.k_norm = nnx.data(None)
        if config.use_qk_norm:
            norm_sharding = None
            self.q_norm = RMSNorm(head_dim, rngs=rngs, dtype=dtype,
                                  sharding=norm_sharding, use_bias=False,
                                  param_dtype=param_dtype, eps=config.layer_norm_epsilon)
            self.k_norm = RMSNorm(head_dim, rngs=rngs, dtype=dtype,
                                  sharding=norm_sharding, use_bias=False,
                                  param_dtype=param_dtype, eps=config.layer_norm_epsilon)

        self.c_proj = make_proj(q_proj_dim, config.d_model)

        self.rope_theta = rope_theta if rope_theta is not None else config.rope_theta
        # Per-call rope_theta resolution: when both `rope_theta` and
        # `rope_theta_sliding` are configured, the actual value depends on whether
        # this layer is *running* as sliding (effective_window_size > 0) or full
        # (effective_window_size == 0). Storing both here lets `__call__` pick
        # per-step from `override_window_size`, which is what makes hybrid attention
        # actually honor `rope_theta != rope_theta_sliding` inside a scan stack.
        # The pre-fix path baked `effective_rope_theta` from `attention_type` here
        # at __init__ time, which silently collapsed scan-stack rope to
        # position_specs[0]'s value -- see [pytorch_phase1_parity_harness]/V1
        # post-mortem.
        self.rope_theta_full = config.rope_theta
        self.rope_theta_sliding_static = getattr(config, 'rope_theta_sliding', None)
        self.max_positions = config.n_positions

        # use cross_attn_pdrop for cross-attention layers
        dropout_rate = config.cross_attn_pdrop if is_cross_attention else config.attn_pdrop
        self.attn_dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        self.attn_pdrop = dropout_rate

        self.is_cross_attention = is_cross_attention
        self.is_causal = is_causal
        self.layer_idx = layer_idx
        self.is_encoder = is_encoder
        self.window_size = config.sliding_window_size

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.d_model = config.d_model
        self.head_dim = head_dim
        self.q_proj_dim = q_proj_dim
        self.kv_proj_dim = kv_proj_dim
        self.d_prime = q_proj_dim  # back-compat alias for q_proj_dim
        # HF Gemma 3 convention: query_pre_attn_scalar parameterizes scaling = scalar ** -0.5.
        default_q_multiplier = float(head_dim) ** -0.5
        if config.query_pre_attn_scalar is not None:
            self.q_multiplier = float(config.query_pre_attn_scalar) ** -0.5
        else:
            self.q_multiplier = default_q_multiplier
        self.use_gqa = (num_kv_heads != num_heads)
        self.kv_groups = num_heads // num_kv_heads
        self._has_custom_q_multiplier = abs(self.q_multiplier - default_q_multiplier) > 1e-12
        # Flag set when at least one new attention feature is active. Forces
        # the inline custom path that handles GQA and Q pre-scaling uniformly.
        # Legacy yamls leave this False, preserving bit-identical behavior on
        # the `dot_product_attention_weights` path.
        self._needs_inline_attn = (
            self.use_gqa
            or self._has_custom_q_multiplier
        )

    def _call_proj(self, proj, x, deterministic=None, rngs=None):
        """Call projection layer."""
        return proj(x)

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: jnp.ndarray,
        position_ids: Optional[jnp.ndarray] = None,
        segment_ids: Optional[jnp.ndarray] = None,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        encoder_position_ids: Optional[jnp.ndarray] = None,
        encoder_segment_ids: Optional[jnp.ndarray] = None,
        init_cache: bool = False,
        past_key_value: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
        rngs: Optional[nnx.Rngs] = None,
        override_window_size: Optional[jnp.ndarray] = None,
    ) -> Tuple[Union[jnp.ndarray, Tuple[jnp.ndarray]], ...]:

        deterministic = self.attn_dropout.deterministic
        use_causal = self.is_causal

        B, q_len, _ = hidden_states.shape

        # 1. QKV projections
        if self.is_cross_attention and encoder_hidden_states is not None:
            query_states_unrotated = checkpoint_name(
                self._call_proj(self.query, hidden_states, deterministic, rngs), "query_proj")
            q_len = hidden_states.shape[1]
            kv_len = encoder_hidden_states.shape[1]
            q_segment_ids = segment_ids
            kv_segment_ids = encoder_segment_ids
            q_position_ids = position_ids
            kv_position_ids = encoder_position_ids

            key_states_unrotated = checkpoint_name(
                self._call_proj(self.key, encoder_hidden_states, deterministic, rngs), "key_proj")
            value_states = checkpoint_name(
                self._call_proj(self.value, encoder_hidden_states, deterministic, rngs), "value_proj")
        else:
            query_states_unrotated = checkpoint_name(
                self._call_proj(self.query, hidden_states, deterministic, rngs), "query_proj")
            key_states_unrotated = checkpoint_name(
                self._call_proj(self.key, hidden_states, deterministic, rngs), "key_proj")
            value_states = checkpoint_name(
                self._call_proj(self.value, hidden_states, deterministic, rngs), "value_proj")
            kv_len = q_len
            q_segment_ids = kv_segment_ids = segment_ids
            q_position_ids = kv_position_ids = position_ids

        # handle KV cache
        if past_key_value is not None:
            past_k, past_v = past_key_value
            # concatenate past and current K/V
            key_states_unrotated = jnp.concatenate([past_k, key_states_unrotated], axis=1)
            value_states = jnp.concatenate([past_v, value_states], axis=1)
            kv_len = key_states_unrotated.shape[1]

        # 2. prepare attention mask first (needed for determining mask format)
        if not self.is_cross_attention: # self-attention
            # MHA uses 4D mask
            # if we have past key values, create attention mask for full sequence
            if past_key_value is not None:
                # create full mask for cached + current kv sequence
                full_mask = jnp.ones((B, kv_len), dtype=attention_mask.dtype)
                padding_mask = full_mask[:, None, None, :]

                if use_causal:
                    # during generation, each query position can only attend to positions before it
                    query_indices = jnp.arange(q_len)[None, :, None]
                    kv_indices = jnp.arange(kv_len)[None, None, :]
                    causal_mask = query_indices >= kv_indices
                    causal_mask = causal_mask[:, None, :, :]
                    final_attention_mask = nnx.combine_masks(padding_mask, causal_mask)
                else:
                    final_attention_mask = padding_mask

            elif attention_mask.ndim == 4:
                # packed training: mask is already 4D (B, 1, seq, seq) with causal+segment
                final_attention_mask = attention_mask

            else:
                # non-packed training: 2D mask (B, seq_len) needs expansion
                padding_mask = attention_mask[:, None, None, :]

                if use_causal:
                    attention_mask_bool = attention_mask[:, :q_len].astype(jnp.bool_)
                    causal_mask = nnx.make_causal_mask(attention_mask_bool)
                    final_attention_mask = nnx.combine_masks(padding_mask, causal_mask)
                else:
                    final_attention_mask = padding_mask

        else: # cross-attention
            # standard: mask applies to encoder states (K/V)
            assert encoder_attention_mask is not None
            if encoder_attention_mask.ndim == 4:
                # packed training: already 4D
                final_attention_mask = encoder_attention_mask
            else:
                # MHA uses 4D mask (B, 1, 1, kv_len)
                final_attention_mask = encoder_attention_mask[:, None, None, :]

        # MHA/GQA: Q gets full num_heads; K/V get num_kv_heads (broadcasts in GQA path).
        query_per_head_unrotated = query_states_unrotated.reshape(B, q_len, self.num_heads, self.head_dim)
        key_per_head_unrotated = key_states_unrotated.reshape(B, kv_len, self.num_kv_heads, self.head_dim)

        # Resolve per-call window AND rope_theta from `override_window_size`. Both
        # depend on whether this layer is RUNNING as sliding (W > 0) or full
        # (W == 0). The pre-fix path baked rope_theta at __init__ from
        # position_specs[0]; inside a scan stack that silently collapsed all
        # layers to the same theta. Resolving here at call time -- driven by the
        # SAME signal that picks the attention window -- makes hybrid attention
        # actually honor `rope_theta != rope_theta_sliding`.
        effective_window_size = override_window_size if override_window_size is not None else self.window_size
        if self.rope_theta_sliding_static is not None:
            # match hidden_states.dtype so freqs_cis stays in model dtype; mirrors
            # the pre-fix path where theta was a Python weak-typed float that
            # promoted to hidden_states.dtype inside _precompute_freqs_cis.
            effective_rope_theta = jnp.where(
                jnp.asarray(effective_window_size) > 0,
                jnp.asarray(self.rope_theta_sliding_static, dtype=hidden_states.dtype),
                jnp.asarray(self.rope_theta_full, dtype=hidden_states.dtype),
            )
        else:
            effective_rope_theta = self.rope_theta

        # apply RoPE based on attention type
        if self.is_cross_attention:
            # NO RoPE for cross-attention (like T5)
            query_states_rotated_per_head = query_per_head_unrotated
            key_states_rotated_per_head = key_per_head_unrotated
        else:
            # self-attention: apply rotary to both Q and K (head-axis-agnostic; broadcasts over the head dim)
            if position_ids is not None:
                # use provided position IDs for packed sequences
                # precompute for max configured length to avoid traced value in jnp.arange
                freqs_cis = _precompute_freqs_cis(
                    dim=self.head_dim,
                    seq_len=self.max_positions,
                    theta=effective_rope_theta,
                    dtype=hidden_states.dtype
                )
                # apply RoPE with position IDs
                query_states_rotated_per_head = _apply_rotary_pos_emb_with_ids(
                    query_per_head_unrotated, freqs_cis, q_position_ids
                )
                key_states_rotated_per_head = _apply_rotary_pos_emb_with_ids(
                    key_per_head_unrotated, freqs_cis, kv_position_ids
                )
            else:
                # original sequential positions
                max_seq_len = max(q_len, kv_len)
                freqs_cis = _precompute_freqs_cis(
                    dim=self.head_dim,
                    seq_len=max_seq_len,
                    theta=effective_rope_theta,
                    dtype=hidden_states.dtype
                )
                freqs_cis_for_query = freqs_cis[kv_len - q_len:kv_len, ...]
                query_states_rotated_per_head = _apply_rotary_pos_emb(query_per_head_unrotated, freqs_cis_for_query)
                freqs_cis_for_kv = freqs_cis[:kv_len, ...]
                key_states_rotated_per_head = _apply_rotary_pos_emb(key_per_head_unrotated, freqs_cis_for_kv)

        # post-RoPE QK-norm (Gemma 3 / T5Gemma 2 order). RMSNorm over head_dim broadcasts across heads.
        if self.q_norm is not None:
            query_states_rotated_per_head = self.q_norm(query_states_rotated_per_head)
        if self.k_norm is not None:
            key_states_rotated_per_head = self.k_norm(key_states_rotated_per_head)

        # pre-scale Q only on the new path (preserves bit-identicalness on legacy
        # configs where dot_product_attention_weights does its own 1/sqrt(d) scale).
        if self._needs_inline_attn:
            q_final = query_states_rotated_per_head * jnp.asarray(
                self.q_multiplier, dtype=query_states_rotated_per_head.dtype
            )
        else:
            q_final = query_states_rotated_per_head
        k_final = key_states_rotated_per_head
        # value states reshaped to num_kv_heads to match K
        v_final = value_states.reshape(B, kv_len, self.num_kv_heads, self.head_dim)

        # 4. call attention (effective_window_size was resolved above alongside
        # effective_rope_theta).
        attn_output_heads = maybe_flash_attention(
            q_final, k_final, v_final,
            mask=final_attention_mask,
            q_segment_ids=q_segment_ids,
            kv_segment_ids=kv_segment_ids,
            window_size=effective_window_size,
            is_cross_attention=self.is_cross_attention,
            causal=use_causal,
            dropout_rng=rngs.dropout() if rngs is not None else None,
            dropout_rate=self.attn_pdrop if not deterministic else 0.0,
            deterministic=deterministic,
            dtype=q_final.dtype,
            num_kv_heads=self.num_kv_heads,
            q_pre_scaled=self._needs_inline_attn,
        )

        # (B, q_len, num_heads, head_dim) -> (B, q_len, q_proj_dim)
        context_layer = attn_output_heads.reshape(B, q_len, self.q_proj_dim)

        # SubLN: RMSNorm before attention output projection
        if self.attn_sub_norm is not None:
            context_layer = self.attn_sub_norm(context_layer)
        # pin [B, S, d_prime] to batch-sharded before c_proj for the same reason
        # as FFN: prevents XLA from interpreting FSDP weight sharding as TP.
        context_layer = maybe_constrain_activation(context_layer)
        projected_output = checkpoint_name(
            self._call_proj(self.c_proj, context_layer, deterministic, rngs), "out_proj")
        projected_output = maybe_constrain_activation(projected_output)

        # prepare present KV cache
        use_cache = init_cache | (past_key_value is not None)

        # use jnp.where to avoid traced boolean in if statement
        if not self.is_cross_attention:
            # for cache, we need the unrotated K/V states
            # extract only the current timestep's K/V (not including past)
            if past_key_value is not None:
                # after concatenation, key_states_unrotated and value_states contain both past and present
                # we need to extract only the NEW tokens (last q_len positions)
                present_k = key_states_unrotated[:, -q_len:, :]
                present_v = value_states[:, -q_len:, :]
            else:
                present_k = key_states_unrotated
                present_v = value_states

            # conditionally set cache based on use_cache (traced)
            # return zeros instead of None to maintain pytree structure
            present_kv = jax.lax.cond(
                use_cache,
                lambda: (present_k, present_v),
                lambda: (jnp.zeros_like(present_k), jnp.zeros_like(present_v))
            )
        else:
            present_kv = None

        return projected_output, present_kv


class FlaxHan2HanMLP(nnx.Module):
    def __init__(
        self,
        config: Han2HanConfig,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
        sharding: tuple = ('model',None),
        intermediate_size: int = None,
        activation_override: Optional[str] = None,
        param_dtype: Optional[jnp.dtype] = None,
    ):
        if param_dtype is None:
            param_dtype = dtype
        inner_dim = intermediate_size if intermediate_size else (config.d_ff if config.d_ff is not None else 4 * config.d_model)

        act_name = activation_override if activation_override is not None else config.ffn_activation
        if act_name == "gelu_new":
            act_name = "gelu"

        self.use_swiglu = (act_name == "swiglu")
        self.use_geglu = (act_name == "geglu")
        self.use_reglu2 = (act_name == "reglu2")
        self.use_relu2 = (act_name == "relu2")
        self.use_gated = self.use_swiglu or self.use_geglu or self.use_reglu2

        if self.use_gated:
            # T5-style gated FFN: two separate projections (wi_0, wi_1, wo naming)
            self.wi_0 = nnx.Linear(in_features=config.d_model, out_features=inner_dim, dtype=dtype,
                                   rngs=rngs, param_dtype=param_dtype, use_bias=config.use_bias,
                                   kernel_init=nnx.with_partitioning(
                                       config.make_kernel_init(dtype=dtype), sharding=sharding,),
                                   bias_init=nnx.with_partitioning(
                                       config.make_bias_init(), sharding=((sharding[0],) if sharding else None),))

            self.wi_1 = nnx.Linear(in_features=config.d_model, out_features=inner_dim, dtype=dtype,
                                   rngs=rngs, param_dtype=param_dtype, use_bias=config.use_bias,
                                   kernel_init=nnx.with_partitioning(
                                       config.make_kernel_init(dtype=dtype), sharding=sharding,),
                                   bias_init=nnx.with_partitioning(
                                       config.make_bias_init(), sharding=((sharding[0],) if sharding else None),))

            self.wo = nnx.Linear(in_features=inner_dim, out_features=config.d_model, dtype=dtype,
                                 rngs=rngs, param_dtype=param_dtype, use_bias=config.use_bias,
                                 kernel_init=nnx.with_partitioning(
                                     config.make_kernel_init(dtype=dtype), sharding=sharding,),
                                 bias_init=nnx.with_partitioning(
                                     config.make_bias_init(), sharding=((sharding[0],) if sharding else None),))
        else:
            # standard dense FFN (c_fc, c_proj naming for checkpoint compatibility)
            self.c_fc = nnx.Linear(in_features=config.d_model, out_features=inner_dim, dtype=dtype,
                                   rngs=rngs, param_dtype=param_dtype, use_bias=config.use_bias,
                                   kernel_init=nnx.with_partitioning(
                                       config.make_kernel_init(dtype=dtype), sharding=sharding,),
                                   bias_init=nnx.with_partitioning(
                                       config.make_bias_init(), sharding=((sharding[0],) if sharding else None),))

            self.c_proj = nnx.Linear(in_features=inner_dim, out_features=config.d_model, dtype=dtype,
                                     rngs=rngs, param_dtype=param_dtype, use_bias=config.use_bias,
                                     kernel_init=nnx.with_partitioning(
                                         config.make_kernel_init(dtype=dtype), sharding=sharding,),
                                     bias_init=nnx.with_partitioning(
                                         config.make_bias_init(), sharding=((sharding[0],) if sharding else None),))

            if not self.use_relu2:
                self.act_fn = getattr(nnx, act_name, nnx.gelu)

        # SubLN: RMSNorm between the activation/gate and wo/c_proj.
        self.sub_norm = nnx.data(None)
        if config.use_sub_ln:
            norm_sharding = (sharding[0],) if sharding else None
            self.sub_norm = RMSNorm(hidden_size=inner_dim, rngs=rngs, dtype=dtype,
                                    sharding=norm_sharding, use_bias=False,
                                    param_dtype=param_dtype, eps=config.layer_norm_epsilon)

        self.dropout = nnx.Dropout(rate=config.resid_pdrop, rngs=rngs)

    def __call__(self, hidden_states, rngs: nnx.Rngs = None, deterministic: bool = None):
        if self.use_gated:
            gate_pre = self.wi_0(hidden_states)
            if self.use_reglu2:
                hidden_gated = nnx.relu(gate_pre) ** 2
            elif self.use_geglu:
                hidden_gated = nnx.gelu(gate_pre)
            else:
                hidden_gated = nnx.silu(gate_pre)
            hidden_gated = checkpoint_name(hidden_gated, "mlpwi_0")
            hidden_linear = self.wi_1(hidden_states)
            hidden_linear = checkpoint_name(hidden_linear, "mlpwi_1")
            hidden_states = hidden_gated * hidden_linear
            if self.sub_norm is not None:
                hidden_states = self.sub_norm(hidden_states)
            # pin [B, S, d_ff] to batch-sharded; without this XLA may resolve the
            # FSDP `('data', None)` weight sharding as Megatron TP and all-gather
            # the full activation in the backward pass.
            hidden_states = maybe_constrain_activation(hidden_states)
            hidden_states = checkpoint_name(self.wo(hidden_states), "mlpwo")
        else:
            hidden_states = checkpoint_name(self.c_fc(hidden_states), "mlpwi")
            if self.use_relu2:
                hidden_states = nnx.relu(hidden_states) ** 2
            else:
                hidden_states = self.act_fn(hidden_states)
            if self.sub_norm is not None:
                hidden_states = self.sub_norm(hidden_states)
            hidden_states = maybe_constrain_activation(hidden_states)
            hidden_states = checkpoint_name(self.c_proj(hidden_states), "mlpwo")

        hidden_states = self.dropout(hidden_states, deterministic=deterministic, rngs=rngs)
        hidden_states = maybe_constrain_activation(hidden_states)
        return hidden_states


class FlaxHan2HanBlock(nnx.Module):
    def __init__(
        self,
        config: Han2HanConfig,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
        is_encoder: bool = False,
        layer_idx: int = 0,
        sharding: tuple = ('model',None),
        ffn_type: str = 'd',
        attention_type: Optional[str] = None,
        cross_attention_type: Optional[str] = None,
        param_dtype: Optional[jnp.dtype] = None,
        jamo_buckets: Optional[np.ndarray] = None,
        char_buckets: Optional[np.ndarray] = None,
    ):
        if param_dtype is None:
            param_dtype = dtype
        self.config = config
        self.layer_idx = layer_idx
        add_cross_attention = not is_encoder
        hidden_size = config.d_model
        norm_sharding = (sharding[0],) if sharding else None

        if is_encoder:
            self.ln_1 = create_encoder_norm(config, hidden_size, rngs, dtype, norm_sharding, param_dtype=param_dtype)
            self.ln_2 = create_encoder_norm(config, hidden_size, rngs, dtype, norm_sharding, param_dtype=param_dtype)
        else:
            self.ln_1 = create_decoder_norm(config, hidden_size, rngs, dtype, norm_sharding, param_dtype=param_dtype)
            self.ln_2 = create_decoder_norm(config, hidden_size, rngs, dtype, norm_sharding, param_dtype=param_dtype)

        if attention_type is None:
            raise ValueError(
                f"attention_type is None for layer {layer_idx}. "
                f"Specify encoder_attention_types/decoder_attention_types or set attention_mechanism."
            )

        self.attention_type = attention_type

        if add_cross_attention:
            if cross_attention_type is None:
                raise ValueError(
                    f"cross_attention_type is None for decoder layer {layer_idx}. "
                    f"Specify decoder_cross_attention_types or set attention_mechanism."
                )
            self.cross_attention_type = cross_attention_type
        else:
            self.cross_attention_type = None

        is_sliding_self = 'sliding' in attention_type or 'local' in attention_type
        effective_rope_theta = (
            config.rope_theta_sliding
            if is_sliding_self and config.rope_theta_sliding is not None
            else config.rope_theta
        )

        self.attn = FlaxHan2HanAttention(config=config, rngs=rngs, is_causal=not is_encoder,
                                         dtype=dtype, sharding=sharding,
                                         layer_idx=layer_idx, is_encoder=is_encoder,
                                         rope_theta=effective_rope_theta,
                                         param_dtype=param_dtype)

        # store effective window sizes for runtime override (scan compatibility)
        self.effective_window_size = 0 if 'sliding' not in attention_type else config.sliding_window_size
        self.effective_rope_theta = effective_rope_theta

        if add_cross_attention:
            self.crossattention = FlaxHan2HanAttention(config=config, rngs=rngs, dtype=dtype,
                                                       is_cross_attention=add_cross_attention,
                                                       is_causal=False, sharding=sharding,
                                                       layer_idx=layer_idx, is_encoder=False,
                                                       param_dtype=param_dtype)

            is_sliding_cross = (
                'sliding' in cross_attention_type or 'local' in cross_attention_type
            )
            self.effective_cross_window_size = (
                config.sliding_window_size if is_sliding_cross else 0
            )

            self.ln_cross_attn = create_decoder_norm(config, hidden_size, rngs, dtype, norm_sharding, param_dtype=param_dtype)

        self.ffn_type = ffn_type
        self.mlp = FlaxHan2HanMLP(
            config, rngs, dtype, sharding,
            activation_override=config.dense_ffn_activation,
            param_dtype=param_dtype,
        )

        self.block_type = "encoder_block" if is_encoder else "decoder_block"

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        segment_ids: Optional[jnp.ndarray] = None,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        encoder_segment_ids: Optional[jnp.ndarray] = None,
        encoder_position_ids: Optional[jnp.ndarray] = None,
        init_cache: bool = False,
        past_key_value: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
        generation_step: Optional[int] = None,
        rngs: nnx.Rngs = None,
        override_window_size: Optional[jnp.ndarray] = None,
    ) -> Union[Tuple[jnp.ndarray], Optional[Tuple[jnp.ndarray, Tuple[jnp.ndarray, ...]]]]:

        # resolve effective window size: override (from scan) > stored default
        effective_ws = override_window_size if override_window_size is not None else self.effective_window_size

        hidden_states = checkpoint_name(hidden_states, "decoder_layer_input")
        # pin block input to batch-sharded; this is the residual stream that
        # all subsequent in-block ops branch off of.
        hidden_states = maybe_constrain_activation(hidden_states)
        residual = hidden_states

        hidden_states = self.ln_1(hidden_states)

        attn_outputs = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
            position_ids=position_ids,
            init_cache=init_cache,
            past_key_value=past_key_value,
            rngs=rngs,
            override_window_size=effective_ws,
        )
        # MHA: (projected_output, present_kv)
        attn_output = attn_outputs[0]
        present_cache = attn_outputs[1]
        outputs = (present_cache,)

        # residual connection
        hidden_states = attn_output + residual
        hidden_states = maybe_constrain_activation(hidden_states)

        if encoder_hidden_states is not None and hasattr(self, "crossattention"):

            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)

            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                segment_ids=segment_ids,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                encoder_position_ids=encoder_position_ids,
                encoder_segment_ids=encoder_segment_ids,
                init_cache=init_cache,
                past_key_value=None,
                rngs=rngs,
                override_window_size=self.effective_cross_window_size,
            )
            attn_output = cross_attn_outputs[0]

            # residual connection
            hidden_states = residual + attn_output
            hidden_states = maybe_constrain_activation(hidden_states)

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)

        feed_forward_hidden_states = self.mlp(hidden_states, rngs=rngs)

        hidden_states = residual + feed_forward_hidden_states
        hidden_states = maybe_constrain_activation(hidden_states)
        outputs = (hidden_states,) + outputs

        return outputs  # hidden_states, present_kv


_SCAN_MAX_PERIOD = 4


def _identify_scan_groups(ffn_types, attn_types, cross_attn_types=None, max_period=_SCAN_MAX_PERIOD):
    """Find periodic blocks of layers that share a scanned layer body.

    At each starting position pick the (period, repeats) maximizing coverage
    (period * repeats), tie-broken toward smaller period for more
    compilation reuse. A period-1 group is the classic contiguous run of
    identical layers; a period-k repeats-r group covers k*r layers whose
    scan-keys match in k-step strides, scanned once via a repeat wrapper.

    MHA variants differ only in masking so they collapse to a single
    scan-key for grouping purposes.

    Returns list of (start, period, repeats, position_specs) where
    position_specs is a length-period list of (ffn_type, attn_type,
    cross_attn_type) tuples from the first occurrence of each sub-position.
    """
    def _scan_key(t):
        if t is None:
            return None
        return 'mha'

    n = len(ffn_types)
    keys = [
        (
            ffn_types[i],
            _scan_key(attn_types[i]),
            _scan_key(cross_attn_types[i]) if cross_attn_types else None,
        )
        for i in range(n)
    ]

    groups = []
    i = 0
    while i < n:
        best_k, best_r, best_cov = 1, 1, 1
        for k in range(1, max_period + 1):
            if i + k > n:
                break
            r = 1
            while i + k * (r + 1) <= n and all(
                keys[i + k * r + j] == keys[i + j] for j in range(k)
            ):
                r += 1
            if k > 1 and r < 2:
                continue
            cov = k * r
            if cov > best_cov or (cov == best_cov and k < best_k):
                best_k, best_r, best_cov = k, r, cov

        position_specs = [
            (
                ffn_types[i + j],
                attn_types[i + j],
                cross_attn_types[i + j] if cross_attn_types else None,
            )
            for j in range(best_k)
        ]
        groups.append((i, best_k, best_r, position_specs))
        i += best_k * best_r

    return groups


class FlaxHan2HanRepeatBlock(nnx.Module):
    """Wrapper holding k sequential FlaxHan2HanBlock sub-layers forming one
    period of a scanned heterogeneous layer stack.

    Under ``@nnx.vmap`` this wrapper is stacked along the repetition axis so
    a single ``jax.lax.scan`` pass applies ``period * repeats`` layers while
    compiling only the inner per-period body once. Each replica has
    identical tree structure (same ffn_type/attn_type per sub-position), so
    vmap stacks each sub-layer's parameters along axis 0 cleanly.
    """
    def __init__(
        self,
        config: Han2HanConfig,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
        is_encoder: bool = False,
        layer_idx_base: int = 0,
        sharding: tuple = ('model', None),
        position_specs: Optional[List[Tuple[str, Optional[str], Optional[str]]]] = None,
        param_dtype: Optional[jnp.dtype] = None,
        jamo_buckets: Optional[np.ndarray] = None,
        char_buckets: Optional[np.ndarray] = None,
    ):
        if position_specs is None or len(position_specs) == 0:
            raise ValueError("FlaxHan2HanRepeatBlock requires non-empty position_specs.")
        if param_dtype is None:
            param_dtype = dtype

        self.sublayers = nnx.List([
            FlaxHan2HanBlock(
                config=config, rngs=rngs, dtype=dtype, is_encoder=is_encoder,
                layer_idx=layer_idx_base + j, sharding=sharding,
                ffn_type=ffn_t,
                attention_type=attn_t,
                cross_attention_type=xattn_t,
                param_dtype=param_dtype,
                jamo_buckets=jamo_buckets,
                char_buckets=char_buckets,
            )
            for j, (ffn_t, attn_t, xattn_t) in enumerate(position_specs)
        ])

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        segment_ids: Optional[jnp.ndarray] = None,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        encoder_segment_ids: Optional[jnp.ndarray] = None,
        encoder_position_ids: Optional[jnp.ndarray] = None,
        init_cache: bool = False,
        past_key_value: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
        generation_step: Optional[int] = None,
        rngs: nnx.Rngs = None,
        sublayer_window_sizes: Optional[jnp.ndarray] = None,
    ):
        for j, sub in enumerate(self.sublayers):
            ws = sublayer_window_sizes[j] if sublayer_window_sizes is not None else None
            out = sub(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                segment_ids=segment_ids,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                encoder_segment_ids=encoder_segment_ids,
                encoder_position_ids=encoder_position_ids,
                init_cache=init_cache,
                past_key_value=past_key_value,
                generation_step=generation_step,
                rngs=rngs,
                override_window_size=ws,
            )
            hidden_states = out[0]

        return hidden_states


@functools.lru_cache(maxsize=None)
def _layer_remat_static_argnums(cls) -> tuple:
    """Positional indices of Python-bool args in ``cls.__call__`` that drive
    in-module branches and must therefore be marked ``static_argnums`` for
    ``nnx.remat``. ``nnx.remat`` resolves kwargs to positions via
    ``inspect.signature(...).bind`` against the unbound ``__call__`` (``self``
    excluded), so indices are 0-based over the user-visible parameters.

    Raises if the expected arg disappears or is renamed so that signature
    drift surfaces immediately instead of as a TracerBoolConversionError deep
    inside the layer body.
    """
    bool_names = ("init_cache",)
    params = list(inspect.signature(cls.__call__).parameters)[1:]
    missing = [n for n in bool_names if n not in params]
    if missing:
        raise ValueError(
            f"{cls.__name__}.__call__ missing expected bool args {missing}; "
            f"update _layer_remat_static_argnums or the layer signature."
        )
    return tuple(params.index(n) for n in bool_names)


class FlaxHan2HanBlockCollection(nnx.Module):
    def __init__(
        self,
        config: Han2HanConfig,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
        gradient_checkpointing: bool = True,
        is_encoder: bool = False,
        sharding: tuple = ('model',None),
        layer_ffn_types: Optional[List[str]] = None,
        layer_attention_types: Optional[List[str]] = None,
        layer_cross_attention_types: Optional[List[str]] = None,
        param_dtype: Optional[jnp.dtype] = None,
        jamo_buckets: Optional[np.ndarray] = None,
        char_buckets: Optional[np.ndarray] = None,
    ):
        if param_dtype is None:
            param_dtype = dtype
        n_layers = config.encoder_nlayer if is_encoder else config.decoder_nlayer

        if layer_ffn_types is None:
            layer_ffn_types = ['d'] * n_layers
        if layer_attention_types is None:
            raise ValueError(
                "layer_attention_types is required. Use config.get_encoder_attention_types() "
                "or config.get_decoder_attention_types()."
            )

        self.layer_attention_types = layer_attention_types
        self.layer_cross_attention_types = layer_cross_attention_types

        self.layerdrop = config.layer_pdrop
        self.is_encoder = is_encoder
        remat_policy_name = config.remat_policy
        if remat_policy_name == "none":
            self.gradient_checkpointing = False
            self.remat_policy = None
        else:
            self.gradient_checkpointing = gradient_checkpointing
            self.remat_policy = get_remat_policy(remat_policy_name)

        # scan layer setup: group layers into periodic blocks for jax.lax.scan
        self._scan_groups = None
        if config.use_scan_layers:
            self._scan_groups = _identify_scan_groups(
                layer_ffn_types, layer_attention_types,
                cross_attn_types=layer_cross_attention_types if not is_encoder else None,
            )

            # bookend groups (period*repeats == 1) get individual FlaxHan2HanBlock
            # instances; multi-layer groups get vmapped stacks. Period-1 stacks
            # hold FlaxHan2HanBlock directly; period-k stacks hold
            # FlaxHan2HanRepeatBlock wrappers containing k sub-layers each.
            self.layers = nnx.List()
            for (start, period, repeats, _) in self._scan_groups:
                if period * repeats <= 1:
                    self.layers.append(FlaxHan2HanBlock(
                        config, rngs, dtype, is_encoder,
                        layer_idx=start, sharding=sharding,
                        ffn_type=layer_ffn_types[start],
                        attention_type=layer_attention_types[start],
                        cross_attention_type=layer_cross_attention_types[start] if not is_encoder else None,
                        param_dtype=param_dtype,
                        jamo_buckets=jamo_buckets,
                        char_buckets=char_buckets,
                    ))

            self._scan_stacks = nnx.List()
            _scan_periods_list = []
            _window_sizes_list = []

            def _ws_for_layer(attn_t):
                return 0 if attn_t == 'mha' else config.sliding_window_size

            for (start, period, repeats, position_specs) in self._scan_groups:
                if period * repeats <= 1:
                    continue

                _param_dtype = param_dtype
                _is_encoder = is_encoder

                if period == 1:
                    # single-block scan: vmap creates stack of FlaxHan2HanBlock
                    spec = position_specs[0]
                    _ffn, _attn, _xattn = spec
                    _jbu = jamo_buckets
                    _cbu = char_buckets

                    @nnx.vmap(transform_metadata={nnx.PARTITION_NAME: None})
                    def create_layer(layer_rngs):
                        return FlaxHan2HanBlock(
                            config, layer_rngs, dtype, _is_encoder,
                            layer_idx=0, sharding=sharding,
                            ffn_type=_ffn, attention_type=_attn,
                            cross_attention_type=_xattn,
                            param_dtype=_param_dtype,
                            jamo_buckets=_jbu,
                            char_buckets=_cbu,
                        )

                    stack = create_layer(rngs.fork(split=repeats))
                    self._scan_stacks.append(stack)

                    ws = tuple(
                        _ws_for_layer(layer_attention_types[start + j])
                        for j in range(repeats)
                    )
                else:
                    # period-k scan: vmap creates stack of FlaxHan2HanRepeatBlock
                    _specs = list(position_specs)
                    _jbu = jamo_buckets
                    _cbu = char_buckets

                    @nnx.vmap(transform_metadata={nnx.PARTITION_NAME: None})
                    def create_layer(layer_rngs):
                        return FlaxHan2HanRepeatBlock(
                            config=config, rngs=layer_rngs, dtype=dtype,
                            is_encoder=_is_encoder, layer_idx_base=0,
                            sharding=sharding, position_specs=_specs,
                            param_dtype=_param_dtype,
                            jamo_buckets=_jbu,
                            char_buckets=_cbu,
                        )

                    stack = create_layer(rngs.fork(split=repeats))
                    self._scan_stacks.append(stack)

                    # (repeats, period) shaped tuples-of-tuples for per-step row
                    ws = tuple(
                        tuple(
                            _ws_for_layer(layer_attention_types[start + i * period + j])
                            for j in range(period)
                        )
                        for i in range(repeats)
                    )

                _scan_periods_list.append(period)
                _window_sizes_list.append(ws)

            self._scan_periods = nnx.static(_scan_periods_list)
            self._scan_window_sizes = nnx.static(_window_sizes_list)
        else:
            self.layers = nnx.List([
                FlaxHan2HanBlock(
                    config, rngs, dtype, is_encoder,
                    layer_idx=i, sharding=sharding,
                    ffn_type=layer_ffn_types[i],
                    attention_type=layer_attention_types[i],
                    cross_attention_type=layer_cross_attention_types[i] if not is_encoder else None,
                    param_dtype=param_dtype,
                    jamo_buckets=jamo_buckets,
                    char_buckets=char_buckets,
                ) for i in range(n_layers)])

    def __call__(
        self,
        hidden_states,
        attention_mask,
        position_ids: Optional[jnp.ndarray] = None,
        segment_ids: Optional[jnp.ndarray] = None,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        encoder_position_ids: Optional[jnp.ndarray] = None,
        encoder_segment_ids: Optional[jnp.ndarray] = None,
        init_cache: bool = False,
        past_key_values: Optional[Tuple[Optional[Tuple[Optional[Tuple[jnp.ndarray, jnp.ndarray]], ...]], ...]] = None,
        generation_step: Optional[int] = None,
        output_hidden_states: bool = False,
        deterministic: bool = True,
        rngs: nnx.Rngs = None,
    ):
        init_cache = (init_cache and not self.is_encoder)

        if self._scan_groups is not None:
            if init_cache or past_key_values is not None:
                return self._forward_generation(
                    hidden_states, attention_mask, position_ids, segment_ids,
                    encoder_hidden_states, encoder_attention_mask,
                    encoder_position_ids, encoder_segment_ids,
                    init_cache, past_key_values,
                    generation_step, output_hidden_states,
                    deterministic, rngs,
                )
            return self._forward_scanned(
                hidden_states, attention_mask, position_ids, segment_ids,
                encoder_hidden_states, encoder_attention_mask,
                encoder_position_ids, encoder_segment_ids,
                deterministic, rngs,
            )

        return self._forward_unrolled(
            hidden_states, attention_mask, position_ids, segment_ids,
            encoder_hidden_states, encoder_attention_mask,
            encoder_position_ids, encoder_segment_ids,
            init_cache, past_key_values,
            generation_step, output_hidden_states,
            deterministic, rngs,
        )

    def _forward_unrolled(
        self,
        hidden_states,
        attention_mask,
        position_ids, segment_ids,
        encoder_hidden_states, encoder_attention_mask,
        encoder_position_ids, encoder_segment_ids,
        init_cache, past_key_values,
        generation_step, output_hidden_states,
        deterministic, rngs,
    ):
        all_hidden_states = () if output_hidden_states else None
        present_key_values = () if (init_cache or past_key_values is not None) else None

        if self.gradient_checkpointing and not deterministic and init_cache:
            init_cache_for_layer = False
        else:
            init_cache_for_layer = init_cache

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            skip_layer = False
            if not deterministic and self.layerdrop > 0 and rngs is not None:
                dropout_probability = jax.random.uniform(rngs.layerdrop())
                skip_layer = dropout_probability < self.layerdrop

            layer_past = past_key_values[i] if past_key_values is not None else None

            layer_module = layer
            if self.gradient_checkpointing:
                layer_module = remat(
                    layer,
                    static_argnums=_layer_remat_static_argnums(type(layer)),
                    prevent_cse=not deterministic,
                    policy=self.remat_policy,
                )

            hidden_states_before = hidden_states

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                segment_ids=segment_ids,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                encoder_position_ids=encoder_position_ids,
                encoder_segment_ids=encoder_segment_ids,
                init_cache=init_cache_for_layer,
                past_key_value=layer_past,
                generation_step=generation_step,
                rngs=rngs,
            )

            hidden_states = jnp.where(skip_layer, hidden_states_before, layer_outputs[0])

            if init_cache_for_layer or layer_past is not None:
                present_key_values += (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        final_outputs = (
            hidden_states,
            all_hidden_states,
            present_key_values,
        )

        return tuple(v for v in final_outputs if v is not None)

    def _forward_generation(
        self,
        hidden_states,
        attention_mask,
        position_ids, segment_ids,
        encoder_hidden_states, encoder_attention_mask,
        encoder_position_ids, encoder_segment_ids,
        init_cache, past_key_values,
        generation_step, output_hidden_states,
        deterministic, rngs,
    ):
        """Generation-compatible forward for scan-configured models.

        Unrolls scan stacks into Python loops so each layer can receive
        its own KV cache slice. Used automatically when generation args
        are passed with use_scan_layers=True.
        """
        all_hidden_states = () if output_hidden_states else None
        present_key_values = () if (init_cache or past_key_values is not None) else None

        layer_idx = 0
        stack_idx = 0
        bookend_idx = 0

        for (start, period, repeats, _) in self._scan_groups:
            total = period * repeats
            if total <= 1:
                layer = self.layers[bookend_idx]
                bookend_idx += 1
                layers_to_run = [(layer, None)]
            else:
                stack = self._scan_stacks[stack_idx]
                window_sizes = self._scan_window_sizes[stack_idx]
                stack_idx += 1
                graphdef, state = nnx.split(stack)
                layers_to_run = []
                for i in range(repeats):
                    wrapper_i_state = jax.tree.map(lambda x, _i=i: x[_i], state)
                    wrapper_i = nnx.merge(graphdef, wrapper_i_state)
                    if period == 1:
                        ws = int(window_sizes[i])
                        layers_to_run.append((wrapper_i, ws))
                    else:
                        # period-k repeat block: iterate its sublayers
                        for j, sub in enumerate(wrapper_i.sublayers):
                            ws = int(window_sizes[i][j])
                            layers_to_run.append((sub, ws))

            for layer, override_ws in layers_to_run:
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                layer_past = past_key_values[layer_idx] if past_key_values is not None else None

                extra_kwargs = {}
                if override_ws is not None:
                    extra_kwargs['override_window_size'] = override_ws

                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    segment_ids=segment_ids,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    encoder_position_ids=encoder_position_ids,
                    encoder_segment_ids=encoder_segment_ids,
                    init_cache=init_cache,
                    past_key_value=layer_past,
                    generation_step=generation_step,
                    rngs=rngs,
                    **extra_kwargs,
                )

                hidden_states = layer_outputs[0]

                if init_cache or layer_past is not None:
                    present_key_values += (layer_outputs[1],)

                layer_idx += 1

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        final_outputs = (
            hidden_states,
            all_hidden_states,
            present_key_values,
        )

        return tuple(v for v in final_outputs if v is not None)

    def _forward_scanned(
        self,
        hidden_states,
        attention_mask,
        position_ids, segment_ids,
        encoder_hidden_states, encoder_attention_mask,
        encoder_position_ids, encoder_segment_ids,
        deterministic, rngs,
    ):
        """Forward pass using jax.lax.scan over homogeneous layer groups.

        Compiles one layer body per group and repeats it N times, reducing
        XLA compilation from O(num_layers) to O(num_unique_groups).
        """
        stack_idx = 0
        bookend_idx = 0
        for (start, period, repeats, _) in self._scan_groups:
            total = period * repeats
            if total <= 1:
                layer = self.layers[bookend_idx]
                bookend_idx += 1
                if self.gradient_checkpointing:
                    layer = remat(
                        layer,
                        static_argnums=_layer_remat_static_argnums(type(layer)),
                        prevent_cse=not deterministic,
                        policy=self.remat_policy,
                    )

                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    segment_ids=segment_ids,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    encoder_position_ids=encoder_position_ids,
                    encoder_segment_ids=encoder_segment_ids,
                    init_cache=False,
                    past_key_value=None,
                    generation_step=None,
                    rngs=rngs,
                )
                hidden_states = layer_outputs[0]

                continue

            stack = self._scan_stacks[stack_idx]
            window_sizes = jnp.array(self._scan_window_sizes[stack_idx])
            layer_rngs = rngs.fork(split=repeats) if rngs is not None else None
            stack_idx += 1

            if period == 1:
                def scan_body(carry, xs):
                    layer_module, ws, layer_rng = xs

                    if self.gradient_checkpointing:
                        layer_fn = remat(
                            layer_module,
                            static_argnums=_layer_remat_static_argnums(type(layer_module)),
                            prevent_cse=not deterministic,
                            policy=self.remat_policy,
                        )
                    else:
                        layer_fn = layer_module

                    outputs = layer_fn(
                        carry,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        segment_ids=segment_ids,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_attention_mask,
                        encoder_position_ids=encoder_position_ids,
                        encoder_segment_ids=encoder_segment_ids,
                        init_cache=False,
                        past_key_value=None,
                        generation_step=None,
                        rngs=layer_rng,
                        override_window_size=ws,
                    )

                    new_h = outputs[0]
                    return new_h, None

                scan_xs = (stack, window_sizes, layer_rngs)
                hidden_states, _ = jax.lax.scan(
                    scan_body,
                    hidden_states,
                    scan_xs,
                )
            else:
                # period-k scan: wrapper runs k sub-layers sequentially per step.
                # window_sizes are (repeats, period)-shaped; scan feeds one
                # length-period row per step to the wrapper.
                def scan_body(carry, xs):
                    wrapper_module, ws_row, layer_rng = xs

                    if self.gradient_checkpointing:
                        layer_fn = remat(
                            wrapper_module,
                            static_argnums=_layer_remat_static_argnums(type(wrapper_module)),
                            prevent_cse=not deterministic,
                            policy=self.remat_policy,
                        )
                    else:
                        layer_fn = wrapper_module

                    new_h = layer_fn(
                        carry,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        segment_ids=segment_ids,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_attention_mask,
                        encoder_position_ids=encoder_position_ids,
                        encoder_segment_ids=encoder_segment_ids,
                        init_cache=False,
                        past_key_value=None,
                        generation_step=None,
                        rngs=layer_rng,
                        sublayer_window_sizes=ws_row,
                    )
                    return new_h, None

                scan_xs = (stack, window_sizes, layer_rngs)
                hidden_states, _ = jax.lax.scan(
                    scan_body,
                    hidden_states,
                    scan_xs,
                )

        return (hidden_states,)


class FlaxHan2HanModule(nnx.Module):
    def __init__(
        self,
        config: Han2HanConfig,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
        gradient_checkpointing: bool = True,
        is_encoder: bool = False,
        sharding: tuple = ('model',None),
        layer_ffn_types: Optional[List[str]] = None,
        layer_attention_types: Optional[List[str]] = None,
        layer_cross_attention_types: Optional[List[str]] = None,
        param_dtype: Optional[jnp.dtype] = None,
        jamo_buckets: Optional[np.ndarray] = None,
        char_buckets: Optional[np.ndarray] = None,
    ):
        self.config = config
        self.is_encoder = is_encoder
        self.layer_ffn_types = layer_ffn_types
        self.layer_attention_types = layer_attention_types
        self.layer_cross_attention_types = layer_cross_attention_types
        if param_dtype is None:
            param_dtype = dtype

        self.wte = nnx.Embed(
            num_embeddings=config.vocab_size,
            features=config.d_model,
            dtype=dtype,
            param_dtype=param_dtype,
            embedding_init=nnx.with_partitioning(
                config.make_kernel_init(), sharding),
            rngs=rngs
        )

        # === SUBWORD EMBEDDING INITIALIZATION ===
        self.wje = nnx.data(None)
        self.wce = nnx.data(None)
        if config.jamo_subwords or config.char_subwords:
            self.subword_proj = nnx.Linear(
                in_features=config.subword_embed_dim,
                out_features=config.d_model,
                param_dtype=param_dtype,
                use_bias=False,
                dtype=dtype,
                kernel_init=nnx.with_partitioning(
                    config.make_kernel_init(),
                    sharding
                ),
                rngs=rngs
            )
            emb_norm_sharding = (sharding[0],) if sharding else None
            if is_encoder:
                self.ln_emb = create_encoder_norm(config, config.d_model, rngs, dtype, emb_norm_sharding, param_dtype=param_dtype)
            else:
                self.ln_emb = create_decoder_norm(config, config.d_model, rngs, dtype, emb_norm_sharding, param_dtype=param_dtype)

        if config.jamo_subwords:
            # use nnx.data() for Flax 0.12 to explicitly mark as data (not static)
            self.wje = nnx.data(nnx.Embed(
                num_embeddings=config.jamo_vocab_size,
                features=config.subword_embed_dim,
                dtype=dtype,
                param_dtype=param_dtype,
                embedding_init=nnx.with_partitioning(
                    config.make_kernel_init(), sharding,),
                rngs=rngs
            ))

        if config.char_subwords:
            # use nnx.data() for Flax 0.12 to explicitly mark as data (not static)
            self.wce = nnx.data(nnx.Embed(
                num_embeddings=config.char_vocab_size,
                features=config.subword_embed_dim,
                dtype=dtype,
                param_dtype=param_dtype,
                embedding_init=nnx.with_partitioning(
                    config.make_kernel_init(), sharding,),
                rngs=rngs
            ))
        # === END SUBWORD EMBEDDING INITIALIZATION ===

        self.drop = nnx.Dropout(config.embd_pdrop, rngs=rngs)
        self.h = FlaxHan2HanBlockCollection(
            config, rngs, dtype, gradient_checkpointing, is_encoder, sharding,
            layer_ffn_types=layer_ffn_types,
            layer_attention_types=layer_attention_types,
            layer_cross_attention_types=layer_cross_attention_types,
            param_dtype=param_dtype,
            jamo_buckets=jamo_buckets,
            char_buckets=char_buckets,
        )
        norm_sharding = (sharding[0],) if sharding else None
        if is_encoder:
            self.ln_f = create_encoder_norm(config, config.d_model, rngs, dtype, norm_sharding, param_dtype=param_dtype)
        else:
            self.ln_f = create_decoder_norm(config, config.d_model, rngs, dtype, norm_sharding, param_dtype=param_dtype)

        self.pad_token_id = config.pad_token_id
        self.n_layers = config.encoder_nlayer if is_encoder else config.decoder_nlayer

    def __call__(
        self,
        input_ids: Optional[jnp.ndarray] = None,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        segment_ids: Optional[jnp.ndarray] = None,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        encoder_position_ids: Optional[jnp.ndarray] = None,
        encoder_segment_ids: Optional[jnp.ndarray] = None,
        input_embeddings: Optional[jnp.ndarray] = None,
        output_hidden_states: Optional[bool] = None,
        init_cache: Optional[bool] = None,
        past_key_values: Optional[dict] = None,
        generation_step: Optional[int] = None,
        return_embeddings_only: bool = False,
        return_dict: Optional[bool] = None,
        deterministic: Optional[bool] = True,
        rngs: nnx.Rngs = None
    ):
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False

        if init_cache is None:
            deterministic = self.drop.deterministic
            init_cache = deterministic and (not self.is_encoder)
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if past_key_values is None and not self.config.use_scan_layers:
            past_key_values = tuple([None] * self.n_layers)

        # if input_embeddings are provided, use them directly
        if input_embeddings is not None:
            hidden_states = input_embeddings
            # we need attention_mask but can't infer from embeddings, so it must be provided
            if attention_mask is None:
                raise ValueError("attention_mask must be provided when using input_embeddings")
        else:
            # standard path with input_ids
            if input_ids is None:
                raise ValueError("Either input_ids or input_embeddings must be provided")

            if attention_mask is None:
                attention_mask = jnp.where(input_ids == self.pad_token_id, 0, 1).astype("i4")

            jamo_input_ids, char_input_ids = None, None
            if self.config.jamo_subwords and self.subword_lookups is not None:
                if "jbu" in self.subword_lookups:
                    jamo_input_ids = jax.lax.stop_gradient(
                        jnp.take(self.subword_lookups["jbu"], input_ids.astype("i4"), axis=0)
                    )
            if self.config.char_subwords and self.subword_lookups is not None:
                if "cbu" in self.subword_lookups:
                    char_input_ids = jax.lax.stop_gradient(
                        jnp.take(self.subword_lookups["cbu"], input_ids.astype("i4"), axis=0)
                    )
            wte_embeds = self.wte(input_ids.astype("i4"))
            # scale embeddings by sqrt(d_model) for decoder only
            # this helps stabilize attention scores during generation
            if not self.is_encoder:
                normalizer = jnp.sqrt(self.config.d_model).astype(wte_embeds.dtype)
                wte_embeds = wte_embeds * normalizer

            # === CONDITIONAL SUBWORD EMBEDDINGS ===
            if hasattr(self, 'subword_proj'):
                subword_features = jnp.zeros((*wte_embeds.shape[:2],
                                              self.config.subword_embed_dim),
                                             dtype=wte_embeds.dtype)

                # embedding dropout: during training, randomly drop embedding modules
                # only apply during training (deterministic=False)
                if not deterministic and self.config.embedding_dropout_rate > 0:
                    # sample dropout masks for each embedding type
                    if rngs is not None and hasattr(rngs, 'dropout'):
                        dropout_key = rngs.dropout()
                        # split key for different embeddings
                        key1, key2, key3 = jax.random.split(dropout_key, 3)
                        # sample dropout masks (1 = keep, 0 = drop)
                        if self.config.char_is_unified_cjk:
                            wte_wje_pdrop = 0.0
                        else:
                            wte_wje_pdrop = self.config.embedding_dropout_rate
                        wte_keep = jax.random.bernoulli(key1, 1.0 - wte_wje_pdrop)
                        wje_keep = jax.random.bernoulli(key2, 1.0 - wte_wje_pdrop)
                        wce_keep = jax.random.bernoulli(key3, 1.0 - self.config.embedding_dropout_rate)

                        # ensure at least one embedding type is kept (prevent total dropout)
                        total_keep = wte_keep + wje_keep + wce_keep
                        all_dropped = total_keep == 0
                        # randomly keep one if all were dropped (jax-friendly)
                        keep_idx = jax.random.choice(key1, 3)
                        wte_keep = jnp.where(all_dropped & (keep_idx == 0), 1.0, wte_keep)
                        wje_keep = jnp.where(all_dropped & (keep_idx == 1), 1.0, wje_keep)
                        wce_keep = jnp.where(all_dropped & (keep_idx == 2), 1.0, wce_keep)
                    else:
                        # fallback if no rng available
                        wte_keep = wje_keep = wce_keep = 1.0
                else:
                    # no dropout during inference
                    wte_keep = wje_keep = wce_keep = 1.0

                # apply dropout to token embeddings and track active embeddings
                hidden_states = wte_embeds * wte_keep
                num_active = wte_keep

                if self.wje is not None and jamo_input_ids is not None:
                    jamo_embeds = self.wje(jamo_input_ids.astype("i4"))
                    jamo_embeds = jnp.sum(jamo_embeds, axis=-2)  # pooling over ngrams
                    # apply dropout mask and add to features
                    subword_features += jamo_embeds * wje_keep
                    num_active += wje_keep

                if self.wce is not None and char_input_ids is not None:
                    char_embeds = self.wce(char_input_ids.astype("i4"))
                    char_embeds = jnp.sum(char_embeds, axis=-2)
                    # apply dropout mask and add to features
                    subword_features += char_embeds * wce_keep
                    num_active += wce_keep

                # scale embeddings to compensate for dropout
                # avoid division by zero
                scale_factor = jnp.where(num_active > 0, 2.0 / jnp.maximum(num_active, 1.0), 1.0)
                subword_features = subword_features * scale_factor

                self_activated = nnx.silu(subword_features) * subword_features # element-wise self-gating
                projected_subwords = self.subword_proj(self_activated)
                # gated fusion: tokens + projected, self-gated subwords
                hidden_states = wte_embeds + projected_subwords
                hidden_states = self.ln_emb(hidden_states)

            else:
                hidden_states = wte_embeds

            # === END CONDITIONAL SUBWORD EMBEDDINGS ===

        if return_embeddings_only:
            return hidden_states

        hidden_states = self.drop(hidden_states, rngs=rngs)

        outputs = self.h(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            segment_ids=segment_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            encoder_segment_ids=encoder_segment_ids,
            encoder_position_ids=encoder_position_ids,
            init_cache=init_cache,
            past_key_values=past_key_values,
            generation_step=generation_step,
            output_hidden_states=output_hidden_states,
            deterministic=deterministic,
            rngs=rngs
        )

        hidden_states = outputs[0]

        hidden_states = self.ln_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = outputs[1] + (hidden_states,)
            outputs = (hidden_states, all_hidden_states) + outputs[2:]
        else:
            outputs = (hidden_states,) + outputs[1:]

        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        # parse outputs tuple - structure depends on what's enabled:
        # (hidden_states, [all_hidden_states,] [present_key_values,])
        idx = 1
        all_hidden_states_out = None
        if output_hidden_states:
            all_hidden_states_out = outputs[idx]
            idx += 1

        present_key_values = None
        if init_cache or past_key_values is not None:
            if idx < len(outputs) and outputs[idx] is not None:
                present_key_values = outputs[idx]
                idx += 1

        return FlaxBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_values,
            hidden_states=all_hidden_states_out,
        )

    def get_embedding_weights(self) -> Dict[str, float]:
        """Get the current embedding fusion weights for analysis."""
        weights_dict = {}

        if hasattr(self, 'subword_proj'):
            # For highway gating, we can report the average gate activation
            # This would require a forward pass, so we'll just report gate parameters
            weights_dict['subword_proj_bias_mean'] = float(jnp.mean(self.subword_proj.bias.value))
            weights_dict['subword_proj_kernel_std'] = float(jnp.std(self.subword_proj.kernel.value))

            # report which embeddings are active
            if self.wje is not None:
                weights_dict['jamo_embedding_active'] = True
            if self.wce is not None:
                weights_dict['char_embedding_active'] = True

        return weights_dict


class TiedLinear(nnx.Module):
    """Linear layer that shares weights with an embedding layer."""
    def __init__(self, embedding_module: nnx.Embed):
        self.embedding_ref = embedding_module

    def __call__(self, inputs):
        return inputs @ self.embedding_ref.embedding.value.T


class FlaxHan2Han(nnx.Module):
    def __init__(
        self,
        config: Han2HanConfig,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: Optional[jnp.dtype] = None,
        sharding: tuple = ('data',None),
        gradient_checkpointing: bool = False,
        jamo_buckets: np.ndarray = None,
        char_buckets: np.ndarray = None,
    ):
        self.config = config
        self.sharding = sharding
        # latent-weight storage dtype; defaults to compute dtype for backward compat.
        # split (param_dtype != dtype) supports f32 weight storage while the
        # forward pass runs in bf16.
        if param_dtype is None:
            param_dtype = dtype

        encoder_ffn_types = decoder_ffn_types = None

        encoder_attention_types = config.get_encoder_attention_types()
        decoder_attention_types = config.get_decoder_attention_types()
        decoder_cross_attention_types = config.get_decoder_cross_attention_types()

        if config.encoder_attention_types is not None or config.decoder_attention_types is not None:
            log_from_main_process(logger, 'info',
                f"Per-layer attention: encoder={encoder_attention_types[:3]}..., "
                f"decoder self={decoder_attention_types[:3]}..., "
                f"decoder cross={decoder_cross_attention_types[:3]}...")

        self.encoder = FlaxHan2HanModule(
            config, rngs, dtype, gradient_checkpointing, is_encoder=True,
            sharding=sharding, layer_ffn_types=encoder_ffn_types,
            layer_attention_types=encoder_attention_types,
            param_dtype=param_dtype,
            jamo_buckets=jamo_buckets,
            char_buckets=char_buckets,
        )
        self.decoder = FlaxHan2HanModule(
            config, rngs, dtype, gradient_checkpointing, is_encoder=False,
            sharding=sharding, layer_ffn_types=decoder_ffn_types,
            layer_attention_types=decoder_attention_types,
            layer_cross_attention_types=decoder_cross_attention_types,
            param_dtype=param_dtype,
            jamo_buckets=jamo_buckets,
            char_buckets=char_buckets,
        )

        if config.tie_input_output_embeddings:
            self.lm_head = TiedLinear(
                embedding_module=self.decoder.wte
            )
        else:
            # for untied lm_head, we need flipped sharding: (hidden_size, vocab_size)
            # should be PartitionSpec(None, 'model') to shard vocab dimension
            # handle both 1D and 2D sharding tuples
            if sharding is None:
                lm_head_sharding = None
            elif len(sharding) == 1:
                # 1d sharding (e.g., data_parallel mode): just use (None, sharding[0])
                lm_head_sharding = (None, sharding[0])
            else:
                # 2d sharding: flip the tuple
                lm_head_sharding = (sharding[1], sharding[0])

            self.lm_head = nnx.Linear(
                in_features=config.d_model,
                out_features=config.vocab_size,
                dtype=dtype,
                param_dtype=param_dtype,
                kernel_init=nnx.with_partitioning(
                    config.make_kernel_init(),
                    sharding=lm_head_sharding,
                ),
                use_bias=False,
                rngs=rngs,
            )

        self.tie_weights()

        self.pad_token_id = config.pad_token_id
        self.decoder_start_token_id = config.decoder_start_token_id
        self.eos_token_id = config.eos_token_id if hasattr(config, 'eos_token_id') else config.decoder_start_token_id
        self.bos_token_id = config.bos_token_id if hasattr(config, 'bos_token_id') else config.decoder_start_token_id

        # load subword tables if they exist
        self.encoder.subword_lookups = nnx.data(None)
        self.decoder.subword_lookups = nnx.data(None)
        if jamo_buckets is not None or char_buckets is not None:
            self.set_subword_tables(jamo_buckets, char_buckets)
        else:
            self._load_subword_tables()

    def tie_weights(self):
        """Establish weight tying between encoder and decoder.

        Call this after checkpoint restoration to ensure ties are preserved,
        since nnx.update() may break Python object references.
        """
        config = self.config

        # fix TiedLinear reference - nnx.update() may have replaced decoder.wte
        if config.tie_input_output_embeddings and isinstance(self.lm_head, TiedLinear):
            self.lm_head.embedding_ref = self.decoder.wte

        if config.tie_word_embeddings:
            self.encoder.wte.embedding = self.decoder.wte.embedding

        if config.tie_subtoken_embeddings:
            if config.jamo_subwords:
                self.encoder.wje.embedding = self.decoder.wje.embedding
            if config.char_subwords:
                self.encoder.wce.embedding = self.decoder.wce.embedding

        if config.tie_encoder_decoder:
            self._tie_encoder_decoder_blocks(config)

    def _tie_encoder_decoder_blocks(self, config):
        """Tie attention + dense-MLP kernels across encoder/decoder blocks.

        Walks both ``self.layers`` (unrolled bookends) and
        ``self._scan_stacks`` (vmap-stacked groups) in parallel between
        encoder and decoder. Attribute assignment on a vmap-stacked Param
        ties its full ``[repeats, ...]`` array, which is what we want:
        encoder layer i shares with decoder layer i for every i in the
        group.

        Tied: attention QKV/output kernels and dense MLP kernels.

        NOT tied: all biases (cheap per-projection offsets give the
        encoder/decoder a degree of freedom to differentiate while the
        kernels share gradient signal).

        Raises if encoder/decoder scan structures differ -- typically
        because the encoder and decoder ffn_types don't align across the
        layer stack. Align the ffn_types to enable tying.
        """
        enc_groups = self.encoder.h._scan_groups
        dec_groups = self.decoder.h._scan_groups

        if (enc_groups is None) != (dec_groups is None):
            raise ValueError(
                "tie_encoder_decoder requires both encoder and decoder to use "
                "the same use_scan_layers setting"
            )

        if enc_groups is None:
            # neither side uses scan -- iterate layers directly
            n = min(len(self.encoder.h.layers), len(self.decoder.h.layers))
            for i in range(n):
                enc_b = self.encoder.h.layers[i]
                dec_b = self.decoder.h.layers[i]
                self._tie_block_pair(enc_b, dec_b, enc_b.ffn_type, config)
            return

        # encoder has no cross-attention so position_specs always carry None
        # in the cross_attn slot; decoder carries its actual type. Strip
        # cross_attn from both before comparing -- only the ffn_type and
        # self-attn type need to match for tying to be structurally valid.
        def _strip_xattn(groups):
            return [
                (s, p, r, tuple((ffn_t, attn_t) for (ffn_t, attn_t, _) in specs))
                for (s, p, r, specs) in groups
            ]

        if _strip_xattn(enc_groups) != _strip_xattn(dec_groups):
            raise ValueError(
                f"tie_encoder_decoder requires matching scan-group structures; "
                f"encoder groups={enc_groups}, decoder groups={dec_groups}. "
                f"Common cause: encoder and decoder ffn_types differ across the "
                f"layer stack. Match the encoder and decoder ffn_types."
            )

        enc_bookend_idx = 0
        dec_bookend_idx = 0
        enc_stack_idx = 0
        dec_stack_idx = 0
        for (start, period, repeats, position_specs) in enc_groups:
            if period * repeats <= 1:
                enc_b = self.encoder.h.layers[enc_bookend_idx]
                dec_b = self.decoder.h.layers[dec_bookend_idx]
                enc_bookend_idx += 1
                dec_bookend_idx += 1
                ffn_t = position_specs[0][0]
                self._tie_block_pair(enc_b, dec_b, ffn_t, config)
                continue

            enc_stack = self.encoder.h._scan_stacks[enc_stack_idx]
            dec_stack = self.decoder.h._scan_stacks[dec_stack_idx]
            enc_stack_idx += 1
            dec_stack_idx += 1
            if period == 1:
                ffn_t = position_specs[0][0]
                self._tie_block_pair(enc_stack, dec_stack, ffn_t, config)
            else:
                for j, (ffn_t, _, _) in enumerate(position_specs):
                    self._tie_block_pair(
                        enc_stack.sublayers[j],
                        dec_stack.sublayers[j],
                        ffn_t, config,
                    )

    def _tie_block_pair(self, enc_block, dec_block, ffn_type, config):
        """Tie params between one encoder/decoder block pair.

        ``enc_block`` and ``dec_block`` are either individual
        ``FlaxHan2HanBlock`` instances (bookend path) or vmap-stacked
        equivalents from ``_scan_stacks`` -- attribute access works the
        same and ``=`` assignment ties the underlying arrays.

        Ties attention QKV/output kernels and dense MLP kernels. Biases
        are left untied: cheap symmetry-breaking that lets the
        encoder/decoder differentiate while the kernels stay shared.
        """
        enc_block.attn.query.kernel = dec_block.attn.query.kernel
        enc_block.attn.key.kernel = dec_block.attn.key.kernel
        enc_block.attn.value.kernel = dec_block.attn.value.kernel
        enc_block.attn.c_proj.kernel = dec_block.attn.c_proj.kernel

        self._tie_dense_mlp_weights(enc_block.mlp, dec_block.mlp, config)

    def _tie_dense_mlp_weights(self, enc_mlp, dec_mlp, config):
        """Tie kernels between two dense MLP modules.

        Handles both naming conventions:
        - T5 style: wi, wo, wi_0, wi_1
        - GPT style: c_fc, c_proj

        Biases are left untied to give encoder/decoder a small degree of
        freedom to differentiate while kernels share gradient signal.
        """
        if hasattr(dec_mlp, 'wi_0') and hasattr(dec_mlp, 'wi_1'):
            enc_mlp.wi_0.kernel = dec_mlp.wi_0.kernel
            enc_mlp.wi_1.kernel = dec_mlp.wi_1.kernel
        elif hasattr(dec_mlp, 'wi'):
            enc_mlp.wi.kernel = dec_mlp.wi.kernel
        elif hasattr(dec_mlp, 'c_fc'):
            enc_mlp.c_fc.kernel = dec_mlp.c_fc.kernel

        if hasattr(dec_mlp, 'wo'):
            enc_mlp.wo.kernel = dec_mlp.wo.kernel
        elif hasattr(dec_mlp, 'c_proj'):
            enc_mlp.c_proj.kernel = dec_mlp.c_proj.kernel

    def _load_subword_tables(self):
        """Load jbu and cbu subword lookup tables from model directory if they exist."""
        import os
        import numpy as np

        frozen_subword_lookups = {}

        # try to get model path from config
        model_path = getattr(self.config, '_name_or_path', None)
        if model_path and os.path.isdir(model_path):
            jbu_path = os.path.join(model_path, "jbu.npy")
            cbu_path = os.path.join(model_path, "cbu.npy")

            if self.config.jamo_subwords and self.encoder.subword_lookups is None:
                if os.path.exists(jbu_path):
                    jbu_array = np.load(jbu_path)
                    # Keep as numpy array - nnx.static doesn't accept JAX arrays
                    frozen_subword_lookups["jbu"] = jbu_array.astype(np.uint32)

            if self.config.char_subwords and self.encoder.subword_lookups is None:
                if os.path.exists(cbu_path):
                    cbu_array = np.load(cbu_path)
                    # Keep as numpy array - nnx.static doesn't accept JAX arrays
                    frozen_subword_lookups["cbu"] = cbu_array.astype(np.uint32)

        # Only create lookups if we actually have subword tables
        if frozen_subword_lookups:
            # SubwordLookups is an nnx.Object with StaticLookup variables
            # share the same object between encoder and decoder
            lookups = SubwordLookups(frozen_subword_lookups)
            self.encoder.subword_lookups = lookups
            self.decoder.subword_lookups = lookups
        else:
            self.encoder.subword_lookups = self.decoder.subword_lookups = None

    def set_subword_tables(self, jbu=None, cbu=None):
        """Set the jbu and cbu subword lookup tables directly.

        Args:
            jbu: Array-like object of jamo subword indices or path to jbu.npy file
            cbu: Array-like object of char subword indices or path to cbu.npy file
        """
        import os
        import numpy as np

        frozen_subword_lookups = {}

        if jbu is not None:
            if isinstance(jbu, str) and os.path.exists(jbu):
                jbu = np.load(jbu)
            frozen_subword_lookups["jbu"] = np.asarray(jbu, dtype=np.uint32)

        if cbu is not None:
            if isinstance(cbu, str) and os.path.exists(cbu):
                cbu = np.load(cbu)
            frozen_subword_lookups["cbu"] = np.asarray(cbu, dtype=np.uint32)

        # only create lookups if we actually have subword tables
        if frozen_subword_lookups:
            # SubwordLookups is an nnx.Object with StaticLookup variables
            # share the same object between encoder and decoder
            # use nnx.data() to override the static attribute status from initial None assignment
            lookups = SubwordLookups(frozen_subword_lookups)
            self.encoder.subword_lookups = lookups
            self.decoder.subword_lookups = lookups

    def resize_token_embeddings(
        self,
        new_num_tokens: int,
        rngs: Optional[nnx.Rngs] = None,
        init_token_id: Optional[int] = None
    ):
        """
        Resize the model's token embeddings and lm_head for vocabulary extension.

        New tokens are initialized with a reference token's embedding values.
        Subword lookup tables (jbu/cbu) are extended with pad token's bucket values.

        Args:
            new_num_tokens: New vocabulary size
            rngs: Optional nnx.Rngs for initialization (uses existing if None)
            init_token_id: Token ID to use for initializing new embeddings.
                           Defaults to pad_token_id. Consider using:
                           - mask_token_id (4) for sentinel tokens
                           - unk_token_id (1) for byte fallback tokens

        Returns:
            The model itself for method chaining
        """
        old_num_tokens = self.config.vocab_size

        if new_num_tokens == old_num_tokens:
            return self

        if new_num_tokens < old_num_tokens:
            raise ValueError(
                f"Cannot shrink vocabulary from {old_num_tokens} to {new_num_tokens}. "
                "Only expansion is supported."
            )

        pad_token_id = self.config.pad_token_id
        embed_init_id = init_token_id if init_token_id is not None else pad_token_id

        self.encoder.wte = self._resize_embedding(
            self.encoder.wte, new_num_tokens, embed_init_id, rngs
        )
        self.decoder.wte = self._resize_embedding(
            self.decoder.wte, new_num_tokens, embed_init_id, rngs
        )

        # note: wje/wce (jamo/char embeddings) are NOT resized - they're indexed by
        # bucket hash, not token ID. only the lookup tables (jbu/cbu) need extension.

        self._resize_subword_lookups(old_num_tokens, new_num_tokens, pad_token_id)

        if not self.config.tie_input_output_embeddings:
            self.lm_head = self._resize_lm_head(
                self.lm_head, new_num_tokens, embed_init_id, rngs
            )

        self.config.vocab_size = new_num_tokens

        self.tie_weights()

        init_token_name = {0: "pad", 1: "unk", 4: "mask"}.get(embed_init_id, str(embed_init_id))
        log_from_main_process(logger, 'info',
            f"Resized vocabulary from {old_num_tokens} to {new_num_tokens} "
            f"(init from <{init_token_name}>)"
        )
        return self

    def _resize_embedding(
        self,
        old_embed: nnx.Embed,
        new_num_tokens: int,
        init_token_id: int,
        rngs: Optional[nnx.Rngs] = None
    ) -> nnx.Embed:
        """Resize an embedding layer, initializing new tokens from a reference token."""
        old_num_tokens = old_embed.num_embeddings
        embedding_dim = old_embed.features

        if new_num_tokens == old_num_tokens:
            return old_embed

        old_weights = old_embed.embedding.value
        init_embedding = old_weights[init_token_id]

        num_new = new_num_tokens - old_num_tokens
        new_rows = jnp.tile(init_embedding[None, :], (num_new, 1))
        new_weights = jnp.concatenate([old_weights, new_rows], axis=0)

        new_embed = nnx.Embed(
            num_embeddings=new_num_tokens,
            features=embedding_dim,
            dtype=old_embed.embedding.value.dtype,
            param_dtype=old_embed.embedding.value.dtype,
            rngs=rngs or nnx.Rngs(0),
        )
        new_embed.embedding.value = new_weights

        return new_embed

    def _resize_lm_head(
        self,
        old_head: nnx.Linear,
        new_num_tokens: int,
        pad_token_id: int,
        rngs: Optional[nnx.Rngs] = None
    ) -> nnx.Linear:
        """Resize the lm_head linear layer."""
        old_num_tokens = old_head.out_features
        in_features = old_head.in_features

        if new_num_tokens == old_num_tokens:
            return old_head

        old_kernel = old_head.kernel.value
        pad_weights = old_kernel[:, pad_token_id]

        num_new = new_num_tokens - old_num_tokens
        new_cols = jnp.tile(pad_weights[:, None], (1, num_new))
        new_kernel = jnp.concatenate([old_kernel, new_cols], axis=1)

        new_head = nnx.Linear(
            in_features=in_features,
            out_features=new_num_tokens,
            dtype=old_kernel.dtype,
            param_dtype=old_kernel.dtype,
            use_bias=old_head.bias is not None,
            rngs=rngs or nnx.Rngs(0),
        )
        new_head.kernel.value = new_kernel

        if old_head.bias is not None:
            old_bias = old_head.bias.value
            pad_bias = old_bias[pad_token_id]
            new_bias_vals = jnp.full((num_new,), pad_bias)
            new_head.bias.value = jnp.concatenate([old_bias, new_bias_vals], axis=0)

        return new_head

    def _resize_subword_lookups(
        self,
        old_num_tokens: int,
        new_num_tokens: int,
        pad_token_id: int
    ):
        """Resize jbu/cbu lookup tables by extending with pad token's bucket values."""
        lookups = self.encoder.subword_lookups
        if lookups is None:
            return

        num_new = new_num_tokens - old_num_tokens
        new_lookups = {}

        if lookups.jbu is not None:
            jbu_array = lookups.jbu.value
            if pad_token_id < len(jbu_array):
                pad_bucket = jbu_array[pad_token_id]
                extension = np.tile(pad_bucket[None, :], (num_new, 1))
                new_jbu = np.concatenate([jbu_array, extension], axis=0)
            else:
                new_jbu = np.pad(
                    jbu_array,
                    ((0, num_new), (0, 0)),
                    mode='constant',
                    constant_values=0
                )
            new_lookups["jbu"] = new_jbu.astype(np.uint32)

        if lookups.cbu is not None:
            cbu_array = lookups.cbu.value
            if pad_token_id < len(cbu_array):
                pad_bucket = cbu_array[pad_token_id]
                extension = np.tile(pad_bucket[None, :], (num_new, 1))
                new_cbu = np.concatenate([cbu_array, extension], axis=0)
            else:
                new_cbu = np.pad(
                    cbu_array,
                    ((0, num_new), (0, 0)),
                    mode='constant',
                    constant_values=0
                )
            new_lookups["cbu"] = new_cbu.astype(np.uint32)

        if new_lookups:
            new_lookups_obj = SubwordLookups(new_lookups)
            self.encoder.subword_lookups = new_lookups_obj
            self.decoder.subword_lookups = new_lookups_obj

    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        segment_ids: Optional[jnp.ndarray] = None,
        decoder_input_ids: Optional[jnp.ndarray] = None,
        decoder_attention_mask: Optional[jnp.ndarray] = None,
        decoder_position_ids: Optional[jnp.ndarray] = None,
        decoder_segment_ids: Optional[jnp.ndarray] = None,
        decoder_input_embeddings: Optional[jnp.ndarray] = None,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        past_key_values: Optional[dict] = None,
        init_cache: bool = False,
        output_hidden_states: bool = False,
        output_sentence_embeddings: bool = False,
        return_dict: Optional[bool] = None,
        rngs: Optional[nnx.Rngs] = None,
        deterministic: Optional[bool] = True,
        precomputed_encoder_last_hidden_state: Optional[jnp.ndarray] = None,
    ) -> Union[Tuple, FlaxSeq2SeqLMOutput]:

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else False
        )
        return_dict = return_dict if return_dict is not None else True

        if encoder_hidden_states is not None:
            encoder_attention_mask = jnp.broadcast_to(encoder_attention_mask[..., None], encoder_hidden_states.shape).astype("i4")

        if attention_mask is None:
            attention_mask = jnp.where(input_ids == self.pad_token_id, 0, 1).astype("i4")

        # caller-supplied encoder outputs short-circuit the encoder forward;
        # used by sft_train_step to share encoder activations between the
        # gold and scheduled-sampling decoder passes (both run in train mode,
        # same encoder inputs, so the encoder output is identical).
        if precomputed_encoder_last_hidden_state is not None:
            encoder_outputs = None
            hidden_states = precomputed_encoder_last_hidden_state
        else:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                segment_ids=segment_ids,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                init_cache=init_cache,
                past_key_values=past_key_values,
                rngs=rngs,
                deterministic=deterministic
            )

            hidden_states = encoder_outputs[0] if not return_dict else encoder_outputs.last_hidden_state

        if decoder_input_ids is None:
            decoder_input_ids = shift_tokens_right(
                input_ids.copy(), self.pad_token_id, self.decoder_start_token_id
            )

        if decoder_attention_mask is None:
            decoder_attention_mask = jnp.where(decoder_input_ids == self.pad_token_id, 0, 1).astype("i4")

        # for BART training, use full encoder sequences; for TSDAE, use pooled embeddings
        if self.config.use_bart_training:
            encoder_hidden_for_decoder = hidden_states  # (batch, seq_len, d_model)
            encoder_attention_mask_for_decoder = attention_mask  # (batch, seq_len)
        else:
            input_mask = jnp.where(input_ids != self.pad_token_id, 1.0, 0.0)
            input_mask_expanded = jnp.expand_dims(input_mask, axis=-1)

            sum_embeddings = jnp.sum(hidden_states * input_mask_expanded, axis=1)
            sum_mask = jnp.sum(input_mask_expanded, axis=1)
            sum_mask = jnp.maximum(sum_mask, 1e-9)
            sentence_embeddings = sum_embeddings / sum_mask

            if output_sentence_embeddings:
                return (sentence_embeddings,)

            encoder_hidden_for_decoder = sentence_embeddings[:, None, :]  # (batch, 1, d_model)
            encoder_attention_mask_for_decoder = attention_mask[..., 0:1]  # (batch, 1)

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            position_ids=decoder_position_ids,
            segment_ids=decoder_segment_ids,
            input_embeddings=decoder_input_embeddings,
            encoder_hidden_states=encoder_hidden_for_decoder,
            encoder_attention_mask=encoder_attention_mask_for_decoder,
            encoder_position_ids=position_ids,
            encoder_segment_ids=segment_ids,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            init_cache=init_cache,
            past_key_values=past_key_values,
            rngs=rngs,
            deterministic=deterministic
        )

        decoder_hidden_states = decoder_outputs[0] if not return_dict else decoder_outputs.last_hidden_state

        if self.config.tie_input_output_embeddings:
            # directly use decoder.wte to avoid TiedLinear reference issues after checkpoint restoration
            logits = decoder_hidden_states @ self.decoder.wte.embedding.value.T
        else:
            logits = self.lm_head(decoder_hidden_states)

        if not return_dict:
            enc_tail = () if encoder_outputs is None else encoder_outputs[1:]
            return (logits,) + enc_tail + decoder_outputs[1:]

        return FlaxSeq2SeqLMOutput(
            logits=logits,
            hidden_states=decoder_hidden_states,
            sentence_embeddings=sentence_embeddings if not self.config.use_bart_training else None,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=(
                hidden_states if encoder_outputs is None else encoder_outputs.last_hidden_state
            ),
            encoder_hidden_states=None if encoder_outputs is None else encoder_outputs.hidden_states,
            encoder_attentions=None if encoder_outputs is None else encoder_outputs.attentions,
        )

    def generate(
        self,
        input_ids: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        decoder_input_ids: Optional[jnp.ndarray] = None,
        max_length: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        min_length: Optional[int] = None,
        do_sample: bool = False,
        early_stopping: bool = True,
        num_beams: int = 1,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        length_penalty: float = 1.0,
        decoder_start_token_id: Optional[int] = None,
        use_cache: bool = True,
        num_return_sequences: int = 1,
        use_fixed_length_generation: bool = True,
        suppress_tokens: Optional[List[int]] = None,
        tokenizer: Any = None,
        return_text: bool = False,
        rngs: Optional[nnx.Rngs] = None,
        local_batch_size: Optional[int] = None,
        **model_kwargs,
    ) -> jnp.ndarray:
        """Generate sequences using Han2HanSampler.

        Uses the efficient Gemma-style sampler with nnx.split/merge pattern for
        optimal JIT compilation and generation speed.

        Supports:
        - Standard KV-cached generation
        - Beam search
        - Temperature, top-k, top-p sampling
        - Repetition penalty and n-gram blocking

        Args:
            input_ids: encoder input token ids (batch_size, seq_len)
            attention_mask: encoder attention mask
            decoder_input_ids: optional decoder prompt/conditioning sequence
            max_length: maximum total length of generated sequences
            max_new_tokens: maximum number of new tokens to generate
            min_length: minimum length before EOS is allowed
            do_sample: whether to use sampling (temperature > 0 enables this)
            early_stopping: stop beam search when num_beams sentences finish
            num_beams: number of beams for beam search (1 = greedy/sampling)
            temperature: sampling temperature (0 = greedy)
            top_k: number of top tokens to consider for top-k sampling
            top_p: cumulative probability for nucleus sampling
            repetition_penalty: penalty for repeated tokens (>1 reduces repetition)
            no_repeat_ngram_size: block n-grams from repeating (0 = disabled)
            bos_token_id: beginning of sentence token id (unused)
            pad_token_id: padding token id
            eos_token_id: end of sentence token id
            length_penalty: beam search length penalty
            decoder_start_token_id: token id to start decoding with
            use_cache: whether to use kv caching
            num_return_sequences: number of sequences to return (unused currently)
            use_fixed_length_generation: unused (kept for API compatibility)
            suppress_tokens: list of token ids to never generate
            tokenizer: optional tokenizer for decoding (if return_text=True)
            return_text: whether to return decoded text along with tokens
            rngs: random number generators for sampling
            local_batch_size: local batch size for SPMD (global array shape != local)
            **model_kwargs: additional model arguments (unused)

        Returns:
            generated token sequences (batch_size, seq_len)
        """
        from han2han_sampler import Han2HanSampler

        if not hasattr(self, '_sampler') or self._sampler is None:
            self._sampler = Han2HanSampler(
                model=self,
                tokenizer=tokenizer,
                max_length=max_length or self.config.n_positions,
            )

        if do_sample and temperature == 1.0:
            effective_temperature = 1.0
        elif do_sample:
            effective_temperature = temperature
        else:
            effective_temperature = 0.0

        seed = None
        if rngs is not None and hasattr(rngs, 'dropout'):
            seed = rngs.dropout()
        elif do_sample:
            seed = jax.random.PRNGKey(42)

        output = self._sampler(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            min_length=min_length or 0,
            num_beams=num_beams,
            temperature=effective_temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
            decoder_start_token_id=(
                decoder_start_token_id
                or getattr(self.config, 'decoder_start_token_id', None)
            ),
            eos_token_id=(
                eos_token_id
                or getattr(self.config, 'eos_token_id', None)
                or decoder_start_token_id
            ),
            pad_token_id=pad_token_id,
            suppress_tokens=suppress_tokens,
            seed=seed,
            return_text=return_text,
            local_batch_size=local_batch_size,
        )

        return output.tokens


class FlaxHan2HanForSequenceClassification(nnx.Module):
    """Classification head on top of FlaxHan2Han encoder-decoder.

    Runs the full encoder-decoder, extracts the last non-padding decoder
    hidden state, then projects through a classification head.

    Head types (config.classifier_head_type):
      - 'linear': dropout -> linear (T5Gemma 2 style)
      - 'mlp': dropout -> dense -> tanh -> dropout -> linear (RoBERTa style)

    For pair tasks (e.g. STS), put one sentence in the encoder and the
    other in the decoder so cross-attention compares them naturally.
    """

    def __init__(
        self,
        config: Han2HanConfig,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: Optional[jnp.dtype] = None,
        sharding: tuple = ('data', None),
        gradient_checkpointing: bool = False,
        jamo_buckets: np.ndarray = None,
        char_buckets: np.ndarray = None,
    ):
        self.config = config
        self.head_type = getattr(config, 'classifier_head_type', 'linear')
        # latent-weight storage dtype; defaults to compute dtype for parity with
        # FlaxHan2Han. Forwarding this to the head matters when the backbone is
        # bf16 and the optimizer state walks both halves: a mixed-dtype param
        # tree breaks optax.partition/MultiSteps' lax.cond (init vs post-update
        # branches end up with different dtypes per leaf).
        if param_dtype is None:
            param_dtype = dtype
        self.model = FlaxHan2Han(
            config, rngs, dtype=dtype, param_dtype=param_dtype, sharding=sharding,
            gradient_checkpointing=gradient_checkpointing,
            jamo_buckets=jamo_buckets, char_buckets=char_buckets,
        )
        self.classifier_dropout = nnx.Dropout(config.classf_pdrop, rngs=rngs)
        if self.head_type == 'mlp':
            self.dense = nnx.Linear(
                config.d_model, config.d_model,
                rngs=rngs, dtype=jnp.float32, param_dtype=param_dtype,
            )
            self.out_dropout = nnx.Dropout(config.classf_pdrop, rngs=rngs)
        self.classifier = nnx.Linear(
            config.d_model, config.num_labels,
            rngs=rngs, dtype=jnp.float32, param_dtype=param_dtype,
        )

    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: jnp.ndarray,
        decoder_input_ids: jnp.ndarray,
        decoder_attention_mask: jnp.ndarray,
        deterministic: bool = True,
        rngs: Optional[nnx.Rngs] = None,
    ) -> FlaxSequenceClassifierOutput:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            deterministic=deterministic,
            rngs=rngs,
            return_dict=True,
        )

        hidden = outputs.hidden_states
        last_non_pad = decoder_attention_mask.sum(axis=-1).astype(jnp.int32) - 1
        last_non_pad = jnp.clip(last_non_pad, 0)
        pooled = hidden[jnp.arange(hidden.shape[0]), last_non_pad]
        pooled = self.classifier_dropout(pooled, deterministic=deterministic)
        if self.head_type == 'mlp':
            pooled = jnp.tanh(self.dense(pooled.astype(jnp.float32)))
            pooled = self.out_dropout(pooled, deterministic=deterministic)
        logits = self.classifier(pooled.astype(jnp.float32))

        return FlaxSequenceClassifierOutput(logits=logits, hidden_states=pooled)


# register with AutoClasses when module is imported
try:
    import register_han2han # noqa F401
except ImportError:
    pass  # registration is optional
