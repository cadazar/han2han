#!/usr/bin/env python3
# coding: utf-8

from dataclasses import dataclass
from typing import Optional, Tuple, Union, Dict

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    ModelOutput,
    QuestionAnsweringModelOutput,
    MultipleChoiceModelOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.generation.utils import GenerationMixin
from transformers.utils import logging

# Heavy/optional deps (FLA fused kernels) are imported lazily inside the modules
# that use them so conversion / inference works on a clean torch+transformers env
# without TPU-flavored extras. The dense MHA path does not need any of these; FLA
# fused RotaryEmbedding / RMSNorm / GatedMLP are opt-in via config flags.
RotaryEmbedding = None  # populated by _import_fla_modules() if available
GatedMLP = None
FLARMSNorm = None


def _import_fla_modules():
    global RotaryEmbedding, GatedMLP, FLARMSNorm
    if RotaryEmbedding is not None:
        return
    from fla.modules import RotaryEmbedding as _Rotary, GatedMLP as _Gated, RMSNorm as _RMSNorm
    RotaryEmbedding = _Rotary
    GatedMLP = _Gated
    FLARMSNorm = _RMSNorm


from han2han_config import Han2HanConfig

logger = logging.get_logger(__name__)


class SimpleRotaryEmbedding(nn.Module):
    """Simple PyTorch-native rotary embedding implementation as CPU fallback."""

    def __init__(self, dim: int, base: float = 10000.0, interleaved: bool = False):
        super().__init__()
        self.dim = dim
        self.base = base
        self.interleaved = interleaved
        self._cos_cached = None
        self._sin_cached = None
        self._seq_len_cached = -1

    def _update_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            # Compute the inverse frequencies
            inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=device, dtype=dtype) / self.dim))
            # Compute the positions
            pos = torch.arange(seq_len, device=device, dtype=dtype)
            # Compute the frequencies
            freqs = torch.einsum("i,j->ij", pos, inv_freq)
            # Cache cos and sin
            self._cos_cached = freqs.cos()
            self._sin_cached = freqs.sin()

    def forward(self, q: torch.Tensor, k: torch.Tensor, seqlen_offsets: int = 0):
        """Apply rotary embeddings to query and key tensors.

        Args:
            q: Query tensor of shape (batch, seq_len, num_heads, head_dim) or (batch, seq_len, head_dim)
            k: Key tensor of shape (batch, seq_len, num_heads, head_dim) or (batch, seq_len, head_dim)
            seqlen_offsets: Offset for position indices (for kv-cache scenarios)

        Returns:
            Tuple of rotated (q, k) tensors
        """
        orig_dtype = q.dtype
        q = q.float()
        k = k.float()

        # Update cache if needed - use max of q and k sequence lengths
        seq_len_q = q.shape[1]
        seq_len_k = k.shape[1]
        max_seq_len = max(seq_len_q, seq_len_k)
        self._update_cos_sin_cache(max_seq_len, q.device, q.dtype)

        # Handle both 3D and 4D inputs
        if q.ndim == 3:
            # Add heads dimension
            q = q.unsqueeze(2)
            k = k.unsqueeze(2)
            squeeze_after = True
        else:
            squeeze_after = False

        # Split the last dimension in half for real and imaginary parts
        q_r, q_i = q.chunk(2, dim=-1)
        k_r, k_i = k.chunk(2, dim=-1)

        # Get cos and sin for each tensor's sequence length
        cos_q = self._cos_cached[:seq_len_q].to(q.device)
        sin_q = self._sin_cached[:seq_len_q].to(q.device)
        cos_k = self._cos_cached[:seq_len_k].to(k.device)
        sin_k = self._sin_cached[:seq_len_k].to(k.device)

        # Reshape for broadcasting
        if q.ndim == 4:  # (batch, seq_len, num_heads, head_dim/2)
            cos_q = cos_q.unsqueeze(1).unsqueeze(0)  # (1, seq_len_q, 1, dim/2)
            sin_q = sin_q.unsqueeze(1).unsqueeze(0)  # (1, seq_len_q, 1, dim/2)
            cos_k = cos_k.unsqueeze(1).unsqueeze(0)  # (1, seq_len_k, 1, dim/2)
            sin_k = sin_k.unsqueeze(1).unsqueeze(0)  # (1, seq_len_k, 1, dim/2)

        # Apply rotation to q and k separately
        q_rot_r = q_r * cos_q - q_i * sin_q
        q_rot_i = q_r * sin_q + q_i * cos_q
        k_rot_r = k_r * cos_k - k_i * sin_k
        k_rot_i = k_r * sin_k + k_i * cos_k

        # Concatenate back
        q_rot = torch.cat([q_rot_r, q_rot_i], dim=-1)
        k_rot = torch.cat([k_rot_r, k_rot_i], dim=-1)

        if squeeze_after:
            q_rot = q_rot.squeeze(2)
            k_rot = k_rot.squeeze(2)

        return q_rot.to(orig_dtype), k_rot.to(orig_dtype)


def apply_rotary_embedding_simple(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings using precomputed cos/sin values.

    Args:
        x: Input tensor of shape (batch, seq_len, head_dim) or (batch, seq_len, num_heads, head_dim)
        cos: Cosine values of shape (seq_len, head_dim/2) or longer
        sin: Sine values of shape (seq_len, head_dim/2) or longer

    Returns:
        Rotated tensor of same shape as input
    """
    orig_dtype = x.dtype
    x = x.float()

    # Get the actual sequence length from input
    seq_len = x.shape[1]

    # Ensure cos/sin match the input sequence length
    if cos.shape[0] > seq_len:
        cos = cos[:seq_len]
        sin = sin[:seq_len]
    elif cos.shape[0] < seq_len:
        raise ValueError(f"Cos/sin cache too small: {cos.shape[0]} < {seq_len}")

    # Split the last dimension in half
    x_r, x_i = x.chunk(2, dim=-1)

    # Reshape cos/sin for broadcasting
    if x.ndim == 4:  # (batch, seq_len, num_heads, head_dim/2)
        cos = cos.unsqueeze(1).unsqueeze(0)  # (1, seq_len, 1, head_dim/2)
        sin = sin.unsqueeze(1).unsqueeze(0)
    elif x.ndim == 3:  # (batch, seq_len, head_dim/2)
        cos = cos.unsqueeze(0)  # (1, seq_len, head_dim/2)
        sin = sin.unsqueeze(0)

    # Apply rotation
    x_rot_r = x_r * cos - x_i * sin
    x_rot_i = x_r * sin + x_i * cos

    # Concatenate back
    x_rot = torch.cat([x_rot_r, x_rot_i], dim=-1)

    return x_rot.to(orig_dtype)


class FlaxLinear(nn.Module):
    """A drop-in replacement for ``nn.Linear`` whose `weight` parameter is stored
    in `(in_features, out_features)` order (matching Flax).  The forward pass
    emulates a standard linear transformation using :pyfunc:`torch.matmul` so
    that the runtime behavior is identical while the parameter layout stays
    compatible with the Flax checkpoints used in unit-tests."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True, *, std: float = 0.02):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Parameter initialisation follows the Flax convention: kernel shape
        # is (in_features, out_features).
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = torch.matmul(input, self.weight.to(input.device))
        if self.bias is not None:
            output = output + self.bias.to(input.device)
        return output

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int) -> torch.Tensor:
    """
    Shift input ids one token to the right, matching Flax behavior.
    """
    shifted_input_ids = torch.zeros_like(input_ids)
    shifted_input_ids[:, 1:] = input_ids[:, :-1]
    shifted_input_ids[:, 0] = decoder_start_token_id
    shifted_input_ids = torch.where(shifted_input_ids == -100, pad_token_id, shifted_input_ids)
    return shifted_input_ids


@dataclass
class Seq2SeqLMOutput(ModelOutput):
    logits: torch.Tensor = None
    hidden_states: Optional[Tuple[torch.Tensor]] = None
    sentence_embeddings: Optional[torch.Tensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.Tensor]] = None
    decoder_attentions: Optional[Tuple[torch.Tensor]] = None
    cross_attentions: Optional[Tuple[torch.Tensor]] = None
    encoder_last_hidden_state: Optional[torch.Tensor] = None
    encoder_hidden_states: Optional[Tuple[torch.Tensor]] = None
    encoder_attentions: Optional[Tuple[torch.Tensor]] = None


@dataclass
class BaseModelOutputWithPastAndCrossAttentions(ModelOutput):
    last_hidden_state: torch.Tensor = None
    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None
    hidden_states: Optional[Tuple[torch.Tensor]] = None
    attentions: Optional[Tuple[torch.Tensor]] = None
    cross_attentions: Optional[Tuple[torch.Tensor]] = None


@dataclass
class BaseModelOutputWithPastAttentionsAndSentenceEmbeddings(ModelOutput):
    last_hidden_state: torch.Tensor = None
    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None
    sentence_embeddings: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.Tensor]] = None
    attentions: Optional[Tuple[torch.Tensor]] = None

@dataclass
class Seq2SeqSequenceClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    sentence_embeddings: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class ClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    sentence_embeddings: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class Seq2SeqModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    sentence_embeddings: Optional[Tuple[torch.FloatTensor]] = None


class Han2HanAttention(nn.Module):
    def __init__(
        self,
        config: Han2HanConfig,
        is_cross_attention: bool = False,
        is_causal: bool = True,
        rope_theta: Optional[float] = None,
        attention_type: Optional[str] = None,
    ):
        super().__init__()

        # Per-layer override > config default > 'mha' fallback. V1 configs set
        # config.attention_mechanism=None and use encoder_attention_types /
        # decoder_attention_types / decoder_cross_attention_types lists instead.
        resolved_attn_type = attention_type or config.attention_mechanism or 'mha'

        # only dense multi-head attention (optionally sliding-window) is supported.
        if 'mha' not in resolved_attn_type:
            raise NotImplementedError(
                f"attention_mechanism={resolved_attn_type!r} is not supported; "
                "use 'mha' or 'mha-sliding'."
            )
        self.attention_mechanism = resolved_attn_type
        self.is_cross_attention = is_cross_attention
        self.is_causal = is_causal

        # GQA-aware projection sizing per modeling_han2han_flax.py:960-968.
        # Defaults preserve the d_prime sizing of the older han2han-ul2-bart_base_v5e-US.yaml
        # checkpoints (head_dim=None and num_kv_heads=None collapse to q_proj_dim==kv_proj_dim==d_prime).
        assert config.num_heads is not None, "num_heads must be specified for MHA attention"
        num_heads = config.num_heads
        head_dim = getattr(config, 'head_dim', None)
        if is_cross_attention:
            num_kv_heads = getattr(config, 'cross_attn_num_kv_heads', None)
        else:
            num_kv_heads = getattr(config, 'num_kv_heads', None)
        if num_kv_heads is None:
            num_kv_heads = num_heads
        if head_dim is None:
            head_dim = config.d_prime // num_heads
        q_proj_dim = num_heads * head_dim
        kv_proj_dim = num_kv_heads * head_dim

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_proj_dim = q_proj_dim
        self.kv_proj_dim = kv_proj_dim
        self.d_model = config.d_model

        def make_proj(in_features, out_features):
            return FlaxLinear(
                in_features, out_features, bias=config.use_bias,
                std=config.initializer_range,
            )

        self.query = make_proj(config.d_model, q_proj_dim)
        self.key = make_proj(config.d_model, kv_proj_dim)
        self.value = make_proj(config.d_model, kv_proj_dim)
        self.c_proj = make_proj(q_proj_dim, config.d_model)

        dropout_rate = config.cross_attn_pdrop if is_cross_attention else config.attn_pdrop
        self.attn_dropout = nn.Dropout(dropout_rate)

        effective_rope_theta = rope_theta if rope_theta is not None else config.rope_theta
        self.rope_theta = effective_rope_theta

        # rotary operates per-head; use head_dim, not d_prime / q_proj_dim. apply
        # via apply_rotary_embedding_simple after the per-head reshape (Flax 1141-1179).
        # FLA fused rotary is opt-in (config.use_fla_fused_rotary). the dense MHA
        # path doesn't use it -- the GQA forward applies RoPE manually via
        # apply_rotary_embedding_simple -- so self.rotary is retained only for
        # callers that flip the flag.
        self.use_fla_fused_rotary = config.use_fla_fused_rotary
        if self.use_fla_fused_rotary:
            try:
                _import_fla_modules()
                self.rotary = RotaryEmbedding(
                    dim=head_dim,
                    base=effective_rope_theta,
                    interleaved=False,
                )
            except Exception as e:
                logger.warning(f"Failed to initialize FLA RotaryEmbedding, falling back to simple implementation: {e}")
                self.use_fla_fused_rotary = False
                self.rotary = SimpleRotaryEmbedding(
                    dim=head_dim,
                    base=effective_rope_theta,
                    interleaved=False,
                )
        else:
            self.rotary = SimpleRotaryEmbedding(
                dim=head_dim,
                base=effective_rope_theta,
                interleaved=False,
            )

        # SubLN: RMSNorm before output projection (applied on (B, Tq, q_proj_dim)).
        # Mirrors modeling_han2han_flax.py:1016-1021. used for extra expressivity
        # between fully-tied encoder/decoder layers.
        # Flax passes eps=config.layer_norm_epsilon to these RMSNorms, so match it here
        # rather than the PT RMSNorm class default (1e-8).
        self.attn_sub_norm = None
        if getattr(config, 'use_sub_ln', False):
            self.attn_sub_norm = RMSNorm(
                q_proj_dim, eps=config.layer_norm_epsilon,
                use_fla_fused=config.use_fla_fused_norm,
            )

        # QK-norm: per-head RMSNorm on Q and K after RoPE (Gemma 3 / T5Gemma 2 style).
        # Mirrors modeling_han2han_flax.py:1026-1033 -- also eps=config.layer_norm_epsilon.
        self.q_norm = None
        self.k_norm = None
        if getattr(config, 'use_qk_norm', False):
            self.q_norm = RMSNorm(
                head_dim, eps=config.layer_norm_epsilon,
                use_fla_fused=config.use_fla_fused_norm,
            )
            self.k_norm = RMSNorm(
                head_dim, eps=config.layer_norm_epsilon,
                use_fla_fused=config.use_fla_fused_norm,
            )

        # Sliding-window self/cross-attention: 'mha-sliding' / 'mha-local' use
        # sliding_window_size with the per-layer effective_rope_theta (caller picks
        # rope_theta_sliding for these layers in Han2HanBlock).
        self.is_sliding = ('sliding' in resolved_attn_type) or ('local' in resolved_attn_type)
        self.window_size = getattr(config, 'sliding_window_size', 0) if self.is_sliding else 0

    def _build_rope_cache(self, max_len: int, device: torch.device, dtype: torch.dtype):
        """Precompute cos/sin for `max_len` positions at our head_dim.
        Returns cos, sin of shape (max_len, head_dim/2) in fp32 (caller casts).
        """
        inv_freq = 1.0 / (self.rope_theta ** (
            torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float32) / self.head_dim
        ))
        pos = torch.arange(max_len, device=device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", pos, inv_freq)
        return freqs.cos(), freqs.sin()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        init_cache: bool = False,
        output_attentions: bool = False,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:

        # standard self / cross-attention.
        if self.is_cross_attention and encoder_hidden_states is not None:
            Q = self.query(hidden_states)
            K = self.key(encoder_hidden_states)
            V = self.value(encoder_hidden_states)
            mask = encoder_attention_mask
        else:
            Q = self.query(hidden_states)
            K = self.key(hidden_states)
            V = self.value(hidden_states)
            mask = attention_mask

        B, Tq, _ = Q.shape
        Tkv = K.shape[1]

        # KV cache: concatenate past with new (unrotated; RoPE is applied per-step
        # below using kv_len-relative positions).
        if past_key_value is not None:
            past_k, past_v = past_key_value
            K = torch.cat([past_k, K], dim=1)
            V = torch.cat([past_v, V], dim=1)
            Tkv = K.shape[1]

        # legacy tuple-cache contract: present holds the FULL concatenated K/V
        # so BlockCollection can stash it as past_key_values[layer_idx] for the
        # next step. Migration to HF EncoderDecoderCache (which would have us
        # return only new tokens here and let the cache class concat) is deferred
        # to a follow-up PR -- see plan A.5.
        use_cache_out = init_cache or (past_key_value is not None)
        present = (K, V) if use_cache_out else None

        causal = self.is_causal and not self.is_cross_attention

        if 'mha' in self.attention_mechanism:
            # Reshape: Q gets num_heads; K/V get num_kv_heads (GQA-friendly).
            q = Q.view(B, Tq, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
            k = K.view(B, Tkv, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
            v = V.view(B, Tkv, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()

            # Apply RoPE per-head AFTER reshape (matches Flax 1141-1179). For cache
            # scenarios Q's positions live at [kv_len-q_len : kv_len], not [0 : q_len].
            if not self.is_cross_attention:
                cos, sin = self._build_rope_cache(Tkv, q.device, q.dtype)
                cos = cos.to(q.dtype)
                sin = sin.to(q.dtype)
                # q is (B, num_heads, Tq, head_dim); transpose to (B, Tq, num_heads, head_dim)
                # for apply_rotary_embedding_simple, then transpose back.
                q_seq = q.permute(0, 2, 1, 3).contiguous()
                k_seq = k.permute(0, 2, 1, 3).contiguous()
                q_seq = apply_rotary_embedding_simple(q_seq, cos[Tkv - Tq:Tkv], sin[Tkv - Tq:Tkv])
                k_seq = apply_rotary_embedding_simple(k_seq, cos[:Tkv], sin[:Tkv])
                q = q_seq.permute(0, 2, 1, 3).contiguous()
                k = k_seq.permute(0, 2, 1, 3).contiguous()

            # Post-RoPE QK-norm (Gemma 3 / T5Gemma 2 order). RMSNorm normalizes the
            # last axis (head_dim) and broadcasts across the head axis.
            if self.q_norm is not None:
                q = self.q_norm(q)
            if self.k_norm is not None:
                k = self.k_norm(k)

            # 4D attention mask matching Flax modeling_han2han_flax.py:1097-1138 +
            # sliding-window when self.is_sliding (modeling_han2han_flax.py:683-704).
            attn_mask = self._build_4d_mask(mask, B, Tq, Tkv, causal, q.device)

            # When the mask already encodes causal structure (cache scenario or 4D
            # input), do not double-apply via SDPA's is_causal flag. Sliding windows
            # always require the explicit mask path.
            sdpa_is_causal = causal and attn_mask is None

            # Convert bool mask to additive float mask using finfo.min, matching
            # Flax `jnp.where(mask, logits, finfo.min)`. SDPA with a bool mask
            # fills False positions with -inf, so a fully-masked row produces
            # softmax(-inf, ...) = NaN. finfo.min is finite, so an all-masked row
            # softmaxes to a uniform distribution (finite output).
            if attn_mask is not None and attn_mask.dtype == torch.bool:
                additive = torch.zeros((), dtype=q.dtype, device=q.device).expand_as(attn_mask).clone()
                additive.masked_fill_(~attn_mask, torch.finfo(q.dtype).min)
                attn_mask = additive

            dropout_p = self.attn_dropout.p if self.training else 0.0
            sdpa_kwargs = dict(
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=sdpa_is_causal,
            )
            if self.num_heads != self.num_kv_heads:
                # Manual GQA broadcast via repeat_interleave. SDPA's enable_gqa
                # kernel produced ~0.15 max_abs encoder drift vs Flax at fp32
                # because of a slightly different reduction order; explicit
                # repeat keeps the matmul order identical to Flax's GQA einsum
                # (`BTKGH,BSKH->BTKGS`) at the cost of K/V memory.
                repeat = self.num_heads // self.num_kv_heads
                k_e = k.repeat_interleave(repeat, dim=1)
                v_e = v.repeat_interleave(repeat, dim=1)
                attn_output = F.scaled_dot_product_attention(q, k_e, v_e, **sdpa_kwargs)
            else:
                attn_output = F.scaled_dot_product_attention(q, k, v, **sdpa_kwargs)

            # (B, num_heads, Tq, head_dim) -> (B, Tq, q_proj_dim)
            attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(B, Tq, self.q_proj_dim)

        # SubLN: normalize over q_proj_dim before the output projection.
        # Matches modeling_han2han_flax.py:1259-1261.
        if self.attn_sub_norm is not None:
            attn_output = self.attn_sub_norm(attn_output)
        attn_output = self.c_proj(attn_output)
        attn_output = self.attn_dropout(attn_output)

        return attn_output, present

    def _build_4d_mask(
        self,
        mask: Optional[torch.Tensor],
        B: int,
        Tq: int,
        Tkv: int,
        causal: bool,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Builds a (B, 1, Tq, Tkv) boolean mask combining padding + causal +
        sliding-window. Returns None when no mask is needed (no padding, no
        windowing, and SDPA's `is_causal` flag can cover the causal case).

        Mirrors modeling_han2han_flax.py:1097-1138 for padding/causal and
        modeling_han2han_flax.py:683-704 / 752-762 for the sliding-window
        semantics:
          causal + sliding:    (diff >= 0) & (diff < W)
          non-causal + sliding: (diff > -W) & (diff < W)       # symmetric
        where diff = (q_idx + q_offset) - kv_idx.
        """
        # Skip window-mask construction when the window already covers everything
        # (W >= max(Tq, Tkv)): the mask is identically True, so building it just
        # forces a redundant SDPA mask path that introduces float-order noise.
        W = self.window_size if (self.is_sliding and self.window_size < max(Tq, Tkv)) else 0

        if self.is_cross_attention:
            # cross-attention mask applies to encoder states (K/V); never causal.
            # Sliding cross-attention (Han2Han uses W around the decoder q position)
            # is symmetric per Flax _build_splash_mask cross branch.
            window_mask = None
            if W > 0:
                q_idx = torch.arange(Tq, device=device)[None, None, :, None]
                kv_idx = torch.arange(Tkv, device=device)[None, None, None, :]
                diff = q_idx - kv_idx
                window_mask = (diff > -W) & (diff < W)  # (1, 1, Tq, Tkv)

            if mask is None and window_mask is None:
                return None
            if mask is not None:
                if mask.dim() == 4:
                    pad = mask.bool()
                elif mask.dim() == 2:
                    pad = mask.bool()[:, None, None, :]
                else:
                    raise ValueError(
                        f"Unsupported encoder_attention_mask ndim={mask.dim()} for cross-attention"
                    )
            else:
                pad = None

            if pad is not None and window_mask is not None:
                return (pad & window_mask).contiguous()
            if pad is not None:
                return pad.contiguous()
            return window_mask.expand(B, 1, Tq, Tkv).contiguous()

        # self-attention
        if mask is not None and mask.dim() == 4:
            base = mask.bool().contiguous()
            if W == 0:
                return base
            q_idx = torch.arange(Tq, device=device)[None, None, :, None]
            kv_idx = torch.arange(Tkv, device=device)[None, None, None, :]
            q_offset = Tkv - Tq
            diff = (q_idx + q_offset) - kv_idx
            if causal:
                window_mask = (diff >= 0) & (diff < W)
            else:
                window_mask = (diff > -W) & (diff < W)
            return (base & window_mask).contiguous()

        if mask is not None and mask.dim() == 2:
            # (B, Tkv_or_Tq). Two regimes: padding mask covers the kv length (which
            # for cache decoding can exceed Tq), or covers Tq with no past.
            if mask.shape[1] >= Tkv:
                padding_mask_kv = mask[:, :Tkv].bool()
            else:
                # left-pad with True for the cached portion if mask only covers new tokens.
                pad = mask.new_ones(B, Tkv - mask.shape[1])
                padding_mask_kv = torch.cat([pad, mask], dim=1).bool()
            padding_mask = padding_mask_kv[:, None, None, :]  # (B, 1, 1, Tkv)
        else:
            padding_mask = None

        # build (causal +/- sliding) structural mask
        structural = None
        if causal or W > 0:
            q_idx = torch.arange(Tq, device=device)[None, None, :, None]
            kv_idx = torch.arange(Tkv, device=device)[None, None, None, :]
            q_offset = Tkv - Tq
            diff = (q_idx + q_offset) - kv_idx
            if causal and W > 0:
                structural = (diff >= 0) & (diff < W)
            elif causal:
                structural = (diff >= 0)
            else:
                # non-causal encoder sliding: symmetric window
                structural = (diff > -W) & (diff < W)

        if structural is not None and padding_mask is not None:
            return (padding_mask & structural).contiguous()
        if structural is not None:
            # Pure causal w/ no padding and no window: defer to SDPA is_causal.
            # Pure windowed (causal or not): build it explicitly.
            if causal and W == 0:
                return None
            return structural.expand(B, 1, Tq, Tkv).contiguous()
        if padding_mask is None:
            return None
        return padding_mask.expand(B, 1, Tq, Tkv).contiguous()


class Han2HanMLP(nn.Module):
    def __init__(self, config: Han2HanConfig):
        super().__init__()
        inner_dim = config.d_ff if config.d_ff is not None else 4 * config.d_model

        act_name = config.ffn_activation
        if act_name == "gelu_new":
            act_name = "gelu"

        self.use_gated = act_name in ("swiglu", "geglu", "reglu2")

        if config.use_fla_fused_mlp and act_name in ["swish", "swiglu"]:
            self.use_fla = True
            _import_fla_modules()
            self.fla_mlp = GatedMLP(
                hidden_size=config.d_model,
                intermediate_size=inner_dim,
                hidden_act="swish",
                fuse_swiglu=True
            )
        else:
            self.use_fla = False

            if self.use_gated:
                # nnx.gelu defaults to approximate=True (tanh approximation), but
                # torch's F.gelu defaults to approximate='none' (exact erfc). Pin
                # the tanh approximation here so geglu in PyTorch matches Flax.
                # reglu2 applies relu(.)**2 to the full wi_0 projection (incl bias),
                # matching FlaxHan2HanMLP (modeling_han2han_flax.py:1413-1414).
                if act_name == "swiglu":
                    self.gate_act = F.silu
                elif act_name == "reglu2":
                    self.gate_act = lambda x: F.relu(x) ** 2
                else:  # geglu
                    self.gate_act = lambda x: F.gelu(x, approximate='tanh')
                self.wi_0 = FlaxLinear(config.d_model, inner_dim, bias=config.use_bias, std=config.initializer_range)
                self.wi_1 = FlaxLinear(config.d_model, inner_dim, bias=config.use_bias, std=config.initializer_range)
                self.wo = FlaxLinear(inner_dim, config.d_model, bias=config.use_bias, std=config.initializer_range)
            else:
                self.c_fc = FlaxLinear(config.d_model, inner_dim, bias=config.use_bias, std=config.initializer_range)
                if act_name == "elu":
                    self.act_fn = lambda hidden_states: F.elu(hidden_states, alpha=1) + 1
                else:
                    self.act_fn = ACT2FN[act_name]
                self.c_proj = FlaxLinear(inner_dim, config.d_model, bias=config.use_bias, std=config.initializer_range)

            # SubLN: RMSNorm over inner_dim, applied between the gated activation
            # (or activation, for non-gated) and the output projection. Mirrors
            # modeling_han2han_flax.py:1402-1406, which passes
            # eps=config.layer_norm_epsilon.
            self.sub_norm = None
            if getattr(config, 'use_sub_ln', False):
                self.sub_norm = RMSNorm(
                    inner_dim, eps=config.layer_norm_epsilon,
                    use_fla_fused=config.use_fla_fused_norm,
                )

            self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_fla:
            return self.fla_mlp(hidden_states)

        if self.use_gated:
            hidden_states = self.gate_act(self.wi_0(hidden_states)) * self.wi_1(hidden_states)
            if self.sub_norm is not None:
                hidden_states = self.sub_norm(hidden_states)
            hidden_states = self.wo(hidden_states)
        else:
            hidden_states = self.c_fc(hidden_states)
            hidden_states = self.act_fn(hidden_states)
            if self.sub_norm is not None:
                hidden_states = self.sub_norm(hidden_states)
            hidden_states = self.c_proj(hidden_states)

        hidden_states = self.dropout(hidden_states) if self.training else hidden_states
        return hidden_states


class Han2HanBlock(nn.Module):
    """Transformer block, matching FlaxHan2HanBlock."""
    def __init__(
        self,
        config: Han2HanConfig,
        is_encoder: bool = False,
        attention_type: Optional[str] = None,
        cross_attention_type: Optional[str] = None,
    ):
        super().__init__()
        self.is_encoder = is_encoder

        # Per-layer effective RoPE theta: sliding/local layers use rope_theta_sliding
        # if set (Gemma 2/3 / T5Gemma 2 style hybrid attention). Mirrors
        # modeling_han2han_flax.py:1480-1485. Legacy checkpoints land here with
        # rope_theta == rope_theta_sliding (rewritten by
        # Han2HanConfig._apply_legacy_rope_quirk), so the per-layer selection
        # collapses to a single value -- matching pre-fix Flax behavior. New
        # checkpoints can set distinct values for true hybrid attention.
        self.attention_type = attention_type or config.attention_mechanism
        self.cross_attention_type = cross_attention_type
        is_sliding_self = ('sliding' in self.attention_type) or ('local' in self.attention_type)
        rope_theta_sliding = getattr(config, 'rope_theta_sliding', None)
        effective_rope_theta = (
            rope_theta_sliding
            if is_sliding_self and rope_theta_sliding is not None
            else config.rope_theta
        )
        self.effective_rope_theta = effective_rope_theta

        self.ln_1 = RMSNorm(config.d_model, eps=config.layer_norm_epsilon, use_fla_fused=config.use_fla_fused_norm)
        self.attn = Han2HanAttention(
            config,
            is_causal=not is_encoder,
            rope_theta=effective_rope_theta,
            attention_type=self.attention_type,
        )

        add_cross_attention = not is_encoder
        if add_cross_attention:
            # cross-attention skips RoPE entirely (MHA), but the cross_attention_type
            # carries 'mha' vs 'mha-sliding' so the sliding-window mask path engages
            # when configured (Han2Han uses a symmetric window around the decoder q).
            self.crossattention = Han2HanAttention(
                config,
                is_cross_attention=True,
                is_causal=False,
                attention_type=self.cross_attention_type,
            )
            self.ln_cross_attn = RMSNorm(config.d_model, eps=config.layer_norm_epsilon, use_fla_fused=config.use_fla_fused_norm)
            self.ca_gate = None
            if hasattr(config, 'gated_cross_attention') and config.gated_cross_attention:
                self.ca_gate = nn.Parameter(torch.tensor(0.7))
                self._ca_reg = None

        self.ln_2 = RMSNorm(config.d_model, eps=config.layer_norm_epsilon, use_fla_fused=config.use_fla_fused_norm)
        self.mlp = Han2HanMLP(config)

        self.block_type = "encoder_block" if is_encoder else "decoder_block"

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        init_cache: bool = False,
        output_attentions: bool = False,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)

        attn_output, present = self.attn(
            hidden_states,
            attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            init_cache,
            output_attentions,
            past_key_value,
        )
        hidden_states = attn_output + residual

        if not self.is_encoder and encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)

            cross_attn_output, _ = self.crossattention(
                hidden_states,
                attention_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                init_cache,
                output_attentions,
                past_key_value=None, # no caching for cross attn
            )
            if self.ca_gate is not None:
                cross_attn_output = self.ca_gate * cross_attn_output
                self._ca_reg = 1e-3 * (1.0 - self.ca_gate).pow(2)
            hidden_states = cross_attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)

        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = residual + feed_forward_hidden_states

        return hidden_states, present


class Han2HanBlockCollection(nn.Module):
    """Collection of transformer blocks, matching FlaxHan2HanBlockCollection."""
    def __init__(self, config: Han2HanConfig, is_encoder: bool = False):
        super().__init__()
        n_layers = config.encoder_nlayer if is_encoder else config.decoder_nlayer

        # per-layer attention types -- enables hybrid sliding/full schedules with
        # rope_theta vs. rope_theta_sliding selection (Gemma 2/3 pattern).
        # Use the config's pattern-expanding accessors so short repeating patterns
        # (e.g., V1 encoder ['mha-sliding']*5+['mha'] for 18 layers) broadcast correctly.
        # The legacy Flax `_scan_key` MHA-collapse quirk is handled upstream by
        # Han2HanConfig._apply_legacy_rope_quirk: legacy checkpoints land here with
        # rope_theta == rope_theta_sliding, so per-layer selection becomes uniform.
        if is_encoder:
            attention_types = config.get_encoder_attention_types()
            cross_types = [None] * n_layers
        else:
            attention_types = config.get_decoder_attention_types()
            cross_types = config.get_decoder_cross_attention_types()

        self.layers = nn.ModuleList([
            Han2HanBlock(
                config,
                is_encoder,
                attention_type=attention_types[i],
                cross_attention_type=cross_types[i],
            )
            for i in range(n_layers)
        ])
        self.layerdrop = config.layer_pdrop
        self.is_encoder = is_encoder
        self.n_layers = n_layers

        # mark the middle decoder layer's cross-attention for weight capture
        if not is_encoder and n_layers > 0:
            middle_idx = n_layers // 2
            if hasattr(self.layers[middle_idx], 'crossattention'):
                self.layers[middle_idx].crossattention.is_middle_layer = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        past_key_values: Optional[Tuple] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_attentions = None
        all_cross_attentions = None
        init_cache = use_cache and (past_key_values is None)
        presents = () if use_cache else None

        for layer_idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.training and self.layerdrop > 0:
                if torch.distributed.is_initialized():
                    drop_layer = torch.rand(1, device=hidden_states.device)
                    torch.distributed.broadcast(drop_layer, src=0)
                    if drop_layer.item() < self.layerdrop:
                        continue
                else:
                    if torch.rand(1).item() < self.layerdrop:
                        continue

            layer_past = past_key_values[layer_idx] if past_key_values is not None else None

            hidden_states, present = layer(
                hidden_states,
                attention_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                init_cache,
                output_attentions,
                past_key_value=layer_past,
            )

            if present is not None:
                presents += (present,)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in (hidden_states, presents, all_hidden_states, all_attentions, all_cross_attentions) if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents if presents else None,
            hidden_states=all_hidden_states if output_hidden_states else None,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions
        )


class Han2HanPreTrainedModel(PreTrainedModel):
    config_class = Han2HanConfig
    supports_gradient_checkpointing = True
    # These buffers are expected and will be loaded by our custom from_pretrained
    _keys_to_ignore_on_load_unexpected = [
        r"encoder\.jbu", r"encoder\.cbu",
        r"decoder\.jbu", r"decoder\.cbu"
    ]
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_tied_weights_keys(self):
        """Initialize tied weights mapping: {secondary: canonical}.

        Canonical weight is decoder.wte.weight for word embeddings,
        decoder.wje/wce.weight for subtoken embeddings, and the decoder
        layer's attention/MLP kernels for tie_encoder_decoder=True.

        NEVER tied (regardless of flag): biases, RMSNorm scales
        (ln_1/ln_2/ln_cross_attn/attn_sub_norm/sub_norm/q_norm/k_norm),
        subword_proj, ln_emb, cross-attention. See
        memory/pytorch_subword_proj_tying_bug.md for context.
        """
        tied = {}

        if self.config.tie_word_embeddings:
            tied["encoder.wte.weight"] = "decoder.wte.weight"

        if getattr(self.config, 'tie_subtoken_embeddings', False):
            if self.config.jamo_subwords:
                tied["encoder.wje.weight"] = "decoder.wje.weight"
            if self.config.char_subwords:
                tied["encoder.wce.weight"] = "decoder.wce.weight"

        if getattr(self.config, 'tie_input_output_embeddings', False):
            tied["lm_head.weight"] = "decoder.wte.weight"

        if getattr(self.config, 'tie_encoder_decoder', False):
            if (getattr(self.config, 'tie_encoder_decoder_experts_only', False)
                    or getattr(self.config, 'tie_encoder_decoder_except_experts', False)):
                raise NotImplementedError(
                    "tie_encoder_decoder_experts_only / tie_encoder_decoder_except_experts "
                    "are MoE-only and deferred until the MoE PyTorch port lands."
                )
            n = min(self.config.encoder_nlayer, self.config.decoder_nlayer)
            for i in range(n):
                base_enc = f"encoder.h.layers.{i}"
                base_dec = f"decoder.h.layers.{i}"
                tied[f"{base_enc}.attn.query.weight"] = f"{base_dec}.attn.query.weight"
                tied[f"{base_enc}.attn.key.weight"]   = f"{base_dec}.attn.key.weight"
                tied[f"{base_enc}.attn.value.weight"] = f"{base_dec}.attn.value.weight"
                tied[f"{base_enc}.attn.c_proj.weight"] = f"{base_dec}.attn.c_proj.weight"
                # dense MLP kernels: gated FFNs (swiglu/geglu/reglu2) use the
                # wi_0/wi_1/wo naming; non-gated use c_fc/c_proj. Must match
                # Han2HanMLP.use_gated so the declared tied keys line up with the
                # actually-shared tensors (HF save_pretrained validates this).
                if self.config.ffn_activation in ('swiglu', 'geglu', 'reglu2'):
                    tied[f"{base_enc}.mlp.wi_0.weight"] = f"{base_dec}.mlp.wi_0.weight"
                    tied[f"{base_enc}.mlp.wi_1.weight"] = f"{base_dec}.mlp.wi_1.weight"
                    tied[f"{base_enc}.mlp.wo.weight"]   = f"{base_dec}.mlp.wo.weight"
                else:
                    tied[f"{base_enc}.mlp.c_fc.weight"]   = f"{base_dec}.mlp.c_fc.weight"
                    tied[f"{base_enc}.mlp.c_proj.weight"] = f"{base_dec}.mlp.c_proj.weight"

        return tied

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Override from_pretrained to properly load jbu/cbu buffers onto encoder/decoder."""
        import os
        from safetensors import safe_open

        # Load model normally
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        # get base model (handles both Han2Han and Han2HanForXXX classes)
        base_model = model.model if hasattr(model, 'model') else model

        # Manually load jbu/cbu buffers from safetensors since HF doesn't load buffers properly
        if not hasattr(base_model.encoder, 'jbu') or not hasattr(base_model.encoder, 'cbu'):
            # Find the safetensors file
            if os.path.isdir(pretrained_model_name_or_path):
                safetensors_path = os.path.join(pretrained_model_name_or_path, "model.safetensors")
            else:
                safetensors_path = pretrained_model_name_or_path

            if os.path.exists(safetensors_path):
                with safe_open(safetensors_path, framework="pt") as f:
                    # Load encoder buffers
                    if "encoder.jbu" in f.keys() and not hasattr(base_model.encoder, 'jbu'):
                        base_model.encoder.register_buffer('jbu', f.get_tensor("encoder.jbu"))
                    if "encoder.cbu" in f.keys() and not hasattr(base_model.encoder, 'cbu'):
                        base_model.encoder.register_buffer('cbu', f.get_tensor("encoder.cbu"))

                    # Load decoder buffers
                    if "decoder.jbu" in f.keys() and not hasattr(base_model.decoder, 'jbu'):
                        base_model.decoder.register_buffer('jbu', f.get_tensor("decoder.jbu"))
                    if "decoder.cbu" in f.keys() and not hasattr(base_model.decoder, 'cbu'):
                        base_model.decoder.register_buffer('cbu', f.get_tensor("decoder.cbu"))

        return model

    def set_subword_tables(self, jbu=None, cbu=None):
        """Set the jbu and cbu subword lookup tables directly.

        Args:
            jbu: Array-like object of jamo subword indices or path to jbu.npy file
            cbu: Array-like object of char subword indices or path to cbu.npy file
        """
        import os
        import numpy as np

        if jbu is not None:
            if isinstance(jbu, str) and os.path.exists(jbu):
                jbu = np.load(jbu)
            jbu_tensor = torch.as_tensor(jbu, dtype=torch.long)
            self.register_buffer("jbu", jbu_tensor)

        if cbu is not None:
            if isinstance(cbu, str) and os.path.exists(cbu):
                cbu = np.load(cbu)
            cbu_tensor = torch.as_tensor(cbu, dtype=torch.long)
            self.register_buffer("cbu", cbu_tensor)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """Override to register jbu/cbu buffers when loading from state dict."""

        # Check for buffers with current prefix (for base model)
        if f"{prefix}jbu" in state_dict and not hasattr(self, 'jbu'):
            self.register_buffer('jbu', state_dict[f"{prefix}jbu"])
        if f"{prefix}cbu" in state_dict and not hasattr(self, 'cbu'):
            self.register_buffer('cbu', state_dict[f"{prefix}cbu"])

        # CRITICAL: Also register buffers on encoder/decoder modules if they exist
        # Buffers are saved as "encoder.jbu", "decoder.cbu" etc in the state dict
        if hasattr(self, 'encoder') and isinstance(self.encoder, Han2HanModule):
            if f"{prefix}encoder.jbu" in state_dict and not hasattr(self.encoder, 'jbu'):
                self.encoder.register_buffer('jbu', state_dict[f"{prefix}encoder.jbu"])
            if f"{prefix}encoder.cbu" in state_dict and not hasattr(self.encoder, 'cbu'):
                self.encoder.register_buffer('cbu', state_dict[f"{prefix}encoder.cbu"])

        if hasattr(self, 'decoder') and isinstance(self.decoder, Han2HanModule):
            if f"{prefix}decoder.jbu" in state_dict and not hasattr(self.decoder, 'jbu'):
                self.decoder.register_buffer('jbu', state_dict[f"{prefix}decoder.jbu"])
            if f"{prefix}decoder.cbu" in state_dict and not hasattr(self.decoder, 'cbu'):
                self.decoder.register_buffer('cbu', state_dict[f"{prefix}decoder.cbu"])

        # Call parent to actually load the state
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def _adjust_bias(self, output_embeddings, input_embeddings):
        """Override HF's _adjust_bias to skip FlaxLinear modules.

        HF's default assumes PyTorch-convention weights `(out, in)` and resizes
        the target's bias to `weight.shape[0]`. Our FlaxLinear stores weights
        as `(in, out)`, so shape[0] is in_features, which would truncate/pad
        the bias to the wrong size. For FlaxLinear, the bias must keep its
        original `(out_features,)` shape (and is tied or not independently of
        the weight tying — biases are NEVER tied under tie_encoder_decoder).
        """
        if isinstance(output_embeddings, FlaxLinear):
            return
        return super()._adjust_bias(output_embeddings, input_embeddings)

    def _init_weights(self, module):
        """Initialize the weights.

        Uses `torch.nn.init.*` functions (not tensor `.data.normal_()` methods)
        so HF v5's `guard_torch_init_functions` context manager can honor the
        `_is_hf_initialized` flag set by `mark_tied_weights_as_initialized`.
        Without this, `_initialize_missing_keys` would overwrite loaded tied
        masters (e.g. decoder.wte) when re-initing their secondary tied keys
        (encoder.wte, lm_head) because the secondaries share storage with the
        master.
        """
        from torch.nn import init
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv1d, FlaxLinear)):
            init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None and not getattr(module.weight, "_is_hf_initialized", False):
                init.zeros_(module.weight[module.padding_idx])
        else:
            norm_classes = [nn.LayerNorm, RMSNorm]
            if FLARMSNorm is not None:
                norm_classes.append(FLARMSNorm)
            if isinstance(module, tuple(norm_classes)):
                if getattr(module, 'weight', None) is not None:
                    init.ones_(module.weight)
                if getattr(module, 'bias', None) is not None:
                    init.zeros_(module.bias)

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None) -> nn.Embedding:
        """
        Resize the model's token embeddings to handle new tokens (e.g., special tokens for RE tasks).
        Uses padding token initialization for new tokens in jbu/cbu arrays.
        """
        if new_num_tokens is None:
            return self.get_input_embeddings()

        old_num_tokens = self.config.vocab_size

        if new_num_tokens == old_num_tokens:
            return self.get_input_embeddings()

        # Handle different model architectures
        # For models with separate encoder/decoder
        if hasattr(self, 'encoder') and hasattr(self.encoder, 'wte'):
            new_embeddings = self._resize_token_embeddings(self.encoder.wte, new_num_tokens)
            self.encoder.wte = new_embeddings

            if hasattr(self, 'decoder') and hasattr(self.decoder, 'wte'):
                new_decoder_embeddings = self._resize_token_embeddings(self.decoder.wte, new_num_tokens)
                self.decoder.wte = new_decoder_embeddings

        # for models that wrap a Han2Han model
        elif hasattr(self, 'model') and hasattr(self.model, 'encoder'):
            if hasattr(self.model.encoder, 'wte'):
                new_embeddings = self._resize_token_embeddings(self.model.encoder.wte, new_num_tokens)
                self.model.encoder.wte = new_embeddings

            if hasattr(self.model, 'decoder') and hasattr(self.model.decoder, 'wte'):
                new_decoder_embeddings = self._resize_token_embeddings(self.model.decoder.wte, new_num_tokens)
                self.model.decoder.wte = new_decoder_embeddings

        # Resize jbu and cbu arrays if they exist
        if hasattr(self.encoder, 'jbu') and self.encoder.jbu is not None:
            self.encoder.jbu = self._resize_subword_table(self.encoder.jbu, old_num_tokens, new_num_tokens)

        if hasattr(self.encoder, 'cbu') and self.encoder.cbu is not None:
            self.encoder.cbu = self._resize_subword_table(self.encoder.cbu, old_num_tokens, new_num_tokens)

        # Resize jbu and cbu arrays if they exist
        if hasattr(self.decoder, 'jbu') and self.decoder.jbu is not None:
            self.decoder.jbu = self._resize_subword_table(self.decoder.jbu, old_num_tokens, new_num_tokens)

        if hasattr(self.decoder, 'cbu') and self.decoder.cbu is not None:
            self.decoder.cbu = self._resize_subword_table(self.decoder.cbu, old_num_tokens, new_num_tokens)

        # update config
        self.config.vocab_size = new_num_tokens

        # update LM head if it exists
        if hasattr(self, 'lm_head'):
            self.lm_head = self._resize_lm_head(self.lm_head, new_num_tokens)
        elif hasattr(self, 'model') and hasattr(self.model, 'lm_head'):
            self.model.lm_head = self._resize_lm_head(self.model.lm_head, new_num_tokens)

        # update other output layers
        if hasattr(self, 'qa_outputs'):
            self.qa_outputs = self._resize_lm_head(self.qa_outputs, new_num_tokens)

        # tie weights if necessary
        if hasattr(self, '_tie_weights'):
            self._tie_weights()
        elif hasattr(self, 'model') and hasattr(self.model, '_tie_weights'):
            self.model._tie_weights()

        return self.get_input_embeddings()

    def _resize_subword_table(self, old_table: torch.Tensor, old_num_tokens: int, new_num_tokens: int) -> torch.Tensor:
        """Resize jbu or cbu subword lookup table by extending with padding token's bucket indices."""
        if new_num_tokens == old_num_tokens:
            return old_table

        # Get the padding token's bucket index
        pad_token_id = self.config.pad_token_id if hasattr(self.config, 'pad_token_id') else 0

        # use the padding token's bucket indices for new tokens
        # jbu/cbu are 2D: [vocab_size, num_buckets] where each token has multiple bucket indices
        if pad_token_id < old_table.size(0):
            # Get the padding token's bucket indices (a 1D tensor of bucket indices)
            pad_bucket_values = old_table[pad_token_id]  # Shape: [num_buckets]
        else:
            # Use zeros if pad token not in range
            pad_bucket_values = torch.zeros(old_table.size(1), dtype=old_table.dtype, device=old_table.device)

        # create new table with extended size
        with torch.no_grad():
            if new_num_tokens > old_num_tokens:
                # Extend the table - repeat pad_bucket_values for each new token
                num_new_tokens = new_num_tokens - old_num_tokens
                extension = pad_bucket_values.unsqueeze(0).expand(num_new_tokens, -1)  # Shape: [num_new_tokens, num_buckets]
                new_table = torch.cat([old_table, extension], dim=0)
            else:
                # shrink the table
                new_table = old_table[:new_num_tokens]

        # register as buffer to maintain the same status as before
        return new_table

    def _resize_token_embeddings(self, old_embeddings: nn.Embedding, new_num_tokens: int) -> nn.Embedding:
        """Resize token embeddings using padding token initialization for new tokens."""
        old_num_tokens = old_embeddings.num_embeddings

        if new_num_tokens == old_num_tokens:
            return old_embeddings

        new_embeddings = nn.Embedding(new_num_tokens, old_embeddings.embedding_dim)
        new_embeddings.to(old_embeddings.weight.device, dtype=old_embeddings.weight.dtype)

        # Copy old weights
        with torch.no_grad():
            new_embeddings.weight[:old_num_tokens] = old_embeddings.weight

            # Initialize new tokens with padding token embedding if it exists
            if hasattr(self.config, 'pad_token_id') and self.config.pad_token_id is not None:
                pad_token_id = self.config.pad_token_id
                if pad_token_id < old_num_tokens:
                    pad_embedding = old_embeddings.weight[pad_token_id].clone()
                    new_embeddings.weight[old_num_tokens:] = pad_embedding
                else:
                    # Fallback to random initialization
                    nn.init.normal_(new_embeddings.weight[old_num_tokens:], std=self.config.initializer_range)
            else:
                # Fallback to random initialization
                nn.init.normal_(new_embeddings.weight[old_num_tokens:], std=self.config.initializer_range)

        return new_embeddings

    def _resize_lm_head(self, old_lm_head: nn.Linear, new_num_tokens: int) -> nn.Linear:
        """Resize LM head for new vocabulary size."""
        old_num_tokens = old_lm_head.out_features

        if new_num_tokens == old_num_tokens:
            return old_lm_head

        new_lm_head = nn.Linear(old_lm_head.in_features, new_num_tokens, bias=old_lm_head.bias is not None)
        new_lm_head.to(old_lm_head.weight.device, dtype=old_lm_head.weight.dtype)

        # copy old weights
        with torch.no_grad():
            new_lm_head.weight[:old_num_tokens] = old_lm_head.weight[:old_num_tokens]

            # initialize new weights
            if hasattr(self.config, 'pad_token_id') and self.config.pad_token_id is not None:
                pad_token_id = self.config.pad_token_id
                if pad_token_id < old_num_tokens:
                    pad_weight = old_lm_head.weight[pad_token_id].clone()
                    new_lm_head.weight[old_num_tokens:] = pad_weight
                else:
                    nn.init.normal_(new_lm_head.weight[old_num_tokens:], std=self.config.initializer_range)
            else:
                nn.init.normal_(new_lm_head.weight[old_num_tokens:], std=self.config.initializer_range)

            if old_lm_head.bias is not None:
                new_lm_head.bias[:old_num_tokens] = old_lm_head.bias[:old_num_tokens]
                new_lm_head.bias[old_num_tokens:] = 0

        return new_lm_head

    @property
    def dummy_inputs(self):
        pad_token = self.config.pad_token_id
        input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]], device=self.device)
        dummy_inputs = {
            "attention_mask": input_ids.ne(pad_token),
            "input_ids": input_ids,
        }
        return dummy_inputs


class Han2HanModule(Han2HanPreTrainedModel):
    """Base module for encoder/decoder, matching FlaxHan2HanModule."""
    def __init__(self, config: Han2HanConfig, is_encoder: bool = False):
        super().__init__(config)
        self.is_encoder = is_encoder

        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        nn.init.normal_(self.wte.weight, std=config.initializer_range)

        # initialize subword embeddings and fusion layers (matching Flax)
        # subword_embed_dim defaults to d_model // 2 in han2han_config when unset
        subword_embed_dim = getattr(config, 'subword_embed_dim', None) or (config.d_model // 2)
        self.wje = None
        self.wce = None
        if config.jamo_subwords or config.char_subwords:
            self.subword_proj = FlaxLinear(
                in_features=subword_embed_dim,
                out_features=config.d_model,
                bias=False,
                std=config.initializer_range,
            )
            self.ln_emb = RMSNorm(config.d_model, eps=config.layer_norm_epsilon, use_fla_fused=config.use_fla_fused_norm)

        if config.jamo_subwords:
            self.wje = nn.Embedding(config.jamo_vocab_size, subword_embed_dim)
            nn.init.normal_(self.wje.weight, std=config.initializer_range)

        if config.char_subwords:
            self.wce = nn.Embedding(config.char_vocab_size, subword_embed_dim)
            nn.init.normal_(self.wce.weight, std=config.initializer_range)

        self.subword_embed_dim = subword_embed_dim

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = Han2HanBlockCollection(config, is_encoder)
        self.ln_f = RMSNorm(config.d_model, eps=config.layer_norm_epsilon, use_fla_fused=config.use_fla_fused_norm)

    def tie_embeddings_and_encoder_decoder(self, *args, **kwargs):
        pass

    def forward(
        self,
        input_ids: torch.Tensor,
        jamo_input_ids: Optional[torch.Tensor] = None,
        char_input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        use_cache: bool = False,
        past_key_values: Optional[Tuple] = None,
        return_dict: bool = True,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        if past_key_values is None:
            past_key_values = tuple([None] * (self.config.encoder_nlayer if self.is_encoder else self.config.decoder_nlayer))

        if attention_mask is None and input_ids is not None:
            attention_mask = input_ids.ne(self.config.pad_token_id)

        # standard token embeddings
        wte_embeds = self.wte(input_ids)

        # decoder-side sqrt(d_model) scaling stabilizes attention scores during
        # generation; encoder side is left unscaled (matches Flax behavior at
        # modeling_han2han_flax.py:2738-2740).
        if not self.is_encoder:
            normalizer = (self.config.d_model ** 0.5)
            wte_embeds = wte_embeds * wte_embeds.new_tensor(normalizer)

        # === CONDITIONAL SUBWORD EMBEDDINGS ===
        if hasattr(self, 'subword_proj'):
            subword_features = torch.zeros(
                (*wte_embeds.shape[:2], self.subword_embed_dim),
                device=wte_embeds.device,
                dtype=wte_embeds.dtype,
            )

            # per-embedding-type dropout with 2.0/num_active rescale (Flax 2750-2807).
            # active only during training; inference path is a no-op.
            embedding_dropout_rate = getattr(self.config, 'embedding_dropout_rate', 0.0)
            char_is_unified_cjk = getattr(self.config, 'char_is_unified_cjk', False)
            apply_emb_dropout = self.training and embedding_dropout_rate > 0.0
            if apply_emb_dropout:
                wte_wje_pdrop = 0.0 if char_is_unified_cjk else embedding_dropout_rate
                wte_keep = float(torch.bernoulli(torch.tensor(1.0 - wte_wje_pdrop)).item())
                wje_keep = float(torch.bernoulli(torch.tensor(1.0 - wte_wje_pdrop)).item())
                wce_keep = float(torch.bernoulli(torch.tensor(1.0 - embedding_dropout_rate)).item())
                # ensure at least one embedding type survives
                if wte_keep + wje_keep + wce_keep == 0:
                    survivor = int(torch.randint(0, 3, ()).item())
                    wte_keep = 1.0 if survivor == 0 else wte_keep
                    wje_keep = 1.0 if survivor == 1 else wje_keep
                    wce_keep = 1.0 if survivor == 2 else wce_keep
            else:
                wte_keep = wje_keep = wce_keep = 1.0

            num_active = wte_keep + wje_keep + wce_keep

            if self.wje is not None:
                if jamo_input_ids is None:
                    if not hasattr(self, 'jbu'):
                        raise ValueError("`jamo_input_ids` not provided and `jbu` lookup buffer is missing.")
                    jamo_input_ids = self.jbu[input_ids.long()].long()
                jamo_embeds = self.wje(jamo_input_ids)
                jamo_embeds = jamo_embeds.sum(dim=-2)  # pooling over ngrams
                subword_features = subword_features + jamo_embeds * wje_keep

            if self.wce is not None:
                if char_input_ids is None:
                    if not hasattr(self, 'cbu'):
                        raise ValueError("`char_input_ids` not provided and `cbu` lookup buffer is missing.")
                    char_input_ids = self.cbu[input_ids.long()].long()
                char_embeds = self.wce(char_input_ids)
                char_embeds = char_embeds.sum(dim=-2)
                subword_features = subword_features + char_embeds * wce_keep

            scale_factor = (2.0 / max(num_active, 1.0)) if num_active > 0 else 1.0
            subword_features = subword_features * scale_factor

            self_activated = F.silu(subword_features) * subword_features  # element-wise self-gating
            projected_subwords = self.subword_proj(self_activated)
            # gated fusion: tokens + projected, self-gated subwords
            hidden_states = wte_embeds + projected_subwords
            hidden_states = self.ln_emb(hidden_states)

        else:
            hidden_states = wte_embeds

        # === END CONDITIONAL SUBWORD EMBEDDINGS ===

        hidden_states = self.drop(hidden_states) if self.training else hidden_states

        outputs = self.h(
            hidden_states,
            attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
            past_key_values,
        )

        if not return_dict:
            hidden_states = outputs[0]
        else:
            hidden_states = outputs.last_hidden_state

        hidden_states = self.ln_f(hidden_states)

        if output_hidden_states and not return_dict:
            outputs = (hidden_states,) + (outputs[1] + (hidden_states,),) + outputs[2:]
        elif output_hidden_states:
            outputs.hidden_states = outputs.hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions
        )


class Han2Han(Han2HanPreTrainedModel, GenerationMixin):
    """Main Han2Han model, matching FlaxHan2Han, with HF compatibility for KLUE evaluation."""
    # Inherit buffer handling from parent class
    _keys_to_ignore_on_load_unexpected = [
        r"encoder\.jbu", r"encoder\.cbu",
        r"decoder\.jbu", r"decoder\.cbu"
    ]

    def __init__(self, config: Han2HanConfig, char_buckets=None, jamo_buckets=None):
        super().__init__(config)

        # this port supports the dense bf16 MHA path only; raise loudly rather than
        # silently dropping into unsupported configurations.
        if getattr(config, 'encoder_num_sparse', 0) > 0 or getattr(config, 'decoder_num_sparse', 0) > 0:
            raise NotImplementedError(
                "sparse (MoE) layers are not supported; this model is dense FFN only. "
                "Set encoder_num_sparse=decoder_num_sparse=0."
            )

        self.gradient_checkpointing = None

        self.encoder = Han2HanModule(config, is_encoder=True)
        self.decoder = Han2HanModule(config, is_encoder=False)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self._tied_weights_keys = self._init_tied_weights_keys()
        self.post_init()

        if char_buckets is not None:
            self.encoder.cbu = self.decoder.cbu = torch.tensor(char_buckets.copy())
            self.register_buffer('cbu', self.encoder.cbu)
        if jamo_buckets is not None:
            self.encoder.jbu = self.decoder.jbu = torch.tensor(jamo_buckets.copy())
            self.register_buffer('jbu', self.encoder.jbu)

    def tie_weights(self, missing_keys=None, *args, **kwargs):
        """Override HF's generic tie_weights.

        HF's ``PreTrainedModel.tie_weights`` ties by name-matching every encoder
        submodule to its decoder twin when ``config.tie_encoder_decoder=True`` +
        ``is_encoder_decoder=True`` (older transformers) and over-ties the
        separately-trained subword tables (wje/wce/subword_proj/ln_emb), which
        Flax keeps untied unless ``tie_subtoken_embeddings``. That clobbers
        encoder.wje with decoder.wje and collapses generation (see
        memory/pytorch_subword_proj_tying_bug.md). Han2Han's tying is fully and
        precisely expressed by ``_tie_weights`` / ``_tie_encoder_decoder_blocks``,
        so route exclusively through those.

        The load path calls ``tie_weights(missing_keys=load_info.missing_keys)``
        and relies on the native implementation to discard tied targets from the
        missing set (so they are not mis-reported as "newly initialized"). Since
        we bypass the native path, replicate that reconciliation here: a tied
        target is populated by tying as long as its source is present.
        """
        self._tie_weights()
        if missing_keys is not None:
            for target, source in (self._tied_weights_keys or {}).items():
                if source not in missing_keys:
                    missing_keys.discard(target)

    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self.encoder.wte.weight = self.decoder.wte.weight

        if getattr(self.config, 'tie_subtoken_embeddings', False):
            if self.config.jamo_subwords:
                self.encoder.wje.weight = self.decoder.wje.weight
            if self.config.char_subwords:
                self.encoder.wce.weight = self.decoder.wce.weight

        if getattr(self.config, 'tie_encoder_decoder', False):
            self._tie_encoder_decoder_blocks()

        if getattr(self.config, 'tie_input_output_embeddings', False):
            self.lm_head.weight = self.decoder.wte.weight

    def _tie_encoder_decoder_blocks(self):
        """Tie encoder/decoder attention QKV/output kernels and dense MLP
        kernels per matching layer pair. Dense-only; MoE tying is deferred
        until the MoE PyTorch port lands.

        Mirrors modeling_han2han_flax.py:3210-3370 (_tie_encoder_decoder_blocks
        and _tie_block_pair). NEVER tied: biases, all RMSNorm scales
        (ln_1/ln_2/ln_cross_attn/attn_sub_norm/sub_norm/q_norm/k_norm),
        subword_proj, ln_emb, cross-attention (encoder has none anyway).
        See memory/pytorch_subword_proj_tying_bug.md.
        """
        if (getattr(self.config, 'tie_encoder_decoder_experts_only', False)
                or getattr(self.config, 'tie_encoder_decoder_except_experts', False)):
            raise NotImplementedError(
                "tie_encoder_decoder_experts_only / tie_encoder_decoder_except_experts "
                "are MoE-only and deferred until the MoE PyTorch port lands."
            )

        enc_layers = self.encoder.h.layers
        dec_layers = self.decoder.h.layers
        n = min(len(enc_layers), len(dec_layers))
        for i in range(n):
            self._tie_block_pair(enc_layers[i], dec_layers[i])

    def _tie_block_pair(self, enc_b, dec_b):
        enc_b.attn.query.weight = dec_b.attn.query.weight
        enc_b.attn.key.weight = dec_b.attn.key.weight
        enc_b.attn.value.weight = dec_b.attn.value.weight
        enc_b.attn.c_proj.weight = dec_b.attn.c_proj.weight
        self._tie_dense_mlp(enc_b.mlp, dec_b.mlp)

    def _tie_dense_mlp(self, enc_mlp, dec_mlp):
        if hasattr(dec_mlp, 'wi_0') and hasattr(dec_mlp, 'wi_1'):
            enc_mlp.wi_0.weight = dec_mlp.wi_0.weight
            enc_mlp.wi_1.weight = dec_mlp.wi_1.weight
        elif hasattr(dec_mlp, 'c_fc'):
            enc_mlp.c_fc.weight = dec_mlp.c_fc.weight
        if hasattr(dec_mlp, 'wo'):
            enc_mlp.wo.weight = dec_mlp.wo.weight
        elif hasattr(dec_mlp, 'c_proj'):
            enc_mlp.c_proj.weight = dec_mlp.c_proj.weight

    def get_input_embeddings(self):
        return self.encoder.wte

    def set_input_embeddings(self, value):
        self.encoder.wte = value
        self._tie_weights()

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = True, assign: bool = False):
        """Load a Flax-style state-dict while automatically transposing dense
        layer kernels when necessary and handling tied weights for encoder/decoder.

        Args:
            state_dict: State dictionary to load
            strict: Whether to enforce strict loading (default: True)
            assign: Whether to assign values to meta tensors instead of copying (default: False)
                    Set to True when loading into a model with meta tensors
        """
        current_model_state = self.state_dict()
        flax_state_adjusted = {}

        # first, adjust lm_head if necessary (transpose)
        for k, v in state_dict.items():
            v_adjusted = v
            if k in current_model_state and v.shape != current_model_state[k].shape:
                if v.ndim == 2 and v.T.shape == current_model_state[k].shape:
                    v_adjusted = v.t()
            flax_state_adjusted[k] = v_adjusted

        final_state_to_load = flax_state_adjusted.copy()

        # handle missing decoder.wte.weight when tie_input_output_embeddings=True
        if self.config.tie_input_output_embeddings and "decoder.wte.weight" not in final_state_to_load:
            if "lm_head.weight" in final_state_to_load:
                # must clone to create actual tensor, not just reference
                final_state_to_load["decoder.wte.weight"] = final_state_to_load["lm_head.weight"].clone()

        model_keys = set(current_model_state.keys())
        loaded_keys = set(final_state_to_load.keys())
        potential_missing_tied_keys = model_keys - loaded_keys

        copied_keys = set()
        for missing_key in potential_missing_tied_keys:
            if missing_key.startswith("encoder."):
                source_key_parts = missing_key.split(".")
                source_key_parts[0] = "decoder"
                source_key = ".".join(source_key_parts)

                if source_key in final_state_to_load:
                    should_copy = False
                    if self.config.tie_word_embeddings and any(tied_part in missing_key for tied_part in
                                                               [".wte.weight", ".wje.weight", ".wce.weight",
                                                                ".subword_proj.weight", ".ln_emb.weight"]):
                        should_copy = True
                    elif self.config.tie_encoder_decoder:
                        # when tie_encoder_decoder=True, share everything EXCEPT cross-attention
                        if ".crossattention." not in missing_key and ".ln_cross_attn." not in missing_key:
                            should_copy = True

                    if should_copy:
                        if missing_key in current_model_state and final_state_to_load[source_key].shape == current_model_state[missing_key].shape:
                           final_state_to_load[missing_key] = final_state_to_load[source_key].clone()
                           copied_keys.add(missing_key)

        # load the state dict with all required tensors
        # pass assign=True when dealing with meta tensors
        result = super().load_state_dict(final_state_to_load, strict=strict, assign=assign)

        self._tie_weights()

        return result

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Union[Tuple, BaseModelOutputWithPastAttentionsAndSentenceEmbeddings]] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        use_cache: Optional[bool] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        output_sentence_embeddings: bool = False,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:

        return_dict = return_dict if return_dict is not None else self.config.return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # handle encoder_outputs - if provided from HF's generate(), use them directly
        if encoder_outputs is None:
            if input_ids is None:
                raise ValueError("You have to specify either input_ids or encoder_outputs")

            encoder_jamo_input_ids = None
            encoder_char_input_ids = None

            if attention_mask is None:
                attention_mask = input_ids.ne(self.config.pad_token_id)

            if self.config.jamo_subwords and hasattr(self.encoder, "jbu") and self.encoder.jbu is not None:
                encoder_jamo_input_ids = self.encoder.jbu[input_ids].long()
            if self.config.char_subwords and hasattr(self.encoder, "cbu") and self.encoder.cbu is not None:
                encoder_char_input_ids = self.encoder.cbu[input_ids].long()

            encoder_outputs = self.encoder(
                input_ids,
                encoder_jamo_input_ids,
                encoder_char_input_ids,
                attention_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions,
                output_hidden_states,
                use_cache,
                past_key_values,
                return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutputWithPastAttentionsAndSentenceEmbeddings):
            # Convert tuple to proper output class if needed
            encoder_outputs = BaseModelOutputWithPastAttentionsAndSentenceEmbeddings(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        if not return_dict:
            hidden_states = encoder_outputs[0]
        else:
            hidden_states = encoder_outputs.last_hidden_state

        sentence_embeddings = None

        if output_sentence_embeddings or not self.config.use_bart_training:
            if input_ids is not None:
                input_mask = (input_ids != self.config.pad_token_id).float()
            elif attention_mask is not None:
                input_mask = attention_mask.float()
            else:
                input_mask = torch.ones(hidden_states.shape[:2], dtype=torch.float32, device=hidden_states.device)

            input_mask_expanded = input_mask.unsqueeze(-1)
            sum_embeddings = (hidden_states * input_mask_expanded).sum(dim=1)
            sum_mask = input_mask_expanded.sum(dim=1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            sentence_embeddings = sum_embeddings / sum_mask

            if output_sentence_embeddings:
                return (sentence_embeddings,)

        if hasattr(self, "decoder"):
            if decoder_input_ids is None and input_ids is not None:
                decoder_input_ids = shift_tokens_right(
                    input_ids.clone(), self.config.pad_token_id, self.config.decoder_start_token_id
                )

            if decoder_input_ids is not None:
                decoder_char_input_ids = None
                decoder_jamo_input_ids = None
                if self.config.jamo_subwords and hasattr(self.decoder, "jbu") and self.decoder.jbu is not None:
                    decoder_jamo_input_ids = self.decoder.jbu[decoder_input_ids].long()
                if self.config.char_subwords and hasattr(self.decoder, "cbu") and self.decoder.cbu is not None:
                    decoder_char_input_ids = self.decoder.cbu[decoder_input_ids].long()
                if decoder_attention_mask is None:
                    decoder_attention_mask = decoder_input_ids.ne(self.config.pad_token_id).long()

            # for BART training, use full encoder sequences; for TSDAE, use pooled embeddings
            if self.config.use_bart_training:
                encoder_hidden_for_decoder = hidden_states  # (batch, seq_len, d_model)
                encoder_attention_mask_for_decoder = attention_mask  # (batch, seq_len)
            else:
                encoder_hidden_for_decoder = sentence_embeddings[:, None, :]  # (batch, 1, d_model)
                encoder_attention_mask_for_decoder = attention_mask[:, 0:1]  # (batch, 1)

            decoder_outputs = self.decoder(
                decoder_input_ids,
                decoder_jamo_input_ids,
                decoder_char_input_ids,
                decoder_attention_mask,
                encoder_hidden_for_decoder,
                encoder_attention_mask_for_decoder,
                output_attentions,
                output_hidden_states,
                use_cache,
                past_key_values,
                return_dict,
            )

            if not return_dict:
                hidden_states = decoder_outputs[0]
            else:
                hidden_states = decoder_outputs.last_hidden_state

        if hasattr(self, "lm_head"):
            lm_logits = self.lm_head(hidden_states)

            if not return_dict:
                return (lm_logits,) + encoder_outputs[1:] + decoder_outputs[1:]

            return Seq2SeqLMOutput(
                logits=lm_logits,
                hidden_states=hidden_states,
                sentence_embeddings=sentence_embeddings if not self.config.use_bart_training else None,
                past_key_values=encoder_outputs.past_key_values,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,
                cross_attentions=decoder_outputs.cross_attentions,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs.hidden_states,
                encoder_attentions=encoder_outputs.attentions,
            )
        if not hasattr(self, 'decoder'):
            if not return_dict:
                return (encoder_outputs, sentence_embeddings)

            return BaseModelOutputWithPastAttentionsAndSentenceEmbeddings(
                last_hidden_state=encoder_outputs.last_hidden_state,
                past_key_values=encoder_outputs.past_key_values,
                sentence_embeddings=sentence_embeddings if not self.config.use_bart_training else None,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions
            )
        else:
            if not return_dict:
                return (decoder_outputs, encoder_outputs, sentence_embeddings)

            return Seq2SeqModelOutput(
                last_hidden_state = hidden_states,
                past_key_values = decoder_outputs.past_key_values,
                decoder_hidden_states = decoder_outputs.hidden_states,
                decoder_attentions = decoder_outputs.attentions,
                cross_attentions = decoder_outputs.cross_attentions,
                encoder_last_hidden_state = encoder_outputs.last_hidden_state,
                encoder_hidden_states = encoder_outputs.hidden_states,
                encoder_attentions=encoder_outputs.attentions,
            )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        encoder_outputs=None,
        decoder_attention_mask=None,
        **kwargs
    ):
        """Prepare inputs for generation compatible with HF's generate."""
        # trim decoder_input_ids if past is used
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        # handle decoder_attention_mask
        if decoder_attention_mask is None and decoder_input_ids is not None:
            decoder_attention_mask = decoder_input_ids.ne(self.config.pad_token_id).long()

        return {
            "input_ids": None,  # encoder_outputs is defined, input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "use_cache": kwargs.get("use_cache")
        }

    def _reorder_cache(self, past_key_values, beam_idx):
        """Reorder past key values for beam search."""
        reordered_past = ()
        for layer_past in past_key_values:
            # cached cross_attention states don't need reordering for beam search
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),
            )
        return reordered_past

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        decoder_input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
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
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        length_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        num_return_sequences: int = 1,
        decoder_start_token_id: Optional[int] = None,
        use_cache: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        use_fixed_length_generation: bool = True,
        **model_kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Generate sequences using the Han2Han model with proper KV caching.

        Supports greedy decoding, beam search, and nucleus/top-k sampling.
        Integrates with Han2Han's encoder-decoder architecture and KV caching.

        Args:
            input_ids: input token ids for the encoder
            decoder_input_ids: optional initial decoder token ids to use as a prompt
            attention_mask: attention mask for encoder inputs
            max_length: maximum length of generated sequences
            max_new_tokens: maximum number of new tokens to generate
            min_length: minimum length of generated sequences
            do_sample: whether to use sampling instead of greedy decoding
            early_stopping: whether to stop beam search when num_beams sentences are finished
            num_beams: number of beams for beam search
            temperature: temperature for sampling
            top_k: number of top tokens to consider for top-k sampling
            top_p: cumulative probability for nucleus sampling
            repetition_penalty: penalty for repeated tokens
            bad_words_ids: list of token ids that should not be generated
            force_words_ids: list of token ids that must be generated
            bos_token_id: beginning of sentence token id
            pad_token_id: padding token id
            eos_token_id: end of sentence token id
            length_penalty: exponential penalty to the length for beam search
            no_repeat_ngram_size: size of n-grams that should not be repeated
            encoder_no_repeat_ngram_size: size of encoder n-grams that should not be repeated
            num_return_sequences: number of sequences to return
            decoder_start_token_id: token id to start decoding with
            use_cache: whether to use kv caching
            use_fixed_length_generation: maintain fixed-length sequences during generation (matches training)
            **model_kwargs: additional model arguments

        Returns:
            generated token sequences
        """
        # set default token ids from config
        bos_token_id = bos_token_id if bos_token_id is not None else getattr(self.config, 'bos_token_id', None)
        eos_token_id = eos_token_id if eos_token_id is not None else getattr(self.config, 'eos_token_id', None)
        pad_token_id = pad_token_id if pad_token_id is not None else getattr(self.config, 'pad_token_id', None)
        decoder_start_token_id = decoder_start_token_id if decoder_start_token_id is not None else getattr(self.config, 'decoder_start_token_id', bos_token_id)

        # set default length constraints
        if max_length is None and max_new_tokens is None:
            max_length = getattr(self.config, 'max_length', 512)
        elif max_new_tokens is not None:
            # max_new_tokens specifies additional tokens beyond prompt
            # we need to add prompt length to get total max_length
            if decoder_input_ids is not None:
                prompt_length = decoder_input_ids.shape[-1]
            else:
                prompt_length = 1  # just the start token
            max_length = prompt_length + max_new_tokens

        min_length = min_length if min_length is not None else getattr(self.config, 'min_length', 0)

        batch_size = input_ids.shape[0]
        device = input_ids.device

        # handle fixed-length generation if enabled
        if use_fixed_length_generation:
            # pad encoder inputs to max_length to match training
            current_length = input_ids.shape[1]
            if current_length < max_length:
                # pad input_ids to max_length
                padding_length = max_length - current_length
                padding = torch.full((batch_size, padding_length), pad_token_id, dtype=input_ids.dtype, device=device)
                input_ids = torch.cat([input_ids, padding], dim=1)

                # update attention mask accordingly
                if attention_mask is not None:
                    mask_padding = torch.zeros((batch_size, padding_length), dtype=attention_mask.dtype, device=device)
                    attention_mask = torch.cat([attention_mask, mask_padding], dim=1)

        # fixed-length generation and kv caching are incompatible
        # fixed-length passes full sequences while cache expects single tokens
        if use_fixed_length_generation and use_cache:
            use_cache = False

        # prepare attention mask
        if attention_mask is None:
            attention_mask = input_ids.ne(pad_token_id) if pad_token_id is not None else torch.ones_like(input_ids)

        # handle jamo/char embeddings
        encoder_jamo_input_ids = None
        encoder_char_input_ids = None
        if self.config.jamo_subwords and hasattr(self.encoder, "jbu") and self.encoder.jbu is not None:
            encoder_jamo_input_ids = self.encoder.jbu[input_ids].long()
        if self.config.char_subwords and hasattr(self.encoder, "cbu") and self.encoder.cbu is not None:
            encoder_char_input_ids = self.encoder.cbu[input_ids].long()

        # encode inputs only once
        encoder_outputs = self.encoder(
            input_ids,
            encoder_jamo_input_ids,
            encoder_char_input_ids,
            attention_mask,
            None,  # encoder_hidden_states
            None,  # encoder_attention_mask
            output_attentions,
            output_hidden_states,
            False,  # use_cache
            None,  # past_key_values
            True,  # return_dict
        )

        # prepare encoder outputs for decoder
        encoder_hidden_states = encoder_outputs.last_hidden_state

        # handle different training modes (bart vs tsdae)
        if not self.config.use_bart_training:
            # for tsdae mode, use sentence embeddings
            input_mask = attention_mask.float()
            input_mask_expanded = input_mask.unsqueeze(-1)
            sum_embeddings = (encoder_hidden_states * input_mask_expanded).sum(dim=1)
            sum_mask = input_mask_expanded.sum(dim=1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            sentence_embeddings = sum_embeddings / sum_mask
            encoder_hidden_states = sentence_embeddings[:, None, :]
            encoder_attention_mask = torch.ones(batch_size, 1, device=device)
        else:
            # for bart mode, use full sequences
            encoder_attention_mask = attention_mask

        # initialize decoder input
        if decoder_start_token_id is None:
            raise ValueError("decoder_start_token_id must be specified for generation")

        # Route to appropriate generation method based on parameters
        if num_beams > 1:
            # Use beam search generation
            return self._beam_search_generate(
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                decoder_start_token_id=decoder_start_token_id,
                decoder_input_ids=decoder_input_ids,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                length_penalty=length_penalty,
                early_stopping=early_stopping,
                num_return_sequences=num_return_sequences,
                batch_size=batch_size,
                device=device,
                use_cache=use_cache,
                use_fixed_length_generation=use_fixed_length_generation,
            )
        else:
            # Use greedy or sampling generation with KV caching
            return self._generate_with_cache(
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                decoder_start_token_id=decoder_start_token_id,
                decoder_input_ids=decoder_input_ids,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                batch_size=batch_size,
                device=device,
                use_cache=use_cache,
                use_fixed_length_generation=use_fixed_length_generation,
            )

    def _generate_with_cache(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        decoder_start_token_id: int,
        decoder_input_ids: Optional[torch.Tensor],
        max_length: int,
        min_length: int,
        do_sample: bool,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        no_repeat_ngram_size: int,
        eos_token_id: Optional[int],
        pad_token_id: Optional[int],
        batch_size: int,
        device: torch.device,
        use_cache: bool = True,
        use_fixed_length_generation: bool = False,
    ) -> torch.Tensor:
        """Generation with proper KV caching for efficiency."""
        # initialize decoder input - use provided prompt or start token
        if decoder_input_ids is not None:
            # use provided decoder prompt
            decoder_tokens = decoder_input_ids.to(device)
            if decoder_tokens.dim() == 1:
                decoder_tokens = decoder_tokens.unsqueeze(0)
            # expand to batch size if needed
            if decoder_tokens.shape[0] == 1 and batch_size > 1:
                decoder_tokens = decoder_tokens.expand(batch_size, -1)
        else:
            # use single start token
            decoder_tokens = torch.full((batch_size, 1), decoder_start_token_id, device=device, dtype=torch.long)

        # track finished sequences
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Initialize KV cache
        past_key_values = None

        # determine initial prompt length for proper loop handling
        initial_length = decoder_tokens.shape[1]

        # determine buffer size for fixed-length generation
        if use_fixed_length_generation:
            # when using fixed-length generation, the buffer must be exactly max_length
            buffer_size = max_length
        else:
            buffer_size = max_length  # for standard generation, just use max_length

        # if we already have a prompt, we need to initialize cache with it first
        if initial_length > 1 and use_cache and not use_fixed_length_generation:
            # initialize cache with the full prompt except the last token
            prompt_tokens = decoder_tokens[:, :-1]

            # Compute jamo/char input_ids for prompt if needed
            prompt_jamo_input_ids = None
            prompt_char_input_ids = None
            if self.config.jamo_subwords and hasattr(self.decoder, "jbu") and self.decoder.jbu is not None:
                prompt_jamo_input_ids = self.decoder.jbu[prompt_tokens].long()
            if self.config.char_subwords and hasattr(self.decoder, "cbu") and self.decoder.cbu is not None:
                prompt_char_input_ids = self.decoder.cbu[prompt_tokens].long()

            # run decoder on prompt to build cache
            attention_mask_prompt = torch.ones_like(prompt_tokens)
            outputs = self.decoder(
                input_ids=prompt_tokens,
                jamo_input_ids=prompt_jamo_input_ids,
                char_input_ids=prompt_char_input_ids,
                attention_mask=attention_mask_prompt,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_values=None,
                use_cache=use_cache,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values

        for step in range(max_length - initial_length):
            current_length = decoder_tokens.shape[1]

            # Prepare input for this step based on generation mode
            if use_fixed_length_generation:
                # when using fixed-length generation, always pass the full buffer
                # this maintains encoder-decoder length alignment
                # pad decoder tokens to buffer size
                if current_length < buffer_size:
                    padding_length = buffer_size - current_length
                    padding = torch.full((batch_size, padding_length), pad_token_id, dtype=decoder_tokens.dtype, device=device)
                    input_ids_step = torch.cat([decoder_tokens, padding], dim=1)
                else:
                    input_ids_step = decoder_tokens[:, :buffer_size]
                # create attention mask for valid positions only
                positions = torch.arange(buffer_size, device=device).unsqueeze(0)
                decoder_attention_mask = (positions < current_length).long()
            elif past_key_values is not None and use_cache:
                # Only use the last generated token when we have cached KV pairs
                input_ids_step = decoder_tokens[:, -1:]
                decoder_attention_mask = torch.ones_like(input_ids_step)
            else:
                # Use full sequence for first step or non-cached generation
                input_ids_step = decoder_tokens
                decoder_attention_mask = decoder_tokens.ne(pad_token_id) if pad_token_id is not None else torch.ones_like(decoder_tokens)

            # Compute jamo/char input_ids if needed
            decoder_jamo_input_ids = None
            decoder_char_input_ids = None
            if self.config.jamo_subwords and hasattr(self.decoder, "jbu") and self.decoder.jbu is not None:
                decoder_jamo_input_ids = self.decoder.jbu[input_ids_step].long()
            if self.config.char_subwords and hasattr(self.decoder, "cbu") and self.decoder.cbu is not None:
                decoder_char_input_ids = self.decoder.cbu[input_ids_step].long()

            # forward pass with KV caching
            outputs = self.decoder(
                input_ids=input_ids_step,
                jamo_input_ids=decoder_jamo_input_ids,
                char_input_ids=decoder_char_input_ids,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                return_dict=True,
            )

            # Update KV cache for next iteration
            if use_cache:
                past_key_values = outputs.past_key_values

            # get logits for next token prediction
            hidden_states = outputs.last_hidden_state
            next_token_logits = self.lm_head(hidden_states[:, -1, :])

            # apply repetition penalty
            if repetition_penalty != 1.0:
                next_token_logits = self._apply_repetition_penalty(
                    next_token_logits, decoder_tokens, repetition_penalty
                )

            # apply no_repeat_ngram constraint
            if no_repeat_ngram_size > 0 and current_length > no_repeat_ngram_size:
                next_token_logits = self._apply_no_repeat_ngram(
                    next_token_logits, decoder_tokens, no_repeat_ngram_size, current_length
                )

            # apply min_length constraint
            if current_length < min_length and eos_token_id is not None:
                next_token_logits[:, eos_token_id] = -float('inf')

            if do_sample:
                # sampling-based generation
                next_token = self._sample_next_token(next_token_logits, temperature, top_k, top_p)
            else:
                # greedy generation
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # update sequences
            decoder_tokens = torch.cat([decoder_tokens, next_token], dim=-1)

            # check for finished sequences
            if eos_token_id is not None:
                finished = finished | (next_token.squeeze(-1) == eos_token_id)
                if finished.all():
                    break

        return decoder_tokens

    def _beam_search_generate(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        decoder_start_token_id: int,
        decoder_input_ids: Optional[torch.Tensor],
        max_length: int,
        min_length: int,
        num_beams: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        no_repeat_ngram_size: int,
        eos_token_id: Optional[int],
        pad_token_id: Optional[int],
        length_penalty: float,
        early_stopping: bool,
        num_return_sequences: int,
        batch_size: int,
        device: torch.device,
        use_cache: bool = True,
        use_fixed_length_generation: bool = False,
    ) -> torch.Tensor:
        """Beam search generation with KV caching."""
        # handle fixed-length generation compatibility
        if use_fixed_length_generation:
            # fixed-length generation and kv caching are incompatible in beam search
            if use_cache:
                logger.warning("Fixed-length generation is incompatible with KV caching in beam search. Disabling cache.")
                use_cache = False

        # Expand inputs for beam search
        beam_batch_size = batch_size * num_beams

        # Expand encoder outputs
        encoder_hidden_states = encoder_hidden_states.unsqueeze(1).repeat(1, num_beams, 1, 1)
        encoder_hidden_states = encoder_hidden_states.view(beam_batch_size, encoder_hidden_states.shape[2], encoder_hidden_states.shape[3])

        encoder_attention_mask = encoder_attention_mask.unsqueeze(1).repeat(1, num_beams, 1)
        encoder_attention_mask = encoder_attention_mask.view(beam_batch_size, encoder_attention_mask.shape[2])

        # Initialize beams
        if decoder_input_ids is None:
            decoder_input_ids = torch.full((beam_batch_size, 1), decoder_start_token_id, device=device, dtype=torch.long)
        else:
            # replicate decoder prompt for all beams
            decoder_input_ids = decoder_input_ids.unsqueeze(1).repeat(1, num_beams, 1)
            decoder_input_ids = decoder_input_ids.view(beam_batch_size, -1)

        # Initialize beam scores
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=device)
        beam_scores[:, 1:] = -float('inf')  # Only first beam is active initially
        beam_scores = beam_scores.view(-1)

        # Track finished beams
        finished_sequences = []
        finished_scores = []

        # Initialize KV cache
        past_key_values = None

        # determine initial prompt length for proper loop handling
        initial_length = decoder_input_ids.shape[1]

        # pre-allocate buffer for fixed-length generation
        if use_fixed_length_generation:
            buffer_size = max_length
        else:
            buffer_size = max_length

        for step in range(max_length - initial_length):
            current_length = decoder_input_ids.shape[1]

            # Prepare input for this step based on generation mode
            if use_fixed_length_generation:
                # when using fixed-length generation, always pass the full buffer
                # pad decoder tokens to buffer size
                if current_length < buffer_size:
                    padding_length = buffer_size - current_length
                    padding = torch.full((beam_batch_size, padding_length), pad_token_id, dtype=decoder_input_ids.dtype, device=device)
                    input_ids_step = torch.cat([decoder_input_ids, padding], dim=1)
                else:
                    input_ids_step = decoder_input_ids[:, :buffer_size]
                # create attention mask for valid positions only
                positions = torch.arange(buffer_size, device=device).unsqueeze(0)
                decoder_attention_mask = (positions < current_length).long()
            elif past_key_values is not None and use_cache:
                input_ids_step = decoder_input_ids[:, -1:]
                decoder_attention_mask = torch.ones_like(input_ids_step)
            else:
                input_ids_step = decoder_input_ids
                decoder_attention_mask = decoder_input_ids.ne(pad_token_id) if pad_token_id is not None else torch.ones_like(decoder_input_ids)

            # Compute jamo/char input_ids if needed
            decoder_jamo_input_ids = None
            decoder_char_input_ids = None
            if self.config.jamo_subwords and hasattr(self.decoder, "jbu") and self.decoder.jbu is not None:
                decoder_jamo_input_ids = self.decoder.jbu[input_ids_step].long()
            if self.config.char_subwords and hasattr(self.decoder, "cbu") and self.decoder.cbu is not None:
                decoder_char_input_ids = self.decoder.cbu[input_ids_step].long()

            # forward pass with KV caching
            outputs = self.decoder(
                input_ids=input_ids_step,
                jamo_input_ids=decoder_jamo_input_ids,
                char_input_ids=decoder_char_input_ids,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                return_dict=True,
            )

            # Update KV cache
            if use_cache:
                past_key_values = outputs.past_key_values

            # get logits for next token prediction
            hidden_states = outputs.last_hidden_state
            next_token_logits = self.lm_head(hidden_states[:, -1, :])

            # apply repetition penalty
            if repetition_penalty != 1.0:
                next_token_logits = self._apply_repetition_penalty(
                    next_token_logits, decoder_input_ids, repetition_penalty
                )

            # apply no_repeat_ngram constraint
            if no_repeat_ngram_size > 0 and current_length > no_repeat_ngram_size:
                next_token_logits = self._apply_no_repeat_ngram(
                    next_token_logits, decoder_input_ids, no_repeat_ngram_size, current_length
                )

            # apply min_length constraint
            if current_length < min_length and eos_token_id is not None:
                next_token_logits[:, eos_token_id] = -float('inf')

            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Calculate scores for all possible next tokens
            next_scores = F.log_softmax(next_token_logits, dim=-1)

            # Apply length penalty
            if length_penalty != 1.0:
                next_scores = next_scores / (decoder_input_ids.shape[1] ** length_penalty)

            # Add to beam scores
            next_scores = next_scores + beam_scores.unsqueeze(1)

            # Reshape for beam selection
            next_scores = next_scores.view(batch_size, num_beams * next_token_logits.shape[-1])

            # Select top beams
            next_scores, next_tokens = torch.topk(next_scores, 2 * num_beams, dim=1, largest=True, sorted=True)

            next_beam_scores = []
            next_beam_tokens = []
            next_beam_indices = []

            for batch_idx in range(batch_size):
                beam_id = 0
                for rank in range(2 * num_beams):
                    if beam_id >= num_beams:
                        break

                    token_id = next_tokens[batch_idx, rank] % next_token_logits.shape[-1]
                    beam_idx = next_tokens[batch_idx, rank] // next_token_logits.shape[-1]

                    # Check if this beam has finished
                    if eos_token_id is not None and token_id.item() == eos_token_id:
                        if decoder_input_ids.shape[1] >= min_length:
                            # Store finished sequence
                            finished_sequences.append(decoder_input_ids[batch_idx * num_beams + beam_idx].clone())
                            finished_scores.append(next_scores[batch_idx, rank].item())

                            # Check early stopping
                            if early_stopping and len(finished_sequences) >= num_beams:
                                break
                        continue

                    next_beam_scores.append(next_scores[batch_idx, rank])
                    next_beam_tokens.append(token_id)
                    next_beam_indices.append(batch_idx * num_beams + beam_idx)
                    beam_id += 1

            # Check if we have enough finished sequences
            if early_stopping and len(finished_sequences) >= batch_size * num_beams:
                break

            # Update beam state
            if len(next_beam_scores) > 0:
                beam_scores = torch.stack(next_beam_scores)
                beam_tokens = torch.stack(next_beam_tokens).unsqueeze(1)
                beam_indices = torch.tensor(next_beam_indices, device=device)

                # Reorder decoder input ids
                decoder_input_ids = decoder_input_ids[beam_indices]
                decoder_input_ids = torch.cat([decoder_input_ids, beam_tokens], dim=-1)

                # Reorder KV cache
                if use_cache and past_key_values is not None:
                    past_key_values = self._reorder_cache(past_key_values, beam_indices)
            else:
                break

        # Return best sequences
        if len(finished_sequences) > 0:
            # Sort by scores and return top sequences
            sorted_indices = sorted(range(len(finished_scores)), key=lambda i: finished_scores[i], reverse=True)
            best_sequences = [finished_sequences[i] for i in sorted_indices[:num_return_sequences]]

            # Pad sequences to same length
            max_len = max(seq.shape[0] for seq in best_sequences)
            padded_sequences = []
            for seq in best_sequences:
                if seq.shape[0] < max_len:
                    padding = torch.full((max_len - seq.shape[0],), pad_token_id if pad_token_id is not None else 0,
                                        device=device, dtype=torch.long)
                    seq = torch.cat([seq, padding])
                padded_sequences.append(seq)

            return torch.stack(padded_sequences)
        else:
            # Return current best beams if no sequences finished
            return decoder_input_ids[:num_return_sequences]

    def _simple_generate(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        decoder_start_token_id: int,
        max_length: int,
        min_length: int,
        do_sample: bool,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        eos_token_id: Optional[int],
        pad_token_id: Optional[int],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Simple generation without kv caching (kept for compatibility)."""
        return self._generate_with_cache(
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            decoder_start_token_id=decoder_start_token_id,
            max_length=max_length,
            min_length=min_length,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            batch_size=batch_size,
            device=device,
            use_cache=False,  # Disable caching for simple generation
        )


    def _sample_next_token(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float
    ) -> torch.Tensor:
        """Sample next token using temperature, top-k, and top-p."""
        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature

        # Apply top-k filtering
        if top_k > 0:
            top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
            logits_filtered = torch.full_like(logits, float('-inf'))
            logits_filtered.scatter_(-1, top_k_indices, top_k_logits)
            logits = logits_filtered

        # Apply top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # Create mask for indices to remove
            indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
            logits = logits.masked_fill(indices_to_remove, float('-inf'))

        # Sample from the filtered distribution
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        return next_token

    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        previous_tokens: torch.Tensor,
        penalty: float
    ) -> torch.Tensor:
        """Apply repetition penalty to logits."""
        if penalty == 1.0:
            return logits

        # Create a mask for tokens that have appeared before
        batch_size, vocab_size = logits.shape
        for batch_idx in range(batch_size):
            for token in previous_tokens[batch_idx]:
                if token < vocab_size:
                    if logits[batch_idx, token] < 0:
                        logits[batch_idx, token] *= penalty
                    else:
                        logits[batch_idx, token] /= penalty

        return logits

    def _apply_no_repeat_ngram(
        self,
        logits: torch.Tensor,
        previous_tokens: torch.Tensor,
        ngram_size: int,
        cur_len: int
    ) -> torch.Tensor:
        """Apply no-repeat-ngram constraint."""
        if ngram_size <= 0:
            return logits

        batch_size = previous_tokens.shape[0]

        for batch_idx in range(batch_size):
            # get the sequence for this batch
            seq = previous_tokens[batch_idx]

            # check if we have enough tokens for ngram comparison
            if cur_len < ngram_size:
                continue

            # get the last (ngram_size - 1) tokens
            ngram_prefix = tuple(seq[cur_len - ngram_size + 1:cur_len].tolist())

            # look for this prefix in the previous sequence
            for i in range(cur_len - ngram_size):
                prev_ngram = tuple(seq[i:i + ngram_size - 1].tolist())

                if prev_ngram == ngram_prefix:
                    # we found a match, ban the next token that would complete this ngram
                    next_token = seq[i + ngram_size - 1]
                    if next_token < logits.shape[-1]:
                        logits[batch_idx, next_token] = -float('inf')

        return logits


class Han2HanClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim: int,
        inner_dim: int,
        num_classes: int,
        pooler_dropout: float,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Normalizes inputs using the RMS (i.e. the L2 norm divided by the square
    root of the dimensionality) rather than the standard deviation employed by
    LayerNorm.  This variant is bias-free and matches the behavior of
    ``nnx.RMSNorm`` used in the Flax reference model.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-8, use_fla_fused: bool = False):
        """Create an RMSNorm layer.

        The weight *parameter name stays constant* (``<prefix>.weight``) so that
        checkpoints are compatible regardless of whether ``use_fla_fused`` was
        True or False at training time.  When fused mode is requested we still
        instantiate the ``FLARMSNorm`` module, but we *reuse* the same weight
        parameter instead of letting it create its own.  This guarantees a
        stable state-dict layout and avoids missing / unexpected key errors
        when switching between fused and unfused builds.
        """
        super().__init__()

        self.eps = eps
        self.use_fla_fused = use_fla_fused
        self.weight = nn.Parameter(torch.ones(hidden_size))

        if use_fla_fused:
            _import_fla_modules()
            self.fla_norm = FLARMSNorm(
                hidden_size=hidden_size,
                elementwise_affine=True,
                bias=False,
                eps=eps,
            )
            delattr(self.fla_norm, "weight")
            self.fla_norm.register_parameter("weight", self.weight)
        else:
            self.fla_norm = None
        self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fla_norm is not None:
            return self.fla_norm(x)
        # compute variance in fp32 for stability regardless of x.dtype, then
        # cast rms back to x.dtype before the division. matches the Flax
        # RMSNorm in normalization.py (MaxText / standard pattern).
        input_dtype = x.dtype
        x_squared = x.float().pow(2)
        rms = x_squared.mean(dim=-1, keepdim=True).add(self.eps).sqrt().to(input_dtype)
        x_norm = x / rms
        return x_norm * self.weight.to(input_dtype)


class Han2HanForSequenceClassification(Han2HanPreTrainedModel):

    def __init__(self, config: Han2HanConfig):
        super().__init__(config)

        self.gradient_checkpointing = None
        self.activate_s_embs = False

        self.encoder = Han2HanModule(config, is_encoder=True)
        self.decoder = Han2HanModule(config, is_encoder=False)

        self.classifier = Han2HanClassificationHead(config.d_model, config.d_model, config.num_labels, config.classf_pdrop)

        tied = self._init_tied_weights_keys()
        tied.pop("lm_head.weight", None)
        self._tied_weights_keys = tied
        self.post_init()

    def get_input_embeddings(self):
        return self.encoder.wte

    def set_input_embeddings(self, value):
        value.padding_idx = self.encoder.wte.padding_idx
        self.encoder.wte = value
        self.decoder.wte = value

    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self.encoder.wte.weight = self.decoder.wte.weight
        tie_subtokens = getattr(self.config, 'tie_subtoken_embeddings', False)
        if tie_subtokens:
            if self.config.jamo_subwords:
                self.encoder.wje.weight = self.decoder.wje.weight
            if self.config.char_subwords:
                self.encoder.wce.weight = self.decoder.wce.weight
        if self.config.tie_encoder_decoder and not self.config.tie_word_embeddings:
            for module in (self.encoder, self.decoder):
                module.wte.weight = nn.Parameter(torch.empty(module.wte.weight.shape))
                module.wte.apply(self._init_weights)
                if self.config.jamo_subwords:
                    module.wje.weight = nn.Parameter(torch.empty(module.wje.weight.shape))
                    module.wje.apply(self._init_weights)
                if self.config.char_subwords:
                    module.wce.weight = nn.Parameter(torch.empty(module.wce.weight.shape))
                    module.wce.apply(self._init_weights)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Union[Tuple, BaseModelOutputWithPastAttentionsAndSentenceEmbeddings]] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        use_cache: Optional[bool] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        output_sentence_embeddings: bool = False,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:

        return_dict = return_dict if return_dict is not None else self.config.return_dict
        init_cache = use_cache if use_cache is not None else self.config.use_cache

        if input_ids is None:
            raise ValueError("You have to specify either input_ids or encoder_outputs")

        encoder_jamo_input_ids = None
        encoder_char_input_ids = None

        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id)

        if self.config.jamo_subwords and hasattr(self.encoder, "jbu") and self.encoder.jbu is not None:
            encoder_jamo_input_ids = self.encoder.jbu[input_ids].long()
        if self.config.char_subwords and hasattr(self.encoder, "cbu") and self.encoder.cbu is not None:
            encoder_char_input_ids = self.encoder.cbu[input_ids].long()

        encoder_outputs = self.encoder(
            input_ids,
            encoder_jamo_input_ids,
            encoder_char_input_ids,
            attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            output_attentions,
            output_hidden_states,
            init_cache,
            past_key_values,
            return_dict,
        )

        if not return_dict:
            encoder_hidden_states = encoder_outputs[0]
        else:
            encoder_hidden_states = encoder_outputs.last_hidden_state

        if self.config.use_bart_training or output_sentence_embeddings:

            input_mask = (input_ids != self.config.pad_token_id).float()
            input_mask_expanded = input_mask.unsqueeze(-1)
            sum_embeddings = (encoder_hidden_states * input_mask_expanded).sum(dim=1)
            sum_mask = input_mask_expanded.sum(dim=1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            sentence_embeddings = sum_embeddings / sum_mask

            if output_sentence_embeddings:
                return (sentence_embeddings,)

        if decoder_input_ids is None and input_ids is not None:
            decoder_input_ids = shift_tokens_right(
                input_ids.clone(), self.config.pad_token_id, self.config.decoder_start_token_id
            )

        if decoder_input_ids is not None:
            decoder_char_input_ids = None
            decoder_jamo_input_ids = None
            if self.config.jamo_subwords and hasattr(self.decoder, "jbu") and self.decoder.jbu is not None:
                decoder_jamo_input_ids = self.decoder.jbu[decoder_input_ids].long()
            if self.config.char_subwords and hasattr(self.decoder, "cbu") and self.decoder.cbu is not None:
                decoder_char_input_ids = self.decoder.cbu[decoder_input_ids].long()
            if decoder_attention_mask is None:
                decoder_attention_mask = decoder_input_ids.ne(self.config.pad_token_id).long()

        # for BART training, use full encoder sequences; for TSDAE, use pooled embeddings
        if self.config.use_bart_training:
            encoder_hidden_for_decoder = encoder_outputs.last_hidden_state if return_dict else encoder_outputs[0]  # (batch, seq_len, d_model)
            encoder_attention_mask_for_decoder = attention_mask  # (batch, seq_len)
        else:
            encoder_hidden_for_decoder = sentence_embeddings[:, None, :]  # (batch, 1, d_model)
            encoder_attention_mask_for_decoder = attention_mask[:, 0:1]  # (batch, 1)

        decoder_outputs = self.decoder(
            decoder_input_ids,
            decoder_jamo_input_ids,
            decoder_char_input_ids,
            decoder_attention_mask,
            encoder_hidden_for_decoder,  # encoder hidden states
            encoder_attention_mask_for_decoder,  # encoder attention mask
            output_attentions,
            output_hidden_states,
            init_cache,
            past_key_values,
            return_dict,
        )

        if not return_dict:
            hidden_states = decoder_outputs[0]
        else:
            hidden_states = decoder_outputs.last_hidden_state

        # mean pool over non-pad decoder positions
        mask = decoder_attention_mask.unsqueeze(-1).float()
        sentence_representation = (hidden_states * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

        logits = self.classifier(sentence_representation)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.config.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.config.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.config.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss(label_smoothing=getattr(self.config, 'label_smoothing', 0.0))
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            past_key_values=None,
            sentence_embeddings=sentence_embeddings,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

class Han2HanForTokenClassification(Han2HanPreTrainedModel):

    def __init__(self, config: Han2HanConfig):
        super().__init__(config)

        self.gradient_checkpointing = None
        self.activate_s_embs = False

        self.encoder = Han2HanModule(config, is_encoder=True)
        self.decoder = Han2HanModule(config, is_encoder=False)

        self.classifier = Han2HanClassificationHead(config.d_model, config.d_model, config.num_labels, config.classf_pdrop)

        tied = self._init_tied_weights_keys()
        tied.pop("lm_head.weight", None)
        self._tied_weights_keys = tied
        self.post_init()

    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self.encoder.wte.weight = self.decoder.wte.weight
        tie_subtokens = getattr(self.config, 'tie_subtoken_embeddings', False)
        if tie_subtokens:
            if self.config.jamo_subwords:
                self.encoder.wje.weight = self.decoder.wje.weight
            if self.config.char_subwords:
                self.encoder.wce.weight = self.decoder.wce.weight
        if self.config.tie_encoder_decoder and not self.config.tie_word_embeddings:
            for module in (self.encoder, self.decoder):
                module.wte.weight = nn.Parameter(torch.empty(module.wte.weight.shape))
                module.wte.apply(self._init_weights)
                if self.config.jamo_subwords:
                    module.wje.weight = nn.Parameter(torch.empty(module.wje.weight.shape))
                    module.wje.apply(self._init_weights)
                if self.config.char_subwords:
                    module.wce.weight = nn.Parameter(torch.empty(module.wce.weight.shape))
                    module.wce.apply(self._init_weights)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        output_sentence_embeddings: bool = False,
        return_dict: bool = True,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        encoder_jamo_input_ids = None
        encoder_char_input_ids = None

        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id)

        if self.config.jamo_subwords and hasattr(self.encoder, "jbu") and self.encoder.jbu is not None:
            encoder_jamo_input_ids = self.encoder.jbu[input_ids].long()
        if self.config.char_subwords and hasattr(self.encoder, "cbu") and self.encoder.cbu is not None:
            encoder_char_input_ids = self.encoder.cbu[input_ids].long()

        encoder_outputs = self.encoder(
            input_ids,
            encoder_jamo_input_ids,
            encoder_char_input_ids,
            attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            output_attentions,
            output_hidden_states,
            init_cache,
            past_key_values,
            return_dict,
        )

        if not return_dict:
            encoder_hidden_states = encoder_outputs[0]
        else:
            encoder_hidden_states = encoder_outputs.last_hidden_state

        # compute sentence embeddings for optional output
        input_mask = (input_ids != self.config.pad_token_id).float()
        input_mask_expanded = input_mask.unsqueeze(-1)
        sum_embeddings = (encoder_hidden_states * input_mask_expanded).sum(dim=1)
        sum_mask = input_mask_expanded.sum(dim=1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        sentence_embeddings = sum_embeddings / sum_mask

        if output_sentence_embeddings:
            return (sentence_embeddings,)

        if decoder_input_ids is None:
            decoder_input_ids = shift_tokens_right(
                input_ids.clone(), self.config.pad_token_id, self.config.decoder_start_token_id
            )

        if decoder_input_ids is not None:
            decoder_char_input_ids = None
            decoder_jamo_input_ids = None
            if self.config.jamo_subwords and hasattr(self.decoder, "jbu") and self.decoder.jbu is not None:
                decoder_jamo_input_ids = self.decoder.jbu[decoder_input_ids].long()
            if self.config.char_subwords and hasattr(self.decoder, "cbu") and self.decoder.cbu is not None:
                decoder_char_input_ids = self.decoder.cbu[decoder_input_ids].long()
            if decoder_attention_mask is None:
                decoder_attention_mask = decoder_input_ids.ne(self.config.pad_token_id).long()

        # for BART training, use full encoder sequences; for TSDAE, use pooled embeddings
        if self.config.use_bart_training:
            encoder_hidden_for_decoder = encoder_hidden_states  # (batch, seq_len, d_model)
            encoder_attention_mask_for_decoder = attention_mask  # (batch, seq_len)
        else:
            encoder_hidden_for_decoder = sentence_embeddings[:, None, :]  # (batch, 1, d_model)
            encoder_attention_mask_for_decoder = attention_mask[:, 0:1]  # (batch, 1)

        decoder_outputs = self.decoder(
            decoder_input_ids,
            decoder_jamo_input_ids,
            decoder_char_input_ids,
            decoder_attention_mask,
            encoder_hidden_for_decoder,
            encoder_attention_mask_for_decoder,
            output_attentions,
            output_hidden_states,
            init_cache,
            past_key_values,
            return_dict,
        )

        if not return_dict:
            hidden_states = decoder_outputs[0]
        else:
            hidden_states = decoder_outputs.last_hidden_state

        logits = self.classifier(hidden_states) # each token is classified

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.config.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.config.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.config.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss(label_smoothing=getattr(self.config, 'label_smoothing', 0.0))
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return ClassifierOutput(
            loss=loss,
            logits=logits,
            past_key_values=None,
            sentence_embeddings=sentence_embeddings,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class Han2HanForQuestionAnswering(Han2HanPreTrainedModel):
    base_model_prefix = "model"

    def __init__(self, config: Han2HanConfig):
        super().__init__(config)

        config.num_labels = 2
        self.num_labels = config.num_labels

        self.gradient_checkpointing = None
        self.activate_s_embs = False

        self.model = Han2Han(config)

        self.qa_outputs = nn.Linear(config.d_model, config.num_labels)

        self._tied_weights_keys = {
            f"model.{k}": f"model.{v}"
            for k, v in self.model._tied_weights_keys.items()
        }
        self.post_init()

    def _tie_weights(self):
        self.model._tie_weights()

    def forward(
        self,
        input_ids: torch.Tensor,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_sentence_embeddings: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (*sequence_length*). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (*sequence_length*). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        if hasattr(self.model, "lm_head"):
            del self.model.lm_head
            assert not hasattr(self.model, "lm_head"), "Cannot have QA model with LM head. Fix this."

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if start_positions is not None and end_positions is not None:
            use_cache = False  # don't cache when training with labels

        encoder_outputs = self.model(
            input_ids,
            attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_sentence_embeddings=output_sentence_embeddings,
            use_cache=use_cache,
            return_dict=return_dict,
        )

        if not return_dict:
            hidden_states = encoder_outputs[0]
        else:
            hidden_states = encoder_outputs.last_hidden_state

        input_mask = (input_ids != self.config.pad_token_id).float()
        input_mask_expanded = input_mask.unsqueeze(-1)
        sum_embeddings = (hidden_states * input_mask_expanded).sum(dim=1)
        sum_mask = input_mask_expanded.sum(dim=1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        sentence_embeddings = sum_embeddings / sum_mask

        if output_sentence_embeddings:
            return (sentence_embeddings,)

        sequence_output = encoder_outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index, label_smoothing=getattr(self.config, 'label_smoothing', 0.0))
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (
                start_logits,
                end_logits,
            ) + outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=encoder_outputs.last_hidden_state,
            attentions=encoder_outputs.encoder_attentions
        )


class Han2HanForMultipleChoice(Han2HanPreTrainedModel):
    base_model_prefix = "model"

    def __init__(self, config: Han2HanConfig):
        super().__init__(config)

        self.model = Han2Han(config)
        self.dropout = nn.Dropout(config.classf_pdrop)
        self.classifier = nn.Linear(config.d_model, 1)

        self.gradient_checkpointing = None
        self.activate_s_embs = False

        self._tied_weights_keys = {
            f"model.{k}": f"model.{v}"
            for k, v in self.model._tied_weights_keys.items()
        }
        self.post_init()

    def _tie_weights(self):
        self.model._tie_weights()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_sentence_embeddings: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else None
        batch_size = input_ids.shape[0] if input_ids is not None else None

        # reshape inputs: (batch_size, num_choices, seq_len) -> (batch_size * num_choices, seq_len)
        if input_ids is not None:
            input_ids_flat = input_ids.view(-1, input_ids.size(-1))
        else:
            input_ids_flat = None

        sep_token_id = self.config.eos_token_id
        pad_token_id = self.config.pad_token_id
        decoder_start_token_id = self.config.decoder_start_token_id

        # Process in batches to improve efficiency
        batch_contexts = []
        batch_choices = []
        batch_decoder_inputs = []
        batch_indices = []

        for i in range(batch_size * num_choices):
            curr_input = input_ids_flat[i]

            # Find the separator position
            sep_positions = (curr_input == sep_token_id).nonzero(as_tuple=True)[0]
            if len(sep_positions) > 0:
                sep_pos = sep_positions[0].item()

                # Split into context (question) and choice (answer)
                context_ids = curr_input[:sep_pos + 1]  # include the separator
                choice_ids = curr_input[sep_pos + 1:]  # exclude the separator

                # Filter out padding from choice
                choice_mask = choice_ids != pad_token_id
                choice_ids = choice_ids[choice_mask]

                if len(choice_ids) > 0:
                    # Prepare decoder input (shift right, add decoder_start_token)
                    decoder_input = torch.cat([
                        torch.tensor([decoder_start_token_id], device=curr_input.device),
                        choice_ids[:-1]
                    ])

                    batch_contexts.append(context_ids)
                    batch_choices.append(choice_ids)
                    batch_decoder_inputs.append(decoder_input)
                    batch_indices.append(i)

        # if we have valid examples to process
        if len(batch_contexts) > 0:
            # pad all sequences to same length for batched processing
            max_context_len = max(len(c) for c in batch_contexts)
            max_decoder_len = max(len(d) for d in batch_decoder_inputs)

            # create padded tensors
            padded_contexts = torch.full((len(batch_contexts), max_context_len), pad_token_id, device=input_ids.device)
            padded_decoder_inputs = torch.full((len(batch_contexts), max_decoder_len), pad_token_id, device=input_ids.device)
            context_masks = torch.zeros((len(batch_contexts), max_context_len), dtype=torch.bool, device=input_ids.device)
            decoder_masks = torch.zeros((len(batch_contexts), max_decoder_len), dtype=torch.bool, device=input_ids.device)

            for j, (ctx, dec) in enumerate(zip(batch_contexts, batch_decoder_inputs)):
                padded_contexts[j, :len(ctx)] = ctx
                padded_decoder_inputs[j, :len(dec)] = dec
                context_masks[j, :len(ctx)] = True
                decoder_masks[j, :len(dec)] = True

            # batch forward pass through the model
            with torch.no_grad() if not self.training else torch.enable_grad():
                outputs = self.model(
                    input_ids=padded_contexts,
                    attention_mask=context_masks,
                    decoder_input_ids=padded_decoder_inputs,
                    decoder_attention_mask=decoder_masks,
                    use_cache=False,
                    return_dict=True,
                )

                # get logits and compute log-likelihood
                logits = outputs.logits

                # compute scores for each example
                batch_scores = []
                for j, choice_ids in enumerate(batch_choices):
                    # get relevant logits for this example
                    example_logits = logits[j, :len(choice_ids), :]

                    # compute log probabilities for this choice
                    log_probs = F.log_softmax(example_logits, dim=-1)

                    # get log probs for the actual tokens in the choice
                    token_log_probs = log_probs.gather(
                        dim=-1,
                        index=choice_ids.unsqueeze(-1)
                    ).squeeze(-1)

                    # average log probability as the score (higher = better)
                    choice_score = token_log_probs.mean()
                    batch_scores.append(choice_score)

            # assign scores to their correct positions
            scores_tensor = torch.full((batch_size * num_choices,), -1e10, device=input_ids.device)
            for idx, score in zip(batch_indices, batch_scores):
                scores_tensor[idx] = score

            reshaped_logits = scores_tensor.view(batch_size, num_choices)
        else:
            # no valid examples, return very low scores
            reshaped_logits = torch.full((batch_size, num_choices), -1e10, device=input_ids.device)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(label_smoothing=getattr(self.config, 'label_smoothing', 0.0))
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,)
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
                loss=loss,
                logits=reshaped_logits,
                hidden_states=None,
                attentions=None,
            )


# register with AutoClasses when module is imported. Failures here mean
# transformers / register_han2han.py is broken; surface them loudly.
import register_han2han  # noqa: F401


if __name__ == "__main__":
    # Test on GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Testing on device: {device}")

    config = Han2HanConfig(
        attention_mechanism='mha',
        jamo_subwords=False,
        char_subwords=False,
        use_fla_fused_rotary=torch.cuda.is_available()  # Use FLA rotary only on GPU
    )
    model = Han2Han(config).to(device)
    input_ids = torch.ones((1, 8), dtype=torch.long).to(device)
    model.gradient_checkpointing_enable()

    outputs = model(input_ids=input_ids)
    print(f"Output shape: {outputs.logits.shape}")

    # Test CPU fallback explicitly
    print("\nTesting CPU fallback with use_fla_fused_rotary=False...")
    config_cpu = Han2HanConfig(
        attention_mechanism='mha',
        jamo_subwords=False,
        char_subwords=False,
        use_fla_fused_rotary=False  # Force simple rotary
    )
    model_cpu = Han2Han(config_cpu).to('cpu')
    input_ids_cpu = torch.ones((1, 8), dtype=torch.long).to('cpu')

    outputs_cpu = model_cpu(input_ids=input_ids_cpu)
    print(f"CPU output shape: {outputs_cpu.logits.shape}")
    print("CPU fallback test successful!")

    # Test cross-attention during generation
    print("\n" + "="*60)
    print("Testing cross-attention during generation...")
    print("="*60)

    # create a simple encoder-decoder model
    config = Han2HanConfig(
        vocab_size=100,
        d_model=64,
        encoder_nlayer=2,
        decoder_nlayer=2,
        num_heads=4,
        d_ff=256,
        attention_mechanism='mha',  # use simple MHA for testing
        use_fla_fused_rotary=False,
        jamo_subwords=False,
        char_subwords=False,
    )

    model = Han2Han(config).to(device)
    model.eval()

    # create test inputs
    batch_size = 2
    encoder_length = 10
    decoder_length = 5

    encoder_input_ids = torch.randint(0, 100, (batch_size, encoder_length), device=device)
    decoder_input_ids = torch.randint(0, 100, (batch_size, decoder_length), device=device)
    labels = torch.randint(0, 100, (batch_size, decoder_length), device=device)

    print(f"\nEncoder input shape: {encoder_input_ids.shape}")
    print(f"Decoder input shape: {decoder_input_ids.shape}")

    # test forward pass (training mode)
    print("\n--- Testing forward pass (training) ---")
    with torch.no_grad():
        outputs = model(
            input_ids=encoder_input_ids,
            decoder_input_ids=decoder_input_ids
        )
    print(f"Forward pass successful! Output shape: {outputs.logits.shape}")

    # add debug hook to check cross-attention
    cross_attn_called = {'count': 0, 'shapes': []}

    def debug_cross_attention(module, args, output):
        if hasattr(module, 'is_cross_attention') and module.is_cross_attention:
            hidden_states, attn_mask, encoder_hidden, encoder_mask = args[:4]
            cross_attn_called['count'] += 1
            cross_attn_called['shapes'].append({
                'hidden_states': hidden_states.shape,
                'encoder_hidden': encoder_hidden.shape if encoder_hidden is not None else None
            })
            print(f"  Cross-attention called! Hidden: {hidden_states.shape}, Encoder: {encoder_hidden.shape if encoder_hidden is not None else 'None'}")
        return output

    # register hooks
    for module in model.modules():
        if isinstance(module, Han2HanAttention):
            module.register_forward_hook(debug_cross_attention)

    # test generation
    print("\n--- Testing generation ---")
    with torch.no_grad():
        # reset counter
        cross_attn_called['count'] = 0
        cross_attn_called['shapes'] = []

        generated = model.generate(
            input_ids=encoder_input_ids,
            max_length=20,
            do_sample=True,
            temperature=1.0,
            top_k=50,
            top_p=0.95,
            pad_token_id=config.pad_token_id,
            bos_token_id=config.bos_token_id,
            eos_token_id=config.eos_token_id,
        )

        print(f"\nGenerated shape: {generated.shape}")
        print(f"Cross-attention was called {cross_attn_called['count']} times during generation")

        if cross_attn_called['count'] == 0:
            print("\n  WARNING: Cross-attention was NOT called during generation!")
        else:
            print("\n Cross-attention is working during generation")
            print(f"Shapes seen: {cross_attn_called['shapes'][:3]}...")  # show first 3

    print("\n" + "="*60)
    print("Cross-attention test complete!")
    print("="*60)