#!/usr/bin/env python3
# coding: utf-8
"""Rotary position embedding helpers shared by the encoder/decoder attention.

These functions implement the standard split-half RoPE rotation along with a
position-id-aware variant used by packed-sequence training and a frequency
precomputation utility.
"""

import jax.numpy as jnp


def _apply_rotary_pos_emb(x: jnp.ndarray, freqs_cis: jnp.ndarray):
    x_r, x_i = jnp.split(x, 2, axis=-1)
    cos_f = freqs_cis[..., 0]
    sin_f = freqs_cis[..., 1]
    if x.ndim == 4:
        cos_f_reshaped = cos_f[None, :, None, :]
        sin_f_reshaped = sin_f[None, :, None, :]
        y_r = x_r * cos_f_reshaped - x_i * sin_f_reshaped
        y_i = x_r * sin_f_reshaped + x_i * cos_f_reshaped
    elif x.ndim == 3:
        cos_f_reshaped = cos_f[None, :, :]
        sin_f_reshaped = sin_f[None, :, :]
        y_r = x_r * cos_f_reshaped - x_i * sin_f_reshaped
        y_i = x_r * sin_f_reshaped + x_i * cos_f_reshaped
    else:
        raise ValueError(f"Input x to RoPE has unexpected ndim: {x.ndim}. Shape: {x.shape}")
    return jnp.concatenate((y_r, y_i), axis=-1)


def _apply_rotary_pos_emb_with_ids(x: jnp.ndarray, freqs_cis: jnp.ndarray, position_ids: jnp.ndarray):
    """Apply RoPE using custom position IDs for packed sequences.

    Args:
        x: Input tensor of shape (batch, seq_len, dim) or (batch, seq_len, n_heads, dim)
        freqs_cis: Precomputed frequencies of shape (max_seq_len, dim//2, 2)
        position_ids: Position IDs of shape (batch, seq_len)
    """
    batch_size = x.shape[0]
    seq_len = x.shape[1]

    flat_pos = position_ids.reshape(-1)

    cos_gathered = freqs_cis[flat_pos, :, 0]
    sin_gathered = freqs_cis[flat_pos, :, 1]

    cos_f = cos_gathered.reshape(batch_size, seq_len, -1)
    sin_f = sin_gathered.reshape(batch_size, seq_len, -1)

    x_r, x_i = jnp.split(x, 2, axis=-1)

    if x.ndim == 4:
        cos_f = cos_f[:, :, None, :]
        sin_f = sin_f[:, :, None, :]

    y_r = x_r * cos_f - x_i * sin_f
    y_i = x_r * sin_f + x_i * cos_f

    return jnp.concatenate((y_r, y_i), axis=-1)


def _compute_inv_freqs(dim: int, theta: float = 10000.0, dtype=jnp.float32):
    inv_freq = 1.0 / (theta ** (jnp.arange(0, dim, 2, dtype=dtype) / dim))
    return inv_freq


def _precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0, dtype=jnp.float32):
    inv_freqs = _compute_inv_freqs(dim, theta, dtype)
    positions = jnp.arange(seq_len, dtype=dtype)
    freqs = jnp.einsum("i,j->ij", positions, inv_freqs)
    return jnp.stack((jnp.cos(freqs), jnp.sin(freqs)), axis=-1)
