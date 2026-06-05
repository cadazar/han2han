#!/usr/bin/env python3
# coding: utf-8
"""
RoPE utilities for handling packed sequences with position resets.
"""

import numpy as np
import jax.numpy as jnp


def create_packed_position_ids(segment_ids: jnp.ndarray, max_length: int) -> jnp.ndarray:
    """
    Create position IDs that reset at document boundaries.

    Args:
        segment_ids: (batch_size, seq_len) array where each unique value represents a document
        max_length: Maximum sequence length

    Returns:
        position_ids: (batch_size, seq_len) array with positions resetting at boundaries
    """
    batch_size, seq_len = segment_ids.shape
    position_ids = jnp.zeros_like(segment_ids)

    for b in range(batch_size):
        current_pos = 0
        prev_segment = segment_ids[b, 0]

        for i in range(seq_len):
            if segment_ids[b, i] != prev_segment:
                # New document, reset position
                current_pos = 0
                prev_segment = segment_ids[b, i]

            # Skip padding (segment_id = 0)
            if segment_ids[b, i] > 0:
                position_ids = position_ids.at[b, i].set(current_pos)
                current_pos += 1

    return position_ids


def apply_rotary_pos_emb_packed(
    x: jnp.ndarray,
    freqs_cis: jnp.ndarray,
    position_ids: jnp.ndarray
):
    """
    Apply RoPE with packed sequences using custom position IDs.

    Args:
        x: Input tensor of shape (batch, seq_len, n_heads, dim) or (batch, seq_len, dim)
        freqs_cis: Precomputed frequencies of shape (max_seq_len, dim//2, 2)
        position_ids: Position IDs of shape (batch, seq_len)

    Returns:
        Rotated tensor with same shape as input
    """
    # split into real and imaginary parts
    x_r, x_i = jnp.split(x, 2, axis=-1)

    # gather frequencies based on position_ids instead of sequential positions
    # position_ids: (batch, seq_len)
    # freqs_cis: (max_seq_len, dim//2, 2)

    # use take_along_axis for proper indexing
    batch_size, seq_len = position_ids.shape

    # expand position_ids to match frequency dimensions
    if x.ndim == 4:  # (batch, seq_len, n_heads, dim)
        # need to broadcast position_ids: (batch, seq_len) -> (batch, seq_len, 1, 1)
        pos_expanded = position_ids[:, :, None, None]
        # gather cos and sin values
        cos_f = jnp.take_along_axis(freqs_cis[..., 0][None, ...], pos_expanded, axis=1)
        sin_f = jnp.take_along_axis(freqs_cis[..., 1][None, ...], pos_expanded, axis=1)
    elif x.ndim == 3:  # (batch, seq_len, dim)
        # position_ids: (batch, seq_len) -> (batch, seq_len, 1)
        pos_expanded = position_ids[:, :, None]
        cos_f = jnp.take_along_axis(freqs_cis[..., 0][None, ...], pos_expanded, axis=1)
        sin_f = jnp.take_along_axis(freqs_cis[..., 1][None, ...], pos_expanded, axis=1)
    else:
        raise ValueError(f"Input x to RoPE has unexpected ndim: {x.ndim}. Shape: {x.shape}")

    # apply rotation
    y_r = x_r * cos_f - x_i * sin_f
    y_i = x_r * sin_f + x_i * cos_f

    return jnp.concatenate((y_r, y_i), axis=-1)


def create_packed_attention_mask(
    segment_ids,
    causal: bool = False,
    dtype=np.float32
):
    """
    Create attention mask that prevents cross-document attention.
    Uses numpy to avoid materializing O(S^2) masks on TPU HBM.

    Args:
        segment_ids: (batch_size, seq_len) array where each unique value represents a document
        causal: Whether to add causal masking (for decoder)
        dtype: Output dtype

    Returns:
        mask: (batch_size, 1, seq_len, seq_len) attention mask
    """
    segment_ids = np.asarray(segment_ids)
    batch_size, seq_len = segment_ids.shape

    # create document mask: tokens can only attend within same segment
    document_mask = segment_ids[:, None, :] == segment_ids[:, :, None]  # (batch, seq_len, seq_len)

    # handle padding: segment_id = 0 means padding
    is_padding = segment_ids == 0  # (batch, seq_len)
    padding_mask = ~(is_padding[:, None, :] | is_padding[:, :, None])  # (batch, seq_len, seq_len)

    mask = document_mask & padding_mask

    if causal:
        causal_mask = np.tril(np.ones((seq_len, seq_len), dtype=bool))
        mask = mask & causal_mask[None, :, :]

    # (batch, 1, seq_len, seq_len), 1.0 = allowed, 0.0 = blocked
    attention_mask = np.where(mask[:, None, :, :], 1.0, 0.0).astype(dtype)

    return attention_mask
