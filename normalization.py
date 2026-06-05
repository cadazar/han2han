#!/usr/bin/env python3
# coding: utf-8
"""Normalization layers.

Provides the RMSNorm used throughout the encoder/decoder, including the SubLN
placement before attention/FFN output projections (``use_sub_ln``).
"""

from typing import Optional

import jax.numpy as jnp
from flax import nnx


class RMSNorm(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        rngs: nnx.Rngs = None,
        dtype: jnp.dtype = jnp.float32,
        sharding: tuple = ('model',),
        use_bias: bool = False,
        param_dtype: Optional[jnp.dtype] = None,
    ):
        self.eps = eps
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        if param_dtype is None:
            param_dtype = dtype
        key = rngs.params()
        self.scale = nnx.Param(nnx.with_partitioning(nnx.initializers.ones, ((sharding[0],) if sharding else None)
                                           )(key, (hidden_size,), param_dtype))
        if use_bias:
            self.bias = nnx.Param(nnx.with_partitioning(nnx.initializers.zeros, ((sharding[0],) if sharding else None)
                                               )(key, (hidden_size,), param_dtype))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # compute rms in f32 for stability regardless of x.dtype, then normalize
        # and scale in x.dtype. scale/bias are stored at param_dtype (may differ
        # from x.dtype) so we promote them here (MaxText / standard Flax pattern).
        x_squared = jnp.square(x.astype(jnp.float32))
        mean_squared = jnp.mean(x_squared, axis=-1, keepdims=True)
        rms = jnp.sqrt(mean_squared + self.eps).astype(x.dtype)
        x_norm = x / rms
        scale = self.scale.astype(x.dtype)
        if self.use_bias:
            return x_norm * scale + self.bias.astype(x.dtype)
        return x_norm * scale
