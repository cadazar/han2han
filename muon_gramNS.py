"""Muon optimizer variant used by the Han2Han pre-training script.

``scale_by_muon_with_variant`` mirrors ``optax.contrib.scale_by_muon`` (same
beta/Nesterov/coeffs/dimension-number contract) with two differences: it uses
literal heavy-ball Nesterov momentum (matching the Moonlight and Dao Gram NS
reference implementations) rather than optax's EMA + bias_correction pattern,
and it can swap the inner orthogonalization for the Gram variant below:

``orthogonalize_via_gram_newton_schulz`` (Zhang/Amsel/Chen/Dao 2026,
https://dao-ailab.github.io/blog/2026/gram-newton-schulz/) iterates on the
small Gram matrix ``R = X X^T`` (n x n where n = min(rows, cols)) rather than
on X itself, accumulating ``Q`` and only multiplying ``Q @ X`` at the very
end. With ``POLAR_EXPRESS_COEFFICIENTS`` and a restart at iteration 2
(paper-recommended), this is a drop-in for the inner ``_orthogonalize`` and
saves FLOPs whenever rows != cols. The Dao reference always casts the
iteration body to fp16 even on the torch fallback path (not just for the
symmetric-GEMM kernel) -- so fp16/fp32 is doing real numerical work that bf16
may not match. We expose the working dtype via ``gram_work_dtype`` so callers
can A/B; the Frobenius prologue always runs in fp32 regardless to prevent
underflow on sparse inputs.

The factory preserves optax's vmap-over-leading-axes contract: for an N-D
weight tensor we treat the last two axes as ``(reduction, output)`` and vmap
over all preceding axes (covers stacked/scanned weights with extra leading
axes). Per-leaf state inherits the parameter's sharding by construction.
Inputs are NaN/inf-guarded before entering persistent state.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple, Union

import chex
import jax
import jax.numpy as jnp
import optax
import optax.tree
from optax._src import base
from optax._src import numerics
from optax.contrib._muon import (
    MuonDimensionNumbers,
    _compute_muon_reshape,
    _is_weight_dim_nums,
    _shape_factor,
)


WeightDimNumOrFn = Union[
    MuonDimensionNumbers,
    base.Params,
    Callable[[base.Params], "base.Params | None"],
]

# Polar Express coefficients (Dao et al. 2026, blog/paper). Safety-factor
# rescaled like the upstream Python copy so the iteration is stable for
# singular values starting in [0, 1].
_POLAR_EXPRESS_RAW = (
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
)
_POLAR_SAFETY = 1.05
POLAR_EXPRESS_COEFFICIENTS: Tuple[Tuple[float, float, float], ...] = tuple(
    (a / _POLAR_SAFETY, b / _POLAR_SAFETY**3, c / _POLAR_SAFETY**5)
    for (a, b, c) in _POLAR_EXPRESS_RAW
)


def _gram_orthogonalize_2d(
    X: jax.Array,
    ns_coeffs: jax.Array,
    reset_iters: Tuple[int, ...],
    eps: float,
    work_dtype: jnp.dtype,
) -> jax.Array:
    """Gram Newton-Schulz on a single 2D matrix ``(m, n)`` with ``m <= n``.

    Caller is responsible for transposing so the shorter axis comes first
    (matches optax's NS convention and lets ``R = X X^T`` be the smaller
    Gram matrix). ``ns_coeffs`` has shape ``(num_steps, 3)``; coefficients
    are per-iteration. ``reset_iters`` lists 0-indexed iteration indices
    immediately BEFORE which to recompute ``R`` from ``Q @ X`` to bound
    Q-drift. The paper uses ``[2]`` with five Polar-Express steps.
    """
    original_dtype = X.dtype

    # Step 1: normalize Frobenius to [0, 1] singular-value range. The norm step
    # runs in fp32 regardless of work_dtype: in bf16 storage, eps values below
    # ~7.8e-3 round to 0, so the Frobenius floor is cosmetic, and on sparse
    # inputs (e.g., starved MoE expert slabs) the norm itself can underflow to
    # 0, producing 0/0 = NaN through the iteration. Casting to fp32 for the
    # norm reproduces the Dao reference (gram_newton_schulz.py:96 explicit
    # `.float()` before the prologue) and prevents both failure modes.
    X_fp32 = X.astype(jnp.float32)
    X_fp32 = X_fp32 / (jnp.linalg.norm(X_fp32) + jnp.float32(eps))
    X = X_fp32.astype(work_dtype)

    R = X @ X.T  # (m, m) Gram matrix.
    m = X.shape[0]
    I_m = jnp.eye(m, dtype=work_dtype)
    Q = None  # lazy: avoid materializing a dead (batch, m, m) zeros tensor.

    num_steps = ns_coeffs.shape[0]
    reset_set = set(int(r) for r in reset_iters)

    # Loop is unrolled at trace time. ns_coeffs is a static (num_steps, 3)
    # tracer; reset_set comes from Python ints at construction time.
    for i in range(num_steps):
        a = ns_coeffs[i, 0].astype(work_dtype)
        b = ns_coeffs[i, 1].astype(work_dtype)
        c = ns_coeffs[i, 2].astype(work_dtype)

        if i in reset_set and i != 0:
            # restart: materialize the partially-orthogonalized X, recompute
            # R from scratch, drop Q so the next iter rebuilds it from Z + a*I.
            X = Q @ X
            R = X @ X.T
            Q = None

        Z = b * R + c * (R @ R)
        if Q is None:
            Q = Z + a * I_m
        else:
            Q = Q @ Z + a * Q

        # Stop updating R on the very last step or just before a restart;
        # both cases throw R away.
        update_R = (i < num_steps - 1) and ((i + 1) not in reset_set)
        if update_R:
            RZ = R @ Z + a * R
            R = Z @ RZ + a * RZ

    X = Q @ X
    return X.astype(original_dtype)


def orthogonalize_via_gram_newton_schulz(
    x: jax.Array,
    ns_coeffs: jax.Array,
    reset_iters: Tuple[int, ...] = (2,),
    eps: float = 1e-7,
    work_dtype: jnp.dtype = jnp.bfloat16,
    dimension_numbers: Optional[MuonDimensionNumbers] = None,
    chunk_size: int = 0,
) -> jax.Array:
    """Drop-in for ``optax.contrib._muon.orthogonalize_via_newton_schulz``.

    Same external contract: takes an N-D weight tensor + dimension numbers,
    flattens leading batch axes and vmaps the inner orthogonalization. The
    inner kernel is Gram NS (operates on ``X X^T``) rather than standard NS
    (operates on ``X`` directly).

    ``chunk_size`` (default 0) controls how the leading batch axis is
    processed. ``0`` means full ``jax.vmap`` (every batch element is
    processed in parallel; trace materializes ``(batch, m, m)`` Gram
    intermediates simultaneously). Any positive value uses
    ``jax.lax.map(..., batch_size=chunk_size)``, which processes the batch
    in chunks of ``chunk_size`` sequentially via inner vmap. Peak HBM for
    the Gram intermediates scales like ``chunk_size / batch``. Useful when
    a large MoE leading axis (e.g. ``num_experts * num_sparse_layers``)
    pushes the trace over the device HBM cap.
    """
    if ns_coeffs.ndim != 2 or ns_coeffs.shape[-1] != 3:
        raise ValueError(
            "Gram NS expects ns_coeffs of shape (num_steps, 3), got "
            f"{ns_coeffs.shape}. The Polar-Express schedule has 5 distinct "
            "per-step triples; do not pass a flat (3,) coefficient."
        )

    if x.ndim != 2 and not isinstance(dimension_numbers, MuonDimensionNumbers):
        raise ValueError(
            "Gram NS requires either a 2D matrix or explicit dimension_numbers "
            f"for higher-rank tensors. Got shape={x.shape}, "
            f"dimension_numbers={dimension_numbers}."
        )
    if x.ndim == 2:
        dimension_numbers = MuonDimensionNumbers(reduction_axis=0, output_axis=1)

    def _per_matrix(x2d: jax.Array) -> jax.Array:
        transposed = False
        if x2d.shape[0] > x2d.shape[1]:
            x2d = x2d.T
            transposed = True
        out = _gram_orthogonalize_2d(x2d, ns_coeffs, reset_iters, eps, work_dtype)
        if transposed:
            out = out.T
        return out

    reshape_fn, inverse_fn = _compute_muon_reshape(x, dimension_numbers)
    flat = reshape_fn(x)
    if chunk_size and chunk_size < flat.shape[0]:
        # jax.lax.map streams the leading axis through inner vmap of width
        # `batch_size`, handling any non-divisible remainder on its own.
        out_flat = jax.lax.map(_per_matrix, flat, batch_size=chunk_size)
    else:
        out_flat = jax.vmap(_per_matrix)(flat)
    return inverse_fn(out_flat)


def _make_orthogonalize_call(
    ns_variant: str,
    ns_steps: int,
    eps: float,
    gram_reset_iters: Tuple[int, ...],
    gram_work_dtype: jnp.dtype,
    gram_chunk_size: int = 0,
) -> Callable[[jax.Array, jax.Array, MuonDimensionNumbers], jax.Array]:
    """Builds the inner orthogonalize callable for either NS variant.

    ``gram_chunk_size`` is forwarded to Gram NS and ignored for standard NS.
    """
    if ns_variant == "gram":
        def _orth(x, coeffs, dim_num):
            return orthogonalize_via_gram_newton_schulz(
                x, coeffs, gram_reset_iters, eps, gram_work_dtype, dim_num,
                chunk_size=gram_chunk_size,
            )
        return _orth
    elif ns_variant == "standard":
        from optax.contrib._muon import orthogonalize_via_newton_schulz

        def _orth(x, coeffs, dim_num):
            return orthogonalize_via_newton_schulz(
                x, coeffs, ns_steps, eps, dim_num,
            )
        return _orth
    else:
        raise ValueError(
            f"Unknown ns_variant={ns_variant!r}; expected 'standard' or 'gram'."
        )


def scale_by_muon_with_variant(
    ns_coeffs: Union[
        Tuple[float, float, float],
        Tuple[Tuple[float, float, float], ...],
    ],
    ns_steps: int,
    beta: float,
    eps: float,
    mu_dtype: Optional[chex.ArrayDType],
    *,
    nesterov: bool = True,
    weight_dimension_numbers: WeightDimNumOrFn | None = None,
    ns_variant: str = "standard",
    gram_reset_iters: Tuple[int, ...] = (2,),
    gram_work_dtype: jnp.dtype = jnp.bfloat16,
    gram_chunk_size: int = 0,
) -> base.GradientTransformation:
    """Muon transform with heavy-ball Nesterov + NaN-guarded inputs.

    Pre-2026-05-30 the ``ns_variant='standard'`` path delegated to
    ``optax.contrib._muon.scale_by_muon``, which uses EMA + bias_correction
    Nesterov. The Moonlight, NorMuon, and Dao Gram NS reference
    implementations all use literal heavy-ball Nesterov instead:

        buf <- beta * buf + g            (no (1-beta) factor)
        nesterov: g_used = g + beta * buf_new
        plain:    g_used = buf_new

    Heavy-ball weights the fresh gradient ~25% less than EMA + bias_correction
    at warmup -- this is the design difference that the May 2026 audits
    identified as a likely cause of the persistent spike-settle global-norm
    signature. We now run our own update_fn for both ``standard`` and
    ``gram`` variants; the only variant-specific piece is the inner
    orthogonalize callable (``_make_orthogonalize_call``).
    """
    from optax.contrib._muon import MuonState
    from optax._src import utils

    mu_dtype = utils.canonicalize_dtype(mu_dtype)
    orth_call = _make_orthogonalize_call(
        ns_variant, ns_steps, eps, gram_reset_iters, gram_work_dtype,
        gram_chunk_size=gram_chunk_size,
    )

    def init_fn(params):
        mu = optax.tree.zeros_like(params, dtype=mu_dtype)
        ns_coeffs_ = jnp.asarray(ns_coeffs)
        if ns_coeffs_.ndim > 2 or ns_coeffs_.shape[-1] != 3:
            raise ValueError(
                f"ns_coeffs must have shape (3,) or (n, 3), got {ns_coeffs_.shape}"
            )
        if ns_variant == "gram" and ns_coeffs_.ndim != 2:
            raise ValueError(
                "ns_variant='gram' requires per-step coefficients of shape "
                f"(num_steps, 3), got {ns_coeffs_.shape}. Pass "
                "POLAR_EXPRESS_COEFFICIENTS or a similar list of triples."
            )
        return MuonState(
            count=jnp.zeros([], jnp.int32),
            mu=mu,
            ns_coeffs=ns_coeffs_,
        )

    def update_fn(updates, state, params=None):
        del params
        if callable(weight_dimension_numbers):
            resolved_wdims = weight_dimension_numbers(updates)
        else:
            resolved_wdims = weight_dimension_numbers

        # NaN/inf guard on inputs: zero out any non-finite gradient before it
        # poisons persistent mu state. See N1 audit (May 2026).
        updates = jax.tree.map(
            lambda g: jnp.where(jnp.isfinite(g), g, jnp.zeros_like(g)),
            updates,
        )

        # Heavy-ball momentum (matches Moonlight / NorMuon / Dao references).
        new_mu = jax.tree.map(
            lambda m, g: beta * m.astype(g.dtype) + g, state.mu, updates,
        )
        count_inc = numerics.safe_increment(state.count)
        if nesterov:
            mu_hat = jax.tree.map(lambda g, m: g + beta * m, updates, new_mu)
        else:
            mu_hat = new_mu

        updates_out = jax.tree.map(
            lambda x, dn: orth_call(x, state.ns_coeffs, dn),
            mu_hat, resolved_wdims, is_leaf=_is_weight_dim_nums,
        )

        factors = jax.tree.map(
            _shape_factor, updates_out, resolved_wdims, is_leaf=_is_weight_dim_nums,
        )
        updates_out = jax.tree.map(
            lambda x, factor: jnp.sqrt(jnp.maximum(1, factor)) * x,
            updates_out, factors,
        )

        mu = optax.tree.cast(new_mu, mu_dtype)
        return updates_out, MuonState(
            count=count_inc,
            mu=mu,
            ns_coeffs=state.ns_coeffs,
        )

    return base.GradientTransformation(init_fn, update_fn)


def resolve_ns_coeffs(
    ns_variant: str,
    hybrid_ns: bool,
) -> Tuple[Tuple[Tuple[float, float, float], ...] | Tuple[float, float, float], int]:
    """Returns (ns_coeffs, ns_steps) for the requested variant.

    standard + hybrid_ns=False : 5-step Keller-Jordan (single triple)
    standard + hybrid_ns=True  : 10-step DeepSeek hybrid (8 fast + 2 stab)
    gram     + (ignores hybrid) : 5-step Polar-Express, restart at 2

    The hybrid flag is ignored under Gram NS because Polar-Express tunes
    each iteration to a specific singular-value distribution; mixing
    iteration counts with the restart pattern is not validated.
    """
    if ns_variant == "gram":
        return POLAR_EXPRESS_COEFFICIENTS, 5
    if hybrid_ns:
        coeffs = tuple(
            [(3.4445, -4.7750, 2.0315)] * 8
            + [(2.0, -1.5, 0.5)] * 2
        )
        return coeffs, 10
    return (3.4445, -4.7750, 2.0315), 5
