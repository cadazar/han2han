import logging
import pickle
import re
from typing import Any, Dict

import numpy as np
import torch
import jax.numpy as jnp

logger = logging.getLogger(__name__)

__all__ = [
    "load_flax_state_dict",
    "convert_to_pt",
    "convert_to_jax",
    "flatten_dict",
]


_SCAN_KEY_RE = re.compile(r'^(?P<prefix>.+\.h)\._scan_stacks\.(?P<g>\d+)\.(?P<rest>.+)$')


def _unroll_scan_stacks(flat: Dict[str, Any]) -> Dict[str, Any]:
    """Expand Flax `_scan_stacks` per-group `[repeats, ...]` arrays into
    per-layer keys at `<prefix>.layers.<layer_idx>.<rest>`.

    Flax stores scanned encoder/decoder bodies as
        encoder.h._scan_stacks.<g>.<param_path>   shape=[repeats, ...]
        decoder.h._scan_stacks.<g>.<param_path>   shape=[repeats, ...]
    PyTorch's Han2HanBlockCollection.layers is a flat ModuleList[n_layers],
    so we slice axis 0 out and emit one key per layer. Scan groups are
    visited in numeric `<g>` order and layer indices accumulate across
    groups, matching how Flax assigns each group a contiguous layer range.

    Bookend (unscanned) keys at `<prefix>.layers.<i>.<param_path>` pass
    through unchanged. Keys that don't match the scan-stack pattern also
    pass through.

    Raises if any leaf in a scan group has a leading dim different from
    the rest of the group -- this would indicate a malformed checkpoint
    rather than a recoverable case (CLAUDE.md "fail loudly").
    """
    by_group: Dict[tuple, list] = {}
    passthrough: Dict[str, Any] = {}

    for k, v in flat.items():
        m = _SCAN_KEY_RE.match(k)
        if m is None:
            passthrough[k] = v
            continue
        prefix = m.group('prefix')
        g_idx = int(m.group('g'))
        rest = m.group('rest')
        by_group.setdefault((prefix, g_idx), []).append((k, rest, v))

    if not by_group:
        return passthrough

    out: Dict[str, Any] = dict(passthrough)
    side_to_groups: Dict[str, list] = {}
    for (prefix, g_idx) in by_group:
        side_to_groups.setdefault(prefix, []).append(g_idx)

    for prefix, gs in side_to_groups.items():
        layer_offset = 0
        for g in sorted(gs):
            entries = by_group[(prefix, g)]
            sample_arr = entries[0][2]
            sample_arr_np = np.asarray(sample_arr)
            repeats = sample_arr_np.shape[0]
            for (orig_key, rest, arr) in entries:
                arr_np = np.asarray(arr)
                if arr_np.shape[0] != repeats:
                    raise ValueError(
                        f"scan-stack inconsistency at {orig_key}: leading dim "
                        f"{arr_np.shape[0]} does not match group repeats={repeats}"
                    )
                for r in range(repeats):
                    layer_idx = layer_offset + r
                    new_key = f"{prefix}.layers.{layer_idx}.{rest}"
                    if new_key in out:
                        raise ValueError(
                            f"scan-stack unroll would overwrite existing key {new_key}; "
                            f"check for bookend/scan layer-index collision"
                        )
                    out[new_key] = arr_np[r]
            layer_offset += repeats

    return out


# ---------------------------------------------------------------------------
# Helpers -------------------------------------------------------------------
# ---------------------------------------------------------------------------

def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """Recursively flattens a nested dictionary using *sep* to join keys.

    For example::

        {
            "encoder": {"wte": {"embedding": arr}}
        }

    becomes::

        {"encoder.wte.embedding": arr}

    Args:
        d: Nested *dict*-like structure (values can themselves be *dict*s).
        parent_key: Prepended key path during recursion.
        sep: Delimiter between path components.
    Returns:
        A new dictionary with flattened keys.
    """
    items: Dict[str, Any] = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def _translate_key_for_pt(key: str) -> str:
    """Maps Flax-style leaf keys to their PyTorch equivalents.

    The saved Flax state dict uses *kernel* / *bias* / *scale* / *embedding*.
    In PyTorch these correspond to *weight* / *bias* / *weight* / *weight*.

    Special handling for subword embeddings:
    - Flax: encoder.subword_lookups.jbu / decoder.subword_lookups.jbu
    - PyTorch: encoder.jbu / decoder.jbu (module-level buffers)

    Note: Keep the encoder/decoder prefix to match PyTorch model structure
    """
    # handle subword lookups - preserve encoder/decoder prefix
    if "subword_lookups.jbu" in key:
        # replace subword_lookups.jbu.value with just jbu, keeping encoder/decoder prefix
        return key.replace("subword_lookups.jbu.value", "jbu").replace("subword_lookups.jbu", "jbu")
    if "subword_lookups.cbu" in key:
        return key.replace("subword_lookups.cbu.value", "cbu").replace("subword_lookups.cbu", "cbu")

    if "kernel" in key:
        key = key.replace("kernel", "weight")
    if "scale" in key:
        key = key.replace("scale", "weight")
    if "embedding" in key:
        key = key.replace("embedding", "weight")
    if key.endswith(".value"):
        return key[: -len(".value")]
    # otherwise leave unchanged (e.g. *.bias* or already *.weight*)
    return key


# ---------------------------------------------------------------------------
# Loading / Conversion -------------------------------------------------------
# ---------------------------------------------------------------------------

def load_flax_state_dict(path: str) -> Dict[str, Any]:
    """Loads the pickled **pure-NumPy** state dict written by the debug snippet.

    Returns the raw nested dictionary with NumPy arrays. No type conversion
    is performed at this stage.
    """
    with open(path, "rb") as f:
        raw_dict = pickle.load(f)
    return raw_dict


def convert_to_pt(raw_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Converts the *raw* state-dict to a PyTorch compatible version.

    * Flattens the nested dict into dotted keys.
    * Expands Flax scan-stacks (`._scan_stacks.<g>.` with leading [repeats, ...])
      into per-layer keys at `.layers.<i>.` matching the PT ModuleList layout.
    * Renames leaf suffixes (kernel/scale/embedding -> weight).
    * Converts NumPy arrays to `torch.Tensor`, preserving *dtype* where
      possible (mapping bfloat16 -> torch.bfloat16, fall-back to float32).
    * Handles subword lookups (jbu/cbu) which are stored in encoder/decoder modules.
    """
    flat = flatten_dict(raw_dict)
    flat = _unroll_scan_stacks(flat)
    pt_state: Dict[str, torch.Tensor] = {}
    for k, arr in flat.items():
        new_key = _translate_key_for_pt(k)
        np_arr = np.asarray(arr)
        try:
            tensor = torch.from_numpy(np_arr.copy())
        except TypeError as e:
            logger.error(
                "dtype conversion failed for %s (np dtype=%s): %s; falling back to float32",
                k, np_arr.dtype, e,
            )
            tensor = torch.from_numpy(np_arr.astype(np.float32))

        pt_state[new_key] = tensor
    return pt_state


def convert_to_jax(raw_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Converts the *raw* state-dict to one backed by *jax.numpy* arrays.

    The nested structure and keys are preserved; only values are converted.
    """
    def _to_jax(x):
        return jnp.asarray(x)

    def _recurse(obj):
        if isinstance(obj, dict):
            return {k: _recurse(v) for k, v in obj.items()}
        return _to_jax(obj)

    return _recurse(raw_dict) 