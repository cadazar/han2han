#!/usr/bin/env python3
# coding: utf-8
"""
Standardized checkpoint save/restore utilities for Han2Han training scripts.

This module provides unified checkpoint handling for Flax 0.12+ models, properly
handling RngState separation and supporting both SPMD and pmap training paradigms.

Key features:
- Proper RngState filtering for Flax 0.12 compatibility
- Support for subword lookups (jbu/cbu) embedded in model state
- Consistent handling of optimizer states across different training scripts
- Clear error messages and logging for debugging
"""

import logging
import json
from typing import Dict, Any, Tuple, Optional, Union
import base64

import jax
import jax.numpy as jnp
from flax import nnx
import orbax.checkpoint as ocp
from etils import epath
import numpy as np

from orbax.checkpoint.transform_utils import RestoreTransform
from orbax.checkpoint.checkpoint_utils import construct_restore_args
from jax.sharding import NamedSharding, PartitionSpec
from logging_utils import log_from_main_process

logger = logging.getLogger(__name__)


def _align_pspecs_to_abs(abs_state, pspecs):
    """Expand collapsed pspecs subtrees so the treedef matches abs_state.

    nnx.get_partition_spec collapses subtrees whose leaves all carry the same
    sharding metadata into a single bare terminal (None when every leaf is
    unsharded; a single PartitionSpec when every leaf shares the same spec),
    which breaks subsequent jax.tree.map(..., abs_state, pspecs) calls because
    abs_state still carries the original NamedTuple/dict structure. Two cases
    we've seen so far:

    - optax.contrib.muon hybrid state: the muon branch is None on params that
      flow through the Adam companion (and vice versa), so the entire branch
      collapses to None on the spec side.
    - MoE optimizer state: a NamedTuple of leaves all derived from one expert
      param shares the param's spec; nnx returns the bare PartitionSpec.

    Walk pspecs treating None and PartitionSpec as terminals; wherever such a
    terminal lines up with an abs subtree of structure, expand it into a tree
    that mirrors that structure. None broadcasts to None-leaves; a
    PartitionSpec broadcasts to itself at every leaf (rank-mismatched leaves
    are truncated to PartitionSpec() by _make_shape_safe_sharding downstream,
    so the broadcast is safe for mixed-rank subtrees).
    """
    def _expand(p, a):
        if p is None:
            return jax.tree.map(lambda _: None, a)
        if isinstance(p, PartitionSpec):
            return jax.tree.map(lambda _: p, a)
        return p

    return jax.tree.map(
        _expand,
        pspecs,
        abs_state,
        is_leaf=lambda x: x is None or isinstance(x, PartitionSpec),
    )


def _make_shape_safe_sharding(arr_shape, spec, mesh):
    """Create sharding compatible with array shape.

    For each spec axis, check if the array dimension can be evenly divided
    by the mesh axis size. If not, replace with None (replicated).

    This prevents Adafactor factored stats (v_row shape [N], v_col shape [M])
    from inheriting incompatible 2D partition specs from their parent parameters.
    """
    if spec is None:
        return NamedSharding(mesh, PartitionSpec())

    spec_tuple = tuple(spec) if hasattr(spec, '__iter__') else (spec,)

    safe_spec = []
    for i, axis_name in enumerate(spec_tuple):
        if i >= len(arr_shape):
            break

        if axis_name is None:
            safe_spec.append(None)
            continue

        if isinstance(axis_name, tuple):
            mesh_size = 1
            for ax in axis_name:
                if ax in mesh.axis_names:
                    mesh_size *= mesh.shape[ax]
        elif axis_name in mesh.axis_names:
            mesh_size = mesh.shape[axis_name]
        else:
            mesh_size = 1

        if arr_shape[i] >= mesh_size and arr_shape[i] % mesh_size == 0:
            safe_spec.append(axis_name)
        else:
            safe_spec.append(None)

    return NamedSharding(mesh, PartitionSpec(*safe_spec))


def split_model_for_checkpoint(model: nnx.Module, skip_lm_head: bool = False) -> Tuple[Any, Any, Any]:
    """
    Split model for checkpoint saving, separating RngState which can't be serialized.

    This is required for Flax 0.12+ as RngState contains JAX PRNG keys that
    cannot be directly serialized by Orbax.

    Args:
        model: The NNX model to split
        skip_lm_head: If True, exclude lm_head from model_state (for tied->untied conversion)

    Returns:
        Tuple of (graphdef, rng_state, model_state) where model_state excludes RngState
        (and optionally excludes lm_head if skip_lm_head=True)
    """
    if skip_lm_head:
        # split out RngState and lm_head separately, then everything else
        graphdef, rng_state, lm_head_state, model_state = nnx.split(
            model, nnx.RngState, nnx.All(nnx.PathContains("lm_head")), ...
        )
        # debug: log what was filtered
        if hasattr(lm_head_state, 'flat_state'):
            lm_flat = lm_head_state.flat_state()
            lm_head_keys = list(lm_flat._keys) if hasattr(lm_flat, '_keys') else []
        else:
            lm_head_keys = 'N/A'
        if hasattr(model_state, 'flat_state'):
            m_flat = model_state.flat_state()
            model_keys = list(m_flat._keys)[:10] if hasattr(m_flat, '_keys') else []
        else:
            model_keys = 'N/A'
        log_from_main_process(logger, 'info',
            f"split_model_for_checkpoint: skip_lm_head=True, filtered lm_head keys: {lm_head_keys}, model_state first 10 keys: {model_keys}")
    else:
        # split out RngState first, then everything else
        graphdef, rng_state, model_state = nnx.split(model, nnx.RngState, ...)
    return graphdef, rng_state, model_state


def save_checkpoint(
    checkpoint_manager: ocp.CheckpointManager,
    model: nnx.Module,
    optimizers: Union[nnx.Optimizer, Tuple[nnx.Optimizer, nnx.Optimizer]],
    metadata: Dict[str, Any],
    step: int,
    force: bool = False,
    use_legacy_subwords: bool = False,
    metrics: Optional[Dict[str, float]] = None,
) -> None:
    """
    Save a checkpoint with proper Flax 0.12 handling.

    Args:
        checkpoint_manager: Orbax checkpoint manager instance
        model: The NNX model to save
        optimizers: Single optimizer or tuple of (optimizer_with_wd, optimizer_no_wd)
        metadata: Dictionary of metadata to save (tokens_seen, global_step, etc.)
        step: The current training step
        force: Whether to force save even if not at a regular interval
        use_legacy_subwords: If True, also save jbu.npy/cbu.npy separately (deprecated)
    """
    # filter out RngState when saving - it can't be serialized
    # split model into graphdef, RngState (discarded), and everything else
    _, _, model_state = split_model_for_checkpoint(model)

    # handle single or dual optimizer case
    if isinstance(optimizers, tuple):
        # dual optimizer case
        optimizer_wd, optimizer_no_wd = optimizers
        opt_wd_state = nnx.state(optimizer_wd)
        opt_no_wd_state = nnx.state(optimizer_no_wd)

        saved = checkpoint_manager.save(
            step,
            args=ocp.args.Composite(
                model=ocp.args.StandardSave(model_state),
                optimizer_wd=ocp.args.StandardSave(opt_wd_state),
                optimizer_no_wd=ocp.args.StandardSave(opt_no_wd_state),
                meta=ocp.args.JsonSave(metadata)
            ),
            metrics=metrics,
            force=force,
        )
    else:
        # single optimizer case
        opt_state = nnx.state(optimizers)

        saved = checkpoint_manager.save(
            step,
            args=ocp.args.Composite(
                model=ocp.args.StandardSave(model_state),
                optimizer=ocp.args.StandardSave(opt_state),
                meta=ocp.args.JsonSave(metadata)
            ),
            metrics=metrics,
            force=force,
        )

    # only log if checkpoint was actually saved (not skipped by interval)
    if saved:
        log_from_main_process(logger, 'info', f"Checkpoint saved at step {step}")

    # deprecated: save legacy subword files if requested
    if use_legacy_subwords:
        _save_legacy_subwords(checkpoint_manager.directory, model_state)


def _merge_optimizer_states_into_single(
    optimizer: nnx.Optimizer,
    state_wd: Any,
    state_no_wd: Any
) -> None:
    """
    Merge dual optimizer states into a single optimizer.

    The dual optimizers used `wrt` filters to track disjoint parameter sets:
    - optimizer_wd: weights/kernels (with weight decay)
    - optimizer_no_wd: biases, layer norms, embeddings (without weight decay)

    Since the parameter sets are disjoint, state_wd + state_no_wd = complete state.

    Args:
        optimizer: The single optimizer to merge states into (modified in-place)
        state_wd: Optimizer state from the weight-decay optimizer (partial dict)
        state_no_wd: Optimizer state from the no-weight-decay optimizer (partial dict)
    """
    # combine the two partial state dicts - they have disjoint keys
    # so we can just update one with the other
    combined_dict = {}

    # recursively merge the nested dicts
    def merge_nested_dicts(d1, d2):
        """Merge two nested dicts, combining all keys."""
        result = dict(d1) if d1 else {}
        if d2:
            for key, value in d2.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = merge_nested_dicts(result[key], value)
                else:
                    result[key] = value
        return result

    combined_dict = merge_nested_dicts(state_wd, state_no_wd)

    # convert to nnx.State and update optimizer
    # the combined dict should match the optimizer's full state structure
    graphdef, _ = nnx.split(optimizer)
    merged_optimizer = nnx.merge(graphdef, combined_dict)
    nnx.update(optimizer, nnx.state(merged_optimizer))

    log_from_main_process(logger, 'info', "Merged dual optimizer states into single optimizer")


def _detect_checkpoint_structure(checkpoint_manager: ocp.CheckpointManager, step: int) -> Dict[str, bool]:
    """
    Detect the structure of a checkpoint (single vs dual optimizer).

    Args:
        checkpoint_manager: Orbax checkpoint manager
        step: Checkpoint step to inspect

    Returns:
        Dict with detection results: {'has_dual_optimizers': bool, 'has_single_optimizer': bool}
    """
    ckpt_dir = checkpoint_manager.directory / str(step)

    # check for dual optimizer structure
    has_optimizer_wd = (ckpt_dir / 'optimizer_wd').exists()
    has_optimizer_no_wd = (ckpt_dir / 'optimizer_no_wd').exists()
    has_dual = has_optimizer_wd and has_optimizer_no_wd

    # check for single optimizer structure
    has_single = (ckpt_dir / 'optimizer').exists()

    return {
        'has_dual_optimizers': has_dual,
        'has_single_optimizer': has_single
    }


def restore_checkpoint(
    checkpoint_manager: ocp.CheckpointManager,
    model: nnx.Module,
    optimizers: Optional[Union[nnx.Optimizer, Tuple[nnx.Optimizer, nnx.Optimizer], Dict[str, nnx.Optimizer]]] = None,
    step: Optional[int] = None,
    mesh: Optional[Any] = None,
    use_abstract_restoration: bool = True,
    model_only: bool = False,
    merge_dual_optimizers: bool = True,
    skip_lm_head: bool = False,
    checkpoint_transforms: Optional[Dict] = None
) -> Tuple[int, Dict[str, Any]]:
    """
    Restore a checkpoint with proper Flax 0.12 handling.

    This function handles the complex restoration process required by Flax 0.12,
    including proper RngState preservation and model state merging.

    Args:
        checkpoint_manager: Orbax checkpoint manager instance
        model: The NNX model to restore into
        optimizers: Single optimizer, tuple of (optimizer_with_wd, optimizer_no_wd),
                   dict with keys matching checkpoint, or None if model_only=True
        step: Specific step to restore (None for latest)
        mesh: Optional mesh for SPMD restoration
        use_abstract_restoration: If True, use abstract model for efficient restoration
        model_only: If True, only restore model (ignore optimizer states)
        merge_dual_optimizers: If True and checkpoint has dual optimizers but single
                              optimizer provided, automatically merge the states
        skip_lm_head: If True, skip restoring lm_head (for tied->untied conversion)
        checkpoint_transforms: Optional dict of regex-keyed RestoreTransform objects
                              for key remapping during restoration. Passed directly
                              to orbax's PyTreeRestore(transforms=...) so that orbax
                              handles both key remapping and resharding internally.
                              Use build_flaxformer_moe_transforms() for Flaxformer
                              MoE checkpoints.

    Returns:
        Tuple of (restored_step, metadata_dict)
    """
    if not model_only and optimizers is None:
        raise ValueError("optimizers must be provided when model_only=False")

    # determine step to restore
    if step is None:
        step = checkpoint_manager.latest_step()
        if step is None:
            logger.info("No checkpoint found to restore")
            return 0, {}

    # detect checkpoint structure
    ckpt_structure = _detect_checkpoint_structure(checkpoint_manager, step)
    has_dual_optimizers_in_ckpt = ckpt_structure['has_dual_optimizers']

    # parse optimizer input format
    single_optimizer_provided = False
    if optimizers is not None:
        if isinstance(optimizers, dict):
            # dict format - keys should match checkpoint
            optimizer_dict = optimizers
        elif isinstance(optimizers, tuple):
            # tuple format - two optimizers
            optimizer_wd, optimizer_no_wd = optimizers
            optimizer_dict = {'optimizer_wd': optimizer_wd, 'optimizer_no_wd': optimizer_no_wd}
        else:
            # single optimizer
            optimizer_dict = {'optimizer': optimizers}
            single_optimizer_provided = True
    else:
        optimizer_dict = {}

    # check if we need to merge dual optimizers from checkpoint
    should_merge = (
        merge_dual_optimizers and
        has_dual_optimizers_in_ckpt and
        single_optimizer_provided and
        not model_only
    )

    if should_merge:
        log_from_main_process(
            logger, 'info',
            f"Detected dual-optimizer checkpoint at step {step}, will merge into single optimizer"
        )

        # create temporary checkpoint manager with dual optimizer handlers for restoration
        # checkpoint_manager.directory points to checkpoints dir, get parent for setup
        ckpt_dir = checkpoint_manager.directory
        parent_dir = str(ckpt_dir.parent) if hasattr(ckpt_dir, 'parent') else str(ckpt_dir)[:-len('/checkpoints')]

        # detect if GCS path
        is_gcs = str(ckpt_dir).startswith('gs://')

        if is_gcs:
            restore_mgr = setup_checkpoint_manager(
                output_dir='.',  # dummy, won't be used
                gcs_output_dir=parent_dir,
                max_to_keep=1,
                single_optimizer=False  # needs dual optimizer handlers
            )
        else:
            restore_mgr = setup_checkpoint_manager(
                output_dir=parent_dir,
                max_to_keep=1,
                single_optimizer=False  # needs dual optimizer handlers
            )
    else:
        restore_mgr = checkpoint_manager

    logger.info(f"Restoring checkpoint from step {step}")

    if use_abstract_restoration and mesh is not None:
        restored_checkpoint = _restore_with_abstract_model(
            restore_mgr, model, optimizer_dict, step, mesh,
            model_only, should_merge, skip_lm_head, checkpoint_transforms
        )
    else:
        _, _, model_state = split_model_for_checkpoint(model, skip_lm_head=skip_lm_head)

        if checkpoint_transforms:
            restore_args = {
                'model': ocp.args.PyTreeRestore(
                    item=model_state,
                    transforms=checkpoint_transforms,
                    restore_args=construct_restore_args(model_state),
                ),
                'meta': ocp.args.JsonRestore(),
            }
            log_from_main_process(logger, 'info',
                "Using PyTreeRestore with transforms for checkpoint key remapping")
        elif skip_lm_head:
            log_from_main_process(logger, 'info', "Skipping lm_head restoration (tied->untied conversion)")
            restore_args = {'model': ocp.args.PyTreeRestore(model_state, partial_restore=True), 'meta': ocp.args.JsonRestore()}
        else:
            restore_args = {'model': ocp.args.StandardRestore(model_state), 'meta': ocp.args.JsonRestore()}

        if not model_only:
            if should_merge:
                restore_args['optimizer_wd'] = ocp.args.StandardRestore()
                restore_args['optimizer_no_wd'] = ocp.args.StandardRestore()
            else:
                for opt_name, opt in optimizer_dict.items():
                    opt_state = nnx.state(opt)
                    if checkpoint_transforms:
                        restore_args[opt_name] = ocp.args.PyTreeRestore(
                            item=opt_state,
                            transforms=checkpoint_transforms,
                            restore_args=construct_restore_args(opt_state),
                        )
                    elif skip_lm_head:
                        restore_args[opt_name] = ocp.args.StandardRestore(opt_state, strict=False)
                    else:
                        restore_args[opt_name] = ocp.args.StandardRestore(opt_state)

        restored_checkpoint = restore_mgr.restore(
            step,
            args=ocp.args.Composite(**restore_args)
        )

    metadata = restored_checkpoint.get('meta', {})

    nnx.update(model, restored_checkpoint['model'])

    if not model_only:
        if should_merge:
            # merge dual optimizer states into single optimizer
            single_opt = optimizer_dict['optimizer']
            _merge_optimizer_states_into_single(
                single_opt,
                restored_checkpoint['optimizer_wd'],
                restored_checkpoint['optimizer_no_wd']
            )
        else:
            # normal restore - update each optimizer from restored checkpoint
            for opt_name, opt in optimizer_dict.items():
                if opt_name in restored_checkpoint:
                    restored_opt = restored_checkpoint[opt_name]
                    # strict=False leaves ShapeDtypeStruct placeholders for missing keys;
                    # replace them with real zero arrays so nnx.update doesn't choke
                    restored_opt = jax.tree.map(
                        lambda x: jnp.zeros(x.shape, x.dtype) if isinstance(x, jax.ShapeDtypeStruct) else x,
                        restored_opt
                    )
                    nnx.update(opt, restored_opt)

    logger.info(f"Successfully restored checkpoint from step {step}")

    return step, metadata


def _restore_with_abstract_model(
    checkpoint_manager: ocp.CheckpointManager,
    model: nnx.Module,
    optimizer_dict: Dict[str, nnx.Optimizer],
    step: int,
    mesh: Any,
    model_only: bool = False,
    should_merge: bool = False,
    skip_lm_head: bool = False,
    checkpoint_transforms: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Restore using abstract model for memory efficiency.

    This is useful for large models where creating a full model just for
    restoration structure would be wasteful.

    Args:
        should_merge: If True, restore dual optimizers to be merged into single optimizer
        skip_lm_head: If True, skip restoring lm_head (for tied->untied conversion)
        checkpoint_transforms: Optional transforms dict for key remapping via PyTreeRestore
    """
    # create abstract states for restoration
    # split out RngState (and optionally lm_head) from model to match checkpoint structure
    _, _, model_state = split_model_for_checkpoint(model, skip_lm_head=skip_lm_head)

    if skip_lm_head:
        log_from_main_process(logger, 'info', "Skipping lm_head restoration (tied->untied conversion)")

    abs_model_state = nnx.eval_shape(lambda: model_state)

    # debug: log keys in abs_model_state
    if hasattr(abs_model_state, 'flat_state'):
        flat = abs_model_state.flat_state()
        abs_keys = list(flat._keys) if hasattr(flat, '_keys') else []
        has_lm_head = any('lm_head' in str(k) for k in abs_keys)
        log_from_main_process(logger, 'info',
            f"_restore_with_abstract_model: abs_model_state has {len(abs_keys)} keys, contains lm_head: {has_lm_head}")

    # create abstract optimizer states
    abs_opt_states = {}
    if not model_only and not should_merge:
        # normal case - one abstract state per optimizer
        for opt_name, opt in optimizer_dict.items():
            abs_opt_states[opt_name] = nnx.eval_shape(lambda: nnx.state(opt))
    # note: for should_merge case, we use empty restore in build_restore_args

    # apply sharding specs if using SPMD
    if mesh is not None:
        with mesh:
            # get partition specs from abstract model
            model_pspecs = nnx.get_partition_spec(abs_model_state)

            # create shape-safe sharding for restoration (ensure spec is compatible with shape)
            def make_model_sharding(arr, spec):
                if not hasattr(arr, 'shape'):
                    return NamedSharding(mesh, PartitionSpec())
                return _make_shape_safe_sharding(arr.shape, spec, mesh)

            model_pspecs = _align_pspecs_to_abs(abs_model_state, model_pspecs)
            abs_model_state = jax.tree.map(
                lambda a, spec: jax.ShapeDtypeStruct(
                    a.shape, a.dtype, sharding=make_model_sharding(a, spec)
                ) if hasattr(a, 'shape') else a,
                abs_model_state, model_pspecs
            )

            # apply shape-safe sharding to optimizer states
            # this is critical for Adafactor where factored stats (v_row, v_col) are 1D
            # but may inherit incompatible 2D partition specs from their parent parameters
            if not model_only:
                for opt_name in abs_opt_states:
                    opt_pspecs = nnx.get_partition_spec(abs_opt_states[opt_name])
                    opt_pspecs = _align_pspecs_to_abs(
                        abs_opt_states[opt_name], opt_pspecs
                    )
                    abs_opt_states[opt_name] = jax.tree.map(
                        lambda a, spec: jax.ShapeDtypeStruct(
                            a.shape, a.dtype, sharding=_make_shape_safe_sharding(a.shape, spec, mesh)
                        ) if hasattr(a, 'shape') else a,
                        abs_opt_states[opt_name], opt_pspecs
                    )

    if checkpoint_transforms:
        restore_args = {
            'model': ocp.args.PyTreeRestore(
                item=abs_model_state,
                transforms=checkpoint_transforms,
                restore_args=construct_restore_args(abs_model_state),
            )
        }
        log_from_main_process(logger, 'info',
            "Using PyTreeRestore with transforms for checkpoint key remapping")
    elif skip_lm_head:
        restore_args = {'model': ocp.args.PyTreeRestore(abs_model_state)}
        log_from_main_process(logger, 'info',
            "skip_lm_head=True, using PyTreeRestore with partial_restore=True")
    else:
        restore_args = {'model': ocp.args.StandardRestore(abs_model_state)}

    if not model_only:
        if should_merge:
            restore_args['optimizer_wd'] = ocp.args.StandardRestore()
            restore_args['optimizer_no_wd'] = ocp.args.StandardRestore()
        else:
            for opt_name, abs_opt_state in abs_opt_states.items():
                if checkpoint_transforms:
                    restore_args[opt_name] = ocp.args.PyTreeRestore(
                        item=abs_opt_state,
                        transforms=checkpoint_transforms,
                        restore_args=construct_restore_args(abs_opt_state),
                    )
                elif skip_lm_head:
                    restore_args[opt_name] = ocp.args.StandardRestore(abs_opt_state)
                else:
                    restore_args[opt_name] = ocp.args.StandardRestore(abs_opt_state)

    restore_args['meta'] = ocp.args.JsonRestore()

    # debug: log what we're about to restore
    log_from_main_process(logger, 'info',
        f"_restore_with_abstract_model: restore_args keys = {list(restore_args.keys())}")
    if 'model' in restore_args and hasattr(restore_args['model'], 'item') and restore_args['model'].item is not None:
        item = restore_args['model'].item
        if hasattr(item, 'flat_state'):
            flat = item.flat_state()
            item_keys = list(flat._keys) if hasattr(flat, '_keys') else []
            has_lm_head_in_item = any('lm_head' in str(k) for k in item_keys)
            log_from_main_process(logger, 'info',
                f"_restore_with_abstract_model: model item has {len(item_keys)} keys, contains lm_head: {has_lm_head_in_item}")

    # restore with abstract states - orbax places data directly on devices
    restored_checkpoint = checkpoint_manager.restore(
        step,
        args=ocp.args.Composite(**restore_args)
    )

    return restored_checkpoint


def _save_legacy_subwords(checkpoint_dir: epath.Path, model_state: Any) -> None:
    """
    Save jbu.npy and cbu.npy files separately for backwards compatibility.

    This is deprecated - subword lookups should be embedded in the model state.
    """
    try:
        # extract subword lookups from model state if they exist
        state_dict = model_state.to_pure_dict() if hasattr(model_state, 'to_pure_dict') else model_state

        for key in ['jbu', 'cbu']:
            if key in state_dict:
                lookup_data = state_dict[key]
                if lookup_data is not None:
                    # handle StaticLookup wrapper if present
                    if hasattr(lookup_data, 'value'):
                        lookup_data = lookup_data.value

                    # save as numpy file
                    lookup_path = checkpoint_dir / f"{key}.npy"
                    with lookup_path.open("wb") as f:
                        np.save(f, lookup_data)
                    logger.info(f"Legacy {key}.npy saved to {lookup_path}")
    except Exception as e:
        logger.warning(f"Could not save legacy subword files: {e}")


def prepare_metadata(
    tokens_seen: int,
    global_step: int,
    additional_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Prepare metadata dictionary for checkpoint saving.

    Args:
        tokens_seen: Number of tokens processed so far
        global_step: Current global training step
        additional_data: Optional additional metadata to include

    Returns:
        Dictionary ready for JSON serialization
    """
    metadata = {
        'tokens_seen': int(tokens_seen),
        'global_step': int(global_step)
    }

    if additional_data:
        # handle special data types that need encoding
        for key, value in additional_data.items():
            if isinstance(value, bytes):
                # encode bytes as base64 for JSON serialization
                metadata[key] = base64.b64encode(value).decode('ascii')
            elif isinstance(value, (np.ndarray, jnp.ndarray)):
                # convert arrays to lists
                metadata[key] = value.tolist()
            else:
                metadata[key] = value

    return metadata


def extract_tokens_from_metadata(metadata: Dict[str, Any]) -> int:
    """
    Extract tokens_seen from checkpoint metadata with proper type handling.

    Args:
        metadata: Metadata dictionary from checkpoint

    Returns:
        Number of tokens seen (int64)
    """
    tokens_seen = metadata.get('tokens_seen', 0)
    # ensure it's int64 to avoid overflow with large token counts
    return np.int64(tokens_seen)


def setup_checkpoint_manager(
    output_dir: str,
    gcs_output_dir: Optional[str] = None,
    max_to_keep: int = 5,
    save_interval_steps: int = 500,
    single_optimizer: bool = True,
    for_pretrained_restoration: bool = False,
    best_fn: Optional[callable] = None,
    best_mode: str = 'max',
) -> ocp.CheckpointManager:
    """
    Setup checkpoint manager with proper directory handling for GCS and local paths.

    Args:
        output_dir: Local output directory
        gcs_output_dir: Optional GCS path for checkpoints
        max_to_keep: Maximum number of checkpoints to keep
        save_interval_steps: How many steps to trigger actual saving
        single_optimizer: If True, use single optimizer checkpoint format (default: True)
        for_pretrained_restoration: If True, only used for restoring `'model'` from checkpoint

    Returns:
        Configured CheckpointManager instance
    """
    # determine checkpoint directory
    if gcs_output_dir:
        ckpt_dir = epath.Path(gcs_output_dir)
        if not str(ckpt_dir).endswith('/checkpoints'):
            ckpt_dir = ckpt_dir / "checkpoints"
    else:
        ckpt_dir = epath.Path(output_dir) / "checkpoints"
        ckpt_dir = ckpt_dir.resolve()

    logger.info(f"Checkpoint directory: {ckpt_dir}")

    # create checkpoint manager options
    options = ocp.CheckpointManagerOptions(
        max_to_keep=max_to_keep,
        save_interval_steps=save_interval_steps,
        best_fn=best_fn,
        best_mode=best_mode,
    )

    # create checkpoint manager with appropriate item names
    if single_optimizer:
        item_names = ('model', 'optimizer', 'meta')
    else:
        item_names = ('model', 'optimizer_wd', 'optimizer_no_wd', 'meta')
    if for_pretrained_restoration and not single_optimizer:
        item_names = ('model', 'meta')

    checkpoint_manager = ocp.CheckpointManager(
        ckpt_dir,
        options=options,
        item_names=item_names
    )

    return checkpoint_manager


def load_config_from_checkpoint(
    checkpoint_dir: str,
    step: Optional[int] = None
) -> Dict[str, Any]:
    """Load model config from checkpoint metadata.

    Checkpoints save metadata (including config) at:
    {checkpoint_dir}/checkpoints/{step}/meta/metadata

    Args:
        checkpoint_dir: Path to checkpoint directory (parent of 'checkpoints/')
        step: Specific step to load (None for latest)

    Returns:
        Config dictionary from checkpoint metadata

    Raises:
        FileNotFoundError: If checkpoint or metadata not found
        KeyError: If 'config' not in metadata
    """
    ckpt_path = epath.Path(checkpoint_dir)

    # handle both "checkpoints/" suffix and parent dir
    if not str(ckpt_path).endswith('/checkpoints'):
        ckpt_path = ckpt_path / "checkpoints"

    # find latest step if not specified
    if step is None:
        steps = sorted([int(p.name) for p in ckpt_path.iterdir() if p.name.isdigit()])
        if not steps:
            raise FileNotFoundError(f"No checkpoint steps found in {ckpt_path}")
        step = steps[-1]
        logger.info(f"Using latest checkpoint step: {step}")

    # read metadata json
    metadata_path = ckpt_path / str(step) / "meta" / "metadata"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found at {metadata_path}")

    with metadata_path.open('r') as f:
        metadata = json.load(f)

    if 'config' not in metadata:
        raise KeyError(f"'config' not found in checkpoint metadata. Available keys: {list(metadata.keys())}")

    logger.info(f"Loaded config from checkpoint step {step}")
    return metadata['config']


def build_flaxformer_moe_transforms() -> Dict[str, RestoreTransform]:
    """Build orbax transforms for restoring a Flaxformer MoE checkpoint into NNX format.

    Flaxformer stores MoE parameters under a nested `moe_params` wrapper:
        routed_moe.moe_params.expert.{wi,wo}.kernel
        routed_moe.moe_params.router.router_weights.w.kernel

    NNX uses a flat structure:
        routed_moe.experts.{wi,wo}.kernel
        routed_moe.router.kernel

    Returns:
        Dict of regex-keyed RestoreTransform objects for use with
        ocp.args.PyTreeRestore(transforms=...).
    """
    return {
        r'(.*routed_moe)/experts(.*)': RestoreTransform(
            original_key=r'\1/moe_params/expert\2'
        ),
        r'(.*routed_moe)/router(.*)': RestoreTransform(
            original_key=r'\1/moe_params/router/router_weights/w\2'
        ),
    }
