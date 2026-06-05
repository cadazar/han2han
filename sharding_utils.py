#!/usr/bin/env python3
"""
Shared utilities for JAX sharding and mesh setup across training scripts.
Extracted from the finetune scripts to avoid duplication.

This module provides a global mesh singleton that can be queried from any module
to determine sharding strategy.
"""

import jax
import numpy as np
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental.mesh_utils import create_device_mesh
from logging_utils import log_from_main_process
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

# global mesh state - set once at training init, queryable from any module
_GLOBAL_MESH = None
_GLOBAL_MESH_CONFIG = None


def get_tpu_topology() -> Tuple[str, Tuple[int, ...]]:
    """Detect TPU type and physical topology.

    Returns:
        (tpu_type, physical_shape) where:
        - tpu_type: 'v4', 'v5e', 'v5p', 'v6e', or 'unknown'
        - physical_shape: tuple of physical dimensions
    """
    devices = jax.devices()
    if not devices:
        return ('unknown', (1,))

    device = devices[-1]
    device_kind = getattr(device, 'device_kind', '')

    log_from_main_process(
        logger, 'info',
        f"TPU topology detection: device_kind='{device_kind}', "
        f"coords={getattr(device, 'coords', 'none')}"
    )

    # check device_kind first - v5e/v6e are logically 2D even if they report 3D coords
    if 'v5e' in device_kind or 'v5 lite' in device_kind.lower():
        num_hosts = jax.process_count()
        devices_per_host = jax.local_device_count()
        return ('v5e', (num_hosts, devices_per_host))
    elif 'v6e' in device_kind:
        num_hosts = jax.process_count()
        devices_per_host = jax.local_device_count()
        return ('v6e', (num_hosts, devices_per_host))

    # check for 3D coords
    if hasattr(device, 'coords') and len(device.coords) == 3:
        x, y, z = device.coords

        # if z=0 for all devices, this is actually a 2D topology (v5e/v6e style)
        all_z_zero = all(d.coords[2] == 0 for d in devices if hasattr(d, 'coords'))
        if all_z_zero:
            num_hosts = jax.process_count()
            devices_per_host = jax.local_device_count()
            log_from_main_process(
                logger, 'info',
                f"Detected 2D topology (z=0 for all devices), treating as v5e-like"
            )
            return ('v5e', (num_hosts, devices_per_host))

        # true 3D topology
        physical_shape = (x + 1, y + 1, z + 1)
        if 'v4' in device_kind:
            return ('v4', physical_shape)
        elif 'v5p' in device_kind:
            return ('v5p', physical_shape)
        else:
            return ('v4-like', physical_shape)
    else:
        num_hosts = jax.process_count()
        devices_per_host = jax.local_device_count()
        return ('2d', (num_hosts, devices_per_host))


def set_global_mesh(mesh: Mesh, config: dict):
    """
    Set the global mesh for all distributed operations.

    Call this once at training startup after creating your mesh:
        mesh, data_sharding, config = setup_mesh_and_sharding(args)
        set_global_mesh(mesh, config)

    Args:
        mesh: JAX Mesh object
        config: Parallelism config dict from get_parallelism_config()
    """
    global _GLOBAL_MESH, _GLOBAL_MESH_CONFIG
    _GLOBAL_MESH = mesh
    _GLOBAL_MESH_CONFIG = config
    log_from_main_process(logger, 'info', f"Global mesh set: {mesh.axis_names}")


def get_global_mesh() -> Mesh:
    """
    Get the global mesh, creating a simple data-parallel one if not set.

    Returns:
        JAX Mesh object
    """
    global _GLOBAL_MESH
    if _GLOBAL_MESH is None:
        devices = jax.devices()
        _GLOBAL_MESH = Mesh(np.array(devices).reshape(-1), axis_names=('data',))
        log_from_main_process(logger, 'warning',
            "Global mesh not set, created default data-parallel mesh")
    return _GLOBAL_MESH


def get_global_mesh_config() -> dict:
    """
    Get the global mesh configuration.

    Returns:
        Parallelism config dict, or a default data_parallel config if not set
    """
    global _GLOBAL_MESH_CONFIG
    if _GLOBAL_MESH_CONFIG is None:
        return {
            'mesh_shape': (jax.device_count(), 1),
            'mesh_axes': ('data', 'model'),
            'model_param_sharding': None,
            'vocab_sharding': None,
            'description': 'Default data parallelism (mesh not explicitly set)'
        }
    return _GLOBAL_MESH_CONFIG


def get_data_partition_spec() -> PartitionSpec:
    """
    Get the data partition spec for batch sharding.

    Returns the appropriate PartitionSpec based on the current parallelism strategy:
    - data_parallel/fsdp: P('data') for standard data sharding
    - other: P() for no sharding

    Returns:
        PartitionSpec for sharding batch dimension
    """
    config = get_global_mesh_config()
    if 'data_partition_spec' in config:
        return config['data_partition_spec']
    mesh_axes = config.get('mesh_axes', ('data', 'model'))
    if 'data' in mesh_axes:
        return PartitionSpec('data')
    return PartitionSpec()


def get_activation_partition_spec(rank: int = 3) -> PartitionSpec:
    """PartitionSpec for activation tensors of shape [B, S, ...].

    Pins the leading batch axis to the current strategy's data sharding and
    leaves the remaining axes unsharded. Mirrors MaxText's
    (activation_batch, activation_length, activation_embed) logical layout for
    the usual path; without these constraints XLA can reinterpret
    FSDP's `('data', None)` weight sharding as Megatron-style tensor parallel,
    which forces full-batch all-gathers in the backward pass.

    For strategies with no batch axis (e.g. data_parallel preset returns P()),
    the result is P() and callers should skip the constraint.
    """
    base = get_data_partition_spec()
    if base == PartitionSpec():
        return PartitionSpec()
    return PartitionSpec(base[0], *([None] * (rank - 1)))


def maybe_constrain_activation(x):
    """Apply get_activation_partition_spec to a rank-3 activation if the
    current strategy has a batch axis; otherwise return unchanged.
    """
    spec = get_activation_partition_spec(rank=x.ndim)
    if spec == PartitionSpec():
        return x
    return jax.lax.with_sharding_constraint(x, spec)


def is_data_parallel_only() -> bool:
    """
    Check if current strategy is pure data parallel (no model/activation sharding).

    Use this to decide whether to enable activation partitioning in components
    like Flaxformer's MlpBlock.

    Returns:
        True if pure data parallel, False if any model/activation sharding
    """
    config = get_global_mesh_config()
    strategy = config.get('description', '')

    # pure data parallel means no activation sharding
    if 'Pure data parallelism' in strategy or 'data_parallel' in strategy.lower():
        return True

    # also check mesh_axes - if only 'data' is meaningful, it's data parallel
    mesh_axes = config.get('mesh_axes', ())
    meaningful_axes = [ax for ax in mesh_axes if ax is not None and ax != 'data']
    return len(meaningful_axes) == 0


def get_activation_partitioning_dims() -> int:
    """
    Get the appropriate activation_partitioning_dims for Flaxformer components.

    Returns:
        1 for data-parallel (no activation sharding), 2 for FSDP/model-parallel
    """
    return 1 if is_data_parallel_only() else 2


def get_mesh_data_axis() -> str:
    """Get the data-parallel axis name from the global mesh."""
    config = get_global_mesh_config()
    mesh_axes = config.get('mesh_axes', ('data',))
    return 'data' if 'data' in mesh_axes else mesh_axes[0]



def get_parallelism_config(args, n_devices=None):
    """Get parallelism configuration based on strategy preset."""
    if n_devices is None:
        n_devices = jax.device_count()
    strategy = args.parallelism_strategy

    log_from_main_process(logger, 'info', f"Configuring parallelism strategy: {strategy}")
    log_from_main_process(logger, 'info', f"Available devices: {n_devices}")

    if strategy == "data_parallel":
        # pure data parallelism - use 2D mesh with model axis size 1
        # sharding=None means no explicit sharding constraints, just eager sharding
        mesh_shape = (n_devices, 1)  # 2D mesh
        mesh_axes = ('data', 'model')  # 2D axes
        model_param_sharding = None  # no explicit sharding, rely on eager sharding
        vocab_sharding = None  # no vocab sharding
        description = "Pure data parallelism"

    elif strategy == "model_parallel":
        # pure model parallelism - minimal data sharding
        mesh_shape = (1, n_devices) if n_devices > 1 else (1, 1)
        mesh_axes = (None, 'model')
        model_param_sharding = ('model', None)  # shard vocab/params across model axis
        vocab_sharding = 'model'  # shard vocab across model axis
        description = "Pure model parallelism"

    elif strategy == "fsdp":
        # fully sharded data parallel - shard parameters and gradients across data axis
        mesh_shape = (n_devices, 1) if n_devices > 1 else (1, 1)
        mesh_axes = ('data', None)
        model_param_sharding = ('data', None)  # shard params across data axis (FSDP style)
        vocab_sharding = 'data'  # shard vocab across data axis
        description = "Fully Sharded Data Parallel (FSDP)"

    elif strategy == "hybrid":
        # default hybrid strategy - current 4×8 setup for TPU v4-64
        if n_devices == 32:
            mesh_shape = (4, 8)
            mesh_axes = ('data', 'model')
            model_param_sharding = ('model', None)  # shard vocab across model axis
            vocab_sharding = 'model'  # 8-way vocab sharding
            description = "Hybrid data+model parallelism (4x8 TPU v4-64 optimized)"
        elif n_devices == 16:
            mesh_shape = (4, 4)
            mesh_axes = ('data', 'model')
            model_param_sharding = ('model', None)
            vocab_sharding = 'model'
            description = "Hybrid data+model parallelism (4x4 TPU v4-32)"
        else:
            # fallback for other device counts
            import math
            sqrt_devices = int(math.sqrt(n_devices))
            if sqrt_devices * sqrt_devices == n_devices:
                mesh_shape = (sqrt_devices, sqrt_devices)
            else:
                mesh_shape = (1, n_devices)
            mesh_axes = ('data', 'model')
            model_param_sharding = ('model', None)
            vocab_sharding = 'model'
            description = f"Hybrid data+model parallelism (auto-detected {mesh_shape})"

    elif strategy == "custom":
        # custom strategy using provided mesh_axes and mesh_shape
        if args.mesh_shape is None:
            raise ValueError("Custom strategy requires --mesh_shape to be specified")
        if len(args.mesh_shape) != len(args.mesh_axes):
            raise ValueError("mesh_shape and mesh_axes must have the same length")
        if np.prod(args.mesh_shape) != n_devices:
            raise ValueError(f"Product of mesh_shape {args.mesh_shape} = {np.prod(args.mesh_shape)} "
                           f"must equal device count {n_devices}")

        mesh_shape = tuple(args.mesh_shape)
        mesh_axes = tuple(args.mesh_axes)

        # determine sharding based on axis names
        if 'model' in mesh_axes:
            model_param_sharding = ('model', None)
            vocab_sharding = 'model'
        elif 'data' in mesh_axes and len(mesh_axes) == 1:
            # pure data parallelism or FSDP
            model_param_sharding = ('data', None)
            vocab_sharding = 'data'
        else:
            # default to no sharding for unknown axes
            model_param_sharding = (None, None)
            vocab_sharding = None

        description = f"Custom parallelism (shape={mesh_shape}, axes={mesh_axes})"

    else:
        raise ValueError(f"Unknown parallelism strategy: {strategy}")

    return {
        'mesh_shape': mesh_shape,
        'mesh_axes': mesh_axes,
        'model_param_sharding': model_param_sharding,
        'vocab_sharding': vocab_sharding,
        'description': description
    }


def setup_mesh_and_sharding(args):
    """Setup JAX SPMD mesh with configurable parallelism strategies."""
    n_devices = jax.device_count()
    log_from_main_process(logger, 'info', f"Setting up mesh with {n_devices} devices: {jax.devices()}")

    # get parallelism configuration
    config = get_parallelism_config(args, n_devices)
    mesh_shape = config['mesh_shape']
    mesh_axes = config['mesh_axes']

    log_from_main_process(logger, 'info', f"Strategy: {config['description']}")
    log_from_main_process(logger, 'info', f"Mesh shape: {mesh_shape}, axes: {mesh_axes}")

    filtered_axes = tuple(axis for axis in mesh_axes if axis is not None)
    filtered_shape = tuple(s for s, a in zip(mesh_shape, mesh_axes) if a is not None)
    strategy = args.parallelism_strategy

    # jax.make_mesh handles topology-aware device placement automatically,
    # mapping larger logical axes to higher-bandwidth physical ICI axes.
    # all axes are Auto so the compiler resolves sharding for dot_general etc.
    from jax.sharding import AxisType
    mesh = jax.make_mesh(
        filtered_shape, filtered_axes,
        axis_types=(AxisType.Auto,) * len(filtered_axes),
    )

    if 'data' in filtered_axes:
        data_partition_spec = PartitionSpec('data')
    else:
        data_partition_spec = PartitionSpec()

    data_sharding = NamedSharding(mesh, data_partition_spec)

    log_from_main_process(logger, 'info', f"Mesh configured: {mesh}")
    log_from_main_process(logger, 'info', f"Data sharding: {data_sharding}")
    log_from_main_process(logger, 'info', f"Parallelism strategy: {strategy}")

    config['data_partition_spec'] = data_partition_spec

    # set JAX thread-local mesh context so bare PartitionSpec works everywhere
    jax.set_mesh(mesh)

    # set global mesh for use by other modules
    set_global_mesh(mesh, config)

    return mesh, data_sharding, config


def shard_batch_to_devices(batch, mesh, partition_spec=None):
    """Shard a batch across devices.

    Uses jax.make_array_from_process_local_data for multi-host, which works
    with any mesh topology (no contiguous-subcube constraint). For data-parallel
    batching it doesn't matter which microbatch goes on each device.

    Args:
        batch: Input batch (dict of arrays, may contain non-array fields)
        mesh: JAX mesh object
        partition_spec: PartitionSpec for sharding (default: uses get_data_partition_spec())

    Returns:
        Sharded batch with arrays distributed across devices
    """
    if partition_spec is None:
        partition_spec = get_data_partition_spec()

    global_sharding = NamedSharding(mesh, partition_spec)

    def maybe_shard(x):
        if not isinstance(x, (np.ndarray, jax.Array)):
            return x
        if jax.process_count() <= 1:
            return jax.device_put(x, global_sharding)
        return jax.make_array_from_process_local_data(global_sharding, x)

    return jax.tree.map(maybe_shard, batch)


def get_data_layout(batch_size, n_devices):
    """Calculate data layout for multi-device training.

    Args:
        batch_size: Target per-host batch size (will be divided across devices)
        n_devices: Number of devices per host

    Returns:
        dict with per_device_batch_size and global_batch_size
    """
    return {
        'per_device_batch_size': batch_size // n_devices,
        'global_batch_size': batch_size,
        'n_devices': n_devices
    }


def derive_param_sharding(strategy, parallelism_config):
    """Derive parameter sharding tuple from parallelism strategy.

    The model expects sharding as a tuple like ('axis1', 'axis2') for 2D params.
    This function provides the correct sharding based on the strategy, which may
    differ from model_param_sharding in the config.

    Args:
        strategy: Parallelism strategy name
        parallelism_config: Config dict from get_parallelism_config()

    Returns:
        Tuple of axis names for parameter sharding, or None for no sharding
    """
    mesh_axes = parallelism_config['mesh_axes']

    if strategy == 'data_parallel':
        return None

    elif strategy == 'fsdp':
        return ('data', None)

    elif strategy == 'hybrid':
        if 'model' in mesh_axes:
            return (None, 'model')
        return None

    elif strategy == 'model_parallel':
        return (None, 'model')

    return parallelism_config.get('model_param_sharding')