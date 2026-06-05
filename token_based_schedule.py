#!/usr/bin/env python3
# coding: utf-8
"""Progress-based learning rate schedules using optax's StatefulSchedule interface.

This module provides clean, framework-aligned implementations of LR scheduling
that work with JIT compilation and optax's inject_hyperparams functionality.

All schedules use normalized progress [0.0, 1.0] instead of absolute token counts,
allowing JAX to remain in 32-bit mode (no x64 needed for large token counts).
"""

import math
import jax.numpy as jnp
import chex
from typing import NamedTuple


class ProgressScheduleState(NamedTuple):
    """State for progress-based learning rate schedule.

    Stores training progress as ratio [0.0, 1.0+] in float32.
    Field is named tokens_seen for checkpoint backward compatibility
    (old checkpoints serialized this NamedTuple field name).
    """
    tokens_seen: chex.Array


# backward compatibility alias
TokenBasedScheduleState = ProgressScheduleState


class ProgressCosineSchedule:
    """A stateful schedule with cosine decay based on training progress.

    This schedule implements a cosine decay with warmup based on normalized
    progress [0.0, 1.0] rather than absolute token counts. It works with
    optax's inject_hyperparams and receives progress via extra_args.

    The schedule has four phases:
    1. Warmup: Linear increase from 0 to peak_lr
    2. Constant: Hold at peak_lr (T5-style)
    3. Cosine decay: Smooth cosine decay from peak_lr to min_lr
    4. Final constant: Stays at min_lr after progress >= 1.0
    """

    def __init__(
        self,
        peak_learning_rate: float,
        warmup_ratio: float,
        constant_ratio: float = 0.0,
        min_lr_ratio: float = 0.15,
        lr_cooldown_ratio: float = 0.0,
        lr_cooldown_type: str = "linear",
    ):
        """Initialize the progress-based cosine schedule.

        Args:
            peak_learning_rate: Peak learning rate after warmup
            warmup_ratio: Fraction of training for warmup [0.0, 1.0]
            constant_ratio: Fraction to hold at peak_lr (T5-style) [0.0, 1.0]
            min_lr_ratio: Minimum learning rate as ratio of peak (default: 0.15)
            lr_cooldown_ratio: Fraction of training for final cooldown (0.0 = disabled)
            lr_cooldown_type: Cooldown shape - "linear" or "sqrt" (1-sqrt(x))
        """
        self.peak_lr = peak_learning_rate
        self.warmup_ratio = warmup_ratio
        self.constant_ratio = constant_ratio
        self.min_lr_ratio = min_lr_ratio
        self.decay_start_ratio = warmup_ratio + constant_ratio
        self.lr_cooldown_ratio = lr_cooldown_ratio
        self.lr_cooldown_type = lr_cooldown_type
        self.cooldown_start = 1.0 - lr_cooldown_ratio

    def init(self) -> ProgressScheduleState:
        """Initialize the schedule state."""
        return ProgressScheduleState(tokens_seen=jnp.zeros([], dtype=jnp.float32))

    def update(
        self,
        state: ProgressScheduleState,
        **extra_args,
    ) -> ProgressScheduleState:
        """Update the schedule state with new progress value.

        Args:
            state: Current schedule state
            **extra_args: Should contain 'progress' key with current progress [0.0, 1.0]

        Returns:
            Updated schedule state
        """
        if 'progress' in extra_args:
            new_progress = jnp.asarray(extra_args['progress'], dtype=jnp.float32)
        else:
            new_progress = state.tokens_seen

        return ProgressScheduleState(tokens_seen=new_progress)

    def __call__(
        self,
        state: ProgressScheduleState,
        **extra_args,
    ) -> chex.Numeric:
        """Compute the learning rate based on current progress.

        Args:
            state: Current schedule state
            **extra_args: Ignored in computation

        Returns:
            Current learning rate (float32)
        """
        progress = state.tokens_seen

        # warmup phase: linear 0 -> peak_lr
        warmup_progress = progress / jnp.maximum(self.warmup_ratio, 1e-8)
        warmup_lr = self.peak_lr * jnp.clip(warmup_progress, 0.0, 1.0)

        # constant phase: hold at peak_lr
        constant_lr = self.peak_lr

        # cosine decay phase
        min_lr = self.peak_lr * self.min_lr_ratio

        if self.lr_cooldown_ratio > 0.0:
            # decay runs from decay_start to cooldown_start, then cooldown takes over
            decay_range = self.cooldown_start - self.decay_start_ratio
            decay_progress = (progress - self.decay_start_ratio) / jnp.maximum(decay_range, 1e-8)
            decay_progress = jnp.clip(decay_progress, 0.0, 1.0)
            cosine_factor = 0.5 * (1.0 + jnp.cos(jnp.pi * decay_progress))
            decay_lr = min_lr + (self.peak_lr - min_lr) * cosine_factor

            # LR at cooldown start (for smooth transition)
            cd_progress = jnp.clip(
                (self.cooldown_start - self.decay_start_ratio)
                / jnp.maximum(decay_range, 1e-8),
                0.0, 1.0,
            )
            cd_cosine = 0.5 * (1.0 + jnp.cos(jnp.pi * cd_progress))
            cooldown_start_lr = jnp.where(
                self.decay_start_ratio >= self.cooldown_start,
                self.peak_lr,
                min_lr + (self.peak_lr - min_lr) * cd_cosine,
            )

            cooldown_progress = (progress - self.cooldown_start) / self.lr_cooldown_ratio
            cooldown_progress = jnp.clip(cooldown_progress, 0.0, 1.0)

            if self.lr_cooldown_type == "sqrt":
                cooldown_factor = 1.0 - jnp.sqrt(cooldown_progress)
            else:
                cooldown_factor = 1.0 - cooldown_progress

            cooldown_lr = min_lr + (cooldown_start_lr - min_lr) * cooldown_factor

            decay_lr = jnp.where(
                progress < self.cooldown_start, decay_lr, cooldown_lr
            )
        else:
            decay_range = 1.0 - self.decay_start_ratio
            decay_progress = (progress - self.decay_start_ratio) / jnp.maximum(decay_range, 1e-8)
            decay_progress = jnp.clip(decay_progress, 0.0, 1.0)
            cosine_factor = 0.5 * (1.0 + jnp.cos(jnp.pi * decay_progress))
            decay_lr = min_lr + (self.peak_lr - min_lr) * cosine_factor

        # combine phases
        lr = jnp.where(
            progress < self.warmup_ratio,
            warmup_lr,
            jnp.where(
                progress < self.decay_start_ratio,
                constant_lr,
                jnp.where(progress < 1.0, decay_lr, min_lr)
            )
        )

        return lr


# backward compatibility alias
TokenBasedCosineSchedule = ProgressCosineSchedule


class ProgressLinearSchedule:
    """A stateful schedule with linear decay based on training progress.

    This schedule implements a linear decay with warmup based on normalized
    progress [0.0, 1.0]. Useful for trapezoidal schedules (constant then linear
    decay) as recommended by T5 authors.

    The schedule has four phases:
    1. Warmup: Linear increase from 0 to peak_lr
    2. Constant: Hold at peak_lr (T5-style)
    3. Linear decay: Linear decay from peak_lr to min_lr
    4. Final constant: Stays at min_lr after progress >= 1.0
    """

    def __init__(
        self,
        peak_learning_rate: float,
        warmup_ratio: float,
        constant_ratio: float = 0.0,
        min_lr_ratio: float = 0.1,
    ):
        """Initialize the progress-based linear schedule.

        Args:
            peak_learning_rate: Peak learning rate after warmup
            warmup_ratio: Fraction of training for warmup [0.0, 1.0]
            constant_ratio: Fraction to hold at peak_lr (T5-style) [0.0, 1.0]
            min_lr_ratio: Minimum learning rate as ratio of peak (default: 0.1)
        """
        self.peak_lr = peak_learning_rate
        self.warmup_ratio = warmup_ratio
        self.constant_ratio = constant_ratio
        self.min_lr_ratio = min_lr_ratio
        self.decay_start_ratio = warmup_ratio + constant_ratio

    def init(self) -> ProgressScheduleState:
        """Initialize the schedule state."""
        return ProgressScheduleState(tokens_seen=jnp.zeros([], dtype=jnp.float32))

    def update(
        self,
        state: ProgressScheduleState,
        **extra_args,
    ) -> ProgressScheduleState:
        """Update the schedule state with new progress value.

        Args:
            state: Current schedule state
            **extra_args: Should contain 'progress' key with current progress [0.0, 1.0]

        Returns:
            Updated schedule state
        """
        if 'progress' in extra_args:
            new_progress = jnp.asarray(extra_args['progress'], dtype=jnp.float32)
        else:
            new_progress = state.tokens_seen

        return ProgressScheduleState(tokens_seen=new_progress)

    def __call__(
        self,
        state: ProgressScheduleState,
        **extra_args,
    ) -> chex.Numeric:
        """Compute the learning rate based on current progress.

        Args:
            state: Current schedule state
            **extra_args: Ignored in computation

        Returns:
            Current learning rate (float32)
        """
        progress = state.tokens_seen

        # warmup phase: linear 0 -> peak_lr
        warmup_progress = progress / jnp.maximum(self.warmup_ratio, 1e-8)
        warmup_lr = self.peak_lr * jnp.clip(warmup_progress, 0.0, 1.0)

        # constant phase: hold at peak_lr
        constant_lr = self.peak_lr

        # linear decay phase
        decay_range = 1.0 - self.decay_start_ratio
        decay_progress = (progress - self.decay_start_ratio) / jnp.maximum(decay_range, 1e-8)
        decay_progress = jnp.clip(decay_progress, 0.0, 1.0)
        min_lr = self.peak_lr * self.min_lr_ratio
        decay_lr = self.peak_lr - (self.peak_lr - min_lr) * decay_progress

        # combine phases
        lr = jnp.where(
            progress < self.warmup_ratio,
            warmup_lr,
            jnp.where(
                progress < self.decay_start_ratio,
                constant_lr,
                jnp.where(progress < 1.0, decay_lr, min_lr)
            )
        )

        return lr


# backward compatibility alias
TokenBasedLinearSchedule = ProgressLinearSchedule


class ProgressConstantSchedule:
    """A constant learning rate schedule with optional warmup and final cooldown.

    The schedule has up to three phases:
    1. Warmup: Linear increase from 0 to peak_lr
    2. Constant: Stays at peak_lr
    3. Cooldown (optional): Decays from peak_lr to min_lr over the final lr_cooldown_ratio

    Exposes `peak_lr` (matching Cosine/Linear/Rsqrt schedules) so the phase 2
    transition can do `lr_scheduler.peak_lr = new_lr` and get an immediate drop
    to the new constant level. `min_lr` is computed dynamically from the
    current peak_lr so the cooldown floor tracks peak_lr changes.
    """

    def __init__(
        self,
        lr: float,
        warmup_ratio: float = 0.0,
        lr_cooldown_ratio: float = 0.0,
        lr_cooldown_type: str = "linear",
        min_lr_ratio: float = 0.0,
    ):
        """Initialize the constant schedule.

        Args:
            lr: Constant learning rate (after warmup)
            warmup_ratio: Fraction of training for warmup [0.0, 1.0]
            lr_cooldown_ratio: Fraction of training for final cooldown (0.0 = disabled)
            lr_cooldown_type: Cooldown shape - "linear" or "sqrt" (1-sqrt(x))
            min_lr_ratio: Minimum LR as ratio of peak (floor for cooldown)
        """
        self.peak_lr = lr
        self.warmup_ratio = warmup_ratio
        self.lr_cooldown_ratio = lr_cooldown_ratio
        self.lr_cooldown_type = lr_cooldown_type
        self.min_lr_ratio = min_lr_ratio
        self.cooldown_start = 1.0 - lr_cooldown_ratio

    @property
    def lr(self):
        return self.peak_lr

    @lr.setter
    def lr(self, value):
        self.peak_lr = value

    @property
    def min_lr(self):
        return self.peak_lr * self.min_lr_ratio

    def init(self) -> ProgressScheduleState:
        """Initialize the schedule state."""
        return ProgressScheduleState(tokens_seen=jnp.zeros([], dtype=jnp.float32))

    def update(
        self,
        state: ProgressScheduleState,
        **extra_args,
    ) -> ProgressScheduleState:
        """Update the schedule state with new progress value.

        Args:
            state: Current schedule state
            **extra_args: Should contain 'progress' key with current progress [0.0, 1.0]

        Returns:
            Updated schedule state
        """
        if 'progress' in extra_args:
            new_progress = jnp.asarray(extra_args['progress'], dtype=jnp.float32)
        else:
            new_progress = state.tokens_seen

        return ProgressScheduleState(tokens_seen=new_progress)

    def __call__(
        self,
        state_or_progress,
        **extra_args,
    ) -> chex.Numeric:
        """Compute the learning rate based on current progress.

        Args:
            state_or_progress: Either a ProgressScheduleState or raw progress value [0.0, 1.0]
            **extra_args: Ignored in computation

        Returns:
            Current learning rate (float32)
        """
        if hasattr(state_or_progress, 'tokens_seen'):
            progress = state_or_progress.tokens_seen
        else:
            progress = jnp.asarray(state_or_progress, dtype=jnp.float32)

        if self.warmup_ratio > 0:
            warmup_progress = progress / self.warmup_ratio
            warmup_lr = self.peak_lr * jnp.clip(warmup_progress, 0.0, 1.0)
            base_lr = jnp.where(progress < self.warmup_ratio, warmup_lr, self.peak_lr)
        else:
            base_lr = jnp.asarray(self.peak_lr, dtype=jnp.float32)

        if self.lr_cooldown_ratio > 0.0:
            cooldown_progress = (progress - self.cooldown_start) / self.lr_cooldown_ratio
            cooldown_progress = jnp.clip(cooldown_progress, 0.0, 1.0)
            if self.lr_cooldown_type == "sqrt":
                cooldown_factor = 1.0 - jnp.sqrt(cooldown_progress)
            else:
                cooldown_factor = 1.0 - cooldown_progress
            min_lr = self.min_lr
            cooldown_lr = min_lr + (self.peak_lr - min_lr) * cooldown_factor
            lr = jnp.where(progress >= self.cooldown_start, cooldown_lr, base_lr)
        else:
            lr = base_lr

        return lr


# backward compatibility alias
ConstantSchedule = ProgressConstantSchedule


class ProgressRsqrtSchedule:
    """A stateful schedule with reciprocal square root decay based on training progress.

    Implements the inverse square root schedule from "Attention Is All You Need"
    and adopted by ViT-22B for long training runs. After warmup, lr decays as
    1/sqrt(progress), which is less aggressive than cosine in the later stages.

    The schedule has up to four phases:
    1. Warmup: Linear increase from 0 to peak_lr
    2. Constant: Hold at peak_lr (T5-style, optional)
    3. Rsqrt decay: peak_lr * sqrt(decay_start / progress)
    4. Linear cooldown (optional): Linear decay to min_lr for pre-SFT convergence
    """

    def __init__(
        self,
        peak_learning_rate: float,
        warmup_ratio: float,
        constant_ratio: float = 0.0,
        min_lr_ratio: float = 0.0,
        lr_cooldown_ratio: float = 0.0,
    ):
        """Initialize the progress-based rsqrt schedule.

        Args:
            peak_learning_rate: Peak learning rate after warmup
            warmup_ratio: Fraction of training for warmup [0.0, 1.0]
            constant_ratio: Fraction to hold at peak_lr (T5-style) [0.0, 1.0]
            min_lr_ratio: Minimum LR as fraction of peak (used as cooldown floor)
            lr_cooldown_ratio: Fraction of training for final linear cooldown
                (e.g., 0.1 = last 10%). 0.0 disables cooldown.
        """
        self.peak_lr = peak_learning_rate
        self.warmup_ratio = warmup_ratio
        self.constant_ratio = constant_ratio
        self.decay_start_ratio = warmup_ratio + constant_ratio
        self.lr_cooldown_ratio = lr_cooldown_ratio
        self.min_lr_ratio = min_lr_ratio
        self.cooldown_start = 1.0 - lr_cooldown_ratio
        # rsqrt value at cooldown start, expressed as a ratio of peak_lr so it
        # tracks peak_lr swaps (e.g. phase 2) without requiring re-init
        if lr_cooldown_ratio > 0.0:
            self._cooldown_start_ratio = math.sqrt(
                self.decay_start_ratio / max(self.cooldown_start, 1e-8)
            )
        else:
            self._cooldown_start_ratio = 0.0

    @property
    def min_lr(self):
        return self.peak_lr * self.min_lr_ratio

    @property
    def cooldown_start_lr(self):
        return self.peak_lr * self._cooldown_start_ratio

    def init(self) -> ProgressScheduleState:
        """Initialize the schedule state."""
        return ProgressScheduleState(tokens_seen=jnp.zeros([], dtype=jnp.float32))

    def update(
        self,
        state: ProgressScheduleState,
        **extra_args,
    ) -> ProgressScheduleState:
        if 'progress' in extra_args:
            new_progress = jnp.asarray(extra_args['progress'], dtype=jnp.float32)
        else:
            new_progress = state.tokens_seen
        return ProgressScheduleState(tokens_seen=new_progress)

    def __call__(
        self,
        state: ProgressScheduleState,
        **extra_args,
    ) -> chex.Numeric:
        progress = state.tokens_seen

        warmup_progress = progress / jnp.maximum(self.warmup_ratio, 1e-8)
        warmup_lr = self.peak_lr * jnp.clip(warmup_progress, 0.0, 1.0)

        constant_lr = self.peak_lr

        rsqrt_lr = self.peak_lr * jnp.sqrt(
            self.decay_start_ratio / jnp.maximum(progress, 1e-8)
        )

        if self.lr_cooldown_ratio > 0.0:
            cooldown_progress = (progress - self.cooldown_start) / self.lr_cooldown_ratio
            cooldown_progress = jnp.clip(cooldown_progress, 0.0, 1.0)
            cooldown_lr = self.cooldown_start_lr + (
                (self.min_lr - self.cooldown_start_lr) * cooldown_progress
            )

            lr = jnp.where(
                progress < self.warmup_ratio,
                warmup_lr,
                jnp.where(
                    progress < self.decay_start_ratio,
                    constant_lr,
                    jnp.where(
                        progress < self.cooldown_start,
                        rsqrt_lr,
                        cooldown_lr,
                    )
                )
            )
        else:
            lr = jnp.where(
                progress < self.warmup_ratio,
                warmup_lr,
                jnp.where(
                    progress < self.decay_start_ratio,
                    constant_lr,
                    rsqrt_lr,
                )
            )

        return lr


def create_progress_schedule(
    learning_rate: float,
    warmup_ratio: float,
    constant_ratio: float = 0.0,
    min_lr_ratio: float = 0.15,
    schedule_type: str = "cosine",
    lr_cooldown_ratio: float = 0.0,
    lr_cooldown_type: str = "linear",
):
    """Create a progress-based schedule with warmup and optional constant phase.

    Args:
        learning_rate: Peak learning rate
        warmup_ratio: Fraction of training for warmup [0.0, 1.0]
        constant_ratio: Fraction to hold at peak_lr (T5-style) [0.0, 1.0]
        min_lr_ratio: Minimum LR as ratio of peak (default: 0.15)
        schedule_type: Type of decay schedule - "cosine" or "linear" (default: "cosine")
        lr_cooldown_ratio: Fraction of training for final cooldown (0.0 = disabled)
        lr_cooldown_type: Cooldown shape - "linear" or "sqrt" (1-sqrt(x))

    Returns:
        A ProgressCosineSchedule, ProgressLinearSchedule, or ProgressRsqrtSchedule instance
    """
    if schedule_type == "linear":
        return ProgressLinearSchedule(
            peak_learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            constant_ratio=constant_ratio,
            min_lr_ratio=min_lr_ratio,
        )
    elif schedule_type == "rsqrt":
        return ProgressRsqrtSchedule(
            peak_learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            constant_ratio=constant_ratio,
            min_lr_ratio=min_lr_ratio,
            lr_cooldown_ratio=lr_cooldown_ratio,
        )
    else:
        return ProgressCosineSchedule(
            peak_learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            constant_ratio=constant_ratio,
            min_lr_ratio=min_lr_ratio,
            lr_cooldown_ratio=lr_cooldown_ratio,
            lr_cooldown_type=lr_cooldown_type,
        )


def create_token_based_schedule(
    learning_rate: float,
    warmup_ratio: float,
    max_tokens: int,
    min_lr_ratio: float = 0.15,
    constant_ratio: float = 0.0,
    schedule_type: str = "cosine",
):
    """Create a progress-based schedule (backward compatible API).

    Note: max_tokens is accepted for API compatibility but ignored.
    The schedule now uses progress ratios directly.

    Args:
        learning_rate: Peak learning rate
        warmup_ratio: Fraction of training for warmup [0.0, 1.0]
        max_tokens: Ignored (kept for backward compatibility)
        min_lr_ratio: Minimum LR as ratio of peak (default: 0.15)
        constant_ratio: Fraction to hold at peak_lr [0.0, 1.0]
        schedule_type: Type of decay schedule - "cosine" or "linear"

    Returns:
        A ProgressCosineSchedule or ProgressLinearSchedule instance
    """
    return create_progress_schedule(
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        constant_ratio=constant_ratio,
        min_lr_ratio=min_lr_ratio,
        schedule_type=schedule_type,
    )


def compute_lr_for_logging(
    progress: float,
    learning_rate: float,
    warmup_ratio: float,
    constant_ratio: float,
    min_lr_ratio: float,
    schedule_type: str = "cosine",
    lr_cooldown_ratio: float = 0.0,
    lr_cooldown_type: str = "linear",
) -> float:
    """Compute current LR for logging (pure Python, not JAX).

    This mirrors the schedule logic for accurate monitoring without
    requiring JAX operations. Use this outside of JIT-compiled functions.

    Args:
        progress: Training progress [0.0, 1.0]
        learning_rate: Peak learning rate
        warmup_ratio: Fraction of training for warmup
        constant_ratio: Fraction at constant peak LR
        min_lr_ratio: Minimum LR as ratio of peak
        schedule_type: "cosine", "linear", or "rsqrt"
        lr_cooldown_ratio: Fraction of training for final cooldown (0.0 = disabled)
        lr_cooldown_type: Cooldown shape - "linear" or "sqrt" (1-sqrt(x))

    Returns:
        Current learning rate as Python float
    """
    decay_start = warmup_ratio + constant_ratio
    min_lr = learning_rate * min_lr_ratio
    cooldown_start = 1.0 - lr_cooldown_ratio

    if progress < warmup_ratio:
        return learning_rate * (progress / max(warmup_ratio, 1e-8))
    elif progress < decay_start:
        return learning_rate
    elif schedule_type == "rsqrt":
        if lr_cooldown_ratio > 0.0 and progress >= cooldown_start:
            cooldown_start_lr = learning_rate * math.sqrt(
                decay_start / max(cooldown_start, 1e-8)
            )
            cooldown_progress = min(
                (progress - cooldown_start) / lr_cooldown_ratio, 1.0
            )
            return cooldown_start_lr + (min_lr - cooldown_start_lr) * cooldown_progress
        return learning_rate * math.sqrt(decay_start / max(progress, 1e-8))
    elif progress < 1.0:
        # check if we're in the cooldown phase
        if lr_cooldown_ratio > 0.0 and progress >= cooldown_start:
            # compute LR at cooldown_start for smooth transition
            if decay_start >= cooldown_start:
                cd_start_lr = learning_rate
            else:
                decay_range = cooldown_start - decay_start
                cd_prog = min((cooldown_start - decay_start) / max(decay_range, 1e-8), 1.0)
                if schedule_type == "linear":
                    cd_start_lr = learning_rate - (learning_rate - min_lr) * cd_prog
                else:
                    cd_cosine = 0.5 * (1.0 + math.cos(math.pi * cd_prog))
                    cd_start_lr = min_lr + (learning_rate - min_lr) * cd_cosine

            cooldown_progress = min(
                (progress - cooldown_start) / lr_cooldown_ratio, 1.0
            )
            if lr_cooldown_type == "sqrt":
                cooldown_factor = 1.0 - math.sqrt(cooldown_progress)
            else:
                cooldown_factor = 1.0 - cooldown_progress

            return min_lr + (cd_start_lr - min_lr) * cooldown_factor

        # normal decay phase
        decay_range = (cooldown_start if lr_cooldown_ratio > 0.0 else 1.0) - decay_start
        decay_progress = (progress - decay_start) / max(decay_range, 1e-8)
        decay_progress = min(max(decay_progress, 0.0), 1.0)

        if schedule_type == "linear":
            return learning_rate - (learning_rate - min_lr) * decay_progress
        else:
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
            return min_lr + (learning_rate - min_lr) * cosine_factor
    else:
        return min_lr
