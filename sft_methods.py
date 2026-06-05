#!/usr/bin/env python3
# coding: utf-8
"""Scheduled sampling and contrastive learning helpers for SFT.

Extracted from the legacy finetune scripts so
that finetune_sft.py can depend on a focused module rather than pulling in
full legacy training loops just to reuse the SS+CL math.

Scheduled sampling follows the confidence-aware approach of Liu et al. 2021
(https://arxiv.org/abs/2107.10427) -- the schedule is implicit and adaptive
via percentile-based confidence thresholds, not a fixed teacher-forcing
probability ramp. Contrastive learning follows He et al. 2024
(https://aclanthology.org/2024.eamt-1.10/) with token-level margin loss
between the gold and scheduled-sampling forward passes.
"""

from functools import partial
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from flax.struct import dataclass, field
from flax.training.common_utils import onehot


@dataclass
class ContrastiveLearningConfig:
    """Configuration for contrastive learning following He et al. 2024."""
    enabled: bool = field(default=False)
    margin: float = field(default=0.01)
    weight: float = field(default=0.1)
    temperature: float = field(default=1.0)


@dataclass
class ScheduledSamplingConfig:
    """Configuration for scheduled sampling.

    The schedule is implicit and adaptive (confidence percentile thresholds
    set per-batch), so no teacher-forcing-probability ramp or annealing
    schedule is exposed -- those would only matter for the Mihaylova & Martins
    2019 fixed-schedule transformer variant (sigmoid/inverse-sigmoid/linear
    decay of teacher-forcing prob), which this module does not implement.

    threshold_mode picks how the high/low-confidence percentile boundaries
    are chosen. 'percentile' (default) computes per-batch quartiles of the
    valid CE distribution -- stable bucket sizes (25/50/25) but the
    semantic meaning of "high confidence" drifts with model quality.
    'fixed' uses the static high_conf_threshold / low_conf_threshold values
    (defaulting to Liu et al. 2021's -log(0.95) / -log(0.9) nats) -- stable
    semantics, but bucket sizes drift; early in training nearly all tokens
    fall in the teacher-forcing bucket. The percentile thresholds are
    always logged via ce_high_conf_threshold / ce_low_conf_threshold so
    'fixed' mode can be calibrated from a 'percentile' prefix run.
    """
    use_soft_embeddings: bool = field(default=False)
    temperature: float = field(default=1.0)
    mixing_method: str = field(default='gumbel')
    threshold_mode: str = field(default='percentile')
    high_conf_threshold: float = field(default=0.05129329438755058)
    low_conf_threshold: float = field(default=0.10536051565782628)


@partial(nnx.jit, static_argnames=('temperature', 'hard'))
def gumbel_softmax_embeddings(
    logits: jnp.ndarray,
    embedding_matrix: nnx.Embed,
    temperature: float = 1.0,
    rng_key: Optional[jax.random.PRNGKey] = None,
    hard: bool = False,
) -> jnp.ndarray:
    """Apply Gumbel-Softmax to get soft embeddings from logits."""
    if rng_key is not None:
        uniform = jax.random.uniform(rng_key, logits.shape, minval=1e-10, maxval=1.0)
        gumbel_noise = -jnp.log(-jnp.log(uniform))
        logits = logits + gumbel_noise

    soft_weights = nnx.softmax(logits / temperature, axis=-1)

    if hard:
        # straight-through estimator
        hard_weights = onehot(jnp.argmax(logits, axis=-1), logits.shape[-1])
        soft_weights = hard_weights - jax.lax.stop_gradient(soft_weights) + soft_weights

    soft_weights, embedding_values = embedding_matrix.promote_dtype(
        (soft_weights, embedding_matrix.embedding.value), dtype=embedding_matrix.dtype
    )
    return jnp.matmul(soft_weights, embedding_values)


@partial(nnx.jit, static_argnames=('temperature',))
def softmax_temperature_embeddings(
    logits: jnp.ndarray,
    embedding_matrix: nnx.Embed,
    temperature: float = 1.0,
) -> jnp.ndarray:
    """Apply softmax with temperature to get soft embeddings."""
    soft_weights = nnx.softmax(logits / temperature, axis=-1)
    soft_weights, embedding_values = embedding_matrix.promote_dtype(
        (soft_weights, embedding_matrix.embedding.value), dtype=embedding_matrix.dtype
    )
    return jnp.matmul(soft_weights, embedding_values)


@partial(nnx.jit, static_argnames=('config',))
def compute_contrastive_loss(
    gold_logits: jnp.ndarray,
    sampled_logits: jnp.ndarray,
    labels: jnp.ndarray,
    label_mask: jnp.ndarray,
    config: ContrastiveLearningConfig,
) -> Tuple[jnp.ndarray, Dict[str, float]]:
    """Token-level contrastive loss from He et al. 2024.

    Objective: max(0, rho + S_negative - S_positive), where S_positive is the
    score of the gold sequence and S_negative is the score of the scheduled
    sampling sequence.
    """
    gold_log_probs = jax.nn.log_softmax(gold_logits / config.temperature, axis=-1)
    sampled_log_probs = jax.nn.log_softmax(sampled_logits / config.temperature, axis=-1)

    batch_size, seq_len = labels.shape
    batch_indices = jnp.arange(batch_size)[:, None]
    seq_indices = jnp.arange(seq_len)[None, :]

    gold_target_log_probs = gold_log_probs[batch_indices, seq_indices, labels]
    sampled_target_log_probs = sampled_log_probs[batch_indices, seq_indices, labels]

    contrastive_loss_per_token = jax.nn.relu(
        config.margin + sampled_target_log_probs - gold_target_log_probs
    )

    masked_contrastive_loss = contrastive_loss_per_token * label_mask
    contrastive_loss = jnp.sum(masked_contrastive_loss) / jnp.maximum(jnp.sum(label_mask), 1.0)

    gold_wins = (gold_target_log_probs > sampled_target_log_probs) * label_mask
    gold_win_rate = jnp.sum(gold_wins) / jnp.maximum(jnp.sum(label_mask), 1.0)

    margin_per_token = (gold_target_log_probs - sampled_target_log_probs) * label_mask
    avg_margin = jnp.sum(margin_per_token) / jnp.maximum(jnp.sum(label_mask), 1.0)

    metrics = {
        'cl_loss': contrastive_loss,
        'gold_win_rate': gold_win_rate,
        'avg_margin': avg_margin,
    }

    return contrastive_loss, metrics


@partial(nnx.jit, static_argnames=('config',))
def scheduled_sampling_step(
    decoder_input_ids: jnp.ndarray,
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    decoder_embeddings: nnx.Embed,
    config: ScheduledSamplingConfig,
    rngs: Optional[nnx.Rngs] = None,
) -> Tuple[jnp.ndarray, Dict[str, float]]:
    """Apply scheduled sampling to decoder inputs.

    The "schedule" is implicit and adaptive, based on Liu et al. 2021
    (https://arxiv.org/abs/2107.10427). Confidence percentiles drive a
    three-way split: high-confidence positions get random tokens, medium get
    soft/predicted tokens, low get teacher forcing.

    Returns:
        Modified decoder_input_ids (or decoder_input_embeddings if
        use_soft_embeddings) and confidence stats.
    """
    batch_size, seq_len = decoder_input_ids.shape
    p_model = 1.0
    should_replace = p_model > 0.0

    if rngs is not None:
        sample_key, gumbel_key = jax.random.split(rngs.default(), 2)
    else:
        sample_key = gumbel_key = None

    # ce computed against original logits aligned with labels
    ce_per_token = optax.softmax_cross_entropy_with_integer_labels(logits, labels)

    # shift logits to align with decoder inputs for soft embeddings:
    # [0, logit_0, logit_1, ..., logit_{n-2}]
    shifted_logits = jnp.concatenate(
        [
            jnp.zeros((logits.shape[0], 1, logits.shape[2]), dtype=logits.dtype),
            logits[:, :-1, :]
        ],
        axis=1,
    )

    if config.use_soft_embeddings:
        if config.mixing_method == 'gumbel':
            soft_embeddings = gumbel_softmax_embeddings(
                shifted_logits, decoder_embeddings,
                temperature=config.temperature,
                rng_key=gumbel_key,
                hard=False,
            )
        else:
            soft_embeddings = softmax_temperature_embeddings(
                shifted_logits, decoder_embeddings,
                temperature=config.temperature,
            )
    else:
        predicted_ids = jnp.argmax(shifted_logits, axis=-1)

    # ce stats for logging
    valid_mask = labels > 0
    valid_ce = jnp.where(valid_mask, ce_per_token, jnp.nan)
    ce_stats = {
        'ce_mean': jnp.nanmean(valid_ce),
        'ce_std': jnp.nanstd(valid_ce),
        'ce_min': jnp.nanmin(valid_ce),
        'ce_max': jnp.nanmax(valid_ce),
        'ce_median': jnp.nanmedian(valid_ce),
    }

    # 'percentile' mode picks the high/low boundaries as per-batch quartiles
    # of the valid (non-pad) ce distribution. Bucket sizes stay at 25/50/25
    # but the semantic confidence threshold drifts with the loss landscape.
    # 'fixed' mode uses static thresholds from config (defaulting to Liu et
    # al. 2021's -log(0.95) / -log(0.9) nats). Semantics stay stable but
    # bucket sizes drift -- early training nearly everything is teacher-
    # forcing, late training the high-conf bucket may saturate.
    if config.threshold_mode == 'fixed':
        high_conf_threshold = jnp.asarray(config.high_conf_threshold, dtype=ce_per_token.dtype)
        low_conf_threshold = jnp.asarray(config.low_conf_threshold, dtype=ce_per_token.dtype)
    else:
        valid_mask_flat = (labels > 0).flatten()
        ce_flat = ce_per_token.flatten()
        valid_ce_for_percentiles = jnp.where(valid_mask_flat, ce_flat, jnp.nan)
        high_conf_threshold = jnp.nanpercentile(valid_ce_for_percentiles, 25)
        low_conf_threshold = jnp.nanpercentile(valid_ce_for_percentiles, 75)

    ce_stats['ce_high_conf_threshold'] = high_conf_threshold
    ce_stats['ce_low_conf_threshold'] = low_conf_threshold

    # three-way split:
    # high confidence (low ce) -> random tokens
    # medium confidence -> soft/predicted tokens
    # low confidence (high ce) -> teacher forcing
    if sample_key is not None:
        schedule_mask = jax.random.uniform(sample_key, (batch_size, seq_len)) < p_model
        random_key, sample_key = jax.random.split(sample_key)
        random_token_ids = jax.random.randint(random_key, (batch_size, seq_len), 0, 15000)
    else:
        schedule_mask = np.random.random((batch_size, seq_len)) < p_model
        schedule_mask = jnp.array(schedule_mask)
        random_token_ids = jnp.array(np.random.randint(0, 15000, (batch_size, seq_len)))

    # 0 = teacher, 1 = soft/predicted, 2 = random
    token_source = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    token_source = jnp.where(ce_per_token > low_conf_threshold, 0, token_source)
    token_source = jnp.where(ce_per_token < high_conf_threshold, 2, token_source)

    # only replace where schedule allows
    token_source = jnp.where(schedule_mask, token_source, 0)
    # never replace position 0 (bos)
    token_source = token_source.at[:, 0].set(0)
    token_source = jax.lax.stop_gradient(token_source)

    if config.use_soft_embeddings:
        original_embeddings = decoder_embeddings(decoder_input_ids)
        random_embeddings = decoder_embeddings(random_token_ids)

        final_embeddings = jnp.where(
            (token_source == 2)[..., None],
            random_embeddings,
            jnp.where(
                (token_source == 1)[..., None],
                soft_embeddings,
                original_embeddings,
            ),
        )

        embeds_to_return = jnp.where(
            should_replace,
            final_embeddings,
            original_embeddings,
        )

        return embeds_to_return, ce_stats
    else:
        final_ids = jnp.where(
            token_source == 2,
            random_token_ids,
            jnp.where(
                token_source == 1,
                predicted_ids,
                decoder_input_ids,
            ),
        )

        ids_to_return = jnp.where(
            should_replace,
            final_ids,
            decoder_input_ids,
        )

        return ids_to_return, ce_stats

