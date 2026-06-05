"""Sampler for Han2Han encoder-decoder transformer.

A sampling class following the Gemma NNX pattern for efficient generation.
Supports standard KV-cached generation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional, Tuple, List, Union

import flax
from flax import nnx
from flax.nnx import graph
from flax.nnx import statelib
import jax
import jax.numpy as jnp



def _sample_top_p(probs: jnp.ndarray, p: float, key: jax.Array) -> jnp.ndarray:
    """Sample a token using top-p (nucleus) sampling."""
    probs_sorted, indices = jax.lax.top_k(probs, k=probs.shape[-1])
    cumsum_probs = jnp.cumsum(probs_sorted, axis=-1)
    mask = cumsum_probs - probs_sorted > p
    probs_sorted = jnp.where(mask, 0.0, probs_sorted)
    probs_sorted = probs_sorted / jnp.sum(probs_sorted, axis=-1, keepdims=True)

    next_token = jax.random.categorical(key, logits=jnp.log(probs_sorted + 1e-10))
    next_token = jnp.take_along_axis(indices, next_token[..., None], axis=-1)
    next_token = jnp.squeeze(next_token, axis=-1)
    return next_token


def _sample_top_k(logits: jnp.ndarray, k: int, key: jax.Array) -> jnp.ndarray:
    """Sample a token using top-k sampling."""
    top_k_logits, top_k_indices = jax.lax.top_k(logits, k=min(k, logits.shape[-1]))
    next_token = jax.random.categorical(key, logits=top_k_logits)
    next_token = jnp.take_along_axis(top_k_indices, next_token[..., None], axis=-1)
    next_token = jnp.squeeze(next_token, axis=-1)
    return next_token


def _apply_repetition_penalty(
    logits: jnp.ndarray,
    generated_tokens: jnp.ndarray,
    penalty: float,
    valid_mask: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Apply repetition penalty to logits based on previously generated tokens.

    Args:
        logits: next token logits (batch_size, vocab_size)
        generated_tokens: tokens generated so far (batch_size, seq_len)
        penalty: repetition penalty factor (>1 reduces repetition)
        valid_mask: optional mask for valid token positions (batch_size, seq_len)

    Returns:
        modified logits with repetition penalty applied
    """
    if penalty == 1.0:
        return logits

    batch_size, vocab_size = logits.shape

    if valid_mask is not None:
        masked_tokens = jnp.where(valid_mask, generated_tokens, vocab_size)
    else:
        masked_tokens = generated_tokens

    token_indices = jnp.arange(vocab_size)[None, None, :]
    generated_expanded = masked_tokens[:, :, None]
    appeared_mask = jnp.any(generated_expanded == token_indices, axis=1)

    positive_mask = logits > 0
    penalized_logits = jnp.where(
        positive_mask,
        logits / penalty,
        logits * penalty
    )

    return jnp.where(appeared_mask, penalized_logits, logits)


def _apply_no_repeat_ngram_blocking(
    logits: jnp.ndarray,
    generated_tokens: jnp.ndarray,
    cur_len: int,
    ngram_size: int,
    vocab_size: int,
) -> jnp.ndarray:
    """Block n-grams that would repeat in the generated sequence.

    Args:
        logits: next token logits (batch_size, vocab_size)
        generated_tokens: tokens generated so far (batch_size, max_len)
        cur_len: current decoding position
        ngram_size: size of n-grams to block
        vocab_size: vocabulary size

    Returns:
        modified logits with repeated n-grams blocked
    """
    if ngram_size <= 1:
        return logits

    batch_size = logits.shape[0]
    max_len = generated_tokens.shape[1]

    def apply_blocking():
        context_start = jnp.maximum(0, cur_len - ngram_size + 1)
        context_len = ngram_size - 1

        batch_indices = jnp.arange(batch_size)[:, None]
        context_indices = jnp.arange(context_len)[None, :] + context_start
        context_indices = jnp.minimum(context_indices, max_len - 1)
        context_tokens = generated_tokens[batch_indices, context_indices]

        num_check_positions = jnp.maximum(0, cur_len - ngram_size + 1)
        max_check_positions = max_len
        position_offsets = jnp.arange(ngram_size - 1)[None, :]
        check_positions = jnp.arange(max_check_positions)[:, None] + position_offsets

        check_positions_clamped = jnp.minimum(check_positions, max_len - 1)
        batch_indices_expanded = jnp.arange(batch_size)[:, None, None]
        check_positions_expanded = check_positions_clamped[None, :, :]

        all_ngrams = generated_tokens[batch_indices_expanded, check_positions_expanded]

        context_expanded = context_tokens[:, None, :]
        matches = jnp.all(all_ngrams == context_expanded, axis=2)

        position_indices = jnp.arange(max_check_positions)[None, :]
        valid_mask = position_indices < num_check_positions
        matches = matches & valid_mask

        follow_indices = jnp.arange(max_check_positions) + ngram_size - 1
        follow_indices_clamped = jnp.minimum(follow_indices, max_len - 1)
        batch_indices_2d = jnp.arange(batch_size)[:, None]
        follow_indices_2d = follow_indices_clamped[None, :]

        follow_tokens = generated_tokens[batch_indices_2d, follow_indices_2d]

        def update_banned_for_batch(batch_idx):
            batch_matches = matches[batch_idx]
            batch_follow_tokens = follow_tokens[batch_idx]

            batch_banned = jnp.zeros(vocab_size, dtype=jnp.bool_)

            def update_for_position(carry, idx):
                banned = carry
                is_match = batch_matches[idx]
                token_to_ban = batch_follow_tokens[idx]
                token_mask = jnp.arange(vocab_size) == token_to_ban
                banned = jnp.where(is_match, banned | token_mask, banned)
                return banned, None

            batch_banned, _ = jax.lax.scan(
                update_for_position, batch_banned, jnp.arange(max_check_positions)
            )
            return batch_banned

        banned_mask = jax.vmap(update_banned_for_batch)(jnp.arange(batch_size))
        return jnp.where(banned_mask, -jnp.inf, logits)

    return jax.lax.cond(
        cur_len >= ngram_size,
        apply_blocking,
        lambda: logits,
    )


@flax.struct.dataclass
class _SamplingState:
    """Internal state for greedy/sampling generation."""
    decoding_step: jnp.int32
    num_input_tokens: jnp.ndarray
    token_buffer: jnp.ndarray
    cache: Optional[Any]
    done: jnp.ndarray
    total_sampling_steps: int = flax.struct.field(pytree_node=False)
    encoder_hidden_states: jnp.ndarray = None
    encoder_attention_mask: jnp.ndarray = None
    eos_token_id: Optional[int] = flax.struct.field(pytree_node=False, default=None)
    pad_token_id: int = flax.struct.field(pytree_node=False, default=0)
    min_length: int = flax.struct.field(pytree_node=False, default=0)
    temperature: float = flax.struct.field(pytree_node=False, default=1.0)
    top_k: int = flax.struct.field(pytree_node=False, default=50)
    top_p: float = flax.struct.field(pytree_node=False, default=1.0)
    repetition_penalty: float = flax.struct.field(pytree_node=False, default=1.0)
    no_repeat_ngram_size: int = flax.struct.field(pytree_node=False, default=0)
    suppress_tokens: Optional[Tuple[int, ...]] = flax.struct.field(
        pytree_node=False, default=None
    )
    seed: jax.Array = None


@flax.struct.dataclass
class _BeamSearchState:
    """Internal state for beam search generation (standard KV cache)."""
    decoding_step: jnp.int32
    decoder_tokens: jnp.ndarray
    beam_scores: jnp.ndarray
    cache: Optional[Any]
    finished_mask: jnp.ndarray
    finished_scores: jnp.ndarray
    finished_sequences: jnp.ndarray
    encoder_hidden_states: jnp.ndarray
    encoder_attention_mask: jnp.ndarray
    batch_size: int = flax.struct.field(pytree_node=False)
    num_beams: int = flax.struct.field(pytree_node=False)
    max_length: int = flax.struct.field(pytree_node=False)
    min_length: int = flax.struct.field(pytree_node=False, default=0)
    eos_token_id: Optional[int] = flax.struct.field(pytree_node=False, default=None)
    pad_token_id: int = flax.struct.field(pytree_node=False, default=0)
    length_penalty: float = flax.struct.field(pytree_node=False, default=1.0)
    early_stopping: bool = flax.struct.field(pytree_node=False, default=True)
    repetition_penalty: float = flax.struct.field(pytree_node=False, default=1.0)
    no_repeat_ngram_size: int = flax.struct.field(pytree_node=False, default=0)
    suppress_tokens: Optional[Tuple[int, ...]] = flax.struct.field(
        pytree_node=False, default=None
    )


class SamplerOutput:
    """Output from the sampler."""

    def __init__(
        self,
        tokens: jnp.ndarray,
        text: Optional[List[str]] = None,
        scores: Optional[jnp.ndarray] = None,
    ):
        self.tokens = tokens
        self.text = text
        self.scores = scores

    def __repr__(self):
        return f"SamplerOutput(tokens={self.tokens.shape}, text={len(self.text) if self.text else None})"


class Han2HanSampler:
    """Sampler for Han2Han encoder-decoder transformer.

    This sampler follows the Gemma NNX pattern:
    1. Split model into graphdef + state at initialization
    2. Merge inside JIT-compiled step function
    3. Use lax.while_loop for efficient generation

    Supports:
    - Standard KV-cached generation
    - Beam search
    - Temperature, top-k, top-p sampling
    - Repetition penalty
    - N-gram blocking
    """

    def __init__(
        self,
        model: nnx.Module,
        tokenizer: Any = None,
        max_length: int = 512,
    ):
        """Initialize the sampler.

        Args:
            model: Han2Han model instance
            tokenizer: tokenizer for encoding/decoding (optional)
            max_length: maximum generation length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.config = model.config

        self._encoder_graphdef, self._encoder_state = nnx.split(model.encoder)
        self._decoder_graphdef, self._decoder_rng_state, self._decoder_state = nnx.split(
            model.decoder, nnx.RngState, ...
        )

        if self.config.tie_input_output_embeddings:
            self._lm_head_graphdef = None
            self._lm_head_state = model.decoder.wte.embedding.value
        else:
            self._lm_head_graphdef, self._lm_head_state = nnx.split(model.lm_head)

        self._compiled_sample_fn = jax.jit(self._sample_fn)
        self._compiled_beam_search_fn = jax.jit(self._beam_search_fn)

    @property
    def encoder(self) -> nnx.Module:
        return nnx.merge(self._encoder_graphdef, self._encoder_state)

    @property
    def decoder(self) -> nnx.Module:
        return nnx.merge(self._decoder_graphdef, self._decoder_rng_state, self._decoder_state)

    @property
    def vocab_size(self) -> int:
        if self._lm_head_graphdef is None:
            # tied embeddings - embedding shape is (vocab_size, d_model)
            return self._lm_head_state.shape[0]
        else:
            # separate lm_head - kernel shape is (d_model, vocab_size)
            return self._lm_head_state['kernel'].value.shape[1]

    def _get_logits(
        self,
        hidden_states: jnp.ndarray,
        lm_head_state: jnp.ndarray,
        lm_head_graphdef: Optional[graph.NodeDef],
    ) -> jnp.ndarray:
        """Compute logits from hidden states."""
        if lm_head_graphdef is None:
            return hidden_states @ lm_head_state.T
        else:
            lm_head = nnx.merge(lm_head_graphdef, lm_head_state)
            return lm_head(hidden_states)

    def _apply_logit_processors(
        self,
        logits: jnp.ndarray,
        state: _SamplingState,
        cur_step: jnp.int32,
    ) -> jnp.ndarray:
        """Apply all logit processing (penalties, blocking, suppression)."""
        vocab_size = logits.shape[-1]

        if state.repetition_penalty != 1.0:
            valid_mask = jnp.arange(state.token_buffer.shape[1]) < cur_step
            logits = _apply_repetition_penalty(
                logits, state.token_buffer, state.repetition_penalty, valid_mask
            )

        if state.no_repeat_ngram_size > 0:
            logits = _apply_no_repeat_ngram_blocking(
                logits, state.token_buffer, cur_step, state.no_repeat_ngram_size, vocab_size
            )

        if state.suppress_tokens is not None and len(state.suppress_tokens) > 0:
            logits = logits.at[:, state.suppress_tokens].set(-jnp.inf)

        if state.eos_token_id is not None and state.min_length > 0:
            eos_blocked = logits.at[:, state.eos_token_id].set(-jnp.inf)
            logits = jnp.where(cur_step < state.min_length, eos_blocked, logits)

        return logits

    def _sample_token(
        self,
        logits: jnp.ndarray,
        temperature: float,
        top_k: int,
        top_p: float,
        key: jax.Array,
    ) -> jnp.ndarray:
        """Sample next token from logits."""
        if temperature <= 0:
            return jnp.argmax(logits, axis=-1)

        logits = logits / temperature

        if top_k > 0 and top_k < logits.shape[-1]:
            return _sample_top_k(logits, top_k, key)

        if top_p < 1.0:
            probs = jax.nn.softmax(logits, axis=-1)
            return _sample_top_p(probs, top_p, key)

        return jax.random.categorical(key, logits, axis=-1)

    def _sample_step(
        self,
        encoder_state: statelib.State,
        decoder_state: statelib.State,
        decoder_rng_state: statelib.State,
        lm_head_state: statelib.State,
        state: _SamplingState,
    ) -> _SamplingState:
        """Single sampling step for KV-cached generation."""
        batch_size = state.token_buffer.shape[0]
        cur_step = jnp.int32(state.decoding_step)

        last_token = jax.lax.dynamic_slice(
            state.token_buffer, (jnp.int32(0), cur_step - 1), (batch_size, 1)
        )

        decoder = nnx.merge(
            self._decoder_graphdef, decoder_rng_state, decoder_state
        )
        decoder.eval()

        outputs = decoder(
            input_ids=last_token,
            attention_mask=jnp.ones_like(last_token),
            encoder_hidden_states=state.encoder_hidden_states,
            encoder_attention_mask=state.encoder_attention_mask,
            past_key_values=state.cache,
            init_cache=False,
            return_dict=True,
            rngs=None,
        )

        hidden_states = outputs.last_hidden_state[:, -1, :]
        logits = self._get_logits(hidden_states, lm_head_state, self._lm_head_graphdef)

        logits = self._apply_logit_processors(logits, state, cur_step)

        key = jax.random.fold_in(state.seed, cur_step) if state.seed is not None else None
        if state.temperature > 0 and key is not None:
            next_token = self._sample_token(
                logits, state.temperature, state.top_k, state.top_p, key
            )
        else:
            next_token = jnp.argmax(logits, axis=-1)

        next_token = next_token.astype(jnp.int32)
        next_token = jnp.where(state.done, state.pad_token_id, next_token)

        new_token_buffer = jax.lax.dynamic_update_slice(
            state.token_buffer, next_token[:, None], (jnp.int32(0), cur_step)
        )

        new_done = state.done
        if state.eos_token_id is not None:
            new_done = state.done | (next_token == state.eos_token_id)

        return _SamplingState(
            decoding_step=state.decoding_step + 1,
            num_input_tokens=state.num_input_tokens,
            token_buffer=new_token_buffer,
            cache=outputs.past_key_values,
            done=new_done,
            total_sampling_steps=state.total_sampling_steps,
            encoder_hidden_states=state.encoder_hidden_states,
            encoder_attention_mask=state.encoder_attention_mask,
            eos_token_id=state.eos_token_id,
            pad_token_id=state.pad_token_id,
            min_length=state.min_length,
            temperature=state.temperature,
            top_k=state.top_k,
            top_p=state.top_p,
            repetition_penalty=state.repetition_penalty,
            no_repeat_ngram_size=state.no_repeat_ngram_size,
            suppress_tokens=state.suppress_tokens,
            seed=state.seed,
        )

    def _sample_fn(
        self,
        encoder_state: statelib.State,
        decoder_state: statelib.State,
        decoder_rng_state: statelib.State,
        lm_head_state: statelib.State,
        initial_state: _SamplingState,
    ) -> _SamplingState:
        """JIT-compiled sampling function using lax.while_loop."""

        def cond_fn(state: _SamplingState) -> jnp.bool_:
            return (state.decoding_step < state.total_sampling_steps) & jnp.any(
                ~state.done
            )

        def body_fn(state: _SamplingState) -> _SamplingState:
            return self._sample_step(
                encoder_state, decoder_state, decoder_rng_state, lm_head_state, state
            )

        return jax.lax.while_loop(cond_fn, body_fn, initial_state)

    def _beam_search_step(
        self,
        decoder_state: statelib.State,
        decoder_rng_state: statelib.State,
        lm_head_state: statelib.State,
        state: _BeamSearchState,
    ) -> _BeamSearchState:
        """Single beam search step."""
        cur_step = jnp.int32(state.decoding_step)
        beam_batch_size = state.batch_size * state.num_beams
        vocab_size = self.vocab_size

        last_token = jax.lax.dynamic_slice(
            state.decoder_tokens, (jnp.int32(0), cur_step - 1), (beam_batch_size, 1)
        )

        decoder = nnx.merge(
            self._decoder_graphdef, decoder_rng_state, decoder_state
        )
        decoder.eval()

        outputs = decoder(
            input_ids=last_token,
            attention_mask=jnp.ones_like(last_token),
            encoder_hidden_states=state.encoder_hidden_states,
            encoder_attention_mask=state.encoder_attention_mask,
            past_key_values=state.cache,
            init_cache=False,
            return_dict=True,
            rngs=None,
        )

        hidden_states = outputs.last_hidden_state[:, -1, :]
        logits = self._get_logits(hidden_states, lm_head_state, self._lm_head_graphdef)

        if state.repetition_penalty != 1.0:
            valid_mask = jnp.arange(state.decoder_tokens.shape[1]) < cur_step
            logits = _apply_repetition_penalty(
                logits, state.decoder_tokens, state.repetition_penalty, valid_mask
            )

        if state.no_repeat_ngram_size > 0:
            logits = _apply_no_repeat_ngram_blocking(
                logits, state.decoder_tokens, cur_step, state.no_repeat_ngram_size, vocab_size
            )

        if state.suppress_tokens is not None and len(state.suppress_tokens) > 0:
            logits = logits.at[:, state.suppress_tokens].set(-jnp.inf)

        if state.eos_token_id is not None and state.min_length > 0:
            eos_blocked = logits.at[:, state.eos_token_id].set(-jnp.inf)
            logits = jnp.where(cur_step < state.min_length, eos_blocked, logits)

        next_scores = jax.nn.log_softmax(logits, axis=-1)
        next_scores = next_scores + state.beam_scores[:, None]

        next_scores_flat = next_scores.reshape(state.batch_size, state.num_beams * vocab_size)
        top_scores, top_indices = jax.lax.top_k(next_scores_flat, 2 * state.num_beams)

        beam_indices = top_indices // vocab_size
        token_ids = top_indices % vocab_size

        eos_id = state.eos_token_id if state.eos_token_id is not None else -1
        is_eos = token_ids == eos_id

        length_penalties = jnp.power(
            jnp.float32(cur_step + 1), jnp.float32(state.length_penalty)
        )
        finished_scores_candidates = top_scores / length_penalties

        continuing_mask = ~is_eos
        continuing_scores = jnp.where(continuing_mask, top_scores, -jnp.inf)

        selected_scores, selected_indices = jax.lax.top_k(continuing_scores, state.num_beams)

        batch_indices = jnp.arange(state.batch_size)[:, None]
        selected_beam_indices = beam_indices[batch_indices, selected_indices]
        selected_tokens = token_ids[batch_indices, selected_indices]

        base_beam_indices = jnp.arange(state.batch_size)[:, None] * state.num_beams
        flat_beam_indices = (base_beam_indices + selected_beam_indices).reshape(-1)

        new_decoder_tokens = state.decoder_tokens[flat_beam_indices]
        new_decoder_tokens = jax.lax.dynamic_update_slice(
            new_decoder_tokens,
            selected_tokens.reshape(-1)[:, None],
            (jnp.int32(0), cur_step),
        )

        new_cache = self._reorder_cache_for_beams(outputs.past_key_values, flat_beam_indices)

        new_finished_mask = is_eos[batch_indices, selected_indices].reshape(-1)

        selected_is_finished = is_eos[batch_indices, selected_indices]
        new_finished_scores = jnp.where(
            selected_is_finished,
            finished_scores_candidates[batch_indices, selected_indices],
            -jnp.inf,
        )
        combined_finished_scores = jnp.maximum(state.finished_scores, new_finished_scores)

        current_sequences = new_decoder_tokens.reshape(state.batch_size, state.num_beams, -1)
        new_finished_sequences = jnp.where(
            new_finished_mask.reshape(state.batch_size, state.num_beams, 1),
            current_sequences,
            state.finished_sequences,
        )

        return _BeamSearchState(
            decoding_step=cur_step + 1,
            decoder_tokens=new_decoder_tokens,
            beam_scores=selected_scores.reshape(-1),
            cache=new_cache,
            finished_mask=new_finished_mask,
            finished_scores=combined_finished_scores,
            finished_sequences=new_finished_sequences,
            encoder_hidden_states=state.encoder_hidden_states,
            encoder_attention_mask=state.encoder_attention_mask,
            batch_size=state.batch_size,
            num_beams=state.num_beams,
            max_length=state.max_length,
            min_length=state.min_length,
            eos_token_id=state.eos_token_id,
            pad_token_id=state.pad_token_id,
            length_penalty=state.length_penalty,
            early_stopping=state.early_stopping,
            repetition_penalty=state.repetition_penalty,
            no_repeat_ngram_size=state.no_repeat_ngram_size,
            suppress_tokens=state.suppress_tokens,
        )

    def _beam_search_fn(
        self,
        decoder_state: statelib.State,
        decoder_rng_state: statelib.State,
        lm_head_state: statelib.State,
        initial_state: _BeamSearchState,
    ) -> _BeamSearchState:
        """JIT-compiled beam search function."""

        def cond_fn(state: _BeamSearchState) -> jnp.bool_:
            not_at_max = state.decoding_step < state.max_length
            not_all_finished = ~jnp.all(state.finished_mask)
            return not_at_max & not_all_finished

        def body_fn(state: _BeamSearchState) -> _BeamSearchState:
            return self._beam_search_step(
                decoder_state, decoder_rng_state, lm_head_state, state
            )

        return jax.lax.while_loop(cond_fn, body_fn, initial_state)

    def _reorder_cache_for_beams(
        self, cache: Optional[Any], beam_indices: jnp.ndarray
    ) -> Optional[Any]:
        """Reorder KV cache for selected beams."""
        if cache is None:
            return None

        def reorder_layer(layer_cache):
            if layer_cache is None:
                return None
            k, v = layer_cache
            return (jnp.take(k, beam_indices, axis=0), jnp.take(v, beam_indices, axis=0))

        return tuple(reorder_layer(layer) for layer in cache)

    def _encode(
        self,
        input_ids: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Encode input sequence."""
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)

        encoder = nnx.merge(self._encoder_graphdef, self._encoder_state)
        encoder.eval()

        encoder_outputs = encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
            rngs=None,
        )

        return encoder_outputs.last_hidden_state, attention_mask

    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        decoder_input_ids: Optional[jnp.ndarray] = None,
        max_length: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        min_length: int = 0,
        num_beams: int = 1,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        length_penalty: float = 1.0,
        early_stopping: bool = True,
        decoder_start_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        suppress_tokens: Optional[List[int]] = None,
        seed: Optional[jax.Array] = None,
        return_text: bool = False,
        local_batch_size: Optional[int] = None,
    ) -> SamplerOutput:
        """Generate sequences from input.

        Args:
            input_ids: encoder input token ids (batch_size, seq_len)
            attention_mask: encoder attention mask
            decoder_input_ids: optional decoder prompt
            max_length: maximum total length
            max_new_tokens: maximum new tokens to generate
            min_length: minimum length before EOS is allowed
            num_beams: number of beams for beam search (1 = greedy/sampling)
            temperature: sampling temperature (0 = greedy)
            top_k: top-k sampling parameter
            top_p: nucleus sampling parameter
            repetition_penalty: penalty for repeated tokens
            no_repeat_ngram_size: block n-grams from repeating
            length_penalty: beam search length penalty
            early_stopping: stop beam search when enough beams finish
            decoder_start_token_id: token to start decoding
            eos_token_id: end of sequence token
            pad_token_id: padding token
            suppress_tokens: tokens to never generate
            seed: random seed for sampling
            return_text: whether to decode tokens to text

        Returns:
            SamplerOutput with generated tokens (and optionally text)
        """
        # use local_batch_size for SPMD (global array shape != local batch)
        batch_size = local_batch_size if local_batch_size is not None else input_ids.shape[0]

        decoder_start_token_id = decoder_start_token_id or getattr(
            self.config, 'decoder_start_token_id', 1
        )
        eos_token_id = eos_token_id or getattr(self.config, 'eos_token_id', 2)
        pad_token_id = pad_token_id or getattr(self.config, 'pad_token_id', 0)

        if max_length is None:
            if max_new_tokens is not None:
                prompt_len = decoder_input_ids.shape[1] if decoder_input_ids is not None else 1
                max_length = prompt_len + max_new_tokens
            else:
                max_length = self.max_length

        encoder_hidden_states, encoder_attention_mask = self._encode(
            input_ids, attention_mask
        )

        if seed is None:
            seed = jax.random.PRNGKey(0)

        suppress_tokens_tuple = tuple(suppress_tokens) if suppress_tokens else None

        if num_beams > 1:
            tokens = self._generate_beam_search(
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_start_token_id=decoder_start_token_id,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                early_stopping=early_stopping,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                suppress_tokens=suppress_tokens_tuple,
                batch_size=batch_size,
            )
        else:
            tokens = self._generate_cached(
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_start_token_id=decoder_start_token_id,
                max_length=max_length,
                min_length=min_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                suppress_tokens=suppress_tokens_tuple,
                seed=seed,
                batch_size=batch_size,
            )

        text = None
        if return_text and self.tokenizer is not None:
            text = [self.tokenizer.decode(t.tolist()) for t in tokens]

        return SamplerOutput(tokens=tokens, text=text)

    def _generate_cached(
        self,
        encoder_hidden_states: jnp.ndarray,
        encoder_attention_mask: jnp.ndarray,
        decoder_input_ids: Optional[jnp.ndarray],
        decoder_start_token_id: int,
        max_length: int,
        min_length: int,
        temperature: float,
        top_k: int,
        top_p: float,
        eos_token_id: int,
        pad_token_id: int,
        repetition_penalty: float,
        no_repeat_ngram_size: int,
        suppress_tokens: Optional[Tuple[int, ...]],
        seed: jax.Array,
        batch_size: int,
    ) -> jnp.ndarray:
        """Generate with KV caching."""
        # use encoder's actual batch dimension to ensure decoder tensors match
        encoder_batch = encoder_hidden_states.shape[0]
        token_buffer = jnp.full((encoder_batch, max_length), pad_token_id, dtype=jnp.int32)

        if decoder_input_ids is not None:
            prompt_len = decoder_input_ids.shape[1]
            token_buffer = token_buffer.at[:, :prompt_len].set(decoder_input_ids)
            cur_len = prompt_len
        else:
            token_buffer = token_buffer.at[:, 0].set(decoder_start_token_id)
            cur_len = 1

        decoder = nnx.merge(
            self._decoder_graphdef, self._decoder_rng_state, self._decoder_state
        )
        decoder.eval()

        init_outputs = decoder(
            input_ids=token_buffer[:, :cur_len],
            attention_mask=jnp.ones((encoder_batch, cur_len), dtype=jnp.int32),
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=None,
            init_cache=True,
            return_dict=True,
            rngs=None,
        )
        cache = init_outputs.past_key_values

        initial_state = _SamplingState(
            decoding_step=jnp.int32(cur_len),
            num_input_tokens=jnp.array([cur_len] * encoder_batch, dtype=jnp.int32),
            token_buffer=token_buffer,
            cache=cache,
            done=jnp.zeros(encoder_batch, dtype=jnp.bool_),
            total_sampling_steps=max_length,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            min_length=min_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            suppress_tokens=suppress_tokens,
            seed=seed,
        )

        final_state = self._compiled_sample_fn(
            self._encoder_state,
            self._decoder_state,
            self._decoder_rng_state,
            self._lm_head_state,
            initial_state,
        )

        return final_state.token_buffer

    def _generate_beam_search(
        self,
        encoder_hidden_states: jnp.ndarray,
        encoder_attention_mask: jnp.ndarray,
        decoder_input_ids: Optional[jnp.ndarray],
        decoder_start_token_id: int,
        max_length: int,
        min_length: int,
        num_beams: int,
        length_penalty: float,
        early_stopping: bool,
        eos_token_id: int,
        pad_token_id: int,
        repetition_penalty: float,
        no_repeat_ngram_size: int,
        suppress_tokens: Optional[Tuple[int, ...]],
        batch_size: int,
    ) -> jnp.ndarray:
        """Generate with beam search."""
        # use encoder's actual batch dimension to ensure decoder tensors match
        encoder_batch = encoder_hidden_states.shape[0]
        beam_batch_size = encoder_batch * num_beams

        encoder_hidden_states = jnp.repeat(encoder_hidden_states, num_beams, axis=0)
        encoder_attention_mask = jnp.repeat(encoder_attention_mask, num_beams, axis=0)

        decoder_tokens = jnp.full(
            (beam_batch_size, max_length), pad_token_id, dtype=jnp.int32
        )

        if decoder_input_ids is not None:
            prompt_len = decoder_input_ids.shape[1]
            decoder_input_ids_expanded = jnp.repeat(decoder_input_ids, num_beams, axis=0)
            decoder_tokens = decoder_tokens.at[:, :prompt_len].set(decoder_input_ids_expanded)
            cur_len = prompt_len
        else:
            decoder_tokens = decoder_tokens.at[:, 0].set(decoder_start_token_id)
            cur_len = 1

        beam_scores = jnp.full((beam_batch_size,), -jnp.inf)
        beam_scores = beam_scores.at[::num_beams].set(0.0)

        decoder = nnx.merge(
            self._decoder_graphdef, self._decoder_rng_state, self._decoder_state
        )
        decoder.eval()

        init_outputs = decoder(
            input_ids=decoder_tokens[:, :cur_len],
            attention_mask=jnp.ones((beam_batch_size, cur_len), dtype=jnp.int32),
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=None,
            init_cache=True,
            return_dict=True,
            rngs=None,
        )
        cache = init_outputs.past_key_values

        initial_state = _BeamSearchState(
            decoding_step=jnp.int32(cur_len),
            decoder_tokens=decoder_tokens,
            beam_scores=beam_scores,
            cache=cache,
            finished_mask=jnp.zeros(beam_batch_size, dtype=jnp.bool_),
            finished_scores=jnp.full((encoder_batch, num_beams), -jnp.inf),
            finished_sequences=jnp.full(
                (encoder_batch, num_beams, max_length), pad_token_id, dtype=jnp.int32
            ),
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            batch_size=encoder_batch,
            num_beams=num_beams,
            max_length=max_length,
            min_length=min_length,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            suppress_tokens=suppress_tokens,
        )

        final_state = self._compiled_beam_search_fn(
            self._decoder_state,
            self._decoder_rng_state,
            self._lm_head_state,
            initial_state,
        )

        finished_scores_per_batch = final_state.finished_scores
        finished_sequences_per_batch = final_state.finished_sequences

        active_scores = final_state.beam_scores.reshape(encoder_batch, num_beams)
        active_sequences = final_state.decoder_tokens.reshape(encoder_batch, num_beams, -1)

        all_scores = jnp.concatenate([finished_scores_per_batch, active_scores], axis=1)
        all_sequences = jnp.concatenate(
            [finished_sequences_per_batch, active_sequences], axis=1
        )

        top_indices = jnp.argsort(all_scores, axis=1)[:, -1:][:, ::-1]

        batch_indices = jnp.arange(encoder_batch)[:, None]
        best_sequences = all_sequences[batch_indices, top_indices]

        return best_sequences.reshape(encoder_batch, -1)

