"""Subword feature calculation for Han2Han.

Han2Han's core innovation: implicit phonetic-to-semantic understanding through
jamo n-gram and character-level features that capture Korean's unique structure.

Usage:
    from subword_features import compute_subword_tables

    jbu, cbu, config_updates = compute_subword_tables(
        tokenizer,
        jamo_subwords=True,
        char_subwords=True,
        ngram_sizes="2,3,4",
    )

    model = FlaxHan2Han(config, jamo_buckets=jbu, char_buckets=cbu)
"""

import math
import re
import numpy as np
from collections import Counter, OrderedDict
from typing import Optional

from han2han_tools import (
    transcribe, jamo_sentence,
    ngrams_bytes, hash_bytes,
    ngrams_bytes_specific,
)

BYTE_TOKEN_PATTERN = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")
SENTINEL_TOKEN_PATTERN = re.compile(r"^<extra_id_\d+>$")


def _is_extended_token(token: str) -> bool:
    """Check if token is a byte fallback or sentinel token.

    These tokens were added during vocab extension and have no meaningful
    jamo/character decomposition. They should use padding in subword tables
    to keep embedding matrices compatible with pre-extension checkpoints.
    """
    return bool(BYTE_TOKEN_PATTERN.match(token) or SENTINEL_TOKEN_PATTERN.match(token))


def compute_subword_tables(
    tokenizer,
    jamo_subwords: bool = True,
    char_subwords: bool = True,
    ngram_sizes: Optional[str] = "2,3,4",
    min_n: int = 2,
    max_n: int = 4,
    pad_idx: int = 0,
    align_to: int = 128,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], dict]:
    """Compute jamo and character subword lookup tables for Han2Han.

    Args:
        tokenizer: tokenizer with get_vocab() method
        jamo_subwords: whether to compute jamo n-gram features
        char_subwords: whether to compute character-level features
        ngram_sizes: comma-separated discrete n-gram sizes (e.g., "2,3,4")
                     if None, uses min_n/max_n range instead
        min_n: minimum n-gram size (used if ngram_sizes is None)
        max_n: maximum n-gram size (used if ngram_sizes is None)
        pad_idx: padding index (0 by default, features use 1-based indexing)
        align_to: align bucket counts to this value for TPU efficiency

    Returns:
        tuple of (jbu, cbu, config_updates) where:
            - jbu: jamo bucket array of shape (vocab_size, max_ngrams), or None
            - cbu: char bucket array of shape (vocab_size, max_chars), or None
            - config_updates: dict with num_jamo_buckets, num_char_buckets, etc.
    """
    tok2idx = tokenizer.get_vocab()

    use_discrete_ngrams = ngram_sizes is not None
    use_range_ngrams = not use_discrete_ngrams

    if use_discrete_ngrams:
        ngram_sizes_list = [int(n.strip()) for n in ngram_sizes.split(",")]
    else:
        ngram_sizes_list = []

    jbu = None
    cbu = None
    config_updates = {
        "jamo_subwords": jamo_subwords,
        "char_subwords": char_subwords,
    }

    if jamo_subwords:
        jbu, num_jamo_buckets = _compute_jamo_buckets(
            tok2idx,
            use_discrete_ngrams,
            use_range_ngrams,
            ngram_sizes_list,
            min_n,
            max_n,
            pad_idx,
            align_to,
        )
        config_updates["num_jamo_buckets"] = num_jamo_buckets

    if char_subwords:
        cbu, num_char_buckets = _compute_char_buckets(
            tok2idx,
            pad_idx,
            align_to,
        )
        config_updates["num_char_buckets"] = num_char_buckets

    return jbu, cbu, config_updates


def _compute_jamo_buckets(
    tok2idx: dict,
    use_discrete_ngrams: bool,
    use_range_ngrams: bool,
    ngram_sizes_list: list,
    min_n: int,
    max_n: int,
    pad_idx: int,
    align_to: int,
) -> tuple[np.ndarray, int]:
    """Compute jamo n-gram hash buckets for each vocabulary token."""

    def jamo_ngram_hashes(word, num_buckets):
        if _is_extended_token(word):
            return []
        try:
            if use_discrete_ngrams:
                encoded_ngrams = ngrams_bytes_specific(
                    jamo_sentence(transcribe(word)), ns=ngram_sizes_list
                )
            elif use_range_ngrams:
                encoded_ngrams = ngrams_bytes(
                    jamo_sentence(transcribe(word)), min_n=min_n, max_n=max_n
                )
            else:
                return []

            hashes = [hash_bytes(n) % max(1, num_buckets) for n in encoded_ngrams]
            return hashes
        except Exception:
            return []

    def calc_num_jamo_buckets(vocab):
        ngrams = set()
        for term in vocab:
            if _is_extended_token(term):
                continue
            try:
                if use_discrete_ngrams:
                    ngrams.update(
                        ngrams_bytes_specific(
                            jamo_sentence(transcribe(term)), ns=ngram_sizes_list
                        )
                    )
                elif use_range_ngrams:
                    ngrams.update(
                        ngrams_bytes(
                            jamo_sentence(transcribe(term)), min_n=min_n, max_n=max_n
                        )
                    )
            except Exception:
                pass
        return max(1, len(ngrams))

    num_jamo_buckets = calc_num_jamo_buckets(tok2idx)
    jamo_buckets_words = OrderedDict(
        {tok: jamo_ngram_hashes(tok, num_jamo_buckets) for tok in tok2idx}
    )
    base_vocab_buckets = [b for tok, b in jamo_buckets_words.items()
                         if not _is_extended_token(tok)]
    longest_jamo_len = max([len(b) for b in base_vocab_buckets] + [0])
    longest_jamo_len = align_to * math.ceil(max(longest_jamo_len, 1) / align_to)

    jbu_list = []
    for tok in tok2idx:
        hashes = jamo_buckets_words.get(tok, [])
        padded_hashes = [h + 1 for h in hashes] + [pad_idx] * (
            longest_jamo_len - len(hashes)
        )
        jbu_list.append(padded_hashes)

    jbu = np.array(jbu_list, dtype=np.int32)

    max_jamo_idx = jbu.max() if jbu.size > 0 else 0
    num_jamo_buckets = align_to * math.ceil(
        max(num_jamo_buckets + 1, max_jamo_idx + 1) / align_to
    )

    return jbu, num_jamo_buckets


def _compute_char_buckets(
    tok2idx: dict,
    pad_idx: int,
    align_to: int,
) -> tuple[np.ndarray, int]:
    """Compute character-level feature indices for each vocabulary token."""
    base_vocab = [tok for tok in tok2idx if not _is_extended_token(tok)]
    cfs = OrderedDict(Counter("".join(base_vocab)).most_common())
    ch2idx = {ch: id + 1 for id, ch in enumerate(cfs)}

    num_chars_calc = len(ch2idx)

    tok2ch = {}
    for tok in tok2idx:
        if _is_extended_token(tok):
            continue
        char_ids = [ch2idx.get(c) for c in tok]
        if all(cid is not None for cid in char_ids):
            tok2ch[tok] = char_ids

    longest_token_len = max([len(t) for t in tok2ch.values()] + [0])
    longest_token_len = align_to * math.ceil(max(longest_token_len, 1) / align_to)

    cbu_list = []
    for tok in tok2idx:
        ids = tok2ch.get(tok, [])
        padded_ids = ids + [pad_idx] * (longest_token_len - len(ids))
        cbu_list.append(padded_ids)

    cbu = np.array(cbu_list, dtype=np.int32)

    max_char_idx = cbu.max() if cbu.size > 0 else 0
    num_char_buckets = align_to * math.ceil(
        max(num_chars_calc + 1, max_char_idx + 1) / align_to
    )

    return cbu, num_char_buckets


def extend_subword_tables(
    jbu: Optional[np.ndarray],
    cbu: Optional[np.ndarray],
    old_vocab_size: int,
    new_vocab_size: int,
    pad_token_id: int = 0,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Extend jbu/cbu lookup tables for vocab extension.

    When restoring a checkpoint trained with base vocab, the jbu/cbu arrays
    have shape (old_vocab_size, ...). This function extends them to
    (new_vocab_size, ...) by appending padding rows.

    Args:
        jbu: Existing jamo bucket array of shape (old_vocab_size, max_ngrams), or None
        cbu: Existing char bucket array of shape (old_vocab_size, max_chars), or None
        old_vocab_size: Original vocabulary size
        new_vocab_size: New (extended) vocabulary size
        pad_token_id: Token ID to use for padding values (default: 0)

    Returns:
        tuple of (extended_jbu, extended_cbu)
    """
    if new_vocab_size <= old_vocab_size:
        return jbu, cbu

    num_new = new_vocab_size - old_vocab_size

    new_jbu = None
    if jbu is not None:
        if pad_token_id < len(jbu):
            pad_bucket = jbu[pad_token_id]
            extension = np.tile(pad_bucket[None, :], (num_new, 1))
            new_jbu = np.concatenate([jbu, extension], axis=0)
        else:
            new_jbu = np.pad(jbu, ((0, num_new), (0, 0)), mode='constant', constant_values=0)
        new_jbu = new_jbu.astype(np.int32)

    new_cbu = None
    if cbu is not None:
        if pad_token_id < len(cbu):
            pad_bucket = cbu[pad_token_id]
            extension = np.tile(pad_bucket[None, :], (num_new, 1))
            new_cbu = np.concatenate([cbu, extension], axis=0)
        else:
            new_cbu = np.pad(cbu, ((0, num_new), (0, 0)), mode='constant', constant_values=0)
        new_cbu = new_cbu.astype(np.int32)

    return new_jbu, new_cbu
