#!/usr/bin/env python3
# coding: utf-8

import dataclasses
from typing import Optional, Dict, Any, List, Union
from threading import Lock
import threading
import math
import pickle
import logging
from pathlib import Path

import polars as pl 

import numpy as np

import jax

from transformers import (
    PreTrainedTokenizerBase,
    BatchEncoding,
    AutoTokenizer
)

from han2han_tools import transcribe, has_hanja
from mecab_morphs_preprocessing import hanja_aware_morpheme_tokenization

logger = logging.getLogger(__name__)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.info("WandB not available for logging Japanese text examples")

# Japanese character ranges
HIRAGANA_RANGE = (0x3040, 0x309F)
KATAKANA_RANGE = (0x30A0, 0x30FF)
KANJI_RANGE = (0x4E00, 0x9FFF)  # CJK unified ideographs (includes both Chinese and Japanese)

def contains_japanese(text: str) -> bool:
    """Check if text contains Japanese characters (Hiragana or Katakana)."""
    for char in text:
        code_point = ord(char)
        if (HIRAGANA_RANGE[0] <= code_point <= HIRAGANA_RANGE[1] or 
            KATAKANA_RANGE[0] <= code_point <= KATAKANA_RANGE[1]):
            return True
    return False

# import mecab for morphological analysis
try:
    from konlpy.tag import Mecab
    MECAB_AVAILABLE = True
except ImportError:
    MECAB_AVAILABLE = False

# import kiwi as the superior alternative
try:
    from kiwipiepy import Kiwi
    KIWI_AVAILABLE = True
except ImportError:
    KIWI_AVAILABLE = False

# === CONSTANTS ===
DEFAULT_BUCKET_SIZES = (128, 256, 512, 1024, 2048, 4096, 8192)


def shift_tokens_right(input_ids: np.ndarray, pad_token_id: int, decoder_start_token_ids) -> np.ndarray:
    """
    Shift input ids one token to the right (numpy version).

    Args:
        input_ids: Label token ids
        pad_token_id: Padding token id
        decoder_start_token_ids: Either int or array of start tokens (one per example)
    """
    shifted_input_ids = np.zeros_like(input_ids)
    shifted_input_ids[:, 1:] = input_ids[:, :-1]

    # Handle per-example start tokens
    if isinstance(decoder_start_token_ids, (list, np.ndarray)):
        # Different start token for each example
        for i, start_token in enumerate(decoder_start_token_ids):
            shifted_input_ids[i, 0] = start_token
    else:
        # Same start token for all examples
        shifted_input_ids[:, 0] = decoder_start_token_ids

    shifted_input_ids = np.where(shifted_input_ids == -100, pad_token_id, shifted_input_ids)
    return shifted_input_ids


# === DATA COLLATOR ===
@dataclasses.dataclass
class DataCollator:
    """
    Handles tokenization, noise application, subword feature lookup, and batching.
    Operates on the local data for the current process using pre-distributed indices.
    Supports gradient accumulation by yielding micro-batches.
    """
    tokenizer: PreTrainedTokenizerBase
    rng: np.random.Generator
    max_length: int = 512
    model_max_length: int = 512
    deletion_ratio: float = 0.6
    dataset: pl.DataFrame = dataclasses.field(default_factory=pl.DataFrame)
    indices_per_bucket: list = dataclasses.field(default_factory=list)
    batch_size: int = 16 # base effective batch size per process (before accumulation)
    eval_batch_size: int = 32 # base effective eval batch size per process
    bucket_sizes: tuple = DEFAULT_BUCKET_SIZES
    use_bucketing: bool = False
    adaptive_batch_size: bool = False
    batch_scale_factor: float = 1.0
    accumulation_steps: int = 1
    resume_rng_state: Optional[bytes] = None
    resume_iter_state: Optional[Dict[str, Any]] = None
    use_mecab_deletion: float = 0.0  # probability of using mecab-based deletion
    mecab_tokenizer: Any = dataclasses.field(default=None, init=False)
    hangul_decoder: Union[bool, str] = True
    hangul_only: bool = False
    han2han_transcription_ratio: Optional[float] = None  # ratio of samples to apply transcription to
    disable_stratified_sampling: bool = False  # flag to disable stratified sampling logic
    
    # Track Japanese text examples for WandB logging
    _japanese_examples: list = dataclasses.field(default_factory=list, init=False)
    _japanese_examples_lock: Lock = dataclasses.field(default_factory=Lock, init=False)

    def log_japanese_example(self, original_text: str, transcribed_text: str):
        """Log examples where Japanese remains after transcription."""
        with self._japanese_examples_lock:
            self._japanese_examples.append({
                "original": original_text,  # Log full text
                "transcribed": transcribed_text,  # Log full text
                "timestamp": logging.Formatter().formatTime(logging.LogRecord("", 0, "", 0, "", (), None))
            })
            
            # Log to WandB every 10 examples if available
            if WANDB_AVAILABLE and len(self._japanese_examples) % 10 == 0:
                try:
                    wandb.log({
                        "japanese_text_examples": wandb.Table(
                            columns=["original", "transcribed", "timestamp"],
                            data=[list(ex.values()) for ex in self._japanese_examples[-10:]]
                        )
                    })
                except Exception as e:
                    logger.debug(f"Failed to log to WandB: {e}")

    def __post_init__(self):
        """Initialize morpheme tokenizers if available."""
        # initialize mecab as primary tokenizer
        self.mecab_instance = None  # store full Mecab instance for hanja_aware function
        if MECAB_AVAILABLE and self.mecab_tokenizer is None:
            try:
                self.mecab_instance = Mecab()
                self.mecab_tokenizer = self.mecab_instance.morphs
                logger.info("MeCab initialized as primary tokenizer")
            except:
                try:
                    self.mecab_instance = Mecab("/usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ko-dic")
                    self.mecab_tokenizer = self.mecab_instance.morphs
                    logger.info("MeCab initialized with explicit dictionary path")
                except Exception as e:
                    logger.warning(f"Failed to initialize Mecab: {e}")
                    self.mecab_tokenizer = None
                    self.mecab_instance = None
        
        # lazy initialization for kiwi as fallback - will be initialized per-worker
        self._kiwi_instances = {}  # {thread_id: Kiwi instance}
        self._kiwi_lock = Lock()
        
        # morphs cache for on-the-fly computation
        self._cache_size_threshold = 10000  # save cache after this many entries
        self._last_saved_cache_size = 0  # track last saved size to avoid redundant saves

        # initialize epoch tracking cache for non-repetition
        self._seen_examples_cache = set()  # tracks all examples seen across epochs
        self._current_epoch_examples = set()  # tracks examples in current epoch

        # initialize source distribution tracking
        self._source_indices_map = {}  # maps source -> list of indices
        self._source_seen_counts = {}  # tracks how many examples seen per source
        self._epoch_count = 0  # track epoch number
        
        # if dataset is provided at init, build source index map
        if len(self.dataset) > 0 and 'source' in self.dataset.columns:
            self._build_source_index_map()

    def _build_source_index_map(self):
        """Build a mapping of source -> indices to track distribution."""
        self._source_indices_map = {}
        self._source_seen_counts = {}

        # create local indices for this collator instance
        # drop any existing index column and create new local indices
        self.dataset = self.dataset.drop('index', strict=False).with_row_index('local_index')
        
        # group by source and get local indices for each source
        grouped = self.dataset.group_by('source').agg(pl.col('local_index'))
        
        # convert to dictionary mapping
        for row in grouped.iter_rows(named=True):
            source = row['source']
            indices = row['local_index']  # this is already a list of local indices
            self._source_indices_map[source] = indices
            self._source_seen_counts[source] = 0

        # log source distribution
        total_samples = len(self.dataset)
        logger.info(f"Source distribution in dataset (total {total_samples} samples):")
        sorted_sources = sorted(self._source_indices_map.items(), key=lambda x: x[0], reverse=True)
        for source, indices in sorted_sources:
            pct = len(indices) / total_samples * 100
            logger.info(f"  {source}: {len(indices)} samples ({pct:.1f}%)")

    def _get_available_indices_by_source(self):
        """Get indices that haven't been seen yet, organized by source."""
        available_by_source = {}
        
        # debug logging
        total_seen = len(self._seen_examples_cache)
        total_available = sum(len(indices) for indices in self._source_indices_map.values())
        logger.debug(f"Cache status: {total_seen}/{total_available} indices seen")
        
        # collect available indices for each source
        for source, all_indices in self._source_indices_map.items():
            # filter out indices that have been seen
            available = [idx for idx in all_indices if idx not in self._seen_examples_cache]
            available_by_source[source] = available
            logger.debug(f"Source {source}: {len(available)}/{len(all_indices)} available")
            
        return available_by_source

    def get_bucket_batch_size(self, bucket_size, for_eval=False):
        """Calculates the per-process micro-batch size for a given bucket."""
        if not self.adaptive_batch_size:
            base_size = self.eval_batch_size if for_eval else self.batch_size
            # return the base size divided by accumulation steps for training
            if not for_eval and self.accumulation_steps > 1:
                return max(1, base_size // self.accumulation_steps)
            return base_size

        # hard-coded global batch sizes (target effective global batch size)
        target_global_batch_size = 32 # default fallback
        if bucket_size <= 128: target_global_batch_size = 2048
        elif bucket_size <= 256: target_global_batch_size = 1024
        elif bucket_size <= 512: target_global_batch_size = 512
        elif bucket_size <= 1024: target_global_batch_size = 256
        elif bucket_size <= 2048: target_global_batch_size = 128
        elif bucket_size <= 4096: target_global_batch_size = 64
        elif bucket_size <= 8192: target_global_batch_size = 32

        if not for_eval:
            scaled_global_size = int(target_global_batch_size * self.batch_scale_factor)
        else:
            scaled_global_size = target_global_batch_size # no scaling for eval, to be safe
        num_processes = jax.process_count()
        local_devices = jax.local_device_count()

        # calculate per-process size
        per_process_bs = max(local_devices, scaled_global_size // num_processes)

        # helper function for TPU alignment
        def align_for_tpu(size):
            if size >= 128:
                return max(local_devices, (size // 128) * 128)  # align to 128 for larger sizes
            else:
                return max(local_devices, (size // 8) * 8)      # align to 8 for smaller sizes

        # apply TPU alignment
        per_process_bs = align_for_tpu(per_process_bs)

        # divide by accumulation_steps for training, but not for evaluation
        if not for_eval and self.accumulation_steps > 1:
            # ensure divisibility by accumulation_steps for clean accumulation
            per_process_bs = max(local_devices, per_process_bs // self.accumulation_steps)
            # ensure TPU alignment after division
            per_process_bs = align_for_tpu(per_process_bs)

        return per_process_bs

    def __call__(self, examples: List[Dict], cooldown_phase=False, bucket_idx=None, tokenizer=None):
        """Collates a list of examples into a batch with clean token flow.
        
        Clean token flow:
        1. Labels: sentences joined with `</s><s>`, ending with `</s>` (no BOS at start)
        2. Encoder Input: metadata + corrupted OR `<s>` + corrupted (cooldown)
        3. Decoder Input: `<eos>` + labels[:-1] (using shift_tokens_right)
        """
        if not examples:
            return {}
        
        if tokenizer is None:
            tokenizer = self.tokenizer

        # determine effective max length
        batch_max_length = self.max_length
        if self.use_bucketing and bucket_idx is not None and bucket_idx < len(self.bucket_sizes):
            batch_max_length = min(self.bucket_sizes[bucket_idx], self.model_max_length)

        # process examples
        all_encoder_inputs = []
        all_labels = []

        for example in examples:
            # get sentences or fallback to original_text
            sentences = example.get('sentences', None)
            if sentences is None:
                # fallback to original_text split by periods if no sentences column
                orig_text = example.get('original_text', '')
                if self.hangul_only:
                    orig_text = transcribe(orig_text) if has_hanja(orig_text) else orig_text
                    # log if Japanese remains
                    if contains_japanese(orig_text):
                        logger.debug(f"Japanese kana detected after transcription in hangul_only mode: {orig_text[:100]}...")
                        self.log_japanese_example(example.get('original_text', ''), orig_text)
                # simple sentence splitting for backward compatibility
                sentences = [s.strip() for s in orig_text.split('.') if s.strip()]
                if not sentences:
                    sentences = [orig_text]  # use whole text if no periods
            
            # get metadata (optional)
            metadata = example.get('metadata', '')
            
            # get example index for sliding window seed
            example_idx = example.get('index', 0)
            
            # calculate actual metadata length if present
            metadata_length = 0
            if metadata and not cooldown_phase:
                metadata_tokens = tokenizer(metadata + " ", add_special_tokens=False).input_ids
                metadata_length = len(metadata_tokens)
            
            # effective window for content after metadata
            content_max_length = batch_max_length - metadata_length

            # join sentences with proper boundaries
            if isinstance(sentences, list):
                # join with sentence boundaries for both labels AND encoder
                full_text = f"{tokenizer.eos_token}".join(sentences)
                full_text = full_text + tokenizer.eos_token  # end with EOS
            else:
                # already a string
                full_text = sentences + tokenizer.eos_token
            
            # initially both label and corruption text are the same
            label_text = full_text
            text_for_corruption = full_text

            # handle hangul translation if needed
            # if han2han_transcription_ratio is set, randomly decide whether to apply transcription
            apply_transcription = True
            if self.han2han_transcription_ratio is not None:
                apply_transcription = self.rng.random() < self.han2han_transcription_ratio

            if apply_transcription:
                if self.hangul_only:
                    # already translated above if using original_text fallback
                    if not isinstance(sentences, str):  # only translate if we have actual sentences
                        text_for_corruption = transcribe(text_for_corruption) if has_hanja(text_for_corruption) else text_for_corruption
                        label_text = transcribe(label_text) if has_hanja(label_text) else label_text
                elif self.hangul_decoder and self.hangul_decoder != "reverse":
                    # normal mode: decoder gets hangul
                    label_text = transcribe(label_text) if has_hanja(label_text) else label_text
                elif self.hangul_decoder == "reverse":
                    # reverse mode: encoder gets hangul, decoder gets original
                    text_for_corruption = transcribe(text_for_corruption) if has_hanja(text_for_corruption) else text_for_corruption
            
            # apply sliding window to both label and corruption text
            # tokenize to check length
            label_tokens = tokenizer(label_text, add_special_tokens=False).input_ids
            corruption_tokens = tokenizer(text_for_corruption, add_special_tokens=False).input_ids
            
            # determine if sliding window is needed
            includes_doc_start = True  # default for short sequences
            
            if len(label_tokens) > content_max_length:
                # need sliding window for labels
                max_start = len(label_tokens) - content_max_length
                # use example index to seed the random selection for consistency
                rng = np.random.RandomState(example_idx + self.current_epoch if hasattr(self, 'current_epoch') else example_idx)
                start_idx = rng.randint(0, max_start + 1)
                label_tokens = label_tokens[start_idx:start_idx + content_max_length]
                includes_doc_start = (start_idx == 0)
                # decode back to text
                label_text = tokenizer.decode(label_tokens, skip_special_tokens=False)
                
                # clean leading special tokens to prevent double </s>
                while label_text.startswith(tokenizer.eos_token) or label_text.startswith(tokenizer.bos_token):
                    if label_text.startswith(tokenizer.eos_token):
                        label_text = label_text[len(tokenizer.eos_token):]
                    elif label_text.startswith(tokenizer.bos_token):
                        label_text = label_text[len(tokenizer.bos_token):]
            
            if len(corruption_tokens) > content_max_length:
                # need sliding window for corruption text (use same logic)
                max_start = len(corruption_tokens) - content_max_length
                rng = np.random.RandomState(example_idx + self.current_epoch if hasattr(self, 'current_epoch') else example_idx)
                start_idx = rng.randint(0, max_start + 1)
                corruption_tokens = corruption_tokens[start_idx:start_idx + content_max_length]
                includes_doc_start = includes_doc_start and (start_idx == 0)
                # decode back to text
                text_for_corruption = tokenizer.decode(corruption_tokens, skip_special_tokens=False)
                
                # clean leading special tokens
                while text_for_corruption.startswith(tokenizer.eos_token) or text_for_corruption.startswith(tokenizer.bos_token):
                    if text_for_corruption.startswith(tokenizer.eos_token):
                        text_for_corruption = text_for_corruption[len(tokenizer.eos_token):]
                    elif text_for_corruption.startswith(tokenizer.bos_token):
                        text_for_corruption = text_for_corruption[len(tokenizer.bos_token):]

            # apply deletion noise to the joined text
            if self.deletion_ratio > 0:
                # tokenize for deletion
                corrupted_ids = tokenizer(text_for_corruption, add_special_tokens=False).input_ids
                # apply deletion
                corrupted_ids = self.delete_tokens(
                    np.array(corrupted_ids), 
                    text_for_corruption, 
                    example, 
                    tokenizer
                )
                # decode back to text
                corrupted_text = tokenizer.decode(corrupted_ids, skip_special_tokens=False)
            else:
                corrupted_text = text_for_corruption

            # build encoder input
            if cooldown_phase:
                # during cooldown: only add BOS if we're at document start
                if includes_doc_start:
                    encoder_input = tokenizer.bos_token + corrupted_text
                else:
                    encoder_input = corrupted_text  # mid-document, no false BOS
            elif metadata:
                # normal training: metadata + content
                encoder_input = metadata + " " + corrupted_text
            else:
                # no metadata available, only add BOS if at doc start
                if includes_doc_start:
                    encoder_input = tokenizer.bos_token + corrupted_text
                else:
                    encoder_input = corrupted_text

            all_encoder_inputs.append(encoder_input)
            all_labels.append(label_text)

        # tokenize all at once
        encoder_batch = tokenizer(
            all_encoder_inputs,
            padding="max_length",
            max_length=batch_max_length,
            truncation=True,
            return_tensors="np",
            add_special_tokens=False  # we already added special tokens manually
        )

        labels_batch = tokenizer(
            all_labels,
            padding="max_length",
            max_length=batch_max_length,
            truncation=True,
            return_tensors="np",
            add_special_tokens=False  # we already added special tokens manually
        )

        # create decoder input by shifting labels with EOS as start token
        decoder_input_ids = shift_tokens_right(
            labels_batch.input_ids,
            tokenizer.pad_token_id,
            tokenizer.eos_token_id  # use EOS as start token instead of BOS
        )
        
        # mask padding in labels
        labels_masked = np.where(
            labels_batch.input_ids == tokenizer.pad_token_id,
            -100,
            labels_batch.input_ids
        )
        
        # create batch dictionary
        batch_dict = {
            "input_ids": encoder_batch.input_ids,
            "decoder_input_ids": decoder_input_ids,
            "labels": labels_masked,
            "attention_mask": encoder_batch.attention_mask.astype(np.int32),
            "decoder_attention_mask": np.where(
                decoder_input_ids != tokenizer.pad_token_id, 1, 0
            ).astype(np.int32),
        }
        
        return BatchEncoding(batch_dict).data

    def get_kiwi(self):
        """Get or create a Kiwi instance for the current thread."""
        thread_id = threading.get_ident()
        if thread_id not in self._kiwi_instances:
            with self._kiwi_lock:
                if thread_id not in self._kiwi_instances:
                    try:
                        self._kiwi_instances[thread_id] = Kiwi(integrate_allomorph=False, num_workers=1)
                        logger.debug(f"Created Kiwi instance for thread {thread_id}")
                    except Exception as e:
                        logger.warning(f"Failed to create Kiwi for thread {thread_id}: {e}")
                        return None
        return self._kiwi_instances[thread_id]

    def get_or_compute_morphs(self, example_data: dict) -> Optional[List[str]]:
        """Get morphs from pre-computed data, cache, or compute on-the-fly."""
        # check pre-computed morphs first
        if example_data and 'morphs' in example_data and example_data['morphs'] is not None:
            return example_data['morphs']

        return None

    def delete_tokens(self, inputs: np.ndarray, text: str = None, example_data: dict = None, tokenizer = None) -> List[int]:
        """Applies token deletion noise using either pre-computed morphs, kiwi, mecab, or unigram tokens."""
        if tokenizer is None:
            tokenizer = self.tokenizer
        # get morphemes from pre-computed, cache, or compute
        morphemes = None
        if text is not None and example_data is not None:
            morphemes = self.get_or_compute_morphs(example_data)

        # decide whether to use morpheme-based deletion for this example  
        use_morpheme = (morphemes is not None and 
                       self.rng.random() < self.use_mecab_deletion)

        if use_morpheme:
            try:
                # apply deletion at morpheme level
                mask = self.rng.random(len(morphemes)) > self.deletion_ratio
                remaining_morphemes = [m for m, keep in zip(morphemes, mask) if keep]

                # reconstruct text — spaces already preserved
                remaining_text = ''.join(remaining_morphemes)

                # tokenize the reconstructed text without special tokens to match unigram path
                remaining_ids = tokenizer(remaining_text, add_special_tokens=False).input_ids
                return remaining_ids
            except Exception as e:
                # fallback to standard deletion if morpheme processing fails
                logger.debug(f"Morpheme deletion failed: {e}, falling back to standard deletion")

        # fallback: try mecab if available and no morphemes yet
        if not use_morpheme and self.mecab_tokenizer is not None and text is not None and self.rng.random() < self.use_mecab_deletion:
            try:
                morphemes = hanja_aware_morpheme_tokenization(text, self.mecab_instance)

                # apply deletion at morpheme level
                mask = self.rng.random(len(morphemes)) > self.deletion_ratio
                remaining_morphemes = [m for m, keep in zip(morphemes, mask) if keep]

                # reconstruct text — spaces already preserved
                remaining_text = ''.join(remaining_morphemes)

                # tokenize the reconstructed text without special tokens to match unigram path
                remaining_ids = tokenizer(remaining_text, add_special_tokens=False).input_ids
                return remaining_ids
            except Exception as e:
                # fallback to standard deletion if morpheme processing fails
                pass

        # standard unigram-based deletion
        probability_matrix = np.full(inputs.shape, self.deletion_ratio, dtype=np.float32)
        # protect special tokens from deletion
        special_token_ids = [
            tokenizer.pad_token_id,
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
            tokenizer.unk_token_id
        ]
        for token_id in special_token_ids:
            if token_id is not None:
                probability_matrix[inputs == token_id] = 0

        idx_to_delete = self.rng.binomial(1, probability_matrix).astype(bool)
        noisy_inputs = inputs[~idx_to_delete]
        return noisy_inputs.tolist() # return list for padding

    def generate_batch_splits(self, samples_idx: np.ndarray, batch_size: int, drop_last=True):
        """
        Generate batches of indices. Handles edge cases.
        Adapts logic from `train_han2han.py` for distributed padding.
        """
        num_samples = len(samples_idx)

        if batch_size <= 0:
             logger.warning(f"generate_batch_splits called with batch_size <= 0 ({batch_size}). returning empty list.")
             # return shape consistent with drop_last=True if needed
             return np.array([], dtype=samples_idx.dtype).reshape(0, 0) if drop_last else []

        local_device_count = jax.local_device_count()
        if batch_size < local_device_count or batch_size % local_device_count != 0:
            # round up to nearest multiple of local_device_count
            orig_batch_size = batch_size
            batch_size = ((batch_size + local_device_count - 1) // local_device_count) * local_device_count
            logger.info(f"Adjusted batch_size from {orig_batch_size} to {batch_size} to be multiple of local_device_count ({local_device_count})")

        if drop_last: # training
            samples_to_remove = num_samples % batch_size
            if samples_to_remove != 0: samples_idx = samples_idx[:-samples_to_remove]
            num_samples = len(samples_idx) # update num_samples
            if num_samples == 0: return np.array([], dtype=samples_idx.dtype).reshape(0, batch_size)
            sections_split = num_samples // batch_size
            try:
                return samples_idx.reshape((sections_split, batch_size)) # (num_batches, batch_size)
            except ValueError as e:
                logger.error(f"reshape error in generate_batch_splits (drop_last=true): {e}. num_samples={num_samples}, batch_size={batch_size}")
                return np.array([], dtype=samples_idx.dtype).reshape(0, batch_size) # return empty on error
        else: # eval
            if num_samples == 0: return []
            sections_split = math.ceil(num_samples / batch_size)
            split_batches_indices = np.array_split(samples_idx, sections_split)

            result_batches = []
            for batch_indices in split_batches_indices:
                if len(batch_indices) == 0:
                    continue
                # if not a multiple of local_device_count, pad the batch
                if len(batch_indices) < local_device_count or len(batch_indices) % local_device_count != 0:
                    # calculate padding needed
                    target_size = ((len(batch_indices) + local_device_count - 1) // local_device_count) * local_device_count
                    padding_needed = target_size - len(batch_indices)
                    # use the first index as padding (will create duplicates, but they can be masked/ignored)
                    padding_indices = np.repeat(batch_indices[0], padding_needed)
                    # concatenate original indices with padding
                    batch_indices = np.concatenate([batch_indices, padding_indices])
                    logger.info(f"Padded eval batch from {len(batch_indices) - padding_needed} to {len(batch_indices)} elements")
                result_batches.append(batch_indices)

            return result_batches

    def get_sequence_length(self, example: dict):
        """Get sequence length from precomputed column."""
        seq_len = example.get("sequence_length") 
        if seq_len is not None and isinstance(seq_len, (int, np.integer)) and seq_len >= 0:
            return int(seq_len)
        else:
            text_preview = example.get("original_text", "N/A")[:50]
            logger.warning(f"Missing or invalid sequence_length in example: {text_preview}... returning 0.")
            return 0

    def assign_to_bucket(self, length: int):
        """Assign a sequence length to a bucket index."""
        for i, bucket_size in enumerate(self.bucket_sizes):
            if length <= bucket_size: return i
        return len(self.bucket_sizes) - 1 # assign to largest if exceeds all

    def get_generator_state(self) -> Dict[str, Any]:
        """Returns the current state of the generator for checkpointing."""
        iter_state = {
            'current_bucket_idx_iter': getattr(self, '_current_bucket_idx_iter', 0),
            'current_batch_offset': getattr(self, '_current_batch_offset', 0),
            'processed_buckets': getattr(self, '_processed_buckets', set()),
            'non_bucketed_offset': getattr(self, '_non_bucketed_offset', 0)
        }
        return {'rng_object': self.rng, 'iter_state': iter_state}

    def batch_generator(self, batch_size_arg_unused, epoch_idx: int = 0, cooldown_phase=False, 
                        is_eval=False, resuming=False, max_steps=None):
        """
        Generator yielding batches using pre-distributed indices per bucket.
        `batch_size_arg_unused` is ignored; batch size is determined by bucket/adaptive logic.
        Handles state restoration for resuming training.
        Alternates bucket processing order based on epoch_idx (even=desc sample count, odd=reversed).

        When accumulation_steps > 1, yields groups of K micro-batches stacked with shape [num_micro_batches, batch_size, ...],
        otherwise yields individual batches.

        Args:
            max_steps: If provided, ensures we sample a representative distribution within this many steps.
                      Ignored if disable_stratified_sampling is True.
        """
        # if stratified sampling is disabled, ignore max_steps
        if self.disable_stratified_sampling and max_steps is not None:
            logger.info(f"Process {jax.process_index()}: Stratified sampling disabled, ignoring max_steps={max_steps}")
            max_steps = None
        # set tokenizer and current epoch for sliding window
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer.name_or_path)  # each thread has its own
        self.current_epoch = epoch_idx
        self._epoch_count = epoch_idx

        # initialize mecab instance for this thread (thread safety)
        thread_mecab = None
        if MECAB_AVAILABLE:
            try:
                thread_mecab = Mecab()
                logger.debug(f"Thread {epoch_idx}: Initialized thread-local MeCab instance")
            except Exception as e:
                logger.warning(f"Thread {epoch_idx}: Failed to initialize MeCab: {e}")
                thread_mecab = None

        # track examples seen in this epoch
        self._current_epoch_examples.clear()

        # reset seen cache when doing natural epochs (max_steps=None)
        # this allows the generator to produce all data each epoch
        if max_steps is None and not is_eval and epoch_idx > 0:
            # clear the seen cache for natural epoch progression
            self._seen_examples_cache.clear()
            logger.info(f"Process {jax.process_index()}: Cleared seen examples cache for natural epoch {epoch_idx+1}")

        # if max_steps is set and we have source tracking, prepare representative sampling
        if max_steps is not None and not is_eval and self._source_indices_map:
            # calculate approximate samples we'll see with max_steps
            # note: this is just for logging, actual calculation happens below with effective_bs
            approx_samples_to_see_log = max_steps * self.batch_size

            # log sampling plan
            if jax.process_index() == 0:
                logger.info(f"Max steps set to {max_steps}, will see approximately {approx_samples_to_see_log} samples (initial estimate)")
                logger.info(f"Batch size: {self.batch_size}, Accumulation steps: {self.accumulation_steps}")
                logger.info(f"Ensuring representative distribution across sources within these steps")
        elif max_steps is None and not is_eval:
            # natural epochs - log that we're processing all data
            if jax.process_index() == 0:
                logger.info(f"Natural epoch mode: will process all available data for epoch {epoch_idx+1}")

        # === State Restoration ===
        start_bucket_idx_iter = 0
        start_batch_offset = 0
        processed_buckets_resume = set()
        non_bucketed_start_offset = 0

        if self.resume_rng_state is not None and not is_eval:
            try:
                logger.info(f"Process {jax.process_index()}: Restoring RNG state for training generator...")
                self.rng = pickle.loads(self.resume_rng_state)
            except Exception as e:
                logger.error(f"Process {jax.process_index()}: Failed to restore RNG state: {e}. Using fresh RNG.")
                self.rng = np.random.default_rng(getattr(self, 'seed', 42))

        if self.resume_iter_state is not None and not is_eval and resuming == True:
            logger.info(f"Process {jax.process_index()}: Restoring iteration state: {self.resume_iter_state}")
            start_bucket_idx_iter = self.resume_iter_state.get('current_bucket_idx_iter', 0)
            start_batch_offset = self.resume_iter_state.get('current_batch_offset', 0)
            processed_buckets_resume = set(self.resume_iter_state.get('processed_buckets', []))
            non_bucketed_start_offset = self.resume_iter_state.get('non_bucketed_offset', 0)
            logger.info(f"  -> Start Bucket Iter: {start_bucket_idx_iter}, Start Batch Offset: {start_batch_offset}, "
                        f"Processed Buckets: {processed_buckets_resume}, Non-Bucketed Offset: {non_bucketed_start_offset}")

        self._current_bucket_idx_iter = start_bucket_idx_iter
        self._current_batch_offset = start_batch_offset
        self._processed_buckets = processed_buckets_resume
        self._non_bucketed_offset = non_bucketed_start_offset

        if len(self.dataset) == 0:
             logger.warning("batch_generator called with empty DataFrame. Returning.")
             return

        # set up for micro-batch fusion when accumulation_steps > 1
        micro_batch_buffer = []
        micro_batch_info_buffer = []
        micro_batch_count = 0  # track number of micro-batches processed

        # === NON-BUCKETED DATA GENERATION ===
        if not self.use_bucketing:
            num_local_samples = len(self.dataset)
            if num_local_samples == 0: return

            # effective_bs was: self.eval_batch_size if is_eval else self.batch_size
            current_bs_per_process = self.eval_batch_size if is_eval else self.batch_size

            if not is_eval and self.accumulation_steps > 1:
                raw_micro_bs = current_bs_per_process // self.accumulation_steps
                num_ld = jax.local_device_count()
                # ensure at least num_ld and multiple of num_ld
                # round up to the nearest multiple of num_ld if raw_micro_bs > 0
                aligned_micro_bs = ((raw_micro_bs + num_ld - 1) // num_ld) * num_ld if raw_micro_bs > 0 else 0
                # ensure it's at least num_ld (handles case where raw_micro_bs was 0)
                effective_bs = max(num_ld, aligned_micro_bs)
            else: # is_eval or accumulation_steps <= 1
                # ensure base batch size is also compatible if used directly for pmap splitting logic
                num_ld = jax.local_device_count()
                if current_bs_per_process % num_ld != 0:
                     # this case implies base batch/eval batch size might be an issue
                     # for now, we'll round up.
                     current_bs_per_process = ((current_bs_per_process + num_ld - 1) // num_ld) * num_ld
                effective_bs = max(num_ld, current_bs_per_process)

            drop = True # not is_eval # drop last incomplete batch for training, keep for eval
            # NOTE: we are ALWAYS dropping last incomplete batch for training, even during eval

            # get available indices respecting seen cache
            if not is_eval and self._source_indices_map:
                # if max_steps is None (natural epochs), don't use seen cache filtering
                if max_steps is None:
                    # natural epochs: use all indices without filtering
                    available_by_source = {}
                    for source, indices in self._source_indices_map.items():
                        available_by_source[source] = indices.copy()
                else:
                    # max_steps mode: use seen cache filtering for stratified sampling
                    available_by_source = self._get_available_indices_by_source()

                # if max_steps is set, sample proportionally from each source
                if max_steps is not None:
                    # calculate total samples we'll see with max_steps
                    # important: when accumulation_steps > 1, each step processes accumulation_steps micro-batches
                    # so the total samples per step is effective_bs * accumulation_steps
                    # however, effective_bs is already the micro-batch size when accumulation_steps > 1
                    # so we just use effective_bs directly (it represents samples per micro-batch)
                    # and multiply by accumulation_steps to get total samples per optimizer step
                    if self.accumulation_steps > 1:
                        # effective_bs is micro-batch size, multiply by accumulation_steps for full batch
                        samples_per_step = effective_bs * self.accumulation_steps
                    else:
                        # effective_bs is the full batch size
                        samples_per_step = effective_bs

                    approx_samples_to_see = max_steps * samples_per_step
                    logger.debug(f"Planning to sample {approx_samples_to_see} indices for {max_steps} steps")
                    logger.debug(f"  effective_bs={effective_bs}, accumulation_steps={self.accumulation_steps}, samples_per_step={samples_per_step}")

                    # use ORIGINAL source proportions, not current available proportions
                    # this ensures consistent distribution across epochs
                    total_original = sum(len(indices) for indices in self._source_indices_map.values())

                    # check if we need to reset cache due to insufficient samples
                    total_available = sum(len(indices) for indices in available_by_source.values())

                    # if we don't have enough samples available OR any source is completely exhausted, reset
                    need_reset = False
                    if total_available < approx_samples_to_see:
                        logger.info(f"Insufficient samples available ({total_available} < {approx_samples_to_see}), will reset cache")
                        need_reset = True
                    else:
                        # check if any source would be unable to provide its proportional share
                        for source in self._source_indices_map.keys():
                            original_size = len(self._source_indices_map[source])
                            source_proportion = original_size / total_original if total_original > 0 else 0
                            needed_from_source = int(approx_samples_to_see * source_proportion)
                            available_from_source = len(available_by_source[source])

                            if needed_from_source > 0 and available_from_source < needed_from_source:
                                logger.info(f"Source {source} cannot provide enough samples ({available_from_source} < {needed_from_source}), will reset cache")
                                need_reset = True
                                break

                    # reset if needed
                    if need_reset:
                        logger.info(f"Process {jax.process_index()}: Resetting seen examples cache to maintain distribution")
                        self._seen_examples_cache.clear()
                        # recalculate available indices after reset
                        available_by_source = self._get_available_indices_by_source()

                    # now sample proportionally from each source
                    # important: we should sample enough for the full max_steps, not just a subset
                    # add a small buffer (10%) to ensure we don't run out of samples
                    samples_with_buffer = int(approx_samples_to_see * 1.1)

                    local_idx_array = []
                    for source, available_indices in available_by_source.items():
                        # use original source size for proportion calculation
                        original_size = len(self._source_indices_map[source])
                        source_proportion = original_size / total_original if total_original > 0 else 0
                        samples_from_source = int(samples_with_buffer * source_proportion)

                        # ensure we don't exceed available indices
                        samples_from_source = min(samples_from_source, len(available_indices))
                        logger.debug(f"Source {source}: sampling {samples_from_source} samples from {len(available_indices)} available")

                        # randomly sample from this source
                        if samples_from_source > 0:
                            source_sample = self.rng.choice(available_indices, size=samples_from_source, replace=False)
                            local_idx_array.extend(source_sample)

                    local_idx_array = np.array(local_idx_array)

                    # safety check: if still empty, we have a serious problem  
                    if len(local_idx_array) == 0:
                        logger.error("No indices available for sampling - dataset might be too small for requested max_steps")
                        # return early to prevent infinite loop
                        return

                    if jax.process_index() == 0:
                        # log detailed sampling info
                        if len(local_idx_array) > 0:
                            logger.info(f"Sampled {len(local_idx_array)} indices maintaining source distribution for {max_steps} steps")
                        else:
                            logger.warning(f"WARNING: Sampled 0 indices for {max_steps} steps - this will cause issues!")
                        # NOTE: Removed expensive per-source counting that was causing 7+ hour hangs
                        # The nested loop checking 153k+ indices against each source was extremely slow
                else:
                    # no max_steps limit, use all available indices
                    local_idx_array = []
                    for source, indices in available_by_source.items():
                        local_idx_array.extend(indices)
                    local_idx_array = np.array(local_idx_array)

                    if jax.process_index() == 0:
                        logger.info(f"Natural epochs: using all {len(local_idx_array)} indices for epoch {epoch_idx+1}")
            else:
                # for eval or if no source mapping, use all indices
                local_idx_array = np.arange(num_local_samples)

            self.rng.shuffle(local_idx_array)

            # safety check: if no indices, return early to prevent infinite loop
            if len(local_idx_array) == 0:
                logger.warning(f"Process {jax.process_index()}: Empty index array for epoch {epoch_idx+1}, returning early")
                # reset for next epoch
                if not is_eval: 
                    self._non_bucketed_offset = 0
                return

            batch_indices_list = self.generate_batch_splits(local_idx_array, effective_bs, drop_last=drop)

            # log expected batches
            if jax.process_index() == 0:
                if max_steps is None:
                    logger.info(f"Natural epoch {epoch_idx+1}: generated {len(batch_indices_list)} batches from {len(local_idx_array)} indices")
                else:
                    logger.info(f"Max-steps epoch {epoch_idx+1}: generated {len(batch_indices_list)} batches from {len(local_idx_array)} sampled indices")
                    logger.info(f"  Target: {max_steps} steps, Expected batches: {len(batch_indices_list)}")
                    if len(batch_indices_list) < max_steps:
                        logger.warning(f"  WARNING: Generated fewer batches ({len(batch_indices_list)}) than max_steps ({max_steps})!")

            # safety check: if no batches were generated, log warning
            if len(batch_indices_list) == 0:
                logger.warning(f"Process {jax.process_index()}: No batches generated for epoch {epoch_idx+1}")
                # reset for next epoch if training and return
                if not is_eval:
                    self._non_bucketed_offset = 0
                return

            # track batches yielded for debugging
            batches_yielded = 0

            for batch_idx, batch_indices in enumerate(batch_indices_list):
                self._non_bucketed_offset = batch_idx # update current position
                if not is_eval and self._non_bucketed_offset < non_bucketed_start_offset:
                    continue

                if len(batch_indices) == 0: continue
                batch_data = self.dataset[batch_indices].to_dicts()
                if not batch_data: continue

                # track seen indices for non-repetition across epochs
                # only track in seen_examples_cache if using max_steps (stratified sampling)
                if not is_eval:
                    if max_steps is not None:
                        # stratified sampling mode: track globally
                        for idx in batch_indices:
                            self._seen_examples_cache.add(idx)
                            self._current_epoch_examples.add(idx)
                    else:
                        # natural epochs: only track within current epoch
                        for idx in batch_indices:
                            self._current_epoch_examples.add(idx)

                collated = self(batch_data, cooldown_phase=cooldown_phase, bucket_idx=None, tokenizer=tokenizer, thread_mecab=thread_mecab)
                bs = len(batch_data)
                sl = collated.get("input_ids", np.array([[]])).shape[1]
                collated["batch_info"] = np.array([bs, sl, -1], dtype=np.int32) # -1 indicates no bucket

                if is_eval or self.accumulation_steps <= 1:
                    # yield individual batches for evaluation or when not using micro-batch fusion
                    batches_yielded += 1
                    yield collated
                else:
                    # add to micro-batch buffer
                    micro_batch_buffer.append(collated)
                    micro_batch_info_buffer.append(collated.pop("batch_info"))
                    micro_batch_count += 1

                    # when buffer reaches accumulation_steps, yield the group
                    if len(micro_batch_buffer) == self.accumulation_steps:
                        # stack micro-batches along a new first dimension for each key
                        batched_inputs = {}
                        for key in micro_batch_buffer[0].keys():
                            try:
                                batched_inputs[key] = np.stack([b[key] for b in micro_batch_buffer])
                            except ValueError as e:
                                logger.warning(f"Failed to stack key {key}: {e}")
                                # for keys that can't be stacked, keep only the first micro-batch's value
                                batched_inputs[key] = micro_batch_buffer[0][key]

                        # stack batch info
                        batched_info = np.stack(micro_batch_info_buffer)

                        # create combined batch with appropriate structure
                        result = {
                            **batched_inputs,
                            "batch_info": batched_info
                        }

                        # clear buffers
                        micro_batch_buffer.clear()
                        micro_batch_info_buffer.clear()

                        batches_yielded += 1
                        yield result

            # handle any remaining batches in buffer at the end (should be none if drop_last=True)
            if not is_eval and self.accumulation_steps > 1 and micro_batch_buffer:
                logger.warning(f"Process {jax.process_index()}: Dropping {len(micro_batch_buffer)} remaining micro-batches at end of non-bucketed data")
                # clear buffers without yielding incomplete group
                micro_batch_buffer.clear()
                micro_batch_info_buffer.clear()

            # log actual batches yielded for debugging
            if max_steps is not None and jax.process_index() == 0:
                logger.info(f"Epoch {epoch_idx+1} completed: yielded {batches_yielded} batches (target was {max_steps} steps)")
                if batches_yielded < max_steps:
                    logger.warning(f"  WARNING: Yielded fewer batches than expected! This will cause early stopping.")

            # reset bucket offset for next epoch if not resuming
            if not is_eval: self._non_bucketed_offset = 0
            return

        # === BUCKETED DATA GENERATION ===
        if not self.indices_per_bucket or len(self.indices_per_bucket) != len(self.bucket_sizes):
            logger.error("batch_generator called with invalid or mismatched indices_per_bucket. Returning.")
            return

        # determine bucket processing order based on epoch_idx
        base_bucket_order = sorted(range(len(self.bucket_sizes)), key=lambda i: len(self.indices_per_bucket[i]), reverse=True)

        # alternate order for odd epochs (epoch_idx starts at 0)
        is_odd_epoch = (epoch_idx % 2 != 0)
        if is_odd_epoch and not is_eval: # only alternate for training
            bucket_order = list(reversed(base_bucket_order))
            if jax.process_index() == 0: logger.info(f"Epoch {epoch_idx+1} (odd): Using reversed bucket order.")
        else:
            bucket_order = base_bucket_order
            if jax.process_index() == 0 and not is_eval: logger.info(f"Epoch {epoch_idx+1} (even/eval): Using standard bucket order (desc sample count).")

        # iterate through buckets based on the determined order
        for bucket_idx_iter, bucket_idx in enumerate(bucket_order):
            self._current_bucket_idx_iter = bucket_idx_iter 

            # === Resume logic ===
            # skip buckets already processed or before the resume point *in this epoch's order*
            if not is_eval:
                if bucket_idx in self._processed_buckets:
                    # need to shuffle anyway to advance RNG state correctly
                    indices_for_this_bucket_dummy = np.array(self.indices_per_bucket[bucket_idx], dtype=np.int64)
                    if len(indices_for_this_bucket_dummy) > 0: self.rng.shuffle(indices_for_this_bucket_dummy)
                    continue
                if self._current_bucket_idx_iter < start_bucket_idx_iter:
                    indices_for_this_bucket_dummy = np.array(self.indices_per_bucket[bucket_idx], dtype=np.int64)
                    if len(indices_for_this_bucket_dummy) > 0: self.rng.shuffle(indices_for_this_bucket_dummy)
                    continue

            indices_for_this_bucket = np.array(self.indices_per_bucket[bucket_idx], dtype=np.int64)
            if len(indices_for_this_bucket) == 0: continue

            self.rng.shuffle(indices_for_this_bucket)

            bucket_len_limit = self.bucket_sizes[bucket_idx]
            # get adaptive batch size per process, or fixed eval/train size
            adaptive_bs_per_proc = self.get_bucket_batch_size(bucket_len_limit, for_eval=is_eval)

            if self.adaptive_batch_size:
                effective_bs = adaptive_bs_per_proc
            else: # not adaptive_batch_size
                if is_eval:
                    current_bs_per_process = self.eval_batch_size
                    num_ld = jax.local_device_count()
                    if current_bs_per_process % num_ld != 0:
                        current_bs_per_process = ((current_bs_per_process + num_ld - 1) // num_ld) * num_ld
                    effective_bs = max(num_ld, current_bs_per_process)
                else: # training, non-adaptive
                    if self.accumulation_steps > 1:
                        raw_micro_bs = self.batch_size // self.accumulation_steps
                        num_ld = jax.local_device_count()
                        aligned_micro_bs = ((raw_micro_bs + num_ld - 1) // num_ld) * num_ld if raw_micro_bs > 0 else 0
                        effective_bs = max(num_ld, aligned_micro_bs)
                    else: # no accumulation
                        current_bs_per_process = self.batch_size
                        num_ld = jax.local_device_count()
                        if current_bs_per_process % num_ld != 0:
                            current_bs_per_process = ((current_bs_per_process + num_ld - 1) // num_ld) * num_ld
                        effective_bs = max(num_ld, current_bs_per_process)

            drop = True # not is_eval # drop last incomplete batch for training, keep for eval
            # NOTE: we are ALWAYS dropping last incomplete batch for training, even during eval

            # generate batches for this specific bucket using its indices
            # since indices were pre-balanced globally, drop_last=True should yield full batches until the end
            bucket_batch_indices_list = self.generate_batch_splits(indices_for_this_bucket, effective_bs, drop_last=drop)

            if jax.process_index() == 0:
                logger.info(f"[Gen Batch] Bucket {bucket_idx} (<= {bucket_len_limit}): Local Indices={len(indices_for_this_bucket)}, "
                            f"Eff BS={effective_bs}, Gen Batches={len(bucket_batch_indices_list)}, Bucket Iter={bucket_idx_iter}")

            for batch_offset, batch_indices in enumerate(bucket_batch_indices_list):
                self._current_batch_offset = batch_offset

                # resume within the current bucket iteration
                if not is_eval and self._current_bucket_idx_iter == start_bucket_idx_iter and self._current_batch_offset < start_batch_offset:
                    continue

                if len(batch_indices) == 0: continue
                # fetch data using indices into self.dataset
                batch_data = self.dataset[batch_indices].to_dicts()
                if not batch_data: continue

                # track seen indices for non-repetition across epochs
                # only track in seen_examples_cache if using max_steps (stratified sampling)
                if not is_eval:
                    if max_steps is not None:
                        # stratified sampling mode: track globally
                        for idx in batch_indices:
                            self._seen_examples_cache.add(idx)
                            self._current_epoch_examples.add(idx)
                    else:
                        # natural epochs: only track within current epoch
                        for idx in batch_indices:
                            self._current_epoch_examples.add(idx)

                collated = self(batch_data, cooldown_phase=cooldown_phase, bucket_idx=bucket_idx, tokenizer=tokenizer, thread_mecab=thread_mecab)
                bs = len(batch_data)
                sl = collated.get("input_ids", np.array([[]])).shape[1]
                collated["batch_info"] = np.array([bs, sl, bucket_idx], dtype=np.int32)

                if is_eval:
                    # yield individual batches for evaluation or when not using micro-batch fusion
                    yield collated
                elif self.accumulation_steps <= 1:
                    yield collated
                else:
                    # add to micro-batch buffer
                    micro_batch_buffer.append(collated)
                    micro_batch_info_buffer.append(collated.pop("batch_info"))
                    micro_batch_count += 1

                    # when buffer reaches accumulation_steps, yield the group
                    if len(micro_batch_buffer) == self.accumulation_steps:
                        # stack micro-batches along a new first dimension for each key
                        batched_inputs = {}
                        for key in micro_batch_buffer[0].keys():
                            try:
                                batched_inputs[key] = np.stack([b[key] for b in micro_batch_buffer])
                            except ValueError as e:
                                logger.warning(f"Failed to stack key {key}: {e}")
                                # for keys that can't be stacked, keep only the first micro-batch's value
                                batched_inputs[key] = micro_batch_buffer[0][key]

                        # stack batch info
                        batched_info = np.stack(micro_batch_info_buffer)

                        # create combined batch with appropriate structure
                        result = {
                            **batched_inputs,
                            "batch_info": batched_info
                        }

                        # clear buffers
                        micro_batch_buffer.clear()
                        micro_batch_info_buffer.clear()
                        
                        yield result

            # handle any remaining batches in buffer at the end of a bucket
            if not is_eval and self.accumulation_steps > 1 and micro_batch_buffer:
                logger.warning(f"Process {jax.process_index()}: Dropping {len(micro_batch_buffer)} remaining micro-batches at end of bucket {bucket_idx}")
                # clear buffers without yielding incomplete group
                micro_batch_buffer.clear()
                micro_batch_info_buffer.clear()

            if not is_eval: self._processed_buckets.add(bucket_idx)
            self._current_batch_offset = 0 # reset offset for the next bucket

        # reset state for the next epoch
        if not is_eval:
            self._current_bucket_idx_iter = 0
            self._current_batch_offset = 0
            self._processed_buckets = set()
            self._non_bucketed_offset = 0

    def get_sliding_window_slice(self, text_tokens, index, max_length):
        """Get a slice of tokens using sliding window.
        
        Note: This method is kept for backward compatibility but is no longer used
        by the updated __call__ method which handles sliding window inline.
        """
        if len(text_tokens) <= max_length:
            return text_tokens, True, True  # has both BOS and EOS

        # use index as seed for reproducible randomness
        rng = np.random.RandomState(index + self.current_epoch 
                                    if hasattr(self, "current_epoch") else index + 1)

        # 30% chance to include start (with BOS)
        # 30% chance to include end (with EOS)  
        # 40% chance for middle
        rand = rng.random()

        if rand < 0.3:  # start slice
            return text_tokens[:max_length], True, False  # has BOS, no EOS
        elif rand < 0.6:  # end slice
            return text_tokens[-max_length:], False, True  # no BOS, has EOS
        else:  # random middle
            max_start = len(text_tokens) - max_length
            start = rng.randint(0, max_start)
            return text_tokens[start:start + max_length], False, False  # no BOS, no EOS


# === BART DATA COLLATOR ===
@dataclasses.dataclass
class BARTCollator(DataCollator):
    """
    BART-style collator with text infilling and sentence permutation.
    Inherits most functionality from DataCollator but applies BART-specific corruptions.
    
    Token Flow:
    - Labels: sentences joined with `</s>`, ending with `</s><hangul/hanja>`" (no BOS at start)
    - Encoder Input: [metadata] + corrupted_content + `</s><hangul/hanja>` OR corrupted_content  + `</s><hangul/hanja>` (cooldown)
    - Decoder Input: `<hangul/hanja>` + labels[:-1] (shift_tokens_right with EOS as start)
    """
    tokenizer: PreTrainedTokenizerBase
    rng: np.random.Generator
    max_length: int = 1024
    model_max_length: int = 1024
    dataset: pl.DataFrame = dataclasses.field(default_factory=pl.DataFrame)
    indices_per_bucket: list = dataclasses.field(default_factory=list)
    batch_size: int = 16 # base effective batch size per process (before accumulation)
    eval_batch_size: int = 32 # base effective eval batch size per process
    tokenizer_lock: Lock = dataclasses.field(default_factory=Lock)
    bucket_sizes: tuple = DEFAULT_BUCKET_SIZES
    use_bucketing: bool = False
    adaptive_batch_size: bool = False
    batch_scale_factor: float = 1.0
    accumulation_steps: int = 1
    resume_rng_state: Optional[bytes] = None
    resume_iter_state: Optional[Dict[str, Any]] = None
    infilling_ratio: float = 0.30           # 30% of tokens replaced with mask
    poisson_lambda: float = 3.0             # for span length sampling (token-based)
    morpheme_lambda: float = 2.0            # for span length sampling (morpheme-based)
    preserve_morpheme_eos: bool = False     # whether to keep EOS tokens in morpheme mode
    sentence_permutation: bool = True       # always permute sentences
    use_morpheme_masking: float = 0.0       # probability of using morpheme-based masking
    hangul_decoder: Union[bool, str] = True
    hangul_only: bool = False
    han2han_transcription_ratio: Optional[float] = None  # ratio of samples to apply transcription to

    def __post_init__(self):
        """Initialize BART-specific components."""
        # call parent's __post_init__ to initialize epoch tracking and caching
        super().__post_init__()

        self.mask_token_id = self.tokenizer.mask_token_id
        if self.mask_token_id is None:
            raise ValueError("Failed to add mask token to tokenizer")         

    def permute_sentences(self, sentences: list) -> list:
        """Randomly permute sentences if there are 3 or more."""
        if len(sentences) < 3:
            return sentences
        
        shuffled = sentences.copy()
        self.rng.shuffle(shuffled)
        return shuffled

    def span_masking(self, input_ids: np.ndarray, text: str = None, 
                     example_data: dict = None, tokenizer = None) -> np.ndarray:
        """Apply text infilling corruption with poisson-sampled spans.

        Args:
            input_ids: Token IDs to corrupt
            text: Original text (optional, used for morpheme-based masking)
            example_data: Example dict containing pre-computed morphs if available
            tokenizer: Tokenizer instance for text encoding/decoding, defaults to self.tokenizer
        """
        morphemes = None
        if self.use_morpheme_masking > 0.0:
            # get morphemes from pre-computed, cache, or compute
            if text is not None and example_data is not None:
                morphemes = self.get_or_compute_morphs(example_data)

        # decide whether to use morpheme-based masking for this example
        use_morpheme = (morphemes is not None and 
                    self.rng.random() < self.use_morpheme_masking)

        if use_morpheme:
            return tokenizer(self._apply_morpheme_masking(input_ids, tokenizer.mask_token),
                             return_tensors="np").input_ids
        else:
            return self._token_based_infilling(input_ids, tokenizer=tokenizer)

    def _token_based_infilling(self, input_ids: np.ndarray, tokenizer = None) -> np.ndarray:
        """Original token-based text infilling."""
        if tokenizer is None:
            tokenizer = self.tokenizer

        length = len(input_ids)

        # calculate number of tokens to mask
        num_to_mask = int(length * self.infilling_ratio)
        if num_to_mask == 0:
            return input_ids

        # track which positions have been masked
        is_masked = np.zeros(length, dtype=bool)

        # protect special tokens from masking
        special_token_ids = set(tokenizer.all_special_ids)
        is_special = np.array([tok in special_token_ids for tok in input_ids])

        # pass 1: mark spans to mask - single pass with budget
        # we want to mask ~30% of tokens total
        # with average span of 3.5, we need ~30%/3.5 ≈ 8.5% of positions to start spans
        # but let's be more careful and track our budget

        total_masked = 0
        i = 0

        # shuffle positions to avoid always masking from the beginning
        positions = np.arange(length)
        self.rng.shuffle(positions)

        for pos in positions:
            if total_masked >= num_to_mask:
                break

            # skip special tokens or already masked
            if is_special[pos] or is_masked[pos]:
                continue

            # decide whether to start a span here
            # use a lower probability since each span masks multiple tokens
            if self.rng.random() < 0.15:  # conservative probability
                # sample span length
                span_length = self.rng.poisson(self.poisson_lambda)
                span_length = max(1, min(span_length, num_to_mask - total_masked))

                # mask the span
                for j in range(pos, min(pos + span_length, length)):
                    if not is_special[j] and not is_masked[j]:
                        is_masked[j] = True
                        total_masked += 1
                        if total_masked >= num_to_mask:
                            break

        # create output with mask tokens (collapse consecutive masks)
        output = []
        i = 0
        special_token_set = set(tokenizer.all_special_ids)
        last_was_mask = False

        while i < length:
            if is_masked[i] and input_ids[i] not in special_token_set:  # masked and not special token
                if not last_was_mask:  # only add mask if previous wasn't mask
                    output.append(self.mask_token_id)
                    last_was_mask = True
                i += 1
            else:
                output.append(input_ids[i])
                last_was_mask = False
                i += 1

        return np.array(output)

    def _token_based_sentinel_masking(
        self,
        input_ids: np.ndarray,
        tokenizer=None,
        infilling_ratio: float = None,
        start_sentinel_idx: int = 0
    ) -> tuple:
        """
        T5-style token-based span corruption with sentinel tokens using T5's deterministic algorithm.

        Unlike BART-style (single <mask> token), T5-style uses unique <extra_id_N>
        tokens for each masked span, and returns both encoder and decoder outputs.

        This implementation follows T5's deterministic span allocation:
        - num_noise_tokens = round(length * noise_density)
        - num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
        - Partitions tokens into exactly that many spans using random segmentation
        - Alternates: non-noise, noise, non-noise, noise, ...

        Args:
            input_ids: Array of token IDs to mask
            tokenizer: Tokenizer with get_sentinel_token_id method
            infilling_ratio: Override infilling ratio (uses self.infilling_ratio if None)
            start_sentinel_idx: Starting sentinel index for continuity across calls

        Returns:
            (encoder_ids, decoder_ids, next_sentinel_idx): Tuple
            - encoder_ids: input with spans replaced by <extra_id_N>
            - decoder_ids: <extra_id_N> + span_N + <extra_id_N+1> + span_N+1 + ...
            - next_sentinel_idx: next available sentinel index for chaining
        """
        if tokenizer is None:
            tokenizer = self.tokenizer

        length = len(input_ids)
        ratio = infilling_ratio if infilling_ratio is not None else self.infilling_ratio

        if ratio == 0.0 or length < 2:
            return input_ids, np.array([], dtype=input_ids.dtype), start_sentinel_idx

        # protect special tokens from masking
        special_token_ids = set(tokenizer.all_special_ids)
        is_special = np.array([tok in special_token_ids for tok in input_ids])

        # count non-special tokens for masking calculations
        num_nonspecial = np.sum(~is_special)
        if num_nonspecial == 0:
            return input_ids, np.array([], dtype=input_ids.dtype), start_sentinel_idx

        # T5-style deterministic span calculation
        num_noise_tokens = int(np.round(num_nonspecial * ratio))
        num_noise_tokens = max(1, min(num_noise_tokens, num_nonspecial - 1))

        # calculate number of spans based on mean span length
        mean_noise_span_length = float(self.poisson_lambda)
        num_noise_spans = int(np.round(num_noise_tokens / mean_noise_span_length))
        num_noise_spans = max(1, min(num_noise_spans, tokenizer.NUM_SENTINEL_TOKENS - 1))

        num_nonnoise_tokens = num_nonspecial - num_noise_tokens

        # partition into spans using T5's random segmentation
        def random_segmentation(num_items, num_segments):
            """partition a sequence of items randomly into non-empty segments."""
            if num_segments == 1:
                return np.array([num_items])
            if num_items < num_segments:
                num_segments = num_items

            # create segment boundaries: randomly choose (num_segments - 1) positions
            # from (num_items - 1) possible positions
            mask = np.zeros(num_items - 1, dtype=bool)
            mask[:num_segments - 1] = True
            self.rng.shuffle(mask)

            # prepend True to mark first segment
            first_in_segment = np.concatenate([[True], mask])

            # compute segment IDs (0-indexed) and lengths
            segment_id = np.cumsum(first_in_segment) - 1
            segment_lengths = np.bincount(segment_id, minlength=num_segments)

            return segment_lengths[:num_segments]

        # get random span lengths for noise and non-noise spans
        noise_span_lengths = random_segmentation(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = random_segmentation(num_nonnoise_tokens, num_noise_spans)

        # interleave non-noise and noise spans
        # pattern: [nonnoise, noise, nonnoise, noise, ...]
        interleaved_span_lengths = np.empty(num_noise_spans * 2, dtype=int)
        interleaved_span_lengths[0::2] = nonnoise_span_lengths
        interleaved_span_lengths[1::2] = noise_span_lengths

        # compute span starts
        span_starts = np.concatenate([[0], np.cumsum(interleaved_span_lengths)[:-1]])

        # create mask for non-special tokens: True = noise, False = non-noise
        is_noise_nonspecial = np.zeros(num_nonspecial, dtype=bool)
        for i in range(1, len(span_starts), 2):
            start = span_starts[i]
            span_len = interleaved_span_lengths[i]
            is_noise_nonspecial[start:start + span_len] = True

        # map back to full sequence, preserving special tokens
        num_tokens = len(input_ids)
        is_masked = np.zeros(num_tokens, dtype=bool)
        nonspecial_idx = 0
        for i in range(num_tokens):
            if not is_special[i]:
                is_masked[i] = is_noise_nonspecial[nonspecial_idx]
                nonspecial_idx += 1

        # pass 2: build encoder with sentinels, collect decoder spans
        encoder_output = []
        decoder_output = []
        sentinel_idx = start_sentinel_idx
        in_span = False

        for i in range(num_tokens):
            if sentinel_idx >= tokenizer.NUM_SENTINEL_TOKENS:
                logger.error(f"Sentinel overflow: idx={sentinel_idx}, max={tokenizer.NUM_SENTINEL_TOKENS}, num_noise_spans={num_noise_spans}")
            if is_masked[i] and input_ids[i] not in special_token_ids:
                if not in_span:
                    # start new span - add sentinel to both
                    sentinel_id = tokenizer.get_sentinel_token_id(sentinel_idx)
                    encoder_output.append(sentinel_id)
                    decoder_output.append(sentinel_id)
                    in_span = True
                # add masked token to decoder only
                decoder_output.append(input_ids[i])
            else:
                if in_span and input_ids[i] not in special_token_ids:
                    # end span only on non-special tokens (special tokens in noise regions shouldn't fragment)
                    sentinel_idx += 1
                    in_span = False
                encoder_output.append(input_ids[i])

        # if ended in a span, increment for next call
        if in_span:
            sentinel_idx += 1

        return np.array(encoder_output), np.array(decoder_output), sentinel_idx

    def _apply_morpheme_masking(self, morphemes: list, mask_token="<mask>", eos_token="</s>") -> list:
        """Apply span masking to morphemes using shuffled single-pass algorithm.

        Preserves EOS tokens (</s>) for alignment with sentences.

        Args:
            morphemes: List of morphemes to mask (may include </s> tokens)
            mask_token: Token to use for masking
            eos_token: EOS token to preserve

        Returns:
            List of morphemes with spans replaced by mask_token
        """
        if not morphemes:
            return []

        length = len(morphemes)

        # calculate target number of morphemes to mask
        # exclude EOS tokens from the count
        non_eos_count = sum(1 for m in morphemes if m != eos_token)
        num_to_mask = int(non_eos_count * self.infilling_ratio)

        if num_to_mask == 0:
            return morphemes

        # pass 1: mark which morphemes to mask (never mask EOS tokens)
        is_masked = [False] * length
        total_masked = 0

        # shuffle positions to avoid always masking from the beginning
        positions = list(range(length))
        self.rng.shuffle(positions)

        for pos in positions:
            if total_masked >= num_to_mask:
                break

            # skip EOS tokens or already masked
            if morphemes[pos] == eos_token or is_masked[pos]:
                continue

            # decide whether to start a span here
            # use conservative probability for morphemes
            if self.rng.random() < 0.15:
                span_length = self.rng.poisson(self.morpheme_lambda)
                span_length = max(1, min(span_length, num_to_mask - total_masked))

                # mask the span
                for j in range(pos, min(pos + span_length, length)):
                    if morphemes[j] != eos_token and not is_masked[j]:
                        is_masked[j] = True
                        total_masked += 1
                        if total_masked >= num_to_mask:
                            break

        # pass 2: build result with collapsed mask tokens
        result = []
        last_was_mask = False

        for i in range(length):
            if is_masked[i]:
                if not last_was_mask:  # only add mask if previous wasn't masked
                    result.append(mask_token)
                    last_was_mask = True
            else:
                result.append(morphemes[i])
                last_was_mask = False

        return result

    def permute_sentences_by_eos_tokens(self, token_ids, eos_token_id):
        """Clean numpy sentence permutation using EOS token boundaries."""
        if not isinstance(token_ids, np.ndarray):
            token_ids = np.array(token_ids)
        eos_positions = np.where(token_ids == eos_token_id)[0]

        if len(eos_positions) < 2:
            return token_ids

        # split into sentences and shuffle
        sentences = np.split(token_ids, eos_positions + 1)[:-1]  # exclude empty last split
        indices = np.arange(len(sentences))
        self.rng.shuffle(indices)

        return np.concatenate([sentences[i] for i in indices])

    def split_sentences(self, text: str) -> List[str]:
        """Simple sentence splitting for multilingual web text."""

        sentences = text.replace('! ', '!<SPLIT>').replace('? ', '?<SPLIT>').replace(
            '… ', '…<SPLIT>').replace('…', '…<SPLIT>').replace('... ', '...<SPLIT>').replace('...', '...<SPLIT>').replace(
            '、、、、、', '、、、、、<SPLIT>').replace('・・・', '・・・<SPLIT>').replace(
            '. ', '.<SPLIT>').replace('.', '.<SPLIT>').replace('。', '。<SPLIT>').replace('。 ', '。<SPLIT>').replace(
            '！ ', '！<SPLIT>').replace('？', '？<SPLIT>').replace('\n', '<SPLIT>').split('<SPLIT>')

        return [s.strip() for s in sentences if s.strip()]

    def __call__(self, examples: List[Dict], cooldown_phase=False, bucket_idx=None, tokenizer=None, thread_mecab=None):
        """Collates a list of examples into a batch with BART corruption.

        Single batch tokenization to reduce overhead.

        Token flow:
        1. Labels: sentences joined with EOS, ending with `<hanja/hangul>`
        2. Encoder Input: metadata + corrupted OR just inputs alone during cooldown + `<hanja/hangul>`
        3. Decoder Input: `<hanja/hangul>` + labels[:-1] (using shift_tokens_right)
        """
        if not examples:
            return {}

        if tokenizer is None:
            tokenizer = self.tokenizer

        # determine effective max length
        batch_max_length = self.max_length
        if self.use_bucketing and bucket_idx is not None and bucket_idx < len(self.bucket_sizes):
            batch_max_length = min(self.bucket_sizes[bucket_idx], self.model_max_length)

        # phase 1: prepare all texts without tokenization
        # store texts and metadata for batch processing
        all_base_texts = []  # texts before corruption
        all_label_texts = []  # final label texts
        all_metadata_texts = []  # metadata for each example
        all_example_indices = []  # for sliding window seed
        all_use_morpheme_flags = []  # per-example morpheme decision

        # phase 1: per-example text preparation (no tokenization)
        for example in examples:
            # get sentences (required)
            sentences = example.get('sentences', [])
            if not sentences:
                # simple fallback, kinda hacky, not happy about it, should happen less than 0.0001% of the time
                sentences = self.split_sentences(example['original_text'])

            # get metadata (optional)
            metadata = example['metadata'] + " "    # shouldn't make a difference but separate from body

            morphs = None
            # if thread_mecab is available and we need morphemes, generate them real-time
            if thread_mecab is not None and self.use_morpheme_masking > 0:
                # generate morphemes for each sentence with EOS markers for alignment
                if isinstance(sentences, list):
                    morphs_with_eos = []
                    for sent in sentences:
                        sent_morphs = hanja_aware_morpheme_tokenization(sent, thread_mecab)
                        morphs_with_eos.extend(sent_morphs)
                        morphs_with_eos.append(tokenizer.eos_token)  # add eos for alignment
                    morphs = morphs_with_eos
                else:
                    morphs = hanja_aware_morpheme_tokenization(sentences, thread_mecab)
                    morphs.append(tokenizer.eos_token)

            # get example index for sliding window seed
            example_idx = example.get('index', 0)

            # decide corruption strategy at example level (for mixed batches)
            use_morpheme = (morphs is not None and
                          self.rng.random() < self.use_morpheme_masking)

            # decide transcription at example level
            apply_transcription = True
            if self.han2han_transcription_ratio is not None:
                apply_transcription = self.rng.random() < self.han2han_transcription_ratio

            # determine transcription direction for "both" mode
            transcription_direction = self.hangul_decoder  # Default to configured mode
            if apply_transcription and self.hangul_decoder == "both":
                # Randomly choose direction for this example
                transcription_direction = "true" if self.rng.random() < 0.5 else "reverse"

            # prepare label text - always from sentences with proper boundaries
            if isinstance(sentences, list):
                # note: don't permute for labels - they need clean structure
                label_text = f"{tokenizer.eos_token}".join(sentences)
                label_text = label_text + tokenizer.eos_token
            else:
                label_text = sentences + tokenizer.eos_token

            # prepare corruption text based on morpheme decision
            if use_morpheme and morphs:
                # morpheme mode: no spaces, (mostly) no punctuation-based boundaries
                # this teaches the model to infer boundaries from linguistic cues
                # apply morpheme masking here before any tokenization
                masked_morphs = self._apply_morpheme_masking(morphs, eos_token=tokenizer.eos_token)
                corruption_text = ''.join(masked_morphs)
            else:
                # sentence mode: same as label but with possible permutation
                if isinstance(sentences, list):
                    # apply sentence permutation for corruption only
                    if self.sentence_permutation:
                        sentences = self.permute_sentences(sentences)
                    # no more BOS tokens: just EOS between sentences
                    corruption_text = f"{tokenizer.eos_token}".join(sentences)
                    corruption_text = corruption_text + tokenizer.eos_token
                else:
                    corruption_text = sentences + tokenizer.eos_token

            # apply transcription based on mode
            if apply_transcription:
                if self.hangul_only:
                    corruption_text = transcribe(corruption_text) if has_hanja(corruption_text) else corruption_text
                    label_text = transcribe(label_text) if has_hanja(label_text) else label_text
                elif transcription_direction == "true" or transcription_direction is True:
                    # normal mode: decoder gets hangul
                    label_text = transcribe(label_text) if has_hanja(label_text) else label_text
                elif transcription_direction == "reverse":
                    # reverse mode: encoder gets hangul
                    corruption_text = transcribe(corruption_text) if has_hanja(corruption_text) else corruption_text

            # store for batch processing
            all_base_texts.append(corruption_text)
            all_label_texts.append(label_text)
            all_metadata_texts.append(metadata if metadata and not cooldown_phase else "")
            all_example_indices.append(example_idx)
            all_use_morpheme_flags.append(use_morpheme)

        # phase 2: single batch tokenization for all texts
        # combine all texts for one tokenization call
        all_texts_to_tokenize = all_base_texts + all_label_texts + all_metadata_texts

        # single tokenization call
        tokenized_all = tokenizer(
            all_texts_to_tokenize,
            add_special_tokens=False,
            padding=False,      # no padding yet, we'll handle it after corruption
            truncation=False    # no truncation yet, we need full sequences for sliding window
        )

        # extract tokenized results
        num_examples = len(all_base_texts)
        tokenized_base = [tokenized_all.input_ids[i] for i in range(num_examples)]
        tokenized_labels = [tokenized_all.input_ids[i + num_examples] for i in range(num_examples)]
        tokenized_metadata = [tokenized_all.input_ids[i + 2*num_examples] for i in range(num_examples)]

        # phase 3: token-level processing for each example
        final_encoder_inputs = []
        final_labels = []
        decoder_start_tokens = []  # track script tokens for decoder start

        eos_token_id = tokenizer.eos_token_id

        for idx in range(num_examples):
            base_tokens = tokenized_base[idx]
            label_tokens = tokenized_labels[idx]
            metadata_tokens = tokenized_metadata[idx]
            example_idx = all_example_indices[idx]
            use_morpheme = all_use_morpheme_flags[idx]

            # calculate effective content max length (leaving room for script token)
            metadata_length = len(metadata_tokens)
            content_max_length = batch_max_length - metadata_length - 1

            # apply sliding window if needed (>90% of max length)
            safety_threshold = int(0.9 * content_max_length)

            # create one random state for both label and base sliding
            sliding_rng = np.random.RandomState(example_idx)
            selected_sentence_idx = None  # track selected sentence for alignment

            if any(l > safety_threshold for l in (len(t) for t in (label_tokens, base_tokens))):
                label_eos_positions = np.where(np.array(label_tokens) == eos_token_id)[0]
                corrupted_eos_positions = np.where(np.array(base_tokens) == eos_token_id)[0]

                # graceful fallback if EOS counts don't match (rare edge case)
                if len(label_eos_positions) != len(corrupted_eos_positions):
                    logger.warning(f"EOS mismatch: labels has {len(label_eos_positions)}, base has {len(corrupted_eos_positions)}. Using simple truncation.")
                    # fall back to simple truncation
                    label_tokens = label_tokens[:content_max_length]
                    base_tokens = base_tokens[:content_max_length]
                elif len(label_eos_positions) > 0:
                    max_sentences = len(label_eos_positions)
                    selected_sentence_idx = sliding_rng.randint(0, max_sentences)
                    sentence_start_idx = selected_sentence_idx

                    if sentence_start_idx == 0: 
                        label_start_token_idx = base_start_token_idx = 0
                    else: 
                        label_start_token_idx = label_eos_positions[sentence_start_idx - 1] + 1
                        base_start_token_idx = corrupted_eos_positions[sentence_start_idx - 1] + 1
                    # slice to fit within content_max_length
                    label_tokens = label_tokens[label_start_token_idx:label_start_token_idx + content_max_length]
                    base_tokens = base_tokens[base_start_token_idx:base_start_token_idx + content_max_length]
                else:
                    # no eos tokens, just simple truncation
                    label_tokens = label_tokens[:content_max_length]
                    base_tokens = base_tokens[:content_max_length]

            # apply corruptions on token arrays
            corrupted_tokens = np.array(base_tokens).copy()

            if use_morpheme and not self.preserve_morpheme_eos:
                # filter out eos tokens to restore "no boundaries" feature
                # but also prevent consecutive mask tokens
                filtered = []
                last_was_mask = False
                for token in corrupted_tokens:
                    if token == eos_token_id:
                        continue  # skip EOS tokens
                    elif token == tokenizer.mask_token_id:
                        if not last_was_mask:
                            filtered.append(token)
                            last_was_mask = True
                    else:
                        filtered.append(token)
                        last_was_mask = False
                corrupted_tokens = np.array(filtered)

            # 1. sentence permutation on tokens (if not morpheme mode)
            if not use_morpheme and self.sentence_permutation:
                corrupted_tokens = self.permute_sentences_by_eos_tokens(corrupted_tokens, eos_token_id)

            # 2. span masking on tokens (only for non-morpheme mode)
            if not use_morpheme:
                corrupted_tokens = self._token_based_infilling(corrupted_tokens, tokenizer)

            # build final encoder input tokens
            encoder_tokens = []

            # determine script tokens based on what encoder/decoder actually see
            encoder_text = all_base_texts[idx]      # what encoder sees after transcription
            decoder_text = all_label_texts[idx]     # what decoder sees after transcription

            # check what script each component actually has
            encoder_has_hanja = has_hanja(encoder_text)
            decoder_has_hanja = has_hanja(decoder_text)

            if cooldown_phase:
                # during cooldown, no metadata prepended, no BOS for encoder ever
                pass
            elif len(metadata_tokens) > 0:
                encoder_tokens.extend(metadata_tokens)

            # add encoder content
            encoder_tokens.extend(corrupted_tokens.tolist() if isinstance(corrupted_tokens, np.ndarray) else corrupted_tokens)

            # add script token to encoder (mBART-style)
            if encoder_has_hanja:
                script_token_id = tokenizer.convert_tokens_to_ids('<hanja>')
            else:
                script_token_id = tokenizer.convert_tokens_to_ids('<hangul>')

            if script_token_id != tokenizer.unk_token_id:
                encoder_tokens.append(script_token_id)

            # no truncation needed - we already accounted for script token in content_max_length
            final_encoder_inputs.append(encoder_tokens)
            final_labels.append(label_tokens)

            # store decoder script token for this example
            # we'll use this as decoder_start_token_id
            if decoder_has_hanja:
                decoder_start_token = tokenizer.convert_tokens_to_ids('<hanja>')
            else:
                decoder_start_token = tokenizer.convert_tokens_to_ids('<hangul>')

            # fall back to EOS if script token not available
            if decoder_start_token == tokenizer.unk_token_id:
                decoder_start_token = eos_token_id

            decoder_start_tokens.append(decoder_start_token)

        # pad all sequences to batch_max_length
        def pad_sequence(seq, max_len, pad_id):
            if len(seq) < max_len:
                return seq + [pad_id] * (max_len - len(seq))
            return seq[:max_len]

        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            # fallback to 0 if no pad token is defined
            pad_token_id = 0

        # pad encoder inputs and labels
        encoder_input_ids = np.array([
            pad_sequence(seq, batch_max_length, pad_token_id)
            for seq in final_encoder_inputs
        ], dtype=np.int64)

        label_ids = np.array([
            pad_sequence(seq + [decoder_start_tokens[i]], batch_max_length, pad_token_id)
            for i, seq in enumerate(final_labels.copy())
        ], dtype=np.int64)

        tokens_for_decoder = np.array([
            pad_sequence(seq, batch_max_length, pad_token_id)
            for seq in final_labels.copy()
        ], dtype=np.int64)

        # create attention masks
        encoder_attention_mask = (encoder_input_ids != pad_token_id).astype(np.int32)

        # create decoder input by shifting labels with script tokens as start tokens
        decoder_input_ids = shift_tokens_right(
            tokens_for_decoder,
            pad_token_id,
            decoder_start_tokens  # use script tokens as start tokens
        )

        # mask padding in labels
        labels_masked = np.where(
            label_ids == pad_token_id,
            -100,
            label_ids
        )

        # create decoder attention mask
        decoder_attention_mask = (decoder_input_ids != pad_token_id).astype(np.int32)

        # create batch dictionary
        batch_dict = {
            "input_ids": encoder_input_ids,
            "decoder_input_ids": decoder_input_ids,
            "labels": labels_masked,
            "attention_mask": encoder_attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
        }

        return BatchEncoding(batch_dict).data


if __name__ == "__main__":
    # test bart collator
    import register_han2han
    from transformers import AutoTokenizer
    from pathlib import Path

    print("=== Testing BARTCollator ===")

    # load tokenizer - use the new final tokenizer with special tokens!
    print("Loading tokenizer from ./final_tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("./final_tokenizer")

    # create collator
    collator = BARTCollator(
        tokenizer=tokenizer,
        rng=np.random.default_rng(43),
        max_length=512,
        model_max_length=512,
        infilling_ratio=0.35,
        poisson_lambda=3.5,
        morpheme_lambda=2.5,
        preserve_morpheme_eos=False,
        sentence_permutation=True,
        use_morpheme_masking=1.0
    )

    print(f"Mask token ID: {collator.mask_token_id}")

    # test 1: sentence permutation
    print("\n=== Test 1: Sentence Permutation ===")
    test_text = "이것은 첫 번째 문장입니다. 이것은 두 번째 문장입니다. 이것은 세 번째 문장입니다. 이것은 네 번째 문장입니다."
    permuted = collator.permute_sentences(test_text.replace(". ", ".<split>").split("<split>"))
    print(f"Original: {test_text}")
    print(f"Permuted: {permuted}")

    # test 2: text infilling
    print("\n=== Test 2: Text Infilling ===")
    test_tokens = tokenizer("이것은 텍스트 인필링 테스트입니다. 몇 개의 토큰이 마스크될 것입니다.", return_tensors=None).input_ids
    print(f"Original tokens ({len(test_tokens)}): {test_tokens[:20]}...")
    infilled = collator.span_masking(np.array(test_tokens))
    print(f"Infilled tokens ({len(infilled)}): {infilled[:20]}...")

    # count mask tokens
    mask_count = np.sum(infilled == collator.mask_token_id)
    print(f"Number of mask tokens: {mask_count}")
    print(f"Mask ratio: {mask_count / len(test_tokens):.2f}")

    # test 3: full collation with real dataset
    print("\n=== Test 3: Full Collation Process with Real Dataset ===")

    # Load data from the actual dataset
    dataset_path = Path("bartsample.arrow")
    if dataset_path.exists():
        full_indata = pl.scan_parquet('/mnt/cadazar/cjd_metadata_morphs/*.parquet'
                                      ).drop('global_idx', 'index', strict=False
                                             ).head(100).collect().with_row_index('index')

        print(f"Loaded {len(full_indata)} samples from dataset")

        # convert to examples format - filter for examples with 3+ sentences
        examples = []
        for row in full_indata.iter_rows(named=True):
            if "sentences" in row and len(row['sentences']) > 3:
                if row['sequence_length'] > collator.max_length:
                    example = {
                        "metadata": row["metadata"],
                        "original_text": row["original_text"],
                        "sentences": row["sentences"],
                        "morphs": row["morphs"],
                        "index": row["index"]
                    }
                    examples.append(example)
            if len(examples) >= 5:  # Get 5 good examples
                break

        print(f"Using {len(examples)} examples for testing")
    else:
        print(f"Dataset not found at {dataset_path}, using synthetic examples...")
        examples = [
            {
                "metadata": "뉴스기사",
                "original_text": "오늘 날씨가 맑습니다. 내일은 비가 올 예정입니다. 주말에는 다시 맑을 것입니다.",
                "sentences": ["오늘 날씨가 맑습니다.", "내일은 비가 올 예정입니다.", "주말에는 다시 맑을 것입니다."]
            },
            {
                "metadata": "위키백과",
                "original_text": "한국은 동아시아에 위치한 나라입니다. 수도는 서울입니다. 인구는 약 5천만명입니다.",
                "sentences": ["한국은 동아시아에 위치한 나라입니다.", "수도는 서울입니다.", "인구는 약 5천만명입니다."]
            },
            {
                "metadata": "일제시대",
                "original_text": "三月一日부터六日에至하는京城手形交換所交換週報는枚數二三二九八枚金額一三,六一六千圓을示하얏는데詳細는如左하더라 (單位千圓)",
                "sentences": ["三月一日부터六日에至하는京城手形交換所交換週報는枚數二三二九八枚金額一三,六一六千圓을示하얏는데詳細는如左하더라 (單位千圓)"]
            }
        ]

    batch = collator(examples, cooldown_phase=False, bucket_idx=None, thread_mecab=Mecab())

    print(f"\nBatch keys: {list(batch.keys())}")
    print(f"Input IDs shape: {batch['input_ids'].shape}")
    print(f"Decoder input IDs shape: {batch['decoder_input_ids'].shape}")
    print(f"Labels shape: {batch['labels'].shape}")
    print(f"Attention mask shape: {batch['attention_mask'].shape}")

    # decode and show examples
    for i in range(min(3, len(examples))):  # Show max 3 examples
        print(f"\n--- Example {i+1} ---")
        print(f"Original text: {examples[i]['original_text'][:100]}...")
        if 'sentences' in examples[i] and examples[i]['sentences']:
            print(f"Sentences field: {examples[i]['sentences'][:3]}..." if isinstance(examples[i]['sentences'], list) else f"Sentences field: {examples[i]['sentences'][:100]}...")
        if 'morphs' in examples[i] and examples[i]['morphs']:
            print(f"Morphs field: {examples[i]['morphs'][:3]}..." if isinstance(examples[i]['morphs'], list) else f"Morphs field: {examples[i]['morphs'][:100]}...")
        
        print(f"Metadata: {examples[i]['metadata']}")

        # decode input (corrupted)
        input_text = tokenizer.decode(batch['input_ids'][i], skip_special_tokens=False)
        print(f"Corrupted input: {input_text.replace('<pad>', '')}")
        print(f"Corrupted tokens: {batch['input_ids'][i].tolist()[:30]}...")

        # decode decoder input
        decoder_text = tokenizer.decode(batch['decoder_input_ids'][i], skip_special_tokens=False)
        print(f"Decoder input: {decoder_text.replace('<pad>', '')}")

        # decode labels (clean, with metadata masked)
        labels = batch['labels'][i]
        valid_labels = labels[labels != -100]
        label_text = tokenizer.decode(valid_labels, skip_special_tokens=False)
        print(f"Target labels: {label_text.replace('<pad>', '')}")

        # check mask count in input
        mask_count = np.sum(batch['input_ids'][i] == collator.mask_token_id)
        print(f"Masks in input: {mask_count}")

    # test metadata token lengths
    print("\n=== Test Metadata Token Lengths ===")
    if dataset_path.exists() and len(examples) > 0:
        metadata_lengths = []
        # Sample more data to get better stats
        sample_size = min(1000, len(full_indata))
        for i in range(sample_size):
            row = full_indata[i]
            meta = row["metadata"][0] if "metadata" in row else ""
            meta_tokens = tokenizer(meta + " ", return_tensors=None).input_ids  # Include space
            metadata_lengths.append(len(meta_tokens))

        print(f"Metadata token length stats (n={len(metadata_lengths)}):")
        print(f"  Min: {np.min(metadata_lengths)}")
        print(f"  Max: {np.max(metadata_lengths)}")
        print(f"  Mean: {np.mean(metadata_lengths):.1f}")
        print(f"  Median: {np.median(metadata_lengths):.1f}")
        print(f"  90th percentile: {np.percentile(metadata_lengths, 90):.1f}")
        print(f"  95th percentile: {np.percentile(metadata_lengths, 95):.1f}")
        print(f"  99th percentile: {np.percentile(metadata_lengths, 99):.1f}")

    # test 4: test with hanja text
    print("\n=== Test 4: Hanja Handling ===")
    hanja_examples = [
        {
            "metadata": "연도: 1929 시기: 초기일제강점기 유형: 출판자료",
            "original_text": "美術展覽 計劃 總督府 事案으로 明年 初次 開設總督府에서는 朝鮮에 在 한 "
            "美術의 發達을 裨補할 목적으로 東京의 帝國美術院展覽會를 倣하 여 每年 一次 美術展覽會를 開할 "
            "方針을 內定하고 時 二十六日 午前 十時 總督府 第二會議室에 朴泳孝 侯, 閔丙奭 子, 書畵協會의 "
            "丁大有, 金敦熙, 李道榮 外 諸氏, 書畵硏究會의 金圭鎭 氏, 日本人側 書畵家로  高木背水 外 數氏 "
            "其他 書畵家에 關係있는 人士를 招請하고 水野 政務總監 以下 學務當局者가 會合하여 此에 關한 相議會를 "
            "開한바 滿場一致로 此計劃에 對한 贊成이 有하였으므로 此에  關한 規定의 發表가 有하리라는데 該展覽會는 "
            "此를 東洋畵(第一部), 西洋畵 及 彫刻(第二部), 書(第三部) 의 三部로 定하고 出品人의 資格은 "
            "制限치 아니하되 出品은 審査委員의 鑑査를  經한 것에 限하여 陳列하며 特別히 前回의 展覽會에서 "
            "一等賞을 受한 者와 其他 特別한 者에 는 無鑑査出品을 認定하여 同一人의 出品은 各部에 二點 "
            "以內로 出品. 一點은 幅二間 以內로 制限하였으며 出品은 製作本人이 반드시 此를 □하되 故人의 "
            "製作은 相續人이 出品함을 得하며 (一) 製作 後 五年을 經過한 것 (二) 該展覽會에 陳列하였던 것 "
            "(三) 治安風敎에 有害하다고 認한 것은 出品치 못하며 出品코자 하는 者는 作 品을 相當히 表裝하고 "
            "每點에 命題 及 出品人 氏名을 記한 出品札을 添附하여 出品願書 解說書와 共히 事務所에 提出함을 要하며 "
            "作品의 鑑査 及 審査를 行하기 爲하여 審査委員會를 設하되 委員長은 政務總監이 此에 當하고 委員 은 "
            "官民을 通하여 學識經驗이 有한 人士로 選하여 一, 二, 三部의 各部에 分屬케 하여 初次에는 出品한 "
            "作品의 陳列할 與否를 鑑査하고 更히 鑑査를 行한 陳列品에 對하여 總히 審査를 行하되 各部 委員 過半數의 "
            "出席과 出席委員 過半數의 同意로써 此를 定하며 審査의 結果 優秀 한 作品에는 一, 二, 三四의 等級을 "
            "定하여 入賞한 作品의 製作者에게는 金牌(一等), 銀牌(二等), 銅牌(三等), 褒狀(四等)의 賞品을 朝鮮總督이 "
            "授與할 터이며 明年 五, 六月의 交에 京城 永樂町 商品陳列館을 會場으로 使用하여 會期 約 三十日間으로 "
            "第一回 展覽會를 開하고 每日 午前 九時부터 午後 五時까지 一般에 公開하기로 方今 計劃 準備中이라더라.",
            "sentences": ["美術展覽 計劃 總督府", "事案으로 明年 初次 開設總督府에서는 朝鮮에 在 한 "
            "美術의 發達을 裨補할 목적으로 東京의 帝國美術院展覽會를 倣하 여 每年 一次 美術展覽會를 開할 "
            "方針을 內定하고 時 二十六日 午前 十時 總督府 第二會議室에 朴泳孝 侯, 閔丙奭 子, 書畵協會의 "
            "丁大有, 金敦熙, 李道榮 外 諸氏, 書畵硏究會의 金圭鎭 氏, 日本人側 書畵家로  高木背水 外 數氏 "
            "其他 書畵家에 關係있는 人士를 招請하고 水野 政務總監 以下 學務當局者가 會合하여 此에 關한 相議會를 "
            "開한바 滿場一致로 此計劃에 對한 贊成이 有하였으므로 此에  關한 規定의 發表가 有하리라는데 該展覽會는 "
            "此를 東洋畵(第一部), 西洋畵 及 彫刻(第二部), 書(第三部) 의 三部로 定하고 出品人의 資格은 "
            "制限치 아니하되 出品은 審査委員의 鑑査를  經한 것에 限하여 陳列하며 特別히 前回의 展覽會에서 "
            "一等賞을 受한 者와 其他 特別한 者에 는 無鑑査出品을 認定하여 同一人의 出品은 各部에 二點 "
            "以內로 出品.", "一點은 幅二間 以內로 制限하였으며 出品은 製作本人이 반드시 此를 □하되 故人의 "
            "製作은 相續人이 出品함을 得하며", "(一) 製作 後 五年을 經過한 것", "(二) 該展覽會에 陳列하였던 것",
            "(三) 治安風敎에 有害하다고 認한 것은 出品치 못하며 出品코자 하는 者는 作 品을 相當히 表裝하고 "
            "每點에 命題 及 出品人 氏名을 記한 出品札을 添附하여 出品願書 解說書와 共히 事務所에 提出함을 要하며 "
            "作品의 鑑査 及 審査를 行하기 爲하여 審査委員會를 設하되 委員長은 政務總監이 此에 當하고 委員 은 "
            "官民을 通하여 學識經驗이 有한 人士로 選하여 一, 二, 三部의 各部에 分屬케 하여 初次에는 出品한 "
            "作品의 陳列할 與否를 鑑査하고 更히 鑑査를 行한 陳列品에 對하여 總히 審査를 行하되 各部 委員 過半數의 "
            "出席과 出席委員 過半數의 同意로써 此를 定하며 審査의 結果 優秀 한 作品에는 一, 二, 三四의 等級을 "
            "定하여 入賞한 作品의 製作者에게는 金牌(一等), 銀牌(二等), 銅牌(三等), 褒狀(四等)의 賞品을 朝鮮總督이 "
            "授與할 터이며 明年 五, 六月의 交에 京城 永樂町 商品陳列館을 會場으로 使用하여 會期 約 三十日間으로 "
            "第一回 展覽會를 開하고 每日 午前 九時부터 午後 五時까지 一般에 公開하기로 方今 計劃 準備中이라더라."]
        }
    ]

    hanja_batch = collator(hanja_examples, cooldown_phase=False, bucket_idx=None, thread_mecab=Mecab())
    print(f"Original: {hanja_examples[0]['metadata']} {hanja_examples[0]['original_text']}")

    input_text = tokenizer.decode(hanja_batch['input_ids'][0], skip_special_tokens=False)
    print(f"Corrupted input: {input_text.replace('<pad>', '')}")

    decoder_text = tokenizer.decode(hanja_batch['decoder_input_ids'][0], skip_special_tokens=False)
    print(f"Decoder input: {decoder_text.replace('<pad>', '')}")

    labels = hanja_batch['labels'][0]
    valid_labels = labels[labels != -100]
    label_text = tokenizer.decode(valid_labels, skip_special_tokens=False) 
    print(f"Target labels (hangul): {label_text.replace('<pad>', '')}")

    # test 5: measure actual corruption rates
    print("\n=== Test 5: Corruption Rate Analysis ===")
    print("Testing actual vs target corruption rates...")

    test_corruption_rates = [0.15, 0.30, 0.45]
    test_morpheme_ratios = [0.0, 0.5, 1.0]

    for corruption_ratio in test_corruption_rates:
        print(f"\nTesting corruption_ratio={corruption_ratio:.0%}")

        for morpheme_ratio in test_morpheme_ratios:
            test_collator = BARTCollator(
                tokenizer=tokenizer,
                rng=np.random.default_rng(43),
                max_length=256,
                model_max_length=256,
                infilling_ratio=corruption_ratio,  # use the test corruption ratio
                use_morpheme_masking=morpheme_ratio,  # use the test morpheme ratio
                poisson_lambda=3.5,
                morpheme_lambda=2.5,
                preserve_morpheme_eos=False,
                sentence_permutation=True
            )

            # test on examples without morphemes (pure token-based)
            token_diffs = []
            mask_counts = []

            for _ in range(1):
                batch = test_collator([hanja_examples[0]], cooldown_phase=False, bucket_idx=None,
                                      thread_mecab=Mecab())
                input_ids = batch['input_ids'][0]
                labels = batch['labels'][0]

                # compare labels to input_ids to see what was actually corrupted
                # remove padding and special tokens from both
                valid_labels = labels[labels != -100]
                valid_input = input_ids[(input_ids != 0) & (input_ids != 2) & (input_ids != 3)]

                # count mask tokens in input
                mask_count = (input_ids == tokenizer.mask_token_id).sum()

                # count original tokens
                original_tokens = len(valid_labels)

                # try to align and count matches (this is approximate due to shuffling)
                # but we can at least count how many tokens were replaced/removed
                tokens_in_corrupted = len(valid_input) - mask_count
                tokens_corrupted = original_tokens - tokens_in_corrupted

                if original_tokens > 0:
                    effective_corruption = float(tokens_corrupted) / float(original_tokens)
                    token_diffs.append(effective_corruption)
                    mask_counts.append(mask_count.item())

            if token_diffs:
                avg_corruption = np.mean(token_diffs) * 100
                std_corruption = np.std(token_diffs) * 100
                avg_masks = np.mean(mask_counts)
                print(f"  morpheme={morpheme_ratio:.0%}: effective={avg_corruption:.1f}%±{std_corruption:.1f}% "
                      f"(target={corruption_ratio*100:.0f}%, masks={avg_masks:.1f})")

    print("\n=== All tests completed! ===")