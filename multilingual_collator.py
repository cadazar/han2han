#!/usr/bin/env python3
# coding: utf-8
"""
Han2Han Multilingual Collator for training on Korean + CJK + English data.
Inherits from BARTCollator and adds script-aware processing.
"""

import sys
import re
import dataclasses
from typing import Dict, List, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from dynamic_data_loader import DataSourceConfig

# initialize logging first, before any other imports
import logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    stream=sys.stdout,
    force=True
)

# initialize absl logging to suppress warnings
from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.WARNING)

import numpy as np

from transformers import AutoTokenizer, BatchEncoding
from datasets import Dataset, IterableDataset
from functools import partial

from h2hcollator import BARTCollator
from han2han_tools import has_hanja, transcribe
from logging_utils import log_from_all_processes, log_from_main_process

logger = logging.getLogger(__name__)


# hanja unicode ranges for efficient detection
HANJA_RANGES = [
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs
    (0x3400, 0x4DBF),   # CJK Extension A
    (0x20000, 0x2A6DF), # CJK Extension B
    (0x2A700, 0x2B73F), # CJK Extension C
    (0x2B740, 0x2B81F), # CJK Extension D
    (0x2B820, 0x2CEAF), # CJK Extension E
    (0x2CEB0, 0x2EBEF), # CJK Extension F
    (0xF900, 0xFAFF),   # CJK Compatibility Ideographs
    (0x2F800, 0x2FA1F), # CJK Compatibility Supplement
]

def is_hanja_char(char: str) -> bool:
    """Check if a character is Hanja."""
    code = ord(char)
    for start, end in HANJA_RANGES:
        if start <= code <= end:
            return True
    return False


@dataclasses.dataclass
class MultilingualCollator(BARTCollator):
    """
    Han2Han multilingual collator with sentence splitting and multilingual
    source sampling. Compatible with tf.data.Dataset.from_generator.
    Fixed sequence length (no bucketing) for simplicity.

    Supports splitting Korean samples into heavy/light Hanja sub-sources
    for curriculum learning.
    """

    # effective sampling ratios when using korean sub-sources
    # these control the actual token-level balancing
    sampling_ratios: Dict[str, float] = dataclasses.field(default_factory=lambda: {
        'korean_hanja_heavy': 0.20,   # korean with >=10% hanja tokens
        'korean_hanja_light': 0.25,   # korean with <20% hanja tokens
        'c4_en':              0.20,   # english web
        'mc4_zh':             0.15,   # chinese web
        'mc4_ja':             0.20,   # japanese web
    })

    # hanja concentration threshold
    hanja_heavy_threshold: float = 0.1  # >=10% hanja tokens = heavy

    # datasets organized by source
    datasets: Dict[str, Dataset] = dataclasses.field(default_factory=dict)

    # pre-loaded eval datasets for sources with stratified splits
    # if a source has has_stratified_split=True and its eval data is here,
    # we skip train_test_split and use this data directly
    eval_datasets: Dict[str, Dataset] = dataclasses.field(default_factory=dict)

    # source configurations with field mappings (single source of truth)
    source_configs: Dict[str, 'DataSourceConfig'] = dataclasses.field(default_factory=dict)

    # fixed sequence length (no bucketing)
    use_bucketing: bool = False

    # evaluation configuration
    eval_split_ratio: float = 0.05  # 5% of data for evaluation by default
    eval_batch_size: int = 4

    # randomness
    seed: int = 42

    # token counting and buffer configuration
    buffer_size: int = 100000  # massive buffer for better token-level balancing (uses ~20GB RAM out of 400GB available)
    token_counts: Dict[str, int] = dataclasses.field(default_factory=dict)  # track tokens per sub-source
    sample_counts: Dict[str, int] = dataclasses.field(default_factory=dict)  # track samples per sub-source

    # precomputed hanja lookup (numpy array for vectorized lookup)
    token_hanja_lookup: np.ndarray = dataclasses.field(default=None)

    # korean sample sub-buffers
    korean_sub_buffers: Dict[str, List[Dict]] = dataclasses.field(default_factory=lambda: {
        'korean_hanja_heavy': [],
        'korean_hanja_light': [],
    })

    def __post_init__(self):
        super().__post_init__()

        # validate sampling ratios for actual token balancing
        total_sampling = sum(self.sampling_ratios.values())
        if abs(total_sampling - 1.0) > 0.01:
            log_from_all_processes(logger, 'warning', f"Sampling ratios sum to {total_sampling}, normalizing to 1.0")
            self.sampling_ratios = {k: v/total_sampling for k, v in self.sampling_ratios.items()}

        # initialize token and sample counters for sampling ratios
        for source in self.sampling_ratios:
            if source not in self.token_counts:
                self.token_counts[source] = 0
            if source not in self.sample_counts:
                self.sample_counts[source] = 0

        # build hanja lookup table for all vocabulary tokens
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            log_from_main_process(logger, 'info', "Building Hanja token lookup table...")
            self._build_hanja_token_lookup()
            if self.token_hanja_lookup is not None:
                log_from_main_process(logger, 'info',
                    f"Hanja token lookup built: {len(self.token_hanja_lookup)} entries, "
                    f"{self.token_hanja_lookup.sum()} contain Hanja")

        if self.datasets:
            self._instantiate_dsets()
            if len(self.datasets) > 1:
                # format ratios with 3 decimal places for readability
                formatted_ratios = {k: f"{v:.3f}" for k, v in self.sampling_ratios.items()}
                log_from_main_process(logger, 'info', f"Sampling ratios (with Korean sub-sources): {formatted_ratios}")
                log_from_main_process(logger, 'info', f"Train/eval split ratio: {self.eval_split_ratio}")
                log_from_main_process(logger, 'info', f"Buffer size for token-level balancing: {self.buffer_size}")
                log_from_main_process(logger, 'info', f"Hanja heavy threshold: {self.hanja_heavy_threshold:.1%}")

    def _build_hanja_token_lookup(self):
        """Build lookup table for which tokens contain Hanja characters."""
        try:
            vocab = self.tokenizer.get_vocab()
            vocab_size = len(self.tokenizer)
            self.token_hanja_lookup = np.zeros(vocab_size, dtype=bool)

            for token, token_id in vocab.items():
                clean_token = token.replace("▁", "")
                contains_hanja = any(is_hanja_char(char) for char in clean_token)
                if token_id < vocab_size:
                    self.token_hanja_lookup[token_id] = contains_hanja

        except Exception as e:
            log_from_main_process(logger, 'error',
                f"Failed to build Hanja token lookup: {e}")
            self.token_hanja_lookup = None

    def _calculate_hanja_ratio(self, sample: Dict[str, np.ndarray]) -> float:
        """
        Calculate the ratio of Hanja tokens in a sample.

        Checks BOTH encoder and decoder inputs since Hanja may appear in either
        due to transcription during data augmentation. Returns the maximum ratio
        to ensure we capture samples with Hanja regardless of position.

        Args:
            sample: Tokenized sample with encoder/decoder inputs

        Returns:
            Maximum Hanja ratio across encoder and decoder (0.0 to 1.0)
        """
        def count_hanja_in_tokens(tokens):
            """Helper to count Hanja in token array - VECTORIZED."""
            if tokens is None or len(tokens) == 0:
                return 0, 0

            # ensure tokens is a numpy array for boolean indexing
            if not isinstance(tokens, np.ndarray):
                tokens = np.array(tokens)

            # filter out padding and special tokens (but keep script tokens)
            valid_mask = (tokens != -100) & (tokens != self.tokenizer.pad_token_id)
            valid_tokens = tokens[valid_mask]

            if len(valid_tokens) == 0:
                return 0, 0

            # count Hanja tokens using lookup table
            if self.token_hanja_lookup is not None:
                # vectorized lookup: index directly into numpy array
                hanja_count = self.token_hanja_lookup[valid_tokens].sum()
                return int(hanja_count), len(valid_tokens)
            else:
                # fallback: check if decoded text has any hanja
                text = self.tokenizer.decode(valid_tokens, skip_special_tokens=True)
                if has_hanja(text):
                    # rough estimate: assume 30% of tokens are Hanja if text contains any
                    return int(len(valid_tokens) * 0.3), len(valid_tokens)
                return 0, len(valid_tokens)

        # check encoder inputs
        encoder_hanja, encoder_total = count_hanja_in_tokens(sample.get('input_ids'))
        encoder_ratio = encoder_hanja / encoder_total if encoder_total > 0 else 0.0

        # check decoder inputs/labels (decoder sees the clean text)
        decoder_tokens = sample.get('labels', sample.get('decoder_input_ids'))
        decoder_hanja, decoder_total = count_hanja_in_tokens(decoder_tokens)
        decoder_ratio = decoder_hanja / decoder_total if decoder_total > 0 else 0.0

        # return maximum ratio - if either side has significant Hanja, it's a Hanja-heavy sample
        # this captures both Hanja→Hangul and Hangul→Hanja transcription tasks
        return max(encoder_ratio, decoder_ratio)

    def _classify_korean_sample(self, sample: Dict[str, np.ndarray]) -> str:
        """
        Classify a Korean sample as heavy or light based on Hanja concentration.

        Returns:
            'korean_hanja_heavy' or 'korean_hanja_light'
        """
        hanja_ratio = self._calculate_hanja_ratio(sample)

        if hanja_ratio >= self.hanja_heavy_threshold:
            return 'korean_hanja_heavy'
        else:
            return 'korean_hanja_light'

    def count_tokens_in_sample(self, sample: Dict[str, np.ndarray]) -> int:
        """Count non-padding tokens in a sample."""
        # count non-padding tokens in labels (which represents the actual content)
        labels = sample['labels']
        return int(np.sum(labels != -100))  # -100 is the ignored index for labels

    def __getstate__(self):
        """Exclude unpicklable iterator/buffer attributes.

        datasets.IterableDataset.from_generator hashes the generator function by pickling
        its closure, which includes self. train_iterators, eval_iterators, and sample_buffer
        hold live generator/iterator objects that cannot be pickled. Excluding them here
        makes the collator picklable for hashing without affecting runtime state.
        """
        state = self.__dict__.copy()
        for key in ('train_iterators', 'eval_iterators', 'sample_buffer'):
            state.pop(key, None)
        return state

    def _instantiate_dsets(self, cooldown_phase=False, new_datasets=None,
                           new_source_configs=None, new_sampling_ratios=None):
        """Create dataset iterators with token-level balancing instead of interleaving.

        When called with ``new_datasets`` (phase-2 transition), the caller should
        also pass ``new_source_configs`` and ``new_sampling_ratios`` from the
        same ``get_streaming_datasets(args)`` return tuple. Without those, the
        collator's transform map and per-source buffer weights stay frozen at
        phase-1 values while the dataset iterators yield phase-2 sources --
        which raises ``ValueError("No configuration found for source ...")``
        the first time a phase-2-only source (e.g. ``finepdfs_korean``) shows
        up at ``_transform_example``.
        """

        # if new datasets provided, replace existing ones
        if new_datasets is not None:
            log_from_main_process(logger, 'info', f"Replacing datasets for cooldown phase transition")
            self.datasets = new_datasets
            if new_source_configs is not None:
                # Track sources that are leaving the active config so _transform_example
                # can silently drop late-arriving examples from previous-phase generators.
                dropped = set(self.source_configs.keys()) - set(new_source_configs.keys())
                stale = self.__dict__.setdefault('_stale_source_names', set())
                stale |= dropped
                # Reset the warn-once announcer so each transition logs again.
                self._announced_stale_drops = set()
                self.source_configs = new_source_configs
                log_from_main_process(logger, 'info',
                    f"Refreshed source_configs for phase transition: "
                    f"{list(new_source_configs.keys())}")
                if dropped:
                    log_from_main_process(logger, 'info',
                        f"Sources dropped this transition (stale generators will drain): "
                        f"{sorted(dropped)}")
            if new_sampling_ratios is not None:
                total = sum(new_sampling_ratios.values())
                if total > 0:
                    self.sampling_ratios = {k: v / total for k, v in new_sampling_ratios.items()}
                else:
                    self.sampling_ratios = dict(new_sampling_ratios)
                # mirror __post_init__: seed counter dicts for any new sources, but
                # preserve existing counts so phase-1 token-budget tracking carries over.
                for source in self.sampling_ratios:
                    self.token_counts.setdefault(source, 0)
                    self.sample_counts.setdefault(source, 0)
                log_from_main_process(logger, 'info',
                    f"Refreshed sampling_ratios for phase transition: "
                    f"{ {k: f'{v:.3f}' for k, v in self.sampling_ratios.items()} }")
            # clear stale per-source keyed attributes (safe: keyed by source name, not live-read)
            # do NOT delete train_iterators / eval_iterators / source_names / sample_buffer here:
            # the tf.data background thread may still be calling get_next_balanced_sample() which
            # reads those attributes. they will be atomically overwritten at the end of this method.
            for attr_name in list(self.__dict__.keys()):
                if attr_name.startswith('_dataset_') or attr_name.startswith('_cooldown_') or attr_name.startswith('_backup_'):
                    delattr(self, attr_name)

        # if datasets already exist and no new ones provided, just update the cooldown flag
        elif hasattr(self, 'sampled_datasets'):
            log_from_main_process(logger, 'info', f"Updating cooldown phase to {cooldown_phase}")
            self.cooldown_phase = cooldown_phase
            # update stored cooldown flags for all datasets
            for dn in self.datasets.keys():
                if hasattr(self, f'_cooldown_{dn}'):
                    setattr(self, f'_cooldown_{dn}', cooldown_phase)
            return

        # otherwise create datasets for the first time
        self.cooldown_phase = cooldown_phase

        # transform and pre-tokenize each dataset using iteration (no .map() to avoid multiprocessing deadlocks)
        tokenized_datasets_train = []
        tokenized_datasets_eval = []
        source_names = []

        for dn, ds in self.datasets.items():
            log_from_main_process(logger, 'info', f"Creating transformed iterator for {dn} dataset...")

            # grab custom info to avoid errors in split
            source_type = ds.info.__dict__.pop('source_type')
            data_type = ds.info.__dict__.pop('data_type', 'denoising')

            # check if this source has pre-defined stratified splits
            source_config = self.source_configs.get(dn)
            has_stratified = source_config and getattr(source_config, 'has_stratified_split', False)
            eval_key = f"{dn}_eval"
            has_preloaded_eval = eval_key in self.eval_datasets

            if has_stratified and has_preloaded_eval:
                # use pre-defined stratified split instead of arbitrary train_test_split
                trainset = ds
                evalset = self.eval_datasets[eval_key]
                log_from_main_process(logger, 'info',
                    f"Using pre-defined stratified split for {dn} (train={len(trainset)}, eval={len(evalset)})")
            else:
                # fallback to arbitrary split for sources without pre-defined splits
                split_set = ds.train_test_split(self.eval_split_ratio, generator=self.rng)
                trainset = split_set['train']
                evalset = split_set['test']
                if has_stratified and not has_preloaded_eval and self.eval_datasets:
                    log_from_main_process(logger, 'warning',
                        f"{dn} has_stratified_split=True but no pre-loaded eval data found (key={eval_key}), using fallback split")

            # add custom info back to each set
            trainset.info.source_type = source_type
            trainset.info.data_type = data_type
            evalset.info.source_type = source_type
            evalset.info.data_type = data_type

            # create infinite transformed iterator that applies transformation and tokenization on-the-fly
            def create_infinite_transformed_iterator(dataset, source_name, is_train=True):
                # capture dataset info and custom attributes before iterating
                # these MUST exist - we set them in prepare_multilingual_data.py
                source_type = dataset.info.source_type
                data_type = dataset.info.data_type

                # store params as instance variables to avoid closure/pickle issues
                # this is a workaround for the generator pickling problem
                # use distinct suffix for train vs eval to avoid attribute overwrite
                attr_suffix = 'train' if is_train else 'eval'
                attr_key = f'{source_name}_{attr_suffix}'

                setattr(self, f'_dataset_{attr_key}', dataset)
                setattr(self, f'_source_type_{attr_key}', source_type)
                setattr(self, f'_data_type_{attr_key}', data_type)
                setattr(self, f'_is_train_{attr_key}', is_train)
                setattr(self, f'_cooldown_{attr_key}', cooldown_phase)

                # create a simple generator function that references self
                # use a unique name for each dataset to avoid conflicts
                def make_generator(collator, dataset_name, attr_key):
                    def infinite_transformed_generator():
                        # retrieve stored params using the unique attr_key
                        dataset_obj = getattr(collator, f'_dataset_{attr_key}')
                        source_type_str = getattr(collator, f'_source_type_{attr_key}')
                        is_train_flag = getattr(collator, f'_is_train_{attr_key}')
                        cooldown = getattr(collator, f'_cooldown_{attr_key}')

                        # create a mock dataset object for _transform_example compatibility
                        class DatasetMock:
                            def __init__(self, source_type):
                                class InfoMock:
                                    pass
                                self.info = InfoMock()
                                self.info.source_type = source_type

                        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer.name_or_path)  # thread safe, one per dataset

                        # create morphological tokenizer (thread-safe, one per dataset)
                        from konlpy.tag import Mecab
                        morpheme_tokenizers = {'ko': Mecab()}

                        mock_dataset = DatasetMock(source_type_str)
                        epoch_count = 0
                        example_count = 0

                        while True:
                            epoch_count += 1
                            if epoch_count % 10 == 1:
                                log_from_main_process(logger, 'info', f"Dataset {dataset_name} ({'train' if is_train_flag else 'eval'}): "
                                                                       f"starting epoch {epoch_count}, processed {example_count} examples so far")

                            if is_train_flag:
                                if hasattr(dataset_obj.info, 'source_type'):
                                    delattr(dataset_obj.info, 'source_type')
                                if hasattr(dataset_obj.info, 'data_type'):
                                    delattr(dataset_obj.info, 'data_type')
                                epoch_dataset = dataset_obj.shuffle(seed=collator.seed + epoch_count)
                            else:
                                epoch_dataset = dataset_obj

                            for raw_example in epoch_dataset:
                                example_count += 1

                                # track raw items consumed for fast checkpoint restoration
                                if not hasattr(collator, '_raw_consumed'):
                                    collator._raw_consumed = {}
                                collator._raw_consumed[source_name] = (
                                    collator._raw_consumed.get(source_name, 0) + 1)

                                # fast-skip: bypass tokenization during iterator restoration
                                fast_skip_key = f'_fast_skip_{source_name}'
                                remaining_skip = getattr(collator, fast_skip_key, 0)
                                if remaining_skip > 0:
                                    setattr(collator, fast_skip_key, remaining_skip - 1)
                                    continue

                                transformed_example = collator._transform_example(raw_example, mock_dataset)

                                # skip if transformation returned None (null/empty required fields)
                                if transformed_example is None:
                                    continue

                                # pre-roll all stochastic corruption decisions; supervised tasks
                                # return None from _sample_corruption_plan and fall through to
                                # the legacy collator path unchanged.
                                if hasattr(collator, '_sample_corruption_plan'):
                                    plan = collator._sample_corruption_plan(
                                        transformed_example, cooldown_phase=cooldown
                                    )
                                    if plan is not None:
                                        transformed_example['_corruption_plan'] = plan
                                        transformed_example['_training_mode'] = plan['training_mode']
                                        if plan.get('prompt') is not None:
                                            transformed_example['metadata'] = plan['prompt']

                                tokenized_example, source = collator(transformed_example, cooldown_phase=cooldown, return_source=True,
                                                                     tokenizer=tokenizer, morpheme_tokenizers=morpheme_tokenizers)

                                # skip None examples (too short or malformed)
                                if tokenized_example is None:
                                    continue

                                # skip examples with too few input tokens (< 20 tokens)
                                if 'input_ids' in tokenized_example:
                                    # count non-padding tokens
                                    input_ids = tokenized_example['input_ids']
                                    if hasattr(input_ids, 'tolist'):
                                        input_ids = input_ids.tolist()
                                    non_pad_count = sum(1 for tok in input_ids if tok != tokenizer.pad_token_id)
                                    if non_pad_count < 20:
                                        continue

                                tokenized_example['_source'] = source
                                tokenized_example['_source_name'] = dataset_name

                                yield tokenized_example
                    return infinite_transformed_generator

                generator_fn = make_generator(self, source_name, attr_key)
                return IterableDataset.from_generator(generator_fn)

            # create infinite transformed and tokenized datasets
            infinite_ds_train = create_infinite_transformed_iterator(trainset, dn, is_train=True)
            infinite_ds_eval = create_infinite_transformed_iterator(evalset, dn, is_train=False)
            tokenized_datasets_train.append(infinite_ds_train)
            tokenized_datasets_eval.append(infinite_ds_eval)
            source_names.append(dn)
            # and add info back to source ds for subsequent calls to this function
            ds.info.source_type = source_type
            ds.info.data_type = data_type

        log_from_main_process(logger, 'info', "Pre-tokenization complete! Configuring data iterators with set probabilities...")

        # store dataset iterators for token-based sampling
        self.train_iterators = {source: iter(ds) for source, ds in zip(source_names, tokenized_datasets_train)}
        self.eval_iterators = {source: iter(ds) for source, ds in zip(source_names, tokenized_datasets_eval)}
        self.source_names = source_names

        # store backup references to infinite datasets for recovery
        for source, ds in zip(source_names, tokenized_datasets_train):
            setattr(self, f'_backup_{source}_train_ds', ds)
        for source, ds in zip(source_names, tokenized_datasets_eval):
            setattr(self, f'_backup_{source}_eval_ds', ds)

        # create buffer for token-level balancing
        self.sample_buffer = {source: [] for source in source_names}
        self._samples_yielded = {source: 0 for source in source_names}
        self.cooldown_phase = cooldown_phase

        # create a simple iterator wrapper for training (with buffering/balancing)
        class TokenBalancedIterator:
            def __init__(self, collator):
                self.collator = collator

            def __iter__(self):
                return self

            def __next__(self):
                sample = self.collator.get_next_balanced_sample(use_eval_iterators=False)
                return sample

        # create a lightweight iterator for eval (no buffering, just round-robin through sources)
        class DirectEvalIterator:
            def __init__(self, collator):
                self.collator = collator
                self.source_idx = 0

            def __iter__(self):
                return self

            def __next__(self):
                if not hasattr(self.collator, 'eval_iterators') or not self.collator.eval_iterators:
                    raise StopIteration("No eval iterators available")

                sources = list(self.collator.eval_iterators.keys())
                if not sources:
                    raise StopIteration("No sources available")

                # round-robin through sources
                source = sources[self.source_idx % len(sources)]
                self.source_idx += 1

                try:
                    sample = next(self.collator.eval_iterators[source])
                    sample['_data_source'] = source
                    return sample
                except StopIteration:
                    # recreate iterator from backup if exhausted
                    backup_attr = f'_backup_{source}_eval_ds'
                    if hasattr(self.collator, backup_attr):
                        self.collator.eval_iterators[source] = iter(getattr(self.collator, backup_attr))
                        sample = next(self.collator.eval_iterators[source])
                        sample['_data_source'] = source
                        return sample
                    raise

        self.sampled_datasets = TokenBalancedIterator(self)
        self.eval_data = DirectEvalIterator(self)

        log_from_main_process(logger, 'info', f"Token-balanced dataset created with {len(source_names)} sources")
        log_from_main_process(logger, 'info', f"Sources: {source_names}")
        log_from_main_process(logger, 'info', f"Sampling ratio keys: {list(self.sampling_ratios.keys())}")

    def _transform_example(self, raw_example: Dict[str, Any], dataset) -> Dict[str, Any]:
        """Transform raw examples using field mappings from DataSourceConfig.

        Returns None if example has null/empty required fields (will be skipped by generator).
        """

        # get source name from dataset info
        source_name = getattr(dataset.info, 'source_type', None) if hasattr(dataset, 'info') else None

        if source_name is None:
            error_msg = (
                f"Dataset has no source_type attribute. "
                f"Example keys: {list(raw_example.keys())}"
            )
            log_from_all_processes(logger, 'error', f"[TRANSFORM ERROR] {error_msg}")
            raise ValueError(error_msg)

        # look up source config
        config = self.source_configs.get(source_name)

        if config is None:
            # Phase-transition drain path: a generator captured during a previous
            # phase is still feeding raw_examples from a source that's no longer
            # in source_configs. The packed generator's closure holds onto the
            # phase-N dataset_obj and keeps iterating it forever (while True);
            # we can't kill it from here, but skipping its yields lets the new
            # phase-(N+1) generators dominate the mix as they drain. Log warn-
            # once per stale source so the audit trail is loud per CLAUDE.md.
            stale = getattr(self, '_stale_source_names', set())
            if source_name in stale:
                announced = self.__dict__.setdefault('_announced_stale_drops', set())
                if source_name not in announced:
                    log_from_main_process(logger, 'warning',
                        f"Dropping examples from stale-phase source '{source_name}' "
                        f"(no longer in source_configs after phase transition). "
                        f"Subsequent drops from this source silenced.")
                    announced.add(source_name)
                return None

            error_msg = (
                f"No configuration found for source '{source_name}'. "
                f"Available sources: {list(self.source_configs.keys())}"
            )
            log_from_all_processes(logger, 'error', f"[TRANSFORM ERROR] {error_msg}")
            raise ValueError(error_msg)

        # validate required text field is not null/empty
        text_value = raw_example.get(config.text_field)
        if text_value is None or text_value == '':
            log_from_all_processes(logger, 'debug',
                f"[TRANSFORM] Skipping example with null/empty {config.text_field} from {source_name}")
            return None

        # validate target field for supervised tasks (ocr, translation, transcription)
        if config.target_field:
            target_value = raw_example.get(config.target_field)
            if target_value is None or target_value == '':
                log_from_all_processes(logger, 'debug',
                    f"[TRANSFORM] Skipping example with null/empty {config.target_field} from {source_name}")
                return None

        # use field mappings from config to transform example
        transformed = {
            'original_text': text_value,
            'metadata': raw_example.get(config.metadata_field, ''),
            'source': config.name,
            'sentences': raw_example.get('sentences', []),
            'data_type': config.data_type,
        }

        # add task-specific fields if present
        if config.target_field:
            transformed['target'] = raw_example[config.target_field]

        if config.year_field and config.year_field in raw_example:
            transformed['year'] = raw_example[config.year_field]

        # for 'mixed' data_type, try to parse year from metadata if not already set
        if config.data_type == 'mixed' and 'year' not in transformed:
            metadata = transformed.get('metadata', '')
            if metadata:
                year_match = re.search(r'연도:\s*(\d{4})', metadata)
                if year_match:
                    transformed['year'] = int(year_match.group(1))

        if config.sentence1_field and config.sentence1_field in raw_example:
            transformed['sentence1'] = raw_example[config.sentence1_field]
            transformed['sentence2'] = raw_example[config.sentence2_field]

        if config.score_field and config.score_field in raw_example:
            transformed['rounded_score'] = raw_example[config.score_field]

        # for STS tasks that already have formatted input/target
        if 'input_text' in raw_example:
            transformed['input_text'] = raw_example['input_text']
        if 'target_text' in raw_example:
            transformed['target_text'] = raw_example['target_text']

        # pass through CoT rationale for chat-format SFT (chat_sft_collator
        # picks this up to wrap in <|think|>...<|assistant|>)
        if 'thinking' in raw_example and raw_example['thinking']:
            transformed['thinking'] = raw_example['thinking']

        return transformed

    def get_generator_state(self) -> Dict[str, Any]:
        """Returns collator state for checkpointing, including per-source sample counts."""
        state = super().get_generator_state()
        state['iter_state']['samples_per_source'] = dict(
            getattr(self, '_samples_yielded', {}))
        state['iter_state']['raw_consumed_per_source'] = dict(
            getattr(self, '_raw_consumed', {}))
        return state

    def advance_iterators(self, samples_per_source: Dict[str, int],
                          raw_consumed_per_source: Optional[Dict[str, int]] = None):
        """Set fast-skip counts so generators bypass tokenization on the next N raw items.

        Uses raw item counts (pre-tokenization) when available, which allows the
        generator to skip cheaply by just reading raw parquet rows. Falls back to
        yielded item counts for old checkpoints that lack raw counts.
        """
        skip_counts = raw_consumed_per_source if raw_consumed_per_source else samples_per_source
        if not skip_counts:
            return

        total = sum(skip_counts.values())
        mode = "raw items" if raw_consumed_per_source else "yielded items (approximate)"
        log_from_main_process(logger, 'info',
            f"Fast-skip iterator restoration: {total} {mode} across {len(skip_counts)} sources")

        for source, count in skip_counts.items():
            if count > 0:
                setattr(self, f'_fast_skip_{source}', count)
                log_from_main_process(logger, 'info',
                    f"  {source}: will fast-skip {count} items")

    def get_next_balanced_sample(self, use_eval_iterators: bool = False) -> Dict[str, np.ndarray]:
        """Get next sample using token-level balancing with Korean sub-source routing.

        Args:
            use_eval_iterators: If True, use eval_iterators instead of train_iterators.
                               This should be True when sampling for evaluation.
        """
        if not hasattr(self, '_samples_yielded'):
            self._samples_yielded = {}

        log_from_all_processes(logger, 'debug', f"=== get_next_balanced_sample called (eval={use_eval_iterators}) ===")

        # select which iterators to use
        iterators_attr = 'eval_iterators' if use_eval_iterators else 'train_iterators'
        backup_suffix = '_eval_ds' if use_eval_iterators else '_train_ds'

        # select buffers
        korean_sub_buffers = self.korean_sub_buffers
        sample_buffer = self.sample_buffer

        log_from_all_processes(logger, 'debug', f"Has {iterators_attr}: {hasattr(self, iterators_attr)}")

        # fill buffers if needed (only if iterators exist)
        if hasattr(self, iterators_attr):
            iterators = getattr(self, iterators_attr)
            log_from_all_processes(logger, 'debug', f"Source names: {self.source_names}")

            # detect which sources need hanja splitting (denoising only)
            korean_sources = []
            direct_sources = []
            # use correct attribute key with suffix (_train or _eval)
            attr_suffix = 'eval' if use_eval_iterators else 'train'
            for source in self.source_names:
                attr_key = f'{source}_{attr_suffix}'
                dtype_attr = f'_data_type_{attr_key}'
                data_type = getattr(self, dtype_attr).lower()
                # split denoising/mixed data into hanja_heavy/hanja_light sub-buffers;
                # supervised tasks (sts, transcription, temporal, instruction) use direct sampling
                if data_type in ('denoising', 'mixed'):
                    korean_sources.append(source)
                else:
                    direct_sources.append(source)

            log_from_all_processes(logger, 'debug', f"Sources for hanja splitting (denoising/mixed): {korean_sources}")
            log_from_all_processes(logger, 'debug', f"Direct sampling sources: {direct_sources}")
            log_from_all_processes(logger, 'debug', f"Buffer size: {self.buffer_size}, Num sources: {len(self.source_names)}")

            # calculate target buffer sizes from sampling_ratios
            korean_heavy_target = int(self.buffer_size * self.sampling_ratios.get('korean_hanja_heavy', 0))
            korean_light_target = int(self.buffer_size * self.sampling_ratios.get('korean_hanja_light', 0))

            # fill korean sub-buffers from all korean denoising sources
            # IMPORTANT: round-robin through sources to use ALL korean datasets evenly
            # source_idx persisted across calls so refills cycle through sources
            # (previous bug: local source_idx=0 always drew from first source)
            korean_total_target = korean_heavy_target + korean_light_target
            active_korean_sources = list(korean_sources)
            if not hasattr(self, '_korean_source_idx'):
                self._korean_source_idx = 0

            while active_korean_sources:
                total_korean_buffer = (
                    len(korean_sub_buffers['korean_hanja_heavy']) +
                    len(korean_sub_buffers['korean_hanja_light'])
                )
                if total_korean_buffer >= korean_total_target:
                    break

                source = active_korean_sources[self._korean_source_idx % len(active_korean_sources)]
                try:
                    sample = next(iterators[source])
                    self._samples_yielded[source] = self._samples_yielded.get(source, 0) + 1

                    # classify korean sample based on hanja concentration
                    sub_source = self._classify_korean_sample(sample)
                    sample['_original_source'] = source
                    sample['_sub_source'] = sub_source

                    # route to appropriate sub-buffer
                    korean_sub_buffers[sub_source].append(sample)
                    self._korean_source_idx += 1

                except StopIteration:
                    log_from_all_processes(logger, 'info', f"Iterator for {source} exhausted, recreating")
                    backup_attr = f'_backup_{source}{backup_suffix}'
                    if hasattr(self, backup_attr):
                        iterators[source] = iter(getattr(self, backup_attr))
                    else:
                        log_from_all_processes(logger, 'error', f"No backup for {source}, removing from rotation")
                        active_korean_sources.remove(source)
                    self._korean_source_idx += 1

            # fill direct-sampling buffers (supervised tasks) using sampling_ratios
            for source in direct_sources:
                target_size = int(self.buffer_size * self.sampling_ratios.get(source, 0))
                while len(sample_buffer[source]) < target_size:
                    try:
                        sample = next(iterators[source])
                        self._samples_yielded[source] = self._samples_yielded.get(source, 0) + 1
                        sample_buffer[source].append(sample)
                    except StopIteration:
                        log_from_all_processes(logger, 'error', f"Iterator for {source} exhausted!")
                        backup_attr = f'_backup_{source}{backup_suffix}'
                        if hasattr(self, backup_attr):
                            iterators[source] = iter(getattr(self, backup_attr))
                        break

        # calculate current token ratios using sampling ratios keys
        all_sources = list(self.sampling_ratios.keys())
        total_tokens = sum(self.token_counts.values()) or 1

        current_ratios = {}
        for source in all_sources:
            current_ratios[source] = self.token_counts.get(source, 0) / total_tokens

        # find source that is furthest below its target ratio
        max_deficit = -float('inf')
        selected_source = None

        for source in all_sources:
            # check buffer availability
            if source.startswith('korean_hanja_'):
                buffer_len = len(korean_sub_buffers[source])
            else:
                buffer_len = len(sample_buffer.get(source, []))

            if buffer_len > 0:
                deficit = self.sampling_ratios[source] - current_ratios[source]
                if deficit > max_deficit:
                    max_deficit = deficit
                    selected_source = source

        # fallback to random selection if all are balanced or no clear choice
        if selected_source is None or max_deficit < 0.01:
            # weighted random selection based on target ratios
            available_sources = []
            weights = []

            for source in all_sources:
                if source.startswith('korean_hanja_'):
                    if len(korean_sub_buffers[source]) > 0:
                        available_sources.append(source)
                        weights.append(self.sampling_ratios[source])
                else:
                    if source in sample_buffer and len(sample_buffer[source]) > 0:
                        available_sources.append(source)
                        weights.append(self.sampling_ratios[source])

            if available_sources:
                weights = np.array(weights) / np.sum(weights)
                selected_source = self.rng.choice(available_sources, p=weights)
                log_from_all_processes(logger, 'debug', f"Randomly selected source: {selected_source} (from {len(available_sources)} available)")
            else:
                # no samples available - this is a real error!
                buffer_status = {
                    'korean_hanja_heavy': len(korean_sub_buffers['korean_hanja_heavy']),
                    'korean_hanja_light': len(korean_sub_buffers['korean_hanja_light']),
                }
                for source in sample_buffer:
                    buffer_status[source] = len(sample_buffer[source])

                error_msg = (
                    f"No samples available in any buffer! "
                    f"Buffer status: {buffer_status}\n"
                    f"Has train_iterators: {hasattr(self, 'train_iterators')}\n"
                    f"Source names: {getattr(self, 'source_names', 'NOT SET')}"
                )
                log_from_all_processes(logger, 'error', f"[BUFFER ERROR] {error_msg}")
                raise RuntimeError(error_msg)

        # get sample from selected source
        if selected_source:
            if selected_source.startswith('korean_hanja_'):
                # korean sub-source
                sample = korean_sub_buffers[selected_source].pop(0)

                # extract tracking fields (keep _data_source for per-task loss tracking)
                _ = sample.pop('_original_source', 'han2han_curated')
                sample.pop('_sub_source', None)
                sample.pop('_source', None)
                sample.pop('_source_name', None)

                # add data source tag for per-task loss tracking
                sample['_data_source'] = selected_source

                # update counters for the sub-source
                tokens = self.count_tokens_in_sample(sample)
                self.token_counts[selected_source] += tokens
                self.sample_counts[selected_source] += 1

            else:
                # non-korean source
                sample = sample_buffer[selected_source].pop(0)

                # extract source tracking fields (keep _data_source for per-task loss tracking)
                actual_source = sample.pop('_source', selected_source)
                sample.pop('_source_name', None)

                # add data source tag for per-task loss tracking
                sample['_data_source'] = actual_source

                # update counters
                tokens = self.count_tokens_in_sample(sample)
                self.token_counts[actual_source] += tokens
                self.sample_counts[actual_source] += 1

            # log statistics periodically with more detail
            total_samples = sum(self.sample_counts.values())
            if total_samples % 1000000 == 0 and total_samples > 0:  # log every 1M samples instead of 10M
                total_tokens = sum(self.token_counts.values())
                log_from_main_process(logger, 'info', f"=== Token distribution after {total_samples:,} samples ({total_tokens:,} tokens) ===")

                # log korean sub-sources
                log_from_main_process(logger, 'info', f"Korean sub-sources:")
                for sub_source in ['korean_hanja_heavy', 'korean_hanja_light']:
                    actual_ratio = current_ratios.get(sub_source, 0.0)
                    target_ratio = self.sampling_ratios.get(sub_source, 0.0)
                    buffer_size = len(korean_sub_buffers[sub_source])
                    tokens = self.token_counts.get(sub_source, 0)
                    samples = self.sample_counts.get(sub_source, 0)
                    log_from_main_process(logger, 'info',
                        f"  {sub_source}: {actual_ratio:.3f} (target: {target_ratio:.3f}), "
                        f"tokens: {tokens:,}, samples: {samples:,}, buffer: {buffer_size}")

                # log other sources (all non-korean sub-sources)
                log_from_main_process(logger, 'info', f"Other sources:")
                korean_sub_sources = {'korean_hanja_heavy', 'korean_hanja_light'}
                for source in sorted(all_sources):
                    if source not in korean_sub_sources:
                        actual_ratio = current_ratios.get(source, 0.0)
                        target_ratio = self.sampling_ratios.get(source, 0.0)
                        buffer_size = len(sample_buffer.get(source, []))
                        tokens = self.token_counts.get(source, 0)
                        samples = self.sample_counts.get(source, 0)
                        log_from_main_process(logger, 'info',
                            f"  {source}: {actual_ratio:.3f} (target: {target_ratio:.3f}), "
                            f"tokens: {tokens:,}, samples: {samples:,}, buffer: {buffer_size}")

            return sample
        else:
            error_msg = (
                f"selected_source is None after all selection logic! "
                f"This should never happen. Debug info:\n"
                f"Has train_iterators: {hasattr(self, 'train_iterators')}\n"
                f"All sources: {all_sources}\n"
                f"Max deficit: {max_deficit}\n"
                f"Buffer status: check logs above"
            )
            log_from_all_processes(logger, 'error', f"[SAMPLING ERROR] {error_msg}")
            raise RuntimeError(error_msg)

    def reset_token_counts(self):
        """Reset token and sample counters."""
        for source in self.sampling_ratios:
            self.token_counts[source] = 0
            self.sample_counts[source] = 0
        log_from_main_process(logger, 'info', "Reset token and sample counters")

    def get_token_statistics(self) -> Dict[str, Any]:
        """Get current token distribution statistics."""
        total_tokens = sum(self.token_counts.values()) or 1
        total_samples = sum(self.sample_counts.values()) or 1

        stats = {
            'total_tokens': total_tokens,
            'total_samples': total_samples,
            'token_ratios': {},
            'sample_ratios': {},
            'tokens_per_source': {},
            'samples_per_source': {},
            'korean_sub_sources': {},
        }

        # include all sources from sampling_ratios (including korean sub-sources)
        for source in self.sampling_ratios:
            tokens = self.token_counts.get(source, 0)
            samples = self.sample_counts.get(source, 0)
            stats['token_ratios'][source] = tokens / total_tokens
            stats['sample_ratios'][source] = samples / total_samples
            stats['tokens_per_source'][source] = tokens
            stats['samples_per_source'][source] = samples

            # track korean sub-sources specifically
            if source.startswith('korean_hanja_'):
                buffer_size = len(self.korean_sub_buffers.get(source, []))
                stats['korean_sub_sources'][source] = {
                    'tokens': tokens,
                    'samples': samples,
                    'ratio': tokens / total_tokens,
                    'buffer_size': buffer_size,
                }

        return stats
    
    def split_sentences(self, text: str) -> List[str]:
        """Simple sentence splitting for multilingual web text."""

        sentences = text.replace('! ', '!<SPLIT>').replace('? ', '?<SPLIT>').replace(
            '… ', '…<SPLIT>').replace('…', '…<SPLIT>').replace('... ', '...<SPLIT>').replace('...', '...<SPLIT>').replace(
            '、、、、、', '、、、、、<SPLIT>').replace('・・・', '・・・<SPLIT>').replace(
            '. ', '.<SPLIT>').replace('.', '.<SPLIT>').replace('。', '。<SPLIT>').replace('。 ', '。<SPLIT>').replace(
            '！ ', '！<SPLIT>').replace('？', '？<SPLIT>').replace('\n', '<SPLIT>').split('<SPLIT>')

        return [s.strip() for s in sentences if s.strip()]

    def __call__(self, example: Dict[str, Any]|List[Dict[str, Any]], cooldown_phase=None, 
                 bucket_idx=None, tokenizer=None, return_source=False, padding=True):
        """Collates single example into mBART format: no BOS tokens, sentence boundaries with `</s>`, script tokens.

        Args:
            example: Single example dict or list of examples
            cooldown_phase: Whether in cooldown phase (no metadata)
            bucket_idx: Bucket index for bucketing
            tokenizer: Tokenizer to use (defaults to self.tokenizer)
            return_source: If True, returns tuple (batch_dict, source) for compatibility with iterators
            padding: Whether to pad the inputs to the configured max_length
        """

        if isinstance(example, list):   # batch of examples passed in, call and stack
            if return_source:
                raise ValueError("return_source=True not supported for batch processing")
            batch = [self(ex, cooldown_phase=cooldown_phase, bucket_idx=bucket_idx, tokenizer=tokenizer) for ex in example]
            batch_keys = batch[0].keys()
            batch = {k: np.stack([b[k] for b in batch]) for k in batch_keys}
            return BatchEncoding(batch).data

        if tokenizer is None:
            tokenizer = self.tokenizer

        cur_max_length = self.max_length
        if self.use_bucketing and bucket_idx is not None and bucket_idx < len(self.bucket_sizes):
            cur_max_length = min(self.bucket_sizes[bucket_idx], self.model_max_length)

        pad_fn = partial(self._pad_single_sample, max_length=cur_max_length, pad_token=tokenizer.pad_token_id)
        if not padding:
            pad_fn = lambda x: x

        if hasattr(self, "cooldown_phase") and cooldown_phase is None:
            cooldown_phase = self.cooldown_phase

        # get required fields
        if 'sentences' in example and example['sentences']:
            sentences = example['sentences']
        elif 'original_text' in example:
            sentences = self.split_sentences(example["original_text"])
        else:
            raise ValueError(f"Data does not contain expected keys: "
                                f"Got {example.keys()}, need either 'sentences' or 'original_text'")

        # ensure sentences is a list
        if isinstance(sentences, str):
            sentences = [sentences]

        # handle script detection and transcription
        encoder_sentences = sentences
        decoder_sentences = sentences

        encoder_script_token_id = None
        decoder_script_token_id = None

        # check if any sentence has Hanja
        full_text = ' '.join(sentences)
        has_hanja_text = has_hanja(full_text)

        if has_hanja_text:
            plan = example.get('_corruption_plan')
            if plan is not None:
                transcription_direction = plan.get('transcription')
            else:
                apply_transcription = (
                    self.han2han_transcription_ratio is not None and
                    self.han2han_transcription_ratio > 0 and
                    np.random.random() < self.han2han_transcription_ratio
                )
                if apply_transcription:
                    transcription_direction = (
                        'hangul_to_hanja' if np.random.random() < 0.5 else 'hanja_to_hangul'
                    )
                else:
                    transcription_direction = None

            if transcription_direction == 'hangul_to_hanja':
                encoder_sentences = [transcribe(s) for s in sentences]
                encoder_script_token_id = tokenizer.convert_tokens_to_ids('<hangul>')
                decoder_script_token_id = tokenizer.convert_tokens_to_ids('<hanja>')
                if plan is None and '_training_mode' in example:
                    example['_training_mode'] = example['_training_mode'] + '_transcription_hangul_to_hanja'
            elif transcription_direction == 'hanja_to_hangul':
                decoder_sentences = [transcribe(s) for s in sentences]
                encoder_script_token_id = tokenizer.convert_tokens_to_ids('<hanja>')
                decoder_script_token_id = tokenizer.convert_tokens_to_ids('<hangul>')
                if plan is None and '_training_mode' in example:
                    example['_training_mode'] = example['_training_mode'] + '_transcription_hanja_to_hangul'
            else:
                encoder_script_token_id = tokenizer.convert_tokens_to_ids('<hanja>')
                decoder_script_token_id = tokenizer.convert_tokens_to_ids('<hanja>')
        else:
            # pure Hangul text on both sides
            encoder_script_token_id = tokenizer.convert_tokens_to_ids('<hangul>')
            decoder_script_token_id = tokenizer.convert_tokens_to_ids('<hangul>')

        # step 1: tokenize encoder sentences (potentially transcribed) and track boundaries
        encoder_content_ids = []
        encoder_eos_positions = []  # track sentence boundaries
        for sentence in encoder_sentences:
            sentence_ids = tokenizer(sentence, add_special_tokens=False).input_ids
            encoder_content_ids.extend(sentence_ids)
            encoder_content_ids.append(tokenizer.eos_token_id)
            encoder_eos_positions.append(len(encoder_content_ids) - 1)  # position of EOS

        # step 1b: tokenize decoder sentences (potentially transcribed) and track boundaries
        clean_content_ids = []
        decoder_eos_positions = []  # track sentence boundaries
        for sentence in decoder_sentences:
            sentence_ids = tokenizer(sentence, add_special_tokens=False).input_ids
            clean_content_ids.extend(sentence_ids)
            clean_content_ids.append(tokenizer.eos_token_id)
            decoder_eos_positions.append(len(clean_content_ids) - 1)  # position of EOS

        # step 2: calculate effective content window size
        metadata = example['metadata'] + " "    # shouldn't make a difference but separate from body
        metadata_ids = tokenizer(metadata, add_special_tokens=False).input_ids if not cooldown_phase else []
        # account for both metadata tokens and script token
        effective_window = cur_max_length - len(metadata_ids) - 1

        # step 3: apply sentence-aligned sliding window if needed
        if len(encoder_content_ids) > effective_window or len(clean_content_ids) > effective_window:
            # use sentence boundaries for alignment
            num_sentences = len(encoder_eos_positions)
            if num_sentences > 1:
                # randomly select starting sentence
                start_sentence_idx = self.rng.integers(0, num_sentences)

                # find token positions for this sentence
                if start_sentence_idx == 0:
                    encoder_start = 0
                    decoder_start = 0
                else:
                    encoder_start = encoder_eos_positions[start_sentence_idx - 1] + 1
                    decoder_start = decoder_eos_positions[start_sentence_idx - 1] + 1

                # slice from start sentence and truncate to effective_window
                windowed_encoder_ids = encoder_content_ids[encoder_start:encoder_start + effective_window]
                windowed_content_ids = clean_content_ids[decoder_start:decoder_start + effective_window]
            else:
                # single sentence or fallback: simple truncation
                windowed_encoder_ids = encoder_content_ids[:effective_window]
                windowed_content_ids = clean_content_ids[:effective_window]
        else:
            windowed_encoder_ids = encoder_content_ids
            windowed_content_ids = clean_content_ids

        # step 4: decide masking style BEFORE permutation
        # sentinel masking is incompatible with sentence permutation because sentinels
        # mark positions in the sequence, but permutation scrambles those positions
        plan = example.get('_corruption_plan')
        if plan is not None:
            use_sentinel = plan.get('form') == 'sentinel'
        else:
            use_sentinel = (hasattr(self, 'sentinel_denoising_ratio') and
                           self.rng.random() < self.sentinel_denoising_ratio)
            if '_training_mode' in example:
                masking_suffix = '_sentinel' if use_sentinel else '_bart'
                example['_training_mode'] = example['_training_mode'] + masking_suffix

        # step 5: permute using eos token boundaries (encoder only, BART-style only)
        # only permute when NOT using sentinels - sentinel positions would be meaningless after permutation
        if self.sentence_permutation and not use_sentinel:
            permuted_encoder_ids = self.permute_sentences_by_eos_tokens(
                np.array(windowed_encoder_ids), tokenizer.eos_token_id).tolist()
        else:
            permuted_encoder_ids = windowed_encoder_ids

        # step 6: apply span masking to permuted encoder content
        if not isinstance(permuted_encoder_ids, np.ndarray):
            permuted_encoder_ids = np.array(permuted_encoder_ids)

        if use_sentinel:
            # strip </s> tokens for sentinel mode - sentence boundaries don't make sense with sentinels
            # (</s> is used for sentence-aligned truncation but should be removed before sentinel masking)
            permuted_encoder_ids = permuted_encoder_ids[permuted_encoder_ids != tokenizer.eos_token_id]
            # T5-style: unique sentinels, decoder only outputs masked spans
            content_ids_corrupted, decoder_span_ids, _ = self._token_based_sentinel_masking(
                permuted_encoder_ids, tokenizer=tokenizer
            )
            if isinstance(content_ids_corrupted, np.ndarray):
                content_ids_corrupted = content_ids_corrupted.tolist()
            if isinstance(decoder_span_ids, np.ndarray):
                decoder_span_ids = decoder_span_ids.tolist()
            # for sentinel mode, decoder content is the spans, not full text
            decoder_content_ids = decoder_span_ids
        else:
            # BART-style: single <mask> token, decoder outputs full text
            content_ids_corrupted = self._token_based_infilling(permuted_encoder_ids, tokenizer=tokenizer)
            if isinstance(content_ids_corrupted, np.ndarray):
                content_ids_corrupted = content_ids_corrupted.tolist()
            # for BART mode, decoder content is the full windowed text
            decoder_content_ids = windowed_content_ids

        # step 6: build sequences with separate encoder/decoder script tokens
        # encoder: metadata + corrupted_content + script_token + [encoder_script_token]
        # decoder input: script_token + [decoder_script_token] + decoder_content
        # labels: decoder_content + script_token + [decoder_script_token]

        # build encoder end tokens list
        encoder_end_tokens = [encoder_script_token_id]

        # build decoder start tokens list (for decoder input)
        decoder_start_tokens = [decoder_script_token_id]

        # build decoder end tokens list (for labels)
        decoder_end_tokens = [decoder_script_token_id]

        # build final sequences using decoder_content_ids (may be full text for BART or spans for sentinel)
        if not cooldown_phase:
            encoder_ids = pad_fn(metadata_ids + content_ids_corrupted + encoder_end_tokens)
            decoder_ids = pad_fn(decoder_start_tokens + decoder_content_ids)
            labels = pad_fn(decoder_content_ids + decoder_end_tokens)
            labels_masked = np.where(labels == self.tokenizer.pad_token_id, -100, labels)
        else:
            encoder_ids = pad_fn(content_ids_corrupted + encoder_end_tokens)
            decoder_ids = pad_fn(decoder_start_tokens + decoder_content_ids)
            labels = pad_fn(decoder_content_ids + decoder_end_tokens)
            labels_masked = np.where(labels == self.tokenizer.pad_token_id, -100, labels)

        batch_dict = {
            "input_ids": encoder_ids,
            "decoder_input_ids": decoder_ids,
            "labels": labels_masked,
            "attention_mask": np.where(encoder_ids != tokenizer.pad_token_id, 1, 0).astype(np.int32),
            "decoder_attention_mask": np.where(decoder_ids != tokenizer.pad_token_id, 1, 0).astype(np.int32),
        }

        # propagate _training_mode from example if present
        if '_training_mode' in example:
            batch_dict['_training_mode'] = example['_training_mode']

        if return_source:
            return batch_dict, example['source']
        else:
            return BatchEncoding(batch_dict).data

    def _pad_single_sample(
        self,
        token_ids: List[int],
        max_length: int,
        pad_token: int,
    ) -> np.ndarray:
        """Pads single sample and returns as NumPy array."""
        if len(token_ids) < max_length:
            token_ids += [pad_token] * (max_length - len(token_ids))
        return np.array(token_ids[:max_length]).astype(np.int32)


def create_dummy_collator(args, tokenizer):
    """Create dummy collator for smoke testing with direct iteration support."""
    log_from_main_process(logger, 'info', "Creating dummy collator for smoke test mode")

    # prepare sample texts
    sample_texts = [
        "한국의 역사는 매우 깊고 풍부하다.",
        "조선시대에 훈민정음이 창제되었다.", 
        "현대 한국 문화는 전통과 현대가 조화를 이룬다.",
    ]

    # prepare metadata samples
    metadata_samples = [
        "연도: 1443 시기: 조선전기 유형: 문헌",
        "연도: 1900 시기: 근대 유형: 신문",
        "연도: 2020 시기: 현대 유형: 웹문서",
        "연도: 1960 시기: 근현대 유형: 학술논문",
    ]

    # script tokens
    script_tokens = ["<hangul>", "<hanja>"]

    # dummy collator class with direct iteration support
    class DummyCollator:
        def __init__(self):
            self.batch_count = 0
            self.cooldown_phase = False

        def _instantiate_dsets(self, cooldown_phase=False):
            """Update cooldown phase for dummy collator."""
            self.cooldown_phase = cooldown_phase
            log_from_main_process(logger, 'info', f"Dummy collator: cooldown phase = {cooldown_phase}")

        def create_dummy_batch(self):
            """Create a single dummy batch matching MultilingualCollator format."""
            batch_size = args.batch_size
            seq_len = args.sequence_length

            input_ids = []
            decoder_input_ids = []
            labels = []
            attention_mask = []
            decoder_attention_mask = []

            for i in range(batch_size):
                # choose random text and metadata
                text = sample_texts[i % len(sample_texts)]
                metadata = metadata_samples[i % len(metadata_samples)]
                script_token = script_tokens[i % len(script_tokens)]

                # simulate span masking: randomly replace some tokens with <mask>
                words = text.split()
                if len(words) > 2 and tokenizer.mask_token:
                    mask_idx = i % len(words)
                    words[mask_idx] = tokenizer.mask_token
                corrupted_text = ' '.join(words)

                # tokenize components
                clean_ids = tokenizer(text, add_special_tokens=False).input_ids
                corrupted_ids = tokenizer(corrupted_text, add_special_tokens=False).input_ids
                metadata_ids = tokenizer(metadata, add_special_tokens=False).input_ids
                script_token_id = tokenizer.convert_tokens_to_ids(script_token)

                # build sequences - adjust for cooldown phase
                if not self.cooldown_phase:
                    # training phase: metadata + corrupted_content + script_token 
                    encoder_ids = metadata_ids + corrupted_ids + [script_token_id]
                else:
                    # cooldown phase: corrupted_content + script_token (no metadata)
                    encoder_ids = corrupted_ids + [script_token_id]

                # decoder input: script_token + clean_content
                decoder_ids = [script_token_id] + clean_ids

                # labels: clean_content + script_token
                label_ids = clean_ids + [script_token_id]

                # pad or truncate to sequence length
                def pad_sequence(seq, max_len, pad_id):
                    if len(seq) > max_len:
                        return seq[:max_len]
                    else:
                        return seq + [pad_id] * (max_len - len(seq))

                encoder_padded = pad_sequence(encoder_ids, seq_len, tokenizer.pad_token_id)
                decoder_padded = pad_sequence(decoder_ids, seq_len, tokenizer.pad_token_id)
                labels_padded = pad_sequence(label_ids, seq_len, tokenizer.pad_token_id)

                # create attention masks
                encoder_mask = [1 if token_id != tokenizer.pad_token_id else 0 for token_id in encoder_padded]
                decoder_mask = [1 if token_id != tokenizer.pad_token_id else 0 for token_id in decoder_padded]

                # mask padding tokens in labels (-100)
                labels_masked = [-100 if token_id == tokenizer.pad_token_id else token_id for token_id in labels_padded]

                input_ids.append(encoder_padded)
                decoder_input_ids.append(decoder_padded)
                labels.append(labels_masked)
                attention_mask.append(encoder_mask)
                decoder_attention_mask.append(decoder_mask)

            # convert to numpy arrays
            batch = {
                "input_ids": np.array(input_ids, dtype=np.int32),
                "decoder_input_ids": np.array(decoder_input_ids, dtype=np.int32),
                "labels": np.array(labels, dtype=np.int32),
                "attention_mask": np.array(attention_mask, dtype=np.int32),
                "decoder_attention_mask": np.array(decoder_attention_mask, dtype=np.int32)
            }

            return batch

        def __iter__(self):
            """Make dummy collator iterable for direct iteration."""
            return self

        def __next__(self):
            """Generate next dummy example (unbatched, 5 keys only)."""
            # create a single example instead of a batch
            sample_idx = self.batch_count % len(sample_texts)
            text = sample_texts[sample_idx]
            metadata = metadata_samples[sample_idx % len(metadata_samples)]
            script_token = script_tokens[sample_idx % len(script_tokens)]

            # simulate span masking: randomly replace some tokens with <mask>
            words = text.split()
            if len(words) > 2 and tokenizer.mask_token:
                mask_idx = sample_idx % len(words)
                words[mask_idx] = tokenizer.mask_token
            corrupted_text = ' '.join(words)

            # tokenize components
            clean_ids = tokenizer(text, add_special_tokens=False).input_ids
            corrupted_ids = tokenizer(corrupted_text, add_special_tokens=False).input_ids
            metadata_ids = tokenizer(metadata, add_special_tokens=False).input_ids
            script_token_id = tokenizer.convert_tokens_to_ids(script_token)
            
            # build sequences - adjust for cooldown phase
            if not self.cooldown_phase:
                # training phase: metadata + corrupted_content + script_token 
                encoder_ids = metadata_ids + corrupted_ids + [script_token_id]
            else:
                # cooldown phase: corrupted_content + script_token (no metadata)
                encoder_ids = corrupted_ids + [script_token_id]

            # decoder input: script_token + clean_content
            decoder_ids = [script_token_id] + clean_ids
            # labels: clean_content + script_token  
            label_ids = clean_ids + [script_token_id]

            # pad or truncate to sequence length
            def pad_sequence(seq, max_len, pad_id):
                if len(seq) > max_len:
                    return seq[:max_len]
                else:
                    return seq + [pad_id] * (max_len - len(seq))

            seq_len = args.sequence_length
            encoder_padded = pad_sequence(encoder_ids, seq_len, tokenizer.pad_token_id)
            decoder_padded = pad_sequence(decoder_ids, seq_len, tokenizer.pad_token_id)
            labels_padded = pad_sequence(label_ids, seq_len, tokenizer.pad_token_id)

            # create attention masks
            encoder_mask = [1 if token_id != tokenizer.pad_token_id else 0 for token_id in encoder_padded]
            decoder_mask = [1 if token_id != tokenizer.pad_token_id else 0 for token_id in decoder_padded]

            # mask padding tokens in labels (-100)
            labels_masked = [-100 if token_id == tokenizer.pad_token_id else token_id for token_id in labels_padded]

            # five model-facing arrays plus the _data_source / _training_mode tags
            # that create_batches pops for per-source / per-task loss tracking
            example = {
                "input_ids": np.array(encoder_padded, dtype=np.int32),
                "decoder_input_ids": np.array(decoder_padded, dtype=np.int32),
                "labels": np.array(labels_masked, dtype=np.int32),
                "attention_mask": np.array(encoder_mask, dtype=np.int32),
                "decoder_attention_mask": np.array(decoder_mask, dtype=np.int32),
                "_data_source": "smoke",
                "_training_mode": "denoising",
            }

            self.batch_count += 1

            return example

    dummy_collator = DummyCollator()
    # create attributes matching the real collator interface
    dummy_collator.tokenizer = tokenizer
    dummy_collator.sampled_datasets = dummy_collator
    dummy_collator.eval_data = dummy_collator  # dummy eval data is same as train
    dummy_collator.create_eval_batches = lambda max_batches=100: [dummy_collator.create_dummy_batch() for _ in range(min(max_batches, 10))]

    return dummy_collator