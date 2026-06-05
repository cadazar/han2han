#!/usr/bin/env python3
# coding: utf-8
"""
Packed Multilingual Han2Han Collator - 100% packing efficiency through generator-level buffering.

Key architectural change: Move buffering from collator into generator itself.
This ensures ALL yielded examples are packed sequences with segment_ids, not standalone examples.
"""

import logging
from transformers import AutoTokenizer
from datasets import IterableDataset

from packed_phase2_collator import PackedPhase2Collator
from logging_utils import log_from_main_process

logger = logging.getLogger(__name__)


class PackedMultilingualCollator(PackedPhase2Collator):
    """
    Multilingual Han2Han collator with document packing at generator level.

    Architecture:
    - Inherits from PackedPhase2Collator for pack_documents() logic
    - Overrides _instantiate_dsets() to add per-source buffering in generators
    - Each source has its own buffer and packed cache
    - Only yields when packed sequences are ready from cache
    - Parent's get_next_balanced_sample() handles source selection ratios

    Result: 100% packing efficiency, all examples come from packed sequences
    """

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
        """
        Create dataset iterators with generator-level packing.

        Overrides parent to add per-source buffering and packing inside generators.
        Each source buffers 64 examples, packs them, and only yields packed sequences.

        See MultilingualCollator._instantiate_dsets for the contract on
        ``new_source_configs`` / ``new_sampling_ratios`` during phase transitions.
        """
        # if packing is disabled, use parent's simpler implementation
        # use getattr with default True because enable_packing may not be set yet
        # during __post_init__ (it's set after super().__init__() in PackedPhase2Collator)
        if not getattr(self, 'enable_packing', True):
            log_from_main_process(logger, 'info',
                "Packing disabled - using parent MultilingualCollator._instantiate_dsets()")
            from multilingual_collator import MultilingualCollator
            return MultilingualCollator._instantiate_dsets(
                self, cooldown_phase, new_datasets,
                new_source_configs=new_source_configs,
                new_sampling_ratios=new_sampling_ratios,
            )

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
            log_from_main_process(logger, 'info', f"Creating packed generator for {dn} dataset...")

            # grab custom info to avoid errors in split
            source_type = ds.info.__dict__.pop('source_type')
            data_type = ds.info.__dict__.pop('data_type')

            # check if this source has pre-defined stratified splits
            source_config = self.source_configs.get(dn)
            has_stratified = source_config and getattr(source_config, 'has_stratified_split', False)
            eval_key = f"{dn}_eval"
            has_preloaded_eval = eval_key in getattr(self, 'eval_datasets', {})

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
            if data_type is not None:
                trainset.info.data_type = data_type
            evalset.info.source_type = source_type
            if data_type is not None:
                evalset.info.data_type = data_type

            # create infinite packed iterator that buffers and packs before yielding
            def create_infinite_packed_iterator(dataset, source_name, is_train=True):
                # capture dataset info and custom attributes before iterating
                source_type = dataset.info.source_type
                data_type = dataset.info.data_type

                # use distinct suffix for train vs eval to avoid attribute overwrite
                attr_suffix = 'train' if is_train else 'eval'
                attr_key = f'{source_name}_{attr_suffix}'

                # store params as instance variables to avoid closure/pickle issues
                setattr(self, f'_dataset_{attr_key}', dataset)
                setattr(self, f'_source_type_{attr_key}', source_type)
                setattr(self, f'_data_type_{attr_key}', data_type)
                setattr(self, f'_is_train_{attr_key}', is_train)
                setattr(self, f'_cooldown_{attr_key}', cooldown_phase)

                # create a simple generator function that references self
                def make_packed_generator(collator, dataset_name, attr_key):
                    def infinite_packed_generator():
                        # retrieve stored params using the unique attr_key
                        dataset_obj = getattr(collator, f'_dataset_{attr_key}')
                        source_type_str = getattr(collator, f'_source_type_{attr_key}')
                        data_type_str = getattr(collator, f'_data_type_{attr_key}')
                        is_train_flag = getattr(collator, f'_is_train_{attr_key}')
                        cooldown = getattr(collator, f'_cooldown_{attr_key}')

                        # create a mock dataset object for _transform_example compatibility
                        class DatasetMock:
                            def __init__(self, source_type, data_type=None):
                                class InfoMock:
                                    pass
                                self.info = InfoMock()
                                self.info.source_type = source_type
                                if data_type is not None:
                                    self.info.data_type = data_type

                        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer.name_or_path)

                        # create morphological tokenizer (thread-safe, one per dataset)
                        from konlpy.tag import Mecab
                        morpheme_tokenizers = {'ko': Mecab()}

                        mock_dataset = DatasetMock(source_type_str, data_type_str)
                        epoch_count = 0
                        example_count = 0

                        # per-mode buffers for packing (keyed by training_mode)
                        # this ensures only same-mode examples get packed together
                        mode_buffers = {}  # {mode: [tokenized_examples]}
                        mode_packed_caches = {}  # {mode: [packed_sequences]}

                        log_from_main_process(logger, 'info',
                            f"Packed generator for {dataset_name}: packed_buffer_size={self.packed_buffer_size}, packing_enabled={collator.enable_packing}, per-mode buffering enabled")

                        while True:
                            epoch_count += 1
                            if epoch_count % 10 == 1:
                                log_from_main_process(logger, 'info',
                                    f"Dataset {dataset_name} ({'train' if is_train_flag else 'eval'}): "
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

                                # yield from all mode caches first (round-robin through modes)
                                for mode in list(mode_packed_caches.keys()):
                                    while len(mode_packed_caches[mode]) > 0:
                                        packed_example = mode_packed_caches[mode].pop(0)
                                        packed_example['_source'] = dataset_name
                                        packed_example['_source_name'] = dataset_name
                                        packed_example['_training_mode'] = mode
                                        yield packed_example

                                # transform and tokenize WITHOUT padding
                                transformed_example = collator._transform_example(raw_example, mock_dataset)

                                # skip if transformation returned None (null/empty required fields)
                                if transformed_example is None:
                                    continue

                                # pre-roll all stochastic corruption decisions and resolve the
                                # composite-key task prompt here, so the collator only performs
                                # the selected corruption form rather than re-sampling per branch.
                                # supervised tasks return None from _sample_corruption_plan and
                                # fall through to the legacy collator path.
                                if hasattr(collator, '_sample_corruption_plan'):
                                    plan = collator._sample_corruption_plan(
                                        transformed_example, cooldown_phase=cooldown
                                    )
                                    if plan is not None:
                                        transformed_example['_corruption_plan'] = plan
                                        transformed_example['_training_mode'] = plan['training_mode']
                                        if plan.get('prompt') is not None:
                                            transformed_example['metadata'] = plan['prompt']

                                tokenized_example = collator(
                                    transformed_example,
                                    cooldown_phase=cooldown,
                                    return_source=False,
                                    padding=False,  # no padding yet - we need actual lengths for packing
                                    tokenizer=tokenizer,
                                    morpheme_tokenizers=morpheme_tokenizers
                                )

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

                                # get training mode from tokenized example (set by collator during __call__)
                                training_mode = tokenized_example.get('_training_mode')
                                if training_mode is None:
                                    raise ValueError(
                                        f"Missing '_training_mode' in tokenized example after collation. "
                                        f"This indicates a bug in the collator - all paths should set _training_mode. "
                                        f"Example keys: {list(tokenized_example.keys())}, "
                                        f"source: {dataset_name}"
                                    )

                                # initialize buffer for this mode if not exists
                                if training_mode not in mode_buffers:
                                    mode_buffers[training_mode] = []
                                    mode_packed_caches[training_mode] = []

                                # route to mode-specific buffer
                                mode_buffers[training_mode].append(tokenized_example)

                                # when any mode buffer is full, pack and add to its cache
                                if len(mode_buffers[training_mode]) >= self.packed_buffer_size:
                                    packed_sequences, buffer_leftovers = collator.pack_documents_with_leftovers(mode_buffers[training_mode])
                                    mode_packed_caches[training_mode].extend(packed_sequences)
                                    mode_buffers[training_mode] = buffer_leftovers

                    return infinite_packed_generator

                generator_fn = make_packed_generator(self, source_name, attr_key)
                return IterableDataset.from_generator(generator_fn)

            # create infinite packed datasets
            infinite_ds_train = create_infinite_packed_iterator(trainset, dn, is_train=True)
            infinite_ds_eval = create_infinite_packed_iterator(evalset, dn, is_train=False)
            tokenized_datasets_train.append(infinite_ds_train)
            tokenized_datasets_eval.append(infinite_ds_eval)
            source_names.append(dn)
            # add info back to source ds for subsequent calls to this function
            ds.info.source_type = source_type
            if data_type is not None:
                ds.info.data_type = data_type

        log_from_main_process(logger, 'info',
            "Packed generator creation complete! Configuring data iterators with set probabilities...")

        # store dataset iterators for token-based sampling
        self.train_iterators = {source: iter(ds) for source, ds in zip(source_names, tokenized_datasets_train)}
        self.eval_iterators = {source: iter(ds) for source, ds in zip(source_names, tokenized_datasets_eval)}
        self.source_names = source_names

        # store backup references to infinite datasets for recovery
        for source, ds in zip(source_names, tokenized_datasets_train):
            setattr(self, f'_backup_{source}_train_ds', ds)
        for source, ds in zip(source_names, tokenized_datasets_eval):
            setattr(self, f'_backup_{source}_eval_ds', ds)

        # create dual buffers for token-level balancing
        self.sample_buffer = {source: [] for source in source_names}

        # create korean sub-buffers for hanja classification
        self.korean_sub_buffers = {'korean_hanja_heavy': [], 'korean_hanja_light': []}

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

        log_from_main_process(logger, 'info', f"Token-balanced packed dataset created with {len(source_names)} sources")
        log_from_main_process(logger, 'info', f"Sources: {', '.join(source_names)}")
        log_from_main_process(logger, 'info', f"All examples will be packed sequences with segment_ids!")