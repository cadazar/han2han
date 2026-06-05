#!/usr/bin/env python3
# coding: utf-8
"""
Han2Han Phase 2 Packed Collator - Packs multiple documents with script token boundaries.
Documents are packed until reaching target length, with script tokens as natural boundaries.
"""

import numpy as np
import jax.numpy as jnp
from typing import Dict, List, Any, Tuple
from transformers import BatchEncoding
import logging
from phase2_collator import Phase2MixedCollator

from logging_utils import log_from_all_processes, log_from_main_process

logger = logging.getLogger(__name__)


class PackedPhase2Collator(Phase2MixedCollator):
    """
    Packed version of Phase2MixedCollator that combines multiple documents
    into single sequences for efficient training. Uses script tokens
    (<hangul>, <hanja>) as document boundaries.
    """

    def __init__(
        self,
        *args,
        enable_packing: bool = True,
        packing_efficiency_threshold: float = 0.8,  # only pack if we can fill 80% of sequence
        packed_buffer_size: int = 64,  # buffer this many examples before packing
        **kwargs
    ):
        # set packing attributes BEFORE super().__init__() because
        # __post_init__ calls _instantiate_dsets() which needs these
        self.enable_packing = enable_packing
        self.packing_efficiency_threshold = packing_efficiency_threshold
        self.packed_buffer_size = packed_buffer_size
        super().__init__(*args, **kwargs)

        # track packing statistics
        self.packing_stats = {
            'total_docs_packed': 0,
            'total_packs_created': 0,
            'avg_docs_per_pack': [],
            'packing_efficiency': [],
            'malformed_examples_skipped': 0
        }

        # script token IDs for boundary detection
        self.script_token_ids = {
            self.tokenizer.convert_tokens_to_ids('<hangul>'),
            self.tokenizer.convert_tokens_to_ids('<hanja>'),
        }

        log_from_main_process(logger, 'info', f"Initialized PackedPhase2Collator with packing={'enabled' if enable_packing else 'disabled'}")
        log_from_main_process(logger, 'info', f"Buffer size before packing: {packed_buffer_size} examples")

    def pack_documents(
        self,
        tokenized_examples: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Pack multiple documents into single sequences, respecting script token boundaries.

        Each document format:
        - Encoder: [content] <script>
        - Decoder: <script> [generated_content]

        Packed format (example with 3 docs):
        - Encoder: [doc1] <script> [doc2] <script> [doc3] <script>
        - Decoder: <script> [gen1] <script> [gen2] <script> [gen3]
        - Segment IDs track document boundaries for attention masking
        """
        if not self.enable_packing:
            return tokenized_examples

        # first-fit-decreasing: sort longest first for tighter bin packing
        tokenized_examples = sorted(
            tokenized_examples,
            key=lambda ex: max(len(ex['input_ids']), len(ex['decoder_input_ids'])),
            reverse=True
        )

        packed_batches = []
        current_pack = {
            'input_ids': [],
            'attention_mask': [],
            'decoder_input_ids': [],
            'decoder_attention_mask': [],
            'labels': [],
            'segment_ids': [],  # for encoder
            'decoder_segment_ids': [],  # for decoder
            'position_ids': [],  # for encoder RoPE
            'decoder_position_ids': [],  # for decoder RoPE
            'num_tokens': 0,
            'num_docs': 0
        }

        current_segment_id = 1  # 0 reserved for padding

        for example in tokenized_examples:
            # check document length
            encoder_len = len(example['input_ids'])
            decoder_len = len(example['decoder_input_ids'])

            # guard: truncate if individual example exceeds max_length (sentinel masking edge case)
            if encoder_len > self.max_length or decoder_len > self.max_length:
                log_from_all_processes(logger, 'debug',
                    f"Truncating oversized example before packing: encoder={encoder_len}, decoder={decoder_len}")
                example['input_ids'] = example['input_ids'][:self.max_length]
                example['decoder_input_ids'] = example['decoder_input_ids'][:self.max_length]
                example['labels'] = example['labels'][:self.max_length]
                encoder_len = len(example['input_ids'])
                decoder_len = len(example['decoder_input_ids'])

            total_len = max(encoder_len, decoder_len)  # since we pad to match

            # check if adding this document would exceed max_length
            if current_pack['num_tokens'] > 0:  # not first doc
                would_exceed_length = current_pack['num_tokens'] + total_len > self.max_length

                if would_exceed_length:
                    # check if current pack meets efficiency threshold
                    efficiency = current_pack['num_tokens'] / self.max_length
                    if efficiency >= self.packing_efficiency_threshold:
                        # finalize and save current pack
                        packed_batches.append(self._finalize_pack(current_pack))
                        # reset for new pack
                        current_pack = self._reset_pack()
                        current_segment_id = 1
                    else:
                        # don't pack this document, add current pack and this doc separately
                        if current_pack['num_tokens'] > 0:
                            packed_batches.append(self._finalize_pack(current_pack))
                        # add this document as standalone (will be padded)
                        standalone = self._create_standalone_example(example)
                        if standalone is not None:
                            packed_batches.append(standalone)
                        current_pack = self._reset_pack()
                        current_segment_id = 1
                        continue

            # add document to current pack
            # pad each example to max(encoder_len, decoder_len) to avoid cross-attention overlap
            max_len = max(encoder_len, decoder_len)
            encoder_pad_len = max_len - encoder_len
            decoder_pad_len = max_len - decoder_len

            # encoder side
            current_pack['input_ids'].extend(example['input_ids'])
            current_pack['attention_mask'].extend([1] * encoder_len)  # real tokens
            current_pack['segment_ids'].extend([current_segment_id] * encoder_len)
            current_pack['position_ids'].extend(range(1, encoder_len + 1))

            # pad encoder if needed (when encoder shorter than decoder)
            if encoder_pad_len > 0:
                current_pack['input_ids'].extend([self.tokenizer.pad_token_id] * encoder_pad_len)
                current_pack['attention_mask'].extend([0] * encoder_pad_len)
                current_pack['segment_ids'].extend([0] * encoder_pad_len)
                current_pack['position_ids'].extend([0] * encoder_pad_len)

            # decoder side
            current_pack['decoder_input_ids'].extend(example['decoder_input_ids'])
            current_pack['decoder_attention_mask'].extend([1] * decoder_len)  # real tokens
            current_pack['labels'].extend(example['labels'])
            current_pack['decoder_segment_ids'].extend([current_segment_id] * decoder_len)
            current_pack['decoder_position_ids'].extend(range(1, decoder_len + 1))

            # pad decoder if needed (when decoder shorter than encoder)
            if decoder_pad_len > 0:
                current_pack['decoder_input_ids'].extend([self.tokenizer.pad_token_id] * decoder_pad_len)
                current_pack['decoder_attention_mask'].extend([0] * decoder_pad_len)
                current_pack['labels'].extend([-100] * decoder_pad_len)
                current_pack['decoder_segment_ids'].extend([0] * decoder_pad_len)
                current_pack['decoder_position_ids'].extend([0] * decoder_pad_len)

            # update counters
            current_pack['num_tokens'] += total_len
            current_pack['num_docs'] += 1
            current_segment_id += 1

            # track statistics
            self.packing_stats['total_docs_packed'] += 1

        # handle remaining pack
        if current_pack['num_tokens'] > 0:
            packed_batches.append(self._finalize_pack(current_pack))

        return packed_batches

    def pack_documents_with_leftovers(
        self,
        tokenized_examples: List[Dict[str, Any]],
        max_recycles: int = 2
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Pack documents like pack_documents(), but return short poorly-fitting
        documents as leftovers instead of emitting them as standalone padded
        sequences.

        Only documents shorter than half of max_length are recycled -- longer
        documents can never pair with another similarly-sized doc, so they emit
        as standalones immediately. Documents that have been recycled
        max_recycles times are also emitted as standalones.

        The trailing incomplete pack is always finalized (FFD already tried its
        best; recycling its docs just creates an accumulation feedback loop).

        Returns:
            (packed_sequences, leftovers) where leftovers are original tokenized
            example dicts that can re-enter packing in the next buffer round.
        """
        if not self.enable_packing:
            return tokenized_examples, []

        tokenized_examples = sorted(
            tokenized_examples,
            key=lambda ex: max(len(ex['input_ids']), len(ex['decoder_input_ids'])),
            reverse=True
        )

        packed_batches = []
        leftovers = []
        current_pack = self._reset_pack()
        current_segment_id = 1
        recycle_ceiling = self.max_length // 2

        for example in tokenized_examples:
            encoder_len = len(example['input_ids'])
            decoder_len = len(example['decoder_input_ids'])

            if encoder_len > self.max_length or decoder_len > self.max_length:
                log_from_all_processes(logger, 'debug',
                    f"Truncating oversized example before packing: encoder={encoder_len}, decoder={decoder_len}")
                example['input_ids'] = example['input_ids'][:self.max_length]
                example['decoder_input_ids'] = example['decoder_input_ids'][:self.max_length]
                example['labels'] = example['labels'][:self.max_length]
                encoder_len = len(example['input_ids'])
                decoder_len = len(example['decoder_input_ids'])

            total_len = max(encoder_len, decoder_len)

            if current_pack['num_tokens'] > 0:
                would_exceed_length = current_pack['num_tokens'] + total_len > self.max_length

                if would_exceed_length:
                    efficiency = current_pack['num_tokens'] / self.max_length
                    if efficiency >= self.packing_efficiency_threshold:
                        packed_batches.append(self._finalize_pack(current_pack))
                        current_pack = self._reset_pack()
                        current_segment_id = 1
                    else:
                        if current_pack['num_tokens'] > 0:
                            packed_batches.append(self._finalize_pack(current_pack))
                        recycle_count = example.get('_recycle_count', 0)
                        can_recycle = (
                            total_len < recycle_ceiling
                            and recycle_count < max_recycles
                        )
                        if can_recycle:
                            example['_recycle_count'] = recycle_count + 1
                            leftovers.append(example)
                        else:
                            standalone = self._create_standalone_example(example)
                            if standalone is not None:
                                packed_batches.append(standalone)
                        current_pack = self._reset_pack()
                        current_segment_id = 1
                        continue

            # add document to current pack
            max_len = max(encoder_len, decoder_len)
            encoder_pad_len = max_len - encoder_len
            decoder_pad_len = max_len - decoder_len

            current_pack['input_ids'].extend(example['input_ids'])
            current_pack['attention_mask'].extend([1] * encoder_len)
            current_pack['segment_ids'].extend([current_segment_id] * encoder_len)
            current_pack['position_ids'].extend(range(1, encoder_len + 1))

            if encoder_pad_len > 0:
                current_pack['input_ids'].extend([self.tokenizer.pad_token_id] * encoder_pad_len)
                current_pack['attention_mask'].extend([0] * encoder_pad_len)
                current_pack['segment_ids'].extend([0] * encoder_pad_len)
                current_pack['position_ids'].extend([0] * encoder_pad_len)

            current_pack['decoder_input_ids'].extend(example['decoder_input_ids'])
            current_pack['decoder_attention_mask'].extend([1] * decoder_len)
            current_pack['labels'].extend(example['labels'])
            current_pack['decoder_segment_ids'].extend([current_segment_id] * decoder_len)
            current_pack['decoder_position_ids'].extend(range(1, decoder_len + 1))

            if decoder_pad_len > 0:
                current_pack['decoder_input_ids'].extend([self.tokenizer.pad_token_id] * decoder_pad_len)
                current_pack['decoder_attention_mask'].extend([0] * decoder_pad_len)
                current_pack['labels'].extend([-100] * decoder_pad_len)
                current_pack['decoder_segment_ids'].extend([0] * decoder_pad_len)
                current_pack['decoder_position_ids'].extend([0] * decoder_pad_len)

            current_pack['num_tokens'] += total_len
            current_pack['num_docs'] += 1
            current_segment_id += 1

            self.packing_stats['total_docs_packed'] += 1

        # always finalize trailing pack
        if current_pack['num_tokens'] > 0:
            packed_batches.append(self._finalize_pack(current_pack))

        return packed_batches, leftovers

    def _reset_pack(self) -> Dict[str, Any]:
        """Create a fresh pack dictionary."""
        return {
            'input_ids': [],
            'attention_mask': [],
            'decoder_input_ids': [],
            'decoder_attention_mask': [],
            'labels': [],
            'segment_ids': [],
            'decoder_segment_ids': [],
            'position_ids': [],
            'decoder_position_ids': [],
            'num_tokens': 0,
            'num_docs': 0
        }

    def _finalize_pack(self, pack: Dict[str, Any]) -> Dict[str, Any]:
        """
        Finalize a pack by padding to max_length and converting to arrays.
        """
        # pad all sequences to max_length
        encoder_len = len(pack['input_ids'])
        decoder_len = len(pack['decoder_input_ids'])

        # encoder padding
        # use segment_id=0 for padding so create_packed_attention_mask properly masks them
        encoder_pad_len = self.max_length - encoder_len
        if encoder_pad_len > 0:
            pack['input_ids'].extend([self.tokenizer.pad_token_id] * encoder_pad_len)
            pack['attention_mask'].extend([0] * encoder_pad_len)
            pack['segment_ids'].extend([0] * encoder_pad_len)
            pack['position_ids'].extend([0] * encoder_pad_len)

        # decoder padding
        decoder_pad_len = self.max_length - decoder_len
        if decoder_pad_len > 0:
            pack['decoder_input_ids'].extend([self.tokenizer.pad_token_id] * decoder_pad_len)
            pack['decoder_attention_mask'].extend([0] * decoder_pad_len)
            pack['decoder_segment_ids'].extend([0] * decoder_pad_len)
            pack['decoder_position_ids'].extend([0] * decoder_pad_len)

        # pad labels separately (might be 1 shorter due to teacher forcing)
        label_len = len(pack['labels'])
        label_pad_len = self.max_length - label_len
        if label_pad_len > 0:
            pack['labels'].extend([-100] * label_pad_len)

        # track statistics
        self.packing_stats['total_packs_created'] += 1
        self.packing_stats['avg_docs_per_pack'].append(pack['num_docs'])
        self.packing_stats['packing_efficiency'].append(
            max(encoder_len, decoder_len) / self.max_length
        )

        # convert to arrays and return
        result = {
            'input_ids': np.array(pack['input_ids'][:self.max_length], dtype=np.int32),
            'attention_mask': np.array(pack['attention_mask'][:self.max_length], dtype=np.int32),
            'decoder_input_ids': np.array(pack['decoder_input_ids'][:self.max_length], dtype=np.int32),
            'decoder_attention_mask': np.array(pack['decoder_attention_mask'][:self.max_length], dtype=np.int32),
            'labels': np.array(pack['labels'][:self.max_length], dtype=np.int32),
            'segment_ids': np.array(pack['segment_ids'][:self.max_length], dtype=np.int32),
            'decoder_segment_ids': np.array(pack['decoder_segment_ids'][:self.max_length], dtype=np.int32),
            'position_ids': np.array(pack['position_ids'][:self.max_length], dtype=np.int32),
            'decoder_position_ids': np.array(pack['decoder_position_ids'][:self.max_length], dtype=np.int32)
        }

        return result

    def _create_standalone_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a standalone (non-packed) example with proper padding and segment IDs.
        """
        encoder_len = len(example['input_ids'])
        decoder_len = len(example['decoder_input_ids'])

        # guard against examples that exceed max_length (can happen with sentinel masking edge cases)
        if encoder_len > self.max_length or decoder_len > self.max_length:
            log_from_all_processes(logger, 'debug',
                f"Truncating oversized example: encoder_len={encoder_len}, decoder_len={decoder_len}, max_length={self.max_length}")
            example['input_ids'] = example['input_ids'][:self.max_length]
            example['decoder_input_ids'] = example['decoder_input_ids'][:self.max_length]
            example['labels'] = example['labels'][:self.max_length]
            encoder_len = len(example['input_ids'])
            decoder_len = len(example['decoder_input_ids'])

        # create segment IDs (all 1 since it's a single document)
        segment_ids = np.ones(encoder_len, dtype=np.int32)
        decoder_segment_ids = np.ones(decoder_len, dtype=np.int32)

        # create position IDs (sequential, starting from 1)
        position_ids = np.arange(1, encoder_len + 1, dtype=np.int32)
        decoder_position_ids = np.arange(1, decoder_len + 1, dtype=np.int32)

        # pad to max_length
        encoder_pad_len = self.max_length - encoder_len
        decoder_pad_len = self.max_length - decoder_len

        if encoder_pad_len > 0:
            segment_ids = np.pad(segment_ids, (0, max(0, encoder_pad_len)), constant_values=0)
            position_ids = np.pad(position_ids, (0, max(0, encoder_pad_len)), constant_values=0)

        if decoder_pad_len > 0:
            decoder_segment_ids = np.pad(decoder_segment_ids, (0, max(0, decoder_pad_len)), constant_values=0)
            decoder_position_ids = np.pad(decoder_position_ids, (0, max(0, decoder_pad_len)), constant_values=0)

        # manually pad all fields to max_length
        result = {}

        # encoder fields
        try:
            result['input_ids'] = np.pad(
                np.array(example['input_ids']),
                (0, max(0, encoder_pad_len)),
                constant_values=self.tokenizer.pad_token_id
            )[:self.max_length]
        except ValueError:
            log_from_all_processes(logger, 'warning', f"\n{'='*80}")
            log_from_all_processes(logger, 'warning', f"Skipping malformed example in _create_standalone_example")
            log_from_all_processes(logger, 'warning', f"encoder_len: {encoder_len}, max_length: {self.max_length}, encoder_pad_len: {encoder_pad_len}")
            log_from_all_processes(logger, 'warning', f"decoder_len: {decoder_len}, decoder_pad_len: {decoder_pad_len}")
            log_from_all_processes(logger, 'warning', f"Decoded input (first 200 chars): {self.tokenizer.decode(example['input_ids'][:200])}")
            log_from_all_processes(logger, 'warning', f"Decoded decoder: {self.tokenizer.decode(example['decoder_input_ids'])}")
            log_from_all_processes(logger, 'warning', f"{'='*80}\n")
            self.packing_stats['malformed_examples_skipped'] += 1
            return None

        # Create attention mask (1s for real tokens, 0s for padding)
        attention_mask = np.ones(encoder_len, dtype=np.int32)
        if encoder_pad_len > 0:
            attention_mask = np.pad(attention_mask, (0, max(0, encoder_pad_len)), constant_values=0)
        result['attention_mask'] = attention_mask[:self.max_length]

        result['segment_ids'] = segment_ids[:self.max_length]
        result['position_ids'] = position_ids[:self.max_length]

        # decoder fields
        try:
            result['decoder_input_ids'] = np.pad(
                np.array(example['decoder_input_ids']),
                (0, max(0, decoder_pad_len)),
                constant_values=self.tokenizer.pad_token_id
            )[:self.max_length]
        except ValueError:
            log_from_all_processes(logger, 'warning', f"\n{'='*80}")
            log_from_all_processes(logger, 'warning', f"Skipping malformed example in decoder padding")
            log_from_all_processes(logger, 'warning', f"encoder_len: {encoder_len}, max_length: {self.max_length}, encoder_pad_len: {encoder_pad_len}")
            log_from_all_processes(logger, 'warning', f"decoder_len: {decoder_len}, decoder_pad_len: {decoder_pad_len}")
            log_from_all_processes(logger, 'warning', f"Decoded encoder input (first 200 chars): {self.tokenizer.decode(example['input_ids'][:200])}")
            log_from_all_processes(logger, 'warning', f"Decoded decoder (first 200 chars): {self.tokenizer.decode(example['decoder_input_ids'][:200])}")
            log_from_all_processes(logger, 'warning', f"Full decoder length: {len(example['decoder_input_ids'])}")
            log_from_all_processes(logger, 'warning', f"{'='*80}\n")
            self.packing_stats['malformed_examples_skipped'] += 1
            return None

        # Create decoder attention mask (1s for real tokens, 0s for padding)
        decoder_attention_mask = np.ones(decoder_len, dtype=np.int32)
        if decoder_pad_len > 0:
            decoder_attention_mask = np.pad(decoder_attention_mask, (0, max(0, decoder_pad_len)), constant_values=0)
        result['decoder_attention_mask'] = decoder_attention_mask[:self.max_length]

        result['decoder_segment_ids'] = decoder_segment_ids[:self.max_length]
        result['decoder_position_ids'] = decoder_position_ids[:self.max_length]

        # labels (might be 1 shorter)
        label_len = len(example['labels'])
        label_pad_len = self.max_length - label_len
        try:
            result['labels'] = np.pad(
                np.array(example['labels']),
                (0, max(0, label_pad_len)),
                constant_values=-100
            )[:self.max_length]
        except ValueError:
            log_from_all_processes(logger, 'warning', f"\n{'='*80}")
            log_from_all_processes(logger, 'warning', f"Skipping malformed example in labels padding")
            log_from_all_processes(logger, 'warning', f"label_len: {label_len}, max_length: {self.max_length}, label_pad_len: {label_pad_len}")
            try:
                decoded_labels = self.tokenizer.decode([l for l in example['labels'] if l != -100][:200])
                log_from_all_processes(logger, 'warning', f"Decoded labels (first 200 chars): {decoded_labels}")
            except Exception:
                log_from_all_processes(logger, 'warning', f"Could not decode labels")
            log_from_all_processes(logger, 'warning', f"{'='*80}\n")
            self.packing_stats['malformed_examples_skipped'] += 1
            return None

        return result

    def __call__(
        self,
        examples: Dict[str, Any] | List[Dict[str, Any]],
        cooldown_phase: bool = False,
        bucket_idx: int = None,
        tokenizer: Any = None,
        morpheme_tokenizers: dict = None,
        return_source: bool = False,
        use_length_sampling: bool = True,
        padding: bool = True
    ) -> BatchEncoding | Tuple[BatchEncoding, str]:
        """
        Main collation function with packing support and internal buffering.
        Buffers single examples and returns packed batches when buffer is full.
        """
        # if packing disabled, use parent directly
        if not self.enable_packing:
            return super().__call__(
                examples, cooldown_phase=cooldown_phase,
                bucket_idx=bucket_idx, tokenizer=tokenizer,
                return_source=return_source,
                use_length_sampling=use_length_sampling,
                padding=padding, morpheme_tokenizers=morpheme_tokenizers
            )

        # handle single example vs batch
        is_single_example = isinstance(examples, dict)

        # single example handling without buffering—handled in generator wrapper
        if is_single_example:
            # simply process and return the single example
            # the generator handles buffering and packing
            return super().__call__(
                examples, cooldown_phase=cooldown_phase,
                bucket_idx=bucket_idx, tokenizer=tokenizer,
                return_source=return_source,
                use_length_sampling=use_length_sampling,
                padding=padding, morpheme_tokenizers=morpheme_tokenizers
            )

        # batch processing path (list of examples)
        examples_list = examples if isinstance(examples, list) else [examples]

        # tokenize examples w/o padding to get actual lengths
        tokenized_examples = []
        sources = []  # track sources if needed

        for example in examples_list:
            # use parent's tokenization with padding=False to get actual lengths!
            result = super().__call__(
                example, cooldown_phase=cooldown_phase,
                padding=False, return_source=return_source,
                morpheme_tokenizers=morpheme_tokenizers
            )

            if return_source:
                tokenized, source = result
                sources.append(source)
            else:
                tokenized = result

            # tokenized is now unpadded sequences!
            # convert to dict format for packing - NO attention masks needed yet!
            single_example = {
                'input_ids': tokenized['input_ids'],
                'decoder_input_ids': tokenized['decoder_input_ids'],
                'labels': tokenized['labels']
            }
            tokenized_examples.append(single_example)

        # pack documents if enabled
        packed_examples = self.pack_documents(tokenized_examples)

        # when called from buffer filling (self.buffer_lock=True), return the list
        # so we can cache individual examples. Otherwise, stack into a batch.
        if self.buffer_lock:
            # return list of packed examples for caching (NOT stacked!)
            # we'll extract them individually in the calling code
            return packed_examples
        else:
            # normal batch mode: stack into BatchEncoding
            batch = {
                key: np.stack([ex[key] for ex in packed_examples])
                for key in packed_examples[0].keys()
            }

            batch_encoding = BatchEncoding(batch)

            # return with source if requested (use first source for packed batch)
            if return_source and sources:
                return batch_encoding, sources[0]
            else:
                return batch_encoding

    def create_packed_attention_masks(self, batch: BatchEncoding|Dict) -> BatchEncoding|Dict:
        """
        Create attention masks that prevent cross-document attention.
        This should be called in the model or training script.
        """
        from packed_rope_utils import create_packed_attention_mask

        # create encoder attention mask (overwrites 2D mask with 4D packed mask)
        if 'segment_ids' in batch:
            attention_mask = create_packed_attention_mask(
                np.asarray(batch['segment_ids'], dtype=np.int32),
                causal=False,
                dtype=np.float32
            )
            batch['attention_mask'] = attention_mask

        # create decoder attention mask
        if 'decoder_segment_ids' in batch:
            decoder_attention_mask = create_packed_attention_mask(
                np.asarray(batch['decoder_segment_ids'], dtype=np.int32),
                causal=True,
                dtype=np.float32
            )
            batch['decoder_attention_mask'] = decoder_attention_mask

        return batch