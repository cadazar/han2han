#!/usr/bin/env python3
# coding: utf-8
"""
Phase 2 Mixed Collator for Continued Pretraining

Inherits from MultilingualCollator and adds new training modes:
1. Denoising (regular) - light corruption (~15-45%), can be token or morpheme-based
2. Denoising (heavy) - heavy corruption (~50%), always token-based (X-denoiser)
3. Continuation - prefix LM for generation (S-denoiser)

UL2-style mixture of denoisers with mode ratios (default: 40/40/20).
Within denoising modes, supports both BART-style (<mask>) and T5-style (<extra_id_N>) corruption.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from multilingual_collator import MultilingualCollator
from han2han_tools import has_hanja, transcribe
from logging_utils import log_from_all_processes

import logging

logger = logging.getLogger(__name__)


class Phase2MixedCollator(MultilingualCollator):
    """
    Phase 2 collator that mixes denoising, continuation, and NLP task training.

    Inherits all denoising logic from MultilingualCollator.
    Adds continuation and NLP task modes for generation training.
    """

    def __init__(
        self,
        *args,
        mode_ratios: Optional[Dict[str, float]] = None,
        seed: Optional[int] = None,
        max_encoder_length: int = None,
        max_decoder_length: int = None,
        morpheme_denoising_ratio: float = 0.5,
        sentinel_denoising_ratio: float = 0.5,
        heavy_infilling_ratio: float = 0.50,
        use_corruption_prompts: bool = False,
        byte_reconstruction_ratio: float = 0.0,
        temporal_continuation_ratio: float = 0.0,
        r_denoiser_configs: Optional[list] = None,
        x_denoiser_configs: Optional[list] = None,
        morpheme_denoiser_configs: Optional[list] = None,
        **kwargs
    ):
        """
        Args:
            *args, **kwargs: Passed to parent MultilingualCollator
            mode_ratios: Dict with 'denoising', 'denoising_heavy', 'continuation' ratios
                        Default: {'denoising': 0.40, 'denoising_heavy': 0.40, 'continuation': 0.20}
            max_encoder_length: Maximum sequence length for the encoder
                        Default: same as max_length from parent
            max_decoder_length: Maximum sequence length for the decoder
                        Default: same as max_length from parent
            morpheme_denoising_ratio: Ratio of regular denoising that uses morpheme-level corruption
                        Default: 0.5 (50% morpheme, 50% token-level)
                        NOTE: Only applies to 'denoising' mode, NOT 'denoising_heavy'
            sentinel_denoising_ratio: Ratio of token-level denoising that uses T5-style sentinels
                        Default: 0.5 (50% sentinel, 50% BART-style <mask>)
            heavy_infilling_ratio: Corruption ratio for heavy denoising (X-denoiser)
                        Default: 0.50 (50% of tokens masked)
            use_corruption_prompts: Whether to add prompts to corruption tasks
                        Default: False
            byte_reconstruction_ratio: Within denoising_heavy, ratio to route byte-containing
                        samples to byte_reconstruction task. Default: 0.0 (disabled)
            temporal_continuation_ratio: For samples with year metadata, ratio to use
                        temporal continuation task (continuation + year estimation).
                        Default: 0.0 (disabled)
            r_denoiser_configs: List of (lambda, ratio) tuples for R-denoiser (regular denoising)
                        Default: [(poisson_lambda, infilling_ratio)]
            x_denoiser_configs: List of (lambda, ratio) tuples for X-denoiser (heavy denoising)
                        Default: [(poisson_lambda, heavy_infilling_ratio)]
            morpheme_denoiser_configs: List of (lambda, ratio) tuples for morpheme denoising
                        Default: [(morpheme_lambda, infilling_ratio)]
            seed: Random seed
        """
        super().__init__(*args, **kwargs)

        # mode sampling probabilities (UL2-style: R=40%, X=40%, S=20%)
        if mode_ratios is None:
            mode_ratios = {
                'denoising': 0.40,        # R-denoiser (regular corruption)
                'denoising_heavy': 0.40,  # X-denoiser (heavy corruption, token-based only)
                'continuation': 0.20,     # S-denoiser (prefix LM)
            }

        self.mode_ratios = mode_ratios
        self.modes = list(mode_ratios.keys())
        self.mode_probs = [mode_ratios[m] for m in self.modes]

        # use parent's max_length as default for encoder/decoder if not specified
        self.max_encoder_length = max_encoder_length if max_encoder_length is not None else self.max_length
        self.max_decoder_length = max_decoder_length if max_decoder_length is not None else self.max_length

        self.rng = np.random.default_rng(seed)

        # denoising configuration
        self.morpheme_denoising_ratio = morpheme_denoising_ratio
        self.sentinel_denoising_ratio = sentinel_denoising_ratio
        self.heavy_infilling_ratio = heavy_infilling_ratio
        self.use_corruption_prompts = use_corruption_prompts
        self.byte_reconstruction_ratio = byte_reconstruction_ratio
        self.temporal_continuation_ratio = temporal_continuation_ratio

        # store base infilling ratio for regular denoising (from parent)
        self.base_infilling_ratio = self.infilling_ratio

        # UL2-style multi-config denoising (backwards compatible with single values)
        self.r_denoiser_configs = r_denoiser_configs if r_denoiser_configs is not None else [(self.poisson_lambda, self.infilling_ratio)]
        self.x_denoiser_configs = x_denoiser_configs if x_denoiser_configs is not None else [(self.poisson_lambda, self.heavy_infilling_ratio)]
        self.morpheme_denoiser_configs = morpheme_denoiser_configs if morpheme_denoiser_configs is not None else [(self.morpheme_lambda, self.infilling_ratio)]

        # Document length tracking for debugging
        self.doc_length_history = []
        self.batch_count = 0

        # script tokens (Han2Han uses <hangul>/<hanja>)
        self.script_tokens = {
            'hangul': self.tokenizer.convert_tokens_to_ids('<hangul>'),
            'hanja': self.tokenizer.convert_tokens_to_ids('<hanja>')
        }

    def _sample_mode(self) -> str:
        """Sample training mode according to mode_ratios."""
        return self.rng.choice(self.modes, p=self.mode_probs)

    def _sample_mode_by_length(self, text: str) -> str:
        """
        Sample mode based on document length for optimal TPU utilization.

        Short docs waste tokens in continuation mode, so prefer denoising.
        Long docs work great for continuation with minimal waste.
        Heavy denoising works well for all lengths (just corrupts more).

        Respects self.mode_ratios when continuation is disabled (ratio=0).
        """
        # if continuation is explicitly disabled in mode_ratios, use _sample_mode
        # (respects user config like phase 1 pure BART denoising)
        if self.mode_ratios.get('continuation', 0.0) == 0.0:
            return self._sample_mode()

        # tokenize to check length
        token_count = len(self.tokenizer.encode(text, add_special_tokens=False))

        # total capacity when using both encoder and decoder
        total_capacity = self.max_encoder_length + self.max_decoder_length - 4

        # get configured ratios (default UL2: 40/40/20)
        base_denoising = self.mode_ratios.get('denoising', 0.40)
        base_heavy = self.mode_ratios.get('denoising_heavy', 0.40)
        base_continuation = self.mode_ratios.get('continuation', 0.20)

        # adaptive mode selection based on document length
        # scale continuation based on how well doc fits in sequence length
        if token_count < total_capacity * 0.3:  # very short (<30% capacity)
            # these would waste 70%+ tokens in continuation, redistribute to denoising
            cont_scale = 0.0
        elif token_count < total_capacity * 0.6:  # short (30-60% capacity)
            # some waste but tolerable, reduce continuation
            cont_scale = 0.5
        elif token_count < total_capacity:  # medium (60-100% capacity)
            # good for continuation, use full configured ratio
            cont_scale = 1.0
        else:  # long (>100% capacity)
            # UL2: never use continuation for docs that exceed capacity
            cont_scale = 0.0

        # apply scaling to continuation and redistribute
        scaled_continuation = base_continuation * cont_scale
        redistribution = base_continuation - scaled_continuation

        mode_weights = {
            'denoising': base_denoising + redistribution / 2,
            'denoising_heavy': base_heavy + redistribution / 2,
            'continuation': scaled_continuation,
        }

        # sample based on length-appropriate weights
        modes = list(mode_weights.keys())
        probs = [mode_weights[m] for m in modes]
        return self.rng.choice(modes, p=probs)

    def _is_eligible_for_byte_reconstruction(self, text: str) -> bool:
        """
        Check if text is eligible for byte reconstruction based on length.

        Bytes are ~3x longer than characters for CJK text. We need to ensure
        the byte representation fits in the encoder with margin for metadata
        and script tokens.

        Args:
            text: Input text to check

        Returns:
            True if text can be represented as bytes within encoder limits
        """
        if self.byte_reconstruction_ratio <= 0:
            return False

        # estimate byte length
        byte_len = len(text.encode('utf-8'))

        # need headroom for: metadata (~32), script tokens (2), masks
        # be conservative since bytes expand significantly for CJK
        max_bytes = self.max_encoder_length - 48

        return byte_len <= max_bytes

    def _sample_denoiser_config(self, denoiser_type: str) -> tuple:
        """Sample (lambda, ratio) config for given denoiser type.

        Args:
            denoiser_type: One of 'r' (regular), 'x' (heavy), 'morpheme'

        Returns:
            tuple: (poisson_lambda, infilling_ratio) sampled from configs
        """
        if denoiser_type == 'r':
            configs = self.r_denoiser_configs
        elif denoiser_type == 'x':
            configs = self.x_denoiser_configs
        elif denoiser_type == 'morpheme':
            configs = self.morpheme_denoiser_configs
        else:
            raise ValueError(f"Unknown denoiser type: {denoiser_type}")
        idx = self.rng.integers(0, len(configs))
        return configs[idx]

    def _intensity_bucket(self, ratio: float) -> str:
        """Bucket a corruption ratio into a coarse intensity label for prompt keying."""
        if ratio < 0.20:
            return 'light'
        if ratio < 0.40:
            return 'medium'
        return 'heavy'

    def _is_corruption_eligible(self, example: Dict[str, Any]) -> bool:
        """
        Decide whether an example flows through the corruption pipeline.

        Supervised tasks (sts/nli/translation/cot_reasoning/etc.) route through
        UnifiedCollator's task handlers and never touch Phase2's mode
        sampling, so they should not receive a corruption plan.
        """
        data_type = example.get('data_type')
        if data_type is None:
            return True
        return data_type in ('denoising', 'mixed')

    def _compose_training_mode(self, plan: Dict[str, Any]) -> str:
        """
        Assemble the suffix-tagged training_mode label that the per-task loss
        logger reads. Centralizes the legacy in-place suffix appending so the
        collator helpers can stop mutating example dicts mid-flight.
        """
        parts = [plan['mode']]
        form = plan.get('form')
        if form is not None:
            parts.append(form)
            if form == 'bart' and plan.get('shuffled'):
                parts.append('shuffled')
        transcription = plan.get('transcription')
        if transcription is not None:
            parts.append(f'transcription_{transcription}')
        return '_'.join(parts)

    def _sample_corruption_plan(
        self, example: Dict[str, Any], cooldown_phase: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Pre-roll every stochastic corruption decision for one example.

        Called from the data-loading generators between _transform_example and
        collator(...). Returns None for supervised tasks that should bypass
        the corruption pipeline. The plan threads downstream via the
        example['_corruption_plan'] stash; the collator gates on its presence
        and skips the matching internal RNG draws.
        """
        if not self._is_corruption_eligible(example):
            return None

        from task_prompts import sample_task_prompt

        text = example.get('text', example.get('original_text', ''))

        plan: Dict[str, Any] = {
            'format_name': 'text',
            'mode': None,
            'training_mode': None,
            'form': None,
            'shuffled': False,
            'intensity': None,
            'r_config': None,
            'transcription': None,
            'prompt': None,
        }

        # priority modes first (matches Phase2MixedCollator.__call__ ordering)
        if (self._is_eligible_for_byte_reconstruction(text) and
                self.rng.random() < self.byte_reconstruction_ratio):
            plan['mode'] = 'byte_reconstruction'
        elif (example.get('year') is not None and
              self.temporal_continuation_ratio > 0 and
              self.rng.random() < self.temporal_continuation_ratio):
            plan['mode'] = 'temporal_continuation'
        else:
            plan['mode'] = self._sample_mode_by_length(text) if text else self._sample_mode()

        # promote morpheme denoising to its own top-level mode for prompt + logging
        if (plan['mode'] == 'denoising' and self.rng.random() < self.morpheme_denoising_ratio):
            plan['mode'] = 'morpheme_denoising'

        # form selection for any mode that uses BART-vs-sentinel corruption.
        # only modes that route through MultilingualCollator.__call__
        # (denoising / denoising_heavy) actually consume the shuffle axis, so
        # restrict shuffled=True to those; otherwise the per-task loss label
        # would split into buckets that share identical inputs.
        if plan['mode'] in ('denoising', 'denoising_heavy', 'morpheme_denoising',
                            'continuation', 'byte_reconstruction'):
            form = 'sentinel' if self.rng.random() < self.sentinel_denoising_ratio else 'bart'
            plan['form'] = form
            if plan['mode'] in ('denoising', 'denoising_heavy'):
                plan['shuffled'] = bool(
                    getattr(self, 'sentence_permutation', False) and form == 'bart'
                )

        # intensity bucket from R-denoiser config sample (R only;
        # denoising_heavy carries intensity implicitly via the mode label)
        if plan['mode'] == 'denoising':
            sampled_lambda, sampled_ratio = self._sample_denoiser_config('r')
            plan['r_config'] = (sampled_lambda, sampled_ratio)
            plan['intensity'] = self._intensity_bucket(sampled_ratio)

        # Han2Han transcription overlay for Korean text containing Hanja.
        # only modes that actually transcribe the text get the overlay:
        # denoising / denoising_heavy via MultilingualCollator.__call__,
        # morpheme_denoising via its own transcription block. continuation
        # and byte_reconstruction collators don't transcribe.
        if (plan['mode'] in ('denoising', 'denoising_heavy', 'morpheme_denoising') and
                text and has_hanja(text)):
            ratio = getattr(self, 'han2han_transcription_ratio', None) or 0.0
            if ratio > 0 and self.rng.random() < ratio:
                plan['transcription'] = (
                    'hangul_to_hanja' if self.rng.random() < 0.5 else 'hanja_to_hangul'
                )

        plan['training_mode'] = self._compose_training_mode(plan)

        if getattr(self, 'use_task_prompts', False) and not cooldown_phase:
            plan['prompt'], _ = sample_task_prompt(
                plan['mode'],
                format_name=plan['format_name'],
                form=plan['form'],
                shuffled=plan['shuffled'],
                intensity=plan['intensity'],
                transcription=plan['transcription'],
            )

        return plan

    def _get_byte_mean_span_length(self, num_bytes: int) -> int:
        """Choose mean span length based on byte sequence length."""
        if num_bytes < 200:
            return 3
        elif num_bytes < 500:
            return 8
        else:
            return 64

    def _split_document(self, text: str, using_sentinel: bool = False, encoder_prefix: str = "") -> Tuple[List[int], List[int]]:
        """
        Split document into encoder and decoder portions with random ratios.

        Favors decoder (60-80%) for better loss token utilization and encourages open generation 
        with less encoder/decoder overlap. At this point the document should already have been
        filtered to fit within the maximum length combined of both encoder and decoder.

        Args:
            text: Full document text
            using_sentinel: whether to subtract 1 from total capacity (sentinels are additional)
            encoder_prefix: Optional prefix to prepend to encoder text before tokenizing

        Returns:
            (encoder_ids, decoder_ids) as token lists
        """
        # prepend encoder prefix if provided (e.g., "generate: ")
        if encoder_prefix:
            text = encoder_prefix + text

        # tokenize to get accurate token boundaries
        tokens = self.tokenizer.encode(text, add_special_tokens=False)

        if len(tokens) < 10:
            # too short, use all as decoder
            return [], tokens

        # check if document is longer than our max capacity
        mask_or_sentinel = (2 if using_sentinel else 1) * 2
        total_capacity = self.max_encoder_length + self.max_decoder_length - mask_or_sentinel

        if len(tokens) <= total_capacity:
            # short/medium document - random split favoring decoder
            # encoder gets 20-40%, decoder gets 60-80%
            encoder_ratio = self.rng.uniform(0.2, 0.4)
            split_idx = int(len(tokens) * encoder_ratio)

            # ensure at least some tokens in each side
            split_idx = max(1, min(split_idx, len(tokens) - 1))

            encoder_half = tokens[:split_idx]
            decoder_half = tokens[split_idx:]
        else:
            # long document - still favor decoder but respect max lengths
            # encoder gets 25-35% of capacity
            encoder_ratio = self.rng.uniform(0.25, 0.35)
            encoder_len = int(min(self.max_encoder_length, total_capacity * encoder_ratio))
            decoder_len = min(self.max_decoder_length, total_capacity - encoder_len)

            # pick a random starting position that leaves room for both
            max_start = len(tokens) - (encoder_len + decoder_len)
            start_idx = self.rng.integers(0, max_start + 1)

            # grab contiguous chunks
            encoder_half = tokens[start_idx:start_idx + encoder_len]
            decoder_half = tokens[start_idx + encoder_len:
                                start_idx + encoder_len + decoder_len]

        return encoder_half, decoder_half

    def _collate_continuation(self, examples: List[Dict], padding=True, cooldown_phase=False) -> Dict[str, np.ndarray]:
        """
        Collate examples in continuation format with random asymmetric splitting. Decoder is given
        BOS token (unlike in denoising or SFT tasks where it receives the script token) to 
        provide extra signal for open generation.

        Document structure for continuation (UL2 S-denoiser style):
            With sentinel (T5-style):
                encoder: [optional_metadata] + [20-40% of doc] + [<extra_id_0>]
                decoder: [<s>] + [<extra_id_0>] + [60-80% of doc]
                labels: [<extra_id_0>] + [60-80% of doc] + [<script_token>]

            With mask (BART-style):
                encoder: [optional_metadata] + [20-40% of doc] + [<mask>] + [<script_token>]
                decoder: [<s>] + [60-80% of doc]
                labels: [60-80% of doc] + [<script_token>]

        The sentinel/mask token marks the boundary between prefix and continuation.
        Metadata is prepended to encoder input when not in `cooldown_phase`.

        Args:
            examples: List of dicts with 'text' optional 'source' keys
            padding: Whether to pad the input(s) to the configured max_length
            cooldown_phase: If True, no metadata/prompts are added

        Returns:
            Batch dict with encoder/decoder inputs, labels, masks
        """
        batch_encoder = []
        batch_decoder = []
        batch_labels = []

        # plan-aware: a pre-rolled plan in examples[0] fixes the form for the batch.
        # legacy callers (no plan) still get a single per-batch RNG draw.
        batch_plan = examples[0].get('_corruption_plan') if examples else None
        if batch_plan is not None:
            use_sentinel = batch_plan.get('form') == 'sentinel'
        else:
            use_sentinel = (hasattr(self, 'sentinel_denoising_ratio') and
                           self.rng.random() < self.sentinel_denoising_ratio)
            masking_suffix = '_sentinel' if use_sentinel else '_bart'
            for ex in examples:
                if '_training_mode' in ex:
                    ex['_training_mode'] = ex['_training_mode'] + masking_suffix

        for ex in examples:
            # Support both 'text' and 'original_text' (parent uses original_text after transform)
            text = ex.get('text', ex.get('original_text', ''))

            if not text.strip():
                continue

            # determine script token based on Hanja presence
            # check for Hanja content to choose appropriate script token
            has_hanja_text = has_hanja(text)
            if has_hanja_text:
                script_token = self.script_tokens['hanja']
            else:
                script_token = self.script_tokens['hangul']

            # build full encoder prefix including metadata and generation mode
            encoder_prefix_parts = []

            # metadata handling based on cooldown phase
            if not cooldown_phase:
                metadata = ex.get('metadata', '') + " " # separate prompt and input
                if metadata:
                    encoder_prefix_parts.append(metadata)

            # combine all prefix parts
            encoder_prefix = "".join(encoder_prefix_parts)

            # split document with full prefix (so _split_document accounts for all tokens)
            encoder_ids, decoder_ids = self._split_document(text, using_sentinel=use_sentinel, 
                                                            encoder_prefix=encoder_prefix)

            # get sentinel or mask token
            if use_sentinel:
                boundary_token = self.tokenizer.get_sentinel_token_id(0)
            else:
                boundary_token = self.tokenizer.mask_token_id

            # encoder: prefix + boundary_token + script_token
            encoder_ids = encoder_ids + [boundary_token]

            # truncate encoder if needed (preserve boundary_token and script_token at end): shouldn't happen
            if len(encoder_ids) > self.max_encoder_length:
                encoder_ids = encoder_ids[:self.max_encoder_length-2] + [boundary_token]

            # decoder and labels depend on sentinel vs mask mode
            if use_sentinel:
                # T5-style: decoder starts with script_token + sentinel + continuation
                decoder_input = [script_token, boundary_token] + decoder_ids
                labels = [boundary_token] + decoder_ids + [script_token]
            else:
                # BART-style: decoder starts with script_token + continuation
                decoder_input = [script_token] + decoder_ids
                labels = decoder_ids + [script_token]

            # truncate decoder if needed (preserve script_token at end)
            if len(decoder_input) > self.max_decoder_length - 1:
                truncate_len = self.max_decoder_length - 2 if use_sentinel else self.max_decoder_length - 1
                decoder_ids = decoder_ids[:truncate_len]
                if use_sentinel:
                    decoder_input = [script_token, boundary_token] + decoder_ids
                    labels = [boundary_token] + decoder_ids + [script_token]
                else:
                    decoder_input = [script_token] + decoder_ids
                    labels = decoder_ids + [script_token]

            batch_encoder.append(encoder_ids)
            batch_decoder.append(decoder_input)
            batch_labels.append(labels)

        if not batch_encoder:
            # empty batch - return dummy
            return self._create_empty_batch()

        # pad to max length in batch
        if padding:
            return self._pad_batch(batch_encoder, batch_decoder, batch_labels)
        else:
            # return unpadded sequences as lists (for packing)
            # attention masks should be lists of 1s matching sequence lengths
            return {
                'input_ids': batch_encoder[0] if len(batch_encoder) == 1 else batch_encoder,
                'decoder_input_ids': batch_decoder[0] if len(batch_decoder) == 1 else batch_decoder,
                'labels': batch_labels[0] if len(batch_labels) == 1 else batch_labels,
                'attention_mask': [1] * len(batch_encoder[0]) if len(batch_encoder) == 1 else [[1] * len(seq) for seq in batch_encoder],
                'decoder_attention_mask': [1] * len(batch_decoder[0]) if len(batch_decoder) == 1 else [[1] * len(seq) for seq in batch_decoder],
            }

    def _collate_temporal_continuation(
        self,
        example: Dict[str, Any],
        cooldown_phase: bool,
        padding: bool
    ) -> Dict[str, np.ndarray]:
        """
        Temporal continuation task: generate period-appropriate continuation + year estimation.

        Similar to continuation but year is appended to text before splitting,
        so decoder naturally outputs continuation ending with " {YEAR_PREFIX} {year}",
        This trains the model to understand temporal writing styles and date texts.

        Structure:
            encoder: [prompt] + [prefix text] + [<mask>] + [<script_token>]
            decoder: [<s>] + [continuation with year]
            labels: [continuation with year] + [<script_token>]

        Args:
            example: Single example with 'text' and 'year' fields
            cooldown_phase: Whether in cooldown (no prompts)
            padding: Whether to pad sequences

        Returns:
            Dict with encoder/decoder inputs, labels, masks
        """
        text = example.get('text', example.get('original_text', ''))
        year = example.get('year')

        if not text.strip() or year is None:
            return self._create_empty_batch()

        # determine script token
        if has_hanja(text):
            script_token_id = self.script_tokens['hanja']
        else:
            script_token_id = self.script_tokens['hangul']

        # build encoder prefix with optional prompt
        encoder_prefix = ""
        if not cooldown_phase:
            metadata = example.get('metadata', '')
            if metadata:
                encoder_prefix = metadata + " "

        # append year to text before splitting - will naturally end up in decoder.
        year_prefix = '연도:'
        text_with_year = text + f" {year_prefix} {year}"

        # split document for continuation (year suffix goes to decoder portion)
        encoder_ids, decoder_ids = self._split_document(
            text_with_year, using_sentinel=False, encoder_prefix=encoder_prefix
        )

        # add boundary marker and script token to encoder (like UL2 continuation)
        encoder_ids = encoder_ids + [self.tokenizer.mask_token_id, script_token_id]

        # decoder input: script_token + continuation (includes year naturally)
        decoder_input = [script_token_id] + decoder_ids

        # labels: continuation + script_token
        labels = decoder_ids + [script_token_id]

        # truncate if needed
        if len(encoder_ids) > self.max_encoder_length:
            encoder_ids = encoder_ids[:self.max_encoder_length - 2] + [self.tokenizer.mask_token_id, script_token_id]

        if len(decoder_ids) > self.max_decoder_length - 2:
            decoder_ids = decoder_ids[:self.max_decoder_length - 2]
            decoder_input = [script_token_id] + decoder_ids
            labels = decoder_ids + [script_token_id]

        # convert to arrays
        encoder_ids = np.array(encoder_ids, dtype=np.int32)
        decoder_input = np.array(decoder_input, dtype=np.int32)
        labels = np.array(labels, dtype=np.int32)

        if padding:
            # truncate first if sequences are too long
            encoder_ids = encoder_ids[:self.max_encoder_length]
            decoder_input = decoder_input[:self.max_decoder_length]
            labels = labels[:self.max_decoder_length]

            enc_pad = self.max_encoder_length - len(encoder_ids)
            dec_pad = self.max_decoder_length - len(decoder_input)
            lbl_pad = self.max_decoder_length - len(labels)

            enc_mask = np.ones(len(encoder_ids), dtype=np.int32)
            dec_mask = np.ones(len(decoder_input), dtype=np.int32)

            if enc_pad > 0:
                encoder_ids = np.pad(encoder_ids, (0, enc_pad), constant_values=self.tokenizer.pad_token_id)
                enc_mask = np.pad(enc_mask, (0, enc_pad), constant_values=0)
            if dec_pad > 0:
                decoder_input = np.pad(decoder_input, (0, dec_pad), constant_values=self.tokenizer.pad_token_id)
                dec_mask = np.pad(dec_mask, (0, dec_pad), constant_values=0)
            if lbl_pad > 0:
                labels = np.pad(labels, (0, lbl_pad), constant_values=-100)
        else:
            enc_mask = np.ones(len(encoder_ids), dtype=np.int32)
            dec_mask = np.ones(len(decoder_input), dtype=np.int32)

        result = {
            'input_ids': encoder_ids,
            'decoder_input_ids': decoder_input,
            'labels': labels,
            'attention_mask': enc_mask,
            'decoder_attention_mask': dec_mask,
        }

        # propagate _training_mode from example if present
        if '_training_mode' in example:
            result['_training_mode'] = example['_training_mode']

        return result

    def _create_empty_batch(self) -> Dict[str, np.ndarray]:
        """Create empty batch for fallback."""
        return {
            'input_ids': np.zeros((1, self.max_encoder_length), dtype=np.int32),  # match parent's key name
            'decoder_input_ids': np.zeros((1, self.max_decoder_length), dtype=np.int32),
            'labels': np.full((1, self.max_decoder_length), -100, dtype=np.int32),
            'attention_mask': np.zeros((1, self.max_encoder_length), dtype=np.int32),  # match parent's key name
            'decoder_attention_mask': np.zeros((1, self.max_decoder_length), dtype=np.int32),
        }

    def _pad_batch(
        self,
        batch_encoder: List[List[int]],
        batch_decoder: List[List[int]],
        batch_labels: List[List[int]]
    ) -> Dict[str, np.ndarray]:
        """Pad sequences to max length in batch."""
        # always pad to configured max lengths for consistent shapes
        max_enc = self.max_encoder_length
        max_dec = self.max_decoder_length

        batch_size = len(batch_encoder)

        encoder_ids = np.full((batch_size, max_enc), self.tokenizer.pad_token_id, dtype=np.int32)
        decoder_ids = np.full((batch_size, max_dec), self.tokenizer.pad_token_id, dtype=np.int32)
        labels = np.full((batch_size, max_dec), -100, dtype=np.int32)

        for i, (enc, dec, lbl) in enumerate(zip(batch_encoder, batch_decoder, batch_labels)):
            enc_len = min(len(enc), max_enc)
            dec_len = min(len(dec), max_dec)

            encoder_ids[i, :enc_len] = enc[:enc_len]
            decoder_ids[i, :dec_len] = dec[:dec_len]
            labels[i, :dec_len] = lbl[:dec_len]

        # attention masks
        enc_mask = (encoder_ids != self.tokenizer.pad_token_id).astype(np.int32)
        dec_mask = (decoder_ids != self.tokenizer.pad_token_id).astype(np.int32)

        return {
            'input_ids': encoder_ids,  # match parent's key name
            'decoder_input_ids': decoder_ids,
            'labels': labels,
            'attention_mask': enc_mask,  # match parent's key name
            'decoder_attention_mask': dec_mask,
        }

    def _apply_sentinel_masking_morphemes(
        self,
        morphemes: list,
        tokenizer,
        eos_token: str = "</s>",
        infilling_ratio: float = None
    ) -> Tuple[list, list, int]:
        """
        Apply T5-style span corruption with unique sentinel tokens per span using T5's deterministic algorithm.

        Unlike BART-style (single <mask> token), T5-style uses unique <extra_id_N>
        tokens for each masked span, and the decoder only outputs the masked spans.

        This implementation follows T5's deterministic span allocation:
        - num_noise_tokens = round(length * noise_density)
        - num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
        - Partitions tokens into exactly that many spans using random segmentation
        - Alternates: non-noise, noise, non-noise, noise, ...

        Args:
            morphemes: List of morphemes to mask (may include </s> tokens)
            tokenizer: Tokenizer with get_sentinel_token method
            eos_token: EOS token to preserve
            infilling_ratio: Override infilling ratio (uses self.infilling_ratio if None)

        Returns:
            (encoder_morphemes, decoder_spans): Tuple
            - encoder_morphemes: input with spans replaced by <extra_id_N>
            - decoder_spans: <extra_id_N> + span_N + <extra_id_N+1> + span_N+1 + ...
        """
        if not morphemes:
            return [], []

        length = len(morphemes)
        ratio = infilling_ratio if infilling_ratio is not None else self.infilling_ratio

        if ratio == 0.0 or length < 2:
            return morphemes, []

        # identify EOS positions
        is_eos = np.array([m == eos_token for m in morphemes])

        # count non-EOS morphemes for masking calculations
        num_non_eos = np.sum(~is_eos)
        if num_non_eos == 0:
            return morphemes, []

        # T5-style deterministic span calculation
        num_noise_tokens = int(np.round(num_non_eos * ratio))
        num_noise_tokens = max(1, min(num_noise_tokens, num_non_eos - 1))

        # calculate number of spans based on mean span length
        # cap at max sentinels (T5 used 100, we have 256)
        mean_noise_span_length = float(self.morpheme_lambda)
        num_noise_spans = int(np.round(num_noise_tokens / mean_noise_span_length))
        num_noise_spans = max(1, min(num_noise_spans, tokenizer.NUM_SENTINEL_TOKENS - 1))

        num_nonnoise_tokens = num_non_eos - num_noise_tokens

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

        # create mask for non-EOS morphemes: True = noise, False = non-noise
        is_noise_non_eos = np.zeros(num_non_eos, dtype=bool)
        for i in range(1, len(span_starts), 2):
            start = span_starts[i]
            span_len = interleaved_span_lengths[i]
            is_noise_non_eos[start:start + span_len] = True

        # map back to full sequence, preserving EOS tokens
        is_masked = [False] * len(morphemes)
        non_eos_idx = 0
        for i in range(len(morphemes)):
            if not is_eos[i]:
                is_masked[i] = is_noise_non_eos[non_eos_idx]
                non_eos_idx += 1

        # pass 2: build encoder with sentinels, collect decoder spans
        encoder_result = []
        decoder_spans = []
        sentinel_idx = 0
        in_span = False

        for i in range(length):
            if is_masked[i]:
                if not in_span:
                    # start new span - add sentinel to both encoder and decoder
                    sentinel = tokenizer.get_sentinel_token(sentinel_idx)
                    encoder_result.append(sentinel)
                    decoder_spans.append(sentinel)
                    in_span = True
                # add masked content to decoder only
                decoder_spans.append(morphemes[i])
            else:
                if in_span and morphemes[i] not in tokenizer._special_tokens:
                    # end span
                    sentinel_idx += 1
                    in_span = False
                encoder_result.append(morphemes[i])

        return encoder_result, decoder_spans

    def _collate_heavy_denoising(
        self,
        example: Dict[str, Any],
        cooldown_phase: bool,
        bucket_idx: int,
        tokenizer: Any,
        padding: bool,
    ) -> Dict[str, np.ndarray]:
        """
        Heavy denoising collation (X-denoiser) - always token-based, 50% corruption.

        This version doesn't use morpheme tokenizers.
        Uses parent's token-level infilling with higher corruption ratio.

        Args:
            example: Single training example with 'text' field
            cooldown_phase: Whether in cooldown phase
            bucket_idx: Bucket index
            tokenizer: Tokenizer for encoding text
            padding: Whether to pad sequences

        Returns:
            Dict with input_ids, decoder_input_ids, labels, and attention masks
        """
        # note: byte reconstruction routing moved to __call__ for unified handling

        # sample X-denoiser config (lambda, ratio)
        sampled_lambda, sampled_ratio = self._sample_denoiser_config('x')

        # temporarily override instance state for parent's denoising
        original_lambda = self.poisson_lambda
        original_ratio = self.infilling_ratio
        self.poisson_lambda = sampled_lambda
        self.infilling_ratio = sampled_ratio

        try:
            # use parent's token-level denoising with sampled X-denoiser config
            result = super().__call__(
                example,
                cooldown_phase=cooldown_phase,
                bucket_idx=bucket_idx,
                tokenizer=tokenizer,
                return_source=False,
                padding=padding
            )
        finally:
            # restore original state
            self.poisson_lambda = original_lambda
            self.infilling_ratio = original_ratio

        return result

    def _convert_byte_spans_to_tokens(
        self,
        decoder_ids: np.ndarray,
        tokenizer
    ) -> list:
        """
        Convert byte token spans in decoder to normal token representation.

        Input: [sentinel, byte1, byte2, byte3, sentinel, byte4, byte5, ...]
        Output: [sentinel, tok1, tok2, sentinel, tok3, ...]
        """
        result = []
        current_bytes = []
        sentinel_start = tokenizer.SENTINEL_TOKEN_BASE_ID
        sentinel_end = sentinel_start + tokenizer.NUM_SENTINEL_TOKENS
        byte_start = tokenizer.BYTE_TOKEN_START
        byte_end = tokenizer.BYTE_TOKEN_END

        for tid in decoder_ids:
            if sentinel_start <= tid < sentinel_end:
                if current_bytes:
                    text = bytes(b - byte_start for b in current_bytes).decode('utf-8', errors='replace')
                    result.extend(tokenizer.encode(text, add_special_tokens=False))
                    current_bytes = []
                result.append(tid)
            elif byte_start <= tid <= byte_end:
                current_bytes.append(tid)
            else:
                result.append(tid)

        if current_bytes:
            text = bytes(b - byte_start for b in current_bytes).decode('utf-8', errors='replace')
            result.extend(tokenizer.encode(text, add_special_tokens=False))

        return result

    def _collate_byte_reconstruction(
        self,
        example: Dict[str, Any],
        cooldown_phase: bool,
        bucket_idx: int,
        tokenizer: Any,
        padding: bool
    ) -> Dict[str, np.ndarray]:
        """
        Byte reconstruction task with BART-style masking.

        Encoder: byte tokens with <mask> replacing corrupted spans + script token
        Decoder: full original text as regular tokens + script token

        This teaches the model to reconstruct text from byte-level representations,
        improving OOV handling and character-level understanding.
        """
        text = example.get('text', example.get('original_text', ''))

        # get metadata tokens if not in cooldown phase
        metadata_ids = []
        if not cooldown_phase:
            metadata = example.get('metadata', '')
            if metadata:
                metadata_ids = tokenizer.encode(metadata, add_special_tokens=False)

        # check for Hanja in text
        if has_hanja(text):
            # for byte reconstruction, encoder has bytes (no script meaning)
            # decoder outputs in original script
            encoder_script_token_id = self.script_tokens['hanja']
            decoder_script_token_id = self.script_tokens['hanja']
        else:
            encoder_script_token_id = self.script_tokens['hangul']
            decoder_script_token_id = self.script_tokens['hangul']

        # character-level window slice (conservative for CJK ~3 bytes per char)
        # account for metadata, script token in max calculation
        available_length = self.max_encoder_length - len(metadata_ids) - 2
        max_chars = available_length // 3
        if len(text) > max_chars:
            start = self.rng.integers(0, len(text) - max_chars)
            text = text[start:start + max_chars]

        # convert to byte tokens for encoder
        byte_ids = np.array(tokenizer.text_to_bytes_only(text))

        if len(byte_ids) == 0:
            return self._create_empty_batch()

        # dynamic mean span length based on byte count
        mean_span = self._get_byte_mean_span_length(len(byte_ids))

        # decide: BART-style (<mask>) or T5-style (sentinel)
        plan = example.get('_corruption_plan')
        if plan is not None:
            use_sentinel = plan.get('form') == 'sentinel'
        else:
            use_sentinel = (hasattr(self, 'sentinel_denoising_ratio') and
                           self.rng.random() < self.sentinel_denoising_ratio)
            if '_training_mode' in example:
                masking_suffix = '_sentinel' if use_sentinel else '_bart'
                example['_training_mode'] = example['_training_mode'] + masking_suffix

        old_lambda = self.poisson_lambda
        self.poisson_lambda = mean_span
        try:
            if use_sentinel:
                # T5-style: unique sentinels, decoder outputs masked spans as tokens
                corrupted_byte_ids, decoder_span_ids, _ = self._token_based_sentinel_masking(
                    byte_ids,
                    tokenizer=tokenizer,
                )
                # convert byte spans in decoder to normal tokens
                decoder_content_ids = self._convert_byte_spans_to_tokens(decoder_span_ids, tokenizer)
            else:
                # BART-style: <mask> tokens, decoder outputs full text
                corrupted_byte_ids = self._token_based_infilling(
                    byte_ids,
                    tokenizer=tokenizer,
                )
                # decoder gets full original text as regular tokens (not bytes)
                decoder_content_ids = tokenizer.encode(text, add_special_tokens=False)
        finally:
            self.poisson_lambda = old_lambda

        # build encoder: metadata + corrupted_bytes + script_token
        if isinstance(corrupted_byte_ids, np.ndarray):
            corrupted_byte_ids = corrupted_byte_ids.tolist()

        if metadata_ids:
            encoder_ids = metadata_ids + corrupted_byte_ids + [encoder_script_token_id]
        else:
            encoder_ids = corrupted_byte_ids + [encoder_script_token_id]

        # build decoder: script_token + content + eos
        decoder_ids = [decoder_script_token_id] + decoder_content_ids + [tokenizer.eos_token_id]

        # truncate if needed
        if len(encoder_ids) > self.max_encoder_length:
            encoder_ids = encoder_ids[:self.max_encoder_length - 1] + [encoder_script_token_id]
        if len(decoder_ids) > self.max_decoder_length:
            decoder_ids = decoder_ids[:self.max_decoder_length - 1] + [tokenizer.eos_token_id]

        # labels: content + script_token (shifted from decoder_input)
        labels = decoder_ids[1:]  # remove leading script token for labels

        # decoder input: script_token + content (no final eos)
        decoder_input_ids = decoder_ids[:-1]

        # convert to arrays
        encoder_ids = np.array(encoder_ids, dtype=np.int32)
        decoder_input_ids = np.array(decoder_input_ids, dtype=np.int32)
        labels = np.array(labels, dtype=np.int32)

        # padding
        if padding:
            # truncate first if sequences are too long
            encoder_ids = encoder_ids[:self.max_encoder_length]
            decoder_input_ids = decoder_input_ids[:self.max_decoder_length]
            labels = labels[:self.max_decoder_length]

            enc_pad = self.max_encoder_length - len(encoder_ids)
            dec_pad = self.max_decoder_length - len(decoder_input_ids)

            encoder_attention_mask = np.ones(len(encoder_ids), dtype=np.int32)
            decoder_attention_mask = np.ones(len(decoder_input_ids), dtype=np.int32)

            if enc_pad > 0:
                encoder_ids = np.pad(encoder_ids, (0, enc_pad), constant_values=tokenizer.pad_token_id)
                encoder_attention_mask = np.pad(encoder_attention_mask, (0, enc_pad), constant_values=0)
            if dec_pad > 0:
                decoder_input_ids = np.pad(decoder_input_ids, (0, dec_pad), constant_values=tokenizer.pad_token_id)
                labels = np.pad(labels, (0, dec_pad), constant_values=-100)
                decoder_attention_mask = np.pad(decoder_attention_mask, (0, dec_pad), constant_values=0)
        else:
            encoder_attention_mask = np.ones(len(encoder_ids), dtype=np.int32)
            decoder_attention_mask = np.ones(len(decoder_input_ids), dtype=np.int32)

        result = {
            "input_ids": encoder_ids,
            "decoder_input_ids": decoder_input_ids,
            "labels": labels,
            "attention_mask": encoder_attention_mask,
            "decoder_attention_mask": decoder_attention_mask
        }

        # propagate _training_mode from example if present
        if '_training_mode' in example:
            result['_training_mode'] = example['_training_mode']

        return result

    def _dispatch_with_plan(
        self,
        example: Dict[str, Any],
        plan: Dict[str, Any],
        cooldown_phase: bool,
        bucket_idx: Optional[int],
        tokenizer: Any,
        return_source: bool,
        padding: bool,
        morpheme_tokenizers: Optional[Dict[str, Any]],
    ) -> Any:
        """
        Dispatch a single example using a pre-rolled corruption plan.

        Authoritative path for plan-driven examples; the legacy branches in
        __call__ stay intact for direct callers that do not pre-roll a plan
        (tests, eval callbacks).
        """
        example['_training_mode'] = plan['training_mode']
        if plan.get('prompt') is not None:
            example['metadata'] = plan['prompt']

        mode = plan['mode']

        if mode == 'byte_reconstruction':
            result = self._collate_byte_reconstruction(
                example, cooldown_phase, bucket_idx, tokenizer, padding
            )
        elif mode == 'temporal_continuation':
            result = self._collate_temporal_continuation(example, cooldown_phase, padding)
        elif mode == 'morpheme_denoising':
            result = self._collate_morpheme_denoising(
                example, cooldown_phase, bucket_idx, tokenizer, padding, morpheme_tokenizers
            )
        elif mode == 'denoising':
            sampled_lambda, sampled_ratio = plan['r_config']
            original_lambda = self.poisson_lambda
            original_ratio = self.infilling_ratio
            self.poisson_lambda = sampled_lambda
            self.infilling_ratio = sampled_ratio
            try:
                return super().__call__(
                    example,
                    cooldown_phase=cooldown_phase,
                    bucket_idx=bucket_idx,
                    tokenizer=tokenizer,
                    return_source=return_source,
                    padding=padding,
                )
            finally:
                self.poisson_lambda = original_lambda
                self.infilling_ratio = original_ratio
        elif mode == 'denoising_heavy':
            result = self._collate_heavy_denoising(
                example, cooldown_phase, bucket_idx, tokenizer, padding
            )
        elif mode == 'continuation':
            batch_result = self._collate_continuation(
                [example], padding=padding, cooldown_phase=cooldown_phase
            )
            result = {}
            for key, value in batch_result.items():
                if isinstance(value, list):
                    if value and isinstance(value[0], list):
                        result[key] = value[0]
                    else:
                        result[key] = value
                elif isinstance(value, np.ndarray):
                    result[key] = value[0] if value.ndim > 1 else value
                else:
                    result[key] = value[0] if hasattr(value, '__getitem__') else value
        else:
            raise ValueError(f"Unknown plan mode: {mode}")

        if '_training_mode' not in result:
            result['_training_mode'] = example.get('_training_mode', plan['training_mode'])

        if return_source:
            source = example.get('source', 'unknown')
            return result, source
        return result

    def __call__(self, examples: Dict[str, Any]|List[Dict[str, Any]],
                 cooldown_phase: bool = False, bucket_idx: int = None,
                 tokenizer: Any = None, return_source: bool = False,
                 use_length_sampling: bool = True, padding=True,
                 morpheme_tokenizers: Dict[str, Any] = None) -> Dict[str, np.ndarray]:
        """
        Main collation function that samples mode and routes to appropriate handler.
        Supports both single example and batch modes for compatibility with parent's iterators.

        Args:
            examples: Single example dict or list of dicts with training data
            cooldown_phase: Whether in cooldown phase (no metadata) - passed through to parent
            bucket_idx: Bucket index for bucketing - passed through to parent
            tokenizer: Tokenizer to use - passed through to parent
            return_source: If True, returns tuple (batch_dict, source) - for single example mode
            use_length_sampling: If True, use length-based mode selection for efficiency
            padding: Whether to pad the input(s) to the configured max_length
            morpheme_tokenizers: Dict of language -> morpheme tokenizer instances (only support for ko)

        Returns:
            Collated batch ready for training (or tuple if return_source=True)
        """
        if isinstance(examples, dict):
            source = examples.get('source', 'UNKNOWN')
        elif isinstance(examples, list) and len(examples) > 0:
            source = examples[0].get('source', 'UNKNOWN')

        # handle single example mode (for compatibility with parent's iterators)
        if not isinstance(examples, list):
            # plan-aware fast path: when the generator pre-rolled a corruption plan,
            # all stochastic decisions are baked into plan and metadata is already
            # resolved to the right composite-key prompt
            plan = examples.get('_corruption_plan')
            if plan is not None:
                return self._dispatch_with_plan(
                    examples, plan,
                    cooldown_phase=cooldown_phase,
                    bucket_idx=bucket_idx,
                    tokenizer=tokenizer,
                    return_source=return_source,
                    padding=padding,
                    morpheme_tokenizers=morpheme_tokenizers,
                )

            text = examples.get('text', examples.get('original_text', ''))

            # check for byte reconstruction BEFORE mode sampling
            # this takes priority when text is short enough for byte representation
            if (self._is_eligible_for_byte_reconstruction(text) and
                self.rng.random() < self.byte_reconstruction_ratio):

                examples['_training_mode'] = 'byte_reconstruction'

                # add task prompt if enabled
                if hasattr(self, 'use_task_prompts') and self.use_task_prompts and not cooldown_phase:
                    from task_prompts import sample_task_prompt
                    prompt, _ = sample_task_prompt('byte_reconstruction')
                    examples['metadata'] = prompt

                result = self._collate_byte_reconstruction(
                    examples, cooldown_phase, bucket_idx, tokenizer, padding
                )
                if return_source:
                    source = examples.get('source', 'unknown')
                    return result, source
                return result

            # check for temporal continuation - requires 'year' field from mixed data_type
            year = examples.get('year')
            if (year is not None and
                self.temporal_continuation_ratio > 0 and
                self.rng.random() < self.temporal_continuation_ratio):

                examples['_training_mode'] = 'temporal_continuation'

                # add task prompt if enabled
                if hasattr(self, 'use_task_prompts') and self.use_task_prompts and not cooldown_phase:
                    from task_prompts import sample_task_prompt
                    prompt, _ = sample_task_prompt('temporal_continuation')
                    examples['metadata'] = prompt

                result = self._collate_temporal_continuation(examples, cooldown_phase, padding)
                if return_source:
                    source = examples.get('source', 'unknown')
                    return result, source
                return result

            # sample mode based on document length or global ratios
            if use_length_sampling and text:
                mode = self._sample_mode_by_length(text)
            else:
                mode = self._sample_mode()

            # store sampled mode so unified collator can add appropriate prompts
            examples['_training_mode'] = mode

            # add task prompt if parent collator wants them (but not during cooldown)
            if hasattr(self, 'use_task_prompts') and self.use_task_prompts and not cooldown_phase:
                from task_prompts import sample_task_prompt
                prompt, _ = sample_task_prompt(mode)
                examples['metadata'] = prompt

            if mode == 'denoising':
                # decide: morpheme or token-level denoising?
                if self.rng.random() < self.morpheme_denoising_ratio:
                    examples['_training_mode'] = 'morpheme_denoising'

                    # update prompt for morpheme denoising specifically (if not in cooldown)
                    if hasattr(self, 'use_task_prompts') and self.use_task_prompts and not cooldown_phase:
                        from task_prompts import sample_task_prompt
                        prompt, _ = sample_task_prompt('morpheme_denoising')
                        examples['metadata'] = prompt

                    result = self._collate_morpheme_denoising(
                        examples, cooldown_phase, bucket_idx, tokenizer, padding, morpheme_tokenizers
                    )
                    if return_source:
                        source = examples.get('source', 'unknown')
                        return result, source
                    return result
                else:
                    # sample R-denoiser config (lambda, ratio) for token-level denoising
                    sampled_lambda, sampled_ratio = self._sample_denoiser_config('r')

                    # temporarily override instance state for parent's denoising
                    original_lambda = self.poisson_lambda
                    original_ratio = self.infilling_ratio
                    self.poisson_lambda = sampled_lambda
                    self.infilling_ratio = sampled_ratio

                    try:
                        # delegate to parent for token-level denoising with sampled R-denoiser config
                        result = super().__call__(examples, cooldown_phase=cooldown_phase,
                                                bucket_idx=bucket_idx, tokenizer=tokenizer,
                                                return_source=return_source, padding=padding)
                        return result
                    finally:
                        # restore original state
                        self.poisson_lambda = original_lambda
                        self.infilling_ratio = original_ratio

            elif mode == 'denoising_heavy':
                # heavy denoising (X-denoiser) - always token-based, 50% corruption
                # update prompt for heavy denoising
                if hasattr(self, 'use_task_prompts') and self.use_task_prompts and not cooldown_phase:
                    from task_prompts import sample_task_prompt
                    prompt, _ = sample_task_prompt('denoising_heavy')
                    examples['metadata'] = prompt

                result = self._collate_heavy_denoising(
                    examples, cooldown_phase, bucket_idx, tokenizer, padding
                )
                if return_source:
                    source = examples.get('source', 'unknown')
                    return result, source
                return result

            elif mode == 'continuation':
                # for continuation mode, batch the single example and process
                result = self._collate_continuation([examples], padding=padding)

                # extract single example from batch
                single_result = {}
                for key, value in result.items():
                    # Check if value is actually a batch (2D) or already a single sequence (1D)
                    if isinstance(value, list):
                        # If it's a list of lists (batch), extract first
                        if value and isinstance(value[0], list):
                            single_result[key] = value[0]
                        else:
                            # Already a single sequence, keep as is
                            single_result[key] = value
                    elif isinstance(value, np.ndarray):
                        # If 2D array (batch), extract first row
                        if value.ndim > 1:
                            single_result[key] = value[0]
                        else:
                            # Already 1D, keep as is
                            single_result[key] = value
                    else:
                        # Unknown type, try to extract but log warning
                        log_from_all_processes(logger, 'warning', f"Unknown type for {key}: {type(value)}")
                        single_result[key] = value[0] if hasattr(value, '__getitem__') else value

                # propagate _training_mode from examples
                if '_training_mode' in examples:
                    single_result['_training_mode'] = examples['_training_mode']

                if return_source:
                    # extract source for compatibility
                    source = examples.get('source', 'unknown')
                    return single_result, source
                return single_result

            else:
                raise ValueError(f"Unknown mode: {mode}. Expected 'denoising', 'denoising_heavy', or 'continuation'.")

        # batch mode - for now sample one mode for whole batch
        # could be improved to handle each example separately
        if use_length_sampling and examples:
            # use average length to decide mode for batch
            avg_length = np.mean([len(self.tokenizer.encode(ex.get('text', ex.get('original_text', '')), add_special_tokens=False))
                                 for ex in examples[:5]])  # sample first 5 for efficiency
            # create fake text of average length for mode selection
            fake_text = 'x' * int(avg_length)
            mode = self._sample_mode_by_length(fake_text)
        else:
            mode = self._sample_mode()

        if mode == 'denoising':
            # decide: morpheme or token-level denoising?
            if self.rng.random() < self.morpheme_denoising_ratio:
                # for batch mode, process each example
                # note: could be optimized to batch morpheme processing
                results = []
                for ex in examples:
                    result = self._collate_morpheme_denoising(
                        ex, cooldown_phase, bucket_idx, tokenizer, padding, morpheme_tokenizers
                    )
                    results.append(result)
                # stack results into batch
                batch = {}
                for key in results[0].keys():
                    batch[key] = np.stack([r[key] for r in results])
                return batch
            else:
                # sample R-denoiser config (lambda, ratio) for token-level denoising
                sampled_lambda, sampled_ratio = self._sample_denoiser_config('r')

                # temporarily override instance state for parent's denoising
                original_lambda = self.poisson_lambda
                original_ratio = self.infilling_ratio
                self.poisson_lambda = sampled_lambda
                self.infilling_ratio = sampled_ratio

                try:
                    # use parent's token-level denoising logic with sampled R-denoiser config
                    result = super().__call__(examples, cooldown_phase=cooldown_phase,
                                            bucket_idx=bucket_idx, tokenizer=tokenizer,
                                            return_source=return_source, padding=padding)
                    return result
                finally:
                    # restore original state
                    self.poisson_lambda = original_lambda
                    self.infilling_ratio = original_ratio

        elif mode == 'denoising_heavy':
            # heavy denoising (X-denoiser) - always token-based, 50% corruption
            # process each example with heavy denoising
            results = []
            for ex in examples:
                result = self._collate_heavy_denoising(
                    ex, cooldown_phase, bucket_idx, tokenizer, padding
                )
                results.append(result)
            # stack results into batch
            batch = {}
            for key in results[0].keys():
                batch[key] = np.stack([r[key] for r in results])
            return batch

        elif mode == 'continuation':
            # use new continuation logic
            return self._collate_continuation(examples, cooldown_phase=cooldown_phase, padding=padding)

        else:
            raise ValueError(f"Unknown mode: {mode}. Expected 'denoising', 'denoising_heavy', or 'continuation'.")

    def _collate_morpheme_denoising(
        self,
        example: Dict[str, Any],
        cooldown_phase: bool,
        bucket_idx: int,
        tokenizer: Any,
        padding: bool,
        morpheme_tokenizers: Dict[str, Any] = None
    ) -> Dict[str, np.ndarray]:
        """
        Morpheme-aware denoising collation following Han2Han denoising paradigm.

        Extracts morphemes using MeCab-Ko morphological tokenizer,
        corrupts them using parent's span masking, and prepares encoder-decoder pairs.
        Properly handles sentence boundaries with </s> tokens.

        Structure for denoising paradigm (NOT continuation):
            encoder: corrupted text + [<script_token>]
            decoder: [<script_token>] + [optional_prompt] + original text
            labels: [optional_-100_for_prompt] + original text + [<script_token>]

        Args:
            example: Single training example with 'text' field
            cooldown_phase: Whether in cooldown (unused for now)
            bucket_idx: Bucket index (unused for now)
            tokenizer: Tokenizer for encoding text
            padding: Whether to pad sequences
            morpheme_tokenizers: Dict mapping language codes to morpheme tokenizers (ko only)

        Returns:
            Dict with input_ids, decoder_input_ids, labels, and attention masks
        """
        # use provided tokenizer or fall back to self.tokenizer
        tok = tokenizer if tokenizer is not None else self.tokenizer

        text = example.get('text', example.get('original_text', ''))

        # get morpheme tokenizer
        morph_tok = morpheme_tokenizers.get('ko') if morpheme_tokenizers else None

        # sample morpheme denoiser config (lambda, ratio)
        sampled_lambda, sampled_ratio = self._sample_denoiser_config('morpheme')

        # temporarily override instance state for morpheme masking
        original_lambda = self.morpheme_lambda
        original_ratio = self.infilling_ratio
        self.morpheme_lambda = sampled_lambda
        self.infilling_ratio = sampled_ratio

        # decide ONCE per example: BART-style (<mask>) or T5-style (sentinel) masking
        plan = example.get('_corruption_plan')
        if plan is not None:
            use_sentinel = plan.get('form') == 'sentinel'
        else:
            use_sentinel = (hasattr(self, 'sentinel_denoising_ratio') and
                           self.rng.random() < self.sentinel_denoising_ratio)
            if '_training_mode' in example:
                masking_suffix = '_sentinel' if use_sentinel else '_bart'
                example['_training_mode'] = example['_training_mode'] + masking_suffix

        # step 1: split text into sentences (only needed for BART-style)
        if 'sentences' in example and example['sentences']:
            sentences = example['sentences']
        else:
            sentences = self.split_sentences(text)

        # helper to get morphemes for a given text
        def get_morphemes_for_text(input_text):
            if morph_tok is None:
                raise ValueError(f"Korean morpheme tokenizer not provided in morpheme_tokenizers dict")
            return get_morphemes_korean(input_text, morph_tok, preserve_spacing=False)

        # step 2: process morphemes and apply corruption
        encoder_corrupted_sentences = []
        decoder_original_sentences = []

        if use_sentinel:
            # T5-style: process WHOLE document at once (not per-sentence)
            # this ensures T5's algorithm naturally bounds the number of spans
            all_morphemes = get_morphemes_for_text(text)
            all_morphemes = [m for m in all_morphemes if m.strip()]

            encoder_morphemes, decoder_spans = self._apply_sentinel_masking_morphemes(
                all_morphemes,
                tokenizer=tok,
                eos_token=tok.eos_token
            )

            encoder_morphemes = [m for m in encoder_morphemes if m.strip()]
            decoder_spans = [m for m in decoder_spans if m.strip()]

            encoder_corrupted_sentences.append(''.join(encoder_morphemes))
            decoder_original_sentences.append(''.join(decoder_spans))
        else:
            # BART-style: process per-sentence (need sentence boundaries for decoder)
            for sentence in sentences:
                morphemes = get_morphemes_for_text(sentence)
                morphemes = [m for m in morphemes if m.strip()]

                corrupted_morphemes = self._apply_morpheme_masking(
                    morphemes,
                    mask_token=tok.mask_token,
                    eos_token=tok.eos_token
                )
                corrupted_morphemes = [m for m in corrupted_morphemes if m.strip()]

                encoder_corrupted_sentences.append(''.join(corrupted_morphemes))
                decoder_original_sentences.append(sentence)

        # step 2b: apply Han2Han transcription for Korean with Hanja (after morpheme corruption)
        encoder_script_token_id = None
        decoder_script_token_id = None

        has_hanja_text = has_hanja(text)
        if has_hanja_text:
            if plan is not None:
                transcription_direction = plan.get('transcription')
            else:
                apply_transcription = (
                    hasattr(self, 'han2han_transcription_ratio') and
                    self.han2han_transcription_ratio is not None and
                    self.han2han_transcription_ratio > 0 and
                    self.rng.random() < self.han2han_transcription_ratio
                )
                if apply_transcription:
                    transcription_direction = (
                        'hangul_to_hanja' if self.rng.random() < 0.5 else 'hanja_to_hangul'
                    )
                else:
                    transcription_direction = None

            if transcription_direction == 'hangul_to_hanja':
                encoder_corrupted_sentences = [transcribe(s) for s in encoder_corrupted_sentences]
                encoder_script_token_id = self.script_tokens['hangul']
                decoder_script_token_id = self.script_tokens['hanja']
                if plan is None and '_training_mode' in example:
                    example['_training_mode'] = example['_training_mode'] + '_transcription_hangul_to_hanja'
            elif transcription_direction == 'hanja_to_hangul':
                decoder_original_sentences = [transcribe(s) for s in decoder_original_sentences]
                encoder_script_token_id = self.script_tokens['hanja']
                decoder_script_token_id = self.script_tokens['hangul']
                if plan is None and '_training_mode' in example:
                    example['_training_mode'] = example['_training_mode'] + '_transcription_hanja_to_hangul'
            else:
                encoder_script_token_id = self.script_tokens['hanja']
                decoder_script_token_id = self.script_tokens['hanja']
        else:
            encoder_script_token_id = self.script_tokens['hangul']
            decoder_script_token_id = self.script_tokens['hangul']

        # step 3: tokenize encoder sentences (corrupted, no spaces) and track boundaries
        encoder_content_ids = []
        encoder_eos_positions = []  # track sentence boundaries for alignment
        for corrupted_sentence in encoder_corrupted_sentences:
            sentence_ids = tok.encode(corrupted_sentence, add_special_tokens=False)
            encoder_content_ids.extend(sentence_ids)
            if not use_sentinel:
                # only add </s> for BART-style (will be removed later for encoder)
                encoder_content_ids.append(tok.eos_token_id)
            encoder_eos_positions.append(len(encoder_content_ids) - 1)

        # step 4: tokenize decoder sentences (original WITH spacing) and track boundaries
        decoder_content_ids = []
        decoder_eos_positions = []  # track sentence boundaries
        for original_sentence in decoder_original_sentences:
            sentence_ids = tok.encode(original_sentence, add_special_tokens=False)
            decoder_content_ids.extend(sentence_ids)
            if not use_sentinel:
                # only add </s> for BART-style (decoder outputs full text with sentence boundaries)
                decoder_content_ids.append(tok.eos_token_id)
            decoder_eos_positions.append(len(decoder_content_ids) - 1)

        # get metadata prompt for encoder prepending (not decoder!)
        # only add metadata if NOT in cooldown phase
        prompt_tokens = []
        if not cooldown_phase:
            metadata = example['metadata']
            if metadata:
                # encode prompt for encoder prepending
                prompt_tokens = tok.encode(metadata + " ", add_special_tokens=False)

        # step 5: truncate if needed while maintaining sentence alignment
        # truncate encoder if needed (keep </s> tokens for now)
        # account for both prompt_tokens and script_token when calculating target_length
        if len(encoder_content_ids) > self.max_encoder_length - len(prompt_tokens) - 1:
            target_length = self.max_encoder_length - len(prompt_tokens) - 1
            # find last complete sentence that fits
            for i, eos_pos in enumerate(encoder_eos_positions):
                if eos_pos > target_length:
                    # use previous sentence boundary if it exists
                    if i > 0:
                        encoder_content_ids = encoder_content_ids[:encoder_eos_positions[i-1]+1]
                        # also truncate decoder to same number of sentences
                        decoder_content_ids = decoder_content_ids[:decoder_eos_positions[i-1]+1]
                    else:
                        # force truncate if even first sentence is too long
                        encoder_content_ids = encoder_content_ids[:target_length]
                        # truncate decoder similarly
                        if decoder_eos_positions:
                            decoder_content_ids = decoder_content_ids[:min(len(decoder_content_ids), decoder_eos_positions[0]+1)]
                    break

        # step 6: NOW remove </s> tokens from encoder only (after alignment is done)
        # remove all </s> tokens from encoder
        encoder_content_ids_no_eos = [tok_id for tok_id in encoder_content_ids if tok_id != tok.eos_token_id]

        # encoder: [prompt] + corrupted text (no </s>) + <script_token>
        encoder_ids = prompt_tokens + encoder_content_ids_no_eos + [encoder_script_token_id]

        # decoder: <script_token> + content
        decoder_input = [decoder_script_token_id] + decoder_content_ids

        # truncate decoder if needed (preserve script_token only - prompt is in encoder!)
        # use sentence-wise truncation
        if len(decoder_input) > self.max_decoder_length:
            reserved_length = 1  # script_token only
            target_length = self.max_decoder_length - reserved_length
            # find last complete sentence that fits
            for i, eos_pos in enumerate(decoder_eos_positions):
                if eos_pos > target_length:
                    # use previous sentence boundary if it exists
                    if i > 0:
                        decoder_content_ids = decoder_content_ids[:decoder_eos_positions[i-1]+1]
                    else:
                        # force truncate if even first sentence is too long
                        decoder_content_ids = decoder_content_ids[:target_length]
                    break
            decoder_input = [decoder_script_token_id] + decoder_content_ids

        # labels: content + <script_token>
        # NO masking needed since prompt is in encoder, not decoder!
        labels = decoder_content_ids + [decoder_script_token_id]

        # restore original state
        self.morpheme_lambda = original_lambda
        self.infilling_ratio = original_ratio

        # pad/truncate if requested
        if padding:
            max_enc_len = self.max_encoder_length or 512
            max_dec_len = self.max_decoder_length or 512

            # truncate if necessary (transcription can expand sequences)
            encoder_ids = encoder_ids[:max_enc_len]
            decoder_input = decoder_input[:max_dec_len]
            labels = labels[:max_dec_len]

            enc_pad_len = max_enc_len - len(encoder_ids)
            dec_pad_len = max_dec_len - len(decoder_input)
            lbl_pad_len = max_dec_len - len(labels)

            encoder_ids = encoder_ids + [tok.pad_token_id] * enc_pad_len
            decoder_input = decoder_input + [tok.pad_token_id] * dec_pad_len
            labels = labels + [-100] * lbl_pad_len

            # attention masks
            enc_mask = [1] * (max_enc_len - enc_pad_len) + [0] * enc_pad_len
            dec_mask = [1] * (max_dec_len - dec_pad_len) + [0] * dec_pad_len
        else:
            enc_mask = [1] * len(encoder_ids)
            dec_mask = [1] * len(decoder_input)

        result = {
            'input_ids': np.array(encoder_ids, dtype=np.int32),
            'decoder_input_ids': np.array(decoder_input, dtype=np.int32),
            'labels': np.array(labels, dtype=np.int32),
            'attention_mask': np.array(enc_mask, dtype=np.int32),
            'decoder_attention_mask': np.array(dec_mask, dtype=np.int32),
        }

        # propagate _training_mode from example if present
        if '_training_mode' in example:
            result['_training_mode'] = example['_training_mode']

        return result

def get_morphemes_korean(text: str, mecab_tokenizer, preserve_spacing: bool = False) -> List[str]:
    """
    Get morphemes for Korean text using Hanja-aware tokenization.

    Args:
        text: Korean text (may contain Hanja)
        mecab_tokenizer: MeCab tokenizer instance
        preserve_spacing: Whether to preserve spaces (False for denoising)

    Returns:
        List of morpheme strings
    """
    from mecab_morphs_preprocessing import hanja_aware_morpheme_tokenization
    return hanja_aware_morpheme_tokenization(text, mecab_tokenizer, preserve_spacing=preserve_spacing)
