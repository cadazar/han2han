#!/usr/bin/env python3
# coding: utf-8
"""
Unified Han2Han Collator - Smart routing with explicit parent calls.

We inherit from PackedMultilingualCollator and override __call__ to:
1. Detect task type from data structure
2. Route to appropriate parent class method explicitly
3. Add task-specific formatting for new data types
4. Maintain all existing functionality through inheritance
"""

import logging

# import all parent classes explicitly for clear method calls
from phase2_collator import Phase2MixedCollator
from packed_multilingual_collator import PackedMultilingualCollator
from logging_utils import log_from_all_processes
from task_prompts import add_task_prompt_to_example
from han2han_tools import transcribe, has_hanja

logger = logging.getLogger(__name__)


class UnifiedCollator(PackedMultilingualCollator):
    """
    Unified collator that inherits everything but routes __call__ explicitly.

    Key benefits:
    - All existing methods available through inheritance
    - Clear routing in __call__ shows exactly which implementation is used
    - Only need to add new methods for new task types
    - No copy-paste maintenance nightmare
    """

    def __init__(self, *args, **kwargs):
        """Initialize with all parent functionality."""
        # optional: disable task prompts
        self.use_task_prompts = kwargs.pop('use_task_prompts', True)

        # initialize parent with all existing functionality
        super().__init__(*args, **kwargs)

        # task routing configuration
        self.task_handlers = {
            'unsupervised_pretraining': self._handle_unsupervised_pretraining,
            'ocr_correction': self._handle_ocr_correction,
            'temporal_classification': self._handle_temporal_classification,
            'temporal_continuation': self._handle_temporal_continuation,
            'sts': self._handle_sts,
            'transcription': self._handle_transcription,
            'topic_classification': self._handle_topic_classification,
            'nli': self._handle_nli,
            'instruction_following': self._handle_instruction_following,
            'multiple_choice': self._handle_multiple_choice,
            'summarization': self._handle_summarization,
            'cot_reasoning': self._handle_cot_reasoning,
        }

    def _detect_task_type(self, examples):
        """Detect task type from example structure."""
        # handle both single example (dict) and batch (list of dicts)
        if isinstance(examples, dict):
            example = examples
        elif isinstance(examples, list) and len(examples) > 0 and isinstance(examples[0], dict):
            example = examples[0]
        else:
            raise ValueError(
                f"Cannot detect task type: examples must be dict or non-empty list of dicts, "
                f"got {type(examples)}"
            )

        # check for explicit task type markers
        if 'task_type' in example:
            task_type = example['task_type']
        elif 'data_type' in example and example['data_type'] != 'mixed':
            task_type = example['data_type']
        else:
            task_type = None

        # map legacy data_type values to new handler names
        if task_type in ('denoising', 'continuation', 'morpheme_denoising'):
            return 'unsupervised_pretraining'
        elif task_type:
            return task_type

        # infer from data structure (or when data_type='mixed')
        # note: we don't infer temporal_classification from year field anymore -
        # mixed data with year goes to unsupervised_pretraining where
        # temporal_continuation_ratio controls routing to temporal_continuation
        if 'corrected' in example:
            return 'ocr_correction'
        elif 'sentence1' in example and 'sentence2' in example:
            return 'sts'
        elif 'transcribed' in example:
            return 'transcription'
        else:
            # default to unsupervised pretraining (denoising/continuation via Phase2MixedCollator)
            return 'unsupervised_pretraining'

    # =========================================================================
    # === Task-Specific Handlers (route to appropriate parent __call__) ===
    # =========================================================================

    def _handle_supervised(self, examples, cooldown_phase=False, bucket_idx=None,
                          tokenizer=None, padding=True, return_source=False, **kwargs):
        """
        Simple supervised task handler for all seq2seq tasks.

        Format:
        - encoder: [prompt] [input_text] <script>
        - decoder: <script> [label_text]
        - labels: [label_text] <script>

        No sentence splitting, no corruption, no permutation.

        Args:
            padding: if False, skip padding (for packing later)
            return_source: if True, return (result, source) tuple
        """
        import numpy as np
        from transformers import BatchEncoding

        if tokenizer is None:
            tokenizer = self.tokenizer

        # handle batch
        if isinstance(examples, list):
            batch = [self._handle_supervised(ex, cooldown_phase, bucket_idx, tokenizer, padding, return_source=False) for ex in examples]
            batch_keys = batch[0].keys()
            batch = {k: np.stack([b[k] for b in batch]) for k in batch_keys}
            result = BatchEncoding(batch).data
            if return_source:
                source = examples[0].get('source')
                if source is None:
                    raise ValueError(f"No 'source' field in supervised example: {list(examples[0].keys())}")
                return result, source
            return result

        # single example
        input_text = examples['original_text']
        label_text = examples['labels']
        metadata = examples.get('metadata', '')

        # determine max lengths (use separate encoder/decoder lengths for SFT efficiency)
        encoder_max = getattr(self, 'max_encoder_length', self.max_length)
        decoder_max = getattr(self, 'max_decoder_length', self.max_length)
        if self.use_bucketing and bucket_idx is not None and bucket_idx < len(self.bucket_sizes):
            encoder_max = min(self.bucket_sizes[bucket_idx], encoder_max)
            decoder_max = min(self.bucket_sizes[bucket_idx], decoder_max)

        # detect input script
        if has_hanja(input_text):
            encoder_token_id = tokenizer.convert_tokens_to_ids('<hanja>')
        else:
            encoder_token_id = tokenizer.convert_tokens_to_ids('<hangul>')

        # determine decoder token from LABELS text
        if has_hanja(label_text):
            decoder_token_id = tokenizer.convert_tokens_to_ids('<hanja>')
        else:
            decoder_token_id = tokenizer.convert_tokens_to_ids('<hangul>')

        # tokenize
        input_ids = tokenizer(input_text, add_special_tokens=False).input_ids
        label_ids = tokenizer(label_text, add_special_tokens=False).input_ids
        metadata_ids = tokenizer(metadata + " ", add_special_tokens=False).input_ids if not cooldown_phase else []

        # truncate text BEFORE adding special tokens to preserve them
        # reserve 1 tokens for the script token
        encoder_text_max = encoder_max - 1
        decoder_text_max = decoder_max - 1
        if len(metadata_ids) + len(input_ids) > encoder_text_max:
            available_for_input = encoder_text_max - len(metadata_ids)
            input_ids = input_ids[:available_for_input]
        if len(label_ids) > decoder_text_max:
            label_ids = label_ids[:decoder_text_max]

        # build encoder: [metadata] [input] <encoder_token>
        encoder_ids = metadata_ids + input_ids + [encoder_token_id]

        # build decoder input: <decoder_token> [labels]
        decoder_input_ids = [decoder_token_id] + label_ids

        # build labels: [labels] <decoder_token>
        labels = label_ids + [decoder_token_id]

        # pad to respective max lengths (if padding enabled)
        if padding:
            encoder_ids = encoder_ids + [tokenizer.pad_token_id] * (encoder_max - len(encoder_ids))
            decoder_input_ids = decoder_input_ids + [tokenizer.pad_token_id] * (decoder_max - len(decoder_input_ids))
            labels = labels + [-100] * (decoder_max - len(labels))

        # create batch dict
        result = {
            'input_ids': np.array(encoder_ids, dtype=np.int32),
            'decoder_input_ids': np.array(decoder_input_ids, dtype=np.int32),
            'labels': np.array(labels, dtype=np.int32),
            'attention_mask': np.array([1 if x != tokenizer.pad_token_id else 0 for x in encoder_ids], dtype=np.int32),
            'decoder_attention_mask': np.array([1 if x != tokenizer.pad_token_id else 0 for x in decoder_input_ids], dtype=np.int32),
        }

        # propagate _training_mode from example if present
        if '_training_mode' in examples:
            result['_training_mode'] = examples['_training_mode']

        if return_source:
            source = examples.get('source')
            if source is None:
                raise ValueError(f"No 'source' field in supervised example: {list(examples.keys())}")
            return result, source
        return result

    def _handle_unsupervised_pretraining(self, examples, cooldown_phase=False, bucket_idx=None,
                                         tokenizer=None, morpheme_tokenizers=None,
                                         return_source=False, padding=True, **kwargs):
        """
        Unsupervised pretraining - denoising vs continuation sampled by Phase2MixedCollator.

        Phase2MixedCollator will randomly choose between:
        - denoising (regular, with morpheme-aware or token-level masking)
        - denoising_heavy (X-denoiser, 50% corruption, token-level only)
        - continuation (prefix LM style, S-denoiser)

        Based on mode_ratios (UL2-style default: 40% denoising, 40% heavy, 20% continuation).
        """
        # Phase2MixedCollator handles mode sampling and task prompts internally
        return Phase2MixedCollator.__call__(
            self, examples,
            cooldown_phase=cooldown_phase,
            bucket_idx=bucket_idx,
            tokenizer=tokenizer,
            morpheme_tokenizers=morpheme_tokenizers,
            return_source=return_source,
            padding=padding
        )

    def _handle_ocr_correction(self, examples, **kwargs):
        """OCR correction task - noisy to clean pairs."""
        is_single = isinstance(examples, dict)
        processed_examples = []

        # handle batch
        if isinstance(examples, list):
            ex_list = examples
        else:
            ex_list = [examples]

        for ex in ex_list:

            # add task prompt if enabled
            if self.use_task_prompts:
                ex, _ = add_task_prompt_to_example(ex, 'ocr_correction')

            # OCR correction: noisy text -> clean text
            # handle both pre-transformed (original_text/target) and raw (text/corrected) formats
            if 'original_text' in ex and 'target' in ex:
                # already transformed by _transform_example
                processed_examples.append({
                    'original_text': ex['original_text'],  # noisy OCR text (input)
                    'labels': ex['target'],  # clean text (target)
                    'metadata': ex['metadata'],  # preserve prompt
                    'source': ex['source'],
                    '_training_mode': 'ocr_correction',
                })
            elif 'text' in ex and 'corrected' in ex:
                # raw format
                processed_examples.append({
                    'original_text': ex['text'],  # noisy OCR text (input)
                    'labels': ex['corrected'],  # clean text (target)
                    'metadata': ex['metadata'],  # preserve prompt
                    'source': ex['source'],
                    '_training_mode': 'ocr_correction',
                })
            else:
                raise ValueError(
                    f"OCR example has neither (original_text+target) nor (text+corrected) fields. "
                    f"Available keys: {list(ex.keys())}. "
                    f"If this is OCR quality metadata (mean_wc_ocr), use data_type='denoising' instead."
                )

        # use supervised handler - pass single dict if input was single, else pass list
        if is_single and len(processed_examples) == 1:
            return self._handle_supervised(processed_examples[0], **kwargs)
        else:
            return self._handle_supervised(processed_examples, **kwargs)

    def _handle_temporal_classification(self, examples, **kwargs):
        """Simple temporal classification: estimate the year from text.

        For SFT/evaluation - just predict the year as a label, no continuation.
        Uses the temporal_classification prompt (e.g., "이 텍스트가 작성된 연도를 추정하시오:")
        """
        is_single = isinstance(examples, dict)
        processed_examples = []

        if isinstance(examples, list):
            ex_list = examples
        else:
            ex_list = [examples]

        for ex in ex_list:
            input_text = ex.get('original_text', ex.get('text', ex.get('input_text', '')))
            if has_hanja(input_text):
                input_text = transcribe(input_text)

            year_or_date = ex.get('year') or ex.get('date')
            year = None
            if year_or_date is not None and year_or_date != '':
                if hasattr(year_or_date, 'year'):
                    year = year_or_date.year
                else:
                    try:
                        year = int(year_or_date)
                    except (ValueError, TypeError):
                        year = None

            if year is None:
                continue

            label_text = str(year)

            if self.use_task_prompts:
                ex, _ = add_task_prompt_to_example(ex, 'temporal_classification')

            processed_examples.append({
                'original_text': input_text,
                'labels': label_text,
                'metadata': ex.get('metadata', ''),
                'source': ex.get('source', 'korean_temporal'),
                '_training_mode': 'temporal_classification',
            })

        if not processed_examples:
            return self._create_empty_batch()

        if is_single and len(processed_examples) == 1:
            return self._handle_supervised(processed_examples[0], **kwargs)
        else:
            return self._handle_supervised(processed_examples, **kwargs)

    def _handle_temporal_continuation(self, examples, cooldown_phase=False, return_source=False, **kwargs):
        """Temporal continuation: continue text in period-appropriate style + estimate year.

        For pretraining - trains the model to continue text considering its time period
        and then estimate the year. Falls back to unsupervised pretraining (denoising)
        if year is missing - we don't want to waste the text data.
        """
        is_single = isinstance(examples, dict)
        padding = kwargs.get('padding', True)

        # handle batch
        if isinstance(examples, list):
            ex_list = examples
        else:
            ex_list = [examples]

        results_with_year = []
        examples_without_year = []

        for ex in ex_list:

            # get year from year or date field
            year_or_date = ex.get('year') or ex.get('date')
            year = None

            if year_or_date is not None and year_or_date != '':
                # extract year from datetime objects or convert to int
                if hasattr(year_or_date, 'year'):
                    year = year_or_date.year
                else:
                    try:
                        year = int(year_or_date)
                    except (ValueError, TypeError):
                        year = None

            # if we have valid year, directly collate as temporal_continuation
            if year is not None:
                input_text = ex.get('original_text', ex.get('text', ''))
                if not input_text:
                    examples_without_year.append(ex)
                    continue

                # add temporal_continuation prompt if enabled
                metadata = ex['metadata']
                if self.use_task_prompts and not cooldown_phase:
                    ex, _ = add_task_prompt_to_example(ex, 'temporal_continuation')
                    metadata = ex['metadata']

                # prepare and collate directly (preserve _data_source for sub-buffer tracking)
                prepared = {
                    'text': input_text,
                    'original_text': input_text,
                    'year': year,
                    'metadata': metadata,
                    '_training_mode': 'temporal_continuation',
                    '_data_source': ex.get('_data_source')
                }
                result = self._collate_temporal_continuation(prepared, cooldown_phase, padding)
                if result is not None:
                    result['_training_mode'] = 'temporal_continuation'
                    if ex.get('_data_source'):
                        result['_data_source'] = ex['_data_source']
                    results_with_year.append(result)
            else:
                # no valid year - route to denoising
                ex.pop('data_type', None)
                examples_without_year.append(ex)

        # if we have results from temporal_continuation, return the first one
        if results_with_year:
            result = results_with_year[0]
            if return_source:
                source = ex_list[0].get('source')
                if source is None:
                    raise ValueError(f"No 'source' field in temporal example: {list(ex_list[0].keys())}")
                return result, source
            return result

        # otherwise fall back to unsupervised pretraining for examples without year
        if examples_without_year:
            if is_single:
                return self._handle_unsupervised_pretraining(
                    examples_without_year[0], cooldown_phase=cooldown_phase,
                    return_source=return_source, **kwargs)
            else:
                return self._handle_unsupervised_pretraining(
                    examples_without_year, cooldown_phase=cooldown_phase,
                    return_source=return_source, **kwargs)

        # fallback
        return self._create_empty_batch()

    def _handle_sts(self, examples, **kwargs):
        """Semantic textual similarity - sentence pairs to scores."""
        is_single = isinstance(examples, dict)
        processed_examples = []

        # handle batch
        if isinstance(examples, list):
            ex_list = examples
        else:
            ex_list = [examples]

        for ex in ex_list:
            # add task prompt if enabled
            if self.use_task_prompts:
                ex, _ = add_task_prompt_to_example(ex, 'sts')

            if 'input_text' in ex and 'target_text' in ex:
                # already formatted from preprocessing
                processed_examples.append({
                    'original_text': ex['input_text'],  # formatted input
                    'labels': ex['target_text'],  # similarity score (target)
                    'source': ex['source'],
                    'metadata': ex['metadata'],  # preserve prompt
                    '_training_mode': 'sts',
                })
            elif 'sentence1' in ex and 'sentence2' in ex:
                # need to format
                input_text = f"sentence1: {ex['sentence1']} sentence2: {ex['sentence2']}"
                score = ex.get('rounded_score', ex.get('label', ''))
                processed_examples.append({
                    'original_text': input_text,  # formatted input
                    'labels': f"{score:.1f}" if isinstance(score, float) else str(score),  # score (target)
                    'source': ex['source'],
                    'metadata': ex['metadata'],  # preserve prompt
                    '_training_mode': 'sts',
                })

        # use supervised handler - pass single dict if input was single, else pass list
        if is_single and len(processed_examples) == 1:
            return self._handle_supervised(processed_examples[0], **kwargs)
        else:
            return self._handle_supervised(processed_examples, **kwargs)

    def _handle_transcription(self, examples, cooldown_phase=False, **kwargs):
        """Transcription task - bidirectional Hanja<->Hangul."""
        is_single = isinstance(examples, dict)
        processed_examples = []

        # handle batch
        if isinstance(examples, list):
            ex_list = examples
        else:
            ex_list = [examples]

        for ex in ex_list:
            # add task prompt if enabled and not in cooldown (eval) phase
            direction = None
            if self.use_task_prompts and not cooldown_phase:
                ex, direction = add_task_prompt_to_example(ex, 'transcription')

            # after transform: text_field='target' becomes 'original_text'
            text_to_transcribe = ex.get('original_text', ex.get('text', ''))

            # apply transcription based on direction
            if direction == 'transcription_hanja_to_hangul':
                # hanja -> hangul: original as input, transcribed as label
                input_text = text_to_transcribe
                label_text = transcribe(text_to_transcribe)
            elif direction == 'transcription_hangul_to_hanja':
                # hangul -> hanja: transcribed as input, original as label
                input_text = transcribe(text_to_transcribe)
                label_text = text_to_transcribe
            else:
                # no direction (shouldn't happen with prompts enabled)
                input_text = text_to_transcribe
                label_text = transcribe(text_to_transcribe)

            processed_examples.append({
                'original_text': input_text,
                'labels': label_text,
                'metadata': ex['metadata'],  # preserve prompt
                'source': ex['source'],
                '_training_mode': 'transcription',
            })

        # use supervised handler - pass single dict if input was single, else pass list
        # IMPORTANT: forward cooldown_phase explicitly since it was captured by our param
        if is_single and len(processed_examples) == 1:
            return self._handle_supervised(processed_examples[0], cooldown_phase=cooldown_phase, **kwargs)
        else:
            return self._handle_supervised(processed_examples, cooldown_phase=cooldown_phase, **kwargs)

    def _handle_topic_classification(self, examples, **kwargs):
        """Topic classification task (KLUE YNAT) - text to category label."""
        is_single = isinstance(examples, dict)
        processed_examples = []

        if isinstance(examples, list):
            ex_list = examples
        else:
            ex_list = [examples]

        for ex in ex_list:
            input_text = ex.get('original_text', ex.get('input_text', ''))
            label_text = ex.get('target', ex.get('target_text', ex.get('labels', '')))

            if self.use_task_prompts:
                ex, _ = add_task_prompt_to_example(ex, 'classification')

            processed_examples.append({
                'original_text': input_text,
                'labels': label_text,
                'metadata': ex.get('metadata', ''),
                'source': ex.get('source', 'klue_ynat'),
                '_training_mode': 'topic_classification',
            })

        if is_single and len(processed_examples) == 1:
            return self._handle_supervised(processed_examples[0], **kwargs)
        else:
            return self._handle_supervised(processed_examples, **kwargs)

    def _handle_nli(self, examples, **kwargs):
        """Natural language inference task (KLUE NLI) - premise+hypothesis to relation."""
        is_single = isinstance(examples, dict)
        processed_examples = []

        if isinstance(examples, list):
            ex_list = examples
        else:
            ex_list = [examples]

        for ex in ex_list:
            input_text = ex.get('original_text', ex.get('input_text', ''))
            label_text = ex.get('target', ex.get('target_text', ex.get('labels', '')))

            if self.use_task_prompts:
                ex, _ = add_task_prompt_to_example(ex, 'nli')

            processed_examples.append({
                'original_text': input_text,
                'labels': label_text,
                'metadata': ex.get('metadata', ''),
                'source': ex.get('source', 'klue_nli'),
                '_training_mode': 'nli',
            })

        if is_single and len(processed_examples) == 1:
            return self._handle_supervised(processed_examples[0], **kwargs)
        else:
            return self._handle_supervised(processed_examples, **kwargs)

    def _handle_multiple_choice(self, examples, **kwargs):
        """Multiple-choice QA task - question + one candidate answer.

        Expected fields: original_text (question), labels (candidate answer).
        Used by MCLogProbCallback which passes each (question, candidate) pair
        as a separate example to get properly tokenized encoder/decoder arrays.
        """
        is_single = isinstance(examples, dict)
        processed_examples = []

        if isinstance(examples, list):
            ex_list = examples
        else:
            ex_list = [examples]

        for ex in ex_list:
            input_text = ex.get('original_text', ex.get('input_text', ''))
            label_text = ex.get('labels', ex.get('target_text', ''))

            if not input_text or not label_text:
                raise ValueError(
                    f"Multiple-choice example missing 'original_text' or 'labels': "
                    f"{list(ex.keys())}"
                )

            if self.use_task_prompts:
                ex, _ = add_task_prompt_to_example(ex, 'multiple_choice')

            processed_examples.append({
                'original_text': input_text,
                'labels': label_text,
                'metadata': ex.get('metadata', ''),
                'source': ex.get('source', 'mc_eval'),
                '_training_mode': 'multiple_choice',
            })

        if is_single and len(processed_examples) == 1:
            return self._handle_supervised(processed_examples[0], **kwargs)
        return self._handle_supervised(processed_examples, **kwargs)

    def _handle_instruction_following(self, examples, **kwargs):
        """Instruction-following task - instruction+input to output.

        When the per-row metadata field is empty (heegyu/open-korean-
        instructions has no <sys> tag for most rows), fall back to a
        sampled generic helpful-assistant prompt from the cot_reasoning
        family so the encoder never emits a bare <|user|> turn. The
        cot_reasoning family is intentionally generic ("you are a helpful
        Korean assistant", "answer the question", etc.) rather than
        reasoning-flavored -- reasoning behavior is toggled on the
        decoder side via <|assistant|> vs <|think|> start tokens, not
        via the encoder prompt.
        """
        is_single = isinstance(examples, dict)
        ex_list = examples if isinstance(examples, list) else [examples]
        processed = []

        for ex in ex_list:
            instruction = ex.get('metadata') or ''
            input_text = ex['original_text']
            label_text = ex['target_text']

            if not instruction and self.use_task_prompts:
                ex, _ = add_task_prompt_to_example(
                    ex, 'cot_reasoning'
                )
                instruction = ex['metadata']

            processed.append({
                'original_text': input_text,
                'labels': label_text,
                'metadata': instruction,
                'source': ex.get('source', 'open_kor_instructions'),
                '_training_mode': 'instruction_following',
            })

        if is_single and len(processed) == 1:
            return self._handle_supervised(processed[0], **kwargs)
        return self._handle_supervised(processed, **kwargs)

    def _handle_summarization(self, examples, **kwargs):
        """AIHub Korean summarization - document to summary.

        The raw metadata field carries the AIHub domain tag (news, legal,
        editorial, books, papers, patents, reports), which is not a natural
        prompt; replace it with a sampled summarization prompt drawn from
        TASK_PROMPTS['summarization'] so the chat collator can route it into
        the system slot.
        """
        is_single = isinstance(examples, dict)
        ex_list = examples if isinstance(examples, list) else [examples]
        processed = []

        for ex in ex_list:
            if self.use_task_prompts:
                ex, _ = add_task_prompt_to_example(
                    ex, 'summarization',
                )

            input_text = ex['original_text']
            label_text = ex.get('target') or ex.get('target_text', '')

            processed.append({
                'original_text': input_text,
                'labels': label_text,
                'metadata': ex.get('metadata', ''),
                'source': ex.get('source', 'aihub_summarization'),
                '_training_mode': 'summarization',
            })

        if is_single and len(processed) == 1:
            return self._handle_supervised(processed[0], **kwargs)
        return self._handle_supervised(processed, **kwargs)

    def _handle_cot_reasoning(self, examples, **kwargs):
        """KAIST Multilingual-CoT reasoning - question to answer with optional
        rationale.

        Source/user text already contains the FLAN-style task instruction, so
        the system slot gets a generic helpful-assistant prompt sampled from
        TASK_PROMPTS['cot_reasoning']. The 'thinking' field, when present,
        gets threaded through to ChatSFTCollator which wraps it between
        <|think|> and <|assistant|>. At inference, decoder start token
        (<|think|> vs <|assistant|>) toggles reasoning.
        """
        is_single = isinstance(examples, dict)
        ex_list = examples if isinstance(examples, list) else [examples]
        processed = []

        for ex in ex_list:
            if self.use_task_prompts:
                ex, _ = add_task_prompt_to_example(
                    ex, 'cot_reasoning'
                )

            input_text = ex['original_text']
            label_text = ex.get('target') or ex.get('target_text', '')
            thinking_text = ex.get('thinking') or ''

            record = {
                'original_text': input_text,
                'labels': label_text,
                'metadata': ex.get('metadata', ''),
                'source': ex.get('source', 'kaist_cot_ko'),
                '_training_mode': 'cot_reasoning',
            }
            if thinking_text:
                record['thinking'] = thinking_text
            processed.append(record)

        if is_single and len(processed) == 1:
            return self._handle_supervised(processed[0], **kwargs)
        return self._handle_supervised(processed, **kwargs)

    # =========================================================================
    # === Main __call__ with Task Routing ===
    # =========================================================================

    def __call__(self, examples, cooldown_phase=False, bucket_idx=None,
                 tokenizer=None, **kwargs):
        """
        Task routing collator - detects task type and routes to handler.

        Routes both batches (list) and single examples (dict), but skips
        routing for already-tokenized examples to avoid re-entry during packing.
        """

        # override tokenizer if provided
        if tokenizer is not None:
            self.tokenizer = tokenizer

        # check if already tokenized (from packing or internal operations)
        is_tokenized = False
        if isinstance(examples, dict):
            is_tokenized = 'input_ids' in examples
        elif isinstance(examples, list) and len(examples) > 0:
            is_tokenized = 'input_ids' in examples[0]

        # task routing for untokenized examples (both batch and single)
        if not is_tokenized:
            task_type = self._detect_task_type(examples)

            if task_type not in self.task_handlers:
                raise ValueError(
                    f"Task type '{task_type}' not supported. "
                    f"Available handlers: {list(self.task_handlers.keys())}"
                )

            log_from_all_processes(logger, 'debug', f"[TASK ROUTING] {task_type}")
            return self.task_handlers[task_type](
                examples,
                cooldown_phase=cooldown_phase,
                bucket_idx=bucket_idx,
                tokenizer=tokenizer,
                **kwargs
            )

        # already tokenized - pass through to parent for final processing
        return PackedMultilingualCollator.__call__(
            self, examples, cooldown_phase=cooldown_phase,
            bucket_idx=bucket_idx, tokenizer=tokenizer, **kwargs
        )

    def __repr__(self):
        """Clear representation showing configuration."""
        return (
            f"UnifiedCollator(\n"
            f"  task_prompts={'enabled' if self.use_task_prompts else 'disabled'},\n"
            f"  packing={'enabled' if self.enable_packing else 'disabled'},\n"
            f"  modes={list(self.mode_ratios.keys()) if self.mode_ratios else 'none'},\n"
            f"  morpheme_masking={self.use_morpheme_masking:.2f},\n"
            f"  sentence_perm={'enabled' if self.sentence_permutation else 'disabled'}\n"
            f")"
        )