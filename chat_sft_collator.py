#!/usr/bin/env python3
# coding: utf-8
"""
Chat-style SFT collator. Drop-in replacement for UnifiedCollator that
formats encoder/decoder using ChatML-style boundaries (<|system|>, <|user|>,
<|assistant|>, <|end_of_turn|>, <|think|>) instead of the script-tagged
boundaries used by pretraining.

Strict pretraining/SFT token separation: this collator never emits <s>, </s>,
<mask>, <extra_id_N>, or script tags. Pretraining tasks routed through this
collator will raise.

Encoder format:
    <|system|>{system_prompt}<|user|>{user_input}<|end_of_turn|>

Decoder target format (thinking off):
    <|assistant|>{response}<|end_of_turn|>

Decoder target format (thinking on):
    <|think|>{reasoning}<|assistant|>{response}<|end_of_turn|>

Right-shift convention (matches T5Gemma 2025): the decoder is primed with
<|think|> (when reasoning) or <|assistant|>; turn closes with <|end_of_turn|>.
decoder_input_ids = target[:-1], labels = target[1:]; both length-decoder_max
after padding.

Inherits all dataset/streaming/sampling/packing infrastructure from
UnifiedCollator. The 9 task-specific stub handlers (_handle_nli,
_handle_topic_classification, etc.) keep working unchanged because they all
delegate to _handle_supervised under the hood, which is overridden here.
"""

import logging
import numpy as np
from transformers import BatchEncoding

from unified_collator import UnifiedCollator

logger = logging.getLogger(__name__)


class ChatSFTCollator(UnifiedCollator):
    """SFT collator with chat-template encoder/decoder formatting."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_chat_token_ids()

    def _init_chat_token_ids(self):
        """Resolve SFT token ids from the tokenizer. Errors loudly if missing."""
        tok = self.tokenizer

        def _resolve(attr_name, token_str):
            # prefer Han2HanTokenizer's dedicated attributes (already cached at init)
            if hasattr(tok, attr_name):
                tid = getattr(tok, attr_name)
                if tid is not None:
                    return tid
            # fall back to standard HF lookup
            tid = (
                tok.convert_tokens_to_ids(token_str)
                if hasattr(tok, 'convert_tokens_to_ids') else None
            )
            unk = getattr(tok, 'unk_token_id', None)
            if tid is None or tid == unk:
                raise ValueError(
                    f"ChatSFTCollator requires {token_str!r} in the tokenizer "
                    f"vocabulary (got id={tid}, unk_id={unk}). For Han2Han, "
                    f"regenerate with prepare_multilingual_tokenizer.py. For "
                    f"the retired Han2HanTokenizer, call map_chat_tokens() "
                    f"after loading to alias chat tokens onto sentinel slots."
                )
            return tid

        self.system_id = _resolve('system_token_id', '<|system|>')
        self.user_id = _resolve('user_token_id', '<|user|>')
        self.assistant_id = _resolve('assistant_token_id', '<|assistant|>')
        self.end_of_turn_id = _resolve('end_of_turn_token_id', '<|end_of_turn|>')
        self.think_id = _resolve('think_token_id', '<|think|>')

        logger.info(
            "ChatSFTCollator token ids: "
            f"system={self.system_id}, user={self.user_id}, "
            f"assistant={self.assistant_id}, end_of_turn={self.end_of_turn_id}, "
            f"think={self.think_id}"
        )

    def _handle_unsupervised_pretraining(self, examples, **kwargs):
        raise ValueError(
            "ChatSFTCollator does not support pretraining tasks. Pretraining "
            "examples should not be routed through this collator. Check "
            "mode_ratios / sampling_ratios in your config -- all denoising "
            "ratios must be 0 for SFT runs, and source data should not "
            "produce data_type='denoising' / 'continuation' / "
            "'morpheme_denoising' examples."
        )

    def _handle_supervised(self, examples, cooldown_phase=False, bucket_idx=None,
                           tokenizer=None, padding=True, return_source=False,
                           **kwargs):
        """Chat-format supervised handler. Replaces parent's script-tag version.

        Per-example record contract (any of):
          - {'original_text', 'labels', ...} legacy SFT shape used by all
            UnifiedCollator task handlers. 'metadata' (when present)
            becomes the system prompt; 'original_text' becomes the user message;
            'labels' becomes the assistant response.
          - 'thinking' (optional): reasoning string that, when present, gets
            wrapped between <|think|> and <|assistant|> in the decoder target.

        Output dict matches the existing collator contract:
            input_ids, decoder_input_ids, labels, attention_mask,
            decoder_attention_mask  -- all np.int32, padded to encoder_max /
            decoder_max respectively.
        """
        if tokenizer is None:
            tokenizer = self.tokenizer

        # batch handling
        if isinstance(examples, list):
            batch = [
                self._handle_supervised(
                    ex, cooldown_phase, bucket_idx, tokenizer, padding,
                    return_source=False
                )
                for ex in examples
            ]
            keys = batch[0].keys()
            stacked = {k: np.stack([b[k] for b in batch]) for k in keys}
            result = BatchEncoding(stacked).data
            if return_source:
                source = examples[0].get('source')
                if source is None:
                    raise ValueError(
                        f"No 'source' field in supervised example: "
                        f"{list(examples[0].keys())}"
                    )
                return result, source
            return result

        # single example
        system_text = examples.get('metadata') or ''
        user_text = examples['original_text']
        assistant_text = examples['labels']
        thinking_text = examples.get('thinking')

        # determine max lengths (matches parent's bucketing logic)
        encoder_max = getattr(self, 'max_encoder_length', self.max_length)
        decoder_max = getattr(self, 'max_decoder_length', self.max_length)
        if (
            self.use_bucketing
            and bucket_idx is not None
            and bucket_idx < len(self.bucket_sizes)
        ):
            encoder_max = min(self.bucket_sizes[bucket_idx], encoder_max)
            decoder_max = min(self.bucket_sizes[bucket_idx], decoder_max)

        # tokenize content (no special tokens added by tokenizer; we add ours)
        sys_ids = (
            tokenizer(system_text, add_special_tokens=False).input_ids
            if system_text else []
        )
        usr_ids = tokenizer(user_text, add_special_tokens=False).input_ids
        rsp_ids = tokenizer(assistant_text, add_special_tokens=False).input_ids
        thk_ids = (
            tokenizer(thinking_text, add_special_tokens=False).input_ids
            if thinking_text else []
        )

        # build encoder: <|system|>{sys}<|user|>{usr}<|end_of_turn|>
        # role tokens are openers; <|end_of_turn|> appears once at the tail
        encoder_ids = []
        if sys_ids:
            encoder_ids.append(self.system_id)
            encoder_ids.extend(sys_ids)
        encoder_ids.append(self.user_id)
        encoder_ids.extend(usr_ids)
        encoder_ids.append(self.end_of_turn_id)

        # truncate encoder; preserve trailing <|end_of_turn|>
        if len(encoder_ids) > encoder_max:
            encoder_ids = encoder_ids[: encoder_max - 1] + [self.end_of_turn_id]

        # build decoder target sequence of length L+1; slice into
        # decoder_input_ids (length L) and labels (length L)
        if thk_ids:
            target = (
                [self.think_id] + thk_ids
                + [self.assistant_id] + rsp_ids
                + [self.end_of_turn_id]
            )
        else:
            target = [self.assistant_id] + rsp_ids + [self.end_of_turn_id]

        # truncate target; preserve trailing <|end_of_turn|>
        target_max = decoder_max + 1
        if len(target) > target_max:
            target = target[: target_max - 1] + [self.end_of_turn_id]

        decoder_input_ids = target[:-1]
        labels = target[1:]

        # padding
        pad_id = tokenizer.pad_token_id
        if padding:
            encoder_ids = (
                encoder_ids + [pad_id] * (encoder_max - len(encoder_ids))
            )
            decoder_input_ids = (
                decoder_input_ids + [pad_id] * (decoder_max - len(decoder_input_ids))
            )
            labels = labels + [-100] * (decoder_max - len(labels))

        attention_mask = [1 if x != pad_id else 0 for x in encoder_ids]
        decoder_attention_mask = [
            1 if x != pad_id else 0 for x in decoder_input_ids
        ]

        result = {
            'input_ids': np.array(encoder_ids, dtype=np.int32),
            'decoder_input_ids': np.array(decoder_input_ids, dtype=np.int32),
            'labels': np.array(labels, dtype=np.int32),
            'attention_mask': np.array(attention_mask, dtype=np.int32),
            'decoder_attention_mask': np.array(decoder_attention_mask, dtype=np.int32),
        }

        if '_training_mode' in examples:
            result['_training_mode'] = examples['_training_mode']

        if return_source:
            source = examples.get('source')
            if source is None:
                raise ValueError(
                    f"No 'source' field in supervised example: "
                    f"{list(examples.keys())}"
                )
            return result, source
        return result

    def __repr__(self):
        return (
            f"ChatSFTCollator(\n"
            f"  task_prompts={'enabled' if self.use_task_prompts else 'disabled'},\n"
            f"  packing={'enabled' if self.enable_packing else 'disabled'},\n"
            f"  chat_token_ids=(system={self.system_id}, user={self.user_id}, "
            f"assistant={self.assistant_id}, "
            f"end_of_turn={self.end_of_turn_id}, think={self.think_id})\n"
            f")"
        )
