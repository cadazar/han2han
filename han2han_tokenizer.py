#!/usr/bin/env python3
# coding: utf-8
"""
Han2Han Tokenizer - Clean SentencePiece Wrapper

A clean tokenizer that wraps SentencePiece directly without HuggingFace complexity.
Based on Han2HanTokenizer design for reliable, predictable tokenization.
"""

import os
import regex as re
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import sentencepiece as spm
from transformers import BatchEncoding

from logging_utils import log_from_main_process

import logging
logger = logging.getLogger(__name__)

try:
    from sentencepiece import sentencepiece_model_pb2
    HAS_SPM_PROTOBUF = True
except ImportError:
    HAS_SPM_PROTOBUF = False
    logger.warning("sentencepiece_model_pb2 not available - vocabulary extension disabled")

_BYTE_PAT = re.compile(r"<0x([0-9A-Fa-f]{2})>")


def _unbytefall(text: str, *, encoding: str = "utf-8", errors: str = "strict") -> str:
    """Convert byte placeholders like '<0xE6><0xA9>' back to characters."""
    parts = []
    last_end = 0
    byte_buffer = b""

    for match in _BYTE_PAT.finditer(text):
        if match.start() > last_end:
            if byte_buffer:
                try:
                    parts.append(byte_buffer.decode(encoding, errors=errors))
                except UnicodeDecodeError:
                    parts.append(byte_buffer.decode(encoding, errors="replace"))
                byte_buffer = b""
            parts.append(text[last_end:match.start()])

        hex_str = match.group(1)
        byte_val = int(hex_str, 16)
        byte_buffer += bytes([byte_val])
        last_end = match.end()

    if byte_buffer:
        try:
            parts.append(byte_buffer.decode(encoding, errors=errors))
        except UnicodeDecodeError:
            parts.append(byte_buffer.decode(encoding, errors="replace"))

    if last_end < len(text):
        parts.append(text[last_end:])

    return "".join(parts)


class Han2HanTokenizer:
    """
    Clean Han2Han tokenizer wrapper around SentencePiece.

    Provides the same interface as HuggingFace tokenizers but without
    all the complexity. Directly uses SentencePiece for reliable,
    predictable tokenization.

    Automatically detects sentinel tokens and byte fallback from the
    loaded SentencePiece model to support both old and new tokenizers.
    """

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, model_path: str, vocab_size: int = None):
        """
        Initialize with SentencePiece model path.

        Args:
            model_path: Path to the .model file
            vocab_size: Override vocab size (auto-detected from model if None)
        """
        self.model_path = model_path

        self._tokenizer = spm.SentencePieceProcessor()
        self._tokenizer.load(model_path)

        self._vocab_size = vocab_size or self._tokenizer.get_piece_size()

        # detect sentinel tokens and byte fallback from the model
        self._detect_special_tokens()

        # probe SPM for known specials so the same class loads both the v1
        # tokenizer (mask=4, hanja=5, hangul=6 via control_symbols) and v2
        # (hanja=4, hangul=5, mask=6, plus chat/tool/reserved as real pieces)
        unk = self._tokenizer.unk_id()

        def _probe(piece):
            tid = self._tokenizer.piece_to_id(piece)
            return tid if (tid != unk or piece == "<unk>") else None

        self._special_tokens = {}
        for piece in ("<pad>", "<unk>", "<s>", "</s>", "<mask>",
                      "<hanja>", "<hangul>"):
            tid = _probe(piece)
            if tid is not None:
                self._special_tokens[piece] = tid

        # v2 chat/tool/reserved -- absent in v1, present in v2
        for piece in ("<|system|>", "<|user|>", "<|assistant|>",
                      "<|end_of_turn|>", "<|think|>",
                      "<|tool_call|>", "<|tool_result|>",
                      "<|reserved_0|>", "<|reserved_1|>", "<|reserved_2|>"):
            tid = _probe(piece)
            if tid is not None:
                self._special_tokens[piece] = tid

        # add sentinel tokens if present in model
        if self._has_sentinel_tokens:
            for i in range(self._num_sentinel_tokens):
                self._special_tokens[f'<extra_id_{i}>'] = self._sentinel_base_id + i

        self._id_to_token = {v: k for k, v in self._special_tokens.items()}

        self.pad_token = "<pad>"
        self.pad_token_id = self._special_tokens["<pad>"]
        self.unk_token = "<unk>"
        self.unk_token_id = self._special_tokens["<unk>"]
        self.bos_token = "<s>"
        self.bos_token_id = self._special_tokens["<s>"]
        self.eos_token = "</s>"
        self.eos_token_id = self._special_tokens["</s>"]
        self.mask_token = "<mask>"
        self.mask_token_id = self._special_tokens["<mask>"]

        self._hanja_token_id = self._special_tokens["<hanja>"]
        self._hangul_token_id = self._special_tokens["<hangul>"]

        # v2: expose chat-token convenience attrs when baked into the SPM
        # (v1 still relies on map_chat_tokens() to alias these onto sentinels)
        for attr_piece in (
            ("system_token", "<|system|>"),
            ("user_token", "<|user|>"),
            ("assistant_token", "<|assistant|>"),
            ("end_of_turn_token", "<|end_of_turn|>"),
            ("think_token", "<|think|>"),
        ):
            attr, piece = attr_piece
            if piece in self._special_tokens:
                setattr(self, attr, piece)
                setattr(self, f"{attr}_id", self._special_tokens[piece])

        # sentinel token IDs (empty dict if not present)
        if self._has_sentinel_tokens:
            self._sentinel_token_ids = {i: self._sentinel_base_id + i for i in range(self._num_sentinel_tokens)}
        else:
            self._sentinel_token_ids = {}

        self.all_special_ids = list(self._special_tokens.values())
        self.all_special_tokens = list(self._special_tokens.keys())

        self._compile_special_token_pattern()

    def _compile_special_token_pattern(self):
        """Compile regex pattern for special token matching (called once at init)."""
        import re
        if self._special_tokens:
            self._special_token_pattern = re.compile(
                '(' + '|'.join(re.escape(tok) for tok in sorted(self._special_tokens.keys(), key=len, reverse=True)) + ')'
            )
        else:
            self._special_token_pattern = None

    def _detect_special_tokens(self):
        """Detect sentinel tokens and byte fallback from the loaded model."""
        # check for sentinel tokens by looking for <extra_id_0>
        sentinel_id = self._tokenizer.piece_to_id('<extra_id_0>')
        if sentinel_id != self._tokenizer.unk_id():
            self._has_sentinel_tokens = True
            self._sentinel_base_id = sentinel_id
            # count how many sentinel tokens exist
            count = 0
            while True:
                token = f'<extra_id_{count}>'
                tid = self._tokenizer.piece_to_id(token)
                if tid == self._tokenizer.unk_id():
                    break
                count += 1
            self._num_sentinel_tokens = count
            log_from_main_process(logger, 'info', f"Detected {count} sentinel tokens starting at ID {sentinel_id}")
        else:
            self._has_sentinel_tokens = False
            self._sentinel_base_id = None
            self._num_sentinel_tokens = 0
            log_from_main_process(logger, 'info', "No sentinel tokens detected in model")

        # check for byte fallback by looking for <0x00>
        byte_id = self._tokenizer.piece_to_id('<0x00>')
        if byte_id != self._tokenizer.unk_id():
            self._has_byte_fallback = True
            self._byte_token_start = byte_id
            self._byte_token_end = byte_id + 255
            log_from_main_process(logger, 'info', f"Detected byte fallback tokens starting at ID {byte_id}")
        else:
            self._has_byte_fallback = False
            self._byte_token_start = None
            self._byte_token_end = None
            log_from_main_process(logger, 'info', "No byte fallback tokens detected in model")

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def name_or_path(self) -> str:
        return os.path.dirname(self.model_path)

    @property
    def NUM_SENTINEL_TOKENS(self) -> int:
        """Backward compatibility property for collator access."""
        return self._num_sentinel_tokens

    @property
    def SENTINEL_TOKEN_BASE_ID(self) -> Optional[int]:
        """Backward compatibility property for collator access."""
        return self._sentinel_base_id

    @property
    def BYTE_TOKEN_START(self) -> Optional[int]:
        """Backward compatibility property for collator access."""
        return self._byte_token_start

    @property
    def BYTE_TOKEN_END(self) -> Optional[int]:
        """Backward compatibility property for collator access."""
        return self._byte_token_end

    @property
    def has_sentinel_tokens(self) -> bool:
        """Whether this tokenizer has sentinel tokens."""
        return self._has_sentinel_tokens

    @property
    def has_byte_fallback(self) -> bool:
        """Whether this tokenizer has byte fallback tokens."""
        return self._has_byte_fallback

    def __len__(self) -> int:
        return self.vocab_size

    def is_byte_token(self, token_id: int) -> bool:
        """Check if token is a byte fallback token (<0xNN>)."""
        if not self._has_byte_fallback:
            return False
        return self._byte_token_start <= token_id <= self._byte_token_end

    def get_byte_token_ids(self) -> set:
        """Get set of all byte token IDs."""
        if not self._has_byte_fallback:
            return set()
        if not hasattr(self, '_byte_token_ids_cache'):
            self._byte_token_ids_cache = set(range(self._byte_token_start, self._byte_token_end + 1))
        return self._byte_token_ids_cache

    def text_to_bytes_only(self, text: str) -> List[int]:
        """Convert text to pure byte token representation."""
        if not self._has_byte_fallback:
            raise ValueError("This tokenizer does not have byte fallback tokens")
        return [self._byte_token_start + b for b in text.encode('utf-8')]

    def get_sentinel_token(self, idx: int) -> str:
        """Get sentinel token string by index."""
        if not self._has_sentinel_tokens:
            raise ValueError("This tokenizer does not have sentinel tokens")
        if 0 <= idx < self._num_sentinel_tokens:
            return f'<extra_id_{idx}>'
        raise ValueError(f"Sentinel index must be 0-{self._num_sentinel_tokens-1}, got {idx}")

    def get_sentinel_token_id(self, idx: int) -> int:
        """Get sentinel token ID by index."""
        if not self._has_sentinel_tokens:
            raise ValueError("This tokenizer does not have sentinel tokens")
        if 0 <= idx < self._num_sentinel_tokens:
            return self._sentinel_base_id + idx
        raise ValueError(f"Sentinel index must be 0-{self._num_sentinel_tokens-1}, got {idx}")

    def get_all_sentinel_ids(self) -> List[int]:
        """Get list of all available sentinel token IDs."""
        return list(self._sentinel_token_ids.values())

    def has_hanja(self, text: str) -> bool:
        """Detect if text contains Hanja characters (CJK Unified Ideographs)."""
        return any('\u4e00' <= char <= '\u9fff' for char in text)

    def split_sentences(self, text: str) -> List[str]:
        """Simple sentence splitting matching h2hcollator.py logic."""
        sentences = text.replace('! ', '!<SPLIT>').replace('? ', '?<SPLIT>').replace(
            '... ', '...<SPLIT>').replace('…', '…<SPLIT>').replace(
            '. ', '.<SPLIT>').replace('。', '。<SPLIT>').replace(
            '！', '！<SPLIT>').replace('？', '？<SPLIT>').replace('\n', '<SPLIT>').split('<SPLIT>')

        filtered = [s.strip() for s in sentences if s.strip()]
        if not filtered:
            return [text.strip()] if text.strip() else []
        return filtered

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        max_length: int = None,
        truncation: bool = False
    ) -> List[int]:
        """
        Encode text to token IDs, properly handling special tokens.

        Args:
            text: Input text to encode
            add_special_tokens: Whether to add BOS/EOS tokens
            max_length: Maximum sequence length
            truncation: Whether to truncate to max_length

        Returns:
            List of token IDs
        """
        if not isinstance(text, str):
            raise ValueError(f"Input must be a string, got {type(text)}")

        token_ids = self._encode_with_special_tokens(text)

        for i, token_id in enumerate(token_ids):
            if token_id >= self.vocab_size:
                log_from_main_process(logger, 'warning', f"Token ID {token_id} out of bounds, replacing with UNK")
                token_ids[i] = self.unk_token_id

        if add_special_tokens:
            token_ids = [self.bos_token_id] + token_ids + [self.eos_token_id]

        if truncation and max_length is not None:
            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]

        return token_ids

    def _encode_with_special_tokens(self, text: str) -> List[int]:
        """
        Encode text while properly handling special token strings.

        Splits text on special tokens, encodes segments with SentencePiece,
        and inserts correct special token IDs. Uses pre-compiled regex pattern.
        """
        if self._special_token_pattern is None:
            return self._tokenizer.encode_as_ids(text)

        parts = self._special_token_pattern.split(text)
        token_ids = []

        for part in parts:
            if not part:
                continue
            if part in self._special_tokens:
                token_ids.append(self._special_tokens[part])
            else:
                token_ids.extend(self._tokenizer.encode_as_ids(part))

        return token_ids

    def decode(
        self,
        token_ids: Union[List[int], int],
        skip_special_tokens: bool = False,
        encoding: str = "utf-8",
        errors: str = "strict"
    ) -> str:
        """Decode token IDs to text with byte fallback support."""
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        elif hasattr(token_ids, 'tolist'):
            token_ids = token_ids.tolist()

        if skip_special_tokens:
            token_ids = [tid for tid in token_ids if tid not in self._id_to_token]
            text = self._tokenizer.decode_ids(token_ids)
        else:
            pieces = []
            for tid in token_ids:
                # prefer aliased name (e.g. <|system|>) over SPM piece name
                # (<extra_id_N>) so chat-token-aliased ids render as their
                # intended chat-template strings
                if tid in self._id_to_token:
                    pieces.append(self._id_to_token[tid])
                else:
                    pieces.append(self._tokenizer.id_to_piece(tid))
            text = "".join(pieces)
            text = text.replace("▁", " ")
            text = text.strip()

        return _unbytefall(text, encoding=encoding, errors=errors)

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        """Convert token(s) to ID(s)."""
        if isinstance(tokens, str):
            if tokens in self._special_tokens:
                return self._special_tokens[tokens]
            token_ids = self._tokenizer.encode_as_ids(tokens)
            if len(token_ids) > 0:
                token_id = token_ids[0]
                if token_id >= self.vocab_size:
                    return self.unk_token_id
                return token_id
            return self.unk_token_id
        else:
            return [self.convert_tokens_to_ids(token) for token in tokens]

    def convert_ids_to_tokens(self, ids: Union[int, List[int]]) -> Union[str, List[str]]:
        """Convert ID(s) to token(s)."""
        if isinstance(ids, int):
            if ids in self._id_to_token:
                return self._id_to_token[ids]
            tokens = self._tokenizer.decode_ids([ids])
            return tokens if tokens else self.unk_token
        else:
            return [self.convert_ids_to_tokens(id_) for id_ in ids]

    def get_vocab(self) -> Dict[str, int]:
        """Get the vocabulary dictionary."""
        vocab = self._special_tokens.copy()
        for i in range(self.vocab_size):
            if i not in self._id_to_token:
                try:
                    piece = self._tokenizer.id_to_piece(i)
                    unique_key = f"{piece}#{i}" if piece in vocab else piece
                    vocab[unique_key] = i
                except:
                    vocab[f"<unk_{i}>"] = i
        return vocab

    def __call__(
        self,
        text: Union[str, List[str]],
        padding: Union[bool, str] = False,
        truncation: bool = False,
        max_length: int = None,
        return_tensors: str = None,
        add_special_tokens: bool = False,
        **kwargs
    ) -> BatchEncoding:
        """
        Tokenize text and return a dictionary like HuggingFace tokenizers.

        Args:
            text: Input text(s) to tokenize
            padding: Padding strategy ('max_length', True/'longest', or False)
            truncation: Whether to truncate to max_length
            max_length: Maximum sequence length
            return_tensors: 'np' for numpy, 'pt' for PyTorch, None for lists
            add_special_tokens: Whether to add BOS/EOS tokens

        Returns:
            BatchEncoding with 'input_ids' and 'attention_mask'
        """
        if isinstance(text, str):
            input_ids = self.encode(text, add_special_tokens=add_special_tokens,
                                   max_length=max_length, truncation=truncation)
            attention_mask = [1] * len(input_ids)
            batch_input_ids = [input_ids]
            batch_attention_mask = [attention_mask]
            is_single = True
        else:
            batch_input_ids = []
            batch_attention_mask = []
            for single_text in text:
                input_ids = self.encode(single_text, add_special_tokens=add_special_tokens,
                                       max_length=max_length, truncation=truncation)
                attention_mask = [1] * len(input_ids)
                batch_input_ids.append(input_ids)
                batch_attention_mask.append(attention_mask)
            is_single = False

        if padding:
            if padding == "max_length" and max_length:
                target_len = max_length
            else:
                target_len = max(len(seq) for seq in batch_input_ids)

            padded_input_ids = []
            padded_attention_mask = []
            for seq, mask in zip(batch_input_ids, batch_attention_mask):
                pad_len = target_len - len(seq)
                if pad_len > 0:
                    padded_input_ids.append(seq + [self.pad_token_id] * pad_len)
                    padded_attention_mask.append(mask + [0] * pad_len)
                else:
                    padded_input_ids.append(seq[:target_len])
                    padded_attention_mask.append(mask[:target_len])
            batch_input_ids = padded_input_ids
            batch_attention_mask = padded_attention_mask

        if is_single:
            output_ids = batch_input_ids[0]
            output_mask = batch_attention_mask[0]
        else:
            output_ids = batch_input_ids
            output_mask = batch_attention_mask

        if return_tensors == "np":
            if is_single:
                output_ids = np.array([output_ids])
                output_mask = np.array([output_mask])
            else:
                output_ids = np.array(output_ids)
                output_mask = np.array(output_mask)

        elif return_tensors == "pt":
            import torch
            if is_single:
                output_ids = torch.tensor([output_ids])
                output_mask = torch.tensor([output_mask])
            else:
                output_ids = torch.tensor(output_ids)
                output_mask = torch.tensor(output_mask)

        return BatchEncoding({
            'input_ids': output_ids,
            'attention_mask': output_mask
        })

    def prepare_for_encoder(
        self,
        text: Union[str, List[str]],
        add_metadata: bool = False,
        metadata: Optional[str] = None,
        **kwargs
    ) -> BatchEncoding:
        """Prepare text for encoder input matching training format.

        Format: text + <hanja|hangul>
        """
        if isinstance(text, list):
            return self.batch_prepare_for_encoder(text, add_metadata, metadata, **kwargs)

        has_hanja_char = self.has_hanja(text)
        script_token_id = self._hanja_token_id if has_hanja_char else self._hangul_token_id

        if add_metadata and metadata:
            encoder_text = f"{metadata} {text}"
        else:
            encoder_text = text

        max_length = kwargs.get('max_length')
        truncation = kwargs.get('truncation', False)
        padding = kwargs.get('padding', False)
        return_tensors = kwargs.get('return_tensors')

        token_ids = self.encode(encoder_text, add_special_tokens=False)
        token_ids.append(self.eos_token_id)
        token_ids.append(script_token_id)

        if truncation and max_length and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]

        attention_mask = [1] * len(token_ids)

        if padding == "max_length" and max_length:
            pad_len = max_length - len(token_ids)
            if pad_len > 0:
                token_ids = token_ids + [self.pad_token_id] * pad_len
                attention_mask = attention_mask + [0] * pad_len

        if return_tensors == "np":
            token_ids = np.array([token_ids])
            attention_mask = np.array([attention_mask])
        elif return_tensors == "pt":
            import torch
            token_ids = torch.tensor([token_ids])
            attention_mask = torch.tensor([attention_mask])

        return BatchEncoding({
            'input_ids': token_ids,
            'attention_mask': attention_mask
        })

    def batch_prepare_for_encoder(
        self,
        texts: List[str],
        add_metadata: bool = False,
        metadata: Optional[Union[str, List[str]]] = None,
        **kwargs
    ) -> BatchEncoding:
        """Batch version of prepare_for_encoder."""
        max_length = kwargs.get('max_length')
        truncation = kwargs.get('truncation', False)
        padding = kwargs.get('padding', False)
        return_tensors = kwargs.get('return_tensors')

        if add_metadata:
            if metadata is None:
                metadata = [""] * len(texts)
            elif isinstance(metadata, str):
                metadata = [metadata] * len(texts)
        else:
            metadata = [""] * len(texts)

        all_input_ids = []
        all_attention_mask = []

        for text, meta in zip(texts, metadata):
            has_hanja_char = self.has_hanja(text)
            script_token_id = self._hanja_token_id if has_hanja_char else self._hangul_token_id

            if meta:
                encoder_text = f"{meta} {text}"
            else:
                encoder_text = text

            token_ids = self.encode(encoder_text, add_special_tokens=False)
            token_ids.append(self.eos_token_id)
            token_ids.append(script_token_id)

            if truncation and max_length and len(token_ids) > max_length:
                token_ids = token_ids[:max_length]

            all_input_ids.append(token_ids)
            all_attention_mask.append([1] * len(token_ids))

        if padding:
            if padding == "max_length" and max_length:
                target_len = max_length
            else:
                target_len = max(len(seq) for seq in all_input_ids)

            padded_ids = []
            padded_mask = []
            for seq, mask in zip(all_input_ids, all_attention_mask):
                pad_len = target_len - len(seq)
                if pad_len > 0:
                    padded_ids.append(seq + [self.pad_token_id] * pad_len)
                    padded_mask.append(mask + [0] * pad_len)
                else:
                    padded_ids.append(seq[:target_len])
                    padded_mask.append(mask[:target_len])
            all_input_ids = padded_ids
            all_attention_mask = padded_mask

        if return_tensors == "np":
            all_input_ids = np.array(all_input_ids)
            all_attention_mask = np.array(all_attention_mask)
        elif return_tensors == "pt":
            import torch
            all_input_ids = torch.tensor(all_input_ids)
            all_attention_mask = torch.tensor(all_attention_mask)

        return BatchEncoding({
            'input_ids': all_input_ids,
            'attention_mask': all_attention_mask
        })

    def prepare_for_decoder(
        self,
        text: Union[str, List[str]],
        **kwargs
    ) -> BatchEncoding:
        """Prepare text for decoder input matching training format.

        Format: <script_token>text</s>

        Unlike encoder (text</s><script_token>), decoder has script token prepended.
        """
        if isinstance(text, list):
            return self.batch_prepare_for_decoder(text, **kwargs)

        has_hanja_char = self.has_hanja(text)
        script_token_id = self._hanja_token_id if has_hanja_char else self._hangul_token_id

        max_length = kwargs.get('max_length')
        truncation = kwargs.get('truncation', False)
        padding = kwargs.get('padding', False)
        return_tensors = kwargs.get('return_tensors')

        token_ids = self.encode(text, add_special_tokens=False)
        token_ids = [script_token_id] + token_ids + [self.eos_token_id]

        if truncation and max_length and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]

        attention_mask = [1] * len(token_ids)

        if padding == "max_length" and max_length:
            pad_len = max_length - len(token_ids)
            if pad_len > 0:
                token_ids = token_ids + [self.pad_token_id] * pad_len
                attention_mask = attention_mask + [0] * pad_len

        if return_tensors == "np":
            token_ids = np.array([token_ids])
            attention_mask = np.array([attention_mask])
        elif return_tensors == "pt":
            import torch
            token_ids = torch.tensor([token_ids])
            attention_mask = torch.tensor([attention_mask])

        return BatchEncoding({
            'input_ids': token_ids,
            'attention_mask': attention_mask
        })

    def batch_prepare_for_decoder(
        self,
        texts: List[str],
        **kwargs
    ) -> BatchEncoding:
        """Batch version of prepare_for_decoder."""
        max_length = kwargs.get('max_length')
        truncation = kwargs.get('truncation', False)
        padding = kwargs.get('padding', False)
        return_tensors = kwargs.get('return_tensors')

        all_input_ids = []
        all_attention_mask = []

        for text in texts:
            has_hanja_char = self.has_hanja(text)
            script_token_id = self._hanja_token_id if has_hanja_char else self._hangul_token_id

            token_ids = self.encode(text, add_special_tokens=False)
            token_ids = [script_token_id] + token_ids + [self.eos_token_id]

            if truncation and max_length and len(token_ids) > max_length:
                token_ids = token_ids[:max_length]

            all_input_ids.append(token_ids)
            all_attention_mask.append([1] * len(token_ids))

        if padding:
            if padding == "max_length" and max_length:
                target_len = max_length
            else:
                target_len = max(len(seq) for seq in all_input_ids)

            padded_ids = []
            padded_mask = []
            for seq, mask in zip(all_input_ids, all_attention_mask):
                pad_len = target_len - len(seq)
                if pad_len > 0:
                    padded_ids.append(seq + [self.pad_token_id] * pad_len)
                    padded_mask.append(mask + [0] * pad_len)
                else:
                    padded_ids.append(seq[:target_len])
                    padded_mask.append(mask[:target_len])
            all_input_ids = padded_ids
            all_attention_mask = padded_mask

        if return_tensors == "np":
            all_input_ids = np.array(all_input_ids)
            all_attention_mask = np.array(all_attention_mask)
        elif return_tensors == "pt":
            import torch
            all_input_ids = torch.tensor(all_input_ids)
            all_attention_mask = torch.tensor(all_attention_mask)

        return BatchEncoding({
            'input_ids': all_input_ids,
            'attention_mask': all_attention_mask
        })

    def prepare_for_labels(
        self,
        text: Union[str, List[str]],
        add_eos_between_sentences: bool = False,
        return_decoder_start_token: bool = False,
        **kwargs
    ) -> BatchEncoding:
        """Prepare text for labels matching training format.

        Format: sentence1</s>sentence2</s>...<hanja|hangul>
        """
        if isinstance(text, list):
            return self.batch_prepare_for_labels(
                text, add_eos_between_sentences, return_decoder_start_token, **kwargs
            )

        has_hanja_char = self.has_hanja(text)
        script_token_id = self._hanja_token_id if has_hanja_char else self._hangul_token_id

        max_length = kwargs.get('max_length')
        truncation = kwargs.get('truncation', False)
        padding = kwargs.get('padding', False)
        return_tensors = kwargs.get('return_tensors')

        if add_eos_between_sentences:
            sentences = self.split_sentences(text)
            token_ids = []
            for i, sentence in enumerate(sentences):
                sent_ids = self.encode(sentence, add_special_tokens=False)
                token_ids.extend(sent_ids)
                if i < len(sentences) - 1:
                    token_ids.append(self.eos_token_id)
            token_ids.append(self.eos_token_id)
        else:
            token_ids = self.encode(text, add_special_tokens=False)
            token_ids.append(self.eos_token_id)

        token_ids.append(script_token_id)

        if truncation and max_length and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]

        attention_mask = [1] * len(token_ids)

        if padding == "max_length" and max_length:
            pad_len = max_length - len(token_ids)
            if pad_len > 0:
                token_ids = token_ids + [self.pad_token_id] * pad_len
                attention_mask = attention_mask + [0] * pad_len

        outputs = {
            'input_ids': token_ids,
            'attention_mask': attention_mask
        }

        if return_decoder_start_token:
            outputs['decoder_start_token_id'] = script_token_id

        if return_tensors == "np":
            outputs['input_ids'] = np.array([outputs['input_ids']])
            outputs['attention_mask'] = np.array([outputs['attention_mask']])
        elif return_tensors == "pt":
            import torch
            outputs['input_ids'] = torch.tensor([outputs['input_ids']])
            outputs['attention_mask'] = torch.tensor([outputs['attention_mask']])

        return BatchEncoding(outputs)

    def batch_prepare_for_labels(
        self,
        texts: List[str],
        add_eos_between_sentences: bool = False,
        return_decoder_start_token: bool = False,
        **kwargs
    ) -> BatchEncoding:
        """Batch version of prepare_for_labels."""
        max_length = kwargs.get('max_length')
        truncation = kwargs.get('truncation', False)
        padding = kwargs.get('padding', False)
        return_tensors = kwargs.get('return_tensors')

        all_input_ids = []
        all_attention_mask = []
        decoder_start_tokens = [] if return_decoder_start_token else None

        for text in texts:
            has_hanja_char = self.has_hanja(text)
            script_token_id = self._hanja_token_id if has_hanja_char else self._hangul_token_id

            if add_eos_between_sentences:
                sentences = self.split_sentences(text)
                token_ids = []
                for i, sentence in enumerate(sentences):
                    sent_ids = self.encode(sentence, add_special_tokens=False)
                    token_ids.extend(sent_ids)
                    if i < len(sentences) - 1:
                        token_ids.append(self.eos_token_id)
                token_ids.append(self.eos_token_id)
            else:
                token_ids = self.encode(text, add_special_tokens=False)
                token_ids.append(self.eos_token_id)

            token_ids.append(script_token_id)

            if truncation and max_length and len(token_ids) > max_length:
                token_ids = token_ids[:max_length]

            all_input_ids.append(token_ids)
            all_attention_mask.append([1] * len(token_ids))

            if return_decoder_start_token:
                decoder_start_tokens.append(script_token_id)

        if padding:
            if padding == "max_length" and max_length:
                target_len = max_length
            else:
                target_len = max(len(seq) for seq in all_input_ids)

            padded_ids = []
            padded_mask = []
            for seq, mask in zip(all_input_ids, all_attention_mask):
                pad_len = target_len - len(seq)
                if pad_len > 0:
                    padded_ids.append(seq + [self.pad_token_id] * pad_len)
                    padded_mask.append(mask + [0] * pad_len)
                else:
                    padded_ids.append(seq[:target_len])
                    padded_mask.append(mask[:target_len])
            all_input_ids = padded_ids
            all_attention_mask = padded_mask

        outputs = {
            'input_ids': all_input_ids,
            'attention_mask': all_attention_mask
        }

        if return_decoder_start_token:
            outputs['decoder_start_token_ids'] = decoder_start_tokens

        if return_tensors == "np":
            outputs['input_ids'] = np.array(outputs['input_ids'])
            outputs['attention_mask'] = np.array(outputs['attention_mask'])
        elif return_tensors == "pt":
            import torch
            outputs['input_ids'] = torch.tensor(outputs['input_ids'])
            outputs['attention_mask'] = torch.tensor(outputs['attention_mask'])

        return BatchEncoding(outputs)

    def extend_vocabulary(
        self,
        add_bytes: bool = False,
        num_sentinels: int = 0,
        output_path: Optional[str] = None
    ) -> Tuple[int, int]:
        """
        Extend vocabulary with byte fallback tokens and/or sentinel tokens.

        Uses SentencePiece protobuf API to add new tokens to the vocabulary.
        This modifies the underlying .model file and reloads the tokenizer.

        Args:
            add_bytes: Add 256 byte fallback tokens (<0x00> through <0xFF>)
            num_sentinels: Number of sentinel tokens to add (<extra_id_0>, etc.)
            output_path: Path to save extended model (default: overwrite current)

        Returns:
            Tuple of (bytes_added, sentinels_added)
        """
        if not HAS_SPM_PROTOBUF:
            raise RuntimeError(
                "sentencepiece_model_pb2 not available. "
                "Install with: pip install sentencepiece[protobuf]"
            )

        if not add_bytes and num_sentinels <= 0:
            logger.info("No tokens to add")
            return (0, 0)

        m = sentencepiece_model_pb2.ModelProto()
        with open(self.model_path, 'rb') as f:
            m.ParseFromString(f.read())

        existing_pieces = {p.piece for p in m.pieces}
        original_size = len(m.pieces)
        bytes_added = 0
        sentinels_added = 0

        if add_bytes and not self._has_byte_fallback:
            for byte_val in range(256):
                token = f'<0x{byte_val:02X}>'
                if token not in existing_pieces:
                    new_piece = sentencepiece_model_pb2.ModelProto.SentencePiece()
                    new_piece.piece = token
                    new_piece.score = 0.0
                    new_piece.type = 4
                    m.pieces.append(new_piece)
                    bytes_added += 1
            logger.info(f"Added {bytes_added} byte fallback tokens")
        elif add_bytes:
            logger.info("Byte fallback tokens already present, skipping")

        if num_sentinels > 0:
            current_sentinels = self._num_sentinel_tokens
            for i in range(current_sentinels, current_sentinels + num_sentinels):
                token = f'<extra_id_{i}>'
                if token not in existing_pieces:
                    new_piece = sentencepiece_model_pb2.ModelProto.SentencePiece()
                    new_piece.piece = token
                    new_piece.score = 0.0
                    new_piece.type = 4
                    m.pieces.append(new_piece)
                    sentinels_added += 1
            logger.info(f"Added {sentinels_added} sentinel tokens (now {current_sentinels + sentinels_added} total)")

        if bytes_added == 0 and sentinels_added == 0:
            logger.info("All requested tokens already exist")
            return (0, 0)

        save_path = output_path or self.model_path
        with open(save_path, 'wb') as f:
            f.write(m.SerializeToString())
        logger.info(f"Saved extended model to {save_path} ({original_size} -> {len(m.pieces)} tokens)")

        self._tokenizer = spm.SentencePieceProcessor()
        self._tokenizer.load(save_path)
        self._vocab_size = self._tokenizer.get_piece_size()
        self.model_path = save_path

        self._detect_special_tokens()

        if self._has_sentinel_tokens:
            for i in range(self._num_sentinel_tokens):
                self._special_tokens[f'<extra_id_{i}>'] = self._sentinel_base_id + i

        self._id_to_token = {v: k for k, v in self._special_tokens.items()}
        self.all_special_ids = list(self._special_tokens.values())
        self.all_special_tokens = list(self._special_tokens.keys())

        if self._has_sentinel_tokens:
            self._sentinel_token_ids = {
                i: self._sentinel_base_id + i for i in range(self._num_sentinel_tokens)
            }

        if hasattr(self, '_byte_token_ids_cache'):
            del self._byte_token_ids_cache

        return (bytes_added, sentinels_added)

    CHAT_TOKENS = (
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
        "<|end_of_turn|>",
        "<|think|>",
    )

    def map_chat_tokens(
        self,
        sentinel_indices: Optional[List[int]] = None,
    ) -> Dict[str, int]:
        """
        Alias chat-template special tokens onto existing sentinel slots
        without modifying the underlying SentencePiece model.

        Reuses 5 sentinel IDs as Python-level aliases for the chat strings
        ``<|system|>``, ``<|user|>``, ``<|assistant|>``, ``<|end_of_turn|>``,
        ``<|think|>``. The .model file is untouched, so existing pretrained
        checkpoints remain bit-identical to load. After this call:

        - ``encode("<|system|>...")`` routes the chat string to its aliased
          sentinel ID via the regex special-token splitter.
        - ``convert_tokens_to_ids("<|system|>")`` returns the sentinel ID.
        - ``decode(..., skip_special_tokens=False)`` renders these IDs as the
          chat-template strings (``<|system|>`` etc.) rather than the
          underlying SPM piece names (``<extra_id_N>``).
        - ``decode(..., skip_special_tokens=True)`` filters chat IDs out of
          generated text (the standard SFT generation path).
        - Convenience attrs ``system_token_id``, ``user_token_id``,
          ``assistant_token_id``, ``end_of_turn_token_id``, ``think_token_id``
          are exposed for the SFT collator.

        Args:
            sentinel_indices: Indices of the sentinels to repurpose. Defaults
                to ``[0, 1, 2, 3, 4]``. Must be length 5, unique, and within
                ``[0, num_sentinel_tokens)``.

        Returns:
            Dict mapping each chat-token string to its aliased sentinel ID.
        """
        chat_tokens = list(self.CHAT_TOKENS)

        # v2 fast path: chat tokens are already real SPM pieces. Skip the
        # sentinel-aliasing logic and just return the real IDs (init has
        # already wired _special_tokens / convenience attrs in this case).
        unk = self._tokenizer.unk_id()
        real_ids = {
            tok: self._tokenizer.piece_to_id(tok) for tok in chat_tokens
        }
        if all(tid != unk for tid in real_ids.values()):
            log_from_main_process(
                logger,
                "info",
                f"Chat tokens already present as real SPM pieces; returning real IDs without aliasing: {real_ids}",
            )
            return real_ids

        if not self._has_sentinel_tokens:
            raise ValueError(
                "Cannot map chat tokens: no sentinel tokens detected in this model. "
                "Run extend_vocabulary(num_sentinels=...) first or use a model with sentinels."
            )

        if sentinel_indices is None:
            sentinel_indices = list(range(len(chat_tokens)))

        if len(sentinel_indices) != len(chat_tokens):
            raise ValueError(
                f"sentinel_indices must have length {len(chat_tokens)} "
                f"(one per chat token); got {len(sentinel_indices)}."
            )

        if len(set(sentinel_indices)) != len(sentinel_indices):
            raise ValueError(
                f"sentinel_indices must be unique; got {sentinel_indices}."
            )

        for idx in sentinel_indices:
            if not 0 <= idx < self._num_sentinel_tokens:
                raise ValueError(
                    f"Sentinel index {idx} out of range "
                    f"[0, {self._num_sentinel_tokens})."
                )

        aliases: Dict[str, int] = {}
        for tok, idx in zip(chat_tokens, sentinel_indices):
            sentinel_id = self._sentinel_base_id + idx
            self._special_tokens[tok] = sentinel_id
            aliases[tok] = sentinel_id

        # rebuild reverse map; chat names take precedence over <extra_id_N>
        # because they were inserted later into _special_tokens
        self._id_to_token = {v: k for k, v in self._special_tokens.items()}

        self.system_token = "<|system|>"
        self.system_token_id = aliases["<|system|>"]
        self.user_token = "<|user|>"
        self.user_token_id = aliases["<|user|>"]
        self.assistant_token = "<|assistant|>"
        self.assistant_token_id = aliases["<|assistant|>"]
        self.end_of_turn_token = "<|end_of_turn|>"
        self.end_of_turn_token_id = aliases["<|end_of_turn|>"]
        self.think_token = "<|think|>"
        self.think_token_id = aliases["<|think|>"]

        self.all_special_ids = list(self._special_tokens.values())
        self.all_special_tokens = list(self._special_tokens.keys())

        self._compile_special_token_pattern()

        log_from_main_process(
            logger,
            "info",
            f"Mapped chat tokens onto sentinel slots: {aliases}",
        )
        return aliases

    def save_pretrained(self, save_directory: str, **kwargs) -> None:
        """
        Save the tokenizer to a directory for later loading with from_pretrained.

        Saves:
        - The SentencePiece model file as 'spiece.model'
        - A tokenizer_config.json for HuggingFace compatibility

        Args:
            save_directory: Directory to save the tokenizer files
        """
        import json
        import shutil

        os.makedirs(save_directory, exist_ok=True)

        dest_model_path = os.path.join(save_directory, "spiece.model")
        if self.model_path != dest_model_path:
            shutil.copy2(self.model_path, dest_model_path)

        config = {
            "tokenizer_class": "Han2HanTokenizer",
            "model_type": "han2han",
            "vocab_size": self._vocab_size,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "mask_token": self.mask_token,
        }

        config_path = os.path.join(save_directory, "tokenizer_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved tokenizer to {save_directory}")

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> 'Han2HanTokenizer':
        """Load tokenizer from a directory."""
        if os.path.isdir(model_path):
            possible_names = [
                "han2han_v2.model",
                "spiece.model",
                "han2han_final.model",
                "tokenizer.model",
                "sentencepiece.bpe.model",
            ]

            sp_model_path = None
            for name in possible_names:
                candidate = os.path.join(model_path, name)
                if os.path.exists(candidate):
                    sp_model_path = candidate
                    break

            if sp_model_path is None:
                raise FileNotFoundError(f"Could not find SentencePiece model in {model_path}")
        else:
            sp_model_path = model_path

        return cls(sp_model_path)


try:
    from transformers import AutoTokenizer
    AutoTokenizer.register("han2han", Han2HanTokenizer)
    log_from_main_process(logger, 'info', "Han2Han tokenizer registered with HuggingFace AutoClasses!")
except ImportError:
    log_from_main_process(logger, 'warning', "Could not register Han2Han tokenizer with HuggingFace")

