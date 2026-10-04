#!/usr/bin/env python3
# coding: utf-8
"""Build the Hugging Face fast tokenizer files from the Han2Han SentencePiece model.

Writes tokenizer.json, tokenizer_config.json and chat_template.jinja, the files
`AutoTokenizer` / `AutoProcessor` load without custom code and the ones
`transformers serve` streams through.

The result encodes like `Han2HanTokenizer` (han2han_tokenizer.py): text is split
on the special tokens and every segment between them is tokenized on its own.
One difference is known and cannot be removed: where two segmentations of a
span have exactly the same Unigram score (runs of digits, mostly), the
`tokenizers` library may order the same pieces differently than SentencePiece,
e.g. `59 4 44` for `59 44 4`.

Usage:
    python scripts/build_hf_tokenizer.py --output_dir <dir>
"""
import argparse
import os

from sentencepiece import sentencepiece_model_pb2
from tokenizers import AddedToken, Regex, Tokenizer, decoders, normalizers, pre_tokenizers
from tokenizers.models import Unigram
from transformers import PreTrainedTokenizerFast

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DEFAULT_SPM = os.path.join(REPO_ROOT, "han2han_v2_tokenizer", "spiece.model")
DEFAULT_TEMPLATE = os.path.join(REPO_ROOT, "han2han_v2_tokenizer", "chat_template.jinja")

# SentencePiece piece types
UNKNOWN, CONTROL, USER_DEFINED = 2, 3, 4


def build_tokenizer(spm_path: str) -> Tokenizer:
    proto = sentencepiece_model_pb2.ModelProto()
    with open(spm_path, "rb") as f:
        proto.ParseFromString(f.read())
    if proto.trainer_spec.model_type != 1:
        raise ValueError(f"expected a Unigram SentencePiece model, got model_type={proto.trainer_spec.model_type}")
    if not proto.trainer_spec.byte_fallback:
        raise ValueError("expected a SentencePiece model trained with byte_fallback")

    tokenizer = Tokenizer(Unigram(
        [(piece.piece, piece.score) for piece in proto.pieces],
        unk_id=proto.trainer_spec.unk_id,
        byte_fallback=True,
    ))
    # the precompiled charsmap maps character by character; NFKC after it composes
    # what SentencePiece composes across characters (decomposed Hangul, compatibility jamo)
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.Precompiled(proto.normalizer_spec.precompiled_charsmap),
        normalizers.NFKC(),
        normalizers.Replace(Regex(" {2,}"), " "),
        normalizers.Strip(),
    ])
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always", split=False)
    tokenizer.decoder = decoders.Sequence([
        decoders.Replace("▁", " "),
        decoders.ByteFallback(),
        decoders.Fuse(),
        decoders.Strip(" ", 1, 0),
    ])
    specials = [piece.piece for piece in proto.pieces if piece.type in (UNKNOWN, CONTROL, USER_DEFINED)]
    tokenizer.add_special_tokens([AddedToken(piece, normalized=False, special=True) for piece in specials])
    for index, piece in enumerate(proto.pieces):
        if tokenizer.token_to_id(piece.piece) != index:
            raise ValueError(
                f"id drift at {index}: {piece.piece!r} maps to {tokenizer.token_to_id(piece.piece)}"
            )
    return tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--spm_model", default=DEFAULT_SPM)
    parser.add_argument("--chat_template", default=DEFAULT_TEMPLATE)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_max_length", type=int, default=2048)
    args = parser.parse_args()

    with open(args.chat_template, encoding="utf-8") as f:
        chat_template = f.read()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=build_tokenizer(args.spm_model),
        pad_token="<pad>",
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        mask_token="<mask>",
        model_max_length=args.model_max_length,
        clean_up_tokenization_spaces=False,
        model_input_names=["input_ids", "attention_mask"],
        chat_template=chat_template,
    )
    os.makedirs(args.output_dir, exist_ok=True)
    fast.save_pretrained(args.output_dir)
    print(f"wrote {sorted(os.listdir(args.output_dir))} to {args.output_dir}")


if __name__ == "__main__":
    main()

