#!/usr/bin/env python3
# coding: utf-8

from han2han_tools import transcribe


def hanja_aware_morpheme_tokenization(text: str, mecab, preserve_spacing: bool = False) -> list[str]:
    """Apply Hanja-aware morpheme tokenization using mecab.

    Preserves original Hanja characters while using morpheme boundaries
    from Hangul tokenization.

    Args:
        text: Input text potentially containing Hanja characters
        mecab: MeCab tokenizer instance
        preserve_spacing: If True, preserves spaces as separate tokens.
                         If False (default), removes all spaces for denoising tasks.
    """
    # guard: empty/whitespace-only text crashes MeCab (returns None)
    if not text or not text.strip():
        return []

    # convert to hangul first for mecab compatibility (1:1 mapping)
    hangul_text = transcribe(text)

    # guard: transcription might produce empty result
    if not hangul_text or not hangul_text.strip():
        return []

    # get morphemes from mecab (this removes spaces)
    # wrap in try-except: MeCab can return None or crash on edge cases
    try:
        hangul_morphemes = mecab.morphs(hangul_text)
        if hangul_morphemes is None:
            return []
    except Exception:
        return []

    # now find where each morpheme appears in the original text
    result = []
    search_pos = 0  # position in the hangul_text (with spaces)

    for morpheme in hangul_morphemes:
        # handle spaces before the morpheme
        while search_pos < len(hangul_text) and hangul_text[search_pos] == ' ':
            if preserve_spacing:
                result.append(' ')
            search_pos += 1

        # find the morpheme starting from search_pos
        morpheme_start = hangul_text.find(morpheme, search_pos)

        if morpheme_start == -1:
            # if not found exactly, the morpheme might be at current position
            morpheme_start = search_pos
            morpheme_end = morpheme_start + len(morpheme)
        else:
            morpheme_end = morpheme_start + len(morpheme)

        # extract the corresponding segment from the ORIGINAL text (preserving Hanja)
        original_segment = text[morpheme_start:morpheme_end]
        result.append(original_segment)

        # update search position
        search_pos = morpheme_end

    return result