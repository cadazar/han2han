#!/usr/bin/env python3
# coding: utf-8
"""
Task prompt library for Han2Han training.

Provides Korean task prompts for different task types. Composite keys
(``task:format:form:shuffle:intensity``) let prompts express the exact
corruption used at training time, with a hierarchical fallback chain that
progressively drops axes until a registered entry is found (legacy
single-token keys are the final fallback rung).

The ``format_name`` axis is future-proofed for mixed text + MIDI training: a
later patch will register ``denoising:remi:...`` / ``denoising:compound_midi:...``
entries without further refactor.
"""

import logging
import random
from typing import Literal, Optional

from logging_utils import log_from_main_process

logger = logging.getLogger(__name__)

# tracks key chains that have already triggered a legacy-fallback warning so
# we do not spam the log on every batch.
_LOGGED_FALLBACK_KEYS: set[tuple] = set()

# task prompts by language. legacy single-token keys (``denoising``,
# ``denoising_heavy``, ...) act as the final fallback rung. composite
# ``task:format:form:shuffle:intensity`` keys are registered below for the
# corruption-aware text pretraining recipe.
TASK_PROMPTS = {
    'denoising': {
        'ko': [
            "다음 텍스트를 복원하시오:",
            "손상된 텍스트를 재구성하시오:",
            "빈칸을 채워 텍스트 데이터를 완성하시오:",
        ]
    },
    'ocr_correction': {
        'ko': [
            "OCR 오류를 수정하시오:",
            "잘못 인식된 텍스트를 교정하시오:",
            "스캔 오류를 바로잡으시오:",
        ],
    },
    'temporal_classification': {
        'ko': [
            "이 텍스트가 작성된 연도를 추정하시오:",
            "다음 글의 시대를 판별하시오:",
            "문서의 작성 시기를 밝히시오:",
        ],
    },
    'sts': {
        'ko': [
            "두 문장의 의미적 유사도를 평가하시오:",
            "문장 간 유사성을 0-5 척도로 측정하시오:",
            "다음 문장 쌍의 의미적 관련성을 판단하시오:",
        ],
    },
    'transcription_hanja_to_hangul': {
        'ko': [
            "한자를 한글로 전사하시오:",
            "다음 한문을 한글로 옮기시오:",
            "한자 표기를 한글 독음으로 변환하시오:",
        ],
    },
    'transcription_hangul_to_hanja': {
        'ko': [
            "한글을 한자로 전사하시오:",
            "다음 한글 텍스트에 적절한 한자를 추가하시오:",
            "한자 표기가 필요한 부분에 한자를 병기하시오:",
        ],
    },
    'continuation': {
        'ko': [
            "다음 텍스트를 이어서 작성하시오:",
            "문장을 완성하시오:",
            "이야기를 계속 써 내려가시오:",
        ],
    },
    'morpheme_denoising': {
        'ko': [
            "띄어쓰기와 문장 구조를 복원하시오:",
            "공백 없는 텍스트에 올바른 띄어쓰기를 적용하시오:",
            "압축된 텍스트를 문장 단위로 재구성하시오:",
        ],
    },
    'denoising_heavy': {
        'ko': [
            "심하게 손상된 텍스트를 복원하시오:",
            "대부분이 가려진 텍스트를 재구성하시오:",
            "절반 이상 손실된 텍스트를 복구하시오:",
        ],
    },
    'byte_reconstruction': {
        'ko': [
            "바이트 수준 텍스트를 복원하시오:",
            "바이트 단위로 손상된 텍스트를 재구성하시오:",
            "바이트 표현에서 원본 텍스트를 복구하시오:",
        ],
    },
    'classification': {
        'ko': [
            "다음 기사의 주제를 분류하시오:",
            "뉴스 기사의 분야를 판별하시오:",
            "텍스트의 카테고리를 결정하시오:",
        ],
    },
    'nli': {
        'ko': [
            "전제와 가설의 관계를 판단하시오:",
            "두 문장 간의 논리적 관계를 추론하시오:",
            "가설이 전제로부터 도출되는지 판단하시오:",
        ],
    },
    'temporal_continuation': {
        'ko': [
            "작성 시대를 고려하여 텍스트를 이어 쓰고, 작성 연도를 추정하시오:",
            "시대적 문체를 유지하며 계속 쓰고 연도를 밝히시오:",
            "문체를 분석하여 글을 이어가고, 작성 시기를 판단하시오:",
        ],
    },
    'multiple_choice': {
        'ko': [
            "다음 질문에 대한 정답을 선택하시오:",
            "보기 중 올바른 답을 고르시오:",
            "다음 문제의 답을 고르시오:",
        ],
    },
    # generic helpful-assistant prompts used as the system slot for KAIST
    # multilingual CoT data. reasoning is toggled by the decoder start token
    # (<|think|> vs <|assistant|>), NOT by an encoder-side "think step by
    # step" instruction; the FLAN-style task framing already lives in the
    # source/user text.
    'cot_reasoning': {
        'ko': [
            "당신은 도움이 되는 한국어 어시스턴트입니다.",
            "주어진 질문에 정확하고 친절하게 답하시오.",
            "다음 작업을 수행하시오.",
            "사용자의 요청에 신중히 응답하시오.",
            "다음 질문에 답하시오.",
        ],
    },
    'summarization': {
        'ko': [
            "다음 글을 요약하시오.",
            "다음 텍스트의 핵심 내용을 요약하시오.",
            "다음 문서를 간결하게 요약해 주세요.",
            "아래 글의 주요 내용을 정리하시오.",
            "다음 글의 요지를 한 문단으로 요약하시오.",
        ],
    },

    # ------------------------------------------------------------------
    # corruption-aware composite keys (text-only). format axis is wired
    # so future MIDI prompt entries slot in via ``denoising:remi:...`` /
    # ``denoising:compound_midi:...`` without further refactor.
    # ------------------------------------------------------------------

    # BART-form, sentences in order, light corruption
    'denoising:text:bart:no_shuffle:light': {
        'ko': [
            "텍스트의 일부가 마스크 토큰으로 가려져 있습니다. 빈칸을 채워 원본을 복원하시오:",
            "마스크된 짧은 구간을 채워 텍스트를 완성하시오:",
            "마스크 토큰 자리에 누락된 표현을 복원하시오:",
        ],
    },
    'denoising:text:bart:no_shuffle:medium': {
        'ko': [
            "텍스트의 여러 구간이 마스크 토큰으로 가려져 있습니다. 가려진 부분을 복원하시오:",
            "마스크된 구간을 찾아 원문을 재구성하시오:",
            "다수의 마스크 토큰으로 손상된 텍스트를 복원하시오:",
        ],
    },
    'denoising:text:bart:no_shuffle:heavy': {
        'ko': [
            "텍스트의 상당 부분이 마스크 토큰으로 가려져 있습니다. 가능한 한 원문을 복원하시오:",
            "대부분이 마스크된 텍스트를 재구성하시오:",
            "심하게 손상된 텍스트의 빈칸을 채우시오:",
        ],
    },

    # BART-form, sentence permutation on
    'denoising:text:bart:shuffled:light': {
        'ko': [
            "텍스트의 일부가 마스크 토큰으로 가려져 있으며, 문장 순서가 뒤섞였을 수 있습니다. 빈칸을 채우고 원래 순서로 복원하시오:",
            "마스크된 짧은 구간을 채우고 뒤바뀐 문장을 올바른 순서로 재배열하시오:",
            "마스크 토큰을 복원하고 문장 순서를 복구하시오:",
        ],
    },
    'denoising:text:bart:shuffled:medium': {
        'ko': [
            "텍스트의 여러 구간이 마스크되었고 문장 순서가 뒤섞였을 수 있습니다. 원문을 복원하시오:",
            "마스크된 구간을 채우고 문장을 올바른 순서로 정렬하시오:",
            "중간 정도로 손상되었고 순서가 바뀐 텍스트를 재구성하시오:",
        ],
    },
    'denoising:text:bart:shuffled:heavy': {
        'ko': [
            "텍스트의 상당 부분이 마스크되었고 문장 순서가 뒤섞였을 수 있습니다. 가능한 한 원문을 복원하시오:",
            "대부분이 마스크되고 순서가 뒤바뀐 텍스트를 재구성하시오:",
            "심하게 손상되고 뒤섞인 텍스트의 원형을 복원하시오:",
        ],
    },

    # sentinel-form (T5 numbered sentinels; never shuffles sentences)
    'denoising:text:sentinel:light': {
        'ko': [
            "짧은 구간이 번호가 매겨진 센티넬 토큰으로 가려져 있습니다. 각 센티넬에 해당하는 원본 토큰을 출력하시오:",
            "센티넬 토큰으로 표시된 짧은 빈칸을 채우시오:",
            "번호별 센티넬에 대응하는 누락된 표현을 복원하시오:",
        ],
    },
    'denoising:text:sentinel:medium': {
        'ko': [
            "여러 구간이 번호가 매겨진 센티넬 토큰으로 가려져 있습니다. 각 센티넬에 해당하는 원본 구간을 복원하시오:",
            "센티넬 토큰으로 표시된 구간들을 순서대로 복원하시오:",
            "다수의 센티넬 표시 구간을 원문에 맞게 채우시오:",
        ],
    },
    'denoising:text:sentinel:heavy': {
        'ko': [
            "상당 부분이 번호가 매겨진 센티넬 토큰으로 가려져 있습니다. 각 센티넬에 해당하는 원본 구간을 가능한 한 복원하시오:",
            "대부분이 센티넬로 가려진 텍스트의 각 구간을 채우시오:",
            "심하게 센티넬 마스크된 텍스트를 복원하시오:",
        ],
    },

    # X-denoiser (always heavy intensity); composite keys without intensity suffix
    'denoising_heavy:text:bart:no_shuffle': {
        'ko': [
            "대부분이 마스크 토큰으로 가려진 텍스트를 복원하시오:",
            "극도로 손상된 텍스트의 마스크된 빈칸을 채우시오:",
            "절반 이상이 가려진 텍스트를 재구성하시오:",
        ],
    },
    'denoising_heavy:text:bart:shuffled': {
        'ko': [
            "대부분이 마스크되고 문장 순서가 뒤섞인 텍스트를 복원하시오:",
            "심하게 손상되고 순서가 뒤바뀐 텍스트를 재구성하시오:",
            "마스크된 구간을 채우고 뒤바뀐 문장 순서를 복원하시오:",
        ],
    },
    'denoising_heavy:text:sentinel': {
        'ko': [
            "대부분이 번호가 매겨진 센티넬 토큰으로 가려진 텍스트를 복원하시오:",
            "극도로 손상된 센티넬 마스크 텍스트의 각 구간을 채우시오:",
            "심하게 센티넬로 가려진 텍스트를 재구성하시오:",
        ],
    },
}


def _build_prompt_key_chain(
    task_type: str,
    format_name: str = 'text',
    form: Optional[str] = None,
    shuffled: bool = False,
    intensity: Optional[str] = None,
) -> list[str]:
    """
    Build hierarchical lookup chain from most specific to legacy fallback.

    The chain progressively drops axes: intensity -> shuffle -> form ->
    format -> legacy. Sentinel form never carries a shuffle tag because
    sentinel-style corruption preserves sentence order by construction.
    """
    keys: list[str] = []
    if form is not None:
        if form == 'sentinel':
            if intensity is not None:
                keys.append(f"{task_type}:{format_name}:{form}:{intensity}")
            keys.append(f"{task_type}:{format_name}:{form}")
        else:
            shuffle_tag = 'shuffled' if shuffled else 'no_shuffle'
            if intensity is not None:
                keys.append(f"{task_type}:{format_name}:{form}:{shuffle_tag}:{intensity}")
            keys.append(f"{task_type}:{format_name}:{form}:{shuffle_tag}")
            keys.append(f"{task_type}:{format_name}:{form}")
    keys.append(f"{task_type}:{format_name}")
    keys.append(task_type)
    return keys


def sample_task_prompt(
    task_type: str,
    *,
    format_name: str = 'text',
    form: Optional[Literal['bart', 'sentinel']] = None,
    shuffled: bool = False,
    intensity: Optional[Literal['light', 'medium', 'heavy']] = None,
    transcription: Optional[Literal['hangul_to_hanja', 'hanja_to_hangul']] = None,
    seed: Optional[int] = None,
) -> tuple[str, str]:
    """
    Sample a task prompt with weighted language selection.

    Resolves a prompt via a hierarchical key chain that captures the
    corruption configuration applied to the example. The chain falls
    back from the most specific composite key to the legacy single-token
    key, so existing call sites that pass only ``task_type`` keep working.

    Args:
        task_type: top-level task label (e.g. ``'denoising'``,
            ``'denoising_heavy'``, ``'temporal_classification'``).
        format_name: modality axis (``'text'`` today; ``'remi'`` /
            ``'compound_midi'`` slot in later without changing the signature).
        form: corruption form for denoising-family tasks. ``'bart'`` is
            single-mask BART-style; ``'sentinel'`` is T5 numbered sentinels.
            ``None`` for non-denoising tasks.
        shuffled: whether sentence permutation was applied (BART-style only).
        intensity: intensity bucket sampled from the R-denoiser config
            (``'light'`` / ``'medium'`` / ``'heavy'``). ``None`` when not
            applicable (e.g. X-denoiser composites carry intensity in the
            ``task_type`` itself).
        transcription: when set, overrides ``task_type`` and resolves
            directly from the ``transcription_*`` prompt pools.
        seed: optional seed for reproducibility.

    Returns:
        Tuple of ``(prompt_string, resolved_task_direction)``. The resolved
        direction matches the ``task_type`` actually used after any
        transcription / translation direction override.
    """
    if seed is not None:
        random.seed(seed)

    # transcription overlay reroutes to the existing transcription pools
    # regardless of the caller's ``task_type`` (e.g. a denoising example
    # that overlaid a hanja/hangul transcription gets a transcription prompt)
    if transcription is not None:
        if transcription not in ('hangul_to_hanja', 'hanja_to_hangul'):
            raise ValueError(
                f"transcription must be 'hangul_to_hanja' or 'hanja_to_hangul', "
                f"got {transcription!r}"
            )
        task_type = f"transcription_{transcription}"
        form = None
        shuffled = False
        intensity = None

    # legacy direction sampling for callers that pass the generic label
    if task_type == 'transcription':
        task_type = random.choice(['transcription_hanja_to_hangul', 'transcription_hangul_to_hanja'])

    key_chain = _build_prompt_key_chain(task_type, format_name, form, shuffled, intensity)
    for idx, key in enumerate(key_chain):
        task_langs = TASK_PROMPTS.get(key)
        if task_langs is None:
            continue
        prompts = task_langs.get('ko')
        if not prompts:
            continue
        # log a one-time warning when we fall all the way to the legacy
        # rung despite the caller specifying a composite axis. that means
        # the registry is missing a specific entry the caller asked for.
        is_legacy_rung = (idx == len(key_chain) - 1)
        had_composite_axes = (form is not None) or (intensity is not None) or (format_name != 'text')
        if is_legacy_rung and had_composite_axes:
            cache_key = (task_type, format_name, form, shuffled, intensity)
            if cache_key not in _LOGGED_FALLBACK_KEYS:
                _LOGGED_FALLBACK_KEYS.add(cache_key)
                log_from_main_process(
                    logger, 'warning',
                    "sample_task_prompt fell back to legacy key %r for "
                    "task=%r format=%r form=%r shuffled=%r intensity=%r "
                    "(no composite entry registered; tried %s)",
                    key, task_type, format_name, form, shuffled, intensity, key_chain,
                )
        return random.choice(prompts), task_type

    raise KeyError(
        f"No prompt registered for task_type={task_type!r}, format_name={format_name!r}, "
        f"form={form!r}, shuffled={shuffled!r}, intensity={intensity!r}. "
        f"Tried keys: {key_chain}"
    )


def add_task_prompt_to_example(
    example: dict,
    task_type: str,
    *,
    format_name: str = 'text',
    form: Optional[Literal['bart', 'sentinel']] = None,
    shuffled: bool = False,
    intensity: Optional[Literal['light', 'medium', 'heavy']] = None,
    transcription: Optional[Literal['hangul_to_hanja', 'hanja_to_hangul']] = None,
) -> tuple[dict, str]:
    """
    Add task prompt to example as metadata field. Pass-through wrapper
    that forwards corruption axes to ``sample_task_prompt``.

    Returns:
        Tuple of (example with ``'metadata'`` field replaced with the
        sampled prompt, resolved task direction).
    """
    prompt, task_direction = sample_task_prompt(
        task_type,
        format_name=format_name,
        form=form,
        shuffled=shuffled,
        intensity=intensity,
        transcription=transcription,
    )
    example['metadata'] = prompt
    return example, task_direction


if __name__ == '__main__':
    print("Task Prompt Sampling Test")
    print("=" * 80)

    # legacy single-token keys still work
    print("\n[legacy single-token keys]")
    for task in ['denoising', 'denoising_heavy', 'continuation', 'temporal_classification']:
        prompt, resolved = sample_task_prompt(task)
        print(f"  {task} -> ({resolved}) {prompt}")

    # full composite key resolution: every (form, shuffled, intensity) combo
    # should hit a non-empty pool
    print("\n[denoising composite keys]")
    combos = [
        ('bart', False, 'light'),
        ('bart', False, 'medium'),
        ('bart', False, 'heavy'),
        ('bart', True, 'light'),
        ('bart', True, 'medium'),
        ('bart', True, 'heavy'),
        ('sentinel', False, 'light'),
        ('sentinel', False, 'medium'),
        ('sentinel', False, 'heavy'),
    ]

    for form, shuffled, intensity in combos:
        prompt, _ = sample_task_prompt(
            'denoising', form=form, shuffled=shuffled, intensity=intensity
        )
        shuf_tag = 'shuffled' if shuffled else 'no_shuffle'
        print(f"  denoising/{form}/{shuf_tag}/{intensity}: {prompt}")

    print("\n[denoising_heavy composite keys]")
    heavy_combos = [
        ('bart', False),
        ('bart', True),
        ('sentinel', False),
    ]
    for form, shuffled in heavy_combos:
        prompt, _ = sample_task_prompt(
            'denoising_heavy', form=form, shuffled=shuffled,
        )
        shuf_tag = 'shuffled' if shuffled else 'no_shuffle'
        print(f"  denoising_heavy/{form}/{shuf_tag}: {prompt}")

    # transcription overlay
    print("\n[transcription overlay]")
    for direction in ['hangul_to_hanja', 'hanja_to_hangul']:
        prompt, resolved = sample_task_prompt(
            'denoising', form='bart', intensity='medium',
            transcription=direction,
        )
        print(f"  {direction} (called via denoising): ({resolved}) {prompt}")

    # progressive fallback: drop intensity -> drop shuffle -> drop form -> legacy
    print("\n[progressive fallback chain for denoising]")
    chains_to_check = [
        ('denoising', 'text', 'bart', True, 'medium'),
        ('denoising', 'text', 'bart', True, None),       # drops intensity
        ('denoising', 'text', 'bart', False, None),      # drops shuffle (no_shuffle has entries)
        ('denoising', 'text', None, False, None),        # drops form -> tries denoising:text -> legacy denoising
    ]
    for chain_args in chains_to_check:
        keys = _build_prompt_key_chain(*chain_args)
        first_hit = next((k for k in keys if k in TASK_PROMPTS), None)
        print(f"  args={chain_args} chain={keys} first_hit={first_hit}")

    # exhaustive non-empty check for every composite key
    print("\n[non-empty pool check]")
    missing = []
    composite_keys = [k for k in TASK_PROMPTS if ':' in k]
    for key in composite_keys:
        if not TASK_PROMPTS[key].get('ko'):
            missing.append(key)
    if missing:
        print(f"  FAIL: empty pools at {missing}")
    else:
        print(f"  OK: all {len(composite_keys)} composite keys have non-empty Korean pools")