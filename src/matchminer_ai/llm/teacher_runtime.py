"""Tokenizer, chat-template, reasoning-parser and sampling choices of teacher requests rendered on the client.

Moved from the training repository's vllm_reasoning_utils.py (2026-11-06) with the teacher pool
(``teacher_pool``) and labeler (``matchminer_ai.labeling``). The values feed label identities, so they
are unchanged.
"""

from __future__ import annotations

from typing import Optional

MODEL_TO_PARSER: dict[str, str] = {
    "google/gemma-4-E2B":     "gemma4",
    "google/gemma-4-E2B-it":  "gemma4",
    "google/gemma-4-31b-it":  "gemma4",
    "Qwen/Qwen3.6-27B":       "qwen3",
    "Qwen/Qwen3.6-27B-FP8":   "qwen3",
    "Qwen/Qwen3.5-9B":        "qwen3",
    "Qwen/Qwen3.5-35B-A3B":   "qwen3",
    "openai/gpt-oss-20b":     "openai_gptoss",
    "openai/gpt-oss-120b":    "openai_gptoss",
}

_SUBSTRING_FALLBACKS: list[tuple[str, str]] = [
    ("nemotron-3.5-super-vl", "nemotron_v3"),
    ("gemma",         "gemma4"),
    ("qwen",          "qwen3"),
    ("gpt-oss",       "openai_gptoss"),
    ("gptoss",        "openai_gptoss"),
    ("deepseek",      "deepseek_r1"),
]


def resolve_parser_name(model: str, explicit: Optional[str] = None) -> str:
    """Pick a vLLM reasoning-parser name for a given model.

    If ``explicit`` is set and not ``"auto"``, it wins. Otherwise consult the
    shared model profiles, then MODEL_TO_PARSER and case-insensitive substring
    fallbacks for other known families.
    """
    if explicit and explicit != "auto":
        return explicit
    from .model_profiles import resolve_model_profile

    profile = resolve_model_profile(model)
    if profile is not None and profile.reasoning_parser is not None:
        return profile.reasoning_parser
    if model in MODEL_TO_PARSER:
        return MODEL_TO_PARSER[model]
    lowered = model.lower()
    for needle, parser in _SUBSTRING_FALLBACKS:
        if needle in lowered:
            return parser
    raise ValueError(
        f"Cannot infer reasoning parser for model {model!r}. "
        f"Pass --reasoning-parser explicitly. Known models: "
        f"{sorted(MODEL_TO_PARSER.keys())}"
    )



def load_tokenizer(model, cache_dir=None, **kwargs):
    """Load Beam's published tokenizer without requiring its model architecture."""
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    from .model_profiles import REFLECTION_BEAM

    loader = PreTrainedTokenizerFast if REFLECTION_BEAM.matches(model) else AutoTokenizer
    return loader.from_pretrained(model, cache_dir=cache_dir, **kwargs)


def resolve_chat_template_kwargs(model: str, *, enable_thinking=True, **overrides) -> dict:
    """Use the shared Beam effort default for local rendering and chat requests.

    Beam always reasons; repair calls requesting thinking off use low effort.
    Explicit template overrides win. Other families retain their thinking switch.
    """
    from .model_profiles import REFLECTION_BEAM

    if REFLECTION_BEAM.matches(str(model or "")):
        template = dict(REFLECTION_BEAM.chat_template_kwargs)
        if not enable_thinking:
            template["reasoning_effort"] = "low"
    else:
        template = {"enable_thinking": enable_thinking}
    return {**template, **overrides}


def resolve_sampling_defaults(model: str, *, fallback=None, enable_thinking=True, **overrides) -> dict:
    """Text sampling from the shared inference profile, with explicit overrides.

    Use None for unspecified CLI values; callers can supply legacy defaults
    separately for unregistered models. No task output budget is inferred.
    """
    from .sampling import vendor_defaults
    defaults, _ = vendor_defaults(model, template={"enable_thinking": enable_thinking})
    # A recognized family must not inherit another task/model's penalties.
    # These are neutral controls, not additional publisher recommendations.
    neutral = dict(min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0) if defaults else {}
    return {**(fallback or {}), **neutral, **defaults,
            **{key: value for key, value in overrides.items() if value is not None}}


__all__ = ["MODEL_TO_PARSER", "load_tokenizer", "resolve_chat_template_kwargs", "resolve_parser_name",
           "resolve_sampling_defaults"]
