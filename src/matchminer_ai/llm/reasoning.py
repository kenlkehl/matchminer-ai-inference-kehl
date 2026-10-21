"""vLLM reasoning parser helpers."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from threading import Lock
from typing import Any


MODEL_TO_REASONING_PARSER: dict[str, str] = {
    "google/gemma-4-e2b": "gemma4",
    "google/gemma-4-e2b-it": "gemma4",
    "google/gemma-4-31b-it": "gemma4",
    "nvidia/gemma-4-31b-it-nvfp4": "gemma4",
    "qwen/qwen3.6-27b": "qwen3",
    "qwen/qwen3.6-27b-fp8": "qwen3",
    "qwen/qwen3.5-9b": "qwen3",
    "qwen/qwen3.5-35b-a3b": "qwen3",
    "openai/gpt-oss-20b": "openai_gptoss",
    "openai/gpt-oss-120b": "openai_gptoss",
}

_SUBSTRING_FALLBACKS: tuple[tuple[str, str], ...] = (
    ("nemotron-3.5-super-vl", "nemotron_v3"),
    ("gemma", "gemma4"),
    ("qwen", "qwen3"),
    ("gpt-oss", "openai_gptoss"),
    ("gptoss", "openai_gptoss"),
    ("deepseek", "deepseek_r1"),
)

_DISABLED = {"", "none", "off", "false", "disabled", "null"}


@dataclass
class _ParserEntry:
    # Retain the tokenizer: its identity cannot be recycled while cached.
    tokenizer: Any
    parser: Any = None
    lock: Any = field(default_factory=Lock)


_PARSER_CACHE: OrderedDict[tuple, _ParserEntry] = OrderedDict()
_PARSER_CACHE_LOCK = Lock()
_PARSER_CACHE_SIZE = 8


def _extract_reasoning(parser_cls, parser_name, tokenizer, options, text, request):
    # Nemotron's non-streaming extract_reasoning resets its engine per response.
    # Other parsers retain their existing lifetime until reuse is verified.
    if parser_name != "nemotron_v3":
        return parser_cls(tokenizer, **options).extract_reasoning(text, request=request)
    thinking = options.get("chat_template_kwargs", {}).get("enable_thinking")
    key = (parser_cls, id(tokenizer), thinking)
    with _PARSER_CACHE_LOCK:
        entry = _PARSER_CACHE.pop(key, None)
        if entry is None:
            entry = _ParserEntry(tokenizer)
        _PARSER_CACHE[key] = entry
        while len(_PARSER_CACHE) > _PARSER_CACHE_SIZE:
            _PARSER_CACHE.popitem(last=False)
    # A parser is mutable. Serialize its use, including first construction, so
    # concurrent worker threads cannot mix response state. Discard failed state.
    with entry.lock:
        try:
            if entry.parser is None:
                entry.parser = parser_cls(tokenizer, **options)
            return entry.parser.extract_reasoning(text, request=request)
        except BaseException:
            entry.parser = None
            raise


def reasoning_system_prompt(model_name: str | None, instruction: str = "") -> str:
    """Add the GPT-OSS text control only for a known GPT-OSS model identity.

    Other families request thinking through their chat-template/API controls.
    Unknown names deliberately get no GPT-OSS header. Task instructions remain
    intact regardless of model family.
    """
    normalized = (model_name or "").lower()
    header = "Reasoning: high" if any(
        name in normalized for name in ("gpt-oss", "gptoss", "gpt_oss")
    ) else ""
    return "\n\n".join(text for text in (header, instruction) if text)


def resolve_reasoning_parser(
    model_name: str,
    explicit: str | None = "auto",
) -> str | None:
    """
    Resolve the vLLM reasoning parser name for a model.

    ``explicit`` wins unless it is ``"auto"``. Passing ``"none"`` or an
    equivalent disabled value turns parsing off.
    """
    requested = (explicit or "auto").strip()
    if requested.lower() in _DISABLED:
        return None
    if requested.lower() != "auto":
        return requested

    from .model_profiles import resolve_model_profile

    profile = resolve_model_profile(model_name)
    if profile is not None and profile.reasoning_parser is not None:
        return profile.reasoning_parser
    normalized = model_name.strip().lower()
    if normalized in MODEL_TO_REASONING_PARSER:
        return MODEL_TO_REASONING_PARSER[normalized]
    for needle, parser_name in _SUBSTRING_FALLBACKS:
        if needle in normalized:
            return parser_name
    raise ValueError(
        f"Cannot infer vLLM reasoning parser for model {model_name!r}. "
        "Set reasoning_parser explicitly or use 'none' to disable parsing."
    )


def resolve_tool_call_parser(model_name: str, explicit: str | None = "auto") -> str | None:
    """Resolve an optional vLLM tool parser, preserving overrides and opt-out."""
    requested = (explicit or "auto").strip()
    if requested.lower() in _DISABLED:
        return None
    if requested.lower() != "auto":
        return requested
    from .model_profiles import resolve_model_profile

    profile = resolve_model_profile(model_name)
    return profile.tool_call_parser if profile else None


def parse_reasoning_output(
    text: str,
    *,
    parser_name: str | None,
    tokenizer: Any,
    enable_thinking: bool | None = None,
) -> tuple[str, str]:
    """Split raw local-mode vLLM output into ``(reasoning, final_text)``."""
    if parser_name is None:
        return "", text.strip()

    if parser_name == "gemma4":
        from vllm.reasoning.gemma4_utils import parse_thinking_output

        parsed = parse_thinking_output(text)
        reasoning = parsed.get("thinking") or ""
        content = parsed.get("answer") or ""
    elif parser_name == "openai_gptoss":
        from vllm.entrypoints.openai.parser.harmony_utils import parse_chat_output

        reasoning, content = parse_chat_output(text)
    else:
        from vllm.reasoning import ReasoningParserManager

        parser_cls = ReasoningParserManager.get_reasoning_parser(parser_name)
        from types import SimpleNamespace
        options = {} if enable_thinking is None else {
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
        reasoning, content = _extract_reasoning(
            parser_cls, parser_name, tokenizer, options, text,
            SimpleNamespace(**options) if options else None,
        )
        if parser_name == "nemotron_v3" and enable_thinking is False:
            content = (content or "").strip()
            while content.endswith("<|im_end|>"):
                content = content[:-len("<|im_end|>")].rstrip()

    return (reasoning or "").strip(), (content or "").strip()


__all__ = [
    "MODEL_TO_REASONING_PARSER",
    "parse_reasoning_output",
    "reasoning_system_prompt",
    "resolve_reasoning_parser",
    "resolve_tool_call_parser",
]
