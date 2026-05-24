"""vLLM reasoning parser helpers."""

from __future__ import annotations

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
    ("gemma", "gemma4"),
    ("qwen", "qwen3"),
    ("gpt-oss", "openai_gptoss"),
    ("gptoss", "openai_gptoss"),
    ("deepseek", "deepseek_r1"),
)

_DISABLED = {"", "none", "off", "false", "disabled", "null"}


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


def parse_reasoning_output(
    text: str,
    *,
    parser_name: str | None,
    tokenizer: Any,
) -> tuple[str, str]:
    """Split raw vLLM output into ``(reasoning, final_text)``."""
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
        parser = parser_cls(tokenizer)
        reasoning, content = parser.extract_reasoning(text, request=None)

    return (reasoning or "").strip(), (content or "").strip()


def compose_raw_text(reasoning: str, content: str) -> str:
    """Reconstruct a readable debug string from structured reasoning output."""
    reasoning = reasoning.strip()
    content = content.strip()
    if reasoning and content:
        return f"{reasoning}\n{content}"
    return reasoning or content


__all__ = [
    "MODEL_TO_REASONING_PARSER",
    "compose_raw_text",
    "parse_reasoning_output",
    "resolve_reasoning_parser",
]
