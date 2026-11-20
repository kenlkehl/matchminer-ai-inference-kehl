"""Shared helpers for picking and running a vLLM reasoning parser.

One place that maps model names to vLLM reasoning-parser names, adds the
standard CLI flag, and parses a raw generated string via the appropriate
parser. Every vLLM-touching script in this repo goes through this module so
swapping models (e.g. the default Nemotron 3.5 Super -> Gemma 4) is a one-flag change.
"""

from __future__ import annotations

import argparse
from typing import Optional, Tuple

# Tokenizer, template, parser and sampling choices of client-rendered teacher requests are shared with the
# inference package's teacher pool and labeler (moved there 2026-11-06).
from matchminer_ai.llm.teacher_runtime import (  # noqa: F401  (re-exported)
    MODEL_TO_PARSER, load_tokenizer, resolve_chat_template_kwargs, resolve_parser_name, resolve_sampling_defaults,
)


# Hybrid Mamba serving flags of Nemotron 3.5 Super (the default model), shared by the labeling fleet,
# the on-prem servers and the scripts that start their own vLLM servers for it.
NEMOTRON_SERVE_FLAGS = ["--trust-remote-code", "--async-scheduling", "--mamba-backend", "flashinfer",
    "--mamba-ssm-cache-dtype", "float16", "--enable-mamba-cache-stochastic-rounding",
    "--mamba-cache-philox-rounds", "5", "--chat-template-content-format", "string", "--mamba-cache-mode", "align"]


def resolve_tool_call_parser(model: str, explicit: Optional[str] = None) -> Optional[str]:
    """Use the inference package's publisher-backed tool-parser registry."""
    from matchminer_ai.llm.reasoning import resolve_tool_call_parser as resolve
    return resolve(model, explicit)


SAMPLING_KEYS = (
    "temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty",
)


def add_sampling_cli_args(parser) -> None:
    """Unspecified CLI values select model defaults; explicit values win."""
    for key in SAMPLING_KEYS:
        flags = list(dict.fromkeys(("--" + key.replace("_", "-"), "--" + key)))
        parser.add_argument(*flags, dest=key, type=int if key == "top_k" else float,
                            default=None, help="Default: detect from model family.")


def sampling_from_args(args, *, model=None, fallback=None) -> dict:
    """Resolve CLI overrides consistently for local and remote generation."""
    model = model or getattr(args, "tokenizer_model", None) or getattr(args, "model", None) or getattr(args, "model_name", None)
    return resolve_sampling_defaults(
        model, fallback=fallback,
        **{key: getattr(args, key, None) for key in SAMPLING_KEYS},
    )


def parse_reasoning_output(
    text: str, parser_name: str, tokenizer, *, enable_thinking: Optional[bool] = None,
) -> Tuple[str, str]:
    """Split raw model output into (reasoning, answer) using vLLM's parser.

    ``gemma4`` uses :func:`vllm.reasoning.gemma4_utils.parse_thinking_output`
    which also strips trailing ``<turn|>`` / ``<eos>`` sentinels — behavior
    the class-based ``extract_reasoning`` skips. ``openai_gptoss`` goes
    through vLLM's harmony helper (its ``extract_reasoning`` raises
    NotImplementedError). ``beam`` uses the shared native-channel splitter,
    without depending on the installed vLLM build. Everything else, including the default
    model's ``nemotron_v3``, uses the standard :class:`ReasoningParser` class API.
    """
    from matchminer_ai.llm.reasoning import parse_reasoning_output as parse

    return parse(text, parser_name=parser_name, tokenizer=tokenizer,
                 enable_thinking=enable_thinking)


def add_reasoning_cli_args(ap: argparse.ArgumentParser) -> None:
    """Add the standard ``--reasoning-parser`` flag.

    Default ``"auto"`` routes through :func:`resolve_parser_name` using the
    script's ``--model`` value.
    """
    ap.add_argument(
        "--reasoning-parser", "--reasoning_parser",
        dest="reasoning_parser",
        default="auto",
        help=(
            "vLLM reasoning parser name (default: auto = infer from --model). "
            "Examples: nemotron_v3, gemma4, qwen3, openai_gptoss, deepseek_r1, beam."
        ),
    )
