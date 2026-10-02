"""Recommended sampling profiles for served models, and endpoint discovery.

A profile records the sampling parameters a model's publisher recommends, so a
config pointed at an OpenAI-compatible endpoint can adopt them automatically
once the endpoint reports which model it serves. Profiles only replace
sampling keys, plus a floor on ``max_tokens`` for models whose reasoning
trace shares the completion budget; larger task budgets are left alone.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx

from matchminer_ai.llm.remote_inference import normalize_openai_base_url

if TYPE_CHECKING:
    from matchminer_ai.config import MMAIConfig

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelProfile:
    """Publisher-recommended request settings for one model family."""

    name: str
    patterns: tuple[str, ...]
    reasoning_parser: str | None
    request_params: Mapping[str, Any]
    extra_body: Mapping[str, Any]
    chat_template_kwargs: Mapping[str, Any] = field(default_factory=dict)
    min_max_tokens: int | None = None
    source: str = ""

    def matches(self, model_name: str) -> bool:
        normalized = model_name.strip().lower()
        return any(pattern in normalized for pattern in self.patterns)


# Matched by substring so quantized re-uploads (for example
# Inferact/Qwen3.8-Flash-Next-NVFP4) resolve to the base model's profile.
QWEN3_8_FLASH_NEXT_THINKING = ModelProfile(
    name="qwen3.8-flash-next-thinking",
    patterns=("qwen3.8-flash-next",),
    reasoning_parser="qwen3",
    request_params={"temperature": 1.0, "top_p": 0.95, "presence_penalty": 0.0},
    extra_body={"top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0},
    chat_template_kwargs={"enable_thinking": True},
    # Thinking runs at the card's default "xhigh" effort and counts against
    # max_tokens; 8,000-token screening budgets were exhausted mid-reasoning.
    min_max_tokens=32768,
    source="Qwen3.8-Flash-Next model card, Best Practices, thinking mode",
)

# Matches Hub IDs (google/gemma-4-31B-it, RedHatAI/Gemma-4-31B-IT-FP8-Dynamic)
# and short served names such as gemma4-31b.
GEMMA_4_THINKING = ModelProfile(
    name="gemma-4-thinking",
    patterns=("gemma-4", "gemma4"),
    reasoning_parser="gemma4",
    request_params={"temperature": 1.0, "top_p": 0.95},
    # The card names no min_p or repetition penalty, so both are set to their
    # neutral values to replace the preset's greedy, penalized decoding.
    extra_body={"top_k": 64, "min_p": 0.0, "repetition_penalty": 1.0},
    chat_template_kwargs={"enable_thinking": True},
    # No max_tokens floor: the card recommends none, and note search defaults
    # Gemma to 8,192. Raise task budgets in config where thinking needs room.
    source=(
        "Gemma 4 31B-it model card, Best Practices (standardized sampling); "
        "vLLM Gemma 4 recipe (gemma4 reasoning parser, enable_thinking)"
    ),
)

MODEL_PROFILES: tuple[ModelProfile, ...] = (
    QWEN3_8_FLASH_NEXT_THINKING,
    GEMMA_4_THINKING,
)


def resolve_model_profile(model_name: str) -> ModelProfile | None:
    """Return the first registered profile matching ``model_name``, if any."""
    for profile in MODEL_PROFILES:
        if profile.matches(model_name):
            return profile
    return None


def discover_served_model(
    server_url: str,
    *,
    api_key: str | None = None,
    timeout: float = 30.0,
) -> str:
    """Return the model ID an OpenAI-compatible server reports at ``/models``."""
    base_url = normalize_openai_base_url(server_url)
    token = api_key or os.environ.get("OPENAI_API_KEY") or "not-needed"
    response = httpx.get(
        f"{base_url.rstrip('/')}/models",
        headers={"Authorization": f"Bearer {token}"},
        timeout=timeout,
    )
    response.raise_for_status()
    model_ids = [
        str(item.get("id") or "").strip()
        for item in response.json().get("data", [])
        if isinstance(item, Mapping) and str(item.get("id") or "").strip()
    ]
    if not model_ids:
        raise RuntimeError(f"{base_url}/models reported no models.")
    if len(model_ids) > 1:
        LOGGER.warning(
            "%s serves %d models; using the first, %s.",
            base_url,
            len(model_ids),
            model_ids[0],
        )
    return model_ids[0]


def discover_served_context_tokens(
    server_url: str,
    *,
    api_key: str | None = None,
    timeout: float = 10.0,
) -> int | None:
    """Return the ``max_model_len`` a vLLM server reports at ``/models``.

    Returns ``None`` when the server lists no positive ``max_model_len``, which
    is normal for OpenAI-compatible servers other than vLLM.
    """
    base_url = normalize_openai_base_url(server_url)
    token = api_key or os.environ.get("OPENAI_API_KEY") or "not-needed"
    response = httpx.get(
        f"{base_url.rstrip('/')}/models",
        headers={"Authorization": f"Bearer {token}"},
        timeout=timeout,
    )
    response.raise_for_status()
    for item in response.json().get("data", []):
        if not isinstance(item, Mapping):
            continue
        try:
            value = int(item.get("max_model_len") or 0)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return None


def _config_section(config: MMAIConfig, path: str) -> dict[str, Any]:
    """Resolve ``section`` or a dotted stage override such as
    ``good_option_catalog.screening_llm``, creating the override if absent."""
    head, *rest = path.split(".")
    section = getattr(config, head, None)
    if not isinstance(section, dict):
        raise ValueError(f"Config has no LLM section named {head!r}.")
    for key in rest:
        section = section.setdefault(key, {})
        if not isinstance(section, dict):
            raise ValueError(f"Config entry {path!r} is not a mapping.")
    return section


def apply_model_profile(
    config: MMAIConfig,
    model_name: str,
    *,
    sections: Sequence[str],
) -> ModelProfile | None:
    """
    Point each config section's remote settings at ``model_name``.

    Sections may be dotted stage overrides that are merged over a base
    section, so their smaller output budgets also receive the floor. When a
    profile matches, its sampling parameters and chat-template kwargs replace
    the section's, and its reasoning parser is recorded on top-level sections.
    Unmatched models keep the section's existing sampling with a warning.
    """
    profile = resolve_model_profile(model_name)
    if profile is None:
        LOGGER.warning(
            "No sampling profile registered for %s; keeping preset sampling.",
            model_name,
        )
    for section_name in sections:
        section = _config_section(config, section_name)
        remote = section.setdefault("remote", {})
        remote["model_name"] = model_name
        remote.pop("tokenizer_name", None)
        if profile is None:
            continue
        request_params = {**remote.get("request_params", {}), **profile.request_params}
        if profile.min_max_tokens is not None:
            request_params["max_tokens"] = max(
                int(request_params.get("max_tokens", 0)), profile.min_max_tokens
            )
        remote["request_params"] = request_params
        extra_body = {**remote.get("extra_body", {}), **profile.extra_body}
        extra_body["chat_template_kwargs"] = {
            **extra_body.get("chat_template_kwargs", {}),
            **profile.chat_template_kwargs,
        }
        remote["extra_body"] = extra_body
        if "." not in section_name:
            section["reasoning_parser"] = profile.reasoning_parser or "none"
    return profile


def configure_served_model(
    config: MMAIConfig,
    server_urls: Sequence[str],
    *,
    sections: Sequence[str],
    model_name: str | None = None,
    api_key: str | None = None,
) -> tuple[str, ModelProfile | None]:
    """
    Enable remote inference against ``server_urls`` and adopt its model.

    The model is discovered from the first server unless ``model_name`` is
    given; every other server must report the same model.
    """
    urls = [normalize_openai_base_url(url) for url in server_urls if url.strip()]
    if not urls:
        raise ValueError("At least one server URL is required.")
    served = [discover_served_model(url, api_key=api_key) for url in urls]
    if len(set(served)) > 1:
        raise ValueError(f"Servers report different models: {dict(zip(urls, served))}")
    resolved_model = model_name or served[0]
    if resolved_model != served[0]:
        raise ValueError(
            f"Requested model {resolved_model!r} but the servers serve {served[0]!r}."
        )
    config.remote["enabled"] = True
    config.remote["server_urls"] = urls
    profile = apply_model_profile(config, resolved_model, sections=sections)
    return resolved_model, profile


__all__ = [
    "GEMMA_4_THINKING",
    "MODEL_PROFILES",
    "ModelProfile",
    "QWEN3_8_FLASH_NEXT_THINKING",
    "apply_model_profile",
    "configure_served_model",
    "discover_served_context_tokens",
    "discover_served_model",
    "resolve_model_profile",
]
