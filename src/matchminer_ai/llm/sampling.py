"""Vendor sampling defaults, resolved from the actual model rather than its task.

Sources (verified 2026-09-24):
https://ai.google.dev/gemma/docs/core/model_card_4
https://huggingface.co/Qwen/Qwen3.8-Flash-Next
https://huggingface.co/Qwen/Qwen3.8-27B
"""

from __future__ import annotations

import copy
import re


def vendor_defaults(
    model_name, *, profile="auto", template=None, reasoning_effort="xhigh"
):
    """Return sampling/template defaults; explicit caller settings take precedence.

    Gemma 4 has a thinking switch, not graded effort. Qwen 3.8 supports xhigh,
    medium and low. Unknown models receive no guessed vendor parameters.
    """
    template = dict(template or {})
    if profile in (None, "none"):
        return {}, template
    if profile not in {"auto", "gemma4", "qwen3.8"}:
        raise ValueError("sampling_profile must be auto, gemma4, qwen3.8, or none")
    model = str(model_name or "").lower()
    family = profile
    if profile == "auto":
        family = (
            "gemma4"
            if re.search(r"gemma[-_ ]?4", model)
            else ("qwen3.8" if re.search(r"qwen[-_ ]?3[._]8", model) else None)
        )
    if family is None:
        return {}, template
    template.setdefault("enable_thinking", True)
    if type(template["enable_thinking"]) is not bool:
        raise ValueError("enable_thinking must be boolean")
    if family == "gemma4":
        return {"temperature": 1.0, "top_p": 0.95, "top_k": 64}, template
    thinking = template["enable_thinking"]
    template.setdefault("preserve_thinking", True)
    if thinking:
        effort = template.get("reasoning_effort", reasoning_effort)
        if effort not in {"xhigh", "medium", "low"}:
            raise ValueError("Qwen 3.8 reasoning_effort must be xhigh, medium, or low")
        template["reasoning_effort"] = effort
    return {
        "temperature": 1.0 if thinking else 0.7,
        "top_p": 0.95 if thinking else 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0 if thinking else 1.5,
        "repetition_penalty": 1.0,
    }, template


def remote_sampling(runtime):
    """Resolve API and vLLM extension parameters without mutating configuration."""
    remote = runtime.get("remote", {})
    params = copy.deepcopy(remote.get("request_params", {}))
    extra = copy.deepcopy(remote.get("extra_body", {}))
    profile = runtime.get("sampling_profile", "none")
    defaults, template = vendor_defaults(
        runtime.get("model_name"),
        profile=profile,
        template=extra.get("chat_template_kwargs"),
        reasoning_effort=params.get(
            "reasoning_effort", runtime.get("reasoning_effort", "xhigh")
        ),
    )
    for key, value in defaults.items():
        (
            extra if key in {"top_k", "min_p", "repetition_penalty"} else params
        ).setdefault(key, value)
    if template:
        extra["chat_template_kwargs"] = template
    if template.get("enable_thinking") and template.get("reasoning_effort"):
        if (
            "reasoning_effort" in params
            and params["reasoning_effort"] != template["reasoning_effort"]
        ):
            raise ValueError(
                "API and chat-template reasoning_effort must agree so token "
                "counting and generation use the same reasoning level"
            )
        params.setdefault("reasoning_effort", template["reasoning_effort"])
    return params, extra
