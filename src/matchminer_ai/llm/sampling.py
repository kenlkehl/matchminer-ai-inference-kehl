"""Vendor sampling defaults, resolved from the actual model rather than its task.

Sources (verified 2026-09-24):
https://ai.google.dev/gemma/docs/core/model_card_4
https://huggingface.co/Qwen/Qwen3.8-Flash-Next
https://huggingface.co/Qwen/Qwen3.8-27B
Qwen 3.5 general-task defaults (verified 2026-10-01):
https://huggingface.co/Qwen/Qwen3.5-4B#best-practices
Reflection Beam defaults (verified 2026-10-16):
https://developers.reflection.ai/serving-vllm
https://huggingface.co/reflection/Beam-1.0-501B-A23B/blob/main/generation_config.json

A section's own sampling keys normally win over these defaults. A section that names no
``sampling_profile`` fixes its own sampling (for example greedy extraction); for a model whose
profile is marked ``replaces_task_sampling`` (Nemotron 3.5 Super, user instruction 2026-11-05)
the profile's sampling replaces that fixed sampling, so every workflow served by Nemotron sends
NVIDIA's settings. A section that names a profile keeps the usual rule (its explicit keys win), and
``sampling_profile: none`` sends only its own (:func:`replacing_profile`,
:func:`apply_replacing_profile`).
"""

from __future__ import annotations

import copy
import re


def vendor_defaults(
    model_name, *, profile="auto", template=None, reasoning_effort="xhigh"
):
    """Return sampling/template defaults; explicit caller settings take precedence.

    Gemma 4 and Qwen 3.5 have a thinking switch, not graded effort. Qwen 3.8 supports xhigh,
    medium and low. Unknown models receive no guessed vendor parameters.
    """
    template = dict(template or {})
    if profile in (None, "none"):
        return {}, template
    if profile not in {"auto", "gemma4", "qwen3.5", "qwen3.8", "nemotron-3.5-super-vl", "reflection-beam", "beam"}:
        raise ValueError("sampling_profile must be auto, gemma4, qwen3.5, qwen3.8, nemotron-3.5-super-vl, reflection-beam, beam, or none")
    from .model_profiles import REFLECTION_BEAM

    if profile in {"reflection-beam", "beam"} or (
        profile == "auto" and REFLECTION_BEAM.matches(str(model_name or ""))
    ):
        # Workspace effort default; the generic reasoning_effort argument is
        # Qwen's setting and must not override Beam with inherited xhigh.
        template = {**REFLECTION_BEAM.chat_template_kwargs, **template}
        return {**REFLECTION_BEAM.request_params, **REFLECTION_BEAM.extra_body}, template
    model = str(model_name or "").lower()
    family = profile
    if profile == "auto":
        family = (
            "gemma4"
            if re.search(r"gemma[-_ ]?4", model)
            else (
                "qwen3.5" if re.search(r"qwen[-_ ]?3[._]5", model)
                else "qwen3.8" if re.search(r"qwen[-_ ]?3[._]8", model) else None
            )
        )
        if "nemotron-3.5-super-vl" in model:
            family = "nemotron-3.5-super-vl"
    if family is None:
        return {}, template
    template.setdefault("enable_thinking", True)
    if type(template["enable_thinking"]) is not bool:
        raise ValueError("enable_thinking must be boolean")
    if family == "nemotron-3.5-super-vl":
        from .model_profiles import NEMOTRON_3_5_SUPER_VL

        # Text workflow. Explicit enable_thinking/medium_effort remain intact;
        # do not translate Qwen's graded effort into Nemotron template kwargs.
        return {
            **NEMOTRON_3_5_SUPER_VL.request_params,
            **NEMOTRON_3_5_SUPER_VL.extra_body,
        }, template
    if family == "gemma4":
        return {"temperature": 1.0, "top_p": 0.95, "top_k": 64}, template
    thinking = template["enable_thinking"]
    if family == "qwen3.5":
        return {
            "temperature": 1.0 if thinking else 0.7,
            "top_p": 0.95 if thinking else 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.5,
            "repetition_penalty": 1.0,
        }, template
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


def replacing_profile(model_name, sampling_profile=None):
    """The registered profile whose sampling replaces a section's fixed sampling, or None.

    ``sampling_profile`` is the section's own setting: None (unset) means the section fixes its
    sampling, which a ``replaces_task_sampling`` profile replaces; any named profile, or ``"none"``,
    keeps the section's explicit keys.
    """
    if sampling_profile is not None:
        return None
    from .model_profiles import resolve_model_profile

    profile = resolve_model_profile(str(model_name or ""))
    return profile if profile is not None and profile.replaces_task_sampling else None


def apply_replacing_profile(model_name, sampling_profile, params, extra=None):
    """Replace the sampling keys of ``params`` (and ``extra``, the vLLM extensions, when given)
    with the replacing profile's, and default its chat-template kwargs. Mutates and returns
    ``(params, extra)``; a model without a replacing profile is left unchanged."""
    profile = replacing_profile(model_name, sampling_profile)
    if profile is None:
        return params, extra
    params.update(profile.request_params)
    if extra is not None:  # vLLM extensions only where an extra body is sent
        extra.update(profile.extra_body)
    if extra is not None:
        template = extra.setdefault("chat_template_kwargs", {})
        for key, value in profile.chat_template_kwargs.items():
            template.setdefault(key, value)  # an explicit thinking switch stays
    return params, extra


def remote_sampling(runtime):
    """Resolve API and vLLM extension parameters without mutating configuration."""
    remote = runtime.get("remote", {})
    params = copy.deepcopy(remote.get("request_params", {}))
    extra = copy.deepcopy(remote.get("extra_body", {}))
    profile = runtime.get("sampling_profile", "none")
    from .model_profiles import REFLECTION_BEAM

    beam = profile in {"beam", "reflection-beam"} or (
        profile == "auto" and REFLECTION_BEAM.matches(str(runtime.get("model_name") or ""))
    )
    if beam and "reasoning_effort" in params:
        extra.setdefault("chat_template_kwargs", {}).setdefault(
            "reasoning_effort", params["reasoning_effort"]
        )
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
    apply_replacing_profile(runtime.get("model_name"), runtime.get("sampling_profile"), params, extra)
    template = extra.get("chat_template_kwargs", {})
    if (beam or template.get("enable_thinking")) and template.get("reasoning_effort"):
        if (
            "reasoning_effort" in params
            and params["reasoning_effort"] != template["reasoning_effort"]
        ):
            raise ValueError(
                "API and chat-template reasoning_effort must agree so token "
                "counting and generation use the same reasoning level"
            )
        if not beam:
            params.setdefault("reasoning_effort", template["reasoning_effort"])
    return params, extra
