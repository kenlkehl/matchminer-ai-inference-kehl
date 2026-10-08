"""Explicit, same-model transport migration without rewriting accepted checkpoints."""

from dataclasses import replace

from matchminer_ai.llm.structured import StructuredConfig

from ._guideline_generation import Client

OPERATIONAL = {
    "base_url", "transport_base_url", "timeout", "attempts", "api_key_env",
    "stream", "max_concurrent_requests",
}


def resume_transport(llm, previous, *, allow_endpoint_change=False):
    """Keep the original request identity while explicitly moving its transport.

    Changed models, context limits, sampling, reasoning, and tokenizer settings
    remain incompatible. Existing raw responses are never copied or rewritten.
    """
    if type(allow_endpoint_change) is not bool:
        raise TypeError("allow_endpoint_change must be boolean")
    if not previous:
        return llm
    saved = previous["llm"]
    endpoint = llm.base_url
    prior_endpoint = saved.get("transport_base_url") or saved["base_url"]
    if endpoint != prior_endpoint:
        if not allow_endpoint_change:
            raise ValueError(
                "Endpoint changed; stop the old worker and explicitly set "
                "allow_endpoint_change=True to resume on a compatible endpoint."
            )
        # Even explicitly pinned model/context settings must be checked against
        # the new server, not accepted as a substitute for discovery.
        info = Client(replace(llm, timeout=30), None).discover()
        if (info["id"] != saved["model"]
                or info.get("max_model_len") != saved["context_window"]):
            raise ValueError(
                "Endpoint migration requires the same served model and context; "
                "use a new output directory for a different model or context."
            )
    if endpoint == saved["base_url"] and not saved.get("transport_base_url"):
        return llm
    def comparable(d):
        return {k: v for k, v in d.items() if k not in OPERATIONAL}
    if comparable(llm.public_dict()) != comparable(saved):
        raise ValueError(
            "Endpoint migration requires identical model, context, sampling, "
            "reasoning and extraction settings; use a new output directory."
        )
    return replace(llm, base_url=saved["base_url"], transport_base_url=endpoint)


def endpoint_history(previous, llm, timestamp):
    """Append one transport event; caller holds the catalog directory lock."""
    history = list((previous or {}).get("endpoint_history", []))
    if not history and previous:
        old = previous["llm"]
        history.append({
            "endpoint": old.get("transport_base_url") or old["base_url"],
            "recorded_utc": None,
            "kind": "original_run",
            "model": old["model"], "context_window": old["context_window"],
        })
    endpoint = llm.transport_base_url or llm.base_url
    if not history or history[-1]["endpoint"] != endpoint:
        history.append({
            "endpoint": endpoint, "recorded_utc": timestamp,
            "kind": "explicit_migration" if previous else "initial",
            "model": llm.model, "context_window": llm.context_window,
        })
    return history


def validate_transport_record(record, settings, history):
    """Reject a transport that is absent from the recorded same-model history."""
    if not history:
        if record.get("transport_base_url"):
            raise ValueError("Request transport has no endpoint history")
        return
    endpoints = set()
    for event in history:
        if (event["model"] != settings["model"]
                or event["context_window"] != settings["context_window"]):
            raise ValueError("Endpoint history changed model or context")
        # Apply the usual URL/credential validation to stored provenance too.
        Client(StructuredConfig(base_url=event["endpoint"]), None)
        endpoints.add(event["endpoint"])
    actual = record.get("transport_base_url") or settings["base_url"]
    if actual not in endpoints:
        raise ValueError("Request transport is absent from endpoint history")
