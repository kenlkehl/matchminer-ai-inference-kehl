"""Opt-in local archives of teacher requests, reasoning, and complete responses."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import uuid

TRACE_VERSION = "llm-response-trace-v1"


def json_value(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if hasattr(value, "model_dump"):
        return json_value(value.model_dump(mode="json"))
    if hasattr(value, "__dict__"):
        return {k: json_value(v) for k, v in vars(value).items()
                if not k.startswith("_") and not callable(v)}
    raise TypeError(f"Unsupported teacher response type: {type(value).__name__}")


def response_record(response, *, messages, provenance=None):
    raw = json_value(response)
    choices = raw.get("choices", [])
    message = choices[0].get("message", {}) if choices else {}
    reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
    encoded = json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()
    return dict(trace_version=TRACE_VERSION, generated_at=datetime.now(timezone.utc).isoformat(),
                messages=json_value(messages), messages_sha256=hashlib.sha256(encoded).hexdigest(),
                reasoning=reasoning, raw_response=raw, provenance=provenance or {})


def write_response_trace(directory, response, *, request, provenance=None):
    """Archive before application validation, including token-limited attempts.

    Credentials, HTTP headers, and unknown request extensions are never copied.
    The opt-in directory contains the original input text and must be appropriate
    for that data. An archive failure propagates rather than silently losing it.
    """
    record = response_record(response, messages=request.get("messages", []), provenance=provenance)
    safe = ("model", "temperature", "top_p", "max_tokens", "max_completion_tokens",
            "presence_penalty", "frequency_penalty", "seed", "reasoning_effort")
    record["request_parameters"] = {k: json_value(request[k]) for k in safe if k in request}
    extra = request.get("extra_body") or {}
    record["request_parameters"]["extra_body"] = {k: json_value(extra[k]) for k in
        ("top_k", "min_p", "repetition_penalty", "chat_template_kwargs", "enable_thinking") if k in extra}
    record["stage"] = os.environ.get("MMAI_LLM_TRACE_STAGE")
    path = Path(directory) / record["messages_sha256"][:2] / (uuid.uuid4().hex + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)
    return str(path.resolve())
