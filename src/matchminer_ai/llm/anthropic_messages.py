"""Claude through a Databricks AI Gateway, behind the structured client's chat contract.

``StructuredClient`` speaks OpenAI chat completions. For provider
``databricks_anthropic`` it hands each request to :func:`send_messages`, which sends
it with the official Anthropic SDK to the gateway's Messages route
(``https://<workspace>/ai-gateway/anthropic``) with a Databricks access token, and
returns the answer in the chat-completions shape the rest of the package reads.

- **Request:** leading system messages become system blocks, the first one cached
  (send stable instructions first, varying ones after); ``reasoning_effort``
  becomes ``output_config.effort``; a JSON-schema response format becomes
  ``output_config.format`` (the SDK moves keywords Claude does not enforce, such as
  ``maxItems``, into descriptions; callers still validate in code). Sampling and
  chat-template fields are dropped: Claude manages its own sampling and thinking.
- **Refusals:** Opus and Sonnet 5.5 requests opt into server-side refusal fallbacks
  (``fallbacks="default"``); the response's ``model`` and ``transport.fallback_used``
  say which model answered.
- **Errors:** only the HTTP status is kept, never provider text.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any
from urllib.parse import urlsplit

from matchminer_ai.cancellation import check_cancelled

from .remote_auth import DatabricksCredentialsError, databricks_token_provider

GATEWAY_PATH = "/ai-gateway/anthropic"
_WORKSPACE_HOST = re.compile(
    r"(?:[a-z0-9-]+\.)+(?:azuredatabricks\.net|cloud\.databricks\.com|gcp\.databricks\.com)"
)
EFFORTS = ("low", "medium", "high", "xhigh", "max")
FALLBACK_BETA = "server-side-fallback-2026-07-01"
#: Request fields Claude takes from the chat body, and fields dropped as not applicable.
_TAKEN = frozenset({"model", "messages", "max_tokens", "reasoning_effort", "response_format",
                    "stream", "stream_options"})
_DROPPED = frozenset({"temperature", "top_p", "top_k", "min_p", "presence_penalty",
                      "frequency_penalty", "repetition_penalty", "chat_template_kwargs"})
_FINISH = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length",
           "model_context_window_exceeded": "length", "refusal": "refusal",
           "pause_turn": "pause_turn", "tool_use": "tool_calls"}


def gateway_url_error(base_url: str) -> str | None:
    """Why ``base_url`` is not a Databricks gateway Messages URL (None when it is)."""
    parsed = urlsplit(base_url)
    try:
        port = parsed.port
    except ValueError:
        port = -1
    if (
        parsed.scheme != "https" or not _WORKSPACE_HOST.fullmatch(parsed.hostname or "")
        or parsed.username or parsed.password or port not in (None, 443)
        or parsed.query or parsed.fragment or parsed.path.rstrip("/") != GATEWAY_PATH
    ):
        return ("Databricks credentials go only to a workspace's gateway URL: "
                f"https://<workspace host>{GATEWAY_PATH}.")
    return None


def request_field_error(fields) -> str | None:
    """Why request fields cannot be sent to Claude (None when they can)."""
    unknown = sorted(set(fields) - _TAKEN - _DROPPED)
    if unknown:
        return "Claude through Databricks does not take these request fields: " + ", ".join(unknown)
    return None


def fallbacks_supported(model: str) -> bool:
    """Server-side refusal fallbacks: accepted on the Opus and Sonnet 5.5 endpoints."""
    return bool(re.search(r"claude-(?:opus|sonnet)-5-5$", model))


def messages_request(body: dict) -> dict:
    """The Messages API request for one chat-completions body."""
    error = request_field_error(body)
    if error:
        raise ValueError(error)
    system, messages = [], []
    for message in body["messages"]:
        role, content = message["role"], message["content"]
        if role == "system" and not messages:
            system.append(content)
        elif role in ("user", "assistant"):
            messages.append({"role": role, "content": content})
        else:
            raise ValueError("Claude requests take system instructions first, then user "
                             "and assistant messages.")
    if not messages or messages[-1]["role"] != "user":
        raise ValueError("Claude requests end with a user message (no assistant prefill).")
    request: dict[str, Any] = {"model": body["model"], "max_tokens": body["max_tokens"],
                               "messages": messages}
    if system:
        # One block per leading system message; the first is cached (the same
        # instructions open every turn of an agent run), later ones may vary.
        request["system"] = [{"type": "text", "text": text} for text in system]
        request["system"][0]["cache_control"] = {"type": "ephemeral"}
    output: dict[str, Any] = {}
    effort = body.get("reasoning_effort")
    if effort is not None:
        if effort not in EFFORTS:
            raise ValueError(f"Claude effort must be one of {', '.join(EFFORTS)}.")
        output["effort"] = effort
    response_format = body.get("response_format") or {}
    if response_format.get("type") == "json_schema":
        from anthropic import transform_schema

        output["format"] = {"type": "json_schema",
                            "schema": transform_schema(response_format["json_schema"]["schema"])}
    if output:
        request["output_config"] = output
    if fallbacks_supported(body["model"]):
        request["betas"] = [FALLBACK_BETA]
        request["fallbacks"] = "default"
    return request


def chat_completion(message, *, streamed: bool, abort_reason: str | None = None) -> dict:
    """A Messages API response in the chat-completions shape (text blocks only)."""
    blocks = list(getattr(message, "content", None) or [])
    text = "".join(block.text for block in blocks if getattr(block, "type", None) == "text")
    usage = getattr(message, "usage", None)
    read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
    written = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
    stop = getattr(message, "stop_reason", None)
    finish = "client_aborted_repetition" if abort_reason else _FINISH.get(stop, str(stop))
    transport = {"streamed": streamed, "stop_reason": stop,
                 "fallback_used": any(getattr(b, "type", None) == "fallback" for b in blocks)}
    if abort_reason:
        transport["abort_reason"] = abort_reason
    return {
        "id": getattr(message, "id", None),
        "model": getattr(message, "model", None),
        "choices": [{"index": 0, "finish_reason": finish,
                     "message": {"role": "assistant", "content": text}}],
        "usage": {
            "prompt_tokens": int(getattr(usage, "input_tokens", 0) or 0) + read + written,
            "completion_tokens": int(getattr(usage, "output_tokens", 0) or 0),
            "cache_read_input_tokens": read,
            "cache_creation_input_tokens": written,
        },
        "transport": transport,
    }


@lru_cache(maxsize=8)
def _client(base_url: str, profile: str):
    """One SDK client per gateway URL and profile; it asks for a token before each request."""
    import anthropic
    from anthropic.lib.credentials import AccessToken

    tokens = databricks_token_provider(profile)
    if (urlsplit(tokens.host).hostname or "").lower() != (urlsplit(base_url).hostname or ""):
        raise DatabricksCredentialsError(
            "The Databricks profile signs in to a different workspace than the gateway URL.")

    def credentials(*, force_refresh: bool = False) -> AccessToken:
        # expires_at=0: the Databricks SDK caches and refreshes the token itself.
        return AccessToken(token=tokens.token(), expires_at=0)

    return anthropic.Anthropic(base_url=base_url, credentials=credentials, max_retries=2)


def _retry_after(response) -> float | None:
    from .structured import _retry_after_seconds

    headers = getattr(response, "headers", None)
    return _retry_after_seconds(headers.get("retry-after")) if headers is not None else None


def send_messages(config, body: dict, *, guards=()) -> dict:
    """Send one chat-completions ``body`` to Claude; return a chat-completions result.

    Raises ``EndpointError`` with only the HTTP status (or the failure's type) for
    transport failures, and ``CredentialsError`` with a login hint when no Databricks
    token is available.
    """
    import anthropic

    from .structured import CredentialsError, EndpointError

    base_url = config.transport_base_url or config.base_url
    request = messages_request(body)
    try:
        client = _client(base_url, config.databricks_profile).with_options(timeout=config.timeout)
        api = client.beta.messages if "betas" in request else client.messages
        if not body.get("stream"):
            return chat_completion(api.create(**request), streamed=False)
        abort_reason = None
        with api.stream(**request) as stream:
            for event in stream:
                check_cancelled()
                if guards and getattr(event, "type", None) == "text":
                    abort_reason = next(
                        (reason for guard, reason in guards if guard.feed(event.text)), None)
                    if abort_reason:
                        break
            message = (stream.current_message_snapshot if abort_reason
                       else stream.get_final_message())
        return chat_completion(message, streamed=True, abort_reason=abort_reason)
    except DatabricksCredentialsError as exc:
        raise CredentialsError(str(exc)) from None
    except anthropic.APIStatusError as exc:
        # Deliberately drop provider bodies: they may echo request content.
        raise EndpointError(f"Endpoint HTTP {exc.status_code} for /chat/completions",
                            http_status=exc.status_code,
                            retry_after=_retry_after(exc.response)) from None
    except anthropic.APIConnectionError as exc:
        raise EndpointError(f"Endpoint request failed: {type(exc).__name__}") from None


def check_gateway(base_url: str, model: str, *, profile: str = "", timeout: float = 60) -> dict:
    """Send a short request with no patient data; report whether Claude answered.

    Returns ``{"available", "http_status", "error"}``; errors are fixed texts.
    """
    import anthropic

    error = gateway_url_error(base_url)
    if error:
        return {"available": False, "http_status": None, "error": error}
    try:
        client = _client(base_url, profile).with_options(timeout=timeout, max_retries=1)
        message = client.messages.create(
            model=model, max_tokens=1024, output_config={"effort": "low"},
            messages=[{"role": "user", "content": "Reply with exactly: OK"}])
    except DatabricksCredentialsError as exc:
        return {"available": False, "http_status": None, "error": str(exc)}
    except anthropic.APIStatusError as exc:
        reasons = {401: "Databricks rejected the login; log in again with databricks auth login.",
                   403: "This Databricks account has no access to the gateway endpoint.",
                   404: "The gateway does not serve this endpoint name for your account; use "
                        "the full three-part name from your onboarding email."}
        return {"available": False, "http_status": exc.status_code,
                "error": reasons.get(exc.status_code,
                                     f"The gateway answered HTTP {exc.status_code}.")}
    except anthropic.APIConnectionError as exc:
        return {"available": False, "http_status": None,
                "error": f"Could not reach the Databricks gateway ({type(exc).__name__})."}
    answered = any(getattr(b, "type", None) == "text" for b in message.content or [])
    return {"available": answered, "http_status": 200,
            "error": None if answered else "Claude returned no text."}


__all__ = [
    "EFFORTS",
    "GATEWAY_PATH",
    "chat_completion",
    "check_gateway",
    "fallbacks_supported",
    "gateway_url_error",
    "messages_request",
    "request_field_error",
    "send_messages",
]
