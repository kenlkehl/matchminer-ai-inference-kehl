"""OpenAI-compatible HTTP client with bounded retries and per-request checkpoints."""

import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass, field, replace
from email.utils import parsedate_to_datetime
from importlib import resources
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from matchminer_ai._storage import atomic_json, digest, read_json
from matchminer_ai.cancellation import cancel_sleep, check_cancelled, run_cancellable

from .remote_auth import (
    GOOGLE_AGENT_PLATFORM_PROVIDER,
    remote_bearer_token,
    remote_provider_name,
)
from .remote_inference import build_remote_request_config, normalize_remote_server_urls


@dataclass(frozen=True)
class StructuredConfig:
    base_url: str = "http://localhost:8000/v1"
    model: str = ""
    api_key_env: str = "OPENAI_API_KEY"
    max_concurrent_requests: int = 32
    max_tokens: int = 100000
    timeout: float = 7200
    attempts: int = 3
    response_format: str = "json_object"
    thinking: str = "on"
    max_prompt_chars: int | None = None
    context_window: int = 262144
    safety_tokens: int = 2048
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 64
    tokenizer_mode: str = "endpoint"
    stream: bool = True

    request_params: dict = field(default_factory=dict)
    extra_body: dict = field(default_factory=dict)
    provider: str = "openai"
    google_project_id: str = ""
    request_start_interval_seconds: float = 0.0
    capacity_retry_initial_seconds: float = 0.0
    capacity_retry_max_seconds: float = 60.0

    def __post_init__(self):
        for name in (
            "request_start_interval_seconds", "capacity_retry_initial_seconds",
            "capacity_retry_max_seconds",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0
            ):
                raise ValueError(f"{name} must be finite and nonnegative.")
        if self.capacity_retry_max_seconds < self.capacity_retry_initial_seconds:
            raise ValueError("Capacity retry maximum must be at least its initial delay.")

    def public_dict(self):
        result = asdict(self)  # Contains only the key's environment-variable NAME.
        # Keep existing checkpoint identities when pacing is not configured.
        for name, default in (
            ("request_start_interval_seconds", 0.0),
            ("capacity_retry_initial_seconds", 0.0),
            ("capacity_retry_max_seconds", 60.0),
        ):
            if result[name] == default:
                result.pop(name)
        if self.provider == "openai" and not self.google_project_id:
            # Preserve existing local endpoint checkpoint identities.
            result.pop("provider")
            result.pop("google_project_id")
        return result


class EndpointError(RuntimeError):
    def __init__(self, message, *, http_status=None, retry_after=None):
        super().__init__(message)
        self.http_status = http_status
        self.retry_after = retry_after


def _retry_after_seconds(value):
    """Parse only a delay; never retain arbitrary provider header text."""
    if not value:
        return None
    try:
        seconds = float(value)
    except (ValueError, TypeError):
        try:
            seconds = parsedate_to_datetime(value).timestamp() - time.time()
        except (ValueError, TypeError, OverflowError):
            return None
    return max(0.0, seconds) if math.isfinite(seconds) else None


class RepairError(EndpointError):
    """A bounded citation repair failed; preserve its clinical response for resume."""


JSON_NORMALIZATION_VERSION = "lossless-lists-v1"


def retry_draft(response):
    """Return only a complete JSON draft, never reasoning or a truncated stream."""
    try:
        choice = response["choices"][0]
        if choice.get("finish_reason") != "stop":
            return None
        value, _ = parse_model_json(choice["message"]["content"])
        return json.dumps(value, ensure_ascii=False)
    except (ValueError, KeyError, TypeError, IndexError):
        return None


def parse_model_json(content):
    """Recover repeated list keys without losing model content or choosing scalar values."""
    if not isinstance(content, str):
        raise ValueError("Expected a text JSON response")
    counts = {
        "identical_duplicate_keys": 0,
        "merged_list_keys": 0,
        "recovered_list_items": 0,
    }

    def object_from_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key not in result:
                result[key] = value
            elif result[key] == value:
                counts["identical_duplicate_keys"] += 1
            elif isinstance(result[key], list) and isinstance(value, list):
                counts["merged_list_keys"] += 1
                for item in value:
                    if item not in result[key]:
                        result[key].append(item)
                        counts["recovered_list_items"] += 1
            else:
                raise ValueError(
                    f"Conflicting duplicate JSON key: {key}; emit each object key once"
                )
        return result

    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
    return json.loads(content, object_pairs_hook=object_from_pairs), counts


def read_chat_stream(lines, *, guards=()):
    """Assemble provider text verbatim; reject sustained degenerate generation early."""
    result = {
        "choices": [
            {
                "index": 0,
                "finish_reason": None,
                "message": {"role": "assistant", "content": ""},
            }
        ],
        "usage": None,
        "transport": {"streamed": True},
    }
    fragments = {"content": [], "reasoning": []}
    tails = {"content": "", "reasoning": ""}
    choice = result["choices"][0]
    for raw in lines:
        check_cancelled()
        line = raw.decode("utf-8").strip() if isinstance(raw, bytes) else raw.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        event = json.loads(data)
        if "error" in event:
            raise EndpointError("Endpoint returned a streaming error event")
        for key in ("id", "model", "created", "system_fingerprint"):
            if key in event:
                result[key] = event[key]
        if event.get("usage") is not None:
            result["usage"] = event["usage"]
        aborted = False
        for item in event.get("choices", []):
            if item.get("index", 0) != 0:
                raise EndpointError("Expected one streamed completion choice")
            delta = item.get("delta", {})
            for key in fragments:
                text = delta.get(key)
                if key == "reasoning" and text is None:
                    text = delta.get("reasoning_content")
                if not isinstance(text, str) or not text:
                    continue
                fragments[key].append(text)
                if key == "content":
                    for guard, reason in guards:
                        if guard.feed(text):
                            choice["finish_reason"] = "client_aborted_repetition"
                            result["transport"]["abort_reason"] = reason
                            aborted = True
                            break
                    if aborted:
                        break
                tails[key] = (tails[key] + text)[-8192:]
                tail = tails[key]
                if len(tail) == 8192:
                    words = tail.split()
                    repetitive = len(words) >= 500 and (
                        len(set(words)) <= 10 or len(set("".join(words))) <= 6
                    )
                    if not words or repetitive:
                        choice["finish_reason"] = "client_aborted_repetition"
                        result["transport"]["abort_reason"] = (
                            f"Sustained repetitive {key} output"
                        )
                        aborted = True
                        break
            if aborted:
                break
            if item.get("finish_reason") is not None:
                choice["finish_reason"] = item["finish_reason"]
        if aborted:
            break
    choice["message"]["content"] = "".join(fragments["content"])
    if fragments["reasoning"]:
        choice["message"]["reasoning"] = "".join(fragments["reasoning"])
    return result


class StructuredClient:
    def __init__(self, config: StructuredConfig, cache_dir: Path | None):
        self.config = config
        self.cache_dir = cache_dir
        self._token_counts: dict[str, int] = {}
        parsed = urlsplit(config.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("base_url must be an HTTP(S) OpenAI-compatible /v1 URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError(
                "Do not put credentials, queries or fragments in base_url; use api_key_env"
            )
        if self.prompt_budget <= 0:
            raise ValueError(
                "Context window must exceed output reserve plus safety tokens"
            )

    def normalize_result(self, value):
        """Normalize representation only; subclasses must not rewrite clinical content."""
        return 0

    def preserved_content(self, value):
        return value

    def stream_guards(self, schema):
        return ()

    def retry_feedback(self, schema, error):
        template = (
            resources.files("matchminer_ai.prompts")
            .joinpath("structured.retry.txt")
            .read_text(encoding="utf-8")
        )
        return template.format(error=error[:1200]).rstrip()

    def retry_feedback_history(self, schema, errors):
        return self.retry_feedback(schema, errors[-1])

    @property
    def prompt_budget(self):
        return (
            self.config.context_window
            - self.config.max_tokens
            - self.config.safety_tokens
        )

    def _http(self, endpoint, body=None, *, server_root=False, output_schema=None):
        return run_cancellable(lambda: self._http_request(
            endpoint, body, server_root=server_root, output_schema=output_schema,
        ))

    def _http_request(self, endpoint, body=None, *, server_root=False, output_schema=None):
        check_cancelled()
        headers = {"Content-Type": "application/json"}
        key = os.environ.get(self.config.api_key_env)
        if self.config.provider == GOOGLE_AGENT_PLATFORM_PROVIDER:
            parsed = urlsplit(self.config.base_url)
            if (
                parsed.scheme != "https"
                or not re.fullmatch(
                    r"(?:[a-z0-9-]+-)?aiplatform\.googleapis\.com", parsed.hostname or ""
                )
                or parsed.username or parsed.password or parsed.port not in (None, 443)
            ):
                raise ValueError("Google credentials require an HTTPS Agent Platform endpoint.")
            key = remote_bearer_token({
                "provider": self.config.provider,
                "google_project_id": self.config.google_project_id,
            })
        if key:
            headers["Authorization"] = "Bearer " + key
        base = self.config.base_url.rstrip("/")
        if server_root and base.endswith("/v1"):
            base = base[:-3]
        request = Request(
            base + endpoint,
            headers=headers,
            data=json.dumps(body).encode() if body is not None else None,
        )
        from .request_limits import endpoint_slot

        # These CPU/metadata routes do not generate tokens. Sharing their slots
        # with long completions can delay each exact-tokenizer packing check by
        # an entire generation. Keep preparation bounded independently.
        preparation = endpoint in {"/models", "/tokenize"}
        limit = self.config.max_concurrent_requests
        if preparation:
            limit = min(limit, 4)
        dispatch_wait = 0.0
        try:
            with endpoint_slot(
                self.config.base_url,
                limit,
                pool="preparation" if preparation else "generation",
            ):
                if endpoint == "/chat/completions" and (
                    self.config.request_start_interval_seconds
                    or self.config.capacity_retry_initial_seconds
                ):
                    from .request_pacing import endpoint_pacer

                    dispatch_wait = endpoint_pacer(
                        self.config.base_url, self.config.model
                    ).wait(self.config.request_start_interval_seconds)
                check_cancelled()
                with urlopen(request, timeout=self.config.timeout) as response:
                    if endpoint == "/chat/completions" and body and body.get("stream"):
                        result = read_chat_stream(
                            response, guards=self.stream_guards(output_schema)
                        )
                    else:
                        result = json.load(response)
                check_cancelled()
                if endpoint == "/chat/completions" and isinstance(result, dict):
                    if not isinstance(result.get("transport"), dict):
                        result["transport"] = {}
                    result["transport"]["dispatch_wait_seconds"] = round(dispatch_wait, 4)
                return result
        except HTTPError as exc:
            # Deliberately don't persist server bodies (they may echo credentials).
            raise EndpointError(
                f"Endpoint HTTP {exc.code} for {endpoint}",
                http_status=exc.code,
                retry_after=_retry_after_seconds(exc.headers.get("Retry-After")) if exc.headers else None,
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise EndpointError(
                f"Endpoint request failed: {type(exc).__name__}: {exc}"
            ) from exc

    def discover(self):
        models = self._http("/models").get("data", [])
        if not models or not models[0].get("id"):
            raise EndpointError(
                "Endpoint returned no model IDs; pass model_name if discovery is unsupported"
            )
        if self.config.model:
            matches = [m for m in models if m.get("id") == self.config.model]
            if not matches:
                raise EndpointError(
                    f"Configured model {self.config.model!r} is not reported by /models"
                )
            return matches[0]
        return models[0]

    def count_tokens(self, messages):
        identity = digest(
            {
                "messages": messages,
                "model": self.config.model,
                "base_url": self.config.base_url,
                "thinking": self.config.thinking,
                "template": self.config.extra_body.get("chat_template_kwargs"),
                "tokenizer_mode": self.config.tokenizer_mode,
                "context_window": self.config.context_window,
            }
        )
        if identity in self._token_counts:
            return self._token_counts[identity]
        token_file = (
            self.cache_dir / "token_counts" / (identity + ".json")
            if self.cache_dir is not None else None
        )
        if token_file is not None and token_file.exists():
            saved = read_json(token_file)
            if (
                saved.get("identity") != identity
                or type(saved.get("count")) is not int
                or saved["count"] < 1
            ):
                raise ValueError("Corrupted tokenizer checkpoint")
            self._token_counts[identity] = saved["count"]
            return saved["count"]
        if self.config.tokenizer_mode == "bytes":
            # Explicit portability fallback; never silently substituted for the model tokenizer.
            count = sum(len(m["content"].encode("utf-8")) + 32 for m in messages) + 64
        else:
            body = {
                "model": self.config.model,
                "messages": messages,
                "add_generation_prompt": True,
            }
            if self.config.thinking != "default":
                body["chat_template_kwargs"] = {
                    "enable_thinking": self.config.thinking == "on"
                }
            if "chat_template_kwargs" in self.config.extra_body:
                body["chat_template_kwargs"] = self.config.extra_body[
                    "chat_template_kwargs"
                ]
            result = self._http("/tokenize", body, server_root=True)
            count = result.get("count")
            if type(count) is not int or count < 1:
                raise EndpointError(
                    "Tokenizer did not return a positive count; configure tokenizer_mode='bytes' for a non-vLLM endpoint"
                )
            server_limit = result.get("max_model_len")
            if (
                isinstance(server_limit, int)
                and self.config.context_window > server_limit
            ):
                raise EndpointError(
                    "Configured context window exceeds tokenizer's reported model limit"
                )
        self._token_counts[identity] = count
        if token_file is not None:
            atomic_json(token_file, {"identity": identity, "count": count})
        return count

    def _complete_in_memory(self, body, messages, schema, validator):
        """Validate transient clinical requests without writing prompts or responses."""
        feedback = None
        for attempt in range(self.config.attempts):
            request = dict(body)
            request["messages"] = messages + (
                [{"role": "user", "content": feedback}] if feedback else []
            )
            if self.config.stream:
                request.update(stream=True, stream_options={"include_usage": True})
            if not self.fits(request["messages"], use_safety_margin=True):
                raise EndpointError("Structured retry exceeds the context budget.")
            try:
                response = self._http("/chat/completions", request, output_schema=schema)
                choice = response["choices"][0]
                if choice.get("finish_reason") != "stop":
                    raise ValueError("Response was incomplete; return the full JSON.")
                value, _ = parse_model_json(choice["message"]["content"])
                validator(value)
                return value
            except (ValueError, KeyError, TypeError, IndexError, EndpointError) as exc:
                if self._capacity_retry(exc, attempt):
                    # Capacity rejection does not invalidate the prompt or add
                    # model feedback. The next HTTP attempt waits at dispatch.
                    continue
                # Do not echo provider output, invalid quotes, or patient content.
                feedback = resources.files("matchminer_ai.prompts").joinpath(
                    "structured.memory_retry.txt"
                ).read_text(encoding="utf-8")
                if attempt + 1 < self.config.attempts:
                    cancel_sleep(min(2 ** attempt, 8))
        raise EndpointError("Structured review failed validation after bounded retries.")

    def _capacity_retry(self, error, attempt):
        if (
            not isinstance(error, EndpointError)
            or error.http_status not in {429, 503}
            or not self.config.capacity_retry_initial_seconds
        ):
            return False
        from .request_pacing import capacity_delay, endpoint_pacer

        delay = capacity_delay(
            attempt, self.config.capacity_retry_initial_seconds,
            self.config.capacity_retry_max_seconds, error.retry_after,
        )
        # Share cooldown even after the last attempt: other question workers
        # must not immediately replace a rejected request with another burst.
        endpoint_pacer(self.config.base_url, self.config.model).defer(delay)
        return True

    def fits(self, messages, *, use_safety_margin=False):
        if (
            self.config.max_prompt_chars is not None
            and sum(len(m["content"]) for m in messages) > self.config.max_prompt_chars
        ):
            return False
        budget = self.prompt_budget + (
            self.config.safety_tokens if use_safety_margin else 0
        )
        return self.count_tokens(messages) <= budget

    def retry_messages(self, messages, schema, errors, draft):
        """Let the model revise its draft while keeping all original source context."""
        feedback = self.retry_feedback_history(schema, errors)
        fallback = messages + [{"role": "user", "content": feedback}]
        if draft is not None:
            instruction = resources.files("matchminer_ai.prompts").joinpath(
                "structured.revise_retry.txt"
            ).read_text(encoding="utf-8").strip()
            revision = messages + [
                {"role": "assistant", "content": draft},
                {"role": "user", "content": instruction + "\n\n" + feedback},
            ]
            # Never trim source context or the output reserve to squeeze in a draft.
            if self.fits(revision, use_safety_margin=True):
                return revision
        return fallback

    def complete(
        self, job, messages, schema, validator, repair_handler=None,
        *, reuse_exhausted=False,
    ):
        from .remote_auth import prepare_messages_for_provider

        messages = prepare_messages_for_provider(
            messages, {"provider": self.config.provider, "model_name": self.config.model}
        )
        body = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_tokens,
        }
        if self.config.top_k:
            body["top_k"] = self.config.top_k
        if self.config.response_format == "json_object":
            body["response_format"] = {"type": "json_object"}
        elif self.config.response_format == "json_schema":
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_response",
                    "strict": True,
                    "schema": schema,
                },
            }
        if self.config.thinking != "default":
            body["chat_template_kwargs"] = {
                "enable_thinking": self.config.thinking == "on"
            }
        body.update(self.config.request_params)
        body.update(self.config.extra_body)
        if (
            self.config.provider == GOOGLE_AGENT_PLATFORM_PROVIDER
            and self.config.model.removeprefix("google/").startswith("gemini-3.8-")
        ):
            for name in (
                "temperature", "top_p", "top_k", "min_p", "presence_penalty",
                "frequency_penalty", "repetition_penalty", "candidate_count",
                "chat_template_kwargs",
            ):
                body.pop(name, None)
        if not self.fits(messages):
            raise ValueError(
                f"{job}: prompt exceeds {self.prompt_budget} available input tokens "
                f"after reserving {self.config.max_tokens} output tokens"
            )
        if self.cache_dir is None:
            if repair_handler is not None:
                raise ValueError("In-memory requests do not support checkpoint repairs.")
            return self._complete_in_memory(body, messages, schema, validator)
        identity = digest(
            {"base_url": self.config.base_url, "body": body, "schema": schema}
        )
        folder = self.cache_dir / identity
        accepted = folder / "accepted.json"
        if accepted.exists():
            cached = read_json(accepted)
            if (
                cached["request_sha256"] != identity
                or digest(cached["result"]) != cached["result_sha256"]
            ):
                raise ValueError(f"Corrupted checkpoint for {job}")
            validator(cached["result"])
            if cached.get("json_normalization_version") == JSON_NORMALIZATION_VERSION:
                return cached["result"]
            for path in sorted(folder.glob("attempt-*.json"), reverse=True):
                response = read_json(path)
                if (
                    response.get("usage") != cached.get("usage")
                    or response.get("model") != cached.get("response_model")
                    or response["choices"][0].get("finish_reason") != "stop"
                ):
                    continue
                try:
                    value, counts = parse_model_json(
                        response["choices"][0]["message"]["content"]
                    )
                    self.normalize_result(value)
                except (ValueError, KeyError, TypeError):
                    continue
                equivalent = (
                    self.preserved_content(value)
                    == self.preserved_content(cached["result"])
                    if cached.get("repair_applied")
                    else value == cached["result"]
                )
                if equivalent:
                    cached.update(
                        json_normalization_version=JSON_NORMALIZATION_VERSION,
                        json_normalization=counts,
                        response_sha256=digest(response),
                    )
                    atomic_json(accepted, cached)
                    return cached["result"]
            # Revalidate old raw responses with the lossless parser. Earlier
            # parsers may have discarded arrays under repeated object keys.
        atomic_json(
            folder / "request.json",
            {
                "job": job,
                "base_url": self.config.base_url,
                "body": body,
                "prompt_tokens": self.count_tokens(messages),
                "context_window": self.config.context_window,
                "reserved_output_tokens": self.config.max_tokens,
                "safety_tokens": self.config.safety_tokens,
                "tokenizer_mode": self.config.tokenizer_mode,
            },
        )

        def accept_response(response, *, allow_repair=True):
            choice = response["choices"][0]
            if choice.get("finish_reason") == "client_aborted_repetition":
                reason = response.get("transport", {}).get(
                    "abort_reason", "Sustained repetitive output"
                )
                raise ValueError(
                    reason + ". Complete the finite JSON "
                    "task without repeating completed records or emitting padding."
                )
            if choice.get("finish_reason") != "stop":
                raise ValueError(
                    f"Incomplete response: finish_reason={choice.get('finish_reason')!r}; "
                    "increase max_tokens or reduce the packet"
                )
            value, json_normalization = parse_model_json(choice["message"]["content"])
            normalized_citations = self.normalize_result(value)
            repair_error = None
            try:
                validator(value)
            except ValueError as exc:
                if repair_handler is None or not allow_repair:
                    raise
                repair_error = str(exc)
                try:
                    value = repair_handler(value, repair_error)
                except EndpointError as repair_exc:
                    raise RepairError(
                        f"{job}: targeted repair failed: {repair_exc}"
                    ) from repair_exc
                validator(value)
            atomic_json(
                accepted,
                {
                    "job": job,
                    "request_sha256": identity,
                    "result": value,
                    "result_sha256": digest(value),
                    "usage": response.get("usage"),
                    "normalized_citation_items": normalized_citations,
                    "json_normalization_version": JSON_NORMALIZATION_VERSION,
                    "json_normalization": json_normalization,
                    "response_sha256": digest(response),
                    "repair_applied": repair_error is not None,
                    "original_validation_error": repair_error,
                    "response_model": response.get("model"),
                },
            )
            return value

        # Recover a response written immediately before a crash/validation interruption.
        previous = sorted(
            folder.glob("attempt-*.json"), key=lambda p: int(p.stem.split("-")[-1])
        )
        previous_numbers = [int(p.stem.split("-")[-1]) for p in previous]
        previous_numbers += [
            int(p.stem.split("-")[-1]) for p in folder.glob("failure-*.json")
        ]
        # A cancelled stream may have a recorded request but no response. Keep
        # that attempt visible instead of overwriting it on resume.
        previous_numbers += [
            int(p.stem.split("-")[-1]) for p in folder.glob("request-attempt-*.json")
        ]
        offset = max(previous_numbers, default=0)
        last_error = None
        last_draft = None
        retry_errors = []
        for path in reversed(previous):
            response = read_json(path)
            try:
                return accept_response(response, allow_repair=False)
            except (ValueError, KeyError, TypeError, IndexError) as exc:
                if last_error is None:
                    last_error = str(exc)
                if last_draft is None:
                    last_draft = retry_draft(response)
                retry_errors.insert(0, str(exc))
                continue
        # Try all saved responses without generation first. If none validates,
        # repair the earliest usable response instead of repeatedly rewriting its
        # clinical records. Failed bounded repairs propagate and stay resumable.
        if repair_handler is not None:
            for path in previous:
                try:
                    return accept_response(read_json(path))
                except (ValueError, KeyError, TypeError, IndexError) as exc:
                    # A failed repair may expose a more specific grounding error
                    # than raw draft validation. Preserve it for regeneration on
                    # resume rather than replaying only the original quote error.
                    last_error = str(exc)
                    retry_errors.append(last_error)
                    continue
        # A caller with an alternate repair strategy need not repeat an already
        # exhausted strategy on every resume. Still recover valid saved answers
        # above, and never count an interrupted request as a failed response.
        if (
            reuse_exhausted
            and len(list(folder.glob("failure-*.json"))) >= self.config.attempts
        ):
            raise EndpointError(f"{job}: saved attempts exhausted: {last_error}")
        last_capacity_error = None
        for attempt in range(1, self.config.attempts + 1):
            attempt_body = dict(body)
            if self.config.stream:
                attempt_body.update(stream=True, stream_options={"include_usage": True})
            response = None
            try:
                if last_error is not None:
                    attempt_body["messages"] = self.retry_messages(
                        messages, schema, retry_errors, last_draft
                    )
                # The initial pack leaves safety space for validation feedback. A retry
                # may use that space but must always preserve the entire output reserve.
                if not self.fits(attempt_body["messages"], use_safety_margin=True):
                    raise ValueError(
                        "Retry prompt exceeds input budget; output reserve will not be reduced"
                    )
                atomic_json(
                    folder / f"request-attempt-{offset + attempt}.json",
                    {
                        "body": attempt_body,
                        "prompt_tokens": self.count_tokens(attempt_body["messages"]),
                        "context_window": self.config.context_window,
                        "reserved_output_tokens": self.config.max_tokens,
                    },
                )
                response = self._http(
                    "/chat/completions", attempt_body, output_schema=schema
                )
                atomic_json(folder / f"attempt-{offset + attempt}.json", response)
                return accept_response(response)
            except RepairError as exc:
                atomic_json(
                    folder / f"failure-{offset + attempt}.json",
                    {"error": str(exc), "job": job},
                )
                raise
            except (ValueError, KeyError, TypeError, IndexError, EndpointError) as exc:
                if self._capacity_retry(exc, attempt - 1):
                    last_capacity_error = str(exc)
                    atomic_json(
                        folder / f"failure-{offset + attempt}.json",
                        {"error": str(exc), "job": job},
                    )
                    continue
                last_error = str(exc)
                draft = retry_draft(response)
                if draft is not None:
                    last_draft = draft
                retry_errors.append(last_error)
                atomic_json(
                    folder / f"failure-{offset + attempt}.json",
                    {"error": last_error, "job": job},
                )
                if attempt < self.config.attempts:
                    cancel_sleep(min(2 ** (attempt - 1), 8))
        raise EndpointError(
            f"{job}: exhausted {self.config.attempts} attempts: {last_error or last_capacity_error}"
        )


def resolve_structured_config(
    runtime, *, cache_dir, previous=None, client_type=StructuredClient
):
    """Resolve the existing remote runtime into a checkpointed structured request.

    A deliberately single-endpoint workflow: discovery and token counting must
    describe the same model serving the requests. Explicit model/context settings
    support providers without discovery; bytes token accounting is opt-in.
    """
    urls = normalize_remote_server_urls(runtime)
    if len(urls) != 1:
        raise ValueError(
            "Structured guideline extraction requires exactly one remote server URL."
        )
    params, extra = build_remote_request_config(runtime)
    if "max_completion_tokens" in params or "max_completion_tokens" in extra:
        raise ValueError(
            "Guideline generation uses request_params.max_tokens; do not set max_completion_tokens."
        )
    reserved = {
        "model",
        "messages",
        "stream",
        "stream_options",
        "response_format",
        "n",
        "tools",
        "tool_choice",
        "extra_headers",
        "extra_query",
        "extra_body",
        "api_key",
    }
    if reserved.intersection(params.keys() | extra.keys()):
        raise ValueError(
            "Guideline request parameters cannot override model/messages, streaming, schema, tools, choice count, or credentials."
        )
    if params.keys() & extra.keys():
        raise ValueError(
            "Do not duplicate guideline request_params keys in extra_body."
        )
    if {"max_tokens", "temperature", "top_p"}.intersection(extra):
        raise ValueError(
            "Put max_tokens, temperature and top_p in guideline.remote.request_params."
        )
    if {"chat_template_kwargs", "top_k"}.intersection(params):
        raise ValueError(
            "Put top_k and chat_template_kwargs in guideline.remote.extra_body."
        )
    kwargs = extra.get("chat_template_kwargs", {})
    if not isinstance(kwargs, dict):
        raise ValueError("chat_template_kwargs must be a mapping.")
    enabled = kwargs.get("enable_thinking")
    if enabled is not None and type(enabled) is not bool:
        raise ValueError("enable_thinking must be boolean.")
    thinking = "default" if enabled is None else "on" if enabled else "off"
    configured_model = runtime.get("model_name")
    configured_context = runtime.get("context_window")
    saved = (previous or {}).get("llm", {})
    same_endpoint = saved.get("base_url") == urls[0]
    model = configured_model or (saved.get("model") if same_endpoint else "") or ""
    context = configured_context or (
        saved.get("context_window")
        if same_endpoint
        and (not configured_model or configured_model == saved.get("model"))
        else None
    )
    maximum = params.pop("max_tokens", 100000)
    safety = runtime.get("safety_tokens", 2048)
    attempts = runtime.get("max_retries", 3)
    for name, value in (
        ("max_tokens", maximum),
        ("safety_tokens", safety),
        ("max_retries", attempts),
    ):
        minimum = 0 if name == "safety_tokens" else 1
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    if configured_context is not None and (
        type(configured_context) is not int or configured_context <= 0
    ):
        raise ValueError(
            "context_window must be a positive integer or None for discovery."
        )
    mode = runtime.get("tokenizer_mode", "endpoint")
    if mode not in {"endpoint", "bytes"}:
        raise ValueError("tokenizer_mode must be 'endpoint' or 'bytes'.")
    response_format = runtime.get("response_format", "json_schema")
    if response_format not in {"json_schema", "json_object", "none"}:
        raise ValueError("response_format must be json_schema, json_object, or none.")
    timeout = runtime.get("request_timeout", 7200)
    if (
        not isinstance(timeout, (int, float))
        or isinstance(timeout, bool)
        or timeout <= 0
    ):
        raise ValueError("request_timeout must be positive.")
    stream = runtime.get("stream", True)
    if type(stream) is not bool:
        raise ValueError("stream must be boolean.")
    config = StructuredConfig(
        base_url=urls[0],
        provider=remote_provider_name(runtime),
        google_project_id=runtime.get("google_project_id", ""),
        request_start_interval_seconds=runtime.get("request_start_interval_seconds", 0.0),
        capacity_retry_initial_seconds=runtime.get("capacity_retry_initial_seconds", 0.0),
        capacity_retry_max_seconds=runtime.get("capacity_retry_max_seconds", 60.0),
        model=model,
        context_window=context or maximum + safety + 1,
        max_tokens=maximum,
        safety_tokens=safety,
        timeout=timeout,
        attempts=attempts,
        max_concurrent_requests=runtime.get("max_concurrent_requests", 32),
        temperature=params.pop("temperature", 1.0),
        top_p=params.pop("top_p", 0.95),
        top_k=extra.pop("top_k", 0),
        thinking=thinking,
        response_format=response_format,
        tokenizer_mode=mode,
        stream=stream,
        api_key_env=runtime.get("api_key_env", "OPENAI_API_KEY"),
        request_params=params,
        extra_body=extra,
    )
    discovery = None
    if not model or context is None:
        discovery = client_type(config, cache_dir).discover()
        model = discovery["id"]
        if context is None:
            context = discovery.get("max_model_len")
        if type(context) is not int or context < 1:
            raise EndpointError(
                "Endpoint does not advertise max_model_len; set guideline.context_window explicitly."
            )
    actual_runtime = {**runtime, "model_name": model}
    final_params, final_extra = build_remote_request_config(actual_runtime)
    final_params.pop("max_tokens", None)
    template = final_extra.get("chat_template_kwargs", {})
    enabled = template.get("enable_thinking")
    config = replace(
        config,
        model=model,
        context_window=context,
        temperature=final_params.pop("temperature", 1.0),
        top_p=final_params.pop("top_p", 0.95),
        top_k=final_extra.pop("top_k", 0),
        thinking="default" if enabled is None else "on" if enabled else "off",
        request_params=final_params,
        extra_body=final_extra,
    )
    # Validate URL and combined budgets before creating any generation artifacts.
    client_type(config, cache_dir)
    return config, {
        "model_name": model,
        "base_url": urls[0],
        "context_window": context,
        "discovery": discovery,
        "source": "configured_remote_endpoint",
    }
