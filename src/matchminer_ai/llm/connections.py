"""Connection profiles shared by the apps: an OpenAI-compatible server, the Vertex (Agent Platform) profiles and
Claude through a Databricks AI Gateway.

A connection names where a preset section's requests go. Three kinds:

- **an OpenAI-compatible server** (vLLM): an address normalized by ``normalize_endpoint`` and a model chosen from
  the server's ``/v1/models`` (``served_models``);
- **a Vertex profile** (``VERTEX_MODELS``): a hosted Gemini or Gemma model in the site config's
  ``vertex.project``, reached with this server's Application Default Credentials. The Google destination is built
  here from the profile (``vertex_profile``), never from browser input. Hosted Gemini manages its own sampling and
  thinking (``GEMINI_BACKENDS``); Gemma MaaS takes a thinking switch and needs request pacing (``GEMMA_PACING``);
- **Claude through a Databricks AI Gateway** (``claude_gateway_endpoint``): the gateway of the site config's
  ``databricks.host``, its Anthropic Messages route, a three-part endpoint name (``CLAUDE_ENDPOINT_NAME``) and the
  Databricks CLI login profile.

The ``vertex_*`` helpers give the parts of a preset section that a Vertex profile fixes (request parameters, extra
body, chat-template switches, context window, sampling profile); the apps apply them to their own sections.
"""

from __future__ import annotations

import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from urllib.parse import urlsplit

from .remote_auth import GOOGLE_AGENT_PLATFORM_PROVIDER
from .remote_inference import normalize_openai_base_url

# ---------------------------------------------------------------------------------------- OpenAI-compatible


def normalize_endpoint(value: str) -> str:
    """Accept a server address or an OpenAI-compatible API base URL (``host:port`` gains ``http://`` and ``/v1``).

    Raises ``ValueError`` for anything but an HTTP(S) URL without credentials, query or fragment.
    """
    value = (value or "").strip()
    if not value or any(ch.isspace() or ord(ch) < 32 for ch in value):
        raise ValueError("Enter an HTTP or HTTPS LLM server address.")
    candidate = value if "://" in value else "http://" + value
    try:
        parsed = urlsplit(candidate)
        valid = bool(
            parsed.scheme in {"http", "https"}
            and parsed.hostname
            and not (parsed.username or parsed.password or parsed.query or parsed.fragment)
        )
        parsed.port  # malformed or out-of-range ports raise here, before any request
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("Use an HTTP(S) server URL without credentials, query, or fragment.")
    return normalize_openai_base_url(f"{parsed.scheme}://{parsed.netloc}{parsed.path}")


async def served_models(endpoint: str, *, timeout: float = 10.0, api_key: str | None = None) -> list[dict]:
    """The models an OpenAI-compatible endpoint serves (``/models``): ``[{"id", "max_model_len"}]``, de-duplicated.

    Raises ``httpx.HTTPError``, ``ValueError``, ``KeyError`` or ``TypeError`` when the list cannot be read.
    """
    import httpx

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(endpoint.rstrip("/") + "/models", headers=headers)
        response.raise_for_status()
        data = response.json()["data"]
    if not isinstance(data, list):
        raise ValueError("expected a model list")
    models = [
        {"id": m["id"].strip(), "max_model_len": m.get("max_model_len")}
        for m in data
        if isinstance(m, dict) and isinstance(m.get("id"), str) and m["id"].strip() and len(m["id"]) <= 512
    ]
    return list({m["id"]: m for m in models}.values())


# ---------------------------------------------------------------------------------------- Vertex profiles

# Local chunk boundaries only; not Gemini's tokenizer or weights.
VERTEX_TOKENIZER = "Qwen/Qwen3-0.6B"
VERTEX_MODELS = {
    # Gemini 4 Hydrogen (GA 2026-11-12): 1,048,576-token window shared by input and output, up to 65,536 output.
    "gemini4_vertex": "google/gemini-4-hydrogen",
    "gemini_vertex": "google/gemini-3.8-flash",
    "gemma_vertex": "google/gemma-4-26b-a4b-it-maas",
}
VERTEX_LABELS = {
    "gemini4_vertex": "Gemini 4 Hydrogen",
    "gemini_vertex": "Gemini 3.8 Flash",
    "gemma_vertex": "Gemma 4 26B MaaS",
}
# Hosted Gemini models manage their own sampling and thinking (reasoning_effort, no chat-template switches).
GEMINI_BACKENDS = frozenset({"gemini4_vertex", "gemini_vertex"})
VERTEX_MAX_OUTPUT_TOKENS = 65536
# The Gemma MaaS endpoint's request pacing (shared remote settings).
GEMMA_PACING = {
    "request_start_interval_seconds": 2.0,
    "capacity_retry_initial_seconds": 15.0,
    "capacity_retry_max_seconds": 60.0,
}


def vertex_project() -> str:
    """The Google Cloud project of the Vertex profiles: the site config's ``vertex.project`` ("" when unset)."""
    from matchminer_ai.site import load_site_config

    return str(load_site_config().get("vertex.project") or "")


def vertex_endpoint(project: str | None = None) -> str:
    project = vertex_project() if project is None else project
    return f"https://aiplatform.googleapis.com/v1/projects/{project}/locations/global/endpoints/openapi"


def vertex_profile(backend: str) -> dict:
    """``{"endpoint", "model", "tokenizer"}`` of a Vertex profile (``KeyError`` for another backend)."""
    return {"endpoint": vertex_endpoint(), "model": VERTEX_MODELS[backend], "tokenizer": VERTEX_TOKENIZER}


def vertex_profiles() -> dict[str, dict]:
    return {backend: vertex_profile(backend) for backend in VERTEX_MODELS}


def vertex_context_tokens(backend: str) -> int:
    return 1048576 if backend in GEMINI_BACKENDS else 262144


def vertex_request_params(backend: str, max_tokens: int = 32768) -> dict:
    """Request parameters a Vertex profile fixes: the output cap and, for Gemini, high reasoning effort."""
    params: dict = {"max_tokens": min(VERTEX_MAX_OUTPUT_TOKENS, int(max_tokens))}
    if backend in GEMINI_BACKENDS:
        params["reasoning_effort"] = "high"
    return params


def vertex_chat_template(backend: str, thinking: bool) -> dict:
    """Chat-template switches: none for Gemini, Gemma's thinking switch."""
    return {} if backend in GEMINI_BACKENDS else {"enable_thinking": thinking}


def vertex_extra_body(backend: str, thinking: bool) -> dict:
    template = vertex_chat_template(backend, thinking)
    return {"chat_template_kwargs": template} if template else {}


def vertex_sampling_profile(backend: str) -> str:
    """Gemini samples on its own ("none"); Gemma takes the model profile's sampling ("auto")."""
    return "none" if backend in GEMINI_BACKENDS else "auto"


def vertex_remote(backend: str) -> dict:
    """The shared ``remote`` settings of a Vertex profile: provider, project, endpoint and Gemma's pacing."""
    values = {
        "provider": GOOGLE_AGENT_PLATFORM_PROVIDER,
        "google_project_id": vertex_project(),
        "server_urls": [vertex_endpoint()],
    }
    if backend not in GEMINI_BACKENDS:
        values.update(GEMMA_PACING)
    return values


async def vertex_available(backend: str, *, timeout: float = 30) -> bool:
    """Whether this server's credentials reach a Vertex profile's model (a short request with no patient data)."""
    import asyncio

    from .vllm_server import check_openai_endpoint

    profile = vertex_profile(backend)
    try:
        checks = await asyncio.to_thread(
            check_openai_endpoint,
            profile["endpoint"],
            provider=GOOGLE_AGENT_PLATFORM_PROVIDER,
            google_project_id=vertex_project(),
            model_name=profile["model"],
            timeout=timeout,
        )
        return bool(checks and checks[0][1] == 200)
    except Exception:
        return False


CREDENTIALS_TIMEOUT = 15.0  # seconds to get a Google token before calling the credentials unavailable
_credentials: dict = {"checked": None, "error": None}
_credentials_lock = threading.Lock()


def vertex_credentials_error(max_age: float = 60.0) -> str | None:
    """None when this server's Application Default Credentials give a Google token, else why not. Checked at most
    once per ``max_age`` seconds and never for longer than ``CREDENTIALS_TIMEOUT``, so missing credentials are
    reported, not waited on."""
    with _credentials_lock:
        checked = _credentials["checked"]
        if checked is not None and time.monotonic() - checked < max_age:
            return _credentials["error"]

    def token():
        import google.auth
        from google.auth.transport.requests import Request

        credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        credentials.refresh(Request())
        if not credentials.token:
            raise ValueError("no token")

    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vertex-credentials")
    error: str | None
    try:
        pool.submit(token).result(timeout=CREDENTIALS_TIMEOUT)
        error = None
    except FutureTimeout:
        error = (
            f"Vertex credentials not available: no Google token within {CREDENTIALS_TIMEOUT:.0f} s from this "
            f"server's Application Default Credentials (project {vertex_project()})."
        )
    except Exception as exc:
        error = (
            f"Vertex credentials not available on this server ({type(exc).__name__}): set up Google Application "
            f"Default Credentials with access to project {vertex_project()}, or choose an OpenAI-compatible "
            f"endpoint."
        )
    finally:
        pool.shutdown(wait=False)
    with _credentials_lock:
        _credentials.update(checked=time.monotonic(), error=error)
    return error


# ---------------------------------------------------------------------------------------- Claude via Databricks

# Gateway endpoint names have three dot-separated parts, the last ending in the Claude model ID.
CLAUDE_ENDPOINT_NAME = re.compile(r"[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]*claude-[a-z0-9-]+")
CLAUDE_CONTEXT_TOKENS = 1_000_000
DEFAULT_DATABRICKS_PROFILE = "unity-gateway"


def databricks_host() -> str:
    """The Databricks workspace whose AI Gateway serves Claude: the site config's ``databricks.host``."""
    from matchminer_ai.site import load_site_config

    return str(load_site_config().get("databricks.host") or "").rstrip("/")


def databricks_profile() -> str:
    """The Databricks CLI login profile (``databricks auth login --profile``): site ``databricks.profile``."""
    from matchminer_ai.site import load_site_config

    return str(load_site_config().get("databricks.profile") or DEFAULT_DATABRICKS_PROFILE)


def claude_gateway_endpoint(host: str | None = None) -> str:
    """The gateway's Anthropic Messages route."""
    return (databricks_host() if host is None else host.rstrip("/")) + "/ai-gateway/anthropic"
