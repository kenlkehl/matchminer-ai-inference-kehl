"""The LLM endpoint the dashboard's LLM checks use (GoodOption check, per-criterion LLM check).

Modeled on the guideline fetcher's connection settings (``guideline-fetcher/settings.py``: ``normalize_endpoint``, the
Vertex profiles, building the inference package's config with ``remote.server_urls`` and the model, and model discovery
from ``/v1/models``), copied here rather than imported. Two kinds of connection:

- a Vertex profile, by default Gemini 4 Hydrogen (the fetcher's ``gemini4_vertex``; user, 2026-11-17): the site
  config's ``vertex.project``, this server's Application Default Credentials, hosted Gemini managing its own sampling
  and thinking.
  The Google destination is built here from the profile, never from browser input. The credentials are checked
  before a request (at most once a minute, bounded by a timeout), so their absence is reported, not waited on;
- an OpenAI-compatible server (vLLM): a URL, by default the site config's ``endpoints.teacher.url`` (an on-prem
  Nemotron 3.5 Super server), and a model chosen from the server's ``/v1/models``.

Sampling comes from the inference package's model profiles: each check's section gets ``sampling_profile: auto`` and
loses the section's own fixed sampling keys (the GoodOption preset's greedy decoding), so the served model's profile
applies (Nemotron 3.5 Super: NVIDIA's). Hosted Gemini gets ``sampling_profile: none`` and no sampling keys. The choice is
set at startup (``--llm-backend``, ``--llm-endpoint``, ``--llm-model``) and can be changed on Data and Models for the
server's session; nothing is written to disk.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from copy import deepcopy
from dataclasses import asdict, dataclass, field
import threading
import time
from urllib.parse import urlsplit, urlunsplit

NEMOTRON = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4"
OPENAI = "openai"  # an OpenAI-compatible server (vLLM)
# Local chunk boundaries only; not Gemini's tokenizer or weights (the fetcher's choice).
VERTEX_TOKENIZER = "Qwen/Qwen3-0.6B"
VERTEX_MODELS = {"gemini4_vertex": "google/gemini-4-hydrogen", "gemini_vertex": "google/gemini-3.8-flash",
                 "gemma_vertex": "google/gemma-4-26b-a4b-it-maas"}


def default_endpoint() -> str:
    """The OpenAI-compatible server offered first: the site config's ``endpoints.teacher.url`` ("" for none)."""
    from matchminer_ai.site import load_site_config

    return load_site_config().get("endpoints.teacher.url") or ""


def vertex_project() -> str:
    """The Google Cloud project of the Vertex profiles: the site config's ``vertex.project``."""
    from matchminer_ai.site import load_site_config

    return load_site_config().get("vertex.project") or ""


def vertex_endpoint() -> str:
    return f"https://aiplatform.googleapis.com/v1/projects/{vertex_project()}/locations/global/endpoints/openapi"


class _VertexProfiles(dict):
    """The Vertex profiles, with the endpoint of the configured project (read when used)."""

    def __getitem__(self, backend):
        return {"endpoint": vertex_endpoint(), "model": VERTEX_MODELS[backend], "tokenizer": VERTEX_TOKENIZER}

    def items(self):
        return [(key, self[key]) for key in VERTEX_MODELS]

    def get(self, backend, default=None):
        return self[backend] if backend in VERTEX_MODELS else default


VERTEX_PROFILES = _VertexProfiles.fromkeys(VERTEX_MODELS)
VERTEX_LABELS = {"gemini4_vertex": "Gemini 4 Hydrogen (Vertex)", "gemini_vertex": "Gemini 3.8 Flash (Vertex)",
                 "gemma_vertex": "Gemma 4 26B MaaS (Vertex)"}
GEMINI_BACKENDS = frozenset({"gemini4_vertex", "gemini_vertex"})  # manage their own sampling and thinking
BACKENDS = (OPENAI, *VERTEX_PROFILES)
DEFAULT_BACKEND = "gemini4_vertex"  # the dashboard's default LLM (user, 2026-11-17); the GB300 stays selectable
CREDENTIALS_TIMEOUT = 15.0  # seconds to get a Vertex token before calling the credentials unavailable
_credentials: dict = {"checked": None, "error": None}
_credentials_lock = threading.Lock()


def vertex_credentials_error(max_age: float = 60.0) -> str | None:
    """None when this server's Application Default Credentials give a Google token, else why not. Checked at most once
    a minute and never for longer than ``CREDENTIALS_TIMEOUT``, so missing credentials are reported, not waited on."""
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
    try:
        pool.submit(token).result(timeout=CREDENTIALS_TIMEOUT)
        error = None
    except FutureTimeout:
        error = (f"Vertex credentials not available: no Google token within {CREDENTIALS_TIMEOUT:.0f} s from this "
                 f"server's Application Default Credentials (project {vertex_project()}).")
    except Exception as exc:
        error = (f"Vertex credentials not available on this server ({type(exc).__name__}): set up Google Application "
                 f"Default Credentials with access to project {vertex_project()}, or choose an OpenAI-compatible "
                 f"endpoint.")
    finally:
        pool.shutdown(wait=False)
    with _credentials_lock:
        _credentials.update(checked=time.monotonic(), error=error)
    return error
NOTICE = "The patient summary is sent to the selected LLM endpoint; use an endpoint authorized for the data."
# The per-criterion check also sends the cohort's criteria (protocol text, for a protocol cohort).
# Sampling keys a section may fix; removed so the served model's profile supplies them.
SAMPLING_KEYS = ("temperature", "top_p", "top_k", "min_p", "repetition_penalty", "presence_penalty",
                 "frequency_penalty")
MAX_IN_FLIGHT = 16  # requests the dashboard keeps in flight on one endpoint (a shared, busy server)


def normalize_endpoint(value: str) -> str:
    """Accept a server address or an OpenAI-compatible API base URL (the fetcher's rule)."""
    value = (value or "").strip()
    if not value or any(ch.isspace() or ord(ch) < 32 for ch in value):
        raise ValueError("Enter an HTTP or HTTPS LLM server address.")
    if "://" not in value:
        value = "http://" + value
    try:
        parsed = urlsplit(value)
        valid = (parsed.scheme in {"http", "https"} and parsed.hostname
                 and not (parsed.username or parsed.password or parsed.query or parsed.fragment))
        parsed.port  # malformed or out-of-range ports raise here, before any request
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("Use an HTTP(S) server URL without credentials, query, or fragment.")
    path = parsed.path.rstrip("/") or "/v1"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


@dataclass(frozen=True)
class Connection:
    backend: str = OPENAI
    endpoint: str = field(default_factory=default_endpoint)  # "" for none (the LLM checks are then disabled)
    model: str = NEMOTRON
    tokenizer: str = ""  # default: the model

    @classmethod
    def make(cls, backend: str = OPENAI, endpoint: str | None = None, model: str | None = None,
             tokenizer: str | None = None) -> "Connection":
        """A validated connection; a Vertex profile fixes its own endpoint, model and tokenizer."""
        backend = (backend or OPENAI).strip()
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {', '.join(BACKENDS)}")
        if backend in VERTEX_PROFILES:
            profile = VERTEX_PROFILES[backend]
            return cls(backend, profile["endpoint"], profile["model"], profile["tokenizer"])
        endpoint = (endpoint or "").strip()
        if endpoint.lower() in ("", "none", "off"):
            return cls(OPENAI, "", (model or "").strip(), (tokenizer or "").strip())
        model = (model or "").strip()
        if not model or len(model) > 512:
            raise ValueError("Choose the model the endpoint serves (check the endpoint for its models).")
        return cls(OPENAI, normalize_endpoint(endpoint), model, (tokenizer or "").strip())

    @property
    def configured(self) -> bool:
        return bool(self.endpoint and self.model)

    @property
    def vertex(self) -> bool:
        return self.backend in VERTEX_PROFILES

    @property
    def gemini(self) -> bool:
        return self.backend in GEMINI_BACKENDS

    @property
    def label(self) -> str:
        return VERTEX_LABELS[self.backend] if self.vertex else f"{self.model} at {self.endpoint}"

    def identity(self) -> dict:
        """What the cached answers are keyed on."""
        return {"backend": self.backend, "endpoint": self.endpoint, "model": self.model}

    def describe(self) -> dict:
        return {**asdict(self), "configured": self.configured, "label": self.label, "vertex": self.vertex,
                "project": vertex_project() if self.vertex else None}

    # ------------------------------------------------------------------ the inference package's configuration

    def configure(self, config, sections: tuple[str, ...]):
        """A copy of ``config`` sending ``sections`` (``llm_good_option``, ``llm_criteria_check``) to this
        connection, with the served model's sampling profile and debug mode (answers, reasoning, parse details)."""
        if not self.configured:
            raise ValueError("No LLM endpoint is configured (Data and Models).")
        c = deepcopy(config)
        c.debug_mode = True
        c.remote.update(enabled=True, server_urls=[self.endpoint], max_concurrent_requests=MAX_IN_FLIGHT,
                        request_timeout=3600, provider="openai")
        c.remote.setdefault("adaptive_concurrency", {}).update(enabled=False)
        if self.vertex:
            c.remote.update(provider="google_agent_platform", google_project_id=vertex_project())
            if not self.gemini:  # the Gemma MaaS endpoint's pacing (the fetcher's)
                c.remote.update(request_start_interval_seconds=2.0, capacity_retry_initial_seconds=15.0,
                                capacity_retry_max_seconds=60.0)
        for name in sections:
            section = getattr(c, name) if hasattr(c, name) else c.raw.get(name)
            if section is None:
                raise ValueError(f"the inference package's preset has no {name} section")
            remote = deepcopy(section.get("remote") or {})
            local = deepcopy(section.get("local") or {})
            params = {k: v for k, v in (remote.get("request_params") or {}).items() if k not in SAMPLING_KEYS}
            extra = {k: v for k, v in (remote.get("extra_body") or {}).items() if k not in SAMPLING_KEYS}
            local["generation"] = {k: v for k, v in (local.get("generation") or {}).items() if k not in SAMPLING_KEYS}
            remote.update(model_name=self.model, tokenizer_name=self.tokenizer or self.model)
            if self.vertex:
                params["max_tokens"] = min(65536, int(params.get("max_tokens", 32768)))
                if self.gemini:
                    params["reasoning_effort"] = "high"
                    extra = {}
                    local["chat_template_kwargs"] = {}
                else:
                    extra = {"chat_template_kwargs": {"enable_thinking": True}}
                local["engine"] = {**(local.get("engine") or {}), "max_model_len": 1048576 if self.gemini else 262144}
                section["sampling_profile"] = "none" if self.gemini else "auto"
                section["structured_outputs"] = False  # hosted models answer free JSON that the package validates
            else:
                section["sampling_profile"] = "auto"
            remote.update(request_params=params, extra_body=extra)
            section.update(remote=remote, local=local)
            if name == "llm_criteria_check":
                section.update(chunk_size=MAX_IN_FLIGHT, chunks_in_flight=1)
        return c


async def discover_models(connection: Connection, endpoint: str | None = None, timeout: float = 10.0) -> dict:
    """The models an OpenAI-compatible endpoint serves (``/v1/models``), with their context lengths; a Vertex profile
    is checked with this server's Application Default Credentials."""
    import httpx

    if connection.vertex:
        from matchminer_ai.llm.vllm_server import check_openai_endpoint
        import asyncio

        error = await asyncio.to_thread(vertex_credentials_error, 0.0)
        if error:
            return {"available": False, "endpoint": connection.endpoint, "models": [], "error": error}
        try:
            checks = await asyncio.to_thread(check_openai_endpoint, connection.endpoint,
                                             provider="google_agent_platform", google_project_id=vertex_project(),
                                             model_name=connection.model, timeout=30)
            available = bool(checks and checks[0][1] == 200)
        except Exception:
            available = False
        return {"available": available, "endpoint": connection.endpoint,
                "models": [{"id": connection.model, "max_model_len": None}] if available else [],
                "error": None if available else (f"Could not reach {connection.label}. Check this server's Google "
                                                  f"Application Default Credentials and model access in project "
                                                  f"{vertex_project()}.")}
    target = normalize_endpoint(endpoint or connection.endpoint)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(target + "/models")
            response.raise_for_status()
            data = response.json()["data"]
        if not isinstance(data, list):
            raise ValueError("expected a model list")
        models = [{"id": m["id"].strip(), "max_model_len": m.get("max_model_len")} for m in data
                  if isinstance(m, dict) and isinstance(m.get("id"), str) and m["id"].strip() and len(m["id"]) <= 512]
        models = list({m["id"]: m for m in models}.values())
        return {"available": bool(models), "endpoint": target, "models": models,
                "error": None if models else "The endpoint returned no served models."}
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        return {"available": False, "endpoint": target, "models": [],
                "error": f"Could not read the endpoint's model list ({type(exc).__name__}). Check the address and "
                         "server."}
