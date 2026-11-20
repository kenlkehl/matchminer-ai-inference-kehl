"""The LLM endpoint the dashboard's LLM checks use (GoodOption check, per-criterion LLM check).

The connection profiles are the package's (``matchminer_ai.llm.connections``, shared with the guideline fetcher):
``normalize_endpoint``, the Vertex profiles and the parts of a section they fix, model discovery from
``/v1/models`` and the Vertex credentials check. Two kinds of connection:

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

from copy import deepcopy
from dataclasses import asdict, dataclass, field

from matchminer_ai.llm import connections
from matchminer_ai.llm.connections import normalize_endpoint, vertex_credentials_error, vertex_project

NEMOTRON = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4"
OPENAI = "openai"  # an OpenAI-compatible server (vLLM)
VERTEX_MODELS = connections.VERTEX_MODELS
VERTEX_LABELS = {key: f"{label} (Vertex)" for key, label in connections.VERTEX_LABELS.items()}
GEMINI_BACKENDS = connections.GEMINI_BACKENDS  # manage their own sampling and thinking
BACKENDS = (OPENAI, *VERTEX_MODELS)
DEFAULT_BACKEND = "gemini4_vertex"  # the dashboard's default LLM (user, 2026-11-17); the on-prem server stays selectable


def default_endpoint() -> str:
    """The OpenAI-compatible server offered first: the site config's ``endpoints.teacher.url`` ("" for none)."""
    from matchminer_ai.site import load_site_config

    return load_site_config().get("endpoints.teacher.url") or ""


NOTICE = "The patient summary is sent to the selected LLM endpoint; use an endpoint authorized for the data."
# The per-criterion check also sends the cohort's criteria (protocol text, for a protocol cohort).
# Sampling keys a section may fix; removed so the served model's profile supplies them.
SAMPLING_KEYS = ("temperature", "top_p", "top_k", "min_p", "repetition_penalty", "presence_penalty",
                 "frequency_penalty")
MAX_IN_FLIGHT = 16  # requests the dashboard keeps in flight on one endpoint (a shared, busy server)


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
        if backend in VERTEX_MODELS:
            profile = connections.vertex_profile(backend)
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
        return self.backend in VERTEX_MODELS

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
        if self.vertex:  # provider and project and, for Gemma MaaS, its pacing (the fetcher's)
            c.remote.update({k: v for k, v in connections.vertex_remote(self.backend).items() if k != "server_urls"})
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
                params.update(connections.vertex_request_params(self.backend, params.get("max_tokens", 32768)))
                extra = connections.vertex_extra_body(self.backend, True)
                if self.gemini:
                    local["chat_template_kwargs"] = {}
                local["engine"] = {**(local.get("engine") or {}),
                                   "max_model_len": connections.vertex_context_tokens(self.backend)}
                section["sampling_profile"] = connections.vertex_sampling_profile(self.backend)
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
    import asyncio

    import httpx

    if connection.vertex:
        error = await asyncio.to_thread(vertex_credentials_error, 0.0)
        if error:
            return {"available": False, "endpoint": connection.endpoint, "models": [], "error": error}
        available = await connections.vertex_available(connection.backend, timeout=30)
        return {"available": available, "endpoint": connection.endpoint,
                "models": [{"id": connection.model, "max_model_len": None}] if available else [],
                "error": None if available else (f"Could not reach {connection.label}. Check this server's Google "
                                                  f"Application Default Credentials and model access in project "
                                                  f"{vertex_project()}.")}
    target = normalize_endpoint(endpoint or connection.endpoint)
    try:
        models = await connections.served_models(target, timeout=timeout)
        return {"available": bool(models), "endpoint": target, "models": models,
                "error": None if models else "The endpoint returned no served models."}
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        return {"available": False, "endpoint": target, "models": [],
                "error": f"Could not read the endpoint's model list ({type(exc).__name__}). Check the address and "
                         "server."}
