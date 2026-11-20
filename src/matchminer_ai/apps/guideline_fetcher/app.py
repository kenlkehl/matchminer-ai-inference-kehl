"""Local web application. Clinical behavior stays in the inference package.

Settings(phi=True) is the PHI deployment: every page and API route except /login and
static assets requires the access token, and PHI population routes are added.
"""

import json
import os
import re
import secrets
import asyncio
import time
from contextlib import asynccontextmanager, suppress
from dataclasses import replace
from typing import Literal

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from matchminer_ai.apps.common import security
from matchminer_ai.llm import connections

from .extraction_workup import SessionExtractions
from .jobs import BusyError, Jobs
from .note_compressions import NoteCompressions
from .population import PopulationSessions
from .population_routes import register_population_routes
from .services import (
    Catalogs, Inference, InputError, parse_upload, synthetic_note_sources, synthetic_notes,
)
from .settings import (
    CLAUDE_BACKENDS, CLAUDE_ENDPOINT_NAME, CLAUDE_LABELS, DATABRICKS_PROFILE, GEMINI_BACKENDS,
    GEMINI_PROFILE, GEMINI_PROJECT, LLM_TASKS, ROOT, SQL_ONLY_BACKENDS, SQL_TASK, VERTEX_LABELS,
    VERTEX_PROFILES, Settings, load_access_token, normalize_endpoint,
)

# PHI mode's cookies are named apart from the non-PHI app's.
SESSION_COOKIES = {"standard": "gf_session", "phi": "gf_phi_session"}
AUTH_COOKIE = "gf_phi_auth"
SECURITY_HEADERS = {"Cache-Control": "no-store", **security.security_headers(security.content_security_policy())}
# index.html puts mode-specific markup between <!-- mode:phi --> and <!-- /mode:phi -->
# (or mode:standard) lines; each marker line is removed whole.
MODE_BLOCK = re.compile(
    r"^[ \t]*<!-- mode:(phi|standard) -->\n(.*?)^[ \t]*<!-- /mode:\1 -->\n", re.S | re.M
)
LOGIN_PAGE = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width,initial-scale=1" />
    <title>Sign in · MatchMiner-AI guideline dashboard (PHI)</title>
    <link rel="icon" href="/static/mark.svg" type="image/svg+xml" />
    <link rel="stylesheet" href="/static/style.css" />
  </head>
  <body class="login-page">
    <main class="panel login-panel">
      <img src="/static/mark.svg" alt="" width="42" height="42" />
      <h1>MatchMiner-AI guideline dashboard</h1>
      <p class="helper">This workspace contains real patient records. Enter the access token
        printed by <code>mmai-guideline-fetcher --phi</code> or stored in its access token file.</p>
      {error}
      <form method="post" action="/login">
        <label for="token">Access token</label>
        <input id="token" name="token" type="password" autocomplete="current-password"
          required autofocus />
        <button class="button primary full" type="submit">Sign in</button>
      </form>
    </main>
  </body>
</html>
"""


def render_index(mode):
    """The dashboard page for one mode: the other mode's blocks removed, the mode named.

    The browser scripts read <html data-mode> to enable PHI-only controls.
    """
    page = (ROOT / "static/index.html").read_text()
    page = MODE_BLOCK.sub(lambda block: block.group(2) if block.group(1) == mode else "", page)
    return page.replace('<html lang="en">', f'<html lang="en" data-mode="{mode}">', 1)


LLMBackend = Literal["local", "gemini4_vertex", "gemini_vertex", "gemma_vertex", "claude_databricks"]
LLMTask = Literal["summarize", "compress", "workup", "workup_sql", "workup_full_backup",
                  "workup_agent_backup"]


def claude_endpoint_name(value):
    if not CLAUDE_ENDPOINT_NAME.fullmatch(value or ""):
        raise ValueError("Enter the full three-part Databricks endpoint name, such as "
                         "ai_gateway_model_services_prod.<team>_<project>.<team>_<project>-claude-sonnet-5-5.")
    return value


class LLMSettings(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    llm_backend: LLMBackend = "local"
    endpoint: str = Field(min_length=1, max_length=2048)
    model: str = Field(min_length=1, max_length=512)
    tokenizer: str = Field(default="", max_length=512)
    task: LLMTask | None = None
    apply_to_all: bool = False
    reasoning: bool = False
    backup_enabled: bool = False
    failure_threshold: int = Field(default=3, ge=1, le=10)

    @model_validator(mode="after")
    def backup_task(self):
        if self.apply_to_all and self.task is None:
            raise ValueError("Choose a source task when applying a model to all tasks.")
        if self.backup_enabled and self.task not in {"workup_full_backup", "workup_agent_backup"}:
            raise ValueError("Enable backup only for a concordance backup task.")
        if self.llm_backend in SQL_ONLY_BACKENDS:
            if self.task != SQL_TASK or self.apply_to_all:
                raise ValueError("Claude through Databricks serves only the SQL agent task.")
            claude_endpoint_name(self.model)
        return self

    @field_validator("endpoint")
    @classmethod
    def endpoint_url(cls, value):
        return normalize_endpoint(value)


class EndpointCheck(BaseModel):
    task: LLMTask | None = None
    llm_backend: LLMBackend | None = None
    endpoint: str | None = Field(default=None, max_length=2048)
    model: str | None = Field(default=None, max_length=512)

    @field_validator("endpoint")
    @classmethod
    def endpoint_url(cls, value):
        return normalize_endpoint(value) if value is not None else None


class Note(BaseModel):
    note_text: str = Field(min_length=1, max_length=2_000_000)
    note_date: str | None = None
    note_type: str | None = Field(default=None, max_length=1000)


class NotesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=2_000_000)
    notes: list[Note] | None = Field(default=None, max_length=5000)


class SummaryRequest(NotesRequest):
    note_source: Literal["original", "compressed"] = "original"


class CompressionRequest(NotesRequest):
    # Preserve the current patient workspace while the independent action runs.
    summary: str = Field(default="", max_length=100_000)
    retrieval_job_id: str | None = Field(default=None, min_length=1, max_length=100)
    note_source: Literal["original", "compressed"] = "original"


class SearchRequest(BaseModel):
    summary: str = Field(min_length=1, max_length=100_000)
    catalogs: list[str] = Field(default_factory=list, max_length=100)
    top_n: int = Field(default=1, ge=1, le=20)
    candidate_k: int = Field(default=5, ge=1, le=100)

    @model_validator(mode="after")
    def check(self):
        if not self.summary.strip() or self.candidate_k < self.top_n:
            raise ValueError("Supply a summary and candidate_k >= top_n")
        return self


class WorkupRequest(SummaryRequest):
    retrieval_job_id: str = Field(min_length=1, max_length=100)
    space_index: int = Field(ge=0, le=19)
    # extractions: compiled extraction queries; extraction_sql: the SQL agent's queries.
    method: Literal["full_notes", "agentic", "extractions", "extraction_sql"] = "full_notes"
    # Extraction methods only: extract each note, or also build a patient-level record.
    extraction_levels: Literal["document", "document_and_patient"] = "document"


EXTRACTION_METHODS = {"extractions", "extraction_sql"}


class ExtractionCostRequest(NotesRequest):
    retrieval_job_id: str = Field(min_length=1, max_length=100)
    space_index: int = Field(ge=0, le=19)
    extraction_levels: Literal["document", "document_and_patient"] = "document"
    method: Literal["extractions", "extraction_sql"] = "extractions"


def create_app(settings=None, inference=None, access_token=None):
    settings = settings or Settings()
    session_cookie = SESSION_COOKIES[settings.mode]
    if settings.phi:
        access_token = access_token or load_access_token()
        sign_in = security.SignIn(access_token, b"guideline-fetcher-phi browser sign-in v1", AUTH_COOKIE)
        valid_token, authorized = sign_in.valid_token, sign_in.authorized

    catalogs = Catalogs(settings.catalog_root)
    jobs = Jobs(settings.ttl)
    note_compressions = NoteCompressions(settings.ttl)
    session_extractions = SessionExtractions(settings.ttl)
    population_sessions = PopulationSessions(settings.ttl)
    workspace_settings = {}

    def clean_settings():
        expired = [owner for owner, entry in workspace_settings.items()
                   if time.monotonic() - entry["updated"] > settings.ttl]
        for owner in expired:
            del workspace_settings[owner]

    def current_settings(owner):
        with jobs.lock:
            clean_settings()
            entry = workspace_settings.get(owner)
            if entry is None:
                return settings
            entry["updated"] = time.monotonic()
            return replace(settings, **entry["values"])

    def patient_inference(owner, task):
        workspace = current_settings(owner)
        selected = workspace.for_task(task)
        # Capture before submitting the job, so a later settings change cannot
        # redirect a running or queued patient's notes to another endpoint.
        worker = inference if selected is settings else inference.with_settings(selected)
        if isinstance(worker, Inference):
            worker = worker.with_compression_cache(
                note_compressions.bind(owner), settings=workspace.for_task("compress")
            )
            if task == "workup":
                worker = worker.with_workup_backups(
                    full=workspace.for_task("workup_full_backup"),
                    agent=workspace.for_task("workup_agent_backup"),
                ).with_extractions(session_extractions, owner)
        return worker

    @asynccontextmanager
    async def lifespan(api):
        async def expire():
            while True:
                await asyncio.sleep(30)
                with jobs.lock:
                    jobs._clean()
                    clean_settings()
                    note_compressions.clean()
                    session_extractions.clean()
                    population_sessions.clean({j["owner"] for j in jobs.items.values()
                                               if j["status"] in {"queued", "running", "cancelling"}})

        cleaner = asyncio.create_task(expire())
        yield
        cleaner.cancel()
        with suppress(asyncio.CancelledError):
            await cleaner
        jobs.shutdown()

    api = FastAPI(title="Guideline Fetcher", docs_url=None, redoc_url=None, lifespan=lifespan)
    api.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)
    inference = inference or Inference(settings)
    api.state.jobs = jobs
    api.state.note_compressions = note_compressions
    api.state.session_extractions = session_extractions
    api.state.population_sessions = population_sessions
    register_population_routes(api, settings=settings, catalogs=catalogs, jobs=jobs,
                               sessions=population_sessions, current_settings=current_settings)

    def secured(response):
        response.headers.update(SECURITY_HEADERS)
        return response

    @api.middleware("http")
    async def session(request, call_next):
        # Reject cross-origin writes, including requests from unrelated local pages.
        refused = security.cross_origin_write(request.method, request.headers)
        if refused:
            return secured(JSONResponse({"detail": refused}, status_code=403))
        if settings.phi:
            # Only the sign-in page and patient-free static assets are public.
            public = request.url.path == "/login" or request.url.path.startswith("/static/")
            if not public and not authorized(request):
                if request.url.path.startswith("/api/"):
                    return secured(JSONResponse({"detail": "Sign in with the access token."}, status_code=401))
                return secured(RedirectResponse("/login", status_code=303))
        token = request.cookies.get(session_cookie)
        if not token or len(token) != 43:
            token = secrets.token_urlsafe(32)
        request.state.owner = token
        response = await call_next(request)
        response.set_cookie(
            session_cookie,
            token,
            httponly=True,
            samesite="strict",
            secure=request.url.scheme == "https",
        )
        return secured(response)

    if settings.phi:
        def signed_in(request):
            return sign_in.set_cookie(request, RedirectResponse("/", status_code=303))

        def login_page(failed=False):
            error = ('<p class="error-banner" role="alert">That access token was not accepted.</p>'
                     if failed else "")
            return HTMLResponse(LOGIN_PAGE.replace("{error}", error), status_code=401 if failed else 200)

        @api.get("/login")
        def login_form(request: Request, token: str | None = None):
            # mmai-guideline-fetcher --phi prints /login?token=… to its own terminal for one-click sign-in.
            if token is not None:
                return signed_in(request) if valid_token(token) else login_page(failed=True)
            return login_page()

        @api.post("/login")
        def login(request: Request, token: str = Form(..., max_length=1000)):
            return signed_in(request) if valid_token(token) else login_page(failed=True)

        @api.post("/logout")
        def logout():
            response = RedirectResponse("/login", status_code=303)
            response.delete_cookie(AUTH_COOKIE, httponly=True, samesite="strict")
            return response

    @api.exception_handler(InputError)
    async def bad_input(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @api.exception_handler(BusyError)
    async def busy(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=409)

    def connection_view(selected):
        from matchminer_ai.llm.sampling import vendor_defaults

        connection = VERTEX_PROFILES.get(selected.llm_backend) or {
            "endpoint": selected.endpoint, "model": selected.model,
            "tokenizer": selected.tokenizer,
        }
        managed = selected.llm_backend in GEMINI_BACKENDS or selected.llm_backend in CLAUDE_BACKENDS
        sampling, _ = vendor_defaults(
            connection["model"], template={"enable_thinking": selected.reasoning}
        ) if not managed else ({}, {})
        return {
            "llm_backend": selected.llm_backend, **connection,
            "reasoning": selected.reasoning,
            "thinking_supported": selected.thinking_supported,
            "sampling": sampling,
            "backup_enabled": selected.backup_enabled,
            "failure_threshold": selected.failure_threshold,
        }

    def config_view(selected):
        return dict(
            **connection_view(selected),
            tasks={task: {"label": label, **connection_view(selected.for_task(task))}
                   for task, label in LLM_TASKS.items()},
            note_compression_available=True,
            gemini_profile={**GEMINI_PROFILE, "project": GEMINI_PROJECT},
            vertex_profiles={key: {**profile, "label": VERTEX_LABELS[key], "project": GEMINI_PROJECT}
                             for key, profile in VERTEX_PROFILES.items()},
            claude_profiles={key: {**profile, "label": CLAUDE_LABELS[key],
                                   "databricks_profile": DATABRICKS_PROFILE, "task": SQL_TASK}
                             for key, profile in CLAUDE_BACKENDS.items()},
            local_defaults={"endpoint": settings.endpoint, "model": settings.model,
                            "tokenizer": settings.tokenizer},
            device=settings.device,
            ttl_minutes=settings.ttl // 60,
            max_upload_mb=settings.max_upload // 1024**2,
            example_available=settings.example.is_file(),
            synthetic_note_sources=synthetic_note_sources(settings),
        )

    @api.get("/api/config")
    def config(request: Request):
        return config_view(current_settings(request.state.owner))

    @api.post("/api/config")
    def update_config(body: LLMSettings, request: Request):
        values = body.model_dump(exclude={"task", "apply_to_all"})
        values["tokenizer"] = body.tokenizer or body.model
        if body.llm_backend in VERTEX_PROFILES:
            # Construct the Google destination on the server; never send ADC to
            # a browser-supplied endpoint or accept browser-supplied credentials.
            values.update(VERTEX_PROFILES[body.llm_backend])
        elif body.llm_backend in CLAUDE_BACKENDS:
            # The same for Databricks tokens: the server fixes the gateway URL; the
            # browser picks only the gateway's endpoint name.
            profile = CLAUDE_BACKENDS[body.llm_backend]
            values.update(endpoint=profile["endpoint"], tokenizer=profile["tokenizer"])
        with jobs.lock:
            clean_settings()
            previous = current_settings(request.state.owner)
            if body.apply_to_all:
                connection = {k: v for k, v in values.items()
                              if k not in {"backup_enabled", "failure_threshold"}}
                tasks = {}
                for task in LLM_TASKS:
                    prior = previous.for_task(task)
                    tasks[task] = {
                        **connection,
                        "backup_enabled": prior.backup_enabled if task.endswith("_backup") else False,
                        "failure_threshold": prior.failure_threshold,
                    }
                # Save the source task's own backup controls without copying
                # those controls onto the other backup or primary tasks.
                tasks[body.task] = values
                values = {
                    **(workspace_settings.get(request.state.owner) or {}).get("values", {}),
                    **connection,
                    "task_connections": tasks,
                }
            elif body.task is not None:
                values = {
                    **(workspace_settings.get(request.state.owner) or {}).get("values", {}),
                    "task_connections": {**previous.task_connections, body.task: values},
                }
            else:
                # Legacy/global updates apply to every task.
                values["task_connections"] = {}
            updated = replace(settings, **values)
            workspace_settings[request.state.owner] = {
                "values": values, "updated": time.monotonic(),
            }
            if connection_view(previous.for_task("compress")) != connection_view(updated.for_task("compress")):
                note_compressions.clear(request.state.owner)
        return config_view(updated)

    @api.get("/api/catalogs")
    def library(refresh: bool = False):
        rows = catalogs.list(refresh)
        return {
            "catalogs": [{k: v for k, v in row.items() if k != "path"} for row in rows],
            "spaces": sum(row["spaces"] for row in rows),
        }

    @api.post("/api/endpoint-check")
    async def endpoint_check(body: EndpointCheck, request: Request):
        selected = current_settings(request.state.owner)
        if body.task:
            selected = selected.for_task(body.task)
        backend = body.llm_backend or selected.llm_backend
        if backend in CLAUDE_BACKENDS:
            from matchminer_ai.llm.anthropic_messages import check_gateway

            profile = CLAUDE_BACKENDS[backend]
            model = body.model if body.model is not None else selected.model
            try:
                claude_endpoint_name(model)
            except ValueError as exc:
                return {"available": False, "endpoint": profile["endpoint"], "models": [],
                        "model": None, "error": str(exc)}
            # A short request with no patient data; errors are fixed texts.
            result = await asyncio.to_thread(check_gateway, profile["endpoint"], model,
                                             profile=DATABRICKS_PROFILE, timeout=60)
            return {"available": result["available"], "endpoint": profile["endpoint"],
                    "models": [model] if result["available"] else [],
                    "model": model if result["available"] else None, "error": result["error"]}
        if backend in VERTEX_PROFILES:
            profile = VERTEX_PROFILES[backend]
            available = await connections.vertex_available(backend, timeout=30)
            return {"available": available, "endpoint": profile["endpoint"],
                    "models": [profile["model"]] if available else [],
                    "model": profile["model"] if available else None,
                    "error": None if available else
                    f"Could not connect to {VERTEX_LABELS[backend]}. Check the dashboard server's Google Application "
                    f"Default Credentials and model access in project {GEMINI_PROJECT}."}
        endpoint = body.endpoint or selected.endpoint
        model = body.model if body.model is not None else selected.model
        try:
            names = [m["id"] for m in await connections.served_models(
                endpoint, timeout=10, api_key=os.environ.get("OPENAI_API_KEY"))]
            return {"available": model in names, "models": names, "endpoint": endpoint,
                    "model": model if model in names else next(iter(names), None),
                    "error": None if names else "The endpoint returned no served models."}
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            return {"available": False, "models": [], "endpoint": endpoint, "model": None,
                    "error": "Could not read the endpoint's model list. Check the address and server."}

    @api.get("/api/example")
    def example():
        # Operator-selected synthetic file, never bundled in the distributable app.
        if not settings.example.is_file():
            raise HTTPException(404, "No synthetic example is configured.")
        data = json.loads(settings.example.read_text())
        return {"summary": data[0]["cancer_history_summary"], "label": "Synthetic NSCLC example"}

    @api.get("/api/synthetic-notes")
    def example_notes(source: str = "demo"):
        try:
            return synthetic_notes(settings, source)
        except InputError:
            raise
        except Exception:
            raise InputError("Could not load the synthetic CSV. Check the sample path or HF connection.") from None

    @api.post("/api/upload")
    async def upload(file: UploadFile = File(...)):
        data = await file.read(settings.max_upload + 1)
        await file.close()
        if len(data) > settings.max_upload:
            raise InputError("File exceeds the 20 MB upload limit.")
        if not data:
            raise InputError("The uploaded file is empty.")
        try:
            # A PDF may need OCR: off the event loop.
            result = await asyncio.to_thread(parse_upload, file.filename or "", data)
        except InputError:
            raise
        except Exception:
            raise InputError(
                "This file could not be read. Check its format and try again."
            ) from None
        if len(result["text"]) > 2_000_000:
            raise InputError(
                "Extracted notes exceed the 2 million character limit. Split the record."
            )
        return result

    def source_notes(body):
        if not body.text.strip():
            raise InputError("Enter patient notes first.")
        notes = [n.model_dump(exclude={"note_type"} if n.note_type is None else set()) for n in body.notes] if body.notes else None
        if notes and sum(len(n["note_text"]) for n in notes) > 2_000_000:
            raise InputError("Too much note text.")
        return notes

    @api.post("/api/compress", status_code=202)
    def compress(body: CompressionRequest, request: Request):
        notes = source_notes(body)
        restore = preprocessing_restore(body, request.state.owner, notes)
        worker = patient_inference(request.state.owner, "compress")

        def compression_job(progress):
            return {**restore, **worker.compress(body.text, notes, progress)}

        key = jobs.submit(request.state.owner, "compress", compression_job, restore=restore)
        return {"job_id": key}

    def preprocessing_restore(body, owner, notes):
        restore = {"notes_input": {"text": body.text, "notes": notes},
                   "summary": body.summary, "note_source": body.note_source}
        if body.retrieval_job_id:
            retrieval = jobs.get(owner, body.retrieval_job_id)
            if not retrieval or retrieval["kind"] != "retrieve" or retrieval["status"] != "complete":
                raise InputError("The previous guideline search is unavailable in this session.")
            restore.update(retrieval=retrieval["result"], retrieval_job_id=body.retrieval_job_id)
        previous = jobs.latest(owner)
        if previous and previous["status"] == "complete":
            result = previous["result"]
            if result.get("notes_input") == restore["notes_input"]:
                if previous["kind"] == "summarize" and result["summary"] == body.summary:
                    restore["summary_result"] = result
                elif previous["kind"] == "workup" and result["retrieval_job_id"] == body.retrieval_job_id:
                    restore["workup_result"] = result
        return restore

    @api.post("/api/summarize", status_code=202)
    def summarize(body: SummaryRequest, request: Request):
        notes = source_notes(body)
        worker = patient_inference(request.state.owner, "summarize")

        def summary_job(progress):
            options = {"note_source": "compressed"} if body.note_source == "compressed" else {}
            result = worker.summarize(body.text, notes, progress, **options)
            result.update(notes_input={"text": body.text, "notes": notes},
                          note_source=body.note_source)
            return result

        key = jobs.submit(
            request.state.owner,
            "summarize",
            summary_job,
            restore={"notes_input": {"text": body.text, "notes": notes},
                     "note_source": body.note_source},
        )
        return {"job_id": key}

    @api.post("/api/retrieve", status_code=202)
    def retrieve(body: SearchRequest, request: Request):
        paths = catalogs.resolve(body.catalogs)

        def retrieve_job(progress):
            result = inference.retrieve(body.summary, paths, body.top_n, body.candidate_k, progress)
            result["search"] = {
                "catalogs": body.catalogs,
                "top_n": body.top_n,
                "candidate_k": body.candidate_k,
            }
            return result

        key = jobs.submit(
            request.state.owner,
            "retrieve",
            retrieve_job,
        )
        return {"job_id": key}

    @api.get("/api/jobs/{key}")
    def job(key: str, request: Request):
        result = jobs.get(request.state.owner, key)
        if result is None:
            raise HTTPException(404, "Job expired or is unavailable in this browser session.")
        return result

    @api.post("/api/jobs/{key}/stop")
    def stop_job(key: str, request: Request):
        result = jobs.cancel(request.state.owner, key)
        if result is None:
            raise HTTPException(404, "Job expired or is unavailable in this browser session.")
        return result

    def workup_space(owner, retrieval_job_id, space_index):
        retrieval = jobs.get(owner, retrieval_job_id)
        if not retrieval or retrieval["kind"] != "retrieve" or retrieval["status"] != "complete":
            raise InputError("Find guideline spaces in this browser session first.")
        spaces = retrieval["result"]["spaces"]
        if space_index >= len(spaces):
            raise InputError("The selected guideline space is unavailable.")
        space = spaces[space_index]
        if not space["diagnostic_workup"]:
            raise InputError("This space has no cataloged diagnostic/workup items.")
        return retrieval, space

    @api.post("/api/workup/extraction-cost")
    def extraction_cost(body: ExtractionCostRequest, request: Request):
        """What an extraction-based review would send; makes no model calls."""
        notes = source_notes(body)
        _, space = workup_space(request.state.owner, body.retrieval_job_id, body.space_index)
        worker = patient_inference(request.state.owner, "workup")
        return worker.extraction_cost(body.text, notes, space, body.extraction_levels,
                                      method=body.method)

    @api.post("/api/workup", status_code=202)
    def workup(body: WorkupRequest, request: Request):
        notes = source_notes(body)
        retrieval, space = workup_space(request.state.owner, body.retrieval_job_id,
                                        body.space_index)
        if body.method in EXTRACTION_METHODS and body.note_source != "original":
            raise InputError(
                "Structured extraction quotes the original notes; select Original notes.")
        worker = patient_inference(request.state.owner, "workup")

        def review_job(progress):
            options = {"patient_summary": retrieval["result"].get("summary")} if body.method == "agentic" else {}
            if body.note_source == "compressed":
                options["note_source"] = "compressed"
            if body.method in EXTRACTION_METHODS:
                options["extraction_levels"] = body.extraction_levels
            review = worker.review_workup(body.text, notes, space, progress, method=body.method, **options)
            return {
                "review": review, "space_index": body.space_index,
                "method": body.method,
                "extraction_levels": body.extraction_levels,
                "note_source": body.note_source,
                "retrieval_job_id": body.retrieval_job_id,
                "retrieval": retrieval["result"],
                "notes_input": {"text": body.text, "notes": notes},
            }

        key = jobs.submit(request.state.owner, "workup", review_job, restore={
            "retrieval": retrieval["result"], "retrieval_job_id": body.retrieval_job_id,
            "notes_input": {"text": body.text, "notes": notes}, "method": body.method,
            "note_source": body.note_source,
            "space_index": body.space_index,
        })
        return {"job_id": key}

    @api.delete("/api/session", status_code=204)
    def clear(request: Request):
        jobs.clear(request.state.owner)
        note_compressions.clear(request.state.owner)
        session_extractions.clear(request.state.owner)
        population_sessions.clear(request.state.owner)
        return Response(status_code=204)

    @api.get("/api/session")
    def latest_job(request: Request):
        return {"job": jobs.latest(request.state.owner, exclude_prefix="population-"),
                "note_compression": note_compressions.latest(request.state.owner)}

    @api.get("/api/note-compression")
    def saved_compression(request: Request):
        return {"note_compression": note_compressions.latest(request.state.owner)}

    @api.delete("/api/note-compression", status_code=204)
    def discard_compression(request: Request):
        note_compressions.clear(request.state.owner)
        return Response(status_code=204)

    @api.get("/")
    def index():
        return HTMLResponse(render_index(settings.mode))

    api.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
    return api


def __getattr__(name):
    # `uvicorn matchminer_ai.apps.guideline_fetcher.app:app` serves the non-PHI app, built on first use; run.py builds its own,
    # so a PHI process never constructs a non-PHI app.
    if name == "app":
        globals()["app"] = create_app()
        return globals()["app"]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
