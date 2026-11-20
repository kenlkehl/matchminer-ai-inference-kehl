"""HTTP routes (FastAPI) over the dashboard state, and the built React frontend (``static/``)."""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from matchminer_ai.apps.common import security

from . import pipeline
from .core import cancer_types
from .core.encoder import EncoderUnavailable
from .evidence import CatalogUnavailable, UnknownEntry
from .llm import Connection, discover_models

from .state import DashState, NotFound

STATIC = Path(__file__).with_name("static")
DIRECTIONS = ("patients_for_cohort", "trials_for_patient")
CSP = security.content_security_policy(inline_styles=True, form_action="'none'")
HEADERS = security.security_headers(CSP)


class PatientsForCohort(BaseModel):
    cohort_id: str
    k: int | None = None
    cancer_type: str = ""
    sex: str = ""


class TrialsForPatient(BaseModel):
    patient_id: str | None = None
    summary: str | None = None  # an entered cancer summary (embedded with the round's encoder)
    boilerplate: str | None = None  # its common-exclusion evidence ("Other relevant history")
    k: int | None = None
    scope: str = "dashboard"  # "dashboard": the cohorts the dashboards show; "all": both versions of every trial
    per_trial: bool = True


class ScoreRequest(BaseModel):
    direction: str
    query_id: str
    target_ids: list[str] = Field(default_factory=list)


class PairRequest(BaseModel):
    patient_id: str
    cohort_id: str


class LLMCheckRequest(BaseModel):
    patient_id: str
    cohort_id: str | None = None  # the per-criterion check
    trial_id: str | None = None  # the GoodOption check
    rerun: bool = False


class ConnectionRequest(BaseModel):
    backend: str = "openai"
    endpoint: str | None = Field(default=None, max_length=2048)
    model: str | None = Field(default=None, max_length=512)
    tokenizer: str | None = Field(default=None, max_length=512)


# ---------------------------------------------------------------------------------------------- shaping


DRUG_BRIEF = ("drug_id", "name", "listed_name", "role", "mechanism_types", "all_classes", "class_ids", "payload",
              "payload_classes")


def trial_summary(trial: dict) -> dict:
    """A trial's fields on every cohort row; its drugs without their arms and mechanism sentence (those are in the
    trial's details)."""
    keys = ("id", "nct_id", "protocol_no", "short_title", "phase", "status", "is_open", "center", "center_id",
            "center_short_name", "site_pi", "overall_pi", "contact", "sites", "primary_version", "state")
    drugs = [{k: d.get(k) for k in DRUG_BRIEF} for d in trial.get("drugs") or []]  # a store built before drugs: none
    return {**{k: trial[k] for k in keys}, "drugs": drugs}


def lines(items: list[dict], kind: str, boilerplate: bool | None = False) -> list[str]:
    return [i["text"] for i in items if i.get("criterion_type") == kind
            and (boilerplate is None or bool(i.get("boilerplate")) == boilerplate)]


def cohort_row(state: DashState, cohort: dict) -> dict:
    """A dashboard row: the cohort with its trial's summary fields."""
    trial = state.store["trials"][cohort["trial_id"]]
    items = cohort["items"]
    version = trial["versions"][cohort["version"]]
    return {"cohort_id": cohort["id"], "trial": trial_summary(trial), "name": cohort["name"], "number": cohort["number"],
            "version": cohort["version"], "merged_from": cohort["merged_from"], "group": cohort.get("group"),
            "group_reason": cohort.get("group_reason"), "cancer_types": cohort["cancer_types"],
            "key_inclusion": lines(items, "inclusion"), "key_exclusion": lines(items, "exclusion"),
            "common_count": sum(1 for i in items if i.get("boilerplate")), "line_count": len(items),
            "completed_at": version["completed_at"], "stale": version["state"] == "stale"}


def cohort_detail(state: DashState, cohort: dict) -> dict:
    return {**cohort_row(state, cohort), "key": cohort["key"], "boilerplate": cohort["boilerplate"],
            "items": cohort["items"]}


def type_labels(keys: list[str]) -> list[str]:
    return [cancer_types.TYPE_LABELS[k] for k in keys if k in cancer_types.TYPE_LABELS]


def create_app(state: DashState) -> FastAPI:
    app = FastAPI(title="MatchMiner-AI dashboard (research use)", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.mmai = state

    @app.middleware("http")
    async def headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.update(HEADERS)
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        elif request.url.path.startswith("/assets/"):
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"  # hashed file names
        return response

    @app.exception_handler(NotFound)
    async def not_found(_, exc):
        return JSONResponse({"error": f"unknown id {exc.args[0]!r}"}, status_code=404)

    @app.exception_handler(UnknownEntry)
    async def unknown_entry(_, exc):
        return JSONResponse({"error": f"the GoodOption catalog has no {exc.args[0]!r}"}, status_code=404)

    @app.exception_handler(CatalogUnavailable)
    async def no_catalog(_, exc):
        return JSONResponse({"error": str(exc)}, status_code=404)

    @app.exception_handler(EncoderUnavailable)
    async def encoder_off(_, exc):
        return JSONResponse({"error": f"the TrialSpace encoder is unavailable: {exc}"}, status_code=409)

    @app.exception_handler(ValueError)
    async def bad_value(_, exc):
        return JSONResponse({"error": str(exc)}, status_code=400)

    # ------------------------------------------------------------------ dashboard

    @app.get("/api/meta")
    def meta():
        s = state.store
        return {"built_at": s["built_at"], "counts": s["counts"], "higher_threshold_groups": s["higher_threshold_groups"],
                "centers": [{"id": c["id"], "name": c["name"], "short_name": c["short_name"],
                             "trials": len(c["trial_ids"]), "cohorts": sum(len(g["cohort_ids"]) for g in c["groups"]),
                             "pending": len(c["pending_trial_ids"])} for c in s["centers"]],
                "cancer_types": cancer_types.describe(), "auto_score_n": state.settings.auto_score_n,
                "default_k": state.settings.default_k, "max_k": state.settings.max_k,
                "round": state.round.number if state.round else None, "site": pipeline.branding(),
                "vectors": state.cohort_vectors.describe(), "patients": len(state.corpus.patients),
                # The GoodOption catalog's state (None: a store built before the drugs), so a page can say why a
                # trial shows no drugs.
                "good_options": ({k: s["good_options"].get(k) for k in ("state", "message")}
                                 if s.get("good_options") else None)}

    @app.get("/api/centers/{center_id}")
    def center(center_id: str):
        found = next((c for c in state.store["centers"] if c["id"] == center_id), None)
        if found is None:
            raise NotFound(center_id)
        cohorts = state.store["cohorts"]
        groups = [{"id": g["id"], "name": g["name"], "trial_count": len(g["trial_ids"]),
                   "rows": [cohort_row(state, cohorts[cid]) for cid in g["cohort_ids"]]} for g in found["groups"]]
        pending = [trial_summary(state.store["trials"][t]) | {
            "versions": {v: {k: x[k] for k in ("state", "excluded_reason")} for v, x in state.store["trials"][t]["versions"].items()}}
            for t in found["pending_trial_ids"]]
        return {"id": found["id"], "name": found["name"], "short_name": found["short_name"], "built_at": state.store["built_at"],
                "trial_count": len(found["trial_ids"]), "groups": groups, "pending": pending,
                "cancer_type_labels": cancer_types.TYPE_LABELS}

    @app.get("/api/trials/{trial_id}")
    def trial(trial_id: str):
        t = state.trial(trial_id)
        versions = {}
        for version, info in t["versions"].items():
            versions[version] = {**{k: v for k, v in info.items() if k != "cohort_ids"},
                                 "cohorts": [cohort_detail(state, state.store["cohorts"][c]) for c in info["cohort_ids"]]}
        return {**{k: v for k, v in t.items() if k not in ("versions", "cohort_ids")}, "drugs": t.get("drugs") or [],
                "versions": versions}

    @app.get("/api/cohorts/{cohort_id}")
    def cohort(cohort_id: str):
        return cohort_detail(state, state.cohort(cohort_id))

    # ------------------------------------------------------------------ GoodOption catalog evidence (read on demand)

    @app.get("/api/drugs/{drug_id}")
    def drug_evidence(drug_id: str):
        return state.evidence.drug(drug_id, state.store)

    @app.get("/api/classes/{class_id}")
    def class_evidence(class_id: str):
        return state.evidence.klass(class_id, state.store)

    # ------------------------------------------------------------------ patients

    @app.get("/api/patients")
    def patients(q: str = "", cancer_type: str = "", sex: str = "", offset: int = 0, limit: int = 50):
        mask = state.patient_mask(cancer_type, "", "")
        out = state.corpus.search_patients(q, "", sex, max(0, offset), max(1, min(limit, 500)), mask, "text")
        out["rows"] = [state.patient_summary(r["id"]) for r in out["rows"]]
        return out

    @app.get("/api/patients/{pid}")
    def patient(pid: str):
        entity = state.patient_entity(pid)
        return {**state.patient_summary(pid), "key": entity["key"], "boilerplate": entity["boilerplate"],
                "cancer_type_labels": type_labels(state.patient_summary(pid).get("cancer_types") or [])}

    @app.get("/api/patients/{pid}/notes")
    def patient_notes(pid: str):
        entity = state.patient_entity(pid)
        if entity["custom"]:
            return {"state": "none", "notes": [], "message": "An entered summary has no notes."}
        current = state.notes.ensure()
        if current != "ready":
            return {"state": current, "notes": [], "message": state.notes.error or "Indexing the notes (about 30 s, once)."}
        return {"state": "ready", "notes": state.notes.notes(pid)}

    # ------------------------------------------------------------------ matching

    def require_round():
        if state.vectors is None:
            raise HTTPException(503, "no valid TrialSpace round")

    @app.post("/api/match/patients-for-cohort")
    def patients_for_cohort(request: PatientsForCohort):
        require_round()
        started = time.monotonic()
        with state.round_lock:
            cohort = state.cohort(request.cohort_id)
            vector = state.cohort_vectors.vector(state.store, cohort["id"])
            if vector is None:
                job = state.embed_cohorts()
                return JSONResponse({"error": "This cohort's TrialSpace vector is still being made; try again shortly.",
                                     "pending": True, "job": job.describe(False) if job else None}, status_code=409)
            k = max(1, min(int(request.k or state.settings.default_k), state.settings.max_k))
            mask = state.patient_mask(request.cancer_type, request.sex, "")
            hits = state.vectors.search(vector, "patients", k, mask)
            searched = len(state.corpus.patients) if mask is None else int(mask.sum())
        rows = [{"rank": rank, "patient": state.patient_summary(state.corpus.patients[i]["id"]), "cosine": round(c, 5)}
                for rank, (i, c) in enumerate(hits, 1)]
        pairs = [state.pair(r["patient"]["id"], cohort["id"]) for r in rows]
        for row, cached in zip(rows, state.decisions.cached_scores(pairs)):
            row["scores"] = cached
        return {"direction": "patients_for_cohort", "query": cohort_detail(state, cohort), "k": k, "searched": searched,
                "of": len(state.corpus.patients), "round": state.round.number,
                "search_ms": round((time.monotonic() - started) * 1000, 1), "results": rows}

    @app.post("/api/match/trials-for-patient")
    def trials_for_patient(request: TrialsForPatient):
        require_round()
        from matchminer_ai.matching.retrieval import stable_top_k

        started = time.monotonic()
        with state.round_lock:
            if request.patient_id:
                entity = state.patient_entity(request.patient_id)
            elif request.summary and request.summary.strip():
                entity = state.add_custom_patient(request.summary, request.boilerplate or "")
            else:
                raise ValueError("choose a patient or enter a cancer summary")
            store = state.store
            scope = [c["id"] for c in store["cohorts"].values() if request.scope == "all" or c["primary"]]
            ids, matrix = state.cohort_vectors.matrix(store, scope)
            if not ids:
                raise HTTPException(409, "The trial cohorts' TrialSpace vectors are still being made; try again shortly.")
            scores = (matrix @ np.asarray(state.patient_vector(entity), dtype=np.float32)).astype(np.float64)
        k = max(1, min(int(request.k or state.settings.default_k), state.settings.max_k))
        order, seen = [], set()
        for i in stable_top_k(scores, len(ids)):
            trial_id = store["cohorts"][ids[i]]["trial_id"]
            if request.per_trial and trial_id in seen:
                continue
            seen.add(trial_id)
            order.append(i)
            if len(order) == k:
                break
        rows = [{"rank": rank, "cohort": cohort_row(state, store["cohorts"][ids[i]]), "cosine": round(float(scores[i]), 5)}
                for rank, i in enumerate(order, 1)]
        pairs = [state.pair(entity["id"], r["cohort"]["cohort_id"]) for r in rows]
        for row, cached in zip(rows, state.decisions.cached_scores(pairs)):
            row["scores"] = cached
        return {"direction": "trials_for_patient", "patient": {**state.patient_summary(entity["id"]), "key": entity["key"],
                                                                "boilerplate": entity["boilerplate"]},
                "k": k, "searched": len(ids), "of": len(scope), "round": state.round.number,
                "search_ms": round((time.monotonic() - started) * 1000, 1), "results": rows}

    @app.post("/api/score")
    def score(request: ScoreRequest):
        if request.direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {DIRECTIONS}")
        if not request.target_ids:
            raise ValueError("give target_ids")
        if len(request.target_ids) > state.settings.max_score_n:
            raise ValueError(f"score at most {state.settings.max_score_n} pairs at a time")
        if request.direction == "patients_for_cohort":
            pairs = [state.pair(t, request.query_id) for t in request.target_ids]
        else:
            pairs = [state.pair(request.query_id, t) for t in request.target_ids]
        job = state.jobs.submit("score", f"{len(pairs)} pairs",
                                lambda progress: {"scores": state.decisions.score(pairs, ("match", "boilerplate"), progress)},
                                device=state.decisions.device)
        return {"job": job.describe(with_result=False)}

    @app.post("/api/criteria")
    def criteria(request: PairRequest):
        pair = state.pair(request.patient_id, request.cohort_id)
        if not pair.items:
            raise ValueError("this cohort has no tagged criteria lines")
        job = state.jobs.submit("criteria", f"{request.patient_id} x {request.cohort_id}",
                                lambda progress: state.decisions.criteria(pair, progress), device=state.decisions.device)
        return {"job": job.describe(with_result=False)}

    # ------------------------------------------------------------------ LLM checks (the configured endpoint)

    @app.get("/api/llm")
    def llm():
        return state.llm.describe()

    @app.post("/api/llm")
    def set_llm(request: ConnectionRequest):
        state.llm.set_connection(Connection.make(request.backend, request.endpoint, request.model, request.tokenizer))
        return state.llm.describe()

    @app.post("/api/llm/models")
    async def llm_models(request: ConnectionRequest):
        connection = (Connection.make(request.backend) if request.backend != "openai"
                      else state.llm.connection if not request.endpoint
                      else Connection("openai", "", "", ""))
        return await discover_models(connection, request.endpoint)

    @app.post("/api/good-option")
    def good_option(request: LLMCheckRequest):
        if not request.trial_id:
            raise ValueError("give trial_id")
        trial = state.trial(request.trial_id)
        entity = state.patient_entity(request.patient_id)
        return state.llm.good_option({"id": entity["id"], "key": entity["key"]}, trial["id"], trial.get("drugs") or [],
                                     request.rerun)

    @app.post("/api/criteria-llm")
    def criteria_llm(request: LLMCheckRequest):
        if not request.cohort_id:
            raise ValueError("give cohort_id")
        cohort = state.cohort(request.cohort_id)
        return state.llm.criteria(state.pair(request.patient_id, cohort["id"]), request.rerun)

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str):
        found = state.jobs.get(job_id) or state.background.get(job_id) or state.llm.jobs.get(job_id)
        if found is None:
            return JSONResponse({"error": "unknown job"}, status_code=404)
        return found.describe()

    # ------------------------------------------------------------------ status

    @app.get("/api/status")
    def status():
        return state.describe()

    @app.post("/api/rebuild")
    def rebuild():
        job = state.background.submit("rebuild", "dashboard store", lambda progress: {"rebuilt": state.check_store()})
        return {"job": job.describe(with_result=False)}

    # ------------------------------------------------------------------ the frontend

    if (STATIC / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=STATIC / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        if path.startswith("api/"):
            return JSONResponse({"error": "not found"}, status_code=404)
        file = (STATIC / path).resolve()
        if path and file.is_file() and STATIC.resolve() in file.parents:
            return FileResponse(file)
        index = STATIC / "index.html"
        if not index.exists():
            return JSONResponse({"error": "frontend not built: cd frontend && pnpm build"}, status_code=503)
        return FileResponse(index, headers={"Cache-Control": "no-cache"})

    return app
