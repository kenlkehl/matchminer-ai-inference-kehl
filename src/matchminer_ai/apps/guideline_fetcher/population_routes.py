"""Session-scoped population routes using the existing cancellable job worker."""

import asyncio
import json
import shutil
import tempfile
from pathlib import Path
from typing import Literal

from fastapi import File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, ConfigDict, Field
from starlette.background import BackgroundTask

from .jobs import BusyError
from .population import (
    PopulationInference,
    adapt_tables,
    browse_databases,
    inspect_database,
    load_population_sample,
    parse_table,
    records,
    resolve_database,
)
from .services import InputError


class PopulationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: str = Field(min_length=1, max_length=100)


class CacheRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    path: str | None = Field(default=None, max_length=4096)


class SampleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    size: int = Field(default=1000, ge=1, le=100_000, strict=True)
    seed: int = Field(default=23, ge=0, le=2**32 - 1, strict=True)


class DatabaseSampleRequest(SampleRequest):
    """PHI mode: sample from an operator-selected DuckDB database."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    database: str | None = Field(default=None, max_length=4096)


class PopulationSearch(PopulationRequest):
    catalog_id: str = Field(min_length=1, max_length=100)
    space_trial_id: str = Field(min_length=1, max_length=1000)
    candidate_k: int = Field(default=50, ge=1, le=100_000, strict=True)
    min_reasonableness: float | None = Field(default=0.75, ge=0, le=1)


class PopulationReview(PopulationRequest):
    note_source: Literal["original", "compressed"] = "original"
    max_concurrent_patients: int = Field(default=4, ge=1, le=32, strict=True)
    # full_notes: the LLM reads every patient's notes. extraction_queries: compiled queries
    # run in code on precomputed extraction records (no patient text reaches an LLM).
    # extraction_sql: the SQL agent explores the store's extracted values (they reach the
    # concordance connection), then each accepted query runs once over the cohort in code.
    method: Literal["full_notes", "extraction_queries", "extraction_sql"] = "full_notes"
    extraction_records: str | None = Field(default=None, max_length=4096)


def register_population_routes(api, *, settings, catalogs, jobs, sessions, current_settings):
    def idle(owner):
        with jobs.lock:
            if any(
                j["owner"] == owner and j["status"] in {"queued", "running", "cancelling"}
                for j in jobs.items.values()
            ):
                raise BusyError("Wait for the current job to finish or stop it first.")

    def entry_for(owner, version=None, active=False):
        entry = sessions.get(owner)
        if not entry or (version is not None and entry["version"] != version):
            raise InputError(
                "Load the population again; this cohort is unavailable or has changed."
            )
        if active and entry["active"] is None:
            raise InputError("Fetch patients for a guideline space first.")
        return entry

    @api.get("/api/population")
    def state(request: Request):
        with jobs.lock:
            mine = [
                j
                for j in jobs.items.values()
                if j["owner"] == request.state.owner and j["kind"].startswith("population-")
            ]
            latest = max(mine, key=lambda j: j["updated"], default=None)
        return {
            "population": sessions.view(request.state.owner),
            "job": jobs.get(request.state.owner, latest["id"]) if latest else None,
            "defaults": {
                "size": settings.population_sample_size,
                "seed": settings.population_seed,
                "extraction_records": str(settings.extraction_records),
                **({"database": str(settings.population_duckdb),
                    "cache": str(settings.population_cache)} if settings.phi else {}),
            },
        }

    if settings.phi:
        @api.get("/api/population/cache")
        def cache(path: str | None = None):
            from .population_cache import inspect_cache

            return inspect_cache(settings, path)

        @api.post("/api/population/cache", status_code=202)
        def load_cache(body: CacheRequest, request: Request):
            from .population_cache import cached_population, open_cache, resolve_cache

            owner = request.state.owner
            resolve_cache(settings, body.path)

            def run(progress):
                progress("Loading preprocessed summaries and TrialSpace embeddings")
                return sessions.save(owner, cached_population(open_cache(settings, body.path)))

            return {"job_id": jobs.submit(owner, "population-cache", run)}

        @api.get("/api/population/databases")
        def databases(path: str | None = None):
            return browse_databases(settings, path)

        @api.get("/api/population/database")
        def database(path: str | None = None):
            return inspect_database(settings, path)

    @api.get("/api/population/extractions")
    def extractions(path: str | None = None):
        """Manifest facts of an extraction record store; no patient rows are read."""
        from .extraction_workup import inspect_store

        return inspect_store(settings, path)

    @api.get("/api/population/spaces")
    def spaces(catalog_id: str):
        from matchminer_ai.trials import load_guideline_catalog

        try:
            frame = load_guideline_catalog(catalogs.resolve([catalog_id]))
            columns = [
                c
                for c in (
                    "space_trial_id", "name", "cohort_key_criteria", "clinical_space_summary",
                    "disease", "title",
                )
                if c in frame
            ]
            return {"spaces": records(frame[columns])}
        except InputError:
            raise
        except Exception:
            raise InputError(
                "Could not validate this completed catalog. Refresh the library."
            ) from None

    # PHI mode samples a DuckDB database; the non-PHI app, its synthetic Parquets.
    sample_request = DatabaseSampleRequest if settings.phi else SampleRequest

    @api.post("/api/population/sample", status_code=202)
    def sample(body: sample_request, request: Request):
        owner = request.state.owner
        database = getattr(body, "database", None)
        if settings.phi:
            resolve_database(settings, database)

        def run(progress):
            cohort = load_population_sample(
                settings, body.size, body.seed, progress, database=database
            )
            return sessions.save(owner, cohort)

        return {"job_id": jobs.submit(owner, "population-sample", run)}

    @api.post("/api/population/tables", status_code=202)
    async def tables(
        request: Request, summaries: UploadFile = File(...), notes: UploadFile = File(...)
    ):
        owner = request.state.owner
        idle(owner)
        try:
            a = await summaries.read(settings.population_upload_limit + 1)
            b = await notes.read(settings.population_upload_limit + 1)
        finally:
            await summaries.close()
            await notes.close()
        if len(a) + len(b) > settings.population_upload_limit:
            raise InputError("Combined cohort tables exceed the 512 MB upload limit.")

        def run(progress):
            from matchminer_ai.population import create_patient_cohort

            progress("Validating patient summaries and source notes")
            try:
                s, n = adapt_tables(
                    parse_table(summaries.filename, a), parse_table(notes.filename, b)
                )
                cohort = create_patient_cohort(s, n, provenance={"source": "uploaded_tables"})
            except InputError:
                raise
            except Exception:
                raise InputError(
                    "Invalid cohort tables. Check unique patient IDs, matching notes, text and dates."
                ) from None
            return sessions.save(owner, cohort)

        return {"job_id": jobs.submit(owner, "population-tables", run)}

    @api.post("/api/population/import")
    async def restore(request: Request, file: UploadFile = File(...)):
        from matchminer_ai.population import load_patient_cohort

        owner = request.state.owner
        idle(owner)
        try:
            with tempfile.TemporaryDirectory(prefix="guideline-cohort-import-") as tmp:
                path = Path(tmp) / "cohort.zip"
                size = 0
                with path.open("wb") as output:
                    while block := await file.read(1024 * 1024):
                        size += len(block)
                        if size > settings.population_archive_limit:
                            raise InputError(
                                "Cohort archive exceeds the configured archive upload limit."
                            )
                        output.write(block)
                try:
                    cohort = await asyncio.to_thread(load_patient_cohort, path)
                except Exception:
                    raise InputError(
                        "Invalid or incompatible cohort archive. Export a fresh cohort file."
                    ) from None
            with jobs.lock:
                idle(owner)
                return {
                    "population": sessions.save(owner, cohort, cohort if cohort.space else None)
                }
        finally:
            await file.close()

    @api.post("/api/population/retrieve", status_code=202)
    def retrieve(body: PopulationSearch, request: Request):
        owner = request.state.owner
        entry = entry_for(owner, body.version)
        paths = catalogs.resolve([body.catalog_id])
        worker = PopulationInference(current_settings(owner))

        def run(progress):
            active = worker.retrieve(
                entry["source"],
                paths,
                body.space_trial_id,
                body.candidate_k,
                body.min_reasonableness,
                progress,
            )
            return sessions.save(owner, entry["source"], active, expected=entry)

        return {"job_id": jobs.submit(owner, "population-retrieve", run)}

    def stage(owner, body, kind, action):
        entry = entry_for(owner, body.version, active=True)
        if entry["active"].summaries.empty:
            raise InputError(
                "No patients met this selection. Adjust the candidate count or threshold."
            )
        worker = PopulationInference(current_settings(owner))

        def run(progress):
            cohort = action(worker, entry["active"], progress)
            return sessions.save(owner, entry["source"], cohort, expected=entry)

        return {"job_id": jobs.submit(owner, kind, run)}

    @api.post("/api/population/compress", status_code=202)
    def compress(body: PopulationRequest, request: Request):
        return stage(
            request.state.owner, body, "population-compress", lambda w, c, p: w.compress(c, p)
        )

    @api.post("/api/population/review", status_code=202)
    def review(body: PopulationReview, request: Request):
        entry = entry_for(request.state.owner, body.version, active=True)
        if body.method in ("extraction_queries", "extraction_sql"):
            from .extraction_workup import resolve_store

            if body.note_source != "original":
                raise InputError("Extraction records quote the original notes; select "
                                 "original notes.")
            path = resolve_store(settings, body.extraction_records)
            return stage(
                request.state.owner, body, "population-review",
                lambda w, c, p: w.review_extractions(c, path, p, method=body.method),
            )
        if body.note_source == "compressed" and entry["active"].compressed_notes is None:
            raise InputError("Compress the cohort notes first, or select original notes.")
        return stage(
            request.state.owner,
            body,
            "population-review",
            lambda w, c, p: w.review(
                c, body.note_source, p, max_concurrent_patients=body.max_concurrent_patients,
            ),
        )

    @api.get("/api/population/items/{number}")
    def patients_for_item(number: int, request: Request, version: str):
        cohort = entry_for(request.state.owner, version, active=True)["active"]
        if not cohort.report or not 0 <= number < len(cohort.report["aggregate"]):
            raise HTTPException(404, "This workup review is unavailable.")
        # Evidence is fetched for an individual patient, avoiding an enormous dashboard payload.
        return {
            "patients": [
                dict(
                    patient_id=p["patient_id"],
                    **{
                        k: v
                        for k, v in p["review"]["assessments"][number].items()
                        if k not in {"evidence", "metadata", "recommendation"}
                    },
                )
                for p in cohort.report["patients"]
            ]
        }

    @api.get("/api/population/patient")
    def patient(request: Request, version: str, patient_id: str):
        cohort = entry_for(request.state.owner, version, active=True)["active"]
        frame = cohort.summaries.loc[cohort.summaries.patient_id == patient_id]
        if frame.empty:
            raise HTTPException(404, "Patient is not in this active cohort.")
        review = next(
            (
                p["review"]
                for p in (cohort.report or {}).get("patients", [])
                if p["patient_id"] == patient_id
            ),
            None,
        )
        from matchminer_ai.patients.extraction.notes import note_ids

        notes = cohort.compressed_notes if cohort.compressed_notes is not None else cohort.notes
        mine = notes.loc[notes.patient_id == patient_id]
        # Extraction evidence names notes by the extraction stages' note ids.
        mine = mine.assign(note_id=note_ids(cohort.notes.loc[cohort.notes.patient_id == patient_id]))
        return {
            "patient": records(frame)[0],
            "notes": records(mine),
            "review": review,
            "space": cohort.space,
        }

    @api.get("/api/population/results")
    def results(request: Request, version: str, format: Literal["json", "csv"] = "json"):
        from matchminer_ai.population import population_results_table

        cohort = entry_for(request.state.owner, version, active=True)["active"]
        if cohort.report is None:
            raise InputError("Run the population review first.")
        if format == "json":
            data = json.dumps(cohort.report, ensure_ascii=False)
            media = "application/json"
        else:
            frame = population_results_table(cohort)

            # Neutralize formula prefixes in all user/model-controlled spreadsheet cells.
            def safe_cell(value):
                return (
                    "'" + value
                    if isinstance(value, str)
                    and value.lstrip().startswith(("=", "+", "-", "@", "\t", "\r"))
                    else value
                )

            data = frame.map(safe_cell).to_csv(index=False)
            media = "text/csv; charset=utf-8"
        return Response(
            data,
            media_type=media,
            headers={"Content-Disposition": f'attachment; filename="population-results.{format}"'},
        )

    @api.get("/api/population/export")
    def export(request: Request, version: str):
        from matchminer_ai.population import save_patient_cohort

        owner = request.state.owner
        idle(owner)
        cohort = entry_for(owner, version, active=True)["active"]
        if cohort.summaries.empty:
            raise InputError("There are no active patients to export.")
        tmp = tempfile.mkdtemp(prefix="guideline-cohort-export-")
        try:
            path = save_patient_cohort(cohort, Path(tmp) / "cohort.zip")
            return FileResponse(
                path,
                media_type="application/zip",
                filename="guideline-cohort.zip",
                background=BackgroundTask(shutil.rmtree, tmp),
            )
        except Exception:
            shutil.rmtree(tmp)
            raise

    @api.delete("/api/population", status_code=204)
    def clear(request: Request):
        with jobs.lock:
            idle(request.state.owner)
            sessions.clear(request.state.owner)
            for key, job in list(jobs.items.items()):
                if job["owner"] == request.state.owner and job["kind"].startswith("population-"):
                    del jobs.items[key]
                    jobs.controls.pop(key, None)
        return Response(status_code=204)
