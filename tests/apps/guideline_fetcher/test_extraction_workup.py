"""Extraction-based workup checks in both modes. Fabricated notes and records only."""

import json
import stat

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from matchminer_ai.apps.guideline_fetcher.app import create_app
from .conftest import mode_settings, phi_settings
from matchminer_ai.apps.guideline_fetcher.extraction_workup import QueryCache, inspect_store, open_store, write_store
from matchminer_ai.apps.guideline_fetcher.services import InputError, Inference
from matchminer_ai.apps.guideline_fetcher.settings import Settings
from .test_app import catalog, wait
from .test_population import NOTE_CSV, SPACE, SUMMARY_CSV, job

from matchminer_ai.llm.backends import LLMGenerationResult
from matchminer_ai.patients.extraction.notes import note_ids
from matchminer_ai.patients.workup_queries.language import EXAMPLES
from matchminer_ai.patients.workup_sql import agent_limits

CT_QUERY = dict(EXAMPLES)["CT chest/abdomen/pelvis (one combined study or separate studies)"]
CT_QUERY = {**CT_QUERY, "criteria": CT_QUERY["criteria"][:1],
            "requirement": {"all_of": CT_QUERY["requirement"]["all_of"][:1]}}


class CompileBackend:
    """Answers every compile request with the CT query, counting requests."""

    def __init__(self):
        self.requests = 0

    def generate_llm_outputs(self, *, prompt_list, llm_config, **kwargs):
        self.requests += len(prompt_list)
        assert all("REQUIREMENT" in p.messages[0]["content"] for p in prompt_list)
        return LLMGenerationResult(
            final_outputs=[json.dumps(CT_QUERY)] * len(prompt_list),
            finish_reasons=["stop"] * len(prompt_list),
            reasoning_outputs=[""] * len(prompt_list), raw_outputs=[], model_metadata={})


def study(note_id, note_date, relation="this_report"):
    return {
        "item_type": "imaging_study", "relation": relation,
        "source_of_document": relation == "this_report",
        "status": "planned" if relation == "planned" else "performed",
        "modality": "ct", "modality_text": "CT chest", "body_regions": ["chest"],
        "body_site_text": "", "contrast": "not_stated", "tracer": "",
        "study_date": ({"value": note_date, "precision": "day", "source": "note_date"}
                       if relation == "this_report" and note_date else
                       {"value": None, "precision": "unknown", "source": "unknown"}),
        "evidence": [{"note_id": note_id, "note_date": note_date, "quote": "CT", "start": 0,
                      "end": 2}],
    }


def document_rows(notes):
    """Fabricated document records: a performed CT for "CT completed", a planned one
    for "CT planned"."""
    rows = []
    for note_id, row in zip(note_ids(notes), notes.itertuples(), strict=True):
        date = None if pd.isna(row.note_date) or not str(row.note_date).strip() else \
            str(row.note_date)[:10]
        items = []
        if "completed" in row.note_text:
            items = [study(note_id, date)]
        elif "planned" in row.note_text:
            items = [study(note_id, date, "planned")]
        record = {
            "schema_version": "mmai-clinical-extraction-v1", "level": "document",
            "patient_id": str(row.patient_id), "note_id": note_id, "note_date": date,
            "note_type": None, "document_type": "imaging_report", "document_type_text": "",
            "extensions": [], "items": {"imaging_studies": items},
        }
        rows.append({"patient_id": str(row.patient_id), "note_id": note_id, "note_date": date,
                     "note_type": None, "document_type": "imaging_report",
                     "extraction_status": "complete", "error": None, "item_count": len(items),
                     "record": record})
    return pd.DataFrame(rows)


def fabricated_notes():
    import io

    return pd.read_csv(io.StringIO(NOTE_CSV), dtype=str, keep_default_na=False)


@pytest.fixture
def backend(monkeypatch):
    fake = CompileBackend()
    monkeypatch.setattr("matchminer_ai.patients.workup_queries.compile.get_llm_backend",
                        lambda _config: fake)
    return fake


def paths(tmp_path, phi):
    side = tmp_path / ("phi" if phi else "no_phi")
    return {"cache": side / "cache", "extraction_records": side / "extractions" / "records"}


def build_store(settings, notes=None):
    notes = fabricated_notes() if notes is None else notes
    return write_store(settings, settings.extraction_records, document_rows(notes),
                       source={"kind": "fabricated", "name": "fabricated"},
                       extraction={"schema_version": "mmai-clinical-extraction-v1",
                                   "model": "fabricated"})


# Stores -----------------------------------------------------------------------


@pytest.mark.parametrize("phi", [False, True], ids=["standard", "phi"])
def test_store_is_written_atomically_on_its_mode_side(tmp_path, phi):
    settings = mode_settings(phi, catalog_root=tmp_path, **paths(tmp_path, phi))
    manifest = build_store(settings)
    assert manifest["mode"] == settings.mode and manifest["counts"]["patients"] == 2
    store = open_store(settings)
    documents, patients = store.records(["a"])
    assert patients is None and list(documents.patient_id) == ["a"]
    assert documents.record[0]["items"]["imaging_studies"][0]["relation"] == "this_report"
    info = inspect_store(settings)
    assert info["counts"]["documents"]["complete"] == 2 and "path" not in info["source"]
    mode = stat.S_IMODE((settings.extraction_records / "document_records.parquet").stat().st_mode)
    assert (mode == 0o600) if phi else True
    with pytest.raises(InputError, match="exists"):
        build_store(settings)
    other = tmp_path / ("no_phi" if phi else "phi") / "x"
    with pytest.raises(InputError, match="no_phi folders" if phi else "phi folder"):
        write_store(settings, other, document_rows(fabricated_notes()), source={},
                    extraction={})
    # A store built for the other mode is refused.
    flipped = mode_settings(not phi, catalog_root=tmp_path,
                            cache=tmp_path / "neutral" / "cache",
                            extraction_records=tmp_path / "neutral" / "records")
    write_store(flipped, flipped.extraction_records, document_rows(fabricated_notes()),
                source={}, extraction={})
    with pytest.raises(InputError):
        open_store(settings, str(flipped.extraction_records))


def test_settings_keep_extraction_records_on_their_mode_side(tmp_path, monkeypatch):
    monkeypatch.setenv("GUIDELINE_EXTRACTION_RECORDS", str(tmp_path / "data/no_phi/records"))
    assert Settings().extraction_records == tmp_path / "data/no_phi/records"
    with pytest.raises(ValueError, match="no_phi"):
        Settings(phi=True, cache=tmp_path / "phi/cache")
    monkeypatch.setenv("GUIDELINE_EXTRACTION_RECORDS", str(tmp_path / "data/phi/records"))
    with pytest.raises(ValueError, match="phi folder"):
        Settings()
    assert Settings(phi=True, cache=tmp_path / "phi/cache").extraction_records == \
        tmp_path / "data/phi/records"


def test_query_cache_round_trips_and_rejects_unsafe_keys(tmp_path):
    cache = QueryCache(tmp_path / "queries")
    key = "a" * 64
    assert cache.get(key) is None and key not in cache
    cache[key] = {"status": "compiled"}
    assert QueryCache(tmp_path / "queries")[key] == {"status": "compiled"}
    with pytest.raises(KeyError):
        cache["../escape"] = {}


def test_extraction_config_maps_the_connection_onto_extraction_sections(tmp_path):
    local = Settings(catalog_root=tmp_path, llm_backend="local", model="served/model",
                     tokenizer="served/tokenizer", endpoint="http://example.invalid:8000/v1")
    config = local.extraction_config()
    for name in ("document_extraction", "patient_extraction", "workup_query_compile"):
        section = getattr(config, name)
        assert section["remote"]["model_name"] == "served/model"
        assert section["structured_outputs"] is True and section["sampling_profile"] == "auto"
    assert config.remote["server_urls"] == ["http://example.invalid:8000/v1"]
    gemma = Settings(catalog_root=tmp_path, llm_backend="gemma_vertex").extraction_config()
    assert gemma.workup_query_compile["structured_outputs"] is False
    assert gemma.workup_query_compile["remote"]["model_name"] == "google/gemma-4-26b-a4b-it-maas"


# Population view ----------------------------------------------------------------


@pytest.fixture
def population_app(tmp_path, monkeypatch, backend):
    catalog(tmp_path)

    def retrieve(summaries, paths, **kwargs):
        frame = summaries.copy()
        frame["space_trial_id"] = "s"
        frame["match_quality_score"] = [0.9, 0.3]
        frame["match_quality_pass"] = True
        frame["similarity_score"] = [0.8, 0.7]
        frame["retrieval_rank"] = [1, 2]
        frame["rank"] = [1, 2]
        return frame, {"space": SPACE, "scored_count": 2, "population_count": 2,
                       "min_reasonableness": kwargs["min_reasonableness"]}

    monkeypatch.setattr("matchminer_ai.matching.retrieve_guideline_patients", retrieve)
    monkeypatch.setattr("matchminer_ai.trials.load_guideline_catalog",
                        lambda *a, **kw: pd.DataFrame([SPACE]))

    def make(phi):
        settings = mode_settings(phi, catalog_root=tmp_path, device="cpu", llm_backend="local",
                                 **paths(tmp_path, phi))
        build_store(settings)
        return create_app(settings), settings

    return make


def active_cohort(client):
    loaded = client.post("/api/population/tables", files={
        "summaries": ("summaries.csv", SUMMARY_CSV), "notes": ("notes.csv", NOTE_CSV)})
    loaded = wait(client, loaded.json()["job_id"]).json()["result"]
    catalog_id = client.get("/api/catalogs").json()["catalogs"][0]["id"]
    return job(client, "/api/population/retrieve", dict(
        version=loaded["version"], catalog_id=catalog_id, space_trial_id="s", candidate_k=7,
        min_reasonableness=0.75))["result"]


@pytest.mark.parametrize("phi", [False, True], ids=["standard", "phi"])
def test_population_review_with_extraction_queries(population_app, backend, phi):
    api, settings = population_app(phi)
    with TestClient(api) as client:
        state = client.get("/api/population").json()
        assert state["defaults"]["extraction_records"] == str(settings.extraction_records)
        info = client.get("/api/population/extractions").json()
        assert info["counts"]["patients"] == 2 and info["levels"] == ["document"]
        cohort = active_cohort(client)
        version = {"version": cohort["version"]}
        response = client.post("/api/population/review", json={
            **version, "method": "extraction_queries", "note_source": "compressed"})
        assert response.status_code == 400
        done = job(client, "/api/population/review", {**version,
                                                       "method": "extraction_queries"})
        report = done["result"]["active"]["report"]
        assert report["method"] == "extraction_queries"
        assert report["extraction_store"]["name"] == "records"
        assert report["compile"]["requests"] == 1 and backend.requests == 1
        row = report["aggregate"][0]
        assert row["statuses"] == {"met": 1, "not_met": 1, "unknown": 0, "not_applicable": 0,
                                 "done_not_recommended": 0}
        assert row["reasons"]["planned_only"] == 1
        assert row["documented_completion_fraction"] == 0.5
        version = {"version": client.get("/api/population").json()["population"]["version"]}
        patient = client.get("/api/population/patient", params={**version,
                                                                 "patient_id": "a"}).json()
        assessment = patient["review"]["assessments"][0]
        assert assessment["status"] == "met"
        assert assessment["evidence"][0]["note_id"] == patient["notes"][0]["note_id"]
        items = client.get("/api/population/items/0", params=version).json()["patients"]
        assert {p["patient_id"]: p["status"] for p in items} == {"a": "met", "b": "not_met"}
        csv = client.get("/api/population/results", params={**version, "format": "csv"})
        assert csv.status_code == 200 and "not_met" in csv.text
        # Compiled queries are cached on disk: a second check sends no request.
        job(client, "/api/population/review", {**version, "method": "extraction_queries"})
        assert backend.requests == 1
        assert list((settings.cache / "workup_queries").glob("*.json"))
        # A full-note review stays available as the other method.
        outside = client.post("/api/population/review", json={
            "version": client.get("/api/population").json()["population"]["version"],
            "method": "extraction_queries", "extraction_records": "/elsewhere/records"})
        assert outside.status_code == 400


def test_population_extraction_paths_stay_on_the_mode_side(population_app, tmp_path):
    api, settings = population_app(False)
    with TestClient(api) as client:
        phi_path = tmp_path / "phi" / "extractions" / "records"
        response = client.get("/api/population/extractions", params={"path": str(phi_path)})
        assert response.status_code == 400 and "phi folder" in response.json()["detail"]
        response = client.get("/api/population/extractions", params={"path": "relative/path"})
        assert response.status_code == 400
    api, settings = population_app(True)
    with TestClient(api) as client:
        no_phi = tmp_path / "no_phi" / "extractions" / "records"
        response = client.get("/api/population/extractions", params={"path": str(no_phi)})
        assert response.status_code == 400 and "no_phi" in response.json()["detail"]


# Patient view -------------------------------------------------------------------


class ExtractionInference(Inference):
    def retrieve(self, summary, paths, top_n, candidate_k, progress):
        return {"summary": summary, "spaces": [
            {"name": "Synthetic space", "space_trial_id": "s",
             "cohort_key_criteria": "Inclusion criteria:\nFabricated cancer",
             "diagnostic_workup": [{"name": "CT chest", "conditions": "At diagnosis",
                                    "category": "2A"}]}]}


@pytest.mark.parametrize("phi", [False, True], ids=["standard", "phi"])
def test_patient_view_extracts_on_demand_and_reuses_session_records(
        tmp_path, monkeypatch, backend, phi):
    catalog(tmp_path)
    calls = []

    def extract(notes, *, config, patient_records=None, return_metadata=False,
                progress_callback=None, **kwargs):
        calls.append(len(notes))
        rows = document_rows(notes)
        progress_callback(len(rows), len(rows))
        return rows, {"counts": {"complete": len(rows), "partial": 0, "failed": 0},
                      "schema_version": "mmai-clinical-extraction-v1",
                      "documents": {n: {"attempts": [{"attempt": 1}]} for n in rows.note_id}}

    monkeypatch.setattr("matchminer_ai.patients.extract_document_records", extract)
    settings = mode_settings(phi, catalog_root=tmp_path, llm_backend="local",
                             **paths(tmp_path, phi))
    with TestClient(create_app(settings, ExtractionInference(settings))) as client:
        key = client.post("/api/retrieve", json={"summary": "Synthetic summary"}).json()["job_id"]
        wait(client, key)
        notes = [{"note_text": "Clinic visit.", "note_date": "2026-01-01T00:00:00"},
                 {"note_text": "CT completed.", "note_date": "2026-01-05T00:00:00",
                  "note_type": "Imaging"}]
        body = {"text": "Synthetic notes", "notes": notes, "retrieval_job_id": key,
                "space_index": 0}
        cost = client.post("/api/workup/extraction-cost", json=body).json()
        assert cost["document_requests"] == 2 and cost["compile_requests"] == 1
        assert cost["reused_extraction"] is False and backend.requests == 0
        assert client.post("/api/workup", json={**body, "method": "extractions",
                                                "note_source": "compressed"}).status_code == 400
        response = client.post("/api/workup", json={**body, "method": "extractions"})
        done = wait(client, response.json()["job_id"]).json()
        assert done["status"] == "complete", done
        review = done["result"]["review"]
        assert review["metadata"]["method"] == "extraction_queries"
        assessment = review["assessments"][0]
        assert assessment["status"] == "met" and assessment["recommendation"]["name"] == "CT chest"
        assert assessment["evidence"][0]["note_number"] == 2
        assert "server memory" in review["notice"]
        assert review["metadata"]["requests"] == 3 and calls == [2]
        again = client.post("/api/workup", json={**body, "method": "extractions"})
        review = wait(client, again.json()["job_id"]).json()["result"]["review"]
        assert calls == [2] and backend.requests == 1
        assert review["metadata"]["extraction"]["reused"] is True
        assert review["metadata"]["requests"] == 0
        cost = client.post("/api/workup/extraction-cost", json=body).json()
        assert cost["reused_extraction"] and cost["requests"] == 0
        client.delete("/api/session")
        assert not client.app.state.session_extractions.entries


def test_free_text_notes_are_one_undated_note(tmp_path, monkeypatch, backend):
    catalog(tmp_path)
    seen = []

    def extract(notes, *, config, return_metadata=False, progress_callback=None, **kwargs):
        seen.append(notes.to_dict("records"))
        rows = document_rows(notes)
        return rows, {"counts": {"complete": 1, "partial": 0, "failed": 0},
                      "schema_version": "x", "documents": {}}

    monkeypatch.setattr("matchminer_ai.patients.extract_document_records", extract)
    settings = Settings(catalog_root=tmp_path, llm_backend="local", **paths(tmp_path, False))
    with TestClient(create_app(settings, ExtractionInference(settings))) as client:
        key = client.post("/api/retrieve", json={"summary": "Synthetic"}).json()["job_id"]
        wait(client, key)
        response = client.post("/api/workup", json={
            "text": "CT completed.", "retrieval_job_id": key, "space_index": 0,
            "method": "extractions"})
        done = wait(client, response.json()["job_id"]).json()
        assert done["status"] == "complete", done
        assert seen[0][0]["note_date"] is None
        assert done["result"]["review"]["metadata"]["undated_notes"] == 1
        assert done["result"]["review"]["assessments"][0]["status"] == "met"


# Offline build script -------------------------------------------------------------


def test_preprocess_extractions_builds_a_synthetic_store(tmp_path, monkeypatch):
    from matchminer_ai.apps.guideline_fetcher import preprocess_extractions
    notes = pd.DataFrame({
        "pseudo_mrn": [1, 1, 2, 3], "synthetic_note": ["CT completed.", "Visit.", "CT planned.",
                                                       "Visit."],
        "date": ["2026-01-01", "2026-01-02", "2026-01-03", None],
        "event_type": ["Imaging", "Visit", "Visit", "Visit"],
    })
    source = tmp_path / "no_phi" / "notes.parquet"
    source.parent.mkdir()
    notes.to_parquet(source)
    seen = {}

    def extract(frame, *, config, patient_records=None, checkpoint_dir=None,
                return_metadata=False, progress_callback=None):
        seen.update(patients=sorted(set(frame.patient_id)), checkpoints=checkpoint_dir,
                    timeout=config.remote["request_timeout"])
        return document_rows(frame), {"schema_version": "mmai-clinical-extraction-v1",
                                      "counts": {"complete": len(frame)}}

    monkeypatch.setattr("matchminer_ai.patients.extract_document_records", extract)
    monkeypatch.setenv("GUIDELINE_EXTRACTION_RECORDS", str(tmp_path / "no_phi/ex/default"))
    monkeypatch.setenv("GUIDELINE_CACHE", str(tmp_path / "no_phi/cache"))
    output = tmp_path / "no_phi/ex/built"
    preprocess_extractions.main(["--notes", str(source), "--output", str(output),
                                 "--limit", "2", "--endpoint", "http://example.invalid/v1"])
    assert len(seen["patients"]) == 2 and seen["timeout"] == 14400
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["mode"] == "standard" and manifest["source"]["patients"] == 2
    assert manifest["source"]["kind"] == "synthetic_notes"
    assert seen["checkpoints"] == output.parent / ".built.checkpoints"
    with pytest.raises(SystemExit, match="exists"):
        preprocess_extractions.main(["--notes", str(source), "--output", str(output)])
    with pytest.raises(SystemExit, match="phi folder"):
        preprocess_extractions.main(["--notes", str(tmp_path / "phi/notes.parquet"),
                                     "--output", str(tmp_path / "no_phi/ex/other")])


def test_preprocess_extractions_phi_reads_a_population_cache_privately(tmp_path, monkeypatch):
    from matchminer_ai.apps.guideline_fetcher import preprocess_extractions
    from matchminer_ai.apps.guideline_fetcher.population_cache import FORMAT, VERSION

    cache = tmp_path / "phi" / "population_cache"
    cache.mkdir(parents=True)
    notes = pd.DataFrame({"patient_id": ["11", "12"], "note_text": ["CT completed.", "Visit."],
                          "note_date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
                          "note_type": ["Imaging", "Visit"]})
    notes.to_parquet(cache / "notes.parquet")
    (cache / "manifest.json").write_text(json.dumps({
        "format": FORMAT, "version": VERSION, "created_utc": "2026-11-12T00:00:00+00:00",
        "counts": {"patients": 2, "notes": 2},
        "files": {"notes.parquet": (cache / "notes.parquet").stat().st_size}}))

    def extract(frame, *, config, patient_records=None, checkpoint_dir=None,
                return_metadata=False, progress_callback=None):
        return document_rows(frame), {"schema_version": "x", "counts": {}}

    def patient_extract(frame, *, config, checkpoint_dir=None, return_metadata=False,
                        progress_callback=None):
        rows = pd.DataFrame([{"patient_id": p, "extraction_status": "complete", "error": None,
                              "failed_chunks": [], "item_count": 0, "record": None}
                             for p in sorted(set(frame.patient_id))])
        return rows, {"schema_version": "x", "counts": {}}

    monkeypatch.setattr("matchminer_ai.patients.extract_document_records", extract)
    monkeypatch.setattr("matchminer_ai.patients.extract_patient_records", patient_extract)
    monkeypatch.setenv("GUIDELINE_DUCKDB_ROOTS", str(tmp_path / "phi"))
    monkeypatch.setenv("GUIDELINE_CACHE", str(tmp_path / "phi/app_cache"))
    monkeypatch.setenv("GUIDELINE_EXTRACTION_RECORDS", str(tmp_path / "phi/ex/default"))
    output = tmp_path / "phi" / "ex" / "records"
    preprocess_extractions.main(["--phi", "--cache", str(cache), "--output", str(output),
                                 "--levels", "document_and_patient"])
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["mode"] == "phi" and manifest["levels"] == ["document", "patient"]
    for name in ("manifest.json", "document_records.parquet", "patient_records.parquet"):
        assert stat.S_IMODE((output / name).stat().st_mode) == 0o600
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    settings = phi_settings(cache=tmp_path / "phi/app_cache")
    _, patients = open_store(settings, str(output)).records(["11"])
    assert patients.note_ids[0] == note_ids(notes.iloc[:1].assign(
        note_date=notes.note_date.iloc[:1]))
    with pytest.raises(SystemExit, match="no_phi"):
        preprocess_extractions.main(["--phi", "--cache", str(cache), "--output",
                                     str(tmp_path / "no_phi/records")])


# SQL agent ----------------------------------------------------------------------

AGENT_SQL = """
SELECT p.patient_key,
  CASE WHEN m.k IS NOT NULL THEN 'met' ELSE 'not_met' END AS status,
  'applies' AS applicability,
  CASE WHEN m.k IS NOT NULL THEN 'documented'
       WHEN pl.k IS NOT NULL THEN 'planned_only' ELSE 'not_documented' END AS reason,
  COALESCE(m.k, pl.k, []::BIGINT[]) AS evidence_item_keys,
  COALESCE(m.d, '') AS detail
FROM patients p
LEFT JOIN (SELECT patient_key, list(item_key) AS k, string_agg(label, '; ') AS d
           FROM imaging_studies WHERE modality = 'ct' AND occurrence = 'performed'
           GROUP BY patient_key) m USING (patient_key)
LEFT JOIN (SELECT patient_key, list(item_key) AS k FROM imaging_studies
           WHERE modality = 'ct' AND occurrence = 'planned' GROUP BY patient_key) pl
  USING (patient_key)
"""


def wait_long(client, key, seconds=60):
    """SQL-agent jobs start isolated DuckDB workers: allow more than the shared second."""
    import time

    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        response = client.get(f"/api/jobs/{key}")
        if response.json()["status"] in {"complete", "failed"}:
            return response
        time.sleep(0.05)
    pytest.fail("Job did not finish")


def job_long(client, route, body):
    response = client.post(route, json=body)
    assert response.status_code == 202, response.text
    result = wait_long(client, response.json()["job_id"]).json()
    assert result["status"] == "complete", result
    return result


class Agent:
    """A scripted SQL agent: search once, submit AGENT_SQL, accept it."""

    def __init__(self):
        self.requests, self.payloads = 0, []

    def complete(self, client, job, messages, schema, validator):
        payload = json.loads(messages[1]["content"])
        self.payloads.append(payload)
        self.requests += 1
        client.requests += 1
        base = {"purpose": "step", "summary": "", "approximations": [],
                "not_expressible_reason": "", "recommends_against": False, "memory": "",
                "sql": "", "terms": []}
        if payload["pending_submission"]:
            value = {**base, "action": "accept"}
        elif not payload["recent_steps"]:
            value = {**base, "action": "search", "terms": ["ct", "computed tomography"]}
        else:
            value = {**base, "action": "submit", "sql": AGENT_SQL, "summary": "A performed CT."}
        return validator(value)


@pytest.fixture
def agent(monkeypatch):
    from matchminer_ai.llm.structured import StructuredConfig
    from matchminer_ai.patients import note_search_qa

    fake = Agent()
    monkeypatch.setattr(note_search_qa._MeasuredClient, "complete",
                        lambda client, *args: fake.complete(client, *args))
    # No endpoint discovery: the scripted agent stands in for the concordance connection.
    monkeypatch.setattr(
        "matchminer_ai.patients.workup_sql.agent.resolve_agent_llm",
        lambda config: StructuredConfig(base_url="http://fabricated.invalid/v1",
                                        model="fabricated-agent", tokenizer_mode="bytes",
                                        response_format="json_schema", attempts=1))
    return fake


@pytest.mark.parametrize("phi", [False, True], ids=["standard", "phi"])
def test_population_review_with_the_sql_agent(population_app, agent, phi):
    api, settings = population_app(phi)
    with TestClient(api) as client:
        cohort = active_cohort(client)
        version = {"version": cohort["version"]}
        assert client.post("/api/population/review", json={
            **version, "method": "extraction_sql", "note_source": "compressed"}).status_code == 400
        done = job_long(client, "/api/population/review", {**version, "method": "extraction_sql"})
        report = done["result"]["active"]["report"]
        assert report["method"] == "extraction_sql" and report["extraction_store"]["name"] == "records"
        assert report["compile"]["requests"] == 3 and agent.requests == 3
        assert report["compile"]["exploration"]["patients"] == 2
        row = report["aggregate"][0]
        assert row["statuses"] == {"met": 1, "not_met": 1, "unknown": 0, "not_applicable": 0,
                                 "done_not_recommended": 0}
        assert row["reasons"]["planned_only"] == 1
        assert report["queries"][0]["sql"].startswith("SELECT p.patient_key")
        assert report["queries"][0]["cohort_run"]["status"] == "ran"
        # The agent saw extracted values, never patient ids or note text.
        seen = json.dumps(agent.payloads)
        assert "CT completed" not in seen and '"a"' not in seen
        version = {"version": client.get("/api/population").json()["population"]["version"]}
        patient = client.get("/api/population/patient", params={**version,
                                                                 "patient_id": "a"}).json()
        assessment = patient["review"]["assessments"][0]
        assert assessment["status"] == "met" and assessment["sql"]
        assert assessment["evidence"][0]["note_id"] == patient["notes"][0]["note_id"]
        assert patient["review"]["metadata"]["method"] == "extraction_sql"
        csv = client.get("/api/population/results", params={**version, "format": "csv"})
        assert "extraction_sql" in csv.text
        # The exploration database and the accepted query are cached on the mode's side.
        databases = list((settings.cache / "extraction_databases").glob("*.duckdb"))
        queries = list((settings.cache / "workup_sql").glob("*.json"))
        assert len(databases) == 1 and len(queries) == 1
        if phi:
            assert stat.S_IMODE(databases[0].stat().st_mode) == 0o600
            assert stat.S_IMODE(databases[0].parent.stat().st_mode) == 0o700
        cached = json.loads(queries[0].read_text())
        assert "CT completed" not in json.dumps(cached) and cached["sql"]
        job_long(client, "/api/population/review", {**version, "method": "extraction_sql"})
        assert agent.requests == 3


@pytest.mark.parametrize("phi", [False, True], ids=["standard", "phi"])
def test_patient_view_sql_agent_explores_the_default_store(tmp_path, monkeypatch, agent, phi):
    catalog(tmp_path)

    def extract(notes, *, config, patient_records=None, return_metadata=False,
                progress_callback=None, **kwargs):
        rows = document_rows(notes)
        return rows, {"counts": {"complete": len(rows), "partial": 0, "failed": 0},
                      "schema_version": "mmai-clinical-extraction-v1",
                      "documents": {n: {"attempts": [{"attempt": 1}]} for n in rows.note_id}}

    monkeypatch.setattr("matchminer_ai.patients.extract_document_records", extract)
    settings = mode_settings(phi, catalog_root=tmp_path, llm_backend="local",
                             **paths(tmp_path, phi))
    with TestClient(create_app(settings, ExtractionInference(settings))) as client:
        key = client.post("/api/retrieve", json={"summary": "Synthetic summary"}).json()["job_id"]
        wait(client, key)
        notes = [{"note_text": "Clinic visit.", "note_date": "2026-01-01T00:00:00"},
                 {"note_text": "CT completed.", "note_date": "2026-01-05T00:00:00"}]
        body = {"text": "Synthetic notes", "notes": notes, "retrieval_job_id": key,
                "space_index": 0}
        # Without a default store the SQL agent has nothing to explore.
        missing = client.post("/api/workup/extraction-cost",
                              json={**body, "method": "extraction_sql"})
        assert missing.status_code == 400 and "preprocess_extractions" in missing.text
        build_store(settings)
        cost = client.post("/api/workup/extraction-cost",
                           json={**body, "method": "extraction_sql"}).json()
        assert cost["sql_agent"]["store"] == "records" and cost["sql_agent"]["patients"] == 2
        assert cost["sql_agent"]["items_to_explore"] == 1 and not cost["sql_agent"]["database_built"]
        assert cost["compile_requests"] == agent_limits(settings.extraction_config()).max_actions
        assert agent.requests == 0
        response = client.post("/api/workup", json={**body, "method": "extraction_sql"})
        done = wait_long(client, response.json()["job_id"]).json()
        assert done["status"] == "complete", done
        review = done["result"]["review"]
        assert review["metadata"]["method"] == "extraction_sql"
        assert review["metadata"]["queries"]["queries"][0]["sql"].startswith("SELECT")
        assessment = review["assessments"][0]
        assert assessment["status"] == "met" and assessment["evidence"][0]["note_number"] == 2
        assert "SQL agent explored the extraction records of records" in review["notice"]
        cost = client.post("/api/workup/extraction-cost",
                           json={**body, "method": "extraction_sql"}).json()
        assert cost["sql_agent"]["items_to_explore"] == 0 and cost["sql_agent"]["database_built"]
        assert cost["requests"] == 0 and cost["reused_extraction"]
        # A failed run saved for diagnosis is explored again, so it is not counted cached.
        from matchminer_ai.apps.guideline_fetcher.extraction_workup import sql_cache

        cache, key = sql_cache(settings), review["metadata"]["queries"]["queries"][0]["cache_key"]
        cache[key] = {**cache[key], "status": "failed", "sql": None}
        cost = client.post("/api/workup/extraction-cost",
                           json={**body, "method": "extraction_sql"}).json()
        assert cost["sql_agent"]["items_to_explore"] == 1


def test_extraction_config_maps_the_sql_agent_section(tmp_path):
    local = Settings(catalog_root=tmp_path, llm_backend="local", model="served/model",
                     tokenizer="served/tokenizer", endpoint="http://example.invalid:8000/v1")
    section = local.extraction_config().workup_sql_agent
    assert section["remote"]["model_name"] == "served/model"
    assert "max_model_len" not in section["local"].get("engine", {})
    gemini = Settings(catalog_root=tmp_path, llm_backend="gemini4_vertex").extraction_config()
    section = gemini.workup_sql_agent
    assert section["remote"]["model_name"] == "google/gemini-4-hydrogen"
    assert section["structured_outputs"] is True
    assert section["local"]["engine"]["max_model_len"] == 1048576
    assert gemini.workup_query_compile["structured_outputs"] is False


# History-and-physical items ------------------------------------------------------


def test_history_and_physical_items_are_left_out_of_workup_checks(population_app, agent,
                                                                  monkeypatch):
    from matchminer_ai.apps.guideline_fetcher.services import display_considerations, is_history_and_physical

    for name in ("History and physical examination", "History and physical (H&P)",
                 "Physical examination and performance status",
                 "History and physical examination with complete skin examination"):
        assert is_history_and_physical({"name": name}), name
    for name in ("Initial history, physical, and laboratory evaluation", "Family history",
                 "History and physical examination with nasal endoscopy", "Chest CT"):
        assert not is_history_and_physical({"name": name}), name
    frame = pd.DataFrame([{"diagnostic_workup": [
        {"name": "History and physical examination", "category": "2A"},
        {"name": "Chest CT", "category": "Not specified in supplied evidence"}],
        "treatment_options": []}])
    shown = display_considerations(frame).iloc[0]["diagnostic_workup"]
    assert [i["name"] for i in shown] == ["Chest CT"]
    # The population view's cohort space leaves them out too.
    space = {**SPACE, "diagnostic_workup": [{"name": "History and physical", "conditions": ""},
                                            *SPACE["diagnostic_workup"]]}

    def retrieve(summaries, paths, **kwargs):
        frame = summaries.copy()
        frame["space_trial_id"] = "s"
        for column, values in (("match_quality_score", [0.9, 0.3]), ("similarity_score",
                                [0.8, 0.7]), ("retrieval_rank", [1, 2]), ("rank", [1, 2])):
            frame[column] = values
        frame["match_quality_pass"] = True
        return frame, {"space": space, "scored_count": 2, "population_count": 2,
                       "min_reasonableness": kwargs["min_reasonableness"]}

    monkeypatch.setattr("matchminer_ai.matching.retrieve_guideline_patients", retrieve)
    api, _ = population_app(False)
    with TestClient(api) as client:
        cohort = active_cohort(client)
        assert [i["name"] for i in cohort["active"]["space"]["diagnostic_workup"]] == ["Chest CT"]


def test_preprocess_extractions_spreads_a_build_over_a_server_registry(tmp_path, monkeypatch):
    """Two servers with different builds of a model: each request names its server's model."""
    from matchminer_ai.apps.guideline_fetcher import preprocess_extractions
    notes = pd.DataFrame({"pseudo_mrn": [1, 2], "synthetic_note": ["CT completed.", "Visit."],
                          "date": ["2026-01-01", "2026-01-02"], "event_type": ["Imaging", "Visit"]})
    source = tmp_path / "no_phi" / "notes.parquet"
    source.parent.mkdir()
    notes.to_parquet(source)
    registry = tmp_path / "servers.json"
    servers = [{"url": "http://a.invalid:8000/v1", "model": "fabricated/model-NVFP4",
                "upstream_concurrency": 128},
               {"url": "http://b.invalid:8002", "model": "fabricated/model-FP8",
                "upstream_concurrency": 24}]
    registry.write_text(json.dumps({"servers": servers}))
    served = {"http://a.invalid:8000/v1": ["fabricated/model-NVFP4"],
              "http://b.invalid:8002/v1": ["fabricated/model-FP8"]}
    monkeypatch.setattr(preprocess_extractions, "served_models", lambda url: served[url])
    seen = {}

    def extract(frame, *, config, patient_records=None, checkpoint_dir=None,
                return_metadata=False, progress_callback=None):
        from matchminer_ai.patients.extraction.common import section_settings

        seen["remote"] = dict(config.remote)
        seen["runtime"] = section_settings(config, "document_extraction", "x", None).runtime
        return document_rows(frame), {"schema_version": "mmai-clinical-extraction-v2",
                                      "counts": {"complete": len(frame)}}

    monkeypatch.setattr("matchminer_ai.patients.extract_document_records", extract)
    monkeypatch.setenv("GUIDELINE_EXTRACTION_RECORDS", str(tmp_path / "no_phi/ex/default"))
    monkeypatch.setenv("GUIDELINE_CACHE", str(tmp_path / "no_phi/cache"))
    output = tmp_path / "no_phi/ex/built"
    preprocess_extractions.main(["--notes", str(source), "--output", str(output),
                                 "--server-registry", str(registry)])
    assert seen["remote"]["server_urls_file"] == str(registry.resolve())
    assert seen["remote"]["server_urls"] == ["http://a.invalid:8000/v1", "http://b.invalid:8002/v1"]
    # The registry reaches the extraction's request runtime, which routes through the pool.
    assert seen["runtime"]["server_urls_file"] == str(registry.resolve())
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["extraction"]["model"] == "fabricated/model-NVFP4, fabricated/model-FP8"
    assert [s["upstream_concurrency"] for s in manifest["extraction"]["servers"]] == [128, 24]

    # The pool sends each request with its server's own model name.
    from matchminer_ai.llm.endpoint_pool import EndpointPool

    sent = []

    class Client:
        def __init__(self, base_url, **kwargs):
            self.base_url = base_url
            self.chat = self
            self.completions = self

        def create(self, **request):
            sent.append((self.base_url, request["model"]))
            return {"ok": True}

    pool = EndpointPool(registry=registry, client_factory=Client)
    for _ in range(4):
        pool.request(model="fabricated/model-NVFP4", messages=[])
    assert {model for _, model in sent} == {"fabricated/model-NVFP4", "fabricated/model-FP8"}
    assert all(model.endswith("FP8") == url.startswith("http://b.invalid") for url, model in sent)

    # A server that does not serve the model its row names is refused before any request.
    served["http://b.invalid:8002/v1"] = ["fabricated/other"]
    with pytest.raises(SystemExit, match="not fabricated/model-FP8"):
        preprocess_extractions.main(["--notes", str(source), "--output", str(output),
                                     "--overwrite", "--server-registry", str(registry)])
    with pytest.raises(SystemExit, match="not both"):
        preprocess_extractions.main(["--notes", str(source), "--output", str(output),
                                     "--endpoint", "http://x.invalid/v1",
                                     "--server-registry", str(registry)])


def test_sql_progress_names_each_item_being_explored_and_its_turn():
    from matchminer_ai.apps.guideline_fetcher.extraction_workup import sql_progress_message

    assert sql_progress_message(3, 14, [], 65) == "SQL agent · 3/14 workup queries ready · 1:05 elapsed"
    exploring = [{"name": "DPYD genotyping consideration before fluoropyrimidine therapy",
                  "action": 14, "max_actions": 20},
                 {"name": "Consider MRI of the liver", "action": 6, "max_actions": 20}]
    text = sql_progress_message(11, 14, exploring, 600)
    assert text == ("SQL agent · 11/14 workup queries ready · exploring 2: DPYD genotyping "
                    "consideration… (turn 14/20), Consider MRI of the liver (turn 6/20) "
                    "· 1 waiting · 10:00 elapsed")


# Claude through Databricks for the SQL agent ----------------------------------------

CLAUDE_MODEL = "ai_gateway_model_services_prod.team_project.team_project-claude-sonnet-5-5"


def claude_sql(**values):
    return {"llm_backend": "claude_databricks", "model": CLAUDE_MODEL,
            "endpoint": "https://workspace.example.cloud.databricks.com/ai-gateway/anthropic",
            "tokenizer": "unused", **values}


def test_the_sql_agent_task_has_its_own_connection(tmp_path):
    from matchminer_ai.patients.workup_sql import agent_contract, resolve_agent_llm

    # Unset: the SQL agent uses the concordance checks' connection.
    gemini = Settings(catalog_root=tmp_path, llm_backend="gemini4_vertex")
    assert gemini.for_task("workup_sql").llm_backend == "gemini4_vertex"
    section = gemini.extraction_config().workup_sql_agent
    assert section["remote"]["provider"] == "google_agent_platform"
    # Claude for the SQL agent; extraction and query compilation stay on Gemini.
    claude = Settings(catalog_root=tmp_path, llm_backend="gemini4_vertex",
                      task_connections={"workup_sql": claude_sql()})
    config = claude.for_task("workup").extraction_config()
    assert config.document_extraction["remote"]["model_name"] == "google/gemini-4-hydrogen"
    assert config.remote["provider"] == "google_agent_platform"
    remote = config.workup_sql_agent["remote"]
    assert remote["provider"] == "databricks_anthropic" and remote["model_name"] == CLAUDE_MODEL
    assert remote["server_urls"] == [claude_sql()["endpoint"]]
    assert remote["databricks_profile"] == "unity-gateway"
    llm = resolve_agent_llm(config)
    assert llm.provider == "databricks_anthropic" and llm.base_url == claude_sql()["endpoint"]
    assert llm.request_params == {"reasoning_effort": "high"} and llm.extra_body == {}
    assert llm.context_window == 1_000_000 and llm.google_project_id == ""
    # Each model's queries are cached apart.
    assert agent_contract(config)["llm"]["model"] == CLAUDE_MODEL
    assert agent_contract(gemini.extraction_config())["llm"]["model"] == "google/gemini-4-hydrogen"
    # A local concordance connection with a Gemini SQL agent: each names its own endpoint.
    mixed = Settings(catalog_root=tmp_path, llm_backend="local", endpoint="http://local.invalid/v1",
                     task_connections={"workup_sql": {"llm_backend": "gemini4_vertex"}})
    config = mixed.extraction_config()
    assert config.remote["server_urls"] == ["http://local.invalid/v1"]
    assert resolve_agent_llm(config).provider == "google_agent_platform"
    with pytest.raises(ValueError, match="only the SQL agent"):
        Settings(catalog_root=tmp_path, **claude_sql()).inference_config()


def test_connection_routes_keep_claude_to_the_sql_agent(tmp_path, monkeypatch):
    import matchminer_ai.llm.anthropic_messages as anthropic_messages

    from matchminer_ai.apps.guideline_fetcher.settings import CLAUDE_PROFILE

    checks = []

    def check(base_url, model, *, profile, timeout):
        checks.append((base_url, model, profile))
        return {"available": True, "http_status": 200, "error": None}

    monkeypatch.setattr(anthropic_messages, "check_gateway", check)
    settings = Settings(catalog_root=tmp_path, llm_backend="gemini4_vertex", device="cpu")
    with TestClient(create_app(settings)) as client:
        config = client.get("/api/config").json()
        assert config["claude_profiles"]["claude_databricks"]["task"] == "workup_sql"
        assert config["tasks"]["workup_sql"]["llm_backend"] == "gemini4_vertex"
        for body in (claude_sql(task="workup"), claude_sql(task="workup_sql", apply_to_all=True),
                     claude_sql(task="workup_sql", model="claude-sonnet-5-5")):
            assert client.post("/api/config", json=body).status_code == 422
        # The browser picks only the endpoint name; the server fixes the gateway URL.
        saved = client.post("/api/config", json=claude_sql(
            task="workup_sql", endpoint="http://untrusted.test/v1")).json()
        assert saved["tasks"]["workup_sql"]["endpoint"] == CLAUDE_PROFILE["endpoint"]
        assert saved["tasks"]["workup_sql"]["model"] == CLAUDE_MODEL
        assert saved["tasks"]["workup"]["llm_backend"] == "gemini4_vertex"
        result = client.post("/api/endpoint-check", json={
            "task": "workup_sql", "llm_backend": "claude_databricks",
            "endpoint": "http://untrusted.test/v1", "model": CLAUDE_MODEL}).json()
        assert result["available"] and result["models"] == [CLAUDE_MODEL]
        assert checks == [(CLAUDE_PROFILE["endpoint"], CLAUDE_MODEL, "unity-gateway")]
        bad = client.post("/api/endpoint-check", json={
            "task": "workup_sql", "llm_backend": "claude_databricks", "model": "x"}).json()
        assert not bad["available"] and "three-part" in bad["error"] and len(checks) == 1


def test_population_sql_review_reports_the_sql_agents_connection(population_app, agent,
                                                                 monkeypatch):
    from matchminer_ai.llm.structured import StructuredConfig
    from matchminer_ai.patients.workup_sql import resolve_agent_llm

    seen = []

    def resolve(config):
        seen.append(config)
        return StructuredConfig(base_url="http://fabricated.invalid/v1", model="fabricated-agent",
                                tokenizer_mode="bytes", response_format="json_schema", attempts=1)

    monkeypatch.setattr("matchminer_ai.patients.workup_sql.agent.resolve_agent_llm", resolve)
    api, settings = population_app(False)
    with TestClient(api) as client:
        assert client.post("/api/config", json=claude_sql(task="workup_sql")).status_code == 200
        cohort = active_cohort(client)
        done = job_long(client, "/api/population/review",
                        {"version": cohort["version"], "method": "extraction_sql"})
        report = done["result"]["active"]["report"]
        assert report["connection"]["provider"] == "claude_databricks"
        assert report["connection"]["model"] == CLAUDE_MODEL
    assert resolve_agent_llm(seen[0]).provider == "databricks_anthropic"
