import pandas as pd
import pytest
from fastapi.testclient import TestClient

from matchminer_ai.apps.guideline_fetcher.app import create_app
from .conftest import phi_settings
from matchminer_ai.apps.guideline_fetcher.population import adapt_tables, load_population_sample
from matchminer_ai.apps.guideline_fetcher.services import InputError
from matchminer_ai.apps.guideline_fetcher.settings import Settings
from .test_app import catalog, wait

SPACE = {
    "space_trial_id": "s",
    "name": "Fabricated guideline",
    "clinical_space_summary": "Cancer type allowed: Fabricated cancer.",
    "cohort_key_criteria": "Inclusion criteria:\nCancer type allowed: Fabricated cancer.",
    "diagnostic_workup": [
        {
            "name": "Chest CT",
            "conditions": "If indicated",
            "evidence": [{"quote": "Fabricated source"}],
        }
    ],
}
SUMMARY_CSV = "patient_id,cancer_history_summary\na,Fabricated cancer A\nb,Fabricated cancer B\n"
NOTE_CSV = "patient_id,note_text,note_date,note_type\na,CT completed.,2026-01-01,Imaging\nb,CT planned.,,Visit\n"


def job(client, route, body):
    response = client.post(route, json=body)
    assert response.status_code == 202, response.text
    result = wait(client, response.json()["job_id"]).json()
    assert result["status"] == "complete", result
    return result


def tables(client):
    response = client.post(
        "/api/population/tables",
        files={"summaries": ("summaries.csv", SUMMARY_CSV), "notes": ("notes.csv", NOTE_CSV)},
    )
    assert response.status_code == 202
    loaded = wait(client, response.json()["job_id"]).json()
    assert loaded["status"] == "complete", loaded
    return loaded["result"]


@pytest.fixture
def harness(tmp_path, monkeypatch):
    catalog(tmp_path)
    calls = {"retrieval": [], "compression": [], "review": []}

    def retrieve(summaries, paths, **kwargs):
        calls["retrieval"].append(kwargs)
        frame = summaries.copy()
        frame["space_trial_id"] = "s"
        frame["match_quality_score"] = [0.9, 0.3]
        frame["match_quality_pass"] = True
        frame["similarity_score"] = [0.8, 0.7]
        frame["retrieval_rank"] = [1, 2]
        frame["rank"] = [1, 2]
        return frame, {
            "space": SPACE,
            "scored_count": 2,
            "population_count": 2,
            "min_reasonableness": kwargs["min_reasonableness"],
        }

    def compress(frame, **kwargs):
        calls["compression"].append(kwargs)
        return frame.assign(compressed_note_text=frame.note_text + " Compact.")

    def review(notes, recommendations, **kwargs):
        # One patient's complete selected notes per call (review_population_workup).
        text = notes.note_text.iloc[0]
        calls["review"].append((text, kwargs))
        return {
            "assessments": [
                dict(
                    name=r["name"],
                    recommendation=r,
                    recommendation_index=i,
                    status="completed" if text.startswith("CT completed") else "not_documented",
                    applicability="applies",
                    bottom_line="Fabricated finding",
                    limitations=[],
                    evidence=[
                        {"note_number": 1, "note_date": notes.note_date.iloc[0], "quote": text}
                    ],
                )
                for i, r in enumerate(recommendations)
            ],
            "metadata": {"method": "full_notes"},
            "notice": "Synthetic review",
        }

    monkeypatch.setattr("matchminer_ai.matching.retrieve_guideline_patients", retrieve)
    monkeypatch.setattr("matchminer_ai.patients.compress_patient_notes", compress)
    monkeypatch.setattr("matchminer_ai.patients.review_patient_workup", review)
    monkeypatch.setattr(
        "matchminer_ai.trials.load_guideline_catalog", lambda *a, **kw: pd.DataFrame([SPACE])
    )
    settings = Settings(catalog_root=tmp_path, device="cpu", llm_backend="local")
    return create_app(settings), calls


def selected(client):
    loaded = tables(client)
    catalog_id = client.get("/api/catalogs").json()["catalogs"][0]["id"]
    return job(
        client,
        "/api/population/retrieve",
        dict(
            version=loaded["version"],
            catalog_id=catalog_id,
            space_trial_id="s",
            candidate_k=7,
            min_reasonableness=0.75,
        ),
    )["result"]


def test_full_workflow_restore_drilldown_download_and_isolation(harness):
    api, calls = harness
    with TestClient(api) as client, TestClient(api) as other:
        loaded = selected(client)
        assert loaded["active"]["patient_count"] == 2

        def v():
            return {"version": client.get("/api/population").json()["population"]["version"]}

        assert client.get("/api/session").json()["job"] is None
        assert other.get("/api/population").json()["population"] is None
        assert (
            other.get("/api/population/patient", params={**v(), "patient_id": "a"}).status_code
            == 400
        )
        job(client, "/api/population/compress", v())
        job(
            client,
            "/api/population/review",
            {**v(), "note_source": "compressed", "max_concurrent_patients": 3},
        )
        assert sorted(text for text, _ in calls["review"]) == [
            "CT completed. Compact.", "CT planned. Compact."
        ]
        assert all(
            kw["population_context"] == SPACE["cohort_key_criteria"]
            and kw["backup_config"] is None
            for _, kw in calls["review"]
        )
        current = client.get("/api/population").json()["population"]
        assert current["active"]["report"]["max_concurrent_patients"] == 3
        assert current["active"]["report"]["method"] == "full_notes"
        row = current["active"]["report"]["aggregate"][0]
        assert row["statuses"]["completed"] == 1 and row["statuses"]["not_documented"] == 1
        assert row["documented_completion_fraction"] == 0.5
        item = client.get("/api/population/items/0", params=v()).json()
        assert len(item["patients"]) == 2 and "evidence" not in item["patients"][0]
        patient = client.get("/api/population/patient", params={**v(), "patient_id": "a"}).json()
        assert patient["notes"][0]["note_text"] == "CT completed."
        assert (
            patient["review"]["assessments"][0]["evidence"][0]["quote"] == "CT completed. Compact."
        )
        assert patient["review"]["metadata"]["note_source"] == "compressed"
        assert patient["review"]["metadata"]["evidence_source"] == "llm_compressed_notes"
        for format in ["json", "csv"]:
            response = client.get("/api/population/results", params={**v(), "format": format})
            assert (
                response.status_code == 200
                and "attachment" in response.headers["content-disposition"]
            )
        archive = client.get("/api/population/export", params=v())
        assert archive.status_code == 200
        response = other.post(
            "/api/population/import", files={"file": ("cohort.zip", archive.content)}
        )
        assert response.status_code == 200, response.text
        restored = response.json()["population"]
        assert restored["active"]["space"]["cohort_key_criteria"] == SPACE["cohort_key_criteria"]
        assert restored["active"]["report"]["aggregate"] == current["active"]["report"]["aggregate"]
        job(other, "/api/population/review", {"version": restored["version"]})
        assert sorted(text for text, _ in calls["review"][2:]) == ["CT completed.", "CT planned."]
        assert client.delete("/api/population").status_code == 204
        assert client.get("/api/population").json()["population"] is None
        assert other.get("/api/population").json()["population"] is not None


def test_stale_version_and_unavailable_compressed_notes_rejected(harness):
    api, calls = harness
    with TestClient(api) as client:
        loaded = selected(client)
        old = {"version": loaded["version"]}
        response = client.post("/api/population/review", json={**old, "note_source": "compressed"})
        assert response.status_code == 400 and "Compress the cohort" in response.json()["detail"]
        job(client, "/api/population/compress", old)
        assert client.post("/api/population/review", json=old).status_code == 400
        assert calls["review"] == []
        # ColBERT encoding and its retrieval controls were removed with ColBERT.
        assert client.post("/api/population/encode", json=old).status_code in {404, 405}
        for route, change in [
            ("review", {"top_n": 20}),
            ("review", {"chunk_size": 64}),
            ("review", {"note_source": "summary"}),
            ("review", {"max_concurrent_patients": 0}),
            ("review", {"max_concurrent_patients": 33}),
            ("review", {"max_concurrent_patients": True}),
            ("sample", {"size": 0}),
        ]:
            body = change if route == "sample" else {**old, **change}
            assert client.post("/api/population/" + route, json=body).status_code == 422


def test_clear_workspace_also_clears_population(harness):
    with TestClient(harness[0]) as client:
        selected(client)
        assert client.delete("/api/session").status_code == 204
        assert client.get("/api/population").json()["population"] is None


def test_population_catalog_uses_readable_names_and_rejects_unknown_catalog(harness):
    with TestClient(harness[0]) as client:
        catalog_id = client.get("/api/catalogs").json()["catalogs"][0]["id"]
        response = client.get("/api/population/spaces", params={"catalog_id": catalog_id})
        assert response.status_code == 200
        assert response.json()["spaces"][0]["name"] == SPACE["name"]
        assert response.json()["spaces"][0]["cohort_key_criteria"] == SPACE["cohort_key_criteria"]
        assert (
            client.get("/api/population/spaces", params={"catalog_id": "unknown"}).status_code
            == 400
        )


def test_training_ids_join_on_pseudo_mrn_and_sample_is_reproducible(tmp_path):
    summary = pd.DataFrame(
        {
            "pseudo_mrn": ["1", "2", "3"],
            "patient_summary": ["Fabricated A", "Fabricated B", "Fabricated C"],
        }
    )
    notes = pd.DataFrame(
        {
            "pseudo_mrn": [1, 2, 3],
            "patient_id": ["prompt-a", "prompt-b", "prompt-c"],
            "synthetic_note": ["Original A", "Original B", "Original C"],
            "date": ["2026-01-01"] * 3,
            "event_type": ["Visit"] * 3,
        }
    )
    a, b = adapt_tables(summary, notes)
    assert b.patient_id.tolist() == [1, 2, 3]
    summary.to_parquet(tmp_path / "summaries.parquet")
    notes.to_parquet(tmp_path / "notes.parquet")
    settings = Settings(
        device="cpu",
        population_summaries=tmp_path / "summaries.parquet",
        population_notes=tmp_path / "notes.parquet",
    )
    c = load_population_sample(settings, 2, 23, lambda _: None)
    d = load_population_sample(settings, 2, 23, lambda _: None)
    assert c.summaries.patient_id.tolist() == d.summaries.patient_id.tolist()
    assert len(c.summaries) == 2 and set(c.notes.patient_id) == set(c.summaries.patient_id)


def fabricated_database(path):
    """A tiny fabricated copy of the real-time MatchMiner-AI schema; no real records."""
    import duckdb

    con = duckdb.connect(str(path))
    con.execute(
        "CREATE TABLE patient_summary (mrn BIGINT, patient_summary VARCHAR, "
        "patient_boilerplate VARCHAR, model_name VARCHAR, is_current BOOLEAN)"
    )
    con.execute(
        "INSERT INTO patient_summary VALUES "
        "(101, 'Old fabricated summary', 'Old', 'm', false),"
        "(101, '  Fabricated lung cancer  ', 'No exclusions', 'm', true),"
        "(102, 'Fabricated colon cancer', NULL, 'm', true),"
        "(103, 'Fabricated breast cancer', '', 'm', true),"
        "(104, 'Fabricated noncurrent only', '', 'm', false),"
        "(105, 'Fabricated without notes', '', 'm', true),"
        "(106, chr(10) || chr(9) || chr(10), '', 'm', true)"
    )
    con.execute(
        "CREATE TABLE patient_notes (id BIGINT, mrn BIGINT, rpt_text VARCHAR, "
        "narrative_text VARCHAR, note_source VARCHAR, event_date TIMESTAMP)"
    )
    con.execute(
        "INSERT INTO patient_notes VALUES "
        "(1, 101, 'CT chest completed.', 'CT', 'imaging_report', '2026-02-01 09:00'),"
        "(2, 101, '   ', 'Narrative only visit.', 'clinical_note', '2026-01-01 08:00'),"
        "(3, 101, 'CT chest completed.', 'CT', 'imaging_report', '2026-02-01 09:00'),"
        "(5, 101, 'Same-time pathology B.', '', 'pathology_report', '2026-03-01'),"
        "(4, 101, 'Same-time pathology A.', '', 'pathology_report', '2026-03-01'),"
        "(6, 102, 'Colonoscopy completed.', '', 'clinical_note', NULL),"
        "(7, 103, chr(10) || chr(9), chr(13) || chr(10) || ' ', 'clinical_note', '2026-01-05'),"
        "(8, 104, 'Noncurrent patient note.', '', 'clinical_note', '2026-01-05'),"
        "(9, 101, chr(10) || chr(10), chr(10) || 'Newline fallback.' || chr(10), "
        "'clinical_note', '2026-01-02'),"
        "(10, 106, 'Visit for a blank summary.', '', 'clinical_note', '2026-01-03')"
    )
    con.close()
    return path


@pytest.fixture
def database(tmp_path):
    return fabricated_database(tmp_path / "oa_fabricated.duckdb")


def database_settings(tmp_path, database, **values):
    return phi_settings(device="cpu", population_duckdb=database, duckdb_roots=[tmp_path], **values)


def test_duckdb_sample_reads_current_summaries_and_original_notes(tmp_path, database):
    settings = database_settings(tmp_path, database)
    before = database.stat().st_mtime_ns
    messages = []
    cohort = load_population_sample(settings, 4, 23, messages.append)
    again = load_population_sample(settings, 4, 23, lambda _: None)
    assert cohort.summaries.patient_id.tolist() == again.summaries.patient_id.tolist()
    # Blank by Python's strip (tabs/newlines too): 103's notes and 106's summary.
    assert set(cohort.summaries.patient_id) == {"101", "102"}
    summary = cohort.summaries.set_index("patient_id")
    assert summary.loc["101", "cancer_history_summary"] == "Fabricated lung cancer"
    assert summary.loc["101", "general_exclusion_criteria_evidence"] == "No exclusions"
    assert summary.loc["102", "general_exclusion_criteria_evidence"] == ""
    notes = cohort.notes.loc[cohort.notes.patient_id == "101"]
    assert notes.note_text.tolist() == [
        "Narrative only visit.", "Newline fallback.", "CT chest completed.",
        "Same-time pathology A.", "Same-time pathology B.",
    ]
    assert notes.note_type.tolist() == [
        "clinical_note", "clinical_note", "imaging_report", "pathology_report",
        "pathology_report",
    ]
    assert notes.note_date.iloc[0].startswith("2026-01-01T08:00:00")
    assert cohort.notes.loc[cohort.notes.patient_id == "102", "note_date"].tolist() == [None]
    provenance = cohort.provenance
    assert provenance["source"] == "duckdb_population"
    assert provenance["database"]["path"] == str(database.resolve())
    assert provenance["eligible_population_count"] == 4
    assert provenance["excluded_without_text_count"] == 2
    assert provenance["duplicate_notes_removed"] == 1
    assert database.stat().st_mtime_ns == before
    assert messages[-1] == "Loaded 2 patients and 6 notes"


def test_duckdb_selection_is_validated(tmp_path, database):
    import duckdb

    settings = database_settings(tmp_path, database)
    with pytest.raises(InputError, match="Only 4 patients"):
        load_population_sample(settings, 5, 23, lambda _: None)
    outside = tmp_path.parent / f"{tmp_path.name}-outside.duckdb"
    fabricated_database(outside)
    try:
        with pytest.raises(InputError, match="configured database folders"):
            load_population_sample(settings, 1, 23, lambda _: None, database=str(outside))
    finally:
        outside.unlink()
    with pytest.raises(InputError, match="absolute path"):
        load_population_sample(settings, 1, 23, lambda _: None, database="oa.duckdb")
    (tmp_path / "notes.csv").write_text("not a database")
    with pytest.raises(InputError, match="existing .duckdb"):
        load_population_sample(settings, 1, 23, lambda _: None, database=str(tmp_path / "notes.csv"))
    (tmp_path / "corrupt.duckdb").write_text("not a database")
    with pytest.raises(InputError, match="read-only DuckDB"):
        load_population_sample(settings, 1, 23, lambda _: None, database=str(tmp_path / "corrupt.duckdb"))
    other = tmp_path / "other.duckdb"
    con = duckdb.connect(str(other))
    con.execute("CREATE TABLE patient_summary (mrn BIGINT, patient_summary VARCHAR)")
    con.close()
    with pytest.raises(InputError, match="patient_notes.mrn"):
        load_population_sample(settings, 1, 23, lambda _: None, database=str(other))


def test_database_browse_lists_only_folders_and_duckdb_files(tmp_path, database):
    from matchminer_ai.apps.guideline_fetcher.population import browse_databases

    (tmp_path / "nested").mkdir()
    (tmp_path / ".hidden").mkdir()
    (tmp_path / "notes.parquet").write_text("x")
    settings = database_settings(tmp_path, database)
    listing = browse_databases(settings)
    assert listing["path"] == str(tmp_path.resolve())
    assert listing["parent"] is None
    assert listing["directories"] == ["nested"]
    assert [f["name"] for f in listing["files"]] == ["oa_fabricated.duckdb"]
    nested = browse_databases(settings, str(tmp_path / "nested"))
    assert nested["parent"] == str(tmp_path.resolve()) and nested["files"] == []
    assert browse_databases(settings, str(database))["path"] == str(tmp_path.resolve())
    for path in (str(tmp_path.parent), str(tmp_path / "nested" / ".." / "..")):
        with pytest.raises(InputError, match="configured database folders"):
            browse_databases(settings, path)
    # A default inside a broader root can still browse up to that root.
    (tmp_path / "nested" / "copy.duckdb").write_text("x")
    inner = database_settings(tmp_path, tmp_path / "nested" / "copy.duckdb")
    assert browse_databases(inner)["parent"] == str(tmp_path.resolve())
    # A default outside the roots stays selectable.
    outer = phi_settings(device="cpu", population_duckdb=database,
                         duckdb_roots=[tmp_path / "nested"])
    assert browse_databases(outer)["path"] == str(tmp_path.resolve())


def test_population_database_routes(tmp_path, database, monkeypatch):
    catalog(tmp_path / "catalogs")
    api = create_app(database_settings(tmp_path, database, catalog_root=tmp_path / "catalogs",
                                       llm_backend="local"))
    with TestClient(api) as client:
        state = client.get("/api/population").json()
        assert state["defaults"]["database"] == str(database)
        listing = client.get("/api/population/databases").json()
        assert [f["path"] for f in listing["files"]] == [str(database.resolve())]
        checked = client.get("/api/population/database", params={"path": str(database)})
        assert checked.json()["name"] == "oa_fabricated.duckdb"
        assert client.get("/api/population/database",
                          params={"path": "/etc/passwd"}).status_code == 400
        rejected = client.post("/api/population/sample",
                               json={"size": 1, "seed": 1, "database": "/etc/hosts"})
        assert rejected.status_code == 400
        loaded = job(client, "/api/population/sample",
                     {"size": 4, "seed": 7, "database": str(database)})["result"]
        assert loaded["source_count"] == 2
        assert loaded["provenance"]["database"]["name"] == "oa_fabricated.duckdb"
        assert client.get("/api/population").json()["population"]["version"] == loaded["version"]


def test_non_phi_app_never_opens_a_duckdb_database(tmp_path, database, monkeypatch):
    import duckdb

    from matchminer_ai.apps.guideline_fetcher.population import browse_databases, inspect_database
    from matchminer_ai.apps.guideline_fetcher.population_cache import inspect_cache

    def refuse(*args, **kwargs):
        raise AssertionError("The non-PHI app opened a DuckDB database")

    monkeypatch.setattr(duckdb, "connect", refuse)
    settings = Settings(device="cpu", population_duckdb=database, duckdb_roots=[tmp_path],
                        population_cache=tmp_path)
    for call in (
        lambda: load_population_sample(settings, 1, 23, lambda _: None, database=str(database)),
        lambda: inspect_database(settings, str(database)),
        lambda: browse_databases(settings),
        lambda: inspect_cache(settings, str(tmp_path)),
    ):
        with pytest.raises(InputError, match="PHI app"):
            call()




def test_task_specific_model_connections(harness):
    api, calls = harness
    with TestClient(api) as client:
        for task in ["compress", "workup", "workup_full_backup"]:
            response = client.post(
                "/api/config",
                json={
                    "task": task,
                    "llm_backend": "local",
                    "endpoint": f'http://{task.replace("_","-")}:8000/v1',
                    "model": "Qwen/Qwen3.5-4B",
                    "tokenizer": "Qwen/Qwen3.5-4B",
                    "backup_enabled": task.endswith("backup"),
                    "failure_threshold": 4,
                },
            )
            assert response.status_code == 200
        selected(client)

        def v():
            return {"version": client.get("/api/population").json()["population"]["version"]}

        job(client, "/api/population/compress", v())
        job(client, "/api/population/review", v())
        assert calls["compression"][0]["config"].remote["server_urls"] == [
            "http://compress:8000/v1"
        ]
        options = calls["review"][0][1]
        assert options["config"].remote["server_urls"] == ["http://workup:8000/v1"]
        assert options["backup_config"].remote["server_urls"] == [
            "http://workup-full-backup:8000/v1"
        ]
        assert options["max_consecutive_failures"] == 4


def test_stop_interrupts_a_long_database_query():
    import threading
    import time

    import duckdb

    from matchminer_ai.cancellation import (
        CancellationToken, InferenceCancelled, cancellation_scope,
    )
    from matchminer_ai.apps.guideline_fetcher.population import _interruptible

    con = duckdb.connect()
    token, messages = CancellationToken(), []
    threading.Timer(1.5, token.cancel).start()
    started = time.monotonic()
    with cancellation_scope(token), pytest.raises(InferenceCancelled):
        _interruptible(con, lambda: con.execute(
            "SELECT count(DISTINCT md5(range::VARCHAR)) FROM range(CAST(2e9 AS BIGINT))"
        ).fetchall(), lambda m: (token.check(), messages.append(m)), "Reading")
    assert time.monotonic() - started < 15
    assert messages and messages[0].startswith("Reading")
    assert con.execute("SELECT 42").fetchone() == (42,)
