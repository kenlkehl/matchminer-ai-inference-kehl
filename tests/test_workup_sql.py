"""SQL-agent workup checks: database, isolated worker, agent loop, cohort evaluation, preset.

Every record, note and requirement here is fabricated; no clinical data is used.
The agent's LLM is scripted; SQL runs in the real isolated DuckDB worker.
"""

import json
import os
import stat
import subprocess
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("duckdb")

from matchminer_ai.config import load_default_preset  # noqa: E402
from matchminer_ai.llm.structured import StructuredConfig  # noqa: E402
from matchminer_ai.patients import note_search_qa as qa  # noqa: E402
from matchminer_ai.patients.extraction.notes import note_ids  # noqa: E402
from matchminer_ai.patients.workup_sql import (  # noqa: E402
    SQLSandbox,
    WorkupSQLLimits,
    agent_limits,
    build_extraction_database,
    check_workup_with_sql,
    compile_workup_sql,
    read_database_info,
    resolve_agent_llm,
    run_final_query,
    write_extraction_database,
)
from matchminer_ai.patients.workup_sql import agent as agent_module  # noqa: E402
from matchminer_ai.population import (  # noqa: E402
    check_population_workup_with_sql,
    create_patient_cohort,
    load_patient_cohort,
    population_results_table,
    save_patient_cohort,
)
from test_workup_queries import biomarker, diagnosis, document, study, when  # noqa: E402

PID = "fabricated-patient-7f3a"

CT_SQL = """
SELECT p.patient_key,
  CASE WHEN m.k IS NOT NULL THEN 'met' ELSE 'not_met' END AS status,
  'applies' AS applicability,
  CASE WHEN m.k IS NOT NULL THEN 'documented'
       WHEN pl.k IS NOT NULL THEN 'planned_only' ELSE 'not_documented' END AS reason,
  COALESCE(m.k, pl.k, []::BIGINT[]) AS evidence_item_keys,
  COALESCE(m.d, '') AS detail
FROM patients p
LEFT JOIN (SELECT patient_key, list(item_key ORDER BY item_key) AS k, string_agg(label, '; ') AS d
           FROM imaging_studies WHERE modality = 'ct' AND list_contains(body_regions, 'chest')
           AND occurrence = 'performed' GROUP BY patient_key) m USING (patient_key)
LEFT JOIN (SELECT patient_key, list(item_key) AS k FROM imaging_studies
           WHERE modality = 'ct' AND occurrence = 'planned' GROUP BY patient_key) pl
  USING (patient_key)
ORDER BY p.patient_key;
"""
REQUIREMENT = {"name": "CT chest", "conditions": "For staging", "category": "Category 2A"}


def rows():
    """Four fabricated patients: CT done, CT only planned, extraction failed, no records."""
    return pd.DataFrame([
        document("n1", "2024-02-01", [study("n1", "2024-02-01"), diagnosis("n1", "2024-02-01")],
                 patient=PID),
        document("n2", "2024-02-03", [study("n2", "2024-02-03", relation="previously_performed",
                                            date=when("2024-01"))], patient=PID),
        document("n3", "2024-02-05", [study("n3", "2024-02-05", relation="planned")],
                 patient="p2"),
        document("n4", "2024-02-07", [biomarker("n4", "2024-02-07", "EGFR")], patient="p3"),
        document("n5", "2024-02-09", [], patient="p3", status="failed"),
    ])


@pytest.fixture
def database():
    return build_extraction_database(rows(), patient_ids=[PID, "p2", "p3", "p4"])


@pytest.fixture
def llm():
    return StructuredConfig(base_url="http://fabricated.invalid/v1", model="fabricated-model",
                            max_tokens=2048, context_window=262144, tokenizer_mode="bytes",
                            response_format="json_schema", attempts=1, timeout=5)


def action(kind, sql="", **values):
    return {"action": kind, "purpose": values.get("purpose", kind), "sql": sql,
            "summary": values.get("summary", ""), "approximations": values.get("approximations", []),
            "not_expressible_reason": values.get("reason", ""),
            "recommends_against": values.get("against", False), "memory": values.get("memory", "")}


def script(monkeypatch, actions):
    """Scripted LLM answers; returns the (payload, schema) of every request."""
    seen, queue = [], list(actions)

    def complete(self, job, messages, schema, validator):
        payload = json.loads(messages[1]["content"])
        seen.append((payload, schema, messages[0]["content"]))
        self.requests += 1
        value = queue.pop(0)(payload) if callable(queue[0]) else queue.pop(0)
        return validator(value)

    monkeypatch.setattr(qa._MeasuredClient, "complete", complete)
    return seen


# Database --------------------------------------------------------------------


def test_database_tables_dates_and_privacy(database):
    tables = database.tables
    assert tables["patients"].num_rows == 4
    patients = tables["patients"].to_pylist()
    assert [p["patient_key"] for p in patients] == [1, 2, 3, 4]
    assert patients[2]["extraction_incomplete"] and patients[2]["failed_document_records"] == 1
    assert patients[3]["item_count"] == 0 and patients[3]["reference_date"] is None
    assert patients[0]["reference_date"] == date(2024, 2, 3)
    studies = {r["note_id"]: r for r in tables["imaging_studies"].to_pylist()}
    assert studies["n1"]["date_lo"] == studies["n1"]["date_hi"] == date(2024, 2, 1)
    # A month-precision date spans its month; an undated planned study is on or after its note.
    assert (studies["n2"]["date_lo"], studies["n2"]["date_hi"]) == (date(2024, 1, 1),
                                                                    date(2024, 1, 31))
    assert studies["n3"]["occurrence"] == "planned" and studies["n3"]["date_bound"] == "on_or_after"
    assert studies["n3"]["date_hi"] is None and studies["n3"]["date_lo"] == date(2024, 2, 5)
    assert studies["n1"]["body_regions"] == ["chest"] and studies["n1"]["tracer"] is None
    # No patient identifier appears in any table; quotes stay in Python.
    for table in tables.values():
        for row in table.to_pylist():
            assert PID not in json.dumps(row, default=str)
    assert database.views[studies["n1"]["item_key"]].item["evidence"][0]["note_id"] == "n1"
    # Columns come from the declarations: the same with no records at all.
    empty = build_extraction_database([], patient_ids=["x"])
    assert {n: t.schema for n, t in empty.tables.items() if n != "patients"}.keys() == \
        {n for n in tables if n != "patients"}
    for name, table in tables.items():
        assert empty.tables[name].schema == table.schema, name


def test_write_and_read_database(tmp_path, database):
    path = write_extraction_database(tmp_path / "x" / "records.duckdb", database, private=True)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    info = read_database_info(path)
    assert info["fingerprint"] == database.fingerprint and info["patients"] == "4"
    assert not list(path.parent.glob(".*partial*"))
    assert build_extraction_database(rows(), patient_ids=[PID, "p2", "p3", "p4"]).fingerprint \
        == database.fingerprint


# Isolated worker -------------------------------------------------------------


def test_sandbox_allows_only_reads(database, tmp_path):
    target = tmp_path / "leak.csv"
    with SQLSandbox(database=database) as box:
        ok = box.query("SELECT modality, count(*) AS n FROM imaging_studies GROUP BY 1",
                       max_rows=5)
        assert ok["error"] is None and ok["rows"] == [["ct", 3]]
        for sql in (f"COPY items TO '{target}'", "SELECT 1; SELECT 2", "ATTACH 'x.db'",
                    "SET threads = 4", "INSTALL httpfs", "CREATE TABLE t (a INT)",
                    "SELECT * FROM read_csv('/etc/passwd')", "SELECT * FROM glob('/etc/*')",
                    "SELECT * FROM 'https://fabricated.invalid/a.csv'"):
            assert box.query(sql, max_rows=5)["error"], sql
        assert box.query("SUMMARIZE imaging_studies", max_rows=50)["error"] is None
        cut = box.query("SELECT repeat('x', 500) AS t FROM range(10)", max_rows=3,
                        max_cell_chars=20)
        assert cut["row_count"] == 10 and len(cut["rows"]) == 3 and len(cut["rows"][0][0]) == 21
        slow = box.query("SELECT count(*) FROM range(1000000000000)", max_rows=1, timeout=0.5)
        assert slow["error"] == "InterruptException"
        assert box.query("SELECT 42", max_rows=1)["rows"] == [[42]]
    assert not target.exists()


def test_sandbox_reads_database_file(tmp_path, database):
    path = write_extraction_database(tmp_path / "records.duckdb", database)
    with SQLSandbox(database_path=path) as box:
        assert box.info["fingerprint"] == database.fingerprint
        assert box.query("SELECT count(*) FROM items", max_rows=1)["rows"] == [[5]]


def test_worker_isolation_blocks_open_socket_and_processes():
    worker = Path(agent_module.__file__).with_name("_sql_worker.py")
    import duckdb
    import pyarrow

    paths = [str(Path(m.__file__).resolve().parent.parent) for m in (duckdb, pyarrow)]
    code = (
        "import importlib.util, json, os, socket, sys\n"
        f"sys.argv = ['w', {json.dumps(json.dumps(paths))}]\n"
        f"spec = importlib.util.spec_from_file_location('w', {str(worker)!r})\n"
        "w = importlib.util.module_from_spec(spec); spec.loader.exec_module(w)\n"
        "w.isolate(512)\n"
        "out = []\n"
        "for name, fn in [('open', lambda: open('/etc/hostname').read()),\n"
        "                 ('socket', lambda: socket.socket()), ('fork', os.fork)]:\n"
        "    try:\n        fn(); out.append(name + ' allowed')\n"
        "    except OSError:\n        out.append(name + ' blocked')\n"
        "sys.stdout.write(json.dumps(out)); sys.stdout.flush()\n"
    )
    result = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True,
                            timeout=60)
    assert json.loads(result.stdout) == ["open blocked", "socket blocked", "fork blocked"]


# Final query contract --------------------------------------------------------


def test_final_query_contract(database):
    limits = WorkupSQLLimits()
    with SQLSandbox(database=database) as box:
        good = run_final_query(box, CT_SQL, 4, limits)
        assert good.problems == []
        assert good.rows[1]["status"] == "met" and len(good.rows[1]["evidence_item_keys"]) == 2
        assert good.rows[2]["reason"] == "planned_only"
        assert good.distribution["status_reason"] == {"not_met/not_documented": 2,
                                                      "met/documented": 1,
                                                      "not_met/planned_only": 1}
        missing = run_final_query(box, CT_SQL.replace("ORDER BY p.patient_key", "LIMIT 2"), 4, limits)
        assert any("2 patients have no row" in p for p in missing.problems)
        wrong = run_final_query(box, CT_SQL.replace("'documented'", "'not_documented'"), 4,
                                limits)
        assert any("reason does not fit" in p for p in wrong.problems)
        foreign = run_final_query(box, CT_SQL.replace("COALESCE(m.k, pl.k, []::BIGINT[])",
                                                      "[1::BIGINT]"), 4, limits)
        assert any("not item_key values of that row's patient" in p for p in foreign.problems)
        columns = run_final_query(box, "SELECT patient_key FROM patients", 4, limits)
        assert columns.problems[0].startswith("Return exactly the columns")
        broken = run_final_query(box, "SELECT nope FROM patients", 4, limits)
        assert broken.problems[0].startswith("The query failed")


# Agent loop ------------------------------------------------------------------


def test_agent_explores_revises_accepts_and_caches(monkeypatch, database, llm):
    bad = CT_SQL.replace("ORDER BY p.patient_key", "LIMIT 1")
    seen = script(monkeypatch, [
        action("query", "SELECT modality, count(*) FROM imaging_studies GROUP BY 1",
               memory="CT is coded ct."),
        action("submit", bad, summary="A performed chest CT."),
        lambda payload: action("submit", CT_SQL, summary="A performed chest CT.",
                               approximations=["Contrast is not checked."]),
        action("accept"),
    ])
    cache = {}
    entries, metadata = compile_workup_sql([REQUIREMENT, dict(REQUIREMENT)], database=database,
                                           population_context="Fabricated NSCLC population",
                                           cache=cache, llm=llm, return_metadata=True)
    assert len(seen) == 4 and metadata["distinct"] == 1 and metadata["requests"] == 4
    entry = entries[0]
    assert entry["status"] == "compiled" and entry["accepted_by"] == "agent"
    assert entry["sql"] == CT_SQL.strip().rstrip(";") and entry["termination"] == "accepted"
    assert entry["approximations"] == ["Contrast is not checked."]
    assert entry["distribution"]["status_reason"]["met/documented"] == 1
    assert [s["outcome"] for s in entry["steps"]] == [
        "1 row", "invalid: patients without a row", "valid"]
    assert entries[1]["cache_key"] == entry["cache_key"]
    first, second, third, fourth = (p for p, _, _ in seen)
    assert "accept" not in first["allowed_actions"] and first["memory"] == ""
    assert second["memory"] == "CT is coded ct."
    assert "ct" in second["recent_steps"][0]["result"]
    assert third["recent_steps"][-1]["result"].startswith("Not valid")
    assert fourth["pending_submission"]["summary"] == "A performed chest CT."
    assert "accept" in fourth["allowed_actions"]
    assert all(PID not in json.dumps(p) for p, _, _ in seen)
    assert "THE DATABASE" in seen[0][2] and "imaging_studies" in seen[0][2]
    assert set(cache) == {entry["cache_key"]} and "cached" not in cache[entry["cache_key"]]
    # Cached: no further model calls.
    again = compile_workup_sql([REQUIREMENT], database=database,
                               population_context="Fabricated NSCLC population",
                               cache=cache, llm=llm)
    assert again[0]["cached"] and again[0]["sql"] == entry["sql"] and len(seen) == 4


def test_agent_not_expressible_and_budget(monkeypatch, database, llm, tmp_path):
    script(monkeypatch, [action("not_expressible", reason="A laboratory panel.")])
    path = write_extraction_database(tmp_path / "db.duckdb", database)
    [entry] = compile_workup_sql([{"name": "CBC"}], database=path, llm=llm)
    assert entry["status"] == "compiled" and entry["expressible"] is False
    assert entry["not_expressible_reason"] == "A laboratory panel."
    config = load_default_preset()
    config.workup_sql_agent["limits"] = {"max_actions": 3, "max_submits": 2}
    # Accepted by the agent within its budget.
    script(monkeypatch, [action("query", "SELECT 1"), action("submit", CT_SQL, summary="CT."),
                         action("accept")])
    [agent] = compile_workup_sql([{"name": "CT"}], database=database, llm=llm, config=config)
    assert agent["status"] == "compiled" and agent["accepted_by"] == "agent"
    # A valid submission on the last turn is used, with a limitation.
    seen = script(monkeypatch, [action("query", "SELECT 1"), action("query", "SELECT 2"),
                                action("submit", CT_SQL, summary="CT.")])
    [last] = compile_workup_sql([{"name": "CT", "conditions": "x"}], database=database, llm=llm,
                                config=config)
    assert seen[-1][0]["allowed_actions"] == ["submit", "not_expressible"]
    assert last["accepted_by"] == "budget" and last["termination"] == "budget_accepted"
    assert "action budget" in last["approximations"][-1]
    # No valid submission within the budget: failed. The failure is saved, with its
    # steps and diagnostics, but never reused.
    cache = {}
    script(monkeypatch, [action("query", "SELECT 1"), action("submit", "SELECT 1", summary="x"),
                         action("submit", "SELECT nope FROM patients", summary="y")])
    [failed] = compile_workup_sql([{"name": "MRI"}], database=database, llm=llm, config=config,
                                  cache=cache)
    assert failed["status"] == "failed" and failed["termination"] == "budget_exhausted"
    assert failed["sql"] is None
    assert [s["outcome"] for s in failed["steps"]] == [
        "1 row", "invalid: wrong columns", "invalid: query error BinderException"]
    assert "the last invalid: query error BinderException" in failed["error"]
    assert set(failed["diagnostics"]) == {"invalid_answers", "finish_reasons",
                                          "transport_errors"}
    saved = cache[failed["cache_key"]]
    assert saved["status"] == "failed" and saved["steps"] == failed["steps"]
    assert "nope" not in json.dumps(saved["steps"][-1]["outcome"])
    assert not agent_module.reusable_sql_entry(saved, saved["contract"])
    seen = script(monkeypatch, [action("query", "SELECT 1"),
                                action("submit", CT_SQL, summary="MRI."), action("accept")])
    [retried] = compile_workup_sql([{"name": "MRI"}], database=database, llm=llm,
                                   config=config, cache=cache)
    assert len(seen) == 3 and retried["status"] == "compiled" and not retried["cached"]
    assert cache[failed["cache_key"]]["status"] == "compiled"


def test_checking_queries_after_a_valid_submission_are_capped(monkeypatch, database, llm):
    """After a valid submission the agent may check it a few times, then must decide."""
    config = load_default_preset()
    config.workup_sql_agent["limits"] = {"max_actions": 12, "max_review_queries": 2}
    seen = script(monkeypatch, [
        action("submit", CT_SQL, summary="CT."),
        action("query", "SELECT 1"), action("query", "SELECT 2"),
        # No query allowed now: a revision restores two checking queries.
        action("submit", CT_SQL, summary="CT, revised."),
        action("query", "SELECT 3"),
        action("accept"),
    ])
    [entry] = compile_workup_sql([REQUIREMENT], database=database, llm=llm, config=config)
    allowed = [p["allowed_actions"] for p, _, _ in seen]
    assert "query" in allowed[1] and "query" in allowed[2]
    assert allowed[3] == ["submit", "accept", "not_expressible"]
    assert [p["checking_queries_left"] for p, _, _ in seen] == [None, 2, 1, 0, 2, 1]
    assert "no checking queries are left" in seen[3][0]["next_step"]
    assert entry["status"] == "compiled" and entry["summary"] == "CT, revised."
    assert entry["action_count"] == 6
    assert "2 checking queries" in seen[1][0]["recent_steps"][-1]["result"]


def test_budget_end_falls_back_to_the_last_valid_submission(monkeypatch, database, llm):
    config = load_default_preset()
    config.workup_sql_agent["limits"] = {"max_actions": 3, "max_submits": 3}
    script(monkeypatch, [action("submit", CT_SQL, summary="CT."),
                         action("submit", "SELECT 1", summary="Broken revision."),
                         action("submit", "SELECT 2", summary="Broken again.")])
    [entry] = compile_workup_sql([REQUIREMENT], database=database, llm=llm, config=config)
    assert entry["status"] == "compiled" and entry["termination"] == "budget_accepted"
    assert entry["summary"] == "CT." and entry["accepted_by"] == "budget"
    assert any("later revision was not valid" in a for a in entry["approximations"])


def test_agent_fails_closed_on_endpoint_errors(monkeypatch, database, llm):
    def complete(self, job, messages, schema, validator):
        raise agent_module.EndpointError("fabricated outage")

    monkeypatch.setattr(qa._MeasuredClient, "complete", complete)
    cache = {}
    [entry] = compile_workup_sql([REQUIREMENT], database=database, llm=llm, cache=cache)
    assert entry["status"] == "failed" and entry["termination"] == "endpoint_failure"
    assert "fabricated outage" not in json.dumps(entry)
    assert "fabricated outage" not in json.dumps(cache)
    assert cache[entry["cache_key"]]["status"] == "failed"


# Cohort evaluation -----------------------------------------------------------


def compiled_entries(*entries):
    base = {"status": "compiled", "expressible": True, "summary": "A performed chest CT.",
            "approximations": ["Contrast is not checked."], "cache_key": "k"}
    return [{**base, "requirement": {"name": f"Item {n}", "conditions": "", "category": ""},
             **entry} for n, entry in enumerate(entries)]


def test_check_workup_with_sql_runs_once_over_the_cohort(monkeypatch):
    calls = []
    original = SQLSandbox.query

    def counted(self, sql, **kwargs):
        calls.append(sql)
        return original(self, sql, **kwargs)

    monkeypatch.setattr(SQLSandbox, "query", counted)
    entries = compiled_entries(
        {"sql": CT_SQL},
        {"expressible": False, "sql": None, "not_expressible_reason": "A laboratory panel."},
        {"status": "failed", "sql": None},
        {"sql": "SELECT nope FROM patients"},
    )
    report = check_workup_with_sql(rows(), entries, patient_ids=[PID, "p2", "p3", "p4"])
    # One full run plus one ownership check per runnable query, regardless of cohort size.
    assert len(calls) == 3
    by_patient = {p["patient_id"]: p["review"]["assessments"] for p in report["patients"]}
    met = by_patient[PID][0]
    assert met["status"] == "met" and met["reason"] == "documented"
    assert met["bottom_line"].startswith("Met: CT (chest)")
    assert {e["note_id"] for e in met["evidence"]} == {"n1", "n2"}
    assert met["evidence"][0]["state"] == "yes" and met["sql"] == CT_SQL
    assert met["limitations"] == ["Contrast is not checked."]
    assert met["reference_date"] == "2024-02-03"
    planned = by_patient["p2"][0]
    assert (planned["status"], planned["reason"]) == ("not_met", "planned_only")
    assert planned["evidence"][0]["state"] == "planned"
    incomplete = by_patient["p3"][0]
    assert (incomplete["status"], incomplete["reason"]) == ("unknown", "incomplete_extraction")
    assert "failed extraction" in incomplete["bottom_line"]
    none = by_patient["p4"][0]
    assert (none["status"], none["reason"]) == ("unknown", "no_extraction")
    assert by_patient[PID][1]["reason"] == "not_expressible"
    assert by_patient[PID][2]["reason"] == "query_unavailable"
    assert by_patient[PID][3]["reason"] == "query_unavailable"
    assert "failed on this cohort" in by_patient[PID][3]["bottom_line"]
    assert report["queries"][3] == {"status": "failed", "problem": "error"}
    assert report["aggregate"][0]["statuses"] == {"met": 1, "not_met": 1, "unknown": 2,
                                                  "not_applicable": 0, "done_not_recommended": 0}
    assert report["patients"][0]["review"]["metadata"]["method"] == "extraction_sql"


def test_population_sql_check_and_archive(tmp_path):
    notes = pd.DataFrame([
        {"patient_id": PID, "note_text": "Fabricated CT chest report.", "note_date": "2024-02-01",
         "note_type": "imaging"},
        {"patient_id": "p2", "note_text": "Fabricated plan: CT chest.", "note_date": "2024-02-05",
         "note_type": "clinic"},
    ])
    summaries = pd.DataFrame([{"patient_id": PID, "cancer_history_summary": "Fabricated."},
                              {"patient_id": "p2", "cancer_history_summary": "Fabricated."}])
    cohort = create_patient_cohort(summaries, notes)
    ids = note_ids(notes)
    records = pd.DataFrame([
        document(ids[0], "2024-02-01", [study(ids[0], "2024-02-01")], patient=PID),
        document(ids[1], "2024-02-05", [study(ids[1], "2024-02-05", relation="planned")],
                 patient="p2"),
        document("note-other", "2024-02-06", [study("note-other", "2024-02-06")], patient="p2"),
    ])
    space = {"space_trial_id": "fabricated-space", "diagnostic_workup": [REQUIREMENT]}
    from dataclasses import replace

    cohort = replace(cohort, space=space)
    checked = check_population_workup_with_sql(cohort, compiled_entries({"sql": CT_SQL}),
                                               document_records=records)
    report = checked.report
    assert report["method"] == "extraction_sql" and report["extraction"]["documents_ignored"] == 1
    statuses = [p["review"]["assessments"][0]["status"] for p in report["patients"]]
    assert statuses == ["met", "not_met"]
    assert report["queries"][0]["sql"] == CT_SQL and report["queries"][0]["cohort_run"]["status"] \
        == "ran"
    assert report["queries"][0]["requirement"] == REQUIREMENT
    table = population_results_table(checked)
    assert set(table.method) == {"extraction_sql"}
    path = tmp_path / "cohort.zip"
    save_patient_cohort(checked, path)
    loaded = load_patient_cohort(path)
    assert loaded.report["aggregate"] == report["aggregate"]


# Preset ------------------------------------------------------------------------


def test_preset_section_and_endpoint_requirement():
    config = load_default_preset()
    section = config.workup_sql_agent
    assert section["structured_outputs"] is True and section["tokenizer_mode"] == "bytes"
    limits = agent_limits(config)
    assert limits.max_actions == 12 and limits.max_concurrent_items == 8
    assert limits.submit_by == 6 and limits.max_review_queries == 3
    config.remote["enabled"] = False
    with pytest.raises(ValueError, match="OpenAI-compatible endpoint"):
        resolve_agent_llm(config)
    config = load_default_preset()
    config.remote.update(enabled=True, server_urls=["http://fabricated.invalid/v1"])
    config.workup_sql_agent["remote"]["model_name"] = "fabricated-model"
    resolved = resolve_agent_llm(config)
    assert resolved.model == "fabricated-model" and resolved.context_window == 131072
    assert resolved.response_format == "json_schema"
    with pytest.raises(ValueError, match="Unknown workup_sql_agent.limits"):
        WorkupSQLLimits.from_mapping({"max_turns": 3})


def test_sandbox_never_runs_without_linux_isolation(monkeypatch, database):
    monkeypatch.setattr(sys, "platform", "darwin")
    from matchminer_ai.patients.workup_sql.sandbox import SQLSandboxError

    with pytest.raises(SQLSandboxError):
        SQLSandbox(database=database)
    assert os.name == "posix"


# Lab tests, counseling, older records and steps the guideline advises against -----------


def lab(note_id, note_date, test="cbc", relation="this_report"):
    return {"item_type": "lab_test", "relation": relation,
            "source_of_document": relation == "this_report",
            "status": "planned" if relation == "planned" else "performed",
            "lab_test": test, "lab_text": test.upper(), "result_text": "", "result_flag":
            "normal", "collection_date": when(note_date, "note_date"),
            "evidence": [{"note_id": note_id, "note_date": note_date, "quote": "CBC", "start": 0,
                          "end": 3}]}


def lab_document(note_id, note_date, items, patient, version="mmai-clinical-extraction-v2"):
    row = document(note_id, note_date, [], patient=patient)
    row["record"]["schema_version"] = version
    row["record"]["items"]["lab_tests"] = items
    row["record"]["items"]["counseling"] = []
    return row


LAB_SQL = """
SELECT p.patient_key,
  CASE WHEN l.k IS NOT NULL THEN 'met' ELSE 'not_met' END AS status, 'applies' AS applicability,
  CASE WHEN l.k IS NOT NULL THEN 'documented' ELSE 'not_documented' END AS reason,
  COALESCE(l.k, []::BIGINT[]) AS evidence_item_keys, '' AS detail
FROM patients p LEFT JOIN (SELECT patient_key, list(item_key) AS k FROM lab_tests
  WHERE lab_test = 'cbc' AND occurrence = 'performed' GROUP BY 1) l USING (patient_key)
"""


def test_lab_counseling_tables_and_older_records():
    rows = pd.DataFrame([
        lab_document("n1", "2024-03-01", [lab("n1", "2024-03-01")], "v2-done"),
        lab_document("n2", "2024-03-02", [], "v2-none"),
        lab_document("n3", "2024-03-03", [], "v1-none", version="mmai-clinical-extraction-v1"),
    ])
    database = build_extraction_database(rows)
    assert {"lab_tests", "counseling", "extraction_coverage"} <= set(database.tables)
    labs = database.tables["lab_tests"].to_pylist()
    assert labs[0]["lab_test"] == "cbc" and labs[0]["date_lo"] == date(2024, 3, 1)
    coverage = {r["item_type"]: r["patients_covered"]
                for r in database.tables["extraction_coverage"].to_pylist()}
    assert coverage["lab_test"] == 2 and coverage["imaging_study"] == 3
    patients = database.tables["patients"].to_pylist()
    assert "lab_test" not in patients[database.patient_ids.index("v1-none")]["extracted_item_types"]
    report = check_workup_with_sql(rows, compiled_entries({"sql": LAB_SQL}))
    by = {p["patient_id"]: p["review"]["assessments"][0] for p in report["patients"]}
    assert by["v2-done"]["status"] == "met"
    assert (by["v2-none"]["status"], by["v2-none"]["reason"]) == ("not_met", "not_documented")
    # The older record could not contain lab tests: not documented becomes unknown.
    assert (by["v1-none"]["status"], by["v1-none"]["reason"]) == ("unknown",
                                                                  "incomplete_extraction")
    assert "without lab_tests" in by["v1-none"]["bottom_line"]


AGAINST_SQL = CT_SQL.replace("THEN 'met'", "THEN 'done_not_recommended'").replace(
    "THEN 'documented'", "THEN 'documented_not_recommended'")


def test_steps_the_guideline_advises_against(database):
    limits = WorkupSQLLimits()
    with SQLSandbox(database=database) as box:
        assert run_final_query(box, AGAINST_SQL, 4, limits, recommends_against=True).problems == []
        met = run_final_query(box, CT_SQL, 4, limits, recommends_against=True)
        assert any("never met" in p for p in met.problems)
        done = run_final_query(box, AGAINST_SQL, 4, limits)
        assert any("only for a requirement that advises against" in p for p in done.problems)
    entries = compiled_entries({"sql": AGAINST_SQL, "recommends_against": True}, {"sql": CT_SQL})
    report = check_workup_with_sql(rows(), entries, patient_ids=[PID, "p2", "p3", "p4"])
    first = report["patients"][0]["review"]["assessments"][0]
    assert first["status"] == "done_not_recommended" and first["recommends_against"] is True
    assert first["bottom_line"].startswith("Done though the guideline advises against it")
    against, normal = report["aggregate"]
    assert against["statuses"]["done_not_recommended"] == 1 and against["recommends_against"]
    assert against["documented_completion_fraction"] is None and against["applicable_assessed"] == 0
    assert normal["recommends_against"] is False and normal["documented_completion_fraction"] == 0.5


def test_agent_requests_get_their_own_timeout_slots_and_capacity_waits():
    config = load_default_preset()
    config.remote.update(enabled=True, server_urls=["http://fabricated.invalid/v1"],
                         request_timeout=7200, max_concurrent_requests=1)
    config.workup_sql_agent["remote"]["model_name"] = "fabricated-model"
    resolved = resolve_agent_llm(config)
    assert resolved.timeout == 600 and resolved.max_concurrent_requests == 8
    assert resolved.capacity_retry_initial_seconds == 10
    config.workup_sql_agent.update(request_timeout=90, max_concurrent_requests=3)
    resolved = resolve_agent_llm(config)
    assert resolved.timeout == 90 and resolved.max_concurrent_requests == 3


def test_activity_reports_the_turn_each_item_is_on(monkeypatch, database, llm):
    """Every exploring item's turn is reported while it runs, from the calling thread."""
    import threading

    caller = threading.current_thread()
    seen, threads = [], set()
    import time

    def slow(value):
        return lambda payload: (time.sleep(0.7), value)[1]

    script(monkeypatch, [slow(action("query", "SELECT 1")),
                         slow(action("submit", CT_SQL, summary="CT.")), slow(action("accept"))])
    compile_workup_sql([REQUIREMENT], database=database, llm=llm,
                       activity_callback=lambda items: (seen.append(items),
                                                        threads.add(threading.current_thread())))
    assert threads == {caller}
    assert seen[-1] == []  # nothing left exploring
    turns = [item["action"] for items in seen for item in items]
    assert turns and turns == sorted(turns) and max(turns) <= 3
    assert all(item == {"name": "CT chest", "action": item["action"], "max_actions": 12}
               for items in seen for item in items)
