"""The LLM checks (matchminer_ai.apps.dashboard/llm.py, llm_checks.py) over the fabricated trials and catalog of test_dash.py, with fakes
for the inference package's check_good_options and check_eligibility_criteria_with_llm and for the endpoint's
/v1/models. No request leaves the machine."""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from .conftest import wait_job
from .test_dash import dash, dash_client, dash_settings, fake_ncit, trial_data  # noqa: F401  (fixtures)

NEMOTRON = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4"
TEACHER = "http://teacher.example:8000/v1"  # the fabricated site config's endpoints.teacher.url
GB300 = {"backend": "openai", "endpoint": TEACHER, "model": NEMOTRON}


@pytest.fixture(autouse=True)
def vertex_credentials(monkeypatch):
    """This server's Vertex credentials, faked (``state["error"]`` is the preflight's answer); no Google call."""
    state = {"error": None, "calls": 0}

    def check(max_age=60.0):
        state["calls"] += 1
        return state["error"]

    monkeypatch.setattr("matchminer_ai.apps.dashboard.llm.vertex_credentials_error", check)
    monkeypatch.setattr("matchminer_ai.apps.dashboard.llm_checks.vertex_credentials_error", check)
    return state


def test_connection_rules_and_sampling():
    from matchminer_ai import load_default_preset

    from matchminer_ai.apps.dashboard.llm import Connection, normalize_endpoint

    assert normalize_endpoint("teacher.example:8000") == TEACHER
    assert normalize_endpoint("https://host/v1/") == "https://host/v1"
    for bad in ("http://user:pw@host/v1", "http://host/v1?x=1", "ftp://host", "two words"):
        with pytest.raises(ValueError):
            normalize_endpoint(bad)
    vertex = Connection.make("gemini4_vertex", "http://elsewhere/v1", "other-model")
    assert vertex.endpoint.startswith("https://aiplatform.googleapis.com/") and vertex.model == "google/gemini-4-hydrogen"
    assert not Connection.make("openai", "none").configured
    with pytest.raises(ValueError, match="model"):
        Connection.make("openai", "http://host:8000/v1", "")
    config = Connection().configure(load_default_preset(), ("llm_good_option",))
    section = config.llm_good_option
    # The preset's fixed greedy sampling is removed so the served model's profile (Nemotron's) applies.
    assert section["sampling_profile"] == "auto" and "temperature" not in section["remote"]["request_params"]
    assert "top_k" not in section["remote"]["extra_body"] and section["remote"]["model_name"] == NEMOTRON
    assert config.remote["server_urls"] == [TEACHER] and config.remote["enabled"] and config.debug_mode
    gemini = Connection.make("gemini4_vertex").configure(load_default_preset(), ("llm_good_option",))
    assert gemini.remote["provider"] == "google_agent_platform"
    assert gemini.llm_good_option["sampling_profile"] == "none" and gemini.llm_good_option["remote"]["extra_body"] == {}


def test_choosing_the_endpoint_and_its_models(dash_client, monkeypatch):
    import httpx

    described = dash_client.get("/api/llm").json()
    # The default is Gemini 4 Hydrogen on Vertex (user, 2026-11-17); the GB300 stays the OpenAI-compatible default.
    assert described["connection"]["backend"] == "gemini4_vertex" and "authorized" in described["notice"]
    assert described["connection"]["model"] == "google/gemini-4-hydrogen" and described["good_option"]["available"]
    assert described["defaults"]["endpoint"] == TEACHER
    assert set(described["vertex_profiles"]) == {"gemini4_vertex", "gemini_vertex", "gemma_vertex"}
    served = {"data": [{"id": NEMOTRON, "max_model_len": 250000}]}
    real = httpx.AsyncClient

    def fake_client(**kwargs):
        return real(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=served)), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", fake_client)
    models = dash_client.post("/api/llm/models", json={"backend": "openai", "endpoint": "otherhost:8001"}).json()
    assert models["available"] and models["endpoint"] == "http://otherhost:8001/v1"
    assert models["models"] == [{"id": NEMOTRON, "max_model_len": 250000}]
    changed = dash_client.post("/api/llm", json={"backend": "openai", "endpoint": "otherhost:8001",
                                                 "model": NEMOTRON}).json()
    assert changed["connection"]["endpoint"] == "http://otherhost:8001/v1"
    assert dash_client.post("/api/llm", json={"backend": "openai", "endpoint": "http://h/v1?q"}).status_code == 400
    off = dash_client.post("/api/llm", json={"backend": "openai", "endpoint": "none"}).json()
    assert not off["good_option"]["available"] and "No LLM endpoint" in off["good_option"]["reason"]
    response = dash_client.post("/api/good-option", json={"patient_id": "patient_a", "trial_id": "NCT90000001"})
    assert response.status_code == 400 and "No LLM endpoint" in response.json()["error"]


@pytest.fixture
def fake_good_options(monkeypatch):
    """check_good_options and the catalog loader, faked: Drugamab (D-INV) and Unsurib (D-UNC) are scored."""
    import matchminer_ai.matching as matching
    import matchminer_ai.trials as trials

    calls = []
    summaries = (SimpleNamespace(drug_id="D-INV", preferred_name="Drugamab"),
                 SimpleNamespace(drug_id="D-UNC", preferred_name="Unsurib"))
    catalog = SimpleNamespace(trial_status=lambda trial: "ok" if trial == "NCT90000001" else "no_scoreable_drug",
                              scoreable_summaries_for_trial=lambda trial: summaries)
    monkeypatch.setattr(trials, "load_good_option_catalog", lambda path, cache=False: catalog)

    def check_good_options(frame, *, catalog, config, return_metadata=False):
        calls.append((frame.to_dict("records"), config))
        row = frame.iloc[0]
        criterion = lambda point: {"point": point, "rationale": "Fabricated rationale."}  # noqa: E731
        assessment = lambda name, points: {"drug_name": name, "targeted_biomarkers": ["FOO"],  # noqa: E731
                                           **{c: criterion(p) for c, p in zip(
                                               ("disease_type_benefit", "common_biomarker_in_disease",
                                                "patient_biomarker_targeted", "biomarker_targeted_benefit"), points)}}
        out = pd.DataFrame([{"patient_id": row["patient_id"], "trial_id": row["trial_id"], "good_option_score": 0.375,
                             "good_option_points": 3, "good_option_max_points": 8, "good_option_drug_count": 2,
                             "good_option_status": "ok", "good_option_method": "llm",
                             "good_option_patient_disease_type": "EGFR-mutant NSCLC",
                             "good_option_drug_assessments": [assessment("Drugamab", (1, 1, 0, 0)),
                                                              assessment("Unsurib", (1, 0, 0, 0))],
                             "good_option_uncertainties": ["Fabricated uncertainty."],
                             "good_option_reasoning_text": "Fabricated reasoning.", "good_option_finish_reason": "stop",
                             "good_option_parse_error": "", "good_option_parse_attempts": 2,
                             "good_option_reasoning_off_fallback_used": False}])
        metadata = {"good_option_check": {"context_tokens": 250000, "context_source": "endpoint"},
                    "evidence_packing": {"prompts": 1, "subjects": 3}, "catalog_compatibility_id": "fabricated",
                    "model_metadata": {"llm_good_option": {"model_name": config.llm_good_option["remote"]["model_name"],
                                                           "responses": [{"long": "x" * 400}]}}}
        return (out, metadata) if return_metadata else out

    monkeypatch.setattr(matching, "check_good_options", check_good_options)
    return calls


def test_vertex_without_credentials_says_so_at_once(dash_client, monkeypatch, vertex_credentials, fake_good_options):
    import matchminer_ai.llm.vllm_server as server

    vertex_credentials["error"] = "Vertex credentials not available on this server (DefaultCredentialsError): ..."
    monkeypatch.setattr(server, "check_openai_endpoint", lambda *a, **k: pytest.fail("no request without credentials"))
    described = dash_client.get("/api/llm").json()
    assert described["good_option"]["reason"].startswith("Vertex credentials not available")
    assert described["criteria"]["reason"].startswith("Vertex credentials not available")
    response = dash_client.post("/api/good-option", json={"patient_id": "patient_a", "trial_id": "NCT90000001"})
    assert response.status_code == 400 and "Vertex credentials not available" in response.json()["error"]
    models = dash_client.post("/api/llm/models", json={"backend": "gemini4_vertex"}).json()
    assert not models["available"] and "Vertex credentials not available" in models["error"]
    assert fake_good_options == []
    # With credentials, the hosted model gets the check with its own sampling and thinking.
    vertex_credentials["error"] = None
    started = dash_client.post("/api/good-option", json={"patient_id": "patient_a", "trial_id": "NCT90000001"}).json()
    result = wait_job(dash_client, started["job"]["id"])["result"]
    config = fake_good_options[0][1]
    assert config.remote["provider"] == "google_agent_platform" and config.llm_good_option["sampling_profile"] == "none"
    assert result["connection"]["model"] == "google/gemini-4-hydrogen"


def test_good_option_check_runs_as_a_job_and_is_cached(dash_client, dash, fake_good_options):
    dash_client.post("/api/llm", json=GB300)
    started = dash_client.post("/api/good-option", json={"patient_id": "patient_a", "trial_id": "NCT90000001"}).json()
    result = wait_job(dash_client, started["job"]["id"])["result"]
    assert result["status"] == "ok" and result["points"] == 3 and result["max_points"] == 8 and result["score"] == 0.375
    assert result["parse_attempts"] == 2 and result["max_parse_attempts"] == 3 and result["reasoning"]
    drugamab = result["assessments"][0]
    assert drugamab["drug_id"] == "D-INV" and drugamab["points"] == 2 and drugamab["role"] == "investigational"
    assert drugamab["class_ids"]["FOO Inhibitor"] == "NCIT:C2" and drugamab["criteria"][0]["point"] == 1
    assert result["model_metadata"] == {"model_name": NEMOTRON}  # long or nested metadata is left out
    sent = fake_good_options[0][0][0]
    assert sent["cancer_history_summary"].startswith("Age: 62") and sent["trial_id"] == "NCT90000001"
    again = dash_client.post("/api/good-option", json={"patient_id": "patient_a", "trial_id": "NCT90000001"}).json()
    assert again["result"]["cached"] and len(fake_good_options) == 1
    rerun = dash_client.post("/api/good-option", json={"patient_id": "patient_a", "trial_id": "NCT90000001",
                                                       "rerun": True}).json()
    wait_job(dash_client, rerun["job"]["id"])
    assert len(fake_good_options) == 2
    # Another model is another answer.
    dash_client.post("/api/llm", json={"backend": "openai", "endpoint": "otherhost:8001", "model": "other/model"})
    other = dash_client.post("/api/good-option", json={"patient_id": "patient_a", "trial_id": "NCT90000001"}).json()
    assert "job" in other and wait_job(dash_client, other["job"]["id"])["result"]["connection"]["model"] == "other/model"
    rubric = __import__("matchminer_ai.apps.dashboard.llm_checks", fromlist=["rubric"]).rubric()
    assert [c["criterion"] for c in rubric["criteria"]] == ["disease_type_benefit", "common_biomarker_in_disease",
                                                            "patient_biomarker_targeted", "biomarker_targeted_benefit"]
    assert rubric["criteria"][0]["question"].startswith("Has this drug")


def test_good_option_errors_are_shown_and_not_cached(dash_client, monkeypatch, fake_good_options):
    import matchminer_ai.matching as matching

    def down(*args, **kwargs):
        raise ConnectionError("endpoint down")

    monkeypatch.setattr(matching, "check_good_options", down)
    started = dash_client.post("/api/good-option", json={"patient_id": "patient_a", "trial_id": "NCT90000001"}).json()
    job = wait_job(dash_client, started["job"]["id"])
    assert job["status"] == "error" and "endpoint down" in job["error"]
    assert "job" in dash_client.post("/api/good-option", json={"patient_id": "patient_a",
                                                                "trial_id": "NCT90000001"}).json()


def test_criteria_llm_check(dash_client, dash, monkeypatch):
    import matchminer_ai.matching as matching
    from matchminer_ai import load_default_preset

    base = load_default_preset()  # an inference package with the check also has its preset section
    base.raw.setdefault("llm_criteria_check", {"structured_outputs": True, "max_attempts": 3, "chunk_size": 64,
                                               "chunks_in_flight": 1, "sampling_profile": "auto",
                                               "remote": {"model_name": NEMOTRON,
                                                          "request_params": {"max_tokens": 32768}}})
    dash.llm._base = base
    dash_client.post("/api/llm", json=GB300)

    monkeypatch.delattr(matching, "check_eligibility_criteria_with_llm", raising=False)
    response = dash_client.post("/api/criteria-llm", json={"patient_id": "patient_a", "cohort_id": "NCT90000001-cohort-0"})
    assert response.status_code == 400 and "criteria-llm-check" in response.json()["error"]
    assert not dash_client.get("/api/llm").json()["criteria"]["available"]
    calls = []

    def check(frame, *, config, return_metadata=False, progress=None):
        calls.append(frame.iloc[0].to_dict())
        import json

        items = json.loads(frame.iloc[0]["cohort_criteria_items"])
        rows = [{"patient_id": "patient_a", "cohort_trial_id": "NCT90000001-cohort-0", "criterion_number": i["number"],
                 "criterion_type": i["criterion_type"], "boilerplate": i["boilerplate"], "criterion_text": i["text"],
                 "llm_criterion_answer": "no" if i["criterion_type"] == "exclusion" else "yes",
                 "llm_criterion_clearly_fails": i["criterion_type"] == "exclusion",
                 "llm_criterion_status": "met", "llm_criterion_reason": "Fabricated reason.",
                 "llm_criterion_finish_reason": "stop", "llm_criterion_attempts": 1,
                 "llm_criterion_parse_status": "parsed", "llm_criterion_reasoning_text": "Fabricated reasoning."}
                for i in items]
        if progress:
            progress(len(rows), len(rows))
        metadata = {"criteria_contract": "7a7eb230", "model_metadata": {"check_eligibility_criteria_with_llm": {
            "requests": len(rows), "retries": 0, "parse_failures": 0, "unchecked_lines": 0,
            "structured_outputs": True}}}
        return (pd.DataFrame(rows), metadata) if return_metadata else pd.DataFrame(rows)

    monkeypatch.setattr(matching, "check_eligibility_criteria_with_llm", check, raising=False)
    started = dash_client.post("/api/criteria-llm", json={"patient_id": "patient_a",
                                                          "cohort_id": "NCT90000001-cohort-0"}).json()
    assert "job" in started, started
    result = wait_job(dash_client, started["job"]["id"])["result"]
    assert [line["fails"] for line in result["lines"]] == [False, False, True]
    assert result["counts"]["clearly_fail"] == 1 and result["counts"]["lines"] == 3 and result["structured_outputs"]
    assert calls[0]["general_exclusion_criteria_evidence"] == "ECOG 1. Creatinine 0.9 mg/dL."
    cached = dash_client.post("/api/criteria-llm", json={"patient_id": "patient_a",
                                                         "cohort_id": "NCT90000001-cohort-0"}).json()
    assert cached["result"]["cached"] and len(calls) == 1
    # Since 2026-11-17 (user) protocol criteria may go to the hosted (Vertex) endpoint too, as may CT.gov cohorts'.
    hosted = dash_client.post("/api/llm", json={"backend": "gemini4_vertex"}).json()
    assert hosted["criteria"]["available"] and "protocol_cohorts" not in hosted["criteria"]
    protocol = dash_client.post("/api/criteria-llm", json={"patient_id": "patient_a",
                                                           "cohort_id": "NCT90000001-cohort-0"}).json()
    assert wait_job(dash_client, protocol["job"]["id"])["status"] == "done" and len(calls) == 2
    public = dash_client.post("/api/criteria-llm", json={"patient_id": "patient_a",
                                                         "cohort_id": "NCT90000001-ctgov-cohort-0"}).json()
    assert wait_job(dash_client, public["job"]["id"])["status"] == "done" and len(calls) == 3
