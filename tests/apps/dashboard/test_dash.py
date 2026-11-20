"""The dashboard (matchminer_ai.apps.dashboard) over a fabricated trial_data folder, OA table and GoodOption catalog (no real protocol
text or drug data), with the fabricated synthetic run, the fake decision model and the fake encoder of conftest.py."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

from .conftest import PATIENTS, FakeEncoder, encoder_identity_for, wait_job, write_json, write_round

THORACIC = "ECC/MAIN Thoracic Oncology"
BREAST = "ECC/MAIN Breast Oncology"


def entry(number, name, lines):
    """A deboilerplate.json cohort: (text, criterion_type, boilerplate) lines under their headings."""
    items = [{"number": i, "label": "Inclusion criteria:" if kind == "inclusion" else "Exclusion criteria:", "text": text,
              "criterion_type": kind, "boilerplate": bp, "attempts": []} for i, (text, kind, bp) in enumerate(lines, 1)]
    key = "\n".join(t for t, _, bp in lines if not bp)
    common = "\n".join(t for t, _, bp in lines if bp)
    return {"cohort_number": number, "cohort_name": name, "input_sha256": "0" * 64, "key_criteria": key,
            "boilerplate_criteria": common, "items": items}


def write_trial(root: Path, version: str, trial_id: str, cohorts: list | None, *, screen="yes", at="2026-11-11T10:00:00+00:00",
                extracted_at="2026-11-11T09:00:00+00:00"):
    folder = root / version / "trials" / trial_id
    if version == "protocol":
        write_json(folder / "extraction.json", {"trial_id": trial_id, "completed_at": extracted_at,
                                                "decision": {"cancer_treatment_trial": screen, "cohorts": cohorts or []}})
    if cohorts is not None:
        write_json(folder / "deboilerplate.json", {"trial_id": trial_id, "cohorts": cohorts, "completed_at": at})


# A fabricated GoodOption catalog: (trial, drug id, name, CT.gov name, role, arms) and each drug's (class, NCIt code).
CATALOG_INDEX = [
    ("NCT90000001", "D-BG", "Testplatin", "testplatin", "background", ["Arm A: Drug A + testplatin"]),
    ("NCT90000001", "D-SUP", "Antiemetol", "Antiemetol", "supportive", []),
    ("NCT90000001", "D-CTL", "Comparatinib", "Comparatinib", "control", ["Arm B: comparatinib"]),
    ("NCT90000001", "D-INV", "Drugamab", "DRG-101", "investigational", ["Arm A: Drug A + testplatin"]),
    ("NCT90000001", "D-UNC", "Unsurib", "Unsurib", "uncertain", []),
    ("NCT90000002", "D-ADC", "Fabtuzumab Payloadecan", "FP-7", "investigational", []),
    ("NCT90000002", "D-SUP", "Antiemetol", "Antiemetol", "supportive", []),
]
CATALOG_CLASSES = {
    "D-INV": [("Anti-FOO Monoclonal Antibody", "C3"), ("Monoclonal Antibody", "C10"), ("FOO Inhibitor", "C2"),
              ("Checkpoint Inhibitor", "C1")],
    "D-BG": [("Platinum Compound", "C20")],
    "D-CTL": [("BAR Kinase Inhibitor", "C40"), ("Kinase Inhibitor", "C41")],
    "D-UNC": [("Invented Class", "")],  # an LLM-made class without an NCIt code
    "D-ADC": [("Exatecan Derivative-containing Antibody-drug Conjugate", "C31"),
              ("Topoisomerase I Inhibitor-containing Antibody-drug Conjugate", "C30"),
              ("Anti-FOO Antibody-drug Conjugate", "C32"), ("Antineoplastic Antibody-Drug Conjugate", "C29")],
    "D-SUP": [("Antiemetic Agent", "C50")],
}
# The fake NCIt: each code's ancestors.
FAKE_ANCESTORS = {"C3": {"C2", "C1", "C10"}, "C2": {"C1"}, "C40": {"C41"}, "C31": {"C30", "C29"}, "C30": {"C29"},
                  "C32": {"C29"}}


INVENTED_CLASS_ID = "0f0f0f0f0f0f0f0f0f0f0f0f"  # an LLM-made class's catalog ID (no NCIt code)
# Drugamab's facts: one facet out of reading order, qualifiers, and a citation with no passage ("web:missing").
DRUGAMAB_FACTS = {
    "limitations": [{"text": "Fabricated limitation.", "scope": "agent", "support_ids": ["pubmed:2"]}],
    "efficacy_by_tumor": [{"text": "Fabricated response rate of 40%.", "scope": "agent", "tumor_type": "lung cancer",
                           "regimen": "drugamab 10 mg/kg", "evidence_level": "phase 2", "extra_note": ["a", "b"],
                           "support_ids": ["pubmed:2", "web:missing"]}],
    "mechanism_and_targets": [{"text": "D-INV blocks a fabricated target.", "scope": "agent",
                               "support_ids": ["pubmed:1", "ncit_definition:C3"]},
                              {"text": "A second fact.", "scope": "agent", "support_ids": ["pubmed:1"]}],
}
# (owner, evidence ID, title, passage, url)
DRUG_EVIDENCE = [
    ("D-INV", "pubmed:1", "Drugamab mechanism", "Drugamab binds FOO.", "https://pubmed.ncbi.nlm.nih.gov/1/"),
    ("D-INV", "ncit_definition:C3", "NCIt definition", "An anti-FOO antibody.", "javascript:alert(1)"),
    ("D-INV", "pubmed:2", "Drugamab phase 2", "Responses in 40% of patients.", ""),
    ("D-BG", "pubmed:2", "Another drug's copy", "Not Drugamab's passage.", ""),
]
CLASS_EVIDENCE = [("NCIT:C2", "pubmed:9", "FOO inhibitors review", "FOO inhibitors block FOO.", "https://example.org/9")]


def class_id_of(code: str) -> str:
    return f"NCIT:{code}" if code else INVENTED_CLASS_ID


def write_catalog(folder: Path, index=CATALOG_INDEX, classes=CATALOG_CLASSES) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    folder.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"trial_id": [r[0] for r in index], "drug_id": [r[1] for r in index],
                             "preferred_name": [r[2] for r in index], "ctgov_name": [r[3] for r in index],
                             "intervention_type": ["DRUG"] * len(index), "role": [r[4] for r in index],
                             "arm_labels_json": [json.dumps(r[5]) for r in index]}), folder / "trial_drug_index.parquet")
    rows = [(drug, name, code) for drug, found in classes.items() for name, code in found]
    pq.write_table(pa.table({"drug_id": [r[0] for r in rows], "class_id": [class_id_of(r[2]) for r in rows],
                             "class_name": [r[1] for r in rows], "ncit_code": [r[2] for r in rows],
                             "assigned_by": ["ncit_hierarchy" if r[2] else "llm_new" for r in rows]}),
                   folder / "drug_classes.parquet")
    names = {r[1]: r[2] for r in index}
    facts = {drug: DRUGAMAB_FACTS if drug == "D-INV" else {"mechanism_and_targets": [
        {"text": f"{drug} blocks a fabricated target.", "scope": "agent"}, {"text": "A second fact.", "scope": "agent"}]}
        for drug in classes}
    pq.write_table(pa.table({"drug_id": list(facts), "preferred_name": [names[d] for d in facts],
                             "ncit_code": ["C3" if d == "D-INV" else "" for d in facts],
                             "research_status": ["complete"] * len(facts), "synthesis_status": ["ok"] * len(facts),
                             "good_option_summary": [f"Drug: {names[d]}\nFull fabricated summary." for d in facts],
                             "evidence_count": [3] * len(facts),
                             "structured_facts_json": [json.dumps(f) for f in facts.values()]}),
                   folder / "drug_summaries.parquet")
    class_names = dict.fromkeys((class_id_of(code), name) for found in classes.values() for name, code in found)
    class_facts = {cid: {"mechanism_and_targets": [{"text": f"{name} is a fabricated class.", "scope": "class",
                                                    "support_ids": ["pubmed:9", "europe_pmc:missing"]
                                                    if cid == "NCIT:C2" else []}]} for cid, name in class_names}
    pq.write_table(pa.table({"class_id": [c for c, _ in class_names], "class_name": [n for _, n in class_names],
                             "synthesis_status": ["ok"] * len(class_names),
                             "structured_facts_json": [json.dumps(class_facts[c]) for c, _ in class_names],
                             "class_option_summary": [f"Drug class: {n}" for _, n in class_names],
                             "evidence_count": [1] * len(class_names)}), folder / "class_summaries.parquet")
    for name, key, evidence in (("drug_evidence", "drug_id", DRUG_EVIDENCE), ("class_evidence", "class_id", CLASS_EVIDENCE)):
        (folder / name).mkdir(exist_ok=True)
        pq.write_table(pa.table({"evidence_id": [e[1] for e in evidence], "drug_id": [e[0] if key == "drug_id" else ""
                                                                                    for e in evidence],
                                 "class_id": [e[0] if key == "class_id" else "" for e in evidence],
                                 "source": ["pubmed"] * len(evidence), "source_type": ["abstract"] * len(evidence),
                                 "title": [e[2] for e in evidence], "passage": [e[3] for e in evidence],
                                 "url": [e[4] for e in evidence], "source_locator": ["PMID"] * len(evidence),
                                 "published_at": ["2025-01-01"] * len(evidence), "license": ["CC BY 4.0"] * len(evidence)}),
                       folder / name / "part-00000.parquet")
    write_json(folder / "manifest.json", {"build_id": "fabricated", "created_at_utc": "2026-11-14T13:22:09+00:00",
                                          "ncit": {"resource": "Thesaurus_fake.FLAT.zip", "version": "fake"},
                                          "counts": {"trials": 2, "unique_drugs": 6}})
    return folder


def fake_ancestors(resource, codes):
    return {code: frozenset(FAKE_ANCESTORS.get(code, ())) for code in codes}


@pytest.fixture
def fake_ncit(monkeypatch):
    """The NCIt hierarchy, faked (the real one is the inference package's bundled thesaurus)."""
    from matchminer_ai.apps.dashboard import drugs

    monkeypatch.setattr(drugs, "ncit_ancestors", fake_ancestors)


@pytest.fixture
def trial_data(tmp_path):
    import duckdb

    from matchminer_ai.apps.dashboard.pipeline import TRIAL_FIELDS, trial_table

    root = tmp_path / "trial_data"
    root.mkdir()
    (root / "inputs.csv").write_text(
        "trial_id,protocol_no,oa_status,phase,short_title,protocol_excluded_reason,ctgov_excluded_reason\n"
        "NCT90000001,25-001,OPEN TO ACCRUAL,II,Drug A in EGFR NSCLC,,\n"
        "NCT90000002,25-002,OPEN TO ACCRUAL,III,Drug B in HR+ breast,no folder in the protocols store,\n"
        "NCT90000003,25-003,OPEN TO ACCRUAL,I,Imaging study,,\n"
        "NCT90000004,25-004,OPEN TO ACCRUAL,I/II,Drug D in lung cancer,,\n")
    write_trial(root, "protocol", "NCT90000001", [
        entry(0, "Cohort A (NSCLC, EGFR exon 19 deletion)", [
            ("Metastatic non-small cell lung cancer with an EGFR exon 19 deletion.", "inclusion", False),
            ("ECOG 0-1.", "inclusion", True), ("Prior osimertinib.", "exclusion", False)]),
        entry(1, "Cohort B (NSCLC, EGFR wild type)", [
            ("Previously treated metastatic non-small cell lung cancer, EGFR wild type.", "inclusion", False),
            ("Adequate organ function.", "inclusion", True)])])
    write_trial(root, "ctgov", "NCT90000001", [entry(0, "Drug A", [("Metastatic NSCLC.", "inclusion", False)])])
    write_trial(root, "ctgov", "NCT90000002", [entry(0, "All participants", [
        ("HR-positive, HER2-negative metastatic breast cancer.", "inclusion", False), ("ECOG 0-1.", "inclusion", True)])])
    write_trial(root, "protocol", "NCT90000003", None, screen="no")
    write_trial(root, "protocol", "NCT90000004", None)  # extracting: no deboilerplate.json yet
    db = tmp_path / "oa.duckdb"
    with duckdb.connect(str(db)) as con:
        table, columns = trial_table()
        con.execute(f"CREATE TABLE {table} ({', '.join(f'{columns[f]} VARCHAR' for f in TRIAL_FIELDS)})")
        sites = repr({"site": [{"site_name": "Example Cancer Center"}, {"site_name": "Example General Hospital"},
                               {"site_name": "Unmapped Site"}]})
        for nct, protocol, center in (("NCT90000001", "25-001", THORACIC), ("NCT90000002", "25-002", BREAST),
                                      ("NCT90000003", "25-003", THORACIC), ("NCT90000004", "25-004", THORACIC)):
            con.execute(f"INSERT INTO {table} VALUES ({', '.join('?' * len(TRIAL_FIELDS))})",
                        [nct, protocol, "Short", "Long title", center, "Pat Coordinator", "pat@example.org",
                         "Lead Study Coordinator", "Dr PI", "Dr Overall", "OPEN TO ACCRUAL", "II", "A summary.", sites, None])
    return {"root": root, "db": db, "catalog": write_catalog(tmp_path / "good_options" / "catalog")}


@pytest.fixture
def dash_settings(dataset, trial_data, tmp_path):
    from matchminer_ai.apps.dashboard.settings import DashSettings

    return DashSettings(synthetic_run=dataset["root"], store_dir=tmp_path / "store", trial_data=trial_data["root"],
                        oa_db=trial_data["db"], good_options_catalog=trial_data["catalog"],
                        embedding_caches=(tmp_path / "hf",), embedding_python=tmp_path / "missing-python", auto_score_n=2)


@pytest.fixture
def dash(dash_settings, fake_model, fake_ncit):
    from matchminer_ai.apps.dashboard.state import DashState

    state = DashState(dash_settings, encoder=FakeEncoder(), log=lambda *_: None)
    job = state.embed_cohorts()
    while job is not None and job.status not in ("done", "error"):
        time.sleep(0.01)
    return state


@pytest.fixture
def dash_client(dash):
    from fastapi.testclient import TestClient

    from matchminer_ai.apps.dashboard.server import create_app

    with TestClient(create_app(dash), base_url="http://127.0.0.1") as test_client:  # a loopback Host
        yield test_client


def test_loopback_host_cross_origin_writes_and_the_access_token(dash):
    from fastapi.testclient import TestClient

    from matchminer_ai.apps.dashboard.server import create_app

    with TestClient(create_app(dash), base_url="http://dashboard.example") as client:
        assert client.get("/api/meta").status_code == 400  # a DNS-rebinding Host is refused
    token = "fabricated-dashboard-access-token-abcdef0123456789"
    with TestClient(create_app(dash, access_token=token), base_url="http://127.0.0.1") as client:
        assert client.get("/api/meta").status_code == 401
        assert client.get("/", follow_redirects=False).headers["location"] == "/login"
        assert client.get("/login?token=wrong-token", follow_redirects=False).status_code == 401
        assert client.get(f"/login?token={token}", follow_redirects=False).status_code == 303
        assert client.get("/api/meta").status_code == 200  # the sign-in cookie
        refused = client.post("/api/llm", json={}, headers={"origin": "http://other.example"})
        assert refused.status_code == 403
    with TestClient(create_app(dash, access_token=token), base_url="http://127.0.0.1") as client:
        assert client.get("/api/meta", headers={"Authorization": f"Bearer {token}"}).status_code == 200


def test_store_versions_centers_and_groups(dash_client, dash):
    store = dash.store
    t1, t2 = store["trials"]["NCT90000001"], store["trials"]["NCT90000002"]
    assert t1["primary_version"] == "protocol" and t1["cohort_ids"] == ["NCT90000001-cohort-0", "NCT90000001-cohort-1"]
    assert t1["versions"]["ctgov"]["cohort_ids"] == ["NCT90000001-ctgov-cohort-0"]  # kept for the details dialog
    assert t2["primary_version"] == "ctgov" and t2["versions"]["protocol"]["state"] == "not_run"
    assert store["trials"]["NCT90000003"]["state"] == "not_treatment"
    assert store["trials"]["NCT90000004"]["state"] == "pending"
    assert t1["sites"] == ["ECC Main", "EGH"]  # mapped sites only
    # TOP-style rules: the EGFR-mutant cohort is in EGFR; "EGFR wild type" is negated, so its sibling is not.
    assert store["cohorts"]["NCT90000001-cohort-0"]["group"] == "EGFR"
    assert store["cohorts"]["NCT90000001-cohort-1"]["group"] == "NSCLC Previously Treated"
    assert store["cohorts"]["NCT90000002-ctgov-cohort-0"]["group"] == "HR-Positive / HER2-Negative"
    meta = dash_client.get("/api/meta").json()
    centers = {c["name"]: c for c in meta["centers"]}
    # The site config's trial table, center prefix, PI label and protocol links reach the store and the frontend.
    assert centers[THORACIC]["short_name"] == "Thoracic Oncology" and t1["site_pi"] == "Dr PI"
    assert meta["site"]["pi_label"] == "ECC PI" and meta["higher_threshold_groups"] == ["ECC Phase I"]
    assert meta["site"]["trial_links"] == [{"label": "Portal", "url": "https://portal.example.org/protocols/{protocol_no}"}]
    assert centers[THORACIC]["trials"] == 3 and centers[THORACIC]["cohorts"] == 2 and centers[THORACIC]["pending"] == 2
    view = dash_client.get(f"/api/centers/{centers[THORACIC]['id']}").json()
    assert [g["name"] for g in view["groups"]] == ["EGFR", "NSCLC Previously Treated"]
    row = view["groups"][0]["rows"][0]
    assert row["key_inclusion"] == ["Metastatic non-small cell lung cancer with an EGFR exon 19 deletion."]
    assert row["key_exclusion"] == ["Prior osimertinib."] and row["common_count"] == 1
    detail = dash_client.get("/api/trials/NCT90000001").json()
    assert len(detail["versions"]["protocol"]["cohorts"]) == 2 and len(detail["versions"]["ctgov"]["cohorts"]) == 1


def test_finished_extraction_is_picked_up(dash, trial_data):
    """A trial's deboilerplate.json appearing (its extraction finishing) puts it on its dashboard and embeds it."""
    assert not dash.check_store()  # nothing changed
    write_trial(trial_data["root"], "protocol", "NCT90000004", [
        entry(0, "All participants", [("Advanced small cell lung cancer after platinum.", "inclusion", False)])])
    assert dash.check_store()
    trial = dash.store["trials"]["NCT90000004"]
    assert trial["state"] == "ready" and trial["cohort_ids"] == ["NCT90000004-cohort-0"]
    assert dash.store["cohorts"]["NCT90000004-cohort-0"]["group"] == "Small Cell Lung Cancer"
    deadline = time.monotonic() + 10
    while dash.cohort_vectors.vector(dash.store, "NCT90000004-cohort-0") is None and time.monotonic() < deadline:
        time.sleep(0.02)
    assert dash.cohort_vectors.vector(dash.store, "NCT90000004-cohort-0") is not None
    # A newer extraction of a shown trial keeps its last complete cohorts, marked stale.
    write_trial(trial_data["root"], "protocol", "NCT90000004", None, extracted_at="2026-11-12T00:00:00+00:00")
    assert dash.check_store()
    assert dash.store["trials"]["NCT90000004"]["state"] == "stale"
    assert dash.store["trials"]["NCT90000004"]["cohort_ids"] == ["NCT90000004-cohort-0"]


def test_both_matching_directions_score_and_check(dash_client, dataset):
    found = dash_client.post("/api/match/patients-for-cohort", json={"cohort_id": "NCT90000001-cohort-0", "k": 3}).json()
    assert len(found["results"]) == 3 and found["of"] == 5
    patient = found["results"][0]["patient"]["id"]
    job = dash_client.post("/api/score", json={"direction": "patients_for_cohort", "query_id": "NCT90000001-cohort-0",
                                               "target_ids": [patient]}).json()["job"]
    scores = wait_job(dash_client, job["id"])["result"]["scores"][0]
    assert scores["match"]["score"] in (0.9, 0.1) and scores["boilerplate"]["pass"] is True

    trials = dash_client.post("/api/match/trials-for-patient", json={"patient_id": "patient_a", "k": 10}).json()
    trial_ids = [r["cohort"]["trial"]["id"] for r in trials["results"]]
    assert len(trial_ids) == len(set(trial_ids)) == 2  # best cohort per trial; only trials with dashboard cohorts
    entered = dash_client.post("/api/match/trials-for-patient",
                               json={"summary": "Age: 60\nCancer type: Lung cancer", "boilerplate": "ECOG 1."}).json()
    assert entered["patient"]["custom"] and len(entered["results"]) == 2

    job = dash_client.post("/api/criteria", json={"patient_id": "patient_a", "cohort_id": "NCT90000001-cohort-0"}).json()["job"]
    lines = wait_job(dash_client, job["id"])["result"]["lines"]
    assert [line["fails"] for line in lines] == [False, False, True]  # the fake fails exclusion lines


def test_trial_drugs_from_the_catalog(dash_client, dash):
    """Drugs in role order without supportive ones; mechanism types are the most specific classes."""
    drugs = dash.store["trials"]["NCT90000001"]["drugs"]
    assert [(d["name"], d["role"]) for d in drugs] == [("Drugamab", "investigational"), ("Unsurib", "uncertain"),
                                                       ("Comparatinib", "control"), ("Testplatin", "background")]
    drug = drugs[0]
    assert drug["mechanism_types"] == ["Anti-FOO Monoclonal Antibody"]  # its NCIt ancestors are left out
    assert drug["all_classes"] == ["Anti-FOO Monoclonal Antibody", "Monoclonal Antibody", "FOO Inhibitor",
                                   "Checkpoint Inhibitor"]
    assert drug["listed_name"] == "DRG-101" and drug["arms"] == ["Arm A: Drug A + testplatin"]
    assert drug["mechanism"] == "D-INV blocks a fabricated target." and drug["payload"] is None
    assert drugs[1]["mechanism_types"] == ["Invented Class"] and drugs[2]["mechanism_types"] == ["BAR Kinase Inhibitor"]
    assert drugs[3]["listed_name"] is None  # the same name in another case
    adc = dash.store["trials"]["NCT90000002"]["drugs"]
    assert [d["name"] for d in adc] == ["Fabtuzumab Payloadecan"]
    assert adc[0]["mechanism_types"] == ["Exatecan Derivative-containing Antibody-drug Conjugate",
                                         "Anti-FOO Antibody-drug Conjugate"]
    assert adc[0]["payload"] == "Exatecan Derivative (Topoisomerase I Inhibitor)"
    assert adc[0]["payload_classes"] == [{
        "label": "Exatecan Derivative", "class_name": "Exatecan Derivative-containing Antibody-drug Conjugate",
        "generic": [{"label": "Topoisomerase I Inhibitor",
                     "class_name": "Topoisomerase I Inhibitor-containing Antibody-drug Conjugate"}]}]
    assert drug["class_ids"]["FOO Inhibitor"] == "NCIT:C2" and drugs[1]["class_ids"] == {"Invented Class": INVENTED_CLASS_ID}
    assert set(drug["class_ids"]) == set(drug["all_classes"]) and drugs[1]["payload_classes"] == []
    assert dash.store["trials"]["NCT90000003"]["drugs"] == []
    info = dash.store["good_options"]
    assert info["state"] == "ready" and info["most_specific"] and info["build_id"] == "fabricated"
    assert info["shown"] == {"trials": 4, "trials_with_drugs": 2, "drugs": 5, "distinct_drugs": 5,
                             "drugs_by_role": {"investigational": 2, "uncertain": 1, "control": 1, "background": 1},
                             "mechanism_types": 4, "filter_classes": 11}
    # The overview's rows carry the drugs without arms or mechanism sentences; the trial's details have them all.
    thoracic = next(c for c in dash_client.get("/api/meta").json()["centers"] if c["name"] == THORACIC)
    row = dash_client.get(f"/api/centers/{thoracic['id']}").json()["groups"][0]["rows"][0]
    assert [d["name"] for d in row["trial"]["drugs"]] == ["Drugamab", "Unsurib", "Comparatinib", "Testplatin"]
    assert "mechanism" not in row["trial"]["drugs"][0] and row["trial"]["drugs"][0]["all_classes"] == drug["all_classes"]
    assert row["trial"]["drugs"][0]["class_ids"] == drug["class_ids"]
    assert dash_client.get("/api/trials/NCT90000001").json()["drugs"] == drugs
    status = dash_client.get("/api/status").json()
    assert status["good_options"]["shown"]["trials_with_drugs"] == 2
    assert dash_client.get("/api/meta").json()["good_options"] == {"state": "ready", "message": None}
    assert status["sources"]["good_options"] == str(dash.settings.good_options_catalog)


def test_drug_and_class_evidence(dash_client, dash):
    """The evidence dialogs: facts by facet with qualifiers, citations resolved to the owner's own passages, a missing
    passage listed, classes with their IDs, the class's drugs and the store's trials with them."""
    from urllib.parse import quote

    found = dash_client.get(f"/api/drugs/{quote('D-INV', safe='')}").json()
    assert found["name"] == "Drugamab" and found["ncit_code"] == "C3" and "LLM-assisted" in found["label"]
    assert [f["facet"] for f in found["facets"]] == ["mechanism_and_targets", "efficacy_by_tumor", "limitations"]
    efficacy = found["facets"][1]["facts"][0]
    assert efficacy["qualifiers"] == [["scope", "agent"], ["evidence_level", "phase 2"], ["tumor_type", "lung cancer"],
                                      ["regimen", "drugamab 10 mg/kg"], ["extra_note", "a; b"]]
    assert efficacy["support_ids"] == ["pubmed:2", "web:missing"]
    assert set(found["evidence"]) == {"pubmed:1", "ncit_definition:C3", "pubmed:2"}
    assert found["evidence"]["pubmed:2"]["passage"] == "Responses in 40% of patients."  # its own, not D-BG's copy
    assert found["evidence"]["pubmed:1"]["url"] == "https://pubmed.ncbi.nlm.nih.gov/1/"
    assert found["evidence"]["ncit_definition:C3"]["url"] == ""  # only web links are passed on
    assert found["missing_citations"] == ["web:missing"]
    assert found["counts"] == {"facts": 4, "citations": 6, "cited_sources": 4, "resolved_sources": 3}
    assert found["summary"].startswith("Drug: Drugamab")
    classes = {c["name"]: c for c in found["classes"]}
    assert classes["FOO Inhibitor"]["id"] == "NCIT:C2" and classes["FOO Inhibitor"]["specific"] is False
    assert classes["Anti-FOO Monoclonal Antibody"]["specific"] is True
    klass = dash_client.get(f"/api/classes/{quote('NCIT:C2', safe='')}").json()
    assert klass["name"] == "FOO Inhibitor" and klass["ncit_code"] == "C2"
    assert klass["evidence"]["pubmed:9"]["title"] == "FOO inhibitors review"
    assert klass["missing_citations"] == ["europe_pmc:missing"] and klass["drugs"] == [{"id": "D-INV", "name": "Drugamab"}]
    assert [(t["id"], t["drugs"]) for t in klass["trials"]] == [("NCT90000001", [{"name": "Drugamab",
                                                                                  "role": "investigational"}])]
    invented = dash_client.get(f"/api/classes/{INVENTED_CLASS_ID}").json()
    assert invented["ncit_code"] is None and invented["counts"]["citations"] == 0 and invented["trials"][0]["id"] == "NCT90000001"
    for path in ("/api/drugs/NOPE:1", "/api/classes/NCIT%3AC999"):
        response = dash_client.get(path)
        assert response.status_code == 404 and "has no" in response.json()["error"]


def test_evidence_without_a_catalog(dash_client, dash, tmp_path, trial_data):
    """A missing catalog: the dialogs' endpoints say so (404) and /api/meta gives the reason; evidence tables missing:
    citations are unresolved."""
    import shutil

    from matchminer_ai.apps.dashboard.evidence import CatalogEvidence
    from matchminer_ai.apps.dashboard.pipeline import Builder, Sources

    dash.store = Builder(Sources(trial_data["root"], trial_data["db"], dash.settings.grouping, tmp_path / "none")).build()
    catalog = dash_client.get("/api/meta").json()["good_options"]
    assert catalog["state"] == "missing" and "without drugs" in catalog["message"]
    dash.store = {k: v for k, v in dash.store.items() if k != "good_options"}  # a store built before the drugs
    assert dash_client.get("/api/meta").json()["good_options"] is None

    dash.evidence = CatalogEvidence(tmp_path / "no-catalog")
    for path in ("/api/drugs/D-INV", "/api/classes/NCIT%3AC2"):
        response = dash_client.get(path)
        assert response.status_code == 404 and "No GoodOption catalog" in response.json()["error"]
    copy = shutil.copytree(trial_data["catalog"], tmp_path / "copy")
    evidence = CatalogEvidence(copy, cache_size=1)
    assert evidence.drug("D-INV")["name"] == "Drugamab"
    renamed = [r if r[1] != "D-INV" else (*r[:2], "Drugamab Renamed", *r[3:]) for r in CATALOG_INDEX]
    summaries = copy / "drug_summaries.parquet"
    before = summaries.stat().st_mtime
    write_catalog(copy, index=renamed)
    os.utime(summaries, (before + 5, before + 5))
    assert evidence.drug("D-INV")["name"] == "Drugamab Renamed"  # the cached answer's stamps no longer match
    shutil.rmtree(copy / "drug_evidence")
    found = evidence.drug("D-INV")
    assert found["evidence"] == {} and found["counts"]["resolved_sources"] == 0 and len(found["missing_citations"]) == 4


def test_drugs_without_the_ncit_hierarchy_or_catalog(trial_data, tmp_path, monkeypatch):
    """Without NCIt every class is kept, in the catalog's order; without a catalog the trials just have no drugs."""
    from matchminer_ai.apps.dashboard import drugs
    from matchminer_ai.apps.dashboard.pipeline import Builder, Sources
    from matchminer_ai.apps.dashboard.settings import DEFAULT_GROUPING

    def unavailable(resource, codes):
        raise FileNotFoundError(resource)

    monkeypatch.setattr(drugs, "ncit_ancestors", unavailable)
    found, info = drugs.read_catalog(trial_data["catalog"])
    assert found["NCT90000001"][0]["mechanism_types"] == [name for name, _ in CATALOG_CLASSES["D-INV"]]
    assert found["NCT90000002"][0]["payload"] == "Exatecan Derivative, Topoisomerase I Inhibitor"
    assert info["state"] == "ready" and not info["most_specific"] and "could not be loaded" in info["hierarchy"]
    from matchminer_ai.apps.dashboard.settings import parse_settings

    chosen = parse_settings(["--store-dir", str(tmp_path / "store"), "--device", "cpu",
                             "--good-options-catalog", str(tmp_path / "no-catalog")])
    assert chosen.good_options_catalog == tmp_path / "no-catalog"
    for catalog in (chosen.good_options_catalog, None):
        store = Builder(Sources(trial_data["root"], trial_data["db"], DEFAULT_GROUPING, catalog)).build()
        assert all(t["drugs"] == [] for t in store["trials"].values()) and store["counts"]["trials"] == 4
        assert store["good_options"]["state"] == "missing" and "without drugs" in store["good_options"]["message"]
        assert store["good_options"]["shown"]["trials_with_drugs"] == 0


def test_catalog_changes_rebuild_the_store(dash, trial_data):
    """A rewritten catalog file changes the store's signature (file stamps), and the store is rebuilt from it."""
    assert not dash.check_store()
    index = [r if r[2] != "Unsurib" else (*r[:4], "investigational", r[5]) for r in CATALOG_INDEX]
    path = trial_data["catalog"] / "trial_drug_index.parquet"
    before = path.stat().st_mtime
    write_catalog(trial_data["catalog"], index=index)
    os.utime(path, (before + 5, before + 5))  # a later whole-second mtime even if the size is the same
    assert dash.builder.sources.signature() != dash.store["signature"]
    assert dash.check_store()
    assert [d["role"] for d in dash.store["trials"]["NCT90000001"]["drugs"]][:2] == ["investigational"] * 2
    assert not dash.check_store()


def test_grouping_negation_and_rule_errors(tmp_path):
    from matchminer_ai.apps.dashboard import grouping

    egfr = grouping._compile(r"\bEGFR\b")
    assert grouping.mentioned(egfr, "NSCLC with an activating EGFR mutation")
    for text in ("NSCLC, EGFR wild type", "no EGFR or ALK alteration", "EGFR-negative NSCLC", "absence of EGFR mutations"):
        assert not grouping.mentioned(egfr, text), text
    bad = tmp_path / "rules.yaml"
    bad.write_text("centers:\n  - match: 'x'\n    groups:\n      - name: g\n        cancer: [not_a_type]\n")
    rules = grouping.load_rules(bad)
    assert rules.centers == [] and "unknown cancer types" in rules.error
    assert grouping.load_rules(Path(grouping.__file__).with_name("grouping.yaml")).error is None


def test_a_round_being_copied_back_is_picked_up_once_whole(dash, dataset):
    """A later TrialSpace round is used only once its outputs and the fitted TrialSpace that made it are whole."""
    trialspace = dash.settings.path("trialspace")
    fitted = trialspace / "models" / "trialspace_round1"
    receipt = "a" * 64
    fitted.mkdir(parents=True)
    (fitted / "model.safetensors").write_bytes(b"12")
    write_json(fitted / "stage_receipt.json", {"format": "mmai-stage-receipt-v1", "kind": "fit", "receipt_id": receipt,
                                               "outputs": {"model.safetensors": 4}})
    mining = write_round(dataset["root"], 2, "corpus-receipt-1", dataset["digests"],
                         encoder_identity_for(model=receipt, revision=None), seed=7)
    assert not dash.refresh_round() and dash.round.number == 1
    assert "fitted TrialSpace trialspace_round1 is incomplete" in dash.skipped_rounds[0]["reason"]
    (fitted / "model.safetensors").write_bytes(b"1234")
    whole = (mining / "patients.npy").read_bytes()
    (mining / "patients.npy").write_bytes(whole[:-8])
    assert not dash.refresh_round() and "still being copied" in dash.skipped_rounds[0]["reason"]
    (mining / "patients.npy").write_bytes(whole)
    assert dash.refresh_round() and dash.round.number == 2 and dash.skipped_rounds == [] and dash.round_switched_at
    assert np.array_equal(dash.patient_vector(dash.patient_entity(PATIENTS[0][0])), np.load(mining / "patients.npy")[0])
    assert not dash.refresh_round() and dash.round.number == 2
    deadline = time.monotonic() + 10  # the trial cohorts are re-embedded for round 2 in the background
    while any(job.status in ("queued", "running") for job in list(dash.background.jobs.values())):
        assert time.monotonic() < deadline
        time.sleep(0.01)
    assert dash.cohort_vectors.identity == dash.round.encoder
    assert dash.cohort_vectors.vector(dash.store, "NCT90000001-cohort-0") is not None


def test_store_is_owner_only(dash, dash_settings):
    assert dash_settings.store_path.stat().st_mode & 0o077 == 0
    assert json.loads(dash_settings.store_path.read_text())["format"] == "mmai-dash-store-v2"


def write_interim(root: Path, step: int, *, complete: bool = True) -> Path:
    """A fabricated interim export (step-<N>) with interim.json listing its files' sizes."""
    from matchminer_ai.apps.dashboard.settings import INTERIM_DECISION_MODELS
    folder = root / INTERIM_DECISION_MODELS / f"step-{step}"
    folder.mkdir(parents=True)
    files = {"config.json": b"{}", "decision_contract.json": b'{"interim": true}', "model.safetensors": b"weights"}
    for name, data in files.items():
        (folder / name).write_bytes(data)
    outputs = {name: len(data) + (0 if complete else 1) for name, data in files.items()}
    write_json(folder / "interim.json", {"format": "mmai-decision-interim-v1", "interim": True, "checkpoint_step": step,
                                         "max_steps": 66136, "epoch": step / 66136, "outputs": outputs})
    return folder


def test_decision_model_prefers_final_then_newest_interim_then_fit2(dataset, tmp_path):
    from matchminer_ai.apps.dashboard.settings import DashSettings

    s = DashSettings(synthetic_run=dataset["root"], store_dir=tmp_path / "store", device="cpu")
    root = s.synthetic_run
    assert "fallback" in s.resolved_decision_model()[2]  # nothing yet: base d1-2B
    smoke = root / "smoke_tests/round1_labels_fit2_20261107/decision_fit/model"
    smoke.mkdir(parents=True)
    (smoke / "config.json").write_text("{}")
    assert s.resolved_decision_model()[0] == str(smoke)
    write_interim(root, 1000, complete=False)  # still copying: sizes differ from interim.json
    assert s.resolved_decision_model()[0] == str(smoke)
    first = write_interim(root, 9000)
    write_interim(root, 10000, complete=False)
    model, revision, why = s.resolved_decision_model()
    assert model == str(first) and revision is None
    assert "INTERIM" in why and "9,000 of 66,136" in why and "not trained yet" in why
    newer = write_interim(root, 12000)
    assert s.resolved_decision_model()[0] == str(newer)
    run = root / "trialspace/models/decision_model"  # the final model replaces the interim ones once complete
    run.mkdir(parents=True)
    files = {"decision_contract.json": b"{}", "model.safetensors": b"weights"}
    for name, data in files.items():
        (run / name).write_bytes(data)
    assert s.resolved_decision_model()[0] == str(newer)  # no stage receipt yet
    write_json(run / "stage_receipt.json", {"format": "mmai-stage-receipt-v1", "kind": "decision-model",
                                            "outputs": {name: len(data) for name, data in files.items()}})
    model, _, why = s.resolved_decision_model()
    assert model == str(run) and "trained decision model" in why
    explicit = DashSettings(synthetic_run=root, store_dir=tmp_path / "store", device="cpu", decision_model=str(smoke))
    assert explicit.resolved_decision_model()[0] == str(smoke)


def test_decision_model_device_tries_gpu0(monkeypatch, tmp_path):
    from matchminer_ai.apps.dashboard import settings as dash_settings_module
    from matchminer_ai.apps.dashboard.settings import DashSettings, parse_settings

    monkeypatch.setattr(dash_settings_module, "cuda_available", lambda: True)
    s = DashSettings(store_dir=tmp_path / "store")
    assert s.device == "cuda:0" and s.device_notice is None
    assert parse_settings(["--store-dir", str(tmp_path / "store"), "--device", "cpu"]).device == "cpu"
    monkeypatch.setenv("MMAI_DASH_DEVICE", "cuda:1")
    assert parse_settings(["--store-dir", str(tmp_path / "store")]).device == "cuda:1"
    monkeypatch.delenv("MMAI_DASH_DEVICE")
    monkeypatch.setattr(dash_settings_module, "cuda_available", lambda: False)
    s = parse_settings(["--store-dir", str(tmp_path / "store")])
    assert s.device == "cpu" and "CUDA is not available" in s.device_notice
    assert s.embedding_device == "cpu"  # the TrialSpace encoder's device is unchanged

    def broken():
        raise RuntimeError("no driver")

    monkeypatch.setattr(dash_settings_module, "cuda_available", broken)
    s = DashSettings(store_dir=tmp_path / "store")
    assert s.device == "cpu" and "Could not check for CUDA" in s.device_notice
