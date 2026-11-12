"""Extraction-based workup checks: query language, evaluator, compile loop and preset.

Every record, note and requirement here is fabricated; no clinical data is used.
"""

import copy
import json

import pandas as pd
import pytest

from matchminer_ai.config import load_default_preset
from matchminer_ai.llm.backends import LLMGenerationResult, build_llm_runtime_config
from matchminer_ai.patients import (
    check_workup_with_extractions,
    compile_workup_queries,
    validate_workup_query,
)
from matchminer_ai.patients.extraction.notes import note_ids
from matchminer_ai.patients.workup_queries import language as lang
from matchminer_ai.patients.workup_queries.compile import (
    render_compile_prompt,
    workup_query_cache_key,
)
from matchminer_ai.patients.workup_queries.evaluate import stage_group
from matchminer_ai.population import (
    check_population_workup_with_extractions,
    create_patient_cohort,
    load_patient_cohort,
    population_results_table,
    save_patient_cohort,
)

MODEL = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4"
EXAMPLES = dict(lang.EXAMPLES)


# Fabricated records ---------------------------------------------------------


def when(value, source="document_text"):
    if value is None:
        return {"value": None, "precision": "unknown", "source": "unknown"}
    return {"value": value, "precision": {4: "year", 7: "month", 10: "day"}[len(value)],
            "source": source}


def quote(note_id, note_date, text):
    return {"note_id": note_id, "note_date": note_date, "quote": text, "start": 3,
            "end": 3 + len(text)}


def study(note_id, note_date, *, relation="this_report", modality="ct", regions=("chest",),
          contrast="with_contrast", date="same", text="CT CHEST"):
    if date == "same":
        date = when(note_date, "note_date") if relation == "this_report" else when(None)
    elif not isinstance(date, dict):
        date = when(date)
    return {
        "item_type": "imaging_study", "relation": relation,
        "source_of_document": relation == "this_report",
        "status": "planned" if relation == "planned" else "performed",
        "modality": modality, "modality_text": text, "body_regions": list(regions),
        "body_site_text": "", "contrast": contrast, "tracer": "", "study_date": date,
        "evidence": [quote(note_id, note_date, text)],
    }


def procedure(note_id, note_date, kind="core_needle_biopsy", date="2024-03-08"):
    groups = {"core_needle_biopsy": "histologic_biopsy", "fine_needle_aspiration":
              "aspiration_or_cytology", "not_stated": None}
    return {
        "item_type": "procedure", "relation": "this_report", "source_of_document": True,
        "status": "performed", "procedure_type": kind, "procedure_group": groups[kind],
        "procedure_text": kind.replace("_", " "), "image_guidance": "ct",
        "specimen_site": "right upper lobe", "laterality": "right",
        "procedure_date": when(date), "specimen_result": "malignant",
        "result_text": "adenocarcinoma", "evidence": [quote(note_id, note_date, "biopsy")],
    }


def diagnosis(note_id, note_date, stage="IIIA", date="2024-01-10"):
    return {
        "item_type": "diagnosis", "primary_site": "lung", "histology": "adenocarcinoma",
        "grade": "", "diagnosis_date": when(date), "stage_group": stage, "t_category": "",
        "n_category": "", "m_category": "", "stage_basis": "clinical", "staging_system": "",
        "extent_at_diagnosis": "not_stated", "evidence": [quote(note_id, note_date, "stage")],
    }


def biomarker(note_id, note_date, name, result="negative"):
    return {
        "item_type": "biomarker", "biomarker": name, "biomarker_type": "mutation",
        "result": result, "alteration": "", "value": "", "method": "ngs", "assay_name": "",
        "sample_type": "tissue", "origin": "somatic", "test_date": when(None),
        "evidence": [quote(note_id, note_date, name)],
    }


def treatment(note_id, note_date, start="2024-04-01", status="ongoing"):
    return {
        "item_type": "treatment", "treatment_type": "systemic_therapy",
        "regimen": "carboplatin/pemetrexed", "agents": ["carboplatin", "pemetrexed"],
        "site": "", "setting": "advanced_or_metastatic", "line_of_therapy": 1,
        "treatment_status": status, "start_date": when(start), "end_date": when(None),
        "best_response": "not_stated", "end_reason": "", "toxicities": [],
        "evidence": [quote(note_id, note_date, "carboplatin")],
    }


KEYS = {"imaging_study": "imaging_studies", "procedure": "procedures", "diagnosis": "diagnoses",
        "biomarker": "biomarkers", "treatment": "treatments"}


def document(note_id, note_date, items=(), *, patient="p1", kind="clinic_note", status="complete"):
    lists = {key: [] for key in (*KEYS.values(), "disease_assessments", "performance_status")}
    for item in items:
        lists[KEYS[item["item_type"]]].append(item)
    record = {
        "schema_version": "mmai-clinical-extraction-v1", "level": "document",
        "patient_id": patient, "note_id": note_id, "note_date": note_date, "note_type": None,
        "document_type": kind, "document_type_text": "", "extensions": [], "items": lists,
    }
    return {"patient_id": patient, "note_id": note_id, "note_date": note_date, "note_type": None,
            "document_type": kind, "extraction_status": status, "error": None,
            "item_count": len(items), "record": None if status == "failed" else record}


def compiled(*queries):
    return [{"status": "compiled", "query": q, "cache_key": f"k{i}",
             "requirement": {"name": f"Requirement {i}", "conditions": "", "category": ""}}
            for i, q in enumerate(queries)]


def criterion(cid, item_type, filters, *, status="performed", anchor="any_time", target="",
              before=-1, after=-1, fallback="unknown", count=1, label=None):
    return {
        "id": cid, "label": label or cid, "item_type": item_type, "status": status,
        "filters": [{"field": f, "op": o, "values": v, "if_not_stated": n}
                    for f, o, v, n in filters],
        "window": {"anchor": anchor, "anchor_criterion": target, "days_before": before,
                   "days_after": after, "if_no_anchor": fallback},
        "min_count": count,
    }


def query(criteria, requirement, applies=(), excluded=()):
    def expression(groups):
        return {"all_of": [{"any_of": [{"criterion": c, "negated": n} for c, n in g]}
                           for g in groups]}

    return {"summary": "Fabricated query.", "expressible": True, "not_expressible_reason": "",
            "approximations": [], "criteria": criteria,
            "applies_when": expression(applies), "not_applicable_when": expression(excluded),
            "requirement": expression(requirement)}


CT = [("modality", "in", ["ct"], "no_match"), ("body_regions", "contains_any", ["chest"], "unknown")]


def ct_query(**window):
    return query([criterion("ct", "imaging_study", CT, **window)], [[("ct", False)]])


def run(rows, *queries, **kwargs):
    report = check_workup_with_extractions(pd.DataFrame(rows), compiled(*queries), **kwargs)
    return report["patients"][0]["review"]["assessments"]


# Query JSON schema and validation -------------------------------------------


def _objects(schema):
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            yield schema
        for value in schema.values():
            yield from _objects(value)
    elif isinstance(schema, list):
        for value in schema:
            yield from _objects(value)


def test_query_schema_is_strict_non_recursive_and_decodable():
    schema = lang.query_response_schema()
    for node in _objects(schema):
        assert node["additionalProperties"] is False
        assert node["required"] == list(node["properties"])
    text = json.dumps(schema)
    for token in ("minLength", "pattern", '"null"', "$ref", "$defs", "anyOf"):
        assert token not in text
    criterion_schema = schema["properties"]["criteria"]["items"]["properties"]
    assert "imaging_study" in criterion_schema["item_type"]["enum"]
    assert set(criterion_schema["filters"]["items"]["properties"]["op"]["enum"]) == set(lang.OPS)
    assert "smoking_history" in json.dumps(lang.query_response_schema(["thoracic"]))
    assert "smoking_history" not in text


def test_prompt_examples_are_valid_queries():
    for text, example in lang.EXAMPLES:
        assert validate_workup_query(example) == [], text


@pytest.mark.parametrize("change, expected", [
    (lambda q: q.pop("summary"), "missing fields ['summary']"),
    (lambda q: q.update(extra=1), "unknown fields ['extra']"),
    (lambda q: q["criteria"][0]["filters"][0].update(field="stage_group"),
     "'stage_group' is not a field of imaging_study"),
    (lambda q: q["criteria"][0]["filters"][0].update(op="text_contains_any"),
     "text_contains_any does not apply to modality"),
    (lambda q: q["criteria"][0]["filters"][0].update(values=["cat_scan"]),
     "['cat_scan'] are not modality codes"),
    (lambda q: q["criteria"][0].update(item_type="scan"), "item_type: must be one of"),
    (lambda q: q["criteria"][0].update(id="CT Chest"), "use 1-24 lowercase letters"),
    (lambda q: q["criteria"][0]["window"].update(days_before=30), "any_time takes no day limits"),
    (lambda q: q["criteria"][0]["window"].update(anchor="criterion_earliest",
                                                 anchor_criterion="ct_chest"),
     "cannot anchor its own window"),
    (lambda q: q["requirement"]["all_of"][0]["any_of"][0].update(criterion="mri"),
     "'mri' is not a criterion id"),
    (lambda q: [lit.update(negated=True) for group in q["requirement"]["all_of"]
                for lit in group["any_of"]],
     "needs at least one criterion that is not negated"),
    (lambda q: q["requirement"].update(all_of=[]), "needs at least one group"),
    (lambda q: q["not_applicable_when"].update(
        all_of=[{"any_of": [{"criterion": "ct_chest", "negated": True}]}]),
     "not_applicable_when: needs a documented fact"),
    (lambda q: q.update(expressible=False), "criteria: must be [] when expressible is false"),
    (lambda q: q["criteria"][0].update(min_count=0), "min_count: must be an integer from 1"),
    (lambda q: q["criteria"][0]["filters"][0].update(if_not_stated="maybe"), "if_not_stated"),
])
def test_validation_names_every_problem(change, expected):
    value = copy.deepcopy(EXAMPLES["CT chest/abdomen/pelvis (one combined study or separate "
                                   "studies)"])
    change(value)
    problems = validate_workup_query(value)
    assert any(expected in p for p in problems), problems


def test_a_fourteen_marker_panel_fits_the_schema():
    markers = ["EGFR", "ALK", "KRAS", "ROS1", "BRAF", "NTRK1", "NTRK2", "NTRK3", "MET", "RET",
               "ERBB2", "NRG1", "HER2", "c-Met"]
    criteria = [criterion(f"m{i}", "biomarker", [("biomarker", "text_contains_any", [m],
                                                  "no_match")], status="any")
                for i, m in enumerate(markers)]
    panel = query(criteria, [[(f"m{i}", False)] for i in range(len(markers))])
    assert validate_workup_query(panel) == []
    schema = lang.query_response_schema()["properties"]
    assert schema["requirement"]["properties"]["all_of"]["maxItems"] >= len(markers)
    assert schema["criteria"]["maxItems"] >= len(markers)
    assert "up to 24 named matchers" in render_compile_prompt(REQUIREMENT)


def test_validation_of_references_values_and_types():
    q = query([
        criterion("a", "imaging_study", CT, anchor="criterion_earliest", target="b"),
        criterion("b", "treatment", [], anchor="criterion_latest", target="a"),
        criterion("c", "biomarker", [("biomarker", "is_stated", ["x"], "unknown")],
                  status="planned"),
        criterion("d", "diagnosis", [("stage_group", "stage_group_in", ["3"], "unknown"),
                                     ("grade", "stage_group_in", ["III"], "unknown")]),
        criterion("e", "treatment", [("line_of_therapy", "number_at_least", ["two"], "unknown")]),
        criterion("unused", "procedure", []),
    ], [[("a", False), ("c", False), ("d", False), ("e", False)]])
    problems = "\n".join(validate_workup_query(q))
    assert "anchor criteria form a cycle" in problems
    assert "is_stated takes no values" in problems
    assert "biomarker items are never planned" in problems
    assert "stage groups must be among" in problems
    assert "stage_group_in does not apply to grade" in problems
    assert "takes exactly one integer" in problems
    assert "'unused': not used" in problems
    assert validate_workup_query("not a query") == ["query: must be an object"]


# Evaluator --------------------------------------------------------------------


def test_day_month_and_year_dates_against_a_window():
    window = dict(anchor="reference_date", before=30, after=0)
    rows = [document("n1", "2024-03-02", [study("n1", "2024-03-02")], kind="imaging_report")]
    assert run(rows, ct_query(**window), reference_dates="2024-03-20")[0]["status"] == "met"
    assert run(rows, ct_query(**window), reference_dates="2024-05-20")[0]["status"] == "not_met"
    month = [document("n1", "2024-03-15", [study("n1", "2024-03-15", date="2024-03")])]
    inside = run(month, ct_query(anchor="reference_date", before=60, after=0),
                 reference_dates="2024-04-30")[0]
    partial = run(month, ct_query(**window), reference_dates="2024-04-10")[0]
    assert inside["status"] == "met"
    assert partial["status"] == "unknown" and partial["reason"] == "uncertain_dates"
    assert "month-precision date" in partial["bottom_line"]
    year = [document("n1", "2024-06-01", [study("n1", "2024-06-01", date="2024")])]
    assert run(year, ct_query(**window), reference_dates="2024-06-20")[0]["status"] == "unknown"
    assert run(year, ct_query(), reference_dates="2024-06-20")[0]["status"] == "met"
    # The reference date defaults to the latest note date.
    assert run(rows, ct_query(**window))[0]["reference_date"] == "2024-03-02"


def test_previously_performed_mentions_are_bounded_by_their_note_and_deduplicated():
    mention = document("n2", "2024-03-10",
                       [study("n2", "2024-03-10", relation="previously_performed", date=None)])
    report = document("n1", "2024-03-02", [study("n1", "2024-03-02")], kind="imaging_report")
    alone = run([mention], ct_query(anchor="reference_date", before=30, after=0),
                reference_dates="2024-03-10")[0]
    assert alone["status"] == "unknown" and alone["reason"] == "uncertain_dates"
    assert "undated mention" in alone["bottom_line"]
    assert run([mention], ct_query())[0]["status"] == "met"
    # One study documented by its report and mentioned in a later note counts once.
    twice = query([criterion("ct", "imaging_study", CT, count=2)], [[("ct", False)]])
    result = run([report, mention], twice, ct_query())
    assert result[0]["status"] == "not_met"
    entry = result[1]["criteria"][0]
    assert entry["definite_count"] == 1 and entry["matches"][0]["mentions"] == 2
    assert entry["matches"][0]["relations"] == ["previously_performed", "this_report"]
    assert {e["note_id"] for e in result[1]["evidence"]} == {"n1", "n2"}
    # Two different studies are two.
    later = document("n3", "2024-05-02", [study("n3", "2024-05-02")], kind="imaging_report")
    assert run([report, mention, later], twice)[0]["status"] == "met"


def test_the_report_decides_fields_a_mention_contradicts():
    report = document("n1", "2024-03-02", [study("n1", "2024-03-02", contrast="without_contrast")])
    mention = document("n2", "2024-03-10", [study("n2", "2024-03-10", contrast="not_stated",
                                                  relation="previously_performed", date=None)])
    contrast = query([criterion("ct", "imaging_study", [
        *CT, ("contrast", "in", ["with_contrast"], "unknown")])], [[("ct", False)]])
    assert run([report, mention], contrast)[0]["status"] == "not_met"
    unstated = run([mention], contrast)[0]
    assert unstated["status"] == "unknown" and unstated["reason"] == "fields_not_stated"
    strict = query([criterion("ct", "imaging_study", [
        *CT, ("contrast", "in", ["with_contrast"], "no_match")])], [[("ct", False)]])
    assert run([mention], strict)[0]["status"] == "not_met"


def test_planned_studies_are_not_performed():
    plan = document("n1", "2024-03-02", [study("n1", "2024-03-02", relation="planned",
                                               date=None)])
    result = run([plan], ct_query())[0]
    assert result["status"] == "not_met" and result["reason"] == "planned_only"
    assert "Planned or ordered" in result["bottom_line"]
    assert result["evidence"][0]["state"] == "planned"


def test_and_or_not_combinations():
    chest = document("n1", "2024-03-02", [study("n1", "2024-03-02")], kind="imaging_report")
    abdomen = document("n2", "2024-03-02", [study("n2", "2024-03-02",
                                                  regions=("abdomen", "pelvis"))])
    cap = EXAMPLES["CT chest/abdomen/pelvis (one combined study or separate studies)"]
    assert run([chest, abdomen], cap)[0]["status"] == "met"
    missing = run([chest], cap)[0]
    assert missing["status"] == "not_met" and "no CT abdomen found" in missing["bottom_line"]
    biopsy = EXAMPLES["Tissue diagnosis by core needle or excisional biopsy"]
    fna = document("n3", "2024-03-09", [procedure("n3", "2024-03-09", "fine_needle_aspiration")])
    core = document("n4", "2024-03-09", [procedure("n4", "2024-03-09")])
    assert run([fna], biopsy)[0]["status"] == "not_met"
    assert run([fna, core], biopsy)[0]["status"] == "met"
    unknown = document("n5", "2024-03-09", [procedure("n5", "2024-03-09", "not_stated")])
    assert run([unknown], biopsy)[0]["status"] == "unknown"
    # NOT: a CT with no FNA documented.
    neither = query([criterion("ct", "imaging_study", CT),
                     criterion("fna", "procedure", [("procedure_type", "in",
                                                     ["fine_needle_aspiration"], "no_match")])],
                    [[("ct", False)], [("fna", True)]])
    assert run([chest], neither)[0]["status"] == "met"
    assert run([chest, fna], neither)[0]["status"] == "not_met"


def test_applicability_needs_documented_facts():
    mri = EXAMPLES["Brain MRI (or head CT with contrast if MRI is not possible) if stage "
                   "III-IV, before systemic therapy starts"]
    brain = study("n2", "2024-03-20", modality="mri", regions=("brain",))
    late = document("n1", "2024-03-01", [diagnosis("n1", "2024-03-01", "IIIA")])
    early = document("n1", "2024-03-01", [diagnosis("n1", "2024-03-01", "Stage IB")])
    unstaged = document("n1", "2024-03-01", [diagnosis("n1", "2024-03-01", "")])
    scan = document("n2", "2024-03-20", [brain], kind="imaging_report")
    treated = document("n3", "2024-04-02", [treatment("n3", "2024-04-02", start="2024-04-01")])
    assert run([late, scan, treated], mri)[0]["status"] == "met"
    result = run([early, scan], mri)[0]
    assert result["status"] == "not_applicable" and result["applicability"] == "not_applicable"
    assert "Stage IB" in result["bottom_line"]
    result = run([unstaged], mri)[0]
    assert result["status"] == "not_met" and result["applicability"] == "uncertain"
    assert "Applicability uncertain" in result["bottom_line"]
    # The MRI must come before the first systemic therapy.
    started = document("n3", "2024-03-01", [treatment("n3", "2024-03-01", start="2024-02-15")])
    assert run([late, scan, started], mri)[0]["status"] == "not_met"
    # Without treatment the reference date (latest note) bounds the window.
    assert run([late, scan], mri)[0]["status"] == "met"
    entry = run([late, scan], mri)[0]["criteria"][0]
    assert entry["anchor"]["fallback"] == "reference_date"
    # An undated treatment may have come first: unknown, never the reference date.
    undated = document("n3", "2024-04-02", [treatment("n3", "2024-04-02", start=None)])
    result = run([late, scan, undated], mri)[0]
    assert result["status"] == "unknown" and "fallback" not in result["criteria"][0]["anchor"]


def test_anchor_without_fallback_is_unknown_and_diagnosis_anchor():
    at_diagnosis = query([criterion("ct", "imaging_study", CT, anchor="diagnosis_date",
                                    before=30, after=30)], [[("ct", False)]])
    scan = document("n2", "2024-02-01", [study("n2", "2024-02-01")], kind="imaging_report")
    result = run([scan], at_diagnosis)[0]
    assert result["status"] == "unknown" and result["reason"] == "anchor_unavailable"
    dx = document("n1", "2024-01-20", [diagnosis("n1", "2024-01-20", date="2024-01-15")])
    assert run([dx, scan], at_diagnosis)[0]["status"] == "met"
    month = document("n1", "2024-01-20", [diagnosis("n1", "2024-01-20", date="2024-01")])
    assert run([month, scan], at_diagnosis)[0]["status"] == "unknown"
    earlier = document("n1", "2024-01-20", [diagnosis("n1", "2024-01-20", date="2023-11")])
    assert run([earlier, scan], at_diagnosis)[0]["status"] == "not_met"
    report = check_workup_with_extractions(pd.DataFrame([scan]), compiled(at_diagnosis),
                                           anchors={"p1": {"diagnosis_date": "2024-01-25"}})
    assert report["patients"][0]["review"]["assessments"][0]["status"] == "met"


def test_text_matching_stage_parsing_and_counts():
    ok = document("n1", "2024-03-01", [biomarker("n1", "2024-03-01", "EGFR"),
                                       biomarker("n1", "2024-03-01", "PD L1", "positive")])
    alkaline = document("n1", "2024-03-01", [biomarker("n1", "2024-03-01",
                                                       "alkaline phosphatase")])
    markers = EXAMPLES["Biomarker testing including EGFR and ALK"]
    assert run([alkaline], markers)[0]["status"] == "not_met"
    pdl1 = query([criterion("pdl1", "biomarker", [("biomarker", "text_contains_any", ["PD-L1"],
                                                   "no_match")], status="any")],
                 [[("pdl1", False)]])
    assert run([ok], pdl1)[0]["status"] == "met"
    assert [stage_group(s) for s in ("IIIA", "Stage IVB", "4", "pathologic stage IIA", "IB",
                                     "extensive stage", "", "0")] == [
        "III", "IV", "IV", "II", "I", None, None, "0"]


def test_incomplete_or_missing_extraction_is_unknown_and_unexpressible_requirements():
    failed = document("n9", "2024-03-05", status="failed")
    result = run([failed], ct_query())[0]
    assert result["status"] == "unknown" and result["reason"] == "incomplete_extraction"
    report = check_workup_with_extractions(
        pd.DataFrame([document("n1", "2024-03-02", [])]),
        compiled(ct_query(), EXAMPLES["History and physical examination; CBC and chemistry "
                                      "profile"]),
        patient_ids=["p1", "p2"], coverage={"p1": {"notes_without_records": 2}},
    )
    first, second = report["patients"]
    assert first["review"]["assessments"][0]["reason"] == "incomplete_extraction"
    assert first["review"]["assessments"][1]["reason"] == "not_expressible"
    assert second["review"]["assessments"][0]["reason"] == "no_extraction"
    assert first["review"]["metadata"]["coverage"]["notes_without_records"] == 2
    aggregate = report["aggregate"][0]
    assert aggregate["statuses"] == {"met": 0, "not_met": 0, "unknown": 2, "not_applicable": 0}
    assert aggregate["documented_completion_fraction"] is None
    failed_query = [{"status": "failed", "query": None, "requirement": {"name": "x"}}]
    result = check_workup_with_extractions(pd.DataFrame([failed]), failed_query)
    assert result["patients"][0]["review"]["assessments"][0]["reason"] == "query_unavailable"


def test_patient_level_items_and_evidence_note_dates():
    patient = {
        "patient_id": "p1", "extraction_status": "complete", "failed_chunks": [],
        "record": {
            "schema_version": "mmai-clinical-extraction-v1", "level": "patient",
            "patient_id": "p1", "note_count": 2, "first_note_date": "2024-01-01",
            "last_note_date": "2024-03-10", "chunk_count": 1, "failed_chunks": [],
            "extensions": [],
            "items": {"imaging_studies": [{
                **study("n2", "2024-03-10", date=when("2024-03-10", "evidence_note_date")),
                "item_id": "I1", "status": "performed"}]},
        },
    }
    for key in ("relation", "source_of_document"):
        patient["record"]["items"]["imaging_studies"][0].pop(key)
    result = run([patient], ct_query(anchor="reference_date", before=30, after=0))[0]
    assert result["status"] == "unknown" and "on or before 2024-03-10" in result["bottom_line"]
    assert run([patient], ct_query())[0]["status"] == "met"


def test_evaluation_is_deterministic_and_does_no_io(monkeypatch):
    import builtins
    import socket

    def refuse(*args, **kwargs):
        raise AssertionError("the evaluator must not do I/O")

    rows = [document("n1", "2024-03-02", [study("n1", "2024-03-02")], kind="imaging_report")]
    first = run(rows, *[q for _, q in lang.EXAMPLES])
    monkeypatch.setattr(builtins, "open", refuse)
    monkeypatch.setattr(socket, "socket", refuse)
    assert run(rows, *[q for _, q in lang.EXAMPLES]) == first


def test_population_check_links_records_to_cohort_notes(tmp_path):
    summaries = pd.DataFrame({"patient_id": ["p1", "p2"],
                              "cancer_history_summary": ["Fabricated A", "Fabricated B"]})
    notes = pd.DataFrame({
        "patient_id": ["p1", "p1", "p2"],
        "note_text": ["CT chest done.", "Clinic visit.", "Clinic visit."],
        "note_date": ["2024-03-02", "2024-03-10", "2024-03-10"],
        "note_type": ["Imaging", "Visit", "Visit"],
    })
    cohort = create_patient_cohort(summaries, notes)
    ids = dict(zip(zip(notes.patient_id, notes.note_text), note_ids(notes), strict=True))
    rows = [
        document(ids["p1", "CT chest done."], "2024-03-02",
                 [study(ids["p1", "CT chest done."], "2024-03-02")], kind="imaging_report"),
        document(ids["p1", "Clinic visit."], "2024-03-10"),
        document("note-stale", "2024-01-01", [study("note-stale", "2024-01-01")], patient="p2"),
    ]
    space = {"space_trial_id": "s", "name": "Fabricated", "diagnostic_workup": [
        {"name": "CT chest", "conditions": "", "category": "2A", "evidence": []}]}
    from dataclasses import replace

    cohort = replace(cohort, space=space)
    checked = check_population_workup_with_extractions(
        cohort, compiled(ct_query()), document_records=pd.DataFrame(rows))
    report = checked.report
    assert report["method"] == "extraction_queries"
    assert report["extraction"]["documents_ignored"] == 1
    first, second = (p["review"]["assessments"][0] for p in report["patients"])
    assert first["status"] == "met" and second["reason"] == "no_extraction"
    assert report["aggregate"][0]["statuses"]["met"] == 1
    table = population_results_table(checked)
    assert list(table.status) == ["met", "unknown"]
    restored = load_patient_cohort(save_patient_cohort(checked, tmp_path / "cohort.zip"))
    assert restored.report["aggregate"] == report["aggregate"]


# Compile step -----------------------------------------------------------------


class FakeBackend:
    """Answers compile requests from a script, in order, and records every prompt."""

    def __init__(self, answers):
        self.answers, self.calls = list(answers), []

    def generate_llm_outputs(self, *, prompt_list, llm_config, **kwargs):
        self.calls.append((prompt_list, llm_config))
        texts, finishes = [], []
        for _ in prompt_list:
            answer = self.answers.pop(0)
            text, finish = answer if isinstance(answer, tuple) else (answer, "stop")
            texts.append(json.dumps(text) if isinstance(text, dict) else text)
            finishes.append(finish)
        return LLMGenerationResult(final_outputs=texts, finish_reasons=finishes,
                                   reasoning_outputs=[""] * len(texts), raw_outputs=[],
                                   model_metadata={"model_sha": "test"})


@pytest.fixture
def config():
    config = load_default_preset()
    config.remote["enabled"] = True
    return config


def install(monkeypatch, backend):
    monkeypatch.setattr("matchminer_ai.patients.workup_queries.compile.get_llm_backend",
                        lambda _config: backend)


REQUIREMENT = {"name": "CT chest with contrast", "conditions": "At diagnosis {field_guide}",
               "category": "Category 2A", "evidence": [{"page_id": "x", "line_ids": [1]}]}


def test_compile_prompt_renders_requirement_population_and_data_model():
    prompt = render_compile_prompt(REQUIREMENT, "Inclusion criteria:\nFabricated cancer")
    assert "Name: CT chest with contrast" in prompt
    # One-pass filling: braces in the requirement text stay literal.
    assert "Conditions: At diagnosis {field_guide}" in prompt
    assert "Inclusion criteria:\nFabricated cancer" in prompt
    assert "previously_performed: An earlier or outside study" in prompt
    assert "core_needle_biopsy: Core needle biopsy" in prompt
    assert "stage_group_in" in prompt and "if_not_stated" in prompt
    assert json.dumps(lang.EXAMPLES[0][1], ensure_ascii=False) in prompt
    assert "{" not in prompt.split("REQUIREMENT")[0].replace("{\"", "").replace("{}", "")[-50:]


def test_compile_retries_malformed_and_invalid_queries_then_caches(config, monkeypatch):
    good = EXAMPLES["CT chest with contrast within 30 days before the reference date"]
    bad = copy.deepcopy(good)
    bad["criteria"][0]["filters"][0]["field"] = "histology"
    backend = FakeBackend([("{not json", "stop"), bad, good])
    install(monkeypatch, backend)
    cache: dict = {}
    seen = []
    entries, metadata = compile_workup_queries(
        [REQUIREMENT, dict(REQUIREMENT, evidence=[])], config=config,
        population_context="Fabricated population", cache=cache, return_metadata=True,
        progress_callback=lambda done, total: seen.append((done, total)),
    )
    assert len(backend.calls) == 3  # the duplicate requirement is compiled once
    assert [e["status"] for e in entries] == ["compiled", "compiled"]
    entry = entries[0]
    assert entry["attempt_count"] == 3 and not entry["first_attempt_valid"]
    assert "not valid JSON" in entry["attempts"][0]["problems"][0]
    assert any("'histology' is not a field" in p for p in entry["attempts"][1]["problems"])
    assert entry["query"] == lang.canonical_query(good)
    second_prompt = backend.calls[1][0][0].messages
    assert "not accepted" in second_prompt[-1]["content"]
    third_prompt = backend.calls[2][0][0].messages
    assert third_prompt[-2]["role"] == "assistant" and "histology" in third_prompt[-1]["content"]
    runtime = backend.calls[0][1]
    assert runtime["sampling_params"]["response_format"]["json_schema"]["schema"] == \
        lang.query_response_schema()
    assert metadata["requests"] == 3 and metadata["first_attempt_valid"] == 0
    assert seen[-1] == (1, 1)
    key = workup_query_cache_key(REQUIREMENT, "Fabricated population")
    assert set(cache) == {key} and entry["cache_key"] == key
    # Cached: no further requests; a changed population is a new key.
    again = compile_workup_queries([REQUIREMENT], config=config,
                                   population_context="Fabricated population", cache=cache)
    assert again[0]["cached"] and len(backend.calls) == 3
    assert workup_query_cache_key(REQUIREMENT, "Other population") != key


def test_compile_fails_after_every_attempt_with_thinking_off_last(config, monkeypatch):
    bad = copy.deepcopy(EXAMPLES["Tissue diagnosis by core needle or excisional biopsy"])
    bad["criteria"][0]["filters"][0]["values"] = ["needle"]
    backend = FakeBackend([bad, bad, bad, ("", "length")])
    install(monkeypatch, backend)
    cache: dict = {}
    entry = compile_workup_queries([REQUIREMENT], config=config, cache=cache)[0]
    assert entry["status"] == "failed" and entry["query"] is None
    assert entry["attempt_count"] == 4 and "no valid query" in entry["error"]
    assert [a["thinking"] for a in entry["attempts"]] == [True, True, True, False]
    template = backend.calls[-1][1]["remote"]["extra_body"]["chat_template_kwargs"]
    assert template["enable_thinking"] is False
    assert cache == {}


# Preset -----------------------------------------------------------------------


def test_compile_preset_uses_nemotron_structured_outputs_and_room_to_reason():
    config = load_default_preset()
    section = config.workup_query_compile
    assert section["local"]["model_name"] == MODEL and section["remote"]["model_name"] == MODEL
    assert section["sampling_profile"] == "auto" and section["structured_outputs"] is True
    assert section["max_attempts"] == 3 and section["reasoning_off_fallback"] is True
    assert section["remote"]["request_params"]["max_tokens"] >= 32768
    assert section["remote"]["request_params"]["max_tokens"] < section["local"]["engine"][
        "max_model_len"]
    assert section["remote"]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True
    config.remote["enabled"] = True
    runtime = build_llm_runtime_config("workup_query_compile", {
        k: section[k] for k in ("local", "remote", "sampling_profile", "reasoning_effort")},
        config=config)
    assert runtime["remote"]["request_params"]["temperature"] == 1.0
    assert runtime["remote"]["extra_body"]["top_k"] == -1
