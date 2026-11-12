"""Structured clinical extraction: data model, prompts, validation and the two stage APIs.

All notes here are fabricated; no clinical data is used.
"""

import json
import re

import pandas as pd
import pytest

from matchminer_ai.config import load_default_preset
from matchminer_ai.llm.backends import LLMGenerationResult
from matchminer_ai.patients import (
    extract_document_records,
    extract_patient_records,
    extraction_items,
)
from matchminer_ai.patients.extraction import schema as s
from matchminer_ai.patients.extraction import vocabulary as v
from matchminer_ai.patients.extraction.common import fill, parse_answer
from matchminer_ai.patients.extraction.documents import check_document_answer
from matchminer_ai.patients.extraction.evidence import locate_quote
from matchminer_ai.patients.extraction.extensions import THORACIC

MODEL = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4"
CT_REPORT = (
    "EXAM: CT CHEST WITH IV CONTRAST\n"
    "INDICATION: Staging of {fabricated} lung mass.\n"
    "FINDINGS: 3.1 cm spiculated mass in the right upper lobe. No pleural effusion.\n"
    "IMPRESSION: Right upper lobe mass, suspicious for primary lung cancer."
)
PATH_REPORT = (
    "SPECIMEN: Lung, right upper lobe, CT-guided core needle biopsy.\n"
    "Date of procedure: 2024-03-08\n"
    "DIAGNOSIS: Adenocarcinoma, acinar predominant.\n"
    "PD-L1 (22C3) tumor proportion score: 60%."
)
CLINIC_NOTE = (
    "Oncology visit. ECOG 1. Outside PET/CT on 2024-02-20 showed an avid RUL mass.\n"
    "Plan: start carboplatin and pemetrexed."
)


def imaging(**changes):
    item = {
        "evidence": [{"quote": "CT CHEST WITH IV CONTRAST"}],
        "relation": "this_report",
        "modality": "ct",
        "modality_text": "CT chest with IV contrast",
        "body_regions": ["chest"],
        "body_site_text": "",
        "contrast": "with_contrast",
        "tracer": "",
        "study_date": "",
    }
    return {**item, **changes}


def procedure(**changes):
    item = {
        "evidence": [{"quote": "Lung, right upper lobe, CT-guided core needle biopsy."}],
        "relation": "this_report",
        "procedure_type": "core_needle_biopsy",
        "procedure_text": "CT-guided core needle biopsy",
        "image_guidance": "ct",
        "specimen_site": "right upper lobe",
        "laterality": "right",
        "procedure_date": "2024-03-08",
        "specimen_result": "malignant",
        "result_text": "Adenocarcinoma, acinar predominant",
    }
    return {**item, **changes}


def document_answer(document_type="imaging_report", **lists):
    answer = {"document_type": document_type, "document_type_text": ""}
    for item_type in s.item_types_for("document"):
        answer[item_type.key] = lists.get(item_type.key, [])
    return answer


class FakeBackend:
    """Answers each request with the next scripted answer for the first marker
    its prompt contains. A dict is sent as JSON; (text, finish_reason) sets both."""

    def __init__(self, script):
        self.script, self.calls = script, []

    def generate_llm_outputs(self, *, prompt_list, llm_config, **kwargs):
        self.calls.append((prompt_list, llm_config))
        answers = []
        for prompt in prompt_list:
            text = "\n".join(m["content"] for m in prompt.messages)
            marker = next(m for m in self.script if m in text)
            queue = self.script[marker]
            value = queue.pop(0) if len(queue) > 1 else queue[0]
            answer, finish = value if isinstance(value, tuple) else (value, "stop")
            answers.append((json.dumps(answer) if isinstance(answer, dict) else answer, finish))
        return LLMGenerationResult(
            final_outputs=[a for a, _ in answers], finish_reasons=[f for _, f in answers],
            reasoning_outputs=[""] * len(answers), raw_outputs=[],
            model_metadata={"model_sha": "test"},
        )


class WordTokenizer:
    """One token per word, with character offsets (as fast tokenizers give)."""

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        spans = [m.span() for m in re.finditer(r"\S+", text)]
        result = {"input_ids": list(range(len(spans)))}
        if return_offsets_mapping:
            result["offset_mapping"] = spans
        return result


@pytest.fixture
def config():
    config = load_default_preset()
    config.remote["enabled"] = True
    return config


def install(monkeypatch, backend):
    for module in ("documents", "patient_records"):
        monkeypatch.setattr(
            f"matchminer_ai.patients.extraction.{module}.get_llm_backend", lambda _c: backend
        )
    monkeypatch.setattr(
        "matchminer_ai.patients.extraction.patient_records.load_tokenizer",
        lambda _name: WordTokenizer(),
    )


# Data model and JSON schemas


def _objects(schema):
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            yield schema
        for value in schema.values():
            yield from _objects(value)
    elif isinstance(schema, list):
        for value in schema:
            yield from _objects(value)


def test_response_schemas_are_strict_and_decodable():
    for schema in (s.document_response_schema(), s.document_response_schema(linking=True),
                   s.patient_response_schema(THORACIC.item_types)):
        for node in _objects(schema):
            assert node["additionalProperties"] is False
            assert node["required"] == list(node["properties"])
        # vLLM's grammar admits no JSON escapes in a minLength string; nulls are not used.
        text = json.dumps(schema)
        assert "minLength" not in text and "pattern" not in text and '"null"' not in text
    document = s.document_response_schema()
    imaging_item = document["properties"]["imaging_studies"]["items"]
    assert imaging_item["properties"]["modality"]["enum"] == list(v.IMAGING_MODALITIES.codes)
    assert imaging_item["properties"]["relation"]["enum"] == list(v.DOCUMENT_RELATIONS.codes)
    assert list(imaging_item["properties"])[0] == "evidence" and "cancer_ids" not in imaging_item["properties"]
    linked = s.document_response_schema(linking=True)["properties"]["procedures"]["items"]
    assert "cancer_ids" in linked["properties"]
    patient = s.patient_response_schema(THORACIC.item_types)["properties"]
    assert "disease_assessments" not in patient and "events" in patient and "smoking_history" in patient
    item = patient["procedures"]["items"]["properties"]
    assert list(item)[:2] == ["item_id", "evidence"] and item["status"]["enum"] == ["performed", "planned"]
    assert set(item["evidence"]["items"]["properties"]) == {"note", "quote"}


def test_record_schema_is_versioned_and_dates_carry_precision():
    record = s.record_json_schema("document", THORACIC.item_types)
    assert record["properties"]["schema_version"]["const"] == s.SCHEMA_VERSION
    items = record["properties"]["items"]["properties"]
    assert {"imaging_studies", "procedures", "performance_status", "smoking_history"} <= set(items)
    study = items["imaging_studies"]["items"]["properties"]
    assert study["study_date"]["properties"]["precision"]["enum"] == ["day", "month", "year", "unknown"]
    assert {"source_of_document", "status", "relation"} <= set(study)
    assert "procedure_group" in items["procedures"]["items"]["properties"]
    evidence = study["evidence"]["items"]["properties"]
    assert set(evidence) == {"note_id", "note_date", "quote", "start", "end"}


def test_vocabularies_cover_imaging_studies_and_pathology_procedures():
    for code in ("ct", "mri", "pet_ct", "xray", "ultrasound", "bone_scan", "mammography"):
        assert code in v.IMAGING_MODALITIES.codes
    assert v.IMAGING_MODALITIES.other == "other" and v.IMAGING_MODALITIES.unknown == "not_stated"
    for code in ("chest", "abdomen", "pelvis", "brain", "whole_body"):
        assert code in v.BODY_REGIONS.codes
    groups = {
        "core_needle_biopsy": "histologic_biopsy",
        "fine_needle_aspiration": "aspiration_or_cytology",
        "ebus_tbna": "aspiration_or_cytology",
        "bronchoscopic_biopsy": "histologic_biopsy",
        "endoscopic_biopsy": "histologic_biopsy",
        "excisional_biopsy": "histologic_biopsy",
        "surgical_resection": "surgical_resection",
        "fluid_cytology": "aspiration_or_cytology",
        "bone_marrow_biopsy": "bone_marrow",
        "other": "other",
    }
    for code, group in groups.items():
        assert v.PATHOLOGY_PROCEDURES.group(code) == group
    assert v.PATHOLOGY_PROCEDURES.group("not_stated") is None
    with pytest.raises(ValueError, match="repeats a code"):
        v.Vocabulary("x", "", (v.Term("a", ""), v.Term("a", "")), other=None, unknown=None)


def test_dates_keep_their_precision():
    assert s.parse_date("2024-03-05") == ("2024-03-05", "day")
    assert s.parse_date("2024-3") == ("2024-03", "month")
    assert s.parse_date("2024") == ("2024", "year")
    assert s.parse_date("") == (None, "unknown")
    for bad in ("March 2024", "2024-02-30", "1850", "2024-13", "03/05/2024"):
        with pytest.raises(ValueError):
            s.parse_date(bad)
    assert s.resolved_date("", fallback="2024-04-01") == {
        "value": "2024-04-01", "precision": "day", "source": "note_date"}


# Prompts


def test_document_prompt_renders_guide_note_and_optional_linking(config, monkeypatch):
    backend = FakeBackend({"CT CHEST": [document_answer(imaging_studies=[imaging()])]})
    install(monkeypatch, backend)
    notes = pd.DataFrame([{"patient_id": "p1", "note_text": CT_REPORT, "note_date": "2024-03-01",
                           "note_type": "CT {chest}"}])
    extract_document_records(notes, config=config)
    content = backend.calls[0][0][0].messages[0]["content"]
    assert "{fabricated}" in content and "Document type label (from the record system): CT {chest}" in content
    assert "Document date (from the record system): 2024-03-01" in content
    assert "core_needle_biopsy: Core needle biopsy" in content and "LINKING ITEMS TO CANCERS" not in content
    assert not re.search(r"\{(field_guide|note_text|linking|extension_guidance)\}", content)
    assert content.index("FIELDS") < content.index("<document>")
    patient_records = pd.DataFrame([{"patient_id": "p1", "record": {"items": {"diagnoses": [
        {"item_id": "D1", "primary_site": "lung", "histology": "adenocarcinoma",
         "diagnosis_date": {"value": "2024-03", "precision": "month", "source": "document_text"}}]}}}])
    answer = document_answer(imaging_studies=[{**imaging()}])
    backend.script["CT CHEST"] = [answer]
    extract_document_records(notes, config=config, patient_records=patient_records)
    content = backend.calls[-1][0][0].messages[0]["content"]
    assert "D1: lung, adenocarcinoma, diagnosed 2024-03" in content
    assert "cancer_ids" in json.dumps(backend.calls[-1][1]["remote"]["request_params"]["response_format"])


def test_fill_is_single_pass():
    assert fill("{a} {b}", {"a": "{b}", "b": "x"}) == "{b} x"
    with pytest.raises(KeyError):
        fill("{missing}", {})


# Parsing and validation


def _note(text):
    return {"patient_id": "p1", "note_id": "n1", "note_date": "2024-03-10", "note_type": None, "text": text}


def test_quotes_must_occur_in_the_note():
    note = _note(PATH_REPORT)
    item_types = s.item_types_for("document")
    good = document_answer("pathology_report", procedures=[procedure()])
    check = check_document_answer(good, note, item_types, linking=False, known_ids=set())
    assert not check.problems
    entry, evidence = check.items["procedures"][0]
    assert evidence == [{"note_id": "n1", "note_date": "2024-03-10",
                         "quote": "Lung, right upper lobe, CT-guided core needle biopsy.",
                         "start": 10, "end": 63}]
    invented = procedure(evidence=[{"quote": "Lung, right upper lobe, CT-guided core needle biopsy performed by Dr. X."}])
    check = check_document_answer(document_answer("pathology_report", procedures=[invented]),
                                  note, item_types, linking=False, known_ids=set())
    assert check.grounding and not check.structural and check.dropped[0]["item_type"] == "procedure"
    # A placeholder item with an empty quote is unsupported, like an invented one.
    placeholder = {"evidence": [{"quote": ""}], "scale": "ecog", "score": 0}
    check = check_document_answer(document_answer("pathology_report", procedures=[procedure()],
                                                  performance_status=[placeholder]),
                                  note, item_types, linking=False, known_ids=set())
    assert not check.structural and check.dropped == [{
        "item_type": "performance_status", "index": 0,
        "reason": "no evidence quote was found in the source"}]
    # Only layout may differ (whitespace, case, markdown marks, list markers,
    # typographic dashes); the record keeps the note's own text at the span.
    assert locate_quote("A  B\nC", "A B C") == (0, 6)
    text = "**Scan type:** CT chest\nBIOPSY:\n- MODERATELY differentiated \u2013 grade 2"
    span = locate_quote(text, "scan type: ct chest")
    assert text[span[0]:span[1]] == "Scan type:** CT chest"
    span = locate_quote(text, "BIOPSY: moderately differentiated - grade 2")
    assert text[span[0]:span[1]] == "BIOPSY:\n- MODERATELY differentiated \u2013 grade 2"
    for invented in ("tumour proportion score", "Scan type: MRI chest", "BIOPSY: well differentiated"):
        assert locate_quote(text + " tumor proportion score", invented) is None


def test_structural_problems_are_reported_for_retry():
    note = _note(CT_REPORT)
    item_types = s.item_types_for("document")
    bad = [
        document_answer(imaging_studies=[imaging(modality="cat_scan")]),
        document_answer(imaging_studies=[imaging(modality="other", modality_text="")]),
        document_answer(imaging_studies=[imaging(study_date="March 2024")]),
        document_answer(imaging_studies=[imaging(relation="previously_performed")]),
        document_answer(performance_status=[{"evidence": [{"quote": "CT"}], "scale": "ecog", "score": 7}]),
        {**document_answer(), "summary": "extra"},
    ]
    for answer in bad:
        check = check_document_answer(answer, note, item_types, linking=False, known_ids=set())
        assert check.structural, answer
    assert "must list the imaging study" in check_document_answer(
        bad[3], note, item_types, linking=False, known_ids=set()).structural[0].message
    with pytest.raises(ValueError, match="cut off"):
        parse_answer("{}", "length")
    with pytest.raises(ValueError, match="Duplicate"):
        parse_answer('{"a": 1, "a": 2}', "stop")


# Document-level API


def test_extract_document_records_builds_grounded_records(config, monkeypatch, tmp_path):
    first = document_answer(imaging_studies=[imaging(evidence=[{"quote": "CT CHEST, WITH CONTRAST"}])])
    backend = FakeBackend({
        "CT CHEST": [first, document_answer(imaging_studies=[imaging()])],
        "SPECIMEN": [document_answer("pathology_report", procedures=[procedure()], biomarkers=[{
            "evidence": [{"quote": "PD-L1 (22C3) tumor proportion score: 60%."}], "biomarker": "PD-L1",
            "biomarker_type": "protein_expression", "result": "positive", "alteration": "",
            "value": "TPS 60%", "method": "ihc", "assay_name": "22C3", "sample_type": "tissue",
            "origin": "somatic", "test_date": ""}])],
        "Oncology visit": [document_answer("clinic_note", imaging_studies=[imaging(
            evidence=[{"quote": "Outside PET/CT on 2024-02-20"}], relation="previously_performed",
            modality="pet_ct", modality_text="PET/CT", body_regions=["whole_body"],
            contrast="not_stated", study_date="2024-02-20")], performance_status=[
            {"evidence": [{"quote": "ECOG 1"}], "scale": "ecog", "score": 1}])],
    })
    install(monkeypatch, backend)
    notes = pd.DataFrame([
        {"patient_id": "p1", "note_text": CT_REPORT, "note_date": "2024-03-01", "note_id": "ct"},
        {"patient_id": "p1", "note_text": PATH_REPORT, "note_date": "2024-03-12", "note_id": "path"},
        {"patient_id": "p1", "note_text": CLINIC_NOTE, "note_date": "2024-03-15", "note_id": "visit"},
        {"patient_id": "p1", "note_text": "  ", "note_date": "2024-03-16", "note_id": "blank"},
    ])
    records, metadata, qc = extract_document_records(
        notes, config=config, checkpoint_dir=tmp_path, return_metadata=True, return_qc=True)
    assert records.note_id.tolist() == ["ct", "path", "visit"]
    assert records.extraction_status.tolist() == ["complete"] * 3
    format_ = backend.calls[0][1]["remote"]["request_params"]["response_format"]
    assert format_["type"] == "json_schema" and format_["json_schema"]["strict"] is True
    # The rejected draft is revised, with its problem named.
    retry = [p for prompts, _ in backend.calls for p in prompts if len(p.messages) == 3]
    assert len(retry) == 1 and json.loads(retry[0].messages[1]["content"]) == first
    assert "quote not found in the document" in retry[0].messages[2]["content"]
    ct = records.record[0]["items"]["imaging_studies"][0]
    assert ct["study_date"] == {"value": "2024-03-01", "precision": "day", "source": "note_date"}
    assert ct["source_of_document"] is True and ct["status"] == "performed"
    assert ct["evidence"][0] == {"note_id": "ct", "note_date": "2024-03-01",
                                 "quote": "CT CHEST WITH IV CONTRAST", "start": 6, "end": 31}
    biopsy = records.record[1]["items"]["procedures"][0]
    assert biopsy["procedure_group"] == "histologic_biopsy"
    assert biopsy["procedure_date"]["source"] == "document_text"
    assert metadata["counts"]["notes_without_text"] == 1 and metadata["structured_outputs"] is True
    assert metadata["documents"]["ct"]["attempts"][0]["problems"]
    assert qc.set_index("metric").loc["documents_failed_extraction", "value"] == 0

    # The questions a workup check asks become table filters.
    items = extraction_items(records)
    studies = items[items.item_type == "imaging_study"]
    ct_chest = studies[(studies.modality == "ct") & (studies.status == "performed")
                       & studies.body_regions.map(lambda regions: "chest" in regions)]
    assert ct_chest.date.tolist() == ["2024-03-01"]
    assert items.loc[items.procedure_type == "core_needle_biopsy", "date"].tolist() == ["2024-03-08"]
    assert items.loc[items.item_type == "performance_status", "date_source"].tolist() == ["note_date"]
    outside = items[items.modality == "pet_ct"].iloc[0]
    assert outside.source_of_document is False and outside.date == "2024-02-20"

    # A rerun reuses the checkpoint without requests.
    calls = len(backend.calls)
    again = extract_document_records(notes, config=config, checkpoint_dir=tmp_path)
    assert len(backend.calls) == calls and again.record.tolist() == records.record.tolist()


def test_last_attempt_drops_ungrounded_items_and_unfixable_answers_fail(config, monkeypatch):
    invented = {"evidence": [{"quote": "invented history"}], "scale": "ecog", "score": 2}
    backend = FakeBackend({
        "CT CHEST": [document_answer(imaging_studies=[imaging()], performance_status=[invented])],
        "SPECIMEN": [("not json", "stop")],
    })
    install(monkeypatch, backend)
    notes = pd.DataFrame([
        {"patient_id": "p1", "note_text": CT_REPORT, "note_date": "2024-03-01", "note_id": "ct"},
        {"patient_id": "p2", "note_text": PATH_REPORT, "note_date": None, "note_id": "path"},
    ])
    records, metadata = extract_document_records(notes, config=config, return_metadata=True)
    ct, path = records.itertuples(index=False)
    assert ct.extraction_status == "partial" and ct.record["items"]["performance_status"] == []
    assert metadata["documents"]["ct"]["dropped_items"][0]["item_type"] == "performance_status"
    # Three attempts, then one with thinking off.
    attempts = metadata["documents"]["path"]["attempts"]
    assert [a["thinking"] for a in attempts] == [True, True, True, False]
    assert path.extraction_status == "failed" and path.record is None and "not valid JSON" in path.error
    fallback = backend.calls[-1][1]
    assert fallback["remote"]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


# Patient-level API


def _patient_answer(**lists):
    return {t.key: lists.get(t.key, []) for t in s.item_types_for("patient")}


def test_extract_patient_records_merges_chunk_updates(config, monkeypatch, tmp_path):
    config.patient_extraction = {**config.patient_extraction, "chunk_tokens": 300,
                                 "chunk_overlap_tokens": 20}
    diagnosis = {
        "item_id": "D1", "evidence": [{"note": 1, "quote": "Right upper lobe mass"}],
        "primary_site": "lung", "histology": "", "grade": "", "diagnosis_date": "",
        "stage_group": "", "t_category": "", "n_category": "", "m_category": "",
        "stage_basis": "not_stated", "staging_system": "", "extent_at_diagnosis": "not_stated",
    }
    study = {
        "item_id": "I1", "evidence": [{"note": 1, "quote": "CT CHEST WITH IV CONTRAST"}],
        "status": "performed", "modality": "ct", "modality_text": "CT chest", "body_regions": ["chest"],
        "body_site_text": "", "contrast": "with_contrast", "tracer": "", "study_date": "",
    }
    biopsy = {
        "item_id": "P1", "evidence": [{"note": 2, "quote": "CT-guided core needle biopsy"}],
        "status": "performed", "procedure_type": "core_needle_biopsy", "procedure_text": "core biopsy",
        "image_guidance": "ct", "specimen_site": "right upper lobe", "laterality": "right",
        "procedure_date": "2024-03-08", "specimen_result": "malignant", "result_text": "Adenocarcinoma",
        "cancer_ids": ["D1"],
    }
    updated = {**diagnosis, "histology": "adenocarcinoma",
               "evidence": [{"note": 2, "quote": "Adenocarcinoma, acinar predominant."}]}
    backend = FakeBackend({
        "PART 1 OF 2": [_patient_answer(diagnoses=[diagnosis], imaging_studies=[study])],
        "PART 2 OF 2": [_patient_answer(diagnoses=[updated], procedures=[{**biopsy, "cancer_ids": ["D9"]}]),
                        _patient_answer(diagnoses=[updated], procedures=[biopsy])],
    })
    install(monkeypatch, backend)
    filler = " ".join(["Fabricated filler text."] * 60)
    notes = pd.DataFrame([
        {"patient_id": "p1", "note_text": CT_REPORT + "\n" + filler, "note_date": "2024-03-01", "note_id": "ct"},
        {"patient_id": "p1", "note_text": PATH_REPORT + "\n" + filler, "note_date": "2024-03-12", "note_id": "path"},
    ])
    records, metadata = extract_patient_records(notes, config=config, checkpoint_dir=tmp_path,
                                                return_metadata=True)
    row = records.iloc[0]
    assert row.extraction_status == "complete" and row.chunk_count == 2
    second = backend.calls[1][0][0].messages[0]["content"]
    assert '{"item_id": "D1", "evidence"' not in second and '"item_id": "D1"' in second
    assert "[Note 2 | 2024-03-12]" in second and "NOTES, PART 2 OF 2" in second
    retry = backend.calls[2][0][0].messages
    assert len(retry) == 3 and retry[1]["role"] == "assistant" and "not diagnosis ids" in retry[2]["content"]
    items = row.record["items"]
    assert items["diagnoses"][0]["histology"] == "adenocarcinoma"
    assert [e["note_id"] for e in items["diagnoses"][0]["evidence"]] == ["ct", "path"]
    assert items["procedures"][0]["cancer_ids"] == ["D1"]
    assert items["imaging_studies"][0]["study_date"] == {
        "value": "2024-03-01", "precision": "day", "source": "evidence_note_date"}
    table = extraction_items(records)
    assert set(table.item_id) == {"D1", "I1", "P1"} and table.level.eq("patient").all()
    assert metadata["counts"]["chunks"] == 2

    calls = len(backend.calls)
    again = extract_patient_records(notes, config=config, checkpoint_dir=tmp_path)
    assert len(backend.calls) == calls and again.record[0] == row.record


# Preset


def test_preset_sections_use_nemotron_structured_outputs_and_room_to_reason():
    config = load_default_preset()
    for name in ("document_extraction", "patient_extraction"):
        section = getattr(config, name)
        assert section["local"]["model_name"] == MODEL and section["remote"]["model_name"] == MODEL
        assert section["sampling_profile"] == "auto" and section["structured_outputs"] is True
        assert section["reasoning_off_fallback"] is True and section["max_attempts"] == 3
        assert section["remote"]["request_params"]["max_tokens"] >= 32768
        assert section["remote"]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True
        assert section["extensions"] == []
    patient = config.patient_extraction
    assert patient["chunk_tokens"] + patient["remote"]["request_params"]["max_tokens"] < (
        patient["local"]["engine"]["max_model_len"])
    from matchminer_ai.llm.backends import build_llm_runtime_config

    config.remote["enabled"] = True
    section = config.document_extraction
    runtime = build_llm_runtime_config("document_extraction", {
        k: section[k] for k in ("local", "remote", "sampling_profile", "reasoning_effort")}, config=config)
    # Nemotron's own sampling.
    assert runtime["remote"]["request_params"]["temperature"] == 1.0
    assert runtime["remote"]["extra_body"]["top_k"] == -1
