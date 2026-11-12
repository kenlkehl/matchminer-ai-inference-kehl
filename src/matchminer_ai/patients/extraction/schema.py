"""The structured clinical extraction data model: item types, JSON schemas, validation.

A record is a JSON object of typed *items* (imaging studies, procedures,
diagnoses, biomarkers, treatments and so on), each with its fields, its dates as
ISO strings with a precision, and its evidence: exact quotes from the source
note with the note's code-derived id, date and character offsets. Two levels
share the item types and vocabularies:

* a **document** record describes one note (:func:`extract_document_records`);
* a **patient** record describes one patient from all of their notes
  (:func:`extract_patient_records`), and gives each item an ``item_id``.

Each item type is declared once here (:class:`ItemType`). The prompt's field
guide, the JSON schema that constrains the model's answer (vLLM structured
outputs), the validator of that answer, and the JSON schema of the returned
records are all derived from the declaration, so they cannot drift apart.
Disease-specific item types are added by extensions (:mod:`.extensions`) without
changing these.

What the model writes and what a record holds differ only in code-owned fields:
the model writes a date as ``"2024-03"`` and the record holds
``{"value": "2024-03", "precision": "month", "source": "document_text"}``; the
model writes ``{"quote": ...}`` and the record holds the quote as it appears in
the note with the note's id, date and offsets.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date

from . import vocabulary as v
from .vocabulary import Vocabulary

SCHEMA_VERSION = "mmai-clinical-extraction-v1"
LEVELS = ("document", "patient")

#: Longest accepted evidence quote, in characters (the shortest supporting span
#: is asked for; this bounds copying whole sections).
MAX_QUOTE_CHARACTERS = 600
MAX_EVIDENCE_ITEMS = 4
MAX_TEXT_CHARACTERS = 2000
MAX_LIST_ENTRIES = 50

_DATE = re.compile(r"(\d{4})(?:-(\d{1,2})(?:-(\d{1,2}))?)?")


@dataclass(frozen=True)
class Field:
    """One field of an item type.

    ``kind`` is ``text``, ``text_list``, ``choice``, ``choice_list``, ``date``
    or ``integer``. A choice field's ``other_text`` names the text field that
    must name the value when the ``other`` code is chosen.
    """

    name: str
    kind: str
    description: str
    vocabulary: Vocabulary | None = None
    other_text: str | None = None
    minimum: int | None = None
    maximum: int | None = None

    def __post_init__(self):
        kinds = {"text", "text_list", "choice", "choice_list", "date", "integer"}
        if self.kind not in kinds:
            raise ValueError(f"Unknown field kind {self.kind!r}")
        if (self.kind in {"choice", "choice_list"}) != (self.vocabulary is not None):
            raise ValueError(f"{self.name}: choice fields, and only they, need a vocabulary")


@dataclass(frozen=True)
class ItemType:
    """One kind of extracted item.

    ``key`` is the record's list of these items and ``name`` one item's
    ``item_type``. ``levels`` are the record levels that extract it.
    ``occurrence`` items (imaging studies, procedures) say how they relate to
    the document (``relation``) or, at the patient level, whether they were
    performed or planned (``status``). ``linkable_levels`` are the levels whose
    items carry ``cancer_ids``: the patient-level ``D`` ids of the diagnoses
    they concern. ``primary_date`` names the field that dates an item for
    timeline queries (None: the document's note date). ``check`` adds
    item-specific validation and returns problem messages.
    """

    name: str
    key: str
    id_prefix: str
    description: str
    fields: tuple[Field, ...]
    levels: frozenset[str]
    occurrence: bool = False
    linkable_levels: frozenset[str] = frozenset()
    primary_date: str | None = None
    guidance: str = ""
    check: Callable[[dict], list[str]] | None = None
    max_items: int = 64

    def __post_init__(self):
        names = [f.name for f in self.fields]
        reserved = {"item_id", "item_type", "relation", "status", "cancer_ids",
                    "evidence", "source_of_document"}
        if len(set(names)) != len(names) or reserved & set(names):
            raise ValueError(f"{self.name}: duplicate or reserved field names")
        if not re.fullmatch(r"[A-Z]{1,3}", self.id_prefix):
            raise ValueError(f"{self.name}: id_prefix must be 1-3 capital letters")
        if not set(self.levels) <= set(LEVELS) or not self.levels:
            raise ValueError(f"{self.name}: levels must be a nonempty subset of {LEVELS}")
        for item_field in self.fields:
            if item_field.other_text is not None and item_field.other_text not in names:
                raise ValueError(f"{self.name}.{item_field.name}: unknown other_text field")
        if self.primary_date is not None and self.primary_date not in names:
            raise ValueError(f"{self.name}: unknown primary_date field")

    def field(self, name: str) -> Field:
        return next(f for f in self.fields if f.name == name)


def _text(name, description):
    return Field(name, "text", description)


def _date(name, description):
    return Field(name, "date", description)


def _choice(name, vocabulary, description, other_text=None):
    return Field(name, "choice", description, vocabulary, other_text)


def _performance_status_check(item: dict) -> list[str]:
    score, scale = item.get("score"), item.get("scale")
    if scale == "ecog" and not 0 <= score <= 5:
        return ["an ECOG score must be 0-5"]
    if scale == "karnofsky" and score % 10:
        return ["a Karnofsky score must be a multiple of 10 from 0 to 100"]
    return []


IMAGING_STUDY = ItemType(
    name="imaging_study",
    key="imaging_studies",
    id_prefix="I",
    description="An imaging study: one item per study (modality, region and date).",
    levels=frozenset(LEVELS),
    occurrence=True,
    primary_date="study_date",
    fields=(
        _choice("modality", v.IMAGING_MODALITIES, "The imaging technique.", "modality_text"),
        _text(
            "modality_text",
            "The study's name as written, for example 'CT chest, abdomen and pelvis "
            "with IV contrast'.",
        ),
        Field(
            "body_regions",
            "choice_list",
            "Every body region the study covers; [] when not stated.",
            v.BODY_REGIONS,
            "body_site_text",
        ),
        _text(
            "body_site_text",
            "The specific site or organ examined, as written, when narrower than the "
            "regions or when a region is other (for example 'liver', 'prostate'); "
            "'' otherwise.",
        ),
        _choice("contrast", v.CONTRAST, "Intravenous contrast."),
        _text("tracer", "PET or nuclear medicine tracer as written (FDG, PSMA); '' otherwise."),
        _date("study_date", "The date the study was performed."),
    ),
    guidance=(
        "A CT- or ultrasound-guided biopsy is a procedure with that image_guidance, "
        "not an imaging study. A combined report of separately ordered studies "
        "(for example CT chest and CT abdomen/pelvis) may be one item per study or "
        "one item listing every region."
    ),
)

PROCEDURE = ItemType(
    name="procedure",
    key="procedures",
    id_prefix="P",
    description=(
        "A procedure that obtained a tissue, cytology, bone marrow or blood specimen: "
        "one item per procedure and site."
    ),
    levels=frozenset(LEVELS),
    occurrence=True,
    linkable_levels=frozenset(LEVELS),
    primary_date="procedure_date",
    fields=(
        _choice(
            "procedure_type",
            v.PATHOLOGY_PROCEDURES,
            "How the specimen was obtained.",
            "procedure_text",
        ),
        _text(
            "procedure_text",
            "The procedure as written, for example 'CT-guided core biopsy of liver "
            "lesion' or 'EBUS-TBNA station 7'.",
        ),
        _choice("image_guidance", v.IMAGE_GUIDANCE, "Imaging that guided the procedure."),
        _text(
            "specimen_site",
            "The site sampled, as written ('right upper lobe', 'level 4R lymph node', "
            "'posterior iliac crest'); '' when not stated.",
        ),
        _choice("laterality", v.LATERALITY, "Side of the site sampled."),
        _date(
            "procedure_date",
            "The date of the procedure or specimen collection (not the report date).",
        ),
        _choice("specimen_result", v.SPECIMEN_RESULTS, "What the specimen showed."),
        _text(
            "result_text",
            "The specimen's diagnosis as written ('adenocarcinoma, acinar predominant'); "
            "'' when not stated.",
        ),
    ),
    guidance=(
        "Name the technique from the text: core biopsy, FNA, EBUS-TBNA and so on. "
        "Use needle_biopsy_not_specified or biopsy_not_specified rather than guess "
        "a technique the text does not state, and do not infer the technique from "
        "the diagnosis."
    ),
)

DIAGNOSIS = ItemType(
    name="diagnosis",
    key="diagnoses",
    id_prefix="D",
    description=(
        "A cancer diagnosis: one item per distinct primary cancer. A recurrence or "
        "metastasis is not a new diagnosis."
    ),
    levels=frozenset(LEVELS),
    linkable_levels=frozenset({"document"}),
    primary_date="diagnosis_date",
    fields=(
        _text("primary_site", "Primary site as written ('lung', 'left breast', 'sigmoid colon')."),
        _text("histology", "Histology as written ('adenocarcinoma', 'diffuse large B-cell lymphoma')."),
        _text("grade", "Grade or differentiation as written; '' when not stated."),
        _date("diagnosis_date", "The date of diagnosis."),
        _text("stage_group", "The grouped stage as written ('IIIA', 'IV', 'extensive stage'); ''."),
        _text("t_category", "T category as written ('T2a', 'pT1c'); '' when not stated."),
        _text("n_category", "N category as written ('N2', 'cN0'); '' when not stated."),
        _text("m_category", "M category as written ('M1b', 'M0'); '' when not stated."),
        _choice("stage_basis", v.STAGE_BASES, "Whether the stage is clinical or pathologic."),
        _text(
            "staging_system",
            "The staging system as written ('AJCC 8th edition', 'FIGO 2018', 'R-ISS'); ''.",
        ),
        _choice(
            "extent_at_diagnosis",
            v.DISEASE_EXTENTS,
            "Extent at diagnosis, only when the text states it or the stage states it.",
        ),
    ),
    guidance=(
        "Report the stage as written. Never put a T, N or M category in stage_group, "
        "and do not derive a stage group the text does not state. Do not record "
        "non-cancer diagnoses."
    ),
)

BIOMARKER = ItemType(
    name="biomarker",
    key="biomarkers",
    id_prefix="B",
    description=(
        "A biomarker result: one item per marker tested (negative results included), "
        "per test."
    ),
    levels=frozenset(LEVELS),
    linkable_levels=frozenset(LEVELS),
    primary_date="test_date",
    fields=(
        _text(
            "biomarker",
            "The gene, protein or marker as written (EGFR, ALK, PD-L1, HER2, ER, MSI, "
            "TMB, PSA).",
        ),
        _choice("biomarker_type", v.BIOMARKER_TYPES, "The kind of biomarker."),
        _choice("result", v.BIOMARKER_RESULTS, "The result category."),
        _text(
            "alteration",
            "The specific variant or finding ('L858R', 'exon 19 deletion', "
            "'EML4-ALK fusion', 'IHC 3+'); '' when none.",
        ),
        _text(
            "value",
            "A quantitative value with its unit ('TPS 60%', 'VAF 12%', '14 mut/Mb', "
            "'4.2 ng/mL'); '' when none.",
        ),
        _choice("method", v.TEST_METHODS, "The test method."),
        _text("assay_name", "The named assay or panel as written; '' when not stated."),
        _choice("sample_type", v.SAMPLE_TYPES, "The sample tested."),
        _choice("origin", v.BIOMARKER_ORIGINS, "Somatic or germline."),
        _date(
            "test_date",
            "The date the specimen was collected or the result reported, whichever "
            "the text gives.",
        ),
    ),
    guidance=(
        "Include negative and wild-type results. A gene name means the gene, never "
        "a laboratory value with a similar abbreviation (EGFR the gene is not eGFR, "
        "the kidney function estimate)."
    ),
)

TREATMENT = ItemType(
    name="treatment",
    key="treatments",
    id_prefix="T",
    description=(
        "An anticancer treatment course: one item per systemic regimen or line, "
        "operation, radiation course or local treatment."
    ),
    levels=frozenset(LEVELS),
    linkable_levels=frozenset(LEVELS),
    primary_date="start_date",
    fields=(
        _choice("treatment_type", v.TREATMENT_TYPES, "The kind of treatment."),
        _text(
            "regimen",
            "The regimen, operation or radiation course as written "
            "('carboplatin/pemetrexed/pembrolizumab', 'right upper lobectomy', "
            "'SBRT 50 Gy in 5 fractions').",
        ),
        Field(
            "agents",
            "text_list",
            "Each drug of a systemic regimen by name; [] for other treatments.",
        ),
        _text("site", "The treated site for surgery, radiation or local therapy; ''."),
        _choice("setting", v.TREATMENT_SETTINGS, "The clinical setting."),
        Field(
            "line_of_therapy",
            "integer",
            "Line number for advanced or metastatic disease when the text states it "
            "(or, in a patient record, counted across this patient's regimens for "
            "advanced disease); 0 otherwise.",
            minimum=0,
            maximum=30,
        ),
        _choice("treatment_status", v.TREATMENT_STATUSES, "Where the treatment stands."),
        _date("start_date", "The start date (the operation date for surgery)."),
        _date("end_date", "The end date."),
        _choice("best_response", v.TREATMENT_RESPONSES, "Best response reported."),
        _text("end_reason", "Why it ended, as written (progression, toxicity); ''."),
        Field("toxicities", "text_list", "Treatment toxicities as written; []."),
    ),
    guidance=(
        "Record treatments given or definitely scheduled, never options merely "
        "discussed. A change of systemic agents for progression or toxicity starts a "
        "new item."
    ),
)
DISEASE_ASSESSMENT = ItemType(
    name="disease_assessment",
    key="disease_assessments",
    id_prefix="A",
    description=(
        "This document's own overall assessment of a cancer's status and the sites "
        "where it says cancer is present: at most one item per cancer."
    ),
    levels=frozenset({"document"}),
    linkable_levels=frozenset({"document"}),
    fields=(
        _choice("disease_status", v.DISEASE_STATUSES, "The overall assessment."),
        Field(
            "disease_sites",
            "text_list",
            "Sites where this document says cancer is present ('liver', 'mediastinal "
            "lymph nodes', 'brain'); sites stated to be free of cancer are not listed.",
        ),
    ),
    guidance=(
        "progression only when disease is growing or new, stable only when it is "
        "stated to be unchanged, mixed when some sites improved and others worsened. "
        "Presence of cancer alone is not progression. Use a comparison with an earlier "
        "study only to judge change; findings restated from that earlier study are "
        "not this document's."
    ),
)

PERFORMANCE_STATUS = ItemType(
    name="performance_status",
    key="performance_status",
    id_prefix="S",
    description="A performance status this document states as a number.",
    levels=frozenset({"document"}),
    fields=(
        _choice("scale", v.PERFORMANCE_SCALES, "The scale."),
        Field("score", "integer", "The score as stated.", minimum=0, maximum=100),
    ),
    guidance=(
        "Only a number the document states counts ('ECOG 1', 'PS 0', 'KPS 80'). "
        "Never infer a score from a description of function and never convert "
        "between scales. For a range ('ECOG 1-2') record the worse value."
    ),
    check=_performance_status_check,
    max_items=4,
)

EVENT = ItemType(
    name="event",
    key="events",
    id_prefix="E",
    description="A key dated event in the patient's cancer history.",
    levels=frozenset({"patient"}),
    linkable_levels=frozenset({"patient"}),
    primary_date="event_date",
    fields=(
        _choice("event_type", v.EVENT_TYPES, "The kind of event."),
        _text("description", "A short description in the text's words."),
        _date("event_date", "The date of the event."),
    ),
    guidance=(
        "Events complement the other items: the dates of recurrence, progression, "
        "new metastases, remission, hospitalizations, hospice, death and trial "
        "enrollment. Do not repeat treatment starts or imaging studies as events."
    ),
)

CORE_ITEM_TYPES: tuple[ItemType, ...] = (
    IMAGING_STUDY,
    PROCEDURE,
    DIAGNOSIS,
    BIOMARKER,
    TREATMENT,
    DISEASE_ASSESSMENT,
    PERFORMANCE_STATUS,
    EVENT,
)


def item_types_for(level: str, extra: Iterable[ItemType] = ()) -> tuple[ItemType, ...]:
    """The core item types of ``level`` followed by ``extra`` (extension) ones."""
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}")
    types = [t for t in (*CORE_ITEM_TYPES, *extra) if level in t.levels]
    for attribute in ("name", "key", "id_prefix"):
        values = [getattr(t, attribute) for t in types]
        if len(set(values)) != len(values):
            raise ValueError(f"Item types repeat a {attribute}: {values}")
    return tuple(types)


# ---------------------------------------------------------------------------
# JSON schemas
# ---------------------------------------------------------------------------


def _object(properties: dict, description: str | None = None) -> dict:
    result = {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }
    if description:
        result["description"] = description
    return result


def _response_field_schema(item_field: Field) -> dict:
    # No minLength or pattern: under vLLM structured outputs a minLength string
    # admitted no JSON escapes (trials/cohort_workflow.py). Code validates them.
    if item_field.kind in {"text", "date"}:
        schema = {"type": "string"}
    elif item_field.kind == "text_list":
        schema = {"type": "array", "items": {"type": "string"}, "maxItems": MAX_LIST_ENTRIES}
    elif item_field.kind == "choice":
        schema = {"type": "string", "enum": list(item_field.vocabulary.codes)}
    elif item_field.kind == "choice_list":
        schema = {
            "type": "array",
            "items": {"type": "string", "enum": list(item_field.vocabulary.codes)},
            "maxItems": len(item_field.vocabulary.codes),
        }
    else:
        schema = {"type": "integer"}
        if item_field.minimum is not None:
            schema["minimum"] = item_field.minimum
        if item_field.maximum is not None:
            schema["maximum"] = item_field.maximum
    return {**schema, "description": item_field.description}


def _evidence_response_schema(level: str) -> dict:
    properties = {"quote": {"type": "string", "description": "An exact quote."}}
    if level == "patient":
        properties = {
            "note": {"type": "integer", "minimum": 1, "description": "The note number."},
            **properties,
        }
    return {
        "type": "array",
        "minItems": 1,
        "maxItems": MAX_EVIDENCE_ITEMS,
        "items": _object(properties),
    }


def _linking(item_type: ItemType, level: str, linking: bool) -> bool:
    return level in item_type.linkable_levels and (level == "patient" or linking)


def item_response_schema(item_type: ItemType, level: str, *, linking: bool = False) -> dict:
    """The JSON schema of one item as the model writes it at ``level``."""
    properties: dict = {}
    if level == "patient":
        properties["item_id"] = {
            "type": "string",
            "description": f"{item_type.id_prefix} followed by a number, such as "
            f"{item_type.id_prefix}1.",
        }
    properties["evidence"] = _evidence_response_schema(level)
    if item_type.occurrence:
        vocabulary = v.DOCUMENT_RELATIONS if level == "document" else v.OCCURRENCE_STATUSES
        properties["relation" if level == "document" else "status"] = {
            "type": "string",
            "enum": list(vocabulary.codes),
        }
    for item_field in item_type.fields:
        properties[item_field.name] = _response_field_schema(item_field)
    if _linking(item_type, level, linking):
        properties["cancer_ids"] = {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 20,
            "description": "Ids of the diagnoses this item concerns; [] when unclear.",
        }
    return _object(properties, item_type.description)


def document_response_schema(
    extra: Iterable[ItemType] = (), *, linking: bool = False
) -> dict:
    """The JSON schema of the model's answer for one document."""
    properties: dict = {
        "document_type": {"type": "string", "enum": list(v.DOCUMENT_TYPES.codes)},
        "document_type_text": {
            "type": "string",
            "description": "The document type as written when document_type is other; ''.",
        },
    }
    for item_type in item_types_for("document", extra):
        properties[item_type.key] = {
            "type": "array",
            "maxItems": item_type.max_items,
            "items": item_response_schema(item_type, "document", linking=linking),
        }
    return _object(properties)


def patient_response_schema(extra: Iterable[ItemType] = ()) -> dict:
    """The JSON schema of the model's answer for one part of a patient's notes:
    the new and changed items, by type."""
    properties = {
        item_type.key: {
            "type": "array",
            "maxItems": 4 * item_type.max_items,
            "items": item_response_schema(item_type, "patient"),
        }
        for item_type in item_types_for("patient", extra)
    }
    return _object(properties)


def response_format(name: str, schema: dict) -> dict:
    """OpenAI-compatible ``response_format`` constraining an answer to ``schema``."""
    return {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}}


_DATE_RECORD = _object(
    {
        "value": {"type": ["string", "null"], "description": "YYYY, YYYY-MM or YYYY-MM-DD."},
        "precision": {"type": "string", "enum": list(v.DATE_PRECISIONS)},
        "source": {
            "type": "string",
            "enum": ["document_text", "note_date", "evidence_note_date", "unknown"],
            "description": "document_text: written in the note; note_date: the note's "
            "own date (the study or procedure a report documents); evidence_note_date: "
            "the earliest note quoted (performed on or before it); unknown: no date.",
        },
    }
)
_EVIDENCE_RECORD = {
    "type": "array",
    "items": _object(
        {
            "note_id": {"type": "string"},
            "note_date": {"type": ["string", "null"]},
            "quote": {"type": "string", "description": "The exact text of the note."},
            "start": {"type": "integer", "description": "Character offset in the note."},
            "end": {"type": "integer"},
        }
    ),
}


def _record_field_schema(item_field: Field) -> dict:
    if item_field.kind == "date":
        return {**_DATE_RECORD, "description": item_field.description}
    return _response_field_schema(item_field)


def item_record_schema(item_type: ItemType, level: str) -> dict:
    """The JSON schema of one item as a returned record holds it."""
    properties: dict = {"item_type": {"const": item_type.name}}
    if level == "patient":
        properties["item_id"] = {"type": "string"}
    if item_type.occurrence:
        if level == "document":
            properties["relation"] = {"type": "string", "enum": list(v.DOCUMENT_RELATIONS.codes)}
            properties["source_of_document"] = {"type": "boolean"}
        properties["status"] = {"type": "string", "enum": list(v.OCCURRENCE_STATUSES.codes)}
    for item_field in item_type.fields:
        properties[item_field.name] = _record_field_schema(item_field)
        if item_field.vocabulary is v.PATHOLOGY_PROCEDURES:
            groups = sorted({t.group for t in v.PATHOLOGY_PROCEDURES.terms if t.group})
            properties["procedure_group"] = {"type": ["string", "null"], "enum": [*groups, None]}
    if level in item_type.linkable_levels:
        properties["cancer_ids"] = {"type": "array", "items": {"type": "string"}}
    properties["evidence"] = _EVIDENCE_RECORD
    schema = _object(properties, item_type.description)
    if any(f.name == "agents" for f in item_type.fields):
        # Present only with ncit_agent_codes: each agent's NCIt concept, or nulls.
        schema["properties"]["agents_ncit"] = {
            "type": "array",
            "items": _object({
                "agent": {"type": "string"},
                "ncit_code": {"type": ["string", "null"]},
                "ncit_name": {"type": ["string", "null"]},
            }),
        }
    return schema


def record_json_schema(level: str, extra: Iterable[ItemType] = ()) -> dict:
    """The JSON schema of the records :func:`extract_document_records` (``level``
    ``"document"``) or :func:`extract_patient_records` (``"patient"``) return."""
    items = {
        t.key: {"type": "array", "items": item_record_schema(t, level)}
        for t in item_types_for(level, extra)
    }
    common = {
        "schema_version": {"const": SCHEMA_VERSION},
        "level": {"const": level},
        "patient_id": {"type": "string"},
        "extensions": {"type": "array", "items": {"type": "string"}},
    }
    if level == "document":
        specific = {
            "note_id": {"type": "string"},
            "note_date": {"type": ["string", "null"]},
            "note_type": {"type": ["string", "null"]},
            "document_type": {"type": "string", "enum": list(v.DOCUMENT_TYPES.codes)},
            "document_type_text": {"type": "string"},
        }
    else:
        specific = {
            "note_count": {"type": "integer"},
            "first_note_date": {"type": ["string", "null"]},
            "last_note_date": {"type": ["string", "null"]},
            "chunk_count": {"type": "integer"},
            "failed_chunks": {"type": "array", "items": {"type": "integer"}},
        }
    schema = _object({**common, **specific, "items": _object(items)})
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["title"] = f"MatchMiner-AI {level} extraction record ({SCHEMA_VERSION})"
    return schema


# ---------------------------------------------------------------------------
# Prompt field guide
# ---------------------------------------------------------------------------


def _describe_field(item_field: Field) -> str:
    kind = {
        "text": "text",
        "text_list": "list of text",
        "choice": "one of",
        "choice_list": "list of",
        "date": "date",
        "integer": "integer",
    }[item_field.kind]
    line = f"    {item_field.name} ({kind}): {item_field.description}"
    if item_field.vocabulary is not None:
        line += "\n" + item_field.vocabulary.describe()
    return line


def render_field_guide(level: str, extra: Iterable[ItemType] = (), *, linking: bool = False) -> str:
    """The prompt's description of every item type and field at ``level``."""
    blocks = []
    for item_type in item_types_for(level, extra):
        lines = [f"{item_type.key}: {item_type.description}"]
        if item_type.guidance:
            lines.append(f"  Guidance: {item_type.guidance}")
        if level == "patient":
            lines.append(f"    item_id: {item_type.id_prefix}1, {item_type.id_prefix}2 and so on.")
        if item_type.occurrence:
            if level == "document":
                lines.append("    relation (one of):\n" + v.DOCUMENT_RELATIONS.describe())
            else:
                lines.append("    status (one of):\n" + v.OCCURRENCE_STATUSES.describe())
        lines.extend(_describe_field(f) for f in item_type.fields)
        if _linking(item_type, level, linking):
            lines.append("    cancer_ids (list of text): see LINKING ITEMS TO CANCERS.")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Validation of model answers
# ---------------------------------------------------------------------------


@dataclass
class Problem:
    """One reason an answer is not accepted.

    ``message`` is for the model (it may repeat a short part of the answer);
    ``code`` is a text-free description for metadata. ``grounding`` problems
    are evidence quotes not found in the source.
    """

    path: str
    message: str
    code: str
    grounding: bool = False


def parse_date(value: str) -> tuple[str | None, str]:
    """``(normalized ISO value or None, precision)`` of a model-written date.

    Raises ValueError for anything but '', YYYY, YYYY-MM or YYYY-MM-DD (one-digit
    months and days are padded) naming a real calendar date from 1900 to 2100.
    """
    text = value.strip()
    if not text:
        return None, "unknown"
    match = _DATE.fullmatch(text)
    if not match:
        raise ValueError("dates must be YYYY-MM-DD, YYYY-MM, YYYY or ''")
    year, month, day = match.groups()
    if not 1900 <= int(year) <= 2100:
        raise ValueError("dates must fall between 1900 and 2100")
    if month is None:
        return year, "year"
    if not 1 <= int(month) <= 12:
        raise ValueError("a date's month must be 01-12")
    if day is None:
        return f"{year}-{int(month):02d}", "month"
    date(int(year), int(month), int(day))  # raises for impossible days
    return f"{year}-{int(month):02d}-{int(day):02d}", "day"


def _check_value(item_field: Field, value, path: str) -> list[Problem]:
    def problem(reason):
        return [Problem(path, f"{path}: {reason}", f"{path}: {reason}")]

    kind = item_field.kind
    if kind in {"text", "date", "choice"}:
        if not isinstance(value, str):
            return problem("must be a string")
        if kind == "text" and len(value) > MAX_TEXT_CHARACTERS:
            return problem(f"must be at most {MAX_TEXT_CHARACTERS} characters")
        if kind == "choice" and value not in item_field.vocabulary.codes:
            return problem(f"must be one of {list(item_field.vocabulary.codes)}")
        if kind == "date":
            try:
                parse_date(value)
            except ValueError as exc:
                return problem(str(exc))
        return []
    if kind in {"text_list", "choice_list"}:
        if not isinstance(value, list) or not all(isinstance(e, str) for e in value):
            return problem("must be a list of strings")
        if len(value) > MAX_LIST_ENTRIES:
            return problem(f"must have at most {MAX_LIST_ENTRIES} entries")
        if kind == "text_list" and any(not e.strip() for e in value):
            return problem("must not contain empty strings")
        if kind == "choice_list":
            unknown = [e for e in value if e not in item_field.vocabulary.codes]
            if unknown or len(set(value)) != len(value):
                return problem(
                    f"must list distinct values of {list(item_field.vocabulary.codes)}"
                )
        return []
    if type(value) is not int:
        return problem("must be an integer")
    if item_field.minimum is not None and value < item_field.minimum:
        return problem(f"must be at least {item_field.minimum}")
    if item_field.maximum is not None and value > item_field.maximum:
        return problem(f"must be at most {item_field.maximum}")
    return []


def check_item(
    item_type: ItemType,
    level: str,
    value,
    path: str,
    *,
    linking: bool = False,
) -> list[Problem]:
    """Structural problems of one model-written item (evidence quotes excluded)."""
    if not isinstance(value, dict):
        return [Problem(path, f"{path}: must be an object", f"{path}: not an object")]
    expected = set(item_response_schema(item_type, level, linking=linking)["properties"])
    if set(value) != expected:
        missing, extra = sorted(expected - set(value)), sorted(set(value) - expected)
        reason = f"missing fields {missing}" if missing else ""
        reason += ("; " if reason and extra else "") + (f"unknown fields {extra}" if extra else "")
        return [Problem(path, f"{path}: {reason}", f"{path}: {reason}")]
    problems: list[Problem] = []
    for item_field in item_type.fields:
        problems += _check_value(item_field, value[item_field.name], f"{path}.{item_field.name}")
    if problems:
        return problems
    for item_field in item_type.fields:
        if item_field.other_text is None or item_field.vocabulary.other is None:
            continue
        chosen = value[item_field.name]
        uses_other = item_field.vocabulary.other in (
            chosen if isinstance(chosen, list) else [chosen]
        )
        if uses_other and not value[item_field.other_text].strip():
            reason = f"{item_field.name} is other, so {item_field.other_text} must name it"
            problems.append(Problem(path, f"{path}: {reason}", f"{path}: {reason}"))
    if item_type.occurrence:
        name = "relation" if level == "document" else "status"
        vocabulary = v.DOCUMENT_RELATIONS if level == "document" else v.OCCURRENCE_STATUSES
        if value[name] not in vocabulary.codes:
            reason = f"{name} must be one of {list(vocabulary.codes)}"
            problems.append(Problem(path, f"{path}: {reason}", f"{path}: {reason}"))
    if "cancer_ids" in value:
        ids = value["cancer_ids"]
        if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
            problems.append(Problem(path, f"{path}.cancer_ids: must be a list of ids",
                                    f"{path}.cancer_ids: not a list of ids"))
    evidence = value["evidence"]
    if not isinstance(evidence, list) or not 1 <= len(evidence) <= MAX_EVIDENCE_ITEMS:
        reason = f"evidence must list 1-{MAX_EVIDENCE_ITEMS} quotes"
        problems.append(Problem(path, f"{path}: {reason}", f"{path}: {reason}"))
    else:
        keys = {"note", "quote"} if level == "patient" else {"quote"}
        for number, quote in enumerate(evidence):
            where = f"{path}.evidence[{number}]"
            if not isinstance(quote, dict) or set(quote) != keys:
                problems.append(Problem(where, f"{where}: must have exactly {sorted(keys)}",
                                        f"{where}: wrong fields"))
            elif not isinstance(quote["quote"], str):
                problems.append(Problem(where, f"{where}: quote must be text",
                                        f"{where}: quote not text"))
            elif len(quote["quote"]) > MAX_QUOTE_CHARACTERS:
                reason = (f"quote must be at most {MAX_QUOTE_CHARACTERS} characters; quote "
                          "the shortest exact span that supports the item")
                problems.append(Problem(where, f"{where}: {reason}", f"{where}: quote too long"))
            elif level == "patient" and type(quote["note"]) is not int:
                problems.append(Problem(where, f"{where}: note must be a note number",
                                        f"{where}: note not an integer"))
    if not problems and item_type.check is not None:
        problems += [Problem(path, f"{path}: {m}", f"{path}: {m}") for m in item_type.check(value)]
    return problems


def resolved_date(value: str, *, fallback: str | None = None, fallback_source: str = "note_date") -> dict:
    """A record's date object for a validated model-written date.

    ``fallback`` (an ISO date) is used, with ``fallback_source``, when the text
    gives no date.
    """
    normalized, precision = parse_date(value)
    if normalized is not None:
        return {"value": normalized, "precision": precision, "source": "document_text"}
    if fallback:
        normalized, precision = parse_date(fallback[:10])
        return {"value": normalized, "precision": precision, "source": fallback_source}
    return {"value": None, "precision": "unknown", "source": "unknown"}


def to_record_item(
    item_type: ItemType,
    level: str,
    value: dict,
    evidence: Sequence[dict],
    *,
    date_fallback: str | None = None,
    date_fallback_source: str = "note_date",
) -> dict:
    """A record item from a validated model-written item and its resolved evidence.

    ``date_fallback`` dates an occurrence item's primary date when the text gives
    none (a report's own study or procedure, from the note date; a performed
    patient-level study, from its earliest evidence note).
    """
    item: dict = {"item_type": item_type.name}
    if level == "patient":
        item["item_id"] = value["item_id"]
    if item_type.occurrence:
        if level == "document":
            item["relation"] = value["relation"]
            item["source_of_document"] = value["relation"] == "this_report"
            item["status"] = "planned" if value["relation"] == "planned" else "performed"
        else:
            item["status"] = value["status"]
    for item_field in item_type.fields:
        raw = value[item_field.name]
        if item_field.kind == "date":
            fallback = date_fallback if item_field.name == item_type.primary_date else None
            item[item_field.name] = resolved_date(
                raw, fallback=fallback, fallback_source=date_fallback_source
            )
        elif item_field.kind in {"text", "text_list"}:
            item[item_field.name] = raw.strip() if isinstance(raw, str) else [e.strip() for e in raw]
        else:
            item[item_field.name] = raw
        if item_field.vocabulary is v.PATHOLOGY_PROCEDURES:
            item["procedure_group"] = v.PATHOLOGY_PROCEDURES.group(raw)
    if level in item_type.linkable_levels:
        item["cancer_ids"] = list(dict.fromkeys(value.get("cancer_ids", [])))
    item["evidence"] = [dict(e) for e in evidence]
    return item


@dataclass
class ResponseCheck:
    """The outcome of checking one answer: accepted items by key, and problems."""

    items: dict[str, list[tuple[dict, list[dict]]]] = field(default_factory=dict)
    problems: list[Problem] = field(default_factory=list)
    dropped: list[dict] = field(default_factory=list)

    @property
    def structural(self) -> list[Problem]:
        return [p for p in self.problems if not p.grounding]

    @property
    def grounding(self) -> list[Problem]:
        return [p for p in self.problems if p.grounding]


def check_items(
    value: dict,
    item_types: Sequence[ItemType],
    level: str,
    locate: Callable[[dict, str], tuple[dict | None, str | None]],
    *,
    linking: bool = False,
) -> ResponseCheck:
    """Check every item list of a parsed answer and resolve its evidence.

    ``locate(quote_item, path)`` returns ``(resolved evidence, None)`` or
    ``(None, reason)``. An item keeps its resolved evidence; quotes that are not
    found are grounding problems, and the item is listed in ``dropped`` when
    none of its quotes was found (callers decide whether to retry or drop it).
    """
    result = ResponseCheck()
    for item_type in item_types:
        entries = value.get(item_type.key)
        if not isinstance(entries, list):
            result.problems.append(Problem(item_type.key, f"{item_type.key}: must be a list",
                                           f"{item_type.key}: not a list"))
            continue
        if len(entries) > (item_type.max_items if level == "document" else 4 * item_type.max_items):
            result.problems.append(Problem(item_type.key, f"{item_type.key}: too many items",
                                           f"{item_type.key}: too many items"))
            continue
        accepted = result.items.setdefault(item_type.key, [])
        for index, entry in enumerate(entries):
            path = f"{item_type.key}[{index}]"
            problems = check_item(item_type, level, entry, path, linking=linking)
            if problems:
                result.problems += problems
                continue
            resolved = []
            for number, quote in enumerate(entry["evidence"]):
                if not quote["quote"].strip():
                    # No quote, no support: the item is ungrounded like an invented one.
                    found, reason = None, "the quote is empty; quote the source or remove the item"
                else:
                    found, reason = locate(quote, f"{path}.evidence[{number}]")
                if found is None:
                    where = f"{path}.evidence[{number}]"
                    snippet = quote["quote"][:120]
                    result.problems.append(Problem(
                        where, f"{where}: {reason}: {snippet!r}", f"{where}: {reason}", True,
                    ))
                elif found not in resolved:
                    resolved.append(found)
            if resolved:
                accepted.append((entry, resolved))
            else:
                result.dropped.append({"item_type": item_type.name, "index": index,
                                       "reason": "no evidence quote was found in the source"})
    return result


__all__ = [
    "BIOMARKER",
    "CORE_ITEM_TYPES",
    "DIAGNOSIS",
    "DISEASE_ASSESSMENT",
    "EVENT",
    "Field",
    "IMAGING_STUDY",
    "ItemType",
    "LEVELS",
    "PERFORMANCE_STATUS",
    "PROCEDURE",
    "Problem",
    "ResponseCheck",
    "SCHEMA_VERSION",
    "TREATMENT",
    "check_item",
    "check_items",
    "document_response_schema",
    "item_record_schema",
    "item_response_schema",
    "item_types_for",
    "parse_date",
    "patient_response_schema",
    "record_json_schema",
    "render_field_guide",
    "resolved_date",
    "response_format",
    "to_record_item",
]
