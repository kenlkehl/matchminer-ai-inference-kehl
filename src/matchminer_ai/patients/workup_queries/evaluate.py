"""Deterministic evaluation of workup queries on one patient's extracted items.

No I/O and no model calls: a query (:mod:`.language`) and a patient's extraction
records go in, and a status with the matching items, their dates and evidence
quotes comes out. The same query and items always give the same result.

Every test is three-valued: **yes**, **unknown** or **no**.

* A filter on a field the note did not state (``not_stated``, ``""``, ``[]``)
  follows the filter's ``if_not_stated`` (unknown by default).
* Dates are intervals. A day-precision date is that day; a month or year is
  every day in it; a ``previously_performed`` mention without a date, or a
  patient-level study dated by its earliest evidence note
  (``evidence_note_date``), was done on or before that note's date; another
  undated item mentioned in a note happened on or before the note (or, when
  planned, after it). An item is in a window when its whole interval is, out of
  it when none of it is, and unknown otherwise. Anchors (a diagnosis date, the
  earliest matching treatment) are intervals too.
* The same study or procedure mentioned in several notes, or at both levels, is
  one *cluster* (same item type and kind, compatible dates, overlapping body
  regions for imaging); a criterion counts clusters, not mentions. An undated
  mention joins the latest compatible dated cluster. A cluster is dated by its
  most precise member and matches a filter when any member does (the report's
  own study, ``this_report``, decides when it contradicts a mention).
* A criterion holds (yes) when at least ``min_count`` clusters certainly match,
  is unknown when enough clusters might, and is no otherwise. Expressions
  combine criteria with Kleene logic (AND = min, OR = max, NOT swaps yes/no).

Statuses: ``met`` (the requirement holds), ``not_met`` (no extracted item
documents it: not documented, which is not evidence it was not done),
``unknown`` (some matching item's date or fields leave it open, its anchor is
missing, the extraction of some notes failed, or the requirement cannot be
expressed with extracted items) and ``not_applicable`` (documented facts in
``not_applicable_when``). Applicability is reported separately as ``applies``,
``not_applicable`` or ``uncertain``; ``not_applicable`` needs documented facts.
"""

from __future__ import annotations

import calendar
import copy
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from .language import (
    STATUS_TYPES,
    canonical_text,
    check_workup_query,
    item_types,
    query_fields,
)

NO, MAYBE, YES = 0, 1, 2
STATE_NAMES = {NO: "no", MAYBE: "unknown", YES: "yes"}
INF = math.inf
STATUSES = ("met", "not_met", "unknown", "not_applicable")
APPLICABILITY = ("applies", "not_applicable", "uncertain")
REASONS = (
    "documented",
    "not_documented",
    "planned_only",
    "uncertain_dates",
    "fields_not_stated",
    "anchor_unavailable",
    "incomplete_extraction",
    "no_extraction",
    "not_expressible",
    "query_unavailable",
    "documented_not_applicable",
)
NOTICE = (
    "Structured-extraction workup check for human review, not a determination. "
    "Queries compiled by an LLM from the guideline text ran in code on items an LLM "
    "extracted from the notes; check every item against its source note. Not documented "
    "means no extracted item matched, not that the step was not done. This does not "
    "establish overall guideline concordance."
)
MAX_CLUSTERS_SHOWN = 5
MAX_EVIDENCE = 16

_MODALITY = {
    "ct": "CT", "mri": "MRI", "pet_ct": "PET/CT", "pet_mri": "PET/MRI", "pet": "PET",
    "xray": "X-ray", "ultrasound": "Ultrasound", "echocardiography": "Echocardiogram",
    "mammography": "Mammogram", "bone_scan": "Bone scan",
    "nuclear_medicine_other": "Nuclear medicine study", "fluoroscopy": "Fluoroscopy",
    "bone_densitometry": "DXA",
}
#: Fields that say what an item is; a near miss must agree on every one a criterion tests.
IDENTITY_FIELDS = {
    "imaging_study": {"modality"},
    "procedure": {"procedure_type", "procedure_group"},
    "biomarker": {"biomarker"},
    "treatment": {"treatment_type", "regimen", "agents"},
    "diagnosis": {"primary_site", "histology"},
    "event": {"event_type"},
    "performance_status": {"scale"},
}
_STAGE = re.compile(r"(?:^|\bstage\s*)(iv|iii|ii|i|0|[1-4])(?=[abc]?[0-9]?(?:\b|$))")
_ROMAN = {"1": "I", "2": "II", "3": "III", "4": "IV"}


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------


def date_interval(value: str | None, precision: str) -> tuple[float, float]:
    """The ordinal-day interval of an extraction date (``(-inf, inf)`` when unknown)."""
    if not value or precision == "unknown":
        return -INF, INF
    year = int(value[:4])
    if precision == "year":
        return date(year, 1, 1).toordinal(), date(year, 12, 31).toordinal()
    month = int(value[5:7])
    if precision == "month":
        last = calendar.monthrange(year, month)[1]
        return date(year, month, 1).toordinal(), date(year, month, last).toordinal()
    day = date(year, month, int(value[8:10])).toordinal()
    return day, day


def parse_date(value) -> tuple[str | None, str]:
    """``(normalized value, precision)`` of a caller-supplied date: YYYY, YYYY-MM,
    YYYY-MM-DD, a datetime or a Timestamp."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None, "unknown"
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d"), "day"
    text = str(value).strip()
    match = re.fullmatch(r"(\d{4})(?:-(\d{1,2})(?:-(\d{1,2}))?)?(?:[T ].*)?", text)
    if not match:
        raise ValueError(f"Unreadable date {text!r}; use YYYY-MM-DD, YYYY-MM or YYYY.")
    year, month, day = match.groups()
    if month is None:
        return year, "year"
    if day is None:
        return f"{year}-{int(month):02d}", "month"
    date(int(year), int(month), int(day))
    return f"{year}-{int(month):02d}-{int(day):02d}", "day"


# ---------------------------------------------------------------------------
# Items
# ---------------------------------------------------------------------------


@dataclass
class ItemView:
    """One extracted item with its record context and date interval."""

    item_type: str
    level: str
    note_id: str | None
    note_date: str | None
    document_type: str | None
    item: dict
    lo: float
    hi: float
    date_value: str | None
    precision: str
    source: str
    status: str
    bound: str | None = None  # "on_or_before" / "on_or_after" for undated items

    @property
    def relation(self) -> str | None:
        return self.item.get("relation")

    @property
    def width(self) -> float:
        return self.hi - self.lo


def _primary_dates() -> dict[str, str | None]:
    return {t.name: t.primary_date for t in item_types(_all_extensions())}


def _all_extensions() -> list[str]:
    from matchminer_ai.patients.extraction.extensions import registered_extensions

    return list(registered_extensions())


def _status(item: dict) -> str:
    if item["item_type"] in {"imaging_study", "procedure"}:
        return item.get("status") or "performed"
    if item["item_type"] == "treatment":
        return "planned" if item.get("treatment_status") == "planned" else "performed"
    return "performed"


def item_view(item: dict, *, level: str, note_id=None, note_date=None, document_type=None,
              primary: Mapping[str, str | None] | None = None) -> ItemView:
    """An :class:`ItemView` of one record item, with the date rules of this module."""
    primary = primary if primary is not None else _primary_dates()
    status = _status(item)
    date_field = primary.get(item["item_type"])
    when = item.get(date_field) if date_field else None
    bound = None
    if date_field is None and note_date:
        value, precision, source = note_date[:10], "day", "note_date"
        lo, hi = date_interval(value, "day")
    elif isinstance(when, dict) and when.get("value"):
        value, precision, source = when["value"], when["precision"], when["source"]
        lo, hi = date_interval(value, precision)
        if source == "evidence_note_date":
            lo, bound = -INF, "on_or_before"
    else:
        value, precision, source = None, "unknown", "unknown"
        lo, hi = -INF, INF
        if level == "document":
            anchor = note_date
        else:
            dates = [e.get("note_date") for e in item.get("evidence", []) if e.get("note_date")]
            anchor = min(dates) if dates else None
        if anchor:
            day = date_interval(anchor[:10], "day")[0]
            if status == "planned":
                lo, bound = day, "on_or_after"
            else:
                hi, bound = day, "on_or_before"
            value = anchor[:10]
    return ItemView(item["item_type"], level, note_id, note_date, document_type, item, lo, hi,
                    value, precision, source, status, bound)


def _record_rows(records) -> list[dict]:
    """Rows of {patient_id, level, status, record, note_id, ...} from any supported input."""
    if isinstance(records, pd.DataFrame):
        return [dict(r) for r in records.to_dict("records")]
    if isinstance(records, dict):
        return [{"patient_id": records.get("patient_id"), "record": records}]
    rows = []
    for entry in records or []:
        if isinstance(entry, pd.DataFrame):
            rows.extend(dict(r) for r in entry.to_dict("records"))
        elif isinstance(entry, dict) and "record" in entry:
            rows.append(dict(entry))
        elif isinstance(entry, dict):
            rows.append({"patient_id": entry.get("patient_id"), "record": entry})
        else:
            raise TypeError("records must be extraction DataFrames, record dicts or item rows")
    return rows


@dataclass
class PatientData:
    """One patient's items and how complete their extraction is."""

    patient_id: str
    items: list[ItemView] = field(default_factory=list)
    documents: int = 0
    failed_documents: int = 0
    partial_documents: int = 0
    patient_records: int = 0
    failed_patient_records: int = 0
    failed_chunks: int = 0
    notes_without_records: int = 0
    last_note_date: str | None = None

    @property
    def has_records(self) -> bool:
        return bool(self.documents or self.patient_records or self.items)

    @property
    def incomplete(self) -> bool:
        return bool(self.failed_documents or self.failed_patient_records or self.failed_chunks
                    or self.notes_without_records)

    def coverage(self) -> dict:
        return {
            "documents": self.documents,
            "failed_documents": self.failed_documents,
            "partial_documents": self.partial_documents,
            "patient_records": self.patient_records,
            "failed_patient_records": self.failed_patient_records,
            "failed_chunks": self.failed_chunks,
            "notes_without_records": self.notes_without_records,
            "items": len(self.items),
            "last_note_date": self.last_note_date,
        }


def _later(a, b):
    b = b[:10] if isinstance(b, str) and b.strip() else None
    return b if a is None or (b is not None and b > a) else a


def patient_data(records, *, coverage: Mapping[str, Mapping] | None = None) -> dict[str, PatientData]:
    """Items and coverage by patient from extraction output.

    ``records`` is ``extract_document_records`` or ``extract_patient_records``
    output, ``extraction_items`` output, record dicts, or a list of these.
    ``coverage`` adds counts by patient, such as ``notes_without_records`` (notes
    the caller holds that no document record covers).
    """
    primary = _primary_dates()
    patients: dict[str, PatientData] = {}

    def get(pid):
        pid = str(pid)
        if pid not in patients:
            patients[pid] = PatientData(pid)
        return patients[pid]

    for row in _record_rows(records):
        if "item" in row and "record" not in row:
            data = get(row["patient_id"])
            level = row.get("level") or "document"
            note_date = row.get("note_date") if isinstance(row.get("note_date"), str) else None
            data.items.append(item_view(
                row["item"], level=level, note_id=row.get("note_id"), note_date=note_date,
                document_type=row.get("document_type") if isinstance(
                    row.get("document_type"), str) else None, primary=primary,
            ))
            data.last_note_date = _later(data.last_note_date, note_date)
            continue
        record = row.get("record")
        status = row.get("extraction_status") or ("complete" if isinstance(record, dict) else
                                                  "failed")
        pid = row.get("patient_id") or (record or {}).get("patient_id")
        if pid is None:
            raise ValueError("Every extraction row needs a patient_id")
        data = get(pid)
        level = (record or {}).get("level") or ("document" if "note_id" in row else "patient")
        if level == "document":
            data.documents += 1
            data.failed_documents += status == "failed"
            data.partial_documents += status == "partial"
            data.last_note_date = _later(data.last_note_date, (record or row).get("note_date"))
        else:
            data.patient_records += 1
            data.failed_patient_records += status == "failed"
            chunks = (record or {}).get("failed_chunks") or row.get("failed_chunks") or []
            data.failed_chunks += len(chunks) if isinstance(chunks, (list, tuple)) else 0
            data.last_note_date = _later(data.last_note_date, (record or {}).get("last_note_date"))
        if not isinstance(record, dict):
            continue
        for entries in record.get("items", {}).values():
            for item in entries:
                data.items.append(item_view(
                    item, level=level, note_id=record.get("note_id"),
                    note_date=record.get("note_date"), document_type=record.get("document_type"),
                    primary=primary,
                ))
    for pid, extra in (coverage or {}).items():
        data = get(pid)
        for key in ("notes_without_records", "failed_documents", "failed_chunks"):
            if key in extra:
                setattr(data, key, getattr(data, key) + int(extra[key]))
        if extra.get("last_note_date"):
            data.last_note_date = _later(data.last_note_date, str(extra["last_note_date"])[:10])
    return patients


# ---------------------------------------------------------------------------
# Field filters
# ---------------------------------------------------------------------------


def stage_group(text) -> str | None:
    """The main stage group (0, I, II, III, IV) of a stage written as text, or None."""
    if not isinstance(text, str):
        return None
    match = _STAGE.search(canonical_text(text))
    if not match:
        return None
    found = match.group(1)
    return _ROMAN.get(found, found.upper())


def _value(view: ItemView, name: str):
    if name == "level":
        return view.level
    if name == "document_type":
        return view.document_type
    return view.item.get(name)


def _stated(field, value) -> bool:
    if value is None:
        return False
    if field.kind == "choice":
        return value != "" and value != field.unknown
    if field.kind in {"choice_list", "text_list"}:
        return isinstance(value, list) and any(str(x).strip() for x in value)
    if field.kind == "text":
        return bool(str(value).strip())
    return not (field.name in {"line_of_therapy"} and value == 0)


def _contains(haystack: str, needle: str) -> bool:
    return f" {canonical_text(needle)} " in f" {canonical_text(haystack)} "


def filter_state(entry: dict, field, view: ItemView) -> tuple[int, str | None]:
    """The state of one filter on one item, and the field left unstated (if any)."""
    op, values = entry["op"], entry["values"]
    value = _value(view, field.name)
    stated = _stated(field, value)
    if op == "is_stated":
        return (YES if stated else NO), None
    if field.kind == "choice" and isinstance(value, str) and value in values:
        return (YES if op == "in" else NO), None
    if not stated:
        rule = entry["if_not_stated"]
        return {"unknown": MAYBE, "no_match": NO, "match": YES}[rule], field.name
    if op == "in":
        return (YES if value in values else NO), None
    if op == "not_in":
        return (NO if value in values else YES), None
    if op in {"contains_any", "contains_all", "contains_none"}:
        present = set(value)
        if op == "contains_any":
            return (YES if present & set(values) else NO), None
        if op == "contains_all":
            return (YES if set(values) <= present else NO), None
        return (NO if present & set(values) else YES), None
    texts = value if isinstance(value, list) else [value]
    texts = [str(t) for t in texts if str(t).strip()]
    if op == "text_contains_any":
        return (YES if any(_contains(t, n) for t in texts for n in values) else NO), None
    if op == "text_contains_none":
        return (NO if any(_contains(t, n) for t in texts for n in values) else YES), None
    if op == "text_equals_any":
        wanted = {canonical_text(n) for n in values}
        return (YES if any(canonical_text(t) in wanted for t in texts) else NO), None
    if op == "stage_group_in":
        group = stage_group(value)
        if group is None:
            return MAYBE, field.name
        return (YES if group in values else NO), None
    number = int(values[0])
    if op == "number_at_least":
        return (YES if value >= number else NO), None
    return (YES if value <= number else NO), None


# ---------------------------------------------------------------------------
# Clusters (the same study or procedure documented more than once)
# ---------------------------------------------------------------------------


def _norm(value) -> str:
    return canonical_text(value) if isinstance(value, str) else ""


def cluster_key(view: ItemView) -> tuple:
    """What must agree for two items to be one study, procedure, result or course."""
    item, kind = view.item, view.item_type
    if kind == "imaging_study":
        return (kind, view.status, item.get("modality"))
    if kind == "procedure":
        return (kind, view.status, item.get("procedure_type"))
    if kind == "biomarker":
        return (kind, _norm(item.get("biomarker")), item.get("result"), _norm(item.get("alteration")))
    if kind == "treatment":
        agents = tuple(sorted(_norm(a) for a in item.get("agents") or []))
        return (kind, view.status, item.get("treatment_type"), agents or _norm(item.get("regimen")))
    if kind == "diagnosis":
        return (kind, _norm(item.get("primary_site")), _norm(item.get("histology")))
    if kind == "event":
        return (kind, item.get("event_type"))
    if kind == "performance_status":
        return (kind, item.get("scale"), item.get("score"))
    if kind == "disease_assessment":
        return (kind, view.note_id, id(item))
    rest = {k: x for k, x in item.items() if k not in {"evidence", "item_id", "cancer_ids"}
            and not (isinstance(x, dict) and "precision" in x)}
    return (kind, json.dumps(rest, sort_keys=True, default=str))


def _contains_interval(outer: ItemView, inner: ItemView) -> bool:
    return outer.lo <= inner.lo and inner.hi <= outer.hi


def _regions_overlap(a: ItemView, b: ItemView) -> bool:
    if a.item_type != "imaging_study":
        return True
    ra, rb = set(a.item.get("body_regions") or []), set(b.item.get("body_regions") or [])
    return not ra or not rb or bool(ra & rb)


@dataclass
class Cluster:
    members: list[ItemView]

    @property
    def rep(self) -> ItemView:
        return self.members[0]


def _tightness(view: ItemView) -> float:
    """Among open-ended intervals, the earlier upper bound (or later lower bound) first."""
    if view.lo == -INF and view.hi < INF:
        return view.hi
    if view.hi == INF and view.lo > -INF:
        return -view.lo
    return 0


def _precision_rank(view: ItemView) -> tuple:
    source = {"document_text": 0, "note_date": 1, "evidence_note_date": 2, "unknown": 3}
    return (view.width, view.relation != "this_report", source.get(view.source, 3),
            view.level != "document", _tightness(view))


def cluster_items(views: Sequence[ItemView]) -> list[Cluster]:
    """Group items that document the same thing; the most precise member leads."""
    clusters: list[Cluster] = []
    keyed: dict[tuple, list[Cluster]] = {}
    for view in sorted(views, key=_precision_rank):
        key = cluster_key(view)
        candidates = [
            c for c in keyed.get(key, [])
            if _contains_interval(view, c.rep) and _regions_overlap(view, c.rep)
        ]
        if candidates:
            max(candidates, key=lambda c: c.rep.hi).members.append(view)
            continue
        cluster = Cluster([view])
        clusters.append(cluster)
        keyed.setdefault(key, []).append(cluster)
    return clusters


# ---------------------------------------------------------------------------
# Criteria and expressions
# ---------------------------------------------------------------------------


@dataclass
class ClusterResult:
    cluster: Cluster
    state: int
    field_state: int
    date_state: int
    reasons: list[str]


@dataclass
class CriterionResult:
    criterion: dict
    state: int
    definite: int
    possible: int
    candidates: int
    matches: list[ClusterResult]
    other_status: list[ClusterResult]
    anchor: dict | None
    reasons: list[str]
    # The closest items that did not match: (cluster, the fields or window they failed).
    near: list[tuple] = field(default_factory=list)


@dataclass
class Context:
    data: PatientData
    reference: tuple[float, float] | None
    reference_text: str | None
    anchors: Mapping[str, object]
    criteria: dict[str, dict] = field(default_factory=dict)
    types: dict[str, object] = field(default_factory=dict)
    by_type: dict[str, list[Cluster]] = field(default_factory=dict)
    results: dict[str, CriterionResult] = field(default_factory=dict)

    def clusters(self, item_type: str) -> list[Cluster]:
        if item_type not in self.by_type:
            views = [v for v in self.data.items if v.item_type == item_type]
            self.by_type[item_type] = cluster_items(views)
        return self.by_type[item_type]


def _date_text(view: ItemView) -> str:
    where = f"note {view.note_id}" if view.note_id else "patient-level record"
    if view.bound == "on_or_before":
        how = ("mentioned as previously performed" if view.relation == "previously_performed"
               else "dated by its earliest evidence note" if view.source == "evidence_note_date"
               else "undated mention")
        return f"undated; on or before {view.date_value} ({how}, {where})"
    if view.bound == "on_or_after":
        return f"undated; on or after {view.date_value} (planned, {where})"
    if view.date_value is None:
        return f"date unknown ({where})"
    source = {"document_text": "", "note_date": ", dated by its report"}.get(view.source, "")
    return f"{view.date_value} ({view.precision} precision{source}, {where})"


def describe_item(view: ItemView) -> str:
    """A short readable description of an item's main fields."""
    item, kind = view.item, view.item_type
    if kind == "imaging_study":
        name = _MODALITY.get(item.get("modality")) or item.get("modality_text") or "Imaging study"
        regions = ", ".join(r.replace("_", " ") for r in item.get("body_regions") or [])
        text = f"{name} ({regions})" if regions else name
        if item.get("contrast") and item["contrast"] != "not_stated":
            text += ", " + item["contrast"].replace("_", " ")
        if item.get("tracer"):
            text += f", {item['tracer']}"
    elif kind == "procedure":
        text = (item.get("procedure_type") or "procedure").replace("_", " ")
        if item.get("specimen_site"):
            text += f" of {item['specimen_site']}"
        if item.get("specimen_result") and item["specimen_result"] != "not_stated":
            text += f", {item['specimen_result'].replace('_', ' ')}"
    elif kind == "diagnosis":
        text = " ".join(x for x in (item.get("histology"), item.get("primary_site")) if x) or "cancer"
        stage = item.get("stage_group") or ""
        if stage:
            text += f", {stage}" if stage.casefold().startswith("stage") else f", stage {stage}"
    elif kind == "biomarker":
        name, alteration = item.get("biomarker") or "", item.get("alteration") or ""
        repeated = alteration and canonical_text(alteration) in canonical_text(name)
        text = name if repeated else " ".join(x for x in (name, alteration) if x)
        if item.get("result") and item["result"] != "not_stated":
            text += f" {item['result']}"
        if item.get("value"):
            text += f" ({item['value']})"
    elif kind == "treatment":
        text = item.get("regimen") or ", ".join(item.get("agents") or []) or "treatment"
        text += f" ({(item.get('treatment_type') or '').replace('_', ' ')}"
        if item.get("treatment_status") and item["treatment_status"] != "not_stated":
            text += f", {item['treatment_status']}"
        text += ")"
    elif kind == "performance_status":
        text = f"{(item.get('scale') or '').upper()} {item.get('score')}"
    elif kind == "disease_assessment":
        text = (item.get("disease_status") or "").replace("_", " ")
        if item.get("disease_sites"):
            text += f" ({', '.join(item['disease_sites'])})"
    elif kind == "event":
        text = f"{(item.get('event_type') or 'event').replace('_', ' ')}: {item.get('description', '')}"
    else:
        text = kind.replace("_", " ")
    return text.strip()


def _anchor(ctx: Context, window: dict) -> tuple[tuple[float, float] | None, dict]:
    """The anchor's interval (None when unavailable) and a description of it."""
    kind = window["anchor"]
    if kind == "reference_date":
        info = {"kind": kind, "date": ctx.reference_text}
        return ctx.reference, info
    if kind == "diagnosis_date":
        supplied = ctx.anchors.get("diagnosis_date")
        if supplied:
            value, precision = parse_date(supplied)
            return date_interval(value, precision), {"kind": kind, "date": value,
                                                     "source": "supplied"}
        diagnoses = [v for v in ctx.data.items if v.item_type == "diagnosis"]
        dated = [v for v in diagnoses if v.lo > -INF]
        if not dated:
            # Undated diagnoses leave the anchor unknown; only no diagnosis at all is absent.
            return None, {"kind": kind, "date": None,
                          "source": "undated diagnosis" if diagnoses else "no match"}
        first = min(dated, key=lambda v: (v.lo, v.hi))
        interval = (min(v.lo for v in dated), min(v.hi for v in dated))
        return interval, {"kind": kind, "date": first.date_value, "precision": first.precision,
                          "source": "earliest extracted diagnosis"}
    target = evaluate_criterion(ctx, window["anchor_criterion"])
    certain = [m for m in target.matches if m.state == YES]
    if not certain:
        # A possible match leaves the anchor unknown; only no match at all is absent.
        uncertain = any(m.state == MAYBE for m in target.matches)
        return None, {"kind": kind, "criterion": window["anchor_criterion"], "date": None,
                      "source": "uncertain" if uncertain else "no match"}
    # Every certain match counts, undated ones too: an undated course may be the earliest.
    pick = min if kind == "criterion_earliest" else max
    interval = (pick(m.cluster.rep.lo for m in certain), pick(m.cluster.rep.hi for m in certain))
    if interval == (-INF, INF):
        return None, {"kind": kind, "criterion": window["anchor_criterion"], "date": None,
                      "source": "undated"}
    dated = [m for m in certain if m.cluster.rep.lo > -INF] or certain
    chosen = pick(dated, key=lambda m: (m.cluster.rep.lo, m.cluster.rep.hi)).cluster.rep
    return interval, {"kind": kind, "criterion": window["anchor_criterion"],
                      "date": chosen.date_value, "precision": chosen.precision,
                      "item": describe_item(chosen)}


def _window_state(view: ItemView, bounds) -> int:
    (pos_lo, pos_hi), (def_lo, def_hi) = bounds
    if def_lo <= view.lo and view.hi <= def_hi:
        return YES
    if view.hi < pos_lo or view.lo > pos_hi:
        return NO
    return MAYBE


def _bounds(anchor: tuple[float, float], window: dict):
    before, after = window["days_before"], window["days_after"]
    a_lo, a_hi = anchor
    pos = (a_lo - before if before >= 0 else -INF, a_hi + after if after >= 0 else INF)
    definite = (a_hi - before if before >= 0 else -INF, a_lo + after if after >= 0 else INF)
    return pos, definite


def _date_reason(view: ItemView) -> str:
    if view.bound == "on_or_before":
        return "undated mention (on or before its note)"
    if view.bound == "on_or_after":
        return "undated plan"
    if view.precision in {"month", "year"}:
        return f"{view.precision}-precision date"
    return "date unknown"


def evaluate_criterion(ctx: Context, cid: str) -> CriterionResult:
    if cid in ctx.results:
        return ctx.results[cid]
    criterion = ctx.criteria[cid]
    item_type = ctx.types[criterion["item_type"]]
    fields = query_fields(item_type)
    window = criterion["window"]
    reasons: list[str] = []
    anchor_info = None
    bounds = None
    anchor_missing = False
    if window["anchor"] != "any_time":
        interval, anchor_info = _anchor(ctx, window)
        absent = anchor_info.get("source") == "no match"
        if interval is None and absent and window["if_no_anchor"] == "reference_date" \
                and ctx.reference:
            interval = ctx.reference
            anchor_info = {**anchor_info, "fallback": "reference_date", "date": ctx.reference_text}
        if interval is None:
            anchor_missing = True
            reasons.append("anchor_unavailable")
        else:
            bounds = _bounds(interval, window)
    clusters = ctx.clusters(criterion["item_type"])
    matches, other, near = [], [], []
    for cluster in clusters:
        member_states = []
        for view in cluster.members:
            state, unstated = YES, []
            for entry in criterion["filters"]:
                s, missing = filter_state(entry, fields[entry["field"]], view)
                state = min(state, s)
                if s == MAYBE and missing:
                    unstated.append(missing)
                if state == NO:
                    break
            member_states.append((state, unstated, view))
        # The report's own study or procedure decides when it is certain either way.
        reports = [s for s, _, v in member_states if v.relation == "this_report"]
        if YES in reports or (NO not in reports and any(s == YES for s, _, _ in member_states)):
            field_state, unstated = YES, []
        elif NO in reports:
            field_state, unstated = NO, []
        elif any(s == MAYBE for s, _, _ in member_states):
            field_state = MAYBE
            unstated = sorted({u for s, us, _ in member_states if s == MAYBE for u in us})
        else:
            field_state, unstated = NO, []
        rep = cluster.rep
        status = criterion["status"]
        wrong_status = status != "any" and item_type.name in STATUS_TYPES and rep.status != status
        if field_state == NO:
            failed = [e["field"] for e in criterion["filters"]
                      if filter_state(e, fields[e["field"]], rep)[0] == NO]
            identity = IDENTITY_FIELDS.get(item_type.name, set())
            if (not wrong_status and len(failed) < len(criterion["filters"])
                    and not identity & set(failed)):
                near.append((cluster, failed))
            continue
        if window["anchor"] == "any_time":
            date_state = YES
        elif anchor_missing:
            date_state = MAYBE
        else:
            date_state = _window_state(rep, bounds)
        cluster_reasons = []
        if unstated:
            cluster_reasons.append("not stated: " + ", ".join(unstated))
        if date_state == MAYBE:
            cluster_reasons.append("anchor date unavailable" if anchor_missing else
                                   _date_reason(rep) if rep.width > 0 else "anchor date imprecise")
        result = ClusterResult(cluster, min(field_state, date_state), field_state, date_state,
                               cluster_reasons)
        if result.state == NO:
            if not wrong_status:
                near.append((cluster, ["date (outside the window)"]))
            continue
        (other if wrong_status else matches).append(result)
    definite = sum(m.state == YES for m in matches)
    possible = sum(m.state == MAYBE for m in matches)
    need = criterion["min_count"]
    state = YES if definite >= need else MAYBE if definite + possible >= need else NO
    candidates = sum(len(c.members) for c in clusters)
    for m in matches:
        reasons.extend(m.reasons)
    def recent(view):
        return view.hi if view.hi < INF else view.lo

    order = sorted(matches, key=lambda m: (-m.state, m.cluster.rep.width > 0,
                                           -recent(m.cluster.rep)))
    near.sort(key=lambda n: (len(n[1]), n[0].rep.width > 0, -recent(n[0].rep)))
    result = CriterionResult(criterion, state, definite, possible, candidates, order, other,
                             anchor_info, sorted(set(reasons)), near[:2])
    ctx.results[cid] = result
    return result


def _literal(ctx, literal) -> int:
    state = evaluate_criterion(ctx, literal["criterion"]).state
    return 2 - state if literal["negated"] else state


def evaluate_expression(ctx: Context, expression: dict) -> tuple[int, list[tuple[int, list]]]:
    """The expression's state and each group's (state, [(literal, state)])."""
    groups = []
    overall = YES
    for group in expression["all_of"]:
        literals = [(lit, _literal(ctx, lit)) for lit in group["any_of"]]
        state = max(s for _, s in literals)
        groups.append((state, literals))
        overall = min(overall, state)
    return overall, groups


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


def _cluster_entry(result: ClusterResult, label: str) -> dict:
    rep = result.cluster.rep
    evidence, seen = [], set()
    for view in result.cluster.members:
        for e in view.item.get("evidence", []):
            key = (e.get("note_id"), e.get("start"), e.get("end"))
            if key not in seen:
                seen.add(key)
                evidence.append({**e, "criterion": label, "item": describe_item(view),
                                 "level": view.level})
    return {
        "state": STATE_NAMES[result.state],
        "item_type": rep.item_type,
        "description": describe_item(rep),
        "date": rep.date_value,
        "date_precision": rep.precision,
        "date_source": rep.source,
        "date_bound": rep.bound,
        "date_text": _date_text(rep),
        "relations": sorted({v.relation for v in result.cluster.members if v.relation}),
        "levels": sorted({v.level for v in result.cluster.members}),
        "note_ids": sorted({v.note_id for v in result.cluster.members if v.note_id}),
        "mentions": len(result.cluster.members),
        "reasons": result.reasons,
        "evidence": evidence,
    }


def _criterion_entry(result: CriterionResult) -> dict:
    c = result.criterion
    return {
        "id": c["id"],
        "label": c["label"],
        "item_type": c["item_type"],
        "status": c["status"],
        "state": STATE_NAMES[result.state],
        "min_count": c["min_count"],
        "definite_count": result.definite,
        "possible_count": result.possible,
        "items_of_type": result.candidates,
        "anchor": result.anchor,
        "reasons": result.reasons,
        "matches": [_cluster_entry(m, c["label"]) for m in result.matches[:MAX_CLUSTERS_SHOWN]],
        "closest_non_matches": [
            {"description": describe_item(cluster.rep), "date_text": _date_text(cluster.rep),
             "differs_in": failed, "note_ids": sorted({v.note_id for v in cluster.members
                                                       if v.note_id})}
            for cluster, failed in result.near
        ],
        "other_status": [_cluster_entry(m, c["label"])
                         for m in result.other_status[:MAX_CLUSTERS_SHOWN]],
    }


def _phrase(ctx: Context, literal: dict, state: int) -> str:
    result = evaluate_criterion(ctx, literal["criterion"])
    label = result.criterion["label"]
    best = result.matches[0] if result.matches else None
    if literal["negated"]:
        if state == YES:
            return f"no {label} documented"
        if state == NO and best:
            return f"{label}: {describe_item(best.cluster.rep)}, {_date_text(best.cluster.rep)}"
        return f"{label} possibly documented" if best else f"no {label} documented"
    if best is None:
        if result.near and not literal["negated"]:
            cluster, failed = result.near[0]
            return (f"no {label} found (closest: {describe_item(cluster.rep)}, "
                    f"{_date_text(cluster.rep)}, which differs in {', '.join(failed)})")
        return f"no {label} found"
    text = f"{label}: {describe_item(best.cluster.rep)}, {_date_text(best.cluster.rep)}"
    if result.criterion["min_count"] > 1:
        text += f" ({result.definite} certain, {result.possible} possible of " \
                f"{result.criterion['min_count']} needed)"
    if state == MAYBE and best.reasons:
        text += " [" + "; ".join(best.reasons) + "]"
    return text


def _explain(ctx: Context, groups, target: int) -> list[str]:
    """Phrases for the groups that decide an expression's state."""
    if target == YES:
        # One phrase per supporting item, naming every group it satisfies.
        shown: dict[int, list] = {}
        phrases_in_order: list = []
        for _, literals in groups:
            literal = next(lit for lit, s in literals if s == YES)
            result = evaluate_criterion(ctx, literal["criterion"])
            label = result.criterion["label"]
            if literal["negated"] or not result.matches or result.criterion["min_count"] > 1:
                phrases_in_order.append(_phrase(ctx, literal, YES))
                continue
            rep = result.matches[0].cluster.rep
            if id(rep) not in shown:
                shown[id(rep)] = [label]
                phrases_in_order.append(rep)
            elif label not in shown[id(rep)]:
                shown[id(rep)].append(label)
        return [
            p if isinstance(p, str)
            else f"{', '.join(shown[id(p)])}: {describe_item(p)}, {_date_text(p)}"
            for p in phrases_in_order
        ]
    phrases = []
    for state, literals in groups:
        if state != target:
            continue
        chosen = ([lit for lit, s in literals if s == MAYBE][:2] if target == MAYBE
                  else [lit for lit, _ in literals][:3])
        phrases.append(" or ".join(_phrase(ctx, lit, state) for lit in chosen))
    return phrases


def _evidence(ctx: Context, groups, states=(YES, MAYBE)) -> list[dict]:
    evidence, seen = [], set()
    for _, literals in groups:
        for literal, _ in literals:
            if literal["negated"]:
                continue
            result = evaluate_criterion(ctx, literal["criterion"])
            for match in result.matches:
                if match.state not in states:
                    continue
                entry = _cluster_entry(match, result.criterion["label"])
                for quote in entry["evidence"]:
                    key = (quote.get("note_id"), quote.get("start"), quote.get("end"))
                    if key in seen:
                        continue
                    seen.add(key)
                    evidence.append({**quote, "state": entry["state"],
                                     "date_text": entry["date_text"]})
    return evidence[:MAX_EVIDENCE]


def _planned(ctx: Context, groups) -> list[ClusterResult]:
    found = []
    for _, literals in groups:
        for literal, _ in literals:
            if not literal["negated"]:
                found.extend(evaluate_criterion(ctx, literal["criterion"]).other_status)
    return found


def evaluate_workup_query(
    query: dict | None,
    patient: PatientData,
    *,
    reference_date=None,
    anchors: Mapping[str, object] | None = None,
    extensions: Iterable[str] | None = None,
) -> dict:
    """Evaluate one compiled query on one patient's items (:func:`patient_data`).

    ``reference_date`` (YYYY-MM-DD) dates ``reference_date`` windows and the
    ``if_no_anchor`` fallback; it defaults to the patient's latest note date.
    ``anchors`` may supply ``diagnosis_date`` instead of the earliest extracted
    diagnosis. Returns ``status``, ``reason``, ``applicability``, a machine-made
    ``explanation``, each criterion's matches with dates and evidence, and the
    flattened ``evidence`` quotes (note id, note date, quote, character offsets).
    """
    extensions = _all_extensions() if extensions is None else list(extensions)
    if query is None:
        return _result("unknown", "query_unavailable", "uncertain",
                       "Unknown: no valid query could be compiled for this requirement.")
    query = check_workup_query(query, extensions)
    if not query["expressible"]:
        return _result("unknown", "not_expressible", "uncertain",
                       "Unknown: not checkable from extracted items ("
                       f"{query['not_expressible_reason'].strip()})", query=query)
    if not patient.has_records:
        return _result("unknown", "no_extraction", "uncertain",
                       "Unknown: no extraction records are available for this patient.",
                       query=query)
    if reference_date is None:
        reference_date = patient.last_note_date
    reference, reference_text = None, None
    if reference_date is not None:
        value, precision = parse_date(reference_date)
        if value is not None:
            reference, reference_text = date_interval(value, precision), value
    ctx = Context(patient, reference, reference_text, anchors or {},
                  criteria={c["id"]: c for c in query["criteria"]},
                  types={t.name: t for t in item_types(extensions)})
    applies, applies_groups = evaluate_expression(ctx, query["applies_when"])
    excluded, excluded_groups = (
        evaluate_expression(ctx, query["not_applicable_when"])
        if query["not_applicable_when"]["all_of"] else (NO, [])
    )
    if excluded == YES and applies == YES and query["applies_when"]["all_of"]:
        applicability = "uncertain"
    elif excluded == YES:
        applicability = "not_applicable"
    elif applies == YES:
        applicability = "applies"
    else:
        applicability = "uncertain"
    met, groups = evaluate_expression(ctx, query["requirement"])
    notes = []
    if applicability == "uncertain" and query["applies_when"]["all_of"] and applies != YES:
        notes.append("Applicability uncertain: " + "; ".join(
            _explain(ctx, applies_groups, MAYBE if applies == MAYBE else NO)) + ".")
    elif applicability == "uncertain":
        notes.append("Applicability uncertain: documented facts both support and contradict it.")
    evidence = _evidence(ctx, groups)
    if applicability == "not_applicable":
        status, reason = "not_applicable", "documented_not_applicable"
        text = "Not applicable: " + "; ".join(_explain(ctx, excluded_groups, YES)) + "."
        evidence = _evidence(ctx, excluded_groups, (YES,))
    elif met == YES:
        status, reason = "met", "documented"
        text = "Met: " + "; ".join(_explain(ctx, groups, YES)) + "."
    elif met == MAYBE:
        status = "unknown"
        found = {r for state, literals in groups if state == MAYBE for lit, s in literals
                 if s == MAYBE for r in evaluate_criterion(ctx, lit["criterion"]).reasons}
        reason = ("anchor_unavailable" if "anchor_unavailable" in found
                  else "uncertain_dates" if any("date" in r or "mention" in r or "plan" in r
                                                for r in found)
                  else "fields_not_stated")
        text = "Unknown: " + "; ".join(_explain(ctx, groups, MAYBE)) + "."
    else:
        planned = _planned(ctx, groups)
        phrases = _explain(ctx, groups, NO)
        count = sum(len(c.members) for t in ctx.by_type for c in ctx.by_type[t])
        if patient.incomplete:
            status, reason = "unknown", "incomplete_extraction"
            text = ("Unknown: not found in the extraction records (" + "; ".join(phrases)
                    + "), but " + _incomplete(patient) + ".")
        elif planned:
            status, reason = "not_met", "planned_only"
            plans = ", ".join(f"{describe_item(p.cluster.rep)} ({_date_text(p.cluster.rep)})"
                              for p in planned[:3])
            text = (f"Not documented as performed: {'; '.join(phrases)}. Planned or ordered: "
                    f"{plans}.")
            evidence = _evidence_planned(planned)
        else:
            status, reason = "not_met", "not_documented"
            text = (f"Not documented in the extraction records: {'; '.join(phrases)} "
                    f"(among {count} extracted items of the queried types). A missing item "
                    "does not mean the step was not done.")
    if notes and status != "not_applicable":
        text += " " + " ".join(notes)
    criteria = [_criterion_entry(evaluate_criterion(ctx, c["id"])) for c in query["criteria"]]
    limitations = list(query["approximations"])
    if patient.incomplete and status != "unknown":
        limitations.append(_incomplete(patient).capitalize() + ".")
    return _result(status, reason, applicability, text, query=query, criteria=criteria,
                   evidence=evidence, limitations=limitations,
                   reference_date=reference_text)


def _evidence_planned(planned: list[ClusterResult]) -> list[dict]:
    evidence = []
    for p in planned[:3]:
        entry = _cluster_entry(p, "planned")
        evidence.extend({**q, "state": "planned", "date_text": entry["date_text"]}
                        for q in entry["evidence"])
    return evidence[:MAX_EVIDENCE]


def _incomplete(patient: PatientData) -> str:
    parts = []
    if patient.failed_documents:
        parts.append(f"{patient.failed_documents} note(s) failed extraction")
    if patient.notes_without_records:
        parts.append(f"{patient.notes_without_records} note(s) have no extraction record")
    if patient.failed_patient_records:
        parts.append("the patient-level extraction failed")
    if patient.failed_chunks:
        parts.append(f"{patient.failed_chunks} patient-level chunk(s) failed")
    return " and ".join(parts) or "the extraction is incomplete"


def _result(status, reason, applicability, explanation, *, query=None, criteria=None,
            evidence=None, limitations=None, reference_date=None) -> dict:
    return {
        "status": status,
        "reason": reason,
        "applicability": applicability,
        "bottom_line": explanation,
        "criteria": criteria or [],
        "evidence": evidence or [],
        "limitations": limitations if limitations is not None else (
            list(query.get("approximations", [])) if query else []),
        "reference_date": reference_date,
        "query_summary": query.get("summary") if query else None,
    }


# ---------------------------------------------------------------------------
# Several requirements and patients
# ---------------------------------------------------------------------------


def check_workup_with_extractions(
    records,
    compiled: Sequence[dict],
    *,
    patient_ids: Sequence[str] | None = None,
    reference_dates: Mapping[str, object] | str | None = None,
    anchors: Mapping[str, Mapping[str, object]] | None = None,
    coverage: Mapping[str, Mapping] | None = None,
) -> dict:
    """Check every compiled workup query on every patient's extraction records.

    ``records``: ``extract_document_records`` and/or ``extract_patient_records``
    output (a DataFrame or a list of them), ``extraction_items`` output, or
    record dicts. ``compiled``: :func:`compile_workup_queries` output (one entry
    per requirement, in order). ``patient_ids`` adds patients with no records
    (their results are unknown) and fixes the order. ``reference_dates`` is one
    date for everyone or dates by patient; ``anchors`` gives a patient's
    ``diagnosis_date``; ``coverage`` adds counts such as
    ``notes_without_records`` by patient.

    Returns ``{"patients": [{"patient_id", "review": {"assessments", "notice",
    "metadata"}}], "aggregate": [...], "notice"}``; each assessment carries the
    requirement, its ``status``, ``applicability``, ``reason``, ``bottom_line``,
    criteria, ``evidence`` and ``limitations``. Pure computation: no I/O.
    """
    data = patient_data(records, coverage=coverage)
    ids = [str(p) for p in patient_ids] if patient_ids is not None else sorted(data)
    if len(set(ids)) != len(ids):
        raise ValueError("patient_ids repeat")
    extensions = _all_extensions()
    queries = []
    for entry in compiled:
        query = entry.get("query") if entry.get("status") == "compiled" else None
        queries.append(check_workup_query(query, extensions) if query is not None else None)
    patients = []
    for pid in ids:
        patient = data.get(pid) or PatientData(pid)
        if isinstance(reference_dates, Mapping):
            reference = reference_dates.get(pid)
        else:
            reference = reference_dates
        assessments = []
        for number, (entry, query) in enumerate(zip(compiled, queries, strict=True)):
            result = evaluate_workup_query(query, patient, reference_date=reference,
                                           anchors=(anchors or {}).get(pid), extensions=extensions)
            assessments.append({
                "recommendation_index": number,
                "recommendation": copy.deepcopy(entry["requirement"]),
                "name": entry["requirement"]["name"],
                **result,
                "query_key": entry.get("cache_key"),
            })
        patients.append({
            "patient_id": pid,
            "review": {
                "assessments": assessments,
                "notice": NOTICE,
                "metadata": {"method": "extraction_queries", "coverage": patient.coverage()},
            },
        })
    requirements = [entry["requirement"] for entry in compiled]
    return {"patients": patients, "aggregate": aggregate_extraction_workup(patients, requirements),
            "notice": NOTICE}


def aggregate_extraction_workup(reviews: Sequence[dict], requirements: Sequence[dict]) -> list[dict]:
    """Counts per requirement: statuses, applicability, reasons, and met among applicable.

    ``completed_applicable / applicable_assessed`` is met / (met + not met) among
    patients the requirement applies to: unknown and not-applicable results stay
    out of the denominator. It is a documentation fraction, not a concordance rate.
    """
    rows = []
    for number, requirement in enumerate(requirements):
        statuses = {s: 0 for s in STATUSES}
        applicability = {a: 0 for a in APPLICABILITY}
        reasons = {r: 0 for r in REASONS}
        met = assessed = 0
        for patient in reviews:
            items = patient["review"]["assessments"]
            if len(items) != len(requirements):
                raise ValueError("Incomplete extraction workup matrix.")
            row = items[number]
            if row["status"] not in statuses or row["applicability"] not in applicability:
                raise ValueError("Invalid extraction workup result.")
            statuses[row["status"]] += 1
            applicability[row["applicability"]] += 1
            reason = row.get("reason") or "not_documented"
            reasons[reason] = reasons.get(reason, 0) + 1
            if row["applicability"] == "applies" and row["status"] in {"met", "not_met"}:
                assessed += 1
                met += row["status"] == "met"
        rows.append({
            "recommendation_index": number,
            "recommendation": copy.deepcopy(requirement),
            "patient_count": len(reviews),
            "statuses": statuses,
            "applicability": applicability,
            "reasons": reasons,
            "applicable_assessed": assessed,
            "completed_applicable": met,
            "documented_completion_fraction": met / assessed if assessed else None,
        })
    return rows


__all__ = [
    "APPLICABILITY",
    "NOTICE",
    "PatientData",
    "REASONS",
    "STATUSES",
    "aggregate_extraction_workup",
    "check_workup_with_extractions",
    "cluster_items",
    "date_interval",
    "describe_item",
    "evaluate_workup_query",
    "item_view",
    "patient_data",
    "stage_group",
]
