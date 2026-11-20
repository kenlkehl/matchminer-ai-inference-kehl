"""Structured extraction records as relational tables, for SQL workup checks.

One table per extracted item type (``imaging_studies``, ``procedures``,
``diagnoses``, ``biomarkers``, ``treatments``, ``disease_assessments``,
``performance_status``, ``lab_tests``, ``counseling``, ``events`` and any extension's), one row per item at
either level, plus ``patients``, ``documents``, an ``items`` view of the columns
every item table shares, ``vocabulary`` (every code and its meaning) and
``database_info``. Columns come from the item type declarations
(:mod:`matchminer_ai.patients.extraction.schema`), never from the data, so a query
written against one database runs on any other of the same ``DATABASE_VERSION``.

Dates follow the interval rules of the compiled-query evaluator
(:func:`~matchminer_ai.patients.workup_queries.evaluate.item_view`): a month- or
year-precision date spans its whole month or year (``date_lo`` to ``date_hi``),
an undated "previously performed" mention was on or before its note's date
(``date_lo`` NULL), and a planned undated item is on or after it (``date_hi``
NULL). NULL bounds are unbounded.

The tables hold no patient identifiers and no note text: patients are numbered
(``patient_key``) and the evidence quotes of each item stay in Python
(:attr:`ExtractionDatabase.views`), where results are linked back to them.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

from matchminer_ai.patients.extraction import vocabulary as v
from matchminer_ai.patients.extraction.schema import (
    SCHEMA_VERSION,
    ItemType,
    record_item_types,
)
from matchminer_ai.patients.workup_queries.evaluate import (
    ItemView,
    PatientData,
    _record_rows,
    date_interval,
    describe_item,
    parse_date,
    patient_data,
)
from matchminer_ai.patients.workup_queries.language import item_types

DATABASE_VERSION = "mmai-extraction-sql-v2"

#: Columns every item table starts with, and the ``items`` view holds.
COMMON_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("item_key", "BIGINT", "Unique item number; cite items by it (evidence_item_keys)."),
    ("patient_key", "INTEGER", "The patient (patients.patient_key)."),
    ("item_type", "VARCHAR", "The item type (imaging_study, procedure, ...)."),
    ("level", "VARCHAR", "document: from one note's record; patient: from the patient-level "
     "record built from all of the patient's notes. The same study is often extracted from "
     "several notes and at both levels."),
    ("note_id", "VARCHAR", "The note a document-level item came from (documents.note_id); "
     "NULL for patient-level items."),
    ("note_date", "DATE", "That note's date (NULL for patient-level items or undated notes)."),
    ("document_type", "VARCHAR", "The type of that note (NULL for patient-level items)."),
    ("record_item_id", "VARCHAR", "A patient-level item's id within its record (I1, D2...)."),
    ("occurrence", "VARCHAR", "performed or planned (derived in code: planned imaging, "
     "procedures and treatments are planned; every other item is performed)."),
    ("date_value", "VARCHAR", "The item's primary date as extracted (YYYY, YYYY-MM or "
     "YYYY-MM-DD), the note date for items dated by their note, or the anchoring note date "
     "of an undated item (see date_bound); NULL when unknown."),
    ("date_precision", "VARCHAR", "day, month, year or unknown."),
    ("date_source", "VARCHAR", "document_text (stated in the note), note_date (dated by its "
     "note), evidence_note_date (a patient-level item dated by its earliest evidence note) or "
     "unknown."),
    ("date_bound", "VARCHAR", "NULL for dated items; on_or_before: undated, happened on or "
     "before date_value; on_or_after: undated and planned, on or after date_value."),
    ("date_lo", "DATE", "Earliest day the item can have happened (NULL: unbounded)."),
    ("date_hi", "DATE", "Latest day the item can have happened (NULL: unbounded)."),
    ("label", "VARCHAR", "A short code-made description of the item's main fields."),
)
_OCCURRENCE_COLUMNS = (
    ("relation", "VARCHAR", "Document-level items only (NULL at the patient level): how the "
     "study or procedure relates to the note.", v.DOCUMENT_RELATIONS),
    ("source_of_document", "BOOLEAN", "Document-level items only: true when this note is "
     "the report of this study or procedure.", None),
    ("status", "VARCHAR", "performed or planned, as extracted.", v.OCCURRENCE_STATUSES),
)
_SQL_TYPES = {"text": "VARCHAR", "text_list": "VARCHAR[]", "choice": "VARCHAR",
              "choice_list": "VARCHAR[]", "integer": "INTEGER"}
PATIENT_COLUMNS = (
    ("patient_key", "INTEGER", "The patient's number in this database."),
    ("reference_date", "DATE", "The date a workup is evaluated at: the patient's latest note "
     "date unless the caller supplies one. Items after it are still in the tables."),
    ("last_note_date", "DATE", "The patient's latest note date."),
    ("document_records", "INTEGER", "Notes with a document-level extraction record."),
    ("failed_document_records", "INTEGER", "Notes whose extraction failed (no items)."),
    ("partial_document_records", "INTEGER", "Notes whose extraction kept only some items."),
    ("has_patient_record", "BOOLEAN", "Whether a patient-level record exists."),
    ("notes_without_records", "INTEGER", "The caller's notes no document record covers."),
    ("extraction_incomplete", "BOOLEAN", "Some notes or chunks failed or lack records, so a "
     "missing item may be an extraction gap (code turns not_met into unknown for these)."),
    ("item_count", "INTEGER", "Items extracted for the patient, both levels."),
    ("extracted_item_types", "VARCHAR[]", "The item types every one of this patient's "
     "records was extracted for (older extraction schemas lack some, such as lab_test and "
     "counseling): an absent type cannot document anything for this patient."),
)
COVERAGE_COLUMNS = (
    ("item_type", "VARCHAR", "An item type (its table is in table_name)."),
    ("table_name", "VARCHAR", "The item type's table."),
    ("patients_covered", "INTEGER", "Patients whose records were all extracted for it."),
    ("patients", "INTEGER", "Patients in this database."),
)
DOCUMENT_COLUMNS = (
    ("patient_key", "INTEGER", "The patient."),
    ("note_id", "VARCHAR", "The note's code-derived id."),
    ("note_date", "DATE", "The note's date (NULL when undated)."),
    ("note_type", "VARCHAR", "The note type supplied with the note."),
    ("document_type", "VARCHAR", "The document type the extraction assigned."),
    ("extraction_status", "VARCHAR", "complete, partial or failed."),
    ("item_count", "INTEGER", "Items extracted from the note."),
)
VOCABULARY_COLUMNS = (
    ("table_name", "VARCHAR", "The table whose column uses the code."),
    ("column_name", "VARCHAR", "The column."),
    ("code", "VARCHAR", "The code stored in the column."),
    ("meaning", "VARCHAR", "What the code means."),
    ("code_group", "VARCHAR", "A coarser group of the code, when the vocabulary has one."),
)


@dataclass(frozen=True)
class Column:
    name: str
    sql_type: str
    description: str
    vocabulary: v.Vocabulary | None = None
    source: str | None = None  # item field name, for fields; None for code-derived columns
    part: str | None = None  # value / precision / source / lo / hi for date fields


def table_columns(item_type: ItemType) -> list[Column]:
    """The columns of ``item_type``'s table, from its declaration alone."""
    columns = [Column(n, t, d) for n, t, d in COMMON_COLUMNS]
    if item_type.occurrence:
        columns += [Column(n, t, d, voc, source=n) for n, t, d, voc in _OCCURRENCE_COLUMNS]
    if item_type.linkable_levels:
        columns.append(Column("cancer_ids", "VARCHAR[]", "Patient-level ids (D1, D2...) of the "
                              "diagnoses the item concerns, when linked; [] otherwise.",
                              source="cancer_ids"))
    for item_field in item_type.fields:
        if item_field.kind == "date":
            name = item_field.name
            columns += [
                Column(name, "VARCHAR", item_field.description + " As extracted (YYYY, YYYY-MM "
                       "or YYYY-MM-DD); NULL when not stated.", source=name, part="value"),
                Column(f"{name}_precision", "VARCHAR", "day, month, year or unknown.",
                       source=name, part="precision"),
                Column(f"{name}_source", "VARCHAR", "Where the date came from.", source=name,
                       part="source"),
                Column(f"{name}_lo", "DATE", f"First day {name} can be (NULL: unknown).",
                       source=name, part="lo"),
                Column(f"{name}_hi", "DATE", f"Last day {name} can be (NULL: unknown).",
                       source=name, part="hi"),
            ]
            continue
        description = item_field.description
        if item_field.kind == "text":
            description += " NULL when not stated."
        columns.append(Column(item_field.name, _SQL_TYPES[item_field.kind], description,
                              item_field.vocabulary, source=item_field.name))
        if item_field.vocabulary is v.PATHOLOGY_PROCEDURES:
            columns.append(Column("procedure_group", "VARCHAR", "The procedure_type's group "
                                  "(derived in code).", source="procedure_group"))
        if item_field.name == "agents":
            columns.append(Column("agents_ncit_codes", "VARCHAR[]", "NCIt codes of the agents "
                                  "when the extraction added them; [] otherwise.",
                                  source="agents_ncit"))
    return columns


def _arrow_type(sql_type):
    import pyarrow as pa

    return {"VARCHAR": pa.string(), "BIGINT": pa.int64(), "INTEGER": pa.int32(),
            "BOOLEAN": pa.bool_(), "DATE": pa.date32(),
            "VARCHAR[]": pa.list_(pa.string())}[sql_type]


def _day(ordinal) -> date | None:
    return None if ordinal is None or math.isinf(ordinal) else date.fromordinal(int(ordinal))


def _date(value) -> date | None:
    if not isinstance(value, str) or len(value) < 10:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _text(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    value = str(value)
    return value if value.strip() else None


def _strings(value) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [str(x) for x in value if x is not None and str(x).strip()]


def _field_value(column: Column, item: dict):
    if column.source == "agents_ncit":
        return [str(e["ncit_code"]) for e in item.get("agents_ncit") or []
                if isinstance(e, dict) and e.get("ncit_code")]
    raw = item.get(column.source)
    if column.part is not None:
        when = raw if isinstance(raw, dict) else {}
        value, precision = when.get("value"), when.get("precision") or "unknown"
        if column.part == "value":
            return _text(value)
        if column.part == "precision":
            return precision
        if column.part == "source":
            return when.get("source") or "unknown"
        lo, hi = date_interval(value, precision) if value else (-math.inf, math.inf)
        return _day(lo if column.part == "lo" else hi)
    if column.sql_type == "VARCHAR[]":
        return _strings(raw)
    if column.sql_type == "INTEGER":
        return int(raw) if isinstance(raw, (int, float)) and not isinstance(raw, bool) else None
    if column.sql_type == "BOOLEAN":
        return bool(raw) if isinstance(raw, bool) else None
    return _text(raw)


def _common_values(view: ItemView, key: int, patient_key: int) -> dict:
    return {
        "item_key": key,
        "patient_key": patient_key,
        "item_type": view.item_type,
        "level": view.level,
        "note_id": view.note_id if view.level == "document" else None,
        "note_date": _date(view.note_date) if view.level == "document" else None,
        "document_type": _text(view.document_type) if view.level == "document" else None,
        "record_item_id": _text(view.item.get("item_id")),
        "occurrence": view.status,
        "date_value": _text(view.date_value),
        "date_precision": view.precision,
        "date_source": view.source,
        "date_bound": view.bound,
        "date_lo": _day(view.lo),
        "date_hi": _day(view.hi),
        "label": describe_item(view),
    }


@dataclass
class ExtractionDatabase:
    """Extraction records as Arrow tables, with the links back to patients and quotes.

    ``tables`` are the database tables by name; ``patient_ids[k - 1]`` is the
    patient with ``patient_key`` k; ``views[item_key]`` is the item (with its
    evidence quotes); ``data`` is :func:`patient_data` by patient id.
    """

    tables: dict
    patient_ids: list[str]
    views: dict[int, ItemView]
    data: dict[str, PatientData]
    item_tables: dict[str, str]
    extensions: list[str]
    fingerprint: str
    covered: dict[str, frozenset] | None = None

    def uncovered(self, patient_id: str, sql: str) -> list[str]:
        """Item tables ``sql`` names that this patient's extraction did not cover."""
        covered = (self.covered or {}).get(patient_id)
        if covered is None:
            return []
        return [table for name, table in self.item_tables.items()
                if name not in covered and re.search(rf"\b{re.escape(table)}\b", sql, re.I)]

    def statements(self) -> list[str]:
        """Trusted DDL run after the tables are loaded: the ``items`` view."""
        return [items_view_sql(self.item_tables.values())]

    @property
    def patient_keys(self) -> dict[str, int]:
        return {pid: n for n, pid in enumerate(self.patient_ids, 1)}

    def summary(self) -> dict:
        return {"patients": len(self.patient_ids), "items": len(self.views),
                "documents": self.tables["documents"].num_rows}


def items_view_sql(tables: Iterable[str]) -> str:
    names = ", ".join(name for name, _, _ in COMMON_COLUMNS)
    parts = [f"SELECT {names} FROM {table}" for table in tables]
    return "CREATE VIEW items AS " + " UNION ALL ".join(parts)


def _vocabulary_rows(types: Sequence[ItemType]) -> list[dict]:
    rows = []

    def add(table, column, vocabulary):
        for term in vocabulary.terms:
            rows.append({"table_name": table, "column_name": column, "code": term.code,
                         "meaning": term.description, "code_group": term.group})

    for item_type in types:
        for column in table_columns(item_type):
            if column.vocabulary is not None:
                add(item_type.key, column.name, column.vocabulary)
        if any(f.vocabulary is v.PATHOLOGY_PROCEDURES for f in item_type.fields):
            for group in sorted({t.group for t in v.PATHOLOGY_PROCEDURES.terms if t.group}):
                rows.append({"table_name": item_type.key, "column_name": "procedure_group",
                             "code": group, "meaning": "A group of procedure_type codes.",
                             "code_group": None})
    add("documents", "document_type", v.DOCUMENT_TYPES)
    for code, meaning in (("performed", "Done (or, for non-occurrence items, documented)."),
                          ("planned", "Ordered, scheduled or planned; not documented as done.")):
        rows.append({"table_name": "items", "column_name": "occurrence", "code": code,
                     "meaning": meaning, "code_group": None})
    return rows


def record_extensions(records) -> list[str]:
    """The extensions the records were extracted with (their ``extensions`` lists)."""
    names = set()
    for row in _record_rows(records):
        record = row.get("record")
        if isinstance(record, dict):
            names.update(str(e) for e in record.get("extensions") or [])
    return sorted(names)


def _fingerprint(tables: Mapping) -> str:
    import pyarrow as pa

    digest = hashlib.sha256(DATABASE_VERSION.encode())
    for name in sorted(tables):
        sink = io.BytesIO()
        with pa.ipc.new_stream(sink, tables[name].schema) as writer:
            writer.write_table(tables[name])
        digest.update(name.encode())
        digest.update(hashlib.sha256(sink.getvalue()).digest())
    return digest.hexdigest()


def _table(columns, rows):
    import pyarrow as pa

    schema = pa.schema([(c[0], _arrow_type(c[1])) for c in columns])
    return pa.table({name: [row.get(name) for row in rows] for name, *_ in columns},
                    schema=schema)


def build_extraction_database(
    records,
    *,
    coverage: Mapping[str, Mapping] | None = None,
    patient_ids: Sequence[str] | None = None,
    reference_dates: Mapping[str, object] | str | None = None,
    extensions: Iterable[str] | None = None,
) -> ExtractionDatabase:
    """The relational tables of extraction records (no I/O).

    ``records`` is any input :func:`~matchminer_ai.patients.workup_queries.patient_data`
    accepts (``extract_document_records`` / ``extract_patient_records`` output, a
    list of them, or record dicts); ``coverage`` adds counts such as
    ``notes_without_records`` by patient id; ``patient_ids`` fixes the patients
    and their order (patients without records get a row with no items);
    ``reference_dates`` is one date or dates by patient id (default: each
    patient's latest note date).
    """
    if extensions is None:
        extensions = record_extensions(records)
    extensions = sorted(set(extensions))
    types = item_types(extensions)
    data = patient_data(records, coverage=coverage)
    ids = [str(p) for p in patient_ids] if patient_ids is not None else sorted(data)
    if len(set(ids)) != len(ids):
        raise ValueError("patient_ids repeat")
    keys = {pid: n for n, pid in enumerate(ids, 1)}
    by_type = {t.name: t for t in types}
    everything = frozenset(by_type)
    covered: dict[str, frozenset] = {}
    for row in _record_rows(records):
        record = row.get("record")
        if not isinstance(record, dict):
            continue
        pid = str(row.get("patient_id") or record.get("patient_id"))
        found = record_item_types(record) & everything
        covered[pid] = covered[pid] & found if pid in covered else found
    columns = {t.name: table_columns(t) for t in types}
    rows: dict[str, list[dict]] = {t.name: [] for t in types}
    views: dict[int, ItemView] = {}
    patients, documents = [], []
    for pid in ids:
        patient = data.get(pid) or PatientData(pid)
        key = keys[pid]
        for view in patient.items:
            if view.item_type not in by_type:
                continue
            item_key = len(views) + 1
            views[item_key] = view
            row = _common_values(view, item_key, key)
            for column in columns[view.item_type][len(COMMON_COLUMNS):]:
                if column.name in {"relation", "source_of_document"} and view.level != "document":
                    row[column.name] = None
                else:
                    row[column.name] = _field_value(column, view.item)
            rows[view.item_type].append(row)
        reference = reference_dates.get(pid) if isinstance(reference_dates, Mapping) \
            else reference_dates
        if reference is None:
            reference = patient.last_note_date
        value, precision = parse_date(reference)
        reference_day = None
        if value is not None:
            reference_day = _day(date_interval(value, precision)[1])
        patients.append({
            "patient_key": key,
            "reference_date": reference_day,
            "last_note_date": _date(patient.last_note_date),
            "document_records": patient.documents,
            "failed_document_records": patient.failed_documents,
            "partial_document_records": patient.partial_documents,
            "has_patient_record": bool(patient.patient_records),
            "notes_without_records": patient.notes_without_records,
            "extraction_incomplete": patient.incomplete,
            "item_count": len(patient.items),
            "extracted_item_types": sorted(covered.get(pid, everything)),
        })
    for row in _record_rows(records):
        record = row.get("record") if isinstance(row.get("record"), dict) else {}
        level = record.get("level") or ("document" if "note_id" in row else "patient")
        pid = str(row.get("patient_id") or record.get("patient_id"))
        if level != "document" or pid not in keys or "item" in row and "record" not in row:
            continue
        documents.append({
            "patient_key": keys[pid],
            "note_id": _text(row.get("note_id") or record.get("note_id")),
            "note_date": _date(row.get("note_date") or record.get("note_date")),
            "note_type": _text(row.get("note_type") or record.get("note_type")),
            "document_type": _text(record.get("document_type") or row.get("document_type")),
            "extraction_status": row.get("extraction_status") or (
                "complete" if record else "failed"),
            "item_count": sum(len(items) for items in (record.get("items") or {}).values()),
        })
    tables = {
        "patients": _table(PATIENT_COLUMNS, patients),
        "documents": _table(DOCUMENT_COLUMNS, documents),
    }
    item_tables = {}
    for item_type in types:
        spec = [(c.name, c.sql_type, c.description) for c in columns[item_type.name]]
        tables[item_type.key] = _table(spec, rows[item_type.name])
        item_tables[item_type.name] = item_type.key
    tables["vocabulary"] = _table(VOCABULARY_COLUMNS, _vocabulary_rows(types))
    coverage = {pid: covered.get(pid, everything) for pid in ids}
    tables["extraction_coverage"] = _table(COVERAGE_COLUMNS, [
        {"item_type": t.name, "table_name": t.key, "patients": len(ids),
         "patients_covered": sum(t.name in c for c in coverage.values())} for t in types])
    return ExtractionDatabase(tables=tables, patient_ids=ids, views=views, data=data,
                              item_tables=item_tables, extensions=list(extensions),
                              fingerprint=_fingerprint(tables), covered=coverage)


def database_info(database: ExtractionDatabase) -> dict:
    return {
        "database_version": DATABASE_VERSION,
        "extraction_schema_version": SCHEMA_VERSION,
        "fingerprint": database.fingerprint,
        "extensions": json.dumps(database.extensions),
        "item_tables": json.dumps(database.item_tables),
        **{k: str(n) for k, n in database.summary().items()},
    }


def write_extraction_database(path, database: ExtractionDatabase, *, private=False) -> Path:
    """Write ``database`` as a DuckDB file atomically (0600 when ``private``).

    The file holds patient-derived values (no ids or note text): keep it where the
    extraction records may be kept. Requires the optional ``duckdb`` dependency.
    """
    import duckdb

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700 if private else 0o777)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    os.close(fd)
    os.unlink(temporary)  # DuckDB creates the file itself.
    try:
        connection = duckdb.connect(temporary)
        try:
            for name, table in database.tables.items():
                connection.register("_incoming", table)
                connection.execute(f"CREATE TABLE {name} AS SELECT * FROM _incoming")
                connection.unregister("_incoming")
            for statement in database.statements():
                connection.execute(statement)
            info = {**database_info(database),
                    "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            connection.execute("CREATE TABLE database_info (key VARCHAR, value VARCHAR)")
            connection.executemany("INSERT INTO database_info VALUES (?, ?)", list(info.items()))
            connection.execute("CHECKPOINT")
        finally:
            connection.close()
        for leftover in (Path(temporary + ".wal"),):
            leftover.unlink(missing_ok=True)
        if private:
            os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        Path(temporary + ".wal").unlink(missing_ok=True)
        raise
    return path


def read_database_info(path) -> dict:
    """The ``database_info`` table of a file from :func:`write_extraction_database`."""
    import duckdb

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No extraction database at {path}")
    connection = duckdb.connect(str(path), read_only=True, config={
        "enable_external_access": False, "autoload_known_extensions": False,
        "autoinstall_known_extensions": False})
    try:
        return dict(connection.execute("SELECT key, value FROM database_info").fetchall())
    finally:
        connection.close()


def render_database_guide(extensions: Iterable[str] | None = ()) -> str:
    """The tables, their columns and every column's codes, for the agent's prompt."""
    types = item_types(extensions)

    def block(title, columns, note=""):
        lines = [title + (f"\n  {note}" if note else "")]
        for column in columns:
            name, sql_type, description, *rest = column
            vocabulary = rest[0] if rest else None
            line = f"  {name} {sql_type}: {description}"
            if vocabulary is not None:
                line += " Codes: " + ", ".join(vocabulary.codes) + "."
            lines.append(line)
        return "\n".join(lines)

    blocks = [
        block("patients (one row per patient)", PATIENT_COLUMNS),
        block("documents (one row per note with a document-level record)", DOCUMENT_COLUMNS),
        block("items (view: the shared columns of every item table)", COMMON_COLUMNS),
    ]
    shared = {name for name, _, _ in COMMON_COLUMNS}
    for item_type in types:
        columns = [(c.name, c.sql_type, c.description, c.vocabulary)
                   for c in table_columns(item_type) if c.name not in shared]
        note = f"{item_type.description} Levels: {', '.join(sorted(item_type.levels))}."
        if item_type.guidance:
            note += f" Extraction guidance: {item_type.guidance}"
        date_note = (f" Primary date (date_lo/date_hi): {item_type.primary_date}."
                     if item_type.primary_date else " Dated by its note.")
        blocks.append(block(f"{item_type.key} (item_type '{item_type.name}'; the shared "
                            "columns, then:)", columns, note + date_note))
    blocks.append(block("vocabulary (every code and its meaning)", VOCABULARY_COLUMNS))
    blocks.append(block("extraction_coverage (which item types the records were extracted "
                        "for)", COVERAGE_COLUMNS, "An item type with few covered patients was "
                        "not extracted for most records (an older extraction schema): an "
                        "empty table then says nothing about whether things were done."))
    return "\n\n".join(blocks)


__all__ = [
    "COMMON_COLUMNS",
    "COVERAGE_COLUMNS",
    "DATABASE_VERSION",
    "ExtractionDatabase",
    "build_extraction_database",
    "database_info",
    "read_database_info",
    "record_extensions",
    "render_database_guide",
    "table_columns",
    "write_extraction_database",
]
