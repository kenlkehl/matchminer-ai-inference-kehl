"""Flatten extraction records into one table row per item, for queries.

Both levels flatten to the same columns, so document and patient items can be
concatenated and filtered alike. Every row has a uniform ``date`` (the item's
primary date, or for a document's assessments the note date) with its
``date_precision`` and ``date_source``, so timeline questions such as "a CT of
the chest within 30 days before a date" or "tissue obtained by core biopsy"
are ordinary DataFrame filters. Dates of month or year precision name their
first day only nominally; a query must respect the precision.
"""

from __future__ import annotations

import pandas as pd

from .extensions import registered_extensions
from .schema import CORE_ITEM_TYPES

ITEM_COLUMNS = [
    "patient_id",
    "level",
    "note_id",
    "note_date",
    "document_type",
    "item_type",
    "item_id",
    "date",
    "date_precision",
    "date_source",
]


def _primary_dates() -> dict[str, str | None]:
    types = list(CORE_ITEM_TYPES)
    for extension in registered_extensions().values():
        types.extend(extension.item_types)
    return {t.name: t.primary_date for t in types}


def extraction_items(records: pd.DataFrame | list[dict]) -> pd.DataFrame:
    """One row per extracted item of document or patient records.

    ``records`` is the output of :func:`extract_document_records` or
    :func:`extract_patient_records` (its ``record`` column), or a list of record
    dicts. Columns: ``ITEM_COLUMNS``, then every field of the items (a date field
    ``x`` becomes ``x``, ``x_precision`` and ``x_source``), ``evidence`` and the
    complete ``item`` dict. Failed records (None) contribute no rows.
    """
    if isinstance(records, pd.DataFrame):
        if "record" not in records:
            raise ValueError("records must have a record column")
        values = list(records["record"])
    else:
        values = list(records)
    primary = _primary_dates()
    rows = []
    for record in values:
        if not isinstance(record, dict):
            continue
        document = record["level"] == "document"
        for items in record["items"].values():
            for item in items:
                row = {
                    "patient_id": record["patient_id"],
                    "level": record["level"],
                    "note_id": record.get("note_id"),
                    "note_date": record.get("note_date"),
                    "document_type": record.get("document_type"),
                    "item_type": item["item_type"],
                    "item_id": item.get("item_id"),
                }
                date_field = primary.get(item["item_type"])
                if date_field is not None and isinstance(item.get(date_field), dict):
                    when = item[date_field]
                    row.update(date=when["value"], date_precision=when["precision"],
                               date_source=when["source"])
                elif document and record.get("note_date"):
                    row.update(date=record["note_date"], date_precision="day",
                               date_source="note_date")
                else:
                    row.update(date=None, date_precision="unknown", date_source="unknown")
                for name, value in item.items():
                    if name in {"item_type", "item_id", "evidence"}:
                        continue
                    if isinstance(value, dict) and {"value", "precision", "source"} <= set(value):
                        row[name] = value["value"]
                        row[f"{name}_precision"] = value["precision"]
                        row[f"{name}_source"] = value["source"]
                    else:
                        row[name] = value
                row["evidence"] = item["evidence"]
                row["item"] = item
                rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=[*ITEM_COLUMNS, "evidence", "item"])
    rest = [c for c in frame.columns if c not in ITEM_COLUMNS and c not in {"evidence", "item"}]
    return frame[[*ITEM_COLUMNS, *rest, "evidence", "item"]]


__all__ = ["ITEM_COLUMNS", "extraction_items"]
