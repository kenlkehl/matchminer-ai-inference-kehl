"""Structured clinical data extraction: document-level and patient-level records.

Research abstraction aids, not clinical documentation: every record needs human
review. :func:`extract_document_records` returns one record per note (its
document type, the imaging study or specimen procedure behind a report, and the
diagnoses, biomarkers, treatments, disease assessments and performance status
it states); :func:`extract_patient_records` returns one record per patient from
all of their notes (diagnoses, biomarkers, treatment lines, procedures, imaging
history and key dated events). Both use the item types and vocabularies of
:mod:`.schema` and :mod:`.vocabulary`, carry exact-quote evidence located in the
notes in code, and flatten to one table with :func:`extraction_items`.
"""

from .documents import extract_document_records
from .extensions import Extension, register_extension, registered_extensions
from .patient_records import extract_patient_records
from .schema import (
    CORE_ITEM_TYPES,
    SCHEMA_VERSION,
    Field,
    ItemType,
    document_response_schema,
    patient_response_schema,
    record_json_schema,
)
from .tables import extraction_items
from .vocabulary import VOCABULARIES, Term, Vocabulary

__all__ = [
    "CORE_ITEM_TYPES",
    "Extension",
    "Field",
    "ItemType",
    "SCHEMA_VERSION",
    "Term",
    "VOCABULARIES",
    "Vocabulary",
    "document_response_schema",
    "extract_document_records",
    "extract_patient_records",
    "extraction_items",
    "patient_response_schema",
    "record_json_schema",
    "register_extension",
    "registered_extensions",
]
