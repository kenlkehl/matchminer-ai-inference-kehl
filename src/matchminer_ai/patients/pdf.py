"""Patient-note PDF preparation helpers."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

from matchminer_ai.documents import PDFOCRMethod, ocr_pdf

PatientPDFPath = str | os.PathLike[str]
PatientPDFInput = PatientPDFPath | Sequence[PatientPDFPath]
PatientPDFProgress = Callable[[int, int, int, int, PDFOCRMethod], None]


def _normalize_patient_pdf_paths(pdf_paths: PatientPDFInput) -> list[Path]:
    if isinstance(pdf_paths, (str, os.PathLike)):
        raw_paths: list[PatientPDFPath] = [pdf_paths]
    elif isinstance(pdf_paths, Sequence) and not isinstance(
        pdf_paths, (bytes, bytearray)
    ):
        raw_paths = list(pdf_paths)
    else:
        raise TypeError(
            "pdf_paths must be a PDF path or an ordered sequence of PDF paths."
        )

    if not raw_paths:
        raise ValueError("At least one patient-record PDF is required.")

    paths: list[Path] = []
    for document_number, raw_path in enumerate(raw_paths, start=1):
        if not isinstance(raw_path, (str, os.PathLike)):
            raise TypeError(
                "Each patient-record PDF must be a string or path-like value; "
                f"item {document_number} is {type(raw_path).__name__}."
            )
        path = Path(raw_path)
        if path.suffix.lower() != ".pdf":
            raise ValueError(
                f"Patient-record item {document_number} must use a .pdf extension."
            )
        if not path.is_file():
            raise FileNotFoundError(
                f"Patient-record PDF {document_number} does not exist: {path}"
            )
        paths.append(path)
    return paths


def concatenate_patient_note_pdfs(
    pdf_paths: PatientPDFInput,
    *,
    force_ocr: bool = False,
    dpi: int = 200,
    min_embedded_text_chars: int = 50,
    min_ocr_confidence: float = 0.5,
    password: str | None = None,
    progress_callback: PatientPDFProgress | None = None,
) -> str:
    """Convert one or more local patient-record PDFs into one long note string.

    PDFs are processed in the supplied order. Each page uses the package's
    embedded-text extraction with local RapidOCR fallback, and numbered
    document markers preserve PDF boundaries in the combined text. Filenames
    and filesystem timestamps are deliberately not inserted into the patient
    text, because neither is reliable clinical provenance.

    Parameters
    ----------
    pdf_paths : str, os.PathLike[str], or sequence of paths
        One PDF path or an ordered sequence of PDF paths for one patient.
    force_ocr, dpi, min_embedded_text_chars, min_ocr_confidence, password
        Passed to :func:`matchminer_ai.documents.ocr_pdf` for each document.
        A supplied password is applied to every PDF.
    progress_callback : callable or None, optional
        Called after every processed page as
        ``callback(document_number, document_count, page_number, page_count,
        method)``, where ``method`` is ``"embedded"`` or ``"ocr"``.

    Returns
    -------
    str
        The extracted document text in input order, with numbered PDF boundary
        markers and form-feed page separators retained.

    Notes
    -----
    OCR is local, but the returned patient text reaches the configured LLM
    backend if it is passed to ``summarize_patients``. Use only an endpoint
    authorized for the source records' sensitivity.
    """
    sources = _normalize_patient_pdf_paths(pdf_paths)
    document_count = len(sources)
    documents: list[str] = []

    with tempfile.TemporaryDirectory(prefix="mmai_patient_pdfs_") as temp_dir:
        output_dir = Path(temp_dir)
        for document_index, source in enumerate(sources):
            document_number = document_index + 1

            def report_page(
                page_number: int,
                page_count: int,
                method: PDFOCRMethod,
                *,
                current_document: int = document_number,
            ) -> None:
                if progress_callback is not None:
                    progress_callback(
                        current_document,
                        document_count,
                        page_number,
                        page_count,
                        method,
                    )

            text_path = ocr_pdf(
                source,
                output_dir / f"patient-record-{document_number}.txt",
                force_ocr=force_ocr,
                dpi=dpi,
                min_embedded_text_chars=min_embedded_text_chars,
                min_ocr_confidence=min_ocr_confidence,
                password=password,
                progress_callback=report_page,
            )
            document_text = text_path.read_text(encoding="utf-8").strip()
            documents.append(
                f"=== Patient Record PDF {document_number} of {document_count} ===\n"
                f"{document_text}"
            )

    return "\n\n".join(documents)


__all__ = [
    "PatientPDFInput",
    "PatientPDFPath",
    "PatientPDFProgress",
    "concatenate_patient_note_pdfs",
]
