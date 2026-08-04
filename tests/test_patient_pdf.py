from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from matchminer_ai.patients import concatenate_patient_note_pdfs
from matchminer_ai.patients import pdf as patient_pdf_module


def test_concatenate_patient_note_pdfs_preserves_input_order_and_progress(
    tmp_path,
    monkeypatch,
):
    sources = [tmp_path / "first.pdf", tmp_path / "second.pdf"]
    for source in sources:
        source.write_bytes(b"%PDF synthetic patient record")

    calls = []

    def fake_ocr(source, output, **kwargs):
        source_path = Path(source)
        calls.append((source_path, kwargs))
        kwargs["progress_callback"](1, 1, "embedded")
        output_path = Path(output)
        output_path.write_text(
            f"Extracted text from {source_path.stem}\n",
            encoding="utf-8",
        )
        return output_path

    monkeypatch.setattr(patient_pdf_module, "ocr_pdf", fake_ocr)
    progress = []

    text = concatenate_patient_note_pdfs(
        sources,
        progress_callback=lambda *args: progress.append(args),
    )

    assert [call[0] for call in calls] == sources
    assert text == (
        "=== Patient Record PDF 1 of 2 ===\n"
        "Extracted text from first\n\n"
        "=== Patient Record PDF 2 of 2 ===\n"
        "Extracted text from second"
    )
    assert progress == [
        (1, 2, 1, 1, "embedded"),
        (2, 2, 1, 1, "embedded"),
    ]


def test_concatenate_patient_note_pdfs_accepts_one_path(tmp_path, monkeypatch):
    source = tmp_path / "record.pdf"
    source.write_bytes(b"%PDF synthetic patient record")

    def fake_ocr(_source, output, **_kwargs):
        output_path = Path(output)
        output_path.write_text("One record", encoding="utf-8")
        return output_path

    monkeypatch.setattr(patient_pdf_module, "ocr_pdf", fake_ocr)

    assert concatenate_patient_note_pdfs(source) == (
        "=== Patient Record PDF 1 of 1 ===\nOne record"
    )


def test_concatenate_patient_note_pdfs_rejects_empty_sequence():
    with pytest.raises(ValueError, match="At least one"):
        concatenate_patient_note_pdfs([])


def test_concatenate_patient_note_pdfs_validates_all_sources_before_ocr(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "record.pdf"
    source.write_bytes(b"%PDF synthetic patient record")
    ocr_mock = MagicMock()
    monkeypatch.setattr(patient_pdf_module, "ocr_pdf", ocr_mock)

    with pytest.raises(ValueError, match="item 2"):
        concatenate_patient_note_pdfs([source, tmp_path / "not-a-pdf.txt"])

    ocr_mock.assert_not_called()
