from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from matchminer_ai.documents import PDFOCRError, PDFOCRNoTextError, ocr_pdf
from matchminer_ai.documents import ocr as ocr_module


class _FakePdfiumError(RuntimeError):
    pass


class _FakeTextPage:
    def __init__(self, text: str) -> None:
        self.text = text
        self.closed = False

    def get_text_bounded(self) -> str:
        return self.text

    def close(self) -> None:
        self.closed = True


class _FakePage:
    def __init__(self, text: str) -> None:
        self.text_page = _FakeTextPage(text)
        self.closed = False

    def get_textpage(self) -> _FakeTextPage:
        return self.text_page

    def close(self) -> None:
        self.closed = True


class _FakeDocument:
    def __init__(self, texts: list[str]) -> None:
        self.pages = [_FakePage(text) for text in texts]
        self.closed = False

    def __len__(self) -> int:
        return len(self.pages)

    def __getitem__(self, index: int) -> _FakePage:
        return self.pages[index]

    def close(self) -> None:
        self.closed = True


def _install_fake_pdfium(monkeypatch, texts: list[str]) -> _FakeDocument:
    document = _FakeDocument(texts)
    module = SimpleNamespace(
        PdfDocument=lambda *args, **kwargs: document,
        PdfiumError=_FakePdfiumError,
    )
    monkeypatch.setitem(sys.modules, "pypdfium2", module)
    return document


def test_ocr_pdf_preserves_embedded_text_and_page_boundaries(tmp_path, monkeypatch):
    source = tmp_path / "trial.pdf"
    source.write_bytes(b"%PDF-fake")
    document = _install_fake_pdfium(
        monkeypatch,
        [
            "  First page has enough embedded trial document text.  \r\n",
            "Second page also has enough embedded text for extraction.",
        ],
    )
    progress = []

    destination = ocr_pdf(
        source,
        min_embedded_text_chars=10,
        progress_callback=lambda page, total, method: progress.append(
            (page, total, method)
        ),
    )

    assert destination == source.with_suffix(".txt")
    assert destination.read_text(encoding="utf-8") == (
        "  First page has enough embedded trial document text.\n\n\f\n\n"
        "Second page also has enough embedded text for extraction.\n"
    )
    assert progress == [(1, 2, "embedded"), (2, 2, "embedded")]
    assert document.closed
    assert all(page.closed and page.text_page.closed for page in document.pages)


def test_ocr_pdf_uses_ocr_for_a_page_without_embedded_text(tmp_path, monkeypatch):
    source = tmp_path / "scan.pdf"
    source.write_bytes(b"%PDF-fake")
    _install_fake_pdfium(monkeypatch, [""])
    engine = object()
    monkeypatch.setattr(ocr_module, "_create_ocr_engine", lambda: engine)
    calls = []

    def fake_ocr_page(page, supplied_engine, *, dpi, min_confidence):
        calls.append((page, supplied_engine, dpi, min_confidence))
        return "Recognized scanned eligibility criteria"

    monkeypatch.setattr(ocr_module, "_ocr_page", fake_ocr_page)

    destination = ocr_pdf(source, dpi=300, min_ocr_confidence=0.75)

    assert destination.read_text(encoding="utf-8") == (
        "Recognized scanned eligibility criteria\n"
    )
    assert len(calls) == 1
    assert calls[0][1:] == (engine, 300, 0.75)


def test_ocr_pdf_uses_ocr_when_embedded_text_extraction_fails(tmp_path, monkeypatch):
    source = tmp_path / "damaged-text-layer.pdf"
    source.write_bytes(b"%PDF-fake")
    _install_fake_pdfium(monkeypatch, ["unavailable"])
    monkeypatch.setattr(ocr_module, "_create_ocr_engine", object)

    def fail_embedded_extraction(_page):
        raise _FakePdfiumError

    monkeypatch.setattr(ocr_module, "_extract_embedded_text", fail_embedded_extraction)
    monkeypatch.setattr(
        ocr_module,
        "_ocr_page",
        lambda *args, **kwargs: "Recovered by OCR",
    )

    destination = ocr_pdf(source)

    assert destination.read_text(encoding="utf-8") == "Recovered by OCR\n"


def test_ocr_pdf_force_ocr_ignores_embedded_text(tmp_path, monkeypatch):
    source = tmp_path / "form.pdf"
    source.write_bytes(b"%PDF-fake")
    document = _install_fake_pdfium(monkeypatch, ["Native text is present."])
    monkeypatch.setattr(ocr_module, "_create_ocr_engine", object)
    monkeypatch.setattr(ocr_module, "_ocr_page", lambda *args, **kwargs: "Form OCR")

    destination = ocr_pdf(source, force_ocr=True)

    assert destination.read_text(encoding="utf-8") == "Form OCR\n"
    assert not document.pages[0].text_page.closed


def test_ocr_pdf_does_not_write_a_file_when_no_text_is_found(tmp_path, monkeypatch):
    source = tmp_path / "blank.pdf"
    source.write_bytes(b"%PDF-fake")
    _install_fake_pdfium(monkeypatch, [""])
    monkeypatch.setattr(ocr_module, "_create_ocr_engine", object)
    monkeypatch.setattr(ocr_module, "_ocr_page", lambda *args, **kwargs: "")

    with pytest.raises(PDFOCRNoTextError, match="No text"):
        ocr_pdf(source)

    assert not source.with_suffix(".txt").exists()


def test_ocr_pdf_does_not_overwrite_by_default(tmp_path):
    source = tmp_path / "trial.pdf"
    destination = tmp_path / "trial.txt"
    source.write_bytes(b"%PDF-fake")
    destination.write_text("keep me", encoding="utf-8")

    with pytest.raises(FileExistsError, match="overwrite=True"):
        ocr_pdf(source)

    assert destination.read_text(encoding="utf-8") == "keep me"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"dpi": 71}, "dpi"),
        ({"min_embedded_text_chars": -1}, "min_embedded_text_chars"),
        ({"min_ocr_confidence": 1.1}, "min_ocr_confidence"),
    ],
)
def test_ocr_pdf_validates_options(tmp_path, kwargs, match):
    source = tmp_path / "trial.pdf"
    source.write_bytes(b"%PDF-fake")

    with pytest.raises(ValueError, match=match):
        ocr_pdf(source, **kwargs)


def test_rapidocr_output_filters_low_confidence_and_empty_lines():
    result = SimpleNamespace(
        txts=("high confidence", "low confidence", " "),
        scores=(0.99, 0.2, 0.99),
    )

    assert ocr_module._text_from_rapidocr_output(result, 0.5) == "high confidence"


def test_pdf_text_normalization_repairs_pdf_hyphens_and_removes_controls():
    assert ocr_module._normalize_page_text("pre\x02screening\x00\x7f") == (
        "pre-screening"
    )


def test_ocr_pdf_wraps_page_failures_without_writing_output(tmp_path, monkeypatch):
    source = tmp_path / "broken.pdf"
    source.write_bytes(b"%PDF-fake")
    _install_fake_pdfium(monkeypatch, [""])
    monkeypatch.setattr(ocr_module, "_create_ocr_engine", object)

    def fail(*args, **kwargs):
        raise ValueError("backend details")

    monkeypatch.setattr(ocr_module, "_ocr_page", fail)

    with pytest.raises(PDFOCRError, match="page 1"):
        ocr_pdf(source)

    assert not source.with_suffix(".txt").exists()
