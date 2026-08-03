"""Local, page-aware text extraction and OCR for PDF documents."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import numpy as np

PDFOCRMethod = Literal["embedded", "ocr"]
PDFOCRProgress = Callable[[int, int, PDFOCRMethod], None]

_PAGE_SEPARATOR = "\n\n\f\n\n"
_PDF_CONTROL_CHARACTER_TRANSLATION: dict[int, str | None] = {
    codepoint: None for codepoint in range(32) if codepoint not in (9, 10)
}
# Some PDF font encodings expose a displayed hyphen as U+0002.
_PDF_CONTROL_CHARACTER_TRANSLATION[2] = "-"
_PDF_CONTROL_CHARACTER_TRANSLATION[127] = None
_PDF_CONTROL_CHARACTER_TRANSLATION[0xAD] = "-"


class PDFOCRError(RuntimeError):
    """Raised when a PDF cannot be processed safely."""


class PDFOCRNoTextError(PDFOCRError):
    """Raised when neither embedded extraction nor OCR finds document text."""


def _load_pdfium() -> Any:
    try:
        import pypdfium2
    except ImportError as exc:  # pragma: no cover - packaging guards this path
        raise ImportError(
            "ocr_pdf requires pypdfium2; reinstall matchminer-ai with its "
            "declared dependencies."
        ) from exc
    return pypdfium2


def _create_ocr_engine() -> Any:
    try:
        from rapidocr import RapidOCR
    except ImportError as exc:  # pragma: no cover - packaging guards this path
        raise ImportError(
            "Image-only PDF pages require RapidOCR; reinstall matchminer-ai "
            "with its declared dependencies."
        ) from exc
    return RapidOCR()


def _normalize_page_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.translate(_PDF_CONTROL_CHARACTER_TRANSLATION)
    normalized = normalized.replace("\N{NO-BREAK SPACE}", " ")
    lines = [line.rstrip() for line in normalized.split("\n")]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def _meaningful_character_count(text: str) -> int:
    return sum(character.isalnum() for character in text)


def _extract_embedded_text(page: Any) -> str:
    text_page = page.get_textpage()
    try:
        return str(text_page.get_text_bounded())
    finally:
        text_page.close()


def _text_from_rapidocr_output(result: Any, min_confidence: float) -> str:
    if result is None:
        return ""

    texts = getattr(result, "txts", None)
    scores = getattr(result, "scores", None)
    if texts is None:
        return ""

    retained: list[str] = []
    for index, text in enumerate(texts):
        normalized = str(text).strip()
        if not normalized:
            continue
        if (
            scores is not None
            and index < len(scores)
            and float(scores[index]) < min_confidence
        ):
            continue
        retained.append(normalized)
    return "\n".join(retained)


def _ocr_page(
    page: Any,
    engine: Any,
    *,
    dpi: int,
    min_confidence: float,
) -> str:
    bitmap = page.render(
        scale=dpi / 72,
        draw_annots=True,
        rev_byteorder=True,
    )
    image = None
    rgb_image = None
    try:
        image = bitmap.to_pil()
        rgb_image = image.convert("RGB")
        image_array = np.asarray(rgb_image).copy()
        result = engine(image_array)
        return _text_from_rapidocr_output(result, min_confidence)
    finally:
        if rgb_image is not None:
            rgb_image.close()
        if image is not None:
            image.close()
        bitmap.close()


def _validate_paths(
    pdf_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str] | None,
    *,
    overwrite: bool,
) -> tuple[Path, Path]:
    source = Path(pdf_path)
    if source.suffix.lower() != ".pdf":
        raise ValueError("pdf_path must point to a file with a .pdf extension.")
    if not source.is_file():
        raise FileNotFoundError(f"PDF file does not exist: {source}")

    destination = (
        Path(output_path) if output_path is not None else source.with_suffix(".txt")
    )
    if destination.suffix.lower() != ".txt":
        raise ValueError("output_path must use a .txt extension.")
    if not destination.parent.is_dir():
        raise FileNotFoundError(
            f"Output directory does not exist: {destination.parent}"
        )
    if destination.exists():
        if destination.is_dir():
            raise IsADirectoryError(f"Output path is a directory: {destination}")
        if not overwrite:
            raise FileExistsError(
                f"Output file already exists: {destination}. "
                "Pass overwrite=True to replace it."
            )
    return source, destination


def _validate_options(
    *,
    dpi: int,
    min_embedded_text_chars: int,
    min_ocr_confidence: float,
) -> None:
    if isinstance(dpi, bool) or not isinstance(dpi, int):
        raise TypeError("dpi must be an integer.")
    if dpi < 72:
        raise ValueError("dpi must be an integer of at least 72.")
    if isinstance(min_embedded_text_chars, bool) or not isinstance(
        min_embedded_text_chars, int
    ):
        raise TypeError("min_embedded_text_chars must be an integer.")
    if min_embedded_text_chars < 0:
        raise ValueError("min_embedded_text_chars must be a non-negative integer.")
    if isinstance(min_ocr_confidence, bool) or not isinstance(
        min_ocr_confidence, (int, float)
    ):
        raise TypeError("min_ocr_confidence must be a number.")
    if not 0 <= float(min_ocr_confidence) <= 1:
        raise ValueError("min_ocr_confidence must be between 0 and 1.")


def ocr_pdf(
    pdf_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str] | None = None,
    *,
    force_ocr: bool = False,
    dpi: int = 200,
    min_embedded_text_chars: int = 50,
    min_ocr_confidence: float = 0.5,
    overwrite: bool = False,
    password: str | None = None,
    progress_callback: PDFOCRProgress | None = None,
) -> Path:
    """Extract text from a PDF and write one UTF-8 text file.

    Pages with a usable embedded text layer retain that exact text. Image-only
    or nearly empty pages are rendered and processed locally with RapidOCR.
    Page order is preserved and form-feed characters delimit pages. No PDF
    content is sent to an LLM, web search, or other network service.

    Parameters
    ----------
    pdf_path : str or os.PathLike[str]
        Local PDF file to process.
    output_path : str or os.PathLike[str] or None, optional
        Destination text file. By default, use the PDF path with a ``.txt``
        suffix.
    force_ocr : bool, optional
        Render and OCR every page, ignoring embedded PDF text. This can help
        with damaged text layers or displayed form values, but embedded text
        is normally more accurate.
    dpi : int, optional
        Resolution used to render OCR pages. Must be at least 72.
    min_embedded_text_chars : int, optional
        Minimum number of alphanumeric characters required to trust a page's
        embedded text in automatic mode.
    min_ocr_confidence : float, optional
        Discard OCR text lines below this threshold in the inclusive range from
        0 to 1.
    overwrite : bool, optional
        Replace an existing destination only when True.
    password : str or None, optional
        Password for an encrypted PDF. The password is used locally and is not
        written to the output.
    progress_callback : callable or None, optional
        Called after each page as ``callback(page_number, page_count, method)``,
        where ``method`` is ``"embedded"`` or ``"ocr"``.

    Returns
    -------
    pathlib.Path
        Path to the completed UTF-8 text file.

    Raises
    ------
    PDFOCRNoTextError
        If the document contains pages but no text can be extracted.
    PDFOCRError
        If the PDF cannot be opened or a page cannot be processed.
    FileExistsError
        If the destination exists and ``overwrite`` is False.
    """
    _validate_options(
        dpi=dpi,
        min_embedded_text_chars=min_embedded_text_chars,
        min_ocr_confidence=min_ocr_confidence,
    )
    source, destination = _validate_paths(
        pdf_path,
        output_path,
        overwrite=overwrite,
    )

    pdfium = _load_pdfium()
    try:
        document = pdfium.PdfDocument(source, password=password)
    except Exception as exc:
        raise PDFOCRError("Could not open the PDF document.") from exc

    pages: list[str] = []
    ocr_engine: Any | None = None
    try:
        page_count = len(document)
        if page_count == 0:
            raise PDFOCRError("The PDF document contains no pages.")

        for page_index in range(page_count):
            page = document[page_index]
            try:
                embedded_text = ""
                if not force_ocr:
                    try:
                        embedded_text = _normalize_page_text(
                            _extract_embedded_text(page)
                        )
                    except pdfium.PdfiumError:
                        # A renderable page can still have a damaged text layer.
                        embedded_text = ""
                if (
                    not force_ocr
                    and _meaningful_character_count(embedded_text)
                    >= min_embedded_text_chars
                ):
                    page_text = embedded_text
                    method: PDFOCRMethod = "embedded"
                else:
                    if ocr_engine is None:
                        ocr_engine = _create_ocr_engine()
                    page_text = _normalize_page_text(
                        _ocr_page(
                            page,
                            ocr_engine,
                            dpi=dpi,
                            min_confidence=float(min_ocr_confidence),
                        )
                    )
                    method = "ocr"
                pages.append(page_text)
                if progress_callback is not None:
                    progress_callback(page_index + 1, page_count, method)
            except Exception as exc:
                if isinstance(exc, (ImportError, PDFOCRError)):
                    raise
                raise PDFOCRError(
                    f"Failed to process PDF page {page_index + 1}."
                ) from exc
            finally:
                page.close()
    finally:
        document.close()

    if not any(_meaningful_character_count(page) for page in pages):
        raise PDFOCRNoTextError("No text was found in the PDF document.")

    document_text = _PAGE_SEPARATOR.join(pages) + "\n"
    mode = "w" if overwrite else "x"
    with destination.open(mode, encoding="utf-8", newline="\n") as handle:
        handle.write(document_text)
    return destination
