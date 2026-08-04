"""Local document text-extraction APIs."""

from .ocr import (
    PDFOCRError,
    PDFOCRMethod,
    PDFOCRNoTextError,
    PDFOCRProgress,
    ocr_pdf,
)

__all__ = [
    "PDFOCRError",
    "PDFOCRMethod",
    "PDFOCRNoTextError",
    "PDFOCRProgress",
    "ocr_pdf",
]
