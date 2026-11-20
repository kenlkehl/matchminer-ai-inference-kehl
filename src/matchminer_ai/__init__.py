"""Public package exports for MatchMiner-AI."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from ._metadata import package_version

if TYPE_CHECKING:
    from .config import load_config, load_default_preset, load_preset
    from .documents import PDFOCRError, PDFOCRNoTextError, ocr_pdf
    from .pipeline import MMAIPipeline

# Imported on first use, so `import matchminer_ai` (and the standard-library-only tools under
# matchminer_ai.evaluation.dual_judge) needs neither PyYAML nor NumPy until a caller asks for them.
_LAZY_ATTRIBUTES = {
    "load_config": ".config",
    "load_default_preset": ".config",
    "load_preset": ".config",
    "PDFOCRError": ".documents",
    "PDFOCRNoTextError": ".documents",
    "ocr_pdf": ".documents",
    "MMAIPipeline": ".pipeline",
}

_LAZY_SUBMODULES = {
    "embedding",
    "good_options",
    "llm",
    "matching",
    "patients",
    "trials",
}

__version__ = package_version()

__all__ = [
    "MMAIPipeline",
    "PDFOCRError",
    "PDFOCRNoTextError",
    "__version__",
    "load_config",
    "load_default_preset",
    "load_preset",
    "ocr_pdf",
]


def __getattr__(name: str) -> Any:
    """Lazily expose the public functions and subpackages."""
    if name in _LAZY_ATTRIBUTES:
        value = getattr(import_module(_LAZY_ATTRIBUTES[name], __name__), name)
        globals()[name] = value
        return value
    if name in _LAZY_SUBMODULES:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
