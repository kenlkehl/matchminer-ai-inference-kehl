"""Disease-specific extensions of the extraction data model.

An extension adds item types and prompt guidance without changing the core item
types, vocabularies or record layout: its items are further lists in a record's
``items``, validated, grounded and flattened like the core ones. Select
extensions by name in ``document_extraction.extensions`` or
``patient_extraction.extensions``; register new ones with
:func:`register_extension`.

The packaged ``thoracic`` extension carries the parts of the lung-cancer
abstraction this model generalizes that are specific to thoracic oncology:
smoking history and thoracic guidance for the core fields.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .schema import CORE_ITEM_TYPES, Field, ItemType
from .vocabulary import Term, Vocabulary


@dataclass(frozen=True)
class Extension:
    """Item types and prompt guidance for one disease area."""

    name: str
    description: str
    item_types: tuple[ItemType, ...] = ()
    guidance: str = ""


SMOKING_STATUSES = Vocabulary(
    name="smoking_status",
    description="Tobacco smoking status.",
    terms=(
        Term("never", "Never smoker (fewer than 100 cigarettes in a lifetime)."),
        Term("former", "Former smoker."),
        Term("current", "Current smoker."),
        Term("not_stated", "Not stated."),
    ),
    other=None,
)

SMOKING_HISTORY = ItemType(
    name="smoking_history",
    key="smoking_history",
    id_prefix="SM",
    description="Tobacco smoking history: at most one item.",
    levels=frozenset({"document", "patient"}),
    fields=(
        Field("smoking_status", "choice", "Smoking status.", SMOKING_STATUSES),
        Field("pack_years", "text", "Pack-years as written; '' when not stated."),
        Field("quit_date", "date", "When a former smoker quit."),
    ),
    max_items=1,
)

THORACIC = Extension(
    name="thoracic",
    description="Thoracic oncology: lung cancer, mesothelioma, thymic tumors.",
    item_types=(SMOKING_HISTORY,),
    guidance=(
        "THORACIC ONCOLOGY\n"
        "- Histology examples: adenocarcinoma, squamous cell carcinoma, "
        "non-small cell carcinoma NOS, small cell carcinoma, large cell "
        "neuroendocrine carcinoma, typical or atypical carcinoid, adenosquamous, "
        "NUT carcinoma, epithelioid or sarcomatoid mesothelioma, thymoma, thymic "
        "carcinoma.\n"
        "- Sites as written, using lobe abbreviations where the text does (RUL, RML, "
        "RLL, LUL, LLL, lingula) and nodal stations (4R, 7, 10L).\n"
        "- Biomarkers commonly tested: EGFR, ALK, ROS1, KRAS, BRAF, MET (exon 14 "
        "skipping or amplification), RET, NTRK, HER2 (ERBB2), PD-L1 (record the tumor "
        "proportion score as the value), TTF-1 and p40.\n"
        "- Thoracic surgery is a treatment with its operation as the regimen "
        "(lobectomy, segmentectomy, wedge resection, pneumonectomy, thymectomy, "
        "pleurectomy/decortication, extrapleural pneumonectomy); its specimen is a "
        "surgical_resection procedure. EBUS-TBNA, navigational bronchoscopy, "
        "mediastinoscopy and thoracentesis cytology have their own procedure types.\n"
        "- Radiation as written: SBRT, SRS, conventionally fractionated (long-course) "
        "radiation, with its dose and site.\n"
        "- smoking_history: status and pack-years as documented."
    ),
)

_REGISTRY: dict[str, Extension] = {THORACIC.name: THORACIC}


def register_extension(extension: Extension, *, replace: bool = False) -> None:
    """Make ``extension`` selectable by name. Its item types must not reuse a core
    item type's name, list key or id prefix."""
    if extension.name in _REGISTRY and not replace:
        raise ValueError(f"Extension {extension.name!r} is already registered")
    core = {(t.name, t.key, t.id_prefix) for t in CORE_ITEM_TYPES}
    taken = {value for triple in core for value in triple}
    for item_type in extension.item_types:
        if {item_type.name, item_type.key, item_type.id_prefix} & taken:
            raise ValueError(f"{item_type.name} reuses a core item type's name, key or prefix")
    _REGISTRY[extension.name] = extension


def registered_extensions() -> dict[str, Extension]:
    return dict(_REGISTRY)


def resolve_extensions(names: Iterable[str] | None) -> tuple[Extension, ...]:
    """The registered extensions named, in order, each once."""
    if names is None:
        return ()
    if isinstance(names, str):
        raise TypeError("extensions must be a list of extension names")
    resolved = []
    for name in dict.fromkeys(names):
        if name not in _REGISTRY:
            raise ValueError(f"Unknown extraction extension {name!r}; known: {sorted(_REGISTRY)}")
        resolved.append(_REGISTRY[name])
    return tuple(resolved)


def extension_item_types(extensions: Iterable[Extension]) -> tuple[ItemType, ...]:
    return tuple(t for extension in extensions for t in extension.item_types)


def extension_guidance(extensions: Iterable[Extension]) -> str:
    return "\n\n".join(e.guidance for e in extensions if e.guidance)


__all__ = [
    "Extension",
    "SMOKING_HISTORY",
    "THORACIC",
    "extension_guidance",
    "extension_item_types",
    "register_extension",
    "registered_extensions",
    "resolve_extensions",
]
