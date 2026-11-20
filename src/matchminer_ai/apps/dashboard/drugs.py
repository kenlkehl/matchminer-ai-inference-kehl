"""Each trial's drugs and their mechanisms, from the GoodOption catalog (the inference package's drug catalog).

The catalog (``matchminer_ai.trials.drug_catalog``) is built from trials' NCT IDs and drug names only, never from
patient data. Three of its tables are read here, in place and read-only:

- ``trial_drug_index.parquet``: each trial's drugs, with their role (investigational, uncertain, control, background,
  supportive) and the CT.gov arms that give them;
- ``drug_classes.parquet``: each drug's classes, its NCIt ancestors below Pharmacologic Substance plus a few an LLM
  added;
- ``drug_summaries.parquet``: each drug's cited facts, of which the first ``mechanism_and_targets`` text is shown.

``manifest.json`` gives the build's identity, counts and NCIt version. Roles and added classes are LLM-assisted, so the
dashboard says so and asks for the protocol to be checked.

A drug's classes mix levels (pembrolizumab: Anti-PD1 Monoclonal Antibody, PD1 Inhibitor, Monoclonal Antibody, Immune
Checkpoint Inhibitor, Immune Checkpoint Modulator). Its ``mechanism_types`` are the most specific: a class that is an
NCIt ancestor of another of the drug's classes is left out, using the inference package's bundled NCIt of the version
the catalog names. If that NCIt cannot be loaded, every class is kept in the catalog's order, and Data and Models says
so. Every class stays in ``all_classes`` for the mechanism filter, and ``class_ids`` maps each class name to its catalog
ID (the evidence dialogs, ``evidence.py``). An antibody-drug conjugate's payload is taken from its NCIt
"<payload>-containing Antibody-drug Conjugate" classes (``payload_classes`` keeps which), else from a ``payload`` its
facts name; there is no other source, so most drugs have none.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re

from .core.cache import file_stamp

TABLES = ("trial_drug_index.parquet", "drug_classes.parquet", "drug_summaries.parquet")
CATALOG_FILES = ("manifest.json", *TABLES)  # the store's signature has their stamps
ROLES = ("investigational", "uncertain", "control", "background")  # the order a trial's drugs are listed in
LEFT_OUT_ROLES = ("supportive",)  # e.g. antiemetics and growth factors given alongside
TYPE_ROLES = ("investigational", "uncertain")  # the Drug / Mechanism column shows their names and types
FILTER_ROLES = ("investigational", "uncertain", "control")  # the mechanism filter offers their classes
DEFAULT_NCIT = "Thesaurus_26.07d.FLAT.zip"
_PAYLOAD_CLASS = re.compile(r"^(?P<payload>.+?)-containing antibody-drug conjugate$", re.IGNORECASE)
_ADC_CLASS = re.compile(r"antibody-drug conjugate", re.IGNORECASE)


def catalog_stamps(directory: Path) -> dict[str, str | None]:
    """Size and mtime of the files read (never hashed)."""
    return {name: file_stamp(Path(directory) / name) for name in CATALOG_FILES}


def ncit_ancestors(resource: str, codes) -> dict[str, frozenset[str]]:
    """Each NCIt code's ancestor classes in the inference package's NCIt (``resource``: a bundled file name or a
    path). Loading takes a few seconds; the hierarchy is dropped afterwards (the builder keeps the results)."""
    from matchminer_ai.patients.ontology import NCIThesaurusDrugIndex

    hierarchy = NCIThesaurusDrugIndex.from_flat_zip(resource).classes
    if hierarchy is None:
        raise ValueError(f"{resource} has no drug-class hierarchy")
    return {code: frozenset(record.code for record in hierarchy.classes_above(code)) for code in codes}


def _text(value) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return ""
    return str(value).strip()


def _json_list(value) -> list:
    try:
        parsed = json.loads(value) if isinstance(value, str) and value else value
    except ValueError:
        return []
    return parsed if isinstance(parsed, list) else []


def _rows(path: Path, columns: list[str]) -> list[dict]:
    """The table's rows with these columns; a column the table lacks (an older catalog) is None."""
    import pyarrow.parquet as pq

    present = set(pq.read_schema(path).names)
    rows = pq.read_table(path, columns=[c for c in columns if c in present]).to_pylist()
    absent = [c for c in columns if c not in present]
    for row in rows if absent else ():
        row.update(dict.fromkeys(absent))
    return rows


def payload_of(classes: list[tuple[str, str]], specific: list[str], ancestors: dict,
               facts: list[dict]) -> tuple[str | None, list[dict]]:
    """An ADC's payload and the classes that name it: its most specific payload classes' names, each with the more
    generic payload classes above it ("Exatecan Derivative (Topoisomerase I Inhibitor)"), else the first ``payload``
    its facts name (no classes). ``[{"label", "class_name", "generic": [{"label", "class_name"}]}]``."""
    found = [(name, code, m["payload"].strip()) for name, code in classes if (m := _PAYLOAD_CLASS.match(name))]
    named, parts = [], []
    for name, code, payload in found:
        if name in specific:
            generic = [(other, p) for other, c, p in found if c and c in ancestors.get(code, ())]
            named.append(f"{payload} ({', '.join(p for _, p in generic)})" if generic else payload)
            parts.append({"label": payload, "class_name": name,
                          "generic": [{"label": p, "class_name": other} for other, p in generic]})
    if named:
        return ", ".join(named), parts
    if any(_ADC_CLASS.search(name) for name, _ in classes):
        for fact in facts:
            if isinstance(fact, dict) and _text(fact.get("payload")):
                return _text(fact["payload"]), []
    return None, []


def read_catalog(directory: Path | None, ancestors_of=None) -> tuple[dict[str, list[dict]], dict]:
    """(each trial's drugs, what was read). A missing or unreadable catalog gives no drugs, never an error.

    ``ancestors_of(resource, codes)`` returns each code's NCIt ancestors (default ``ncit_ancestors``)."""
    info = {"path": str(directory) if directory else None, "state": "missing", "message": None, "build_id": None,
            "created_at": None, "ncit": None, "counts": None, "most_specific": False, "hierarchy": None}
    if directory is None or not (Path(directory) / TABLES[0]).is_file():
        info["message"] = f"No GoodOption catalog at {directory}: the trials are shown without drugs."
        return {}, info
    directory = Path(directory)
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        manifest = {}
    info.update(build_id=manifest.get("build_id"), created_at=manifest.get("created_at_utc"),
                ncit=manifest.get("ncit"), counts=manifest.get("counts"))
    try:
        index = _rows(directory / TABLES[0], ["trial_id", "drug_id", "preferred_name", "ctgov_name", "role",
                                              "arm_labels_json"])
        missing = [name for name in TABLES[1:] if not (directory / name).is_file()]
        class_rows = [] if TABLES[1] in missing else _rows(directory / TABLES[1],
                                                           ["drug_id", "class_id", "class_name", "ncit_code"])
        summary_rows = [] if TABLES[2] in missing else _rows(directory / TABLES[2], ["drug_id", "structured_facts_json"])
    except Exception as exc:  # e.g. a table being replaced: tried again when its stamp changes
        info.update(state="error", message=f"Could not read the GoodOption catalog ({type(exc).__name__}: {exc}): "
                                           "the trials are shown without drugs.")
        return {}, info
    classes: dict[str, list[tuple[str, str]]] = {}
    class_ids: dict[str, str] = {}  # class name -> catalog class ID (one to one in the catalog)
    for row in class_rows:
        name, code = _text(row["class_name"]), _text(row["ncit_code"])
        found = classes.setdefault(_text(row["drug_id"]), [])
        if name and all(name != other for other, _ in found):
            found.append((name, code))
        if name and _text(row["class_id"]):
            class_ids.setdefault(name, _text(row["class_id"]))
    facts: dict[str, list[dict]] = {}
    for row in summary_rows:
        try:
            value = json.loads(row["structured_facts_json"] or "{}")
        except ValueError:
            value = {}
        mechanisms = value.get("mechanism_and_targets") if isinstance(value, dict) else None
        facts[_text(row["drug_id"])] = [f for f in mechanisms or [] if isinstance(f, dict)]

    resource = _text((manifest.get("ncit") or {}).get("resource")) or DEFAULT_NCIT
    codes = sorted({code for found in classes.values() for _, code in found if code})
    ancestors: dict[str, frozenset[str]] = {}
    try:
        ancestors = (ancestors_of or ncit_ancestors)(resource, codes)
        info.update(most_specific=True, hierarchy=f"NCIt {resource}, from the inference package")
    except Exception as exc:
        info["hierarchy"] = (f"NCIt {resource} could not be loaded ({type(exc).__name__}: {exc}): every class of a "
                             "drug is shown, in the catalog's order, not only its most specific")
    notes = [f"missing {', '.join(missing)}"] if missing else []

    def drug(drug_id: str) -> dict:
        found = classes.get(drug_id, [])
        above = set().union(*(ancestors.get(code, ()) for _, code in found if code))
        types = [name for name, code in found if not (code and code in above)]
        mechanism = next((_text(f.get("text")) for f in facts.get(drug_id, []) if _text(f.get("text"))), None)
        payload, payload_classes = payload_of(found, types, ancestors, facts.get(drug_id, []))
        return {"mechanism_types": types, "all_classes": [name for name, _ in found],
                "class_ids": {name: class_ids[name] for name, _ in found if name in class_ids},
                "payload": payload, "payload_classes": payload_classes, "mechanism": mechanism}

    rank = {role: i for i, role in enumerate(ROLES)}
    views: dict[str, dict] = {}
    listed: dict[str, list[tuple[tuple[int, int], dict]]] = {}
    for position, row in enumerate(index):
        role = _text(row["role"]).lower()
        trial_id, drug_id = _text(row["trial_id"]), _text(row["drug_id"])
        if role in LEFT_OUT_ROLES or not trial_id:
            continue
        if drug_id not in views:
            views[drug_id] = drug(drug_id)
        name, ctgov = _text(row["preferred_name"]), _text(row["ctgov_name"])
        entry = {"drug_id": drug_id, "name": name or ctgov or drug_id,
                 "listed_name": ctgov if ctgov and ctgov.casefold() != (name or ctgov).casefold() else None,
                 "role": role or "uncertain", "arms": [_text(a) for a in _json_list(row["arm_labels_json"]) if _text(a)],
                 **views[drug_id]}
        listed.setdefault(trial_id, []).append(((rank.get(entry["role"], len(ROLES)), position), entry))
    drugs: dict[str, list[dict]] = {}
    for trial_id, entries in listed.items():
        seen: set[str] = set()
        for _, entry in sorted(entries, key=lambda e: e[0]):
            if entry["drug_id"] not in seen:  # a drug listed twice keeps its first role in ROLES' order
                seen.add(entry["drug_id"])
                drugs.setdefault(trial_id, []).append(entry)
    info.update(state="ready", message="; ".join(notes) or None)
    return drugs, info


def shown_counts(trials: dict) -> dict:
    """What the store shows: trials with drugs, drug entries by role, distinct mechanism types and filter classes."""
    entries = [d for t in trials.values() for d in t.get("drugs") or []]
    return {"trials": len(trials), "trials_with_drugs": sum(1 for t in trials.values() if t.get("drugs")),
            "drugs": len(entries), "distinct_drugs": len({d["drug_id"] for d in entries}),
            "drugs_by_role": dict(Counter(d["role"] for d in entries)),
            "mechanism_types": len({x for d in entries if d["role"] in TYPE_ROLES for x in d["mechanism_types"]}),
            "filter_classes": len({x for d in entries if d["role"] in FILTER_ROLES for x in d["all_classes"]})}
