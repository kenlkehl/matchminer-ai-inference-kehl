"""One drug's or one class's GoodOption catalog evidence, read on demand for the dashboard's evidence dialogs.

The catalog (``drugs.py``) holds, per drug and per class, an LLM-written synthesis of cited sources:
``drug_summaries.parquet`` / ``class_summaries.parquet`` give its structured facts by facet (each fact citing
evidence IDs in ``support_ids``) and the full summary text; ``drug_evidence/part-*.parquet`` /
``class_evidence/part-*.parquet`` hold the cited passages (PubMed, Europe PMC, CT.gov, CIViC, DailyMed, NCI, NCIt,
web), keyed by the drug or class and ``evidence_id``.

The evidence tables are about 190 MB, nearly all passages, each a single row group, so nothing is loaded at startup or
put into the store: each request reads only the drug's or class's rows with DuckDB (the filter is applied in the scan
and only the needed columns are decoded; about 0.1 s once the files are in the page cache, under 1 s cold). The answers
for the last ``cache_size`` drugs and classes are kept, keyed by the catalog files' stamps (size and mtime, never
hashed), so a rebuilt catalog is read afresh. A cited ID with no passage is listed in ``missing_citations``; the
dialog shows it as "source not in catalog".
"""
from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path
import re
import threading

from .core.cache import file_stamp
from .drugs import CATALOG_FILES, catalog_stamps

LABEL = "GoodOption catalog (LLM-assisted synthesis of the cited sources; verify before use)"
FACETS = {"mechanism_and_targets": "Mechanism and targets", "efficacy_by_tumor": "Efficacy by tumor",
          "biomarker_directed_efficacy": "Biomarker-directed efficacy", "biomarker_prevalence": "Biomarker prevalence",
          "safety": "Safety", "limitations": "Limitations"}  # in this order; any other facet follows
# A fact's qualifiers in reading order; any other key follows, alphabetically.
QUALIFIERS = ("scope", "evidence_level", "tumor_type", "histology", "biomarker", "regimen", "comparator", "population",
              "prevalence", "denominator", "target", "targets", "target_expression", "mechanism", "payload", "pathway",
              "study_design", "source_kind", "source_level", "evidence_gap")
EVIDENCE_COLUMNS = ("evidence_id", "source", "source_type", "title", "passage", "url", "source_locator",
                    "published_at", "license")
KINDS = {"drug": ("drug_id", "drug_summaries.parquet", "drug_evidence"),
         "class": ("class_id", "class_summaries.parquet", "class_evidence")}
_WEB_URL = re.compile(r"^https?://", re.IGNORECASE)


class CatalogUnavailable(Exception):
    """No GoodOption catalog (or not one with this table) at the configured path."""


class UnknownEntry(KeyError):
    pass


def _text(value) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return ""
    return str(value).strip()


def _plain(value) -> str:
    """A qualifier's value as text: lists joined, objects as JSON."""
    if isinstance(value, (list, tuple)):
        return "; ".join(t for t in (_plain(v) for v in value) if t)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return _text(value)


def facets_of(facts_json) -> list[dict]:
    """``structured_facts_json`` as facets in reading order, each fact with its text, qualifiers and citations."""
    try:
        value = json.loads(facts_json) if isinstance(facts_json, str) and facts_json else (facts_json or {})
    except ValueError:
        value = {}
    if not isinstance(value, dict):
        return []
    out = []
    for facet in [*FACETS, *sorted(k for k in value if k not in FACETS)]:
        facts = []
        for item in value.get(facet) or []:
            if not isinstance(item, dict) or not _text(item.get("text")):
                continue
            keys = [k for k in QUALIFIERS if k in item] + sorted(k for k in item if k not in QUALIFIERS
                                                                  and k not in ("text", "support_ids"))
            facts.append({"text": _text(item["text"]),
                          "qualifiers": [[k, _plain(item[k])] for k in keys if _plain(item[k])],
                          "support_ids": list(dict.fromkeys(_text(s) for s in item.get("support_ids") or [] if _text(s)))})
        if facts:
            out.append({"facet": facet, "label": FACETS.get(facet, facet.replace("_", " ").capitalize()), "facts": facts})
    return out


def _sql_files(paths) -> str:
    return "[" + ", ".join("'" + str(p).replace("'", "''") + "'" for p in paths) + "]"


class CatalogEvidence:
    """The evidence dialogs' data, read lazily from the catalog directory (see the module docstring)."""

    def __init__(self, directory: Path | None, cache_size: int = 32):
        self.directory = Path(directory) if directory is not None else None
        self.cache_size = cache_size
        self._cache: OrderedDict[tuple[str, str], tuple[dict, dict]] = OrderedDict()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ reading

    def _evidence_files(self, kind: str) -> list[Path]:
        folder = self.directory / KINDS[kind][2]
        return sorted(folder.glob("part-*.parquet")) if folder.is_dir() else []

    def _stamps(self, kind: str) -> dict:
        stamps = catalog_stamps(self.directory)
        stamps[KINDS[kind][1]] = file_stamp(self.directory / KINDS[kind][1])
        for path in self._evidence_files(kind):
            stamps[f"{KINDS[kind][2]}/{path.name}"] = file_stamp(path)
        return stamps

    def _require(self, kind: str) -> None:
        if self.directory is None or not (self.directory / KINDS[kind][1]).is_file():
            raise CatalogUnavailable(f"No GoodOption catalog with {KINDS[kind][1]} at {self.directory}: the catalog's "
                                     "evidence cannot be shown.")

    @staticmethod
    def _query(sql: str, params: list) -> list[dict]:
        import duckdb

        with duckdb.connect() as con:  # one connection per request: thread-safe under the server's worker threads
            cursor = con.execute(sql, params)
            names = [d[0] for d in cursor.description]
            return [dict(zip(names, row)) for row in cursor.fetchall()]

    def _columns(self, path: Path) -> set[str]:
        import pyarrow.parquet as pq

        return set(pq.read_schema(path).names)

    def _select(self, path: Path, wanted: tuple[str, ...], where: str, params: list) -> list[dict]:
        present = self._columns(path)
        columns = ", ".join(f'"{c}"' if c in present else f'NULL AS "{c}"' for c in wanted)
        return self._query(f"SELECT {columns} FROM read_parquet({_sql_files([path])}) WHERE {where}", params)

    def _evidence(self, kind: str, owner: str, cited: list[str]) -> dict[str, dict]:
        files = self._evidence_files(kind)
        if not cited or not files:
            return {}
        key = KINDS[kind][0]
        present = set.intersection(*(self._columns(p) for p in files))
        columns = ", ".join(f'"{c}"' if c in present else f'NULL AS "{c}"' for c in EVIDENCE_COLUMNS)
        rows = self._query(f"SELECT {columns} FROM read_parquet({_sql_files(files)}, union_by_name = true) "
                           f'WHERE "{key}" = ? AND evidence_id IN ({", ".join("?" * len(cited))})', [owner, *cited])
        found = {}
        for row in rows:
            url = _text(row["url"])
            found.setdefault(_text(row["evidence_id"]), {
                "evidence_id": _text(row["evidence_id"]), "source": _text(row["source"]),
                "source_type": _text(row["source_type"]), "title": _text(row["title"]),
                "passage": _text(row["passage"]), "url": url if _WEB_URL.match(url) else "",
                "locator": _text(row["source_locator"]), "published_at": _text(row["published_at"]),
                "license": _text(row["license"])})
        return found

    def _manifest(self) -> dict:
        try:
            manifest = json.loads((self.directory / CATALOG_FILES[0]).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            manifest = {}
        return {"path": str(self.directory), "build_id": manifest.get("build_id"),
                "created_at": manifest.get("created_at_utc")}

    def _facts_and_evidence(self, kind: str, owner: str, facts_json) -> dict:
        facets = facets_of(facts_json)
        cited = list(dict.fromkeys(s for f in facets for fact in f["facts"] for s in fact["support_ids"]))
        evidence = self._evidence(kind, owner, cited)
        return {"facets": facets, "evidence": evidence, "missing_citations": [s for s in cited if s not in evidence],
                "counts": {"facts": sum(len(f["facts"]) for f in facets),
                           "citations": sum(len(fact["support_ids"]) for f in facets for fact in f["facts"]),
                           "cited_sources": len(cited), "resolved_sources": len(evidence)}}

    def _read_drug(self, drug_id: str) -> dict:
        directory = self.directory
        rows = self._select(directory / "drug_summaries.parquet",
                            ("drug_id", "preferred_name", "ncit_code", "research_status", "synthesis_status",
                             "good_option_summary", "evidence_count", "structured_facts_json"), "drug_id = ?", [drug_id])
        if not rows:
            raise UnknownEntry(drug_id)
        row = rows[0]
        classes = []
        if (directory / "drug_classes.parquet").is_file():
            classes = [{"id": _text(c["class_id"]), "name": _text(c["class_name"]), "ncit_code": _text(c["ncit_code"]),
                        "assigned_by": _text(c["assigned_by"])}
                       for c in self._select(directory / "drug_classes.parquet",
                                             ("class_id", "class_name", "ncit_code", "assigned_by"), "drug_id = ?",
                                             [drug_id]) if _text(c["class_name"])]
        return {"kind": "drug", "id": drug_id, "name": _text(row["preferred_name"]) or drug_id,
                "ncit_code": _text(row["ncit_code"]) or None, "label": LABEL,
                "summary": _text(row["good_option_summary"]) or None, "evidence_count": row["evidence_count"],
                "status": {"research": _text(row["research_status"]), "synthesis": _text(row["synthesis_status"])},
                "classes": classes, **self._facts_and_evidence("drug", drug_id, row["structured_facts_json"]),
                "catalog": self._manifest()}

    def _read_class(self, class_id: str) -> dict:
        directory = self.directory
        rows = self._select(directory / "class_summaries.parquet",
                            ("class_id", "class_name", "synthesis_status", "class_option_summary", "evidence_count",
                             "structured_facts_json"), "class_id = ?", [class_id])
        if not rows:
            raise UnknownEntry(class_id)
        row = rows[0]
        drugs = []
        if (directory / "drug_classes.parquet").is_file():
            ids = [_text(r["drug_id"]) for r in self._select(directory / "drug_classes.parquet", ("drug_id",),
                                                             "class_id = ?", [class_id])]
            names = {}
            if ids:
                names = {_text(r["drug_id"]): _text(r["preferred_name"]) for r in self._select(
                    directory / "drug_summaries.parquet", ("drug_id", "preferred_name"),
                    f"drug_id IN ({', '.join('?' * len(ids))})", ids)}
            drugs = sorted(({"id": d, "name": names.get(d) or d} for d in dict.fromkeys(ids)),
                           key=lambda d: d["name"].casefold())
        ncit = class_id.split(":", 1)[1] if class_id.startswith("NCIT:") else None
        return {"kind": "class", "id": class_id, "name": _text(row["class_name"]) or class_id, "ncit_code": ncit,
                "label": LABEL, "summary": _text(row["class_option_summary"]) or None,
                "evidence_count": row["evidence_count"], "status": {"synthesis": _text(row["synthesis_status"])},
                "drugs": drugs, **self._facts_and_evidence("class", class_id, row["structured_facts_json"]),
                "catalog": self._manifest()}

    def _cached(self, kind: str, entry_id: str) -> dict:
        self._require(kind)
        stamps = self._stamps(kind)
        with self._lock:
            hit = self._cache.get((kind, entry_id))
            if hit and hit[0] == stamps:
                self._cache.move_to_end((kind, entry_id))
                return hit[1]
        value = self._read_drug(entry_id) if kind == "drug" else self._read_class(entry_id)
        with self._lock:
            self._cache[(kind, entry_id)] = (stamps, value)
            self._cache.move_to_end((kind, entry_id))
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return value

    # ------------------------------------------------------------------ the dialogs

    def drug(self, drug_id: str, store: dict | None = None) -> dict:
        """A drug's facts, cited passages, classes (``specific``: among its mechanism types in this store, or None
        when no trial in the store lists it) and full summary."""
        value = self._cached("drug", drug_id)
        entries = [d for t in (store or {}).get("trials", {}).values() for d in t.get("drugs") or []
                   if d.get("drug_id") == drug_id]
        specific = {name for d in entries for name in d.get("mechanism_types") or []}
        classes = [{**c, "specific": (c["name"] in specific) if entries else None} for c in value["classes"]]
        return {**value, "classes": classes}

    def klass(self, class_id: str, store: dict | None = None) -> dict:
        """A class's facts, cited passages, catalog drugs and full summary, with the store's trials whose drugs are
        in the class."""
        value = self._cached("class", class_id)
        return {**value, "trials": trials_in_class(store or {}, class_id, value["name"])}


def trials_in_class(store: dict, class_id: str, class_name: str) -> list[dict]:
    """The store's trials with a drug in this class (any role shown), with those drugs."""
    out = []
    for trial in store.get("trials", {}).values():
        drugs = [d for d in trial.get("drugs") or []
                 if (class_id in (d.get("class_ids") or {}).values()
                     or (not d.get("class_ids") and class_name in (d.get("all_classes") or [])))]
        if drugs:
            out.append({"id": trial["id"], "protocol_no": trial.get("protocol_no") or "",
                        "short_title": trial.get("short_title") or "", "center": trial.get("center") or "",
                        "state": trial.get("state"), "drugs": [{"name": d["name"], "role": d["role"]} for d in drugs]})
    return sorted(out, key=lambda t: (t["protocol_no"] or t["id"]))
