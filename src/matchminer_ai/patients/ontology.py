"""Local ontology access for structured patient summaries."""

from __future__ import annotations

import json
import re
import unicodedata
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from difflib import SequenceMatcher, get_close_matches
from functools import lru_cache
from importlib import resources
from io import TextIOWrapper
from pathlib import Path
from typing import Any, Iterator


_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")


def normalize_ontology_text(value: str) -> str:
    """Normalize a label for local ontology lookup."""
    ascii_text = unicodedata.normalize("NFKD", str(value)).encode(
        "ascii", "ignore"
    ).decode("ascii")
    return _NON_ALPHANUMERIC.sub(" ", ascii_text.casefold()).strip()


@contextmanager
def _resource_path(resource_name: str) -> Iterator[Path]:
    configured_path = Path(resource_name).expanduser()
    if configured_path.is_file():
        yield configured_path
        return

    resource = resources.files("matchminer_ai.data").joinpath(resource_name)
    if not resource.is_file():
        raise FileNotFoundError(
            f"Ontology resource {resource_name!r} was not found as a file or "
            "bundled matchminer_ai.data resource."
        )
    with resources.as_file(resource) as materialized:
        yield materialized


@dataclass(frozen=True)
class OncoTreeNode:
    """One OncoTree node and its immediate descendants."""

    code: str
    name: str
    main_type: str | None
    tissue: str | None
    children: tuple["OncoTreeNode", ...]

    def prompt_record(self, index: int) -> dict[str, Any]:
        """Return the bounded node representation exposed to the LLM."""
        return {
            "index": index,
            "code": self.code,
            "name": self.name,
            "main_type": self.main_type,
            "tissue": self.tissue,
            "has_children": bool(self.children),
        }


def _parse_oncotree_node(raw: dict[str, Any]) -> OncoTreeNode:
    children_raw = raw.get("children") or {}
    if not isinstance(children_raw, dict):
        raise ValueError("OncoTree node children must be a mapping.")
    children = tuple(
        sorted(
            (_parse_oncotree_node(child) for child in children_raw.values()),
            key=lambda node: (node.name.casefold(), node.code),
        )
    )
    return OncoTreeNode(
        code=str(raw.get("code") or "").strip(),
        name=str(raw.get("name") or "").strip(),
        main_type=(str(raw["mainType"]).strip() if raw.get("mainType") else None),
        tissue=(str(raw["tissue"]).strip() if raw.get("tissue") else None),
        children=children,
    )


@lru_cache(maxsize=4)
def load_oncotree(resource_name: str) -> OncoTreeNode:
    """Load and cache a bundled or explicitly configured OncoTree snapshot."""
    with _resource_path(resource_name) as path:
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    if not isinstance(raw, dict) or not raw:
        raise ValueError("OncoTree resource must contain a non-empty mapping.")
    root_raw = raw.get("TISSUE")
    if not isinstance(root_raw, dict):
        raise ValueError("OncoTree resource is missing its TISSUE root node.")
    return _parse_oncotree_node(root_raw)


@dataclass(frozen=True)
class NCItDrugRecord:
    """The NCIt fields needed for local drug normalization."""

    code: str
    preferred_name: str
    synonyms: tuple[str, ...]
    definition: str | None
    semantic_types: tuple[str, ...]

    def selection_record(self, index: int, query: str) -> dict[str, Any]:
        """Return a compact record without the definition for candidate choice."""
        normalized_query = normalize_ontology_text(query)
        ranked_synonyms = sorted(
            self.synonyms,
            key=lambda synonym: SequenceMatcher(
                None,
                normalized_query,
                normalize_ontology_text(synonym),
            ).ratio(),
            reverse=True,
        )
        return {
            "index": index,
            "ncit_code": self.code,
            "preferred_name": self.preferred_name,
            "synonyms": ranked_synonyms[:6],
        }

    def detail_record(self, index: int) -> dict[str, Any]:
        """Return the selected record fields used to verify a match."""
        return {
            "index": index,
            "ncit_code": self.code,
            "preferred_name": self.preferred_name,
            "synonyms": list(self.synonyms[:12]),
            "definition": self.definition,
            "semantic_types": list(self.semantic_types),
        }


class NCIThesaurusDrugIndex:
    """In-memory search index over NCIt pharmacologic-substance records."""

    def __init__(self, records: tuple[NCItDrugRecord, ...]) -> None:
        if not records:
            raise ValueError("NCIt drug index must contain at least one record.")
        self.records = records
        exact_aliases: dict[str, list[int]] = {}
        token_index: dict[str, set[int]] = {}
        preferred_lookup: dict[str, list[int]] = {}
        for record_index, record in enumerate(records):
            aliases = dict.fromkeys((record.preferred_name, *record.synonyms))
            for alias in aliases:
                normalized = normalize_ontology_text(alias)
                if not normalized:
                    continue
                exact_aliases.setdefault(normalized, []).append(record_index)
                for token in normalized.split():
                    if len(token) >= 3:
                        token_index.setdefault(token, set()).add(record_index)
            preferred = normalize_ontology_text(record.preferred_name)
            if preferred:
                preferred_lookup.setdefault(preferred, []).append(record_index)
        self._exact_aliases = exact_aliases
        self._token_index = token_index
        self._preferred_lookup = preferred_lookup
        self._preferred_names = tuple(preferred_lookup)

    @classmethod
    def from_flat_zip(cls, resource_name: str) -> "NCIThesaurusDrugIndex":
        """Load active pharmacologic substances from an NCIt flat archive."""
        records: list[NCItDrugRecord] = []
        with _resource_path(resource_name) as path:
            with zipfile.ZipFile(path) as archive:
                members = [name for name in archive.namelist() if name.endswith(".txt")]
                if len(members) != 1:
                    raise ValueError(
                        "NCIt flat archive must contain exactly one text file."
                    )
                with archive.open(members[0]) as raw_handle:
                    with TextIOWrapper(raw_handle, encoding="utf-8") as handle:
                        for line in handle:
                            fields = line.rstrip("\r\n").split("\t")
                            if len(fields) < 9:
                                fields.extend([""] * (9 - len(fields)))
                            semantic_types = tuple(
                                item.strip()
                                for item in fields[7].split("|")
                                if item.strip()
                            )
                            if "Pharmacologic Substance" not in semantic_types:
                                continue
                            concept_status = fields[6].strip().casefold()
                            if (
                                "retired" in concept_status
                                or "obsolete" in concept_status
                            ):
                                continue
                            synonyms = tuple(
                                dict.fromkeys(
                                    item.strip()
                                    for item in fields[3].split("|")
                                    if item.strip()
                                )
                            )
                            display_name = fields[5].strip()
                            preferred_name = display_name or (
                                synonyms[0] if synonyms else ""
                            )
                            if not preferred_name:
                                continue
                            records.append(
                                NCItDrugRecord(
                                    code=fields[0].strip(),
                                    preferred_name=preferred_name,
                                    synonyms=synonyms,
                                    definition=fields[4].strip() or None,
                                    semantic_types=semantic_types,
                                )
                            )
        return cls(tuple(records))

    def _record_score(self, record_index: int, query: str) -> float:
        normalized_query = normalize_ontology_text(query)
        record = self.records[record_index]
        best = 0.0
        for alias in (record.preferred_name, *record.synonyms):
            normalized_alias = normalize_ontology_text(alias)
            if not normalized_alias:
                continue
            if normalized_alias == normalized_query:
                return 1000.0
            if (
                normalized_query in normalized_alias
                or normalized_alias in normalized_query
            ):
                containment = min(len(normalized_query), len(normalized_alias)) / max(
                    len(normalized_query), len(normalized_alias)
                )
                best = max(best, 700.0 + containment)
            query_tokens = set(normalized_query.split())
            alias_tokens = set(normalized_alias.split())
            if query_tokens and alias_tokens:
                overlap = len(query_tokens & alias_tokens) / len(
                    query_tokens | alias_tokens
                )
                best = max(best, 400.0 * overlap)
            best = max(
                best,
                100.0 * SequenceMatcher(
                    None, normalized_query, normalized_alias
                ).ratio(),
            )
        return best

    def search(self, query: str, *, limit: int = 8) -> list[NCItDrugRecord]:
        """Return a bounded, locally ranked NCIt candidate page."""
        normalized_query = normalize_ontology_text(query)
        if not normalized_query or limit < 1:
            return []

        exact = self._exact_aliases.get(normalized_query, [])
        if exact:
            candidate_indices = set(exact)
        else:
            candidate_indices: set[int] = set()
            for token in normalized_query.split():
                candidate_indices.update(self._token_index.get(token, set()))
            if not candidate_indices:
                close_names = get_close_matches(
                    normalized_query,
                    self._preferred_names,
                    n=max(limit * 3, limit),
                    cutoff=0.45,
                )
                for name in close_names:
                    candidate_indices.update(self._preferred_lookup[name])

        ranked = sorted(
            candidate_indices,
            key=lambda index: (
                -self._record_score(index, query),
                self.records[index].preferred_name.casefold(),
                self.records[index].code,
            ),
        )
        return [self.records[index] for index in ranked[:limit]]


@lru_cache(maxsize=4)
def load_ncit_drug_index(resource_name: str) -> NCIThesaurusDrugIndex:
    """Load and cache the NCIt pharmacologic-substance search index."""
    return NCIThesaurusDrugIndex.from_flat_zip(resource_name)


__all__ = [
    "NCItDrugRecord",
    "NCIThesaurusDrugIndex",
    "OncoTreeNode",
    "load_ncit_drug_index",
    "load_oncotree",
    "normalize_ontology_text",
]
