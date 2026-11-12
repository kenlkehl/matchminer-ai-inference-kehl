"""Local NCIt drug index and drug-class hierarchy for the GoodOption catalog."""

from __future__ import annotations

import math
import re
import unicodedata
import zipfile
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from difflib import SequenceMatcher, get_close_matches
from functools import lru_cache
from importlib import resources
from io import TextIOWrapper
from pathlib import Path
from types import MappingProxyType
from typing import Iterator


_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")
_LETTER_DIGIT_BOUNDARY = re.compile(r"(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])")
_PHARMACOLOGIC_SUBSTANCE = "Pharmacologic Substance"
#: Words a class name can carry without the text having to repeat them.
_CLASS_FILLER_WORDS = frozenset(
    {
        "a", "an", "and", "anti", "agent", "agents", "containing",
        "for", "of", "or", "preparation", "recombinant", "targeted", "targeting",
        "the", "therapeutic", "with",
    }
)

#: NCIt concepts above the drug classes, too broad to be classes themselves.
#: Fixed by decision (2026-11-11), not derived from data; every concept below
#: them is kept, e.g. Protein Kinase Inhibitor, Tyrosine Kinase Inhibitor,
#: Monoclonal Antibody, Cytotoxic Chemotherapeutic Agent and Immune Checkpoint
#: Inhibitor.
NCIT_UMBRELLA_CLASSES: Mapping[str, str] = MappingProxyType(
    {
        "C1908": "Drug, Food, Chemical or Biomedical Material",
        "C1909": "Pharmacologic Substance",
        "C274": "Antineoplastic Agent",
        "C308": "Immunotherapeutic Agent",
        "C307": "Biological Agent",
        "C471": "Enzyme Inhibitor",
        "C2189": "Signal Transduction Inhibitor",
        "C129824": "Antineoplastic Protein Inhibitor",
        "C129821": "Antineoplastic Biological Agent",
        "C163758": "Targeted Therapy Agent",
        "C129825": "Antineoplastic Enzyme Inhibitor",
        "C129822": "Antineoplastic Antibody",
        "C129820": "Antineoplastic Immunomodulating Agent",
    }
)
#: Classes come only from below Pharmacologic Substance. Drug concepts also sit
#: under C1913 Drug or Chemical by Structure (Organic Chemical, Nitrogen
#: Compound, Lipid, ...), whose chemistry groups agents that do not act alike.
NCIT_PHARMACOLOGIC_ROOT = "C1909"


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
class NCItDrugRecord:
    """The NCIt fields needed for local drug normalization."""

    code: str
    preferred_name: str
    synonyms: tuple[str, ...]
    definition: str | None
    semantic_types: tuple[str, ...]


@dataclass(frozen=True)
class NCItClassRecord:
    """One NCIt concept usable as a drug class."""

    code: str
    preferred_name: str
    synonyms: tuple[str, ...]
    definition: str | None


def _search_tokens(normalized: str) -> Counter[str]:
    """Words of normalized text, plus the joins and splits that spell them alike.

    "PD-1" normalizes to "pd 1" and "Bcr-Abl1" to "bcr abl1", while NCIt names
    the classes "PD1 Inhibitor" and "BCR-ABL Inhibitor", so adjacent words are
    also counted joined and letter-digit words split.
    """

    words = normalized.split()
    tokens = Counter(words)
    tokens.update(left + right for left, right in zip(words, words[1:]))
    for word in words:
        parts = [part for part in _LETTER_DIGIT_BOUNDARY.split(word) if part]
        if len(parts) > 1:
            tokens.update(parts)
    return tokens


class NCItClassHierarchy:
    """The NCIt drug classes and the parent links that lead to them.

    A class is an active NCIt concept below Pharmacologic Substance
    (``NCIT_PHARMACOLOGIC_ROOT``) that is a transitive ancestor of an active
    pharmacologic-substance concept, is not one itself, and is not one of the
    ``NCIT_UMBRELLA_CLASSES``.
    """

    def __init__(
        self,
        classes: Sequence[NCItClassRecord],
        parents: Mapping[str, Sequence[str]],
    ) -> None:
        self.classes = {record.code: record for record in classes}
        self._parents = {code: tuple(values) for code, values in parents.items()}
        aliases: dict[str, tuple[tuple[str, ...], ...]] = {}
        document_frequency: Counter[str] = Counter()
        for record in self.classes.values():
            names = dict.fromkeys(
                normalized
                for value in (record.preferred_name, *record.synonyms)
                if (normalized := normalize_ontology_text(value))
            )
            aliases[record.code] = tuple(tuple(name.split()) for name in names)
            document_frequency.update(
                {token for name in aliases[record.code] for token in name}
            )
        total = len(self.classes)
        self._aliases = aliases
        self._idf = {
            token: math.log((1 + total) / count)
            for token, count in document_frequency.items()
        }

    def get(self, code: str) -> NCItClassRecord | None:
        return self.classes.get(str(code or "").strip())

    def classes_above(self, code: str) -> tuple[NCItClassRecord, ...]:
        """Every class above one concept, nearest first, umbrellas skipped."""

        start = str(code or "").strip()
        found: list[NCItClassRecord] = []
        seen = {start}
        level = list(self._parents.get(start, ()))
        while level:
            following: list[str] = []
            for value in level:
                if value in seen:
                    continue
                seen.add(value)
                if value in self.classes:
                    found.append(self.classes[value])
                following.extend(self._parents.get(value, ()))
            level = following
        return tuple(found)

    def search(
        self,
        text: str,
        *,
        exclude: Collection[str] = (),
        limit: int = 25,
    ) -> list[NCItClassRecord]:
        """Rank the classes whose name or a synonym the text mentions.

        An alias matches when the text contains each of its words other than
        fillers such as "anti" or "agent", anywhere. Classes rank by the
        rarity, across class names, of the words their best alias matched, a
        whole-phrase mention counting a quarter more, so "FLT3 inhibitor"
        outranks "cytokines"; ties go to more mentions of the rarest word.
        """

        normalized = normalize_ontology_text(text)
        if not normalized or limit < 1:
            return []
        tokens = _search_tokens(normalized)
        phrase_text = f" {normalized} "
        excluded = set(exclude)
        ranked: list[tuple[tuple[float, int], str]] = []
        for code, aliases in self._aliases.items():
            if code in excluded:
                continue
            best: tuple[float, int] | None = None
            for alias in aliases:
                content = [token for token in alias if token not in _CLASS_FILLER_WORDS]
                if not content or not all(tokens[token] for token in content):
                    continue
                matched = sum(self._idf[token] for token in alias if tokens[token])
                rarest = max(content, key=self._idf.__getitem__)
                joined = " ".join(alias)
                phrase = phrase_text.count(f" {joined} ") + phrase_text.count(
                    f" {joined}s "
                )
                score = (
                    round(matched * (1.25 if phrase else 1.0), 6),
                    phrase or tokens[rarest],
                )
                if best is None or score > best:
                    best = score
            if best is not None:
                ranked.append((best, code))
        ranked.sort(
            key=lambda item: (
                -item[0][0],
                -item[0][1],
                self.classes[item[1]].preferred_name.casefold(),
                item[1],
            )
        )
        return [self.classes[code] for _score, code in ranked[:limit]]


def _flat_rows(path: Path) -> Iterator[list[str]]:
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.endswith(".txt")]
        if len(members) != 1:
            raise ValueError("NCIt flat archive must contain exactly one text file.")
        with archive.open(members[0]) as raw_handle:
            with TextIOWrapper(raw_handle, encoding="utf-8") as handle:
                for line in handle:
                    fields = line.rstrip("\r\n").split("\t")
                    if len(fields) < 9:
                        fields.extend([""] * (9 - len(fields)))
                    yield fields


def _drug_class_codes(
    parents: Mapping[str, tuple[str, ...]],
    drug_codes: Collection[str],
    *,
    not_classes: Collection[str],
) -> tuple[set[str], set[str]]:
    """Return (class codes, every concept above a drug) for the flat file."""

    above: set[str] = set()
    stack = [parent for code in drug_codes for parent in parents.get(code, ())]
    while stack:
        code = stack.pop()
        if code in above or code not in parents:
            continue
        above.add(code)
        stack.extend(parents[code])

    @lru_cache(maxsize=None)
    def pharmacologic(code: str) -> bool:
        return code == NCIT_PHARMACOLOGIC_ROOT or any(
            pharmacologic(parent) for parent in parents.get(code, ())
        )

    classes = {
        code
        for code in above
        if code not in NCIT_UMBRELLA_CLASSES
        and code not in not_classes
        and pharmacologic(code)
    }
    return classes, above


class NCIThesaurusDrugIndex:
    """In-memory search index over NCIt pharmacologic-substance records.

    ``classes`` holds the drug-class hierarchy above them when the index was
    loaded from a flat file.
    """

    def __init__(
        self,
        records: tuple[NCItDrugRecord, ...],
        classes: NCItClassHierarchy | None = None,
    ) -> None:
        if not records:
            raise ValueError("NCIt drug index must contain at least one record.")
        self.records = records
        self.classes = classes
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
        """Load active pharmacologic substances, and the classes above them.

        The archive is read twice: once for drugs and parent links, once for
        the records of the few concepts that turn out to be classes.
        """
        records: list[NCItDrugRecord] = []
        parents: dict[str, tuple[str, ...]] = {}
        not_classes: set[str] = set()
        with _resource_path(resource_name) as path:
            for fields in _flat_rows(path):
                code = fields[0].strip()
                parents[code] = tuple(
                    item.strip() for item in fields[2].split("|") if item.strip()
                )
                semantic_types = tuple(
                    item.strip() for item in fields[7].split("|") if item.strip()
                )
                concept_status = fields[6].strip().casefold()
                inactive = "retired" in concept_status or "obsolete" in concept_status
                if inactive or _PHARMACOLOGIC_SUBSTANCE in semantic_types:
                    not_classes.add(code)
                if _PHARMACOLOGIC_SUBSTANCE not in semantic_types or inactive:
                    continue
                synonyms = tuple(
                    dict.fromkeys(
                        item.strip() for item in fields[3].split("|") if item.strip()
                    )
                )
                preferred_name = fields[5].strip() or (synonyms[0] if synonyms else "")
                if not preferred_name:
                    continue
                records.append(
                    NCItDrugRecord(
                        code=code,
                        preferred_name=preferred_name,
                        synonyms=synonyms,
                        definition=fields[4].strip() or None,
                        semantic_types=semantic_types,
                    )
                )
            drug_codes = {record.code for record in records}
            class_codes, above = _drug_class_codes(
                parents, drug_codes, not_classes=not_classes
            )
            class_records: list[NCItClassRecord] = []
            for fields in _flat_rows(path):
                code = fields[0].strip()
                if code not in class_codes:
                    continue
                synonyms = tuple(
                    dict.fromkeys(
                        item.strip() for item in fields[3].split("|") if item.strip()
                    )
                )
                preferred_name = fields[5].strip() or (synonyms[0] if synonyms else "")
                if preferred_name:
                    class_records.append(
                        NCItClassRecord(
                            code=code,
                            preferred_name=preferred_name,
                            synonyms=synonyms,
                            definition=fields[4].strip() or None,
                        )
                    )
        kept = drug_codes | above
        classes = NCItClassHierarchy(
            class_records,
            {code: values for code, values in parents.items() if code in kept},
        )
        return cls(tuple(records), classes=classes)

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
    """Load and cache the NCIt drug index and its drug-class hierarchy."""
    return NCIThesaurusDrugIndex.from_flat_zip(resource_name)


__all__ = [
    "NCIT_PHARMACOLOGIC_ROOT",
    "NCIT_UMBRELLA_CLASSES",
    "NCItClassHierarchy",
    "NCItClassRecord",
    "NCItDrugRecord",
    "NCIThesaurusDrugIndex",
    "load_ncit_drug_index",
    "normalize_ontology_text",
]
