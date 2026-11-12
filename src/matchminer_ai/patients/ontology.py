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
from functools import cached_property, lru_cache
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


#: Greek letters, which ASCII folding would drop, as NCIt spells them.
_GREEK_LETTERS = MappingProxyType(
    {"α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon",
     "κ": "kappa", "λ": "lambda", "ω": "omega"}
)
#: Words that join the agents of one product, as "/", "+" and "&" do:
#: "Daratumumab and Hyaluronidase" is "Daratumumab-hyaluronidase".
_JOINING_WORDS = frozenset({"and", "plus", "with"})
#: Salts written two ways: abbreviated, or in INN rather than USAN spelling.
_SALT_SPELLINGS = MappingProxyType(
    {
        "hcl": "hydrochloride", "hbr": "hydrobromide", "mesilate": "mesylate",
        "besilate": "besylate", "tosilate": "tosylate", "camsilate": "camsylate",
        "edisilate": "edisylate", "isetionate": "isethionate",
    }
)
#: Counter-ions and hydrates. NCIt files a salt as its own concept beside the
#: parent molecule ("Irinotecan Hydrochloride", "Irinotecan").
_SALT_WORDS = frozenset(
    {
        "hydrochloride", "dihydrochloride", "trihydrochloride", "monohydrochloride",
        "hydrobromide", "mesylate", "dimesylate", "besylate", "tosylate",
        "ditosylate", "camsylate", "edisylate", "isethionate", "maleate",
        "dimaleate", "malate", "fumarate", "hemifumarate", "citrate", "tartrate",
        "bitartrate", "sulfate", "sulphate", "bisulfate", "phosphate",
        "diphosphate", "acetate", "diacetate", "succinate", "lactate", "gluconate",
        "chloride", "dichloride", "bromide", "sodium", "disodium", "potassium",
        "calcium", "magnesium", "meglumine", "hydrate", "monohydrate", "dihydrate",
        "trihydrate", "hemihydrate", "sesquihydrate", "anhydrous",
    }
)
#: Words for how, or as whose product, an agent is given rather than which
#: agent it is: "Trastuzumab SC", "Rituximab Biosimilar".
_PRODUCT_WORDS = frozenset(
    {
        "sc", "subcutaneous", "subq", "iv", "intravenous", "oral", "injection",
        "injectable", "infusion", "tablet", "tablets", "capsule", "capsules",
        "biosimilar",
    }
)
#: Therapeutic and imaging radionuclide elements, by name.
_RADIONUCLIDE_SYMBOLS = MappingProxyType(
    {
        "actinium": "ac", "astatine": "at", "bismuth": "bi", "copper": "cu",
        "fluorine": "f", "gallium": "ga", "holmium": "ho", "indium": "in",
        "iodine": "i", "lead": "pb", "lutetium": "lu", "radium": "ra",
        "rhenium": "re", "samarium": "sm", "strontium": "sr", "technetium": "tc",
        "terbium": "tb", "thorium": "th", "yttrium": "y", "zirconium": "zr",
    }
)
_ELEMENT_SYMBOLS = frozenset(_RADIONUCLIDE_SYMBOLS.values())
#: Symbols read after a separate mass number ("177 Lu"): two letters, and not
#: "in" or "at", which follow numbers as words.
_SYMBOLS_AFTER_MASS = frozenset(
    symbol for symbol in _ELEMENT_SYMBOLS if len(symbol) == 2
) - {"in", "at"}
_MASS_NUMBER = re.compile(r"\d{2,3}m?")
_MASS_THEN_SYMBOL = re.compile(r"(\d{2,3}m?)([a-z]{1,2})")
_SYMBOL_THEN_MASS = re.compile(r"([a-z]{1,2})(\d{2,3}m?)")
_LIPOSOME_WORDS = frozenset({"liposome", "liposomes", "liposomal"})


def _isotope_follows(words: Sequence[str], index: int, symbol: str) -> bool:
    """Whether ``words[index:]`` begin with this element's mass number."""

    word = words[index] if index < len(words) else ""
    following = words[index + 1] if index + 1 < len(words) else ""
    if word == symbol:
        return bool(_MASS_NUMBER.fullmatch(following))
    if _MASS_NUMBER.fullmatch(word):
        return True
    for pattern, group in ((_MASS_THEN_SYMBOL, 2), (_SYMBOL_THEN_MASS, 1)):
        matched = pattern.fullmatch(word)
        if matched and matched.group(group) == symbol:
            return True
    return False


def _canonical_isotopes(words: Sequence[str]) -> list[str]:
    """Write every radionuclide as symbol then mass: "lu177".

    Registries and NCIt spell one isotope as "[177Lu]Lu-", "177Lu-", "Lu-177",
    "Lutetium-177", or "Lutetium Lu 177". The element's name is dropped before
    its mass, a mass written first moves after the symbol, and the symbol
    repeated after "[177Lu]" is dropped. "Lu 177" needs nothing: it is
    "lu177" once the words are joined.
    """

    canonical: list[str] = []
    index = 0
    while index < len(words):
        word = words[index]
        symbol = _RADIONUCLIDE_SYMBOLS.get(word)
        if symbol is not None and _isotope_follows(words, index + 1, symbol):
            following = words[index + 1]
            if _MASS_NUMBER.fullmatch(following):
                canonical.append(symbol + following)
                index += 2
            else:
                index += 1
            continue
        matched = _MASS_THEN_SYMBOL.fullmatch(word)
        if matched and matched.group(2) in _ELEMENT_SYMBOLS:
            canonical.append(matched.group(2) + matched.group(1))
            index += 1
            if index < len(words) and words[index] == matched.group(2):
                index += 1
            continue
        following = words[index + 1] if index + 1 < len(words) else ""
        if _MASS_NUMBER.fullmatch(word) and following in _SYMBOLS_AFTER_MASS:
            canonical.append(following + word)
            index += 2
            continue
        canonical.append(word)
        index += 1
    return canonical


def _drug_name_words(value: str) -> list[str]:
    """Words of a drug name, with the spellings that mean the same agent unified.

    On top of ``normalize_ontology_text`` (case, hyphens, slashes, plus signs):
    Greek letters are spelled out, isotopes written one way
    (``_canonical_isotopes``), salt abbreviations and INN salt spellings
    written in full, "and", "plus" and "with" read as separators like "/",
    "PEG" read as "pegylated", and "X liposome" read as "liposomal X".
    """

    text = str(value).casefold()
    for letter, name in _GREEK_LETTERS.items():
        text = text.replace(letter, f" {name} ")
    words = _canonical_isotopes(normalize_ontology_text(text).split())
    unified: list[str] = []
    liposomal = False
    for word in words:
        word = _SALT_SPELLINGS.get(word, word)
        if word in _JOINING_WORDS:
            continue
        if word in _LIPOSOME_WORDS:
            liposomal = True
            continue
        unified.append("pegylated" if word == "peg" else word)
    return ["liposomal", *unified] if liposomal else unified


def drug_name_key(value: str) -> str:
    """Return the comparison key two spellings of one drug name share.

    The words of ``_drug_name_words`` joined without spaces, so "L19IL2" and
    "L19-IL2", or "Interleukin 2" and "Interleukin-2", are one key. NCIt
    aliases and screened names are compared on it, and a drug NCIt cannot
    match is identified by it.
    """

    return "".join(_drug_name_words(value))


#: Shortest word a one-letter misspelling is corrected in. Every word within one
#: edit of a word this long shares its first or its last four letters.
_CLOSE_SPELLING_MIN_LENGTH = 8


@dataclass(frozen=True)
class _DrugNameKeys:
    """``drug_name_key`` forms of every NCIt alias, to the records they name."""

    spelled: Mapping[str, set[int]]
    salt_free: Mapping[str, set[int]]
    #: Salt-free keys of aliases that name no salt: the parent molecules.
    parents: Mapping[str, set[int]]
    #: Alphabetic alias words of ``_CLOSE_SPELLING_MIN_LENGTH`` or more
    #: letters, by "<" + their first four or ">" + their last four letters.
    words_by_end: Mapping[str, set[str]]


def _edit_position(left: str, right: str) -> int | None:
    """Where one insertion, deletion, or substitution turns left into right.

    None when the words are equal or more than one edit apart.
    """

    if left == right or abs(len(left) - len(right)) > 1:
        return None
    shorter, longer = sorted((left, right), key=len)
    start = 0
    while start < len(shorter) and shorter[start] == longer[start]:
        start += 1
    if len(shorter) == len(longer):
        same = shorter[start + 1:] == longer[start + 1:]
    else:
        same = shorter[start:] == longer[start + 1:]
    return start if same else None


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

    @cached_property
    def _name_keys(self) -> _DrugNameKeys:
        spelled: dict[str, set[int]] = {}
        salt_free: dict[str, set[int]] = {}
        parents: dict[str, set[int]] = {}
        words_by_end: dict[str, set[str]] = {}
        for record_index, record in enumerate(self.records):
            for alias in dict.fromkeys((record.preferred_name, *record.synonyms)):
                words = _drug_name_words(alias)
                if not words:
                    continue
                spelled.setdefault("".join(words), set()).add(record_index)
                kept = [word for word in words if word not in _SALT_WORDS]
                if kept:
                    salt_free.setdefault("".join(kept), set()).add(record_index)
                    if len(kept) == len(words):
                        parents.setdefault("".join(kept), set()).add(record_index)
                for word in words:
                    if len(word) >= _CLOSE_SPELLING_MIN_LENGTH - 1 and word.isalpha():
                        words_by_end.setdefault(f"<{word[:4]}", set()).add(word)
                        words_by_end.setdefault(f">{word[-4:]}", set()).add(word)
        return _DrugNameKeys(spelled, salt_free, parents, words_by_end)

    def resolve_drug_name(self, names: Sequence[str]) -> NCItDrugRecord | None:
        """Return the one concept the names spell, by the strictest rule that finds one.

        For names ``search`` cannot match exactly. Each rule compares the
        ``drug_name_key`` of the names with that of every NCIt alias, and accepts
        a concept only when it is the only one that rule finds. The rules run in
        order, each over all the names before the next:

        1. The same key: "Daratumumab-hyaluronidase" is Daratumumab and
           Hyaluronidase, "Ac-225 rosopatamab tetraxetan" is Actinium Ac 225
           Rosopatamab Tetraxetan, "Liposomal irinotecan" is Irinotecan Liposome.
        2. The same key without salts and hydrates ("5-ALA HCl" is
           Aminolevulinic Acid); when several salts of one molecule tie, the
           parent molecule is taken.
        3. Route, dosage-form and "biosimilar" words moved first ("Azacitidine
           oral" is Oral Azacitidine), or else dropped and rules 1 and 2 applied
           to the rest ("Trastuzumab SC" is Trastuzumab).
        4. One misspelled word corrected: a word of at least
           ``_CLOSE_SPELLING_MIN_LENGTH`` letters that no NCIt alias uses, one
           letter inserted, deleted or changed from words aliases do use, when
           exactly one concept is then spelled ("5-Flurouracil" is
           Fluorouracil). A word one edit inside its first two letters from an
           NCIt word is never corrected (a d-, l- or e- prefix can name another
           drug), nor is a word with digits, so "BMS-986500" is never
           "BMS-986501".
        """

        keys = self._name_keys
        word_lists = [words for name in names if (words := _drug_name_words(name))]
        rules = (
            self._by_spelling,
            self._without_salt,
            self._without_product_words,
            self._by_close_spelling,
        )
        for rule in rules:
            for words in word_lists:
                found = rule(words, keys)
                if found is not None:
                    return self.records[found]
        return None

    @staticmethod
    def _only(found: Collection[int]) -> int | None:
        return next(iter(found)) if len(found) == 1 else None

    def _by_spelling(self, words: Sequence[str], keys: _DrugNameKeys) -> int | None:
        return self._only(keys.spelled.get("".join(words), ()))

    def _without_salt(self, words: Sequence[str], keys: _DrugNameKeys) -> int | None:
        key = "".join(word for word in words if word not in _SALT_WORDS)
        found = keys.salt_free.get(key, set())
        if len(found) > 1:
            found = found & keys.parents.get(key, set())
        return self._only(found)

    def _without_product_words(
        self, words: Sequence[str], keys: _DrugNameKeys
    ) -> int | None:
        product = [word for word in words if word in _PRODUCT_WORDS]
        rest = [word for word in words if word not in _PRODUCT_WORDS]
        if not product or not rest:
            return None
        found = self._by_spelling([*product, *rest], keys)
        if found is None:
            found = self._by_spelling(rest, keys)
        if found is None:
            found = self._without_salt(rest, keys)
        return found

    def _by_close_spelling(
        self, words: Sequence[str], keys: _DrugNameKeys
    ) -> int | None:
        found: set[int] = set()
        for position, word in enumerate(words):
            if len(word) < _CLOSE_SPELLING_MIN_LENGTH or not word.isalpha():
                continue
            candidates = keys.words_by_end.get(
                f"<{word[:4]}", set()
            ) | keys.words_by_end.get(f">{word[-4:]}", set())
            if word in candidates:
                continue
            near = {
                candidate: edit
                for candidate in candidates
                if (edit := _edit_position(word, candidate)) is not None
            }
            # An edit in the first two letters can be a prefix (d-, l-, e-)
            # that names another stereoisomer or drug, not a typo.
            if any(edit < 2 for edit in near.values()):
                continue
            for candidate in near:
                corrected = [*words[:position], candidate, *words[position + 1:]]
                found |= keys.spelled.get("".join(corrected), set())
        return self._only(found)


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
    "drug_name_key",
    "load_ncit_drug_index",
    "normalize_ontology_text",
]
