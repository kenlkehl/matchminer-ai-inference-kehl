"""Each disease center's trial groups, as the TOP view organizes its trials (setting, biomarker, disease).

mm-dash's groups are curated by each program. Here they come from ``grouping.yaml``: per center (a regular
expression over its OnCore management group name), an ordered list of groups, each a rule over a cohort's
cancer types (``cancer_types.py``) and regular expressions over its name, key inclusion lines and trial titles.
A cohort joins the first group whose rule matches; cohorts no rule matches are grouped by cancer type. Rules
are keyword screens for browsing, not a classification, and are meant to be edited: the dashboard rebuilds when
the file changes.

Rule keys (all optional; all given keys must hold):

- ``cancer``: the cohort has at least one of these cancer-type keys.
- ``sole``: (with ``cancer``) the cohort names no other specific cancer type (baskets and multi-tumor cohorts
  fall through to later rules).
- ``exclude_cancer``: the cohort has none of these cancer-type keys.
- ``any``: at least ``any_count`` (default 1) of these patterns are mentioned (not negated).
- ``all``: every pattern is mentioned.
- ``none``: no pattern is mentioned.
- ``in``: where ``any``/``all``/``none`` look: ``name``, ``inclusion`` (key inclusion lines) and ``title``
  (default all three).

A group name may appear in several rules (e.g. a setting read from titles, then from explicit criteria); the
group shows at its first rule's place.

A mention is negated when its clause says "no", "not", "without", "absence of", "negative for" ... before it, or
"negative", "wild-type", "not detected" ... right after it ("EGFR wild type", "no prior adjuvant therapy").
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from pathlib import Path

from .core import cancer_types

TEXT_PARTS = ("name", "inclusion", "title")
PART_LABELS = {"name": "cohort name", "inclusion": "key inclusion criteria", "title": "trial title"}
BEFORE = re.compile(r"\b(?:no|not|nor|without|absence\s+of|absent|lack\s+of|lacking|negative\s+for|free\s+of|"
                    r"excluding|except|other\s+than|ineligible\s+for|unsuitable\s+for|cannot\s+receive)\b[^.;:\n]*$",
                    re.I)
AFTER = re.compile(r"^[\s-]*(?:\(?-\)?(?=\s|$|[,;/)])|negative\b|neg\b|wild[\s-]?type|wt\b|not\s+(?:detected|amplified|"
                   r"mutated|present|expressed)|mutation[\s-]negative|unmutated|non[\s-]?mutated)", re.I)
MULTIPLE = "Multiple tumor types"
OTHER = "Other"


def _compile(pattern: str) -> re.Pattern:
    try:
        return re.compile(pattern, re.I)
    except re.error as exc:
        raise ValueError(f"grouping rule pattern {pattern!r}: {exc}") from exc


def mention(pattern: re.Pattern, text: str) -> re.Match | None:
    """The first occurrence of ``pattern`` in ``text`` without a negation around it."""
    for match in pattern.finditer(text):
        clause_start = max(text.rfind(c, 0, match.start()) for c in ".;:\n")
        before = text[max(clause_start + 1, match.start() - 80):match.start()]
        if BEFORE.search(before) or AFTER.match(text[match.end():match.end() + 30]):
            continue
        return match
    return None


def mentioned(pattern: re.Pattern, text: str) -> bool:
    return mention(pattern, text) is not None


def snippet(text: str, match: re.Match, width: int = 60) -> str:
    start, end = max(0, match.start() - width), min(len(text), match.end() + width)
    return ("…" if start else "") + " ".join(text[start:end].split()) + ("…" if end < len(text) else "")


@dataclass
class Rule:
    name: str
    cancer: frozenset = frozenset()
    sole: bool = False
    exclude_cancer: frozenset = frozenset()
    any: list = field(default_factory=list)
    any_count: int = 1
    all: list = field(default_factory=list)
    none: list = field(default_factory=list)
    parts: tuple = TEXT_PARTS

    def matches(self, types: set, texts: dict) -> bool:
        if self.cancer and not (self.cancer & types):
            return False
        if self.sole and specific_types(types) - self.cancer:
            return False
        if self.exclude_cancer & types:
            return False
        text = "\n".join(texts[p] for p in self.parts)
        if self.any and sum(mentioned(p, text) for p in self.any) < self.any_count:
            return False
        if self.all and not all(mentioned(p, text) for p in self.all):
            return False
        return not (self.none and any(mentioned(p, text) for p in self.none))

    def explain(self, types: set, texts: dict) -> str:
        """Why this rule took the cohort: the first mention it relied on (where and a snippet), or the type."""
        for part in self.parts:
            for pattern in self.any + self.all:
                found = mention(pattern, texts[part])
                if found:
                    return f"{PART_LABELS[part]}: {snippet(texts[part], found)}"
        labels = [cancer_types.TYPE_LABELS[t] for t in cancer_types.TYPE_LABELS if t in self.cancer & types]
        return f"cancer type: {', '.join(labels)}" if labels else ""


@dataclass
class CenterRules:
    match: re.Pattern
    groups: list


@dataclass
class Rules:
    centers: list = field(default_factory=list)
    error: str | None = None

    def for_center(self, name: str) -> list[Rule]:
        for center in self.centers:
            if center.match.search(name):
                return center.groups
        return []

    def describe(self) -> list[dict]:
        out = [{"match": c.match.pattern, "groups": [g.name for g in c.groups]} for c in self.centers]
        return out + ([{"error": self.error}] if self.error else [])


def load_rules(path: Path) -> Rules:
    """The rules in ``path``; a file that does not parse gives no rules (every center by cancer type) and says why."""
    import yaml

    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        centers = []
        for entry in raw.get("centers") or []:
            groups = []
            for g in entry.get("groups") or []:
                unknown = (set(g.get("cancer") or []) | set(g.get("exclude_cancer") or [])) - set(cancer_types.TYPE_LABELS)
                if unknown:
                    raise ValueError(f"group {g.get('name')!r}: unknown cancer types {sorted(unknown)}")
                parts = tuple(g.get("in") or TEXT_PARTS)
                if set(parts) - set(TEXT_PARTS):
                    raise ValueError(f"group {g.get('name')!r}: 'in' must name {TEXT_PARTS}")
                unknown_keys = set(g) - {"name", "cancer", "sole", "exclude_cancer", "any", "any_count", "all", "none", "in"}
                if unknown_keys:
                    raise ValueError(f"group {g.get('name')!r}: unknown keys {sorted(unknown_keys)}")
                groups.append(Rule(str(g["name"]), frozenset(g.get("cancer") or []), bool(g.get("sole")),
                                   frozenset(g.get("exclude_cancer") or []),
                                   [_compile(p) for p in g.get("any") or []], int(g.get("any_count") or 1),
                                   [_compile(p) for p in g.get("all") or []], [_compile(p) for p in g.get("none") or []],
                                   parts))
            centers.append(CenterRules(_compile(str(entry["match"])), groups))
        return Rules(centers)
    except FileNotFoundError:
        return Rules([], f"{path} not found")
    except (ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        return Rules([], f"{type(exc).__name__}: {exc}")


def key_inclusion(items: list[dict]) -> str:
    return "\n".join(i["text"] for i in items if i.get("criterion_type") == "inclusion" and not i.get("boilerplate"))


def cohort_cancer_types(cohort: dict, trial: dict) -> list[str]:
    """The cohort's cancer types from its name and key inclusion lines (``cancer_types.cohort_types``); a cohort
    those name none takes its trial titles' types."""
    found = cancer_types.cohort_types({"cohort_name": cohort.get("name"), "items": cohort.get("items")})
    if not found:
        titles = {*cancer_types.tag_text(trial.get("short_title") or ""), *cancer_types.tag_text(
            trial.get("official_title") or ""), *cancer_types.tag_text(trial.get("long_title") or "")}
        found = [key for key, *_ in cancer_types.TYPES if key in titles]
    return found


def specific_types(types) -> set[str]:
    """The cancer types other than baskets, with "Lung (any)" dropped beside a specific lung type."""
    specific = {t for t in types if t not in cancer_types.BASKET}
    if {"lung_nsclc", "lung_sclc"} & specific:
        specific.discard("lung_any")
    return specific


def fallback_group(types: list[str]) -> str:
    """A cohort no rule took: its one cancer type's label, else "Multiple tumor types" (baskets, several types)."""
    specific = [t for t in types if t in specific_types(types)]
    if any(t in cancer_types.BASKET for t in types) or len(specific) > 1:
        return MULTIPLE
    return cancer_types.TYPE_LABELS[specific[0]] if specific else OTHER


def group_id(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "group"


def assign_groups(center: dict, trials: dict, cohorts: dict, rules: Rules) -> None:
    """Set each dashboard cohort's ``group`` and the center's ordered ``groups``."""
    center_rules = rules.for_center(center["name"])
    members: dict[str, list[str]] = {}
    for trial_id in center["trial_ids"]:
        trial = trials[trial_id]
        title = "\n".join(filter(None, (trial.get("short_title"), trial.get("official_title"), trial.get("long_title"))))
        for cid in trial["cohort_ids"]:
            cohort = cohorts[cid]
            texts = {"name": cohort.get("name") or "", "inclusion": key_inclusion(cohort.get("items") or []),
                     "title": title}
            types = set(cohort["cancer_types"])
            rule = next((r for r in center_rules if r.matches(types, texts)), None)
            cohort["group"] = rule.name if rule else fallback_group(cohort["cancer_types"])
            cohort["group_reason"] = (rule.explain(types, texts) if rule else
                                      "cancer type" + (f": {cohort['group']}" if cohort["group"] not in (MULTIPLE, OTHER) else
                                                       "s: " + ", ".join(cancer_types.TYPE_LABELS[t] for t in cohort["cancer_types"]) or ": none recognized"))
            members.setdefault(cohort["group"], []).append(cid)
    ruled = list(dict.fromkeys(r.name for r in center_rules if r.name in members))  # a name may repeat
    rest = sorted((g for g in members if g not in ruled and g not in (MULTIPLE, OTHER)),
                  key=lambda g: (-len(members[g]), g))
    order = ruled + rest + [g for g in (MULTIPLE, OTHER) if g in members and g not in ruled]

    def sort_key(cid):
        cohort = cohorts[cid]
        return (trials[cohort["trial_id"]]["protocol_no"], cohort["trial_id"], cohort["number"])

    center["groups"] = []
    for name in order:
        ids = sorted(members[name], key=sort_key)
        center["groups"].append({"id": group_id(name), "name": name, "cohort_ids": ids,
                                 "trial_ids": list(dict.fromkeys(cohorts[c]["trial_id"] for c in ids))})
    center["pending_trial_ids"] = [t for t in center["trial_ids"] if not trials[t]["cohort_ids"]]
