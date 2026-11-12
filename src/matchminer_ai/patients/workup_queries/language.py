"""The workup query language: a declarative JSON query over extracted clinical items.

A guideline workup requirement ("CT chest/abdomen/pelvis with contrast", "tissue
diagnosis by core or excisional biopsy", "brain MRI if stage III-IV") is compiled
once, by an LLM, into a query of this language (:mod:`.compile`); code then runs
the query on every patient's extraction items (:mod:`.evaluate`). The language is
data, not code: a query can name item types, fields, vocabulary codes, text to
look for, date windows and counts, combined with AND, OR and NOT, and nothing
else. It cannot read files, call functions or reach the network, so a query
written by a model can only select and count a patient's own extracted items.

A query is one JSON object::

    {
      "summary": "A performed CT covering the chest, abdomen and pelvis, any time.",
      "expressible": true,
      "not_expressible_reason": "",
      "approximations": [],
      "criteria": [
        {"id": "ct_chest", "label": "CT chest", "item_type": "imaging_study",
         "status": "performed",
         "filters": [
           {"field": "modality", "op": "in", "values": ["ct", "pet_ct"], "if_not_stated": "no_match"},
           {"field": "body_regions", "op": "contains_any", "values": ["chest", "whole_body"],
            "if_not_stated": "unknown"}],
         "window": {"anchor": "any_time", "anchor_criterion": "", "days_before": -1,
                    "days_after": -1, "if_no_anchor": "unknown"},
         "min_count": 1}
      ],
      "applies_when": {"all_of": []},
      "not_applicable_when": {"all_of": []},
      "requirement": {"all_of": [{"any_of": [{"criterion": "ct_chest", "negated": false}]}]}
    }

``criteria`` are named item matchers; ``requirement``, ``applies_when`` and
``not_applicable_when`` are AND-of-OR expressions over them (each literal may be
negated). Every query is validated in code (:func:`validate_workup_query`) before
it is used; the JSON schema (:func:`query_response_schema`) constrains a model's
answer under vLLM structured outputs and has no recursion, ``null``,
``minLength`` or ``pattern``, which that decoder handles poorly.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from matchminer_ai.patients.extraction import vocabulary as v
from matchminer_ai.patients.extraction.extensions import resolve_extensions
from matchminer_ai.patients.extraction.schema import CORE_ITEM_TYPES, SCHEMA_VERSION, ItemType

QUERY_LANGUAGE_VERSION = "mmai-workup-query-v1"

MAX_CRITERIA = 12
MAX_FILTERS = 8
MAX_VALUES = 24
MAX_GROUPS = 8
MAX_LITERALS = 8
MAX_APPROXIMATIONS = 8
MAX_DAYS = 36500
MAX_COUNT = 20
MAX_TEXT = 400

STATUSES = ("performed", "planned", "any")
#: Item types whose items are performed or planned; items of other types are
#: always performed (a criterion's ``performed`` is read as ``any`` for them).
STATUS_TYPES = frozenset({"imaging_study", "procedure", "treatment"})
ANCHORS = ("any_time", "reference_date", "diagnosis_date", "criterion_earliest", "criterion_latest")
NO_ANCHOR = ("unknown", "reference_date")
IF_NOT_STATED = ("unknown", "no_match", "match")
STAGE_GROUPS = ("0", "I", "II", "III", "IV")
#: Integer fields whose 0 means "not stated" (performance scores of 0 are real).
ZERO_NOT_STATED = frozenset({"line_of_therapy"})

#: Operators and the field kinds they apply to.
OPS: dict[str, frozenset[str]] = {
    "in": frozenset({"choice"}),
    "not_in": frozenset({"choice"}),
    "contains_any": frozenset({"choice_list"}),
    "contains_all": frozenset({"choice_list"}),
    "contains_none": frozenset({"choice_list"}),
    "text_contains_any": frozenset({"text", "text_list"}),
    "text_contains_none": frozenset({"text", "text_list"}),
    "text_equals_any": frozenset({"text", "text_list"}),
    "stage_group_in": frozenset({"text"}),
    "number_at_least": frozenset({"integer"}),
    "number_at_most": frozenset({"integer"}),
    "is_stated": frozenset({"choice", "choice_list", "text", "text_list", "integer"}),
}
OP_GUIDE = {
    "in": "the field's code is one of values",
    "not_in": "the field's code is none of values",
    "contains_any": "the list field contains at least one of values",
    "contains_all": "the list field contains every one of values",
    "contains_none": "the list field contains none of values",
    "text_contains_any": "the text (or any entry of a text list) contains one of values as a "
    "whole word or phrase, ignoring case and punctuation",
    "text_contains_none": "the text (and every entry of a text list) contains none of values",
    "text_equals_any": "the text (or an entry of a text list) equals one of values, ignoring "
    "case and punctuation",
    "stage_group_in": "stage_group only: the stage's main group (0, I, II, III or IV, read "
    "from text such as 'IIIA', 'Stage IV' or '4') is one of values",
    "number_at_least": "the integer is at least values[0]",
    "number_at_most": "the integer is at most values[0]",
    "is_stated": "the field has a value (values must be [])",
}

_ID = re.compile(r"[a-z][a-z0-9_]{0,23}")


@dataclass(frozen=True)
class QueryField:
    """A field a filter can test: an item field, or one every item carries."""

    name: str
    kind: str
    description: str
    vocabulary: v.Vocabulary | None = None
    codes: tuple[str, ...] = ()

    @property
    def unknown(self) -> str | None:
        return self.vocabulary.unknown if self.vocabulary is not None else None


def _groups_vocabulary() -> tuple[str, ...]:
    return tuple(sorted({t.group for t in v.PATHOLOGY_PROCEDURES.terms if t.group}))


def item_types(extensions: Iterable[str] | None = ()) -> tuple[ItemType, ...]:
    """The queryable item types: the core ones, then those of the named extensions."""
    extra = tuple(t for e in resolve_extensions(list(extensions or ())) for t in e.item_types)
    return (*CORE_ITEM_TYPES, *extra)


def query_fields(item_type: ItemType) -> dict[str, QueryField]:
    """The fields a filter on ``item_type`` items may name, by name."""
    fields = {
        "level": QueryField(
            "level", "choice", "document: from one note's record; patient: from the "
            "patient-level record built from all notes.", codes=("document", "patient"),
        ),
        "document_type": QueryField(
            "document_type", "choice", "The type of the note the item was extracted from "
            "(document-level items only; not stated for patient-level items).",
            v.DOCUMENT_TYPES, v.DOCUMENT_TYPES.codes,
        ),
    }
    if item_type.occurrence:
        fields["relation"] = QueryField(
            "relation", "choice", "Document-level items only (not stated for patient-level "
            "items): how the study or procedure relates to the note.",
            v.DOCUMENT_RELATIONS, v.DOCUMENT_RELATIONS.codes,
        )
    for item_field in item_type.fields:
        if item_field.kind == "date":
            continue  # The primary date is tested by the window, not by filters.
        codes = item_field.vocabulary.codes if item_field.vocabulary is not None else ()
        fields[item_field.name] = QueryField(
            item_field.name, item_field.kind, item_field.description, item_field.vocabulary, codes
        )
        if item_field.vocabulary is v.PATHOLOGY_PROCEDURES:
            fields["procedure_group"] = QueryField(
                "procedure_group", "choice", "The procedure_type's group (derived in code).",
                codes=_groups_vocabulary(),
            )
    return fields


def field_names(types: Sequence[ItemType]) -> list[str]:
    names = {name for t in types for name in query_fields(t)}
    return sorted(names)


# ---------------------------------------------------------------------------
# JSON schema
# ---------------------------------------------------------------------------


def _object(properties: dict, description: str | None = None) -> dict:
    schema = {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }
    if description:
        schema["description"] = description
    return schema


def _expression(description: str) -> dict:
    literal = _object({
        "criterion": {"type": "string", "description": "A criterion id."},
        "negated": {"type": "boolean", "description": "true: the criterion must not hold."},
    })
    group = _object({
        "any_of": {"type": "array", "maxItems": MAX_LITERALS, "items": literal,
                   "description": "Holds when any of its literals holds."},
    })
    return _object({
        "all_of": {"type": "array", "maxItems": MAX_GROUPS, "items": group,
                   "description": "Holds when every group holds; [] always holds."},
    }, description)


def query_response_schema(extensions: Iterable[str] | None = ()) -> dict:
    """The JSON schema of a query, as a model writes it (strict, non-recursive).

    Code checks what a schema cannot: that a field belongs to its item type, that
    an operator fits the field, that values are the field's codes, that ids are
    unique and every reference resolves (:func:`validate_workup_query`).
    """
    types = item_types(extensions)
    window = _object({
        "anchor": {"type": "string", "enum": list(ANCHORS)},
        "anchor_criterion": {
            "type": "string",
            "description": "The criterion whose earliest or latest matching item dates "
            "the window (criterion_earliest and criterion_latest only); '' otherwise.",
        },
        "days_before": {"type": "integer", "minimum": -1, "maximum": MAX_DAYS,
                        "description": "Days before the anchor; -1 for no limit."},
        "days_after": {"type": "integer", "minimum": -1, "maximum": MAX_DAYS,
                       "description": "Days after the anchor; -1 for no limit."},
        "if_no_anchor": {"type": "string", "enum": list(NO_ANCHOR)},
    })
    item_filter = _object({
        "field": {"type": "string", "enum": field_names(types)},
        "op": {"type": "string", "enum": list(OPS)},
        "values": {"type": "array", "maxItems": MAX_VALUES, "items": {"type": "string"}},
        "if_not_stated": {"type": "string", "enum": list(IF_NOT_STATED)},
    })
    criterion = _object({
        "id": {"type": "string", "description": "Lowercase letters, digits and _; unique."},
        "label": {"type": "string", "description": "A short readable name."},
        "item_type": {"type": "string", "enum": [t.name for t in types]},
        "status": {"type": "string", "enum": list(STATUSES)},
        "filters": {"type": "array", "maxItems": MAX_FILTERS, "items": item_filter},
        "window": window,
        "min_count": {"type": "integer", "minimum": 1, "maximum": MAX_COUNT},
    })
    return _object({
        "summary": {"type": "string", "description": "One sentence: what the query checks."},
        "expressible": {"type": "boolean"},
        "not_expressible_reason": {"type": "string"},
        "approximations": {"type": "array", "maxItems": MAX_APPROXIMATIONS,
                           "items": {"type": "string"}},
        "criteria": {"type": "array", "maxItems": MAX_CRITERIA, "items": criterion},
        "applies_when": _expression("Documented facts showing the requirement applies."),
        "not_applicable_when": _expression("Documented facts showing it does not apply."),
        "requirement": _expression("What must be documented for the requirement to be met."),
    })


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_EXPRESSIONS = ("applies_when", "not_applicable_when", "requirement")
_TOP = ("summary", "expressible", "not_expressible_reason", "approximations", "criteria",
        *_EXPRESSIONS)


class QueryError(ValueError):
    """A query that does not validate; ``problems`` names every problem."""

    def __init__(self, problems: list[str]):
        super().__init__("; ".join(problems[:5]))
        self.problems = problems


def _keys(value, keys, path, problems) -> bool:
    if not isinstance(value, dict):
        problems.append(f"{path}: must be an object")
        return False
    missing, extra = [k for k in keys if k not in value], sorted(set(value) - set(keys))
    if missing:
        problems.append(f"{path}: missing fields {missing}")
    if extra:
        problems.append(f"{path}: unknown fields {extra}")
    return not missing and not extra


def _string(value, path, problems, *, blank=True) -> bool:
    if not isinstance(value, str):
        problems.append(f"{path}: must be a string")
        return False
    if len(value) > MAX_TEXT:
        problems.append(f"{path}: must be at most {MAX_TEXT} characters")
        return False
    if not blank and not value.strip():
        problems.append(f"{path}: must not be empty")
        return False
    return True


def _integer(value, path, problems, low, high) -> bool:
    if type(value) is not int or not low <= value <= high:
        problems.append(f"{path}: must be an integer from {low} to {high}")
        return False
    return True


def _check_filter(entry, path, item_type, fields, problems):
    if not _keys(entry, ("field", "op", "values", "if_not_stated"), path, problems):
        return
    name, op, values = entry["field"], entry["op"], entry["values"]
    if entry["if_not_stated"] not in IF_NOT_STATED:
        problems.append(f"{path}.if_not_stated: must be one of {list(IF_NOT_STATED)}")
    if name not in fields:
        problems.append(f"{path}.field: {name!r} is not a field of {item_type.name} items; "
                        f"use one of {sorted(fields)}")
        return
    field = fields[name]
    if op not in OPS:
        problems.append(f"{path}.op: must be one of {list(OPS)}")
        return
    if field.kind not in OPS[op] or (op == "stage_group_in" and name != "stage_group"):
        allowed = [o for o, kinds in OPS.items() if field.kind in kinds
                   and (o != "stage_group_in" or name == "stage_group")]
        problems.append(f"{path}.op: {op} does not apply to {name} ({field.kind}); use one of "
                        f"{allowed}")
        return
    if not isinstance(values, list) or len(values) > MAX_VALUES or not all(
        isinstance(x, str) for x in values
    ):
        problems.append(f"{path}.values: must be a list of at most {MAX_VALUES} strings")
        return
    if op == "is_stated":
        if values:
            problems.append(f"{path}.values: is_stated takes no values; use []")
        return
    if not values:
        problems.append(f"{path}.values: {op} needs at least one value")
        return
    if len(set(values)) != len(values):
        problems.append(f"{path}.values: values repeat")
    if field.kind in {"choice", "choice_list"}:
        unknown = [x for x in values if x not in field.codes]
        if unknown:
            problems.append(f"{path}.values: {unknown} are not {name} codes; use only "
                            f"{list(field.codes)}")
    elif op == "stage_group_in":
        unknown = [x for x in values if x not in STAGE_GROUPS]
        if unknown:
            problems.append(f"{path}.values: stage groups must be among {list(STAGE_GROUPS)}")
    elif field.kind == "integer":
        if len(values) != 1 or not re.fullmatch(r"-?\d{1,4}", values[0].strip()):
            problems.append(f"{path}.values: {op} takes exactly one integer, such as [\"2\"]")
    elif any(not x.strip() or len(x) > 200 for x in values):
        problems.append(f"{path}.values: text values must be nonblank and at most 200 characters")
    elif any(not canonical_text(x) for x in values):
        problems.append(f"{path}.values: text values need letters or digits")


def _check_expression(value, path, ids, problems) -> list[tuple[str, bool]]:
    literals: list[tuple[str, bool]] = []
    if not _keys(value, ("all_of",), path, problems):
        return literals
    groups = value["all_of"]
    if not isinstance(groups, list) or len(groups) > MAX_GROUPS:
        problems.append(f"{path}.all_of: must be a list of at most {MAX_GROUPS} groups")
        return literals
    for g, group in enumerate(groups):
        where = f"{path}.all_of[{g}]"
        if not _keys(group, ("any_of",), where, problems):
            continue
        entries = group["any_of"]
        if not isinstance(entries, list) or not 1 <= len(entries) <= MAX_LITERALS:
            problems.append(f"{where}.any_of: must list 1-{MAX_LITERALS} literals")
            continue
        for n, literal in enumerate(entries):
            at = f"{where}.any_of[{n}]"
            if not _keys(literal, ("criterion", "negated"), at, problems):
                continue
            if not isinstance(literal["negated"], bool):
                problems.append(f"{at}.negated: must be true or false")
            if literal["criterion"] not in ids:
                problems.append(f"{at}.criterion: {literal['criterion']!r} is not a criterion id")
                continue
            literals.append((literal["criterion"], bool(literal["negated"])))
    return literals


def validate_workup_query(query, extensions: Iterable[str] | None = ()) -> list[str]:
    """Every problem of ``query``, as messages a model can act on ([] when valid)."""
    problems: list[str] = []
    if not _keys(query, _TOP, "query", problems):
        return problems
    _string(query["summary"], "summary", problems, blank=False)
    if not isinstance(query["expressible"], bool):
        problems.append("expressible: must be true or false")
        return problems
    _string(query["not_expressible_reason"], "not_expressible_reason", problems)
    notes = query["approximations"]
    if not isinstance(notes, list) or len(notes) > MAX_APPROXIMATIONS:
        problems.append(f"approximations: must be a list of at most {MAX_APPROXIMATIONS} strings")
    else:
        for i, note in enumerate(notes):
            _string(note, f"approximations[{i}]", problems, blank=False)
    criteria = query["criteria"]
    if not isinstance(criteria, list) or len(criteria) > MAX_CRITERIA:
        problems.append(f"criteria: must be a list of at most {MAX_CRITERIA} criteria")
        return problems
    if not query["expressible"]:
        if criteria:
            problems.append("criteria: must be [] when expressible is false")
        for name in _EXPRESSIONS:
            if query[name] != {"all_of": []}:
                problems.append(f"{name}: must be {{\"all_of\": []}} when expressible is false")
        if isinstance(query["not_expressible_reason"], str) and not query[
            "not_expressible_reason"
        ].strip():
            problems.append("not_expressible_reason: say why the requirement cannot be checked")
        return problems
    if isinstance(query["not_expressible_reason"], str) and query["not_expressible_reason"].strip():
        problems.append("not_expressible_reason: must be '' when expressible is true")
    types = {t.name: t for t in item_types(extensions)}
    ids: dict[str, dict] = {}
    anchors: dict[str, str] = {}
    for c, criterion in enumerate(criteria):
        path = f"criteria[{c}]"
        keys = ("id", "label", "item_type", "status", "filters", "window", "min_count")
        if not _keys(criterion, keys, path, problems):
            continue
        cid = criterion["id"]
        if not isinstance(cid, str) or not _ID.fullmatch(cid):
            problems.append(f"{path}.id: use 1-24 lowercase letters, digits or _, starting "
                            "with a letter")
        elif cid in ids:
            problems.append(f"{path}.id: {cid!r} is used twice")
        else:
            ids[cid] = criterion
        _string(criterion["label"], f"{path}.label", problems, blank=False)
        item_type = types.get(criterion["item_type"])
        if item_type is None:
            problems.append(f"{path}.item_type: must be one of {sorted(types)}")
            continue
        status = criterion["status"]
        if status not in STATUSES:
            problems.append(f"{path}.status: must be one of {list(STATUSES)}")
        elif status == "planned" and item_type.name not in STATUS_TYPES:
            problems.append(f"{path}.status: {item_type.name} items are never planned; use any")
        _integer(criterion["min_count"], f"{path}.min_count", problems, 1, MAX_COUNT)
        filters = criterion["filters"]
        if not isinstance(filters, list) or len(filters) > MAX_FILTERS:
            problems.append(f"{path}.filters: must be a list of at most {MAX_FILTERS} filters")
        else:
            fields = query_fields(item_type)
            for f, entry in enumerate(filters):
                _check_filter(entry, f"{path}.filters[{f}]", item_type, fields, problems)
        window = criterion["window"]
        where = f"{path}.window"
        wkeys = ("anchor", "anchor_criterion", "days_before", "days_after", "if_no_anchor")
        if not _keys(window, wkeys, where, problems):
            continue
        anchor = window["anchor"]
        if anchor not in ANCHORS:
            problems.append(f"{where}.anchor: must be one of {list(ANCHORS)}")
            continue
        if window["if_no_anchor"] not in NO_ANCHOR:
            problems.append(f"{where}.if_no_anchor: must be one of {list(NO_ANCHOR)}")
        before_ok = _integer(window["days_before"], f"{where}.days_before", problems, -1, MAX_DAYS)
        after_ok = _integer(window["days_after"], f"{where}.days_after", problems, -1, MAX_DAYS)
        if anchor == "any_time":
            if before_ok and after_ok and (window["days_before"], window["days_after"]) != (-1, -1):
                problems.append(f"{where}: any_time takes no day limits (use -1 and -1); for a "
                                "time limit choose an anchor such as reference_date")
        if anchor.startswith("criterion_"):
            if not isinstance(window["anchor_criterion"], str) or not window["anchor_criterion"]:
                problems.append(f"{where}.anchor_criterion: {anchor} needs a criterion id")
            elif isinstance(cid, str):
                anchors[cid] = window["anchor_criterion"]
        elif window["anchor_criterion"] != "":
            problems.append(f"{where}.anchor_criterion: must be '' unless the anchor is "
                            "criterion_earliest or criterion_latest")
    for cid, target in anchors.items():
        if target not in ids:
            problems.append(f"criteria {cid!r}: anchor_criterion {target!r} is not a criterion id")
        elif target == cid:
            problems.append(f"criteria {cid!r}: a criterion cannot anchor its own window")
    seen_cycle = set()
    for start in anchors:
        chain, node = [], start
        while node in anchors and node not in chain:
            chain.append(node)
            node = anchors[node]
        if node in chain and node not in seen_cycle and len(chain) > 1:
            seen_cycle.update(chain)
            problems.append(f"criteria {sorted(set(chain))}: anchor criteria form a cycle")
    used: set[str] = set(anchors.values())
    for name in _EXPRESSIONS:
        literals = _check_expression(query[name], name, ids, problems)
        used.update(cid for cid, _ in literals)
        if name == "requirement":
            groups = query[name].get("all_of") if isinstance(query[name], dict) else None
            if isinstance(groups, list) and not groups:
                problems.append("requirement.all_of: an expressible requirement needs at least "
                                "one group")
            elif literals and all(negated for _, negated in literals):
                problems.append("requirement: needs at least one criterion that is not negated")
        if name == "not_applicable_when" and literals and all(n for _, n in literals):
            problems.append(
                "not_applicable_when: needs a documented fact, a criterion that is not negated; "
                "a step is not applicable only when documented facts say so, never because "
                "something is not documented (leave it {\"all_of\": []} instead)")
    for cid in ids:
        if cid not in used:
            problems.append(f"criteria {cid!r}: not used by any expression or window anchor")
    return problems


def canonical_query(query: dict) -> dict:
    """A validated query with ``performed`` read as ``any`` where items are never planned."""
    query = json.loads(json.dumps(query))
    for criterion in query["criteria"]:
        if criterion["status"] == "performed" and criterion["item_type"] not in STATUS_TYPES:
            criterion["status"] = "any"
    return query


def check_workup_query(query, extensions: Iterable[str] | None = ()) -> dict:
    """``query`` in canonical form; raises :class:`QueryError` when it is invalid."""
    problems = validate_workup_query(query, extensions)
    if problems:
        raise QueryError(problems)
    return canonical_query(query)


# ---------------------------------------------------------------------------
# Text matching (shared with the evaluator, so validation and evaluation agree)
# ---------------------------------------------------------------------------

_NON_WORD = re.compile(r"[^0-9a-z]+")


def canonical_text(value: str) -> str:
    """Lowercase words separated by single spaces: punctuation and case ignored."""
    return _NON_WORD.sub(" ", value.casefold()).strip()


# ---------------------------------------------------------------------------
# Prompt guide
# ---------------------------------------------------------------------------


_COMMON = ("level", "document_type")
_KINDS = {"choice": "code", "choice_list": "list of codes", "text": "text",
          "text_list": "list of text", "integer": "integer"}


def _describe(name: str, field: QueryField) -> list[str]:
    lines = [f"  {name} ({_KINDS[field.kind]}): {field.description}"]
    if field.vocabulary is not None:
        lines.append(field.vocabulary.describe())
    elif field.codes:
        lines.append("      " + ", ".join(field.codes))
    return lines


def render_query_field_guide(extensions: Iterable[str] | None = ()) -> str:
    """Every queryable item type and field, with each choice field's codes."""
    types = item_types(extensions)
    common = query_fields(types[0])
    blocks = ["Fields of every item type:\n" + "\n".join(
        line for name in _COMMON for line in _describe(name, common[name]))]
    for item_type in types:
        lines = [f"{item_type.name} (levels: {', '.join(sorted(item_type.levels))}): "
                 f"{item_type.description}"]
        if item_type.name in STATUS_TYPES:
            lines.append("  status: performed or planned (see the criterion's status).")
        dated = item_type.primary_date or "the note's date (document-level items)"
        lines.append(f"  date tested by the window: {dated}")
        for name, field in query_fields(item_type).items():
            if name not in _COMMON:
                lines.extend(_describe(name, field))
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def render_operator_guide() -> str:
    return "\n".join(f"  {op}: {text}" for op, text in OP_GUIDE.items())


def _criterion(cid, label, item_type, filters, *, status="performed", window=None, count=1):
    return {
        "id": cid, "label": label, "item_type": item_type, "status": status,
        "filters": [
            {"field": f, "op": o, "values": list(vals), "if_not_stated": n}
            for f, o, vals, n in filters
        ],
        "window": window or {"anchor": "any_time", "anchor_criterion": "", "days_before": -1,
                             "days_after": -1, "if_no_anchor": "unknown"},
        "min_count": count,
    }


def _all(*groups):
    return {"all_of": [{"any_of": [{"criterion": c, "negated": neg} for c, neg in group]}
                       for group in groups]}


_NONE = {"all_of": []}

#: Worked examples rendered into the compile prompt (and validated by the tests).
EXAMPLES: list[tuple[str, dict]] = [
    (
        "CT chest with contrast within 30 days before the reference date",
        {
            "summary": "A performed contrast CT of the chest in the 30 days before the "
            "reference date.",
            "expressible": True, "not_expressible_reason": "", "approximations": [],
            "criteria": [_criterion(
                "ct_chest", "CT chest with contrast", "imaging_study",
                [("modality", "in", ["ct"], "no_match"),
                 ("body_regions", "contains_any", ["chest"], "unknown"),
                 ("contrast", "in", ["with_contrast", "with_and_without_contrast"], "unknown")],
                window={"anchor": "reference_date", "anchor_criterion": "", "days_before": 30,
                        "days_after": 0, "if_no_anchor": "unknown"},
            )],
            "applies_when": _NONE, "not_applicable_when": _NONE,
            "requirement": _all([("ct_chest", False)]),
        },
    ),
    (
        "CT chest/abdomen/pelvis (one combined study or separate studies)",
        {
            "summary": "Performed CT imaging that covers the chest, the abdomen and the pelvis.",
            "expressible": True, "not_expressible_reason": "", "approximations": [],
            "criteria": [
                _criterion("ct_chest", "CT chest", "imaging_study",
                           [("modality", "in", ["ct"], "no_match"),
                            ("body_regions", "contains_any", ["chest"], "unknown")]),
                _criterion("ct_abdomen", "CT abdomen", "imaging_study",
                           [("modality", "in", ["ct"], "no_match"),
                            ("body_regions", "contains_any", ["abdomen"], "unknown")]),
                _criterion("ct_pelvis", "CT pelvis", "imaging_study",
                           [("modality", "in", ["ct"], "no_match"),
                            ("body_regions", "contains_any", ["pelvis"], "unknown")]),
            ],
            "applies_when": _NONE, "not_applicable_when": _NONE,
            "requirement": _all([("ct_chest", False)], [("ct_abdomen", False)],
                                [("ct_pelvis", False)]),
        },
    ),
    (
        "Tissue diagnosis by core needle or excisional biopsy",
        {
            "summary": "A performed core needle or excisional biopsy.",
            "expressible": True, "not_expressible_reason": "", "approximations": [],
            "criteria": [_criterion(
                "core_or_excision", "Core or excisional biopsy", "procedure",
                [("procedure_type", "in", ["core_needle_biopsy", "excisional_biopsy"],
                  "unknown")],
            )],
            "applies_when": _NONE, "not_applicable_when": _NONE,
            "requirement": _all([("core_or_excision", False)]),
        },
    ),
    (
        "Brain MRI (or head CT with contrast if MRI is not possible) if stage III-IV, before "
        "systemic therapy starts",
        {
            "summary": "Brain MRI, or brain CT, before the first systemic therapy (or by the "
            "reference date if none), when stage III-IV is documented.",
            "expressible": True, "not_expressible_reason": "",
            "approximations": ["Whether MRI was possible is not extracted; head CT is accepted "
                               "as the alternative."],
            "criteria": [
                _criterion("brain_mri", "Brain MRI", "imaging_study",
                           [("modality", "in", ["mri"], "no_match"),
                            ("body_regions", "contains_any", ["brain"], "unknown")],
                           window={"anchor": "criterion_earliest",
                                   "anchor_criterion": "systemic", "days_before": -1,
                                   "days_after": 0, "if_no_anchor": "reference_date"}),
                _criterion("head_ct", "Head CT with contrast", "imaging_study",
                           [("modality", "in", ["ct"], "no_match"),
                            ("body_regions", "contains_any", ["brain"], "unknown"),
                            ("contrast", "in", ["with_contrast", "with_and_without_contrast"],
                             "unknown")],
                           window={"anchor": "criterion_earliest",
                                   "anchor_criterion": "systemic", "days_before": -1,
                                   "days_after": 0, "if_no_anchor": "reference_date"}),
                _criterion("systemic", "Systemic therapy", "treatment",
                           [("treatment_type", "in", ["systemic_therapy"], "no_match")]),
                _criterion("stage_3_4", "Stage III-IV", "diagnosis",
                           [("stage_group", "stage_group_in", ["III", "IV"], "unknown")],
                           status="any"),
                _criterion("stage_0_2", "Stage 0-II", "diagnosis",
                           [("stage_group", "stage_group_in", ["0", "I", "II"], "no_match")],
                           status="any"),
            ],
            "applies_when": _all([("stage_3_4", False)]),
            "not_applicable_when": _all([("stage_0_2", False)], [("stage_3_4", True)]),
            "requirement": _all([("brain_mri", False), ("head_ct", False)]),
        },
    ),
    (
        "Biomarker testing including EGFR and ALK",
        {
            "summary": "EGFR and ALK results documented.",
            "expressible": True, "not_expressible_reason": "", "approximations": [],
            "criteria": [
                _criterion("egfr", "EGFR result", "biomarker",
                           [("biomarker", "text_contains_any", ["EGFR"], "no_match"),
                            ("result", "in", ["positive", "negative", "equivocal"], "unknown")],
                           status="any"),
                _criterion("alk", "ALK result", "biomarker",
                           [("biomarker", "text_contains_any", ["ALK"], "no_match"),
                            ("result", "in", ["positive", "negative", "equivocal"], "unknown")],
                           status="any"),
            ],
            "applies_when": _NONE, "not_applicable_when": _NONE,
            "requirement": _all([("egfr", False)], [("alk", False)]),
        },
    ),
    (
        "History and physical examination; CBC and chemistry profile",
        {
            "summary": "Not checkable from extracted items.",
            "expressible": False,
            "not_expressible_reason": "Examinations and laboratory panels are not extracted "
            "item types.",
            "approximations": [], "criteria": [],
            "applies_when": _NONE, "not_applicable_when": _NONE, "requirement": _NONE,
        },
    ),
]


def render_examples() -> str:
    return "\n\n".join(
        f"Requirement: {text}\n{json.dumps(query, ensure_ascii=False)}" for text, query in EXAMPLES
    )


__all__ = [
    "ANCHORS",
    "EXAMPLES",
    "IF_NOT_STATED",
    "OPS",
    "QUERY_LANGUAGE_VERSION",
    "QueryError",
    "QueryField",
    "SCHEMA_VERSION",
    "STAGE_GROUPS",
    "STATUSES",
    "STATUS_TYPES",
    "canonical_query",
    "canonical_text",
    "check_workup_query",
    "item_types",
    "query_fields",
    "query_response_schema",
    "render_examples",
    "render_operator_guide",
    "render_query_field_guide",
    "validate_workup_query",
]
