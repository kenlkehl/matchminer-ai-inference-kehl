"""Strict output shapes and deterministic evidence checks (not clinical validation)."""

import re

from .prompt_builder import load_prompt_text

FIELDS = {
    "age_range_allowed": "Age range allowed",
    "sex_allowed": "Sex allowed",
    "cancer_type_allowed": "Cancer type allowed",
    "histology_allowed": "Histology allowed",
    "cancer_burden_allowed": "Cancer burden allowed",
    "prior_treatment_required": "Prior treatment required",
    "prior_treatment_excluded": "Prior treatment excluded",
    "biomarkers_required": "Biomarkers required",
    "biomarkers_excluded": "Biomarkers excluded",
}


def obj(properties):
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def arr(items):
    return {"type": "array", "items": items}


STRING = {"type": "string", "minLength": 1}
STRINGS = arr(STRING)
FIELD_DESCRIPTIONS = {
    "age_range_allowed": load_prompt_text(
        "guideline.field.age_range_allowed.txt"
    ).rstrip("\n"),
    "sex_allowed": load_prompt_text("guideline.field.sex_allowed.txt").rstrip("\n"),
    "cancer_type_allowed": load_prompt_text(
        "guideline.field.cancer_type_allowed.txt"
    ).rstrip("\n"),
    "histology_allowed": load_prompt_text(
        "guideline.field.histology_allowed.txt"
    ).rstrip("\n"),
    "cancer_burden_allowed": load_prompt_text(
        "guideline.field.cancer_burden_allowed.txt"
    ).rstrip("\n"),
    "prior_treatment_required": load_prompt_text(
        "guideline.field.prior_treatment_required.txt"
    ).rstrip("\n"),
    "prior_treatment_excluded": load_prompt_text(
        "guideline.field.prior_treatment_excluded.txt"
    ).rstrip("\n"),
    "biomarkers_required": load_prompt_text(
        "guideline.field.biomarkers_required.txt"
    ).rstrip("\n"),
    "biomarkers_excluded": load_prompt_text(
        "guideline.field.biomarkers_excluded.txt"
    ).rstrip("\n"),
}
SPACE = obj({key: {**STRING, "description": FIELD_DESCRIPTIONS[key]} for key in FIELDS})
EVIDENCE_ITEM = obj(
    {"page_id": STRING, "line_ids": arr({"type": "integer", "minimum": 1})}
)
EVIDENCE = arr(EVIDENCE_ITEM)
OPTION = obj(
    {"name": STRING, "conditions": STRING, "category": STRING, "evidence": EVIDENCE}
)
STATE = obj(
    {
        "name": STRING,
        "space": SPACE,
        "evidence": EVIDENCE,
        "diagnostic_workup": arr(OPTION),
        "treatment_options": arr(OPTION),
        "uncertainties": STRINGS,
    }
)
CANDIDATE = obj(
    {
        **STATE["properties"],
        "defining_branch": {
            **EVIDENCE_ITEM,
            "description": load_prompt_text("guideline.defining_branch.txt").strip(),
        },
    }
)
LEGACY_EXTRACTION = obj(
    {
        "candidates": arr(STATE),
        "page_coverage": arr(
            obj(
                {
                    "page_id": STRING,
                    "disposition": {
                        "type": "string",
                        "enum": [
                            "states_extracted",
                            "context_only",
                            "no_disease_states",
                            "uncertain",
                        ],
                    },
                    "reason": STRING,
                }
            )
        ),
        "uncertainties": STRINGS,
    }
)
EXTRACTION = obj({**LEGACY_EXTRACTION["properties"], "candidates": arr(CANDIDATE)})
# Legacy schema retained only for reading/auditing earlier development catalogs.
# Active consolidation uses the content-only schema in canonical.py.
CANONICAL = obj(
    {
        "groups": arr(
            obj(
                {
                    "candidate_ids": STRINGS,
                    "name": STRING,
                    "space": SPACE,
                    "rationale": STRING,
                }
            )
        ),
        "context_only_candidates": arr(obj({"candidate_id": STRING, "reason": STRING})),
        "uncertainties": STRINGS,
    }
)
DETAIL = STATE


def validate_shape(value, schema, path="response"):
    kind = schema["type"]
    if kind == "object":
        if not isinstance(value, dict):
            raise ValueError(f"{path}: expected exactly {list(schema['properties'])}")
        if set(value) != set(schema["properties"]):
            missing = [key for key in schema["properties"] if key not in value]
            unexpected = [key for key in value if key not in schema["properties"]]
            raise ValueError(
                f"{path}: expected exactly {list(schema['properties'])}; "
                f"missing keys={missing}; unexpected keys={unexpected}. "
                "Keep each field at the nesting level specified by the schema."
            )
        for key, child in schema["properties"].items():
            validate_shape(value[key], child, f"{path}.{key}")
    elif kind == "array":
        if not isinstance(value, list):
            raise ValueError(f"{path}: expected array")
        for i, child in enumerate(value):
            validate_shape(child, schema["items"], f"{path}[{i}]")
    elif kind == "string":
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{path}: expected nonempty string")
        if "enum" in schema and value not in schema["enum"]:
            raise ValueError(f"{path}: expected one of {schema['enum']}")
    elif kind == "integer":
        if type(value) is not int or value < schema.get("minimum", 0):
            raise ValueError(f"{path}: expected positive integer")


def normalized(text):
    return " ".join(text.split())


def format_space(space):
    validate_shape(space, SPACE, "space")
    for value in space.values():
        if re.search(
            r"\b(same as (above|before|#?\d)|as above)\b", value, re.IGNORECASE
        ):
            raise ValueError("A TrialSpace must stand alone, without cross-references")
    return " ".join(
        f"{label}: {normalized(space[key]).rstrip('.')}."
        for key, label in FIELDS.items()
    )


def validate_evidence(evidence, pages):
    if not evidence:
        raise ValueError(
            "Each state and option needs at least one source-line citation"
        )
    for item in evidence:
        page_id = item["page_id"]
        if page_id not in pages:
            raise ValueError(f"Unknown or unsupplied evidence page: {page_id}")
        ids = item["line_ids"]
        if not ids or len(ids) != len(set(ids)) or len(ids) > 12:
            raise ValueError(
                f"{page_id}: cite 1-12 distinct source line numbers per evidence item"
            )
        lines = pages[page_id].text.splitlines()
        missing = [i for i in ids if i > len(lines)]
        if missing:
            raise ValueError(
                f"{page_id}: nonexistent source line IDs {missing}; maximum is {len(lines)}"
            )
        if not any(lines[i - 1].strip() for i in ids):
            raise ValueError(
                f"{page_id}: evidence may not consist entirely of blank lines"
            )


def normalize_evidence_lists(value):
    """Deduplicate line references and split long citations without altering clinical text.

    Invalid IDs and malformed objects are deliberately left for the validator.
    Returns the number of citation objects whose representation was normalized.
    """
    if not isinstance(value, dict):
        return 0
    states = value.get("candidates", value.get("states", [value]))
    if not isinstance(states, list):
        return 0
    count = 0
    for state in states:
        if not isinstance(state, dict):
            continue
        owners = [state]
        for kind in ("diagnostic_workup", "treatment_options"):
            items = state.get(kind, [])
            if isinstance(items, list):
                owners.extend(item for item in items if isinstance(item, dict))
        for owner in owners:
            items = owner.get("evidence")
            if not isinstance(items, list):
                continue
            normalized = []
            for item in items:
                if (
                    not isinstance(item, dict)
                    or set(item) != {"page_id", "line_ids"}
                    or not isinstance(item["line_ids"], list)
                    or not item["line_ids"]
                    or any(type(i) is not int or i < 1 for i in item["line_ids"])
                ):
                    normalized.append(item)
                    continue
                ids = list(dict.fromkeys(item["line_ids"]))
                if ids == item["line_ids"] and len(ids) <= 12:
                    normalized.append(item)
                    continue
                count += 1
                normalized.extend(
                    {"page_id": item["page_id"], "line_ids": ids[i : i + 12]}
                    for i in range(0, len(ids), 12)
                )
            owner["evidence"] = normalized
    return count


def materialize_evidence(state, pages):
    """Attach exact source text in code; the endpoint selects lines but never invents quotes."""
    import copy

    result = copy.deepcopy(state)
    all_evidence = [result["evidence"]]
    all_evidence += [
        item["evidence"]
        for kind in ("diagnostic_workup", "treatment_options")
        for item in result[kind]
    ]
    for evidence in all_evidence:
        for item in evidence:
            page = pages[item["page_id"]]
            lines = page.text.splitlines()
            item["line_ids"] = sorted(item["line_ids"])
            item["source_lines"] = [
                {"line": i, "text": lines[i - 1]} for i in item["line_ids"]
            ]
            item["quote"] = "\n".join(lines[i - 1] for i in item["line_ids"])
            item["pdf_page"] = page.number
            item["printed_label"] = page.label
    return result


def validate_state(state, pages):
    validate_shape(state, STATE)
    format_space(state["space"])
    validate_evidence(state["evidence"], pages)
    for key in ("diagnostic_workup", "treatment_options"):
        for option in state[key]:
            validate_evidence(option["evidence"], pages)


def validate_extraction(value, pages, primary_ids, *, require_ownership=True):
    validate_shape(value, EXTRACTION if require_ownership else LEGACY_EXTRACTION)
    ids = [row["page_id"] for row in value["page_coverage"]]
    if len(ids) != len(set(ids)) or set(ids) != set(primary_ids):
        raise ValueError(
            "page_coverage must account for every PRIMARY page exactly once"
        )
    # Citation repairs freeze every branch and clinical field. Check all of them
    # before any supporting citation can send the response into a repair call.
    for state in value["candidates"]:
        format_space(state["space"])
        if require_ownership:
            branch = state["defining_branch"]
            if branch["page_id"] not in primary_ids:
                raise ValueError(
                    "Extraction ownership: defining_branch must be on a PRIMARY page. "
                    "A CONTEXT-only branch belongs to that page's extraction call; "
                    "omit it here rather than changing its citation to claim ownership."
                )
            try:
                validate_evidence([branch], pages)
            except ValueError as exc:
                raise ValueError(
                    f"Extraction ownership: invalid defining_branch: {exc}"
                ) from exc
            disposition = next(
                row["disposition"]
                for row in value["page_coverage"]
                if row["page_id"] == branch["page_id"]
            )
            if disposition != "states_extracted":
                raise ValueError(
                    "Extraction ownership: the defining PRIMARY page must be marked "
                    "states_extracted. If it is only discussion/routing context, omit "
                    "the candidate; do not relabel background to duplicate another page's branch."
                )
    for state in value["candidates"]:
        validate_state({key: state[key] for key in STATE["properties"]}, pages)
    claimed = {e["page_id"] for s in value["candidates"] for e in s["evidence"]}
    if require_ownership:
        claimed.update(s["defining_branch"]["page_id"] for s in value["candidates"])
    claimed.update(
        e["page_id"]
        for s in value["candidates"]
        for kind in ("diagnostic_workup", "treatment_options")
        for option in s[kind]
        for e in option["evidence"]
    )
    missing = [
        row["page_id"]
        for row in value["page_coverage"]
        if row["disposition"] == "states_extracted" and row["page_id"] not in claimed
    ]
    if missing:
        raise ValueError(
            f"{', '.join(missing)}: states_extracted requires state or option evidence from each page"
        )


def validate_canonical(value, candidate_ids, *, allow_empty=False):
    validate_shape(value, CANONICAL)
    seen = []
    spaces = set()
    for group in value["groups"]:
        if not group["candidate_ids"] or len(set(group["candidate_ids"])) != len(
            group["candidate_ids"]
        ):
            raise ValueError(
                "Each canonical group needs distinct supporting candidate IDs"
            )
        seen.extend(group["candidate_ids"])
        summary = format_space(group["space"]).casefold()
        if summary in spaces:
            raise ValueError(
                "Duplicate canonical space; combine its supporting candidates into one group"
            )
        spaces.add(summary)
    context = [item["candidate_id"] for item in value["context_only_candidates"]]
    if len(context) != len(set(context)) or set(context) & set(seen):
        raise ValueError(
            "Context-only IDs must be distinct and not also members of a group"
        )
    if set(seen) | set(context) != set(candidate_ids):
        accounted = set(seen) | set(context)
        raise ValueError(
            "Account for all candidate IDs in groups or explicit context-only dispositions; "
            f"missing={sorted(set(candidate_ids) - accounted)}, "
            f"unknown={sorted(accounted - set(candidate_ids))}"
        )
    if not value["groups"] and not allow_empty:
        raise ValueError(
            "No canonical disease states; review whether the guideline supplies enough evidence"
        )


def normalize_identical_groups(value):
    """Union lineage for byte-identical definitions, without semantic merging.

    Keep the first name and all distinct model-written rationales. Definitions
    differing in even one field stay separate; invalid shapes still fail.
    """
    validate_shape(value, CANONICAL)
    grouped = {}
    result = []
    count = 0
    for group in value["groups"]:
        key = tuple(group["space"][field] for field in FIELDS)
        if key not in grouped:
            grouped[key] = group
            result.append(group)
            continue
        target = grouped[key]
        target["candidate_ids"] = list(
            dict.fromkeys(target["candidate_ids"] + group["candidate_ids"])
        )
        if group["rationale"] != target["rationale"]:
            target["rationale"] += "\n" + group["rationale"]
        count += 1
    value["groups"] = result
    return count
