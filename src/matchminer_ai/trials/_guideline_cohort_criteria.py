"""Cohort-style key criteria for each guideline space (since 2026-11-14, user instruction).

Round-2 TrialSpace and the decision model learned from trial cohorts' key criteria: protocol-style sentences under
"Inclusion criteria:" and "Exclusion criteria:". A guideline space is nine labeled fields, so after canonicalization
fixes a space, one request per space asks the endpoint to write the cohort a trial would state for that population.
Every non-NA field must be expressed, and nothing the fields do not contain may be added. Each criterion names the
fields it expresses, and code checks that coverage. The request sees the space's name, its nine fields and the
guideline source lines its evidence cites (context only), plus style examples from trial protocols in other cancers.

Records gain ``key_criteria`` (the rendered text, read by ``load_guideline_catalog`` as ``cohort_key_criteria``) and
``key_criteria_items`` (``[{type, text, fields}]``). ``space``, ``space_trial_id`` and ``clinical_space_summary`` do
not change, and catalogs without these fields keep the derived layout (``format_key_criteria``).
"""

import json
import re

from ._guideline_schema import FIELDS
from .cohort_boilerplate import gather_criteria
from .prompt_builder import load_prompt_text

VERSION = "cohort-criteria-v1"
JOB_PREFIX = "cohort-criteria-"
PROMPT_FILES = (
    "guideline.cohort_criteria.system.txt",
    "guideline.cohort_criteria.txt",
    "guideline.cohort_criteria.examples.txt",
)
SYSTEM = load_prompt_text("guideline.cohort_criteria.system.txt").strip()
TASK = load_prompt_text("guideline.cohort_criteria.txt").strip()
EXAMPLES = load_prompt_text("guideline.cohort_criteria.examples.txt").strip()
MAX_SOURCE_LINES = 60
INCLUSION, EXCLUSION = "Inclusion criteria:", "Exclusion criteria:"

SCHEMA = {
    "type": "object",
    "properties": {"criteria": {"type": "array", "minItems": 1, "items": {
        "type": "object",
        "properties": {
            "type": {"type": "string", "enum": ["inclusion", "exclusion"]},
            "text": {"type": "string", "minLength": 1},
            "fields": {"type": "array", "minItems": 1, "items": {"type": "string", "enum": list(FIELDS)}},
        },
        "required": ["type", "text", "fields"],
        "additionalProperties": False,
    }}},
    "required": ["criteria"],
    "additionalProperties": False,
}

# Generic trial requirements a guideline population never states unless a field does.
BOILERPLATE_TERMS = (
    "informed consent", "ecog", "karnofsky", "performance status", "life expectancy", "organ function",
    "adequate hematologic", "contracept", "pregnan", "measurable disease", "recist", "willing and able",
)
_CROSS_REFERENCE = re.compile(r"\b(as above|same as|see (above|below|section)|as defined (above|below))\b", re.I)
_BOOLEAN_SYNTAX = re.compile(r"\(\s*\(|\bAND\b|\bOR\b")
_NUMBERING = re.compile(r"^\s*(\d+[.)]|[a-z][.)]|[-•*])\s")
_NA_WORD = re.compile(r"\bN/?A\b")


def is_na(value):
    return str(value).strip().rstrip(".").casefold() in {"na", "n/a"}


def active_fields(space):
    """The fields that restrict the population (not NA), in ``FIELDS`` order."""
    return {key: space[key] for key in FIELDS if not is_na(space[key])}


def validate_criteria(space, value):
    """Raise ValueError naming every problem; return ``value`` when the criteria fit the space."""
    active = active_fields(space)
    items = value.get("criteria") or []
    problems = []
    if not any(item["type"] == "inclusion" for item in items):
        problems.append("Write at least one inclusion criterion.")
    cited = {key for item in items for key in item["fields"]}
    for key, text in active.items():
        if key not in cited:
            problems.append(f"Field {key} ({text!r}) is not NA but no criterion expresses it.")
    field_text = " ".join(active.values()).casefold()
    for number, item in enumerate(items, 1):
        text = item["text"].strip()
        for key in item["fields"]:
            if key not in active:
                problems.append(
                    f"Criterion {number} cites {key}, which is NA (no restriction); cite only fields it expresses."
                )
        if not text:
            problems.append(f"Criterion {number} is empty.")
        if "\n" in text:
            problems.append(f"Criterion {number} spans several lines; write one criterion per item.")
        if _NUMBERING.match(text):
            problems.append(f"Criterion {number} starts with a number or bullet; write the sentence only.")
        if _BOOLEAN_SYNTAX.search(text):
            problems.append(
                f"Criterion {number} uses Boolean syntax (nested parentheses or capitalized AND/OR); "
                "state the logic in words."
            )
        if _NA_WORD.search(text):
            problems.append(f"Criterion {number} mentions NA; NA fields get no criterion.")
        if _CROSS_REFERENCE.search(text):
            problems.append(f"Criterion {number} refers to other text; make it self-contained.")
        low = text.casefold()
        for term in BOILERPLATE_TERMS:
            if term in low and term not in field_text:
                problems.append(
                    f"Criterion {number} adds a generic trial requirement ({term!r}) that the fields do not contain."
                )
    if problems:
        raise ValueError("The criteria do not meet the requirements:\n- " + "\n- ".join(dict.fromkeys(problems)))
    return value


def render_criteria(items):
    """A trial cohort's key-criteria layout: inclusion criteria, then exclusion criteria, one per line."""
    ordered = sorted(items, key=lambda item: item["type"] != "inclusion")
    pairs = [(INCLUSION if item["type"] == "inclusion" else EXCLUSION, item["text"].strip()) for item in ordered]
    return gather_criteria(pairs, [False] * len(pairs))[0]


def cited_source_lines(record):
    """The guideline lines the record's evidence cites, deduplicated, in page and line order."""
    seen, lines = set(), []
    for item in record.get("evidence") or []:
        for line in item.get("source_lines") or []:
            key = (item.get("page_id"), line.get("line"))
            text = str(line.get("text", "")).rstrip()
            if key in seen or not text.strip():
                continue
            seen.add(key)
            lines.append((str(item.get("printed_label") or item.get("page_id")), int(line.get("line") or 0), text))
    lines.sort(key=lambda entry: (entry[0], entry[1]))
    return lines[:MAX_SOURCE_LINES]


def build_messages(record, title):
    """The request for one record: task, style examples, then the population (name, fields, cited lines)."""
    population = [
        f"GUIDELINE: {title}",
        f"POPULATION NAME: {record['name']}",
        "FIELDS (the definition; NA = no restriction):\n"
        + json.dumps({key: record["space"][key] for key in FIELDS}, ensure_ascii=False, indent=1),
    ]
    lines = cited_source_lines(record)
    if lines:
        population.append("GUIDELINE SOURCE LINES (context only):\n" + "\n".join(
            f"[{page} line {number}] {text.strip()}" for page, number, text in lines))
    user = "\n\n".join([
        TASK,
        EXAMPLES,
        "THE POPULATION TO REWRITE\n" + "\n\n".join(population),
        'Return JSON: {"criteria": [{"type": "inclusion" or "exclusion", "text": "...", "fields": ["<field key>", ...]}]}',
    ])
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def normalized_items(value):
    """The response's criteria as stored: text stripped, keys in a fixed order."""
    return [{"type": item["type"], "text": item["text"].strip(), "fields": list(item["fields"])}
            for item in value["criteria"]]


def with_cohort_criteria(record, value):
    """The record with ``key_criteria_items`` and the rendered ``key_criteria``."""
    items = normalized_items(value)
    return {**record, "key_criteria": render_criteria(items), "key_criteria_items": items}


def write_cohort_criteria(client, records, title, run_jobs, workers, *, notify=lambda message: None):
    """One checkpointed request per record (jobs ``cohort-criteria-NNNN``); raise if any fails."""

    def one(key, record):
        value = client.complete(key, build_messages(record, title), SCHEMA,
                                lambda v: validate_criteria(record["space"], v))
        return with_cohort_criteria(record, value)

    notify(f"writing cohort-style key criteria for {len(records)} spaces")
    values, failures = run_jobs(
        [(f"{JOB_PREFIX}{number:04d}", record) for number, record in enumerate(records, 1)], workers, one)
    if failures:
        raise RuntimeError(f"{len(failures)} cohort-criteria jobs failed; rerun to resume successful checkpoints")
    return [value for _, value in sorted(values.items())]


def audit_cohort_criteria(rows, accepted_results):
    """Every row's criteria are an accepted response, still validate, and render to its ``key_criteria``.

    ``accepted_results`` holds the accepted ``cohort-criteria-*`` checkpoint results.
    """
    from matchminer_ai._storage import digest

    accepted = {digest(normalized_items(result)) for result in accepted_results}
    for row in rows:
        items = row.get("key_criteria_items")
        if not isinstance(items, list) or not isinstance(row.get("key_criteria"), str):
            raise ValueError(f"{row.get('paradigm_id')}: missing cohort-style key criteria")
        validate_criteria(row["space"], {"criteria": items})
        if row["key_criteria"] != render_criteria(items):
            raise ValueError(f"{row.get('paradigm_id')}: key_criteria differs from its items")
        if digest(items) not in accepted:
            raise ValueError(f"{row.get('paradigm_id')}: key criteria are not an accepted endpoint response")
