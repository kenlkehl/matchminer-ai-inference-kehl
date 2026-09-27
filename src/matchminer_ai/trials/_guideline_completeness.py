"""Bounded clinical-content coverage review; internal identifiers never reach the LLM."""

from concurrent.futures import ThreadPoolExecutor
import json

from matchminer_ai._storage import digest

from ._guideline_context import ContextBudgetError
from ._guideline_schema import STRING, STRINGS, arr, format_space, obj, validate_shape
from .prompt_builder import load_prompt_text

VERSION = "population-coverage-v1"
PROMPT_FILES = (
    "guideline.canonical_coverage.txt",
    "guideline.canonical_coverage_retry.txt",
)
TASK = load_prompt_text(PROMPT_FILES[0]).strip()
RETRY_TASK = load_prompt_text(PROMPT_FILES[1]).strip()
REVIEW = obj({
    "input_name": STRING,
    "status": {"type": "string", "enum": ["represented", "missing"]},
    "matched_population_names": STRINGS,
    "reason": STRING,
})
COVERAGE = obj({"reviews": arr(REVIEW)})
MAX_INPUTS_PER_REVIEW = 12


class CoverageError(ValueError):
    """A syntactically valid catalog lost one or more input populations."""


def population(value):
    return {"name": value["name"], "space": value["space"]}


def validate_reviews(value, inputs, states):
    validate_shape(value, COVERAGE)
    if len(value["reviews"]) != len(inputs):
        raise ValueError("Coverage review must include every input population exactly once")
    known = {state["name"] for state in states}
    for item, source in zip(value["reviews"], inputs):
        if item["input_name"] != source["name"]:
            raise ValueError("Coverage reviews must retain input clinical names in input order")
        matched = item["matched_population_names"]
        if set(matched) - known:
            raise ValueError("Coverage review names a population absent from the proposed catalog")
        if item["status"] == "represented" and not matched:
            raise ValueError("A represented input must name its matching proposed population")


def require_coverage(report):
    missing = [r for r in report["reviews"] if r["status"] != "represented"]
    if missing:
        examples = "; ".join(
            f"{row['input_name'][:105]}: {row['reason'][:105]}" for row in missing[:4]
        )
        raise CoverageError(
            f"Catalog omitted or broadened {len(missing)} of {len(report['reviews'])} "
            "input populations. Restore all distinct supplied populations, not just "
            f"these examples: {examples}"
        )


def validate_report(report, candidates, states, accepted_reviews=None):
    if (
        report.get("version") != VERSION
        or report.get("input_sha256") != digest([population(c) for c in candidates])
        or report.get("catalog_sha256") != digest([population(s) for s in states])
    ):
        raise ValueError("Population coverage report differs from its input or final catalog")
    validate_reviews({"reviews": report["reviews"]}, candidates, states)
    covered = set()
    for batch in report["review_batches"]:
        positions = batch["input_positions"]
        if any(type(i) is not int or not 0 <= i < len(candidates) for i in positions):
            raise ValueError("Invalid coverage-review input positions")
        if len(set(positions)) != len(positions) or covered.intersection(positions):
            raise ValueError("Duplicate coverage-review input positions")
        result = {"reviews": [report["reviews"][i] for i in positions]}
        if digest(result) != batch["result_sha256"] or (
            accepted_reviews is not None
            and (batch["job"], batch["result_sha256"]) not in accepted_reviews
        ):
            raise ValueError("Population coverage differs from its accepted review response")
        covered.update(positions)
    exact = {(s["name"], format_space(s["space"])) for s in states}
    for i, candidate in enumerate(candidates):
        if i not in covered and (
            (candidate["name"], format_space(candidate["space"])) not in exact
            or report["reviews"][i]["matched_population_names"] != [candidate["name"]]
        ):
            raise ValueError("Nonidentical population has no accepted coverage review")
    require_coverage(report)


def review_coverage(client, candidates, states):
    """Keep full output headroom; split review inputs rather than truncate definitions."""
    inputs = [population(c) for c in candidates]
    proposed = [population(s) for s in states]
    exact = {(s["name"], format_space(s["space"])) for s in states}
    reviews = [None] * len(inputs)
    pending = []
    for index, item in enumerate(inputs):
        if (item["name"], format_space(item["space"])) in exact:
            reviews[index] = {
                "input_name": item["name"], "status": "represented",
                "matched_population_names": [item["name"]],
                "reason": "The same clinical name and nine-field definition are present.",
            }
        else:
            pending.append((index, item))

    jobs = []

    def pack(items):
        messages = [
            {"role": "system", "content": TASK},
            {"role": "user", "content": json.dumps({
                "input_populations": [item for _, item in items],
                "proposed_catalog": proposed,
                "required_json_schema": COVERAGE,
            }, ensure_ascii=False)},
        ]
        if not client.fits(messages):
            if len(items) == 1:
                raise ContextBudgetError("Complete catalog exceeds population-coverage context budget")
            middle = len(items) // 2
            pack(items[:middle])
            pack(items[middle:])
        else:
            jobs.append((items, messages))

    for start in range(0, len(pending), MAX_INPUTS_PER_REVIEW):
        pack(pending[start:start + MAX_INPUTS_PER_REVIEW])

    def review(packed):
        items, messages = packed
        job = "catalog-coverage-" + digest(messages)[:16]
        value = client.complete(
            job, messages, COVERAGE,
            lambda v: validate_reviews(v, [item for _, item in items], proposed),
        )
        return items, value, {
            "input_positions": [index for index, _ in items],
            "job": job, "result_sha256": digest(value),
        }

    review_batches = []
    if jobs:
        with ThreadPoolExecutor(max_workers=min(len(jobs), client.config.max_concurrent_requests)) as pool:
            for items, result, receipt in pool.map(review, jobs):
                review_batches.append(receipt)
                for (index, _), row in zip(items, result["reviews"]):
                    reviews[index] = row
    report = {
        "version": VERSION,
        "input_sha256": digest(inputs),
        "catalog_sha256": digest(proposed),
        "reviews": reviews,
        "review_batches": review_batches,
    }
    validate_reviews({"reviews": reviews}, inputs, proposed)
    return report
