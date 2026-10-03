"""Bounded clinical-content coverage review; internal identifiers never reach the LLM."""

from concurrent.futures import ThreadPoolExecutor
import json

from matchminer_ai._storage import digest
from matchminer_ai.llm.structured import EndpointError

from ._guideline_context import ContextBudgetError
from ._guideline_schema import STRING, STRINGS, arr, format_space, obj, validate_shape
from .prompt_builder import load_prompt_text

VERSION = "population-coverage-v1"
PROMPT_FILES = (
    "guideline.canonical_coverage.txt",
    "guideline.canonical_coverage_retry.txt",
    "guideline.canonical_restore.txt",
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


def identical_population_key(value):
    # Normalize only typographic spellings of the same comparison operators.
    # No thresholds, clinical words, Boolean grouping, or input records change.
    text = format_space(value["space"]).replace("≥", ">=").replace("≤", "<=")
    return value["name"], text


def validate_reviews(value, inputs, states):
    validate_shape(value, COVERAGE)
    if len(value["reviews"]) != len(inputs):
        raise ValueError("Coverage review must include every input population exactly once")
    known = {state["name"] for state in states}
    for position, (item, source) in enumerate(zip(value["reviews"], inputs), 1):
        if item["input_name"] != source["name"]:
            raise ValueError(
                "Coverage reviews must retain input clinical names in input order. "
                f"Review {position}: copy input_name exactly as "
                f"{json.dumps(source['name'], ensure_ascii=False)}; received "
                f"{json.dumps(item['input_name'], ensure_ascii=False)}. "
                "Do not abbreviate, expand, or rewrite the input name."
            )
        matched = item["matched_population_names"]
        if set(matched) - known:
            raise ValueError(
                "Coverage review names a population absent from the proposed catalog: "
                + json.dumps(sorted(set(matched) - known), ensure_ascii=False)
                + ". Copy matched_population_names exactly from proposed_catalog names; "
                "do not abbreviate, expand, or rewrite them."
            )
        if item["status"] == "represented" and not matched:
            raise ValueError("A represented input must name its matching proposed population")


def require_coverage(report):
    missing = [r for r in report["reviews"] if r["status"] != "represented"]
    if missing:
        raise CoverageError(
            f"Catalog omitted or broadened {len(missing)} of {len(report['reviews'])} "
            "input populations. Recheck the complete original source and preserve "
            "all distinct supported populations. Complete coverage findings:\n"
            + json.dumps(missing, ensure_ascii=False)
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
    exact = {identical_population_key(s) for s in states}
    for i, candidate in enumerate(candidates):
        if i not in covered and (
            identical_population_key(candidate) not in exact
            or report["reviews"][i]["matched_population_names"] != [candidate["name"]]
        ):
            raise ValueError("Nonidentical population has no accepted coverage review")
    require_coverage(report)


def review_coverage(client, candidates, states):
    """Keep full output headroom; split review inputs rather than truncate definitions."""
    inputs = [population(c) for c in candidates]
    proposed = [population(s) for s in states]
    exact = {identical_population_key(s) for s in states}
    reviews = [None] * len(inputs)
    pending = []
    for index, item in enumerate(inputs):
        if identical_population_key(item) in exact:
            reviews[index] = {
                "input_name": item["name"], "status": "represented",
                "matched_population_names": [item["name"]],
                "reason": "The same clinical name and nine-field definition are present (equivalent comparison-operator notation allowed).",
            }
        else:
            pending.append((index, item))

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
            return pack(items[:middle]) + pack(items[middle:])
        return [(items, messages)]

    jobs = []
    for start in range(0, len(pending), MAX_INPUTS_PER_REVIEW):
        jobs.extend(pack(pending[start:start + MAX_INPUTS_PER_REVIEW]))

    def review(packed):
        items, messages = packed
        job = "catalog-coverage-" + digest(messages)[:16]
        try:
            value = client.complete(
                job, messages, COVERAGE,
                lambda v: validate_reviews(v, [item for _, item in items], proposed),
                reuse_exhausted=len(items) > 1,
            )
        except EndpointError as error:
            # Retry bookkeeping failures with smaller complete clinical inputs.
            # Never infer a missing judgment, rename model output, split a
            # singleton, or turn a transport/clinical coverage failure into success.
            message = str(error)
            if len(items) <= 1 or "exhausted" not in message or not any(
                marker in message for marker in (
                    "Coverage reviews must retain input clinical names in input order",
                    "Coverage review must include every input population exactly once",
                    "Coverage review names a population absent from the proposed catalog",
                )
            ):
                raise
            middle = len(items) // 2
            return [result for child in pack(items[:middle]) + pack(items[middle:])
                    for result in review(child)]
        return [(items, value, {
            "input_positions": [index for index, _ in items],
            "job": job, "result_sha256": digest(value),
        })]

    review_batches = []
    if jobs:
        with ThreadPoolExecutor(max_workers=min(len(jobs), client.config.max_concurrent_requests)) as pool:
            for completed in pool.map(review, jobs):
                for items, result, receipt in completed:
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
