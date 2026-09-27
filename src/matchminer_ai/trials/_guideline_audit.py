"""Offline checks of exported catalogs, provenance, coverage and recorded requests."""

import copy
import csv
import json
from collections import Counter

from matchminer_ai._storage import digest, read_json

from ._guideline_canonical import (
    CONTENT_VERSIONS,
    LEAN_STATE,
    selected_states,
    validate_catalog,
)
from matchminer_ai.llm.structured import JSON_NORMALIZATION_VERSION, parse_model_json
from ._guideline_generation import clinical_content
from ._guideline_completeness import VERSION as COVERAGE_VERSION
from ._guideline_completeness import validate_report as validate_coverage_report
from ._guideline_ownership import VERSION as OWNERSHIP_VERSION
from ._guideline_ownership import branch_ledger
from ._guideline_quotes import VERSION as QUOTE_VERSION
from ._guideline_quotes import materialize_quoted_state, resolve_excerpt
from ._guideline_schema import (
    STATE,
    format_space,
    materialize_evidence,
    normalize_evidence_lists,
    validate_canonical,
    validate_extraction,
    validate_state,
)
from ._guideline_specificity import REVISION as SPECIFICITY_REVISION
from ._guideline_specificity import VERSION as SPECIFICITY_VERSION
from ._guideline_specificity import validate_decision_fields


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit_accepted_response(folder, accepted):
    """Verify accepted content against the exact, unmodified provider response."""
    matches = []
    for path in folder.glob("attempt-*.json"):
        response = read_json(path)
        if digest(response) == accepted.get("response_sha256"):
            matches.append(response)
    require(
        bool(matches), "Accepted response hash has no matching raw provider response"
    )
    response = matches[0]
    require(
        response["choices"][0]["finish_reason"] == "stop",
        "Accepted incomplete provider response",
    )
    value, counts = parse_model_json(response["choices"][0]["message"]["content"])
    require(
        counts == accepted["json_normalization"], "JSON normalization counts differ"
    )
    normalize_evidence_lists(value)
    if accepted.get("repair_applied"):
        require(
            clinical_content(value) == clinical_content(accepted["result"]),
            "Citation repair changed clinical content",
        )
    else:
        require(
            value == accepted["result"],
            "Accepted result dropped or changed provider content",
        )
    return counts


def audit_catalog(guideline, output):
    """Validate saved results without LLM calls or a clinical-completeness claim."""
    status = read_json(output / "status.json")
    require(status["status"] == "complete", "Catalog is not complete")
    config = read_json(output / "run_config.json")
    require(
        config["source_fingerprint"] == guideline.fingerprint,
        "Source fingerprint changed",
    )
    settings = config["llm"]
    quoted_details = config.get("stage_versions", {}).get("detail_citations") == QUOTE_VERSION
    current_json = (
        config.get("stage_versions", {}).get("json_normalization")
        == JSON_NORMALIZATION_VERSION
    )
    normalized_results = {"extract": set(), "detail": set()}
    if current_json:
        for path in (output / "checkpoints").glob("*/accepted.json"):
            accepted = read_json(path)
            if accepted.get("json_normalization_version") != JSON_NORMALIZATION_VERSION:
                continue
            for kind in normalized_results:
                if accepted["job"].startswith(kind + "-"):
                    result = accepted["result"]
                    if kind == "detail" and "space" in result:
                        materialize = materialize_quoted_state if quoted_details else materialize_evidence
                        normalized_results[kind].add(digest(materialize(result, guideline.pages)))
                    elif kind == "extract" and "candidates" in result:
                        normalized_results[kind].add(digest(result))
    rows = [
        json.loads(line)
        for line in (output / "paradigms.jsonl").read_text().splitlines()
    ]
    require(bool(rows), "Empty catalog")
    require(len(rows) == status["paradigms"], "Catalog count differs from status")
    require(
        len(rows) == len({row["paradigm_id"] for row in rows}), "Duplicate paradigm IDs"
    )
    candidates = read_json(output / "candidates.json")
    canonical = read_json(output / "canonical_groups.json")
    if config.get("stage_versions", {}).get("catalog_coverage") == COVERAGE_VERSION:
        accepted_reviews = set()
        for path in (output / "checkpoints").glob("*/accepted.json"):
            accepted = read_json(path)
            if accepted["job"].startswith("catalog-coverage-"):
                accepted_reviews.add((accepted["job"], digest(accepted["result"])))
        validate_coverage_report(
            read_json(output / "canonical_coverage.json"), candidates, canonical["groups"],
            accepted_reviews,
        )
    if canonical.get("version") in CONTENT_VERSIONS:
        batches = read_json(output / "canonical_batches.json")
        require(not batches["failures"], "Canonical batches contain failures")
        processed = []
        for batch in batches["batches"].values():
            processed.extend(batch["input_candidate_ids"])
            validate_catalog(
                batch["result"],
                {p: guideline.pages[p] for p in batch["context"]["included_page_ids"]},
            )
        require(
            len(processed) == len(set(processed))
            and set(processed) == {c["candidate_id"] for c in candidates},
            "Code-side batching did not process every candidate exactly once",
        )
        for group in canonical["groups"]:
            validate_catalog(
                {
                    "states": [{k: group[k] for k in LEAN_STATE["properties"]}],
                    "context_only_topics": [],
                    "uncertainties": [],
                },
                guideline.pages,
            )
        if (output / "canonical_selection.json").exists():
            selection = read_json(output / "canonical_selection.json")
            require(
                selected_states(selection["result"], selection["available_states"])
                == canonical["groups"],
                "Selected clinical definitions differ from canonical catalog",
            )
    else:
        validate_canonical(canonical, [c["candidate_id"] for c in candidates])
    require(
        len(canonical["groups"]) == len(rows),
        "Catalog differs from canonical group count",
    )
    content_ids = {page.id for page in guideline.primary}
    evidence_count = workup_count = treatment_count = 0
    for row, group in zip(rows, canonical["groups"]):
        if (
            config.get("stage_versions", {}).get("decision_fields")
            == SPECIFICITY_VERSION
        ):
            validate_decision_fields(row, guideline.metadata["title"])
        require(
            row["space"] == group["space"],
            "Final state differs from canonical definition",
        )
        lineage_field = (
            "source_batch_numbers"
            if canonical.get("version") in CONTENT_VERSIONS
            else "candidate_ids"
        )
        require(row[lineage_field] == group[lineage_field], "Source lineage differs")
        require(
            row["clinical_space_summary"] == format_space(row["space"]),
            "TrialSpace text differs",
        )
        require(
            row["source"]["source_fingerprint"] == guideline.fingerprint,
            "Row source differs",
        )
        require(row["review_required"] is True, "Missing review requirement")
        require(row["space_trial_id"] == row["paradigm_id"], "TrialSpace ID differs")
        included, omitted = (
            set(row["context_page_ids"]),
            set(row["omitted_source_page_ids"]),
        )
        require(
            not included & omitted and included | omitted == content_ids,
            "Context page accounting is incomplete",
        )
        packing = row["context_packing"]
        require(
            packing["reserved_output_tokens"] == settings["max_tokens"],
            "Output reserve differs",
        )
        require(
            packing["context_window"] == settings["context_window"],
            "Context window differs",
        )
        require(
            packing["prompt_tokens"]
            + packing["reserved_output_tokens"]
            + packing["safety_tokens"]
            <= packing["context_window"],
            "Recorded context exceeds token budget",
        )
        value = copy.deepcopy({key: row[key] for key in STATE["properties"]})
        materialized_digest = digest(value)
        evidence_lists = [value["evidence"]]
        for kind in ("diagnostic_workup", "treatment_options"):
            evidence_lists.extend(item["evidence"] for item in value[kind])
        for items in evidence_lists:
            for item in items:
                page = guideline.pages[item["page_id"]]
                lines = page.text.splitlines()
                expected = [{"line": i, "text": lines[i - 1]} for i in item["line_ids"]]
                require(
                    item["source_lines"] == expected,
                    "Source lines differ from original",
                )
                if quoted_details:
                    derived_ids, exact_quote = resolve_excerpt(page, item["quote"])
                    require(derived_ids == item["line_ids"], "Excerpt line addresses differ")
                    require(exact_quote == item["quote"], "Excerpt differs from exact source text")
                else:
                    require(
                        item["quote"] == "\n".join(line["text"] for line in expected),
                        "Evidence quotation differs from original",
                    )
                require(item["pdf_page"] == page.number, "Evidence PDF page differs")
                for key in set(item) - {"page_id", "line_ids"}:
                    del item[key]
                evidence_count += 1
        validate_state(value, {key: guideline.pages[key] for key in included})
        if current_json:
            require(
                materialized_digest in normalized_results["detail"],
                "Exported clinical content differs from every losslessly parsed model response",
            )
        workup_count += len(row["diagnostic_workup"])
        treatment_count += len(row["treatment_options"])

    with (output / "trial_spaces.csv").open(newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    require(len(csv_rows) == len(rows), "CSV row count differs")
    require(
        all(all(a[key] == str(b[key]) for key in a) for a, b in zip(csv_rows, rows)),
        "CSV records differ from JSONL",
    )
    coverage = read_json(output / "coverage.json")
    covered = [page["page_id"] for page in coverage["page_coverage"]]
    require(
        len(covered) == len(set(covered)) and set(covered) == content_ids,
        "Coverage does not account for every content page exactly once",
    )
    extraction = read_json(output / "extraction.json")
    require(not extraction["failures"], "Extraction contains failures")
    owns_branches = (
        config.get("stage_versions", {}).get("ownership") == OWNERSHIP_VERSION
    )
    for packet in extraction["packets"].values():
        if current_json:
            require(
                digest(packet["result"]) in normalized_results["extract"],
                "Extraction differs from every losslessly parsed model response",
            )
        validate_extraction(
            packet["result"],
            {key: guideline.pages[key] for key in packet["context_page_ids"]},
            packet["primary_page_ids"],
            require_ownership=owns_branches,
        )
    if owns_branches:
        require(
            read_json(output / "extraction_ownership.json")
            == branch_ledger(extraction["packets"], content_ids),
            "Extraction ownership ledger differs",
        )
        expected_candidates = [
            {"candidate_id": f"{job}-c{i:03d}", **candidate}
            for job, packet in sorted(extraction["packets"].items())
            for i, candidate in enumerate(packet["result"]["candidates"], 1)
        ]
        require(
            candidates == expected_candidates,
            "Candidates differ from their owning extractions",
        )

    requests = list((output / "checkpoints").glob("*/request.json"))
    require(bool(requests), "No recorded endpoint requests")
    accepted_count = normalized_citations = checked_raw_responses = 0
    json_counts = {
        "identical_duplicate_keys": 0,
        "merged_list_keys": 0,
        "recovered_list_items": 0,
    }
    for path in requests:
        request = read_json(path)
        body = request["body"]
        for key in ("model", "temperature", "top_p", "max_tokens"):
            require(
                body[key] == settings[key], f"Request {key} differs from configuration"
            )
        require(body.get("top_k", 0) == settings["top_k"], "Request top_k differs")
        expected_thinking = (
            None
            if settings["thinking"] == "default"
            else {"enable_thinking": settings["thinking"] == "on"}
        )
        expected_thinking = settings.get("extra_body", {}).get(
            "chat_template_kwargs", expected_thinking
        )
        require(
            body.get("chat_template_kwargs") == expected_thinking,
            "Request thinking mode differs",
        )
        for key, value in {
            **settings.get("request_params", {}),
            **settings.get("extra_body", {}),
        }.items():
            require(body.get(key) == value, f"Request extension {key} differs")
        require(
            request["context_window"] == settings["context_window"],
            "Request context differs",
        )
        require(
            request["reserved_output_tokens"] == settings["max_tokens"],
            "Request reserve differs",
        )
        require(
            request["prompt_tokens"]
            + request["reserved_output_tokens"]
            + request["safety_tokens"]
            <= request["context_window"],
            "Request exceeds token budget",
        )
        accepted_path = path.with_name("accepted.json")
        if accepted_path.exists():
            accepted = read_json(accepted_path)
            require(
                accepted["request_sha256"] == path.parent.name,
                "Checkpoint identity differs",
            )
            require(
                accepted["result_sha256"] == digest(accepted["result"]),
                "Checkpoint is corrupted",
            )
            accepted_count += 1
            normalized_citations += accepted.get("normalized_citation_items", 0)
            if accepted.get("json_normalization_version") == JSON_NORMALIZATION_VERSION:
                counts = audit_accepted_response(path.parent, accepted)
                checked_raw_responses += 1
                for key in json_counts:
                    json_counts[key] += counts[key]
        for attempt_path in path.parent.glob("request-attempt-*.json"):
            attempt = read_json(attempt_path)
            actual = attempt["body"]
            for key in {
                "model",
                "temperature",
                "top_p",
                "max_tokens",
                "top_k",
                "chat_template_kwargs",
                *settings.get("request_params", {}),
                *settings.get("extra_body", {}),
            }:
                require(actual.get(key) == body.get(key), f"Retry changed {key}")
            require(
                attempt["reserved_output_tokens"] == settings["max_tokens"],
                "Retry reserve differs",
            )
            require(
                attempt["prompt_tokens"] + attempt["reserved_output_tokens"]
                <= settings["context_window"],
                "Retry exceeds full context window",
            )
    return {
        "status": "passed",
        "paradigms": len(rows),
        "diagnostic_workup_items": workup_count,
        "extraction_ownership": OWNERSHIP_VERSION
        if owns_branches
        else "legacy_unchecked",
        "decision_field_validation_revision": (
            SPECIFICITY_REVISION
            if config.get("stage_versions", {}).get("decision_fields")
            == SPECIFICITY_VERSION
            else "legacy"
        ),
        "treatment_options": treatment_count,
        "verified_evidence_items": evidence_count,
        "detail_citation_format": QUOTE_VERSION if quoted_details else "legacy-line-ids",
        "covered_content_pages": len(covered),
        "source_pdf_pages": len(guideline.pages),
        "coverage_dispositions": dict(
            Counter(p["disposition"] for p in coverage["page_coverage"])
        ),
        "source_fingerprint": guideline.fingerprint,
        "thinking": settings["thinking"],
        "sampling": {key: settings[key] for key in ("temperature", "top_p", "top_k")},
        "context_window": settings["context_window"],
        "reserved_output_tokens": settings["max_tokens"],
        "paradigms_sha256": digest((output / "paradigms.jsonl").read_bytes()),
        "recorded_requests": len(requests),
        "accepted_requests": accepted_count,
        "normalized_citation_items": normalized_citations,
        "verified_raw_provider_responses": checked_raw_responses,
        "json_normalization": json_counts,
        "raw_attempt_files": len(
            list((output / "checkpoints").glob("*/attempt-*.json"))
        ),
        "clinical_correctness": "not established by these structural and provenance checks",
    }
