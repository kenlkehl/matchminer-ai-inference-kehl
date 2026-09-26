"""Checkpointed source-support review with immutable clinical content and exact excerpts."""

import copy
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from matchminer_ai._storage import (
    atomic_json,
    atomic_text,
    digest,
    directory_lock,
    read_json,
)

from ._guideline_context import pack_messages
from ._guideline_quotes import QUOTED_EVIDENCE, resolve_excerpt
from ._guideline_schema import (
    STATE,
    STRING,
    STRINGS,
    arr,
    materialize_evidence,
    obj,
    validate_shape,
)
from .prompt_builder import load_prompt_text

VERSION = "citation-review-v1"
ITEM = obj({"name": STRING, "evidence": QUOTED_EVIDENCE, "issues": STRINGS})
REVIEW = obj(
    {"population": ITEM, "diagnostic_workup": arr(ITEM), "treatment_options": arr(ITEM)}
)


def owners(record):
    return [record, *record["diagnostic_workup"], *record["treatment_options"]]


def apply_review(original, review, pages):
    """Only replace citations and attach review metadata; never edit clinical assertions."""
    validate_shape(review, REVIEW)
    for kind in ("diagnostic_workup", "treatment_options"):
        if len(review[kind]) != len(original[kind]):
            raise ValueError(
                f"Review must retain exactly one entry per {kind} item in input order"
            )
    result = copy.deepcopy(original)
    errors = []
    findings = [
        review["population"],
        *review["diagnostic_workup"],
        *review["treatment_options"],
    ]
    for owner, finding in zip(owners(result), findings):
        if finding["name"] != owner["name"]:
            raise ValueError("Review names and order must exactly match the input")
        if not finding["evidence"] and not finding["issues"]:
            raise ValueError(
                "An empty evidence list requires a specific source-support issue"
            )
        evidence = []
        for number, ref in enumerate(finding["evidence"], 1):
            try:
                if ref["page_id"] not in pages:
                    raise ValueError(
                        f"Unknown or unsupplied evidence page: {ref['page_id']}"
                    )
                ids, quote = resolve_excerpt(pages[ref["page_id"]], ref["source_text"])
            except ValueError as exc:
                errors.append(f"{owner['name']} excerpt {number}: {exc}")
                continue
            shell = {
                "evidence": [{"page_id": ref["page_id"], "line_ids": ids}],
                "diagnostic_workup": [],
                "treatment_options": [],
            }
            item = materialize_evidence(shell, pages)["evidence"][0]
            item["quote"] = quote
            evidence.append(item)
        owner["evidence"] = evidence
        owner["citation_review"] = {
            "version": VERSION,
            "status": "unresolved"
            if finding["issues"]
            else "supported_by_model_review",
            "issues": finding["issues"],
        }
    if errors:
        raise ValueError(
            "Correct every invalid excerpt in the draft:\n" + "\n".join(errors)
        )
    return result


def build_review_call(client, guideline, record):
    task = load_prompt_text("guideline.citation_review.txt")
    original = {key: record[key] for key in STATE["properties"]}
    required = {e["page_id"] for owner in owners(record) for e in owner["evidence"]}
    query = (
        record["clinical_space_summary"]
        + " "
        + " ".join(o["name"] for o in owners(record))
    )
    return pack_messages(
        client,
        guideline,
        required,
        query,
        task,
        {},
        REVIEW,
        tail="EXISTING RECORD WITH UNVERIFIED CITATIONS:\n"
        + json.dumps(original, ensure_ascii=False),
        population_guidance=False,
        system_message=load_prompt_text("guideline.citation_system.txt"),
    )


def review_record(client, guideline, record, job):
    context, omitted, messages = build_review_call(client, guideline, record)
    pages = {page.id: page for page in context}
    value = client.complete(
        job, messages, REVIEW, lambda v: apply_review(record, v, pages)
    )
    return {
        "job": job,
        "original_record_sha256": digest(record),
        "result": value,
        "context_page_ids": list(pages),
        "omitted_source_page_ids": omitted,
        "context_packing": {
            "prompt_tokens": client.count_tokens(messages),
            "context_window": client.config.context_window,
            "reserved_output_tokens": client.config.max_tokens,
            "safety_tokens": client.config.safety_tokens,
        },
    }


def run_review(guideline, original_path, output, client, *, workers, notify=None):
    from ._guideline_pipeline import export, now

    notify = notify or (lambda message: None)

    def progress(message):
        try:
            notify(message)
        except Exception:
            logging.getLogger(__name__).exception(
                "Citation review progress callback failed"
            )

    original_bytes = original_path.read_bytes()
    originals = [
        json.loads(line)
        for line in original_bytes.decode().splitlines()
        if line.strip()
    ]
    # Exact input bytes, settings, prompt and source hashes make resume explicit.
    identity = {
        "citation_review_version": VERSION,
        "original_catalog_sha256": digest(original_bytes),
        "source_fingerprint": guideline.fingerprint,
        "llm": client.config.public_dict(),
        "prompt_resources_sha256": {
            name: digest(load_prompt_text(name))
            for name in (
                "guideline.citation_system.txt",
                "guideline.citation_review.txt",
                "structured.retry.txt",
            )
        },
    }
    # Operational concurrency, timeout and retry count may change on resume.
    for key in ("timeout", "attempts", "max_concurrent_requests"):
        identity["llm"].pop(key, None)
    output.mkdir(parents=True, exist_ok=True)
    with directory_lock(output):
        config_path = output / "run_config.json"
        if config_path.exists() and read_json(config_path) != identity:
            raise ValueError(
                "Citation review inputs, model or prompts changed; use a new output directory"
            )
        atomic_json(config_path, identity)
        atomic_text(output / "original_paradigms.jsonl", original_bytes.decode())
        atomic_json(output / "sources.json", originals[0]["source"])
        reviews = output / "reviews"
        reviews.mkdir(exist_ok=True)
        status = {
            "status": "running",
            "stage": "citation_review",
            "paradigms": len(originals),
            "reviewed_spaces": 0,
            "unresolved_items": 0,
            "failures": {},
            "citation_review_version": VERSION,
            "updated_utc": now(),
        }
        atomic_json(output / "status.json", status)
        records = {}

        def work(index, row):
            job = f"citation-{index:04d}"
            path = reviews / f"{job}.json"
            value = review_record(client, guideline, row, job)
            atomic_json(path, value)
            supplied = {key: guideline.pages[key] for key in value["context_page_ids"]}
            revised = apply_review(row, value["result"], supplied)
            revised["citation_review_context"] = {
                k: value[k]
                for k in (
                    "context_page_ids",
                    "omitted_source_page_ids",
                    "context_packing",
                )
            }
            return revised

        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                pending = {
                    pool.submit(work, i, row): i for i, row in enumerate(originals, 1)
                }
                for future in as_completed(pending):
                    index = pending[future]
                    try:
                        records[index] = future.result()
                        status["reviewed_spaces"] += 1
                        status["unresolved_items"] += sum(
                            bool(o["citation_review"]["issues"])
                            for o in owners(records[index])
                        )
                    except Exception as exc:
                        status["failures"][str(index)] = f"{type(exc).__name__}: {exc}"
                    status["updated_utc"] = now()
                    atomic_json(output / "status.json", status)
                    progress(
                        f"{guideline.disease}: reviewed {status['reviewed_spaces']}/{len(originals)} spaces; "
                        f"{status['unresolved_items']} items need source review; {len(status['failures'])} failed"
                    )
            if status["failures"]:
                raise RuntimeError(
                    f"{len(status['failures'])} citation jobs failed; rerun to resume"
                )
            rows = [records[i] for i in sorted(records)]
            export(output, rows)
            status.update(
                status="complete",
                stage="export",
                updated_utc=now(),
                clinical_completeness="not_established",
                review_required=True,
            )
            atomic_json(output / "status.json", status)
            audit = audit_review(guideline, output)
            atomic_json(output / "validation.json", audit)
            return rows, audit
        except BaseException as exc:
            status.update(
                status="interrupted"
                if isinstance(exc, KeyboardInterrupt)
                else "failed",
                error=f"{type(exc).__name__}: {exc}",
                updated_utc=now(),
            )
            atomic_json(output / "status.json", status)
            raise


def audit_review(guideline, output):
    """Reconstruct exports from exact raw model reviews and unchanged original records."""
    import csv
    from ._guideline_audit import audit_accepted_response, require

    config = read_json(output / "run_config.json")
    status = read_json(output / "status.json")
    require(
        config["citation_review_version"] == VERSION, "Unknown citation review version"
    )
    require(status["status"] == "complete", "Citation review is not complete")
    require(
        config["source_fingerprint"] == guideline.fingerprint,
        "Source fingerprint changed",
    )
    original_bytes = (output / "original_paradigms.jsonl").read_bytes()
    require(
        digest(original_bytes) == config["original_catalog_sha256"],
        "Original catalog changed",
    )
    originals = [
        json.loads(line)
        for line in original_bytes.decode().splitlines()
        if line.strip()
    ]
    rows = [
        json.loads(line)
        for line in (output / "paradigms.jsonl").read_text().splitlines()
    ]
    require(len(rows) == len(originals) == status["paradigms"], "Catalog count changed")
    accepted = {}
    for path in (output / "checkpoints").glob("*/accepted.json"):
        value = read_json(path)
        require(
            value["result_sha256"] == digest(value["result"]),
            "Corrupted review checkpoint",
        )
        request = read_json(path.with_name("request.json"))
        require(
            value["request_sha256"]
            == path.parent.name
            == digest(
                {
                    "base_url": config["llm"]["base_url"],
                    "body": request["body"],
                    "schema": REVIEW,
                }
            ),
            "Review request identity differs",
        )
        for key in ("model", "max_tokens", "temperature", "top_p"):
            require(request["body"][key] == config["llm"][key], f"Review {key} changed")
        require(
            request["body"].get("top_k", 0) == config["llm"]["top_k"],
            "Review top_k differs",
        )
        thinking = (
            config["llm"]
            .get("extra_body", {})
            .get(
                "chat_template_kwargs",
                (
                    None
                    if config["llm"]["thinking"] == "default"
                    else {"enable_thinking": config["llm"]["thinking"] == "on"}
                ),
            )
        )
        require(
            request["body"].get("chat_template_kwargs") == thinking,
            "Review thinking differs",
        )
        for key, val in {
            **config["llm"].get("request_params", {}),
            **config["llm"].get("extra_body", {}),
        }.items():
            require(request["body"].get(key) == val, f"Review {key} changed")
        require(
            request["prompt_tokens"]
            + request["reserved_output_tokens"]
            + request["safety_tokens"]
            <= config["llm"]["context_window"],
            "Review exceeds token budget",
        )
        require(
            request["reserved_output_tokens"] == config["llm"]["max_tokens"]
            and request["context_window"] == config["llm"]["context_window"],
            "Review context differs",
        )
        for attempt_path in path.parent.glob("request-attempt-*.json"):
            attempt = read_json(attempt_path)
            require(
                {
                    k: v
                    for k, v in attempt["body"].items()
                    if k not in {"messages", "stream", "stream_options"}
                }
                == {
                    k: v
                    for k, v in request["body"].items()
                    if k not in {"messages", "stream", "stream_options"}
                },
                "Review retry changed inference settings",
            )
            require(
                attempt["prompt_tokens"] + attempt["reserved_output_tokens"]
                <= config["llm"]["context_window"],
                "Review retry exceeded context",
            )
        audit_accepted_response(path.parent, value)
        accepted.setdefault(value["job"], []).append((value["result"], request))
    unresolved = 0
    for index, (original, row) in enumerate(zip(originals, rows), 1):
        saved = read_json(output / "reviews" / f"citation-{index:04d}.json")
        require(
            saved["original_record_sha256"] == digest(original), "Review input changed"
        )
        matching = [
            request
            for value, request in accepted.get(saved["job"], [])
            if value == saved["result"]
        ]
        require(bool(matching), "Review differs from raw response")
        included = set(saved["context_page_ids"])
        omitted = set(saved["omitted_source_page_ids"])
        require(
            not included & omitted
            and omitted == {p.id for p in guideline.primary} - included,
            "Review source page accounting differs",
        )
        require(
            original["source"]["source_fingerprint"] == guideline.fingerprint,
            "Original source differs",
        )
        prompt = matching[0]["body"]["messages"][-1]["content"]
        require(
            set(re.findall(r"<source_page id=(p\d+) ", prompt)) == included,
            "Reviewed pages differ from recorded request",
        )
        require(
            json.dumps(
                {k: original[k] for k in STATE["properties"]}, ensure_ascii=False
            )
            in prompt,
            "Recorded review prompt differs from original record",
        )
        require(
            saved["context_packing"]
            == {
                k: matching[0][k]
                for k in (
                    "prompt_tokens",
                    "context_window",
                    "reserved_output_tokens",
                    "safety_tokens",
                )
            },
            "Saved review context budget differs from request",
        )
        expected = apply_review(
            original, saved["result"], {k: guideline.pages[k] for k in included}
        )
        expected["citation_review_context"] = {
            k: saved[k]
            for k in ("context_page_ids", "omitted_source_page_ids", "context_packing")
        }
        require(
            row == expected,
            "Export changed clinical content or differs from reviewed citations",
        )
        unresolved += sum(bool(o["citation_review"]["issues"]) for o in owners(row))
    with (output / "trial_spaces.csv").open(newline="") as stream:
        exported = list(csv.DictReader(stream))
    require(
        len(exported) == len(rows)
        and all(
            all(str(row[k]) == value for k, value in item.items())
            for row, item in zip(rows, exported)
        ),
        "TrialSpace CSV differs from catalog",
    )
    require(unresolved == status["unresolved_items"], "Unresolved issue count differs")
    return {
        "status": "passed",
        "paradigms": len(rows),
        "citation_review_version": VERSION,
        "unresolved_items": unresolved,
        "source_fingerprint": guideline.fingerprint,
        "original_catalog_sha256": config["original_catalog_sha256"],
        "paradigms_sha256": digest((output / "paradigms.jsonl").read_bytes()),
        "verified_raw_provider_responses": sum(map(len, accepted.values())),
        "clinical_correctness": "not established; model-assessed source gaps are explicitly retained",
    }
