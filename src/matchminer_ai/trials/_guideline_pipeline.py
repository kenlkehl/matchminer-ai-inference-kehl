"""A finite three-stage pipeline: extract packets, canonicalize, populate each state."""

import csv
import io
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from functools import partial
from importlib import resources
from pathlib import Path

from matchminer_ai import __version__
from matchminer_ai._storage import (
    atomic_json,
    atomic_text,
    digest,
    directory_lock,
    read_json,
)

from . import _guideline_prompts as prompts
from ._guideline_canonical import SELECT_TASK, consolidate
from ._guideline_canonical import TASK as CATALOG_TASK
from ._guideline_canonical import VERSION as CATALOG_VERSION
from ._guideline_completeness import PROMPT_FILES as COVERAGE_PROMPT_FILES
from ._guideline_completeness import VERSION as COVERAGE_VERSION
from ._guideline_context import pack_messages
from ._guideline_details import build_detail_call
from matchminer_ai.llm.structured import JSON_NORMALIZATION_VERSION, StructuredConfig
from ._guideline_generation import Client, RETRY_ONLY_PROMPT_FILES
from ._guideline_ownership import VERSION as OWNERSHIP_VERSION
from ._guideline_ownership import branch_ledger, page_owners
from ._guideline_repairs import repair_response
from ._guideline_quotes import QUOTED_DETAIL, materialize_quoted_state
from ._guideline_quotes import VERSION as QUOTE_VERSION
from ._guideline_quote_repair import PROMPT_FILES as QUOTE_REPAIR_PROMPT_FILES
from ._guideline_quote_repair import VERSION as QUOTE_REPAIR_VERSION
from ._guideline_quote_repair import repair_quoted_response
from ._guideline_specificity import validate_decision_field_batch
from ._guideline_schema import (
    EXTRACTION,
    format_space,
    validate_extraction,
    validate_shape,
)
from ._guideline_sources import Guideline, packets
from ._guideline_specificity import VERSION as SPECIFICITY_VERSION
from ._guideline_specificity import validate_decision_fields

NOTICE = (
    "Research extraction requiring human review; not clinical advice, a treatment recommendation, "
    "an eligibility determination, or an NCCN-endorsed catalog. Source-line checks establish text "
    "occurrence, not clinical correctness. PDF flowchart relationships require source review."
)


def now():
    return datetime.now(UTC).isoformat()


def log(message):
    logging.getLogger(__name__).info(message)


def parallel_jobs(jobs, workers, function, *, notify=log):
    """Save successes independently, but never publish a complete catalog after any job fails."""
    values, errors = {}, {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(function, key, data): key for key, data in jobs}
        for future in as_completed(pending):
            key = pending[future]
            try:
                values[key] = future.result()
                notify(f"{key}: complete")
            except Exception as exc:
                errors[key] = f"{type(exc).__name__}: {exc}"
                notify(f"{key}: failed ({errors[key]})")
    return values, errors


def run_guideline(
    guideline: Guideline,
    output: Path,
    llm: StructuredConfig,
    *,
    workers=2,
    packet_chars=None,
    packet_pages=8,
    context_chars=None,
    progress_callback=None,
    finalize=None,
):
    output.mkdir(parents=True, exist_ok=True)
    with directory_lock(output):
        status = _run(
            guideline,
            output,
            llm,
            workers,
            packet_chars,
            packet_pages,
            context_chars,
            progress_callback,
            audit_pending=finalize is not None,
        )
        if finalize is not None:
            try:
                finalize()
            except (Exception, KeyboardInterrupt) as exc:
                status.update(
                    status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                    stage="audit",
                    updated_utc=now(),
                    error=f"{type(exc).__name__}: {exc}",
                )
                atomic_json(output / "status.json", status)
                raise
            if status.get("status") == "running" and status.get("stage") == "audit":
                status.update(status="complete", stage="export", updated_utc=now())
                atomic_json(output / "status.json", status)
        return status


def _run(
    guideline,
    output,
    llm,
    workers,
    packet_chars,
    packet_pages,
    context_chars,
    progress_callback=None,
    *,
    audit_pending=False,
):
    def notify(message):
        log(message)
        if progress_callback is not None:
            try:
                progress_callback(message)
            except Exception:
                logging.getLogger(__name__).exception(
                    "Guideline progress callback failed"
                )

    run_jobs = partial(parallel_jobs, notify=notify)
    prompt_files = resources.files("matchminer_ai.prompts")
    prompt_resources = {
        p.name: digest(p.read_bytes())
        for p in prompt_files.iterdir()
        if (p.name.startswith("guideline.") and p.name.endswith(".txt"))
        or p.name == "structured.retry.txt"
    }
    config = {
        "schema_version": 2,
        "package_version": __version__,
        "implementation": "matchminer_ai.trials.summarize_guidelines-v1",
        "prompt_resources_sha256": prompt_resources,
        "prompt_version": prompts.PROMPT_VERSION,
        "catalog_version": CATALOG_VERSION,
        "decision_fields_version": SPECIFICITY_VERSION,
        "ownership_version": OWNERSHIP_VERSION,
        "prompt_rules_sha256": digest(
            {
                "system": prompts.SYSTEM,
                "extract": prompts.EXTRACT_TASK,
                "detail": prompts.DETAIL_TASK,
                "population": prompts.population_rules(guideline.metadata["title"]),
                "catalog": CATALOG_TASK,
                "selection": SELECT_TASK,
            }
        ),
        "source_fingerprint": guideline.fingerprint,
        "llm": llm.public_dict(),
        "packet_chars": packet_chars,
        "packet_pages": packet_pages,
        "context_chars": context_chars,
    }
    # Runtime timeout/retry/concurrency settings don't change the requested extraction.
    identity_config = json.loads(json.dumps(config))
    for key in ("timeout", "attempts", "api_key_env", "stream", "max_concurrent_requests"):
        identity_config["llm"].pop(key)
    config_id = digest(identity_config)
    config_path = output / "run_config.json"
    if config_path.exists():
        previous_config = read_json(config_path)
        if previous_config["config_sha256"] != config_id:
            # A former broad structured.* glob included the unrelated in-memory
            # patient-review retry prompt. Verify the old digest before removing
            # only this unused dependency; clinical prompts/settings must match.
            old_identity = {key: previous_config[key] for key in config}
            old_identity = json.loads(json.dumps(old_identity))
            for key in ("timeout", "attempts", "api_key_env", "stream", "max_concurrent_requests"):
                old_identity["llm"].pop(key)
            valid_old_digest = digest(old_identity) == previous_config["config_sha256"]
            old_identity["prompt_resources_sha256"].pop("structured.memory_retry.txt", None)
            compatible_identity = json.loads(json.dumps(identity_config))
            # Adding an independent coverage guard does not change existing
            # extraction/consolidation/detail requests. Reuse them only after
            # all old settings match and the new guard revalidates consolidation.
            # Changes to an existing prompt, model, source or budget still fail.
            for name in (*COVERAGE_PROMPT_FILES, *QUOTE_REPAIR_PROMPT_FILES):
                if name not in old_identity["prompt_resources_sha256"]:
                    compatible_identity["prompt_resources_sha256"].pop(name, None)
            # These explicitly named prompts only guide rejected responses;
            # they never enter an initial generation request or validate an
            # accepted response. Keep their hashes in the run audit record, but
            # permit additions/revisions after verifying the old config digest.
            # Core clinical prompts, model/source/settings still must match.
            for name in RETRY_ONLY_PROMPT_FILES:
                old_identity["prompt_resources_sha256"].pop(name, None)
                compatible_identity["prompt_resources_sha256"].pop(name, None)
            if not valid_old_digest or digest(old_identity) != digest(compatible_identity):
                raise ValueError(
                    "Source, model, prompt or extraction settings changed; choose a new output_dir"
                )
        # A collection summary can lag behind a successful independent recovery.
        # Preserve that disease's audited result instead of rebuilding its
        # intermediate files and resending coverage reviews. Only an identical
        # configuration can take this path; migrations still run all new guards.
        status_path = output / "status.json"
        audit_path = output / "validation.json"
        export_path = output / "paradigms.jsonl"
        if (
            previous_config["config_sha256"] == config_id
            and status_path.exists() and audit_path.exists() and export_path.exists()
        ):
            saved_status = read_json(status_path)
            saved_audit = read_json(audit_path)
            if (
                saved_status.get("status") == "complete"
                and saved_audit.get("status") == "passed"
                and saved_audit.get("source_fingerprint") == guideline.fingerprint
                and saved_audit.get("paradigms", 0) > 0
                and saved_audit["paradigms"] == saved_status.get("paradigms")
                and saved_audit.get("paradigms_sha256") == digest(export_path.read_bytes())
            ):
                notify(f"{guideline.disease}: reusing audited completed catalog")
                # run_guideline still executes its final audit callback under
                # the directory lock before the public API returns success.
                return saved_status
    atomic_json(
        config_path,
        {
            **config,
            "config_sha256": config_id,
            "runtime": {"workers": workers},
            "stage_versions": {
                "extraction": prompts.PROMPT_VERSION,
                "catalog": CATALOG_VERSION,
                "catalog_coverage": COVERAGE_VERSION,
                "decision_fields": SPECIFICITY_VERSION,
                "ownership": OWNERSHIP_VERSION,
                "json_normalization": JSON_NORMALIZATION_VERSION,
                "detail_citations": QUOTE_VERSION,
                "detail_citation_repairs": QUOTE_REPAIR_VERSION,
            },
        },
    )
    metadata = guideline.metadata
    source_info = {
        "disease": guideline.disease,
        "title": metadata["title"],
        "version": metadata["version"],
        "source_pdf": metadata["source_pdf"],
        "source_sha256": metadata["source_sha256"],
        "source_fingerprint": guideline.fingerprint,
        "input_directory": str(guideline.directory),
        "conversion_warnings": metadata.get("warnings", []),
    }
    atomic_json(
        output / "sources.json",
        {
            **source_info,
            "pages": [
                {
                    "page_id": p.id,
                    "pdf_page": p.number,
                    "label": p.label,
                    "markdown_path": p.path,
                    "markdown_sha256": p.sha256,
                }
                for p in guideline.pages.values()
            ],
        },
    )
    client = Client(llm, output / "checkpoints")
    status = {
        "status": "running",
        "stage": "extract",
        "updated_utc": now(),
        "notice": NOTICE,
    }
    atomic_json(output / "status.json", status)
    try:
        chunks = list(packets(guideline, packet_chars, packet_pages))
        jobs = [(f"extract-{i:04d}", p) for i, p in enumerate(chunks, 1)]
        owners = page_owners(
            {key: [p.id for p in pages] for key, pages in jobs},
            [p.id for p in guideline.primary],
        )
        atomic_json(
            output / "extraction_ownership.json",
            {"version": OWNERSHIP_VERSION, "page_owners": owners, "branches": []},
        )
        previous_packets = (
            read_json(output / "extraction.json").get("packets", {})
            if (output / "extraction.json").exists()
            else {}
        )
        accepted_extractions = {}
        for path in (output / "checkpoints").glob("*/accepted.json"):
            saved = read_json(path)
            if saved["job"].startswith("extract-") and "candidates" in saved["result"]:
                if saved["result_sha256"] != digest(saved["result"]):
                    raise ValueError("Corrupted extraction checkpoint")
                accepted_extractions[saved["job"]] = saved

        def extract(key, primary):
            ids = [p.id for p in primary]
            previous = previous_packets.get(key)
            accepted = accepted_extractions.get(key)
            if (
                previous is not None
                and accepted is not None
                and accepted.get("json_normalization_version")
                == JSON_NORMALIZATION_VERSION
                and previous["primary_page_ids"] == ids
                and digest(previous["result"]) == accepted["result_sha256"]
            ):
                # After a parser upgrade, rebuild stale packet metadata through
                # the exact request/cache path rather than reusing old content.
                supplied = {p: guideline.pages[p] for p in previous["context_page_ids"]}
                if (
                    set(supplied) & set(previous["omitted_page_ids"])
                    or set(supplied) | set(previous["omitted_page_ids"])
                    != {p.id for p in guideline.primary}
                    or previous["reserved_output_tokens"] != llm.max_tokens
                    or previous["prompt_tokens"] > client.prompt_budget
                ):
                    raise ValueError(
                        "Saved extraction context differs from source or token budget"
                    )
                validate_extraction(previous["result"], supplied, ids)
                for candidate in previous["result"]["candidates"]:
                    validate_decision_fields(candidate, metadata["title"])
                return previous
            payload = {
                "guideline": metadata["title"],
                "version": metadata["version"],
                "primary_page_ids": ids,
            }
            context, omitted, messages = pack_messages(
                client,
                guideline,
                ids,
                " ".join(p.label for p in primary),
                prompts.EXTRACT_TASK,
                payload,
                EXTRACTION,
                context_chars,
                ids,
            )
            supplied = {p.id: p for p in context}

            def validate(v):
                validate_shape(v, EXTRACTION)
                validate_decision_field_batch(v["candidates"], metadata["title"])
                validate_extraction(v, supplied, ids)

            value = client.complete(
                key,
                messages,
                EXTRACTION,
                validate,
                repair_handler=lambda v, error: repair_response(
                    client,
                    guideline,
                    supplied,
                    ids,
                    key,
                    v,
                    error,
                    validate,
                    context_chars,
                ),
            )
            return {
                "result": value,
                "primary_page_ids": ids,
                "context_page_ids": list(supplied),
                "omitted_page_ids": omitted,
                "prompt_tokens": client.count_tokens(messages),
                "reserved_output_tokens": llm.max_tokens,
            }

        notify(
            f"{guideline.disease}: extracting {len(chunks)} packets from {len(guideline.primary)} pages"
        )
        extracted, failures = run_jobs(jobs, workers, extract)
        atomic_json(
            output / "extraction.json", {"packets": extracted, "failures": failures}
        )
        if failures:
            raise RuntimeError(
                f"{len(failures)} extraction packets failed; rerun to resume successful checkpoints"
            )
        atomic_json(
            output / "extraction_ownership.json",
            branch_ledger(extracted, [p.id for p in guideline.primary]),
        )
        candidates, coverage, uncertainties = [], [], []
        for key, packet in sorted(extracted.items()):
            result = packet["result"]
            coverage.extend(result["page_coverage"])
            uncertainties.extend(result["uncertainties"])
            for i, candidate in enumerate(result["candidates"], 1):
                candidates.append({"candidate_id": f"{key}-c{i:03d}", **candidate})
        atomic_json(output / "candidates.json", candidates)
        atomic_json(
            output / "coverage.json",
            {
                "page_coverage": coverage,
                "excluded_front_matter": [
                    p.id for p in guideline.pages.values() if p not in guideline.primary
                ],
                "uncertainties": uncertainties,
            },
        )
        if not candidates:
            raise RuntimeError(
                "No source-supported disease states extracted; inspect page coverage"
            )
        status.update(
            stage="canonicalize", candidates=len(candidates), updated_utc=now()
        )
        atomic_json(output / "status.json", status)
        notify(f"{guideline.disease}: canonicalizing {len(candidates)} candidates")
        canonical = consolidate(
            client,
            guideline,
            candidates,
            output,
            workers,
            run_jobs,
            notify,
            context_chars,
        )
        atomic_json(output / "canonical_groups.json", canonical)
        status.update(
            stage="details", paradigms=len(canonical["groups"]), updated_utc=now()
        )
        atomic_json(output / "status.json", status)

        def detail(key, group):
            context, omitted, messages = build_detail_call(
                client, guideline, group, candidates, context_chars
            )
            supplied = {p.id: p for p in context}

            def validate(v):
                validate_shape(v, QUOTED_DETAIL)
                validate_decision_fields(v, metadata["title"])
                if v["space"] != group["space"]:
                    raise ValueError(
                        "Final detail space must exactly equal the requested canonical space"
                    )
                materialize_quoted_state(v, supplied)

            value = client.complete(
                key,
                messages,
                QUOTED_DETAIL,
                validate,
                repair_handler=lambda v, error: repair_quoted_response(
                    client, guideline, supplied, key, v, error, validate,
                    output=output, notify=notify,
                ),
            )
            # Stable within a source edition and canonical definition, independent of completion order.
            pid = f"nccn-{guideline.disease}-{digest({'pdf': metadata['source_sha256'], 'space': group['space']})[:16]}"
            return {
                "paradigm_id": pid,
                "space_trial_id": pid,
                "trial_id": f"guideline:{guideline.disease}:{metadata['version']}",
                "clinical_space_summary": format_space(group["space"]),
                **materialize_quoted_state(value, supplied),
                "source_batch_numbers": group["source_batch_numbers"],
                "source": source_info,
                "context_page_ids": list(supplied),
                "omitted_source_page_ids": omitted,
                "context_packing": {
                    "prompt_tokens": client.count_tokens(messages),
                    "context_window": llm.context_window,
                    "reserved_output_tokens": llm.max_tokens,
                    "safety_tokens": llm.safety_tokens,
                    "tokenizer_mode": llm.tokenizer_mode,
                },
                "review_required": True,
                "notice": NOTICE,
            }

        notify(
            f"{guideline.disease}: populating {len(canonical['groups'])} canonical states"
        )
        details, failures = run_jobs(
            [
                (f"detail-{i:04d}", group)
                for i, group in enumerate(canonical["groups"], 1)
            ],
            workers,
            detail,
        )
        atomic_json(output / "details.json", {"records": details, "failures": failures})
        if failures:
            raise RuntimeError(
                f"{len(failures)} detail jobs failed; rerun to resume successful checkpoints"
            )
        rows = [v for _, v in sorted(details.items())]
        for number, row in enumerate(rows, 1):
            row["clinical_space_number"] = number
            row["general_exclusion_criteria"] = "NA"
        if len({r["paradigm_id"] for r in rows}) != len(rows):
            raise RuntimeError(
                "Duplicate canonical definitions; inspect canonical_groups.json before export"
            )
        export(output, rows)
        status.update(
            status="running" if audit_pending else "complete",
            stage="audit" if audit_pending else "export",
            updated_utc=now(),
            paradigms=len(rows),
            candidates=len(candidates),
            source_pages=len(guideline.pages),
            extracted_pages=len(coverage),
            canonicalization_uncertainties=canonical["uncertainties"],
            uncertain_pages=[
                r["page_id"] for r in coverage if r["disposition"] == "uncertain"
            ],
            clinical_completeness="not_established",
            review_required=True,
        )
        atomic_json(output / "status.json", status)
        notify(f"{guideline.disease}: exported {len(rows)} canonical spaces")
        return status
    except (Exception, KeyboardInterrupt) as exc:
        status.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            updated_utc=now(),
            error=f"{type(exc).__name__}: {exc}",
        )
        atomic_json(output / "status.json", status)
        raise


def export(output, rows):
    atomic_text(
        output / "paradigms.jsonl",
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
    )
    stream = io.StringIO(newline="")
    fields = [
        "space_trial_id",
        "trial_id",
        "clinical_space_number",
        "clinical_space_summary",
        "general_exclusion_criteria",
        "paradigm_id",
    ]
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(output / "trial_spaces.csv", stream.getvalue())
    lines = ["# Extracted disease-state paradigms", "", NOTICE, ""]
    for row in rows:
        lines.extend(
            [
                f"## {row['name']}",
                "",
                f"`{row['paradigm_id']}`",
                "",
                row["clinical_space_summary"],
                "",
            ]
        )
        if row.get("citation_review", {}).get("issues"):
            lines.extend(["**Population source support unresolved:**", "",
                          *[f"- {s}" for s in row["citation_review"]["issues"]], ""])
        for key, title in (
            ("diagnostic_workup", "Diagnostic / workup considerations"),
            ("treatment_options", "Treatment / management options"),
        ):
            lines.extend([f"### {title}", ""])
            for item in row[key]:
                cites = ", ".join(sorted({e["page_id"] for e in item["evidence"]}))
                lines.append(
                    f"- **{item['name']}** — {item['conditions']} Category: {item['category']}. Sources: {cites}."
                )
                for issue in item.get("citation_review", {}).get("issues", []):
                    lines.append(f"  - **Source support unresolved:** {issue}")
            if not row[key]:
                lines.append("No supported items extracted; see uncertainties.")
            lines.append("")
        if row["uncertainties"]:
            lines.extend(
                ["### Uncertainties", "", *[f"- {s}" for s in row["uncertainties"]], ""]
            )
        if row["omitted_source_page_ids"]:
            lines.extend(
                [
                    f"Context omitted {len(row['omitted_source_page_ids'])} pages; IDs are recorded in JSONL.",
                    "",
                ]
            )
        lines.extend(["### Source pages", ""])
        citations = {e["page_id"] for e in row["evidence"]}
        for key in ("diagnostic_workup", "treatment_options"):
            citations.update(
                e["page_id"] for item in row[key] for e in item["evidence"]
            )
        import os

        source = row["source"]
        manifest = read_json(Path(source["input_directory"]) / "manifest.json")
        for p in manifest["pages"]:
            if f"p{p['pdf_page']:04d}" in citations:
                path = Path(source["input_directory"]) / p["path"]
                link = os.path.relpath(path, output).replace(" ", "%20")
                pdf = Path(source["input_directory"]).parents[1] / source["source_pdf"]
                pdf_link = os.path.relpath(pdf, output).replace(" ", "%20")
                lines.append(
                    f"- [p{p['pdf_page']:04d}: {p['label']}]({link}) · [PDF page {p['pdf_page']}]({pdf_link}#page={p['pdf_page']})"
                )
        lines.append("")
    atomic_text(output / "report.md", "\n".join(lines))
