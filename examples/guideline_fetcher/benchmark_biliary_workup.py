"""Compare both app workup methods on the provenance-verified synthetic biliary case.

Catalogs, source notes, annotations, and outputs remain outside this repository.
No arbitrary patient input or raw provider responses/reasoning are accepted/logged.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
from pathlib import Path
import re
import statistics
from threading import Lock, local
import time


from matchminer_ai.llm.structured import EndpointError, StructuredClient  # noqa: E402
from matchminer_ai.llm import structured  # noqa: E402
from matchminer_ai.trials import load_guideline_catalog  # noqa: E402
from matchminer_ai.apps.guideline_fetcher.services import Inference, synthetic_notes  # noqa: E402
from matchminer_ai.apps.guideline_fetcher.settings import GEMMA_PROFILE, Settings, data_path  # noqa: E402


def synthetic_source():
    """The non-PHI synthetic notes the benchmark's biliary notes were drawn from."""
    return data_path("no_phi/all_synthetic_notes.parquet")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs(settings, catalog, space_id, annotations):
    """Reject stale labels or notes before sending synthetic content to Vertex."""
    note_path = settings.biliary_synthetic_notes
    provenance = json.loads(note_path.with_suffix(".provenance.json").read_text())
    if provenance.get("source") != str(synthetic_source()):
        raise ValueError("Benchmark requires the known non-PHI synthetic source provenance.")
    if provenance.get("csv_sha256") != sha256(note_path):
        raise ValueError("Synthetic notes do not match their provenance hash.")
    source = synthetic_notes(settings, "biliary")
    if len(source["notes"]) != provenance["note_count"]:
        raise ValueError("Synthetic note count does not match provenance.")
    gold = json.loads(annotations.read_text())
    if gold.get("notes_sha256") != sha256(note_path):
        raise ValueError("Annotations describe different synthetic notes.")
    if gold.get("catalog_sha256") != sha256(catalog / "paradigms.jsonl"):
        raise ValueError("Annotations describe a different guideline catalog snapshot.")
    frame = load_guideline_catalog(catalog)
    selected = frame.loc[frame.space_trial_id.eq(space_id)]
    if len(selected) != 1 or gold.get("space_trial_id") != space_id:
        raise ValueError("Select exactly the annotated guideline space.")
    space = selected.iloc[0].to_dict()
    items = space["diagnostic_workup"]
    if len(items) != len(gold["items"]):
        raise ValueError("Workup items and annotations differ.")
    for item, label in zip(items, gold["items"], strict=True):
        if item["name"] != label["name"] or not label["accepted_statuses"]:
            raise ValueError("Annotation names/statuses do not match the workup menu.")
    return source, space, gold


def score(review, gold, notes):
    rows = review["assessments"]
    matches, scored, details = 0, 0, []
    evidence_count, invalid_evidence = 0, 0
    for row, label in zip(rows, gold["items"], strict=True):
        accepted = row["status"] in label["accepted_statuses"]
        included = label.get("score", True)
        scored += included
        matches += included and accepted
        details.append(
            {
                "name": label["name"],
                "status": row["status"],
                "applicability": row["applicability"],
                "accepted_statuses": label["accepted_statuses"],
                "scored": included,
                "accepted": accepted,
                "rationale": label["rationale"],
            }
        )
        for evidence in row["evidence"]:
            evidence_count += 1
            number = evidence.get("note_number")
            valid = type(number) is int and 1 <= number <= len(notes)
            if valid:
                original = notes[number - 1]
                original_date = datetime.fromisoformat(original["note_date"])
                if original_date.tzinfo is None:
                    original_date = original_date.replace(tzinfo=timezone.utc)
                observed_date = datetime.fromisoformat(evidence["note_date"])
                valid = (
                    evidence["quote"] in original["note_text"] and observed_date == original_date
                )
            invalid_evidence += not valid
    return {
        "status_matches": matches,
        "scored_items": scored,
        "excluded_ambiguous_items": len(rows) - scored,
        "items": details,
        "evidence_items": evidence_count,
        "invalid_source_quotes_or_dates": invalid_evidence,
    }


def reused_summary(path, gold):
    saved = json.loads(path.read_text())
    if (
        saved.get("synthetic_only") is not True
        or saved.get("model") != GEMMA_PROFILE["model"]
        or type(saved.get("thinking")) is not bool
        or any(
            saved.get(k) != gold[k] for k in ("notes_sha256", "catalog_sha256", "space_trial_id")
        )
        or not isinstance(saved.get("summary"), str)
        or not saved["summary"].strip()
        or len(saved["summary"]) > 100_000
    ):
        raise ValueError("Reuse only a summary from the same verified synthetic Gemma case.")
    return saved["summary"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--space-id", required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--reuse-summary", type=Path)
    parser.add_argument("--first-method", choices=["full_notes", "agentic"], default="full_notes")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("Use at least one paired repeat.")
    multiprocessing.set_start_method("spawn", force=True)
    settings = Settings(llm_backend="gemma_vertex", device="cpu")
    source, space, gold = load_inputs(settings, args.catalog, args.space_id, args.annotations)
    worker = Inference(settings)
    from transformers import AutoTokenizer

    # Both preparation and tokenizer loading are excluded from review latency.
    AutoTokenizer.from_pretrained(GEMMA_PROFILE["tokenizer"], use_fast=True)
    if args.reuse_summary:
        summary = reused_summary(args.reuse_summary, gold)
        preparation_seconds = 0
        print("Reusing the verified synthetic patient summary", flush=True)
    else:
        print("Preparing the common synthetic patient summary", flush=True)
        prepared = time.perf_counter()
        summary = worker.summarize(
            source["text"], source["notes"], lambda stage: print(stage, flush=True)
        )["summary"]
        preparation_seconds = time.perf_counter() - prepared
    runtime = settings.inference_config().remote
    output = {
        "synthetic_only": True,
        "model": GEMMA_PROFILE["model"],
        "endpoint": GEMMA_PROFILE["endpoint"],
        "thinking": False,
        "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 64},
        "context_window": 262144,
        "reserved_output_tokens": 65536,
        "note_count": len(source["notes"]),
        "input_chars": len(source["text"]),
        "notes_sha256": gold["notes_sha256"],
        "catalog_sha256": gold["catalog_sha256"],
        "space_trial_id": args.space_id,
        "space_name": space["name"],
        "workup_items": len(space["diagnostic_workup"]),
        "annotations_sha256": sha256(args.annotations),
        "accuracy_rubric": gold["rubric"],
        "summary": summary,
        "summary_preparation_seconds": round(preparation_seconds, 3),
        "summary_reused": bool(args.reuse_summary),
        "dispatch_policy": {
            key: runtime[key]
            for key in (
                "request_start_interval_seconds",
                "capacity_retry_initial_seconds",
                "capacity_retry_max_seconds",
            )
        },
        "runs": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    original_http = StructuredClient._http
    original_urlopen = structured.urlopen
    lock = Lock()
    thread = local()
    measurements = []
    active, peak = 0, 0
    network_active, network_peak = 0, 0
    run_started = 0.0

    def tracked_urlopen(request, *args, **kwargs):
        if not request.full_url.endswith("/chat/completions"):
            return original_urlopen(request, *args, **kwargs)

        @contextmanager
        def response():
            nonlocal network_active, network_peak
            dispatched = time.perf_counter()
            with lock:
                network_active += 1
                network_peak = max(network_peak, network_active)
                entry = thread.entry
                entry["pre_dispatch_seconds"] = round(dispatched - thread.started, 4)
                entry["dispatch_offset_seconds"] = round(dispatched - run_started, 4)
            try:
                with original_urlopen(request, *args, **kwargs) as result:
                    yield result
            finally:
                with lock:
                    entry["network_seconds"] = round(time.perf_counter() - dispatched, 4)
                    network_active -= 1

        return response()

    def measured(self, endpoint, body=None, **kwargs):
        nonlocal active, peak
        if endpoint != "/chat/completions":
            return original_http(self, endpoint, body, **kwargs)
        entry = {
            "model": body.get("model"),
            "usage": {},
            "thinking": body.get("chat_template_kwargs", {}).get("enable_thinking"),
            "sampling": {key: body.get(key) for key in ("temperature", "top_p", "top_k")},
            "reasoning_replayed": any(
                "reasoning_content" in message for message in body["messages"]
            ),
        }
        started = time.perf_counter()
        thread.entry, thread.started = entry, started
        with lock:
            active += 1
            peak = max(peak, active)
            measurements.append(entry)
        try:
            response = original_http(self, endpoint, body, **kwargs)
            entry.update(
                usage=response.get("usage") or {},
                finish_reason=response.get("choices", [{}])[0].get("finish_reason"),
            )
            return response
        except EndpointError as exc:
            entry["transport_error"] = True
            match = re.fullmatch(r"Endpoint HTTP (\d{3}) for /chat/completions", str(exc))
            if match:
                entry["http_status"] = int(match[1])
            raise
        finally:
            with lock:
                entry["seconds"] = round(time.perf_counter() - started, 4)
                active -= 1

    StructuredClient._http = measured
    structured.urlopen = tracked_urlopen
    try:
        for repeat in range(args.repeats):
            methods = [
                args.first_method,
                "agentic" if args.first_method == "full_notes" else "full_notes",
            ]
            if repeat % 2:
                methods.reverse()
            for method in methods:
                measurements.clear()
                peak = network_peak = 0
                print(f"Starting {method}, pair {repeat + 1}", flush=True)
                started = time.perf_counter()
                run_started = started
                review = worker.review_workup(
                    source["text"],
                    source["notes"],
                    space,
                    lambda stage: print(stage, flush=True),
                    method=method,
                    patient_summary=summary,
                )
                accuracy = score(review, gold, source["notes"])
                usage_complete = all(
                    type(m["usage"].get(key)) is int
                    for m in measurements
                    for key in ("prompt_tokens", "completion_tokens")
                )
                run = {
                    "method": method,
                    "repeat": repeat + 1,
                    "wall_seconds": round(time.perf_counter() - started, 3),
                    "requests": len(measurements),
                    "peak_inflight_requests": network_peak,
                    "peak_client_requests_including_queued": peak,
                    "prompt_tokens": sum(m["usage"]["prompt_tokens"] for m in measurements)
                    if usage_complete
                    else None,
                    "completion_tokens": sum(m["usage"]["completion_tokens"] for m in measurements)
                    if usage_complete
                    else None,
                    "request_metrics": measurements.copy(),
                    "accuracy": accuracy,
                    "review": review,
                    "http_429": sum(m.get("http_status") == 429 for m in measurements),
                }
                assert all(
                    m["model"] == GEMMA_PROFILE["model"]
                    and m["thinking"] is output["thinking"]
                    and not m["reasoning_replayed"]
                    for m in measurements
                )
                assert all(m["sampling"] == output["sampling"] for m in measurements)
                output["runs"].append(run)
                args.output.write_text(json.dumps(output, indent=2))
                print(
                    {
                        "method": method,
                        "repeat": repeat + 1,
                        "seconds": run["wall_seconds"],
                        "requests": run["requests"],
                        "http_429": run["http_429"],
                        "accuracy": f"{accuracy['status_matches']}/{accuracy['scored_items']}",
                        "invalid_evidence": accuracy["invalid_source_quotes_or_dates"],
                    },
                    flush=True,
                )
    finally:
        StructuredClient._http = original_http
        structured.urlopen = original_urlopen
    output["median_seconds"] = {
        method: statistics.median(
            run["wall_seconds"] for run in output["runs"] if run["method"] == method
        )
        for method in ("full_notes", "agentic")
    }
    output["full_over_agentic_wall_ratio"] = (
        output["median_seconds"]["full_notes"] / output["median_seconds"]["agentic"]
    )
    args.output.write_text(json.dumps(output, indent=2))
    print(
        {
            "median_seconds": output["median_seconds"],
            "full_over_agentic_wall_ratio": output["full_over_agentic_wall_ratio"],
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
