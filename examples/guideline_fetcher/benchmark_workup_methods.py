"""Compare frontend workup methods on fabricated records using Gemini or vLLM.

Run from guideline-fetcher with the inference package available:
  python examples/benchmark_workup_methods.py --output /tmp/workup-benchmark.json
No real patient input is accepted. No raw model responses or reasoning are logged.
"""

import argparse
from collections import defaultdict
import json
import re
from pathlib import Path
import statistics
from threading import Lock
import time
from urllib.request import urlopen


from matchminer_ai.llm.structured import StructuredClient  # noqa: E402
from matchminer_ai.apps.guideline_fetcher.services import Inference  # noqa: E402
from matchminer_ai.apps.guideline_fetcher.settings import GEMINI_PROFILE, Settings, normalize_endpoint  # noqa: E402


ITEMS = [
    {"name": "CT chest", "conditions": "For staging"},
    {"name": "MRI brain", "conditions": "If clinically indicated"},
    {"name": "Molecular tumor panel", "conditions": "If clinically indicated"},
    {"name": "ECG", "conditions": "If clinically indicated"},
    {"name": "Complete blood count (CBC)", "conditions": "For baseline assessment"},
    {"name": "LDH blood test", "conditions": "If clinically indicated"},
    {"name": "Echocardiogram", "conditions": "If clinically indicated"},
    {"name": "Tumor biopsy with pathology", "conditions": "For diagnosis"},
    {"name": "Serum creatinine", "conditions": "For baseline assessment"},
    {"name": "Liver ultrasound", "conditions": "If clinically indicated"},
    {"name": "Audiogram", "conditions": "If clinically indicated"},
    {"name": "Bone scan", "conditions": "If clinically indicated"},
]
EXPECTED = ["completed", "not_done", "planned", "not_documented",
            "completed", "planned", "not_done", "completed",
            "completed", "not_done", "not_documented", "not_documented"]
POPULATION = "Fabricated adult solid tumor staging population"


def fabricated_notes(target_chars, components=4):
    events = [
        "Synthetic adult with a newly diagnosed solid tumor undergoing staging. CT chest and MRI brain ordered.",
        "CT chest performed. Radiology report: no thoracic metastases.",
        "MRI brain canceled by the patient and explicitly never performed.",
        "Molecular tumor panel ordered; sample submission scheduled for next week.",
    ]
    if components == 12:
        additional = [
            "CBC was performed; laboratory report available. Tumor biopsy performed; pathology confirmed carcinoma.",
            "Serum creatinine was measured; laboratory report available. LDH blood test ordered for next week.",
            "Echocardiogram was declined and was not performed.",
            "Liver ultrasound was canceled and never performed.",
        ]
        events = [a + " " + b for a, b in zip(events, additional, strict=True)]
    notes = []
    for i, event in enumerate(events):
        lines, size, j = [], 0, 0
        while size < target_chars // 4:
            line = f"Fictional administrative entry {i}-{j:05d}: scheduling desk updated its routine contact log.\n"
            lines.append(line)
            size += len(line)
            j += 1
        lines.insert(len(lines) // 2, f"\nFabricated clinical event: {event}\n")
        notes.append({"note_text": "".join(lines), "note_date": f"2026-0{i + 1}-01"})
    return notes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[128000, 512000])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--components", type=int, choices=[4, 12], default=4)
    parser.add_argument("--methods", nargs="+",
                        choices=["full_notes", "agentic", "agentic_4", "agentic_low", "agentic_off"],
                        default=["full_notes", "agentic"])
    parser.add_argument("--endpoint", help="Use an explicit vLLM endpoint for agentic benchmarks.")
    parser.add_argument("--model", help="Select one of the models advertised by --endpoint.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or any(s < 1000 for s in args.sizes):
        parser.error("Use positive repeats and sizes of at least 1000 characters.")
    if args.endpoint:
        if "full_notes" in args.methods or "agentic_low" in args.methods:
            parser.error("Local endpoints support agentic, agentic_4 and agentic_off here.")
        endpoint = normalize_endpoint(args.endpoint)
        with urlopen(endpoint + "/models", timeout=30) as response:
            models = json.load(response)["data"]
        selected = [m for m in models if m["id"] == args.model] if args.model else models
        if len(selected) != 1:
            parser.error("Choose exactly one advertised model with --model.")
        model = selected[0]["id"]
        context_window = selected[0].get("max_model_len")
        if type(context_window) is not int or context_window <= 65536:
            parser.error("Endpoint must advertise a context larger than the 65536 output reserve.")
        if "agentic_off" in args.methods and not re.search(r"gemma[-_ ]?4", model.lower()):
            parser.error("agentic_off currently verifies the Gemma 4 thinking switch only.")
        settings = Settings(llm_backend="local", endpoint=endpoint, model=model,
                            tokenizer=model, device="cpu")
    else:
        if args.model or "agentic_off" in args.methods:
            parser.error("--model and agentic_off require an explicit --endpoint.")
        settings = Settings(llm_backend="gemini_vertex", device="cpu")
        endpoint, model = GEMINI_PROFILE["endpoint"], GEMINI_PROFILE["model"]
        context_window = 1048576
    if "full_notes" in args.methods:
        from transformers import AutoTokenizer

        # Exclude initial tokenizer loading from the first timed method.
        print("Preparing the full-note workflow's existing chunk tokenizer", flush=True)
        AutoTokenizer.from_pretrained(GEMINI_PROFILE["tokenizer"], use_fast=True)
    worker = Inference(settings)

    def assessment_effort(method):
        return None if args.endpoint else ("low" if method == "agentic_low" else "high")

    items, expected = ITEMS[:args.components], EXPECTED[:args.components]
    runs = []
    summary = {"model": model, "endpoint": endpoint, "context_window": context_window,
               "reserved_output_tokens": 65536, "tokenizer_mode": "bytes",
               "assessment_reasoning_efforts": {
                   method: assessment_effort(method)
                   for method in args.methods
               },
               "thinking": {method: method != "agentic_off" for method in args.methods},
               "synthetic_only": True, "items_per_record": len(items),
               "full_note_recommendation_batch_size": 6,
               "agentic_max_parallel_questions": len(items), "agentic_max_calls_per_question": 16,
               "agentic_search_reasoning_effort": None if args.endpoint else "low",
               "runs": runs, "comparisons": []}
    original = StructuredClient._http
    lock = Lock()
    measurements = []
    active, peak = 0, 0

    def measured(self, endpoint, body=None, **kwargs):
        nonlocal active, peak
        if endpoint != "/chat/completions":
            return original(self, endpoint, body, **kwargs)
        started = time.perf_counter()
        entry = {"usage": {}, "finish_reason": None,
                 "reasoning_effort": (body or {}).get("reasoning_effort"),
                 "enable_thinking": (body or {}).get("chat_template_kwargs", {}).get("enable_thinking"),
                 "sampling": {k: (body or {}).get(k) for k in ("temperature", "top_p", "top_k")}}
        with lock:
            active += 1
            peak = max(peak, active)
            measurements.append(entry)
        try:
            response = original(self, endpoint, body, **kwargs)
            with lock:
                entry.update({
                    "usage": response.get("usage") or {},
                    "finish_reason": response.get("choices", [{}])[0].get("finish_reason"),
                })
            return response
        except Exception as exc:
            # Retain only a numeric HTTP status, never provider text or a body.
            status = re.search(r"Endpoint HTTP (\d{3}) for /chat/completions", str(exc))
            with lock:
                entry["error"] = f"HTTP {status[1]}" if status else "request_failed"
            raise
        finally:
            with lock:
                entry["seconds"] = time.perf_counter() - started
                active -= 1

    StructuredClient._http = measured
    try:
        for size in args.sizes:
            notes = fabricated_notes(size, args.components)
            text = "\n".join(n["note_text"] for n in notes)
            for repeat in range(args.repeats):
                # Reverse order on alternating pairs to reduce ordering effects.
                methods = args.methods if repeat % 2 == 0 else list(reversed(args.methods))
                for method in methods:
                    measurements.clear()
                    peak = 0
                    print(f"Starting {method}: {len(text):,} characters, pair {repeat + 1}", flush=True)
                    started = time.perf_counter()
                    if args.endpoint or method in {"agentic_4", "agentic_low"}:
                        import pandas as pd
                        from matchminer_ai.patients import review_patient_workup_with_note_search

                        config = worker.settings.inference_config()
                        if args.endpoint:
                            # Match the hosted benchmark's byte-based budgeting
                            # and output reserve; use the actual server context.
                            config.patient.update(context_window=context_window, tokenizer_mode="bytes")
                            config.patient["local"]["engine"]["max_model_len"] = context_window
                            config.patient["remote"]["request_params"]["max_tokens"] = 65536
                            template = config.patient["remote"]["extra_body"].setdefault("chat_template_kwargs", {})
                            template["enable_thinking"] = method != "agentic_off"
                        concurrency = 4 if method == "agentic_4" else len(items)
                        config.remote["max_concurrent_requests"] = concurrency
                        if method == "agentic_low":
                            # Benchmark-only override. LOW is Gemini 3.8's
                            # lowest supported effort, not reasoning disabled.
                            config.patient["reasoning_effort"] = "low"
                            config.patient["remote"]["request_params"]["reasoning_effort"] = "low"
                        result = review_patient_workup_with_note_search(
                            pd.DataFrame(notes), items, config=config, population_context=POPULATION,
                            max_parallel_questions=concurrency,
                            thinking="off" if method == "agentic_off" else "on",
                            progress_callback=lambda p: print(p, flush=True),
                        )
                    else:
                        result = worker.review_workup(
                            text, notes,
                            {"diagnostic_workup": items, "clinical_space_summary": POPULATION},
                            lambda p: print(p, flush=True), method=method,
                        )
                    elapsed = time.perf_counter() - started
                    statuses = [r["status"] for r in result["assessments"]]
                    complete_usage = all(
                        type(m["usage"].get(k)) is int for m in measurements
                        for k in ("prompt_tokens", "completion_tokens")
                    )
                    run = {
                        "method": method, "target_chars": size, "input_chars": len(text),
                        "assessment_reasoning_effort": assessment_effort(method),
                        "thinking": method != "agentic_off",
                        "evidence_selection": result["metadata"].get("evidence_selection"),
                        "repeat": repeat + 1, "wall_seconds": round(elapsed, 3),
                        "requests": len(measurements),
                        "peak_inflight_requests": peak,
                        "prompt_tokens": sum(m["usage"]["prompt_tokens"] for m in measurements) if complete_usage else None,
                        "completion_tokens": sum(m["usage"]["completion_tokens"] for m in measurements) if complete_usage else None,
                        "finish_reasons": [m["finish_reason"] for m in measurements],
                        "request_errors": [m["error"] for m in measurements if "error" in m],
                        "request_metrics": [
                            {"seconds": round(m["seconds"], 4),
                             "reasoning_effort": m["reasoning_effort"],
                             "enable_thinking": m["enable_thinking"],
                             "sampling": m["sampling"],
                             "reasoning_tokens": (m["usage"].get("completion_tokens_details") or {}).get("reasoning_tokens"),
                             "finish_reason": m["finish_reason"],
                             "error": m.get("error")}
                            for m in measurements
                        ],
                        "statuses": statuses, "expected_statuses": expected,
                        "expected_status_matches": sum(a == b for a, b in zip(statuses, expected, strict=True)),
                        "failed_items": sum(r["status"] == "error" for r in result["assessments"]),
                        "cell_counts": [r.get("metadata", {}).get("cells") for r in result["assessments"]],
                        "requests_per_question": [r.get("metadata", {}).get("requests") for r in result["assessments"]],
                        "initial_search_requests": [r.get("metadata", {}).get("initial_search_requests",
                            r.get("metadata", {}).get("requests")) for r in result["assessments"]],
                        "followup_review": [r.get("metadata", {}).get("followup_review", {})
                            for r in result["assessments"]],
                        "validation_errors": [r.get("metadata", {}).get("validation_errors", []) for r in result["assessments"]],
                        "cell_errors": [r.get("metadata", {}).get("cell_errors", []) for r in result["assessments"]],
                        "worker_startup_seconds": [r.get("metadata", {}).get("worker_startup_seconds") for r in result["assessments"]],
                        "cell_seconds": [r.get("metadata", {}).get("cell_seconds", []) for r in result["assessments"]],
                        "termination_reasons": [r.get("metadata", {}).get("termination_reason") for r in result["assessments"]],
                    }
                    if method.startswith("agentic"):
                        assert all(n <= 16 for n in run["initial_search_requests"])
                        assert all(audit.get("requests", 0) <= audit.get("max_calls", 0)
                                   for audit in run["followup_review"])
                        assert all(total == initial + audit.get("requests", 0)
                                   for total, initial, audit in zip(run["requests_per_question"],
                                       run["initial_search_requests"], run["followup_review"], strict=True))
                    if method == "agentic_low":
                        assert all(m["reasoning_effort"] == "low" for m in measurements)
                    if args.endpoint:
                        assert all(m["enable_thinking"] == (method != "agentic_off") for m in measurements)
                    runs.append(run)
                    args.output.write_text(json.dumps(summary, indent=2))
                    print(json.dumps(run), flush=True)
    finally:
        StructuredClient._http = original
    groups = defaultdict(dict)
    for size in args.sizes:
        for method in args.methods:
            selected = [r for r in runs if r["target_chars"] == size and r["method"] == method]
            groups[size][method] = statistics.median(r["wall_seconds"] for r in selected)
        comparison = {"target_chars": size, "median_seconds": groups[size]}
        if {"full_notes", "agentic"}.issubset(groups[size]):
            comparison["full_over_agentic_wall_ratio"] = groups[size]["full_notes"] / groups[size]["agentic"]
        if {"agentic_4", "agentic"}.issubset(groups[size]):
            comparison["four_over_all_wall_ratio"] = groups[size]["agentic_4"] / groups[size]["agentic"]
        if {"agentic", "agentic_low"}.issubset(groups[size]):
            comparison["high_over_low_wall_ratio"] = groups[size]["agentic"] / groups[size]["agentic_low"]
        if {"agentic", "agentic_off"}.issubset(groups[size]):
            comparison["on_over_off_wall_ratio"] = groups[size]["agentic"] / groups[size]["agentic_off"]
        summary["comparisons"].append(comparison)
    args.output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["comparisons"], indent=2), flush=True)


if __name__ == "__main__":
    main()
