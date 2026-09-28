"""Extract a local guideline collection with one concurrency cap per endpoint.

All sources, checkpoints, model responses and catalogs stay in external directories.
Completed catalogs can be supplied to skip already processed, identical editions.
Repeat the same invocation to resume checkpoints and retry failed diseases.
"""

import argparse
import copy
import json
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

from matchminer_ai import load_default_preset
from matchminer_ai._storage import atomic_json, digest
from matchminer_ai.llm.remote_inference import normalize_openai_base_url
from matchminer_ai.llm.structured import StructuredClient, StructuredConfig
from matchminer_ai.trials import (
    load_guideline_catalog,
    summarize_guidelines,
)
from matchminer_ai.trials.guidelines import _output_directory


def validate_endpoints(endpoints, model):
    """Check replica model/context compatibility before sending guideline content."""
    discovered = {}
    for endpoint in endpoints:
        client = StructuredClient(
            StructuredConfig(base_url=endpoint, model=model or "", timeout=30), None
        )
        info = client.discover()
        discovered[endpoint] = {
            "model": info["id"],
            "context_window": info.get("max_model_len"),
        }
    identities = {(v["model"], v["context_window"]) for v in discovered.values()}
    if len(identities) != 1 or any(
        type(v["context_window"]) is not int or v["context_window"] <= 102048
        for v in discovered.values()
    ):
        raise ValueError(
            "Additional endpoints must report the same model and context window."
        )
    return discovered


def assign_endpoints(diseases, output, endpoints, previous, new_endpoint=None):
    """Retain checkpoint owners; balance only unstarted work across replicas."""
    assigned = {}
    load = Counter({endpoint: 0 for endpoint in endpoints})
    for disease in diseases:
        saved = output / disease / "run_config.json"
        prior = previous.get(disease, {})
        if saved.exists():
            endpoint = normalize_openai_base_url(
                json.loads(saved.read_text())["llm"]["base_url"]
            )
            if endpoint not in endpoints:
                raise ValueError(
                    f"Existing checkpoints for {disease} require endpoint {endpoint}."
                )
        else:
            endpoint = new_endpoint or prior.get("endpoint")
        if endpoint in endpoints:
            assigned[disease] = endpoint
            if prior.get("status") != "complete":
                load[endpoint] += 1
    for disease in diseases:
        if disease not in assigned:
            endpoint = min(endpoints, key=lambda e: load[e])
            assigned[disease] = endpoint
            load[endpoint] += 1
    return assigned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--source-override", nargs=2, action="append", default=[],
        metavar=("DISEASE", "SOURCE"),
        help="Use another verified local library for one disease of the same source edition",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument(
        "--additional-endpoint",
        action="append",
        default=[],
        help="Same-model replica for unstarted diseases; repeat to add servers",
    )
    parser.add_argument(
        "--new-disease-endpoint",
        help="Route every catalog without checkpoints to this configured endpoint",
    )
    parser.add_argument("--model")
    parser.add_argument(
        "--concurrency", type=int, default=32, help="Prompt cap per endpoint"
    )
    parser.add_argument(
        "--disease-workers", type=int, default=8, help="Disease workers per endpoint"
    )
    parser.add_argument("--completed-catalog", type=Path, action="append", default=[])
    parser.add_argument("--disease", action="append", help="Optional explicit subset")
    parser.add_argument(
        "--defer-disease", action="append", default=[],
        help="Keep a disease in collection identity/status but leave it to another worker",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Resume only failed diseases from collection.json; write recovery.json",
    )
    args = parser.parse_args()
    if args.concurrency < 1 or args.disease_workers < 1:
        parser.error("Concurrency and disease workers must be positive")
    endpoints = list(
        dict.fromkeys(
            normalize_openai_base_url(e)
            for e in [args.endpoint, *args.additional_endpoint]
        )
    )
    new_endpoint = (
        normalize_openai_base_url(args.new_disease_endpoint)
        if args.new_disease_endpoint
        else None
    )
    if new_endpoint is not None and new_endpoint not in endpoints:
        parser.error(
            "--new-disease-endpoint must match --endpoint or --additional-endpoint"
        )
    endpoint_models = (
        validate_endpoints(endpoints, args.model) if len(endpoints) > 1 else {}
    )
    output = _output_directory(args.source, args.output_dir)
    config = load_default_preset()
    config.remote.update(
        enabled=True,
        server_urls=[args.endpoint],
        max_concurrent_requests=args.concurrency,
        request_timeout=7200,
    )
    config.guideline["remote"]["model_name"] = args.model
    from matchminer_ai.trials._guideline_sources import inventory

    _, records = inventory(args.source, check_presence=False)
    completed = {}
    for path in args.completed_catalog:
        frame, metadata = load_guideline_catalog(path, return_metadata=True)
        if not all(
            a["recorded_source_audit"] == "passed_export_hash_verified"
            for a in metadata["artifacts"]
        ):
            parser.error(f"A completed catalog requires a passing stored audit: {path}")
        for source in frame.source:
            completed[(source["disease"], source["source_sha256"])] = str(
                path.resolve()
            )
    if args.disease:
        unknown = set(args.disease) - {row["disease"] for row in records}
        if unknown:
            parser.error(f"Unknown diseases: {sorted(unknown)}")
        records = [row for row in records if row["disease"] in args.disease]
    # The immutable collection manifest identifies the current PDF edition.
    from matchminer_ai.trials._guideline_sources import library_root, safe_child

    manifest = json.loads((library_root(args.source) / "manifest.json").read_text())
    hashes = {row["folder"]: row.get("source_sha256") for row in manifest["guidelines"]}
    source_overrides = {}
    for disease, source in args.source_override:
        if disease not in hashes or disease in source_overrides:
            parser.error("Source overrides must name distinct diseases in the main collection")
        alternate = library_root(Path(source).resolve())
        alternate_manifest = json.loads((alternate / "manifest.json").read_text())
        entries = {row["folder"]: row for row in alternate_manifest["guidelines"]}
        if disease not in entries or entries[disease].get("source_sha256") != hashes[disease]:
            parser.error(f"Source override for {disease} must have the identical PDF edition/hash")
        alternate_disease = json.loads((safe_child(alternate, disease) / "manifest.json").read_text())
        if alternate_disease.get("source_sha256") != hashes[disease]:
            parser.error(f"Source override manifest for {disease} has a different PDF hash")
        _output_directory(alternate, args.output_dir)
        source_overrides[disease] = str(alternate)
    skipped = {
        row["disease"]: completed[(row["disease"], hashes[row["disease"]])]
        for row in records
        if (row["disease"], hashes[row["disease"]]) in completed
    }
    diseases = [row["disease"] for row in records if row["disease"] not in skipped]
    identity = {
        "source": str(args.source.resolve()),
        "endpoint": args.endpoint,
        "model": args.model,
        "diseases": diseases,
        "skipped": skipped,
        "manifest_sha256": digest(
            (library_root(args.source) / "manifest.json").read_bytes()
        ),
    }
    if source_overrides:
        identity["source_overrides"] = source_overrides
    saved = output / "collection.json"
    previous = json.loads(saved.read_text()) if saved.exists() else {}
    if previous and previous["identity"] != identity:
        parser.error("Collection inputs changed; use a new output directory")
    deferred = set(args.defer_disease)
    if deferred - set(diseases):
        parser.error(f"Deferred diseases are outside this collection: {sorted(deferred - set(diseases))}")
    if args.retry_failed:
        if not saved.exists():
            parser.error("--retry-failed requires an existing collection.json")
        diseases = [
            d for d in diseases if previous["diseases"][d]["status"] == "failed"
        ]
        # A live original runner owns collection.json. Keep its writes independent;
        # each retried disease still uses the same locked, checkpointed output.
        saved = output / "recovery.json"
    assignments = assign_endpoints(
        diseases,
        output,
        endpoints,
        previous.get("diseases", {}),
        new_endpoint=new_endpoint,
    )
    state = {
        "identity": identity,
        "status": "running",
        "concurrency": args.concurrency,
        "disease_workers": args.disease_workers,
        "endpoints": endpoints,
        "endpoint_models": endpoint_models,
        "new_disease_endpoint": new_endpoint,
        "deferred_diseases": sorted(deferred & set(diseases)),
        "diseases": {
            d: {
                **previous.get("diseases", {}).get(d, {"status": "pending"}),
                "endpoint": assignments[d],
            }
            for d in diseases
        },
    }
    lock = threading.Lock()

    def update(disease=None, **values):
        with lock:
            if disease:
                state["diseases"][disease].update(values)
            else:
                state.update(values)
            state["updated_at"] = datetime.now(timezone.utc).isoformat()
            atomic_json(saved, state)
            print(json.dumps({"disease": disease, **values}), flush=True)

    update()

    def run(disease):
        update(disease, status="running", endpoint=assignments[disease], error=None)
        try:
            disease_config = copy.deepcopy(config)
            disease_config.remote["server_urls"] = [assignments[disease]]
            frame = summarize_guidelines(
                source_overrides.get(disease, args.source),
                disease=disease,
                output_dir=output / disease,
                config=disease_config,
                progress_callback=lambda message: update(disease, progress=message),
            )
            update(disease, status="complete", spaces=len(frame))
        except Exception as exc:
            update(disease, status="failed", error=f"{type(exc).__name__}: {exc}")

    # Separate pools let a new server start immediately even while the original
    # pool is resuming long-running diseases. Request limits remain shared across
    # all disease clients targeting the same endpoint within this process.
    with ExitStack() as stack:
        executors = {
            e: stack.enter_context(ThreadPoolExecutor(max_workers=args.disease_workers))
            for e in endpoints
        }
        # A resume should repair known failures promptly instead of leaving them
        # behind every already-running disease's potentially long consolidation.
        priority = {"failed": 0, "running": 1, "pending": 2, "complete": 3}
        ordered = sorted(
            (d for d in diseases if d not in deferred),
            key=lambda d: priority.get(state["diseases"][d]["status"], 2),
        )
        for future in as_completed(
            [executors[assignments[d]].submit(run, d) for d in ordered]
        ):
            future.result()
    failed = [
        d for d, status in state["diseases"].items()
        if d not in deferred and status["status"] != "complete"
    ]
    outstanding_deferred = [
        d for d in state["deferred_diseases"]
        if state["diseases"][d]["status"] != "complete"
    ]
    update(
        status="failed" if failed else "deferred" if outstanding_deferred else "complete",
        failed_diseases=failed,
    )
    return bool(failed)


if __name__ == "__main__":
    raise SystemExit(main())
