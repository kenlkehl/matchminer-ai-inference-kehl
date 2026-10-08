"""Snapshot complete NCI PDQ summaries and extract resumable TrialSpace catalogs.

All sources and outputs belong outside the code repository. Ctrl-C, SIGTERM or
an output-directory STOP file interrupt the batch, preserving atomic checkpoints.
Remove STOP before resuming. Endpoint changes require --allow-endpoint-change and
an identical advertised model/context/settings; different models need a new run.
"""

import argparse
import json
import os
import signal
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from matchminer_ai import load_default_preset
from matchminer_ai._storage import atomic_json, digest, directory_lock, read_json
from matchminer_ai.cancellation import (
    CancellationToken, InferenceCancelled, cancellation_scope, submit_cancellable,
)
from matchminer_ai.trials import prepare_pdq_library, summarize_pdq
from matchminer_ai.trials.guidelines import _output_directory


def run_batch(inventory, source, output, config, *, disease_workers=2,
              allow_endpoint_change=False, token=None):
    """A single batch owner with a shared endpoint prompt cap and durable status."""
    output = _output_directory(source, output)
    output.mkdir(parents=True, exist_ok=True)
    token = token or CancellationToken()
    with directory_lock(output), cancellation_scope(token):
        path = output / "collection.json"
        identity = {
            "source": str(Path(source).resolve()),
            "diseases": inventory.disease.tolist(),
            "manifest_sha256": digest((Path(source) / "markdown" / "manifest.json").read_bytes()),
        }
        previous = read_json(path) if path.exists() else {}
        if previous and previous["identity"] != identity:
            raise ValueError("PDQ collection sources changed; use a new output directory")
        state = {
            "identity": identity, "status": "running", "pid": os.getpid(),
            "endpoint": config.remote["server_urls"][0],
            "concurrency": config.remote["max_concurrent_requests"],
            "disease_workers": disease_workers,
            "diseases": {d: previous.get("diseases", {}).get(d, {"status": "pending"})
                         for d in inventory.disease},
        }
        lock = threading.Lock()

        def update(disease=None, **values):
            with lock:
                (state["diseases"][disease] if disease else state).update(values)
                state["updated_utc"] = datetime.now(timezone.utc).isoformat()
                atomic_json(path, state)
                print(json.dumps({"disease": disease, **values}), flush=True)

        def run(disease):
            update(disease, status="running", error=None)
            try:
                frame, metadata = summarize_pdq(
                    source, disease=disease, output_dir=output / disease,
                    config=config, return_metadata=True,
                    allow_endpoint_change=allow_endpoint_change,
                    progress_callback=lambda message: update(disease, progress=message),
                )
                update(disease, status="complete", spaces=len(frame),
                       audit=metadata["validation"]["status"])
            except InferenceCancelled:
                update(disease, status="interrupted")
                raise
            except Exception as exc:
                update(disease, status="failed", error=f"{type(exc).__name__}: {exc}")

        update()
        try:
            with ThreadPoolExecutor(max_workers=disease_workers) as pool:
                futures = [submit_cancellable(pool, run, d) for d in inventory.disease]
                for future in as_completed(futures):
                    future.result()
        except InferenceCancelled:
            update(status="interrupted")
            return 130
        failed = [d for d, row in state["diseases"].items() if row["status"] != "complete"]
        update(status="failed" if failed else "complete")
        return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summaries", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--endpoint")
    parser.add_argument("--model", help="Exact served ID; omit for endpoint discovery")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--disease-workers", type=int, default=2)
    parser.add_argument("--request-timeout", type=float, default=7200)
    parser.add_argument("--allow-endpoint-change", action="store_true")
    args = parser.parse_args()
    if min(args.concurrency, args.disease_workers, args.request_timeout) <= 0:
        parser.error("Concurrency, disease workers and timeout must be positive")
    if not args.download_only and (not args.endpoint or not args.output_dir):
        parser.error("Extraction requires an explicit --endpoint and --output-dir")
    summaries = json.loads(args.summaries.read_text())
    inventory = prepare_pdq_library(summaries, output_dir=args.source_dir, progress_callback=print)
    print(inventory.to_string(index=False), flush=True)
    if args.download_only:
        return 0
    config = load_default_preset()
    config.remote.update(
        enabled=True, server_urls=[args.endpoint],
        max_concurrent_requests=args.concurrency, request_timeout=args.request_timeout,
    )
    config.guideline["remote"]["model_name"] = args.model
    token = CancellationToken()
    done = threading.Event()
    stop_file = args.output_dir / "STOP"
    if stop_file.exists():
        parser.error(f"Remove {stop_file} to resume this batch")

    def watch_stop():
        while not done.wait(1):
            if stop_file.exists():
                token.cancel()
                return

    previous = {sig: signal.signal(sig, lambda *_: token.cancel())
                for sig in (signal.SIGINT, signal.SIGTERM)}
    watcher = threading.Thread(target=watch_stop, daemon=True)
    watcher.start()
    try:
        return run_batch(
            inventory, args.source_dir, args.output_dir, config,
            disease_workers=args.disease_workers,
            allow_endpoint_change=args.allow_endpoint_change, token=token,
        )
    except InferenceCancelled:
        return 130
    finally:
        done.set()
        watcher.join()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
