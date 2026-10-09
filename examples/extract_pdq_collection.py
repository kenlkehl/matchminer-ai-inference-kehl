"""Run a resumable PDQ job plan with overlapping catalogs and bounded downloads.

Each job maps a disease slug to source_directory, output_directory and optional
urls. Ctrl-C, SIGTERM or output-dir/STOP cancels requests with checkpoints intact.
Use a new collection output directory when expanding a plan; per-disease paths
can stay unchanged. The endpoint is always an explicit operator choice.
"""

import argparse
import json
import signal
import threading
from pathlib import Path

from matchminer_ai import load_default_preset
from matchminer_ai.cancellation import CancellationToken, InferenceCancelled, cancellation_scope
from matchminer_ai.trials import summarize_pdq_collection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", help="Exact served ID; omit for endpoint discovery")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--source-workers", type=int, default=2)
    parser.add_argument("--disease-workers", type=int)
    parser.add_argument("--request-timeout", type=float, default=7200)
    parser.add_argument("--allow-endpoint-change", action="store_true")
    args = parser.parse_args()
    if min(args.concurrency, args.source_workers, args.request_timeout) <= 0 or (
        args.disease_workers is not None and args.disease_workers <= 0
    ):
        parser.error("Worker counts and request timeout must be positive")
    stop = args.output_dir / "STOP"
    if stop.exists():
        parser.error(f"Remove {stop} to resume")
    config = load_default_preset()
    config.remote.update(enabled=True, server_urls=[args.endpoint],
                         max_concurrent_requests=args.concurrency,
                         request_timeout=args.request_timeout)
    config.guideline["remote"]["model_name"] = args.model
    token, done = CancellationToken(), threading.Event()

    def watch():
        while not done.wait(1):
            if stop.exists():
                token.cancel()
                return

    handlers = {sig: signal.signal(sig, lambda *_: token.cancel())
                for sig in (signal.SIGINT, signal.SIGTERM)}
    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        with cancellation_scope(token):
            state = summarize_pdq_collection(
                json.loads(args.plan.read_text()), output_dir=args.output_dir,
                config=config, source_workers=args.source_workers,
                disease_workers=args.disease_workers,
                allow_endpoint_change=args.allow_endpoint_change,
                progress_callback=lambda item: print(json.dumps(item), flush=True),
            )
        return 0 if state["status"] == "complete" else 1
    except InferenceCancelled:
        return 130
    finally:
        done.set()
        watcher.join()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
