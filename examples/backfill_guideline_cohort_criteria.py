"""Add cohort-style key criteria to completed guideline catalogs, as derived editions (2026-11-14).

Catalogs extracted before the cohort-criteria step have only nine-field spaces. For every completed catalog under
``--catalog-root`` (``<root>/<disease>/paradigms.jsonl`` with a passing ``validation.json``), this writes
``<output-dir>/<disease>/``: the same records plus ``key_criteria`` / ``key_criteria_items`` written by the configured
endpoint, with a derived-edition audit. The base catalogs are only read. Rerun the same command to resume.

    python examples/backfill_guideline_cohort_criteria.py --catalog-root RUN_DIR --output-dir NEW_RUN_DIR \\
        --endpoint http://host:8000/v1 [--model MODEL] [--concurrency 128] [--disease-workers 8] [--disease aml ...]
"""

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from matchminer_ai import load_default_preset
from matchminer_ai._storage import atomic_json, read_json
from matchminer_ai.trials import add_guideline_cohort_criteria


def completed_catalogs(root):
    found = {}
    for folder in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        status, audit = folder / "status.json", folder / "validation.json"
        if (folder / "paradigms.jsonl").exists() and status.exists() and audit.exists():
            if read_json(status).get("status") == "complete" and read_json(audit).get("status") == "passed":
                found[folder.name] = folder
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model")
    parser.add_argument("--concurrency", type=int, default=128,
                        help="Requests in flight to the endpoint, shared by all diseases")
    parser.add_argument("--disease-workers", type=int, default=8)
    parser.add_argument("--disease", action="append", default=[])
    args = parser.parse_args()
    if args.concurrency < 1 or args.disease_workers < 1:
        parser.error("Concurrency and disease workers must be positive")
    root, output = args.catalog_root.resolve(), args.output_dir.resolve()
    if output == root or output.is_relative_to(root):
        parser.error("Write derived editions outside the base run directory")
    catalogs = completed_catalogs(root)
    if args.disease:
        unknown = set(args.disease) - set(catalogs)
        if unknown:
            parser.error(f"No completed catalog for: {sorted(unknown)}")
        catalogs = {d: catalogs[d] for d in args.disease}
    # The structured client's request slots are shared per endpoint across every disease in this process
    # (llm.request_limits.endpoint_slot), so this is the total cap, not a per-disease share.
    config = load_default_preset()
    config.remote.update(enabled=True, server_urls=[args.endpoint], max_concurrent_requests=args.concurrency,
                         request_timeout=7200)
    config.guideline["remote"]["model_name"] = args.model
    output.mkdir(parents=True, exist_ok=True)
    summary_path, lock = output / "backfill.json", threading.Lock()
    summary = read_json(summary_path) if summary_path.exists() else {}
    summary.update(catalog_root=str(root), endpoint=args.endpoint, model=args.model,
                   started_utc=datetime.now(UTC).isoformat(), diseases=summary.get("diseases", {}))

    def one(disease, folder):
        def progress(message):
            print(f"{disease}: {message}", flush=True)

        frame, audit = add_guideline_cohort_criteria(folder, output_dir=output / disease, config=config,
                                                     return_metadata=True, progress_callback=progress)
        return {"status": "complete", "paradigms": len(frame), "accepted_requests": audit["accepted_requests"]}

    with ThreadPoolExecutor(args.disease_workers) as pool:
        futures = {pool.submit(one, d, f): d for d, f in catalogs.items()}
        for future in as_completed(futures):
            disease = futures[future]
            try:
                result = future.result()
            except Exception as error:  # recorded per disease; rerun resumes from checkpoints
                result = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
            with lock:
                summary["diseases"][disease] = {**result, "updated_utc": datetime.now(UTC).isoformat()}
                atomic_json(summary_path, summary)
            print(f"{disease}: {result['status']}", flush=True)
    failed = sorted(d for d, r in summary["diseases"].items() if r["status"] != "complete")
    print(json.dumps({"complete": len(summary["diseases"]) - len(failed), "failed": failed}))
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
