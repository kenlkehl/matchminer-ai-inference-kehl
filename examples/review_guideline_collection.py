"""Resume source-citation reviews of caller-supplied disease catalogs.

Input is a JSON object mapping disease names to completed catalog directories.
All outputs, checkpoints, original records and live progress stay outside the repo.
"""

import argparse
import copy
import json
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from matchminer_ai import load_default_preset
from matchminer_ai._storage import atomic_json, directory_lock, read_json
from matchminer_ai.trials import review_guideline_citations
from matchminer_ai.trials.guidelines import _output_directory


def run_collection(catalogs, output, config, *, disease_workers=4):
    output = Path(output).resolve()
    identity = {key: str(Path(value).resolve()) for key, value in catalogs.items()}
    for value in identity.values():
        original = Path(value)
        original = original if original.is_dir() else original.parent
        source = read_json(original / "sources.json")
        _output_directory(Path(source["input_directory"]).parent, output)
        if output.is_relative_to(original) or original.is_relative_to(output):
            raise ValueError("Collection output must be separate from original catalogs")
    output.mkdir(parents=True, exist_ok=True)
    saved = output / "collection.json"
    lock = threading.Lock()
    with directory_lock(output):
        previous = read_json(saved) if saved.exists() else {}
        if previous and previous["catalogs"] != identity:
            raise ValueError("Collection inputs changed; use a new output directory")
        state = {
            "catalogs": identity,
            "status": "running",
            "diseases": {
                key: previous.get("diseases", {}).get(key, {"status": "pending"})
                for key in identity
            },
        }

        def update(disease=None, **values):
            with lock:
                if disease:
                    state["diseases"][disease].update(values)
                else:
                    state.update(values)
                state["updated_utc"] = datetime.now(timezone.utc).isoformat()
                state["counts"] = dict(
                    Counter(v["status"] for v in state["diseases"].values())
                )
                atomic_json(saved, state)
                print(json.dumps({"disease": disease, **values}), flush=True)

        def work(disease):
            update(disease, status="running", error=None)
            try:
                _, metadata = review_guideline_citations(
                    identity[disease],
                    output_dir=output / disease,
                    config=copy.deepcopy(config),
                    return_metadata=True,
                    progress_callback=lambda message: update(disease, progress=message),
                )
                audit = metadata["validation"]
                update(
                    disease,
                    status="complete",
                    spaces=audit["paradigms"],
                    unresolved_items=audit["unresolved_items"],
                )
            except Exception as exc:
                update(disease, status="failed", error=f"{type(exc).__name__}: {exc}")

        update()
        with ThreadPoolExecutor(max_workers=disease_workers) as pool:
            futures = [pool.submit(work, disease) for disease in identity]
            for future in as_completed(futures):
                future.result()
        update(
            status="failed"
            if any(v["status"] == "failed" for v in state["diseases"].values())
            else "complete"
        )
        return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--disease-workers", type=int, default=4)
    args = parser.parse_args()
    if args.concurrency < 1 or args.disease_workers < 1:
        parser.error("Concurrency and disease workers must be positive")
    catalogs = read_json(args.catalog_manifest)
    if (
        not isinstance(catalogs, dict)
        or not catalogs
        or any(
            not isinstance(k, str)
            or Path(k).name != k
            or k in {".", ".."}
            or not isinstance(v, str)
            for k, v in catalogs.items()
        )
    ):
        parser.error(
            "Catalog manifest must map simple disease folder names to catalog paths"
        )
    config = load_default_preset()
    config.remote.update(
        enabled=True,
        server_urls=[args.endpoint],
        max_concurrent_requests=args.concurrency,
        request_timeout=7200,
    )
    config.guideline["remote"]["model_name"] = args.model
    state = run_collection(
        catalogs, args.output_dir, config, disease_workers=args.disease_workers
    )
    raise SystemExit(0 if state["status"] == "complete" else 1)


if __name__ == "__main__":
    main()
