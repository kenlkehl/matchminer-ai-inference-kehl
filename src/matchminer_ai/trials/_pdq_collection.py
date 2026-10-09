"""Feed independent PDQ catalogs to one shared endpoint without a download barrier."""

import threading
import re
import logging
from datetime import datetime, timezone
from pathlib import Path

from matchminer_ai._storage import atomic_json, digest, directory_lock, read_json
from matchminer_ai.cancellation import InferenceCancelled, check_cancelled
from ._catalog_scheduling import catalog_worker_counts, run_catalog_jobs
from ._pdq_sources import pdq_url


def run_collection(jobs, *, output_dir, config, source_workers=2,
                   disease_workers=None, allow_endpoint_change=False,
                   progress_callback=None):
    """Run a caller-owned plan; downloading one source does not block ready catalogs."""
    from .pdq import _sources, prepare_pdq_library, summarize_pdq

    if not isinstance(jobs, dict) or not jobs:
        raise ValueError("jobs must be a nonempty disease-to-job mapping")
    if type(source_workers) is not int or source_workers < 1:
        raise ValueError("source_workers must be a positive integer")
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable or None")
    endpoints = config.remote.get("server_urls", [])
    if not config.remote.get("enabled") or len(endpoints) != 1:
        raise ValueError("Configure exactly one explicit remote endpoint")
    cap = config.remote["max_concurrent_requests"]
    work = Path(output_dir).resolve()
    package = Path(__file__).resolve().parents[1]
    checkout = package.parent.parent

    def external(path):
        path = Path(path).resolve()
        for root in [package, *([checkout] if (checkout / "pyproject.toml").exists() else [])]:
            if path.is_relative_to(root) or root.is_relative_to(path):
                raise ValueError("PDQ collection directories must be outside the code repository")
        return path

    external(work)
    plan = {}
    for disease, job in jobs.items():
        if not isinstance(disease, str) or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", disease):
            raise ValueError("PDQ disease names must be lowercase alphanumeric slugs")
        if (not isinstance(job, dict)
            or not {"source_directory", "output_directory"} <= set(job)
            or set(job) - {"source_directory", "output_directory", "urls"}):
            raise ValueError("A job requires source_directory, output_directory, and optional urls")
        source, output = external(job["source_directory"]), external(job["output_directory"])
        for target in [work, output]:
            if target.is_relative_to(source) or source.is_relative_to(target):
                raise ValueError("Catalog and collection outputs must be separate from sources")
        item = {"source_directory": str(source), "output_directory": str(output)}
        if "urls" in job:
            urls = _sources({disease: job["urls"]})[disease]
            item["urls"] = [pdq_url(url) for url in urls]
        plan[disease] = item
    outputs = [Path(j["output_directory"]) for j in plan.values()]
    if any(a == b or a.is_relative_to(b) or b.is_relative_to(a)
           for i, a in enumerate(outputs) for b in outputs[i+1:]):
        raise ValueError("Disease output directories must be distinct and not nested")
    assignments = {d: endpoints[0] for d in plan}
    counts = catalog_worker_counts(assignments, cap, disease_workers)
    work.mkdir(parents=True, exist_ok=True)
    with directory_lock(work):
        path = work / "collection.json"
        previous = read_json(path) if path.exists() else {}
        identity = {"jobs": plan, "plan_sha256": digest(plan)}
        if previous and previous["identity"] != identity:
            raise ValueError("PDQ collection plan changed; use a new collection output directory")
        state = {
            "identity": identity, "status": "running", "endpoint": endpoints[0],
            "concurrency": cap, "source_workers": source_workers,
            "effective_disease_workers": counts,
            "allocation_policy": "shared-endpoint-capacity-v1",
            "diseases": {d: {**previous.get("diseases", {}).get(d, {}),
                             "status": "pending", "output_directory": j["output_directory"]}
                         for d, j in plan.items()},
        }
        import os
        state["pid"] = os.getpid()
        lock, sources = threading.Lock(), threading.BoundedSemaphore(source_workers)

        def update(disease=None, **values):
            with lock:
                (state["diseases"][disease] if disease else state).update(values)
                state["updated_utc"] = datetime.now(timezone.utc).isoformat()
                state["completed_catalogs"] = [j["output_directory"] for j in state["diseases"].values()
                                               if j["status"] == "complete"]
                atomic_json(path, state)
            if progress_callback:
                try:
                    progress_callback({"disease": disease, **values})
                except Exception:
                    logging.getLogger(__name__).exception("PDQ collection progress callback failed")

        def run(disease):
            job = plan[disease]
            try:
                if "urls" in job:
                    update(disease, status="running", stage="source_wait", error=None)
                    while not sources.acquire(timeout=0.1):
                        check_cancelled()
                    try:
                        check_cancelled()
                        update(disease, stage="source")
                        prepare_pdq_library(
                            {disease: job["urls"]}, output_dir=job["source_directory"],
                            progress_callback=lambda message: update(disease, progress=message),
                        )
                    finally:
                        sources.release()
                check_cancelled()
                update(disease, status="running", stage="catalog", error=None)
                frame, metadata = summarize_pdq(
                    job["source_directory"], disease=disease,
                    output_dir=job["output_directory"], config=config,
                    allow_endpoint_change=allow_endpoint_change, return_metadata=True,
                    progress_callback=lambda message: update(disease, progress=message),
                )
                update(disease, status="complete", stage="export", spaces=len(frame),
                       audit=metadata["validation"]["status"])
            except InferenceCancelled:
                update(disease, status="interrupted")
                raise
            except Exception as exc:
                update(disease, status="failed", error=f"{type(exc).__name__}: {exc}")

        update()
        try:
            run_catalog_jobs(assignments, run, concurrency=cap, disease_workers=disease_workers)
        except InferenceCancelled:
            update(status="interrupted")
            raise
        update(status="complete" if all(j["status"] == "complete" for j in state["diseases"].values()) else "failed")
        return state
