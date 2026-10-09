"""Work-conserving collection dispatch for NCCN and PDQ catalog pipelines."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack

from matchminer_ai.cancellation import submit_cancellable


def catalog_worker_counts(assignments, concurrency, disease_workers=None):
    """Start enough catalogs to use the endpoint cap, without dividing it.

    A catalog can have a serial stage followed by many independent detail jobs.
    Automatic disease dispatch therefore allows up to the request cap in active
    catalogs per endpoint. An explicit disease-worker limit bounds preparation
    overhead, but never reduces the request capacity of a remaining catalog.
    """
    if type(concurrency) is not int or concurrency < 1:
        raise ValueError("Concurrency must be a positive integer")
    if disease_workers is not None and (
        type(disease_workers) is not int or disease_workers < 1
    ):
        raise ValueError("Disease workers must be a positive integer or None")
    limit = concurrency if disease_workers is None else disease_workers
    return {
        endpoint: min(limit, count)
        for endpoint, count in Counter(assignments.values()).items()
    }


def run_catalog_jobs(assignments, run, *, concurrency, disease_workers=None):
    """Dispatch independent catalogs; request slots remain shared, not reserved.

    ``run`` must pass the full endpoint concurrency to each public catalog API.
    The structured client's process-wide endpoint limiter bounds actual streamed
    requests. A finishing catalog releases its slots immediately for others;
    queued catalogs start as soon as a disease worker is free. Distinct endpoints
    have independent pools, preserving explicitly assigned checkpoint owners.
    Cancellation propagates into each disease workflow and its nested jobs.
    """
    counts = catalog_worker_counts(assignments, concurrency, disease_workers)
    with ExitStack() as stack:
        pools = {
            endpoint: stack.enter_context(ThreadPoolExecutor(max_workers=count))
            for endpoint, count in counts.items()
        }
        futures = [
            submit_cancellable(pools[endpoint], run, disease)
            for disease, endpoint in assignments.items()
        ]
        for future in as_completed(futures):
            future.result()
