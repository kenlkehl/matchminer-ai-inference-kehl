"""Bound pair-chain concurrency by the live teacher pool, including direct runs."""
import asyncio
import resource

DEFAULT_CEILING = 768
DEFAULT_INITIAL = 96
DEFAULT_AUTO_WORKERS_MAX = 16384


def worker_target(workers, capacity, maximum=DEFAULT_AUTO_WORKERS_MAX):
    if workers < 0 or maximum < 1:
        raise ValueError('Pair workers must be nonnegative and the automatic maximum positive')
    # Keep 25% headroom for checkpoint writes and preparation between requests.
    # Explicit positive settings remain a fixed user-selected bound.
    return workers or min(maximum, max(1, (max(0, capacity) * 5 + 3) // 4))


def reserve_descriptors(workers):
    """Raise the process's soft limit, never the system hard limit."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    required = 2 * workers + 2048
    if soft == resource.RLIM_INFINITY or soft >= required:
        return
    if hard != resource.RLIM_INFINITY and hard < required:
        raise RuntimeError(f'{workers} label workers require an open-file limit of {required}; '
                           'raise the hard limit or reduce the label worker limit')
    resource.setrlimit(resource.RLIMIT_NOFILE, (required, hard))


async def run_workers(items, process, capacity, *, workers=0,
                      maximum=DEFAULT_AUTO_WORKERS_MAX, on_resize=None, poll_seconds=5):
    """Resize between complete pair chains; never cancel a chain to shrink.

    Workers own the whole gated question chain, including durable response writes.
    A growing endpoint registry creates workers without restarting the labeler.
    Explicit limits, bounded prefetch, exceptions, and cancellation are preserved.
    """
    target = worker_target(workers, 0, maximum)
    iterator, sentinel = iter(items), object()
    tasks = set()
    active = 0
    exhausted = False
    last_target = None

    async def worker():
        nonlocal active, exhausted
        try:
            while not exhausted:
                if active > target:
                    return
                item = next(iterator, sentinel)
                if item is sentinel:
                    exhausted = True
                    return
                await process(item)
        finally:
            active -= 1

    try:
        while tasks or not exhausted:
            target = worker_target(workers, capacity() if not workers else 0, maximum)
            if target != last_target:
                reserve_descriptors(target)
                last_target = target
                if on_resize:
                    on_resize(target)
            while not exhausted and active < target:
                active += 1
                tasks.add(asyncio.create_task(worker()))
            if not tasks:
                break
            done, _ = await asyncio.wait(tasks, timeout=poll_seconds,
                                         return_when=asyncio.FIRST_COMPLETED)
            tasks.difference_update(done)
            # Retrieve every exception before propagating one and cancelling
            # remaining workers; no task failures disappear during a resize.
            errors = [task.exception() for task in done if not task.cancelled()]
            if any(task.cancelled() for task in done):
                raise asyncio.CancelledError()
            for error in errors:
                if error is not None:
                    raise error
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
