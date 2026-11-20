"""Background jobs with polled progress. Model work runs one job at a time (one CPU-bound model in process)."""
from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import threading
import time
import traceback
import uuid


@dataclass
class Job:
    id: str
    kind: str
    label: str
    status: str = "queued"  # queued, running, done, error
    done: int = 0
    total: int = 0
    message: str = ""
    device: str | None = None  # where the model runs ("cpu", "cuda:3", "endpoint"), shown with the progress
    result: object = None
    error: str | None = None
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None

    def describe(self, with_result: bool = True) -> dict:
        value = {k: getattr(self, k) for k in ("id", "kind", "label", "status", "done", "total", "message", "error",
                                               "device", "created", "started", "finished")}
        value["seconds"] = round((self.finished or time.time()) - self.started, 1) if self.started else None
        if with_result:
            value["result"] = self.result
        return value


class JobManager:
    def __init__(self, workers: int = 1, keep: int = 200):
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="mmai-job")
        self.jobs: OrderedDict[str, Job] = OrderedDict()
        self.lock = threading.Lock()
        self.keep = keep

    def submit(self, kind: str, label: str, function, device: str | None = None) -> Job:
        """Run ``function(progress)``; ``progress(done, total, message)`` updates the job."""
        job = Job(uuid.uuid4().hex[:12], kind, label, device=device)
        with self.lock:
            self.jobs[job.id] = job
            while len(self.jobs) > self.keep:
                self.jobs.popitem(last=False)

        def progress(done, total, message=""):
            job.done, job.total, job.message = done, total, message

        def run():
            job.status, job.started = "running", time.time()
            try:
                job.result = function(progress)
                job.status = "done"
            except Exception as exc:  # shown in the page
                job.error = f"{type(exc).__name__}: {exc}"
                job.message = traceback.format_exc(limit=3)[-800:]
                job.status = "error"
            finally:
                job.finished = time.time()

        self.pool.submit(run)
        return job

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def recent(self, n: int = 20) -> list[dict]:
        return [job.describe(with_result=False) for job in list(self.jobs.values())[-n:]][::-1]

    def shutdown(self):
        self.pool.shutdown(wait=False, cancel_futures=True)
