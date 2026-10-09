"""Synthetic collection saturation and tail-capacity regressions."""

import importlib.util
import io
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, Lock

import pandas as pd
import pytest

from matchminer_ai import load_default_preset
from matchminer_ai.cancellation import CancellationToken, cancellation_scope
from matchminer_ai.llm.structured import StructuredClient, StructuredConfig
from matchminer_ai.trials._catalog_scheduling import (
    catalog_worker_counts,
    run_catalog_jobs,
)
from test_guideline_extraction import make_library


def test_last_catalog_can_borrow_every_released_stream_slot(monkeypatch):
    """Finishing a sibling immediately expands the tail's actual HTTP capacity."""
    first_entered, release_first = Event(), Event()
    tail_full, release_tail = Event(), Event()
    lock = Lock()
    active = tail_active = peak = 0

    def respond(request, timeout):
        nonlocal active, tail_active, peak
        name = json.loads(request.data)["name"]
        with lock:
            active += 1
            peak = max(peak, active)
            if name == "tail":
                tail_active += 1
                if tail_active == 3:
                    tail_full.set()
            else:
                first_entered.set()
        try:
            assert (release_tail if name == "tail" else release_first).wait(5)
            return io.BytesIO(b'{"choices":[]}')
        finally:
            with lock:
                active -= 1
                if name == "tail":
                    tail_active -= 1

    monkeypatch.setattr("matchminer_ai.llm.structured.urlopen", respond)
    endpoint = "http://catalog-tail.synthetic.invalid/v1"

    def run(disease):
        client = StructuredClient(
            StructuredConfig(base_url=endpoint, max_concurrent_requests=3), None,
        )
        if disease == "early":
            client._http("/chat/completions", {"name": "early"})
        else:
            assert first_entered.wait(3)
            with ThreadPoolExecutor(max_workers=3) as jobs:
                futures = [jobs.submit(client._http, "/chat/completions",
                                       {"name": "tail"}) for _ in range(3)]
                for future in futures:
                    future.result()

    with ThreadPoolExecutor(max_workers=1) as driver:
        future = driver.submit(
            run_catalog_jobs, {"early": endpoint, "tail": endpoint}, run,
            concurrency=3,
        )
        try:
            assert first_entered.wait(3)
            assert not tail_full.wait(0.1)  # early still holds one of three slots
            release_first.set()
            assert tail_full.wait(3)  # tail now uses all three without a restart
        finally:
            release_first.set()
            release_tail.set()
        future.result(timeout=3)
    assert peak == 3 and active == 0


@pytest.mark.parametrize("kind", ["guideline_collection", "pdq"])
def test_runner_auto_dispatches_serial_catalogs_up_to_shared_cap(
    tmp_path, monkeypatch, kind,
):
    """Both default runners can fill nine slots during one-call serial stages."""
    path = Path(__file__).parents[1] / f"examples/extract_{kind}.py"
    spec = importlib.util.spec_from_file_location(f"auto_{kind}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = make_library(tmp_path)
    diseases = ["fictional", *[f"invented-{i}" for i in range(9)]]
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["guidelines"] = [dict(manifest["guidelines"][0], folder=d)
                              for d in diseases]
    manifest_path.write_text(json.dumps(manifest))
    output = tmp_path / "results"
    endpoint = "http://auto.synthetic.invalid/v1"
    full, release = Event(), Event()
    lock = Lock()
    calls = []

    def summarize(*args, **kwargs):
        config = kwargs["config"]
        assert config.remote["max_concurrent_requests"] == 9
        assert config.remote["server_urls"] == [endpoint]
        assert config.guideline["remote"]["request_params"]["max_tokens"] == 100000
        assert config.guideline["reasoning_effort"] == "xhigh"
        with lock:
            calls.append(kwargs["disease"])
            if len(calls) >= 9:
                full.set()
        assert release.wait(5)
        kwargs["progress_callback"]("Synthetic serial stage complete")
        return ([1], {"validation": {"status": "passed"}}) if kind == "pdq" else [1]

    if kind == "pdq":
        monkeypatch.setattr(module, "summarize_pdq", summarize)
        config = load_default_preset()
        config.remote.update(enabled=True, server_urls=[endpoint], max_concurrent_requests=9)
        # The normal PDQ snapshot lives above markdown; use the same synthetic manifest.
        pdq_source = tmp_path / "pdq-source"
        (pdq_source / "markdown").mkdir(parents=True)
        (pdq_source / "markdown/manifest.json").write_bytes(manifest_path.read_bytes())
        def run():
            return module.run_batch(pd.DataFrame({"disease": diseases}),
                                    pdq_source, output, config)
    else:
        monkeypatch.setattr(module, "summarize_guidelines", summarize)
        monkeypatch.setattr(sys, "argv", [
            "runner", "--source", str(source), "--output-dir", str(output),
            "--endpoint", endpoint, "--concurrency", "9",
        ])
        run = module.main
    with ThreadPoolExecutor(max_workers=1) as driver:
        future = driver.submit(run)
        try:
            assert full.wait(3), "Default disease limit strands allowed request slots"
        finally:
            release.set()
        assert not future.result(timeout=5)
    state = json.loads((output / "collection.json").read_text())
    assert state["effective_disease_workers"] == {endpoint: 9}
    assert state["allocation_policy"] == "shared-endpoint-capacity-v1"
    assert state["status"] == "complete" and set(calls) == set(diseases)


def test_explicit_preparation_limit_and_replica_allocation():
    assignments = {f"a-{i}": "chosen-a" for i in range(5)} | {"b": "chosen-b"}
    assert catalog_worker_counts(assignments, 4) == {"chosen-a": 4, "chosen-b": 1}
    assert catalog_worker_counts(assignments, 4, 2) == {"chosen-a": 2, "chosen-b": 1}
    assert catalog_worker_counts({"last": "chosen-a"}, 96) == {"chosen-a": 1}
    assert catalog_worker_counts({}, 4) == {}


@pytest.mark.parametrize("concurrency,workers", [(0, None), (True, None), (3, 0), (3, True)])
def test_invalid_allocation_fails_before_dispatch(concurrency, workers):
    with pytest.raises(ValueError):
        run_catalog_jobs({"invented": "chosen"}, lambda _: pytest.fail("Dispatched"),
                         concurrency=concurrency, disease_workers=workers)


def test_catalog_dispatch_propagates_cancellation():
    from matchminer_ai.cancellation import InferenceCancelled, check_cancelled

    token = CancellationToken()

    def run(_):
        token.cancel()
        check_cancelled()

    with cancellation_scope(token), pytest.raises(InferenceCancelled):
        run_catalog_jobs({"invented": "chosen"}, run, concurrency=3)
