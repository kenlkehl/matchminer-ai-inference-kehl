"""Invented sources and jobs: independent dispatch, bounded downloads and resume."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock

import pytest

from matchminer_ai import load_default_preset
from matchminer_ai._storage import read_json
from matchminer_ai.cancellation import CancellationToken, InferenceCancelled, cancellation_scope, check_cancelled
from matchminer_ai.trials import summarize_pdq_collection
from matchminer_ai.trials._pdq_discovery import INDEXES, treatment_inventory
from matchminer_ai.trials._pdq_sources import pdq_url

URL = "https://www.cancer.gov/types/invented/hp/invented-treatment-pdq"


def config():
    result = load_default_preset()
    result.remote.update(enabled=True, server_urls=["http://pdq.synthetic.invalid/v1"],
                         max_concurrent_requests=3)
    return result


def jobs(tmp_path, names):
    return {name: {"source_directory": str(tmp_path / "sources" / name),
                   "output_directory": str(tmp_path / "catalogs" / name)} for name in names}


def test_slow_download_does_not_block_existing_catalog_and_sources_are_bounded(tmp_path, monkeypatch):
    plan = jobs(tmp_path, ["existing", "fresh-a", "fresh-b"])
    for name in ["fresh-a", "fresh-b"]:
        plan[name]["urls"] = [URL]
    source_entered, existing_entered, release = Event(), Event(), Event()
    lock = Lock()
    active = peak = 0
    calls = []

    def prepare(*args, **kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        source_entered.set()
        try:
            assert release.wait(5)
        finally:
            with lock:
                active -= 1

    def summarize(source, **kwargs):
        name = kwargs["disease"]
        assert source == plan[name]["source_directory"]
        assert kwargs["output_dir"] == plan[name]["output_directory"]
        assert kwargs["config"].remote["max_concurrent_requests"] == 3
        assert kwargs["allow_endpoint_change"] is True
        calls.append(name)
        if name == "existing":
            existing_entered.set()
        return [1, 2], {"validation": {"status": "passed"}}

    monkeypatch.setattr("matchminer_ai.trials.pdq.prepare_pdq_library", prepare)
    monkeypatch.setattr("matchminer_ai.trials.pdq.summarize_pdq", summarize)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(summarize_pdq_collection, plan, output_dir=tmp_path / "collection",
                             config=config(), source_workers=1, allow_endpoint_change=True)
        try:
            assert source_entered.wait(3)
            assert existing_entered.wait(3), "Ready catalog waited for unrelated downloads"
        finally:
            release.set()
        result = future.result(5)
    assert peak == 1 and set(calls) == set(plan)
    assert result["status"] == "complete"
    assert len(result["completed_catalogs"]) == 3
    assert all(row["audit"] == "passed" for row in result["diseases"].values())


def test_failed_source_is_visible_and_other_jobs_continue_then_resume(tmp_path, monkeypatch):
    plan = jobs(tmp_path, ["failed", "ready"])
    plan["failed"]["urls"] = [URL]
    calls = []

    def fail(*args, **kwargs):
        raise ValueError("Invented source integrity failure")

    def summarize(source, **kwargs):
        calls.append(kwargs["disease"])
        return [1], {"validation": {"status": "passed"}}

    monkeypatch.setattr("matchminer_ai.trials.pdq.prepare_pdq_library", fail)
    monkeypatch.setattr("matchminer_ai.trials.pdq.summarize_pdq", summarize)
    run = lambda: summarize_pdq_collection(plan, output_dir=tmp_path / "collection", config=config(), disease_workers=1)
    state = run()
    assert state["status"] == "failed" and calls == ["ready"]
    assert "integrity failure" in state["diseases"]["failed"]["error"]
    monkeypatch.setattr("matchminer_ai.trials.pdq.prepare_pdq_library", lambda *a, **kw: None)
    state = run()
    assert state["status"] == "complete"
    assert state["diseases"]["failed"]["error"] is None
    changed = {**plan, **jobs(tmp_path, ["new"])}
    with pytest.raises(ValueError, match="plan changed"):
        summarize_pdq_collection(changed, output_dir=tmp_path / "collection", config=config())


def test_collection_cancellation_and_callback_exception(tmp_path, monkeypatch):
    token = CancellationToken()

    def summarize(*args, **kwargs):
        token.cancel()
        check_cancelled()

    def callback(_):
        raise RuntimeError("Broken UI callback")

    monkeypatch.setattr("matchminer_ai.trials.pdq.summarize_pdq", summarize)
    with cancellation_scope(token), pytest.raises(InferenceCancelled):
        summarize_pdq_collection(jobs(tmp_path, ["invented"]), output_dir=tmp_path / "collection",
                                 config=config(), progress_callback=callback)
    state = read_json(tmp_path / "collection/collection.json")
    assert state["status"] == "interrupted"
    assert state["diseases"]["invented"]["status"] == "interrupted"


def index_page(*links):
    return ('<main id="main-content"><article><h1>Invented treatment index</h1><ul>'
            + ''.join(f'<li>{label}<ul><li><a href="{url}">health professional</a></li></ul></li>'
                      for label, url in links) + '</ul></article></main>').encode()


def test_discovery_deduplicates_aliases_and_records_excluded_topics():
    pages = {
        INDEXES["adult"]: index_page(("Invented A", URL), ("Alias A", URL),
            ("Late Effects", "/types/childhood-cancers/late-effects-hp-pdq")),
        INDEXES["pediatric"]: index_page(("Invented A in Children", URL)),
    }
    result = treatment_inventory(pages)["entries"]
    assert len(result) == 2
    included, excluded = result
    assert included["labels"] == ["Invented A", "Alias A", "Invented A in Children"]
    assert len(included["indexes"]) == 2 and included["disease"] == "invented-treatment"
    assert not excluded["included"] and excluded["exclusion_reason"]


@pytest.mark.parametrize("url", ["https://evil.invalid/summary", "https://www.cancer.gov/unexpected"])
def test_unknown_index_links_fail_instead_of_silently_disappearing(url):
    pages = {index: index_page(("Invented", url)) for index in INDEXES.values()}
    with pytest.raises(ValueError):
        treatment_inventory(pages)


def test_same_slug_adult_and_child_sources_remain_distinct():
    pages = {index: index_page(("Invented", f"/types/invented/hp/{age}/same-treatment-pdq"))
             for index, age in zip(INDEXES.values(), ["adult", "child"])}
    entries = treatment_inventory(pages)["entries"]
    assert {e["disease"] for e in entries} == {"invented-adult-same-treatment", "invented-child-same-treatment"}


@pytest.mark.parametrize("path", [
    "/about-cancer/treatment/types/agnostic-cancer-therapies-hp-pdq",
    "/types/lymphoma/hp/mantle-cell-lymphoma-treatment",
    "/types/myeloproliferative/hp/myeloproliferative-neoplasms-treatment",
    "/types/cardiac/hp-child-cardiac-treatment-pdq",
    "/types/leukemia/hp/child-aml-treatment-pdq/childhood-apl-treatment-pdq",
])
def test_official_treatment_url_variants(path):
    assert pdq_url("https://www.cancer.gov" + path) == "https://www.cancer.gov" + path


def test_invalid_collection_paths_fail_before_any_dispatch(tmp_path):
    plan = jobs(tmp_path, ["a", "b"])
    plan["b"]["output_directory"] = plan["a"]["output_directory"]
    with pytest.raises(ValueError, match="distinct"):
        summarize_pdq_collection(plan, output_dir=tmp_path / "collection", config=config())


def test_subset_recovery_updates_parent_preserves_completed_and_can_resume(tmp_path, monkeypatch):
    from matchminer_ai._storage import atomic_json, digest
    plan = jobs(tmp_path, ["done", "failed", "another-failure"])
    completed = {"status": "complete", "output_directory": plan["done"]["output_directory"], "spaces": 7, "audit": "passed"}
    parent = tmp_path / "parent/collection.json"
    atomic_json(parent, {"identity": {"jobs": plan, "plan_sha256": digest(plan)}, "status": "failed",
        "diseases": {"done": completed, **{d: {"status": "failed", "output_directory": plan[d]["output_directory"]}
                                           for d in ["failed", "another-failure"]}}})
    monkeypatch.setattr("matchminer_ai.trials.pdq.summarize_pdq",
                        lambda *a, **kw: ([1], {"validation": {"status": "passed"}}))
    for _ in range(2):
        result = summarize_pdq_collection({"failed": plan["failed"]}, output_dir=tmp_path / "recovery",
                                         config=config(), parent_collection=parent)
        assert result["status"] == "complete"
        updated = read_json(parent)
        assert updated["status"] == "failed"  # Other failure remains visible.
        assert updated["diseases"]["done"] == completed
        assert updated["diseases"]["failed"]["status"] == "complete"
        assert updated["diseases"]["another-failure"]["status"] == "failed"
    with pytest.raises(ValueError, match="completed parent"):
        summarize_pdq_collection({"done": plan["done"]}, output_dir=tmp_path / "unrelated-recovery",
                                 config=config(), parent_collection=parent)
