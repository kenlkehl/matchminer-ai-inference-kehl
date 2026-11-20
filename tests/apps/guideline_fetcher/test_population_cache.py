import json
import stat

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from matchminer_ai.apps.guideline_fetcher.app import create_app
from matchminer_ai.apps.guideline_fetcher.population import browse_databases, load_population_sample
from matchminer_ai.apps.guideline_fetcher.services import InputError
from matchminer_ai.apps.guideline_fetcher.settings import Settings
from .test_app import catalog, wait
from .test_population import SPACE, database_settings, fabricated_database

IDENTITY = {"encoder": "fabricated-revision"}


@pytest.fixture
def fake_trialspace(monkeypatch):
    calls = []

    def embed(frame, *, entity_type, **kwargs):
        calls.append((entity_type, len(frame)))
        key = "patient_id" if entity_type == "patient" else "space_trial_id"
        text = "cancer_history_summary" if entity_type == "patient" else "cohort_key_criteria"
        result = frame[[key]].copy()
        result["embedding"] = [[float(len(t)), 1.0] for t in frame[text]]
        return result, {"model_metadata": {"embed_for_matching": {"model_name": "fake/TrialSpace"}}}

    monkeypatch.setattr("matchminer_ai.embedding.embed_for_matching", embed)
    monkeypatch.setattr("matchminer_ai.embedding.inference._embedding_cache_identity",
                        lambda config, model: IDENTITY)
    monkeypatch.setattr("matchminer_ai.matching.guideline_encoder_identity",
                        lambda config=None: IDENTITY)
    return calls


@pytest.fixture
def built(tmp_path, fake_trialspace):
    from matchminer_ai.apps.guideline_fetcher.population_cache import build_population_cache

    database = fabricated_database(tmp_path / "oa_fabricated.duckdb")
    output = tmp_path / "guideline_population_cache" / "oa_fabricated"
    settings = database_settings(tmp_path, database, population_cache=output)
    messages = []
    manifest = build_population_cache(settings, output, batch_size=1, log=messages.append)
    return settings, output, manifest, messages


def test_build_writes_a_private_complete_cache(built, fake_trialspace):
    settings, output, manifest, messages = built
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    for name in ("manifest.json", "summaries.parquet", "embeddings.npy", "notes.parquet"):
        assert stat.S_IMODE((output / name).stat().st_mode) == 0o600
    assert not list(output.parent.glob(".*partial*")) and not (output / "duckdb_spill").exists()
    assert manifest["counts"] == {"patients": 2, "notes": 6}
    source = manifest["source"]
    # 106's summary is blank; 103 has only blank notes and 105 has none.
    assert source["current_summaries"] == 5 and source["excluded_blank_summary_count"] == 1
    assert source["excluded_without_notes_count"] == 2 and source["duplicate_notes_removed"] == 1
    assert manifest["embedding"]["encoder_identity"] == IDENTITY
    assert manifest == json.loads((output / "manifest.json").read_text())
    summaries = pd.read_parquet(output / "summaries.parquet")
    assert summaries.patient_id.tolist() == ["101", "102"]
    assert summaries.note_count.tolist() == [5, 1]
    assert summaries.cancer_history_summary.tolist()[0] == "Fabricated lung cancer"
    vectors = np.load(output / "embeddings.npy")
    assert vectors.dtype == np.float32 and vectors.shape == (2, 2)
    assert vectors[0, 0] == len("Fabricated lung cancer")
    assert ("patient", 1) in fake_trialspace
    # Logs carry counts and progress, never patient identifiers or text.
    assert not any("101" in m or "Fabricated" in m for m in messages)


def test_cache_notes_match_the_duckdb_sample_loader(built):
    from matchminer_ai.population import create_patient_cohort

    from matchminer_ai.apps.guideline_fetcher.population_cache import open_cache

    settings, output, manifest, _ = built
    cache = open_cache(settings)
    assert open_cache(settings) is cache
    sample = load_population_sample(settings, 4, 23, lambda _: None)
    cached = create_patient_cohort(cache.summaries, cache.notes(["101", "102"]))
    pd.testing.assert_frame_equal(cached.notes, sample.notes)
    assert cache.notes(["102"]).note_text.tolist() == ["Colonoscopy completed."]
    assert cache.embeddings.patient_ids == ("101", "102")
    with pytest.raises(InputError, match="rebuilt"):
        open_cache(settings, expected="stale")


def test_a_cache_from_another_encoder_is_refused(built, monkeypatch):
    """Vectors made by an earlier TrialSpace (such as 0526) never load beside the current model's."""
    from matchminer_ai.apps.guideline_fetcher import population_cache
    settings, output, manifest, _ = built
    population_cache._LOADED.clear()
    monkeypatch.setattr("matchminer_ai.matching.guideline_encoder_identity",
                        lambda config=None: {"encoder": "current-round-2"})
    with pytest.raises(InputError, match="another TrialSpace encoder"):
        population_cache.open_cache(settings)
    assert not population_cache._LOADED


def test_build_is_atomic_and_never_silently_replaces(built, monkeypatch):
    from matchminer_ai.apps.guideline_fetcher.population_cache import build_population_cache

    settings, output, manifest, _ = built
    with pytest.raises(InputError, match="--overwrite"):
        build_population_cache(settings, output, log=lambda _: None)

    def broken(*args, **kwargs):
        raise RuntimeError("TrialSpace unavailable")

    monkeypatch.setattr("matchminer_ai.embedding.embed_for_matching", broken)
    with pytest.raises(RuntimeError):
        build_population_cache(settings, output, overwrite=True, log=lambda _: None)
    assert json.loads((output / "manifest.json").read_text()) == manifest
    assert sorted(p.name for p in output.parent.iterdir()) == ["oa_fabricated"]
    monkeypatch.undo()


def test_overwrite_and_trial_limit(built, fake_trialspace):
    from matchminer_ai.apps.guideline_fetcher.population_cache import build_population_cache, inspect_cache

    settings, output, manifest, _ = built
    rebuilt = build_population_cache(settings, output, overwrite=True, limit=1,
                                     log=lambda _: None)
    assert rebuilt["source"]["sample_limit"] == 1
    assert rebuilt["counts"]["patients"] <= 1
    assert sorted(p.name for p in output.parent.iterdir()) == ["oa_fabricated"]
    info = inspect_cache(settings)
    assert info["sample_limit"] == 1 and info["embedding_model"] == "fake/TrialSpace"


def test_cache_selection_is_validated(built, tmp_path):
    from matchminer_ai.apps.guideline_fetcher.population_cache import inspect_cache

    settings, output, manifest, _ = built
    assert inspect_cache(settings)["patients"] == 2
    with pytest.raises(InputError, match="No population cache"):
        inspect_cache(settings, str(tmp_path))
    with pytest.raises(InputError, match="configured database folders"):
        inspect_cache(settings, "/etc")
    listing = browse_databases(settings, str(output.parent))
    assert listing["caches"] == [{"name": "oa_fabricated", "path": str(output)}]
    assert listing["directories"] == []
    (output / "notes.parquet").write_bytes(b"truncated")
    with pytest.raises(InputError, match="incomplete or changed"):
        inspect_cache(settings)


def test_cached_population_routes_load_notes_only_for_matched_patients(
    built, tmp_path, monkeypatch
):
    from matchminer_ai.matching import PatientEmbeddings

    settings, output, manifest, _ = built
    calls = []

    def retrieve(summaries, paths, **kwargs):
        calls.append(kwargs)
        assert isinstance(kwargs["patient_embeddings"], PatientEmbeddings)
        frame = summaries.loc[summaries.patient_id.eq("102")].copy()
        frame["space_trial_id"] = "s"
        frame["match_quality_score"] = [0.9]
        frame["match_quality_pass"] = True
        frame["similarity_score"] = [0.8]
        frame["retrieval_rank"] = [1]
        frame["rank"] = [1]
        return frame, {"space": SPACE, "scored_count": 2, "population_count": 2,
                       "min_reasonableness": kwargs["min_reasonableness"]}

    monkeypatch.setattr("matchminer_ai.matching.retrieve_guideline_patients", retrieve)
    catalog(tmp_path / "catalogs")
    settings.catalog_root = tmp_path / "catalogs"
    api = create_app(settings)
    with TestClient(api) as client:
        assert client.get("/api/population").json()["defaults"]["cache"] == str(output)
        assert client.get("/api/population/cache").json()["notes"] == 6
        assert client.post("/api/population/cache", json={"path": "/etc"}).status_code == 400
        response = client.post("/api/population/cache", json={"path": str(output)})
        loaded = wait(client, response.json()["job_id"]).json()["result"]
        assert loaded["source_count"] == 2 and loaded["source_note_count"] == 6
        assert loaded["notes_on_demand"] is True
        assert loaded["provenance"]["cache"]["name"] == "oa_fabricated"
        catalog_id = client.get("/api/catalogs").json()["catalogs"][0]["id"]
        response = client.post("/api/population/retrieve", json={
            "version": loaded["version"], "catalog_id": catalog_id, "space_trial_id": "s",
        })
        active = wait(client, response.json()["job_id"]).json()["result"]["active"]
        assert active["patient_count"] == 1 and active["note_count"] == 1
        patient = client.get("/api/population/patient", params={
            "version": client.get("/api/population").json()["population"]["version"],
            "patient_id": "102",
        }).json()
        assert [n["note_text"] for n in patient["notes"]] == ["Colonoscopy completed."]
    assert len(calls) == 1


def test_population_caches_are_phi_only(built, tmp_path):
    from matchminer_ai.apps.guideline_fetcher.population_cache import build_population_cache, inspect_cache, open_cache

    settings, output, manifest, _ = built
    standard = Settings(device="cpu", population_duckdb=settings.population_duckdb,
                        duckdb_roots=[tmp_path], population_cache=output)
    for call in (lambda: inspect_cache(standard), lambda: open_cache(standard),
                 lambda: build_population_cache(standard, tmp_path / "other", log=print)):
        with pytest.raises(InputError, match="PHI app"):
            call()
    # A population cache holds PHI and never lands in a no_phi folder.
    with pytest.raises(InputError, match="no_phi"):
        build_population_cache(settings, tmp_path / "no_phi" / "cache", log=lambda _: None)
    assert not (tmp_path / "no_phi").exists()


def test_summaries_can_come_from_parquet_files(tmp_path, fake_trialspace):
    from matchminer_ai.apps.guideline_fetcher.population_cache import build_population_cache

    database = fabricated_database(tmp_path / "oa_fabricated.duckdb")
    folder = tmp_path / "summaries"
    folder.mkdir()
    pd.DataFrame(dict(patient_id=["101"], cancer_history_summary=["  Fabricated serial lung summary "],
                      general_exclusion_criteria_evidence=[None])).to_parquet(folder / "bucket-001.parquet")
    # 103's only note is blank; 104 has notes but no current database summary.
    pd.DataFrame(dict(patient_id=["103", "104"], cancer_history_summary=[
        "Fabricated serial breast summary", "Fabricated serial other summary"])).to_parquet(
        folder / "bucket-002.parquet")
    output = tmp_path / "guideline_population_cache" / "from_files"
    settings = database_settings(tmp_path, database, population_cache=output)
    manifest = build_population_cache(settings, output, summaries=folder, batch_size=1,
                                      log=lambda _: None)
    cached = pd.read_parquet(output / "summaries.parquet")
    assert cached.patient_id.tolist() == ["101", "104"]
    assert cached.cancer_history_summary.tolist() == [
        "Fabricated serial lung summary", "Fabricated serial other summary"]
    assert cached.general_exclusion_criteria_evidence.tolist() == ["", ""]
    assert manifest["source"]["tables"]["summaries"] == "Parquet: summaries (2 files)"
    assert manifest["source"]["summary_files"]["files"] == 2
    assert manifest["source"]["excluded_without_notes_count"] == 1
    assert set(pd.read_parquet(output / "notes.parquet").patient_id) == {"101", "104"}

    pd.DataFrame(dict(patient_id=["101"], cancer_history_summary=["Again"])).to_parquet(
        folder / "bucket-003.parquet")
    with pytest.raises(InputError, match="more than one row"):
        build_population_cache(settings, tmp_path / "guideline_population_cache" / "duplicate",
                               summaries=folder, log=lambda _: None)
    named = tmp_path / "named.parquet"
    pd.DataFrame(dict(patient_id=["P-101"], cancer_history_summary=["x"])).to_parquet(named)
    with pytest.raises(InputError, match="numeric MRNs"):
        build_population_cache(settings, tmp_path / "guideline_population_cache" / "named",
                               summaries=named, log=lambda _: None)
