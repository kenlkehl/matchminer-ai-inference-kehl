"""Synthetic-only tests of exact catalog lookup and model-ranked menu retrieval."""

import copy
import json
import os
from pathlib import Path

import pandas as pd
import pytest

from matchminer_ai._storage import atomic_json, digest
from matchminer_ai.matching import (
    retrieve_guideline_considerations,
    write_guideline_considerations_report,
)
from matchminer_ai.trials import get_guideline_considerations, load_guideline_catalog
from matchminer_ai.trials._guideline_schema import format_space, materialize_evidence
from matchminer_ai.trials._guideline_sources import load_guideline
from test_guideline_extraction import make_library, state


@pytest.fixture
def catalog_frame(tmp_path):
    root = make_library(tmp_path)
    guideline = load_guideline(root, "fictional")
    rows = []
    for identifier in ["a", "b", "c"]:
        value = materialize_evidence(state(), guideline.pages)
        value["name"] = "Fictional population " + identifier
        value["space"]["cancer_burden_allowed"] = "State " + identifier
        value.update(
            space_trial_id=identifier,
            trial_id="guideline:fictional:1.0",
            clinical_space_summary=format_space(value["space"]),
            omitted_source_page_ids=[],
            source={
                "title": "Fictional guideline",
                "disease": "fictional",
                "version": "1.0",
                "source_sha256": "synthetic-pdf-hash",
                "source_fingerprint": "synthetic-source-hash",
                "source_pdf": "fictional.pdf",
                "input_directory": str(root / "fictional"),
            },
        )
        rows.append(value)
    return pd.DataFrame(rows)


@pytest.fixture
def patients():
    return pd.DataFrame(
        [
            {
                "patient_id": "P2",
                "cancer_history_summary": "Synthetic patient beta",
                "unused_note": "not for model input",
            },
            {
                "patient_id": "P1",
                "cancer_history_summary": "Synthetic patient alpha",
                "unused_note": "not for model input",
            },
        ]
    )


@pytest.fixture
def model_stages(monkeypatch):
    calls = {"embedding": [], "checker": []}
    model = {"model_name": "synthetic-embedding", "model_sha": "same-revision"}

    def embed(frame, *, entity_type, config, return_metadata):
        assert return_metadata
        calls["embedding"].append((entity_type, frame.copy(), config))
        if entity_type == "patient":
            assert list(frame.columns) == ["patient_id", "cancer_history_summary"]
            result = pd.DataFrame(
                {"patient_id": frame.patient_id, "embedding": [[1.0, 0.0]] * len(frame)}
            )
        else:
            assert list(frame.columns) == ["space_trial_id", "clinical_space_summary"]
            vectors = {"a": [1.0, 0.0], "b": [0.8, 0.6], "c": [0.6, 0.8]}
            result = pd.DataFrame(
                {
                    "space_trial_id": frame.space_trial_id,
                    "embedding": [vectors[s] for s in frame.space_trial_id],
                }
            )
        return result, {"model_metadata": {"embedding_model": model}}

    def checker(frame, *, config, filter_low_quality, return_metadata):
        assert filter_low_quality is False
        assert return_metadata
        assert "diagnostic_workup" not in frame and "unused_note" not in frame
        calls["checker"].append(frame.copy())
        result = frame[["patient_id", "space_trial_id"]].copy()
        result["match_quality_score"] = result.space_trial_id.map(
            {"a": 0.1, "b": 0.9, "c": 0.8}
        )
        result["match_quality_pass"] = result.match_quality_score >= 0.2
        return result.iloc[::-1], {
            "model_metadata": {
                "match_quality_checker": {
                    "model_name": "synthetic-checker",
                    "model_sha": "checker-revision",
                }
            }
        }

    monkeypatch.setattr("matchminer_ai.embedding.embed_for_matching", embed)
    monkeypatch.setattr("matchminer_ai.matching.score_match_quality", checker)
    return calls


def test_lookup_preserves_exact_stored_menus_and_does_not_mutate_catalog(catalog_frame):
    original = copy.deepcopy(catalog_frame.iloc[0].diagnostic_workup)
    result = get_guideline_considerations(catalog_frame, space_trial_id="a")
    assert result.iloc[0].diagnostic_workup == original
    assert result.iloc[0].clinical_space_number == 1
    assert result.iloc[0].general_exclusion_criteria == "NA"
    result.iloc[0].diagnostic_workup[0]["conditions"] = "Modified returned copy"
    assert catalog_frame.iloc[0].diagnostic_workup == original
    assert (
        len(
            get_guideline_considerations(
                catalog_frame,
                clinical_space_summary=catalog_frame.iloc[0].clinical_space_summary,
            )
        )
        == 1
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"space_trial_id": "absent"},
        {"clinical_space_summary": "close but absent"},
        {"space_trial_id": "a", "clinical_space_summary": "wrong text"},
    ],
)
def test_exact_lookup_never_substitutes_a_similar_population(catalog_frame, kwargs):
    with pytest.raises(KeyError, match="not present"):
        get_guideline_considerations(catalog_frame, **kwargs)


def test_duplicate_text_in_different_editions_is_not_silently_collapsed(catalog_frame):
    second = copy.deepcopy(catalog_frame.iloc[0].to_dict())
    second["space_trial_id"] = "another-edition"
    second["source"]["version"] = "2.0"
    second["source"]["source_fingerprint"] = "different-edition"
    combined = pd.concat([catalog_frame, pd.DataFrame([second])], ignore_index=True)
    result = get_guideline_considerations(
        combined, clinical_space_summary=second["clinical_space_summary"]
    )
    assert result.space_trial_id.tolist() == ["a", "another-edition"]


def test_loader_checks_completion_and_saved_export_hash(catalog_frame, tmp_path):
    folder = tmp_path / "completed"
    folder.mkdir()
    content = "".join(json.dumps(r) + "\n" for r in catalog_frame.to_dict("records"))
    (folder / "paradigms.jsonl").write_text(content)
    atomic_json(folder / "status.json", {"status": "running"})
    with pytest.raises(ValueError, match="not complete"):
        load_guideline_catalog(folder)
    atomic_json(folder / "status.json", {"status": "complete"})
    atomic_json(
        folder / "validation.json",
        {"status": "passed", "paradigms_sha256": digest(content.encode())},
    )
    result, metadata = load_guideline_catalog(folder, return_metadata=True)
    assert len(result) == 3
    assert (
        metadata["artifacts"][0]["recorded_source_audit"]
        == "passed_export_hash_verified"
    )
    (folder / "paradigms.jsonl").write_text(content + "\n")
    with pytest.raises(ValueError, match="export hash differs"):
        load_guideline_catalog(folder)


@pytest.fixture
def saved_catalog(catalog_frame, tmp_path):
    folder = tmp_path / "saved-catalog"
    folder.mkdir()
    content = catalog_frame.to_json(orient="records", lines=True).encode()
    (folder / "paradigms.jsonl").write_bytes(content)
    atomic_json(folder / "status.json", {"status": "complete"})
    atomic_json(
        folder / "validation.json",
        {"status": "passed", "paradigms_sha256": digest(content)},
    )
    return folder


def test_unchanged_catalog_reuses_validation_without_content_reads(
    saved_catalog, monkeypatch
):
    from matchminer_ai.trials import guideline_catalog as module

    first, first_meta = load_guideline_catalog(saved_catalog, return_metadata=True)
    original = copy.deepcopy(first.to_dict("records"))
    first.iloc[0].diagnostic_workup[0]["conditions"] = "Caller mutation"
    first.iloc[0].evidence[0]["quote"] = "Caller mutation"
    first_meta["artifacts"][0]["sha256"] = "Caller mutation"

    def unexpected(*args, **kwargs):
        pytest.fail("An unchanged catalog was read or schema-validated again")

    original_bytes, original_text = Path.read_bytes, Path.read_text

    def forbid_catalog_reads(method):
        def read(path, *args, **kwargs):
            if path.parent == saved_catalog:
                unexpected()
            return method(path, *args, **kwargs)

        return read

    monkeypatch.setattr(Path, "read_bytes", forbid_catalog_reads(original_bytes))
    monkeypatch.setattr(Path, "read_text", forbid_catalog_reads(original_text))
    monkeypatch.setattr(module, "_validate_catalog", unexpected)
    progress = []
    second, metadata = load_guideline_catalog(
        saved_catalog / "paradigms.jsonl",
        return_metadata=True,
        progress_callback=progress.append,
    )
    assert second.to_dict("records") == original
    assert metadata["catalog_sha256"] == first_meta["catalog_sha256"]
    assert metadata["artifacts"][0]["sha256"] != "Caller mutation"
    assert metadata["validation_cache"]["cached_files"] == 1
    assert metadata["validation_cache"]["loaded_files"] == 0
    assert any("reused 1 validated catalogs" in message for message in progress)


@pytest.mark.parametrize("changed", ["catalog", "status", "audit", "missing"])
def test_cached_catalog_changes_fail_closed(saved_catalog, changed):
    load_guideline_catalog(saved_catalog)
    if changed == "catalog":
        with (saved_catalog / "paradigms.jsonl").open("a") as stream:
            stream.write("\n")
    elif changed == "status":
        atomic_json(saved_catalog / "status.json", {"status": "running"})
    elif changed == "audit":
        atomic_json(saved_catalog / "validation.json", {"status": "failed"})
    else:
        (saved_catalog / "paradigms.jsonl").unlink()
    for _ in range(2):
        with pytest.raises((ValueError, FileNotFoundError)):
            load_guideline_catalog(saved_catalog)


def test_same_size_edit_with_restored_mtime_invalidates_cache(saved_catalog):
    load_guideline_catalog(saved_catalog)
    path = saved_catalog / "paradigms.jsonl"
    before = path.stat()
    content = path.read_bytes()
    edited = content.replace(b"Fictional population", b"Different population")
    assert len(edited) == len(content) and edited != content
    path.write_bytes(edited)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(ValueError, match="export hash differs"):
        load_guideline_catalog(saved_catalog)


def test_added_and_removed_audit_files_update_cached_provenance(saved_catalog):
    _, first = load_guideline_catalog(saved_catalog, return_metadata=True)
    assert (
        first["artifacts"][0]["recorded_source_audit"] == "passed_export_hash_verified"
    )
    audit = saved_catalog / "validation.json"
    audit.unlink()
    _, second = load_guideline_catalog(saved_catalog, return_metadata=True)
    assert second["validation_cache"]["loaded_files"] == 1
    assert second["artifacts"][0]["recorded_source_audit"] == "unavailable"
    atomic_json(audit, {"status": "passed", "paradigms_sha256": "wrong"})
    with pytest.raises(ValueError, match="export hash differs"):
        load_guideline_catalog(saved_catalog)
    audit.unlink()
    (saved_catalog / "status.json").unlink()
    load_guideline_catalog(saved_catalog)
    atomic_json(saved_catalog / "status.json", {"status": "running"})
    with pytest.raises(ValueError, match="not complete"):
        load_guideline_catalog(saved_catalog)


def test_new_catalogs_and_new_valid_versions_reload_only_changed_files(
    saved_catalog, catalog_frame, tmp_path, monkeypatch
):
    from matchminer_ai.trials import guideline_catalog as module

    load_guideline_catalog(saved_catalog)
    second = tmp_path / "another.jsonl"
    other = catalog_frame.iloc[:1].copy()
    other["space_trial_id"] = "new-edition"
    other["trial_id"] = "guideline:other:2.0"
    second.write_text(other.to_json(orient="records", lines=True))
    original_validator = module._validate_catalog
    validated = []

    def validate(frame):
        validated.append(frame.space_trial_id.tolist())
        return original_validator(frame)

    monkeypatch.setattr(module, "_validate_catalog", validate)
    first, metadata = load_guideline_catalog(
        [saved_catalog, second], return_metadata=True
    )
    assert validated == [["new-edition"]]
    assert metadata["validation_cache"]["cached_files"] == 1
    assert metadata["validation_cache"]["loaded_files"] == 1
    other["name"] = "Updated population"
    second.write_text(other.to_json(orient="records", lines=True))
    second_result, second_meta = load_guideline_catalog(
        [second, saved_catalog], return_metadata=True
    )
    assert validated == [["new-edition"], ["new-edition"]]
    assert second_result.iloc[0]["name"] == "Updated population"
    assert second_result.space_trial_id.tolist() == ["new-edition", "a", "b", "c"]
    assert first.iloc[-1]["name"] != "Updated population"
    assert second_meta["catalog_sha256"] != metadata["catalog_sha256"]


def test_cached_files_still_check_cross_catalog_uniqueness(saved_catalog):
    load_guideline_catalog(saved_catalog)
    with pytest.raises(ValueError, match="unique"):
        load_guideline_catalog([saved_catalog, saved_catalog])


def test_symlink_uses_target_audit_and_detects_retargeting(saved_catalog, tmp_path):
    link = tmp_path / "linked.jsonl"
    link.symlink_to(saved_catalog / "paradigms.jsonl")
    _, metadata = load_guideline_catalog(link, return_metadata=True)
    assert (
        metadata["artifacts"][0]["recorded_source_audit"]
        == "passed_export_hash_verified"
    )
    atomic_json(saved_catalog / "status.json", {"status": "running"})
    with pytest.raises(ValueError, match="not complete"):
        load_guideline_catalog(link)
    other = tmp_path / "invalid.jsonl"
    other.write_text("{}\n")
    link.unlink()
    link.symlink_to(other)
    with pytest.raises(ValueError, match="missing columns"):
        load_guideline_catalog(link)


def test_changed_catalog_schema_is_revalidated_even_with_matching_audit(saved_catalog):
    load_guideline_catalog(saved_catalog)
    path = saved_catalog / "paradigms.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["diagnostic_workup"][0]["evidence"] = []
    content = "".join(json.dumps(row) + "\n" for row in rows).encode()
    path.write_bytes(content)
    atomic_json(
        saved_catalog / "validation.json",
        {"status": "passed", "paradigms_sha256": digest(content)},
    )
    with pytest.raises(ValueError, match="must retain evidence"):
        load_guideline_catalog(saved_catalog)


def test_refresh_forces_full_validation_and_dataframe_inputs_are_never_cached(
    saved_catalog, catalog_frame, monkeypatch
):
    from matchminer_ai.trials import guideline_catalog as module

    load_guideline_catalog(saved_catalog)
    original = module._validate_catalog
    calls = []

    def validate(frame):
        calls.append(True)
        return original(frame)

    monkeypatch.setattr(module, "_validate_catalog", validate)
    _, metadata = load_guideline_catalog(
        saved_catalog, refresh=True, return_metadata=True
    )
    assert metadata["validation_cache"]["loaded_files"] == 1
    load_guideline_catalog(catalog_frame)
    catalog_frame.loc[0, "clinical_space_summary"] = "Invalid edit"
    with pytest.raises(ValueError, match="differs"):
        load_guideline_catalog(catalog_frame)
    assert len(calls) == 3


def test_catalog_changed_during_validation_is_not_cached(saved_catalog, monkeypatch):
    from matchminer_ai.trials import guideline_catalog as module

    original = module._validate_catalog

    def validate(frame):
        original(frame)
        with (saved_catalog / "paradigms.jsonl").open("a") as stream:
            stream.write("\n")

    monkeypatch.setattr(module, "_validate_catalog", validate)
    with pytest.raises(ValueError, match="changed while loading"):
        load_guideline_catalog(saved_catalog)
    with pytest.raises(ValueError, match="export hash differs"):
        load_guideline_catalog(saved_catalog)


def test_parallel_callers_share_validation_without_sharing_mutable_records(
    saved_catalog, monkeypatch
):
    from concurrent.futures import ThreadPoolExecutor
    from matchminer_ai.trials import guideline_catalog as module

    original = module._validate_catalog
    validated = []

    def validate(frame):
        validated.append(True)
        return original(frame)

    monkeypatch.setattr(module, "_validate_catalog", validate)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(load_guideline_catalog, [saved_catalog] * 4))
    assert len(validated) == 1
    results[0].iloc[0].source["title"] = "Modified"
    assert results[1].iloc[0].source["title"] != "Modified"


@pytest.mark.parametrize(
    "limit", ["_CATALOG_CACHE_MAX_FILES", "_CATALOG_CACHE_MAX_BYTES"]
)
def test_catalog_cache_is_bounded(saved_catalog, monkeypatch, limit):
    from matchminer_ai.trials import guideline_catalog as module

    monkeypatch.setattr(module, limit, 0)
    load_guideline_catalog(saved_catalog)
    _, metadata = load_guideline_catalog(saved_catalog, return_metadata=True)
    assert metadata["validation_cache"]["loaded_files"] == 1


def test_patient_retrieval_reuses_catalog_validation(
    saved_catalog, patients, model_stages
):
    for expected_hits in (0, 1):
        progress = []
        _, metadata = retrieve_guideline_considerations(
            patients,
            saved_catalog,
            top_n=1,
            candidate_k=2,
            return_metadata=True,
            progress_callback=progress.append,
        )
        assert metadata["catalog"]["validation_cache"]["cached_files"] == expected_hits
        assert any(
            "Checking guideline catalog files for changes" in p for p in progress
        )
        assert any("reused" in p for p in progress)


def test_loader_rejects_missing_menus_duplicate_ids_and_changed_space(catalog_frame):
    with pytest.raises(ValueError, match="missing columns"):
        load_guideline_catalog(catalog_frame.drop(columns=["diagnostic_workup"]))
    with pytest.raises(ValueError, match="unique"):
        load_guideline_catalog(pd.concat([catalog_frame, catalog_frame.iloc[:1]]))
    altered = copy.deepcopy(catalog_frame)
    altered.loc[0, "clinical_space_summary"] += " Extra unsupported text."
    with pytest.raises(ValueError, match="differs"):
        load_guideline_catalog(altered)


def test_mixed_legacy_and_current_exports_keep_existing_space_numbers(
    catalog_frame, tmp_path
):
    rows = catalog_frame.to_dict("records")
    rows[1]["clinical_space_number"] = 1
    rows[1]["general_exclusion_criteria"] = "Existing criterion"
    paths = []
    for index, row in enumerate(rows):
        path = tmp_path / f"catalog-{index}.jsonl"
        path.write_text(json.dumps(row) + "\n")
        paths.append(path)
    result = load_guideline_catalog(paths)
    assert result.clinical_space_number.tolist() == [2, 1, 3]
    assert result.general_exclusion_criteria.tolist() == [
        "NA",
        "Existing criterion",
        "NA",
    ]
    assert result.diagnostic_workup.tolist() == catalog_frame.diagnostic_workup.tolist()


def test_trialchecker_orders_top_n_spaces_without_guideline_deduplication(
    catalog_frame, patients, model_stages
):
    progress = []
    result, metadata = retrieve_guideline_considerations(
        patients,
        catalog_frame,
        top_n=2,
        candidate_k=3,
        return_metadata=True,
        progress_callback=progress.append,
    )
    assert result.patient_id.tolist() == ["P2", "P2", "P1", "P1"]
    assert result.space_trial_id.tolist() == ["b", "c", "b", "c"]
    assert result["rank"].tolist() == [1, 2, 1, 2]
    assert result.retrieval_rank.tolist() == [2, 3, 2, 3]
    assert len(model_stages["embedding"]) == 2  # catalog embedded once for the batch
    assert len(model_stages["checker"]) == 1
    assert len(model_stages["checker"][0]) == 6
    assert metadata["retrieval"]["scored_pairs"] == 6
    assert "cancer_history_summary" not in result
    assert result.iloc[0].diagnostic_workup == catalog_frame.iloc[1].diagnostic_workup
    assert any("TrialChecker" in p for p in progress)


def test_candidate_limit_applies_before_checker_and_low_scores_are_preserved(
    catalog_frame, patients, model_stages
):
    result = retrieve_guideline_considerations(
        patients.iloc[:1], catalog_frame, top_n=2, candidate_k=2
    )
    assert result.space_trial_id.tolist() == ["b", "a"]
    assert result.match_quality_pass.tolist() == [True, False]
    assert set(model_stages["checker"][0].space_trial_id) == {"a", "b"}


def test_all_candidates_and_small_catalog_return_bounded_result(
    catalog_frame, patients, model_stages
):
    result = retrieve_guideline_considerations(
        patients.iloc[:1], catalog_frame, top_n=5, candidate_k=None
    )
    assert len(result) == 3
    assert result.space_trial_id.tolist() == ["b", "c", "a"]


@pytest.mark.parametrize("top_n,candidate_k", [(0, 20), (True, 20), (5, 4), (2, 2.5)])
def test_bad_limits_fail_before_model_calls(
    catalog_frame, patients, model_stages, top_n, candidate_k
):
    with pytest.raises(ValueError):
        retrieve_guideline_considerations(
            patients, catalog_frame, top_n=top_n, candidate_k=candidate_k
        )
    assert model_stages["embedding"] == []


def test_missing_checker_scores_fail_closed(
    catalog_frame, patients, model_stages, monkeypatch
):
    def incomplete(frame, **kwargs):
        return pd.DataFrame(
            columns=["patient_id", "space_trial_id", "match_quality_score"]
        ), {}

    monkeypatch.setattr("matchminer_ai.matching.score_match_quality", incomplete)
    with pytest.raises(ValueError, match="exactly one score"):
        retrieve_guideline_considerations(
            patients, catalog_frame, top_n=2, candidate_k=3
        )


def test_different_embedding_models_are_rejected_before_scoring(
    catalog_frame, patients, model_stages, monkeypatch
):
    from matchminer_ai.embedding import embed_for_matching

    def changed_model(frame, **kwargs):
        result, metadata = embed_for_matching(frame, **kwargs)
        metadata = copy.deepcopy(metadata)
        if kwargs["entity_type"] == "trial":
            metadata["model_metadata"]["embedding_model"]["model_sha"] = (
                "different-revision"
            )
        return result, metadata

    monkeypatch.setattr("matchminer_ai.embedding.embed_for_matching", changed_model)
    with pytest.raises(ValueError, match="refusing to mix embeddings"):
        retrieve_guideline_considerations(
            patients, catalog_frame, top_n=2, candidate_k=3
        )
    assert model_stages["checker"] == []


def test_report_renders_input_conditions_sources_and_rank_order(
    catalog_frame, patients, model_stages, tmp_path
):
    patients.loc[0, "cancer_history_summary"] += "\n# 2020: Synthetic therapy"
    result, metadata = retrieve_guideline_considerations(
        patients.iloc[:1], catalog_frame, top_n=2, candidate_k=3, return_metadata=True
    )
    output = write_guideline_considerations_report(
        result,
        patients,
        tmp_path / "report.md",
        metadata=metadata,
        patient_provenance={"synthetic": True},
    )
    text = output.read_text()
    assert "Synthetic patient beta" in text
    assert "> \\# 2020: Synthetic therapy  \n" in text
    assert "Synthetic patient alpha" not in text
    assert "unused_note" not in text
    assert text.index("### 1. Fictional population b") < text.index(
        "### 2. Fictional population c"
    )
    assert "When finding B is present" in text
    assert "Fictional disease state alpha" in text
    assert "#page=2" in text
    assert "not probabilities" in text
    assert "checker-revision" in text
    with pytest.raises(ValueError, match="outside the code"):
        write_guideline_considerations_report(
            result, patients, __file__.replace(".py", ".md")
        )


@pytest.fixture
def cached_stages(model_stages, monkeypatch):
    from types import SimpleNamespace
    from matchminer_ai.embedding import inference

    revision = SimpleNamespace(_commit_hash="actual-loaded-revision")
    tokens = {"model": {"vocab": {"synthetic": 1}}, "padding": {"length": 3}}
    first = SimpleNamespace(
        auto_model=SimpleNamespace(config=revision),
        tokenizer=SimpleNamespace(
            backend_tokenizer=SimpleNamespace(to_str=lambda: json.dumps(tokens)),
            special_tokens_map={"pad_token": "[PAD]"},
        ),
    )
    encoder = SimpleNamespace(_first_module=lambda: first, _modules={})
    monkeypatch.setattr(inference, "_get_embedding_model", lambda *args: encoder)
    return model_stages, revision, tokens


def cached_retrieve(patients, catalog, directory, **kwargs):
    return retrieve_guideline_considerations(
        patients,
        catalog,
        top_n=2,
        candidate_k=3,
        embedding_cache_dir=directory,
        return_metadata=True,
        **kwargs,
    )


def test_saved_vectors_reused_with_reordered_catalog(
    catalog_frame, patients, cached_stages, tmp_path
):
    import shutil
    from pathlib import Path

    calls, _, tokens = cached_stages
    catalog_frame["clinical_space_number"] = [1, 2, 3]
    first, metadata = cached_retrieve(patients, catalog_frame, tmp_path / "cache")
    assert metadata["guideline_embedding_cache"]["embedded_texts"] == 3
    source = metadata["guideline_embedding_cache"]["path"]
    target = tmp_path / "copied"
    target.mkdir()
    shutil.copyfile(source, target / Path(source).name)
    tokens["padding"]["length"] = 40  # Batch-dependent padding must not invalidate.
    second, metadata = cached_retrieve(
        patients.iloc[::-1], catalog_frame.iloc[::-1], target
    )
    assert metadata["guideline_embedding_cache"]["cached_spaces"] == 3
    assert metadata["guideline_embedding_cache"]["embedded_texts"] == 0
    assert [c[0] for c in calls["embedding"]] == ["patient", "trial", "patient"]
    pd.testing.assert_frame_equal(
        first.sort_values(["patient_id", "rank"]).reset_index(drop=True),
        second.sort_values(["patient_id", "rank"]).reset_index(drop=True),
    )
    disk = Path(metadata["guideline_embedding_cache"]["path"]).read_bytes()
    assert b"Synthetic patient" not in disk
    assert catalog_frame.iloc[0].clinical_space_summary.encode() not in disk


def test_only_added_and_changed_guideline_texts_are_embedded(
    catalog_frame, patients, cached_stages, tmp_path
):
    calls, _, _ = cached_stages
    cached_retrieve(patients, catalog_frame.iloc[:2], tmp_path)
    _, metadata = cached_retrieve(patients, catalog_frame, tmp_path)
    assert metadata["guideline_embedding_cache"]["cached_spaces"] == 2
    assert calls["embedding"][-1][1].space_trial_id.tolist() == ["c"]
    changed = catalog_frame.copy()
    changed.at[0, "space"] = dict(
        changed.iloc[0].space, cancer_burden_allowed="Changed state"
    )
    changed.at[0, "clinical_space_summary"] = format_space(changed.iloc[0].space)
    _, metadata = cached_retrieve(patients, changed, tmp_path)
    assert metadata["guideline_embedding_cache"]["cached_spaces"] == 2
    assert calls["embedding"][-1][1].space_trial_id.tolist() == ["a"]


@pytest.mark.parametrize("change", ["revision", "tokenizer", "prompt", "length"])
def test_encoder_changes_never_reuse_incompatible_vectors(
    catalog_frame, patients, cached_stages, tmp_path, monkeypatch, change
):
    from matchminer_ai.config import load_default_preset
    from matchminer_ai.embedding import inference

    _, revision, tokens = cached_stages
    config = load_default_preset()
    _, before = cached_retrieve(patients, catalog_frame, tmp_path, config=config)
    if change == "revision":
        revision._commit_hash = "new-loaded-revision"
    elif change == "tokenizer":
        tokens["model"]["vocab"]["new"] = 2
    elif change == "prompt":
        monkeypatch.setattr(inference, "_load_prompt_text", lambda _: "New prompt")
    else:
        config.embedding["max_seq_length"] -= 1
    _, after = cached_retrieve(patients, catalog_frame, tmp_path, config=config)
    assert after["guideline_embedding_cache"]["cached_spaces"] == 0
    assert after["guideline_embedding_cache"]["embedded_texts"] == 3
    assert (
        before["guideline_embedding_cache"]["path"]
        != after["guideline_embedding_cache"]["path"]
    )


def test_damaged_vector_is_recomputed_and_repaired(
    catalog_frame, patients, cached_stages, tmp_path
):
    import sqlite3

    _, metadata = cached_retrieve(patients, catalog_frame, tmp_path)
    with sqlite3.connect(metadata["guideline_embedding_cache"]["path"]) as db:
        db.execute("UPDATE embeddings SET vector = ? WHERE rowid = 1", (b"damaged",))
    _, metadata = cached_retrieve(patients, catalog_frame, tmp_path)
    assert metadata["guideline_embedding_cache"]["cached_spaces"] == 2
    assert metadata["guideline_embedding_cache"]["embedded_texts"] == 1
    _, metadata = cached_retrieve(patients, catalog_frame, tmp_path)
    assert metadata["guideline_embedding_cache"]["cached_spaces"] == 3


def test_unreadable_cache_does_not_prevent_fresh_retrieval(
    catalog_frame, patients, cached_stages, tmp_path
):
    from pathlib import Path

    first, metadata = cached_retrieve(patients, catalog_frame, tmp_path)
    Path(metadata["guideline_embedding_cache"]["path"]).write_bytes(b"broken database")
    second, metadata = cached_retrieve(patients, catalog_frame, tmp_path)
    assert metadata["guideline_embedding_cache"]["status"] == "unavailable"
    assert metadata["guideline_embedding_cache"]["embedded_texts"] == 3
    pd.testing.assert_frame_equal(first, second)


def test_unversioned_local_encoder_bypasses_persistent_cache(
    catalog_frame, patients, cached_stages, tmp_path
):
    from matchminer_ai.config import load_default_preset

    config = load_default_preset()
    local = tmp_path / "local-model"
    local.mkdir()
    config.embedding["model_path"] = str(local)
    _, metadata = cached_retrieve(
        patients, catalog_frame, tmp_path / "cache", config=config
    )
    assert metadata["guideline_embedding_cache"]["status"] == "unverified_encoder"
    assert metadata["guideline_embedding_cache"]["path"] is None
    assert not (tmp_path / "cache").exists()


def test_review_issues_survive_ranking_and_report(catalog_frame, patients, model_stages, tmp_path):
    issue = {"version": "citation-review-v1", "status": "unresolved",
             "issues": ["Synthetic source supports only a narrower test."]}
    records = copy.deepcopy(catalog_frame.to_dict("records"))
    for row in records:
        row["citation_review"] = copy.deepcopy(issue)
        row["diagnostic_workup"][0].update(evidence=[], citation_review=copy.deepcopy(issue))
    matches = retrieve_guideline_considerations(patients, pd.DataFrame(records), candidate_k=2, top_n=1)
    assert matches.iloc[0].diagnostic_workup[0]["citation_review"] == issue
    report = write_guideline_considerations_report(matches, patients, tmp_path / "review.md").read_text()
    assert "Population source support unresolved" in report
    assert "Source support unresolved" in report
    assert "Synthetic source supports only a narrower test" in report
