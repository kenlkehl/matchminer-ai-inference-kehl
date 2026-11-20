"""The shared core (``matchminer_ai.apps.dashboard.core``): settings, corpus loading, round selection and vector identity, search,
cancer types, the notes index, the decision model's wrapping and score cache, and jobs."""
import json
import os
import shutil

import numpy as np
import pytest

from .conftest import COHORTS, PATIENTS, encoder_identity_for, write_json, write_round
from matchminer_ai.apps.dashboard.core.cache import JSONCache
from matchminer_ai.apps.dashboard.core.corpus import Corpus, parse_header
from matchminer_ai.apps.dashboard.core.settings import BASE_DECISION_MODEL, Settings
from matchminer_ai.apps.dashboard.core.vectors import (VectorStore, discover_rounds, encoder_settings, find_hub_cache, npy_shape,
                                    select_round, validate_round)


def corpus_for(settings):
    return Corpus(settings.path("corpus"), settings.path("patient_metadata"), JSONCache(settings.cache_dir))


# ---------------------------------------------------------------- settings

def test_defaults_point_at_the_synthetic_run(tmp_path, site_config, monkeypatch):
    run = tmp_path / "data" / "no_phi" / "synthetic_run"  # under the data root without a site value
    s = Settings(cache_dir=tmp_path)
    assert s.synthetic_run == run and "no_phi" in str(s.synthetic_run)
    assert s.path("corpus") == run / "trialspace" / "trialspace_corpus"
    assert s.host == "127.0.0.1" and s.device == "cpu" and not s.allow_downloads
    assert all(str(p).startswith(str(run)) for p in s.paths.values())
    site_config.write_text(site_config.read_text() + f"  synthetic_run: {tmp_path / 'site-run'}\n")
    os.utime(site_config, (site_config.stat().st_atime, site_config.stat().st_mtime + 5))  # a new stamp
    assert Settings(cache_dir=tmp_path).synthetic_run == tmp_path / "site-run"
    json.dumps(s.describe())


def test_environment_and_command_line_overrides(tmp_path, monkeypatch):
    from matchminer_ai.apps.dashboard.settings import parse_settings

    store = ["--store-dir", str(tmp_path / "store")]
    monkeypatch.setenv("MMAI_DASH_SYNTHETIC_RUN", str(tmp_path / "env-root"))
    monkeypatch.setenv("MMAI_DASH_THREADS", "4")
    s = parse_settings(store + ["--port", "9999", "--device", "cuda:3", "--cache-dir", str(tmp_path / "c")])
    assert s.port == 9999 and s.device == "cuda:3" and s.threads == 4 and s.cache_dir == tmp_path / "c"
    assert s.path("corpus") == tmp_path / "env-root" / "trialspace" / "trialspace_corpus"
    assert (parse_settings(store + ["--synthetic-run", str(tmp_path / "cli"), "--device", "cpu"]).synthetic_run
            == tmp_path / "cli")


def test_decision_model_default_and_fallback(tmp_path, settings):
    model, revision, why = settings.resolved_decision_model()
    assert model == BASE_DECISION_MODEL and revision and "fallback" in why
    smoke = settings.synthetic_run / "smoke_tests/round1_labels_fit2_20261107/decision_fit/model"
    smoke.mkdir(parents=True)
    (smoke / "config.json").write_text("{}")
    model, revision, why = settings.resolved_decision_model()
    assert model == str(smoke) and revision is None and "fit 2" in why and "not trained yet" in why
    run = settings.synthetic_run / "trialspace/models/decision_model"  # the run's own model, once its receipt is complete
    run.mkdir(parents=True)
    files = {"decision_contract.json": b"{}", "model.safetensors": b"weights"}
    for name, data in files.items():
        (run / name).write_bytes(data)
    assert "fit 2" in settings.resolved_decision_model()[2]
    receipt = {"format": "mmai-stage-receipt-v1", "kind": "decision-model",
               "outputs": {name: len(data) + (name == "model.safetensors") for name, data in files.items()}}
    (run / "stage_receipt.json").write_text(json.dumps(receipt))
    assert settings.resolved_decision_model()[0] == str(smoke)  # a size differs: still copying
    receipt["outputs"]["model.safetensors"] = len(files["model.safetensors"])
    (run / "stage_receipt.json").write_text(json.dumps(receipt))
    model, revision, why = settings.resolved_decision_model()
    assert model == str(run) and revision is None and "trained decision model" in why


def test_main_refuses_non_loopback_host(tmp_path):
    from matchminer_ai.apps.dashboard.__main__ import main

    with pytest.raises(SystemExit):
        main(["--host", "0.0.0.0", "--store-dir", str(tmp_path / "store"), "--device", "cpu"])


# ---------------------------------------------------------------- corpus

def test_corpus_rows_headers_metadata_and_digests(settings, dataset):
    corpus = corpus_for(settings)
    assert len(corpus.patients) == len(PATIENTS) and len(corpus.cohorts) == len(COHORTS)
    assert corpus.digests == dataset["digests"]
    assert corpus.receipt_id == "corpus-receipt-1"
    summary = corpus.patient_summary("patient_a")
    assert summary["age"] == "62" and summary["cancer_type"] == "Lung cancer" and summary["cls"] == "positive"
    assert summary["source_cohort"] == "NCT00000001-cohort-0"
    assert corpus.from_cohort["NCT00000001-cohort-0"] == ["patient_a", "patient_b"]
    assert corpus.trial_cohorts[("NCT00000002", "protocol")] == ["NCT00000002-cohort-0", "NCT00000002-cohort-1"]


def test_corpus_digests_are_cached_per_file_stamp(settings, monkeypatch):
    corpus_for(settings)
    import matchminer_ai.labeling.files as files

    monkeypatch.setattr(files, "digest", lambda value: (_ for _ in ()).throw(AssertionError("recomputed")))
    assert corpus_for(settings).digests["patients"]


def test_patient_and_cohort_search(settings):
    corpus = corpus_for(settings)
    assert corpus.search_patients("lung")["total"] == 2
    assert [r["id"] for r in corpus.search_patients("breast", cls="positive")["rows"]] == ["patient_c"]
    assert corpus.search_patients(sex="male")["total"] == 2
    assert corpus.search_patients("NCT00000002-cohort-1")["total"] == 2
    assert corpus.search_cohorts(source="ctgov")["total"] == 2
    assert [r["id"] for r in corpus.search_cohorts("her2-negative")["rows"]] == ["NCT00000002-cohort-1"]
    assert corpus.search_cohorts("egfr", titles={"NCT00000001": "EGFR trial"})["total"] == 2


def ids(found):
    return [r["id"] for r in found["rows"]]


def test_list_search_is_regex_terms_over_every_whole_text(settings):
    corpus = corpus_for(settings)
    # The whole summary (key section and other relevant history) of every patient, not only the header fields.
    assert ids(corpus.search_patients("L858R")) == ["patient_a"]
    assert ids(corpus.search_patients("myocardial")) == ["patient_e"]
    assert corpus.search_patients("myocardial", scope="fields")["total"] == 0
    # Case-insensitive regex terms that must all match; quotes keep spaces; ^ anchors a line.
    assert ids(corpus.search_patients("lung|colorectal")) == ["patient_a", "patient_b", "patient_e"]
    assert ids(corpus.search_patients("adenocarcinoma LUNG")) == ["patient_a"]
    assert ids(corpus.search_patients('"invasive (ductal|lobular)"')) == ["patient_c", "patient_d"]
    assert ids(corpus.search_patients(r"her2\s+pos")) == ["patient_c"]
    assert ids(corpus.search_patients("^histology: squamous")) == ["patient_b"]
    assert ids(corpus.search_patients("^squamous", scope="fields")) == ["patient_b"]
    page = corpus.search_patients("adenocarcinoma", offset=1, limit=1)
    assert (page["total"], page["of"], ids(page)) == (2, 5, ["patient_e"])
    with pytest.raises(ValueError, match="invalid pattern '\\(abc': missing \\)"):
        corpus.search_patients("(abc")
    with pytest.raises(ValueError, match="scope"):
        corpus.search_patients("lung", scope="everything")
    # Cohorts: the key and boilerplate criteria too.
    assert ids(corpus.search_cohorts("pregnancy")) == ["NCT00000001-ctgov-cohort-0"]
    assert corpus.search_cohorts("pregnancy", scope="fields")["total"] == 0
    assert ids(corpus.search_cohorts("egfr inhibitor")) == ["NCT00000001-cohort-0"]


def test_text_index_terms_and_chunks():
    from matchminer_ai.apps.dashboard.core.textsearch import TextIndex, parse_terms

    assert parse_terms('EGFR "exon 19|L858R" "still typing') == ["EGFR", "exon 19|L858R", "still typing"]
    assert parse_terms('  ""  ') == []
    texts = [f"row {i}\nvalue {i % 3}" for i in range(11)]
    one, many = TextIndex(texts, parts=1), TextIndex(texts, parts=4)
    assert len(many.chunks) == 4 and many.match("^value [12]$").tolist() == one.match("^value [12]$").tolist()
    assert many.match("value 0").sum() == 5  # two terms: "value" and "0" (row 10 has a 0)
    assert many.match('"value 0"').sum() == 4 and TextIndex([]).match("x").tolist() == []


def test_parse_header_tolerates_missing_lines():
    assert parse_header("Age: 50\nCancer type: Melanoma\nSomething else") == {"age": "50", "cancer_type": "Melanoma"}


# ---------------------------------------------------------------- rounds and vectors

def test_round_one_is_selected_and_its_encoder_identity_rebuilt(settings):
    corpus = corpus_for(settings)
    info, skipped = select_round(settings.path("trialspace"), corpus)
    assert info.number == 1 and skipped == []
    assert info.model.kind == "hub" and info.model.model == "google/embeddinggemma-2"
    rebuilt, same = encoder_settings(info, cache_dir=None)
    assert same and rebuilt.prompts["patient_summary"].endswith("Patient history:\n")


def test_rebuilt_identity_refuses_a_different_encoder_implementation(settings):
    info, _ = select_round(settings.path("trialspace"), corpus_for(settings))
    info.manifest["encoder"] = dict(info.manifest["encoder"], implementation="mmai-encoder-v0")
    assert encoder_settings(info, cache_dir=None)[1] is False


def test_higher_round_without_receipt_is_ignored(settings):
    (settings.path("trialspace") / "trialspace_round2" / "mining").mkdir(parents=True)
    assert discover_rounds(settings.path("trialspace")) == [1]


def test_round_two_with_fitted_model_is_preferred(settings, dataset):
    fitted = settings.path("trialspace") / "models" / "trialspace_round1"
    receipt = "a" * 64
    write_json(fitted / "stage_receipt.json", {"format": "mmai-stage-receipt-v1", "kind": "finetune", "receipt_id": receipt})
    encoder = dict(encoder_identity_for(model=receipt, revision=None))
    write_round(dataset["root"], 2, "corpus-receipt-1", dataset["digests"], encoder, seed=7)
    corpus = corpus_for(settings)
    info, skipped = select_round(settings.path("trialspace"), corpus)
    assert info.number == 2 and skipped == []
    assert info.model.kind == "local" and info.model.model == str(fitted)
    rebuilt, same = encoder_settings(info, cache_dir=None)
    assert same and rebuilt.model_identity == receipt


@pytest.mark.parametrize("break_it, reason", [
    ("corpus", "corpus"), ("shape", "shape"), ("encoder", "encoder differs"), ("digest", "digest"), ("size", "size"),
    ("model", "fitted TrialSpace"), ("kind", "mine receipt")])
def test_invalid_higher_round_falls_back_with_a_reason(settings, dataset, break_it, reason):
    root, digests = dataset["root"], dict(dataset["digests"])
    encoder, corpus_id, rows = encoder_identity_for(), "corpus-receipt-1", None
    if break_it == "corpus":
        corpus_id = "another-corpus"
    if break_it == "shape":
        rows = len(PATIENTS) - 1
    if break_it == "digest":
        digests["patients"] = "0" * 64
    if break_it == "model":
        encoder = encoder_identity_for(model="b" * 64, revision=None)
    mining = write_round(root, 2, corpus_id, digests, encoder, seed=3, patients=rows)
    receipt = json.loads((mining / "stage_receipt.json").read_text())
    if break_it == "encoder":
        receipt["identity"]["encoder"] = dict(encoder, max_seq_length=4096)
    if break_it == "size":
        receipt["outputs"]["patients.npy"] += 1
    if break_it == "kind":
        receipt["kind"] = "label"
    write_json(mining / "stage_receipt.json", receipt)
    info, skipped = select_round(settings.path("trialspace"), corpus_for(settings))
    assert info.number == 1
    assert skipped and skipped[0]["round"] == 2 and reason in skipped[0]["reason"]


def test_no_valid_round(settings):
    shutil.rmtree(settings.path("trialspace") / "trialspace_round1")
    info, skipped = select_round(settings.path("trialspace"), corpus_for(settings))
    assert info is None and skipped == []


def test_exact_search_filters_groups_and_ties(settings):
    corpus = corpus_for(settings)
    info = validate_round(1, settings.path("trialspace"), corpus)
    store = VectorStore(info)
    query = store.patients[0]
    expected = np.argsort(-(store.cohorts @ query), kind="stable")[:3].tolist()
    assert [i for i, _ in store.search(query, "cohorts", 3)] == expected
    only_ctgov = np.array([row["source"] == "ctgov" for row in corpus.cohorts])
    assert all(corpus.cohorts[i]["source"] == "ctgov" for i, _ in store.search(query, "cohorts", 5, only_ctgov))
    grouped = store.search(query, "cohorts", 5, group=[row["trial_id"] for row in corpus.cohorts])
    assert len(grouped) == 2 and len({corpus.cohorts[i]["trial_id"] for i, _ in grouped}) == 2
    tie = np.zeros_like(query)
    assert [i for i, _ in store.search(tie, "patients", 3)] == [0, 1, 2]  # ties keep row order
    with pytest.raises(ValueError):
        store.scores(np.ones(3, dtype=np.float32), "cohorts")


def test_npy_header_and_hub_cache(tmp_path, settings):
    info = validate_round(1, settings.path("trialspace"), corpus_for(settings))
    assert npy_shape(info.directory / "patients.npy") == ((len(PATIENTS), 8), "float32")
    snapshot = tmp_path / "hf" / "models--google--embeddinggemma-2" / "snapshots" / "rev1"
    snapshot.mkdir(parents=True)
    assert find_hub_cache([tmp_path / "hf"], "google/embeddinggemma-2", "rev1") is None
    (snapshot / "model.safetensors").write_text("x")
    assert find_hub_cache([tmp_path / "nowhere", tmp_path / "hf"], "google/embeddinggemma-2", "rev1") == tmp_path / "hf"


def test_encoder_client_reports_why_it_is_unavailable(settings):
    from matchminer_ai.apps.dashboard.core.encoder import EncoderClient, EncoderUnavailable

    info = validate_round(1, settings.path("trialspace"), corpus_for(settings))
    client = EncoderClient(info, settings, settings.cache_dir / "w.log")
    assert not client.available and "embedding Python not found" in client.problem
    with pytest.raises(EncoderUnavailable):
        client.encode(["text"], "patient_summary")
    settings.embedding_python = settings.cache_dir  # exists, but the weights are not cached and downloads are off
    assert "weights not found" in EncoderClient(info, settings, settings.cache_dir / "w.log").problem


# ---------------------------------------------------------------- cancer types

@pytest.mark.parametrize("text, types", [
    ("Locally advanced or metastatic NSCLC.", {"lung_any", "lung_nsclc"}),
    ("Extensive-stage small cell lung cancer (ES-SCLC)", {"lung_any", "lung_sclc"}),
    ("Patients with ovarian, fallopian tube, or primary peritoneal cancer", {"ovarian"}),
    ("HER2-positive gastric or gastroesophageal junction adenocarcinoma", {"gastric"}),
    ("Squamous cell carcinoma of the uterine cervix", {"cervical"}),
    ("Liver metastases from colorectal cancer", {"colorectal"}),
    ("Advanced solid tumors except breast cancer", {"solid_any"}),
    ("including but not limited to breast cancer, gastric cancer", {"breast", "gastric"}),
    ("Advanced malignant solid tumors (except sarcoma), including melanoma", {"solid_any", "melanoma"}),
    ("Measurable disease per Response Evaluation Criteria in Solid Tumors (RECIST) v1.1", set()),
    ("ALK (anaplastic lymphoma kinase) rearranged NSCLC", {"lung_any", "lung_nsclc"}),
    ("Relapsed B-ALL or Philadelphia chromosome-positive ALL", {"leukemia"}),
    ("ALL of the following must be met; women who are breastfeeding are excluded", set()),
    ("Adequate renal function", set()),
])
def test_cancer_type_patterns(text, types):
    from matchminer_ai.apps.dashboard.core.cancer_types import tag_text

    assert tag_text(text) == types


@pytest.mark.parametrize("cancer, histology, types", [
    ("Lung cancer", "Small cell carcinoma", ["lung_any", "lung_sclc"]),
    ("Lung cancer", "Invasive adenocarcinoma, acinar predominant", ["lung_any", "lung_nsclc"]),
    ("Cecal adenocarcinoma", "Moderately differentiated adenocarcinoma", ["colorectal"]),
    ("Urothelial carcinoma of the right renal pelvis", "High-grade urothelial carcinoma", ["bladder"]),
    ("Cervical squamous cell carcinoma", "Invasive squamous cell carcinoma", ["cervical"]),
    ("Carcinoma of unknown primary", "Adenocarcinoma", ["unknown_primary"]),
    ("Metastatic solid tumor", "", []),  # no basket types for patients
])
def test_patient_cancer_types_from_the_header_lines(cancer, histology, types):
    from matchminer_ai.apps.dashboard.core.cancer_types import patient_types

    assert patient_types({"cancer_type": cancer, "histology": histology}) == types


def test_cancer_types_read_only_key_inclusion_lines_and_the_name():
    from matchminer_ai.apps.dashboard.core.cancer_types import cohort_types

    row = {"cohort_name": "Cohort B (melanoma)", "items": [
        {"text": "Metastatic NSCLC.", "criterion_type": "inclusion", "boilerplate": False},
        {"text": "Breast cancer survivors allowed.", "criterion_type": "inclusion", "boilerplate": True},
        {"text": "Prior prostate cancer.", "criterion_type": "exclusion", "boilerplate": False}]}
    assert cohort_types(row) == ["lung_any", "lung_nsclc", "melanoma"]


# ---------------------------------------------------------------- notes

def test_notes_index_builds_once_and_reads_one_patient(settings):
    from matchminer_ai.apps.dashboard.core.notes import NotesIndex

    cache = JSONCache(settings.cache_dir)
    notes = NotesIndex(settings.path("notes"), cache)
    assert notes.ensure() in ("building", "ready")
    notes.thread.join(10)
    assert notes.state == "ready" and notes.index["patient_a"] == [0, 1]
    rows = notes.notes("patient_a")
    assert [r["date"] for r in rows] == ["2024-03-01", "2024-05-02"] and rows[0]["text"] == "Biopsy: adenocarcinoma."
    assert notes.notes("patient_e") == []
    assert NotesIndex(settings.path("notes"), cache).state == "ready"  # reused from the cache


# ---------------------------------------------------------------- decision model and jobs

def corpus_pair(corpus, pid, cid):
    from matchminer_ai.apps.dashboard.core.decisions import PairText

    p, c = corpus.patient(pid), corpus.cohort(cid)
    return PairText(pid, cid, p["text"], p.get("boilerplate") or "", c["text"], c.get("boilerplate") or "", c.get("items"))


def test_scores_map_columns_are_cached_and_skip_empty_boilerplate(settings, fake_model, tmp_path):
    from matchminer_ai.apps.dashboard.core.cache import ScoreCache
    from matchminer_ai.apps.dashboard.core.decisions import DecisionService

    corpus = corpus_for(settings)
    service = DecisionService(settings, ScoreCache(tmp_path / "s.sqlite"), settings.cache_dir)
    pairs = [corpus_pair(corpus, "patient_a", "NCT00000001-cohort-0"), corpus_pair(corpus, "patient_a", "NCT00000002-cohort-0")]
    scores = {s["cohort_id"]: s for s in service.score(pairs)}
    assert scores["NCT00000001-cohort-0"]["match"]["score"] == 0.9 and scores["NCT00000001-cohort-0"]["match"]["pass"]
    assert scores["NCT00000002-cohort-0"]["match"]["reasonableness"] == 0.1
    assert scores["NCT00000001-cohort-0"]["boilerplate"] == {"p": 0.7, "pass": True}
    assert "skipped" in scores["NCT00000002-cohort-0"]["boilerplate"]  # that cohort has no boilerplate criteria
    assert fake_model == {"match": 2, "boilerplate": 1, "criteria": 0, "attribution": 0}
    assert service.score(pairs) == list(scores.values()) and fake_model["match"] == 2  # answered from the score cache
    assert service.cached_scores(pairs)[0]["match"]["score"] == 0.9
    lines = service.criteria(pairs[0])["lines"]
    assert [line["number"] for line in lines] == [1, 2, 3] and [line["fails"] for line in lines] == [False, False, True]
    assert service.criteria(pairs[0])["cached"] and fake_model["criteria"] == 1


def test_decision_contract_warnings(settings, tmp_path):
    from matchminer_ai.apps.dashboard.core.cache import ScoreCache
    from matchminer_ai.apps.dashboard.core.decisions import DecisionService

    model = tmp_path / "ckpt"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "decision_contract.json").write_text(json.dumps({"fingerprint": "f" * 64, "questions_sha256": "q" * 64,
                                                              "states": {"match": 10, "boilerplate": 0, "criterion": 0}}))
    settings.decision_model = str(model)
    service = DecisionService(settings, ScoreCache(tmp_path / "s.sqlite"), settings.cache_dir)
    assert service.untrained_families() == ["boilerplate", "criterion"]
    assert any("different question texts" in w for w in service.warnings())
    assert service.config.raw["decision_model"]["model_name"] == str(model)
    assert service.config.raw["decision_model"]["device"] == "cpu"
    assert service.describe()["fingerprint"] == "f" * 64


def test_jobs_capture_progress_and_errors():
    from matchminer_ai.apps.dashboard.core.jobs import JobManager

    jobs = JobManager()

    def work(progress):
        progress(1, 2, "half")
        return {"ok": True}

    job = jobs.submit("t", "label", work)
    jobs.pool.shutdown(wait=True)
    assert job.status == "done" and job.result == {"ok": True} and (job.done, job.total) == (1, 2)
    jobs = JobManager()
    failed = jobs.submit("t", "label", lambda progress: 1 / 0)
    jobs.pool.shutdown(wait=True)
    assert failed.status == "error" and "ZeroDivisionError" in failed.error
