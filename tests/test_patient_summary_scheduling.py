"""Continuous and round scheduling of serial patient summaries, and resuming them."""

import asyncio
import hashlib
import json
from collections import Counter

import pandas as pd
import pytest

from matchminer_ai.config import load_default_preset
from matchminer_ai.llm.backends import LLMGenerationResult
from matchminer_ai.llm.prompt_rendering import Prompt
from matchminer_ai.llm.remote_inference import ModelResult, RemoteRequestSession
from matchminer_ai.patients.checkpoints import (
    CHUNK_JOURNAL_FILENAME,
    ChunkJournal,
    load_chunk_journal,
    save_prepared_chunks,
    save_round_checkpoint,
)
from matchminer_ai.patients.summarize import summarize_patient_notes
from test_patient_summarization import _config

NOTES = pd.DataFrame([{"patient_id": "unused", "note_text": "x", "note_date": "2024-01-01"}])
METADATA = {"model_name": "fake-model", "model_sha": "sha"}


def _prepared(chunks):
    return pd.DataFrame(
        [
            {
                "patient_id": patient_id,
                "chunk_index": index,
                "first_date": f"2024-01-{index + 1:02d}",
                "last_date": f"2024-01-{index + 1:02d}",
                "chunk_text": text,
            }
            for patient_id, texts in chunks.items()
            for index, text in enumerate(texts)
        ]
    )


def _answer(prompt_text, times_asked):
    """Deterministic: depends only on the prompt (its prior and chunk) and the attempt."""
    chunk = prompt_text.rsplit("=>", 1)[1]
    if "[error]" in chunk:
        return "ERROR: request failed", "", "error"
    if "[draft]" in chunk and times_asked == 1:
        return "Wait, I should check the dates.", "drafting", "stop"
    digest = hashlib.sha256(prompt_text.encode()).hexdigest()[:8]
    return (f"Summary {digest} through {chunk}\nBoilerplate conditions:\nNone",
            f"reasoning {digest}", "stop")


class _PromptPool:
    """Stands in for the prompt-building processes: the prompt shows its prior and chunk."""

    def map(self, func, work_items, chunksize=1):
        return [
            Prompt(row_idx=item.row_idx, max_tokens=7,
                   prompt_text=f"{item.prior_summary_text}=>{item.chunk_text}",
                   messages=[{"role": "user", "content": f"{item.prior_summary_text}=>{item.chunk_text}"}])
            for item in work_items
        ]


class _Session:
    def __init__(self, backend):
        self.backend, self.capacity = backend, backend.capacity
        self.model_metadata = dict(METADATA)

    async def __aenter__(self):
        self.backend.sessions += 1
        return self

    async def __aexit__(self, *exc):
        return None

    async def generate(self, prompt, llm_config):
        self.backend.sent.append(prompt.prompt_text)
        await self.backend.before_answer(prompt)
        summary, reasoning, finish = self.backend.answer(prompt)
        return ModelResult(prompt.row_idx, reasoning, summary, finish)


class _Backend:
    """A remote-style backend: batches for rounds, a request session for continuous."""

    def __init__(self, capacity=2):
        self.capacity, self.sessions, self.batches = capacity, 0, 0
        self.asked, self.sent = Counter(), []

    def answer(self, prompt):
        self.asked[prompt.prompt_text] += 1
        return _answer(prompt.prompt_text, self.asked[prompt.prompt_text])

    async def before_answer(self, prompt):
        await asyncio.sleep(0)

    def generate_llm_outputs(self, *, prompt_list, llm_config, model_metadata_cache_dir=None):
        self.batches += 1
        self.sent.extend(prompt.prompt_text for prompt in prompt_list)
        answers = [self.answer(prompt) for prompt in prompt_list]
        return LLMGenerationResult(
            final_outputs=[a[0] for a in answers], reasoning_outputs=[a[1] for a in answers],
            finish_reasons=[a[2] for a in answers], raw_outputs=[], model_metadata=dict(METADATA))

    def open_request_session(self, llm_config, model_metadata_cache_dir=None):
        return _Session(self)


def _run(monkeypatch, checkpoint_dir, chunks, *, backend, scheduling="continuous",
         existing=None, debug=True):
    monkeypatch.setattr("matchminer_ai.patients.summarize.prep_prompt_pool",
                        lambda patient_config, n_workers: _PromptPool())
    monkeypatch.setattr("matchminer_ai.patients.summarize.shutdown_prompt_pool", lambda pool: None)
    monkeypatch.setattr("matchminer_ai.patients.summarize.get_llm_backend", lambda config: backend)
    monkeypatch.setattr(
        "matchminer_ai._qc.patients.patient_summary_qc_report",
        lambda rows, **kwargs: pd.DataFrame([kwargs["failed_inference_qc_artifact"],
                                             kwargs["noninformative_summary_qc_artifact"]]))
    if chunks is not None:
        save_prepared_chunks(checkpoint_dir, _prepared(chunks))
    config = _config(debug_mode=debug)
    config.patient = {**config.patient, "scheduling": scheduling}
    return summarize_patient_notes(NOTES, config=config, existing_summaries=existing,
                                   checkpoint_dir=checkpoint_dir, return_qc=True)


def _journal(checkpoint_dir):
    return sorted(load_chunk_journal(checkpoint_dir),
                  key=lambda row: (row["patient_id"], row["chunk_index"]))


def test_continuous_and_round_scheduling_give_identical_results(monkeypatch, tmp_path):
    """Uneven chains, a seeded prior, a quality retry and a failure: byte-identical outputs."""
    chunks = {"P1": ["a", "b", "c [draft]", "d"], "P2": ["e"], "P3": ["f", "g [error]", "h"],
              "P4": ["i", "j"]}
    existing = pd.DataFrame([
        {"patient_id": "P4", "patient_summary": "Existing P4 history"},
        {"patient_id": "P9", "patient_summary": "Retained P9 history\nBoilerplate conditions:\nNone"},
    ])
    runs = {}
    for scheduling in ("continuous", "rounds"):
        backend = _Backend(capacity=2)
        result, metadata, qc = _run(monkeypatch, tmp_path / scheduling, chunks, backend=backend,
                                    scheduling=scheduling, existing=existing)
        metadata["config_snapshot"]["patient"].pop("scheduling")
        runs[scheduling] = dict(
            result=result.to_json(orient="records"), qc=qc.to_json(orient="records"),
            metadata=json.dumps(metadata, sort_keys=True, default=str),
            journal=_journal(tmp_path / scheduling), asked=backend.asked, backend=backend)

    continuous, rounds = runs["continuous"], runs["rounds"]
    # Each scheduler really ran: requests on their own, versus one batch per attempt.
    assert continuous["backend"].sessions == 1 and continuous["backend"].batches == 0
    assert rounds["backend"].sessions == 0 and rounds["backend"].batches > 0
    for key in ("result", "qc", "metadata", "journal", "asked"):
        assert continuous[key] == rounds[key], key

    assert [row["patient_id"] for row in json.loads(continuous["result"])] == ["P1", "P2", "P4", "P9"]
    asked = continuous["asked"]
    assert asked["None=>f"] == 1 and sum(asked[p] for p in asked if p.endswith("g [error]")) == 3
    assert not any(prompt.endswith("=>h") for prompt in asked)  # the failure stops P3 only
    assert sum(asked[p] for p in asked if p.endswith("c [draft]")) == 2
    assert "Existing P4 history=>i" in asked
    metadata = json.loads(continuous["metadata"])
    assert metadata["model_metadata"] == METADATA
    assert metadata["summary_quality"] == {"attempts": 12, "rejected_attempts": 4}
    assert json.loads(continuous["qc"])[0]["ids"] == ["P3"]


def test_continuous_scheduling_starts_the_longest_chain_first_without_round_barriers(
        monkeypatch, tmp_path):
    """P0's later chunks are sent while P1's first chunk is still in flight."""
    backend = _Backend(capacity=2)
    p0_third_sent = asyncio.Event()

    async def before_answer(prompt):
        if prompt.prompt_text.endswith("=>p0-2"):
            p0_third_sent.set()
        if prompt.prompt_text.endswith("=>p1-0"):
            # A round barrier would wait for this answer before sending p0-1.
            await asyncio.wait_for(p0_third_sent.wait(), timeout=10)

    backend.before_answer = before_answer
    chunks = {"P0": ["p0-0", "p0-1", "p0-2", "p0-3"], "P1": ["p1-0"], "P2": ["p2-0"]}
    result, _, _ = _run(monkeypatch, tmp_path, chunks, backend=backend)

    sent = [prompt.rsplit("=>", 1)[1] for prompt in backend.sent]
    # Two slots: P0 (4 chunks) and P1 start; while P1 waits, each freed slot goes
    # to P0, whose chain stays longer than P2's, so P2 starts only at the end.
    assert sorted(sent[:2]) == ["p0-0", "p1-0"]
    assert sent[2:4] == ["p0-1", "p0-2"]
    assert sorted(sent[4:]) == ["p0-3", "p2-0"]
    assert result["patient_id"].tolist() == ["P0", "P1", "P2"]


@pytest.mark.parametrize("layout", ["round files", "journal cut mid-chain", "round files and journal"])
def test_continuous_scheduling_resumes_each_patient_from_saved_chunks(monkeypatch, tmp_path, layout):
    """Old round files, a journal and a mix resume to the uninterrupted run's results."""
    chunks = {"P1": ["a", "b", "c"], "P2": ["d", "e"], "P3": ["f"]}
    full_backend = _Backend()
    expected, _, _ = _run(monkeypatch, tmp_path / "full", chunks, backend=full_backend)
    saved = {(row["patient_id"], row["chunk_index"]): row for row in _journal(tmp_path / "full")}
    columns = ["patient_id", "prior_summary", "patient_reasoning_text", "patient_answer_text",
               "finish_reason"]

    def rounds(*indexes):
        for index in indexes:
            rows = [row for (_, chunk), row in sorted(saved.items()) if chunk == index]
            save_round_checkpoint(directory, index, pd.DataFrame(rows)[columns])

    directory = tmp_path / "resumed"
    directory.mkdir()
    if layout == "round files":
        rounds(0, 1)
        done = {("P1", 0), ("P2", 0), ("P3", 0), ("P1", 1), ("P2", 1)}
    elif layout == "journal cut mid-chain":
        journal = ChunkJournal(directory)
        journal.append([saved["P1", 0], saved["P2", 0], saved["P1", 1]])
        journal.close()
        with (directory / CHUNK_JOURNAL_FILENAME).open("ab") as handle:
            handle.write(json.dumps(saved["P3", 0]).encode()[:40])  # interrupted mid-write
        done = {("P1", 0), ("P2", 0), ("P1", 1)}
    else:
        rounds(0)
        journal = ChunkJournal(directory)
        stale = {**saved["P2", 1], "prior_summary": "a summary from another chain"}
        duplicate = {**saved["P1", 0], "patient_answer_text": "a later copy of chunk 0"}
        journal.append([saved["P1", 1], stale, duplicate])
        journal.close()
        done = {("P1", 0), ("P2", 0), ("P3", 0), ("P1", 1)}

    backend = _Backend()
    resumed, _, _ = _run(monkeypatch, directory, chunks, backend=backend)

    # Exactly the unsaved chunks are asked, each with its chain's saved prior summary.
    expected_prompts = {
        f"{saved[patient_id, chunk]['prior_summary']}=>{chunks[patient_id][chunk]}"
        for patient_id, chunk in saved if (patient_id, chunk) not in done
    }
    assert set(backend.asked) == expected_prompts
    assert set(backend.asked) <= set(full_backend.asked)
    assert resumed.drop(columns="patient_summary_attempts").equals(
        expected.drop(columns="patient_summary_attempts"))
    # The interrupted line was dropped, not continued: every line parses.
    lines = (directory / CHUNK_JOURNAL_FILENAME).read_bytes().split(b"\n")
    assert lines[-1] == b"" and all(json.loads(line) for line in lines[:-1])
    journal_rows = load_chunk_journal(directory)
    assert {(row["patient_id"], row["chunk_index"]) for row in journal_rows} >= set(saved) - done


def test_a_failed_chunk_stops_only_that_patient_including_after_resuming(monkeypatch, tmp_path):
    chunks = {"P1": ["a", "b", "c"], "P2": ["x [error]", "y"]}
    backend = _Backend()
    result, _, qc = _run(monkeypatch, tmp_path, chunks, backend=backend)

    assert result["patient_id"].tolist() == ["P1"]
    assert qc.loc[0, "ids"] == ["P2"]
    assert not any(prompt.endswith("=>y") for prompt in backend.asked)
    assert [row["finish_reason"] for row in _journal(tmp_path)] == ["stop", "stop", "stop", "error"]

    again = _Backend()
    result, _, qc = _run(monkeypatch, tmp_path, None, backend=again)
    assert not again.asked
    assert result["patient_id"].tolist() == ["P1"] and qc.loc[0, "ids"] == ["P2"]


def test_patient_scheduling_defaults_to_continuous_and_a_local_engine_keeps_rounds(
        monkeypatch, tmp_path):
    assert load_default_preset().patient["scheduling"] == "continuous"

    class BatchOnly:
        """Like a local engine: batches only, no request session."""

        def __init__(self):
            self.calls = []

        def generate_llm_outputs(self, *, prompt_list, llm_config, model_metadata_cache_dir=None):
            self.calls.append([prompt.prompt_text for prompt in prompt_list])
            return _Backend().generate_llm_outputs(prompt_list=prompt_list, llm_config=llm_config)

    backend = BatchOnly()
    _run(monkeypatch, tmp_path / "local", {"P1": ["a", "b"], "P2": ["c"]}, backend=backend)
    assert [[prompt.rsplit("=>", 1)[1] for prompt in call] for call in backend.calls] == [
        ["a", "c"], ["b"]]
    with pytest.raises(ValueError, match="patient.scheduling"):
        _run(monkeypatch, tmp_path / "bad", {"P1": ["a"]}, backend=_Backend(), scheduling="batch")


def test_pooled_request_capacity_follows_the_registry(tmp_path):
    """The continuous scheduler keeps every registry server's ceiling in flight."""
    registry = tmp_path / "servers.json"
    registry.write_text(json.dumps({"servers": [
        {"url": "http://127.0.0.1:9/v1", "upstream_concurrency": 3},
        {"url": "http://127.0.0.1:10/v1", "upstream_concurrency": 5}]}))
    pooled = {"server_urls": [], "server_urls_file": str(registry), "max_concurrent_requests": 2,
              "request_timeout": 60, "model_name": "fake-model",
              "adaptive_concurrency": {"enabled": False}}
    listed = {**pooled, "server_urls": ["http://a/v1", "http://b/v1"], "server_urls_file": None}

    async def capacities():
        async with RemoteRequestSession(pooled, "not-needed", {}) as session:
            first = session.request_capacity()
            registry.write_text(json.dumps({"servers": [
                {"url": "http://127.0.0.1:9/v1", "upstream_concurrency": 4}]}))
            session.pool.last_refresh = float("-inf")  # as if its refresh interval passed
            edited = session.request_capacity()
        # Without a pool, each listed server takes max_concurrent_requests.
        async with RemoteRequestSession(listed, "not-needed", {}) as session:
            return first, edited, session.request_capacity()

    assert asyncio.run(capacities()) == (8, 4, 4)
