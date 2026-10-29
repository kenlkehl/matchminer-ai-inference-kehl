"""Registry-seeded synthetic protocol generation."""

from copy import deepcopy
import json

import pandas as pd
import pytest

from matchminer_ai.config import load_default_preset
from matchminer_ai.llm.backends import LLMGenerationResult
from matchminer_ai.trials import generate_synthetic_protocols
from matchminer_ai.trials.synthetic_protocols import (
    MISSING, build_synthetic_protocol_messages, choose_layout, load_synthetic_protocol_layouts,
)

SEEDS = pd.DataFrame([
    {"trial_id": "NCT00000001", "title": "Phase 2 {alpha} study", "brief_summary": "Summary one",
     "detailed_description": None, "eligibility_criteria": "Inclusion: adults"},
    {"trial_id": "NCT00000002", "title": "Second study", "brief_summary": "Summary two",
     "detailed_description": "Two parts", "eligibility_criteria": "Inclusion: solid tumors"},
])


@pytest.fixture
def backend(monkeypatch):
    config = load_default_preset()
    config.remote["enabled"] = True
    calls, answers = [], []

    class Backend:
        def generate_llm_outputs(self, *, prompt_list, llm_config, **kwargs):
            calls.append((prompt_list, deepcopy(llm_config)))
            batch = answers.pop(0)
            assert len(batch) == len(prompt_list)
            return LLMGenerationResult(
                final_outputs=[text for text, _ in batch], finish_reasons=[finish for _, finish in batch],
                reasoning_outputs=["reasoning"] * len(batch), raw_outputs=[], model_metadata={"model_sha": "test"},
            )

    monkeypatch.setattr("matchminer_ai.trials.synthetic_protocols.get_llm_backend", lambda _: Backend())
    return config, calls, answers


def content(prompt):
    return json.dumps(prompt.messages) if getattr(prompt, "messages", None) else prompt.prompt_text


def test_messages_keep_seed_text_literal_and_mark_missing_fields():
    layouts = load_synthetic_protocol_layouts()
    messages = build_synthetic_protocol_messages(SEEDS.iloc[0].to_dict(), layouts[0]["text"])
    assert [m["role"] for m in messages] == ["system", "user"]
    user = messages[1]["content"]
    assert "Phase 2 {alpha} study" in user and MISSING in user and layouts[0]["text"] in user
    assert "{title}" not in user and "{eligibility_layout}" not in user


def test_layouts_are_weighted_and_stable():
    layouts = load_synthetic_protocol_layouts()
    assert [entry["weight"] for entry in layouts] == [20, 9, 1, 1]
    picks = [choose_layout(f"NCT{i:08d}", layouts) for i in range(3100)]
    assert picks == [choose_layout(f"NCT{i:08d}", layouts) for i in range(3100)]
    assert 0.55 < picks.count(0) / len(picks) < 0.75 and set(picks) == {0, 1, 2, 3}


def test_outputs_are_written_and_completed_trials_are_skipped(tmp_path, backend):
    config, calls, answers = backend
    answers.append([("# Protocol one", "stop"), ("# Protocol two", "stop")])
    first = generate_synthetic_protocols(SEEDS, config=config, output_dir=tmp_path)
    assert first.protocol.isna().all() and first.finish_reason.eq("stop").all()
    meta = json.loads((tmp_path / "NCT00000001" / "meta.json").read_text())
    assert meta["layout_index"] == choose_layout("NCT00000001", load_synthetic_protocol_layouts())
    assert (tmp_path / "NCT00000002" / "protocol.md").read_text() == "# Protocol two"
    again, metadata = generate_synthetic_protocols(SEEDS, config=config, output_dir=tmp_path, return_metadata=True)
    assert len(calls) == 1 and metadata["skipped_completed"] == 2 and len(again) == 2


def test_empty_or_truncated_outputs_are_retried_then_reported(tmp_path, backend):
    config, calls, answers = backend
    config.synthetic_protocol_generation["max_attempts"] = 2
    answers.extend([[("", "stop"), ("# Cut", "length")], [("# Retried", "stop"), ("# Cut again", "length")]])
    result, metadata = generate_synthetic_protocols(SEEDS, config=config, output_dir=tmp_path, return_metadata=True)
    assert len(calls[1][0]) == 2 and "Phase 2 {alpha} study" in content(calls[1][0][0])
    assert result.trial_id.tolist() == ["NCT00000001"] and result.attempts.tolist() == [2]
    assert set(metadata["trial_errors"]) == {"NCT00000002"} and not (tmp_path / "NCT00000002").exists()


def test_in_memory_results_and_chunking(backend):
    config, calls, answers = backend
    config.synthetic_protocol_generation.update(chunk_size=1, chunks_in_flight=1)
    answers.extend([[("# One", "stop")], [("# Two", "stop")]])
    result = generate_synthetic_protocols(SEEDS, config=config)
    assert len(calls) == 2 and result.protocol.tolist() == ["# One", "# Two"]


@pytest.mark.parametrize("seeds", [SEEDS.drop(columns="trial_id"), pd.concat([SEEDS, SEEDS]),
                                   SEEDS.assign(trial_id=["NCT1", " "])])
def test_invalid_seeds_fail_before_inference(backend, seeds):
    config, calls, _ = backend
    with pytest.raises(ValueError):
        generate_synthetic_protocols(seeds, config=config)
    assert calls == []


def test_chunks_overlap_so_fast_endpoints_are_not_left_idle(tmp_path, monkeypatch):
    import threading

    config = load_default_preset()
    config.remote["enabled"] = True
    config.synthetic_protocol_generation.update(chunk_size=1, chunks_in_flight=2)
    second_started = threading.Event()

    class Backend:
        def generate_llm_outputs(self, *, prompt_list, llm_config, **kwargs):
            text = content(prompt_list[0])
            if "Phase 2 {alpha} study" in text:
                # The first chunk cannot finish until the second chunk is also in flight.
                assert second_started.wait(timeout=10)
            else:
                second_started.set()
            name = "first" if "Phase 2 {alpha} study" in text else "second"
            return LLMGenerationResult(final_outputs=[f"# {name}"], finish_reasons=["stop"],
                                       reasoning_outputs=[""], raw_outputs=[], model_metadata={})

    monkeypatch.setattr("matchminer_ai.trials.synthetic_protocols.get_llm_backend", lambda _: Backend())
    result = generate_synthetic_protocols(SEEDS, config=config, output_dir=tmp_path)
    assert result.trial_id.tolist() == ["NCT00000001", "NCT00000002"]
    assert (tmp_path / "NCT00000001" / "protocol.md").read_text() == "# first"
    assert (tmp_path / "NCT00000002" / "protocol.md").read_text() == "# second"
