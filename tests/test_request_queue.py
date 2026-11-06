"""The trial stages keep requests in flight: one slow answer must not hold up the others."""
import json
from threading import Event, Lock
from types import SimpleNamespace

import pandas as pd

from matchminer_ai.config import load_default_preset
from matchminer_ai.trials import remove_boilerplate_criteria


def test_remote_requests_are_sent_one_by_one_so_a_slow_answer_holds_up_only_itself(monkeypatch, tmp_path):
    import openai
    from matchminer_ai.llm.endpoint_pool import _POOLS
    _POOLS.clear()
    registry = tmp_path / "servers.json"
    registry.write_text(json.dumps({"ready": True, "servers": [{"url": "http://replica/v1", "upstream_concurrency": 8}]}))
    lines = ["Slow line.", "Line two.", "Line three.", "Line four.", "Line five."]
    lock, others_done, answered, sent = Lock(), Event(), [], []

    def respond(request):
        item = request["messages"][-1]["content"].rsplit("ITEM TO CLASSIFY", 1)[1].split("] ", 1)[1].split("\n")[0]
        with lock:
            sent.append(item)
        if item == "Slow line.":
            # In batches of two, the slow line's batch-mate would wait for it and the other lines never start.
            assert others_done.wait(10)
        with lock:
            answered.append(item)
            if len(answered) == len(lines) - 1:
                others_done.set()
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(
            content=json.dumps({"criterion_type": "inclusion", "boilerplate": False}), reasoning="test"))])

    monkeypatch.setattr(openai, "OpenAI", lambda **settings: SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=lambda **request: respond(request)))))
    monkeypatch.setattr("matchminer_ai.llm.backends.get_model_metadata", lambda *a, **k: {"model_sha": "test"})
    config = load_default_preset()
    config.remote.update(enabled=True, server_urls=[], server_urls_file=str(registry))
    config.trial["cohorts"] = {**config.trial["cohorts"], "chunk_size": 2, "chunks_in_flight": 1}  # two in flight
    frame = pd.DataFrame([{"trial_id": "T1", "cohort_number": 0, "cohort_name": "Arm A",
                           "cohort_eligibility_criteria": "\n".join(lines)}])
    result, metadata = remove_boilerplate_criteria(frame, config=config, output_dir=tmp_path / "trials",
                                                   return_metadata=True)
    assert sent[:2] == lines[:2] and answered[-1] == "Slow line."
    assert result.cohort_key_criteria.iloc[0].split("\n") == lines and not metadata["unclassified_items"]
    saved = json.loads((tmp_path / "trials" / "T1" / "deboilerplate.json").read_text())
    assert [i["attempts"][0]["response_metadata"]["endpoint"] for i in saved["cohorts"][0]["items"]] == [
        "http://replica/v1"] * len(lines)
