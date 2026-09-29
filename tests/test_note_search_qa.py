"""Fabricated records only; exercise real isolated Python workers without an LLM."""

from dataclasses import replace
import ast
import json
from threading import Event, Lock

import pytest

from matchminer_ai.llm.structured import EndpointError
from matchminer_ai.patients import (
    NoteSearchLimits,
    NoteSearchLLMConfig,
    answer_patient_question_batch,
    answer_patient_questions,
)
from matchminer_ai.patients import note_search_qa as qa
from matchminer_ai.patients._note_repl import NoteREPL, NoteREPLError


@pytest.fixture
def llm():
    return NoteSearchLLMConfig(
        base_url="http://fabricated.invalid/v1",
        model="fabricated-model",
        max_tokens=2048,
        context_window=32768,
        tokenizer_mode="bytes",
        response_format="json_schema",
        attempts=1,
        timeout=5,
    )


def cell(code, memory=""):
    return dict(
        action="python",
        code=code.splitlines(),
        memory=memory,
        status="unknown",
        answer="",
        limitations=[],
    )


def final(history, quote=None, answer="Documented."):
    return dict(
        action="final",
        code=[],
        memory="",
        status="answered" if quote else "unknown",
        answer=answer,
        limitations=[] if quote else ["No supporting documentation found."],
    )


def mock_model(monkeypatch, handler):
    messages_seen = []

    def complete(self, job, messages, schema, validator):
        payload = json.loads(messages[1]["content"])
        messages_seen.append(messages)
        value = handler(payload)
        validator(value)
        return value

    monkeypatch.setattr(qa._MeasuredClient, "complete", complete)
    return messages_seen


def test_repl_state_exact_unicode_offsets_and_pagination():
    history = "é: test planned.\nLater: test completed.\nRepeat: test pending."
    with NoteREPL(history, NoteSearchLimits()) as worker:
        first = worker.execute("hits = search('test', context=0, limit=1); hits")
        assert "'start': 3" in first["output"]
        assert "'has_more': True" in first["output"]
        second = worker.execute(
            "search('test', start=hits['next_start'], context=0, limit=1)"
        )
        assert f"'start': {history.index('test', 4)}" in second["output"]
        exact = worker.execute("read(23, 38)")
        assert repr(history[23:38]) in exact["output"]
        assert worker.execute("1 / 0")["error"] == "ZeroDivisionError"
        assert "'hits'" in worker.execute("hits")["output"]


def test_scan_covers_synonyms_and_later_updates_with_exact_unicode_spans():
    history = "é: CT ordered.\n" + "Routine contact log.\n" * 5000
    history += "Computed tomography of chest completed."
    with NoteREPL(history, NoteSearchLimits()) as worker:
        output = worker.execute("scan([r'\\bCT\\b', 'computed tomography'])")
    assert output["error"] is None and output["truncated"] is False
    result = ast.literal_eval(output["output"])
    assert result["match_count"] == 2
    assert not result["omitted"] and not result["truncated"]
    assert "CT ordered" in result["hits"][0]["quote"]
    assert "chest completed" in result["hits"][-1]["quote"]
    for hit in result["hits"]:
        assert history[hit["start"] : hit["end"]] == hit["quote"]


def test_scan_dense_matches_preserve_early_late_and_report_omissions():
    history = "Assay ordered.\n" + "Assay pending.\n" * 10000 + "Assay completed."
    with NoteREPL(history, NoteSearchLimits()) as worker:
        output = worker.execute("scan(['assay'], context=20, limit=6)")
        missing = ast.literal_eval(worker.execute("scan(['ECG', 'EKG'])")["output"])
    result = ast.literal_eval(output["output"])
    assert result["match_count"] == 10002 and result["omitted"]
    assert len(result["hits"]) == 6
    assert "ordered" in result["hits"][0]["quote"]
    assert "completed" in result["hits"][-1]["quote"]
    assert missing == {
        "hits": [],
        "match_count": 0,
        "omitted": False,
        "truncated": False,
    }


def test_scan_keeps_all_sparse_hits_in_one_region_and_deduplicates_synonyms():
    history = "CT ordered. CT scheduled. CT performed. CT normal.\n" + "x" * 100000
    with NoteREPL(history, NoteSearchLimits()) as worker:
        output = worker.execute("scan([r'\\bCT\\b', 'CT'])")
    result = ast.literal_eval(output["output"])
    assert result["match_count"] == len(result["hits"]) == 4
    assert result["omitted"] is False
    assert [h["match_start"] for h in result["hits"]] == [0, 12, 26, 40]


def test_scan_bounds_its_result_without_cutting_off_the_return_shape():
    history = ("é" * 100 + "imaginary assay" + "\\'\n" * 100) * 30
    with NoteREPL(history, NoteSearchLimits(max_output_chars=500)) as worker:
        output = worker.execute("scan(['assay'], context=2000)")
    assert output["error"] is None and not output["truncated"]
    result = ast.literal_eval(output["output"])
    assert result["match_count"] == 30 and result["omitted"] and result["truncated"]
    for hit in result["hits"]:
        assert history[hit["start"] : hit["end"]] == hit["quote"]


def test_worker_has_no_host_file_credentials_network_or_process_access(
    tmp_path, monkeypatch
):
    secret = tmp_path / "host-secret.txt"
    secret.write_text("do not expose")
    monkeypatch.setenv("FABRICATED_CREDENTIAL", "must-not-be-in-worker")
    with NoteREPL("Synthetic notes.", NoteSearchLimits()) as worker:
        assert (
            worker.execute(f"open({str(secret)!r}).read()")["error"]
            == "PermissionError"
        )
        assert (
            "must-not-be-in-worker"
            not in worker.execute("import os; dict(os.environ)")["output"]
        )
        assert worker.execute("os.fork()")["error"] == "PermissionError"
        assert worker.execute("os.system('true')")["output"].strip() == "32512"
        # Test socket creation at the syscall boundary, independent of import access.
        assert (
            worker.execute(
                "import ctypes; libc = ctypes.CDLL(None, use_errno=True); "
                "(libc.socket(2, 1, 0), ctypes.get_errno())"
            )["output"].strip()
            == "(-1, 1)"
        )
        assert (
            worker.execute(
                "import resource; resource.setrlimit(resource.RLIMIT_AS, (-1, -1))"
            )["error"]
            is not None
        )


def test_cell_output_is_bounded_and_reports_truncation():
    limits = NoteSearchLimits(max_output_chars=200)
    with NoteREPL("x" * 2000, limits) as worker:
        value = worker.execute("print(history)")
        assert value["truncated"] is True and len(value["output"]) == 200
        assert worker.execute("len(history)")["output"].strip() == "2000"


@pytest.mark.parametrize(
    "code",
    [
        "while True: pass",
        "re.search('(a+)+$', 'a' * 10000 + '!')",
    ],
)
def test_timeout_kills_worker_including_catastrophic_regex(code):
    limits = NoteSearchLimits(cell_timeout_seconds=0.4)
    with NoteREPL("Fabricated text.", limits) as worker:
        with pytest.raises(NoteREPLError, match="wall-time"):
            worker.execute(code)
        assert worker.process.poll() is not None


def test_worker_memory_limit_rejects_large_allocation():
    with NoteREPL("Fabricated text.", NoteSearchLimits(worker_memory_mb=128)) as worker:
        result = worker.execute("large = 'a' * (1024**3)")
        assert result["error"] == "MemoryError"


def test_search_loop_carries_memory_without_replaying_full_history(monkeypatch, llm):
    history = "PRIVATE_UNRELATED_PREFIX " * 1000 + "Imaginary assay completed."
    quote = "Imaginary assay completed."
    calls = 0

    def handler(payload):
        nonlocal calls
        calls += 1
        if calls == 1:
            assert payload["last_cell_result"] is None
            return cell(
                "hits = search('imaginary', context=0); hits", "Look for result."
            )
        if calls == 2:
            assert payload["memory"] == "Look for result."
            return cell(
                "read(hits['hits'][0]['match_start'], len(history))",
                "Keep assay offsets.",
            )
        assert payload["memory"] == "Keep assay offsets."
        assert quote in payload["last_cell_result"]["output"]
        return final(history, quote)

    messages = mock_model(monkeypatch, handler)
    answer = answer_patient_questions(history, ["Was the assay done?"], llm=llm)[
        "answers"
    ][0]
    assert answer["status"] == "answered"
    assert answer["evidence"] == [
        dict(start=history.index(quote), end=len(history), quote=quote, note_date=None)
    ]
    assert answer["metadata"]["cells"] == 2
    assert all("PRIVATE_UNRELATED_PREFIX" not in json.dumps(m) for m in messages)
    assert all(len(m) == 2 for m in messages)


@pytest.mark.parametrize("span", [[-1, 4], [False, 4], [0, 400], [3, 2]])
def test_invalid_worker_source_spans_are_rejected(span):
    with pytest.raises(NoteREPLError, match="invalid source spans"):
        qa._source_observation(
            {"source_spans": [span], "sources_truncated": False, "output": ""},
            "note",
            NoteSearchLimits(),
        )


def test_model_never_selects_ids_or_copies_quotes_and_worker_text_is_untrusted(
    monkeypatch, llm
):
    history = 'é: Assay "A" ordered.\nLater: completed.\\lab\n' * 2

    def handler(payload):
        if payload["last_cell_result"] is None:
            # Even mutated helper return text cannot corrupt parent-owned quotes.
            return cell(
                "hit = read(0, len(history)); hit['quote'] = 'FORGED'; print('calculation')"
            )
        assert payload["last_cell_result"]["source_excerpts"] == [{"quote": history}]
        return final(history, history)

    messages = mock_model(monkeypatch, handler)
    result = answer_patient_questions(history, ["Assay?"], llm=llm)["answers"][0]
    assert result["status"] == "answered"
    assert result["evidence"] == [
        dict(start=0, end=len(history), quote=history, note_date=None)
    ]
    assert result["metadata"]["evidence_selection"] == "automatic_reviewed_excerpts"
    assert "evidence" not in qa._schema(NoteSearchLimits())["properties"]
    observed = json.loads(messages[-1][1]["content"])["last_cell_result"]
    assert "FORGED" not in json.dumps(observed["source_excerpts"])


def test_unsupported_answer_cannot_use_printed_text_as_source():
    with pytest.raises(ValueError, match="source excerpts"):
        qa._validate(final("note", "note"), "note", NoteSearchLimits(), False, 1)


def test_source_capture_has_a_shared_output_budget_and_reports_omissions():
    history = "é\\\n" * 2000
    observation = {
        "source_spans": [[0, 4000]],
        "sources_truncated": False,
        "output": history[:200],
        "truncated": False,
    }
    sources = qa._source_observation(
        observation, history, NoteSearchLimits(max_output_chars=200)
    )
    assert not sources and observation["sources_truncated"]
    assert len(observation["output"]) <= 200
    assert not observation["source_excerpts"]


def test_scan_registers_only_its_returned_excerpts_not_unshown_matches():
    history = "Assay pending.\n" * 100
    with NoteREPL(history, NoteSearchLimits()) as worker:
        result = worker.execute("scan(['assay'], context=0, limit=2)")
        hits = ast.literal_eval(result["output"])["hits"]
        assert result["source_spans"] == [[h["start"], h["end"]] for h in hits]
        assert worker.execute("pass")["source_spans"] == []


def test_prior_excerpts_survive_without_model_tracking(monkeypatch, llm):
    history = "Ordered.\n" + "unrelated " * 100 + "Completed."
    calls = 0

    def handler(payload):
        nonlocal calls
        calls += 1
        if calls == 1:
            return cell("read(0, 8)")
        if calls == 2:
            return cell("read(len(history)-10, len(history))", "Completion found.")
        assert payload["last_cell_result"]["source_excerpts"] == [
            {"quote": "Completed."}
        ]
        return final(history, "Completed.")

    mock_model(monkeypatch, handler)
    answer = answer_patient_questions(history, ["Status?"], llm=llm)["answers"][0]
    assert [e["quote"] for e in answer["evidence"]] == ["Ordered.", "Completed."]


def test_excerpts_not_sent_due_to_context_limit_are_not_retained(monkeypatch, llm):
    mock_model(monkeypatch, lambda _: cell("read(0, len(history))"))
    monkeypatch.setattr(
        qa._MeasuredClient,
        "fits",
        lambda self, messages: (
            json.loads(messages[1]["content"])["last_cell_result"] is None
        ),
    )
    answer = answer_patient_questions("synthetic", ["q"], llm=llm)["answers"][0]
    assert answer["metadata"]["termination_reason"] == "context_limit"
    assert answer["evidence"] == []


def test_worker_bounds_automatic_source_registration():
    with NoteREPL("x" * 100, NoteSearchLimits()) as worker:
        result = worker.execute("for i in range(100):\n    _ = read(i, i+1)")
    assert len(result["source_spans"]) == 64
    assert result["sources_truncated"] is True


def test_unused_answer_fields_do_not_reject_a_valid_python_step():
    value = cell("search('term')")
    value.update(
        status="answered",
        answer="These fields are not a final answer.",
    )
    qa._validate(value, "note", NoteSearchLimits(), False, 0)


def test_final_turn_enforced_and_unknown_is_explicit(monkeypatch, llm):
    def handler(payload):
        if not payload["final_only"]:
            return cell("search('absentterm')")
        assert payload["cells_remaining"] == 0
        assert "'hits': []" in payload["last_cell_result"]["output"]
        return final(
            "No relevant fabricated information.", answer="Unknown from these notes."
        )

    messages = mock_model(monkeypatch, handler)
    answer = answer_patient_questions(
        "No relevant fabricated information.",
        ["Was the event completed?"],
        llm=llm,
        limits=NoteSearchLimits(max_cells=2),
    )["answers"][0]
    assert len(messages) == 3 and answer["status"] == "unknown"
    assert answer["evidence"] == [] and answer["limitations"]
    with pytest.raises(ValueError, match="final answer"):
        qa._validate(cell("1"), "note", NoteSearchLimits(), True, 2)
    with pytest.raises(ValueError, match="exact evidence"):
        qa._validate(final("note", "note"), "note", NoteSearchLimits(), False, 0)


def test_concurrency_within_and_across_patients_preserves_order(monkeypatch, llm):
    lock, full = Lock(), Event()
    active, peak, entered_patients = {}, {}, set()
    total, peak_total = 0, 0

    def answer(history, question, index, *_args):
        nonlocal total, peak_total
        with lock:
            active[history] = active.get(history, 0) + 1
            peak[history] = max(peak.get(history, 0), active[history])
            entered_patients.add(history)
            total += 1
            peak_total = max(peak_total, total)
            if total == 4:
                full.set()
        assert full.wait(4), "Both patients and both questions should run together"
        with lock:
            active[history] -= 1
            total -= 1
        return {"question_index": index, "question": question, "status": "unknown"}

    monkeypatch.setattr(qa, "_answer", answer)
    progress = []
    result = answer_patient_question_batch(
        [
            dict(
                patient_id=f"p{i}",
                history=f"synthetic-{i}",
                questions=[f"q{j}" for j in range(3)],
            )
            for i in range(3)
        ],
        llm=llm,
        max_parallel_patients=2,
        max_parallel_questions=2,
        max_active_questions=4,
        progress_callback=progress.append,
    )
    assert peak_total == 4 and all(n <= 2 for n in peak.values())
    assert peak["synthetic-0"] == peak["synthetic-1"] == 2
    assert len(entered_patients) == 3
    assert [p["patient_id"] for p in result["patients"]] == ["p0", "p1", "p2"]
    assert all(
        [a["question"] for a in p["answers"]] == ["q0", "q1", "q2"]
        for p in result["patients"]
    )
    assert len(progress) == 9
    assert all(
        set(p) == {"patient_index", "question_index", "stage", "status"}
        for p in progress
    )


def test_workers_do_not_share_patient_or_question_state(monkeypatch, llm):
    def handler(payload):
        if payload["last_cell_result"] is None:
            return cell(
                "assert 'previous' not in globals(); previous = history; read(0, len(history))"
            )
        history = (
            "first fabricated note"
            if payload["question"].startswith("first")
            else "second fabricated note"
        )
        other = (
            "second fabricated note"
            if history.startswith("first")
            else "first fabricated note"
        )
        assert history in payload["last_cell_result"]["output"]
        assert other not in payload["last_cell_result"]["output"]
        return final(history, history)

    mock_model(monkeypatch, handler)
    result = answer_patient_question_batch(
        [
            dict(
                patient_id="a",
                history="first fabricated note",
                questions=["first?", "first again?"],
            ),
            dict(
                patient_id="b", history="second fabricated note", questions=["second?"]
            ),
        ],
        llm=llm,
    )
    assert all(
        a["status"] == "answered" for p in result["patients"] for a in p["answers"]
    )


def test_one_endpoint_failure_does_not_erase_other_answers(monkeypatch, llm):
    def handler(payload):
        if payload["question"] == "bad":
            raise EndpointError("MUST_NOT_APPEAR_IN_RESULT")
        if payload["last_cell_result"] is None:
            return cell("search('synthetic')")
        return final("synthetic", "synthetic")

    mock_model(monkeypatch, handler)
    answers = answer_patient_questions("synthetic", ["bad", "good"], llm=llm)["answers"]
    assert [a["status"] for a in answers] == ["error", "answered"]
    assert "MUST_NOT_APPEAR" not in json.dumps(answers)


def test_failed_cells_are_errors_not_missing_documentation(monkeypatch, llm):
    def handler(payload):
        if payload["last_cell_result"] is None:
            return cell("undefined_name")
        assert payload["last_cell_result"]["code"] == "undefined_name"
        assert "undefined_name" in payload["last_cell_result"]["error_detail"]
        return final("synthetic", answer="Unknown.")

    mock_model(monkeypatch, handler)
    result = answer_patient_questions("synthetic", ["q"], llm=llm)["answers"][0]
    assert result["status"] == "error" and result["answer"] is None
    assert result["metadata"]["successful_cells"] == 0
    assert result["metadata"]["cell_errors"] == ["NameError"]
    assert result["metadata"]["termination_reason"] == "cell_execution_failure"


def test_model_can_recover_from_cell_error(monkeypatch, llm):
    def handler(payload):
        last = payload["last_cell_result"]
        if last is None:
            return cell("undefined_name")
        if last["error"]:
            return cell("for term in ['synthetic']:\n    print(search(term))")
        assert "synthetic" in last["output"]
        assert "\n    print" in last["code"]
        return final("synthetic", "synthetic")

    mock_model(monkeypatch, handler)
    result = answer_patient_questions("synthetic", ["q"], llm=llm)["answers"][0]
    assert result["status"] == "answered"
    assert result["metadata"]["successful_cells"] == 1


def test_small_context_abstains_before_generation(monkeypatch, llm):
    mock_model(
        monkeypatch, lambda _: pytest.fail("Should not send an oversized prompt")
    )
    answer = answer_patient_questions(
        "synthetic",
        ["Question?"],
        llm=replace(
            llm,
            context_window=4200,
        ),
    )["answers"][0]
    assert answer["status"] == "unknown" and "context budget" in answer["answer"]


def test_actual_wire_usage_and_no_checkpoint_writes(monkeypatch, llm, tmp_path):
    import io

    seen = []

    def respond(request, timeout):
        body = json.loads(request.data)
        seen.append(body)
        payload = json.loads(body["messages"][1]["content"])
        value = (
            cell("search('synthetic')")
            if payload["last_cell_result"] is None
            else final("synthetic", "synthetic")
        )
        return io.BytesIO(
            json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(value),
                                "reasoning": "DO_NOT_REPLAY_REASONING_SENTINEL",
                                "reasoning_content": "DO_NOT_REPLAY_REASONING_SENTINEL",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 50},
                }
            ).encode()
        )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("matchminer_ai.llm.structured.urlopen", respond)
    answer = answer_patient_questions(
        "synthetic", ["Question?"], llm=replace(llm, stream=False)
    )["answers"][0]
    assert answer["metadata"]["requests"] == 2
    assert answer["metadata"]["prompt_tokens"] == 200
    assert answer["metadata"]["completion_tokens"] == 100
    assert answer["metadata"]["usage_complete"] is True
    assert len(answer["metadata"]["request_metrics"]) == 2
    assert len(answer["metadata"]["cell_seconds"]) == 1
    assert answer["metadata"]["worker_startup_seconds"] > 0
    assert len(seen) == 2 and not list(tmp_path.iterdir())
    assert all("tools" not in b and "tool_choice" not in b for b in seen)
    assert "DO_NOT_REPLAY_REASONING_SENTINEL" not in json.dumps(seen)
    assert all([m["role"] for m in b["messages"]] == ["system", "user"] for b in seen)
    assert all(b["chat_template_kwargs"]["enable_thinking"] is True for b in seen)
    assert all(b["chat_template_kwargs"]["preserve_thinking"] is False for b in seen)


@pytest.mark.parametrize(
    "model,provider,efforts",
    [
        ("Qwen/Qwen3.8-27B", "openai", ["low", "xhigh"]),
        ("google/gemini-3.8-flash", "google_agent_platform", ["low", "high"]),
        ("google/gemma-4-31B-it", "openai", [None, None]),
        ("unrecognized-model", "openai", [None, None]),
    ],
)
def test_two_call_path_uses_low_search_then_original_assessment_effort(
    monkeypatch, llm, model, provider, efforts
):
    import io

    seen = []

    def respond(request, timeout):
        body = json.loads(request.data)
        seen.append(body)
        payload = json.loads(body["messages"][1]["content"])
        searching = payload["search_only"]
        assert searching == (len(seen) == 1)
        schema = body["response_format"]["json_schema"]["schema"]
        assert schema["properties"]["action"]["enum"] == (
            ["python"] if searching else ["final", "python"]
        )
        value = (
            cell("scan(['synthetic'])")
            if searching
            else final("synthetic", "synthetic")
        )
        return io.BytesIO(
            json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(value),
                                "reasoning": "NO_REPLAY",
                            },
                            "finish_reason": "stop",
                        }
                    ]
                }
            ).encode()
        )

    monkeypatch.setattr("matchminer_ai.llm.structured.urlopen", respond)
    monkeypatch.setattr(
        "matchminer_ai.llm.structured.remote_bearer_token", lambda _: "test"
    )
    config = replace(
        llm,
        model=model,
        provider=provider,
        stream=False,
        base_url="https://aiplatform.googleapis.com/v1",
    )
    result = answer_patient_questions(
        "synthetic",
        ["q"],
        llm=config,
        limits=NoteSearchLimits(max_cells=2, max_calls=3),
    )
    answer = result["answers"][0]
    assert answer["status"] == "answered" and answer["metadata"]["requests"] == 2
    assert answer["metadata"]["cells"] == 1
    assert [b.get("reasoning_effort") for b in seen] == efforts
    assert "NO_REPLAY" not in json.dumps(seen)
    if model.startswith("Qwen"):
        assert [b["chat_template_kwargs"]["reasoning_effort"] for b in seen] == efforts
    if provider == "openai":
        assert all(b["chat_template_kwargs"]["enable_thinking"] for b in seen)
        assert all(not b["chat_template_kwargs"]["preserve_thinking"] for b in seen)
    else:
        assert all(
            "chat_template_kwargs" not in b and "temperature" not in b for b in seen
        )


def test_search_effort_override_preserves_sampling_and_caller_config():
    config = NoteSearchLLMConfig(
        model="Qwen/Qwen3.8-27B",
        base_url="http://synthetic.invalid/v1",
        temperature=0.4,
        top_k=12,
        reasoning_effort="medium",
        extra_body={"chat_template_kwargs": {"reasoning_effort": "medium"}},
    )
    assessment = qa._resolve_llm(config)
    search = qa._resolve_search_llm(config, assessment)
    assert search.temperature == assessment.temperature == 0.4
    assert search.top_k == assessment.top_k == 12
    assert search.max_tokens == assessment.max_tokens
    assert search.request_params["reasoning_effort"] == "low"
    assert search.extra_body["chat_template_kwargs"]["reasoning_effort"] == "low"
    assert assessment.request_params["reasoning_effort"] == "medium"
    assert config.extra_body["chat_template_kwargs"]["reasoning_effort"] == "medium"
    assert (
        qa._resolve_search_llm(
            replace(config, search_reasoning_effort=None), assessment
        )
        == assessment
    )
    with pytest.raises(ValueError, match="search_reasoning_effort"):
        qa._resolve_search_llm(
            replace(config, search_reasoning_effort="invalid"), assessment
        )


@pytest.mark.parametrize("failure", ["http", "validation"])
def test_reserved_third_call_recovers_a_failed_early_answer(monkeypatch, llm, failure):
    import io
    from urllib.error import HTTPError

    seen = []

    def respond(request, timeout):
        body = json.loads(request.data)
        seen.append(body)
        payload = json.loads(body["messages"][1]["content"])
        if len(seen) == 2 and failure == "http":
            raise HTTPError(request.full_url, 500, "fabricated failure", {}, None)
        value = (
            cell("scan(['synthetic'])")
            if len(seen) == 1
            else final("synthetic", "synthetic")
        )
        if len(seen) == 2:
            value = {}
        if len(seen) == 3:
            assert payload["final_only"] is True
            assert "synthetic" in payload["last_cell_result"]["output"]
        return io.BytesIO(
            json.dumps(
                {
                    "choices": [
                        {
                            "message": {"content": json.dumps(value)},
                            "finish_reason": "stop",
                        }
                    ]
                }
            ).encode()
        )

    monkeypatch.setattr("matchminer_ai.llm.structured.urlopen", respond)
    answer = answer_patient_questions(
        "synthetic",
        ["q"],
        llm=replace(llm, stream=False),
        limits=NoteSearchLimits(max_calls=3, max_cells=2),
    )["answers"][0]
    assert answer["status"] == "answered" and len(seen) == 3
    assert answer["metadata"]["requests"] == 3
    assert len(answer["metadata"]["request_metrics"]) == 3
    assert answer["metadata"]["cells"] == 1


@pytest.mark.parametrize(
    "mode", ["valid", "repair_first", "invalid_final", "invalid_always"]
)
def test_total_call_limit_includes_final_answer_and_validation_retries(
    monkeypatch, llm, mode
):
    import io

    seen = []

    def respond(request, timeout):
        body = json.loads(request.data)
        seen.append(body)
        payload = json.loads(body["messages"][1]["content"])
        value = (
            final("synthetic", "synthetic")
            if payload["final_only"]
            else cell("search('synthetic')")
        )
        if (
            mode == "invalid_always"
            or (mode == "repair_first" and len(seen) == 1)
            or (mode == "invalid_final" and payload["final_only"])
        ):
            value = {}
        return io.BytesIO(
            json.dumps(
                {
                    "choices": [
                        {
                            "message": {"content": json.dumps(value)},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 50},
                }
            ).encode()
        )

    monkeypatch.setattr("matchminer_ai.llm.structured.urlopen", respond)
    monkeypatch.setattr("matchminer_ai.llm.structured.time.sleep", lambda _: None)
    answer = answer_patient_questions(
        "synthetic",
        ["q"],
        llm=replace(llm, attempts=5, stream=False),
        limits=NoteSearchLimits(max_calls=3),
    )["answers"][0]
    assert len(seen) == answer["metadata"]["requests"] <= 3
    if mode in {"valid", "repair_first"}:
        assert answer["status"] == "answered" and len(seen) == 3
        assert answer["metadata"]["cells"] == (1 if mode == "repair_first" else 2)
        assert json.loads(seen[-1]["messages"][1]["content"])["final_only"] is True
    else:
        assert answer["status"] == "error"
        if mode == "invalid_final":
            assert answer["metadata"]["termination_reason"] == "call_budget"


@pytest.mark.parametrize("value", [0, 1, -1, True, 3.5])
def test_invalid_total_call_limits(value):
    with pytest.raises(ValueError, match="max_calls"):
        NoteSearchLimits(max_calls=value)


@pytest.mark.parametrize(
    "model,k,budget",
    [
        ("Inferact/Qwen3.8-Flash-Next-NVFP4", 20, 32768),
        ("Qwen/Qwen3.8-27B", 20, 8192),
        ("google/gemma-4-31B-it", 64, 8192),
        ("nvidia/Gemma-4-31B-IT-NVFP4", 64, 8192),
    ],
)
def test_shared_vendor_defaults_and_no_reasoning_preservation(model, k, budget):
    config = NoteSearchLLMConfig(base_url="http://synthetic.invalid/v1", model=model)
    resolved = qa._resolve_llm(config)
    assert resolved.top_k == k and resolved.max_tokens == budget
    assert resolved.temperature == 1.0 and resolved.top_p == 0.95
    template = resolved.extra_body["chat_template_kwargs"]
    assert template["enable_thinking"] is True
    assert template["preserve_thinking"] is False
    assert config.extra_body == {}  # No mutation of caller configuration.
    if k == 20:
        assert template["reasoning_effort"] == "xhigh"
        assert resolved.request_params["reasoning_effort"] == "xhigh"
        assert resolved.extra_body["min_p"] == 0.0
        assert resolved.extra_body["repetition_penalty"] == 1.0
    else:
        assert "reasoning_effort" not in template


def test_sampling_overrides_use_shared_resolver():
    config = NoteSearchLLMConfig(
        base_url="http://synthetic.invalid/v1",
        model="Qwen/Qwen3.8-27B",
        temperature=0.4,
        top_k=12,
        reasoning_effort="medium",
        extra_body={"chat_template_kwargs": {"preserve_thinking": True}},
    )
    resolved = qa._resolve_llm(config)
    assert resolved.temperature == 0.4 and resolved.top_k == 12
    assert resolved.request_params["reasoning_effort"] == "medium"
    assert resolved.extra_body["chat_template_kwargs"]["preserve_thinking"] is False
    nonthinking = qa._resolve_llm(
        replace(config, thinking="off", temperature=None, top_k=None)
    )
    assert nonthinking.temperature == 0.7 and nonthinking.top_p == 0.8


def test_isolation_unavailable_fails_before_any_llm_request(monkeypatch, llm):
    def fail(*_):
        raise NoteREPLError("isolation unavailable")

    monkeypatch.setattr(qa, "NoteREPL", fail)
    mock_model(monkeypatch, lambda _: pytest.fail("Must not call endpoint"))
    with pytest.raises(NoteREPLError, match="isolation unavailable"):
        answer_patient_questions("synthetic", ["question"], llm=llm)


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_cells", 0),
        ("max_scan_patterns", 0),
        ("max_scan_patterns", True),
        ("max_output_chars", True),
        ("cell_timeout_seconds", float("nan")),
        ("worker_memory_mb", 64),
    ],
)
def test_invalid_limits_fail_early(field, value):
    with pytest.raises(ValueError):
        NoteSearchLimits(**{field: value})


def test_input_validation_precedes_worker_or_endpoint(monkeypatch, llm):
    monkeypatch.setattr(qa, "NoteREPL", lambda *_: pytest.fail("No worker yet"))
    with pytest.raises(ValueError, match="endpoint URL explicitly"):
        answer_patient_questions("synthetic", ["q"], llm=replace(llm, base_url=""))
    with pytest.raises(ValueError, match="nonempty list"):
        answer_patient_questions("synthetic", [], llm=llm)
    with pytest.raises(ValueError, match="positive integer"):
        answer_patient_questions("synthetic", ["q"], llm=llm, max_parallel_questions=0)
    with pytest.raises(ValueError, match="unique"):
        answer_patient_question_batch(
            [dict(patient_id="x", history="a", questions=["q"])] * 2, llm=llm
        )


@pytest.mark.parametrize("count", [13, 128])
def test_scan_large_pattern_lists_preserve_counts_provenance_and_output_bounds(count):
    history = "é: " + " ".join(f"F{i:03d}" for i in range(count))
    with NoteREPL(history, NoteSearchLimits(max_output_chars=1200)) as worker:
        observation = worker.execute(
            rf"scan([r'\bF%03d\b' % i for i in range({count})], context=0)"
        )
    assert observation["error"] is None and not observation["truncated"]
    result = ast.literal_eval(observation["output"])
    assert result["match_count"] == count and result["omitted"]
    assert 2 <= len(result["hits"]) <= 12
    assert result["hits"][0]["quote"] == "F000"
    assert result["hits"][-1]["quote"] == f"F{count - 1:03d}"
    assert len(observation["output"]) <= 1200
    for hit in result["hits"]:
        assert history[hit["start"] : hit["end"]] == hit["quote"]
        assert [hit["start"], hit["end"]] in observation["source_spans"]


def test_scan_enforces_configured_pattern_cap_and_deduplicates_large_lists():
    with NoteREPL("Assay completed.", NoteSearchLimits()) as worker:
        result = ast.literal_eval(worker.execute("scan(['assay'] * 128)")["output"])
        assert result["match_count"] == len(result["hits"]) == 1
        assert not result["omitted"]
        rejected = worker.execute("scan(['assay'] * 129)")
        assert rejected["error"] == "ValueError"
        assert "1-128 regex strings" in rejected["error_detail"]
        assert rejected["source_spans"] == []
    with NoteREPL("Assay completed.", NoteSearchLimits(max_scan_patterns=16)) as worker:
        assert worker.execute("scan(['assay'] * 16)")["error"] is None
        rejected = worker.execute("scan(['assay'] * 17)")
        assert rejected["error"] == "ValueError"
        assert "1-16 regex strings" in rejected["error_detail"]


def test_large_pattern_question_reaches_worker_and_renders_configured_limit(
    monkeypatch, llm
):
    history = "Fabricated note: imaginary assay completed."
    patterns = [f"unused{i}" for i in range(13)] + ["imaginary assay"]

    def handler(payload):
        if payload["last_cell_result"] is None:
            return cell(f"scan({patterns!r})")
        return final(history, history)

    messages = mock_model(monkeypatch, handler)
    result = answer_patient_questions(
        history,
        ["Was the imaginary assay completed?"],
        llm=llm,
        limits=NoteSearchLimits(max_scan_patterns=24),
    )
    answer = result["answers"][0]
    assert answer["status"] == "answered"
    assert answer["metadata"]["successful_cells"] == 1
    assert answer["metadata"]["cell_errors"] == []
    assert answer["metadata"]["max_scan_patterns"] == 24
    assert result["metadata"]["max_scan_patterns_per_call"] == 24
    assert answer["evidence"][0]["quote"] == history
    assert len(messages) == 2
    for message in messages:
        assert "1 to 24 regex strings" in message[0]["content"]
        assert "{max_scan_patterns}" not in message[0]["content"]
