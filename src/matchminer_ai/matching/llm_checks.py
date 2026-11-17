"""LLM-based match quality, exclusion screening and per-criterion checking helpers."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from importlib import resources
import json
from typing import TYPE_CHECKING, Any

import pandas as pd

from matchminer_ai._metadata import package_metadata
from matchminer_ai.config import config_snapshot, load_default_preset
from matchminer_ai.llm.backends import (
    build_llm_runtime_config,
    get_llm_backend,
)
from matchminer_ai.llm.prompt_rendering import build_prompt_list
from matchminer_ai.llm.reasoning import reasoning_system_prompt
from matchminer_ai.llm.request_queue import RequestQueue

if TYPE_CHECKING:
    from matchminer_ai.config import MMAIConfig


# Highest match score: reasonableness 1, cancer type 1, burden 1, treatment history 1, biomarker 2.
MAX_MATCH_QUALITY_SCORE = 6


def _load_prompt_template(filename: str) -> str:
    """Load an LLM checker prompt template from package prompt resources."""
    prompt_path = resources.files("matchminer_ai.prompts").joinpath(filename)
    with prompt_path.open("r", encoding="utf-8") as handle:
        return handle.read().strip()


def _build_messages(user_content: str) -> list[dict[str, str]]:
    """Wrap task-specific user text in the chat message format used by checks."""
    return [
        {"role": "system", "content": ""},
        {"role": "user", "content": user_content},
    ]


def _parse_exclusion_result(response_text: str) -> tuple[bool | None, str]:
    """Parse an LLM exclusion answer into pass/fail/unknown and parse status."""
    tail = response_text[-10:].upper()
    if "YES!" in tail:
        return False, "parsed"
    if "NO!" in tail:
        return True, "parsed"
    if "YES" in tail:
        return False, "parsed_fallback"
    if "NO" in tail:
        return True, "parsed_fallback"
    return None, "parse_failed"


def _run_llm_check(
    rows: pd.DataFrame,
    *,
    config: "MMAIConfig",
    section_name: str,
    messages_list: list[list[dict[str, str]]],
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Render prompts and run the configured LLM backend for one check task."""
    llm_config = dict(getattr(config, section_name))
    if not llm_config:
        raise ValueError(f"Config is missing '{section_name}' settings.")
    runtime_config = build_llm_runtime_config(
        section_name,
        llm_config,
        config=config,
    )
    # A check's own system instructions follow the model family's reasoning header, if any.
    messages_list = [
        [{"role": "system", "content": reasoning_system_prompt(runtime_config.get("model_name"),
                                                               messages[0]["content"])}, *messages[1:]]
        for messages in messages_list
    ]
    prompt_list = build_prompt_list(messages_list, llm_config=runtime_config)
    backend = get_llm_backend(config)
    generation = backend.generate_llm_outputs(
        prompt_list=prompt_list,
        llm_config=runtime_config,
        model_metadata_cache_dir=config.model_metadata_cache_dir,
    )
    if len(generation.final_outputs) != len(rows):
        raise ValueError("LLM returned a different number of outputs than input rows.")
    return (
        generation.final_outputs,
        generation.reasoning_outputs,
        generation.model_metadata,
    )


def score_match_quality_with_llm(
    candidate_pairs: pd.DataFrame,
    *,
    config: "MMAIConfig | None" = None,
    return_metadata: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """
    Label candidate patient–cohort pairs with the five match labels, in one LLM prompt per pair.

    The prompt (``llm_match_quality.prompt_file``, by default the training teacher's
    ``match_quality.labels.system.txt``) asks whether the patient is a reasonable candidate on the
    cohort's key criteria and whether the cohort specifically requires, and the patient documentedly
    satisfies, its cancer type, burden, treatment history and biomarker requirements
    (:mod:`matchminer_ai.matching.match_labels`). Only key text is read: boilerplate is
    ``exclusion_criteria_check_with_llm``'s.

    Parameters
    ----------
    candidate_pairs : pd.DataFrame
        One row per candidate pair: ``patient_id``; ``cohort_trial_id`` (or historical
        ``space_trial_id``); the patient's key section, ``cancer_history_summary`` (or a complete
        ``patient_summary``, whose key section is taken); and the cohort's key criteria,
        ``cohort_key_criteria`` (else ``cohort_eligibility_criteria``, else ``clinical_space_summary``).
    config : MMAIConfig, optional
        MMAI configuration with ``llm_match_quality`` settings; default preset when omitted.
    return_metadata : bool, default False
        When True, also return a metadata dict with the config snapshot and model metadata.

    Returns
    -------
    pd.DataFrame
        ``patient_id``, the trial identifier, ``llm_match_quality_score`` (0–6 from
        :func:`~matchminer_ai.matching.match_labels.match_score`, or -1 when the answer fails the
        contract) and one boolean ``llm_<label>`` column per label (None when unparsed). With
        debug_mode: ``llm_match_quality_answer_text``, ``llm_match_quality_reasoning_text`` and
        ``llm_match_quality_parse_status`` (``parsed`` or the validation error).
    tuple[pd.DataFrame, dict]
        When return_metadata is True, returns the DataFrame plus metadata.
    """
    from matchminer_ai._texts import patient_key_texts, trial_id_column, trial_key_texts
    from .match_labels import LABELS, build_match_label_messages, match_score, parse_match_labels

    if "patient_id" not in candidate_pairs.columns:
        raise ValueError("candidate_pairs is missing required columns: patient_id")
    trial_id = trial_id_column(candidate_pairs)
    patients, cohorts = patient_key_texts(candidate_pairs), trial_key_texts(candidate_pairs)

    resolved_config = config or load_default_preset()
    llm_config = dict(resolved_config.llm_match_quality)
    prompt = _load_prompt_template(str(llm_config["prompt_file"]).strip())
    messages_list = [build_match_label_messages(patient, cohort, prompt) for patient, cohort in zip(patients, cohorts)]
    responses, reasonings, model_metadata = _run_llm_check(
        candidate_pairs,
        config=resolved_config,
        section_name="llm_match_quality",
        messages_list=messages_list,
    )
    parsed, statuses = [], []
    for response in responses:
        try:
            parsed.append(parse_match_labels(response))
            statuses.append("parsed")
        except ValueError as exc:
            parsed.append(None)
            statuses.append(f"parse_failed: {exc}")

    output = candidate_pairs[["patient_id", trial_id]].copy()
    output["llm_match_quality_score"] = [match_score(p) if p else -1 for p in parsed]
    for label in LABELS:
        output[f"llm_{label}"] = pd.Series([p[label]["answer"] == "yes" if p else None for p in parsed],
                                           index=output.index, dtype=object)
    if resolved_config.debug_mode:
        output["llm_match_quality_answer_text"] = responses
        output["llm_match_quality_reasoning_text"] = reasonings
        output["llm_match_quality_parse_status"] = statuses
    output = output.reset_index(drop=True)

    if return_metadata:
        metadata_payload = {
            "package": package_metadata(),
            "config_snapshot": config_snapshot(resolved_config),
            "model_metadata": {
                "score_match_quality_with_llm": model_metadata,
            },
        }
        return output, metadata_payload
    return output


def exclusion_criteria_check_with_llm(
    matches: pd.DataFrame,
    *,
    config: "MMAIConfig | None" = None,
    return_metadata: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """
    Evaluate a cohort's boilerplate criteria with the configured LLM prompt.

    The prompt (``llm_exclusion_criteria.prompt_file``; the training teacher's boilerplate-label
    prompt) asks whether the patient clearly fails any of the cohort's boilerplate criteria,
    inclusion requirements as well as exclusions, from the patient summary's boilerplate section.

    Parameters
    ----------
    matches : pd.DataFrame
        DataFrame of candidate patient–cohort pairs.

        Expected columns
        ----------------
        patient_id : str
            Patient identifier.
        cohort_trial_id : str
            Cohort identifier (historical ``space_trial_id`` or ``trial_id`` is accepted).
        cohort_boilerplate_criteria : str
            The cohort's boilerplate criteria (historical ``general_exclusion_criteria`` is accepted).
        general_exclusion_criteria_evidence : str
            The patient summary's boilerplate section (or a complete ``patient_summary``, whose
            boilerplate section is taken).

    config : MMAIConfig, optional
        MMAI configuration containing LLM exclusion checker settings.
        Uses default preset when omitted.
    return_metadata : bool, default False
        When True, also return a metadata dict containing the config snapshot
        and model metadata for this run.

    Returns
    -------
    pd.DataFrame
        Derived output table containing:

        Columns
        -------
        patient_id : str
            Patient identifier.
        cohort_trial_id, space_trial_id or trial_id : str
            The input's trial identifier.
        llm_exclusion_criteria_pass : bool | None
            Whether the LLM judged that the patient passes exclusion criteria,
            or None when parsing failed.
        Debug Columns
        -------------
        llm_exclusion_criteria_answer_text : str
            Text the package treated as the LLM answer and used for parsing.
            Included only when debug_mode is true.
        llm_exclusion_criteria_reasoning_text : str
            Optional separate reasoning trace returned by the backend or
            extracted by the configured reasoning parser. Included only when
            debug_mode is true.
        llm_exclusion_criteria_parse_status : str
            Whether the package could parse the answer text. Values are
            ``parsed``, ``parsed_fallback``, or ``parse_failed``. Included only
            when debug_mode is true.
    tuple[pd.DataFrame, dict]
        When return_metadata is True, returns the DataFrame plus metadata.
    """
    from .exclusion_check import _pair_inputs

    trial_id, evidence, criteria = _pair_inputs(matches)
    resolved_config = config or load_default_preset()
    llm_config = dict(resolved_config.llm_exclusion_criteria)
    prompt_template = _load_prompt_template(str(llm_config["prompt_file"]).strip())
    messages_list = [
        _build_messages(prompt_template.format(patient_boilerplate=patient, trial_boilerplate=cohort))
        for patient, cohort in zip(evidence, criteria, strict=True)
    ]
    responses, reasonings, model_metadata = _run_llm_check(
        matches,
        config=resolved_config,
        section_name="llm_exclusion_criteria",
        messages_list=messages_list,
    )
    parsed = [_parse_exclusion_result(response) for response in responses]

    output = matches[["patient_id", trial_id]].copy()
    output["llm_exclusion_criteria_pass"] = [passed for passed, _ in parsed]
    if resolved_config.debug_mode:
        output["llm_exclusion_criteria_answer_text"] = responses
        output["llm_exclusion_criteria_reasoning_text"] = reasonings
        output["llm_exclusion_criteria_parse_status"] = [
            parse_status for _, parse_status in parsed
        ]
    output = output.reset_index(drop=True)

    if return_metadata:
        metadata_payload = {
            "package": package_metadata(),
            "config_snapshot": config_snapshot(resolved_config),
            "model_metadata": {
                "exclusion_criteria_check_with_llm": model_metadata,
            },
        }
        return output, metadata_payload
    return output


CRITERIA_SECTION = "llm_criteria_check"
_CRITERIA_LLM_KEYS = {"local", "remote", "reasoning_parser", "sampling_profile", "reasoning_effort"}


def _criteria_runtime(config: "MMAIConfig") -> tuple[dict[str, Any], dict[str, Any]]:
    """(runtime, settings) of ``llm_criteria_check``: the request settings, and the queue and retry settings."""
    settings = deepcopy(dict(config.raw.get(CRITERIA_SECTION) or {}))
    if not settings:
        raise ValueError(f"Config is missing '{CRITERIA_SECTION}' settings.")
    llm = {key: value for key, value in settings.items() if key in _CRITERIA_LLM_KEYS}
    runtime = build_llm_runtime_config(CRITERIA_SECTION, llm, config=config)
    structured = bool(settings.get("structured_outputs", True)) and runtime["backend_mode"] == "remote"
    if structured:  # the label contract's answer schema, as the training teacher decoded it
        from .decision_model import CRITERION_LABEL_JSON_SCHEMA

        runtime["remote"]["request_params"]["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "criterion_check", "strict": True, "schema": CRITERION_LABEL_JSON_SCHEMA},
        }
    return runtime, {"max_attempts": max(1, int(settings.get("max_attempts", 3))),
                     "chunk_size": max(1, int(settings.get("chunk_size", 64))),
                     "chunks_in_flight": max(1, int(settings.get("chunks_in_flight", 1))),
                     "structured": structured}


def check_eligibility_criteria_with_llm(
    pairs: pd.DataFrame,
    *,
    config: "MMAIConfig | None" = None,
    filter_failing: bool = False,
    return_metadata: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """
    Check every eligibility-criterion line of each candidate pair with an LLM, one request per line.

    The LLM alternative to :func:`~matchminer_ai.matching.check_eligibility_criteria`, with the same
    inputs and the same polarity. Each request is the per-criterion teacher prompt that the decision
    model's question 7 is trained on: ``criterion_check.labels.user.txt`` around the decision model's
    own criterion state and question
    (:func:`~matchminer_ai.matching.decision_model.criterion_teacher_messages`; label contract
    ``matchminer_ai.labeling.contract_hash("criteria")``), that is the complete patient summary, the
    cohort's complete numbered criteria and one line restated with its number and type. On a remote
    backend the answer is decoded under ``CRITERION_LABEL_JSON_SCHEMA``
    (``llm_criteria_check.structured_outputs``) and always validated by
    :func:`~matchminer_ai.matching.decision_model.parse_criterion_label`; an invalid answer is asked
    again with the validation error, up to ``llm_criteria_check.max_attempts``. A pair's lines are
    sent through :class:`~matchminer_ai.llm.request_queue.RequestQueue`, so they run in parallel and
    share the prompt prefix.

    **Polarity: an answer of yes means the patient does NOT clearly fail the line** (it is met, its
    information is missing, or the line does not bind this patient); **no means the patient clearly
    fails it: an inclusion line clearly not met, or an exclusion line clearly met.**

    Parameters
    ----------
    pairs : pd.DataFrame
        Candidate patient–cohort pairs, as for :func:`~matchminer_ai.matching.check_eligibility_criteria`:
        ``patient_id``; ``cohort_trial_id`` (or historical ``space_trial_id``);
        ``cohort_criteria_items`` (the cohort's tagged lines, a JSON list of ``number``, ``heading``,
        ``text``, ``criterion_type`` and ``boilerplate``); and the patient summary's two sections,
        ``cancer_history_summary`` and ``general_exclusion_criteria_evidence`` (or one complete
        ``patient_summary``).
    config : MMAIConfig, optional
        ``llm_criteria_check`` (the model, its sampling profile, structured outputs, attempts and
        requests in flight) and ``remote``. Uses the default preset when omitted. With
        ``debug_mode`` the last attempt's answer text and reasoning are returned too.
    filter_failing : bool, default False
        If True, return only the lines the LLM judged clearly failed.
    return_metadata : bool, default False
        When True, also return the config snapshot, model metadata and request counts.
    progress : callable, optional
        ``progress(done, total)`` after each line's final answer (``total`` lines are asked).

    Returns
    -------
    pd.DataFrame
        One row per pair and checkable line (a line whose type never validated is skipped and
        counted in the metadata's ``unchecked_lines``):

        patient_id, cohort_trial_id or space_trial_id
            The pair.
        criterion_number, criterion_type, boilerplate, criterion_text
            The line, as :func:`~matchminer_ai.matching.check_eligibility_criteria` returns it.
        llm_criterion_answer : str or None
            ``"yes"`` (does not clearly fail the line) or ``"no"`` (clearly fails it); None when no
            attempt validated.
        llm_criterion_clearly_fails : bool or None
            True when the answer is no; None when no attempt validated.
        llm_criterion_status : str or None
            How the line binds the patient: ``met``, ``not_met``, ``insufficient_information`` or
            ``not_applicable``.
        llm_criterion_reason : str
            The model's brief explanation (empty when no attempt validated).
        llm_criterion_finish_reason : str
            The last attempt's finish reason (``stop`` when complete).
        llm_criterion_attempts : int
            Requests sent for the line.
        llm_criterion_parse_status : str
            ``parsed``, or ``parse_failed: <the last validation error>``.
        llm_criterion_answer_text, llm_criterion_reasoning_text : str
            The last attempt's answer and reasoning; with ``debug_mode`` only.
    tuple[pd.DataFrame, dict]
        When return_metadata is True, returns the DataFrame plus metadata: ``package``,
        ``config_snapshot``, ``criteria_contract`` and ``model_metadata``
        (``check_eligibility_criteria_with_llm``: the endpoint's metadata plus ``unchecked_lines``,
        ``lines``, ``requests``, ``retries``, ``parse_failures`` and ``structured_outputs``).
    """
    from matchminer_ai._texts import patient_full_texts, trial_id_column
    from matchminer_ai.labeling.tasks import contract_hash
    from matchminer_ai.trials.cohort_boilerplate import ITEMS_COLUMN, load_criteria_items
    from .decision_model import checkable_items, criterion_teacher_messages, parse_criterion_label

    if "patient_id" not in pairs.columns:
        raise ValueError("pairs is missing required columns: patient_id")
    if ITEMS_COLUMN not in pairs.columns:
        raise ValueError(f"pairs needs {ITEMS_COLUMN}: tag the cohorts' criteria lines with "
                         "matchminer_ai.trials.remove_boilerplate_criteria first")
    trial_id = trial_id_column(pairs)
    resolved_config = config or load_default_preset()
    summaries = patient_full_texts(pairs)
    keys, requests, unchecked = [], {}, 0
    for summary, value in zip(summaries, pairs[ITEMS_COLUMN], strict=True):
        items = load_criteria_items(value)
        checkable = checkable_items(items)
        unchecked += len(items) - len(checkable)
        key = json.dumps([summary, items], ensure_ascii=False)
        keys.append(key)
        if key not in requests and checkable:  # identical summary and criteria are asked once
            requests[key] = (summary, items, checkable)
    total = sum(len(checkable) for _, _, checkable in requests.values())
    answers: dict[tuple[str, int], dict[str, Any]] = {}
    counts = {"requests": 0, "retries": 0}
    model_metadata: dict[str, Any] = {}
    structured = False
    if total:
        runtime, settings = _criteria_runtime(resolved_config)
        structured = settings["structured"]
        header = reasoning_system_prompt(runtime.get("model_name"))
        queue = RequestQueue(get_llm_backend(resolved_config), {None: runtime}, chunk_size=settings["chunk_size"],
                             chunks_in_flight=settings["chunks_in_flight"],
                             model_metadata_cache_dir=resolved_config.model_metadata_cache_dir)
        originals: dict[tuple[str, int], list[dict[str, str]]] = {}
        types: dict[tuple[str, int], str] = {}
        for key, (summary, items, checkable) in requests.items():
            for item in checkable:
                number = int(item["number"])
                messages = criterion_teacher_messages(summary, items, number)
                if header:  # only a model family that needs a system reasoning control gets one
                    messages = [{"role": "system", "content": header}, *messages]
                originals[key, number], types[key, number] = messages, str(item["criterion_type"])
                queue.add((key, number, 1), messages)
                counts["requests"] += 1

        def record(item, answer, reasoning, finish_reason, _provenance):
            key, number, attempt = item
            finish = str(finish_reason or "stop")
            state = answers.setdefault((key, number), {})
            state.update(attempts=attempt, answer_text=str(answer or ""), reasoning_text=str(reasoning or ""),
                         finish_reason=finish)
            try:
                state.update(parsed=parse_criterion_label(str(answer or ""), types[key, number], finish), error=None)
            except ValueError as exc:
                state.update(parsed=None, error=str(exc))
                if attempt < settings["max_attempts"]:
                    queue.add((key, number, attempt + 1), originals[key, number] + [{"role": "user", "content": (
                        f"The preceding attempt failed validation: {exc}. Return a complete replacement JSON object "
                        "following the original instructions.")}])
                    counts["requests"] += 1
                    counts["retries"] += 1
                    return
            if progress is not None:
                progress(sum(1 for s in answers.values() if s["parsed"] is not None or s["attempts"]
                             >= settings["max_attempts"]), total)

        model_metadata = dict(queue.run(record))

    records = []
    for (_, row), key in zip(pairs.iterrows(), keys, strict=True):
        if key not in requests:
            continue
        for item in requests[key][2]:
            state = answers.get((key, int(item["number"])), {})
            parsed = state.get("parsed")
            line = {"patient_id": row["patient_id"], trial_id: row[trial_id],
                      "criterion_number": int(item["number"]), "criterion_type": item["criterion_type"],
                      "boilerplate": bool(item.get("boilerplate")), "criterion_text": str(item["text"]).strip(),
                      "llm_criterion_answer": parsed["answer"] if parsed else None,
                      "llm_criterion_clearly_fails": (parsed["answer"] == "no") if parsed else None,
                      "llm_criterion_status": parsed["criterion_status"] if parsed else None,
                      "llm_criterion_reason": parsed["reason"].strip() if parsed else "",
                      "llm_criterion_finish_reason": state.get("finish_reason", ""),
                      "llm_criterion_attempts": int(state.get("attempts", 0)),
                      "llm_criterion_parse_status": "parsed" if parsed else f"parse_failed: {state.get('error')}"}
            if resolved_config.debug_mode:
                line["llm_criterion_answer_text"] = state.get("answer_text", "")
                line["llm_criterion_reasoning_text"] = state.get("reasoning_text", "")
            records.append(line)
    columns = ["patient_id", trial_id, "criterion_number", "criterion_type", "boilerplate", "criterion_text",
               "llm_criterion_answer", "llm_criterion_clearly_fails", "llm_criterion_status", "llm_criterion_reason",
               "llm_criterion_finish_reason", "llm_criterion_attempts", "llm_criterion_parse_status"]
    if resolved_config.debug_mode:
        columns += ["llm_criterion_answer_text", "llm_criterion_reasoning_text"]
    output = pd.DataFrame(records, columns=columns)
    for column in ("llm_criterion_answer", "llm_criterion_clearly_fails", "llm_criterion_status"):
        # None stays None for an unanswered line (a string dtype would turn it into NaN)
        output[column] = pd.Series([r[column] for r in records], index=output.index, dtype=object)
    if filter_failing:
        output = output.loc[output["llm_criterion_clearly_fails"].eq(True)].reset_index(drop=True)
    if return_metadata:
        metadata = {**model_metadata, "unchecked_lines": unchecked, "lines": total, **counts,
                    "parse_failures": sum(1 for s in answers.values() if s.get("parsed") is None),
                    "structured_outputs": structured}
        return output, {"package": package_metadata(), "config_snapshot": config_snapshot(resolved_config),
                        "criteria_contract": contract_hash("criteria"),
                        "model_metadata": {"check_eligibility_criteria_with_llm": metadata}}
    return output


__all__ = [
    "CRITERIA_SECTION",
    "check_eligibility_criteria_with_llm",
    "exclusion_criteria_check_with_llm",
    "score_match_quality_with_llm",
]
