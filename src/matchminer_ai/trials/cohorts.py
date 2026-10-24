"""Combined treatment screening and cohort extraction for summarize_trials."""

from __future__ import annotations

from copy import deepcopy
import hashlib

import pandas as pd

from matchminer_ai._qc.common import build_qc_artifact, qc_artifact_to_report_row
from matchminer_ai.llm.backends import build_llm_runtime_config, get_llm_backend
from matchminer_ai.llm.prompt_rendering import build_prompt_list

from .cohort_schema import (
    COHORT_PROMPT, COHORT_SCHEMA_VERSION, build_cohort_messages,
    load_cohort_prompt, parse_cohort_response,
)
from .prompt_builder import build_trial_text

# The model's JSON schema stays v2; this versions the derived row representation.
COHORT_REPRESENTATION_VERSION = "trial-treatment-cohorts-v3"

COHORT_COLUMNS = [
    "cohort_trial_id", "trial_id", "cohort_number", "cohort_eligibility_criteria",
    "cohort_specific_eligibility_criteria", "common_eligibility_criteria",
    "cohort_criteria_source", "cancer_treatment_trial", "multiple_cohorts",
]


def summarize_trial_cohorts(trials, config):
    """Return accepted cohort rows, decision/error provenance, and cohort QC."""
    frame = trials.copy().reset_index(drop=True)
    ids = frame["trial_id"]
    if ids.isna().any() or ids.astype(str).str.strip().eq("").any():
        raise ValueError("trial_id must be nonempty")
    frame["trial_id"] = ids.astype(str)
    if not frame.trial_id.is_unique:
        raise ValueError("trial_id must be unique")
    for column in ("trial_title", "brief_summary", "eligibility_criteria"):
        if not frame[column].dropna().map(lambda v: isinstance(v, str)).all():
            raise ValueError(f"{column} must contain strings or missing values")
    frame["trial_text"] = build_trial_text(frame)
    if frame.trial_text.str.strip().eq("").any():
        raise ValueError("Each trial must contain nonempty source text")

    trial_config = deepcopy(config.trial)
    settings = trial_config.pop("cohorts", {})
    max_attempts = settings.get("max_attempts", 3)
    if type(max_attempts) is not int or not 1 <= max_attempts <= 10:
        raise ValueError("trial.cohorts.max_attempts must be an integer from 1 to 10")
    trial_config.pop("extraction_method", None)
    trial_config.pop("hierarchy", None)
    filename = trial_config.get("prompt_files", {}).get("cohorts", COHORT_PROMPT)
    prompt = load_cohort_prompt(filename)
    runtime = build_llm_runtime_config("trial", trial_config, config=config)
    metadata = {
        "extraction_method": "cohorts", "schema_version": COHORT_SCHEMA_VERSION,
        "representation_version": COHORT_REPRESENTATION_VERSION,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "model_metadata": {}, "trial_decisions": {}, "trial_rejections": {},
        "trial_errors": {}, "trial_attempts": {},
    }
    if config.debug_mode:
        metadata["trial_responses"] = {}
    messages = {
        i: build_cohort_messages(text, prompt=prompt)
        for i, text in enumerate(frame.trial_text)
    }
    pending = list(messages)
    backend = get_llm_backend(config) if pending else None
    last_responses = {}
    last_finishes = {}
    for attempt in range(1, max_attempts + 1):
        if not pending:
            break
        generation = backend.generate_llm_outputs(
            prompt_list=build_prompt_list([messages[i] for i in pending], llm_config=runtime),
            llm_config=runtime,
            model_metadata_cache_dir=config.model_metadata_cache_dir,
        )
        metadata["model_metadata"] = generation.model_metadata
        next_pending = []
        provenance = {item['row_idx']: item for item in
                      generation.model_metadata.get('responses', [])}
        for response_index, (i, answer, reasoning, finish) in enumerate(zip(
            pending, generation.final_outputs, generation.reasoning_outputs,
            generation.finish_reasons, strict=True,
        )):
            trial_id = frame.at[i, "trial_id"]
            last_responses[i] = (answer, reasoning)
            last_finishes[i] = finish
            record = {"attempt": attempt, "finish_reason": finish}
            if response_index in provenance:
                record['response_metadata'] = provenance[response_index]
            if config.debug_mode:
                record.update(answer=answer, reasoning=reasoning)
            try:
                parsed = parse_cohort_response(answer, finish)
            except ValueError as exc:
                record["validation_error"] = str(exc)
                metadata["trial_errors"][trial_id] = str(exc)
                next_pending.append(i)
                # Keep the complete source and original instructions. Retry
                # only this trial, without changing vendor sampling/thinking.
                messages[i] = build_cohort_messages(frame.at[i, "trial_text"], prompt=prompt)
                messages[i].append({
                    "role": "user",
                    "content": (
                        f"The preceding attempt failed validation: {exc}. "
                        "Return a complete replacement JSON object following the "
                        "original instructions and full source document."
                    ),
                })
            else:
                metadata["trial_errors"].pop(trial_id, None)
                metadata["trial_decisions"][trial_id] = parsed
                if parsed["cancer_treatment_trial"] == "no":
                    metadata["trial_rejections"][trial_id] = parsed
            metadata["trial_attempts"].setdefault(trial_id, []).append(record)
            if config.debug_mode:
                metadata["trial_responses"][trial_id] = {
                    "answer": answer, "reasoning": reasoning, "finish_reason": finish,
                }
        pending = next_pending

    rows = []
    for i, source in frame.iterrows():
        parsed = metadata["trial_decisions"].get(source.trial_id)
        if parsed is None or parsed["cancer_treatment_trial"] == "no":
            continue
        multiple = parsed["multiple_cohorts"] == "yes"
        # Screening uses the full document, but a single population carries only
        # the original eligibility field. Never substitute title/summary text.
        if not multiple and (not isinstance(source.eligibility_criteria, str)
                             or not source.eligibility_criteria.strip()):
            metadata["trial_errors"][source.trial_id] = (
                "Single-population trial has no original eligibility_criteria"
            )
            continue
        criteria = parsed["cohorts"] if multiple else [source.eligibility_criteria]
        common = parsed["common_eligibility_criteria"] if multiple else None
        for number, text in enumerate(criteria):
            complete = text
            if multiple and common:
                complete = (
                    f"Cohort-specific eligibility criteria:\n{text}\n\n"
                    f"Common eligibility criteria:\n{common}"
                )
            row = {
                "cohort_trial_id": f"{source.trial_id}-cohort-{number}",
                "trial_id": source.trial_id, "cohort_number": number,
                "cohort_eligibility_criteria": complete,
                "cohort_specific_eligibility_criteria": text if multiple else None,
                "common_eligibility_criteria": common,
                "cohort_criteria_source": "llm" if multiple else "eligibility_criteria",
                "cancer_treatment_trial": "yes",
                "multiple_cohorts": parsed["multiple_cohorts"],
            }
            if config.debug_mode:
                row.update(
                    trial_input_text=source.trial_text,
                    trial_answer_text=last_responses[i][0],
                    trial_reasoning_text=last_responses[i][1],
                )
            rows.append(row)
    columns = COHORT_COLUMNS + ([
        "trial_input_text", "trial_answer_text", "trial_reasoning_text",
    ] if config.debug_mode else [])
    result = pd.DataFrame(rows, columns=columns)
    result["cohort_number"] = result.cohort_number.astype("int64")
    # Retain decisions even when return_metadata=False, including all-rejected runs.
    result.attrs.update({key: metadata[key] for key in (
        "extraction_method", "schema_version", "representation_version", "prompt_sha256",
        "trial_decisions", "trial_rejections", "trial_errors",
    )})
    qc = []
    for metric, trial_ids in (
        ("trials_rejected_cancer_treatment", list(metadata["trial_rejections"])),
        ("trials_failed_inference", list(metadata["trial_errors"])),
        ("trials_truncated_llm_response", [
            frame.at[i, "trial_id"] for i, finish in last_finishes.items() if finish == "length"
        ]),
    ):
        qc.append(qc_artifact_to_report_row(build_qc_artifact(
            metric=metric, ids=trial_ids, denominator=len(frame),
        )))
    return result, metadata, pd.DataFrame(qc)
