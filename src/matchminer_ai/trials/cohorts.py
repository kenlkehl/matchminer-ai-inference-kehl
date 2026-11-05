"""Treatment screening and cohort extraction from ctgov listings for extract_cohorts."""

from __future__ import annotations

from copy import deepcopy

import pandas as pd

from matchminer_ai._qc.common import build_qc_artifact, qc_artifact_to_report_row
from matchminer_ai.llm.backends import build_llm_runtime_config, get_llm_backend

from .cohort_workflow import (
    COHORT_COLUMNS, REPRESENTATION_VERSION, SCHEMA_VERSION, decision_rows, load_prompts, prompt_sha256,
    ctgov_trial_text, run_cohort_workflow, step_runtimes,
)

# Kept as names for callers of the earlier single-request module.
COHORT_REPRESENTATION_VERSION = REPRESENTATION_VERSION
COHORT_SCHEMA_VERSION = SCHEMA_VERSION


def cohort_settings(settings, section):
    """Validated attempts, chunking and structured-output settings of one config section."""
    max_attempts = settings.get("max_attempts", 3)
    if type(max_attempts) is not int or not 1 <= max_attempts <= 10:
        raise ValueError(f"{section}.max_attempts must be an integer from 1 to 10")
    chunk_size = settings.get("chunk_size", 500)
    chunks_in_flight = settings.get("chunks_in_flight", 2)
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError(f"{section}.chunk_size must be a positive integer")
    if type(chunks_in_flight) is not int or not 1 <= chunks_in_flight <= 16:
        raise ValueError(f"{section}.chunks_in_flight must be an integer from 1 to 16")
    return max_attempts, chunk_size, chunks_in_flight, bool(settings.get("structured_outputs", True))


def workflow_metadata(states, prompts, structured, extraction_method):
    """Decisions, rejections, errors and attempts per trial, as both extraction APIs report them."""
    hashes = prompt_sha256(prompts)
    metadata = {
        "extraction_method": extraction_method, "schema_version": SCHEMA_VERSION,
        "representation_version": REPRESENTATION_VERSION, "structured_outputs": structured,
        "prompt_sha256": hashes["combined"], "prompt_sha256_by_step": {s: hashes[s] for s in prompts},
        "trial_decisions": {}, "trial_rejections": {}, "trial_errors": {}, "trial_attempts": {},
        "trial_cohort_names": {}, "trial_merged_cohorts": {},
    }
    for trial_id, state in states.items():
        metadata["trial_attempts"][trial_id] = state["attempts"]
        if state["cohort_names"] is not None:
            metadata["trial_cohort_names"][trial_id] = state["cohort_names"]
        if state["merged_cohorts"]:
            metadata["trial_merged_cohorts"][trial_id] = state["merged_cohorts"]
        decision = state["decision"]
        if decision is None:
            metadata["trial_errors"][trial_id] = state["error"] or "No validated decision"
            continue
        metadata["trial_decisions"][trial_id] = decision
        if decision["cancer_treatment_trial"] == "no":
            metadata["trial_rejections"][trial_id] = decision
    return metadata


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
    if frame[["trial_title", "brief_summary", "eligibility_criteria"]].fillna("").apply(
            lambda column: column.str.strip()).eq("").all(axis=1).any():
        raise ValueError("Each trial must contain nonempty source text")
    # The same document shape for every listing: title, brief summary, eligibility criteria.
    frame["trial_text"] = [ctgov_trial_text(*values) for values in
                           frame[["trial_title", "brief_summary", "eligibility_criteria"]].itertuples(index=False)]

    trial_config = deepcopy(config.trial)
    settings = trial_config.pop("cohorts", {})
    max_attempts, chunk_size, chunks_in_flight, structured_setting = cohort_settings(settings, "trial.cohorts")
    trial_config.pop("extraction_method", None)
    trial_config.pop("hierarchy", None)
    prompts = load_prompts(settings.get("prompt_files"))
    runtime = build_llm_runtime_config("trial", trial_config, config=config)
    structured = structured_setting and runtime["backend_mode"] == "remote"
    documents = list(zip(frame.trial_id, frame.trial_text))
    backend = get_llm_backend(config) if documents else None
    states, model_metadata = run_cohort_workflow(
        documents, prompts=prompts, runtimes=step_runtimes(runtime, structured), backend=backend,
        max_attempts=max_attempts, model_metadata_cache_dir=config.model_metadata_cache_dir,
        chunk_size=chunk_size, chunks_in_flight=chunks_in_flight, debug=config.debug_mode,
    ) if documents else ({}, {})
    metadata = workflow_metadata(states, prompts, structured, "cohorts")
    metadata["model_metadata"] = model_metadata
    if config.debug_mode:
        metadata["trial_responses"] = {trial_id: state["responses"] for trial_id, state in states.items()}

    rows = []
    for source in frame.itertuples(index=False):
        decision = metadata["trial_decisions"].get(source.trial_id)
        if decision is None or decision["cancer_treatment_trial"] == "no":
            continue
        for row in decision_rows(source.trial_id, decision, source="ctgov"):
            if config.debug_mode:
                row["trial_input_text"] = source.trial_text
            rows.append(row)
    result = pd.DataFrame(rows, columns=COHORT_COLUMNS + (["trial_input_text"] if config.debug_mode else []))
    result["cohort_number"] = result.cohort_number.astype("int64")
    # Retain decisions even when return_metadata=False, including all-rejected runs.
    result.attrs.update({key: metadata[key] for key in (
        "extraction_method", "schema_version", "representation_version", "prompt_sha256",
        "structured_outputs", "trial_decisions", "trial_rejections", "trial_errors",
    )})
    qc = []
    for metric, trial_ids in (
        ("trials_rejected_cancer_treatment", list(metadata["trial_rejections"])),
        ("trials_failed_inference", list(metadata["trial_errors"])),
        ("trials_truncated_llm_response", [t for t, state in states.items() if state["truncated"]]),
    ):
        qc.append(qc_artifact_to_report_row(build_qc_artifact(
            metric=metric, ids=trial_ids, denominator=len(frame),
        )))
    return result, metadata, pd.DataFrame(qc)
