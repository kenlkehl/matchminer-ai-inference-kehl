"""Remove the boilerplate criteria from extracted cohorts.

Cohort extraction (cohort_workflow) copies every eligibility criterion that applies to a cohort
verbatim, including boilerplate that most cancer trials share: consent, compliance, sample and
biopsy requirements, measurable disease, contraception, pregnancy and lactation, vaccine
restrictions, study-staff and catch-all exclusions. :func:`remove_boilerplate_criteria` sends
each cohort's criteria, without the trial document, to ``trial.cohorts.deboilerplate.user.txt``
and keeps the answer: the same criteria with the boilerplate removed. It is optional at inference;
training uses it for the cohorts that seed synthetic patients, after
:func:`~matchminer_ai.trials.merge_equivalent_cohorts`.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from matchminer_ai._storage import atomic_json
from matchminer_ai.config import MMAIConfig, load_default_preset
from matchminer_ai.llm.backends import get_llm_backend
from matchminer_ai.llm.prompt_rendering import build_prompt_list
from matchminer_ai.llm.prompts import load_prompt_text

from .cohort_merge import _section
from .cohort_workflow import JSON_SCHEMAS, parse_criteria

DEBOILERPLATE_PROMPT = "trial.cohorts.deboilerplate.user.txt"
DEBOILERPLATE_SCHEMA_VERSION = "trial-cohorts-deboilerplate-v1"
DEBOILERPLATE_JSON_SCHEMA = JSON_SCHEMAS["criteria"]


def build_deboilerplate_messages(name: str, criteria: str, prompt: str) -> list[dict[str, str]]:
    """One cohort's name and criteria in the prompt (formatted once so braces stay literal)."""
    return [{"role": "user", "content": prompt.format(cohort_name=name, eligibility_criteria=criteria)}]


def parse_deboilerplated(text: str, finish_reason: str = "stop") -> str:
    """The cohort's criteria without boilerplate: the criteria step's contract."""
    return parse_criteria(text, finish_reason)


def remove_boilerplate_criteria(
    cohorts: pd.DataFrame,
    *,
    config: MMAIConfig | None = None,
    section: str = "trial",
    output_dir: str | Path | None = None,
    return_metadata: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    """Each cohort's eligibility criteria with the boilerplate criteria removed.

    ``cohorts`` holds cohort rows (``trial_id``, ``cohort_number``, ``cohort_name``,
    ``cohort_eligibility_criteria``; other columns are kept), extracted or merged. Each row gets
    one LLM request with ``trial.cohorts.deboilerplate.user.txt`` and the LLM settings of
    ``section`` ("trial" for registry listings, "protocol_cohort_extraction" for protocols); the
    answer is validated like an extracted cohort's criteria and retried on failure up to that
    section's ``max_attempts``. A cohort whose answer never validates keeps its criteria unchanged
    and is reported in the metadata's ``deboilerplate_errors`` (by ``cohort_trial_id`` when the
    rows have it, otherwise ``<trial_id>-cohort-<cohort_number>``).

    Returns the rows in their original order with ``cohort_eligibility_criteria`` replaced.
    Every attempt in the metadata's ``cohort_attempts`` keeps the model's answer and reasoning.
    With ``output_dir``, each trial's results are saved to
    ``<output_dir>/<trial_id>/deboilerplate.json`` (with their attempts) and a cohort's result is
    reused when the prompt and that cohort's name and criteria are unchanged.
    """
    resolved = config or load_default_preset()
    frame = cohorts.reset_index(drop=True).copy()
    missing = {"trial_id", "cohort_number", "cohort_name", "cohort_eligibility_criteria"} - set(frame)
    if missing:
        raise ValueError(f"cohorts is missing columns: {sorted(missing)}")
    prompt = load_prompt_text(DEBOILERPLATE_PROMPT)
    prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
    root = Path(output_dir) if output_dir is not None else None
    keys = (list(frame.cohort_trial_id) if "cohort_trial_id" in frame
            else [f"{t}-cohort-{n}" for t, n in zip(frame.trial_id, frame.cohort_number)])
    names = [str(n) for n in frame.cohort_name.fillna("Cohort")]
    inputs = [hashlib.sha256(json.dumps([name, text]).encode()).hexdigest()
              for name, text in zip(names, frame.cohort_eligibility_criteria)]
    metadata = {"schema_version": DEBOILERPLATE_SCHEMA_VERSION, "prompt_sha256": prompt_sha,
                "deboilerplate_errors": {}, "cohort_attempts": {}, "skipped_completed": 0}
    results: dict[int, str] = {}
    rows_of: dict[Any, list[int]] = {}
    for row, trial_id in enumerate(frame.trial_id):
        rows_of.setdefault(trial_id, []).append(row)

    saved: dict[str, dict[str, dict[str, Any]]] = {}  # trial -> input_sha256 -> saved cohort entry
    for trial_id in rows_of:
        path = root / str(trial_id) / "deboilerplate.json" if root else None
        record = json.loads(path.read_text(encoding="utf-8")) if path is not None and path.exists() else {}
        current = (record.get("schema_version") == DEBOILERPLATE_SCHEMA_VERSION
                   and record.get("prompt_sha256") == prompt_sha)
        saved[trial_id] = {e["input_sha256"]: e for e in record.get("cohorts", []) if current and "eligibility_criteria" in e}

    pending = []
    for row, trial_id in enumerate(frame.trial_id):
        entry = saved[trial_id].get(inputs[row])
        if entry is not None:
            results[row] = entry["eligibility_criteria"]
            metadata["skipped_completed"] += 1
            continue
        messages = build_deboilerplate_messages(names[row], frame.cohort_eligibility_criteria.iloc[row], prompt)
        pending.append((row, 1, messages, messages))

    def save(trial_id):
        entries = []
        for r in rows_of[trial_id]:
            entry = {"cohort_number": int(frame.cohort_number.iloc[r]), "cohort_name": names[r], "input_sha256": inputs[r]}
            if r in results:
                entry["eligibility_criteria"] = results[r]
            elif keys[r] in metadata["deboilerplate_errors"]:
                entry["error"] = metadata["deboilerplate_errors"][keys[r]]
            else:
                continue
            if keys[r] in metadata["cohort_attempts"]:
                entry["attempts"] = metadata["cohort_attempts"][keys[r]]
            elif inputs[r] in saved[trial_id]:
                entry["attempts"] = saved[trial_id][inputs[r]].get("attempts", [])
            entries.append(entry)
        atomic_json(root / str(trial_id) / "deboilerplate.json", {
            "trial_id": trial_id, "schema_version": DEBOILERPLATE_SCHEMA_VERSION, "prompt_sha256": prompt_sha,
            "cohorts": entries, "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})

    if pending:
        runtime, settings = _section(resolved, section, "trial_cohorts_deboilerplate", DEBOILERPLATE_JSON_SCHEMA)
        metadata["structured_outputs"] = settings["structured"]
        backend = get_llm_backend(resolved)
        while pending:
            chunks = [pending[i:i + settings["chunk_size"]] for i in range(0, len(pending), settings["chunk_size"])]
            with ThreadPoolExecutor(max_workers=settings["chunks_in_flight"]) as pool:
                generations = pool.map(lambda chunk: (chunk, backend.generate_llm_outputs(
                    prompt_list=build_prompt_list([item[2] for item in chunk], llm_config=runtime),
                    llm_config=runtime, model_metadata_cache_dir=resolved.model_metadata_cache_dir)), chunks)
                pending = []
                for chunk, generation in generations:  # each chunk is saved as soon as it returns
                    provenance = {item["row_idx"]: item for item in generation.model_metadata.get("responses", [])}
                    touched = set()
                    for position, ((row, attempt, _, original), answer, reasoning, finish) in enumerate(zip(
                            chunk, generation.final_outputs, generation.reasoning_outputs, generation.finish_reasons,
                            strict=True)):
                        entry = {"attempt": attempt, "finish_reason": finish, "answer": answer, "reasoning": reasoning}
                        if position in provenance:
                            entry["response_metadata"] = provenance[position]
                        metadata["cohort_attempts"].setdefault(keys[row], []).append(entry)
                        try:
                            results[row] = parse_deboilerplated(answer, finish)
                        except ValueError as exc:
                            entry["validation_error"] = str(exc)
                            if attempt < settings["max_attempts"]:
                                pending.append((row, attempt + 1, original + [{"role": "user", "content": (
                                    f"The preceding attempt failed validation: {exc}. Return a complete replacement "
                                    "JSON object following the original instructions.")}], original))
                                continue
                            metadata["deboilerplate_errors"][keys[row]] = str(exc)
                        touched.add(frame.trial_id.iloc[row])
                    if root is not None:
                        for trial_id in touched:
                            save(trial_id)

    result = frame.copy()
    result["cohort_eligibility_criteria"] = [results.get(r, text) for r, text in enumerate(frame.cohort_eligibility_criteria)]
    metadata["cohorts"] = len(frame)
    return (result, metadata) if return_metadata else result


__all__ = ["DEBOILERPLATE_JSON_SCHEMA", "DEBOILERPLATE_PROMPT", "DEBOILERPLATE_SCHEMA_VERSION",
           "build_deboilerplate_messages", "parse_deboilerplated", "remove_boilerplate_criteria"]
