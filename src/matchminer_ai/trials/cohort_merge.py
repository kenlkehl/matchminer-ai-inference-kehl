"""Combine a trial's extracted cohorts whose eligibility criteria are essentially the same.

Production extraction (cohort_workflow) lists every cohort a trial defines, which can include
nearly redundant ones, such as arms that differ only in treatment or dose. Training wants one
row per distinct eligibility population, so :func:`merge_equivalent_cohorts` asks the LLM,
once per trial with more than one cohort, which cohorts have essentially the same criteria
(``trial.cohorts.merge.user.txt``). The model returns only a grouping by cohort number; no
criteria are regenerated. Each group becomes one row whose ``cohort_name`` joins the merged
names with "; " and whose criteria are those of the group's first cohort.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from matchminer_ai._storage import atomic_json
from matchminer_ai.config import MMAIConfig, load_default_preset
from matchminer_ai.llm.backends import build_llm_runtime_config, get_llm_backend
from matchminer_ai.llm.prompt_rendering import build_prompt_list
from matchminer_ai.llm.prompts import load_prompt_text

from .cohort_schema import _unique_keys, unwrap_json_fence
from .cohort_workflow import COHORT_COLUMNS

MERGE_PROMPT = "trial.cohorts.merge.user.txt"
MERGE_SCHEMA_VERSION = "trial-cohorts-merge-v1"
MERGE_JSON_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["groups"],
    "properties": {"groups": {"type": "array", "minItems": 1,
                              "items": {"type": "array", "minItems": 1, "items": {"type": "integer"}}}},
}
MERGED_FROM = "merged_from"


def build_merge_messages(names: list[str], criteria: list[str], prompt: str) -> list[dict[str, str]]:
    """One trial's cohorts, numbered from 1, in the merge prompt (formatted once so braces stay literal)."""
    blocks = "\n\n".join(f"### Cohort {n}: {name}\n{text}" for n, (name, text) in enumerate(zip(names, criteria), 1))
    return [{"role": "user", "content": prompt.format(cohorts=blocks)}]


def parse_merge_groups(text: str, finish_reason: str, cohort_count: int) -> list[list[int]]:
    """A partition of cohort numbers 1..cohort_count; returned as zero-based groups ordered by first member."""
    if finish_reason != "stop":
        raise ValueError(f"Incomplete response: {finish_reason}")
    try:
        result = json.loads(unwrap_json_fence(text), object_pairs_hook=_unique_keys)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Response must be valid JSON: {exc}") from exc
    if not isinstance(result, dict) or list(result) != ["groups"]:
        raise ValueError('Response must be a JSON object with exactly the field "groups"')
    groups = result["groups"]
    if not isinstance(groups, list) or not groups or not all(isinstance(g, list) and g for g in groups):
        raise ValueError("groups must be a nonempty array of nonempty arrays")
    numbers = [n for g in groups for n in g]
    if not all(type(n) is int for n in numbers):
        raise ValueError("Cohort numbers must be integers")
    if sorted(numbers) != list(range(1, cohort_count + 1)):
        raise ValueError(f"Every cohort number from 1 to {cohort_count} must appear in exactly one group")
    return sorted((sorted(n - 1 for n in g) for g in groups), key=lambda g: g[0])


def _section(config: MMAIConfig, section: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """(runtime, settings) of the trial or protocol_cohort_extraction section, as their extraction uses them."""
    from .cohorts import cohort_settings
    if section == "trial":
        llm = deepcopy(config.trial)
        settings = llm.pop("cohorts", {})
        for key in ("extraction_method", "hierarchy"):
            llm.pop(key, None)
    elif section == "protocol_cohort_extraction":
        settings = dict(config.protocol_cohort_extraction)
        llm = {key: deepcopy(value) for key, value in settings.items()
               if key in {"local", "remote", "reasoning_parser", "sampling_profile", "reasoning_effort"}}
    else:
        raise ValueError('section must be "trial" or "protocol_cohort_extraction"')
    max_attempts, chunk_size, chunks_in_flight, structured = cohort_settings(settings, section)
    runtime = build_llm_runtime_config(section, llm, config=config)
    if structured and runtime["backend_mode"] == "remote":
        runtime["remote"]["request_params"]["response_format"] = {
            "type": "json_schema", "json_schema": {"name": "trial_cohorts_merge", "strict": True, "schema": MERGE_JSON_SCHEMA}}
    return runtime, dict(max_attempts=max_attempts, chunk_size=chunk_size, chunks_in_flight=chunks_in_flight,
                         structured=structured and runtime["backend_mode"] == "remote")


def merge_equivalent_cohorts(
    cohorts: pd.DataFrame,
    *,
    config: MMAIConfig | None = None,
    section: str = "trial",
    output_dir: str | Path | None = None,
    return_metadata: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    """Combine each trial's cohorts whose eligibility criteria are essentially the same.

    ``cohorts`` holds extraction rows (``COHORT_COLUMNS``, any extra columns are kept), for one
    or many trials. Each trial with more than one cohort gets one LLM request with
    ``trial.cohorts.merge.user.txt`` and the LLM settings of ``section`` ("trial" for registry
    listings, "protocol_cohort_extraction" for protocols); its answer is validated as a
    partition of the cohort numbers and retried on failure up to that section's
    ``max_attempts``. A trial whose answer never validates keeps its cohorts unmerged and is
    reported in the metadata's ``merge_errors``.

    Returns one row per group, in order of the group's first cohort: the first cohort's row
    with ``cohort_name`` set to the merged names joined by "; ", ``cohort_number`` and
    ``cohort_trial_id`` renumbered, ``multiple_cohorts`` recomputed, and ``merged_from``
    listing the original cohort numbers (JSON). The criteria are the first cohort's, unchanged.
    Every attempt in the metadata's ``trial_attempts`` keeps the model's answer and reasoning.
    With ``output_dir``, each trial's grouping is saved to ``<output_dir>/<trial_id>/merge.json``
    (with its attempts) and reused when the prompt and the trial's cohort names and criteria are
    unchanged.
    """
    resolved = config or load_default_preset()
    frame = cohorts.copy()
    missing = {"trial_id", "cohort_number", "cohort_name", "cohort_eligibility_criteria"} - set(frame)
    if missing:
        raise ValueError(f"cohorts is missing columns: {sorted(missing)}")
    frame["_order"] = range(len(frame))
    trials = list(dict.fromkeys(frame.trial_id))
    groups_of = {t: g.sort_values("cohort_number") for t, g in frame.groupby("trial_id", sort=False)}
    prompt = load_prompt_text(MERGE_PROMPT)
    prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
    root = Path(output_dir) if output_dir is not None else None
    metadata = {"schema_version": MERGE_SCHEMA_VERSION, "prompt_sha256": prompt_sha, "trial_groups": {},
                "merge_errors": {}, "trial_attempts": {}, "skipped_completed": 0}

    def identity(group):
        content = json.dumps([[n, c] for n, c in zip(group.cohort_name, group.cohort_eligibility_criteria)])
        return {"schema_version": MERGE_SCHEMA_VERSION, "prompt_sha256": prompt_sha,
                "input_sha256": hashlib.sha256(content.encode()).hexdigest()}

    pending = []
    for trial_id in trials:
        group = groups_of[trial_id]
        if len(group) < 2:
            metadata["trial_groups"][trial_id] = [[0]]
            continue
        saved = root / str(trial_id) / "merge.json" if root else None
        if saved is not None and saved.exists():
            record = json.loads(saved.read_text(encoding="utf-8"))
            if all(record.get(k) == v for k, v in identity(group).items()):
                metadata["trial_groups"][trial_id] = record["groups"]
                metadata["skipped_completed"] += 1
                continue
        names = [str(n) for n in group.cohort_name.fillna("Cohort")]
        messages = build_merge_messages(names, list(group.cohort_eligibility_criteria), prompt)
        pending.append((trial_id, 1, messages, messages))

    if pending:
        runtime, settings = _section(resolved, section)
        metadata["structured_outputs"] = settings["structured"]
        backend = get_llm_backend(resolved)
        while pending:
            chunks = [pending[i:i + settings["chunk_size"]] for i in range(0, len(pending), settings["chunk_size"])]
            with ThreadPoolExecutor(max_workers=settings["chunks_in_flight"]) as pool:
                results = list(pool.map(lambda chunk: (chunk, backend.generate_llm_outputs(
                    prompt_list=build_prompt_list([item[2] for item in chunk], llm_config=runtime),
                    llm_config=runtime, model_metadata_cache_dir=resolved.model_metadata_cache_dir)), chunks))
            pending = []
            for chunk, generation in results:
                provenance = {item["row_idx"]: item for item in generation.model_metadata.get("responses", [])}
                for position, ((trial_id, attempt, _, original), answer, reasoning, finish) in enumerate(zip(
                        chunk, generation.final_outputs, generation.reasoning_outputs, generation.finish_reasons,
                        strict=True)):
                    entry = {"attempt": attempt, "finish_reason": finish, "answer": answer, "reasoning": reasoning}
                    if position in provenance:
                        entry["response_metadata"] = provenance[position]
                    metadata["trial_attempts"].setdefault(trial_id, []).append(entry)
                    try:
                        parsed = parse_merge_groups(answer, finish, len(groups_of[trial_id]))
                    except ValueError as exc:
                        entry["validation_error"] = str(exc)
                        if attempt < settings["max_attempts"]:
                            pending.append((trial_id, attempt + 1, original + [{"role": "user", "content": (
                                f"The preceding attempt failed validation: {exc}. Return a complete replacement "
                                "JSON object following the original instructions.")}], original))
                        else:
                            metadata["merge_errors"][trial_id] = str(exc)
                            metadata["trial_groups"][trial_id] = [[i] for i in range(len(groups_of[trial_id]))]
                        continue
                    metadata["trial_groups"][trial_id] = parsed
                    if root is not None:
                        atomic_json(root / str(trial_id) / "merge.json", {
                            "trial_id": trial_id, **identity(groups_of[trial_id]), "groups": parsed,
                            "cohort_names": list(groups_of[trial_id].cohort_name),
                            "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            "attempts": metadata["trial_attempts"][trial_id]})

    rows = []
    for trial_id in trials:
        group = groups_of[trial_id].reset_index(drop=True)
        partition = metadata["trial_groups"][trial_id]
        for number, members in enumerate(partition):
            row = group.iloc[members[0]].to_dict()
            row.update(cohort_name="; ".join(str(group.cohort_name.iloc[i]) for i in members),
                       cohort_number=number, cohort_trial_id=f"{trial_id}-cohort-{number}",
                       multiple_cohorts="yes" if len(partition) > 1 else "no",
                       **{MERGED_FROM: json.dumps([int(group.cohort_number.iloc[i]) for i in members])})
            rows.append(row)
    extra = [c for c in frame.columns if c not in COHORT_COLUMNS and c not in ("_order", MERGED_FROM)]
    result = pd.DataFrame(rows, columns=[c for c in COHORT_COLUMNS if c in frame.columns] + extra + [MERGED_FROM])
    if "cohort_number" in result:
        result["cohort_number"] = result.cohort_number.astype("int64")
    metadata["cohorts_in"], metadata["cohorts_out"] = len(frame), len(result)
    return (result, metadata) if return_metadata else result


__all__ = ["MERGE_JSON_SCHEMA", "MERGE_PROMPT", "MERGE_SCHEMA_VERSION", "MERGED_FROM", "build_merge_messages",
           "merge_equivalent_cohorts", "parse_merge_groups"]
