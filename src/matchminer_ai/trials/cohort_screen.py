"""Screen trial documents for cancer treatment trials: the cohort workflow's first step on its own.

Training draws its trial sample from trials this screen accepts, so the same prompt
(``trial.cohorts.screen.user.txt``), settings and response contract as the workflow's
``screen`` step decide which trials can be sampled, and every answer and its reasoning is
kept. :func:`screen_trials` takes ClinicalTrials.gov listings (title, brief summary and
eligibility criteria as one document, as :func:`registry_trial_text` builds it) or full
documents.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from matchminer_ai.config import MMAIConfig, load_default_preset
from matchminer_ai.llm.backends import get_llm_backend
from matchminer_ai.llm.prompt_rendering import build_prompt_list

from .cohort_merge import _section
from .cohort_workflow import JSON_SCHEMAS, build_messages, load_prompts, parse_screen, registry_trial_text

SCREEN_SCHEMA_VERSION = "trial-cohorts-screen-v1"
SCREEN_FILE = "screen.jsonl"


def _documents(frame: pd.DataFrame) -> list[str]:
    if "trial_text" in frame:
        return [text if isinstance(text, str) else "" for text in frame.trial_text]
    title = "trial_title" if "trial_title" in frame else "title"
    missing = {title, "brief_summary", "eligibility_criteria"} - set(frame)
    if missing:
        raise ValueError(f"trials needs trial_text or registry columns; missing {sorted(missing)}")
    return [registry_trial_text(t, s, e) for t, s, e in zip(frame[title], frame.brief_summary, frame.eligibility_criteria)]


def screen_trials(
    trials: pd.DataFrame,
    *,
    config: MMAIConfig | None = None,
    section: str = "trial",
    output_dir: str | Path | None = None,
    return_metadata: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    """Decide for each trial whether it is a cancer treatment trial.

    ``trials`` holds ``trial_id`` and either ``trial_text`` or a listing's ``title`` (or
    ``trial_title``), ``brief_summary`` and ``eligibility_criteria``. Each trial gets the
    workflow's screen request with the LLM settings of ``section`` ("trial" or
    "protocol_cohort_extraction"); an answer that fails validation is retried up to that
    section's ``max_attempts``, and a trial whose answer never validates is not rejected but
    reported with its error. With ``output_dir``, each trial's attempts (answers, reasoning and
    response metadata) are appended to ``<output_dir>/screen.jsonl`` as its requests return, and a
    saved decision is reused when the prompt and the trial's document are unchanged.

    Returns one row per trial, in input order: ``trial_id``, ``cancer_treatment_trial`` ("yes",
    "no", or None when no answer validated) and ``screen_error``. The metadata holds the schema
    version, prompt hash, counts, errors and every trial's attempts.
    """
    resolved = config or load_default_preset()
    frame = trials.reset_index(drop=True)
    if "trial_id" not in frame or frame.trial_id.isna().any() or frame.trial_id.duplicated().any():
        raise ValueError("trials needs unique, present trial_id values")
    ids = [str(t) for t in frame.trial_id]
    texts = _documents(frame)
    prompts = load_prompts()
    prompt_sha = hashlib.sha256(prompts["screen"].encode()).hexdigest()
    inputs = [hashlib.sha256(text.encode()).hexdigest() for text in texts]
    path = Path(output_dir) / SCREEN_FILE if output_dir is not None else None
    saved: dict[tuple[str, str], dict[str, Any]] = {}
    if path is not None and path.exists():
        for line in path.read_text().splitlines():
            entry = json.loads(line)
            if entry.get("prompt_sha256") == prompt_sha and entry.get("decision") in ("yes", "no"):
                saved[entry["trial_id"], entry["input_sha256"]] = entry
    decisions: dict[int, str] = {}
    errors: dict[int, str] = {}
    attempts: dict[int, list[dict[str, Any]]] = {}
    metadata: dict[str, Any] = {"schema_version": SCREEN_SCHEMA_VERSION, "prompt_sha256": prompt_sha,
                                "trials": len(frame), "reused": 0, "structured_outputs": None}
    pending = []
    for row, (trial_id, digest) in enumerate(zip(ids, inputs)):
        if (trial_id, digest) in saved:
            decisions[row], attempts[row] = saved[trial_id, digest]["decision"], saved[trial_id, digest]["attempts"]
            metadata["reused"] += 1
        else:
            messages = build_messages("screen", texts[row], prompts)
            pending.append((row, 1, messages, messages))

    def write(rows):
        if path is None or not rows:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with path.open("a") as handle:
            for row in rows:
                entry = {"trial_id": ids[row], "schema_version": SCREEN_SCHEMA_VERSION, "prompt_sha256": prompt_sha,
                         "input_sha256": inputs[row], "decision": decisions.get(row),
                         **({"error": errors[row]} if row in errors else {}),
                         "attempts": attempts.get(row, []), "completed_at": now}
                handle.write(json.dumps(entry) + "\n")

    if pending:
        runtime, settings = _section(resolved, section, "trial_cohorts_screen", JSON_SCHEMAS["screen"])
        metadata["structured_outputs"] = settings["structured"]
        backend = get_llm_backend(resolved)
        while pending:
            chunks = [pending[i:i + settings["chunk_size"]] for i in range(0, len(pending), settings["chunk_size"])]
            with ThreadPoolExecutor(max_workers=settings["chunks_in_flight"]) as pool:
                generations = pool.map(lambda chunk: (chunk, backend.generate_llm_outputs(
                    prompt_list=build_prompt_list([item[2] for item in chunk], llm_config=runtime),
                    llm_config=runtime, model_metadata_cache_dir=resolved.model_metadata_cache_dir)), chunks)
                pending = []
                for chunk, generation in generations:  # each chunk's finished trials are saved as it returns
                    provenance = {item["row_idx"]: item for item in generation.model_metadata.get("responses", [])}
                    finished = []
                    for position, ((row, attempt, _, original), answer, reasoning, finish) in enumerate(zip(
                            chunk, generation.final_outputs, generation.reasoning_outputs, generation.finish_reasons,
                            strict=True)):
                        entry = {"attempt": attempt, "finish_reason": finish, "answer": answer, "reasoning": reasoning}
                        if position in provenance:
                            entry["response_metadata"] = provenance[position]
                        attempts.setdefault(row, []).append(entry)
                        try:
                            decisions[row] = parse_screen(answer, finish)
                        except ValueError as exc:
                            entry["validation_error"] = str(exc)
                            if attempt < settings["max_attempts"]:
                                pending.append((row, attempt + 1, original + [{"role": "user", "content": (
                                    f"The preceding attempt failed validation: {exc}. Return a complete replacement "
                                    "JSON object following the original instructions and the full document.")}],
                                    original))
                                continue
                            errors[row] = str(exc)
                        finished.append(row)
                    write(finished)

    result = pd.DataFrame({"trial_id": ids,
                           "cancer_treatment_trial": pd.Series([decisions.get(r) for r in range(len(ids))], dtype=object),
                           "screen_error": pd.Series([errors.get(r) for r in range(len(ids))], dtype=object)})
    metadata.update(accepted=sum(d == "yes" for d in decisions.values()),
                    rejected=sum(d == "no" for d in decisions.values()),
                    screen_errors={ids[r]: e for r, e in errors.items()},
                    trial_attempts={ids[r]: attempts.get(r, []) for r in range(len(ids))})
    return (result, metadata) if return_metadata else result


__all__ = ["SCREEN_FILE", "SCREEN_SCHEMA_VERSION", "screen_trials"]
