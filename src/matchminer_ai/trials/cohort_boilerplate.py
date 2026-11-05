"""Separate each cohort's key criteria from its boilerplate criteria.

Cohort extraction (cohort_workflow) copies every eligibility criterion that applies to a cohort
verbatim, including boilerplate that most cancer trials share: consent, compliance, sample and
biopsy requirements, measurable disease, prognosis, performance status, organ function, brain
metastases, toxicities of earlier treatment, washout periods of 30 days or less from radiation,
systemic or trial therapy, recovery from recent major surgery, non-cancer conditions, unrelated
other cancers, participation in another trial at the same time, contraception, pregnancy and
lactation, vaccine restrictions, study-staff and catch-all exclusions (each unless it is central
to what the trial studies). :func:`remove_boilerplate_criteria` splits each cohort's criteria into
items, one per line (:func:`split_criteria`), and asks ``trial.cohorts.deboilerplate.user.txt``
whether each item is boilerplate, one request per item showing the cohort's numbered criteria
without the trial document. Code then sorts the unchanged items into the key criteria (those that
are not boilerplate: what distinguishes the cohort's population) and the boilerplate criteria
(:func:`gather_criteria`); the model never copies the criteria, so none can be lost or reworded.
TrialSpace retrieval and TrialChecker read the key criteria; BoilerplateChecker reads the
boilerplate criteria. It is optional at inference; training uses it after
:func:`~matchminer_ai.trials.merge_equivalent_cohorts`.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import pandas as pd

from matchminer_ai._storage import atomic_json
from matchminer_ai.config import MMAIConfig, load_default_preset
from matchminer_ai.llm.backends import get_llm_backend
from matchminer_ai.llm.prompt_rendering import build_prompt_list
from matchminer_ai.llm.prompts import load_prompt_text

from .cohort_merge import _section
from .cohort_workflow import _load_object

DEBOILERPLATE_PROMPT = "trial.cohorts.deboilerplate.user.txt"
DEBOILERPLATE_SCHEMA_VERSION = "trial-cohorts-deboilerplate-v3"
DEBOILERPLATE_JSON_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["boilerplate"],
    "properties": {"boilerplate": {"type": "boolean"}},
}
KEY_COLUMN = "cohort_key_criteria"
BOILERPLATE_COLUMN = "cohort_boilerplate_criteria"
_LABEL = re.compile(r"\s*(inclusion|exclusion)\s+criteria\s*:?\s*", re.IGNORECASE)


def split_criteria(criteria: str) -> list[tuple[str, str]]:
    """The cohort's items in order: (label, line) for each nonblank line other than an
    "Inclusion criteria:" or "Exclusion criteria:" label, where label is the last such label
    above the line ("" before any), and line keeps its text and indentation."""
    items, label = [], ""
    for line in criteria.split("\n"):
        match = _LABEL.fullmatch(line)
        if match:
            label = f"{match.group(1).capitalize()} criteria:"
        elif line.strip():
            items.append((label, line.rstrip()))
    return items


def gather_criteria(items: list[tuple[str, str]], boilerplate: list[bool]) -> tuple[str, str]:
    """(criteria that are not boilerplate, boilerplate criteria): the items in their original order,
    each under its label, in the list its classification gives; a list without items is ""."""
    lists: dict[bool, list[str]] = {False: [], True: []}
    labels: dict[bool, str | None] = {False: None, True: None}
    for (label, line), flag in zip(items, boilerplate, strict=True):
        if label and label != labels[flag]:
            lists[flag].append(label)
        labels[flag] = label
        lists[flag].append(line)
    return "\n".join(lists[False]), "\n".join(lists[True])


def build_deboilerplate_messages(name: str, items: list[tuple[str, str]], number: int, prompt: str) -> list[dict[str, str]]:
    """The request for item ``number`` (from 1). Everything before the item is the same for all
    of a cohort's items, so a server's prefix cache can reuse it (formatted once so braces stay literal)."""
    numbered, label = [], None
    for n, (item_label, line) in enumerate(items, 1):
        if item_label and item_label != label:
            numbered.append(item_label)
        label = item_label
        numbered.append(f"[{n}] {line}")
    return [{"role": "user", "content": prompt.format(cohort_name=name, eligibility_criteria="\n".join(numbered),
                                                      item_number=number, item=items[number - 1][1].strip())}]


def parse_deboilerplated(text: str, finish_reason: str = "stop") -> bool:
    """Whether the item is boilerplate."""
    value = _load_object(text, finish_reason, ["boilerplate"])["boilerplate"]
    if not isinstance(value, bool):
        raise ValueError("boilerplate must be true or false")
    return value


def remove_boilerplate_criteria(
    cohorts: pd.DataFrame,
    *,
    config: MMAIConfig | None = None,
    section: str = "trial",
    output_dir: str | Path | None = None,
    return_metadata: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    """Each cohort's eligibility criteria separated into key and boilerplate criteria.

    ``cohorts`` holds cohort rows (``trial_id``, ``cohort_number``, ``cohort_name``,
    ``cohort_eligibility_criteria``; other columns are kept), extracted or merged. Each row's
    criteria are split into items (:func:`split_criteria`); each item gets one LLM request with
    ``trial.cohorts.deboilerplate.user.txt`` and the LLM settings of ``section`` ("trial" for
    ctgov listings, "protocol_cohort_extraction" for protocols), whose answer is validated and
    retried on failure up to that section's ``max_attempts``. An item whose answer never validates
    counts as not boilerplate and is reported in the metadata's ``unclassified_items``. The items
    are then gathered unchanged (:func:`gather_criteria`). A cohort without items, or whose items
    are all boilerplate, keeps its criteria unchanged, with no boilerplate criteria, and is
    reported in ``deboilerplate_errors``. Both are keyed by ``cohort_trial_id`` when the rows have
    it, otherwise ``<trial_id>-cohort-<cohort_number>``.

    Returns the rows in their original order, ``cohort_eligibility_criteria`` unchanged (all the
    criteria), with a ``cohort_key_criteria`` column holding the key criteria (the criteria that are
    not boilerplate; all the criteria for a cohort in ``deboilerplate_errors``) and a
    ``cohort_boilerplate_criteria`` column holding the boilerplate criteria (None for a cohort in
    ``deboilerplate_errors``). Every item's attempts in
    the metadata's ``trial_cohorts`` (each trial's results as ``deboilerplate.json`` stores them)
    keep the model's answer and reasoning. With ``output_dir``, each trial's results are saved to
    ``<output_dir>/<trial_id>/deboilerplate.json`` as its requests return, and an item's
    classification is reused when the prompt and its cohort's name and criteria are unchanged.
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
    texts = [str(t) for t in frame.cohort_eligibility_criteria.fillna("")]
    inputs = [hashlib.sha256(json.dumps([name, text]).encode()).hexdigest() for name, text in zip(names, texts)]
    items = [split_criteria(text) for text in texts]
    metadata = {"schema_version": DEBOILERPLATE_SCHEMA_VERSION, "prompt_sha256": prompt_sha,
                "deboilerplate_errors": {}, "unclassified_items": {}, "skipped_completed": 0,
                "items": sum(map(len, items)), "items_reused": 0}
    labels: dict[tuple[int, int], bool] = {}  # (row, item number) -> boilerplate
    attempts: dict[tuple[int, int], list[dict[str, Any]]] = {}
    failed: dict[tuple[int, int], str] = {}
    rows_of: dict[Any, list[int]] = {}
    for row, trial_id in enumerate(frame.trial_id):
        rows_of.setdefault(trial_id, []).append(row)

    pending = []
    for trial_id, rows in rows_of.items():
        path = root / str(trial_id) / "deboilerplate.json" if root else None
        record = json.loads(path.read_text(encoding="utf-8")) if path is not None and path.exists() else {}
        current = (record.get("schema_version") == DEBOILERPLATE_SCHEMA_VERSION
                   and record.get("prompt_sha256") == prompt_sha)
        saved = {e["input_sha256"]: e for e in record.get("cohorts", []) if current}
        for row in rows:
            done = {i["number"]: i for i in saved.get(inputs[row], {}).get("items", []) if "boilerplate" in i}
            for number in range(1, len(items[row]) + 1):
                if number in done:
                    labels[row, number], attempts[row, number] = done[number]["boilerplate"], done[number].get("attempts", [])
                    metadata["items_reused"] += 1
                else:  # a cohort's items stay together so their shared prefix is cached
                    messages = build_deboilerplate_messages(names[row], items[row], number, prompt)
                    pending.append((row, number, 1, messages, messages))
            if items[row] and all((row, n) in labels for n in range(1, len(items[row]) + 1)):
                metadata["skipped_completed"] += 1

    def result(row):
        """(criteria, boilerplate or None, error or None) once every item has an answer, else None."""
        numbers = range(1, len(items[row]) + 1)
        if not all((row, n) in labels or (row, n) in failed for n in numbers):
            return None
        criteria, boilerplate = gather_criteria(items[row], [labels.get((row, n), False) for n in numbers])
        if not criteria:
            return texts[row], None, ("no criteria to classify" if not items[row]
                                      else "every item was classified as boilerplate; criteria kept as extracted")
        return criteria, boilerplate, None

    def entries_of(trial_id):
        entries = []
        for r in rows_of[trial_id]:
            numbered = [{"number": n, "label": label, "text": line,
                         **({"boilerplate": labels[r, n]} if (r, n) in labels else {}),
                         **({"error": failed[r, n]} if (r, n) in failed else {}),
                         "attempts": attempts.get((r, n), [])}
                        for n, (label, line) in enumerate(items[r], 1) if (r, n) in attempts]
            outcome = result(r)
            if not numbered and outcome is None:
                continue
            entry = {"cohort_number": int(frame.cohort_number.iloc[r]), "cohort_name": names[r], "input_sha256": inputs[r]}
            if outcome is not None:
                if outcome[2] is None:
                    entry["key_criteria"], entry["boilerplate_criteria"] = outcome[0], outcome[1]
                else:
                    entry["error"] = outcome[2]
                if any((r, n) in failed for n in range(1, len(items[r]) + 1)):
                    entry["unclassified_items"] = [n for n in range(1, len(items[r]) + 1) if (r, n) in failed]
            entries.append({**entry, "items": numbered})
        return entries

    def save(trial_id):
        atomic_json(root / str(trial_id) / "deboilerplate.json", {
            "trial_id": trial_id, "schema_version": DEBOILERPLATE_SCHEMA_VERSION, "prompt_sha256": prompt_sha,
            "cohorts": entries_of(trial_id), "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})

    if pending:
        runtime, settings = _section(resolved, section, "trial_cohorts_deboilerplate", DEBOILERPLATE_JSON_SCHEMA)
        metadata["structured_outputs"] = settings["structured"]
        backend = get_llm_backend(resolved)
        while pending:
            chunks = [pending[i:i + settings["chunk_size"]] for i in range(0, len(pending), settings["chunk_size"])]
            with ThreadPoolExecutor(max_workers=settings["chunks_in_flight"]) as pool:
                generations = pool.map(lambda chunk: (chunk, backend.generate_llm_outputs(
                    prompt_list=build_prompt_list([item[3] for item in chunk], llm_config=runtime),
                    llm_config=runtime, model_metadata_cache_dir=resolved.model_metadata_cache_dir)), chunks)
                pending = []
                for chunk, generation in generations:  # each chunk is saved as soon as it returns
                    provenance = {item["row_idx"]: item for item in generation.model_metadata.get("responses", [])}
                    touched = set()
                    for position, ((row, number, attempt, _, original), answer, reasoning, finish) in enumerate(zip(
                            chunk, generation.final_outputs, generation.reasoning_outputs, generation.finish_reasons,
                            strict=True)):
                        entry = {"attempt": attempt, "finish_reason": finish, "answer": answer, "reasoning": reasoning}
                        if position in provenance:
                            entry["response_metadata"] = provenance[position]
                        attempts.setdefault((row, number), []).append(entry)
                        touched.add(frame.trial_id.iloc[row])
                        try:
                            labels[row, number] = parse_deboilerplated(answer, finish)
                        except ValueError as exc:
                            entry["validation_error"] = str(exc)
                            if attempt < settings["max_attempts"]:
                                pending.append((row, number, attempt + 1, original + [{"role": "user", "content": (
                                    f"The preceding attempt failed validation: {exc}. Return a complete replacement "
                                    "JSON object following the original instructions.")}], original))
                                continue
                            failed[row, number] = str(exc)
                    if root is not None:
                        for trial_id in touched:
                            save(trial_id)

    outcomes = [result(r) for r in range(len(frame))]
    for r, (_, _, error) in enumerate(outcomes):
        if error is not None:
            metadata["deboilerplate_errors"][keys[r]] = error
        unclassified = [n for n in range(1, len(items[r]) + 1) if (r, n) in failed]
        if unclassified:
            metadata["unclassified_items"][keys[r]] = {n: failed[r, n] for n in unclassified}
    result_frame = frame.copy()
    result_frame[KEY_COLUMN] = [criteria for criteria, _, _ in outcomes]
    result_frame[BOILERPLATE_COLUMN] = pd.Series([boilerplate for _, boilerplate, _ in outcomes], dtype=object)
    metadata["cohorts"] = len(frame)
    metadata["trial_cohorts"] = {trial_id: entries_of(trial_id) for trial_id in rows_of}
    if root is not None:
        for trial_id in rows_of:  # cohorts whose items were all reused, or that have no items
            if not (root / str(trial_id) / "deboilerplate.json").exists():
                save(trial_id)
    return (result_frame, metadata) if return_metadata else result_frame


__all__ = ["BOILERPLATE_COLUMN", "KEY_COLUMN", "DEBOILERPLATE_JSON_SCHEMA", "DEBOILERPLATE_PROMPT", "DEBOILERPLATE_SCHEMA_VERSION",
           "build_deboilerplate_messages", "gather_criteria", "parse_deboilerplated", "remove_boilerplate_criteria",
           "split_criteria"]
