"""Three-step cohort extraction shared by ctgov listings and full protocols.

One trial document (a ClinicalTrials.gov listing's title, brief summary and
eligibility criteria, or a full protocol) goes through the same three prompts:

1. ``screen``: is it a cancer treatment trial (``trial.cohorts.screen.user.txt``)?
2. ``list``: the names of the cohorts the trial defines (``trial.cohorts.list.user.txt``), without
   comparing their criteria.
3. ``criteria``: one request per cohort name for that cohort's complete
   eligibility criteria, copied verbatim and including boilerplate
   (``trial.cohorts.criteria.user.txt``).

Two optional steps follow, used by training: ``merge_equivalent_cohorts``
(cohort_merge) and ``remove_boilerplate_criteria`` (cohort_boilerplate).

Every prompt starts with the same preamble and document, and the criteria prompt
puts the cohort list and target name after its instructions, so a server's prefix
cache can reuse a trial's document across all of its requests. Each step's answer
is validated with its own strict contract and, for remote backends, constrained
by its JSON schema; only failed requests are retried. A trial's decision is
complete when every cohort's criteria are accepted; one cohort that fails every
attempt fails the trial.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
import hashlib
import json
import re
from typing import Any

from matchminer_ai.llm.prompt_rendering import build_prompt_list
from matchminer_ai.llm.prompts import load_prompt_text

from .cohort_schema import _PLACEHOLDER, _unique_keys, unwrap_json_fence

STEPS = ("screen", "list", "criteria")
PROMPT_FILES = {
    "screen": "trial.cohorts.screen.user.txt",
    "list": "trial.cohorts.list.user.txt",
    "criteria": "trial.cohorts.criteria.user.txt",
}
SCHEMA_VERSION = "trial-cohorts-stepwise-v1"
# Rows carry each cohort's complete criteria from its own request; there is no
# specific/common split (v4 rows had one).
REPRESENTATION_VERSION = "trial-treatment-cohorts-v5"
COHORT_COLUMNS = [
    "cohort_trial_id", "trial_id", "cohort_number", "cohort_name", "cohort_eligibility_criteria",
    "cohort_criteria_source", "cancer_treatment_trial", "multiple_cohorts",
]

# No minLength: under vLLM structured outputs a minLength string admitted no JSON
# escape. The parsers reject empty names and criteria instead.
_TEXT = {"type": "string"}
JSON_SCHEMAS = {
    "screen": {"type": "object", "additionalProperties": False, "required": ["cancer_treatment_trial"],
               "properties": {"cancer_treatment_trial": {"enum": ["yes", "no"]}}},
    "list": {"type": "object", "additionalProperties": False, "required": ["cohorts"],
             "properties": {"cohorts": {"type": "array", "minItems": 1, "items": _TEXT}}},
    "criteria": {"type": "object", "additionalProperties": False, "required": ["eligibility_criteria"],
                 "properties": {"eligibility_criteria": _TEXT}},
}


def response_format(step: str) -> dict[str, Any]:
    """OpenAI-compatible response_format constraining one step's answer to its schema."""
    return {"type": "json_schema",
            "json_schema": {"name": f"trial_cohorts_{step}", "strict": True, "schema": JSON_SCHEMAS[step]}}


def load_prompts(overrides: dict[str, str] | None = None) -> dict[str, str]:
    """The three prompt templates; ``overrides`` maps a step to another packaged prompt file."""
    files = {**PROMPT_FILES, **(overrides or {})}
    if set(files) != set(STEPS):
        raise ValueError(f"Cohort prompt files must be given for exactly {list(STEPS)}")
    return {step: load_prompt_text(files[step]) for step in STEPS}


def prompt_sha256(prompts: dict[str, str]) -> dict[str, str]:
    """Per-step template hashes plus ``combined``, which identifies the prompt set."""
    hashes = {step: hashlib.sha256(prompts[step].encode()).hexdigest() for step in STEPS}
    hashes["combined"] = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return hashes


def ctgov_trial_text(title: Any, brief_summary: Any, eligibility_criteria: Any) -> str:
    """A ClinicalTrials.gov listing as one document: title, brief summary, eligibility criteria."""
    text = lambda value: value.strip() if isinstance(value, str) else ""
    return (f"Title: {text(title)}\n\nBrief summary:\n{text(brief_summary)}\n\n"
            f"Eligibility criteria:\n{text(eligibility_criteria)}")


def build_messages(step: str, trial_text: str, prompts: dict[str, str], *,
                   cohort_names: list[str] | None = None, cohort_name: str | None = None) -> list[dict[str, str]]:
    """Format once, so braces in the document and names stay literal."""
    values = {"trial_text": trial_text}
    if step == "criteria":
        values.update(cohort_list="\n".join(f"- {name}" for name in cohort_names), cohort_name=cohort_name)
    return [{"role": "user", "content": prompts[step].format(**values)}]


def _load_object(text: str, finish_reason: str, fields: list[str]) -> dict[str, Any]:
    if finish_reason != "stop":
        raise ValueError(f"Incomplete response: {finish_reason}")
    try:
        # Only a complete outer JSON fence is removed; JSON is never repaired.
        result = json.loads(unwrap_json_fence(text), object_pairs_hook=_unique_keys)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Response must be valid JSON: {exc}") from exc
    if not isinstance(result, dict) or list(result) != fields:
        raise ValueError(f"Response must be a JSON object with exactly the fields {fields}")
    return result


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def parse_screen(text: str, finish_reason: str = "stop") -> str:
    """The screening decision, "yes" or "no"; malformed output is never a rejection."""
    value = _load_object(text, finish_reason, ["cancer_treatment_trial"])["cancer_treatment_trial"]
    if value not in ("yes", "no"):
        raise ValueError('cancer_treatment_trial must be the string "yes" or "no"')
    return value


def parse_cohort_list(text: str, finish_reason: str = "stop") -> list[str]:
    """One or more distinct, nonempty cohort names."""
    names = _load_object(text, finish_reason, ["cohorts"])["cohorts"]
    if not isinstance(names, list) or not names:
        raise ValueError("cohorts must be a nonempty array of cohort names")
    if not all(isinstance(name, str) and name.strip() for name in names):
        raise ValueError("Every cohort name must be a nonempty string")
    names = [name.strip() for name in names]
    if any(_PLACEHOLDER.fullmatch(name) for name in names):
        raise ValueError("A cohort name is a placeholder")
    if len({_normalized(name) for name in names}) != len(names):
        raise ValueError("Cohort names must all be different")
    return names


def parse_criteria(text: str, finish_reason: str = "stop") -> str:
    """One cohort's nonempty eligibility criteria."""
    criteria = _load_object(text, finish_reason, ["eligibility_criteria"])["eligibility_criteria"]
    if not isinstance(criteria, str) or not criteria.strip():
        raise ValueError("eligibility_criteria must be a nonempty string")
    if _PLACEHOLDER.fullmatch(criteria):
        raise ValueError("eligibility_criteria is a placeholder, not eligibility criteria")
    return criteria.strip()


PARSERS = {"screen": parse_screen, "list": parse_cohort_list, "criteria": parse_criteria}


def assemble_decision(names: list[str], criteria: list[str]) -> tuple[dict[str, Any], list[list[int]]]:
    """The trial decision; cohorts whose extracted criteria are identical are combined.

    Identical complete criteria mean one eligible population, so their names are
    joined and the first position is kept. Returns the decision and the groups merged.
    """
    groups: dict[str, list[int]] = {}
    for index, text in enumerate(criteria):
        groups.setdefault(_normalized(text), []).append(index)
    cohorts = [{"cohort_name": "; ".join(names[i] for i in members), "eligibility_criteria": criteria[members[0]]}
               for members in groups.values()]
    decision = {"cancer_treatment_trial": "yes", "multiple_cohorts": "yes" if len(cohorts) > 1 else "no",
                "cohorts": cohorts}
    return decision, [members for members in groups.values() if len(members) > 1]


def validate_decision(decision: Any) -> dict[str, Any]:
    """Check a saved or assembled stepwise decision."""
    if not isinstance(decision, dict):
        raise ValueError("Decision must be a JSON object")
    if decision.get("cancer_treatment_trial") == "no":
        if list(decision) != ["cancer_treatment_trial"]:
            raise ValueError("A rejected trial's decision has only cancer_treatment_trial")
        return decision
    if list(decision) != ["cancer_treatment_trial", "multiple_cohorts", "cohorts"] or decision["cancer_treatment_trial"] != "yes":
        raise ValueError("Accepted decision fields must be cancer_treatment_trial, multiple_cohorts, cohorts")
    cohorts = decision["cohorts"]
    if not isinstance(cohorts, list) or not cohorts or decision["multiple_cohorts"] != ("yes" if len(cohorts) > 1 else "no"):
        raise ValueError("multiple_cohorts must match the number of cohorts")
    for cohort in cohorts:
        if (not isinstance(cohort, dict) or list(cohort) != ["cohort_name", "eligibility_criteria"]
                or not all(isinstance(v, str) and v.strip() for v in cohort.values())):
            raise ValueError("Every cohort has exactly a nonempty cohort_name and eligibility_criteria")
    return decision


def decision_rows(trial_id: str, decision: dict[str, Any], *, source: str) -> list[dict[str, Any]]:
    """Cohort rows of one accepted decision; ``source`` is "ctgov" or "protocol"."""
    return [{
        "cohort_trial_id": f"{trial_id}-cohort-{number}", "trial_id": trial_id, "cohort_number": number,
        "cohort_name": cohort["cohort_name"], "cohort_eligibility_criteria": cohort["eligibility_criteria"],
        "cohort_criteria_source": source, "cancer_treatment_trial": "yes",
        "multiple_cohorts": decision["multiple_cohorts"],
    } for number, cohort in enumerate(decision["cohorts"])]


def step_runtimes(runtime: dict[str, Any], structured: bool) -> dict[str, dict[str, Any]]:
    """One runtime per step; remote ones carry that step's response schema."""
    runtimes = {}
    for step in STEPS:
        runtimes[step] = deepcopy(runtime)
        if structured:
            # vLLM applies the grammar after the reasoning section when the
            # server runs a reasoning parser, so thinking models keep reasoning.
            runtimes[step]["remote"]["request_params"]["response_format"] = response_format(step)
    return runtimes


Decided = Callable[[str, dict[str, Any], dict[str, Any]], None]


def run_cohort_workflow(
    documents: Iterable[tuple[str, str]],
    *,
    prompts: dict[str, str],
    runtimes: dict[str, dict[str, Any]],
    backend: Any,
    max_attempts: int,
    model_metadata_cache_dir: str | None = None,
    chunk_size: int = 500,
    chunks_in_flight: int = 2,
    debug: bool = False,
    on_decision: Decided | None = None,
    on_progress: Callable[[dict[str, dict[str, Any]]], None] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Run the three steps for ``(trial_id, document_text)`` pairs; return one state per trial and model metadata.

    A state has ``decision`` (None until accepted or rejected), ``error`` (the last
    validation error of a failed trial), ``cohort_names``, ``attempts`` per step
    (criteria attempts per cohort index), ``merged_cohorts`` and ``truncated``.
    Requests of one step go to the backend in chunks of ``chunk_size``, up to
    ``chunks_in_flight`` at once; later steps are scheduled first so trials finish
    early. ``on_decision(trial_id, decision, state)`` runs in this thread when a
    trial is decided.
    """
    texts = dict(documents)
    states = {trial_id: {"decision": None, "error": None, "cohort_names": None, "merged_cohorts": [],
                         "truncated": False, "attempts": {"screen": [], "list": [], "criteria": {}},
                         **({"responses": {"screen": None, "list": None, "criteria": {}}} if debug else {})}
              for trial_id in texts}
    queues: dict[str, list[tuple]] = {step: [] for step in STEPS}
    criteria: dict[str, dict[int, str]] = {}
    model_metadata: dict[str, Any] = {}

    def messages(trial_id, step, index):
        names = states[trial_id]["cohort_names"]
        return build_messages(step, texts[trial_id], prompts, cohort_names=names,
                              cohort_name=names[index] if step == "criteria" else None)

    for trial_id in texts:
        queues["screen"].append((trial_id, "screen", None, 1, messages(trial_id, "screen", None)))

    def finish(trial_id, decision):
        states[trial_id]["decision"] = decision
        states[trial_id]["error"] = None
        if on_decision is not None:
            on_decision(trial_id, decision, states[trial_id])

    def record(item, answer, reasoning, finish_reason, provenance):
        trial_id, step, index, attempt, sent = item
        state = states[trial_id]
        if state["decision"] is not None or (state["error"] is not None and state.get("failed")):
            return  # another cohort's request already failed the trial
        entry = {"attempt": attempt, "finish_reason": finish_reason}
        if provenance is not None:
            entry["response_metadata"] = provenance
        if debug:
            entry.update(answer=answer, reasoning=reasoning)
            if step == "criteria":
                state["responses"]["criteria"][index] = {"answer": answer, "reasoning": reasoning}
            else:
                state["responses"][step] = {"answer": answer, "reasoning": reasoning}
        log = state["attempts"]["criteria"].setdefault(index, []) if step == "criteria" else state["attempts"][step]
        log.append(entry)
        state["truncated"] |= finish_reason == "length"
        try:
            parsed = PARSERS[step](answer, finish_reason)
        except ValueError as exc:
            entry["validation_error"] = str(exc)
            state["error"] = f"{step}{'' if index is None else f' (cohort {index})'}: {exc}"
            if attempt < max_attempts:
                # The complete document and original instructions, plus the reason.
                retry = messages(trial_id, step, index) + [{"role": "user", "content": (
                    f"The preceding attempt failed validation: {exc}. Return a complete replacement "
                    "JSON object following the original instructions and the full document.")}]
                queues[step].append((trial_id, step, index, attempt + 1, retry))
            else:
                state["failed"] = True
            return
        state["error"] = None
        if step == "screen":
            if parsed == "no":
                finish(trial_id, {"cancer_treatment_trial": "no"})
            else:
                queues["list"].append((trial_id, "list", None, 1, messages(trial_id, "list", None)))
        elif step == "list":
            state["cohort_names"] = parsed
            criteria[trial_id] = {}
            for i in range(len(parsed)):
                queues["criteria"].append((trial_id, "criteria", i, 1, messages(trial_id, "criteria", i)))
        else:
            criteria[trial_id][index] = parsed
            names = state["cohort_names"]
            if len(criteria[trial_id]) == len(names):
                decision, merged = assemble_decision(names, [criteria[trial_id][i] for i in range(len(names))])
                state["merged_cohorts"] = merged
                finish(trial_id, decision)

    def next_chunk():
        for step in reversed(STEPS):
            if queues[step]:
                chunk, queues[step][:] = queues[step][:chunk_size], queues[step][chunk_size:]
                return step, chunk
        return None

    with ThreadPoolExecutor(max_workers=chunks_in_flight) as executor:
        running = {}
        while running or any(queues.values()):
            while len(running) < chunks_in_flight and (picked := next_chunk()) is not None:
                step, chunk = picked
                running[executor.submit(
                    backend.generate_llm_outputs,
                    prompt_list=build_prompt_list([item[4] for item in chunk], llm_config=runtimes[step]),
                    llm_config=runtimes[step], model_metadata_cache_dir=model_metadata_cache_dir,
                )] = chunk
            finished, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in finished:
                chunk = running.pop(future)
                generation = future.result()
                model_metadata = {k: v for k, v in generation.model_metadata.items() if k != "responses"}
                provenance = {item["row_idx"]: item for item in generation.model_metadata.get("responses", [])}
                for position, (item, answer, reasoning, finish_reason) in enumerate(zip(
                        chunk, generation.final_outputs, generation.reasoning_outputs,
                        generation.finish_reasons, strict=True)):
                    record(item, answer, reasoning, finish_reason, provenance.get(position))
            if on_progress is not None:
                on_progress(states)
    for state in states.values():
        state.pop("failed", None)
    return states, model_metadata


__all__ = [
    "COHORT_COLUMNS", "JSON_SCHEMAS", "PROMPT_FILES", "REPRESENTATION_VERSION", "SCHEMA_VERSION", "STEPS",
    "assemble_decision", "build_messages", "decision_rows", "load_prompts", "parse_cohort_list",
    "parse_criteria", "parse_screen", "prompt_sha256", "ctgov_trial_text", "response_format",
    "run_cohort_workflow", "step_runtimes", "validate_decision",
]
