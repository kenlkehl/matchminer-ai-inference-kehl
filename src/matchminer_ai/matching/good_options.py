"""Catalog-backed patient-specific GoodOption prompting and scoring."""

from __future__ import annotations

import contextlib
import json
import logging
import math
import re
import warnings
from collections.abc import Mapping, MutableMapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from matchminer_ai.config import config_snapshot, load_default_preset
from matchminer_ai.help_me_choose import normalize_nct_id
from matchminer_ai.llm.backends import (
    LLMGenerationResult,
    build_llm_runtime_config,
    get_llm_backend,
    remote_enabled,
)
from matchminer_ai.llm.model_profiles import discover_served_context_tokens
from matchminer_ai.llm.remote_auth import (
    OPENAI_COMPATIBLE_PROVIDER,
    remote_bearer_token,
    remote_provider_name,
)
from matchminer_ai.llm.remote_inference import normalize_remote_server_urls
from matchminer_ai.llm.prompt_rendering import build_prompt_list
from matchminer_ai.llm.prompts import load_prompt_text
from matchminer_ai.matching.inference import run_checker
from matchminer_ai.trials.drug_catalog import (
    EVIDENCE_GRANULARITIES,
    load_good_option_catalog,
    render_fact_summary,
)
from matchminer_ai.trials.drug_evidence import (
    GOOD_OPTION_INPUT_VERSION,
    GOOD_OPTION_PROMPT_VERSION,
    RUBRIC_CRITERIA,
    DrugSummary,
    GoodOptionCatalog,
    ParsedGoodOptionResult,
    TrialClassEvidence,
)

if TYPE_CHECKING:
    from matchminer_ai.config import MMAIConfig

LOGGER = logging.getLogger(__name__)


def _clean_text(value: Any, *, max_chars: int) -> str:
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", str(value or ""))
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        return f"{text[: max_chars - 1].rstrip()}…"
    return text


def _find_json_mapping(text: str, *, required_array: str) -> Mapping[str, Any] | None:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", cleaned):
        with contextlib.suppress(json.JSONDecodeError):
            value, _ = decoder.raw_decode(cleaned[match.start() :])
            if isinstance(value, Mapping) and isinstance(
                value.get(required_array), list
            ):
                return value
    return None


def _require_catalog(
    catalog: GoodOptionCatalog | None, *, research: Any = None
) -> GoodOptionCatalog:
    if research is not None:
        raise ValueError(
            "Legacy snippet research is incompatible with GoodOption v2. "
            "Build and load a GoodOptionCatalog, then pass catalog=."
        )
    if not isinstance(catalog, GoodOptionCatalog):
        raise TypeError("catalog must be a loaded, validated GoodOptionCatalog.")
    return catalog


#: Versions patient-time evidence packing separately from the prompt text, whose
#: version is part of the catalog compatibility id. Recorded in scoring metadata.
GOOD_OPTION_EVIDENCE_PACKING_VERSION = "good-option-evidence-packing-v1"

#: Relative claim on the evidence budget when every subject overflows. Drug
#: sections carry the agent-specific evidence the rubric prefers; a class block is
#: supporting context shared by the drugs it covers.
_SUBJECT_WEIGHTS = {"drug": 2, "class": 1}
_SUBJECT_SEPARATOR = "\n\n"
_UNBOUNDED_CHARS = 10**9
#: Detail used when no level fits whole and ranked truncation must drop facts.
#: Coarsening is worth it only while it keeps every fact visible; once facts are
#: dropped anyway, the survivors are the strongest evidence and should keep
#: their numbers. At 300 characters nearly every statement renders intact.
_OVERFLOW_GRANULARITY = "condensed"
#: Room kept for a parse retry, which appends the bounded previous response and
#: the correction template to the original conversation.
_RETRY_RESERVE_CHARS = 14_000


@dataclass(frozen=True)
class _EvidenceSubject:
    kind: str
    subject_id: str
    name: str
    label: str
    facts: Mapping[str, Any]
    stored_summary: str


def _evidence_subjects(
    drug_summaries: Sequence[DrugSummary],
    class_evidence: Sequence[TrialClassEvidence],
) -> list[_EvidenceSubject]:
    """List drug sections, then one labelled block per class.

    Class and agent evidence are synthesized separately and merged only here, so
    neither crowds the other out of a shared budget. A class block says which
    drugs it covers because a point earned from class evidence belongs to those
    drugs and to no others in the trial.
    """

    subjects = [
        _EvidenceSubject(
            kind="drug",
            subject_id=summary.drug_id,
            name=summary.preferred_name,
            label="",
            facts=summary.structured_facts,
            stored_summary=summary.good_option_summary,
        )
        for summary in drug_summaries
    ]
    for item in class_evidence:
        covers = ", ".join(item.drug_names) or "the drugs above"
        subjects.append(
            _EvidenceSubject(
                kind="class",
                subject_id=item.class_id,
                name=item.class_name,
                label=f"DRUG CLASS EVIDENCE — {item.class_name} (covers: {covers})\n",
                facts=item.structured_facts,
                stored_summary=item.class_option_summary,
            )
        )
    return subjects


def _render_subject(
    subject: _EvidenceSubject, *, granularity: str, max_chars: int
) -> str:
    """Render one subject from its facts, or bound its stored summary.

    Catalogs predating stored class facts, and hand-built summaries, carry only
    the rendered projection; those are used as stored and truncated if needed.
    """

    body_chars = max(1, max_chars - len(subject.label))
    if isinstance(subject.facts, Mapping) and subject.facts:
        body = render_fact_summary(
            subject.name,
            subject.facts,
            kind=subject.kind,
            max_chars=body_chars,
            granularity=granularity,
        )
    else:
        body = subject.stored_summary
        if len(body) > body_chars:
            body = f"{body[: body_chars - 1].rstrip()}…"
    return f"{subject.label}{body}"


def _weighted_fair_shares(
    demands: Sequence[int], weights: Sequence[int], available: int
) -> list[int]:
    """Split ``available`` by weighted max-min fair share over ``demands``."""

    budgets = [0] * len(demands)
    pending = {index: demand for index, demand in enumerate(demands)}
    remaining = max(0, available)
    while pending:
        unit = remaining / sum(weights[index] for index in pending)
        satisfied = {
            index: demand
            for index, demand in pending.items()
            if demand <= unit * weights[index]
        }
        if not satisfied:
            for index in pending:
                budgets[index] = int(unit * weights[index])
            break
        for index, demand in satisfied.items():
            budgets[index] = demand
            remaining -= demand
            del pending[index]
    return budgets


def pack_good_option_evidence(
    drug_summaries: Sequence[DrugSummary],
    class_evidence: Sequence[TrialClassEvidence] = (),
    *,
    max_chars: int | None = None,
    max_drug_chars: int | None = None,
    max_class_chars: int | None = None,
    render_cache: MutableMapping[tuple[str, ...], str] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Fit a trial's drug and class evidence into one prompt budget.

    Each drug and class is re-rendered from its structured facts, so how much
    detail it gets follows how much evidence the trial has to integrate rather
    than a fixed per-summary cap set when the catalog was built. A trial with one
    drug and one class shows everything; a trial with many large classes shares
    the budget by weighted fair share, with drugs weighted ahead of classes.

    A subject whose share is below its full render steps down to a coarser
    granularity (shorter attribution and statements) if that fits, which keeps
    every fact visible. Subjects that fit at a coarser level release the rest of
    their share to the others. Only when even the coarsest render overflows does
    ranked truncation drop facts, weakest evidence first, with an omission
    count; the survivors are then rendered at ``condensed`` detail so the
    strongest results keep their numbers. ``max_drug_chars`` and
    ``max_class_chars`` bound any one subject.

    Rendering is patient-independent, so callers packing many prompts from one
    catalog can pass the same ``render_cache`` to reuse whole-subject renders.

    Returns the evidence text and one record per subject describing its demand,
    budget, rendered size and granularity.
    """

    subjects = _evidence_subjects(drug_summaries, class_evidence)
    if not subjects:
        return "", []
    cache = {} if render_cache is None else render_cache

    def whole(subject: _EvidenceSubject, level: str) -> str:
        key = (subject.kind, subject.subject_id, subject.label, level)
        if key not in cache:
            cache[key] = _render_subject(
                subject, granularity=level, max_chars=_UNBOUNDED_CHARS
            )
        return cache[key]

    levels = tuple(EVIDENCE_GRANULARITIES)
    lengths = [
        {level: len(whole(subject, level)) for level in levels}
        for subject in subjects
    ]
    caps = [
        (max_class_chars if subject.kind == "class" else max_drug_chars)
        for subject in subjects
    ]
    ceilings = [
        length["full"] if cap is None else min(length["full"], max(1, int(cap)))
        for length, cap in zip(lengths, caps, strict=True)
    ]
    weights = [_SUBJECT_WEIGHTS[subject.kind] for subject in subjects]
    separators = len(_SUBJECT_SEPARATOR) * (len(subjects) - 1)
    available = (
        sum(ceilings)
        if max_chars is None
        else max(0, int(max_chars) - separators)
    )

    def fitting_level(index: int, budget: int) -> str | None:
        return next(
            (level for level in levels if lengths[index][level] <= budget), None
        )

    # Demands only fall, each subject at most once per coarser level, so this
    # settles in a few rounds.
    demands = list(ceilings)
    for _round in range(len(levels) * len(subjects) + 1):
        budgets = _weighted_fair_shares(demands, weights, available)
        changed = False
        for index, budget in enumerate(budgets):
            if budget >= demands[index]:
                continue
            level = fitting_level(index, budget)
            if level is not None and lengths[index][level] < demands[index]:
                demands[index] = lengths[index][level]
                changed = True
        if not changed:
            break

    sections: list[str] = []
    report: list[dict[str, Any]] = []
    for index, (subject, budget) in enumerate(zip(subjects, budgets, strict=True)):
        level = fitting_level(index, budget)
        truncated = level is None
        level = level or _OVERFLOW_GRANULARITY
        text = (
            _render_subject(subject, granularity=level, max_chars=budget)
            if truncated
            else whole(subject, level)
        )
        sections.append(text)
        report.append(
            {
                "kind": subject.kind,
                "name": subject.name,
                "full_chars": lengths[index]["full"],
                "budget_chars": budget,
                "rendered_chars": len(text),
                "granularity": level,
                "truncated": truncated,
                "from_structured_facts": bool(subject.facts),
            }
        )
    return _SUBJECT_SEPARATOR.join(sections), report


def good_option_evidence_budget(
    config: MMAIConfig, *, fixed_prompt_chars: int
) -> dict[str, Any]:
    """Derive the evidence character budget from the teacher's context window.

    The budget is what remains of the context after the completion allowance,
    a safety margin, parse-retry room and the fixed parts of the prompt, at a
    conservative characters-per-token rate.
    """

    settings = dict(config.raw.get("good_option_prompt") or {})
    runtime = build_llm_runtime_config(
        "llm_good_option", dict(config.llm_good_option), config=config
    )
    context_tokens = int(
        settings.get("context_tokens") or runtime.get("max_model_len") or 0
    )
    if context_tokens <= 0:
        raise ValueError(
            "Set good_option_prompt.context_tokens or "
            "llm_good_option.local.engine.max_model_len."
        )
    sampling = dict(runtime.get("sampling_params") or {})
    output_tokens = int(
        sampling.get("max_tokens") or sampling.get("max_completion_tokens") or 0
    )
    chars_per_token = float(settings.get("chars_per_token", 3.5))
    safety_tokens = int(settings.get("safety_tokens", 4096))

    def section_cap(key: str) -> int | None:
        value = settings.get(key)
        return None if value is None else int(float(value) * chars_per_token)

    prompt_tokens = context_tokens - output_tokens - safety_tokens
    evidence_chars = (
        int(prompt_tokens * chars_per_token) - fixed_prompt_chars - _RETRY_RESERVE_CHARS
    )
    return {
        "packing_version": GOOD_OPTION_EVIDENCE_PACKING_VERSION,
        "context_tokens": context_tokens,
        "output_tokens": output_tokens,
        "safety_tokens": safety_tokens,
        "chars_per_token": chars_per_token,
        "evidence_max_chars": evidence_chars,
        "max_drug_chars": section_cap("max_drug_section_tokens"),
        "max_class_chars": section_cap("max_class_section_tokens"),
    }


def _build_good_option_messages(
    *,
    patient_summary: str,
    drug_summaries: Sequence[DrugSummary],
    class_evidence: Sequence[TrialClassEvidence],
    config: MMAIConfig,
    render_cache: MutableMapping[tuple[str, ...], str] | None = None,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if not drug_summaries:
        raise ValueError("At least one scoreable drug summary is required.")
    patient = _clean_text(patient_summary, max_chars=16000)
    if not patient:
        raise ValueError("patient_summary must be non-empty.")
    system = load_prompt_text("llm_good_option.system.txt")
    rubric = load_prompt_text("llm_good_option.rubric.txt")
    names_json = json.dumps(
        [summary.preferred_name for summary in drug_summaries], ensure_ascii=False
    )
    template = load_prompt_text("llm_good_option.user.txt")

    def user_text(drug_sections: str) -> str:
        return template.format(
            patient=patient,
            drug_sections=drug_sections,
            rubric=rubric,
            names_json=names_json,
        )

    budget = good_option_evidence_budget(
        config, fixed_prompt_chars=len(system) + len(user_text(""))
    )
    minimum = 1000 * (len(drug_summaries) + len(class_evidence))
    if budget["evidence_max_chars"] < minimum:
        raise ValueError(
            "The GoodOption teacher context leaves "
            f"{budget['evidence_max_chars']} characters for evidence, below the "
            f"{minimum}-character minimum for this trial. Increase the context "
            "window or lower llm_good_option max_tokens."
        )
    drug_sections, subjects = pack_good_option_evidence(
        drug_summaries,
        class_evidence,
        max_chars=budget["evidence_max_chars"],
        max_drug_chars=budget["max_drug_chars"],
        max_class_chars=budget["max_class_chars"],
        render_cache=render_cache,
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_text(drug_sections)},
    ]
    return messages, {**budget, "subjects": subjects}


def build_good_option_messages(
    *,
    patient_summary: str,
    drug_summaries: Sequence[DrugSummary],
    class_evidence: Sequence[TrialClassEvidence] = (),
    config: MMAIConfig | None = None,
) -> list[dict[str, str]]:
    """Build a metadata-free patient prompt from a trial's drug and class evidence.

    Evidence is packed to the teacher's context window by
    :func:`pack_good_option_evidence`, using ``good_option_prompt`` settings and
    the ``llm_good_option`` completion allowance from ``config``.
    """

    messages, _packing = _build_good_option_messages(
        patient_summary=patient_summary,
        drug_summaries=drug_summaries,
        class_evidence=class_evidence,
        config=config or load_default_preset(),
    )
    return messages


def parse_good_option_response(
    text: str, *, drug_summaries: Sequence[DrugSummary]
) -> ParsedGoodOptionResult:
    """Validate four binary labels per expected drug and derive the trial score."""

    expected = tuple(summary.preferred_name for summary in drug_summaries)
    if not expected:
        return ParsedGoodOptionResult(status="no_scoreable_drug")
    parsed = _find_json_mapping(text, required_array="drug_assessments")
    if parsed is None:
        return ParsedGoodOptionResult(
            drug_count=len(expected),
            max_points=4 * len(expected),
            parse_error="No JSON object containing drug_assessments was found.",
        )
    disease_type = _clean_text(parsed.get("patient_disease_type"), max_chars=500)
    if not disease_type:
        return ParsedGoodOptionResult(
            drug_count=len(expected),
            max_points=4 * len(expected),
            parse_error="patient_disease_type must be non-empty.",
        )
    by_name: dict[str, Mapping[str, Any]] = {}
    for raw in parsed.get("drug_assessments", []):
        if isinstance(raw, Mapping):
            name = _clean_text(raw.get("drug_name"), max_chars=300)
            if name and name.casefold() not in by_name:
                by_name[name.casefold()] = raw
    if set(by_name) != {name.casefold() for name in expected}:
        return ParsedGoodOptionResult(
            drug_count=len(expected),
            max_points=4 * len(expected),
            parse_error="Drug assessments did not exactly match supplied scoreable drugs.",
        )
    points = 0
    assessments: list[dict[str, Any]] = []
    for name in expected:
        raw = by_name[name.casefold()]
        flat_criteria_present = any(criterion in raw for criterion in RUBRIC_CRITERIA)
        nested_criteria = raw.get("criteria")
        criterion_results = (
            nested_criteria
            if not flat_criteria_present and isinstance(nested_criteria, Mapping)
            else raw
        )
        biomarkers = raw.get("targeted_biomarkers", [])
        if not isinstance(biomarkers, Sequence) or isinstance(biomarkers, (str, bytes)):
            biomarkers = []
        assessment: dict[str, Any] = {
            "drug_name": name,
            "targeted_biomarkers": [
                cleaned
                for value in biomarkers
                if (cleaned := _clean_text(value, max_chars=500))
            ],
        }
        for criterion in RUBRIC_CRITERIA:
            result = criterion_results.get(criterion)
            if not isinstance(result, Mapping):
                return ParsedGoodOptionResult(
                    drug_count=len(expected),
                    max_points=4 * len(expected),
                    parse_error=f"{name}: {criterion} must be an object.",
                )
            point = result.get("point")
            rationale = _clean_text(result.get("rationale"), max_chars=4000)
            if isinstance(point, bool) or point not in {0, 1} or not rationale:
                return ParsedGoodOptionResult(
                    drug_count=len(expected),
                    max_points=4 * len(expected),
                    parse_error=f"{name}: invalid {criterion} result.",
                )
            points += int(point)
            assessment[criterion] = {"point": int(point), "rationale": rationale}
        assessments.append(assessment)
    max_points = 4 * len(expected)
    uncertainties = parsed.get("key_uncertainties", [])
    if not isinstance(uncertainties, Sequence) or isinstance(
        uncertainties, (str, bytes)
    ):
        uncertainties = []
    return ParsedGoodOptionResult(
        score=points / max_points,
        points=points,
        max_points=max_points,
        drug_count=len(expected),
        status="ok",
        patient_disease_type=disease_type,
        drug_assessments=tuple(assessments),
        uncertainties=tuple(
            cleaned
            for value in uncertainties
            if (cleaned := _clean_text(value, max_chars=1000))
        ),
    )


_CHECKER_DEPRECATION = (
    "{name} is deprecated: the trained GoodOptionChecker classifier is no longer "
    "maintained. Use score_good_options_with_llm (method='llm'), whose teacher "
    "prompt carries both drug and class evidence."
)


def _warn_checker_deprecated(name: str) -> None:
    warnings.warn(
        _CHECKER_DEPRECATION.format(name=name), DeprecationWarning, stacklevel=3
    )


def build_good_option_checker_text(
    patient_summary: str, drug_summary: DrugSummary
) -> str:
    """Render one patient-drug input for the four-logit checker.

    .. deprecated::
        The GoodOptionChecker is deprecated; score with
        :func:`score_good_options_with_llm`. This input carries one drug summary
        and no class evidence, unlike the teacher prompt.
    """

    _warn_checker_deprecated("build_good_option_checker_text")
    return _checker_text(patient_summary, drug_summary)


def _checker_text(patient_summary: str, drug_summary: DrugSummary) -> str:
    return load_prompt_text("good_option_checker_template.txt").format(
        patient_summary=str(patient_summary or "").strip(),
        drug_summary=drug_summary.good_option_summary,
    )


def _candidate_records(candidate_pairs: pd.DataFrame) -> pd.DataFrame:
    frame = candidate_pairs.copy()
    if "trial_id" not in frame.columns and "nct_id" in frame.columns:
        frame["trial_id"] = frame["nct_id"]
    if "patient_id" not in frame.columns and "pseudo_mrn" in frame.columns:
        frame["patient_id"] = frame["pseudo_mrn"]
    if (
        "cancer_history_summary" not in frame.columns
        and "patient_summary" in frame.columns
    ):
        frame["cancer_history_summary"] = frame["patient_summary"]
    required = {"patient_id", "trial_id", "cancer_history_summary"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"candidate_pairs is missing required columns: {missing}")
    frame["patient_id"] = frame["patient_id"].fillna("").astype(str)
    frame["trial_id"] = frame["trial_id"].map(normalize_nct_id)
    frame["cancer_history_summary"] = (
        frame["cancer_history_summary"].fillna("").astype(str)
    )
    return frame.drop_duplicates(["patient_id", "trial_id"], keep="first").reset_index(
        drop=True
    )


def _empty_result_row(
    *, patient_id: str, trial_id: str, method: str, status: str, drug_count: int = 0
) -> dict[str, Any]:
    return {
        "patient_id": patient_id,
        "trial_id": trial_id,
        "good_option_score": math.nan,
        "good_option_points": pd.NA,
        "good_option_max_points": 4 * drug_count if drug_count else pd.NA,
        "good_option_drug_count": drug_count,
        "good_option_status": status,
        "good_option_method": method,
        "good_option_patient_disease_type": "",
        "good_option_drug_assessments": [],
        "good_option_uncertainties": [],
    }


def _run_good_option_llm(
    messages_list: list[list[dict[str, str]]], *, config: MMAIConfig
) -> LLMGenerationResult:
    llm_config = dict(config.llm_good_option)
    if not llm_config:
        raise ValueError("Config is missing llm_good_option settings.")
    runtime_config = build_llm_runtime_config(
        "llm_good_option", llm_config, config=config
    )
    prompts = build_prompt_list(messages_list, llm_config=runtime_config)
    return get_llm_backend(config).generate_llm_outputs(
        prompt_list=prompts,
        llm_config=runtime_config,
        model_metadata_cache_dir=config.model_metadata_cache_dir,
    )


_TOKEN_LIMIT_FINISH_REASONS = frozenset({"length", "max_tokens"})
_RETRY_RESPONSE_MAX_CHARS = 12_000


def _parse_good_option_generation(
    response: str,
    *,
    finish_reason: str,
    drug_summaries: Sequence[DrugSummary],
) -> ParsedGoodOptionResult:
    """Parse one generation, rejecting output explicitly stopped by its limit."""

    if finish_reason.strip().casefold() in _TOKEN_LIMIT_FINISH_REASONS:
        return ParsedGoodOptionResult(
            drug_count=len(drug_summaries),
            max_points=4 * len(drug_summaries),
            parse_error=(
                "The GoodOption teacher response reached its output token limit "
                f"(finish_reason={finish_reason})."
            ),
        )
    return parse_good_option_response(response, drug_summaries=drug_summaries)


def _bounded_retry_response(response: str) -> str:
    """Bound malformed assistant text before placing it in a correction turn."""

    text = str(response or "")
    if len(text) <= _RETRY_RESPONSE_MAX_CHARS:
        return text
    half = (_RETRY_RESPONSE_MAX_CHARS - 80) // 2
    return (
        text[:half]
        + "\n...[middle of previous response omitted for retry context]...\n"
        + text[-half:]
    )


def _append_good_option_retry_feedback(
    messages: Sequence[Mapping[str, str]],
    *,
    response: str,
    parse_error: str,
    finish_reason: str,
) -> list[dict[str, str]]:
    """Return a follow-up conversation containing the exact validator failure."""

    retry_messages = [
        {"role": str(message["role"]), "content": str(message["content"])}
        for message in messages
    ]
    previous_response = _bounded_retry_response(response)
    if previous_response.strip():
        retry_messages.append({"role": "assistant", "content": previous_response})
    finish_reason_line = (
        f"\nFinish reason: {str(finish_reason).strip()}"
        if str(finish_reason or "").strip()
        else ""
    )
    feedback = load_prompt_text("llm_good_option.retry.txt").format(
        parse_error=str(parse_error or "unknown parse failure").strip(),
        finish_reason_line=finish_reason_line,
    )
    retry_messages.append({"role": "user", "content": feedback})
    return retry_messages


def _good_option_reasoning_disabled_config(config: MMAIConfig) -> MMAIConfig:
    """Clone config and disable thinking for one final parse-recovery attempt."""

    fallback = deepcopy(config)
    local = fallback.llm_good_option.setdefault("local", {})
    local_template = dict(local.get("chat_template_kwargs", {}))
    local_template["enable_thinking"] = False
    local["chat_template_kwargs"] = local_template

    remote = fallback.llm_good_option.setdefault("remote", {})
    extra_body = dict(remote.get("extra_body", {}))
    remote_template = dict(extra_body.get("chat_template_kwargs", {}))
    remote_template["enable_thinking"] = False
    extra_body["chat_template_kwargs"] = remote_template
    remote["extra_body"] = extra_body
    return fallback


def _packing_summary(packings: Any) -> dict[str, Any]:
    """Summarize how much evidence was condensed or truncated across prompts."""

    packings = list(packings)
    subjects = [subject for packing in packings for subject in packing["subjects"]]
    return {
        "prompts": len(packings),
        "subjects": len(subjects),
        "granularity_counts": {
            level: sum(subject["granularity"] == level for subject in subjects)
            for level in EVIDENCE_GRANULARITIES
        },
        "truncated_subjects": sum(bool(subject["truncated"]) for subject in subjects),
        "min_evidence_max_chars": min(
            (packing["evidence_max_chars"] for packing in packings), default=None
        ),
    }


def score_good_options_with_llm(
    candidate_pairs: pd.DataFrame,
    *,
    catalog: GoodOptionCatalog | None = None,
    research: Any = None,
    config: MMAIConfig | None = None,
    max_parse_attempts: int = 1,
    reasoning_off_fallback: bool = False,
    return_metadata: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    """Apply the four-point rubric to a trial's catalog drug and class evidence.

    Evidence is re-rendered from the catalog's structured facts and packed to the
    teacher's context window; see :func:`pack_good_option_evidence`.

    Parse retries are selective: only invalid generations are submitted again.
    Each retry includes the prior invalid answer and the code validator's exact
    error. If requested, one additional attempt disables model thinking after
    all ordinary parse attempts have failed.
    """

    resolved_catalog = _require_catalog(catalog, research=research)
    resolved_config = config or load_default_preset()
    parse_attempts = int(max_parse_attempts)
    if parse_attempts < 1:
        raise ValueError("max_parse_attempts must be at least 1.")
    frame = _candidate_records(candidate_pairs)
    rows: list[dict[str, Any] | None] = [None] * len(frame)
    message_indices: list[int] = []
    base_messages_by_index: dict[int, list[dict[str, str]]] = {}
    messages_by_index: dict[int, list[dict[str, str]]] = {}
    summaries_by_index: dict[int, tuple[DrugSummary, ...]] = {}
    packing_by_index: dict[int, dict[str, Any]] = {}
    render_cache: dict[tuple[str, ...], str] = {}
    for index, source in frame.iterrows():
        trial_id = str(source["trial_id"])
        status = resolved_catalog.trial_status(trial_id)
        assignments = resolved_catalog.assignments_for_trial(
            trial_id, scoreable_only=True
        )
        if status != "ok":
            rows[index] = _empty_result_row(
                patient_id=str(source["patient_id"]),
                trial_id=trial_id,
                method="llm",
                status=status,
                drug_count=len(assignments),
            )
            continue
        summaries = resolved_catalog.scoreable_summaries_for_trial(trial_id)
        summaries_by_index[index] = summaries
        base_messages, packing = _build_good_option_messages(
            patient_summary=str(source["cancer_history_summary"]),
            drug_summaries=summaries,
            class_evidence=resolved_catalog.class_evidence_for_trial(trial_id),
            config=resolved_config,
            render_cache=render_cache,
        )
        packing_by_index[index] = packing
        base_messages_by_index[index] = base_messages
        messages_by_index[index] = base_messages
        message_indices.append(index)
    generation: LLMGenerationResult | None = None
    latest_by_index: dict[int, tuple[ParsedGoodOptionResult, str, str, str]] = {}
    pending_indices = list(message_indices)

    def run_pending(
        indices: Sequence[int], *, generation_config: MMAIConfig
    ) -> list[int]:
        nonlocal generation
        wave_messages = [messages_by_index[index] for index in indices]
        generation = _run_good_option_llm(
            wave_messages,
            config=generation_config,
        )
        if len(generation.final_outputs) != len(wave_messages):
            raise RuntimeError("GoodOption LLM returned an unexpected output count.")
        failed: list[int] = []
        for output_index, frame_index in enumerate(indices):
            response = str(generation.final_outputs[output_index] or "")
            reasoning = (
                str(generation.reasoning_outputs[output_index] or "")
                if output_index < len(generation.reasoning_outputs)
                else ""
            )
            finish_reason = (
                str(generation.finish_reasons[output_index] or "")
                if output_index < len(generation.finish_reasons)
                else ""
            )
            parsed = _parse_good_option_generation(
                response,
                finish_reason=finish_reason,
                drug_summaries=summaries_by_index[frame_index],
            )
            latest_by_index[frame_index] = (
                parsed,
                response,
                reasoning,
                finish_reason,
            )
            if parsed.status == "ok":
                continue
            messages_by_index[frame_index] = _append_good_option_retry_feedback(
                base_messages_by_index[frame_index],
                response=response,
                parse_error=parsed.parse_error,
                finish_reason=finish_reason,
            )
            failed.append(frame_index)
        return failed

    for _attempt in range(parse_attempts):
        if not pending_indices:
            break
        pending_indices = run_pending(
            pending_indices,
            generation_config=resolved_config,
        )

    if pending_indices and reasoning_off_fallback:
        pending_indices = run_pending(
            pending_indices,
            generation_config=_good_option_reasoning_disabled_config(resolved_config),
        )

    for frame_index in message_indices:
        source = frame.iloc[frame_index]
        parsed, response, reasoning, finish_reason = latest_by_index[frame_index]
        row = _empty_result_row(
            patient_id=str(source["patient_id"]),
            trial_id=str(source["trial_id"]),
            method="llm",
            status=parsed.status,
            drug_count=parsed.drug_count,
        )
        row.update(
            {
                "good_option_score": parsed.score,
                "good_option_points": (
                    parsed.points if parsed.status == "ok" else pd.NA
                ),
                "good_option_max_points": parsed.max_points or pd.NA,
                "good_option_patient_disease_type": parsed.patient_disease_type,
                "good_option_drug_assessments": list(parsed.drug_assessments),
                "good_option_uncertainties": list(parsed.uncertainties),
            }
        )
        if resolved_config.debug_mode:
            row.update(
                {
                    "good_option_answer_text": response,
                    "good_option_reasoning_text": reasoning,
                    "good_option_finish_reason": finish_reason,
                    "good_option_parse_error": parsed.parse_error,
                    "good_option_evidence_packing": packing_by_index[frame_index],
                }
            )
        rows[frame_index] = row
    output = pd.DataFrame([row for row in rows if row is not None])
    metadata = {
        "config_snapshot": config_snapshot(resolved_config),
        "method": "llm",
        "catalog_compatibility_id": resolved_catalog.compatibility_id,
        "prompt_version": GOOD_OPTION_PROMPT_VERSION,
        "evidence_packing_version": GOOD_OPTION_EVIDENCE_PACKING_VERSION,
        "evidence_packing": _packing_summary(packing_by_index.values()),
        "rubric_criteria": list(RUBRIC_CRITERIA),
        "max_parse_attempts": parse_attempts,
        "reasoning_off_fallback": bool(reasoning_off_fallback),
        "model_metadata": (
            {"llm_good_option": generation.model_metadata} if generation else {}
        ),
    }
    return (output, metadata) if return_metadata else output


def _criterion_probabilities(prediction: Any) -> dict[str, float]:
    if isinstance(prediction, Mapping) and isinstance(
        prediction.get("criterion_probabilities"), Mapping
    ):
        probabilities = {
            str(key): float(value)
            for key, value in prediction["criterion_probabilities"].items()
        }
    elif isinstance(prediction, Sequence) and not isinstance(prediction, (str, bytes)):
        probabilities = {}
        for item in prediction:
            if not isinstance(item, Mapping):
                continue
            label = str(item.get("label") or "").strip()
            if label.startswith("LABEL_"):
                with contextlib.suppress(ValueError, IndexError):
                    label = RUBRIC_CRITERIA[int(label.rsplit("_", 1)[1])]
            probabilities[label] = float(item.get("score"))
    else:
        probabilities = {}
    missing = [
        criterion for criterion in RUBRIC_CRITERIA if criterion not in probabilities
    ]
    if missing:
        raise ValueError(
            "GoodOptionChecker must return four criterion probabilities; missing "
            + ", ".join(missing)
        )
    bounded = {criterion: probabilities[criterion] for criterion in RUBRIC_CRITERIA}
    if any(not 0.0 <= value <= 1.0 for value in bounded.values()):
        raise ValueError("GoodOptionChecker criterion probabilities must be in [0, 1].")
    return bounded


def score_good_options(
    candidate_pairs: pd.DataFrame,
    *,
    catalog: GoodOptionCatalog | None = None,
    research: Any = None,
    config: MMAIConfig | None = None,
    return_metadata: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    """Score patient-drug rows with a four-logit checker and aggregate by trial.

    .. deprecated::
        The GoodOptionChecker is deprecated in favor of the LLM teacher,
        :func:`score_good_options_with_llm`. No checker artifact is published,
        and the checker sees one drug summary and no class evidence.
    """

    _warn_checker_deprecated("score_good_options")
    resolved_catalog = _require_catalog(catalog, research=research)
    resolved_config = config or load_default_preset()
    checker_config = dict(resolved_config.raw.get("good_option_checker", {}))
    if not str(checker_config.get("model_name") or "").strip():
        raise ValueError("GoodOptionChecker model_name is not configured.")
    frame = _candidate_records(candidate_pairs)
    rows: list[dict[str, Any] | None] = [None] * len(frame)
    prompts: list[str] = []
    prompt_targets: list[tuple[int, DrugSummary]] = []
    expected_counts: dict[int, int] = {}
    for index, source in frame.iterrows():
        trial_id = str(source["trial_id"])
        status = resolved_catalog.trial_status(trial_id)
        assignments = resolved_catalog.assignments_for_trial(
            trial_id, scoreable_only=True
        )
        if status != "ok":
            rows[index] = _empty_result_row(
                patient_id=str(source["patient_id"]),
                trial_id=trial_id,
                method="classifier",
                status=status,
                drug_count=len(assignments),
            )
            continue
        summaries = resolved_catalog.scoreable_summaries_for_trial(trial_id)
        expected_counts[index] = len(summaries)
        for summary in summaries:
            prompts.append(
                _checker_text(str(source["cancer_history_summary"]), summary)
            )
            prompt_targets.append((index, summary))
    model_metadata: dict[str, Any] = {}
    if prompts:
        predictions, model_metadata = run_checker(
            prompts,
            checker_config=checker_config,
            model_metadata_cache_dir=resolved_config.model_metadata_cache_dir,
            return_all_scores=True,
        )
        if len(predictions) != len(prompts):
            raise RuntimeError("GoodOptionChecker returned an unexpected output count.")
        assessments_by_row: dict[int, list[dict[str, Any]]] = {}
        for (frame_index, summary), prediction in zip(
            prompt_targets, predictions, strict=True
        ):
            probabilities = _criterion_probabilities(prediction)
            assessments_by_row.setdefault(frame_index, []).append(
                {
                    "drug_name": summary.preferred_name,
                    "criterion_probabilities": probabilities,
                }
            )
        for frame_index, assessments in assessments_by_row.items():
            source = frame.iloc[frame_index]
            if len(assessments) != expected_counts[frame_index]:
                raise RuntimeError(
                    "A scoreable drug was omitted from checker aggregation."
                )
            values = [
                float(assessment["criterion_probabilities"][criterion])
                for assessment in assessments
                for criterion in RUBRIC_CRITERIA
            ]
            row = _empty_result_row(
                patient_id=str(source["patient_id"]),
                trial_id=str(source["trial_id"]),
                method="classifier",
                status="ok",
                drug_count=len(assessments),
            )
            row["good_option_score"] = sum(values) / len(values)
            row["good_option_drug_assessments"] = assessments
            rows[frame_index] = row
    output = pd.DataFrame([row for row in rows if row is not None])
    metadata = {
        "config_snapshot": config_snapshot(resolved_config),
        "method": "classifier",
        "catalog_compatibility_id": resolved_catalog.compatibility_id,
        "checker_input_version": GOOD_OPTION_INPUT_VERSION,
        "model_metadata": {"good_option_checker": model_metadata},
    }
    return (output, metadata) if return_metadata else output


def evaluate_good_options(
    candidate_pairs: pd.DataFrame,
    *,
    catalog: GoodOptionCatalog | None = None,
    research: Any = None,
    method: str = "llm",
    config: MMAIConfig | None = None,
    return_metadata: bool = False,
    max_parse_attempts: int = 1,
    reasoning_off_fallback: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    """Dispatch catalog-backed GoodOption scoring to the LLM or checker.

    ``method="classifier"`` is deprecated; use the default ``"llm"``. For
    on-demand checks prefer :func:`check_good_options`, which applies the
    ``good_option_check`` production settings.
    """

    normalized = str(method or "").strip().casefold()
    if normalized == "llm":
        return score_good_options_with_llm(
            candidate_pairs,
            catalog=catalog,
            research=research,
            config=config,
            max_parse_attempts=max_parse_attempts,
            reasoning_off_fallback=reasoning_off_fallback,
            return_metadata=return_metadata,
        )
    if normalized == "classifier":
        return score_good_options(
            candidate_pairs,
            catalog=catalog,
            research=research,
            config=config,
            return_metadata=return_metadata,
        )
    raise ValueError("method must be 'llm' or 'classifier'.")


def _discover_good_option_context(
    config: MMAIConfig, *, timeout: float
) -> tuple[int | None, str]:
    """Smallest ``max_model_len`` across the configured vLLM servers, if any."""

    remote = dict(config.remote or {})
    if remote_provider_name(remote) != OPENAI_COMPATIBLE_PROVIDER:
        return None, ""
    try:
        urls = normalize_remote_server_urls(remote)
    except ValueError as exc:
        return None, str(exc)
    token = remote_bearer_token(remote)
    discovered: list[int] = []
    for url in urls:
        try:
            value = discover_served_context_tokens(url, api_key=token, timeout=timeout)
        except Exception as exc:  # noqa: BLE001 - reported, then fall back
            return None, f"{url}: {exc}"
        if value:
            discovered.append(value)
    return (min(discovered) if discovered else None), ""


def _good_option_check_config(
    config: MMAIConfig, settings: Mapping[str, Any]
) -> tuple[MMAIConfig, dict[str, Any]]:
    """Apply the on-demand output cap and endpoint context to a config copy."""

    checked = deepcopy(config)
    max_output_tokens = settings.get("max_output_tokens")
    if max_output_tokens is not None:
        cap = int(max_output_tokens)
        if cap < 1:
            raise ValueError("good_option_check.max_output_tokens must be positive.")
        local = checked.llm_good_option.setdefault("local", {})
        local.setdefault("generation", {})["max_tokens"] = cap
        request_params = checked.llm_good_option.setdefault(
            "remote", {}
        ).setdefault("request_params", {})
        key = (
            "max_completion_tokens"
            if "max_completion_tokens" in request_params
            and "max_tokens" not in request_params
            else "max_tokens"
        )
        request_params[key] = cap

    prompt_settings = checked.raw.setdefault("good_option_prompt", {})
    report: dict[str, Any] = {
        "max_output_tokens": max_output_tokens,
        "context_tokens": prompt_settings.get("context_tokens"),
        "context_source": "configured",
        "context_warning": "",
    }
    if prompt_settings.get("context_tokens"):
        return checked, report
    report["context_source"] = "preset"
    if remote_enabled(checked) and settings.get("discover_context_tokens", True):
        context_tokens, warning = _discover_good_option_context(
            checked,
            timeout=float(settings.get("discovery_timeout_seconds", 10.0)),
        )
        if context_tokens:
            prompt_settings["context_tokens"] = context_tokens
            report.update(context_tokens=context_tokens, context_source="endpoint")
        elif warning:
            report["context_warning"] = (
                f"Could not read the endpoint context length ({warning}); "
                "packing evidence for the configured context."
            )
            LOGGER.warning("%s", report["context_warning"])
    return checked, report


def check_good_options(
    candidate_pairs: pd.DataFrame,
    *,
    catalog: GoodOptionCatalog | str | Path,
    config: MMAIConfig | None = None,
    return_metadata: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    """Run the on-demand LLM GoodOption check for patient-trial pairs.

    This is the supported entry point for checking whether a trial's
    experimental drugs are evidence-backed options for a patient. It wraps
    :func:`score_good_options_with_llm` with the ``good_option_check`` settings:

    - a catalog path is loaded, validated and cached until its files change;
    - the completion is capped at ``max_output_tokens`` (50,000 by default) so
      more of the context carries evidence;
    - unless ``good_option_prompt.context_tokens`` is set, the evidence budget
      follows the smallest ``max_model_len`` the configured vLLM servers report;
    - invalid answers get ``max_parse_attempts`` attempts in total, then one
      attempt with thinking disabled when ``reasoning_off_fallback`` is true.

    ``candidate_pairs`` needs ``patient_id``, ``trial_id`` and
    ``cancer_history_summary``. Output columns match
    :func:`score_good_options_with_llm`. The score is an evidence fraction for
    human review, not an eligibility decision or treatment recommendation.
    """

    resolved_config = config or load_default_preset()
    settings = dict(resolved_config.raw.get("good_option_check") or {})
    resolved_catalog = (
        catalog
        if isinstance(catalog, GoodOptionCatalog)
        else load_good_option_catalog(catalog, cache=True)
    )
    checked_config, context_report = _good_option_check_config(
        resolved_config, settings
    )
    output, metadata = score_good_options_with_llm(
        candidate_pairs,
        catalog=resolved_catalog,
        config=checked_config,
        max_parse_attempts=int(settings.get("max_parse_attempts", 3)),
        reasoning_off_fallback=bool(settings.get("reasoning_off_fallback", True)),
        return_metadata=True,
    )
    metadata["good_option_check"] = context_report
    return (output, metadata) if return_metadata else output


__all__ = [
    "GOOD_OPTION_EVIDENCE_PACKING_VERSION",
    "build_good_option_checker_text",
    "build_good_option_messages",
    "check_good_options",
    "evaluate_good_options",
    "good_option_evidence_budget",
    "pack_good_option_evidence",
    "parse_good_option_response",
    "score_good_options",
    "score_good_options_with_llm",
]
