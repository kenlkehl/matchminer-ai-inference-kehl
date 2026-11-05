"""LLM-based match quality and exclusion screening helpers."""

from __future__ import annotations

from importlib import resources
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

    The prompt (``llm_exclusion_criteria.prompt_file``; the training teacher's BoilerplateChecker
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


__all__ = [
    "exclusion_criteria_check_with_llm",
    "score_match_quality_with_llm",
]
