"""Per-criterion eligibility check with the decision model's criterion question.

For each candidate pair, the decision model reads one state, the patient's complete summary and the
cohort's complete numbered eligibility criteria, and answers one question per criteria line, which
restates the line with its number and inclusion/exclusion type.

**Polarity: P(yes) is the probability that the patient does NOT clearly fail the line** (the line is
met, its information is missing, or the patient could meet it by waiting). **No means the patient
clearly fails it: an inclusion line clearly not met, or an exclusion line clearly met.** A low
``criterion_pass_probability`` therefore flags a likely failing line for review, whichever list the
line is on.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pandas as pd

from matchminer_ai._metadata import package_metadata
from matchminer_ai.config import config_snapshot, load_default_preset
from .decision_model import checkable_items, criteria_state, criterion_name, criterion_question, run_decisions

if TYPE_CHECKING:
    from matchminer_ai.config import MMAIConfig


def check_eligibility_criteria(
    pairs: pd.DataFrame,
    *,
    config: MMAIConfig | None = None,
    filter_failing: bool = False,
    return_metadata: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """
    Check every eligibility-criterion line of each candidate pair with the decision model.

    Parameters
    ----------
    pairs : pd.DataFrame
        Candidate patient–cohort pairs.

        Expected columns
        ----------------
        patient_id : str
            Patient identifier.
        cohort_trial_id : str
            Cohort identifier (historical ``space_trial_id`` is accepted).
        cohort_criteria_items : str
            The cohort's tagged criteria lines, as
            :func:`~matchminer_ai.trials.remove_boilerplate_criteria` returns them (a JSON list of
            ``number``, ``heading``, ``text``, ``criterion_type`` and ``boilerplate``).
        cancer_history_summary and general_exclusion_criteria_evidence : str
            The patient summary's two sections (or one complete ``patient_summary``).

    config : MMAIConfig, optional
        ``decision_model`` (the model) and ``criteria_check.fail_cutoff`` (default 0.5). Uses the
        default preset when omitted.
    filter_failing : bool, default False
        If True, return only the lines the patient likely fails (``criterion_clearly_fails``).
    return_metadata : bool, default False
        When True, also return the config snapshot and model metadata.

    Returns
    -------
    pd.DataFrame
        One row per pair and checkable line (a line whose type never validated is skipped and
        counted in the metadata's ``unchecked_lines``):

        patient_id, cohort_trial_id or space_trial_id
            The pair.
        criterion_number, criterion_type, boilerplate, criterion_text
            The line: its number in the cohort's criteria, "inclusion" or "exclusion", whether it
            is boilerplate, and its text.
        criterion_pass_probability : float
            P(yes): the probability that the patient does **not** clearly fail this line.
        criterion_clearly_fails : bool
            True when ``criterion_pass_probability`` is below ``criteria_check.fail_cutoff``: the
            patient likely fails the line (an inclusion line not met, or an exclusion line met).
    tuple[pd.DataFrame, dict]
        When return_metadata is True, returns the DataFrame plus metadata.
    """
    from matchminer_ai._texts import patient_full_texts, trial_id_column
    from matchminer_ai.trials.cohort_boilerplate import ITEMS_COLUMN, load_criteria_items

    if "patient_id" not in pairs.columns:
        raise ValueError("pairs is missing required columns: patient_id")
    if ITEMS_COLUMN not in pairs.columns:
        raise ValueError(f"pairs needs {ITEMS_COLUMN}: tag the cohorts' criteria lines with "
                         "matchminer_ai.trials.remove_boilerplate_criteria first")
    trial_id = trial_id_column(pairs)
    resolved_config = config or load_default_preset()
    cutoff = float(resolved_config.raw.get("criteria_check", {}).get("fail_cutoff", 0.5))
    summaries = patient_full_texts(pairs)
    keys, requests, lines, unchecked = [], {}, {}, 0
    for summary, value in zip(summaries, pairs[ITEMS_COLUMN], strict=True):
        items = load_criteria_items(value)
        checkable = checkable_items(items)
        unchecked += len(items) - len(checkable)
        key = json.dumps([summary, items], ensure_ascii=False)
        keys.append(key)
        if key not in requests and checkable:  # identical summary and criteria are asked once
            requests[key] = (criteria_state(summary, items), {criterion_name(i): criterion_question(i) for i in checkable})
            lines[key] = checkable
    ordered = list(requests)
    answers, model_metadata = run_decisions([requests[k] for k in ordered], config=resolved_config)
    by_key = dict(zip(ordered, answers, strict=True))

    records = []
    for (_, row), key in zip(pairs.iterrows(), keys, strict=True):
        for item in lines.get(key, []):
            probability = by_key[key][criterion_name(item)]
            records.append({"patient_id": row["patient_id"], trial_id: row[trial_id],
                            "criterion_number": int(item["number"]), "criterion_type": item["criterion_type"],
                            "boilerplate": bool(item.get("boilerplate")), "criterion_text": str(item["text"]).strip(),
                            "criterion_pass_probability": probability, "criterion_clearly_fails": probability < cutoff})
    columns = ["patient_id", trial_id, "criterion_number", "criterion_type", "boilerplate", "criterion_text",
               "criterion_pass_probability", "criterion_clearly_fails"]
    output = pd.DataFrame(records, columns=columns)
    if filter_failing:
        output = output.loc[output["criterion_clearly_fails"]].reset_index(drop=True)
    if return_metadata:
        model_metadata = {**model_metadata, "unchecked_lines": unchecked}
        return output, {"package": package_metadata(), "config_snapshot": config_snapshot(resolved_config),
                        "model_metadata": {"check_eligibility_criteria": model_metadata}}
    return output


__all__ = ["check_eligibility_criteria"]
