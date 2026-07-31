"""Ontology-grounded structured clinical-trial-space generation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

from matchminer_ai.config import MMAIConfig, config_snapshot, load_default_preset
from matchminer_ai.llm.backends import build_llm_runtime_config
from matchminer_ai.patients.ontology import (
    load_ncit_drug_index,
    load_oncotree,
    normalize_ontology_text,
)
from matchminer_ai.patients.structure import (
    PatientStructuringError,
    _fill_prompt,
    _JsonAgentRunner,
    _load_prompt_text,
    _normalize_drug,
    _optional_string,
    _select_oncotree_diagnosis,
)


TrialSpaceStructuringProgress = Callable[[str, int, int, str], None]

_BIOMARKER_TYPES = {
    "mutation",
    "fusion",
    "expression",
    "copy_number_alteration",
}
_BURDEN_VALUES = {
    "early_or_curative_intent",
    "advanced_or_palliative_intent",
}
_SEX_VALUES = {"female", "male", "other"}


class TrialSpaceStructuringError(ValueError):
    """Raised when structured trial-space output cannot be validated."""


def _emit_progress(
    callback: TrialSpaceStructuringProgress | None,
    stage: str,
    completed: int,
    total: int,
    detail: str,
) -> None:
    if callback is not None:
        callback(stage, completed, total, detail)


def _task_runtime_config(
    structuring_config: dict[str, Any],
    *,
    config: MMAIConfig,
) -> dict[str, Any]:
    llm_only_config = {
        key: deepcopy(structuring_config[key])
        for key in ("reasoning_parser", "local", "remote")
        if key in structuring_config
    }
    return build_llm_runtime_config(
        "trial_space_structuring",
        llm_only_config,
        config=config,
    )


def _normalize_age_bound(value: Any) -> int | float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        age = float(value)
    except (TypeError, ValueError):
        return None
    if age < 0 or age > 130:
        return None
    return int(age) if age.is_integer() else age


def _normalize_sex_allowed(value: Any) -> list[str]:
    if not isinstance(value, list):
        raise TrialSpaceStructuringError("sex_allowed must be a list.")
    normalized = list(
        dict.fromkeys(
            str(item).strip().casefold()
            for item in value
            if str(item).strip().casefold() in _SEX_VALUES
        )
    )
    if not normalized:
        raise TrialSpaceStructuringError(
            "sex_allowed must include at least one supported value."
        )
    return normalized


def _normalize_burden_allowed(value: Any) -> list[str]:
    if not isinstance(value, list):
        raise TrialSpaceStructuringError("cancer_burden_allowed must be a list.")
    normalized = list(
        dict.fromkeys(
            str(item).strip().casefold()
            for item in value
            if str(item).strip().casefold() in _BURDEN_VALUES
        )
    )
    if not normalized:
        raise TrialSpaceStructuringError(
            "cancer_burden_allowed must include at least one supported value."
        )
    return normalized


def _normalize_biomarkers(raw_items: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_items, list):
        return []
    biomarkers: list[dict[str, Any]] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        marker = _optional_string(raw.get("marker"))
        result = _optional_string(raw.get("result"))
        marker_type = (
            str(raw.get("type") or "")
            .strip()
            .casefold()
            .replace("-", "_")
            .replace(" ", "_")
        )
        if marker_type == "copy_number":
            marker_type = "copy_number_alteration"
        screening_value = raw.get("screening_assessment_required")
        screening_required = (
            screening_value if isinstance(screening_value, bool) else None
        )
        if not marker or not result or marker_type not in _BIOMARKER_TYPES:
            continue
        biomarkers.append(
            {
                "marker": marker,
                "type": marker_type,
                "result": result,
                "screening_assessment_required": screening_required,
            }
        )
    return biomarkers


def _normalize_treatment_constraints(raw_items: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_items, list):
        return []
    constraints: list[dict[str, Any]] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        requirement = _optional_string(raw.get("requirement"))
        if not requirement:
            continue
        raw_drugs = raw.get("drug_mentions")
        drug_mentions: list[str] = []
        if isinstance(raw_drugs, list):
            drug_mentions = list(
                dict.fromkeys(
                    drug
                    for item in raw_drugs
                    if (drug := _optional_string(item)) is not None
                )
            )
        constraints.append(
            {
                "requirement": requirement,
                "drug_mentions": drug_mentions,
                "response_requirement": _optional_string(
                    raw.get("response_requirement")
                ),
            }
        )
    return constraints


def _validate_extracted_space(raw: dict[str, Any]) -> dict[str, Any]:
    cancer_description = _optional_string(raw.get("cancer_description"))
    if not cancer_description:
        raise TrialSpaceStructuringError(
            "Extracted JSON must include cancer_description."
        )
    minimum_age = _normalize_age_bound(raw.get("minimum_age"))
    maximum_age = _normalize_age_bound(raw.get("maximum_age"))
    if (
        minimum_age is not None
        and maximum_age is not None
        and minimum_age > maximum_age
    ):
        raise TrialSpaceStructuringError(
            "minimum_age cannot be greater than maximum_age."
        )
    return {
        "minimum_age": minimum_age,
        "maximum_age": maximum_age,
        "sex_allowed": _normalize_sex_allowed(raw.get("sex_allowed")),
        "cancer_description": cancer_description,
        "histology_description": _optional_string(
            raw.get("histology_description")
        ),
        "cancer_burden_allowed": _normalize_burden_allowed(
            raw.get("cancer_burden_allowed")
        ),
        "prior_treatment_required": _normalize_treatment_constraints(
            raw.get("prior_treatment_required")
        ),
        "prior_treatment_excluded": _normalize_treatment_constraints(
            raw.get("prior_treatment_excluded")
        ),
        "biomarkers_required": _normalize_biomarkers(
            raw.get("biomarkers_required")
        ),
        "biomarkers_excluded": _normalize_biomarkers(
            raw.get("biomarkers_excluded")
        ),
    }


def _unique_drug_mentions(space: dict[str, Any]) -> list[str]:
    mentions: list[str] = []
    seen: set[str] = set()
    for field in ("prior_treatment_required", "prior_treatment_excluded"):
        for constraint in space[field]:
            for source_name in constraint["drug_mentions"]:
                normalized = normalize_ontology_text(source_name)
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                mentions.append(source_name)
    return mentions


def _finalize_treatment_constraints(
    constraints: list[dict[str, Any]],
    normalized_drugs: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    finalized: list[dict[str, Any]] = []
    for constraint in constraints:
        finalized.append(
            {
                "requirement": constraint["requirement"],
                "drugs": [
                    deepcopy(normalized_drugs[normalize_ontology_text(source_name)])
                    for source_name in constraint["drug_mentions"]
                    if normalize_ontology_text(source_name) in normalized_drugs
                ],
                "response_requirement": constraint["response_requirement"],
            }
        )
    return finalized


def _structure_trial_space(
    clinical_space_summary: str,
    *,
    trial_id: str | None,
    space_trial_id: str | None,
    config: MMAIConfig,
    progress_callback: TrialSpaceStructuringProgress | None,
) -> tuple[dict[str, Any], _JsonAgentRunner, dict[str, Any]]:
    structuring_config = dict(config.trial_space_structuring)
    if not structuring_config:
        raise ValueError("Config is missing trial_space_structuring settings.")
    retry_limit = max(0, int(structuring_config.get("ontology_retry_limit", 2)))
    runner = _JsonAgentRunner(
        config=config,
        runtime_config=_task_runtime_config(structuring_config, config=config),
        retry_limit=retry_limit,
    )

    _emit_progress(progress_callback, "extract", 0, 1, "Extracting space criteria")
    extract_template = _load_prompt_text("trial_space_structure.extract.user.txt")
    extracted = runner.generate(
        _fill_prompt(
            extract_template,
            clinical_space_summary=clinical_space_summary.strip(),
        )
    )
    space = _validate_extracted_space(extracted)
    _emit_progress(progress_callback, "extract", 1, 1, "Space criteria extracted")

    tree = load_oncotree(str(structuring_config["oncotree_resource"]))
    _emit_progress(progress_callback, "oncotree", 0, 1, "Coding allowed diagnosis")
    path = _select_oncotree_diagnosis(
        space,
        tree_root=tree,
        runner=runner,
        prompt_filename="trial_space_structure.oncotree.user.txt",
        max_depth=max(1, int(structuring_config.get("oncotree_max_depth", 10))),
        retry_limit=retry_limit,
    )
    _emit_progress(progress_callback, "oncotree", 1, 1, path[-1].name)

    ncit_index = load_ncit_drug_index(str(structuring_config["ncit_resource"]))
    mentions = _unique_drug_mentions(space)
    normalized_drugs: dict[str, dict[str, Any]] = {}
    for drug_number, source_name in enumerate(mentions, start=1):
        _emit_progress(
            progress_callback,
            "ncit",
            drug_number - 1,
            len(mentions),
            source_name,
        )
        normalized_drugs[normalize_ontology_text(source_name)] = _normalize_drug(
            source_name,
            index=ncit_index,
            runner=runner,
            select_prompt_filename="patient_structure.ncit_select.user.txt",
            resolve_prompt_filename="patient_structure.ncit_resolve.user.txt",
            candidate_limit=max(
                1, int(structuring_config.get("ncit_candidate_limit", 8))
            ),
            max_agent_steps=max(
                1, int(structuring_config.get("ncit_max_agent_steps", 3))
            ),
        )
        _emit_progress(
            progress_callback,
            "ncit",
            drug_number,
            len(mentions),
            source_name,
        )

    site_node = path[0]
    diagnosis_node = path[-1]
    histology_node = (
        diagnosis_node
        if space["histology_description"] is not None and len(path) > 1
        else None
    )
    cancer_type_node = site_node if histology_node is not None else diagnosis_node
    cancer_type_name = (
        cancer_type_node.main_type
        if cancer_type_node is site_node and cancer_type_node.main_type
        else cancer_type_node.name
    )
    result = {
        "trial_id": _optional_string(trial_id),
        "space_trial_id": _optional_string(space_trial_id),
        "age_range": {
            "minimum_age": space["minimum_age"],
            "maximum_age": space["maximum_age"],
        },
        "sex_allowed": space["sex_allowed"],
        "cancer_type": {
            "name": cancer_type_name,
            "oncotree_code": cancer_type_node.code,
        },
        "histology": (
            {
                "name": histology_node.name,
                "oncotree_code": histology_node.code,
            }
            if histology_node is not None
            else None
        ),
        "cancer_burden_allowed": space["cancer_burden_allowed"],
        "prior_treatment_required": _finalize_treatment_constraints(
            space["prior_treatment_required"], normalized_drugs
        ),
        "prior_treatment_excluded": _finalize_treatment_constraints(
            space["prior_treatment_excluded"], normalized_drugs
        ),
        "biomarkers_required": space["biomarkers_required"],
        "biomarkers_excluded": space["biomarkers_excluded"],
    }
    _emit_progress(progress_callback, "complete", 1, 1, "Structured space ready")
    return result, runner, structuring_config


def structure_trial_space(
    clinical_space_summary: str,
    *,
    trial_id: str | None = None,
    space_trial_id: str | None = None,
    config: MMAIConfig | None = None,
    return_metadata: bool = False,
    progress_callback: TrialSpaceStructuringProgress | None = None,
) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any]]:
    """
    Transform one clinical-space summary into ontology-grounded JSON data.

    The function preserves eligibility constraints for age, sex, cancer burden,
    biomarkers, and required or excluded prior treatment. OncoTree descent
    exposes only immediate children to the LLM. NCIt normalization exposes only
    bounded search candidates and definitions selected for inspection.

    The output is a research abstraction and does not replace review of the
    complete, current protocol or establish patient eligibility.
    """
    if (
        not isinstance(clinical_space_summary, str)
        or not clinical_space_summary.strip()
    ):
        raise ValueError("clinical_space_summary must be a non-empty string.")
    resolved_config = config or load_default_preset()
    if not isinstance(resolved_config, MMAIConfig):
        raise TypeError("config must be an MMAIConfig instance or None.")
    try:
        result, runner, structuring_config = _structure_trial_space(
            clinical_space_summary,
            trial_id=trial_id,
            space_trial_id=space_trial_id,
            config=resolved_config,
            progress_callback=progress_callback,
        )
    except PatientStructuringError as exc:
        raise TrialSpaceStructuringError(str(exc)) from exc

    if return_metadata:
        return result, {
            "config_snapshot": config_snapshot(resolved_config),
            "model_metadata": {"trial_space_structurer": runner.model_metadata},
            "ontology_versions": {
                "oncotree": str(structuring_config.get("oncotree_version", "")),
                "ncit": str(structuring_config.get("ncit_version", "")),
            },
        }
    return result


__all__ = [
    "TrialSpaceStructuringError",
    "TrialSpaceStructuringProgress",
    "structure_trial_space",
]
