"""Ontology-grounded structured patient-summary generation."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from importlib import resources
from typing import Any, Callable

from matchminer_ai.config import MMAIConfig, config_snapshot, load_default_preset
from matchminer_ai.llm.backends import build_llm_runtime_config, get_llm_backend
from matchminer_ai.llm.prompt_rendering import build_prompt_list

from .ontology import (
    NCItDrugRecord,
    NCIThesaurusDrugIndex,
    OncoTreeNode,
    load_ncit_drug_index,
    load_oncotree,
    normalize_ontology_text,
)


PatientStructuringProgress = Callable[[str, int, int, str], None]

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
_SEX_ALIASES = {
    "f": "female",
    "female": "female",
    "m": "male",
    "male": "male",
    "intersex": "intersex",
    "other": "other",
    "unknown": "unknown",
    "not specified": "unknown",
    "unspecified": "unknown",
}


class PatientStructuringError(ValueError):
    """Raised when a structured-summary agent response cannot be validated."""


def _load_prompt_text(filename: str) -> str:
    prompt_path = resources.files("matchminer_ai.prompts").joinpath(filename)
    with prompt_path.open("r", encoding="utf-8") as handle:
        return handle.read().strip()


def _fill_prompt(template: str, **values: Any) -> str:
    """Replace named prompt slots without interpreting literal JSON braces."""
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{" + key + "}", str(value))
    return rendered


def _extract_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for offset, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _end = decoder.raw_decode(text[offset:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise PatientStructuringError("The LLM response did not contain a JSON object.")


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
        "patient_structuring",
        llm_only_config,
        config=config,
    )


@dataclass
class _JsonAgentRunner:
    config: MMAIConfig
    runtime_config: dict[str, Any]
    retry_limit: int
    model_metadata: dict[str, Any] = field(default_factory=dict)

    def generate(self, user_prompt: str) -> dict[str, Any]:
        messages = [
            {
                "role": "system",
                "content": (
                    "Reasoning: high. Follow the requested JSON contract exactly."
                ),
            },
            {"role": "user", "content": user_prompt},
        ]
        backend = get_llm_backend(self.config)
        last_error: Exception | None = None
        for _attempt in range(self.retry_limit + 1):
            prompt_list = build_prompt_list(
                [messages],
                llm_config=self.runtime_config,
            )
            generation = backend.generate_llm_outputs(
                prompt_list=prompt_list,
                llm_config=self.runtime_config,
                model_metadata_cache_dir=self.config.model_metadata_cache_dir,
            )
            if not self.model_metadata:
                self.model_metadata = dict(generation.model_metadata)
            if len(generation.final_outputs) != 1:
                raise PatientStructuringError(
                    "The LLM returned a different number of outputs than prompts."
                )
            response_text = str(generation.final_outputs[0])
            try:
                return _extract_json_object(response_text)
            except PatientStructuringError as exc:
                last_error = exc
                messages.extend(
                    [
                        {"role": "assistant", "content": response_text},
                        {
                            "role": "user",
                            "content": (
                                "That response was not a valid JSON object. Return "
                                "only one JSON object matching the requested schema."
                            ),
                        },
                    ]
                )
        raise PatientStructuringError(
            "The LLM did not return valid JSON after retries."
        ) from last_error


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.casefold() in {
        "n/a",
        "na",
        "not applicable",
        "not documented",
        "null",
        "none",
        "unknown",
    }:
        return None
    return text


def _normalize_age(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        age = int(value)
    except (TypeError, ValueError):
        return None
    if 0 <= age <= 130:
        return age
    return None


def _normalize_sex(value: Any) -> str:
    normalized = str(value or "unknown").strip().casefold()
    return _SEX_ALIASES.get(normalized, "unknown")


def _normalize_biomarkers(raw_items: Any) -> list[dict[str, str]]:
    if not isinstance(raw_items, list):
        return []
    biomarkers: list[dict[str, str]] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        marker = _optional_string(raw.get("marker"))
        result = _optional_string(raw.get("result"))
        raw_type = str(raw.get("type") or "").strip().casefold()
        marker_type = raw_type.replace("-", "_").replace(" ", "_")
        if marker_type == "copy_number":
            marker_type = "copy_number_alteration"
        if not marker or not result or marker_type not in _BIOMARKER_TYPES:
            continue
        biomarkers.append(
            {
                "marker": marker,
                "type": marker_type,
                "result": result,
            }
        )
    return biomarkers


def _normalize_treatments(raw_items: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_items, list):
        return []
    treatments: list[dict[str, Any]] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        treatment = _optional_string(raw.get("treatment"))
        if not treatment:
            continue
        raw_drugs = raw.get("drug_mentions")
        drug_mentions = []
        if isinstance(raw_drugs, list):
            drug_mentions = list(
                dict.fromkeys(
                    drug
                    for item in raw_drugs
                    if (drug := _optional_string(item)) is not None
                )
            )
        treatments.append(
            {
                "treatment": treatment,
                "start_date": _optional_string(raw.get("start_date")),
                "end_date": _optional_string(raw.get("end_date")),
                "drug_mentions": drug_mentions,
                "response": _optional_string(raw.get("response")),
            }
        )
    return treatments


def _validate_extracted_facts(raw: dict[str, Any]) -> dict[str, Any]:
    raw_cancers = raw.get("cancers")
    if not isinstance(raw_cancers, list):
        raise PatientStructuringError("Extracted JSON must contain a cancers list.")

    cancers: list[dict[str, Any]] = []
    for raw_cancer in raw_cancers:
        if not isinstance(raw_cancer, dict):
            continue
        cancer_description = _optional_string(raw_cancer.get("cancer_description"))
        if not cancer_description:
            continue
        burden = str(raw_cancer.get("cancer_burden") or "").strip().casefold()
        if burden not in _BURDEN_VALUES:
            raise PatientStructuringError(
                "Each active cancer must have one of the two cancer_burden values."
            )
        cancers.append(
            {
                "cancer_description": cancer_description,
                "histology_description": _optional_string(
                    raw_cancer.get("histology_description")
                ),
                "biomarkers": _normalize_biomarkers(
                    raw_cancer.get("biomarkers")
                ),
                "treatment_history": _normalize_treatments(
                    raw_cancer.get("treatment_history")
                ),
                "cancer_burden": burden,
            }
        )

    return {
        "age": _normalize_age(raw.get("age")),
        "sex": _normalize_sex(raw.get("sex")),
        "cancers": cancers,
    }


def _emit_progress(
    callback: PatientStructuringProgress | None,
    stage: str,
    completed: int,
    total: int,
    detail: str,
) -> None:
    if callback is not None:
        callback(stage, completed, total, detail)


def _select_oncotree_diagnosis(
    cancer: dict[str, Any],
    *,
    tree_root: OncoTreeNode,
    runner: _JsonAgentRunner,
    prompt_filename: str,
    max_depth: int,
    retry_limit: int,
) -> tuple[OncoTreeNode, ...]:
    template = _load_prompt_text(prompt_filename)
    current = tree_root
    selected_path: list[OncoTreeNode] = []
    for _depth in range(max_depth):
        choices = current.children
        if not choices:
            break
        options_json = json.dumps(
            [choice.prompt_record(index) for index, choice in enumerate(choices)],
            ensure_ascii=True,
            indent=2,
        )
        prompt = _fill_prompt(
            template,
            cancer_description=cancer["cancer_description"],
            histology_description=cancer["histology_description"] or "Not documented",
            current_path=" > ".join(
                [tree_root.name, *(node.name for node in selected_path)]
            ),
            options_json=options_json,
        )
        selected_index: int | None = None
        response: dict[str, Any] = {}
        for attempt in range(retry_limit + 1):
            response = runner.generate(prompt)
            raw_index = response.get("selected_index")
            if isinstance(raw_index, int) and not isinstance(raw_index, bool):
                if 0 <= raw_index < len(choices):
                    selected_index = raw_index
                    break
            prompt += (
                "\n\nThe previous selected_index was invalid. Select exactly one "
                f"integer from 0 through {len(choices) - 1}. Retry {attempt + 1}."
            )
        if selected_index is None:
            raise PatientStructuringError(
                "The OncoTree agent did not select a valid child index."
            )
        selected = choices[selected_index]
        selected_path.append(selected)
        if bool(response.get("stop")) or not selected.children:
            break
        current = selected

    if not selected_path:
        raise PatientStructuringError("The OncoTree traversal selected no diagnosis.")
    return tuple(selected_path)


def _candidate_page(
    index: NCIThesaurusDrugIndex,
    queries: list[str],
    *,
    limit: int,
) -> list[NCItDrugRecord]:
    records: list[NCItDrugRecord] = []
    seen_codes: set[str] = set()
    for query in queries:
        for record in index.search(query, limit=limit):
            if record.code in seen_codes:
                continue
            records.append(record)
            seen_codes.add(record.code)
            if len(records) >= limit:
                return records
    return records


def _retry_queries(response: dict[str, Any]) -> list[str]:
    raw_queries = response.get("retry_queries")
    if not isinstance(raw_queries, list):
        return []
    return list(
        dict.fromkeys(
            query
            for item in raw_queries
            if (query := _optional_string(item)) is not None
        )
    )


def _unmatched_drug(source_name: str) -> dict[str, Any]:
    return {
        "source_name": source_name,
        "normalized_name": source_name,
        "ncit_code": None,
        "target": None,
        "mechanism_of_action": None,
        "normalization_status": "unmatched",
    }


def _normalize_drug(
    source_name: str,
    *,
    index: NCIThesaurusDrugIndex,
    runner: _JsonAgentRunner,
    select_prompt_filename: str,
    resolve_prompt_filename: str,
    candidate_limit: int,
    max_agent_steps: int,
) -> dict[str, Any]:
    select_template = _load_prompt_text(select_prompt_filename)
    resolve_template = _load_prompt_text(resolve_prompt_filename)
    pending_queries = [source_name]
    queries_tried: list[str] = []

    for _step in range(max_agent_steps):
        current_queries = [
            query for query in pending_queries if query not in queries_tried
        ]
        if not current_queries:
            return _unmatched_drug(source_name)
        queries_tried.extend(current_queries)
        candidates = _candidate_page(
            index,
            current_queries,
            limit=candidate_limit,
        )
        candidates_json = json.dumps(
            [
                record.selection_record(candidate_index, source_name)
                for candidate_index, record in enumerate(candidates)
            ],
            ensure_ascii=True,
            indent=2,
        )
        selection = runner.generate(
            _fill_prompt(
                select_template,
                source_name=source_name,
                queries_tried=json.dumps(queries_tried, ensure_ascii=True),
                candidates_json=candidates_json,
            )
        )
        raw_indices = selection.get("selected_indices")
        selected_indices: list[int] = []
        if isinstance(raw_indices, list):
            selected_indices = list(
                dict.fromkeys(
                    item
                    for item in raw_indices
                    if isinstance(item, int)
                    and not isinstance(item, bool)
                    and 0 <= item < len(candidates)
                )
            )[:3]
        if not selected_indices:
            pending_queries = _retry_queries(selection)
            if not pending_queries:
                return _unmatched_drug(source_name)
            continue

        inspected = [candidates[item] for item in selected_indices]
        records_json = json.dumps(
            [
                record.detail_record(position)
                for position, record in enumerate(inspected)
            ],
            ensure_ascii=True,
            indent=2,
        )
        resolution = runner.generate(
            _fill_prompt(
                resolve_template,
                source_name=source_name,
                records_json=records_json,
            )
        )
        status = str(resolution.get("status") or "").strip().casefold()
        if status == "matched":
            raw_index = resolution.get("selected_index")
            if (
                isinstance(raw_index, int)
                and not isinstance(raw_index, bool)
                and 0 <= raw_index < len(inspected)
            ):
                record = inspected[raw_index]
                return {
                    "source_name": source_name,
                    "normalized_name": record.preferred_name,
                    "ncit_code": record.code,
                    "target": _optional_string(resolution.get("target")),
                    "mechanism_of_action": _optional_string(
                        resolution.get("mechanism_of_action")
                    ),
                    "normalization_status": "matched",
                }
            raise PatientStructuringError(
                "The NCIt agent returned an invalid inspected-record index."
            )
        if status == "retry":
            pending_queries = _retry_queries(resolution)
            if pending_queries:
                continue
        return _unmatched_drug(source_name)

    return _unmatched_drug(source_name)


def _unique_drug_mentions(cancers: list[dict[str, Any]]) -> list[str]:
    mentions: list[str] = []
    seen: set[str] = set()
    for cancer in cancers:
        for treatment in cancer["treatment_history"]:
            for source_name in treatment["drug_mentions"]:
                normalized = normalize_ontology_text(source_name)
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                mentions.append(source_name)
    return mentions


def structure_patient_summary(
    patient_summary: str,
    *,
    config: MMAIConfig | None = None,
    return_metadata: bool = False,
    progress_callback: PatientStructuringProgress | None = None,
) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any]]:
    """
    Transform an unstructured cancer-history summary into grounded JSON data.

    Demographics are patient-level. The ``cancers`` array contains one object
    for every active cancer, each with its own OncoTree-coded diagnosis,
    biomarkers, treatment history, normalized NCIt drugs, treatment responses,
    and binary cancer-burden category. Inactive cancers are not promoted into
    that array.

    OncoTree selection is hierarchical: each LLM request sees only the current
    node's immediate children. NCIt normalization uses a bounded local search
    harness: the LLM selects candidate indices to inspect and may request new
    search queries. Complete ontology contents are never sent to the LLM.

    This output is a research abstraction and does not establish diagnosis,
    prognosis, treatment intent, or clinical-trial eligibility.
    """
    if not isinstance(patient_summary, str) or not patient_summary.strip():
        raise ValueError("patient_summary must be a non-empty string.")
    resolved_config = config or load_default_preset()
    if not isinstance(resolved_config, MMAIConfig):
        raise TypeError("config must be an MMAIConfig instance or None.")
    structuring_config = dict(resolved_config.patient_structuring)
    if not structuring_config:
        raise ValueError("Config is missing patient_structuring settings.")

    runtime_config = _task_runtime_config(
        structuring_config,
        config=resolved_config,
    )
    retry_limit = max(0, int(structuring_config.get("ontology_retry_limit", 2)))
    runner = _JsonAgentRunner(
        config=resolved_config,
        runtime_config=runtime_config,
        retry_limit=retry_limit,
    )

    _emit_progress(progress_callback, "extract", 0, 1, "Extracting patient facts")
    extract_template = _load_prompt_text("patient_structure.extract.user.txt")
    extracted = runner.generate(
        _fill_prompt(extract_template, patient_summary=patient_summary.strip())
    )
    facts = _validate_extracted_facts(extracted)
    _emit_progress(progress_callback, "extract", 1, 1, "Patient facts extracted")

    tree = load_oncotree(str(structuring_config["oncotree_resource"]))
    ncit_index = load_ncit_drug_index(str(structuring_config["ncit_resource"]))
    cancers = facts["cancers"]
    oncotree_paths: list[tuple[OncoTreeNode, ...]] = []
    for cancer_number, cancer in enumerate(cancers, start=1):
        _emit_progress(
            progress_callback,
            "oncotree",
            cancer_number - 1,
            len(cancers),
            cancer["cancer_description"],
        )
        oncotree_paths.append(
            _select_oncotree_diagnosis(
                cancer,
                tree_root=tree,
                runner=runner,
                prompt_filename="patient_structure.oncotree.user.txt",
                max_depth=max(
                    1, int(structuring_config.get("oncotree_max_depth", 10))
                ),
                retry_limit=retry_limit,
            )
        )
        _emit_progress(
            progress_callback,
            "oncotree",
            cancer_number,
            len(cancers),
            oncotree_paths[-1][-1].name,
        )

    mentions = _unique_drug_mentions(cancers)
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

    structured_cancers: list[dict[str, Any]] = []
    for cancer, path in zip(cancers, oncotree_paths, strict=True):
        site_node = path[0]
        diagnosis_node = path[-1]
        histology_node = (
            diagnosis_node
            if cancer["histology_description"] is not None and len(path) > 1
            else None
        )
        cancer_type_node = site_node if histology_node is not None else diagnosis_node
        cancer_type_name = (
            cancer_type_node.main_type
            if cancer_type_node is site_node and cancer_type_node.main_type
            else cancer_type_node.name
        )
        treatment_history: list[dict[str, Any]] = []
        for treatment in cancer["treatment_history"]:
            drugs = [
                deepcopy(normalized_drugs[normalize_ontology_text(source_name)])
                for source_name in treatment["drug_mentions"]
                if normalize_ontology_text(source_name) in normalized_drugs
            ]
            treatment_history.append(
                {
                    "treatment": treatment["treatment"],
                    "start_date": treatment["start_date"],
                    "end_date": treatment["end_date"],
                    "drugs": drugs,
                    "response": treatment["response"],
                }
            )
        structured_cancers.append(
            {
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
                "biomarkers": cancer["biomarkers"],
                "treatment_history": treatment_history,
                "cancer_burden": cancer["cancer_burden"],
            }
        )

    result = {
        "age": facts["age"],
        "sex": facts["sex"],
        "cancers": structured_cancers,
    }
    _emit_progress(progress_callback, "complete", 1, 1, "Structured summary ready")

    if return_metadata:
        return result, {
            "config_snapshot": config_snapshot(resolved_config),
            "model_metadata": {"patient_structurer": runner.model_metadata},
            "ontology_versions": {
                "oncotree": str(structuring_config.get("oncotree_version", "")),
                "ncit": str(structuring_config.get("ncit_version", "")),
            },
        }
    return result


__all__ = [
    "PatientStructuringError",
    "PatientStructuringProgress",
    "structure_patient_summary",
]
