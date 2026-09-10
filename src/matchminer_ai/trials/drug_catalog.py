"""Build, validate, and load versioned patient-free GoodOption catalogs."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import inspect
import json
import os
import re
import shutil
import tempfile
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import pandas as pd

from matchminer_ai.config import config_snapshot, load_default_preset
from matchminer_ai.help_me_choose import fetch_trial_study, normalize_nct_id
from matchminer_ai.llm.backends import (
    build_llm_runtime_config,
    get_llm_backend,
)
from matchminer_ai.llm.prompt_rendering import build_prompt_list
from matchminer_ai.llm.prompts import load_prompt_text
from matchminer_ai.patients.ontology import (
    NCItDrugRecord,
    load_ncit_drug_index,
    normalize_ontology_text,
)

from .drug_evidence import (
    CATALOG_SCHEMA_VERSION,
    DRUG_ROLES,
    GOOD_OPTION_INPUT_VERSION,
    GOOD_OPTION_LABEL_SCHEMA_VERSION,
    GOOD_OPTION_PROJECTION_VERSION,
    GOOD_OPTION_PROMPT_VERSION,
    HELP_ME_CHOOSE_PROJECTION_VERSION,
    ROLE_POLICY_VERSION,
    SCOREABLE_ROLES,
    SYNTHESIS_SCHEMA_VERSION,
    DrugIdentity,
    DrugSummary,
    EvidencePassage,
    GoodOptionCatalog,
    ResearchAttempt,
    TrialDrugAssignment,
)
from .drug_research import (
    DrugEvidenceSource,
    GeneralWebProvider,
    ResearchSettings,
    clean_text,
    default_sources,
    research_drug,
    utc_now,
)

if TYPE_CHECKING:
    from matchminer_ai.config import MMAIConfig


CONTROL_ARM_TYPES = frozenset(
    {"ACTIVE_COMPARATOR", "PLACEBO_COMPARATOR", "SHAM_COMPARATOR", "NO_INTERVENTION"}
)
ACTIVE_INTERVENTION_TYPES = frozenset({"DRUG", "BIOLOGICAL"})
ROLE_PROMPT_VERSION = "trial-cancer-treatment-agent-screen-v3"
SYNTHESIS_PROMPT_VERSION = "drug-evidence-synthesis-v3"
CATALOG_CHECKPOINT_SCHEMA_VERSION = "good-option-catalog-checkpoints-v1"
INTERVENTION_SCREENING_DISPOSITIONS = frozenset({"include", "exclude", "uncertain"})
INTERVENTION_EXCLUSION_CATEGORIES = frozenset(
    {
        "none",
        "not_a_concrete_agent",
        "supportive_or_procedural",
        "diagnostic_or_imaging",
        "prevention_or_non_treatment",
        "non_anticancer_therapy",
        "unspecified_standard_of_care",
        "insufficient_context",
        "other",
    }
)
_SYNTHESIS_CATEGORIES = (
    "mechanism_and_targets",
    "efficacy_by_tumor",
    "biomarker_prevalence",
    "biomarker_directed_efficacy",
    "safety",
    "limitations",
)


@dataclass(frozen=True)
class _RegistryIntervention:
    trial_id: str
    trial_title: str
    trial_brief_summary: str
    trial_conditions: tuple[str, ...]
    registry_name: str
    intervention_type: str
    aliases: tuple[str, ...]
    description: str
    arm_labels: tuple[str, ...]
    arm_types: tuple[str, ...]
    arm_descriptions: tuple[str, ...]
    initial_role: str
    role_confidence: str
    role_rationale: str


@dataclass(frozen=True)
class _CatalogLLMOutput:
    text: str
    finish_reason: str = "stop"
    reasoning: str = ""


RoleResolver = Callable[
    [str, Sequence[_RegistryIntervention]],
    Mapping[str, Mapping[str, Any]] | Awaitable[Mapping[str, Mapping[str, Any]]],
]
SummarySynthesizer = Callable[
    [DrugIdentity, Sequence[EvidencePassage]],
    Mapping[str, Any] | Awaitable[Mapping[str, Any]],
]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _compatibility_id(*, ncit_version: str) -> str:
    payload = {
        "catalog_schema": CATALOG_SCHEMA_VERSION,
        "role_policy": ROLE_POLICY_VERSION,
        "role_prompt": ROLE_PROMPT_VERSION,
        "synthesis_schema": SYNTHESIS_SCHEMA_VERSION,
        "synthesis_prompt": SYNTHESIS_PROMPT_VERSION,
        "good_option_projection": GOOD_OPTION_PROJECTION_VERSION,
        "help_me_choose_projection": HELP_ME_CHOOSE_PROJECTION_VERSION,
        "patient_prompt": GOOD_OPTION_PROMPT_VERSION,
        "label_schema": GOOD_OPTION_LABEL_SCHEMA_VERSION,
        "checker_input": GOOD_OPTION_INPUT_VERSION,
        "ncit_version": ncit_version,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _stable_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _callable_identity(value: Any, *, default: str) -> str:
    if value is None:
        return default
    target = value if inspect.isfunction(value) else type(value)
    module = getattr(target, "__module__", "")
    legacy_modules = {
        "matchminer_ai.trials.drug_research": "matchminer_ai.good_options.research",
        "matchminer_ai.trials.drug_catalog": "matchminer_ai.good_options.catalog",
    }
    module = legacy_modules.get(module, module)
    return f"{module}.{getattr(target, '__qualname__', '')}"


def _checkpoint_run_spec(
    *,
    nct_ids: Sequence[str],
    config: MMAIConfig,
    settings: ResearchSettings,
    sources: Sequence[DrugEvidenceSource],
    role_resolver: RoleResolver | None,
    synthesizer: SummarySynthesizer | None,
    ncit_resource: str,
    ncit_version: str,
) -> dict[str, Any]:
    teacher_fingerprint = _stable_sha256(
        {
            "remote_enabled": bool(config.remote.get("enabled", False)),
            "remote_provider": str(config.remote.get("provider") or ""),
            "llm_good_option": config.llm_good_option,
            "good_option_catalog": config.good_option_catalog,
        }
    )
    return {
        "catalog_compatibility_id": _compatibility_id(ncit_version=ncit_version),
        "nct_ids": list(nct_ids),
        "ncit_resource": ncit_resource,
        "ncit_version": ncit_version,
        "research_settings": asdict(settings),
        "sources": [
            {
                "name": str(getattr(source, "name", "")),
                "source_type": str(getattr(source, "source_type", "")),
                "implementation": _callable_identity(source, default=""),
            }
            for source in sources
        ],
        "role_resolver": _callable_identity(
            role_resolver,
            default="matchminer_ai.default_intervention_screening_resolver",
        ),
        "synthesizer": _callable_identity(
            synthesizer, default="matchminer_ai.default_summary_synthesizer"
        ),
        "teacher_fingerprint_sha256": teacher_fingerprint,
        "versions": {
            "checkpoint_schema": CATALOG_CHECKPOINT_SCHEMA_VERSION,
            "catalog_schema": CATALOG_SCHEMA_VERSION,
            "role_policy": ROLE_POLICY_VERSION,
            "role_prompt": ROLE_PROMPT_VERSION,
            "synthesis_schema": SYNTHESIS_SCHEMA_VERSION,
            "synthesis_prompt": SYNTHESIS_PROMPT_VERSION,
        },
    }


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


@dataclass
class _CatalogCheckpointStore:
    root: Path
    manifest: dict[str, Any]

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        run_spec: Mapping[str, Any],
        reset: bool,
    ) -> "_CatalogCheckpointStore":
        unresolved_root = Path(path).expanduser()
        if unresolved_root.is_symlink():
            raise ValueError(
                f"Catalog checkpoint path cannot be a symlink: {unresolved_root}"
            )
        root = unresolved_root.resolve()
        manifest_path = root / "manifest.json"
        if root == Path(root.anchor):
            raise ValueError(f"Catalog checkpoint path is too broad: {root}")
        if reset and root.exists():
            if not manifest_path.is_file():
                raise ValueError(
                    "Refusing to reset an unrecognized checkpoint directory without "
                    f"manifest.json: {root}"
                )
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing.get("schema_version") != CATALOG_CHECKPOINT_SCHEMA_VERSION:
                raise ValueError(
                    "Refusing to reset a checkpoint directory with an unsupported "
                    f"schema: {root}"
                )
            shutil.rmtree(root)
        if root.exists() and not root.is_dir():
            raise ValueError(f"Catalog checkpoint path is not a directory: {root}")

        expected_fingerprint = _stable_sha256(run_spec)
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("schema_version") != CATALOG_CHECKPOINT_SCHEMA_VERSION:
                raise ValueError(
                    "Unsupported GoodOption catalog checkpoint schema "
                    f"{manifest.get('schema_version')!r}; expected "
                    f"{CATALOG_CHECKPOINT_SCHEMA_VERSION!r}."
                )
            actual_fingerprint = str(manifest.get("run_fingerprint_sha256") or "")
            stored_spec = manifest.get("run_spec")
            if (
                not isinstance(stored_spec, Mapping)
                or _stable_sha256(stored_spec) != actual_fingerprint
            ):
                raise ValueError(
                    f"Catalog checkpoint manifest fingerprint is invalid: {manifest_path}"
                )
            if actual_fingerprint != expected_fingerprint:
                raise ValueError(
                    "Catalog checkpoints are incompatible with this run "
                    f"({actual_fingerprint or 'missing fingerprint'} != "
                    f"{expected_fingerprint}). Use a different checkpoint directory or "
                    "explicitly reset the existing catalog checkpoints."
                )
            return cls(root=root, manifest=manifest)

        if root.exists():
            entries = list(root.iterdir())
            orphan_manifest_temps = [
                item
                for item in entries
                if item.is_file()
                and item.name.startswith(".manifest.json.")
                and item.name.endswith(".tmp")
            ]
            if len(orphan_manifest_temps) == len(entries):
                for item in orphan_manifest_temps:
                    item.unlink()
            elif entries:
                raise ValueError(
                    "Catalog checkpoint directory is non-empty but has no "
                    f"manifest.json: {root}"
                )
        root.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": CATALOG_CHECKPOINT_SCHEMA_VERSION,
            "run_fingerprint_sha256": expected_fingerprint,
            "created_at_utc": utc_now(),
            "updated_at_utc": utc_now(),
            "run_spec": dict(run_spec),
            "completed_catalog": "",
            "completed_at_utc": "",
        }
        _write_json_atomic(manifest_path, manifest)
        return cls(root=root, manifest=manifest)

    @property
    def run_fingerprint(self) -> str:
        return str(self.manifest["run_fingerprint_sha256"])

    def _item_path(self, stage: str, key: str) -> Path:
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(key)).strip("._-")[:64]
        digest = hashlib.sha256(str(key).encode("utf-8")).hexdigest()[:16]
        return self.root / stage / f"{slug or 'item'}-{digest}.json"

    def load(
        self, stage: str, key: str, *, input_value: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        path = self._item_path(stage, key)
        if not path.is_file():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        expected_input = _stable_sha256(input_value)
        expected = {
            "schema_version": CATALOG_CHECKPOINT_SCHEMA_VERSION,
            "run_fingerprint_sha256": self.run_fingerprint,
            "stage": stage,
            "key": str(key),
            "input_fingerprint_sha256": expected_input,
        }
        mismatched = {
            field: (value.get(field), expected_value)
            for field, expected_value in expected.items()
            if value.get(field) != expected_value
        }
        if mismatched:
            raise ValueError(f"Incompatible catalog checkpoint {path}: {mismatched}")
        data = value.get("data")
        if not isinstance(data, Mapping):
            raise ValueError(f"Catalog checkpoint has invalid data: {path}")
        if value.get("data_fingerprint_sha256") != _stable_sha256(data):
            raise ValueError(f"Catalog checkpoint data fingerprint is invalid: {path}")
        return dict(data)

    def save(
        self,
        stage: str,
        key: str,
        *,
        input_value: Mapping[str, Any],
        data: Mapping[str, Any],
    ) -> None:
        _write_json_atomic(
            self._item_path(stage, key),
            {
                "schema_version": CATALOG_CHECKPOINT_SCHEMA_VERSION,
                "run_fingerprint_sha256": self.run_fingerprint,
                "stage": stage,
                "key": str(key),
                "input_fingerprint_sha256": _stable_sha256(input_value),
                "saved_at_utc": utc_now(),
                "data_fingerprint_sha256": _stable_sha256(data),
                "data": dict(data),
            },
        )

    def mark_complete(self, catalog_path: Path) -> None:
        self.manifest["updated_at_utc"] = utc_now()
        self.manifest["completed_at_utc"] = utc_now()
        self.manifest["completed_catalog"] = str(catalog_path)
        _write_json_atomic(self.root / "manifest.json", self.manifest)


def _attempt_from_record(record: Mapping[str, Any]) -> ResearchAttempt:
    retry_after = record.get("retry_after_seconds")
    return ResearchAttempt(
        drug_id=str(record.get("drug_id") or ""),
        facet=str(record.get("facet") or ""),
        source=str(record.get("source") or ""),
        query=str(record.get("query") or ""),
        attempt=int(record.get("attempt") or 0),
        status=str(record.get("status") or ""),
        started_at=str(record.get("started_at") or ""),
        finished_at=str(record.get("finished_at") or ""),
        result_count=int(record.get("result_count") or 0),
        error_type=str(record.get("error_type") or ""),
        error_message=str(record.get("error_message") or ""),
        retry_after_seconds=(float(retry_after) if retry_after is not None else None),
    )


def _evidence_from_record(record: Mapping[str, Any]) -> EvidencePassage:
    attributes = record.get("attributes", record.get("attributes_json", {}))
    if isinstance(attributes, str):
        attributes = json.loads(attributes or "{}")
    return EvidencePassage(
        evidence_id=str(record.get("evidence_id") or ""),
        drug_id=str(record.get("drug_id") or ""),
        facet=str(record.get("facet") or ""),
        source=str(record.get("source") or ""),
        source_type=str(record.get("source_type") or ""),
        title=str(record.get("title") or ""),
        passage=str(record.get("passage") or ""),
        url=str(record.get("url") or ""),
        source_locator=str(record.get("source_locator") or ""),
        published_at=str(record.get("published_at") or ""),
        retrieved_at=str(record.get("retrieved_at") or ""),
        license=str(record.get("license") or ""),
        query=str(record.get("query") or ""),
        content_sha256=str(record.get("content_sha256") or ""),
        attributes=(attributes if isinstance(attributes, Mapping) else {}),
    )


def _extract_registry_interventions(
    trial_id: str, study: Mapping[str, Any]
) -> tuple[_RegistryIntervention, ...]:
    protocol = study.get("protocolSection") or {}
    identification = protocol.get("identificationModule") or {}
    description_module = protocol.get("descriptionModule") or {}
    conditions_module = protocol.get("conditionsModule") or {}
    trial_title = clean_text(
        identification.get("briefTitle") or identification.get("officialTitle"),
        max_chars=1000,
    )
    trial_brief_summary = clean_text(
        description_module.get("briefSummary"), max_chars=5000
    )
    trial_conditions = tuple(
        dict.fromkeys(
            clean_text(value, max_chars=300)
            for value in conditions_module.get("conditions", []) or []
            if clean_text(value, max_chars=300)
        )
    )
    module = protocol.get("armsInterventionsModule") or {}
    arms = [
        value for value in module.get("armGroups", []) if isinstance(value, Mapping)
    ]
    arms_by_label = {
        clean_text(value.get("label"), max_chars=300).casefold(): value
        for value in arms
        if clean_text(value.get("label"), max_chars=300)
    }
    extracted: list[_RegistryIntervention] = []
    for raw in module.get("interventions", []) or []:
        if not isinstance(raw, Mapping):
            continue
        intervention_type = clean_text(raw.get("type"), max_chars=40).upper()
        name = clean_text(raw.get("name"), max_chars=500)
        if intervention_type not in ACTIVE_INTERVENTION_TYPES or not name:
            continue
        if re.search(r"\b(?:placebo|sham)\b", name, re.IGNORECASE):
            continue
        labels = tuple(
            dict.fromkeys(
                clean_text(value, max_chars=300)
                for value in raw.get("armGroupLabels", []) or []
                if clean_text(value, max_chars=300)
            )
        )
        linked_arms = [
            arms_by_label[label.casefold()]
            for label in labels
            if label.casefold() in arms_by_label
        ]
        arm_types = tuple(
            dict.fromkeys(
                clean_text(value.get("type"), max_chars=80).upper()
                for value in linked_arms
                if clean_text(value.get("type"), max_chars=80)
            )
        )
        arm_descriptions = tuple(
            clean_text(value.get("description"), max_chars=2000)
            for value in linked_arms
            if clean_text(value.get("description"), max_chars=2000)
        )
        description = clean_text(raw.get("description"), max_chars=3000)
        role_text = " ".join((name, description, *arm_descriptions)).casefold()
        if arm_types and set(arm_types).issubset(CONTROL_ARM_TYPES):
            role, confidence, rationale = (
                "control",
                "high",
                "Assigned only to structured control-type arms.",
            )
        elif re.search(
            r"\b(?:premedication|pre-medication|supportive care|rescue medication|antiemetic)\b",
            role_text,
        ):
            role, confidence, rationale = (
                "supportive",
                "high",
                "Registry text explicitly identifies supportive use.",
            )
        elif re.search(
            r"\b(?:standard[- ]of[- ]care|standard backbone|background therapy|backbone therapy)\b",
            role_text,
        ):
            role, confidence, rationale = (
                "background",
                "medium",
                "Registry text identifies standard/background therapy.",
            )
        else:
            role, confidence, rationale = (
                "uncertain",
                "low",
                "Structured registry metadata does not by itself establish the drug role.",
            )
        aliases = tuple(
            dict.fromkeys(
                clean_text(value, max_chars=300)
                for value in raw.get("otherNames", []) or []
                if clean_text(value, max_chars=300)
                and not re.search(r"\b(?:placebo|sham)\b", str(value), re.IGNORECASE)
            )
        )
        extracted.append(
            _RegistryIntervention(
                trial_id=trial_id,
                trial_title=trial_title,
                trial_brief_summary=trial_brief_summary,
                trial_conditions=trial_conditions,
                registry_name=name,
                intervention_type=intervention_type,
                aliases=aliases,
                description=description,
                arm_labels=labels,
                arm_types=arm_types,
                arm_descriptions=arm_descriptions,
                initial_role=role,
                role_confidence=confidence,
                role_rationale=rationale,
            )
        )
    return tuple(extracted)


def build_intervention_screening_messages(
    trial_id: str, interventions: Sequence[_RegistryIntervention]
) -> list[dict[str, str]]:
    """Build a patient-free screen for cancer-treatment drug candidates."""

    first = interventions[0] if interventions else None
    payload = {
        "trial": {
            "nct_id": trial_id,
            "title": first.trial_title if first is not None else "",
            "brief_summary": first.trial_brief_summary if first is not None else "",
            "conditions": list(first.trial_conditions) if first is not None else [],
        },
        "interventions": [
            {
                "index": index,
                "registry_name": item.registry_name,
                "intervention_type": item.intervention_type,
                "description": item.description,
                "aliases": list(item.aliases),
                "arm_labels": list(item.arm_labels),
                "arm_types": list(item.arm_types),
                "arm_descriptions": list(item.arm_descriptions),
            }
            for index, item in enumerate(interventions)
        ],
    }
    system = load_prompt_text("trial_drug_screen.system.txt")
    user = load_prompt_text("trial_drug_screen.user.txt").format(
        payload=json.dumps(payload, ensure_ascii=False, indent=2)
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_role_resolution_messages(
    trial_id: str, interventions: Sequence[_RegistryIntervention]
) -> list[dict[str, str]]:
    """Backward-compatible alias for the combined intervention screen."""

    return build_intervention_screening_messages(trial_id, interventions)


def _find_json_mapping(
    text: str, required_key: str, *, allow_bare_array: bool = False
) -> Mapping[str, Any] | None:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    decoder = json.JSONDecoder()
    opening_pattern = r"[\{\[]" if allow_bare_array else r"\{"
    for start in [
        0,
        *(match.start() for match in re.finditer(opening_pattern, cleaned)),
    ]:
        with contextlib.suppress(json.JSONDecodeError):
            value, _ = decoder.raw_decode(cleaned[start:])
            if isinstance(value, Mapping) and required_key in value:
                return value
            if allow_bare_array and isinstance(value, list):
                return {required_key: value}
    return None


def _run_llm_messages(
    messages_list: Sequence[Sequence[Mapping[str, str]]],
    *,
    config: MMAIConfig,
    stage: str,
) -> list[_CatalogLLMOutput]:
    def merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(base)
        for key, value in overlay.items():
            if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
                result[key] = merge(result[key], value)
            else:
                result[key] = value
        return result

    override = config.good_option_catalog.get(f"{stage}_llm", {})
    llm_config = merge(
        config.llm_good_option,
        override if isinstance(override, Mapping) else {},
    )
    if not llm_config:
        raise ValueError("Config is missing llm_good_option settings.")
    runtime = build_llm_runtime_config("llm_good_option", llm_config, config=config)
    prompts = build_prompt_list(
        [list(messages) for messages in messages_list], llm_config=runtime
    )
    result = get_llm_backend(config).generate_llm_outputs(
        prompt_list=prompts,
        llm_config=runtime,
        model_metadata_cache_dir=config.model_metadata_cache_dir,
    )
    if len(result.final_outputs) != len(messages_list):
        raise RuntimeError("LLM returned a different number of outputs than prompts.")
    return [
        _CatalogLLMOutput(
            text=str(text or ""),
            finish_reason=(
                str(result.finish_reasons[index] or "")
                if index < len(result.finish_reasons)
                else ""
            ),
            reasoning=(
                str(result.reasoning_outputs[index] or "")
                if index < len(result.reasoning_outputs)
                else ""
            ),
        )
        for index, text in enumerate(result.final_outputs)
    ]


def _coerce_catalog_llm_output(value: Any) -> _CatalogLLMOutput:
    """Normalize production results and simple string test doubles."""

    if isinstance(value, _CatalogLLMOutput):
        return value
    return _CatalogLLMOutput(text=str(value or ""))


def _token_limited_finish_reason(value: str) -> bool:
    return str(value or "").strip().casefold() in {"length", "max_tokens"}


async def _resolve_roles_with_default_llm(
    by_trial: Mapping[str, Sequence[_RegistryIntervention]], *, config: MMAIConfig
) -> dict[str, Mapping[str, Mapping[str, Any]]]:
    """Return the last parsed output for each attempted trial.

    Invalid outputs are retained so the catalog builder can report the exact
    validation failure after retries are exhausted.
    """

    trial_ids = [trial_id for trial_id, items in by_trial.items() if items]
    if not trial_ids:
        return {}
    outputs_by_trial: dict[str, Mapping[str, Mapping[str, Any]]] = {}
    validation_errors: dict[str, str] = {}
    pending = list(trial_ids)
    max_attempts = max(
        1, int(config.good_option_catalog.get("screening_max_attempts", 3))
    )
    for _attempt in range(1, max_attempts + 1):
        if not pending:
            break
        messages_list = []
        for trial_id in pending:
            messages = build_intervention_screening_messages(
                trial_id, by_trial[trial_id]
            )
            previous_error = validation_errors.get(trial_id)
            if previous_error:
                messages[-1] = {
                    **messages[-1],
                    "content": (
                        load_prompt_text("trial_drug_screen.retry.txt").format(
                            previous_content=messages[-1]["content"],
                            attempt=_attempt,
                            max_attempts=max_attempts,
                            previous_error=previous_error,
                        )
                    ),
                }
            messages_list.append(messages)
        outputs = await asyncio.to_thread(
            _run_llm_messages,
            messages_list,
            config=config,
            stage="screening",
        )
        retry: list[str] = []
        for trial_id, raw_output in zip(pending, outputs, strict=True):
            output = _coerce_catalog_llm_output(raw_output)
            if _token_limited_finish_reason(output.finish_reason):
                outputs_by_trial[trial_id] = {}
                validation_errors[trial_id] = (
                    "the response reached its output token limit "
                    f"(finish_reason={output.finish_reason})"
                )
                retry.append(trial_id)
                continue
            if not output.text.strip():
                outputs_by_trial[trial_id] = {}
                validation_errors[trial_id] = "the final response was blank"
                retry.append(trial_id)
                continue
            parsed = (
                _find_json_mapping(output.text, "interventions", allow_bare_array=True)
                or {}
            )
            records: dict[str, Mapping[str, Any]] = {}
            for item in parsed.get("interventions", []) or []:
                if not isinstance(item, Mapping):
                    continue
                raw_index = item.get("index")
                if isinstance(raw_index, bool):
                    continue
                if isinstance(raw_index, int) or str(raw_index).isdigit():
                    records[str(int(raw_index))] = item
            outputs_by_trial[trial_id] = records
            validation_error = _role_output_validation_error(
                by_trial[trial_id], records
            )
            if validation_error is not None:
                validation_errors[trial_id] = validation_error
                retry.append(trial_id)
        pending = retry
    return outputs_by_trial


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def _supported_active_names(
    intervention: _RegistryIntervention, raw_names: Any
) -> tuple[str, ...]:
    if not isinstance(raw_names, Sequence) or isinstance(raw_names, (str, bytes)):
        raw_names = []
    support_fields = (
        intervention.registry_name,
        *intervention.aliases,
        intervention.description,
        *intervention.arm_descriptions,
    )
    compact_support = tuple(
        re.sub(r"[^a-z0-9]+", "", normalize_ontology_text(value))
        for value in support_fields
        if value
    )
    accepted: list[str] = []
    for value in raw_names:
        name = clean_text(value, max_chars=300)
        key = re.sub(r"[^a-z0-9]+", "", normalize_ontology_text(name))
        exact_field_match = bool(key) and key in compact_support
        supported_span = len(key) >= 3 and any(
            key in candidate for candidate in compact_support
        )
        if exact_field_match or supported_span:
            accepted.append(name)
    return tuple(dict.fromkeys(accepted))


def _exact_ncit_match(
    name: str, candidates: Sequence[NCItDrugRecord]
) -> NCItDrugRecord | None:
    normalized = normalize_ontology_text(name)
    stripped = normalize_ontology_text(
        re.sub(
            r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|iv|oral|tablet|capsule)s?\b",
            " ",
            name,
            flags=re.IGNORECASE,
        )
    )
    for candidate in candidates:
        aliases = {
            normalize_ontology_text(value)
            for value in (candidate.preferred_name, *candidate.synonyms)
        }
        if normalized in aliases or stripped in aliases:
            return candidate
    return None


def _canonicalize_drug(
    name: str, aliases: Sequence[str], *, ncit_index: Any
) -> DrugIdentity:
    candidate: NCItDrugRecord | None = None
    for query in (name, *aliases):
        candidate = _exact_ncit_match(query, ncit_index.search(query, limit=8))
        if candidate is not None:
            break
    if candidate is not None:
        canonical_aliases = tuple(
            dict.fromkeys(
                clean_text(value, max_chars=300)
                for value in (name, *aliases, *candidate.synonyms[:12])
                if clean_text(value, max_chars=300)
                and clean_text(value, max_chars=300).casefold()
                != candidate.preferred_name.casefold()
            )
        )
        return DrugIdentity(
            drug_id=f"NCIT:{candidate.code}",
            preferred_name=candidate.preferred_name,
            ncit_code=candidate.code,
            aliases=canonical_aliases,
            definition=str(candidate.definition or ""),
        )
    normalized = normalize_ontology_text(name)
    digest = hashlib.sha256(normalized.encode()).hexdigest()[:20]
    return DrugIdentity(
        drug_id=f"NAME:{digest}",
        preferred_name=clean_text(name, max_chars=300),
        aliases=tuple(
            dict.fromkeys(
                clean_text(value, max_chars=300)
                for value in aliases
                if clean_text(value, max_chars=300)
            )
        ),
    )


def _ncit_definition_evidence(
    drug: DrugIdentity, *, ncit_version: str
) -> tuple[EvidencePassage, ...]:
    """Materialize the bundled NCIt definition as citable ledger evidence."""

    definition = clean_text(drug.definition, max_chars=10_000)
    code = clean_text(drug.ncit_code, max_chars=80)
    if not definition or not code:
        return ()
    return (
        EvidencePassage(
            evidence_id=f"ncit_definition:{code}",
            drug_id=drug.drug_id,
            facet="mechanism_targets",
            source="ncit_ontology",
            source_type="ontology_definition",
            title=f"NCI Thesaurus definition for {drug.preferred_name}",
            passage=definition,
            url="",
            source_locator=f"NCIt {code}",
            license="CC BY 4.0",
            query="",
            content_sha256=hashlib.sha256(definition.encode("utf-8")).hexdigest(),
            attributes={"ncit_code": code, "ncit_version": ncit_version},
        ),
    )


def build_synthesis_messages(
    drug: DrugIdentity, evidence: Sequence[EvidencePassage]
) -> list[dict[str, str]]:
    """Build a compact synthesis prompt without URLs or retrieval metadata."""

    records = []
    for index, item in enumerate(evidence, start=1):
        records.append(
            {
                "passage_id": f"P{index}",
                "facet": item.facet,
                "source_kind": item.source_type,
                "publication_year": str(item.published_at or "")[:4],
                "text": item.passage,
            }
        )
    system = load_prompt_text("trial_drug_synthesis.system.txt")
    user = load_prompt_text("trial_drug_synthesis.user.txt").format(
        payload=json.dumps(
            {
                "drug": drug.preferred_name,
                "passages": records,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _supported_passage_ids(
    support: Any, *, passage_id_map: Mapping[str, str]
) -> list[str]:
    """Resolve safe P# citations plus known aliases for the supplied NCIt passage."""

    if isinstance(support, (str, bytes)):
        raw_values: Sequence[Any] = [support]
    elif isinstance(support, Sequence):
        raw_values = support
    else:
        raw_values = []
    allowed = {str(key).casefold(): str(key) for key in passage_id_map}
    ncit_passage = next(
        (
            str(key)
            for key, evidence_id in passage_id_map.items()
            if str(evidence_id).startswith("ncit_definition:")
        ),
        "",
    )
    ncit_aliases = {
        "ncit",
        "ncit_definition",
        "ncit definition",
        "ncit_ontology_definition",
    }
    resolved: list[str] = []
    for raw in raw_values:
        text = str(raw or "").strip()
        exact = allowed.get(text.casefold())
        if exact:
            resolved.append(exact)
            continue
        if ncit_passage and text.casefold() in ncit_aliases:
            resolved.append(ncit_passage)
            continue
        for token in re.findall(r"\bP\d+\b", text, flags=re.IGNORECASE):
            if matched := allowed.get(token.casefold()):
                resolved.append(matched)
    return list(dict.fromkeys(resolved))


def _synthesis_fact_validation_error(
    value: Mapping[str, Any], *, evidence: Sequence[EvidencePassage]
) -> str | None:
    passage_id_map = {
        f"P{index}": item.evidence_id for index, item in enumerate(evidence, start=1)
    }
    invalid = 0
    for category in _SYNTHESIS_CATEGORIES:
        if category == "limitations":
            continue
        for raw in value.get(category, []) or []:
            if not isinstance(raw, Mapping):
                invalid += 1
                continue
            claim = clean_text(raw.get("claim"), max_chars=3000)
            support_ids = _supported_passage_ids(
                raw.get("support_ids", []), passage_id_map=passage_id_map
            )
            if not claim or (evidence and not support_ids):
                invalid += 1
    if invalid:
        return (
            f"{invalid} fact item(s) lacked a non-empty claim or a supported P# "
            "citation; every non-limitation fact must cite supplied passage IDs"
        )
    return None


def _validate_structured_facts(
    value: Mapping[str, Any], *, evidence: Sequence[EvidencePassage]
) -> dict[str, list[dict[str, Any]]]:
    supplied_map = value.get("__passage_id_map__", {})
    passage_id_map = (
        {str(key): str(item) for key, item in supplied_map.items()}
        if isinstance(supplied_map, Mapping)
        else {}
    )
    if not passage_id_map:
        passage_id_map = {
            f"P{index}": item.evidence_id
            for index, item in enumerate(evidence, start=1)
        }
    allowed_ids = set(passage_id_map)
    validated: dict[str, list[dict[str, Any]]] = {}
    for category in _SYNTHESIS_CATEGORIES:
        items: list[dict[str, Any]] = []
        raw_items = value.get(category, [])
        if not isinstance(raw_items, Sequence) or isinstance(raw_items, (str, bytes)):
            raw_items = []
        for raw in raw_items:
            if not isinstance(raw, Mapping):
                continue
            claim = re.sub(
                r"https?://\S+|www\.\S+",
                "",
                clean_text(raw.get("claim"), max_chars=3000),
                flags=re.IGNORECASE,
            ).strip()
            if not claim:
                continue
            support_ids = _supported_passage_ids(
                raw.get("support_ids", []), passage_id_map=passage_id_map
            )
            support_ids = [item for item in support_ids if item in allowed_ids]
            if category != "limitations" and evidence and not support_ids:
                continue
            record = {
                str(key): value
                for key, value in raw.items()
                if key not in {"claim", "support_ids"}
            }
            record["claim"] = claim
            record["support_ids"] = [passage_id_map[item] for item in support_ids]
            items.append(record)
        validated[category] = items
    return validated


def _render_summary(
    drug: DrugIdentity,
    facts: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    include_safety: bool,
    max_chars: int,
) -> str:
    labels = {
        "mechanism_and_targets": "Mechanism and targets",
        "efficacy_by_tumor": "Human efficacy by tumor type",
        "biomarker_prevalence": "Biomarker prevalence",
        "biomarker_directed_efficacy": "Biomarker-directed human efficacy",
        "safety": "Safety",
        "limitations": "Evidence limitations",
    }
    categories = list(_SYNTHESIS_CATEGORIES)
    if not include_safety:
        categories.remove("safety")
    header = f"Drug: {drug.preferred_name}"
    heading_chars = sum(len(labels[category]) + 2 for category in categories)
    newline_chars = 2 * (len(categories) + 1)
    available = max(
        256 * len(categories),
        max_chars - len(header) - heading_chars - newline_chars,
    )
    section_budget = max(256, available // len(categories))
    lines = [header]
    for category in categories:
        lines.append(f"{labels[category]}:")
        items = list(facts.get(category, ()))
        if not items:
            lines.append(
                "- No relevant evidence for this category was identified in the "
                "retrieved passages."
            )
            continue
        used = 0
        for item in items:
            remaining = section_budget - used
            if remaining <= 3:
                break
            claim = clean_text(item.get("claim"), max_chars=min(1500, remaining - 2))
            if not claim:
                continue
            line = f"- {claim}"
            lines.append(line)
            used += len(line) + 1
    rendered = "\n".join(lines)
    if len(rendered) > max_chars:
        rendered = f"{rendered[: max_chars - 1].rstrip()}…"
    return rendered


async def _default_synthesize_many(
    drugs_and_evidence: Sequence[tuple[DrugIdentity, Sequence[EvidencePassage]]],
    *,
    config: MMAIConfig,
) -> dict[str, Mapping[str, Any]]:
    if not drugs_and_evidence:
        return {}
    token_limit = int(
        config.good_option_catalog.get("synthesis_evidence_max_tokens", 64_000)
    )
    character_limit = max(4_000, token_limit * 4)

    def bounded(items: Sequence[EvidencePassage]) -> list[EvidencePassage]:
        by_source: dict[str, list[EvidencePassage]] = {}
        for item in items:
            by_source.setdefault(item.source, []).append(item)
        selected: list[EvidencePassage] = []
        used = 0
        while by_source and used < character_limit:
            for source in list(by_source):
                item = by_source[source].pop(0)
                remaining = character_limit - used
                if remaining <= 0:
                    break
                if len(item.passage) > remaining:
                    item = EvidencePassage(
                        **{
                            **asdict(item),
                            "passage": clean_text(item.passage, max_chars=remaining),
                        }
                    )
                selected.append(item)
                used += len(item.passage)
                if not by_source[source]:
                    del by_source[source]
        return selected

    bounded_inputs = [
        (drug, bounded(evidence)) for drug, evidence in drugs_and_evidence
    ]
    parsed: dict[str, Mapping[str, Any]] = {}
    last_schema_valid: dict[str, Mapping[str, Any]] = {}
    validation_errors: dict[str, str] = {}
    pending = list(bounded_inputs)
    max_attempts = max(
        1, int(config.good_option_catalog.get("synthesis_max_attempts", 3))
    )
    for _attempt in range(1, max_attempts + 1):
        if not pending:
            break
        messages_list: list[list[dict[str, str]]] = []
        for drug, evidence in pending:
            messages = build_synthesis_messages(drug, evidence)
            previous_error = validation_errors.get(drug.drug_id)
            if previous_error:
                messages[-1] = {
                    **messages[-1],
                    "content": (
                        load_prompt_text("trial_drug_synthesis.retry.txt").format(
                            previous_content=messages[-1]["content"],
                            attempt=_attempt,
                            max_attempts=max_attempts,
                            previous_error=previous_error,
                        )
                    ),
                }
            messages_list.append(messages)
        outputs = await asyncio.to_thread(
            _run_llm_messages,
            messages_list,
            config=config,
            stage="synthesis",
        )
        retry: list[tuple[DrugIdentity, Sequence[EvidencePassage]]] = []
        for (drug, bounded_evidence), raw_output in zip(pending, outputs, strict=True):
            output = _coerce_catalog_llm_output(raw_output)
            if _token_limited_finish_reason(output.finish_reason):
                validation_errors[drug.drug_id] = (
                    "the response reached its output token limit "
                    f"(finish_reason={output.finish_reason})"
                )
                retry.append((drug, bounded_evidence))
                continue
            if not output.text.strip():
                validation_errors[drug.drug_id] = "the final response was blank"
                retry.append((drug, bounded_evidence))
                continue
            raw_value = _find_json_mapping(output.text, "mechanism_and_targets")
            if not isinstance(raw_value, Mapping) or any(
                not isinstance(raw_value.get(category), list)
                for category in _SYNTHESIS_CATEGORIES
            ):
                validation_errors[drug.drug_id] = (
                    "the final response did not contain every required JSON array"
                )
                retry.append((drug, bounded_evidence))
                continue
            raw = dict(raw_value)
            raw["__passage_id_map__"] = {
                f"P{index}": item.evidence_id
                for index, item in enumerate(bounded_evidence, start=1)
            }
            last_schema_valid[drug.drug_id] = raw
            fact_error = _synthesis_fact_validation_error(
                raw, evidence=bounded_evidence
            )
            if fact_error:
                validation_errors[drug.drug_id] = fact_error
                retry.append((drug, bounded_evidence))
                continue
            if bounded_evidence and all(
                not raw.get(category) for category in _SYNTHESIS_CATEGORIES
            ):
                validation_errors[drug.drug_id] = (
                    "all arrays were empty despite supplied passages; include relevant "
                    "supported facts or a limitations item explaining the evidence gap"
                )
                retry.append((drug, bounded_evidence))
                continue
            parsed[drug.drug_id] = raw
        pending = retry
    for drug, _evidence in pending:
        if drug.drug_id in last_schema_valid:
            parsed.setdefault(drug.drug_id, last_schema_valid[drug.drug_id])
    return parsed


async def _fetch_registry_with_retries(
    trial_id: str,
    *,
    client: httpx.AsyncClient,
    settings: ResearchSettings,
) -> tuple[Mapping[str, Any] | None, list[ResearchAttempt]]:
    attempts: list[ResearchAttempt] = []
    for attempt in range(1, settings.registry_max_attempts + 1):
        started = utc_now()
        try:
            study = await fetch_trial_study(trial_id, client=client)
        except Exception as error:  # noqa: BLE001 - catalog audit boundary.
            attempts.append(
                ResearchAttempt(
                    drug_id="",
                    facet="trial_registry",
                    source="clinicaltrials_gov",
                    query=trial_id,
                    attempt=attempt,
                    status="failed",
                    started_at=started,
                    finished_at=utc_now(),
                    error_type=type(error).__name__,
                    error_message=clean_text(error, max_chars=1000),
                )
            )
            retryable = isinstance(
                error, (httpx.TimeoutException, httpx.TransportError)
            )
            if isinstance(error, httpx.HTTPStatusError):
                retryable = (
                    error.response.status_code in {408, 425, 429}
                    or error.response.status_code >= 500
                )
            if not retryable or attempt >= settings.registry_max_attempts:
                return None, attempts
            await asyncio.sleep(
                min(
                    settings.maximum_backoff,
                    settings.initial_backoff * (2 ** (attempt - 1)),
                )
            )
            continue
        attempts.append(
            ResearchAttempt(
                drug_id="",
                facet="trial_registry",
                source="clinicaltrials_gov",
                query=trial_id,
                attempt=attempt,
                status="ok",
                started_at=started,
                finished_at=utc_now(),
                result_count=1,
            )
        )
        return study, attempts
    return None, attempts


def _registry_row(trial_id: str, study: Mapping[str, Any] | None) -> dict[str, Any]:
    if study is None:
        return {
            "trial_id": trial_id,
            "title": "",
            "overall_status": "",
            "phases_json": "[]",
            "brief_summary": "",
            "last_update_post_date": "",
            "retrieved_at": utc_now(),
            "registry_status": "blocked",
            "study_sha256": "",
            "study_json": "",
        }
    protocol = study.get("protocolSection") or {}
    ident = protocol.get("identificationModule") or {}
    status = protocol.get("statusModule") or {}
    raw_json = json.dumps(study, ensure_ascii=False, sort_keys=True)
    return {
        "trial_id": trial_id,
        "title": clean_text(
            ident.get("briefTitle") or ident.get("officialTitle"), max_chars=1000
        ),
        "overall_status": clean_text(status.get("overallStatus"), max_chars=100),
        "phases_json": json.dumps(
            (protocol.get("designModule") or {}).get("phases", []) or [],
            ensure_ascii=False,
        ),
        "brief_summary": clean_text(
            (protocol.get("descriptionModule") or {}).get("briefSummary"),
            max_chars=5000,
        ),
        "last_update_post_date": clean_text(
            (status.get("lastUpdatePostDateStruct") or {}).get("date"),
            max_chars=40,
        ),
        "retrieved_at": utc_now(),
        "registry_status": "ok",
        "study_sha256": hashlib.sha256(raw_json.encode()).hexdigest(),
        "study_json": raw_json,
    }


def _merge_evidence(
    *groups: Sequence[EvidencePassage],
) -> list[EvidencePassage]:
    by_id: dict[str, EvidencePassage] = {}
    for group in groups:
        for item in group:
            by_id[item.evidence_id] = item
    return list(by_id.values())


def _role_output_validation_error(
    interventions: Sequence[_RegistryIntervention],
    output: Mapping[str, Mapping[str, Any]],
) -> str | None:
    expected = {str(index) for index in range(len(interventions))}
    if set(output) != expected:
        return (
            f"expected intervention indexes {sorted(expected)}, got "
            f"{sorted(str(index) for index in output)}"
        )
    for index in sorted(expected, key=int):
        item = output.get(index)
        if not isinstance(item, Mapping):
            return f"intervention {index} is not an object"
        role = clean_text(item.get("role"), max_chars=40).casefold()
        if role not in DRUG_ROLES:
            return f"intervention {index} has invalid role {role!r}"
        disposition = clean_text(
            item.get("research_disposition"), max_chars=40
        ).casefold()
        category = clean_text(item.get("exclusion_category"), max_chars=80).casefold()
        confidence = clean_text(item.get("confidence"), max_chars=20).casefold()
        raw_names = item.get("active_entity_names")
        if disposition not in INTERVENTION_SCREENING_DISPOSITIONS:
            return (
                f"intervention {index} has invalid research_disposition {disposition!r}"
            )
        if category not in INTERVENTION_EXCLUSION_CATEGORIES:
            return f"intervention {index} has invalid exclusion_category {category!r}"
        if confidence not in {"high", "medium", "low"}:
            return f"intervention {index} has invalid confidence {confidence!r}"
        if not clean_text(item.get("rationale"), max_chars=1000):
            return f"intervention {index} has no rationale"
        if not isinstance(raw_names, Sequence) or isinstance(raw_names, (str, bytes)):
            return f"intervention {index} active_entity_names is not an array"
        if disposition == "include":
            if role == "supportive":
                return f"intervention {index} includes a supportive agent"
            if category != "none":
                return (
                    f"intervention {index} is included with exclusion_category "
                    f"{category!r}"
                )
            if not _supported_active_names(interventions[int(index)], raw_names):
                returned_names = [
                    clean_text(value, max_chars=300) for value in raw_names
                ]
                return (
                    f"intervention {index} has no active entity name supported by "
                    "its registry name, aliases, or descriptions: "
                    f"{returned_names!r}"
                )
        elif category == "none":
            return (
                f"intervention {index} is {disposition!r} with "
                "exclusion_category='none'"
            )
        elif list(raw_names):
            return (
                f"intervention {index} is {disposition!r} but returned nonempty "
                "active_entity_names"
            )
    return None


def _role_output_is_complete(
    interventions: Sequence[_RegistryIntervention],
    output: Mapping[str, Mapping[str, Any]],
) -> bool:
    return _role_output_validation_error(interventions, output) is None


def _fail_closed_role_output(
    interventions: Sequence[_RegistryIntervention],
    output: Mapping[str, Mapping[str, Any]],
    *,
    validation_error: str,
) -> dict[str, Mapping[str, Any]]:
    """Preserve valid intervention items and exclude invalid ones fail-closed."""

    error = clean_text(validation_error, max_chars=500)
    fallback: dict[str, Mapping[str, Any]] = {}
    for index, intervention in enumerate(interventions):
        role = (
            intervention.initial_role
            if intervention.initial_role in DRUG_ROLES
            else "uncertain"
        )
        fallback[str(index)] = {
            "research_disposition": "uncertain",
            "exclusion_category": "insufficient_context",
            "role": role,
            "confidence": "low",
            "rationale": (
                "Automated intervention screening remained invalid after all "
                f"attempts ({error}); excluded fail-closed."
            ),
            "active_entity_names": [],
        }

    for index, item in output.items():
        if index not in fallback or not isinstance(item, Mapping):
            continue
        candidate = {**fallback, index: item}
        if _role_output_validation_error(interventions, candidate) is None:
            fallback[index] = dict(item)
    return fallback


async def build_good_option_catalog(
    nct_ids: Sequence[str],
    output_path: str | Path,
    *,
    config: MMAIConfig | None = None,
    settings: ResearchSettings | None = None,
    sources: Sequence[DrugEvidenceSource] | None = None,
    web_provider: GeneralWebProvider | None = None,
    role_resolver: RoleResolver | None = None,
    synthesizer: SummarySynthesizer | None = None,
    checkpoint_path: str | Path | None = None,
    reset_checkpoint: bool = False,
    overwrite: bool = False,
    progress_callback: Callable[[str, int, int, str], None] | None = None,
) -> GoodOptionCatalog:
    """Screen and research cancer-treatment agents with public-only checkpoints."""

    normalized_ids = tuple(dict.fromkeys(normalize_nct_id(value) for value in nct_ids))
    if not normalized_ids:
        raise ValueError("At least one NCT ID is required to build a catalog.")
    output = Path(output_path).expanduser().resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"Catalog output already exists: {output}")
    resolved_config = config or load_default_preset()
    resolved_settings = settings or ResearchSettings()
    catalog_config = dict(resolved_config.raw.get("good_option_catalog", {}))
    ncit_resource = str(
        catalog_config.get("ncit_resource")
        or resolved_config.patient_structuring.get("ncit_resource")
        or ""
    )
    ncit_version = str(
        catalog_config.get("ncit_version")
        or resolved_config.patient_structuring.get("ncit_version")
        or "unknown"
    )
    if not ncit_resource:
        raise ValueError("GoodOption catalog configuration is missing ncit_resource.")
    ncit_index = load_ncit_drug_index(ncit_resource)
    resolved_sources = tuple(sources or default_sources(web_provider))
    resolved_checkpoint_path = (
        Path(checkpoint_path).expanduser().resolve()
        if checkpoint_path is not None
        else output.with_name(f"{output.name}_checkpoints")
    )
    if (
        resolved_checkpoint_path == output
        or resolved_checkpoint_path.is_relative_to(output)
        or output.is_relative_to(resolved_checkpoint_path)
    ):
        raise ValueError(
            "Catalog output and checkpoint paths must be separate, non-nested "
            "locations."
        )
    checkpoint = _CatalogCheckpointStore.open(
        resolved_checkpoint_path,
        run_spec=_checkpoint_run_spec(
            nct_ids=normalized_ids,
            config=resolved_config,
            settings=resolved_settings,
            sources=resolved_sources,
            role_resolver=role_resolver,
            synthesizer=synthesizer,
            ncit_resource=ncit_resource,
            ncit_version=ncit_version,
        ),
        reset=reset_checkpoint,
    )
    timeout = httpx.Timeout(max(1.0, resolved_settings.request_timeout))
    registry_rows: list[dict[str, Any]] = []
    attempts: list[ResearchAttempt] = []
    studies: dict[str, Mapping[str, Any]] = {}
    interventions_by_trial: dict[str, tuple[_RegistryIntervention, ...]] = {}
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        for index, trial_id in enumerate(normalized_ids, start=1):
            checkpoint_input = {"trial_id": trial_id}
            saved = checkpoint.load("registry", trial_id, input_value=checkpoint_input)
            prior_attempts: list[ResearchAttempt] = []
            study: Mapping[str, Any] | None = None
            registry_row: dict[str, Any] | None = None
            resumed = False
            if saved is not None:
                prior_attempts = [
                    _attempt_from_record(value)
                    for value in saved.get("attempts", [])
                    if isinstance(value, Mapping)
                ]
                raw_study = saved.get("study")
                raw_row = saved.get("registry_row")
                if (
                    isinstance(raw_study, Mapping)
                    and isinstance(raw_row, Mapping)
                    and str(raw_row.get("registry_status") or "") == "ok"
                ):
                    study = raw_study
                    registry_row = dict(raw_row)
                    resumed = True
            if not resumed:
                study, new_attempts = await _fetch_registry_with_retries(
                    trial_id, client=client, settings=resolved_settings
                )
                trial_attempts = [*prior_attempts, *new_attempts]
                registry_row = _registry_row(trial_id, study)
                checkpoint.save(
                    "registry",
                    trial_id,
                    input_value=checkpoint_input,
                    data={
                        "registry_row": registry_row,
                        "study": study,
                        "attempts": [item.to_record() for item in trial_attempts],
                    },
                )
            else:
                trial_attempts = prior_attempts
            attempts.extend(trial_attempts)
            if registry_row is None:
                raise RuntimeError(
                    f"Registry checkpoint produced no row for {trial_id}."
                )
            registry_rows.append(registry_row)
            if study is not None:
                studies[trial_id] = study
                interventions_by_trial[trial_id] = _extract_registry_interventions(
                    trial_id, study
                )
            if progress_callback:
                suffix = " (checkpoint)" if resumed else ""
                progress_callback(
                    "registry", index, len(normalized_ids), f"{trial_id}{suffix}"
                )

    screening_inputs = {
        trial_id: items for trial_id, items in interventions_by_trial.items() if items
    }
    role_outputs: dict[str, Mapping[str, Mapping[str, Any]]] = {}
    pending_roles: dict[str, tuple[_RegistryIntervention, ...]] = {}
    screening_completed = 0
    for trial_id, items in screening_inputs.items():
        checkpoint_input = {
            "trial_id": trial_id,
            "interventions": [asdict(item) for item in items],
        }
        saved = checkpoint.load(
            "intervention_screening", trial_id, input_value=checkpoint_input
        )
        saved_output = saved.get("output") if saved is not None else None
        if isinstance(saved_output, Mapping):
            normalized_output = {
                str(key): value
                for key, value in saved_output.items()
                if isinstance(value, Mapping)
            }
        else:
            normalized_output = {}
        if _role_output_is_complete(items, normalized_output):
            role_outputs[trial_id] = normalized_output
            screening_completed += 1
            if progress_callback:
                progress_callback(
                    "screening",
                    screening_completed,
                    len(screening_inputs),
                    f"{trial_id} (checkpoint)",
                )
        else:
            pending_roles[trial_id] = items
    if role_resolver is None:
        pending_role_items = list(pending_roles.items())
        role_batch_size = max(
            1, int(catalog_config.get("screening_checkpoint_batch_size", 64))
        )
        for start in range(0, len(pending_role_items), role_batch_size):
            role_batch = dict(pending_role_items[start : start + role_batch_size])
            new_role_outputs = await _resolve_roles_with_default_llm(
                role_batch, config=resolved_config
            )
            completed_screening: set[str] = set()
            for trial_id, output_value in new_role_outputs.items():
                if not _role_output_is_complete(pending_roles[trial_id], output_value):
                    continue
                checkpoint_input = {
                    "trial_id": trial_id,
                    "interventions": [asdict(item) for item in pending_roles[trial_id]],
                }
                checkpoint.save(
                    "intervention_screening",
                    trial_id,
                    input_value=checkpoint_input,
                    data={"output": dict(output_value)},
                )
                role_outputs[trial_id] = output_value
                completed_screening.add(trial_id)
                screening_completed += 1
                if progress_callback:
                    progress_callback(
                        "screening",
                        screening_completed,
                        len(screening_inputs),
                        trial_id,
                    )
            missing_screening = sorted(set(role_batch) - completed_screening)
            for trial_id in missing_screening:
                output_value = new_role_outputs.get(trial_id, {})
                validation_error = _role_output_validation_error(
                    pending_roles[trial_id], output_value
                )
                fallback_output = _fail_closed_role_output(
                    pending_roles[trial_id],
                    output_value,
                    validation_error=validation_error or "no parsed result",
                )
                role_outputs[trial_id] = fallback_output
                screening_completed += 1
                if progress_callback:
                    progress_callback(
                        "screening",
                        screening_completed,
                        len(screening_inputs),
                        f"{trial_id} (fail-closed: "
                        f"{validation_error or 'no parsed result'})",
                    )
    else:
        for trial_id, items in pending_roles.items():
            output_value = await _maybe_await(role_resolver(trial_id, items))
            if not isinstance(output_value, Mapping):
                output_value = {}
            checkpoint_input = {
                "trial_id": trial_id,
                "interventions": [asdict(item) for item in items],
            }
            validation_error = _role_output_validation_error(items, output_value)
            if validation_error is not None:
                raise ValueError(
                    f"Custom intervention screen returned an invalid result for "
                    f"{trial_id}: {validation_error}."
                )
            checkpoint.save(
                "intervention_screening",
                trial_id,
                input_value=checkpoint_input,
                data={"output": dict(output_value)},
            )
            role_outputs[trial_id] = {
                str(key): value
                for key, value in output_value.items()
                if isinstance(value, Mapping)
            }
            screening_completed += 1
            if progress_callback:
                progress_callback(
                    "screening",
                    screening_completed,
                    len(screening_inputs),
                    trial_id,
                )

    identities: dict[str, DrugIdentity] = {}
    assignment_candidates: list[TrialDrugAssignment] = []
    screening_rows: list[dict[str, Any]] = []
    for trial_id, interventions in interventions_by_trial.items():
        outputs = role_outputs.get(trial_id, {})
        for intervention_index, intervention in enumerate(interventions):
            role_output = outputs.get(str(intervention_index), {})
            disposition = clean_text(
                role_output.get("research_disposition"), max_chars=40
            ).casefold()
            if disposition not in INTERVENTION_SCREENING_DISPOSITIONS:
                disposition = "uncertain"
            exclusion_category = clean_text(
                role_output.get("exclusion_category"), max_chars=80
            ).casefold()
            if exclusion_category not in INTERVENTION_EXCLUSION_CATEGORIES:
                exclusion_category = "insufficient_context"
            if disposition == "include":
                exclusion_category = "none"
            elif exclusion_category == "none":
                exclusion_category = "insufficient_context"

            role = intervention.initial_role
            confidence = intervention.role_confidence
            rationale = intervention.role_rationale
            proposed_role = clean_text(role_output.get("role"), max_chars=40).casefold()
            screening_confidence = clean_text(
                role_output.get("confidence"), max_chars=20
            ).casefold()
            screening_rationale = clean_text(
                role_output.get("rationale"), max_chars=1000
            )
            if proposed_role in DRUG_ROLES and not (
                intervention.initial_role in {"control", "supportive"}
                and intervention.role_confidence == "high"
            ):
                role = proposed_role
                confidence = screening_confidence or "low"
            if screening_rationale:
                rationale = screening_rationale
            elif not role_output:
                rationale = (
                    "Intervention screening did not return a complete valid result; "
                    "the entry was excluded fail-closed."
                )
            active_names = (
                _supported_active_names(
                    intervention, role_output.get("active_entity_names", [])
                )
                if disposition == "include"
                else ()
            )
            if disposition == "include" and not active_names:
                disposition = "uncertain"
                exclusion_category = "not_a_concrete_agent"
                rationale = (
                    f"{rationale} No supported concrete active-entity name was "
                    "returned; the entry was excluded fail-closed."
                ).strip()
            screening_rows.append(
                {
                    "trial_id": trial_id,
                    "intervention_index": intervention_index,
                    "registry_name": intervention.registry_name,
                    "intervention_type": intervention.intervention_type,
                    "research_disposition": disposition,
                    "included": disposition == "include",
                    "exclusion_category": exclusion_category,
                    "confidence": screening_confidence or "low",
                    "role": role,
                    "rationale": rationale,
                    "active_entity_names_json": json.dumps(
                        active_names, ensure_ascii=False
                    ),
                    "arm_labels_json": json.dumps(
                        intervention.arm_labels, ensure_ascii=False
                    ),
                    "arm_types_json": json.dumps(
                        intervention.arm_types, ensure_ascii=False
                    ),
                }
            )
            if disposition != "include":
                continue
            for active_name in active_names:
                identity = _canonicalize_drug(
                    active_name, intervention.aliases, ncit_index=ncit_index
                )
                existing = identities.get(identity.drug_id)
                if existing is not None:
                    identity = DrugIdentity(
                        drug_id=existing.drug_id,
                        preferred_name=existing.preferred_name,
                        ncit_code=existing.ncit_code,
                        aliases=tuple(
                            dict.fromkeys((*existing.aliases, *identity.aliases))
                        ),
                        definition=existing.definition or identity.definition,
                    )
                identities[identity.drug_id] = identity
                assignment_candidates.append(
                    TrialDrugAssignment(
                        trial_id=trial_id,
                        drug_id=identity.drug_id,
                        preferred_name=identity.preferred_name,
                        registry_name=intervention.registry_name,
                        intervention_type=intervention.intervention_type,
                        role=role,
                        role_confidence=confidence,
                        scoreable=role in SCOREABLE_ROLES,
                        arm_labels=intervention.arm_labels,
                        arm_types=intervention.arm_types,
                        role_rationale=rationale,
                    )
                )

    role_priority = {
        "investigational": 5,
        "uncertain": 4,
        "control": 3,
        "background": 2,
        "supportive": 1,
    }
    assignments_by_key: dict[tuple[str, str], TrialDrugAssignment] = {}
    for assignment in assignment_candidates:
        key = (assignment.trial_id, assignment.drug_id)
        existing = assignments_by_key.get(key)
        if (
            existing is None
            or role_priority[assignment.role] > role_priority[existing.role]
        ):
            assignments_by_key[key] = assignment
    assignments = tuple(assignments_by_key.values())

    evidence_by_drug: dict[str, list[EvidencePassage]] = {}
    status_by_drug: dict[str, tuple[str, list[str]]] = {}
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        for index, drug in enumerate(identities.values(), start=1):
            checkpoint_input = {"drug": asdict(drug)}
            definition_evidence = _ncit_definition_evidence(
                drug, ncit_version=ncit_version
            )
            saved = checkpoint.load(
                "research", drug.drug_id, input_value=checkpoint_input
            )
            prior_evidence: list[EvidencePassage] = []
            prior_attempts: list[ResearchAttempt] = []
            resumed = False
            if saved is not None:
                prior_evidence = [
                    _evidence_from_record(value)
                    for value in saved.get("evidence", [])
                    if isinstance(value, Mapping)
                ]
                prior_attempts = [
                    _attempt_from_record(value)
                    for value in saved.get("attempts", [])
                    if isinstance(value, Mapping)
                ]
                if str(saved.get("status") or "") == "complete":
                    evidence = _merge_evidence(definition_evidence, prior_evidence)
                    drug_attempts = prior_attempts
                    status = "complete"
                    failures = [str(value) for value in saved.get("failures", [])]
                    resumed = True
            if not resumed:
                new_evidence, new_attempts, status, failures = await research_drug(
                    drug,
                    sources=resolved_sources,
                    settings=resolved_settings,
                    client=client,
                )
                evidence = _merge_evidence(
                    definition_evidence, prior_evidence, new_evidence
                )
                drug_attempts = [*prior_attempts, *new_attempts]
                web_items = [
                    item for item in evidence if item.source_type == "general_web"
                ]
                non_web_items = [
                    item for item in evidence if item.source_type != "general_web"
                ]
                web_document_limit = min(
                    resolved_settings.max_web_results_per_drug,
                    resolved_settings.max_web_documents_per_drug,
                )
                evidence = [*non_web_items, *web_items[:web_document_limit]]
                checkpoint.save(
                    "research",
                    drug.drug_id,
                    input_value=checkpoint_input,
                    data={
                        "drug": asdict(drug),
                        "evidence": [item.to_record() for item in evidence],
                        "attempts": [item.to_record() for item in drug_attempts],
                        "status": status,
                        "failures": list(failures),
                    },
                )
            evidence_by_drug[drug.drug_id] = evidence
            attempts.extend(drug_attempts)
            status_by_drug[drug.drug_id] = (status, failures)
            if progress_callback:
                suffix = " (checkpoint)" if resumed else ""
                progress_callback(
                    "research",
                    index,
                    len(identities),
                    f"{drug.preferred_name}{suffix}",
                )

    synthesis_inputs = [
        (drug, evidence_by_drug[drug.drug_id])
        for drug in identities.values()
        if status_by_drug[drug.drug_id][0] == "complete"
    ]
    synthesized: dict[str, Mapping[str, Any]] = {}
    pending_synthesis: list[tuple[DrugIdentity, Sequence[EvidencePassage]]] = []
    synthesis_completed = 0
    for drug, evidence in synthesis_inputs:
        checkpoint_input = {
            "drug": asdict(drug),
            "evidence": [item.to_record() for item in evidence],
        }
        saved = checkpoint.load("synthesis", drug.drug_id, input_value=checkpoint_input)
        facts = saved.get("facts") if saved is not None else None
        if isinstance(facts, Mapping) and all(
            isinstance(facts.get(category), list) for category in _SYNTHESIS_CATEGORIES
        ):
            synthesized[drug.drug_id] = facts
            synthesis_completed += 1
            if progress_callback:
                progress_callback(
                    "synthesis",
                    synthesis_completed,
                    len(synthesis_inputs),
                    f"{drug.preferred_name} (checkpoint)",
                )
        else:
            pending_synthesis.append((drug, evidence))

    if synthesizer is None:
        nonempty: list[tuple[DrugIdentity, Sequence[EvidencePassage]]] = []
        for drug, evidence in pending_synthesis:
            if not evidence:
                value = {category: [] for category in _SYNTHESIS_CATEGORIES}
                synthesized[drug.drug_id] = value
                checkpoint.save(
                    "synthesis",
                    drug.drug_id,
                    input_value={
                        "drug": asdict(drug),
                        "evidence": [],
                    },
                    data={"facts": value},
                )
                synthesis_completed += 1
                if progress_callback:
                    progress_callback(
                        "synthesis",
                        synthesis_completed,
                        len(synthesis_inputs),
                        drug.preferred_name,
                    )
            else:
                nonempty.append((drug, evidence))
        synthesis_batch_size = max(
            1, int(catalog_config.get("synthesis_checkpoint_batch_size", 64))
        )
        for start in range(0, len(nonempty), synthesis_batch_size):
            batch = nonempty[start : start + synthesis_batch_size]
            batch_outputs = await _default_synthesize_many(
                batch, config=resolved_config
            )
            for drug, evidence in batch:
                value = batch_outputs.get(drug.drug_id)
                if isinstance(value, Mapping):
                    synthesized[drug.drug_id] = value
                    checkpoint.save(
                        "synthesis",
                        drug.drug_id,
                        input_value={
                            "drug": asdict(drug),
                            "evidence": [item.to_record() for item in evidence],
                        },
                        data={"facts": dict(value)},
                    )
                synthesis_completed += 1
                if progress_callback:
                    progress_callback(
                        "synthesis",
                        synthesis_completed,
                        len(synthesis_inputs),
                        drug.preferred_name,
                    )
    else:
        for drug, evidence in pending_synthesis:
            value = await _maybe_await(synthesizer(drug, evidence))
            if isinstance(value, Mapping) and all(
                isinstance(value.get(category), list)
                for category in _SYNTHESIS_CATEGORIES
            ):
                synthesized[drug.drug_id] = value
                checkpoint.save(
                    "synthesis",
                    drug.drug_id,
                    input_value={
                        "drug": asdict(drug),
                        "evidence": [item.to_record() for item in evidence],
                    },
                    data={"facts": dict(value)},
                )
            synthesis_completed += 1
            if progress_callback:
                progress_callback(
                    "synthesis",
                    synthesis_completed,
                    len(synthesis_inputs),
                    drug.preferred_name,
                )

    summaries: list[DrugSummary] = []
    good_option_max_chars = max(
        1000,
        int(catalog_config.get("good_option_summary_max_tokens", 1500)) * 4,
    )
    help_me_choose_max_chars = max(
        1000,
        int(catalog_config.get("help_me_choose_summary_max_tokens", 3000)) * 4,
    )
    for drug in identities.values():
        research_status, failures = status_by_drug[drug.drug_id]
        raw_facts = synthesized.get(drug.drug_id, {})
        facts = _validate_structured_facts(
            raw_facts if isinstance(raw_facts, Mapping) else {},
            evidence=evidence_by_drug[drug.drug_id],
        )
        synthesis_status = (
            "ok"
            if research_status == "complete" and drug.drug_id in synthesized
            else "blocked"
        )
        if research_status == "complete" and synthesis_status == "blocked":
            failures = [
                *failures,
                "Drug evidence synthesis did not return the required JSON schema.",
            ]
        summaries.append(
            DrugSummary(
                drug_id=drug.drug_id,
                preferred_name=drug.preferred_name,
                ncit_code=drug.ncit_code,
                research_status=research_status,
                synthesis_status=synthesis_status,
                structured_facts=facts,
                good_option_summary=(
                    _render_summary(
                        drug,
                        facts,
                        include_safety=False,
                        max_chars=good_option_max_chars,
                    )
                    if synthesis_status == "ok"
                    else ""
                ),
                help_me_choose_summary=(
                    _render_summary(
                        drug,
                        facts,
                        include_safety=True,
                        max_chars=help_me_choose_max_chars,
                    )
                    if synthesis_status == "ok"
                    else ""
                ),
                evidence_count=len(evidence_by_drug[drug.drug_id]),
                technical_failures=tuple(failures),
            )
        )

    trial_registry = pd.DataFrame(registry_rows)
    trial_intervention_screening = pd.DataFrame(
        screening_rows,
        columns=[
            "trial_id",
            "intervention_index",
            "registry_name",
            "intervention_type",
            "research_disposition",
            "included",
            "exclusion_category",
            "confidence",
            "role",
            "rationale",
            "active_entity_names_json",
            "arm_labels_json",
            "arm_types_json",
        ],
    )
    trial_drug_index = pd.DataFrame(
        [item.to_record() for item in assignments],
        columns=[
            "trial_id",
            "drug_id",
            "preferred_name",
            "registry_name",
            "intervention_type",
            "role",
            "role_confidence",
            "scoreable",
            "role_rationale",
            "arm_labels_json",
            "arm_types_json",
        ],
    )
    drug_summaries = pd.DataFrame(
        [item.to_record() for item in summaries],
        columns=[
            "drug_id",
            "preferred_name",
            "ncit_code",
            "research_status",
            "synthesis_status",
            "good_option_summary",
            "help_me_choose_summary",
            "evidence_count",
            "structured_facts_json",
            "technical_failures_json",
        ],
    )
    evidence_items = [item for values in evidence_by_drug.values() for item in values]
    drug_evidence = pd.DataFrame(
        [item.to_record() for item in evidence_items],
        columns=[
            "evidence_id",
            "drug_id",
            "facet",
            "source",
            "source_type",
            "title",
            "passage",
            "url",
            "source_locator",
            "published_at",
            "retrieved_at",
            "license",
            "query",
            "content_sha256",
            "attributes_json",
        ],
    )
    research_attempts = pd.DataFrame([item.to_record() for item in attempts])
    _write_catalog_bundle(
        output,
        trial_registry=trial_registry,
        trial_intervention_screening=trial_intervention_screening,
        trial_drug_index=trial_drug_index,
        drug_summaries=drug_summaries,
        drug_evidence=drug_evidence,
        research_attempts=research_attempts,
        config=resolved_config,
        ncit_resource=ncit_resource,
        ncit_version=ncit_version,
        settings=resolved_settings,
        nct_ids=normalized_ids,
        checkpoint_run_fingerprint=checkpoint.run_fingerprint,
        overwrite=overwrite,
    )
    catalog = load_good_option_catalog(output)
    checkpoint.mark_complete(output)
    return catalog


def _write_catalog_bundle(
    output: Path,
    *,
    trial_registry: pd.DataFrame,
    trial_intervention_screening: pd.DataFrame,
    trial_drug_index: pd.DataFrame,
    drug_summaries: pd.DataFrame,
    drug_evidence: pd.DataFrame,
    research_attempts: pd.DataFrame,
    config: MMAIConfig,
    ncit_resource: str,
    ncit_version: str,
    settings: ResearchSettings,
    nct_ids: Sequence[str],
    checkpoint_run_fingerprint: str,
    overwrite: bool,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Catalog output already exists: {output}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        evidence_dir = temporary / "drug_evidence"
        evidence_dir.mkdir()
        trial_registry.to_parquet(temporary / "trial_registry.parquet", index=False)
        trial_intervention_screening.to_parquet(
            temporary / "trial_intervention_screening.parquet", index=False
        )
        trial_drug_index.to_parquet(temporary / "trial_drug_index.parquet", index=False)
        drug_summaries.to_parquet(temporary / "drug_summaries.parquet", index=False)
        research_attempts.to_parquet(
            temporary / "drug_research_attempts.parquet", index=False
        )
        drug_evidence.to_parquet(evidence_dir / "part-00000.parquet", index=False)
        relative_files = sorted(
            path.relative_to(temporary).as_posix()
            for path in temporary.rglob("*.parquet")
        )
        manifest = {
            "schema_version": CATALOG_SCHEMA_VERSION,
            "build_id": hashlib.sha256(
                f"{utc_now()}\0{len(trial_registry)}\0{len(drug_summaries)}".encode()
            ).hexdigest()[:24],
            "created_at_utc": utc_now(),
            "compatibility_id": _compatibility_id(ncit_version=ncit_version),
            "trial_ids": list(nct_ids),
            "versions": {
                "role_policy": ROLE_POLICY_VERSION,
                "role_prompt": ROLE_PROMPT_VERSION,
                "synthesis_schema": SYNTHESIS_SCHEMA_VERSION,
                "synthesis_prompt": SYNTHESIS_PROMPT_VERSION,
                "good_option_projection": GOOD_OPTION_PROJECTION_VERSION,
                "help_me_choose_projection": HELP_ME_CHOOSE_PROJECTION_VERSION,
                "good_option_prompt": GOOD_OPTION_PROMPT_VERSION,
                "good_option_label_schema": GOOD_OPTION_LABEL_SCHEMA_VERSION,
                "good_option_checker_input": GOOD_OPTION_INPUT_VERSION,
            },
            "ncit": {"version": ncit_version, "resource": ncit_resource},
            "counts": {
                "trials": len(trial_registry),
                "screened_interventions": len(trial_intervention_screening),
                "excluded_interventions": int(
                    (
                        ~trial_intervention_screening.get(
                            "included", pd.Series(dtype=bool)
                        ).astype(bool)
                    ).sum()
                ),
                "trial_drug_assignments": len(trial_drug_index),
                "unique_drugs": len(drug_summaries),
                "evidence_passages": len(drug_evidence),
                "research_attempts": len(research_attempts),
                "blocked_drugs": int(
                    drug_summaries.get("research_status", pd.Series(dtype=str))
                    .astype(str)
                    .eq("blocked")
                    .sum()
                ),
                "blocked_summaries": int(
                    drug_summaries.get("synthesis_status", pd.Series(dtype=str))
                    .astype(str)
                    .eq("blocked")
                    .sum()
                ),
            },
            "research_settings": asdict(settings),
            "teacher_config": config_snapshot(config).get("llm_good_option", {}),
            "catalog_checkpoint": {
                "schema_version": CATALOG_CHECKPOINT_SCHEMA_VERSION,
                "run_fingerprint_sha256": checkpoint_run_fingerprint,
            },
            "files": {
                name: {"sha256": _sha256_file(temporary / name)}
                for name in relative_files
            },
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        validate_good_option_catalog(temporary)
        if output.exists():
            shutil.rmtree(output)
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _read_evidence(path: Path) -> pd.DataFrame:
    files = sorted((path / "drug_evidence").glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat((pd.read_parquet(file) for file in files), ignore_index=True)


def validate_good_option_catalog(path: str | Path) -> dict[str, Any]:
    """Validate schema, hashes, identities, roles, and terminal research states."""

    root = Path(path).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Catalog is missing manifest.json: {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != CATALOG_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported GoodOption catalog schema {manifest.get('schema_version')!r}; "
            f"expected {CATALOG_SCHEMA_VERSION!r}."
        )
    expected_versions = {
        "role_policy": ROLE_POLICY_VERSION,
        "role_prompt": ROLE_PROMPT_VERSION,
        "synthesis_schema": SYNTHESIS_SCHEMA_VERSION,
        "synthesis_prompt": SYNTHESIS_PROMPT_VERSION,
        "good_option_projection": GOOD_OPTION_PROJECTION_VERSION,
        "help_me_choose_projection": HELP_ME_CHOOSE_PROJECTION_VERSION,
        "good_option_prompt": GOOD_OPTION_PROMPT_VERSION,
        "good_option_label_schema": GOOD_OPTION_LABEL_SCHEMA_VERSION,
        "good_option_checker_input": GOOD_OPTION_INPUT_VERSION,
    }
    versions = manifest.get("versions") or {}
    mismatched_versions = {
        key: (versions.get(key), expected)
        for key, expected in expected_versions.items()
        if versions.get(key) != expected
    }
    if mismatched_versions:
        raise ValueError(
            "Catalog component versions are incompatible with this package: "
            f"{mismatched_versions}"
        )
    ncit_version = str((manifest.get("ncit") or {}).get("version") or "unknown")
    if manifest.get("compatibility_id") != _compatibility_id(ncit_version=ncit_version):
        raise ValueError(
            "Catalog compatibility ID does not match its version manifest."
        )
    for relative, metadata in manifest.get("files", {}).items():
        target = root / relative
        if not target.is_file():
            raise ValueError(f"Catalog is missing declared file: {relative}")
        if _sha256_file(target) != metadata.get("sha256"):
            raise ValueError(f"Catalog file hash mismatch: {relative}")
    registry = pd.read_parquet(root / "trial_registry.parquet")
    screening = pd.read_parquet(root / "trial_intervention_screening.parquet")
    index = pd.read_parquet(root / "trial_drug_index.parquet")
    summaries = pd.read_parquet(root / "drug_summaries.parquet")
    evidence = _read_evidence(root)
    declared_trial_ids = manifest.get("trial_ids")
    if declared_trial_ids is not None and list(registry["trial_id"].astype(str)) != [
        str(value) for value in declared_trial_ids
    ]:
        raise ValueError("Catalog trial_ids do not match trial_registry row order.")
    if registry["trial_id"].astype(str).duplicated().any():
        raise ValueError("trial_registry contains duplicate trial_id values.")
    required_screening_columns = {
        "trial_id",
        "intervention_index",
        "registry_name",
        "research_disposition",
        "included",
        "exclusion_category",
        "role",
        "active_entity_names_json",
    }
    missing_screening_columns = sorted(
        required_screening_columns - set(screening.columns)
    )
    if missing_screening_columns:
        raise ValueError(
            "trial_intervention_screening is missing columns: "
            + ", ".join(missing_screening_columns)
        )
    if not screening.empty:
        if screening[["trial_id", "intervention_index"]].astype(str).duplicated().any():
            raise ValueError(
                "trial_intervention_screening contains duplicate intervention rows."
            )
        if set(screening["trial_id"].astype(str)) - set(
            registry["trial_id"].astype(str)
        ):
            raise ValueError(
                "trial_intervention_screening references unknown trial IDs."
            )
        unknown_dispositions = sorted(
            set(screening["research_disposition"].astype(str))
            - INTERVENTION_SCREENING_DISPOSITIONS
        )
        if unknown_dispositions:
            raise ValueError(
                "trial_intervention_screening contains unknown dispositions: "
                f"{unknown_dispositions}"
            )
        unknown_categories = sorted(
            set(screening["exclusion_category"].astype(str))
            - INTERVENTION_EXCLUSION_CATEGORIES
        )
        if unknown_categories:
            raise ValueError(
                "trial_intervention_screening contains unknown exclusion categories: "
                f"{unknown_categories}"
            )
        unknown_screening_roles = sorted(
            set(screening["role"].astype(str)) - DRUG_ROLES
        )
        if unknown_screening_roles:
            raise ValueError(
                "trial_intervention_screening contains unknown roles: "
                f"{unknown_screening_roles}"
            )
        included = screening["research_disposition"].astype(str).eq("include")
        if not screening["included"].astype(bool).eq(included).all():
            raise ValueError(
                "trial_intervention_screening included flags do not match dispositions."
            )
        for record in screening.to_dict(orient="records"):
            names = json.loads(str(record.get("active_entity_names_json") or "[]"))
            if not isinstance(names, list):
                raise ValueError(
                    "trial_intervention_screening active_entity_names_json must be an array."
                )
            is_included = str(record.get("research_disposition")) == "include"
            if is_included != bool(names):
                raise ValueError(
                    "Included intervention screens require active entity names, and "
                    "excluded screens must not contain them."
                )
    if not index.empty:
        if index[["trial_id", "drug_id"]].astype(str).duplicated().any():
            raise ValueError("trial_drug_index contains duplicate trial-drug pairs.")
        unknown_roles = sorted(set(index["role"].astype(str)) - DRUG_ROLES)
        if unknown_roles:
            raise ValueError(
                f"trial_drug_index contains unknown roles: {unknown_roles}"
            )
        if set(index["trial_id"].astype(str)) - set(registry["trial_id"].astype(str)):
            raise ValueError("trial_drug_index references unknown trial IDs.")
        included_pairs = set(
            screening.loc[
                screening["included"].astype(bool), ["trial_id", "registry_name"]
            ]
            .astype(str)
            .itertuples(index=False, name=None)
        )
        indexed_pairs = set(
            index[["trial_id", "registry_name"]]
            .astype(str)
            .itertuples(index=False, name=None)
        )
        if indexed_pairs - included_pairs:
            raise ValueError(
                "trial_drug_index contains an intervention excluded by cancer-treatment "
                "agent screening."
            )
    if summaries["drug_id"].astype(str).duplicated().any():
        raise ValueError("drug_summaries contains duplicate drug_id values.")
    if set(index.get("drug_id", pd.Series(dtype=str)).astype(str)) - set(
        summaries["drug_id"].astype(str)
    ):
        raise ValueError("trial_drug_index references unknown drug IDs.")
    terminal = summaries["research_status"].astype(str).isin({"complete", "blocked"})
    if not terminal.all():
        raise ValueError(
            "Every drug must have a terminal complete/blocked research status."
        )
    synthesis_status = summaries["synthesis_status"].astype(str)
    if not synthesis_status.isin({"ok", "blocked"}).all():
        raise ValueError("Every drug must have a terminal ok/blocked synthesis status.")
    research_blocked_but_synthesized = summaries["research_status"].astype(str).eq(
        "blocked"
    ) & synthesis_status.ne("blocked")
    if research_blocked_but_synthesized.any():
        raise ValueError("A research-blocked drug cannot have a completed synthesis.")
    synthesized = synthesis_status.eq("ok")
    invalid_projection = synthesized & (
        summaries["good_option_summary"].fillna("").astype(str).str.strip().eq("")
        | summaries["help_me_choose_summary"].fillna("").astype(str).str.strip().eq("")
    )
    if invalid_projection.any():
        raise ValueError(
            "Every completed synthesis requires both clean summary projections."
        )
    known_drugs = set(summaries["drug_id"].astype(str))
    evidence_drugs = set(evidence.get("drug_id", pd.Series(dtype=str)).astype(str))
    if evidence_drugs - known_drugs:
        raise ValueError("drug_evidence references unknown drug IDs.")
    evidence_ids_by_drug = (
        {
            drug_id: set(group["evidence_id"].astype(str))
            for drug_id, group in evidence.groupby("drug_id", sort=False)
        }
        if not evidence.empty
        else {}
    )
    for row in summaries.loc[synthesized].to_dict(orient="records"):
        facts = json.loads(str(row.get("structured_facts_json") or "{}"))
        if not isinstance(facts, Mapping):
            raise ValueError("structured_facts_json must contain a JSON object.")
        known_evidence = evidence_ids_by_drug.get(str(row["drug_id"]), set())
        for category in _SYNTHESIS_CATEGORIES:
            for fact in facts.get(category, []) or []:
                if not isinstance(fact, Mapping):
                    raise ValueError("Structured summary facts must be JSON objects.")
                support_ids = fact.get("support_ids", []) or []
                if not isinstance(support_ids, Sequence) or isinstance(
                    support_ids, (str, bytes)
                ):
                    raise ValueError("Structured fact support_ids must be an array.")
                unsupported = set(map(str, support_ids)) - known_evidence
                if unsupported:
                    raise ValueError(
                        "Structured summary references unknown evidence IDs: "
                        + ", ".join(sorted(unsupported))
                    )
    return manifest


def load_good_option_catalog(
    path: str | Path, *, validate: bool = True
) -> GoodOptionCatalog:
    """Load a Parquet bundle and optionally verify its full manifest contract."""

    root = Path(path).expanduser().resolve()
    manifest = (
        validate_good_option_catalog(root)
        if validate
        else json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    )
    return GoodOptionCatalog(
        path=root,
        manifest=manifest,
        trial_registry=pd.read_parquet(root / "trial_registry.parquet"),
        trial_intervention_screening=pd.read_parquet(
            root / "trial_intervention_screening.parquet"
        ),
        trial_drug_index=pd.read_parquet(root / "trial_drug_index.parquet"),
        drug_summaries=pd.read_parquet(root / "drug_summaries.parquet"),
        drug_evidence=_read_evidence(root),
        drug_research_attempts=pd.read_parquet(root / "drug_research_attempts.parquet"),
    )


__all__ = [
    "CATALOG_CHECKPOINT_SCHEMA_VERSION",
    "INTERVENTION_EXCLUSION_CATEGORIES",
    "INTERVENTION_SCREENING_DISPOSITIONS",
    "ROLE_PROMPT_VERSION",
    "SYNTHESIS_PROMPT_VERSION",
    "RoleResolver",
    "SummarySynthesizer",
    "build_good_option_catalog",
    "build_intervention_screening_messages",
    "build_role_resolution_messages",
    "build_synthesis_messages",
    "load_good_option_catalog",
    "validate_good_option_catalog",
]
