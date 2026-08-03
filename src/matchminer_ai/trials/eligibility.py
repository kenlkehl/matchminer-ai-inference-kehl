"""Trial-space-specific eligibility-criteria extraction from OCR text files."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

from matchminer_ai.config import MMAIConfig, config_snapshot, load_default_preset
from matchminer_ai.llm.backends import build_llm_runtime_config, get_llm_backend
from matchminer_ai.llm.prompt_rendering import build_prompt_list

TrialSpaceCriteriaProgress = Callable[[str, int, int, str], None]

_EXPECTED_RESPONSE_KEYS = {
    "coverage_complete",
    "exclusion_criteria",
    "inclusion_criteria",
}


class TrialSpaceCriteriaExtractionError(ValueError):
    """Raised when a trial-space criteria response cannot be validated."""


def _emit_progress(
    callback: TrialSpaceCriteriaProgress | None,
    stage: str,
    completed: int,
    total: int,
    detail: str,
) -> None:
    if callback is not None:
        callback(stage, completed, total, detail)


def _load_prompt_text(filename: str) -> str:
    prompt_path = resources.files("matchminer_ai.prompts").joinpath(filename)
    with prompt_path.open("r", encoding="utf-8") as handle:
        return handle.read().strip()


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
    raise TrialSpaceCriteriaExtractionError(
        "The LLM response did not contain a JSON object."
    )


def _task_runtime_config(
    extraction_config: dict[str, Any],
    *,
    config: MMAIConfig,
) -> dict[str, Any]:
    llm_only_config = {
        key: deepcopy(extraction_config[key])
        for key in ("reasoning_parser", "local", "remote")
        if key in extraction_config
    }
    return build_llm_runtime_config(
        "trial_space_criteria_extraction",
        llm_only_config,
        config=config,
    )


def _normalize_for_grounding(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = re.sub(r"\s+", " ", normalized).strip().casefold()
    return re.sub(r"(?<=\d)\s+(?=(?:st|nd|rd|th)\b)", "", normalized)


def _page_edge_signature(line: str) -> str:
    normalized = unicodedata.normalize("NFKC", line).casefold()
    normalized = re.sub(r"\d+", "#", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _remove_repeated_page_edge_boilerplate(text: str) -> str:
    pages = text.split("\f")
    if len(pages) < 2:
        return text

    page_lines = [page.splitlines() for page in pages]
    edge_signatures_by_page: list[set[str]] = []
    for lines in page_lines:
        nonempty_indices = [index for index, line in enumerate(lines) if line.strip()]
        edge_indices = set(nonempty_indices[:8] + nonempty_indices[-8:])
        edge_signatures_by_page.append(
            {
                signature
                for index in edge_indices
                if len(signature := _page_edge_signature(lines[index])) >= 12
                and any(character.isalpha() for character in signature)
            }
        )

    signature_page_counts: dict[str, int] = {}
    for signatures in edge_signatures_by_page:
        for signature in signatures:
            signature_page_counts[signature] = signature_page_counts.get(signature, 0) + 1
    repeated_signatures = {
        signature for signature, count in signature_page_counts.items() if count >= 2
    }
    if not repeated_signatures:
        return text

    cleaned_pages: list[str] = []
    for lines in page_lines:
        nonempty_indices = [index for index, line in enumerate(lines) if line.strip()]
        edge_indices = set(nonempty_indices[:8] + nonempty_indices[-8:])
        cleaned_pages.append(
            "\n".join(
                line
                for index, line in enumerate(lines)
                if not (
                    index in edge_indices
                    and _page_edge_signature(line) in repeated_signatures
                )
            )
        )
    return "\f".join(cleaned_pages)


def _remove_referenced_page_end_footnotes(text: str) -> str:
    cleaned_pages: list[str] = []
    for page in text.split("\f"):
        lines = page.splitlines()
        nonempty_indices = [index for index, line in enumerate(lines) if line.strip()]
        footnote_start: int | None = None
        for index in nonempty_indices[-8:]:
            match = re.match(r"^\s*(\d{1,2}|[*†‡])\s+\S", lines[index])
            if match is None:
                continue
            marker = re.escape(match.group(1))
            preceding_text = "\n".join(lines[:index])
            if re.search(rf"[A-Za-z0-9)\]]{marker}(?=\s|[,.]|$)", preceding_text):
                footnote_start = index
                break
        if footnote_start is not None:
            lines = lines[:footnote_start]
        cleaned_pages.append("\n".join(lines))
    return "\f".join(cleaned_pages)


def _validate_criteria_list(
    value: Any,
    *,
    field_name: str,
    normalized_document: str,
    max_criteria: int,
) -> list[str]:
    if not isinstance(value, list):
        raise TrialSpaceCriteriaExtractionError(f"{field_name} must be a list.")
    if len(value) > max_criteria:
        raise TrialSpaceCriteriaExtractionError(
            f"{field_name} exceeds the configured limit of {max_criteria}."
        )

    criteria: list[str] = []
    seen: set[str] = set()
    for position, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise TrialSpaceCriteriaExtractionError(
                f"{field_name}[{position}] must be a non-empty string."
            )
        criterion = re.sub(r"\s+", " ", item).strip()
        normalized_criterion = _normalize_for_grounding(criterion)
        if normalized_criterion not in normalized_document:
            raise TrialSpaceCriteriaExtractionError(
                f"{field_name}[{position}] is not a verbatim excerpt from the "
                "OCR document after conservative OCR normalization."
            )
        if normalized_criterion in seen:
            raise TrialSpaceCriteriaExtractionError(
                f"{field_name} contains a duplicate criterion at position {position}."
            )
        seen.add(normalized_criterion)
        criteria.append(criterion)
    return criteria


def _validate_response(
    payload: dict[str, Any],
    *,
    document_text: str,
    max_criteria: int,
) -> dict[str, list[str]]:
    if set(payload) != _EXPECTED_RESPONSE_KEYS:
        raise TrialSpaceCriteriaExtractionError(
            "The response must contain exactly coverage_complete, "
            "inclusion_criteria, and exclusion_criteria."
        )
    if payload["coverage_complete"] is not True:
        raise TrialSpaceCriteriaExtractionError(
            "coverage_complete must be true after the entire document is reviewed."
        )

    normalized_document = _normalize_for_grounding(document_text)
    inclusion = _validate_criteria_list(
        payload["inclusion_criteria"],
        field_name="inclusion_criteria",
        normalized_document=normalized_document,
        max_criteria=max_criteria,
    )
    exclusion = _validate_criteria_list(
        payload["exclusion_criteria"],
        field_name="exclusion_criteria",
        normalized_document=normalized_document,
        max_criteria=max_criteria,
    )
    if not inclusion and not exclusion:
        raise TrialSpaceCriteriaExtractionError(
            "The response did not contain any eligibility criteria."
        )
    overlap = {
        _normalize_for_grounding(criterion) for criterion in inclusion
    }.intersection(_normalize_for_grounding(criterion) for criterion in exclusion)
    if overlap:
        raise TrialSpaceCriteriaExtractionError(
            "The same criterion cannot appear in both inclusion and exclusion lists."
        )
    return {
        "inclusion_criteria": inclusion,
        "exclusion_criteria": exclusion,
    }


@dataclass
class _CriteriaExtractionRunner:
    config: MMAIConfig
    runtime_config: dict[str, Any]
    retry_limit: int
    model_metadata: dict[str, Any] = field(default_factory=dict)
    generation_calls: int = 0

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        document_text: str,
        max_criteria: int,
    ) -> dict[str, list[str]]:
        working_messages = [dict(message) for message in messages]
        backend = get_llm_backend(self.config)
        last_error: TrialSpaceCriteriaExtractionError | None = None

        for _attempt in range(self.retry_limit + 1):
            prompt_list = build_prompt_list(
                [working_messages],
                llm_config=self.runtime_config,
            )
            generation = backend.generate_llm_outputs(
                prompt_list=prompt_list,
                llm_config=self.runtime_config,
                model_metadata_cache_dir=self.config.model_metadata_cache_dir,
            )
            self.generation_calls += 1
            if not self.model_metadata:
                self.model_metadata = dict(generation.model_metadata)
            if len(generation.final_outputs) != 1:
                raise TrialSpaceCriteriaExtractionError(
                    "The LLM returned a different number of outputs than prompts."
                )

            response_text = str(generation.final_outputs[0])
            try:
                finish_reason = (
                    str(generation.finish_reasons[0]).strip().casefold()
                    if generation.finish_reasons
                    else ""
                )
                if finish_reason in {"length", "max_tokens"}:
                    raise TrialSpaceCriteriaExtractionError(
                        "The LLM response reached its token limit, so full-document "
                        "criteria coverage cannot be established."
                    )
                return _validate_response(
                    _extract_json_object(response_text),
                    document_text=document_text,
                    max_criteria=max_criteria,
                )
            except TrialSpaceCriteriaExtractionError as exc:
                last_error = exc
                working_messages.extend(
                    [
                        {"role": "assistant", "content": response_text},
                        {
                            "role": "user",
                            "content": (
                                f"VALIDATION ERROR: {exc} Return one corrected JSON "
                                "object matching the requested schema. Criteria must "
                                "be verbatim OCR excerpts after conservative OCR "
                                "normalization."
                            ),
                        },
                    ]
                )

        raise TrialSpaceCriteriaExtractionError(
            "The LLM did not return valid, source-grounded criteria after retries. "
            f"Last validation error: {last_error}"
        ) from last_error


def _read_ocr_document(
    path: str | Path,
    *,
    max_characters: int,
) -> tuple[str, bytes]:
    document_path = Path(path)
    if document_path.suffix.lower() != ".txt":
        raise ValueError("eligibility_checklist_path must use a .txt extension.")
    if not document_path.is_file():
        raise FileNotFoundError(
            f"OCR eligibility checklist does not exist: {document_path}"
        )
    document_bytes = document_path.read_bytes()
    try:
        document_text = document_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(
            "OCR eligibility checklist must be UTF-8 encoded text."
        ) from exc
    document_text = document_text.replace("\x00", "").strip()
    if not document_text:
        raise ValueError("OCR eligibility checklist must contain non-empty text.")
    if len(document_text) > max_characters:
        raise ValueError(
            "OCR eligibility checklist exceeds the configured "
            f"max_document_characters limit of {max_characters}."
        )
    return document_text, document_bytes


def extract_trial_space_eligibility_criteria(
    trial_space: str,
    eligibility_checklist_path: str | Path,
    *,
    config: MMAIConfig | None = None,
    return_metadata: bool = False,
    progress_callback: TrialSpaceCriteriaProgress | None = None,
) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any]]:
    """Extract complete, source-grounded criteria for one clinical trial space.

    The function reads one local UTF-8 OCR text file and sends its full text,
    together with the supplied trial-space summary, to the configured local or
    remote LLM backend. It returns the original trial space plus separate lists
    of applicable inclusion and exclusion criteria. Every returned criterion is
    required to occur verbatim in the OCR text after conservative OCR
    normalization.

    The OCR document is treated as untrusted prompt data. No web search is used.
    If a remote backend is configured, the entire document reaches that endpoint;
    use only protocol documents without patient data unless the endpoint is
    authorized for the document's sensitivity. The result is a research
    abstraction, not an eligibility determination, and requires review against
    the complete, current protocol.
    """
    if not isinstance(trial_space, str) or not trial_space.strip():
        raise ValueError("trial_space must be a non-empty string.")
    trial_space = trial_space.strip()
    resolved_config = config or load_default_preset()
    if not isinstance(resolved_config, MMAIConfig):
        raise TypeError("config must be an MMAIConfig instance or None.")
    extraction_config = dict(resolved_config.trial_space_criteria_extraction)
    if not extraction_config:
        raise ValueError("Config is missing trial_space_criteria_extraction settings.")

    max_characters = int(extraction_config.get("max_document_characters", 120000))
    max_criteria = int(extraction_config.get("max_criteria_per_type", 256))
    retry_limit = int(extraction_config.get("response_retry_limit", 2))
    if max_characters < 1:
        raise ValueError("max_document_characters must be positive.")
    if max_criteria < 1:
        raise ValueError("max_criteria_per_type must be positive.")
    if retry_limit < 0:
        raise ValueError("response_retry_limit cannot be negative.")

    _emit_progress(
        progress_callback,
        "read",
        0,
        1,
        "Reading OCR eligibility checklist",
    )
    document_text, document_bytes = _read_ocr_document(
        eligibility_checklist_path,
        max_characters=max_characters,
    )
    grounding_document_text = _remove_referenced_page_end_footnotes(
        _remove_repeated_page_edge_boilerplate(document_text)
    )
    _emit_progress(
        progress_callback,
        "read",
        1,
        1,
        "OCR eligibility checklist loaded",
    )

    runtime_config = _task_runtime_config(
        extraction_config,
        config=resolved_config,
    )
    runner = _CriteriaExtractionRunner(
        config=resolved_config,
        runtime_config=runtime_config,
        retry_limit=retry_limit,
    )
    messages = [
        {
            "role": "system",
            "content": _load_prompt_text("trial_space_criteria.extract.system.txt"),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "trial_space": trial_space,
                    "ocr_eligibility_checklist": document_text,
                },
                ensure_ascii=False,
            ),
        },
    ]
    _emit_progress(
        progress_callback,
        "extract",
        0,
        1,
        "Extracting trial-space inclusion and exclusion criteria",
    )
    extracted = runner.generate(
        messages,
        document_text=grounding_document_text,
        max_criteria=max_criteria,
    )
    result = {
        "trial_space": trial_space,
        "inclusion_criteria": extracted["inclusion_criteria"],
        "exclusion_criteria": extracted["exclusion_criteria"],
    }
    _emit_progress(
        progress_callback,
        "extract",
        1,
        1,
        "Trial-space criteria extracted",
    )
    _emit_progress(
        progress_callback,
        "complete",
        1,
        1,
        "Trial-space eligibility criteria ready for human review",
    )

    if not return_metadata:
        return result
    return result, {
        "config_snapshot": config_snapshot(resolved_config),
        "model_metadata": {
            "trial_space_criteria_extractor": runner.model_metadata,
        },
        "source_document": {
            "sha256": hashlib.sha256(document_bytes).hexdigest(),
            "character_count": len(document_text),
            "page_count": document_text.count("\f") + 1,
        },
        "execution": {
            "llm_generation_calls": runner.generation_calls,
            "inclusion_criteria_count": len(result["inclusion_criteria"]),
            "exclusion_criteria_count": len(result["exclusion_criteria"]),
        },
    }


__all__ = [
    "TrialSpaceCriteriaExtractionError",
    "TrialSpaceCriteriaProgress",
    "extract_trial_space_eligibility_criteria",
]
