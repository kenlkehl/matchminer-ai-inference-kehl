"""Treatment screening and cohort extraction from full protocol documents."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
from typing import Any

import pandas as pd

from matchminer_ai.config import MMAIConfig, config_snapshot, load_default_preset
from matchminer_ai.documents import ocr_pdf
from matchminer_ai.llm.backends import build_llm_runtime_config, get_llm_backend
from matchminer_ai.llm.prompt_rendering import build_prompt_list

from .cohort_schema import (
    PROTOCOL_COHORT_JSON_SCHEMA, PROTOCOL_COHORT_PROMPT, PROTOCOL_COHORT_SCHEMA_VERSION,
    build_protocol_cohort_messages, load_cohort_prompt, parse_cohort_response,
)
from .cohorts import COHORT_COLUMNS, COHORT_REPRESENTATION_VERSION

ProtocolCohortProgress = Callable[[str, int, int, str], None]


def _emit(callback, stage, completed, total, detail):
    if callback is not None:
        callback(stage, completed, total, detail)


def _read_protocols(protocols, *, text_dir, keep_text, max_characters, progress):
    """Return (trial_id, text, provenance) per row; PDFs are converted locally."""
    if not isinstance(protocols, pd.DataFrame):
        raise TypeError("protocols must be a pandas DataFrame.")
    missing = {"trial_id", "protocol_path"} - set(protocols.columns)
    if missing:
        raise ValueError(f"protocols is missing required columns: {sorted(missing)}")
    ids = protocols["trial_id"]
    if ids.isna().any() or ids.astype(str).str.strip().eq("").any():
        raise ValueError("trial_id must be nonempty")
    if not ids.astype(str).is_unique:
        raise ValueError("trial_id must be unique")
    paths = [Path(p) for p in protocols["protocol_path"]]
    for path in paths:
        if path.suffix.lower() not in {".pdf", ".txt"}:
            raise ValueError(f"protocol_path must be a .pdf or .txt file: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"Protocol document does not exist: {path}")
    documents = []
    for done, (trial_id, path) in enumerate(zip(ids.astype(str), paths)):
        _emit(progress, "read", done, len(paths), f"Reading protocol for {trial_id}")
        source_bytes = path.read_bytes()
        text_path = path
        if path.suffix.lower() == ".pdf":
            # Text extraction and OCR run locally; nothing leaves the machine here.
            text_path = ocr_pdf(path, Path(text_dir) / f"{trial_id}__{path.stem}.txt", overwrite=True)
        try:
            text = text_path.read_bytes().decode("utf-8-sig").replace("\x00", "").strip()
        except UnicodeDecodeError as exc:
            raise ValueError(f"Protocol text must be UTF-8: {text_path}") from exc
        if not text:
            raise ValueError(f"Protocol document contains no text: {path}")
        if len(text) > max_characters:
            raise ValueError(
                f"Protocol for {trial_id} has {len(text)} characters, above the configured "
                f"max_document_characters limit of {max_characters}; it was not truncated."
            )
        documents.append((trial_id, text, {
            "protocol_path": str(path), "sha256": hashlib.sha256(source_bytes).hexdigest(),
            "character_count": len(text), "page_count": text.count("\f") + 1,
            "text_path": str(text_path) if keep_text or text_path == path else None,
        }))
    _emit(progress, "read", len(paths), len(paths), "Protocol text ready")
    return documents


def _runtime(settings, config):
    """Task runtime; remote requests constrain the answer with the protocol JSON schema."""
    llm_only = {key: deepcopy(value) for key, value in settings.items()
                if key in {"local", "remote", "reasoning_parser", "sampling_profile", "reasoning_effort"}}
    runtime = build_llm_runtime_config("protocol_cohort_extraction", llm_only, config=config)
    structured = bool(settings.get("structured_outputs", True)) and runtime["backend_mode"] == "remote"
    if structured:
        # vLLM applies the grammar after the reasoning section when the server
        # runs a reasoning parser, so thinking models keep their reasoning.
        runtime["remote"]["request_params"]["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "protocol_cohorts", "strict": True, "schema": PROTOCOL_COHORT_JSON_SCHEMA},
        }
    return runtime, structured


def _rows(trial_id, parsed):
    if parsed["multiple_cohorts"] == "no":
        criteria = parsed["eligibility_criteria"]
        return [{
            "cohort_trial_id": f"{trial_id}-cohort-0", "trial_id": trial_id, "cohort_number": 0,
            "cohort_name": None, "cohort_eligibility_criteria": criteria,
            "cohort_specific_eligibility_criteria": None, "common_eligibility_criteria": None,
            "cohort_criteria_source": "protocol", "cancer_treatment_trial": "yes", "multiple_cohorts": "no",
        }]
    common = parsed["common_eligibility_criteria"]
    rows = []
    for number, cohort in enumerate(parsed["cohorts"]):
        specific = cohort["cohort_specific_eligibility_criteria"]
        complete = (f"Cohort-specific eligibility criteria:\n{specific}\n\n"
                    f"Common eligibility criteria:\n{common}") if common else specific
        rows.append({
            "cohort_trial_id": f"{trial_id}-cohort-{number}", "trial_id": trial_id, "cohort_number": number,
            "cohort_name": cohort["cohort_name"], "cohort_eligibility_criteria": complete,
            "cohort_specific_eligibility_criteria": specific, "common_eligibility_criteria": common,
            "cohort_criteria_source": "protocol", "cancer_treatment_trial": "yes", "multiple_cohorts": "yes",
        })
    return rows


def extract_protocol_cohorts(
    protocols: pd.DataFrame,
    *,
    config: MMAIConfig | None = None,
    text_output_dir: str | Path | None = None,
    return_metadata: bool = False,
    progress_callback: ProtocolCohortProgress | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    """Screen full protocol documents and extract their cohort eligibility criteria.

    ``protocols`` has one row per trial with a unique ``trial_id`` and a
    ``protocol_path`` to a PDF or UTF-8 text file. PDFs are converted locally
    with :func:`matchminer_ai.ocr_pdf`; pass ``text_output_dir`` to keep the
    text files, otherwise they are written to a temporary directory and removed.
    Each document is sent whole, without truncation, to the configured LLM with
    ``trial.cohorts.protocol.user.txt``. With a remote backend and
    ``protocol_cohort_extraction.structured_outputs`` enabled (the default), the
    request constrains the answer to the protocol cohort JSON schema with vLLM
    structured outputs; local in-process generation relies on validation and
    retries. Every answer is checked with the ``trial-protocol-cohorts-v1``
    contract, and only failed trials are retried, up to ``max_attempts``.

    Returns one row per accepted cohort with the same columns as
    :func:`summarize_trials` cohort extraction (``COHORT_COLUMNS``); a
    single-cohort protocol yields one row whose criteria come from the protocol,
    and ``cohort_criteria_source`` is ``"protocol"``. Non-treatment trials and
    trials that fail validation produce no rows and are reported in
    ``DataFrame.attrs`` and the metadata.

    Protocol text is treated as untrusted data. The entire document reaches the
    configured LLM endpoint; use only documents without patient data unless the
    endpoint is authorized for their sensitivity. Results are research
    abstractions for human review, not eligibility determinations.
    """
    resolved = config or load_default_preset()
    if not isinstance(resolved, MMAIConfig):
        raise TypeError("config must be an MMAIConfig instance or None.")
    settings = dict(resolved.protocol_cohort_extraction)
    if not settings:
        raise ValueError("Config is missing protocol_cohort_extraction settings.")
    max_attempts = settings.get("max_attempts", 3)
    max_characters = settings.get("max_document_characters", 800000)
    if type(max_attempts) is not int or not 1 <= max_attempts <= 10:
        raise ValueError("protocol_cohort_extraction.max_attempts must be an integer from 1 to 10")
    if type(max_characters) is not int or max_characters < 1:
        raise ValueError("protocol_cohort_extraction.max_document_characters must be a positive integer")

    with tempfile.TemporaryDirectory(prefix="mmai-protocol-text-") as scratch:
        text_dir = Path(text_output_dir) if text_output_dir is not None else Path(scratch)
        text_dir.mkdir(parents=True, exist_ok=True)
        documents = _read_protocols(protocols, text_dir=text_dir, keep_text=text_output_dir is not None,
                                    max_characters=max_characters, progress=progress_callback)

    prompt = load_cohort_prompt(settings.get("prompt_file", PROTOCOL_COHORT_PROMPT))
    runtime, structured = _runtime(settings, resolved)
    metadata = {
        "extraction_method": "protocol_cohorts", "schema_version": PROTOCOL_COHORT_SCHEMA_VERSION,
        "representation_version": COHORT_REPRESENTATION_VERSION,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "structured_outputs": structured,
        "model_metadata": {}, "trial_decisions": {}, "trial_rejections": {}, "trial_errors": {},
        "trial_attempts": {}, "source_documents": {trial_id: info for trial_id, _, info in documents},
    }
    if resolved.debug_mode:
        metadata["trial_responses"] = {}
    texts = {i: text for i, (_, text, _) in enumerate(documents)}
    messages = {i: build_protocol_cohort_messages(text, prompt=prompt) for i, text in texts.items()}
    pending = list(messages)
    backend = get_llm_backend(resolved) if pending else None
    for attempt in range(1, max_attempts + 1):
        if not pending:
            break
        _emit(progress_callback, "extract", len(documents) - len(pending), len(documents),
              f"Extracting cohorts, attempt {attempt} for {len(pending)} protocol(s)")
        generation = backend.generate_llm_outputs(
            prompt_list=build_prompt_list([messages[i] for i in pending], llm_config=runtime),
            llm_config=runtime, model_metadata_cache_dir=resolved.model_metadata_cache_dir,
        )
        metadata["model_metadata"] = generation.model_metadata
        next_pending = []
        for i, answer, reasoning, finish in zip(pending, generation.final_outputs, generation.reasoning_outputs,
                                                generation.finish_reasons, strict=True):
            trial_id = documents[i][0]
            record = {"attempt": attempt, "finish_reason": finish}
            if resolved.debug_mode:
                record.update(answer=answer, reasoning=reasoning)
                metadata["trial_responses"][trial_id] = {"answer": answer, "reasoning": reasoning, "finish_reason": finish}
            try:
                parsed = parse_cohort_response(answer, finish, schema_version=PROTOCOL_COHORT_SCHEMA_VERSION)
            except ValueError as exc:
                record["validation_error"] = str(exc)
                metadata["trial_errors"][trial_id] = str(exc)
                next_pending.append(i)
                # Retry with the complete document and original instructions.
                messages[i] = build_protocol_cohort_messages(texts[i], prompt=prompt) + [{
                    "role": "user",
                    "content": (f"The preceding attempt failed validation: {exc}. Return a complete "
                                "replacement JSON object following the original instructions and full protocol."),
                }]
            else:
                metadata["trial_errors"].pop(trial_id, None)
                metadata["trial_decisions"][trial_id] = parsed
                if parsed["cancer_treatment_trial"] == "no":
                    metadata["trial_rejections"][trial_id] = parsed
            metadata["trial_attempts"].setdefault(trial_id, []).append(record)
        pending = next_pending
    _emit(progress_callback, "extract", len(documents) - len(pending), len(documents), "Cohort extraction finished")

    rows = []
    for trial_id, _, _ in documents:
        parsed = metadata["trial_decisions"].get(trial_id)
        if parsed is not None and parsed["cancer_treatment_trial"] == "yes":
            rows.extend(_rows(trial_id, parsed))
    result = pd.DataFrame(rows, columns=COHORT_COLUMNS)
    result["cohort_number"] = result.cohort_number.astype("int64")
    result.attrs.update({key: metadata[key] for key in (
        "extraction_method", "schema_version", "representation_version", "prompt_sha256",
        "structured_outputs", "trial_decisions", "trial_rejections", "trial_errors",
    )})
    _emit(progress_callback, "complete", 1, 1, "Protocol cohorts ready for human review")
    if not return_metadata:
        return result
    metadata["config_snapshot"] = config_snapshot(resolved)
    return result, metadata


__all__ = ["ProtocolCohortProgress", "extract_protocol_cohorts"]
