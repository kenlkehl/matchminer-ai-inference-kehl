"""Treatment screening and cohort extraction from full protocol documents (the registry workflow's prompts)."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

import pandas as pd

from matchminer_ai._storage import atomic_json
from matchminer_ai.config import MMAIConfig, config_snapshot, load_default_preset
from matchminer_ai.documents import ocr_pdf
from matchminer_ai.llm.backends import build_llm_runtime_config, get_llm_backend

from .cohort_workflow import (
    COHORT_COLUMNS, REPRESENTATION_VERSION, SCHEMA_VERSION, decision_rows, load_prompts, prompt_sha256,
    run_cohort_workflow, step_runtimes, validate_decision,
)
from .cohorts import cohort_settings, workflow_metadata

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
        if path.suffix.lower() not in {".pdf", ".txt", ".md"}:
            raise ValueError(f"protocol_path must be a .pdf, .txt or .md file: {path}")
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
    """Task runtime from the protocol settings' LLM sections."""
    llm_only = {key: deepcopy(value) for key, value in settings.items()
                if key in {"local", "remote", "reasoning_parser", "sampling_profile", "reasoning_effort"}}
    return build_llm_runtime_config("protocol_cohort_extraction", llm_only, config=config)


def _saved_decision(trial_dir, identity):
    """A saved decision is reused only for the same prompts, schema and document bytes."""
    path = trial_dir / "extraction.json"
    if not path.exists():
        return None
    saved = json.loads(path.read_text(encoding="utf-8"))
    if not all(saved.get(key) == value for key, value in identity.items()):
        return None
    validate_decision(saved["decision"])
    return saved


def extract_protocol_cohorts(
    protocols: pd.DataFrame,
    *,
    config: MMAIConfig | None = None,
    text_output_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    return_metadata: bool = False,
    progress_callback: ProtocolCohortProgress | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    """Screen full protocol documents and extract their cohort eligibility criteria.

    Each cohort's criteria are copied verbatim, boilerplate included. Two optional steps take the
    returned rows, as for registry listings (:func:`~matchminer_ai.trials.summarize_trials`):
    :func:`~matchminer_ai.trials.merge_equivalent_cohorts` and
    :func:`~matchminer_ai.trials.remove_boilerplate_criteria`, both with
    ``section="protocol_cohort_extraction"`` (training uses both, in that order).

    ``protocols`` has one row per trial with a unique ``trial_id`` and a
    ``protocol_path`` to a PDF or UTF-8 text file (``.txt`` or ``.md``). PDFs are converted locally
    with :func:`matchminer_ai.ocr_pdf`; pass ``text_output_dir`` to keep the
    text files, otherwise they are written to a temporary directory and removed.
    Each document is sent whole, without truncation, through the same three
    prompts as registry listings in :func:`summarize_trials`
    (:mod:`matchminer_ai.trials.cohort_workflow`): a cancer-treatment screen, the
    list of cohort names, and one request per cohort for its eligibility
    criteria. With a remote backend and ``protocol_cohort_extraction.structured_outputs``
    enabled (the default), each request constrains its answer to that step's JSON
    schema with vLLM structured outputs; local in-process generation relies on
    validation and retries. Only failed requests are retried, up to ``max_attempts``.

    Requests go through the backend in chunks of ``chunk_size``, with up to
    ``chunks_in_flight`` chunks running at once so fast endpoints are not left
    idle while an earlier chunk finishes. With ``output_dir``, each decided trial
    is written to ``<output_dir>/<trial_id>/extraction.json`` (parsed decision,
    cohort names, attempts per step with serving endpoints, source-document hash)
    with ``reasoning.json`` (each step's reasoning) as soon as it is decided; a
    rerun reuses saved decisions made with the same prompts and schema for the
    same document bytes and sends only the rest.

    Returns one row per accepted cohort with the same columns as
    :func:`summarize_trials` cohort extraction (``COHORT_COLUMNS``): each row's
    ``cohort_eligibility_criteria`` are that cohort's complete criteria, and
    ``cohort_criteria_source`` is ``"protocol"``. Non-treatment trials and
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
    max_attempts, chunk_size, chunks_in_flight, structured_setting = cohort_settings(
        settings, "protocol_cohort_extraction")
    max_characters = settings.get("max_document_characters", 800000)
    if type(max_characters) is not int or max_characters < 1:
        raise ValueError("protocol_cohort_extraction.max_document_characters must be a positive integer")

    with tempfile.TemporaryDirectory(prefix="mmai-protocol-text-") as scratch:
        text_dir = Path(text_output_dir) if text_output_dir is not None else Path(scratch)
        text_dir.mkdir(parents=True, exist_ok=True)
        documents = _read_protocols(protocols, text_dir=text_dir, keep_text=text_output_dir is not None,
                                    max_characters=max_characters, progress=progress_callback)

    prompts = load_prompts(settings.get("prompt_files"))
    runtime = _runtime(settings, resolved)
    structured = structured_setting and runtime["backend_mode"] == "remote"
    hashes = prompt_sha256(prompts)
    identity = {"prompt_sha256": hashes["combined"], "schema_version": SCHEMA_VERSION}
    root = Path(output_dir) if output_dir is not None else None
    info = {trial_id: provenance for trial_id, _, provenance in documents}
    saved, pending = {}, []
    for trial_id, text, provenance in documents:
        record = _saved_decision(root / trial_id, {**identity, "source_sha256": provenance["sha256"]}) if root else None
        if record is None:
            pending.append((trial_id, text))
        else:
            saved[trial_id] = record

    def save(trial_id, decision, state):
        if root is None:
            return
        reasoning = {"screen": None, "list": None, "criteria": {}}
        if "responses" in state:
            reasoning = {step: (value or {}).get("reasoning") if step != "criteria"
                         else {i: r.get("reasoning") for i, r in value.items()}
                         for step, value in state["responses"].items()}
        atomic_json(root / trial_id / "reasoning.json", reasoning)
        atomic_json(root / trial_id / "extraction.json", {
            "trial_id": trial_id, **identity, "source_sha256": info[trial_id]["sha256"],
            "representation_version": REPRESENTATION_VERSION, "structured_outputs": structured,
            "prompt_sha256_by_step": {step: hashes[step] for step in prompts},
            "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source_document": info[trial_id], "decision": decision, "cohort_names": state["cohort_names"],
            "merged_cohorts": state["merged_cohorts"], "attempts": state["attempts"],
        })

    def report(states):
        decided = len(saved) + sum(state["decision"] is not None for state in states.values())
        _emit(progress_callback, "extract", decided, len(documents),
              f"{decided} of {len(documents)} protocols decided")

    report({})
    states, model_metadata = ({}, {})
    if pending:
        # Debug responses are kept for reasoning.json even outside debug mode.
        states, model_metadata = run_cohort_workflow(
            pending, prompts=prompts, runtimes=step_runtimes(runtime, structured),
            backend=get_llm_backend(resolved), max_attempts=max_attempts,
            model_metadata_cache_dir=resolved.model_metadata_cache_dir, chunk_size=chunk_size,
            chunks_in_flight=chunks_in_flight, debug=resolved.debug_mode or root is not None,
            on_decision=save, on_progress=report)
    metadata = workflow_metadata(states, prompts, structured, "protocol_cohorts")
    for trial_id, record in saved.items():
        metadata["trial_decisions"][trial_id] = record["decision"]
        metadata["trial_attempts"][trial_id] = record["attempts"]
        if record.get("cohort_names") is not None:
            metadata["trial_cohort_names"][trial_id] = record["cohort_names"]
        if record["decision"]["cancer_treatment_trial"] == "no":
            metadata["trial_rejections"][trial_id] = record["decision"]
    metadata.update(model_metadata=model_metadata, skipped_completed=len(saved), source_documents=info)
    if resolved.debug_mode:
        metadata["trial_responses"] = {trial_id: state["responses"] for trial_id, state in states.items()}

    rows = []
    for trial_id, _, _ in documents:
        decision = metadata["trial_decisions"].get(trial_id)
        if decision is not None and decision["cancer_treatment_trial"] == "yes":
            rows.extend(decision_rows(trial_id, decision, source="protocol"))
    result = pd.DataFrame(rows, columns=COHORT_COLUMNS)
    result["cohort_number"] = result.cohort_number.astype("int64")
    result.attrs.update({key: metadata[key] for key in (
        "extraction_method", "schema_version", "representation_version", "prompt_sha256",
        "structured_outputs", "trial_decisions", "trial_rejections", "trial_errors", "skipped_completed",
    )})
    _emit(progress_callback, "complete", 1, 1, "Protocol cohorts ready for human review")
    if not return_metadata:
        return result
    metadata["config_snapshot"] = config_snapshot(resolved)
    return result, metadata


__all__ = ["ProtocolCohortProgress", "extract_protocol_cohorts"]
