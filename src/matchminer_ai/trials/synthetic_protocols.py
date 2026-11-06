"""Synthetic full protocol documents seeded by ClinicalTrials.gov (ctgov) records."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from importlib import resources
import itertools
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from matchminer_ai.config import MMAIConfig, config_snapshot, load_default_preset
from matchminer_ai.llm.backends import build_llm_runtime_config, get_llm_backend
from matchminer_ai.llm.prompts import load_prompt_text
from matchminer_ai.llm.request_queue import RequestQueue

SYNTHETIC_PROTOCOL_SYSTEM_PROMPT = "trial.synthetic_protocol.system.txt"
SYNTHETIC_PROTOCOL_USER_PROMPT = "trial.synthetic_protocol.user.txt"
SYNTHETIC_PROTOCOL_LAYOUTS = "synthetic_protocol_layouts.json"
SEED_FIELDS = ("title", "brief_summary", "detailed_description", "eligibility_criteria")
MISSING = "Not provided in the registry record."
SyntheticProtocolProgress = Callable[[str, int, int, str], None]


def load_synthetic_protocol_layouts(filename: str = SYNTHETIC_PROTOCOL_LAYOUTS) -> list[dict[str, Any]]:
    """Eligibility layouts, each {"weight": int, "text": str}, in their fixed order."""
    layouts = json.loads(resources.files("matchminer_ai.data").joinpath(filename).read_text(encoding="utf-8"))
    if not layouts or not all(type(e.get("weight")) is int and e["weight"] > 0 and isinstance(e.get("text"), str)
                              for e in layouts):
        raise ValueError("Synthetic protocol layouts need positive integer weights and text")
    return layouts


def choose_layout(trial_id: str, layouts: list[dict[str, Any]]) -> int:
    """Stable weighted choice from a hash of the trial ID, so reruns keep a trial's layout."""
    point = int(hashlib.sha256(str(trial_id).encode()).hexdigest(), 16) % sum(e["weight"] for e in layouts)
    return next(i for i, bound in enumerate(itertools.accumulate(e["weight"] for e in layouts)) if point < bound)


def _seed_text(value: Any) -> str:
    # Missing table cells arrive as None or NaN; blank ctgov fields are marked too.
    return (value.strip() or MISSING) if isinstance(value, str) else MISSING


def build_synthetic_protocol_messages(
    seed: dict[str, Any], layout_text: str, *, system: str | None = None, user: str | None = None,
) -> list[dict[str, str]]:
    """Format once so braces in ctgov text remain literal."""
    fields = {key: _seed_text(seed.get(key)) for key in SEED_FIELDS}
    system = load_prompt_text(SYNTHETIC_PROTOCOL_SYSTEM_PROMPT) if system is None else system
    user = load_prompt_text(SYNTHETIC_PROTOCOL_USER_PROMPT) if user is None else user
    return [{"role": "system", "content": system},
            {"role": "user", "content": user.format(**fields, eligibility_layout=layout_text)}]


def _emit(callback, stage, completed, total, detail):
    if callback is not None:
        callback(stage, completed, total, detail)


def _write_atomic(path: Path, text: str) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _completed(run_dir: Path, prompt_sha256: str) -> dict[str, Any] | None:
    meta_path = run_dir / "meta.json"
    if not (meta_path.exists() and (run_dir / "protocol.md").exists()):
        return None
    meta = json.loads(meta_path.read_text())
    return meta if meta.get("finish_reason") == "stop" and meta.get("prompt_sha256") == prompt_sha256 else None


def generate_synthetic_protocols(
    seeds: pd.DataFrame,
    *,
    config: MMAIConfig | None = None,
    output_dir: str | Path | None = None,
    return_metadata: bool = False,
    progress_callback: SyntheticProtocolProgress | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    """Write one synthetic full protocol per ctgov seed.

    ``seeds`` has one row per trial with a unique ``trial_id`` and optional
    ``title``, ``brief_summary``, ``detailed_description`` and
    ``eligibility_criteria`` strings (missing fields are marked as not provided).
    The record is a seed, not a specification: the protocol may differ from it,
    and its eligibility section follows one of the weighted layouts in
    ``synthetic_protocol_layouts.json``, chosen by a stable hash of the trial ID.

    Requests go through a :class:`~matchminer_ai.llm.request_queue.RequestQueue`:
    remote backends keep ``synthetic_protocol_generation.chunk_size *
    chunks_in_flight`` requests in flight, each sent as another returns, so one
    slow protocol holds up only itself; other backends get chunks of
    ``chunk_size``, up to ``chunks_in_flight`` at once. Remote runs use the shared
    endpoint pool, so adaptive concurrency and endpoint registries
    (``remote.server_urls_file``) apply as for other workloads. With
    ``output_dir``, each trial's ``protocol.md``, ``reasoning.txt`` and
    ``meta.json`` are written when its request returns, trials already completed
    with the same prompt are skipped, and the result holds file paths instead of
    text. Empty or truncated outputs are retried up to ``max_attempts``.

    Outputs are fabricated training documents, not real protocols, and must not
    be presented as trial criteria. ctgov text reaches the configured LLM.
    """
    resolved = config or load_default_preset()
    if not isinstance(resolved, MMAIConfig):
        raise TypeError("config must be an MMAIConfig instance or None.")
    settings = dict(resolved.synthetic_protocol_generation)
    if not settings:
        raise ValueError("Config is missing synthetic_protocol_generation settings.")
    if not isinstance(seeds, pd.DataFrame) or "trial_id" not in seeds:
        raise ValueError("seeds must be a DataFrame with a trial_id column.")
    ids = seeds["trial_id"]
    if ids.isna().any() or ids.astype(str).str.strip().eq("").any() or not ids.astype(str).is_unique:
        raise ValueError("trial_id must be nonempty and unique")
    chunk_size = settings.get("chunk_size", 500)
    chunks_in_flight = settings.get("chunks_in_flight", 2)
    max_attempts = settings.get("max_attempts", 3)
    for name, value, upper in (("chunk_size", chunk_size, None), ("chunks_in_flight", chunks_in_flight, 16),
                               ("max_attempts", max_attempts, 10)):
        if type(value) is not int or value < 1 or (upper and value > upper):
            raise ValueError(f"synthetic_protocol_generation.{name} must be an integer from 1"
                             + (f" to {upper}" if upper else ""))

    system, user = load_prompt_text(SYNTHETIC_PROTOCOL_SYSTEM_PROMPT), load_prompt_text(SYNTHETIC_PROTOCOL_USER_PROMPT)
    layouts = load_synthetic_protocol_layouts(settings.get("layouts_file", SYNTHETIC_PROTOCOL_LAYOUTS))
    prompt_sha256 = hashlib.sha256(json.dumps([system, user, layouts], sort_keys=True).encode()).hexdigest()
    llm_only = {key: deepcopy(value) for key, value in settings.items()
                if key in {"local", "remote", "reasoning_parser", "sampling_profile", "reasoning_effort"}}
    runtime = build_llm_runtime_config("synthetic_protocol_generation", llm_only, config=resolved)
    root = Path(output_dir) if output_dir is not None else None

    records = seeds.assign(trial_id=ids.astype(str)).to_dict("records")
    rows, pending, metadata = {}, [], {
        "prompt_sha256": prompt_sha256, "trial_errors": {}, "trial_attempts": {}, "model_metadata": {},
        "skipped_completed": 0,
    }
    for record in records:
        trial_id = record["trial_id"]
        layout_index = choose_layout(trial_id, layouts)
        done = _completed(root / trial_id, prompt_sha256) if root else None
        if done:
            metadata["skipped_completed"] += 1
            rows[trial_id] = dict(trial_id=trial_id, layout_index=layout_index, finish_reason="stop",
                                  attempts=done.get("attempt"), protocol=None,
                                  protocol_path=str(root / trial_id / "protocol.md"))
            continue
        pending.append((record, layout_index, 1))
    total = len(records)
    _emit(progress_callback, "generate", len(rows), total, f"{len(pending)} protocol(s) to generate")

    def messages(record, layout_index):
        return build_synthetic_protocol_messages(record, layouts[layout_index]["text"], system=system, user=user)

    def record_answer(item, text, reasoning, finish, provenance):
        """Save an accepted output, or queue a retry."""
        record, layout_index, attempt = item
        trial_id, text = record["trial_id"], str(text or "")
        attempt_record = {"attempt": attempt, "finish_reason": finish, "characters": len(text)}
        if provenance is not None:
            attempt_record["response_metadata"] = provenance
        metadata["trial_attempts"].setdefault(trial_id, []).append(attempt_record)
        if finish != "stop" or not text.strip():
            metadata["trial_errors"][trial_id] = f"finish_reason={finish}, {len(text)} characters"
            if attempt < max_attempts:
                queue.add((record, layout_index, attempt + 1), messages(record, layout_index))
            return
        metadata["trial_errors"].pop(trial_id, None)
        row = dict(trial_id=trial_id, layout_index=layout_index, finish_reason=finish, attempts=attempt,
                   protocol=text, protocol_path=None)
        if root:
            run_dir = root / trial_id
            run_dir.mkdir(parents=True, exist_ok=True)
            _write_atomic(run_dir / "protocol.md", text)
            _write_atomic(run_dir / "reasoning.txt", str(reasoning or ""))
            seed = {key: record.get(key) for key in ("trial_id", *SEED_FIELDS) if key in record}
            _write_atomic(run_dir / "meta.json", json.dumps(dict(
                trial_id=trial_id, prompt_sha256=prompt_sha256, layout_index=layout_index, attempt=attempt,
                finish_reason=finish, completed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                response_metadata=attempt_record.get("response_metadata"), seed=seed,
            ), indent=2, default=str))
            row.update(protocol=None, protocol_path=str(run_dir / "protocol.md"))
        rows[trial_id] = row

    if pending:
        queue = RequestQueue(get_llm_backend(resolved), {None: runtime}, chunk_size=chunk_size,
                             chunks_in_flight=chunks_in_flight,
                             model_metadata_cache_dir=resolved.model_metadata_cache_dir)
        for record, layout_index, attempt in pending:
            queue.add((record, layout_index, attempt), messages(record, layout_index))
        # Results, files and retry bookkeeping are handled only in this thread.
        metadata["model_metadata"] = queue.run(record_answer, progress=lambda: _emit(
            progress_callback, "generate", len(rows), total,
            f"{len(rows)} of {total} generated; {len(metadata['trial_errors'])} awaiting retry or failed"))

    columns = ["trial_id", "layout_index", "finish_reason", "attempts", "protocol", "protocol_path"]
    result = pd.DataFrame([rows[r["trial_id"]] for r in records if r["trial_id"] in rows], columns=columns)
    result.attrs.update(prompt_sha256=prompt_sha256, trial_errors=dict(metadata["trial_errors"]))
    _emit(progress_callback, "complete", len(rows), total, "Synthetic protocol generation finished")
    if not return_metadata:
        return result
    metadata["config_snapshot"] = config_snapshot(resolved)
    return result, metadata


__all__ = [
    "SyntheticProtocolProgress", "build_synthetic_protocol_messages", "choose_layout",
    "generate_synthetic_protocols", "load_synthetic_protocol_layouts",
]
