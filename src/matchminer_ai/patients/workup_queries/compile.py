"""Compile guideline workup requirements into validated queries, once per requirement.

The model's only power is to answer with a query: it gets the requirement text,
the guideline population, the extraction data model and the query language, and
no tools, files, browsing or shell. Each answer is decoded under the query's
JSON schema (remote backends with ``structured_outputs``) and validated in code
(:func:`~.language.validate_workup_query`). A malformed or invalid answer is
retried with its problems named, up to ``max_attempts`` times, then once with
thinking off (``reasoning_off_fallback``). Queries are compiled per requirement,
never per patient, and an optional ``cache`` keyed by the requirement text, the
population context, the extraction schema version, the query language version
and the prompt and response-schema hashes (:func:`workup_query_cache_key`)
reuses earlier compilations.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, MutableMapping, Sequence
from datetime import datetime, timezone
from typing import Any

from matchminer_ai._storage import digest
from matchminer_ai.config import MMAIConfig, load_default_preset
from matchminer_ai.llm.backends import get_llm_backend
from matchminer_ai.llm.prompts import load_prompt_text
from matchminer_ai.llm.request_queue import RequestQueue
from matchminer_ai.patients.extraction.common import (
    attempt_entry,
    attempt_group,
    fill,
    parse_answer,
    section_settings,
)

from .language import (
    QUERY_LANGUAGE_VERSION,
    SCHEMA_VERSION,
    canonical_query,
    query_response_schema,
    render_examples,
    render_operator_guide,
    render_query_field_guide,
    validate_workup_query,
)

logger = logging.getLogger(__name__)

SECTION = "workup_query_compile"
PROMPT_FILE = "patient.workup_query.compile.txt"
_MAX_PROBLEMS = 12


def requirement_text(requirement: dict) -> dict:
    """The requirement fields a query is compiled from: name, conditions, category."""
    if not isinstance(requirement, dict):
        raise ValueError("Each requirement must be a dict with a name.")
    name = requirement.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Each requirement needs a nonempty name.")
    values = {"name": name.strip()}
    for key in ("conditions", "category"):
        value = requirement.get(key, "")
        if value is None:
            value = ""
        if not isinstance(value, str):
            raise ValueError(f"A requirement's {key} must be text.")
        values[key] = value.strip()
    return values


def default_extensions(config: MMAIConfig | None = None) -> list[str]:
    """The extraction extensions the configuration's extraction sections select."""
    config = config or load_default_preset()
    names: list[str] = []
    for section in ("document_extraction", "patient_extraction"):
        names.extend((getattr(config, section, None) or {}).get("extensions") or [])
    return list(dict.fromkeys(names))


def compile_contract(extensions: Iterable[str] = ()) -> dict:
    """What a compiled query depends on besides its requirement and population."""
    extensions = list(extensions)
    template = load_prompt_text(PROMPT_FILE)
    return {
        "language_version": QUERY_LANGUAGE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "extensions": extensions,
        "prompt_sha256": digest({
            "template": template,
            "field_guide": render_query_field_guide(extensions),
            "operators": render_operator_guide(),
            "examples": render_examples(),
        }),
        "response_schema_sha256": digest(query_response_schema(extensions)),
    }


def workup_query_cache_key(
    requirement: dict, population_context: str | None = None, *, extensions: Iterable[str] = (),
    contract: dict | None = None,
) -> str:
    """The cache key of one requirement's compiled query."""
    contract = contract or compile_contract(extensions)
    return digest({
        "requirement": requirement_text(requirement),
        "population": (population_context or "").strip(),
        **contract,
    })


def render_compile_prompt(
    requirement: dict, population_context: str | None = None, *, extensions: Iterable[str] = ()
) -> str:
    """The compile request for one requirement."""
    extensions = list(extensions)
    values = requirement_text(requirement)
    return fill(load_prompt_text(PROMPT_FILE), {
        "field_guide": render_query_field_guide(extensions),
        "operators": render_operator_guide(),
        "examples": render_examples(),
        "population": (population_context or "").strip() or "(not given)",
        "name": values["name"],
        "conditions": values["conditions"] or "(none stated)",
        "category": values["category"] or "(not stated)",
    })


def _retry(original, answer, parsed, problems, error):
    if parsed:
        listed = "\n".join(f"- {p}" for p in problems[:_MAX_PROBLEMS])
        if len(problems) > _MAX_PROBLEMS:
            listed += f"\n- and {len(problems) - _MAX_PROBLEMS} more problems"
        return original + [
            {"role": "assistant", "content": answer},
            {"role": "user", "content": (
                "The preceding assistant message is your query, which was not accepted:\n"
                f"{listed}\n"
                "Fix every problem, check the same rule throughout the query, keep what is "
                "correct, and return the complete corrected JSON query."
            )},
        ]
    return original + [{"role": "user", "content": (
        f"The preceding answer was not accepted: {error}. Return one complete JSON query "
        "object that follows the original instructions."
    )}]


def compile_workup_queries(
    requirements: Sequence[dict],
    *,
    config: MMAIConfig | None = None,
    population_context: str | None = None,
    cache: MutableMapping[str, dict] | None = None,
    extensions: Iterable[str] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    return_metadata: bool = False,
) -> list[dict] | tuple[list[dict], dict]:
    """Compile each workup requirement into a validated query.

    ``requirements`` are guideline ``diagnostic_workup`` items (``name``,
    ``conditions``, ``category``; other keys are ignored). ``population_context``
    is the guideline population's key criteria, shown to the model as context.
    Each distinct requirement is one request with ``config.workup_query_compile``;
    a cached compilation (``cache[key]`` from :func:`workup_query_cache_key`) is
    revalidated and reused without a request, and each new successful one is
    stored in ``cache``.

    Returns one entry per requirement, in order: ``requirement``, ``cache_key``,
    ``status`` (``compiled`` or ``failed``), ``query`` (None when failed),
    ``error``, ``attempts`` (finish reasons and problems; no reasoning),
    ``first_attempt_valid``, ``cached``, ``model`` and the compile ``contract``.
    No patient data is involved.
    """
    resolved = config or load_default_preset()
    if not isinstance(resolved, MMAIConfig):
        raise TypeError("config must be an MMAIConfig instance or None.")
    if population_context is not None and not isinstance(population_context, str):
        raise TypeError("population_context must be text or None.")
    texts = [requirement_text(r) for r in requirements]
    extensions = default_extensions(resolved) if extensions is None else list(extensions)
    schema = query_response_schema(extensions)
    contract = compile_contract(extensions)
    keys = [workup_query_cache_key(t, population_context, contract=contract) for t in texts]
    settings = section_settings(resolved, SECTION, "workup_query", schema)
    model = settings.runtime.get("model_name")
    progress = progress_callback or (lambda done, total: None)
    entries: dict[str, dict] = {}
    for key in dict.fromkeys(keys):
        saved = cache.get(key) if cache is not None else None
        if (
            isinstance(saved, dict)
            and saved.get("status") == "compiled"
            and saved.get("contract") == contract
            and not validate_workup_query(saved.get("query"), extensions)
        ):
            entries[key] = {**saved, "cached": True}
    pending = [(key, texts[keys.index(key)]) for key in dict.fromkeys(keys) if key not in entries]
    total, done = len(set(keys)), len(entries)
    progress(done, total)
    model_metadata: dict[str, Any] = {}
    if pending:
        queue = RequestQueue(
            get_llm_backend(resolved), settings.runtimes, chunk_size=settings.batch_size,
            chunks_in_flight=settings.batches_in_flight,
            model_metadata_cache_dir=resolved.model_metadata_cache_dir,
        )
        attempts: dict[str, list] = {}
        requirement_of = dict(pending)
        for key, text in pending:
            original = [{"role": "user", "content": render_compile_prompt(
                text, population_context, extensions=extensions)}]
            queue.add((key, 1, original), original, "main")

        def finish(key, status, query, error):
            nonlocal done
            history = attempts.get(key, [])
            entries[key] = {
                "cache_key": key,
                "requirement": requirement_of[key],
                "population_context_sha256": digest((population_context or "").strip()),
                "status": status,
                "query": query,
                "error": error,
                "attempts": history,
                "attempt_count": len(history),
                "first_attempt_valid": status == "compiled" and len(history) == 1,
                "contract": contract,
                "model": model,
                "compiled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "cached": False,
            }
            if status == "compiled" and cache is not None:
                cache[key] = {k: v for k, v in entries[key].items() if k != "cached"}
            done += 1
            progress(done, total)

        def record(item, answer, reasoning, finish_reason, provenance):
            key, attempt, original = item
            group = attempt_group(attempt, settings)
            entry = attempt_entry(attempt, group, finish_reason, None, debug=False,
                                  answer=None, reasoning=None)
            attempts.setdefault(key, []).append(entry)
            if finish_reason == "error":
                entry["problems"] = ["request failed"]
                finish(key, "failed", None, "the LLM request failed after its transport retries")
                return
            try:
                value = parse_answer(answer, finish_reason)
            except ValueError as exc:
                value, problems, error = None, [], str(exc)
            else:
                problems, error = validate_workup_query(value, extensions), None
            if error is None and not problems:
                finish(key, "compiled", canonical_query(value), None)
                return
            entry["problems"] = [error] if error else problems[:_MAX_PROBLEMS]
            if attempt < settings.total_attempts:
                queue.add((key, attempt + 1, original),
                          _retry(original, answer, value is not None, problems, error),
                          attempt_group(attempt + 1, settings))
                return
            finish(key, "failed", None, "no valid query after every attempt: "
                   + "; ".join(entry["problems"][:5]))

        logger.info("Compiling %d workup queries.", len(pending))
        model_metadata = queue.run(record)
    results = [{**entries[key], "requirement": text} for key, text in zip(keys, texts, strict=True)]
    if not return_metadata:
        return results
    metadata = {
        "contract": contract,
        "model": model,
        "structured_outputs": settings.structured,
        "requirements": len(texts),
        "distinct": total,
        "cached": sum(1 for key in set(keys) if entries[key].get("cached")),
        "compiled": sum(1 for key in set(keys) if entries[key]["status"] == "compiled"),
        "requests": sum(entries[key].get("attempt_count", 0) for key in set(keys)
                        if not entries[key].get("cached")),
        "first_attempt_valid": sum(1 for key in set(keys) if not entries[key].get("cached")
                                   and entries[key].get("first_attempt_valid")),
        "model_metadata": {k: v for k, v in model_metadata.items() if k != "responses"},
    }
    return results, metadata


__all__ = [
    "PROMPT_FILE",
    "SECTION",
    "compile_contract",
    "compile_workup_queries",
    "default_extensions",
    "render_compile_prompt",
    "requirement_text",
    "workup_query_cache_key",
]
