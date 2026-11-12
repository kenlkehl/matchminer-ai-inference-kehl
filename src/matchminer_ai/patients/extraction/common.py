"""Settings, prompt filling, answer parsing and retry policy shared by both levels."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from matchminer_ai._qc.common import build_qc_artifact, qc_artifact_to_report_row
from matchminer_ai.config import MMAIConfig
from matchminer_ai.llm.backends import build_llm_runtime_config

from .extensions import Extension, resolve_extensions
from .schema import Problem, response_format

#: Section keys that configure the LLM; the others configure the stage.
_LLM_KEYS = ("local", "remote", "reasoning_parser", "sampling_profile", "reasoning_effort")
#: Problems listed in one retry message.
_MAX_FEEDBACK_PROBLEMS = 12


@dataclass
class SectionSettings:
    """A validated extraction section: its LLM runtimes and stage settings."""

    runtime: dict[str, Any]
    fallback_runtime: dict[str, Any] | None
    max_attempts: int
    batch_size: int
    batches_in_flight: int
    structured: bool
    extensions: tuple[Extension, ...]
    ncit_agent_codes: bool
    values: dict[str, Any]

    @property
    def runtimes(self) -> dict[str, dict[str, Any]]:
        """RequestQueue groups: the main runtime, then the thinking-off fallback."""
        groups = {"main": self.runtime}
        if self.fallback_runtime is not None:
            groups["fallback"] = self.fallback_runtime
        return groups

    @property
    def total_attempts(self) -> int:
        return self.max_attempts + (self.fallback_runtime is not None)


def _integer(values, name, default, minimum, maximum, section):
    value = values.get(name, default)
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{section}.{name} must be an integer from {minimum} to {maximum}")
    return value


def section_settings(
    config: MMAIConfig, section: str, schema_name: str, schema: dict | None
) -> SectionSettings:
    """Resolve ``config.<section>``; ``schema`` constrains remote answers when
    ``structured_outputs`` is on (None: settings only, as for a resumed run)."""
    values = dict(getattr(config, section, None) or {})
    if not values:
        raise ValueError(f"The configuration has no {section} section")
    llm = {key: deepcopy(values[key]) for key in _LLM_KEYS if key in values}
    runtime = build_llm_runtime_config(section, llm, config=config)
    structured = bool(values.get("structured_outputs", True)) and runtime["backend_mode"] == "remote"
    if structured and schema is not None:
        runtime["remote"]["request_params"]["response_format"] = response_format(schema_name, schema)
    fallback = None
    if values.get("reasoning_off_fallback", True):
        fallback = deepcopy(runtime)
        if fallback["backend_mode"] == "remote":
            template = fallback["remote"].setdefault("extra_body", {}).setdefault(
                "chat_template_kwargs", {}
            )
            template["enable_thinking"] = False
        fallback["chat_template_kwargs"] = {
            **(fallback.get("chat_template_kwargs") or {}),
            "enable_thinking": False,
        }
    return SectionSettings(
        runtime=runtime,
        fallback_runtime=fallback,
        max_attempts=_integer(values, "max_attempts", 3, 1, 10, section),
        batch_size=_integer(values, "batch_size", 256, 1, 100000, section),
        batches_in_flight=_integer(values, "batches_in_flight", 2, 1, 16, section),
        structured=structured,
        extensions=resolve_extensions(values.get("extensions") or []),
        ncit_agent_codes=bool(values.get("ncit_agent_codes", False)),
        values=values,
    )


_PLACEHOLDER = re.compile(r"\{([a-z_]+)\}")


def fill(template: str, values: dict[str, str]) -> str:
    """Fill ``{name}`` placeholders in one pass, so text inserted for one
    placeholder (a note, a record) is never itself filled."""

    def replace(match):
        name = match.group(1)
        if name not in values:
            raise KeyError(f"Prompt placeholder {{{name}}} has no value")
        return str(values[name])

    return _PLACEHOLDER.sub(replace, template)


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def parse_answer(text: str, finish_reason: str) -> dict:
    """The JSON object of a complete answer; raises ValueError otherwise.

    Only a complete outer code fence is removed; JSON is never repaired.
    """
    if finish_reason == "length":
        raise ValueError(
            "the answer was cut off at the output limit; reason more briefly and "
            "return the complete JSON object"
        )
    if finish_reason != "stop":
        raise ValueError(f"the answer did not complete (finish_reason {finish_reason!r})")
    content = (text or "").strip()
    fenced = re.fullmatch(r"```(?:json)?[ \t]*\r?\n(.*?)\r?\n[ \t]*```", content, re.DOTALL)
    if fenced:
        content = fenced.group(1)
    try:
        value = json.loads(content, object_pairs_hook=_unique_keys)
    except ValueError as exc:
        raise ValueError(f"the answer is not valid JSON ({exc})") from exc
    if not isinstance(value, dict):
        raise ValueError("the answer must be one JSON object")
    return value


_REVISE = (
    "The preceding assistant message is your answer, which was not accepted. Revise it "
    "against the source notes and the original instructions: fix every problem listed "
    "below, check the same rule throughout the answer, and keep the items that are "
    "correct. Copy each quote exactly from the source text; when the source does not "
    "support an item, remove the item. Return the complete revised JSON object."
)


def feedback_message(problems: list[Problem] | None, error: str | None = None) -> str:
    """The problems of a rejected answer, as a retry names them."""
    lines = ["The preceding answer was not accepted:"]
    if error:
        lines.append(f"- {error}")
    for problem in (problems or [])[:_MAX_FEEDBACK_PROBLEMS]:
        lines.append(f"- {problem.message}")
    if problems and len(problems) > _MAX_FEEDBACK_PROBLEMS:
        lines.append(f"- and {len(problems) - _MAX_FEEDBACK_PROBLEMS} more problems")
    return "\n".join(lines)


def retry_messages(
    original: list[dict], answer: str, parsed: bool, problems: list[Problem] | None,
    error: str | None,
) -> list[dict]:
    """The next attempt's messages: the original request, then the rejected answer
    to revise when it was complete JSON, then its problems."""
    feedback = feedback_message(problems, error)
    if parsed:
        return original + [
            {"role": "assistant", "content": answer},
            {"role": "user", "content": f"{_REVISE}\n\n{feedback}"},
        ]
    return original + [{"role": "user", "content": (
        f"{feedback}\nReturn a complete JSON object that follows the original "
        "instructions. Every evidence quote must be copied exactly from the source text."
    )}]


def attempt_group(attempt: int, settings: SectionSettings) -> str:
    """The runtime group of attempt number ``attempt`` (1-based)."""
    return "fallback" if attempt > settings.max_attempts else "main"


def attempt_entry(attempt, group, finish_reason, provenance, *, debug, answer, reasoning):
    entry = {"attempt": attempt, "thinking": group == "main", "finish_reason": finish_reason}
    if provenance is not None:
        entry["response_metadata"] = provenance
    if debug:
        entry.update(answer=answer, reasoning=reasoning)
    return entry


def qc_rows(metrics: list[tuple[str, list[str]]], denominator: int) -> list[dict]:
    return [
        qc_artifact_to_report_row(build_qc_artifact(metric=m, ids=ids, denominator=denominator))
        for m, ids in metrics
    ]


_NCIT_INDEX = None


def ncit_agent_codes(agents: list[str], resource: str) -> list[dict]:
    """NCIt concepts of drug names, from the bundled NCIt snapshot.

    An exact NCIt name or synonym wins; otherwise the index's spelling rules
    may find one concept. An unmatched agent gets nulls.
    """
    global _NCIT_INDEX
    from matchminer_ai.patients.ontology import load_ncit_drug_index, normalize_ontology_text

    if _NCIT_INDEX is None:
        _NCIT_INDEX = load_ncit_drug_index(resource)
    found = []
    for agent in agents:
        key = normalize_ontology_text(agent)
        record = next(
            (
                r
                for r in _NCIT_INDEX.search(agent, limit=8)
                if key in {normalize_ontology_text(n) for n in (r.preferred_name, *r.synonyms)}
            ),
            None,
        ) or _NCIT_INDEX.resolve_drug_name([agent])
        found.append(
            {
                "agent": agent,
                "ncit_code": record.code if record else None,
                "ncit_name": record.preferred_name if record else None,
            }
        )
    return found


__all__ = [
    "SectionSettings",
    "attempt_entry",
    "attempt_group",
    "feedback_message",
    "fill",
    "retry_messages",
    "ncit_agent_codes",
    "parse_answer",
    "qc_rows",
    "section_settings",
]
