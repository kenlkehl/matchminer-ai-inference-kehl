"""Trial-specific validation hooks for shared checkpointed structured generation."""

import json
import re

from matchminer_ai.llm.structured import (
    StructuredClient,
)
from matchminer_ai.llm.structured import (
    read_chat_stream as _read_chat_stream,
)

from ._guideline_schema import (
    DETAIL,
    EXTRACTION,
    SPACE,
    format_space,
    normalize_evidence_lists,
)
from .prompt_builder import load_prompt_text
from ._guideline_quotes import QUOTED_DETAIL


def clinical_content(value):
    """Retain every field except those citation-only repair may replace."""
    if isinstance(value, dict):
        return {
            key: clinical_content(item)
            for key, item in value.items()
            if key not in {"evidence", "page_coverage"}
        }
    if isinstance(value, list):
        return [clinical_content(item) for item in value]
    return value


class SpaceRepetitionGuard:
    """Track complete clinical definitions, not arbitrary model-generated identifiers."""

    def __init__(self):
        self.buffer = ""
        self.started = self.finished = False
        self.counts = {}
        self.repeated_since_new = 0

    def feed(self, text):
        if self.finished:
            return False
        self.buffer += text
        if not self.started:
            match = re.search(r'"spaces"\s*:\s*\[', self.buffer)
            if match is None:
                return False
            self.buffer = self.buffer[match.end() :]
            self.started = True
        if "}" not in text and "]" not in text:
            return False
        while True:
            self.buffer = self.buffer.lstrip(" \t\n\r,")
            if self.buffer.startswith("]"):
                self.finished = True
                self.buffer = ""
                return False
            try:
                value, end = json.JSONDecoder().raw_decode(self.buffer)
            except ValueError:
                return False
            self.buffer = self.buffer[end:]
            try:
                key = format_space(value).casefold()
            except (ValueError, TypeError, AttributeError):
                continue
            self.repeated_since_new = (
                self.repeated_since_new + 1 if key in self.counts else 0
            )
            self.counts[key] = self.counts.get(key, 0) + 1
            if self.repeated_since_new >= 24:
                return True


class OptionRepetitionGuard:
    """Detect repeated clinical options within one state, including repeated JSON keys."""

    def __init__(self):
        self.buffer = ""
        self.starts = []
        self.in_string = self.escaped = False
        self.counts = {}
        self.repeated_since_new = 0

    def feed(self, text):
        offset = len(self.buffer)
        self.buffer += text
        for position, char in enumerate(text, offset):
            if self.in_string:
                if self.escaped:
                    self.escaped = False
                elif char == "\\":
                    self.escaped = True
                elif char == '"':
                    self.in_string = False
                continue
            if char == '"':
                self.in_string = True
            elif char == "{":
                self.starts.append(position)
            elif char == "}" and self.starts:
                start = self.starts.pop()
                try:
                    value = json.loads(self.buffer[start : position + 1])
                except ValueError:
                    continue
                fields = ("name", "conditions", "category")
                if not all(
                    isinstance(value.get(key), str) for key in fields
                ) or not isinstance(value.get("evidence"), list):
                    continue
                key = tuple(value[field].strip().casefold() for field in fields)
                self.repeated_since_new = (
                    self.repeated_since_new + 1 if key in self.counts else 0
                )
                self.counts[key] = self.counts.get(key, 0) + 1
                if self.repeated_since_new >= 12:
                    return True
        return False


def read_chat_stream(lines, *, unique_spaces=False, unique_options=False):
    guards = []
    if unique_spaces:
        guards.append(
            (
                SpaceRepetitionGuard(),
                "Repeated 24 complete clinical definitions without any new definition",
            )
        )
    if unique_options:
        guards.append(
            (
                OptionRepetitionGuard(),
                "Repeated 12 complete clinical options without any new option",
            )
        )
    return _read_chat_stream(lines, guards=guards)


class Client(StructuredClient):
    def normalize_result(self, value):
        return normalize_evidence_lists(value)

    def preserved_content(self, value):
        return clinical_content(value)

    def stream_guards(self, schema):
        guards = []
        if (
            schema
            and schema.get("properties", {}).get("spaces", {}).get("items") == SPACE
        ):
            guards.append(
                (
                    SpaceRepetitionGuard(),
                    "Repeated 24 complete clinical definitions without any new definition",
                )
            )
        if schema in (DETAIL, QUOTED_DETAIL):
            guards.append(
                (
                    OptionRepetitionGuard(),
                    "Repeated 12 complete clinical options without any new option",
                )
            )
        return guards

    def retry_feedback(self, schema, error):
        feedback = super().retry_feedback(schema, error)
        if schema == EXTRACTION:
            feedback += " " + load_prompt_text("guideline.extraction_retry.txt").strip()
        return feedback

    def retry_feedback_history(self, schema, errors):
        if schema != EXTRACTION:
            return super().retry_feedback_history(schema, errors)
        # Whole-packet regeneration can reintroduce an earlier field error.
        # Retain a bounded set of diagnostics, including on checkpoint resume.
        recent = list(dict.fromkeys(errors))[-4:]
        return "\n\n".join(self.retry_feedback(schema, error) for error in recent)
