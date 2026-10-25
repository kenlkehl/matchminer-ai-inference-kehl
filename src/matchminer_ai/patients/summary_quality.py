"""Read-only checks for drafting contamination and repeated summary content.

Findings are review signals, not clinical validation or instructions to delete
text. Multiple primary cancers are supported; repeated headings alone are valid.
No patient text is included in the returned diagnostics, only character offsets.
"""

from __future__ import annotations

from collections import defaultdict
import re
import unicodedata

SUMMARY_QUALITY_VERSION = "patient-summary-quality-v1"

# Match the start of a sentence/line, not clinical uses such as "watch and wait"
# or a patient's quoted "I should ..." within a clinical sentence.
_SELF_EDIT = re.compile(
    r"(?:^|(?<=[.!?])\s+)[ \t]*(?:"
    r"I\s+(?:should|must|need\s+to|have\s+to|will)\b|"
    r"let\s+me\b|let['’]s\s+(?:think|check|review|verify)\b|"
    r"wait\s*[,!?:]|hold\s+on\s*[,!?:])",
    re.IGNORECASE | re.MULTILINE,
)
_COMMENTARY = re.compile(
    r"\b(?:the\s+(?:prompt|instructions?|user)\s+"
    r"(?:says?|asks?|requires?|requests?|wants?|specif(?:y|ies))|"
    r"(?:follow|following|according\s+to)\s+(?:the\s+)?(?:prompt|instructions)|"
    r"as\s+an?\s+(?:AI|language\s+model))\b|"
    r"^[ \t]*(?:#{1,6}\s*)?(?:\*\*)?(?:"
    r"final\s+(?:polish|construction|answer)|formatting\s+check|"
    r"one\s+last\s+(?:look|check)|final\s+check\s+on\b[^\n:]*|"
    r"(?:here\s+is|here['’]s)\s+(?:the|your|my)\s+(?:updated\s+)?summary)"
    r"(?:\*\*)?[ \t]*[:.!]|"
    r"^[ \t]*no\s+markdown\b",
    re.IGNORECASE | re.MULTILINE,
)
_REASONING = re.compile(
    r"</?think>|<\|(?:channel|im_start|im_end|start_header_id|end_header_id)\|?>|"
    r"\[/?THINK\]",
    re.IGNORECASE,
)
_HEADINGS = re.compile(
    r"^[ \t]*(?:#{1,6}[ \t]+|[-*][ \t]+)?(?:\*\*)?"
    r"(?P<field>Age|Sex|Cancer type|Histology|Current extent|Biomarkers|"
    r"Treatment history|Other relevant history|Boilerplate(?: conditions)?)"
    r"(?:\*\*)?[ \t]*:[ \t]*(?:\*\*)?[ \t]*",
    re.IGNORECASE | re.MULTILINE,
)
_UNKNOWN = {"", "none", "unknown", "not documented", "not available", "unspecified", "n/a"}
_CORE = ("age", "cancer type", "histology", "biomarkers")
_CLINICAL = ("cancer type", "histology", "current extent", "biomarkers", "treatment history")


def _normalized(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _sections(text: str) -> list[dict]:
    headings = list(_HEADINGS.finditer(text))
    sections = []
    current = None
    age = ""
    for i, match in enumerate(headings):
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        field = match["field"].lower()
        value = _normalized(text[match.end():end])
        if field == "age":
            age = value
        elif field == "cancer type":
            current = {"start": match.start(), "end": end, "fields": {"age": age, field: value}}
            sections.append(current)
        elif field in _CLINICAL and current is not None:
            current["fields"][field] = value
            current["end"] = end
        elif field.startswith(("other relevant history", "boilerplate")):
            current = None
    return sections


def check_patient_summary(text: str | None) -> dict:
    """Return versioned, JSON-compatible flags without rewriting the summary.

    ``high`` means strong evidence of malformed output, not clinical certainty.
    ``review`` includes ambiguous first-person prose and repeated core fields:
    two distinct primaries can legitimately share those field values. No flag
    alone establishes that a cancer section can safely be removed.
    """
    if text is None:
        text = ""
    if not isinstance(text, str):
        raise TypeError("Patient summary must be text or None")
    findings = []

    def add(code, confidence, start, end, message, **extra):
        findings.append(dict(code=code, confidence=confidence, start=start, end=end,
                             line=text.count("\n", 0, start) + 1, message=message, **extra))

    if not text.strip() or text.lstrip().startswith("ERROR:"):
        add("empty_or_failed", "high", 0, len(text), "Summary is empty or marked as a generation error.")
    for pattern, code, confidence, message in (
        (_SELF_EDIT, "self_editing_language", "review", "Possible first-person drafting or self-correction."),
        (_COMMENTARY, "prompt_commentary", "high", "Prompt, formatting, or answer-construction commentary."),
        (_REASONING, "reasoning_delimiter", "high", "Raw model or reasoning delimiter in the summary."),
    ):
        for match in pattern.finditer(text):
            add(code, confidence, match.start(), match.end(), message)

    sections = _sections(text)
    core_seen = {}
    full_seen = {}
    for section in sections:
        fields = section["fields"]
        full_signature = tuple(fields.get(k, "") for k in ("age", *_CLINICAL))
        # Whole sections can be copied even when biomarkers are undocumented.
        # Missing headings are different from explicitly documented unknowns.
        complete = all(fields.get(k) for k in _CLINICAL)
        if complete and full_signature in full_seen:
            previous = full_seen[full_signature]
            add("duplicate_cancer_section", "high", section["start"], section["end"],
                "All five cancer fields repeat an earlier section.",
                previous_line=text.count("\n", 0, previous["start"]) + 1,
                fields=list(("age", *_CLINICAL)))
            continue
        if complete:
            full_seen[full_signature] = section
        signature = tuple(fields.get(k, "") for k in _CORE)
        # Missing/unknown fields do not establish duplication.
        if any(v.rstrip(".") in _UNKNOWN for v in signature):
            continue
        previous = core_seen.get(signature)
        if previous is not None:
            add("repeated_cancer_fields", "review", section["start"], section["end"],
                "Age, cancer type, histology and biomarkers repeat; review for distinct primaries before changing anything.",
                previous_line=text.count("\n", 0, previous["start"]) + 1,
                fields=list(_CORE))
        else:
            core_seen[signature] = section

    # Catch repeated long prose even when it has no recognizable field headers.
    paragraphs = defaultdict(list)
    for match in re.finditer(r"\S[^\n]*(?:\n(?![ \t]*\n)[^\n]*)*", text):
        normalized = _normalized(match.group())
        if len(normalized.split()) >= 40 and not _HEADINGS.search(match.group()):
            paragraphs[normalized].append(match)
    for matches in paragraphs.values():
        for match in matches[1:]:
            add("repeated_long_paragraph", "review", match.start(), match.end(),
                "A paragraph of at least 40 words repeats verbatim after whitespace/case normalization.",
                previous_line=text.count("\n", 0, matches[0].start()) + 1)

    findings.sort(key=lambda finding: (finding["start"], finding["code"]))
    return dict(version=SUMMARY_QUALITY_VERSION, flagged=bool(findings),
                cancer_sections=len(sections), findings=findings)
