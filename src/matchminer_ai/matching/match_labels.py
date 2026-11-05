"""The five match labels of a patient–cohort pair, asked in one prompt.

``match_quality.labels.system.txt`` asks one LLM request for five yes/no answers about one
patient's key summary section and one cohort's key criteria:

- ``reasonableness``: no clear evidence that the patient fails any of the key criteria;
- ``cancer_type``, ``burden``, ``treatment_history``, ``biomarker``: whether the cohort
  specifically requires that dimension and the patient documentedly satisfies it.

Every answer carries an ``evidence_status`` and a ``reason``. The training teacher labels pairs
with this contract, the multi-head TrialChecker learns one head per label, and
``score_match_quality_with_llm`` uses the same prompt. :func:`match_score` combines the answers
into the 0–6 match score: 0 unless reasonable, else 1 plus 1 each for cancer type, burden and
treatment history and 2 for biomarker.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from matchminer_ai.llm.prompts import load_prompt_text

MATCH_LABELS_PROMPT = "match_quality.labels.system.txt"
MATCH_LABELS_SCHEMA_VERSION = "trial-match-labels-v1"
LABELS = ("reasonableness", "cancer_type", "burden", "treatment_history", "biomarker")
EVIDENCE_STATUSES = ("supported", "contradicted", "insufficient_information", "not_applicable")
POINTS = {"reasonableness": 1, "cancer_type": 1, "burden": 1, "treatment_history": 1, "biomarker": 2}
MAX_SCORE = sum(POINTS.values())
_ANSWER = {"type": "object", "additionalProperties": False, "required": ["answer", "evidence_status", "reason"],
           "properties": {"answer": {"enum": ["yes", "no"]}, "evidence_status": {"enum": list(EVIDENCE_STATUSES)},
                          "reason": {"type": "string"}}}
MATCH_LABELS_JSON_SCHEMA = {"type": "object", "additionalProperties": False, "required": list(LABELS),
                            "properties": {label: _ANSWER for label in LABELS}}


def match_labels_prompt() -> str:
    return load_prompt_text(MATCH_LABELS_PROMPT)


def match_labels_contract() -> dict[str, Any]:
    """The identity of the contract: labels, points, schema and prompt digest."""
    prompt = match_labels_prompt()
    return {"schema_version": MATCH_LABELS_SCHEMA_VERSION, "labels": list(LABELS), "points": POINTS,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}


def build_match_label_messages(patient_key_summary: str, cohort_key_criteria: str,
                               prompt: str | None = None) -> list[dict[str, str]]:
    """System prompt, then the patient's key section and the cohort's key criteria."""
    return [{"role": "system", "content": prompt if prompt is not None else match_labels_prompt()},
            {"role": "user", "content": f"PATIENT HISTORY\n{patient_key_summary}\n\n"
                                        f"COHORT KEY ELIGIBILITY CRITERIA\n{cohort_key_criteria}"}]


def parse_match_labels(text: str, finish_reason: str = "stop") -> dict[str, dict[str, str]]:
    """The five answers; any deviation from the contract raises ValueError."""
    from matchminer_ai.trials.cohort_schema import _unique_keys, unwrap_json_fence
    if finish_reason != "stop":
        raise ValueError(f"Incomplete response: {finish_reason}")
    try:
        result = json.loads(unwrap_json_fence(text.strip()), object_pairs_hook=_unique_keys)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Response must be valid JSON: {exc}") from exc
    if not isinstance(result, dict) or set(result) != set(LABELS):
        raise ValueError(f"Response must be a JSON object with exactly the fields {list(LABELS)}")
    for label in LABELS:
        value = result[label]
        if not isinstance(value, dict) or set(value) != {"answer", "evidence_status", "reason"}:
            raise ValueError(f"{label} must have exactly answer, evidence_status and reason")
        if value["answer"] not in ("yes", "no"):
            raise ValueError(f"{label}.answer must be yes or no")
        if value["evidence_status"] not in EVIDENCE_STATUSES:
            raise ValueError(f"{label}.evidence_status must be one of {list(EVIDENCE_STATUSES)}")
        if not isinstance(value["reason"], str) or not value["reason"].strip():
            raise ValueError(f"{label}.reason must be a nonempty explanation")
    return {label: result[label] for label in LABELS}


def match_score(answers: dict[str, Any]) -> int:
    """0–6: 0 unless reasonable, else the points of every yes answer.

    ``answers`` maps each label to "yes"/"no", a bool, or a parsed answer object.
    """
    def yes(label):
        value = answers[label]
        value = value["answer"] if isinstance(value, dict) else value
        return value is True or value == "yes"
    if not yes("reasonableness"):
        return 0
    return sum(points for label, points in POINTS.items() if yes(label))


__all__ = ["LABELS", "MATCH_LABELS_JSON_SCHEMA", "MATCH_LABELS_PROMPT", "MATCH_LABELS_SCHEMA_VERSION", "MAX_SCORE",
           "POINTS", "build_match_label_messages", "match_labels_contract", "match_labels_prompt", "match_score",
           "parse_match_labels"]
