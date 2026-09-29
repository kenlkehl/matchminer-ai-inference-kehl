"""Small live smoke test containing only fabricated records, never patient data.

Run with an explicitly selected endpoint and its verified model ID. Output is
JSON with original reviewed excerpts and usage; this is not a clinical quality benchmark.
"""

import argparse
import json

from matchminer_ai.patients import (
    NoteSearchLLMConfig,
    NoteSearchLimits,
    answer_patient_question_batch,
)


def fabricated_patients():
    filler = "".join(
        f"Fabricated routine entry {i:04d}: routine scheduling and administrative "
        "follow-up only.\n"
        for i in range(1500)
    )
    return [
        {
            "patient_id": "synthetic-a",
            "history": (
                "Fabricated patient A.\n2025-01-02: HER2 testing ordered, not yet performed.\n"
                + filler
                + "\n2025-02-15: HER2 testing completed. Final laboratory result has not returned.\n"
            ),
            "questions": [
                "Was the HER2 test performed? Distinguish the initial order from the later update.",
                "Is the actual HER2 test laboratory result documented?",
            ],
        },
        {
            "patient_id": "synthetic-b",
            "history": (
                "Fabricated patient B.\n2025-03-02: CT chest ordered for next month.\n"
                + filler
                + "\n2025-04-09: CT chest was canceled and was never performed. No rescheduling planned.\n"
            ),
            "questions": [
                "Was the CT chest performed? Distinguish the initial order from the later update.",
                "Is a completed electrocardiogram (ECG) or its result documented?",
            ],
        },
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--context-window", type=int, default=262144)
    parser.add_argument(
        "--reasoning-effort", choices=["xhigh", "medium", "low"], default="xhigh"
    )
    args = parser.parse_args()
    patients = fabricated_patients()
    result = answer_patient_question_batch(
        patients,
        llm=NoteSearchLLMConfig(
            base_url=args.endpoint,
            model=args.model,
            context_window=args.context_window,
            reasoning_effort=args.reasoning_effort,
            timeout=300,
            max_concurrent_requests=4,
        ),
        limits=NoteSearchLimits(max_cells=4),
        max_parallel_patients=2,
        max_parallel_questions=2,
        max_active_questions=4,
    )
    result["synthetic_input_chars"] = [len(p["history"]) for p in patients]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
