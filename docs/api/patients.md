# Patients

::: matchminer_ai.patients
    options:
      members:
        - summarize_patients
        - structure_patient_summary

## Structured patient summary schema

`structure_patient_summary` accepts a free-text cancer-history summary and
returns a JSON-compatible dictionary. `age` and `sex` are patient-level. The
`cancers` array contains one entry per active cancer; each entry contains:

- `cancer_type` and `histology` objects grounded to OncoTree codes;
- one biomarker object per marker, with `marker`, `type`, and `result`;
- one treatment object per source treatment-history line, retaining dates and
  response;
- a `drugs` array with source name, normalized NCIt preferred name and code,
  NCIt-definition-grounded target and mechanism, and normalization status; and
- `cancer_burden`, constrained to `early_or_curative_intent` or
  `advanced_or_palliative_intent`.

The OncoTree agent sees only one hierarchy level at a time. The NCIt agent sees
only bounded search candidates and pulls full definitions only for candidate
indices it selects. A complete ontology is never placed in LLM context.
