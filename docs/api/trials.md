# Trials

::: matchminer_ai.trials
    options:
      members:
        - extract_trial_space_eligibility_criteria
        - summarize_trials
        - list_guidelines
        - summarize_guidelines
        - audit_guideline_catalog
        - load_guideline_catalog
        - get_guideline_considerations
        - structure_trial_space

`summarize_trials` preserves source-field formatting and starts newly generated
clinical-space numbers at zero. With the remote backend, exhausted requests are
reported in `trials_failed_inference` QC and excluded from the returned spaces,
while successful trials remain available. Run metadata includes the installed
package name and version.

Earlier Kehl snapshots used one-based space IDs. Keep saved spaces, embeddings,
and paradigm memberships from the same snapshot together; see
[trial-space numbering compatibility](space-paradigms.md#trial-space-numbering-compatibility)
before combining old artifacts with newly summarized trials.

## Trial-space eligibility criteria

`extract_trial_space_eligibility_criteria` accepts one clinical-space summary
and the path to a UTF-8 `.txt` file produced by OCR. It sends the complete text
and trial space to the configured local or OpenAI-compatible LLM backend and
returns one JSON-compatible object:

```python
from matchminer_ai.trials import extract_trial_space_eligibility_criteria

criteria = extract_trial_space_eligibility_criteria(
    clinical_space_summary,
    "Eligibility-Eligibility_Checklist.txt",
    config=config,
)
```

```json
{
  "trial_space": "the original clinical-space summary",
  "inclusion_criteria": ["complete relevant criterion"],
  "exclusion_criteria": ["complete relevant criterion"]
}
```

The original trial space is attached by code. Every criterion must be a
verbatim excerpt from the OCR text after conservative OCR normalization. This
collapses whitespace, joins a line-broken ordinal suffix such as `9 th`,
removes repeated page-edge boilerplate, and skips a trailing footnote only when
its marker is referenced earlier on that page. Invalid, ungrounded, duplicated,
or overlapping lists are retried and then rejected.
Universal criteria and criteria whose scope is ambiguous remain in scope, while
criteria explicitly limited to an incompatible population are omitted.

No web search is used. With a remote backend, the entire OCR document reaches
the configured endpoint. Use protocol documents without patient data unless
the endpoint is authorized for the document's sensitivity. OCR and LLM outputs
must be compared with the complete, current protocol and do not establish
eligibility.

## Structured trial-space schema

`structure_trial_space` accepts one free-text clinical-space summary and returns
a JSON-compatible dictionary containing its `trial_id` and `space_trial_id`.
The package trial summarizer defines each space as a single cancer-type and
histology combination, so this API structures one space at a time. Its output
contains:

- `age_range` and `sex_allowed`;
- OncoTree-grounded `cancer_type` and `histology` objects;
- `cancer_burden_allowed`, constrained to
  `early_or_curative_intent` and/or `advanced_or_palliative_intent`;
- `prior_treatment_required` and `prior_treatment_excluded`, preserving each
  source requirement, response requirement, and NCIt-normalized drug; and
- `biomarkers_required` and `biomarkers_excluded`, with one object per marker
  and an explicit screening-assessment flag.

The OncoTree agent sees only one hierarchy level at a time. The NCIt agent sees
only bounded search candidates and retrieves full definitions only for
candidate indices it selects. A complete ontology is never placed in LLM
context.
