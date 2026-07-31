# Trial-space Contextualization

`contextualize_trial_spaces` retrieves source-grounded diagnostic and
therapeutic considerations for disease contexts represented by MatchMiner-AI
trial spaces. It accepts only:

- `space_trial_id`
- `trial_id`
- `clinical_space_summary`

Patient-bearing columns are rejected before any network request. The result
contains a Markdown synthesis table, normalized evidence records with URLs and
retrieval provenance, and run metadata including partial source failures.
Spaces with no evidence skip LLM inference instead of falling back to the
model's intrinsic clinical knowledge.

Retrieval includes disease-relevant, heading-aware NCI PDQ passages; separate
diagnostic/staging, molecular-testing, and treatment-guidance PubMed searches;
and selected guideline or consensus full-text sections from Europe PMC when
article metadata explicitly reports a CC BY or CC0 license. The evidence packer
reserves space for diagnostic material and sends up to a default 12,000 raw
evidence tokens to the configured synthesis model.

The diagnostic synthesis separates workup generally expected before the
represented disease state from testing that may be appropriate at the next
decision point. Where supported by retrieved evidence, it covers pathologic
confirmation, staging and imaging, biomarker or genomic testing, specimens and
assays, baseline assessments, and conditions for repeat or confirmatory
testing. Unsupported steps are reported as evidence gaps rather than supplied
from the model's intrinsic knowledge.

```python
from matchminer_ai.contextualization import contextualize_trial_spaces

context = contextualize_trial_spaces(trial_spaces)
print(context.contexts.iloc[0]["contextualization_markdown"])
context.evidence.to_parquet("trial_space_evidence.parquet")
```

The adapters have deliberately different meanings:

- NCI PDQ is an evidence-based health-professional summary, not a clinical
  practice guideline.
- FDA companion-diagnostic listings and DailyMed structured product labels are
  US regulatory artifacts.
- CIViC is a CC0 community-curated evidence database; only accepted evidence
  items are requested.
- PubMed contributes citation metadata and abstracts, not publisher full text.
- Europe PMC contributes section-aware full text only for disease-relevant
  guideline/consensus records with an allowlisted CC BY or CC0 license; source
  jurisdiction and publication type still require review.

Use `personalize_trial_space_context` only after retrieval when a separate
patient-specific review is wanted. That API performs no web retrieval and sends
patient text only to the configured local vLLM or OpenAI-compatible endpoint.
The review distinguishes diagnostic work explicitly documented as completed
from potentially outstanding or repeat workup and from information that is
simply absent from the supplied patient summary.

::: matchminer_ai.contextualization
    options:
      members:
        - contextualize_trial_spaces
        - personalize_trial_space_context
        - TrialSpaceContextualizationResult
        - EvidenceItem
