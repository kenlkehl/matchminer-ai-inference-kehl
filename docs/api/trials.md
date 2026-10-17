# Trials

::: matchminer_ai.trials
    options:
      members:
        - extract_trial_space_eligibility_criteria
        - summarize_trials
        - list_guidelines
        - summarize_guidelines
        - prepare_pdq_library
        - summarize_pdq
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

## Hierarchical extraction

The default `trial.extraction_method` is `hierarchical`. It builds a tree in this
fixed order: age, sex, cancer primary site, histology, disease state, required
prior treatment, excluded prior treatment, required biomarkers, excluded
biomarkers. One request handles one parent and one field, returning a JSON array
of alternative values rather than a full list of spaces. Independent parents
are batched through the configured backend. A trial with one branch throughout
uses ten logical calls: nine fields and one separate boilerplate call.

Every population-field request follows this order:

1. An introduction explaining the target-population hierarchy and the current
   field as the next level of granularity.
2. The complete trial source.
3. The full authored field definitions, examples, population rules and boilerplate
   boundaries from `trial.user.primer.txt`, followed by instructions for this step.
4. The current candidate's locked fields and internal `scope`.
5. The request for alternative child spaces across this field and the JSON contract.

Every call sees all field definitions, so it can distinguish tumor biomarkers
from general exclusions such as HIV/hepatitis or organ dysfunction. One-shot
numbering and output commands are omitted. Each branch adds only the current
field; it must retain shared requirements rather than turn cumulative criteria
into optional alternatives. The separate boilerplate call uses the same source
and definitions, a trial-wide scope, and its own closing response instruction.

The internal `scope` describes the applicable eligibility pathway. It preserves
cross-field dependencies and conditional exceptions; it is not included in the
final space text. Equal values with different scopes remain separate until the
remaining fields are resolved. No Cartesian product of independent field lists
is made. Python copies ancestors into each completed leaf, removes exact
duplicate leaves, renders all nine labels, and adds numbering. Public column
names and zero-based space IDs remain unchanged.

The model returns `{"branches": [{"value": "...", "scope": "..."}],
"prune_reason": ""}` for a field. Empty branches require an explicit source
incompatibility and cannot be used at the root. Boilerplate instead returns
`{"exclusions": ["..."]}` and retains subgroup qualifications. Malformed JSON,
wrong keys, multiline values and truncated answers are rejected. Retries include
the specific error and latest rejected answer. After two invalid answers,
thinking flips; Beam uses lower reasoning effort instead. Exhausted nodes fail
the trial rather than exporting only its successful branches. A trial for which
all branches are pruned is also reported as failed for review.

Configuration under `trial.hierarchy`:

| Setting | Default | Meaning |
| --- | --- | --- |
| `max_attempts` | 4 | Validation attempts per node; backend transport retries remain separate. |
| `max_nodes` | 4096 | Maximum branches at a field; exceeding it fails rather than truncates. |
| `max_calls` | 20000 | Logical node calls per trial, excluding retries. |
| `batch_size` | 256 | Maximum parent requests passed to one backend batch. |
| `checkpoint_dir` | unset | Optional directory of successful node responses. |

Node checkpoints fingerprint source, parent, scope, prompts, algorithm version,
and runtime configuration. Changed inputs/configuration never reuse old nodes.
Debug `trial_reasoning_text` contains a JSON audit of the node responses,
reasoning and validation attempts. Configured context limits reserve output
space without truncating the source. Model sampling and parser selection still
use the existing backend profiles.

The shared introduction, step instructions and closing response contracts live
in `src/matchminer_ai/prompts/trial.hierarchy.*.txt`. The full semantic rules are
loaded directly from `trial.user.primer.txt`, so edits to those definitions affect
hierarchical extraction too. The loader preserves their wording and examples;
if the primer's section boundaries change, it fails explicitly rather than
silently dropping guidance. `trial.user.question.txt` and the primer's one-shot
output format apply only to `trial.extraction_method = "one_shot"`, retained for
comparison and reproduction. The shared engine, hierarchy prompts and primer
are mirrored in training, with cross-repository contract tests.

This prompt layout is `trial-hierarchy-v2`. Prompt and version fingerprints prevent
reuse of v1 extraction checkpoints; compatible treatment screening can be retained.

This design increases the number of calls and repeats source text. It improves
structural control but does not establish clinical completeness: an erroneous
early branch or an unsupported model interpretation can still propagate. Review
the node traces and final populations against the source before adopting new
spaces or regenerating embeddings. Existing saved spaces are not migrated.

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
