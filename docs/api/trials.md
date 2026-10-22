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
fixed order: age, cancer primary site, sex, histology, disease state, prior
treatment requirements, and biomarker requirements. One request handles one parent and one field, returning a JSON array
of alternative values rather than a full list of spaces. Independent parents
are batched through the configured backend. A trial with one branch throughout
uses eight logical calls: seven fields and one separate boilerplate call.

Every population-field request includes:

1. The hierarchy's purpose and the current field.
2. The complete trial source.
3. "For a patient with this set of characteristics:" followed by the inherited
   characteristics of exactly one candidate branch and its source-grounded scope.
4. "What sets of [age ranges / prior treatment requirements / biomarker
   requirements / etc.] are relevant for determining eligibility for this trial?"
5. Shared anti-boilerplate instructions and authored omissions.
6. Only the current field's verbatim instructions and examples from the active
   `trial.user.primer.txt`, followed by the JSON response contract.

Other fields' definitions are absent. The shared anti-boilerplate boundary uses
the primer's definition and examples of general exclusions; the omission text
retains its measurable-disease/biopsy-target and treatment-washout instructions.
The primary-site request includes the exact diagnosis-enumeration
paragraph now contained in that authored section. Biomarker requests also receive the exact treatment-defining population
example. Each branch adds only the current field and retains shared requirements.
A condition assigned to this field must survive in its value, not only its scope.
Biomarker fields include only restrictions on results: positive or negative status,
specific alterations, or expression thresholds. A requirement to test or know a
biomarker's status, while allowing either result, is omitted; use `NA` when no
result restriction remains. Screening-assessment notes accompany actual result
restrictions only. This rule is shared with one-shot extraction through the primer.
The separate single boilerplate request uses the full source, a trial-wide scope,
the authored boilerplate paragraph (including logical negation examples), and
shared omissions. It does not receive the population fields' definitions.

The internal `scope` describes the applicable eligibility pathway. It preserves
cross-field dependencies and conditional exceptions; it is not included in the
final space text. Equal values with different scopes remain separate until the
remaining fields are resolved. No Cartesian product of independent field lists
is made. Python copies ancestors into each completed leaf, removes exact
duplicate leaves, renders all seven labels, and adds numbering. Public column
names and zero-based space IDs remain unchanged.

The model returns `{"branches": [{"value": "...", "scope": "..."}],
"prune_reason": ""}` for a field. Empty branches require an explicit source
incompatibility and cannot be used at the root. Boilerplate instead returns
`{"exclusions": ["..."]}` and retains subgroup qualifications. Malformed JSON,
wrong keys, multiline values and truncated answers are rejected. Retries include
the specific error, latest rejected answer, and a valid JSON structure, including
explicit array/object closure guidance. After two invalid answers (or on the last
retry with a smaller budget), thinking stays off; Beam stays at low reasoning
effort. After repeated failures, the malformed draft is omitted and the model
regenerates from the original source plus the error and schema. Exhausted nodes still fail the trial rather than exporting
only its successful branches.

When every population branch is explicitly pruned, a separate study-scope call
reviews the full source. It must independently confirm that there is no human
therapeutic cancer population and supply a verbatim source excerpt. Confirmed
rejections produce no spaces and are recorded in `metadata.trial_rejections`,
separately from inference failures. A `review` decision triggers targeted
re-extraction of the pruned fields, with the review explanation and original
parent context. Up to two recovery rounds are allowed per trial. Recovery
feedback does not authorize inventing eligibility, changing parent fields, or
overriding the source. Corrective requests have distinct checkpoint keys and
retain their recovery round and replaced request key in traces. Completed nodes
remain reusable; replaying a previously failed trial reaches these new recovery
requests instead of permanently repeating its cached pruning decision.
Exhausted recovery, malformed reviews and unsupported rejection excerpts remain
failures. This is a targeted
review of empty trees, not a replacement screening pass. Original field prompts,
successful node identities, and the seven-field space format are unchanged.

Configuration under `trial.hierarchy`:

| Setting | Default | Meaning |
| --- | --- | --- |
| `max_attempts` | 4 | Validation attempts per node; backend transport retries remain separate. |
| `max_nodes` | 4096 | Maximum branches at a field; exceeding it fails rather than truncates. |
| `max_calls` | 20000 | Logical node calls per trial, excluding retries. |
| `batch_size` | 256 | Maximum parent requests in a synchronous local backend batch. |
| `active_trials` | 32 | Minimum rolling remote trial window; enlarged to at least total endpoint concurrency. |
| `checkpoint_dir` | unset | Optional directory of successful node responses. |

Remote extraction uses one dependency-ready queue across trials and field levels.
Each validated parent immediately releases its children, without waiting for
siblings or other trials. Boilerplate starts independently of the field tree.
A finished trial admits another immediately. Persistent clients share the
configured `remote.max_concurrent_requests` cap per endpoint across all these
requests, including retries. Only parent-child dependencies impose ordering;
calls for the same branch cannot skip ahead. Leaves retain the model's original
branch order, so response timing does not affect space IDs and cached wave-based
responses keep their existing numbering. Synchronous local vLLM
generation retains batched field waves with one resident model.

Set the remote request cap in the top-level `remote` configuration, for example
`config.remote["max_concurrent_requests"] = 32`, or in YAML:

```yaml
remote:
  max_concurrent_requests: 32
```

The default is 32 per server. With N endpoints the total ceiling is N times
this value, shared across trials, field branches, boilerplate and retries within
one extraction call. `trial.hierarchy.active_trials` controls how much work is
available; it does not multiply the request cap. Separately launched clients
each have their own cap. Set the value before starting extraction.

Node checkpoints fingerprint source, parent, scope, prompts, algorithm version,
and runtime configuration. Changed inputs/configuration never reuse old nodes.
Trial extraction defaults to a 50,000-token output ceiling (reasoning plus final
answer) and a 100,000-token local context. Each request still reduces its output
allowance when necessary to fit the configured context without truncating source.
Debug `trial_reasoning_text` contains a JSON audit of the node responses,
reasoning and validation attempts. Configured context limits reserve output
space without truncating the source. Model sampling and parser selection still
use the existing backend profiles.

The shared introduction and JSON response contracts live in
`src/matchminer_ai/prompts/trial.hierarchy.*.txt`. Each field component expands
`{{field_rules}}` with its exact section from `trial.user.primer.txt`, under
`FIELD-SPECIFIC INSTRUCTIONS (verbatim):`. The combined treatment field receives the
original treatment section, and the combined biomarker field receives the original
biomarker section. Age retains its existing instructions because the primer has
no separate age section. Changed section boundaries fail explicitly rather than
silently dropping guidance. One-shot extraction uses the same active primer;
it now emits the same seven labels. `trial.user.question.txt` is unchanged.
The shared engine, hierarchy prompts and primer are mirrored in training, with
cross-repository contract tests.

This hierarchy is `trial-hierarchy-v5`, the combined-requirements prompt contract.
Primary site is still fixed before sex; final text retains the established age,
sex, primary-site label order. Prompt and version fingerprints prevent reuse of
v1/v2/v3/v4 extraction checkpoints. Compatible treatment screening can be retained.

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

New space text uses `Cancer primary site allowed`, `Prior treatment requirements`,
and `Biomarker requirements`. Treatment and biomarker fields each retain both
positive and negative conditions. Extraction postprocessing, QC, contextualization,
and training's 0c sampling accept these seven fields as well as saved legacy
nine-field spaces; they do not rewrite saved text or mix extraction checkpoints.
TrialChecker and synthetic-patient prompts preserve clause polarity and AND/OR
logic. The structured ontology API and PDQ catalog retain their separate JSON
required/excluded fields, with explicit prompt adapters for the combined input.
Existing trained models can consume the new text, but their quality on this prompt
generation still requires evaluation; regenerate embeddings for changed spaces.

Hierarchical response parsing accepts one complete outer Markdown code fence
(`json`, case-insensitive, or no language tag). It then applies the same strict
JSON/schema, duplicate-key, and completion-status validation. Prose outside the
fence, malformed JSON, and truncated responses still fail. Raw responses and
reasoning remain unchanged in the audit. This presentation-only parser fix is
compatible with existing v5 prompts and successful node checkpoints.
