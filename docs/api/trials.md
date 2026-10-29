# Trials

::: matchminer_ai.trials
    options:
      members:
        - extract_trial_space_eligibility_criteria
        - extract_protocol_cohorts
        - summarize_trials
        - list_guidelines
        - summarize_guidelines
        - prepare_pdq_library
        - summarize_pdq
        - audit_guideline_catalog
        - load_guideline_catalog
        - get_guideline_considerations
        - structure_trial_space

## Combined treatment screening and cohort extraction

The default `trial.extraction_method` is `cohorts`. `summarize_trials` accepts
one row per trial with `trial_id`, `trial_title`, `brief_summary`, and
`eligibility_criteria`. IDs must be unique and nonempty. Source-field formatting
is preserved. Each trial uses one combined prompt, with up to three validation
attempts (`trial.cohorts.max_attempts`); only failed trials are retried, using the
same sampling and thinking settings. Existing local and remote backends are used.

The model must return exactly one of these JSON shapes, in the indicated order:

```json
{"cancer_treatment_trial": "no"}
```

```json
{"cancer_treatment_trial": "yes", "multiple_cohorts": "no"}
```

```json
{
  "cancer_treatment_trial": "yes",
  "multiple_cohorts": "yes",
  "cohorts": [
    {"cohort_name": "Arm A", "cohort_specific_eligibility_criteria": "Criteria specific to the first cohort"},
    {"cohort_name": "Arm B", "cohort_specific_eligibility_criteria": "Criteria specific to the second cohort"}
  ],
  "common_eligibility_criteria": "Inclusion and exclusion criteria shared by every population"
}
```

Values are lowercase strings, not booleans. A negative treatment decision stops
before cohort assessment. Diagnostic-only, supportive-care and primary-prevention
studies do not qualify merely because participants have cancer or receive
background cancer therapy. Direct treatment studies qualify even with only
PK/PD, safety, dose, or surrogate endpoints. The combined prompt owns these rules;
the training screen imports it rather than maintaining another screening prompt.

Arms/cohorts count separately only when eligibility differs. Equivalent groups
are combined. Each object contains exactly two nonempty strings: `cohort_name`
and `cohort_specific_eligibility_criteria`. The name retains the source arm/cohort
label (or `Cohort 1`, `Cohort 2`, etc. when unnamed); the criteria field starts
directly with eligibility requirements, without a cohort-name heading.
Each criteria field contains only that population's specific
inclusion/exclusion criteria. `common_eligibility_criteria` is one string containing
criteria that apply identically to every population, or `""` when none are
established. Criteria shared by only some populations are repeated in their
respective strings. If a rule differs by cohort, each string carries its complete
applicable version; the common field must not introduce a conflicting default.
Each cohort string together with the common field must be complete and
self-contained, preserve exceptions and AND/OR logic, and explicitly flag
incomplete source criteria without inventing missing details.

This is schema `trial-treatment-cohorts-v3`. Live responses require named objects
and the shared field for multiple populations. Historical v1/v2 string outputs
are readable only with explicit `schema_version="trial-treatment-cohorts-v1"`
or `schema_version="trial-treatment-cohorts-v2"`. They are never accepted as new
v3 responses. Different cohort names do not make duplicate criteria distinct.
Prompt hashes and schema versions distinguish the generations.

The returned DataFrame contains accepted cohorts only:

| Column | Meaning |
| --- | --- |
| `cohort_trial_id` | `<trial_id>-cohort-<zero-based number>`; distinct from legacy space IDs. |
| `trial_id` | Source trial identifier. |
| `cohort_number` | Zero-based population index in source order. |
| `cohort_name` | Source cohort/arm name or deterministic fallback from the model; null for single-population trials. Kept out of complete eligibility text and embeddings. |
| `cohort_eligibility_criteria` | Cohort-specific plus common criteria joined in code, or the original eligibility field for one population. |
| `cohort_specific_eligibility_criteria` | The model's specific criteria string; null for single-population trials. |
| `common_eligibility_criteria` | The model's shared criteria string, repeated in each row; null for single-population trials. |
| `cohort_criteria_source` | `llm` for multiple populations; `eligibility_criteria` for a single population. |
| `cancer_treatment_trial` | `yes` for every returned row. |
| `multiple_cohorts` | The accepted model decision. |

For a single eligibility population, the model deliberately emits no criteria.
Code copies the original `eligibility_criteria` field verbatim, without the
title or summary. The combined title/summary/eligibility document remains the
input to screening and cohort detection. If an accepted single-population trial
has missing or blank eligibility, it produces no row and records a `trial_errors`
entry; it is not reclassified as a treatment rejection. Derived row metadata uses
`representation_version="trial-treatment-cohorts-v4"` to distinguish the separate
name column; the LLM JSON schema is v3. Rejected trials produce no rows. Empty/all-rejected runs retain the same
column schema. General exclusions remain in either the specific or common field
according to applicability. The common field is not the legacy boilerplate
extraction: it includes all shared eligibility dimensions, not just exclusions.
Raw model JSON in metadata keeps the common field only once per trial; the
complete row text adds headings and concatenates the two strings without another
model call. When the common string is empty, the specific string is used as-is.

```python
from matchminer_ai.trials import summarize_trials

cohorts, metadata, qc = summarize_trials(
    trials, config=config, return_metadata=True, return_qc=True
)
print(metadata["trial_decisions"])
print(metadata["trial_rejections"])
print(metadata["trial_errors"])
```

Decision, rejection, error, schema-version and prompt-hash metadata also appear
in `cohorts.attrs` when optional metadata is not requested. Returned metadata
includes per-trial attempts and model/config provenance. Debug mode additionally
retains input, final answer and reasoning for accepted rows, and all final
responses/attempts in metadata. QC separates treatment rejections from exhausted
inference errors and truncated responses. Duplicate keys, extra/missing fields,
wrong order/types, empty/duplicate cohort descriptions, surrounding prose, and
non-`stop` finishes fail validation; they never become negative decisions.
One complete outer Markdown fence is accepted when it is unlabeled or labeled
`json` (case-insensitive), with only whitespace outside it. The enclosed JSON
still undergoes all schema and completion checks. Debug responses retain the
original fenced answer; accepting this presentation wrapper avoids unnecessary
model retries without changing the extracted criteria.

This changes the default schema and text semantics. Do not rename these columns
to legacy space columns or reuse saved space embeddings as if the representations
were interchangeable. Existing matching/structuring consumers require adaptation
and evaluation. `hierarchical` and `one_shot` remain explicit legacy modes with
their previous space schema; the current Space applications opt into that mode.
Saved trial-space artifacts remain unchanged.

## Cohort extraction from full protocol documents

`extract_protocol_cohorts` applies the same treatment and cohort decisions to
complete protocol documents instead of registry text. It accepts one row per
trial with a unique `trial_id` and a `protocol_path` to a PDF or UTF-8 text
file. PDFs are converted locally with `ocr_pdf`; pass `text_output_dir` to keep
the converted text, which is otherwise written to a temporary directory and
removed. Each document is sent whole with `trial.cohorts.protocol.user.txt`;
documents longer than `protocol_cohort_extraction.max_document_characters`
fail rather than being truncated.

The prompt locates eligibility anywhere in the document (study design, part or
module sections, appendices), prefers body text over abbreviated synopsis
criteria and current text over superseded amendment text, excludes crossover,
rollover and optional sub-study criteria, and resolves cross-references by
restating the referenced content. Because a protocol has no registry
eligibility field to fall back on, a single-cohort answer carries that
cohort's criteria:

```json
{"cancer_treatment_trial": "yes", "multiple_cohorts": "no", "eligibility_criteria": "Inclusion criteria: ..."}
```

Multi-cohort and negative answers have the registry shapes. This is schema
`trial-protocol-cohorts-v1`. With a remote backend and
`protocol_cohort_extraction.structured_outputs: true` (the default), requests
set an OpenAI-compatible `response_format` with `PROTOCOL_COHORT_JSON_SCHEMA`,
so vLLM structured outputs constrain the answer; vLLM applies the grammar after
the reasoning section when the server runs a reasoning parser. Local in-process
generation is not grammar-constrained and relies on validation. Every answer is
checked strictly, and only failed trials are retried with the complete document
and the validation error, up to `max_attempts`.

The result has the `summarize_trials` cohort columns, with
`cohort_criteria_source` set to `"protocol"`; `return_metadata=True` adds
per-trial decisions, attempts and errors, source-document hashes and page
counts, and the config snapshot. The whole document reaches the configured LLM
endpoint, so use only documents without patient data unless that endpoint is
authorized for their sensitivity. Outputs are research abstractions for human
review, not eligibility determinations.

## Hierarchical extraction

Set `trial.extraction_method` to `hierarchical` to use the legacy space workflow.
It builds a tree in this
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
The legacy trial-space summarizer defines each space as a single cancer-type and
histology combination, so this API structures one legacy space at a time. Full
cohort criteria from the new default summarizer are not this API's input contract. Its output
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
