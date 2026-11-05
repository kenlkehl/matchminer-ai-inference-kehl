# Trials

::: matchminer_ai.trials
    options:
      members:
        - extract_trial_space_eligibility_criteria
        - extract_protocol_cohorts
        - generate_synthetic_protocols
        - summarize_trials
        - merge_equivalent_cohorts
        - remove_boilerplate_criteria
        - screen_trials
        - list_guidelines
        - summarize_guidelines
        - prepare_pdq_library
        - summarize_pdq
        - audit_guideline_catalog
        - load_guideline_catalog
        - get_guideline_considerations
        - structure_trial_space

## Three-step treatment screening and cohort extraction

The default `trial.extraction_method` is `cohorts`. `summarize_trials` accepts
one row per trial with `trial_id`, `trial_title`, `brief_summary`, and
`eligibility_criteria`; IDs must be unique and nonempty. Each listing becomes
one document, `Title: ...`, `Brief summary:` and `Eligibility criteria:` with
source formatting preserved, and goes through the same workflow and prompts as
full protocols (`extract_protocol_cohorts`, below;
`matchminer_ai.trials.cohort_workflow`). Every trial gets separate requests:

1. **Screen** (`trial.cohorts.screen.user.txt`): `{"cancer_treatment_trial": "yes"}`
   or `{"cancer_treatment_trial": "no"}`. Only direct anti-cancer treatment
   strategies qualify (including PK/PD, safety, dose or surrogate-endpoint
   studies); diagnostic, supportive-care, survivorship, primary-prevention and
   non-cancer trials do not. A "no" ends the trial.
2. **Cohort names** (`trial.cohorts.list.user.txt`):
   `{"cohorts": ["Part A, dose escalation", "Part B, dose expansion (disease X)"]}`,
   one or more distinct names in source order: the cohorts the trial defines
   (parts, phases, arms or groups that enroll participants), at the lowest level
   of any hierarchy, without comparing their criteria. Stratification factors,
   crossover, extension and rollover phases, and optional sub-studies are not
   cohorts; closed cohorts are kept, and cohorts limited to countries that
   exclude the United States are omitted. A single population is one cohort
   (for example "All participants").
3. **Criteria, one request per cohort** (`trial.cohorts.criteria.user.txt`):
   `{"eligibility_criteria": "Inclusion criteria:\n...\nExclusion criteria:\n..."}`,
   the complete criteria of the named cohort: criteria stated for every
   participant, for a part or group containing it, or for a disease it can
   include, and no criterion that applies only to other cohorts. The request
   lists every cohort name and ends with the target name.

The criteria prompt keeps enrollment criteria only and copies every criterion
that applies to the cohort verbatim, boilerplate included: consent, compliance,
samples or biopsies, measurable disease, contraception, pregnancy and
lactation, vaccine restrictions, study-staff and catch-all investigator-judgment
exclusions are never left out. The optional boilerplate step
([below](#key-and-boilerplate-criteria)) separates them afterwards. Criteria
that apply only to people already in the study (crossover, retreatment,
rollover) or to optional sub-studies are omitted. Only United States versions
of country-specific criteria are kept, references are resolved by restating
their content, applicability lead-ins and mentions of other cohorts are removed
once a criterion's placement is decided, qualifiers are never deleted, and gaps
or damaged source text are flagged with a "Source limitation:" statement.

Every prompt starts with the same preamble and document, and the criteria
prompt puts the cohort list and target name last, so a server's prefix cache
reuses a trial's document across its requests. Each answer is validated by its
own contract (`parse_screen`, `parse_cohort_list`, `parse_criteria`): exactly
the one field, lowercase "yes"/"no", distinct nonempty names, nonempty criteria
that are not placeholders, a `stop` finish. One complete outer Markdown fence
is accepted; duplicate keys, extra fields, surrounding prose and truncation
fail validation and never become negative decisions. With a remote backend and
`trial.cohorts.structured_outputs: true` (the default), each request sets a
`response_format` with its step's schema (`JSON_SCHEMAS`). Only failed
requests are retried, with the full document, the original instructions and
the validation error, up to `trial.cohorts.max_attempts`; a cohort that fails
every attempt fails its trial. Requests of one step go to the backend in
chunks of `trial.cohorts.chunk_size`, `chunks_in_flight` at once. Cohorts whose
extracted criteria are identical are combined into one row with both names.

A trial's decision (schema `trial-cohorts-stepwise-v1`) is

```json
{
  "cancer_treatment_trial": "yes",
  "multiple_cohorts": "yes",
  "cohorts": [
    {"cohort_name": "Part A", "eligibility_criteria": "Inclusion criteria: ..."},
    {"cohort_name": "Part B", "eligibility_criteria": "Inclusion criteria: ..."}
  ]
}
```

or `{"cancer_treatment_trial": "no"}`; `multiple_cohorts` is derived from the
number of cohorts. The returned DataFrame contains accepted cohorts only:

| Column | Meaning |
| --- | --- |
| `cohort_trial_id` | `<trial_id>-cohort-<zero-based number>`; distinct from legacy space IDs. |
| `trial_id` | Source trial identifier. |
| `cohort_number` | Zero-based population index in source order. |
| `cohort_name` | The cohort's name from step 2, also for a single population. Kept out of eligibility text and embeddings. |
| `cohort_eligibility_criteria` | That cohort's complete criteria from its own request. |
| `cohort_criteria_source` | `registry` (`summarize_trials`) or `protocol` (`extract_protocol_cohorts`). Older outputs used `llm` and `eligibility_criteria`. |
| `cancer_treatment_trial` | `yes` for every returned row. |
| `multiple_cohorts` | `yes` when the trial has more than one cohort. |

Rows use `representation_version="trial-treatment-cohorts-v5"`. Earlier
generations (single-request schemas `trial-registry-cohorts-v1` and
`trial-protocol-cohorts-v2`, rows v4 with `cohort_specific_eligibility_criteria`
and `common_eligibility_criteria`) remain readable with
`cohort_schema.parse_cohort_response` for audits, but are not produced. Rejected
trials produce no rows; empty and all-rejected runs keep the column schema.

```python
from matchminer_ai.trials import summarize_trials

cohorts, metadata, qc = summarize_trials(
    trials, config=config, return_metadata=True, return_qc=True
)
print(metadata["trial_decisions"])
print(metadata["trial_cohort_names"])
print(metadata["trial_errors"])
```

Decisions, rejections, errors, schema and representation versions, and the
prompt-set hash (`prompt_sha256`, with `prompt_sha256_by_step`) also appear in
`cohorts.attrs`. Returned metadata includes each trial's attempts per step (with
serving endpoints), cohort names, combined cohorts and model provenance. Debug
mode adds every step's answer and reasoning and each row's input document. QC
separates treatment rejections, exhausted inference errors and truncated
responses.

Do not rename these columns to legacy space columns or reuse saved space
embeddings as if the representations were interchangeable. `hierarchical` and
`one_shot` remain explicit legacy modes with their previous space schema; the
current Space applications opt into that mode.

## Cohort extraction from full protocol documents

`extract_protocol_cohorts` runs the same three steps and prompts on complete
protocol documents. It accepts one row per trial with a unique `trial_id` and a
`protocol_path` to a PDF or a UTF-8 text file (`.txt` or `.md`). PDFs are
converted locally with `ocr_pdf`; pass `text_output_dir` to keep the converted
text, which is otherwise written to a temporary directory and removed. Each
document is sent whole; documents longer than
`protocol_cohort_extraction.max_document_characters` fail rather than being
truncated. The prompts cover protocol-specific sources: eligibility anywhere in
the document (study design, part or module sections, appendices), body text
over abbreviated synopsis criteria, current text over superseded amendment
text, and PDF-conversion damage.

Settings come from `protocol_cohort_extraction` (`max_attempts`, `chunk_size`,
`chunks_in_flight`, `structured_outputs`, and optional `prompt_files` mapping
`screen`/`list`/`criteria` to other packaged prompt files). With `output_dir`,
each decided trial is saved as soon as it is decided to
`<output_dir>/<trial_id>/extraction.json` (decision, cohort names, combined
cohorts, every attempt per step with its serving endpoint, prompt hashes,
schema version and source-document SHA-256) with `reasoning.json`. A rerun with
the same `output_dir` reuses a saved decision only when the prompts, schema and
document bytes are unchanged, and sends the rest.

The result has the `summarize_trials` cohort columns, with
`cohort_criteria_source` set to `"protocol"`; `return_metadata=True` adds
per-trial decisions, attempts and errors, source-document hashes and page
counts, and the config snapshot. The whole document reaches the configured LLM
endpoint, so use only documents without patient data unless that endpoint is
authorized for their sensitivity. Outputs are research abstractions for human
review, not eligibility determinations.

## Merging equivalent cohorts

`merge_equivalent_cohorts` takes cohort rows from either extractor and asks the
LLM once per trial with more than one cohort (`trial.cohorts.merge.user.txt`)
which cohorts have essentially the same criteria, such as arms that differ only
in treatment or dose. The model returns only a grouping of cohort numbers; each
group becomes its first cohort's row, with the merged names joined by "; ",
renumbered `cohort_number` and `cohort_trial_id`, and `merged_from`. Criteria
are never regenerated. A trial whose answer never validates keeps its cohorts
unmerged (`merge_errors`). With `output_dir`, each grouping is saved to
`<output_dir>/<trial_id>/merge.json` and reused while the prompt and the
trial's cohorts are unchanged.

## Key and boilerplate criteria

`remove_boilerplate_criteria` separates each cohort's **key criteria**, the
criteria that define its population, from its **boilerplate criteria**, the
general requirements most cancer trials share: consent, compliance, samples and
biopsies, measurable disease, prognosis, performance status, organ function,
brain metastases, toxicities of earlier treatment, washout periods of 30 days
or less, recovery from recent major surgery, non-cancer conditions, unrelated
other cancers, simultaneous participation in another trial, contraception,
pregnancy and lactation, vaccines, and study-staff and catch-all exclusions,
each unless it is central to what the trial studies. Each cohort's criteria are
split into one item per line, and each item gets its own request
(`trial.cohorts.deboilerplate.user.txt`) asking only whether it is boilerplate.
Code sorts the unchanged items, so no criterion is lost or reworded.

The result keeps `cohort_eligibility_criteria` unchanged (all the criteria) and
adds:

| Column | Meaning |
| --- | --- |
| `cohort_key_criteria` | The criteria that are not boilerplate; all the criteria for a cohort in `deboilerplate_errors`. Read by TrialSpace retrieval and TrialChecker. |
| `cohort_boilerplate_criteria` | The boilerplate criteria; None for a cohort in `deboilerplate_errors`. Read by BoilerplateChecker. |

A cohort without items, or whose items are all classified as boilerplate, is
reported in `deboilerplate_errors`; an item whose answer never validates counts
as key and is reported in `unclassified_items`. With `output_dir`, each trial's
results are saved to `<output_dir>/<trial_id>/deboilerplate.json`: per cohort,
`key_criteria` and `boilerplate_criteria` (or `error`) and every item's attempts
with their answers and reasoning. Classifications are reused while the prompt
and the cohort's name and criteria are unchanged.

Both steps are optional at inference. They take `section="trial"` (registry
listings) or `"protocol_cohort_extraction"` (protocols) for their LLM settings;
training runs the merge first, then the boilerplate step.

## Cancer-treatment screen only

`screen_trials` runs only the workflow's first step
(`trial.cohorts.screen.user.txt`, with the same settings and response contract)
on listings (`title` or `trial_title`, `brief_summary`, `eligibility_criteria`)
or full documents (`trial_text`). It returns `trial_id`,
`cancer_treatment_trial` ("yes", "no", or None when no answer validated) and
`screen_error`; a trial whose answer never validates is reported, not rejected.
With `output_dir`, every attempt's answer and reasoning is appended to
`<output_dir>/screen.jsonl` as requests return, and saved decisions are reused
while the prompt and the trial's document are unchanged. Training draws its
trial sample from the trials this screen accepts.

## Synthetic protocols seeded by registry records

`generate_synthetic_protocols` writes one fabricated full protocol per registry
seed, for building training corpora of realistic, often messily specified
eligibility text. Seeds have a unique `trial_id` and optional `title`,
`brief_summary`, `detailed_description` and `eligibility_criteria` strings;
missing fields are marked as not provided. The record is a seed, not a
specification. The prompt pair `trial.synthetic_protocol.system.txt` and
`trial.synthetic_protocol.user.txt` asks for a prospective sponsor-style
protocol in Markdown with no synthetic-document disclaimer, and assigns each
trial one of the weighted eligibility layouts in
`data/synthetic_protocol_layouts.json` (stable per trial ID).

Requests run through the configured backend in chunks, with several chunks in
flight. Remote runs use the shared endpoint pool, so adaptive concurrency,
vLLM load feedback and `remote.server_urls_file` registries (which can give
each endpoint its own served model ID) apply. With `output_dir`, each trial's
`protocol.md`, `reasoning.txt` and `meta.json` are written as its chunk
returns, completed trials with the same prompt hash are skipped on rerun, and
the returned frame holds file paths. Empty or truncated outputs are retried.
Outputs are fabricated and must be labeled as synthetic wherever they are used.

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
