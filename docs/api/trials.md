# Trials

`matchminer_ai.trials` turns trial documents into cohorts: the groups of people a trial enrolls, each with its
complete eligibility criteria, split into key and boilerplate criteria and tagged line by line.
[How matching works](../user-guide/patient-centric-matching.md) explains how the later steps use them. The outputs
are research abstractions for human review, not eligibility determinations.

## The trial steps

| Step | Function | Adds |
| --- | --- | --- |
| 1. Screen for cancer treatment | `extract_cohorts` (ctgov listings) or `extract_protocol_cohorts` (full protocols) | Rejected trials get no rows |
| 2. List the cohorts and extract each one's complete criteria | the same extractor | `cohort_name`, `cohort_eligibility_criteria` |
| 3. Merge cohorts whose criteria are equivalent | `merge_equivalent_cohorts` | `merged_from` |
| 4. Tag every criteria line inclusion or exclusion, and key or boilerplate | `remove_boilerplate_criteria` | `cohort_key_criteria`, `cohort_boilerplate_criteria`, `cohort_criteria_items` |

`process_trials` runs all four, in this order:

```python
import pandas as pd
from matchminer_ai import load_preset
from matchminer_ai.trials import process_trials

config = load_preset("default")

listings = pd.DataFrame([{
    "trial_id": "NCT01234567",
    "trial_title": "A Phase 2 Study of Drug X in Advanced Solid Tumors",
    "brief_summary": "This study tests drug X in two cohorts ...",
    "eligibility_criteria": "Inclusion Criteria:\n- ...\nExclusion Criteria:\n- ...",
}])
cohorts, metadata = process_trials(listings, config=config, output_dir="runs/trials", return_metadata=True)

protocols = pd.DataFrame([{"trial_id": "NCT01234567", "protocol_path": "protocols/NCT01234567.pdf"}])
protocol_cohorts = process_trials(protocols, source="protocol", config=config, output_dir="runs/protocols")
```

- **`source`:** `"ctgov"` (the default) reads ClinicalTrials.gov listings with `extract_cohorts` and uses the `trial`
  settings for every step; it needs `trial.extraction_method: cohorts`, the preset's value. `"protocol"` reads full
  protocols with `extract_protocol_cohorts` and uses the `protocol_cohort_extraction` settings for every step;
  `text_output_dir` and `progress_callback` apply only here.
- **Result:** one row per merged cohort of each accepted trial, with all the columns below. The extractor's
  decisions stay in `cohorts.attrs`.
- **Metadata:** with `return_metadata=True`, each step's metadata under `extraction`, `merge` and `boilerplate`.
  Rejected trials are in `metadata["extraction"]["trial_rejections"]`.
- **`output_dir`:** saves per-trial results and reuses them; see [saved results](#saved-results-and-reuse).

The steps are also public functions, so a caller can run some and skip others. Without step 4, TrialSpace and the
decision model's match questions read `cohort_eligibility_criteria` (all the criteria), while
`exclusion_criteria_check` and `check_eligibility_criteria` need its columns. The merge and boilerplate steps take
`section="trial"` (listings, the default) or `section="protocol_cohort_extraction"` (protocols) for their settings.

## Inputs

**ctgov listings** (`extract_cohorts`): one row per trial with a unique, nonempty `trial_id` and the text columns
`trial_title`, `brief_summary` and `eligibility_criteria` (strings or missing; each trial needs some text). Each
listing becomes one document, `Title: ...`, `Brief summary:` and `Eligibility criteria:`, with the source formatting
kept. [`ctgov_records`](#clinicaltrialsgov-records) fetches a listing by NCT ID.

**Full protocols** (`extract_protocol_cohorts`): one row per trial with a unique `trial_id` and a `protocol_path` to
a PDF or a UTF-8 `.txt` or `.md` file.

- **PDFs** are converted locally with [`ocr_pdf`](documents.md). Pass `text_output_dir` to keep the text; otherwise
  it goes to a temporary directory that is removed.
- **Length:** each request carries the whole document. A document longer than
  `protocol_cohort_extraction.max_document_characters` (800,000) fails; it is never truncated.
- **Privacy:** the whole document reaches the configured LLM endpoint. Use documents without patient data unless the
  endpoint is authorized for their sensitivity.

## Cohort extraction

Both extractors send each trial document through the same three prompts (`matchminer_ai.trials.cohort_workflow`):

1. **Screen** (`trial.cohorts.screen.user.txt`), answered `{"cancer_treatment_trial": "yes"}` or `"no"`. Yes means
   the trial evaluates a direct anti-cancer treatment in people with cancer. That includes adjuvant, maintenance and
   recurrence-prevention treatment, comparisons of treatment strategies, and studies whose only endpoints are PK, PD,
   safety, dose or other surrogate endpoints. Diagnostic, screening, supportive-care, toxicity-prevention,
   survivorship, primary-prevention and non-cancer trials are no, as are anti-cancer drugs given to people without
   cancer. A "no" ends the trial.
2. **Cohort names** (`trial.cohorts.list.user.txt`), answered `{"cohorts": ["Part A, dose escalation", ...]}`: the
   parts, phases, arms or groups the trial enrolls, at the lowest level of any nesting, in document order.
    - A group is a cohort even without its own eligibility section: regimens given to separate groups, parts the
      summary names or the numbering implies, cohorts the criteria name. Closed cohorts are kept.
    - Not cohorts: stratification factors; crossover, extension, retreatment and rollover phases; optional
      sub-studies; cohorts limited to countries that exclude the United States.
    - A name only identifies a group and never narrows it. Examples given with "such as" keep that wording or are
      left out.
3. **Criteria**, one request per cohort (`trial.cohorts.criteria.user.txt`), answered
   `{"eligibility_criteria": "Inclusion criteria:\n...\nExclusion criteria:\n..."}`: the cohort's complete
   enrollment criteria. The request lists every cohort name and ends with the target cohort.

The criteria prompt's rules:

- **Complete:** every criterion that applies to the cohort (stated for every participant, for a part or group that
  contains it, or for a disease it can include) and none that applies only to other cohorts. Boilerplate is copied
  too: consent, compliance, samples and biopsies, measurable disease, contraception, pregnancy and lactation,
  vaccines, study-staff and investigator-judgment exclusions. Step 4 separates it.
- **Verbatim:** wording, numbers, units, operators and negation are copied. The only changes: lines split by PDF
  conversion are rejoined, lead-ins that only say where a criterion applies ("For Part 2 only:") are removed, and
  references ("see Section 5.1", "criterion 9") are replaced by their content. Qualifiers that limit a criterion to
  some people are kept.
- **Defined by the document:** what defines the cohort (disease, histology, biomarker, prior therapy, age, sex) is
  stated as separate criteria, taken from the document and never from the cohort's name. Features given only as
  examples are not requirements, and a document criterion is never rewritten to fit the cohort.
- **Enrollment only:** no treatment, discontinuation or in-study rules (crossover, retreatment, rollover), no
  optional sub-studies, and only the United States version of a country-specific criterion.
- **Gaps:** missing references, conflicts and text damaged by conversion get an explicit "Source limitation:"
  statement; nothing is invented.
- **Protocols:** eligibility is searched for throughout the document (study design, part or module sections,
  appendices, tables, footnotes). The body wins over an abbreviated synopsis, and current text over superseded
  amendment text.

How requests run:

- **Validation:** each answer is checked by its own contract (`parse_screen`, `parse_cohort_list`,
  `parse_criteria`): exactly its one field, lowercase "yes" or "no", distinct nonempty names, nonempty criteria
  that are not a placeholder, and a `stop` finish. One complete outer Markdown fence is accepted. Duplicate keys,
  extra fields, surrounding prose and truncation fail validation; an invalid answer is never a rejection.
- **Structured outputs:** with a remote backend and `structured_outputs: true` (the default), each request carries
  its step's JSON schema as `response_format`. Local generation relies on validation.
- **Retries:** only a failed request is retried, with the full document, the original instructions and the
  validation error, up to `max_attempts` (default 3). A cohort whose criteria fail every attempt fails its trial.
- **Concurrency:** a remote backend keeps `chunk_size * chunks_in_flight` requests (500 × 2) in flight, each sent
  as another returns, and sends later steps first so started trials finish. A local backend gets batches of
  `chunk_size` requests of one step, `chunks_in_flight` at once.
- **Prefix cache:** every prompt starts with the same preamble and document, and the criteria prompt puts the cohort
  list and target last, so a server's prefix cache reuses a trial's document across its requests.
- **Identical criteria:** cohorts whose extracted criteria are identical (ignoring whitespace and case) become one
  row, with the names joined by "; ".

These settings are under `trial.cohorts` for listings and `protocol_cohort_extraction` for protocols, each with an
optional `prompt_files` mapping `screen`, `list` and `criteria` to other packaged prompt files. The section's other
keys set the LLM. See [configuration](../reference/configuration.md#trial).

### Cohort rows

| Column | Meaning |
| --- | --- |
| `cohort_trial_id` | `<trial_id>-cohort-<n>`, with `n` the zero-based cohort number. |
| `trial_id` | The source trial. |
| `cohort_number` | Zero-based, in document order. |
| `cohort_name` | The name from step 2, also for a single population (for example "All participants"). It identifies the cohort and is not part of its criteria or embeddings. |
| `cohort_eligibility_criteria` | The cohort's complete criteria from its own request. |
| `cohort_criteria_source` | `ctgov` or `protocol` (older ctgov outputs say `registry`). |
| `cancer_treatment_trial` | `yes` on every row. |
| `multiple_cohorts` | `yes` when the trial has more than one cohort. |
| `trial_input_text` | The document sent, with `debug_mode` (listings only). |

Rejected and failed trials get no rows; an empty or all-rejected result keeps the columns.

### Decisions, metadata and QC

A trial's decision (schema `trial-cohorts-stepwise-v1`) is `{"cancer_treatment_trial": "no"}` or

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

- **`attrs`:** the result's `attrs` always hold `extraction_method`, `schema_version`, `representation_version`
  (`trial-treatment-cohorts-v5`), `prompt_sha256`, `structured_outputs`, `trial_decisions`, `trial_rejections`
  (rejected trial ID to decision) and `trial_errors` (failed trial ID to its last validation error).
- **`return_metadata=True`:** the same keys plus `prompt_sha256_by_step`, `trial_cohort_names`,
  `trial_merged_cohorts` (cohorts combined for identical criteria), `trial_attempts` (per step, and per cohort for
  criteria: finish reason, validation error and serving endpoint), `model_metadata` and `config_snapshot`.
  `extract_protocol_cohorts` adds `source_documents` (path, SHA-256, character and page counts, text path) and
  `skipped_completed`. `debug_mode` adds `trial_responses`, every step's answer and reasoning.
- **`return_qc=True`** (`extract_cohorts` only): QC rows `trials_rejected_cancer_treatment`,
  `trials_failed_inference` and `trials_truncated_llm_response`; see [QC metrics](../reference/qc-metrics.md).

```python
from matchminer_ai.trials import extract_cohorts

cohorts, metadata, qc = extract_cohorts(listings, config=config, return_metadata=True, return_qc=True)
metadata["trial_rejections"]    # {"NCT07654321": {"cancer_treatment_trial": "no"}}
metadata["trial_errors"]        # trial ID -> last validation error
```

## Merging equivalent cohorts

`merge_equivalent_cohorts` takes cohort rows from either extractor (other columns are kept) and sends one request
per trial with more than one cohort (`trial.cohorts.merge.user.txt`), listing the cohorts by number with their
criteria.

- **Rule:** cohorts belong together when everyone who meets one cohort's criteria would meet the other's. Only the
  treatment the trial assigns (agent, dose, schedule) may differ. A required background or concurrent therapy is
  eligibility and keeps cohorts apart. Differences of wording, or a shared criterion one extraction left out, do
  not. When unsure, the model keeps cohorts apart.
- **Answer:** only a grouping, `{"groups": [[1, 3], [2]]}`, validated as a partition of the cohort numbers and
  retried on failure.
- **Result:** each group becomes its first cohort's row, with the names joined by "; ", `cohort_number` and
  `cohort_trial_id` renumbered, `multiple_cohorts` recomputed, and `merged_from` listing the original cohort numbers
  (JSON). The criteria are the first cohort's; none are regenerated.
- **Failures:** a trial whose answer never validates keeps its cohorts unmerged and is listed in `merge_errors`.
- **Metadata:** `trial_groups`, `merge_errors`, `trial_attempts` (with answers and reasoning), `skipped_completed`,
  `cohorts_in` and `cohorts_out`, with the schema version and prompt hash.

## Key and boilerplate criteria

`remove_boilerplate_criteria` separates each cohort's **key criteria**, which define its population, from its
**boilerplate criteria**, the general requirements most cancer trials share, and tags every line as an inclusion or
exclusion criterion.

Boilerplate covers consent, compliance, samples and biopsies, measurable disease, prognosis, performance status,
organ function, brain metastases, toxicities of earlier treatment, washout periods of 30 days or less, recovery from
recent major surgery, non-cancer conditions, unrelated other cancers, simultaneous participation in another trial,
legal and administrative requirements, vaccines, investigator-judgment and study-staff exclusions, and
contraception, pregnancy and lactation. Each is boilerplate unless it is central to what the trial studies, such as
brain metastasis criteria in a trial for brain metastases.

- **One request per line:** each cohort's criteria are split into items, one per nonblank line other than the
  "Inclusion criteria:" and "Exclusion criteria:" headings. Each item gets its own request
  (`trial.cohorts.deboilerplate.user.txt`) showing the cohort's numbered criteria, answered
  `{"criterion_type": "inclusion" | "exclusion", "boilerplate": true | false}`.
- **Inclusion or exclusion:** an item under a heading takes that heading's type, even when worded negatively.
  Sub-items, notes and introductory lines take the type of the criterion they belong to.
- **No rewriting:** code sorts the unchanged lines, so no criterion is lost or reworded.

The rows keep `cohort_eligibility_criteria` unchanged and gain:

| Column | Meaning |
| --- | --- |
| `cohort_key_criteria` | The lines that are not boilerplate, under their headings. Read by TrialSpace retrieval and the decision model's match questions. |
| `cohort_boilerplate_criteria` | The boilerplate lines. Read by the decision model's boilerplate question. |
| `cohort_criteria_items` | Every line with its tags, as a JSON string: a list of `{"number", "heading", "text", "criterion_type", "boilerplate"}` in the original order. Read by [`check_eligibility_criteria`](criteria-check.md). |

- **Unanswered lines:** a line whose answer never validates counts as key, has `criterion_type` null and is listed
  in `unclassified_items`.
- **Unsplittable cohorts:** a cohort without lines, or whose lines are all boilerplate, keeps all its criteria as
  `cohort_key_criteria`, has `cohort_boilerplate_criteria` None and is listed in `deboilerplate_errors`.
- **Metadata:** also `criterion_type_heading_disagreements` (lines whose type differs from their heading),
  `items`, `items_reused`, `skipped_completed` and `trial_cohorts` (every line's attempts with answers and
  reasoning).

Guideline catalog spaces have no boilerplate criteria. `load_guideline_catalog` gives each space a
`cohort_key_criteria`: the protocol-style criteria the catalog stores, or for a catalog without them, the space's
nine population fields laid out as criteria. See the
[guideline extraction guide](../user-guide/guideline-extraction.md#key-criteria).

## Saved results and reuse

With `output_dir`, each step writes its per-trial results atomically as each trial finishes, and a rerun with the
same directory sends only the requests whose inputs changed.

| File | Written by | Contents | Reused while unchanged |
| --- | --- | --- | --- |
| `<trial_id>/extraction.json`, `<trial_id>/reasoning.json` | `extract_protocol_cohorts` | The decision, cohort names, combined cohorts, every attempt per step with its serving endpoint, prompt hashes, schema and representation versions and source-document SHA-256; each step's reasoning | Prompts, schema and document bytes |
| `<trial_id>/merge.json` | `merge_equivalent_cohorts` | The grouping, cohort names and attempts | Prompt and the trial's cohort names and criteria |
| `<trial_id>/deboilerplate.json` | `remove_boilerplate_criteria` | Per cohort, `key_criteria` and `boilerplate_criteria` (or `error`) and every line's tags and attempts; written once all the trial's lines are answered | Per line: prompt, schema and its cohort's name and criteria |
| `screen.jsonl` | `screen_trials` | One line per trial: decision and attempts | Prompt and the trial's document |

`extract_cohorts` saves nothing, so `process_trials(source="ctgov")` extracts listings again on every run and reuses
merge and boilerplate results only where the extracted cohorts come out the same.

`matchminer_ai.trials.cohort_run.read_trial_results(trial_dir)` reads one trial's files during or after a run and
reports its state (`complete`, `stale`, `not_treatment`, `extracting` or `pending`) and cohorts; the dashboard uses
it.

## Cancer-treatment screen only

`screen_trials` runs only the screen, with the same prompt, settings (`section`) and response contract. Rows hold
`trial_id` and either `trial_text` or a listing's `title` (or `trial_title`), `brief_summary` and
`eligibility_criteria`. It returns `trial_id`, `cancer_treatment_trial` ("yes", "no", or None when no answer
validated) and `screen_error`; a trial whose answer never validates is reported, not rejected. Training draws its
trial sample from the trials this screen accepts.

## ClinicalTrials.gov records

`matchminer_ai.trials.ctgov_records` fetches one study from the public ClinicalTrials.gov API v2. Requests carry
only the NCT ID, never patient context.

- **`fetch_trial_ctgov_document`** takes an NCT ID or an official study URL and returns the official (or brief)
  title, brief summary, detailed description and eligibility criteria, with the study URL, retrieval time and last
  update date. The request always goes to ClinicalTrials.gov with the normalized NCT ID, whatever host a URL names.
- **`fetch_trial_eligibility_criteria`** returns only the eligibility criteria, with line breaks kept, and the
  same provenance.
- **`fetch_trial_study`** (async) returns the raw study; the GoodOption catalog uses it.

```python
from matchminer_ai.trials.ctgov_records import fetch_trial_ctgov_document

document = fetch_trial_ctgov_document("NCT01234567")
listings = pd.DataFrame([{"trial_id": document.nct_id, "trial_title": document.trial_title,
                          "brief_summary": document.brief_summary,
                          "eligibility_criteria": document.eligibility_criteria}])
```

## Criteria for one population from an OCR text file

`extract_trial_space_eligibility_criteria(trial_space, eligibility_checklist_path, config=config)` takes a
description of one trial population (a "trial space", such as a cohort's name and key criteria) and a UTF-8 text
file produced by OCR, such as an eligibility checklist. It sends both to the LLM configured under `trial_space_criteria_extraction` and returns the
criteria that apply to that population:

```json
{
  "trial_space": "the population description, as given",
  "inclusion_criteria": ["complete relevant criterion"],
  "exclusion_criteria": ["complete relevant criterion"]
}
```

- **Grounding:** every criterion must occur verbatim in the OCR text after conservative normalization (whitespace
  collapsed, a split ordinal such as `9 th` joined, repeated page-edge text and referenced page-end footnotes
  removed). Invalid, ungrounded, duplicated or overlapping lists are retried and then rejected.
- **Scope:** universal criteria and criteria of ambiguous scope are kept; criteria explicitly limited to an
  incompatible population are left out.
- **Privacy:** no web search is used; the whole document reaches the configured endpoint. Compare the output with
  the complete, current protocol.

## Synthetic protocols

`generate_synthetic_protocols` writes one fabricated full protocol per ctgov seed (`trial_id` and optional `title`,
`brief_summary`, `detailed_description` and `eligibility_criteria`), for training corpora with realistic, often
messily specified eligibility text. Each trial gets one of the weighted eligibility layouts in the packaged
`synthetic_protocol_layouts.json`. With `output_dir`, each trial's `protocol.md`, `reasoning.txt` and `meta.json`
are written as it returns and completed trials are skipped on rerun. Outputs are fabricated and must be labeled
synthetic wherever they are used. See [synthetic protocols](../training/synthetic-protocols.md).

## Legacy space extraction

`trial.extraction_method: hierarchical` or `one_shot` makes `extract_cohorts` produce the older trial-space table
instead of cohorts: `space_trial_id`, `trial_id`, `clinical_space_number`, `clinical_space_summary` and
`general_exclusion_criteria` (settings `trial.hierarchy` and `trial.prompt_files`). Space text is not
interchangeable with cohort criteria, and their embeddings must not be compared. `process_trials` and the merge and
boilerplate steps need cohorts.

## Guideline and PDQ catalogs

The guideline functions below build and read NCCN and NCI PDQ catalogs, whose spaces are matched like cohorts. They
are described in [guideline extraction](../user-guide/guideline-extraction.md) and
[NCI PDQ catalogs](../user-guide/pdq-catalogs.md).

## API reference

::: matchminer_ai.trials
    options:
      members:
        - process_trials
        - extract_cohorts
        - extract_protocol_cohorts
        - merge_equivalent_cohorts
        - remove_boilerplate_criteria
        - screen_trials
        - extract_trial_space_eligibility_criteria
        - generate_synthetic_protocols
        - list_guidelines
        - summarize_guidelines
        - add_guideline_cohort_criteria
        - review_guideline_citations
        - audit_guideline_catalog
        - load_guideline_catalog
        - get_guideline_considerations
        - prepare_pdq_library
        - summarize_pdq
        - summarize_pdq_collection

::: matchminer_ai.trials.ctgov_records
    options:
      members:
        - fetch_trial_ctgov_document
        - fetch_trial_eligibility_criteria
        - normalize_nct_reference

::: matchminer_ai.trials.cohort_run
    options:
      members:
        - read_trial_results
        - read_run_cohorts
