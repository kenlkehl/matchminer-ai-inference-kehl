# Population workup review

!!! warning "Documentation review, not concordance"
    Results say whether a patient's notes, or the items extracted from them, document a guideline workup step.
    *Not documented* means not located, not that the step was not done. Match-quality scores rank patients; they
    are not eligibility decisions. Timing, applicability and source completeness need clinical review. Research
    use only.

`matchminer_ai.population` checks one guideline population's diagnostic workup across a caller-supplied patient
population:

1. **Cohort.** Wrap the population's summaries and source notes in a `PatientCohort`.
2. **Retrieval.** Find the patients who match one guideline space (NCCN or PDQ catalog) with TrialSpace and the
   decision model's match-quality score.
3. **Review.** Check every selected patient against every `diagnostic_workup` item of the space, by one of three
   methods:
    - **full-note review** (`review_population_workup`): an LLM reads each patient's notes;
    - **compiled queries** (`check_population_workup_with_extractions`): code runs LLM-compiled queries over
      [structured extraction](../api/extraction.md) records;
    - **SQL agent** (`check_population_workup_with_sql`): code runs SQL queries that an LLM agent wrote over the
      same records.
4. **Export and archive** the results.

Nothing here samples patients, chooses data paths or writes files except an explicit `save_patient_cohort`. The
[guideline fetcher](../apps/guideline-fetcher.md)'s population view runs the same steps, and leaves workup items
that ask only for a history and physical out of every check.

## Retrieve a cohort

```python
from matchminer_ai import load_default_preset
from matchminer_ai.matching import retrieve_guideline_patients
from matchminer_ai.population import create_patient_cohort, subset_patient_cohort
from matchminer_ai.trials import load_guideline_catalog

# summaries: patient_id, cancer_history_summary; optional general_exclusion_criteria_evidence.
# notes: patient_id, note_text; optional note_date and note_type.
population = create_patient_cohort(summaries, notes, provenance={"source": "my-cohort"})

config = load_default_preset()
config.embedding["model_path"] = "/path/to/trialspace"    # your TrialSpace checkpoint
config.remote.update(enabled=True, server_urls=["http://localhost:8000/v1"])  # for the review step

catalog = "catalogs/nsclc"                                 # a completed guideline catalog directory
space_id = load_guideline_catalog(catalog).space_trial_id.iloc[0]
selected, metadata = retrieve_guideline_patients(
    population.summaries, catalog, space_trial_id=space_id,
    candidate_k=50, min_match_score=0.2, config=config, progress_callback=print,
)
active = subset_patient_cohort(
    population, selected.patient_id.tolist(), space=metadata.pop("space"),
    matches=selected.drop(columns="cancer_history_summary").to_dict("records"),
    provenance={"retrieval": metadata},
)
```

**Inputs:**

- `create_patient_cohort` keeps `patient_id`, `cancer_history_summary` and `general_exclusion_criteria_evidence`
  from the summaries (`summarize_patients` output works as is) and drops other columns; rename training columns
  such as `pseudo_mrn` first. Patient ids become strings and must be unique. Every summary needs notes and every
  note a summary, with nonempty text. Notes are sorted stably by patient and date; undated notes stay undated and
  sort last.
- The catalog argument is a completed catalog directory, its `paradigms.jsonl`, a list of them, or a loaded
  catalog DataFrame. The lookup by `space_trial_id` is exact, and the catalog must be complete and pass its
  recorded audit.

**How retrieval works:**

- The guideline space is the query and the patient summaries are the corpus. The space is embedded and scored by
  its `cohort_key_criteria`: the cohort-style key criteria stored with the space, or for a catalog without them
  its nine fields laid out as criteria (see the [guideline extraction guide](guideline-extraction.md#key-criteria)).
- The `candidate_k` most similar patients (capped at the population size) get the decision model's
  match-quality score. Patients scoring at least `min_match_score` are selected; `None` keeps every candidate.
- Embedding and scoring run in batches of 32, with progress and cancellation checks between batches. Both sides
  must use the same embedding model. Raw notes never enter retrieval.
- `metadata` keeps the space record, catalog provenance, model metadata, every candidate's rank and scores, and
  the scored, population and selected counts.

### Reuse patient embeddings

To search one large population repeatedly, embed it once:

```python
from matchminer_ai.matching import embed_guideline_patients

vectors = embed_guideline_patients(population.summaries, config=config, batch_size=1024)
selected, metadata = retrieve_guideline_patients(
    population.summaries, catalog, space_trial_id=space_id,
    config=config, patient_embeddings=vectors,
)
```

`PatientEmbeddings` holds float32 vectors in `patient_ids` order, each summary's SHA-256 and the verified
TrialSpace encoder identity: the loaded revision or, for a local model directory, a digest of the loaded weights
and its training stage receipt, plus the tokenizer, prompts, sequence length, encoding settings and library
versions. Retrieval then embeds only the guideline space. It rejects the vectors if a requested patient is missing,
a summary changed, or the current encoder differs or cannot be verified; rebuild them in those cases. A subset of
the embedded population may be searched.

Nothing is written to disk. If you persist vectors, store them as privately as the summaries, and compare their
saved identity with `guideline_encoder_identity(config)` before loading them.

## Cohort objects

`PatientCohort` holds `summaries`, `notes`, `provenance`, the optional `compressed_notes` and `compression`
report, the selected `space`, the retrieval `matches` and the review `report`. It is immutable: every stage returns
a new cohort, so the previous stage's result survives a failure or cancellation. An empty retrieval is a valid
cohort, but compression and review need at least one patient.

## Optional note compression

```python
from matchminer_ai.population import compress_patient_cohort

active = compress_patient_cohort(active, config=compression_config)   # thinking="off" by default
```

Each note is sent on its own to the configured patient endpoint, eight at a time. The original notes stay in the
cohort, and losslessness is not verified (`compression["losslessness_verified"]` is false). No other stage
compresses implicitly, and recompressing clears an earlier review report.

## Full-note review

```python
from matchminer_ai.population import review_population_workup

active = review_population_workup(
    active, config=config, note_source="original",   # or "compressed"
    max_concurrent_patients=4, progress_callback=print,
)
print(active.report["aggregate"])
```

Each patient's selected notes go to [`review_patient_workup`](../api/patients.md#workup-documentation-review),
which reads every chunk in order and carries its assessment of every workup item forward. Nothing is retrieved or
skipped, so the configured endpoint receives every selected note and must be authorized for them. The space's
`cohort_key_criteria` (else its `clinical_space_summary`) is the population context.

- **Concurrency:** patients share a pool of `max_concurrent_patients` (default 4, from 1 to 64). Results keep the
  input patient order, and progress counts finished patients.
- **Backup endpoint:** `backup_config` and `max_consecutive_failures` (default 3) apply per patient, as in
  `review_patient_workup`.
- **Failures:** a patient whose review fails gets one `error` row per item. Its `limitations` name only the
  exception type, because provider messages can echo patient text. An error never becomes `not_documented`.
- **Cancellation** propagates; a partial matrix is never published.
- **Report:** `method: full_notes`, `version: 2`, `note_source`, `max_concurrent_patients`, `aggregate`,
  `patients` (one review per patient) and the cohort's provenance.

Statuses are those of `review_patient_workup` (`completed`, `partially_completed`, `planned`, `not_done`,
`not_documented`, `unclear`) plus `error`; applicability is `applies`, `not_applicable` or `uncertain`.

## Extraction-based checks

Both extraction methods read document and patient extraction records instead of the notes. Writing the queries
calls an LLM once per distinct workup item (and is cached); applying them to the cohort is code only:

```python
from matchminer_ai.patients import compile_workup_queries, compile_workup_sql, extract_document_records
from matchminer_ai.population import (
    check_population_workup_with_extractions, check_population_workup_with_sql,
)

workup = active.space["diagnostic_workup"]
context = active.space["cohort_key_criteria"]
query_cache, sql_cache = {}, {}                                     # any mappings; reuse them across calls
documents = extract_document_records(active.notes, config=config)   # or records from a store

compiled = compile_workup_queries(workup, config=config, population_context=context, cache=query_cache)
checked = check_population_workup_with_extractions(active, compiled, document_records=documents)

# store.duckdb: an exploration database built from the whole extraction store
accepted = compile_workup_sql(workup, database="store.duckdb", config=config,
                              population_context=context, cache=sql_cache)
checked = check_population_workup_with_sql(active, accepted, document_records=documents, config=config)
```

- **Queries:** pass one compiled entry per `diagnostic_workup` item, in order. See
  [Extraction-Based Workup Checks](../api/workup-queries.md) and [SQL-Agent Workup Checks](../api/workup-sql.md)
  for how queries are written, cached and evaluated. Compiling a query sends guideline text to the LLM; the SQL
  agent's exploration of `store.duckdb` also sends extracted values from that store, and extraction itself sends
  the notes.
- **Linking:** records are linked to the cohort's own notes by note id. The cohort's notes have no `note_id`
  column, so their ids are the extraction stages' code-derived ones (a hash of patient, date, type and text):
  extract from the same note table without a `note_id` column. Document records of other notes are ignored, and
  cohort notes without a document record count as `notes_without_records`, which turns not met into unknown. A
  patient record is used only when its `note_ids` column (or else its `note_count`) matches the patient's notes.
- **Options:** `patient_records` adds patient-level records, and `reference_dates` sets the evaluation date (one
  date, or dates by patient id; by default each patient's latest note date).
- **Report:** `method` is `extraction_queries` or `extraction_sql`, with `aggregate`, `patients`, `extraction`
  (records used and ignored, notes without records) and `queries` (each item's compiled query, or the SQL
  agent's query and how it ran on this cohort; never query results).

Statuses are `met`, `not_met`, `unknown`, `not_applicable` and, for SQL-agent steps the guideline advises against,
`done_not_recommended`, each with a `reason`.

## Aggregates and export

**Full-note aggregate** (`aggregate_population_workup`): for each item, the counts of each status (errors
included), each applicability value, and their cross-tabulation. `documented_completion_fraction` is
`completed_applicable / applicable_assessed`: completed among patients the item applies to. Errors, uncertain
applicability and not-applicable findings stay out of the denominator; a zero denominator gives `None`.

**Extraction aggregate** (`aggregate_extraction_workup`): statuses, applicability and reasons per item. The
fraction is met among applicable patients whose result is met or not met. A step the guideline advises against
has no fraction; its `done_not_recommended` count says how many patients had it anyway.

Every selected patient is counted in every item's status totals. These are descriptive documentation summaries,
not a validated concordance measure.

```python
from matchminer_ai.population import population_results_table

rows = population_results_table(active)
```

The table has one row per patient and workup item, failures included: `patient_id`, `space_trial_id`,
`recommendation_index`, `workup`, `conditions`, `status`, `applicability`, `bottom_line`,
`match_quality_score`, `note_source`, `method`, `evidence_json` and `limitations_json`.

## Archives

```python
from matchminer_ai.population import load_patient_cohort, save_patient_cohort

save_patient_cohort(active, "private/cohort.zip")
restored = load_patient_cohort("private/cohort.zip")
```

**Contents:** original notes, summaries, any compressed notes and compression report, the guideline space with its
workup items and source evidence, match scores and provenance, and the review report. Archives hold patient data;
you control their storage, transfer, retention and access.

**Saving:** the ZIP is written atomically, readable by its owner only (0600), as JSON with a SHA-256 manifest; no
pickle is used.

**Loading:**

- Archive paths are never extracted. The uncompressed size is bounded by `max_uncompressed_bytes` (default
  16 GiB).
- An archive must hold exactly `cohort.json` and `manifest.json`; other entries, such as patient indexes written by
  older releases, are rejected.
- Checksums are verified (they detect corruption, not a forged archive). Compressed notes must match their source
  notes, and the report's patients the cohort's patients.
- The aggregate is recomputed for the report's method. No model is loaded.
