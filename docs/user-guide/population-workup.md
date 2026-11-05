# Population workup documentation review

This opt-in workflow composes existing model stages across a caller-supplied
population. It does not change serial patient summarization, prompt templates,
model defaults, or NCCN extraction. All sample selection and data-path choices
belong to the caller. No clinical data is written except by explicit save calls.

```python
from matchminer_ai import load_default_preset
from matchminer_ai.matching import retrieve_guideline_patients
from matchminer_ai.population import (
    create_patient_cohort, subset_patient_cohort, compress_patient_cohort,
    review_population_workup, population_results_table,
    save_patient_cohort, load_patient_cohort,
)

# summaries: patient_id, cancer_history_summary; optional exclusion evidence.
# notes: patient_id, note_text, optional note_date and note_type.
# Use caller-supplied, authorized DataFrames. Every patient needs source notes.
population = create_patient_cohort(summaries, notes, provenance={"source": "my-cohort"})
config = load_default_preset()
# Configure config.patient/config.remote for your authorized LLM endpoint.
selected, metadata = retrieve_guideline_patients(
    population.summaries, completed_catalog_path, space_trial_id=selected_space_id,
    candidate_k=50, min_match_score=0.2, config=config, progress_callback=print,
)
active = subset_patient_cohort(
    population, selected.patient_id.tolist(), space=metadata.pop("space"),
    matches=selected.drop(columns="cancer_history_summary").to_dict("records"),
    provenance={"retrieval": metadata},
)
# Optional, independent remote operation; originals are preserved:
# active = compress_patient_cohort(active, config=compression_config)
active = review_population_workup(
    active, config=config, note_source="original",  # or "compressed"
    max_concurrent_patients=4, progress_callback=print,
)
print(active.report["aggregate"])
rows = population_results_table(active)  # one row per patient × workup item
save_patient_cohort(active, "/authorized/private/cohort.zip")
restored = load_patient_cohort("/authorized/private/cohort.zip")
```

`candidate_k` controls TrialSpace candidates that undergo TrialChecker, capped
at population size. Embedding and scoring use batches of 32 records, with
progress and cancellation checks between batches. `min_match_score=None` retains all candidates. The trial space
is the query and patient summaries are the corpus. Both sides use the same loaded
embedding model metadata; raw notes never enter these matching stages. A catalog
lookup validates completion, source hashes and exact space identity. The result
retains candidate ranks, scores, selection counts and source provenance. Scores
are ranking signals, not probabilities or eligibility determinations.

To search one large population repeatedly, embed it once and reuse the vectors:

```python
from matchminer_ai.matching import embed_guideline_patients

vectors = embed_guideline_patients(population.summaries, config=config, batch_size=1024)
selected, metadata = retrieve_guideline_patients(
    population.summaries, completed_catalog_path, space_trial_id=selected_space_id,
    config=config, patient_embeddings=vectors,
)
```

`PatientEmbeddings` holds float32 vectors in `patient_ids` order, each summary's
SHA-256 and the verified TrialSpace encoder identity (loaded revision, tokenizer,
configured and model-saved prompts, sequence length and library versions). Retrieval embeds only the guideline
space and rejects the vectors if any requested patient is missing, any summary text
changed, or the current encoder identity differs or cannot be verified; rebuild them
in those cases. A subset of the embedded population may be searched. Nothing is
written to disk: callers that persist vectors must store them as privately as the
summaries they derive from.

`PatientCohort` stages return new cohort objects, leaving the prior completed
stage available on failure or cancellation. IDs are normalized to strings and
must be unique at the summary level. Notes are stably sorted by patient/date;
undated notes remain undated. The backend accepts public runtime columns only;
training vocabulary must be adapted by the caller. Empty retrieval selections
are supported, but compression and review require a nonempty cohort.

Compression sends each note to its configured endpoint independently with bounded
concurrency. No stage implicitly starts compression. Compression preserves
originals but does not certify losslessness; recompression clears a prior report.

`review_population_workup` reviews every selected patient × workup item over each
patient's **full** selected notes (`note_source="original"` or `"compressed"`):
each patient's notes go to `patients.review_patient_workup`, which visits every
chunk serially, carrying forward its assessment of every item (see the
[patient API guide](../api/patients.md#workup-documentation-review)). Nothing is
retrieved or skipped, so the configured endpoint receives each patient's complete
selected notes and must be authorized for them. Patients share a bounded pool (`max_concurrent_patients`,
default 4, from 1 to 64); results keep input patient order. The configured
endpoint, sampling, pacing and retries remain in effect, and an explicit
`backup_config` with `max_consecutive_failures` (default 3) applies per patient
as in `review_patient_workup`. A patient whose review fails keeps one `error` row
per item, whose `limitations` name only the exception type (provider messages can echo patient
text, so they are not kept); errors never become
`not_documented`. Cancellation propagates rather than publishing a partial
matrix. Progress reports completed patients. The report records
`method="full_notes"`, `version=2`, the note source and the concurrency limit.

`aggregate_population_workup` preserves seven documentation categories (including
error), three applicability categories, and their cross-tabulation for each item.
`completed_applicable / applicable_assessed` excludes errors, uncertain
applicability and not-applicable findings. A zero denominator has fraction `None`.
Every active patient remains represented in status totals. These are descriptive
documentation summaries, not a validated clinical concordance measure. Missing
documentation means not located in the reviewed notes; timing, source completeness
and actual clinical applicability need human review.

Archives contain original notes, summaries, any compressed text, exact guideline
menus and source evidence, match scores/provenance and review results. Saves are
atomic and owner-readable; no pickle is used. Loads validate checksums, compressed
notes against their source table and review membership, and recompute aggregates.
ZIP contents are read without extracting archive paths, and sizes are bounded
(`max_uncompressed_bytes`, default 16 GiB). Checksums detect corruption, not
malicious source claims. Archives written with ColBERT patient indexes by earlier
releases are rejected. Callers control storage, transfer, retention, access and
source-content permissions. No models are loaded simply to restore an archive.
