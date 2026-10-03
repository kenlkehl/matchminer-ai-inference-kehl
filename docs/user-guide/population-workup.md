# Population workup documentation review

This opt-in workflow composes existing model stages across a caller-supplied
population. It does not change serial patient summarization, prompt templates,
model defaults, or NCCN extraction. All sample selection and data-path choices
belong to the caller. No clinical data is written except by explicit save calls.

```python
from matchminer_ai import load_default_preset
from matchminer_ai.matching import retrieve_guideline_patients
from matchminer_ai.patients import ColBERTConfig
from matchminer_ai.population import (
    create_patient_cohort, subset_patient_cohort, compress_patient_cohort,
    encode_patient_cohort, review_population_workup_with_colbert,
    population_results_table, save_patient_cohort, load_patient_cohort,
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
retrieval = ColBERTConfig(chunk_size=64)
active = encode_patient_cohort(active, config=retrieval, note_source="original",
                               progress_callback=print)
active = review_population_workup_with_colbert(
    active, config=config, retrieval_config=retrieval, top_n=20,
    max_concurrent_requests=8,
    progress_callback=print,
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

`PatientCohort` stages return new cohort objects, leaving the prior completed
stage available on failure or cancellation. IDs are normalized to strings and
must be unique at the summary level. Notes are stably sorted by patient/date;
undated notes remain undated. The backend accepts public runtime columns only;
training vocabulary must be adapted by the caller. Empty retrieval selections
are supported, but compression, indexing and review require a nonempty cohort.

Compression sends each note to its configured endpoint independently with bounded
concurrency. ColBERT encoding batches across patients while preserving separate
indexes. Review validates every index against that patient's selected source
notes and requested encoding settings before making any assessment calls.
Changing note source, model settings or chunking requires re-encoding; changing
top N or the answering endpoint does not. No stage implicitly starts compression
or encoding. Compression preserves originals but does not certify losslessness.

Patient–question pairs share one bounded concurrent queue (default eight,
`max_concurrent_requests` from 1 to 32). The same limit reaches the HTTP dispatcher;
there are no nested per-patient pools. Local retrieval prepares each patient's
questions while LLM requests run concurrently. Each request contains one question
and only that patient's excerpts; results retain input patient/question order even
when responses finish out of order. Set the limit to 1 for sequential dispatch.
The configured endpoint, sampling, pacing and retries remain in effect. An explicit
backup switches failed and pending requests for the run; primary requests already
in flight may finish and retain their actual model provenance. Retrieval failures
produce `error` for that patient's items; answering failures affect their own items,
never becoming `not_documented`. Cancellation reaches all workers rather than
publishing a partial matrix. Progress reports completed checks, completed patients,
requests in flight, the concurrency limit and patients retrieved. Errors do not
contain provider/patient exception text.

`aggregate_population_workup` preserves seven documentation categories (including
error), three applicability categories, and their cross-tabulation for each item.
`completed_applicable / applicable_assessed` excludes errors, uncertain
applicability and not-applicable findings. A zero denominator has fraction `None`.
Every active patient remains represented in status totals. These are descriptive
documentation summaries, not a validated clinical concordance measure. Missing
documentation means not located in retrieved excerpts; timing, source completeness
and actual clinical applicability need human review.

Archives contain original notes, summaries, any compressed text, exact guideline
menus and source evidence, match scores/provenance, review results, and each
patient's NPZ token vectors. Saves are atomic and owner-readable; index filenames
never derive from patient IDs. Loads disable pickle and validate checksums,
patient membership, note fingerprints, chunk provenance and expected source.
ZIP contents are read without extracting archive paths. Outer and nested sizes
are bounded (`max_uncompressed_bytes`, default 16 GiB). Checksums detect corruption,
not malicious source claims. The actual encoder signature is checked by retrieval.
Callers control storage, transfer, retention, access and source-content permissions.
No models are loaded simply to restore an archive.
