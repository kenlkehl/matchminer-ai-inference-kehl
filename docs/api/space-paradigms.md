# Space-paradigm roll-up

::: matchminer_ai.paradigms.ranking
    options:
      members:
        - PatientParadigmRankingResult
        - rank_patient_space_paradigms

`rank_patient_space_paradigms` is an optional patient-centric layer after the
ordinary trial-space matching stages. It:

1. retrieves a configurable TrialSpace pool (100 spaces per patient by default);
2. scores that pool with TrialChecker;
3. retains up to 10 leading TrialChecker-passing spaces by default;
4. joins their exact `space_trial_id` values to a caller-supplied membership
   graph; and
5. collapses repeated support for the same paradigm while preserving
   one-to-many space memberships.

```python
from matchminer_ai.paradigms import rank_patient_space_paradigms

result = rank_patient_space_paradigms(
    patient_summaries,
    trial_spaces,
    space_paradigm_memberships,
    paradigm_catalog,
    retrieval_k=100,
    top_space_count=10,
)

print(result.space_matches)
print(result.paradigm_matches)
```

The package does not bundle or assert a canonical paradigm catalog. Callers
must supply:

- one searchable row per `space_trial_id` in `trial_spaces`;
- exact `space_trial_id` to `paradigm_id` edges in
  `space_paradigm_memberships`; and
- one row per `paradigm_id`, including its one-line `paradigm_label`, in
  `paradigm_catalog`.

The paradigm rank follows the best supporting TrialChecker-reranked space, then
the number of retained spaces supporting that paradigm. It is deliberately not
reported as a new calibrated paradigm probability. Keep `document_status` or
equivalent caveat metadata in the catalog so downstream consumers do not treat
incomplete paradigms as fully reviewed.

The default can return fewer than ten spaces, including zero, rather than force
a patient into a low-quality paradigm match. Set
`require_match_quality_pass=False` only when below-cutoff results are useful for
research diagnostics; the returned `match_quality_pass` column remains visible.

Trial embeddings may be cached, but `trial_embeddings` and the metadata returned
when they were created must be supplied together. Reuse fails closed when the
TrialSpace model name, immutable model revision, prompt configuration, or maximum
sequence length differs from the patient embedding run. When the TrialSpace
generation changes, regenerate the entire trial embedding corpus.

These outputs are research prioritization signals. They do not establish a
diagnosis, trial eligibility, or a treatment recommendation.

## Trial-space numbering compatibility

After the upstream integration, `summarize_trials` numbers new spaces from zero
(`NCT12345678-0`, `NCT12345678-1`, ...). Earlier Kehl runs numbered them from one.
The ranking API treats supplied IDs as opaque keys: existing one-based spaces,
embeddings, and membership edges can still be used together without renumbering.

Keep the original trial-space snapshot with its embeddings and membership graph.
Do not join newly summarized zero-based spaces to an older one-based graph:
overlapping IDs can silently refer to different spaces, even when every join
key exists. ID validation alone cannot detect this mismatch.

To migrate an unchanged snapshot, build an explicit old-to-new ID mapping and
apply it together to the space table, embedding keys, and every membership edge,
preserving one-to-many memberships. Verify the trial ID and exact space text
before reusing a vector. If trials are summarized again, establish correspondence
from their content and review memberships again; subtracting one from a suffix
does not establish that regenerated spaces have the same meaning or order.
This integration does not rewrite existing catalogs or membership files.
