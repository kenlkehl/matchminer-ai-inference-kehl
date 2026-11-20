# Guideline consideration retrieval

These functions match patients to the populations ("spaces") of a guideline catalog built from NCCN or NCI PDQ
guidelines (see [guideline extraction](../user-guide/guideline-extraction.md) and
[PDQ catalogs](../user-guide/pdq-catalogs.md)). Each space is identified by `space_trial_id` and described by
cohort-style key criteria (`cohort_key_criteria`, from `trials.load_guideline_catalog`). Retrieval uses the same
steps as trial matching: TrialSpace embeds the patient's `cancer_history_summary` and the space's key criteria,
`generate_candidate_matches` retrieves by cosine similarity, and `score_match_quality` ranks the candidates. No
generative LLM is called.

- **Patient-centric:** `retrieve_guideline_considerations` returns, for each patient, the `top_n` spaces with the
  highest `match_quality_score` among the `candidate_k` nearest, with their complete catalog records, including the
  diagnostic and treatment menus. `write_guideline_considerations_report` writes them to a Markdown review report.
- **Trial-centric:** `retrieve_guideline_patients` scores the `candidate_k` patients of a population nearest to one
  space and keeps those whose `match_quality_score` is at least `min_match_score` (default 0.2; `None` keeps all).
  `embed_guideline_patients` embeds a population once for repeated searches. See
  [population workup review](../user-guide/population-workup.md).

`retrieve_guideline_considerations` scores with `filter_low_quality=False`, so a low score never removes one of the
top N; `match_quality_pass` flags it. Scores are prioritization signals, not probabilities of clinical
appropriateness. The stored menus are returned unchanged and have not been adjudicated for the individual patient,
so source and clinician review remain necessary.

::: matchminer_ai.matching.guidelines
    options:
      members:
        - retrieve_guideline_considerations

::: matchminer_ai.matching.guideline_report
    options:
      members:
        - write_guideline_considerations_report

::: matchminer_ai.matching.guideline_patients
    options:
      members:
        - PatientEmbeddings
        - embed_guideline_patients
        - retrieve_guideline_patients
        - guideline_encoder_identity
