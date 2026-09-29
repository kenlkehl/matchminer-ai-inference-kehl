# Help Me Choose

Help Me Choose compares already-matched trials using a caller-supplied,
validated GoodOption catalog. It performs no live web search. Its APIs enforce
this ordering:

1. An offline catalog build receives the complete run's NCT IDs and completes
   patient-free, drug-only research and synthesis.
2. `evaluate_good_options` introduces one patient's cancer-history summary and
   scores only investigational or unresolved drugs with the four-point-per-drug
   LLM rubric (the four-logit GoodOptionChecker is deprecated).
3. `build_comparison_messages` combines the patient context with the catalog's
   clean Help Me Choose drug projections. Raw passages, snippets, URLs, search
   queries, source metadata, trial registry metadata, and research failures are
   excluded from the LLM prompt.
4. `generate_trial_comparison` uses the configured package LLM backend, and
   `format_report` appends code-generated links from the catalog evidence ledger.

The report ranks trials, but does not establish eligibility or recommend a
treatment. Every per-trial section discusses drug mechanism, efficacy, and
safety before patient-specific advantages, concerns, evidence gaps, and
questions for the trial team. Missing or blocked catalog entries are shown as
unscored and never trigger a hidden live-research fallback.

`fetch_trial_eligibility_criteria` is a separate ClinicalTrials.gov helper used
by consumers that have an NCT ID but not the complete eligibility text. It reads
`protocolSection.eligibilityModule.eligibilityCriteria` from the API v2 study,
preserves multiline criteria formatting, and returns the NCT ID, public study
URL, retrieval timestamp, and registry last-update date with the text. It
accepts no patient context.

`fetch_trial_registry_document` also accepts an official ClinicalTrials.gov
study URL. It downloads the API v2 study and wrangles the official or brief
title, brief summary, detailed description, and eligibility criteria into
explicit fields for `summarize_trials` consumers. URL parsing never changes the
request host: the API request is always sent to ClinicalTrials.gov using the
normalized NCT ID.

::: matchminer_ai.help_me_choose
    options:
      members:
        - fetch_trial_eligibility_criteria
        - fetch_trial_registry_document
        - build_comparison_messages
        - generate_trial_comparison
        - format_report
