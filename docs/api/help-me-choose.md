# Help Me Choose

Help Me Choose identifies and researches experimental `DRUG` and `BIOLOGICAL`
interventions for matched ClinicalTrials.gov records, scores how well supported
each option is for one patient, then builds a patient-specific comparison. Its
APIs enforce this ordering:

1. `research_good_options` receives NCT IDs, uses registry arm metadata to
   exclude standard-of-care/control-only interventions, and constructs
   experimental-drug-only web queries.
2. `evaluate_good_options` introduces one patient's context after web research
   and uses either the four-point-per-drug LLM rubric or a configured local
   GoodOptionChecker classifier.
3. `build_comparison_messages` includes the resulting evidence score and
   remaps research citation labels into the report's per-trial namespace.
4. `generate_trial_comparison` runs the selected scorer when results were not
   precomputed, uses the configured package LLM backend for comparison, and
   appends code-generated source links.

The report ranks trials, but does not establish eligibility or recommend a
treatment. Every per-trial section discusses drug mechanism, efficacy, and
safety before patient-specific advantages, concerns, evidence gaps, and
questions for the trial team.

`fetch_trial_eligibility_criteria` is a separate ClinicalTrials.gov helper used
by consumers that have an NCT ID but not the complete eligibility text. It reads
`protocolSection.eligibilityModule.eligibilityCriteria` from the API v2 study,
preserves multiline criteria formatting, and returns the NCT ID, public study
URL, retrieval timestamp, and registry last-update date with the text. It
accepts no patient context.

`fetch_trial_registry_document` also accepts an official ClinicalTrials.gov
study URL. It downloads the API v2 study and wrangles the official/brief title,
brief summary, detailed description, and eligibility criteria into explicit
fields for `summarize_trials` consumers. URL parsing never changes the request
host: the actual API request is always sent to ClinicalTrials.gov using the
normalized NCT ID.

::: matchminer_ai.help_me_choose
    options:
      members:
        - fetch_trial_eligibility_criteria
        - fetch_trial_registry_document
        - build_comparison_messages
        - generate_trial_comparison
        - format_report
