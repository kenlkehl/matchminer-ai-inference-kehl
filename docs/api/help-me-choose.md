# Help Me Choose

Help Me Choose researches structured `DRUG` and `BIOLOGICAL` interventions for
matched ClinicalTrials.gov records, then builds a patient-specific comparison.
Its APIs enforce this ordering:

1. `research_trials` receives NCT IDs and constructs drug-name-only web queries.
2. `build_comparison_messages` introduces patient context after web research.
3. `generate_trial_comparison` uses the configured package LLM backend and
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

::: matchminer_ai.help_me_choose
    options:
      members:
        - fetch_trial_eligibility_criteria
        - research_trials
        - build_comparison_messages
        - generate_trial_comparison
        - format_report
