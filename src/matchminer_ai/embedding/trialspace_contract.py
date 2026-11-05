"""TrialSpace retrieval text contract shared by runtime and training."""
DIRECTIONS = ("patient_to_cohort", "cohort_to_patient")
INSTRUCTION = ('Find potentially eligible patient–cohort matches, prioritizing matching cancer type, '
               'disease burden, prior treatment, and required biomarkers.')
# Prompt names a TrialSpace model trained by matchminer-ai-training saves with its
# weights: patients (the summary's key section) and cohorts (their key criteria).
PATIENT_PROMPT = "patient_summary"
COHORT_PROMPT = "this_space"
