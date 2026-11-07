"""TrialSpace retrieval text contract shared by runtime and training."""
DIRECTIONS = ("patient_to_cohort", "cohort_to_patient")
INSTRUCTION = ('Find potentially eligible patient–cohort matches, prioritizing matching cancer type, '
               'disease burden, prior treatment, and required biomarkers.')
# Prompt names a TrialSpace model trained by matchminer-ai-training saves with its
# weights: patients (the summary's key section) and cohorts (their key criteria).
PATIENT_PROMPT = "patient_summary"
COHORT_PROMPT = "this_space"

# EmbeddingGemma formatting around INSTRUCTION (moved from the training repository's
# trialspace_prompt_profiles.py, 2026-11-06). The ``trialspace`` profile (production) gives patients and
# cohorts their own prompts, saved with a fine-tuned model under PATIENT_PROMPT and COHORT_PROMPT and
# injected into the base model when mining with it.
SYMMETRIC_PREFIX = 'task: sentence similarity | query: '
PROFILES = ('sentence_similarity', 'trialspace')


def model_prompts(profile='sentence_similarity'):
    """The prompt-name -> prefix map of a TrialSpace prompt profile."""
    if profile not in PROFILES:
        raise ValueError('Unknown TrialSpace prompt profile')
    prefix = SYMMETRIC_PREFIX
    if profile == 'trialspace':
        prefix += INSTRUCTION + '\n\n'
    prompts = {name: prefix for name in ('SentenceSimilarity', 'query', 'document')}
    if profile == 'trialspace':
        prompts.update({PATIENT_PROMPT: prefix + 'Patient history:\n',
                        COHORT_PROMPT: prefix + 'Cohort eligibility criteria:\n'})
    return prompts
