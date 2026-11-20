"""EmbeddingGemma formatting around the shared clinical retrieval instruction.

The profiles are the inference package's (matchminer_ai.embedding.trialspace_contract, since 2026-11-06): the
``trialspace`` profile (production) gives patients and cohorts their own prompts, saved with the model under
the names inference uses; both carry the shared INSTRUCTION.
"""
from matchminer_ai.embedding.trialspace_contract import (  # noqa: F401  (re-exported)
    COHORT_PROMPT, PATIENT_PROMPT, PROFILES, SYMMETRIC_PREFIX, model_prompts,
)


def training_prompts(profile='sentence_similarity'):
    prompts = model_prompts(profile)
    return ({name: prompts[name] for name in (PATIENT_PROMPT, COHORT_PROMPT)}
            if profile == 'trialspace' else SYMMETRIC_PREFIX)
