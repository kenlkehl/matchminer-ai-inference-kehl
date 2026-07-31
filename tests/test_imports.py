import matchminer_ai
from matchminer_ai import MMAIPipeline, load_config
from matchminer_ai.contextualization import (
    contextualize_trial_spaces,
    personalize_trial_space_context,
)
from matchminer_ai.embedding import embed_for_matching
from matchminer_ai.matching import (
    exclusion_criteria_check,
    generate_candidate_matches,
    score_match_quality,
)
from matchminer_ai.patients import (
    answer_question_with_raw_patient_notes,
    structure_patient_summary,
    summarize_patients,
)
from matchminer_ai.trials import structure_trial_space, summarize_trials


def test_imports():
    assert matchminer_ai is not None
    assert isinstance(matchminer_ai.__version__, str)
    assert MMAIPipeline is not None
    assert load_config is not None
    assert summarize_trials is not None
    assert structure_trial_space is not None
    assert summarize_patients is not None
    assert structure_patient_summary is not None
    assert answer_question_with_raw_patient_notes is not None
    assert embed_for_matching is not None
    assert generate_candidate_matches is not None
    assert score_match_quality is not None
    assert exclusion_criteria_check is not None
    assert contextualize_trial_spaces is not None
    assert personalize_trial_space_context is not None
