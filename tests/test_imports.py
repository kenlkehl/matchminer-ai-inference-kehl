import matchminer_ai
from matchminer_ai import MMAIPipeline, load_config, ocr_pdf
from matchminer_ai.contextualization import (
    contextualize_trial_spaces,
    personalize_trial_space_context,
)
from matchminer_ai.embedding import embed_for_matching
from matchminer_ai.good_options import (
    evaluate_good_options,
    research_good_options,
    score_good_options,
    score_good_options_with_llm,
)
from matchminer_ai.matching import (
    assess_trial_centric_cutoff_stability,
    exclusion_criteria_check,
    find_trial_centric_cutoff,
    generate_candidate_matches,
    interpret_exclusion_criteria,
    interpret_match_quality,
    score_match_quality,
)
from matchminer_ai.paradigms import (
    PatientParadigmRankingResult,
    rank_patient_space_paradigms,
)
from matchminer_ai.patients import (
    answer_question_with_raw_patient_notes,
    concatenate_patient_note_pdfs,
    full_patient_screen,
    structure_patient_summaries,
    structure_patient_summary,
    summarize_patients,
)
from matchminer_ai.trials import (
    extract_trial_space_eligibility_criteria,
    structure_trial_space,
    summarize_trials,
)


def test_imports():
    assert matchminer_ai is not None
    assert isinstance(matchminer_ai.__version__, str)
    assert MMAIPipeline is not None
    assert load_config is not None
    assert ocr_pdf is not None
    assert summarize_trials is not None
    assert extract_trial_space_eligibility_criteria is not None
    assert structure_trial_space is not None
    assert summarize_patients is not None
    assert structure_patient_summaries is not None
    assert structure_patient_summary is not None
    assert answer_question_with_raw_patient_notes is not None
    assert concatenate_patient_note_pdfs is not None
    assert full_patient_screen is not None
    assert embed_for_matching is not None
    assert research_good_options is not None
    assert score_good_options is not None
    assert score_good_options_with_llm is not None
    assert evaluate_good_options is not None
    assert generate_candidate_matches is not None
    assert find_trial_centric_cutoff is not None
    assert assess_trial_centric_cutoff_stability is not None
    assert score_match_quality is not None
    assert PatientParadigmRankingResult is not None
    assert rank_patient_space_paradigms is not None
    assert exclusion_criteria_check is not None
    assert interpret_match_quality is not None
    assert interpret_exclusion_criteria is not None
    assert contextualize_trial_spaces is not None
    assert personalize_trial_space_context is not None
