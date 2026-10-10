"""Response criteria are part of the runtime TrialChecker prompt contract."""
from matchminer_ai.matching.llm_checks import _load_prompt_template


def test_response_constraints_render_without_changing_score_contract():
    template = _load_prompt_template('llm_match_quality.user.txt')
    prompt = template.format(
        patient_summary='Fabricated patient: progressed after chemotherapy. Marker {unknown}.',
        trial_summary='Requires non-progressive Ewing sarcoma after chemotherapy.',
    )
    assert 'Marker {unknown}.' in prompt
    assert 'currently non-progressive disease' in prompt
    assert 'latest patient information documents progression' in prompt
    assert 'final score 0; skip the remaining categories' in prompt
    assert 'Changing or stopping treatment does not, by itself' in prompt
    assert 'older progression does not establish current progression' in prompt
    assert 'These are not treatment washout periods' in prompt
    assert 'Unknown information alone does not establish a contradiction' in prompt
    assert 'unknown response' in prompt
    assert 'cannot earn specificity credit' in prompt
    assert 'Final score: X' in prompt
    assert prompt.count('1) REASONABLENESS') == 1
