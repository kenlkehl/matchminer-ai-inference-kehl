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
    assert 'ANY clear mismatch, the final score is 0 — skip the remaining categories' in prompt
    assert 'Do not add any specificity points after a failed reasonableness gate' in prompt
    assert 'Changing or stopping treatment does not, by itself' in prompt
    assert 'older progression does not establish current progression' in prompt
    assert 'These are not treatment washout periods' in prompt
    assert 'Unknown information alone does not establish a contradiction' in prompt
    assert 'Missing response information is not proof of either response or progression' in prompt
    assert 'when its outcome is also a specified requirement for the trial' in prompt
    assert 'Final score: X' in prompt
    assert prompt.count('1) REASONABLENESS') == 1


def test_zero_score_gate_covers_non_response_mismatches():
    template = _load_prompt_template('llm_match_quality.user.txt')
    prompt = template.format(
        patient_summary='Fabricated adult with metastatic lung cancer, EGFR wild-type.',
        trial_summary='Adults with metastatic lung cancer and an EGFR mutation; no prior treatment or response requirement.',
    )
    gate = prompt.split('1) REASONABLENESS', 1)[1].split('2) CANCER TYPE SPECIFICITY', 1)[0]
    assert 'ANY clear mismatch, the final score is 0' in gate
    assert 'applies to every requirement listed above' in gate
    assert 'age, sex, cancer type/histology, burden, biomarkers, required or excluded prior treatments' in gate
    assert 'An explicit response or disease-status contradiction makes the final score 0' not in gate
    assert 'Unknown information alone does not establish a contradiction' in gate
