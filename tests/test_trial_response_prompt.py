"""The runtime scoring rubric keeps fixed instructions before variable inputs."""
from matchminer_ai.matching.llm_checks import _load_prompt_template


def test_scoring_prefix_is_identical_across_patients_and_trials():
    template = _load_prompt_template('llm_match_quality.user.txt')
    first = template.format(
        patient_summary='Fabricated patient: progressed after chemotherapy. Marker {unknown}.',
        trial_summary='Requires non-progressive Ewing sarcoma after chemotherapy.',
    )
    second = template.format(
        patient_summary='Another fabricated patient.',
        trial_summary='Another fabricated trial.',
    )
    prefix, _, context = first.partition('YOUR PATIENT SUMMARY:')
    assert prefix == second.partition('YOUR PATIENT SUMMARY:')[0]
    assert 'SCORING INSTRUCTIONS' in prefix and 'OUTPUT FORMAT' in prefix
    assert 'GENERAL GUIDANCE' in prefix and 'Final score: X' in prefix
    assert 'Marker {unknown}.' in context
    assert context.index('Fabricated patient:') < context.index('YOUR TRIAL SUMMARY:')
    assert context.rstrip().endswith('Evaluate this patient–trial match using the instructions above.')
    assert 'SCORING INSTRUCTIONS' not in context


def test_current_gate_preserves_treatment_response_and_biomarker_exceptions():
    template = _load_prompt_template('llm_match_quality.user.txt')
    gate = template.split('1) REASONABLENESS', 1)[1].split('2) CANCER TYPE SPECIFICITY', 1)[0]
    assert 'required response or progression on that prior therapy' in gate
    assert 'but not during the specified line, the trial is not reasonable' in gate
    specificity = template.split('4) PRIOR TREATMENT SPECIFICITY', 1)[1].split('5) BIOMARKER SPECIFICITY', 1)[0]
    assert 'did not receive it during that line of therapy' in specificity
    assert 'standard of care' in gate and 'novel biomarkers' in gate
    assert 'ignore washout details' in gate
    assert 'your final score is 0, and you must skip the following specificity components' in gate
    assert 'For trials that are reasonable, give 1 point for reasonableness' in gate
