"""Combined positive/negative clauses survive consumers of trial-space text."""
import pandas as pd
import pytest

from matchminer_ai._trial_space_format import has_required_space_fields
from matchminer_ai._qc.trials import trial_qc_report
from matchminer_ai.contextualization.query import build_trial_space_query
from matchminer_ai.matching.llm_checks import _load_prompt_template
from matchminer_ai.trials.postprocess import flatten_trial_to_spaces
from matchminer_ai.trials.hierarchy import FIELDS, OUTPUT_FIELDS, VERSION

SPACE = (
    'Age range allowed: 18+. Sex allowed: Any. Cancer primary site allowed: lung. '
    'Histology allowed: adenocarcinoma. Disease state allowed: metastatic. '
    'Prior treatment requirements: prior cisplatin AND no prior pembrolizumab. '
    'Biomarker requirements: KRAS G12D AND no EGFR mutation.'
)
LEGACY = SPACE.replace('Cancer primary site allowed', 'Cancer type allowed').replace(
    'Prior treatment requirements: prior cisplatin AND no prior pembrolizumab',
    'Prior treatment required: cisplatin. Prior treatment excluded: pembrolizumab'
).replace('Biomarker requirements: KRAS G12D AND no EGFR mutation',
          'Biomarkers required: KRAS G12D. Biomarkers excluded: EGFR mutation')


def test_seven_level_contract_matches_authored_layout():
    assert VERSION == 'trial-hierarchy-v5'
    assert len(FIELDS) == len(OUTPUT_FIELDS) == 7
    assert [key for key, _ in FIELDS][-2:] == [
        'prior_treatment_requirements', 'biomarker_requirements']
    assert all(f'{label}:' in SPACE for _, label in OUTPUT_FIELDS)


@pytest.mark.parametrize('space', [SPACE, LEGACY])
def test_both_formats_survive_postprocess_and_qc_without_rewriting(space):
    assert has_required_space_fields(space)
    source = pd.DataFrame([dict(trial_id='T', trial_answer_text=f'1. {space}\nBoilerplate exclusions:\nNone')])
    rows = flatten_trial_to_spaces(source, 'Boilerplate exclusions')
    assert rows.clinical_space_summary.tolist() == [space]
    rows['space_trial_id'] = 'T-0'
    rows['general_exclusion_criteria'] = 'None'
    empty = dict(numerator=0, denominator=1, ids=[])
    qc = trial_qc_report(rows, trial_source=source, unfiltered_spaces=rows,
        truncated_llm_qc_artifact=dict(metric='truncated', **empty),
        failed_llm_qc_artifact=dict(metric='failed', **empty))
    missing = qc[qc.metric.str.contains('missing_keyword')]
    assert len(missing) == 7
    assert (missing.value == 0).all()


@pytest.mark.parametrize('label', ['Prior treatment requirements:', 'Biomarker requirements:'])
def test_missing_combined_field_is_rejected(label):
    assert not has_required_space_fields(SPACE.replace(label, 'Unrecognized:'))


def test_missing_half_of_legacy_field_is_rejected():
    assert not has_required_space_fields(LEGACY.replace('Biomarkers excluded:', 'Unrecognized:'))


def test_context_query_retains_negative_clauses_without_guessing_a_split():
    query = build_trial_space_query(dict(space_trial_id='T-0', trial_id='T', clinical_space_summary=SPACE))
    assert query.disease == 'lung'
    assert query.prior_treatment_requirements == 'prior cisplatin AND no prior pembrolizumab'
    assert query.biomarker_requirements == 'KRAS G12D AND no EGFR mutation'
    assert query.biomarker_query == query.biomarker_requirements
    assert query.prior_treatment_required == query.prior_treatment_excluded == ''
    assert query.biomarkers_required == query.biomarkers_excluded == ''


def test_trialchecker_receives_complete_combined_space_and_polarity_guidance():
    template = _load_prompt_template('llm_match_quality.user.txt')
    prompt = template.format(patient_summary='Fabricated patient.', trial_summary=SPACE)
    assert prompt.count(SPACE) == 1
    assert 'prohibits pembrolizumab exposure' in prompt
    assert 'excludes an EGFR mutation' in prompt
    assert '(f) Excluded biomarkers' in prompt
    classifier = _load_prompt_template('match_quality_checker_template.txt').format(SPACE, 'Fabricated patient.')
    assert classifier.startswith(SPACE)
