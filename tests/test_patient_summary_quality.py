import pandas as pd
import pytest

from matchminer_ai.patients import check_patient_summary
from matchminer_ai._qc.patients import patient_summary_qc_report


_QC_ARTIFACTS = dict(
    noninformative_summary_qc_artifact=dict(
        metric="patients_dropped_noninformative_summary", ids=[], denominator=1
    ),
    failed_inference_qc_artifact=dict(
        metric="patients_failed_inference", ids=[], denominator=1
    ),
)


def cancer(kind='Lung cancer', histology='Adenocarcinoma', biomarkers='EGFR L858R',
           extent='Metastatic', treatment='2024: osimertinib; partial response'):
    return (f'Age: 62\nSex: Male\nCancer type: {kind}\nHistology: {histology}\n'
            f'Current extent: {extent}\nBiomarkers: {biomarkers}\nTreatment history:\n{treatment}')


def codes(text):
    return {f['code'] for f in check_patient_summary(text)['findings']}


def test_distinct_primary_cancers_with_repeated_demographics_are_valid():
    summary = cancer() + '\n\n' + cancer('Follicular lymphoma', 'Grade 1-2', 'CD20 positive',
                                        'Stage II', '2004-present: watch and wait')
    result = check_patient_summary(summary + '\n\nOther relevant history:\nHypertension.')
    assert result['cancer_sections'] == 2
    assert not result['flagged']


def test_same_site_with_different_biomarker_must_not_be_collapsed():
    assert not check_patient_summary(cancer() + '\n\n' + cancer(biomarkers='EGFR wild-type'))['flagged']


def test_shared_core_fields_but_different_tumor_history_only_requests_review():
    summary = cancer(extent='Left primary', treatment='2020: left lobectomy') + '\n\n' + cancer(
        extent='Right primary', treatment='2024: right lobectomy')
    result = check_patient_summary(summary)
    assert codes(summary) == {'repeated_cancer_fields'}
    assert all(f['confidence'] == 'review' for f in result['findings'])


def test_exact_duplicate_cancer_fields_and_history_detected():
    first = cancer() + '\n\nOther relevant history:\nECOG 1.'
    summary = first + '\n\nFinal Polish:\n' + first
    result = check_patient_summary(summary)
    assert codes(summary) >= {'duplicate_cancer_section', 'prompt_commentary'}
    duplicate = next(f for f in result['findings'] if f['code']=='duplicate_cancer_section')
    assert duplicate['previous_line'] == 3
    assert duplicate['line'] > duplicate['previous_line']
    assert 'EGFR' not in str(result)  # Diagnostics contain offsets, not patient text.


def test_duplicate_field_normalization_accepts_case_spacing_and_markdown():
    second = cancer().upper().replace('CANCER TYPE:', '**CANCER TYPE:**').replace('Age:', 'Age:  ')
    assert 'duplicate_cancer_section' in codes(cancer() + '\n\n' + second)


def test_unknown_or_missing_fields_do_not_establish_duplication():
    value = 'Age: 62\nCancer type: Lung\nHistology: Unknown\nBiomarkers: Not documented.'
    assert not check_patient_summary(value + '\n\n' + value)['flagged']


def test_complete_duplicate_with_unknown_biomarkers_still_flags():
    value = cancer(biomarkers='Not documented.')
    assert 'duplicate_cancer_section' in codes(value + '\n\n' + value)


def test_multiple_cancers_can_share_one_demographics_header():
    second = cancer('Follicular lymphoma', 'Grade 1-2', 'CD20 positive').split('Cancer type:', 1)[1]
    assert not check_patient_summary(cancer() + '\nCancer type:' + second)['flagged']


@pytest.mark.parametrize('phrase', ['I should check this.', 'Wait, that is wrong.', 'I need to rewrite this.',
                                  'Let me verify the summary.', "Let's think again.", 'Hold on: recheck.'])
def test_self_editing_phrases(phrase):
    assert 'self_editing_language' in codes(cancer() + '\n\n' + phrase)


@pytest.mark.parametrize('phrase', ['Final Polish:', 'Formatting check:', 'Final construction.',
                                  'The prompt says to include history.', 'The user requests no markdown.',
                                  'No markdown. No Unicode.', 'One last look:', 'Final check on drug names:'])
def test_commentary(phrase):
    assert 'prompt_commentary' in codes(cancer() + '\n\n' + phrase)


@pytest.mark.parametrize('text', [
    'Other relevant history:\nWatch and wait for lymphoma. Await imaging. Waiting for pathology.',
    'Other relevant history:\nPatient said, "I should take my medication."',
    'Other relevant history:\nWait time for surgery was three weeks.',
    'No cancer diagnosis documented as of 2025-01-01.\nOther relevant history:\nHypertension.',
])
def test_clinical_prose_and_quotes_are_not_drafting(text):
    assert not check_patient_summary(text)['flagged']


@pytest.mark.parametrize('marker', ['<think>', '</think>', '<|channel|>thought', '<|channel>thought', '[THINK]'])
def test_raw_reasoning_delimiters(marker):
    assert 'reasoning_delimiter' in codes(marker + cancer())


@pytest.mark.parametrize('value', [None, '', '  ', 'ERROR: generation failed'])
def test_empty_or_failed(value):
    assert codes(value) == {'empty_or_failed'}


def test_long_repeated_prose():
    paragraph = ' '.join(['The patient underwent treatment and developed persistent neuropathy.'] * 6)
    assert 'repeated_long_paragraph' in codes(paragraph + '\n\n' + paragraph)


def test_qc_includes_quality_flags_without_mutating_input():
    frame = pd.DataFrame([dict(patient_id='good', patient_summary=cancer()),
                          dict(patient_id='bad', patient_summary=cancer()+'\nFinal Polish:\n'+cancer())])
    before = frame.copy(deep=True)
    qc = patient_summary_qc_report(frame, **_QC_ARTIFACTS).set_index('metric')
    assert qc.loc['patients_summary_quality_flagged', 'ids'] == ['bad']
    pd.testing.assert_frame_equal(frame, before)
