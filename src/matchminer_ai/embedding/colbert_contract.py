"""Retrieval text contract shared by runtime and gated regression training."""
DIRECTIONS = ("patient_to_cohort", "cohort_to_patient")
INSTRUCTION = ('Find potentially eligible patient–cohort matches, prioritizing matching cancer type, '
               'disease burden, prior treatment, and required biomarkers.')


def regression_query(direction, text):
    if direction not in DIRECTIONS:
        raise ValueError('Unknown retrieval direction')
    subject = 'Patient history' if direction == 'patient_to_cohort' else 'Cohort eligibility criteria'
    return f'{INSTRUCTION}\n\n{subject}:\n{text}'

