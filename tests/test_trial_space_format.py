"""Combined positive/negative clauses survive consumers of saved trial-space text."""
import pytest

from matchminer_ai._trial_space_format import has_required_space_fields

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


@pytest.mark.parametrize('space', [SPACE, LEGACY])
def test_both_formats_have_the_required_fields(space):
    assert has_required_space_fields(space)


@pytest.mark.parametrize('label', ['Prior treatment requirements:', 'Biomarker requirements:'])
def test_missing_combined_field_is_rejected(label):
    assert not has_required_space_fields(SPACE.replace(label, 'Unrecognized:'))


def test_missing_half_of_legacy_field_is_rejected():
    assert not has_required_space_fields(LEGACY.replace('Biomarkers excluded:', 'Unrecognized:'))


def test_decision_model_receives_the_complete_combined_space():
    from matchminer_ai.matching.decision_model import match_state
    assert SPACE.strip() in match_state('Fabricated patient.', SPACE)
