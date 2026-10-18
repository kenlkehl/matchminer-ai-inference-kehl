"""Label checks for current and saved legacy trial-space text.

Mirrored as training's trial_space_format.py. These checks preserve the original
text; they never infer polarity or convert split criteria to combined criteria.
"""

REQUIRED_TRIAL_SPACE_KEYWORDS = (
    "Age",
    "Sex",
    "Cancer primary site",
    "Histology",
    "Disease state",
    "Prior treatment requirements",
    "Biomarker requirements",
)

_ALTERNATIVES = {
    "Cancer primary site": (("Cancer primary site allowed:",), ("Cancer type allowed:",)),
    "Disease state": (("Disease state allowed:",), ("Cancer burden allowed:",)),
    "Prior treatment requirements": (
        ("Prior treatment requirements:",),
        ("Prior treatment required:", "Prior treatment excluded:"),
    ),
    "Biomarker requirements": (
        ("Biomarker requirements:",),
        ("Biomarkers required:", "Biomarkers excluded:"),
    ),
}


def has_space_keyword(text: str, keyword: str) -> bool:
    if not isinstance(text, str):
        return False
    return any(
        all(term in text for term in terms)
        for terms in _ALTERNATIVES.get(keyword, ((keyword,),))
    )


def has_required_space_fields(text: str) -> bool:
    return all(has_space_keyword(text, key) for key in REQUIRED_TRIAL_SPACE_KEYWORDS)
