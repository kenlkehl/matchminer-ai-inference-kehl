"""Check explicit prompt contracts, without inferring or rewriting clinical facts."""

import re

VERSION = "trialspace-field-semantics-v3"
# A backward-compatible checker fix does not change generation prompts or the
# representation contract. Offline audits record the implementation revision.
REVISION = "trialspace-field-semantics-v3.3-numeric-thresholds"

_ORDINALS = {
    word: str(i)
    for i, word in enumerate(
        (
            "first",
            "second",
            "third",
            "fourth",
            "fifth",
            "sixth",
            "seventh",
            "eighth",
            "ninth",
            "tenth",
        ),
        1,
    )
}
_RANK = r"(?:" + "|".join(_ORDINALS) + r"|\d+(?:st|nd|rd|th)?)"
_LINE = re.compile(r"\b(" + _RANK + r"(?:\s*(?:/|or|and)\s*" + _RANK + r")*)\s+line\b")
_HR = re.compile(
    r"\b(?:hr|hormone\s+receptor)[\s-]*(?:positive|negative|[+−-])", re.IGNORECASE
)
_PRIOR_LINES = re.compile(
    r"\bprior\s+(?:(?:systemic|endocrine|cytotoxic|treatment|therapy)\s+)*lines?\b",
    re.IGNORECASE,
)
_UNRESTRICTED_LINE_LABEL = re.compile(
    r"\b(?:(?:first|1st|1)\s+line|1l)\s+(?:or|and)\s+(?:any\s+)?(?:subsequent|later)\b"
)
# Reject a bare disease-state alternative, not contextual treatment history such
# as 'prior chemotherapy for visceral crisis'. This is a bounded representation
# check, not a clinical concept classifier.
_BARE_VISCERAL_CRISIS = re.compile(
    r"(?:^|[;(]|\b(?:and|or)\b)\s*(?:(?:with|without|no|presence of|absence of)\s+)?"
    r"visceral[\s-]+crisis\s*(?=$|[;).]|\b(?:and|or)\b)",
    re.IGNORECASE,
)
# A terminal numeric comparison is one predicate, not a population-level OR.
# Keep this bounded: "2 OR higher-risk disease" remains an actual alternative.
_NUMERIC_THRESHOLD = re.compile(
    r"(?<![\w.])\d+(?:\.\d+)?\s+(?P<operator>or)\s+"
    r"(?:higher|lower|greater|less|more|fewer|older|younger|above|below)"
    r"(?=\s*(?:$|[),.;:]|\b(?:and|or)\b))"
)


def explicit_lines(text):
    text = re.sub(r"[-‐‑–—]", " ", text.casefold())
    values = set(re.findall(r"\b([1-9]\d*)l\b", text))
    for match in _LINE.finditer(text):
        for rank in re.findall(_RANK, match.group(1)):
            values.add(_ORDINALS.get(rank, re.sub(r"(?:st|nd|rd|th)$", "", rank)))
    return values


def has_ungrouped_mixed_logic(text):
    """Check each parenthesis level for mixed operators, without interpreting concepts."""
    text = text.casefold()
    # Mask only the comparison's operator while tokenizing. Never rewrite the
    # stored field or turn an ungrouped population alternative into conjunction.
    for match in reversed(list(_NUMERIC_THRESHOLD.finditer(text))):
        start, end = match.span("operator")
        text = text[:start] + " " * (end - start) + text[end:]
    levels = [set()]
    for token in re.findall(r"[()]|\band\s*/\s*or\b|\b(?:and|or)\b", text):
        if token == "(":
            levels.append(set())
        elif token == ")":
            if len(levels) > 1:
                levels.pop()
        else:
            # 'and/or' is one inclusive disjunction, not two competing operators.
            # This only tokenizes the check; the original field stays unchanged.
            if "/" in token:
                token = "or"
            levels[-1].add(token)
            if len(levels[-1]) > 1:
                return True
    return False


def validate_decision_fields(state, guideline_title):
    """Generation-only checks; legacy catalogs keep their original validation contract.

    This checks stated labels, not which treatment line is medically appropriate.
    It never fills a field or normalizes a clinical result automatically.
    """
    space = state["space"]
    for field in ("prior_treatment_required", "prior_treatment_excluded"):
        if _BARE_VISCERAL_CRISIS.search(space[field]):
            raise ValueError(
                f"{state['name'][:300]!r}: {field} contains visceral crisis as a "
                "standalone criterion. It is a disease-severity state for "
                "cancer_burden_allowed, not prior treatment. Preserve OR logic: "
                "during extraction/consolidation split alternatives spanning burden "
                "and treatment response into separate complete spaces. Moving both "
                "sides into different fields of one space would change OR to AND. "
                "Never rewrite a fixed definition in a detail or citation-only task."
            )
    for field, value in space.items():
        if has_ungrouped_mixed_logic(value):
            raise ValueError(
                f"{state['name'][:300]!r}: {field} mixes AND and OR at the same "
                f"parenthesis level. Rejected field text: {value[:500]!r}. "
                "Keep only population criteria in this field; place explanatory "
                "workup/treatment prose in the corresponding considerations. "
                "Lowercase conjunctions in prose also count; semicolons do not group logic. "
                "Explicitly group alternatives to preserve the "
                "source population, e.g. '(A OR B) AND C' when C applies to both. "
                "Do not rely on an unstated operator precedence, omit a condition, "
                "or change OR to AND. Group the complete concepts, not just their labels."
            )
    burden = space["cancer_burden_allowed"]
    if explicit_lines(burden) or _PRIOR_LINES.search(burden):
        raise ValueError(
            f"{state['name'][:300]!r}: cancer_burden_allowed contains treatment-line "
            "criteria. Burden describes extent, stage, resectability and prognostic risk. "
            "Encode the source-defined sequence/setting and bounded prior-line counts "
            "in prior_treatment_required / prior_treatment_excluded instead. "
            "Do not change the represented clinical population."
        )
    # First-line OR any subsequent line imposes no history restriction. Mask
    # only that explicit phrase; another bounded line label still requires fields.
    name = re.sub(r"[-‐‑–—]", " ", state["name"].casefold())
    restricted_labels = explicit_lines(_UNRESTRICTED_LINE_LABEL.sub("", name))
    if restricted_labels and all(
        space[f].strip().casefold() == "na"
        for f in ("prior_treatment_required", "prior_treatment_excluded")
    ):
        raise ValueError(
            f"{state['name'][:300]!r}: the treatment-line boundary is only in the name. "
            "Encode its source-defined history in prior_treatment_required / "
            "prior_treatment_excluded, with both bounds for a bounded line/bin; "
            "do not put treatment history in cancer_burden_allowed."
        )
    if re.match(
        r"(?:no|without|absence of)\s+(?:any\s+)?prior\b",
        space["prior_treatment_excluded"].strip(),
        re.IGNORECASE,
    ):
        raise ValueError(
            f"{state['name'][:300]!r}: prior_treatment_excluded must name the treatment "
            "that must NOT have been received, rather than 'No prior treatment', "
            "which reverses exclusion logic. Preserve the source-defined sequence/setting."
        )
    if "breast" in guideline_title.casefold():
        for field in ("biomarkers_required", "biomarkers_excluded"):
            if _HR.search(state["space"][field]):
                raise ValueError(
                    f"{state['name'][:300]!r}, {field}: expand generic breast HR status into ER/PR criteria: "
                    "negative = ER-negative AND PR-negative; positive = ER-positive OR "
                    "PR-positive (either or both). Preserve specific source-stated results."
                )
