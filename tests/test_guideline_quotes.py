import copy
from dataclasses import replace

import pytest

from matchminer_ai.trials._guideline_quotes import (
    materialize_quoted_state, resolve_excerpt,
)
from matchminer_ai.trials._guideline_sources import Page
from test_guideline_extraction import quoted_state


@pytest.fixture
def page():
    return Page("p0002", 2, "SYN-1", "synthetic.md", (
        "Fictional metastatic disease.\n\n"
        "Obtain synthetic blood test A.                 An unrelated column about imaging.\n"
        "Obtain synthetic chemistry panel B.            Another unrelated column.\n"
    ), "synthetic", ())


def test_quote_first_addresses_come_from_source_and_exclude_adjacent_columns(page):
    value = quoted_state()
    value["evidence"] = [{"page_id": page.id, "source_text": "Fictional metastatic disease."}]
    value["diagnostic_workup"][0]["evidence"] = [
        {"page_id": page.id, "source_text": "Obtain synthetic blood test A."}]
    before = copy.deepcopy(value)
    result = materialize_quoted_state(value, {page.id: page})
    evidence = result["diagnostic_workup"][0]["evidence"][0]
    assert evidence["line_ids"] == [3]
    assert evidence["quote"] == "Obtain synthetic blood test A."
    assert "unrelated column" in evidence["source_lines"][0]["text"]
    assert evidence["pdf_page"] == 2
    assert value == before


def test_wrong_numeric_reference_is_not_an_accepted_final_citation(page):
    value = quoted_state()
    value["evidence"] = [{"page_id": page.id, "line_ids": [4]}]
    with pytest.raises(ValueError, match="source_text"):
        materialize_quoted_state(value, {page.id: page})


@pytest.mark.parametrize("text", [
    "Obtain a different synthetic blood test.",
    "Obtain synthetic blood test A. Obtain synthetic chemistry panel B.",
    "Obtain synthetic ... A.",
])
def test_invented_paraphrased_or_cross_column_joined_quotes_fail(page, text):
    with pytest.raises(ValueError, match="does not occur verbatim"):
        resolve_excerpt(page, text)


def test_ambiguous_excerpt_requires_more_context(page):
    with pytest.raises(ValueError, match="occurs 2 times"):
        resolve_excerpt(page, "Obtain synthetic")


def test_only_whitespace_normalization_is_permitted_and_original_bytes_are_returned(page):
    altered = replace(page, text="Use synthetic\t\tpanel B.\nFor the stated population.")
    ids, quote = resolve_excerpt(altered, "Use synthetic panel B.\nFor the stated population.")
    assert ids == [1, 2]
    assert quote == altered.text
    with pytest.raises(ValueError, match="does not occur verbatim"):
        resolve_excerpt(altered, "Use Synthetic panel B.")


def test_unsupplied_page_and_excessive_spans_are_rejected(page):
    with pytest.raises(ValueError, match="Unknown or unsupplied"):
        materialize_quoted_state(quoted_state(), {})
    long_page = replace(page, text="\n".join(f"Synthetic line {i}." for i in range(13)))
    with pytest.raises(ValueError, match="more than 12 lines"):
        resolve_excerpt(long_page, long_page.text)
