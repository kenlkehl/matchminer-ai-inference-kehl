from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest

from matchminer_ai.contextualization import (
    EvidenceItem,
    SourceNotice,
    TrialSpaceContextualizationResult,
    contextualize_trial_spaces,
    personalize_trial_space_context,
)
from matchminer_ai.contextualization.api import _pack_prompt_evidence
from matchminer_ai.contextualization.query import (
    build_trial_space_query,
    parse_clinical_space_summary,
)


SPACE_SUMMARY = (
    "Age range allowed: 18+. Sex allowed: Any. "
    "Cancer type allowed: non-small cell lung cancer. "
    "Histology allowed: adenocarcinoma. "
    "Cancer burden allowed: metastatic disease. "
    "Prior treatment required: platinum chemotherapy. "
    "Prior treatment excluded: prior osimertinib. "
    "Biomarkers required: EGFR L858R mutation. "
    "Biomarkers excluded: ALK rearrangement."
)


class WordTokenizer:
    def encode(self, text, *, add_special_tokens=False):
        del add_special_tokens
        return text.split()

    def decode(
        self,
        tokens,
        *,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ):
        del skip_special_tokens, clean_up_tokenization_spaces
        return " ".join(tokens)


@pytest.fixture(autouse=True)
def _use_network_free_context_tokenizer(monkeypatch):
    monkeypatch.setattr(
        "matchminer_ai.contextualization.api._resolve_evidence_tokenizer",
        lambda config, settings: (
            WordTokenizer(),
            {
                "kind": "test_tokenizer",
                "model_name": "word-tokenizer",
                "warning": "",
            },
        ),
    )


def _spaces(**extra: Any) -> pd.DataFrame:
    row = {
        "space_trial_id": "NCT12345678-1",
        "trial_id": "NCT12345678",
        "clinical_space_summary": SPACE_SUMMARY,
    }
    row.update(extra)
    return pd.DataFrame([row])


class EvidenceAdapter:
    name = "test_evidence"

    def __init__(self) -> None:
        self.queries = []

    async def fetch(self, query, *, client, max_items, settings):
        del client, max_items, settings
        self.queries.append(query)
        return [
            EvidenceItem(
                evidence_id="test:1",
                space_trial_id=query.space_trial_id,
                trial_id=query.trial_id,
                source="nci_pdq",
                evidence_type="evidence_summary",
                title="Synthetic source title",
                excerpt="Synthetic evidence excerpt.",
                url="https://example.org/evidence",
                source_locator="test item 1",
                query=query.disease_query,
            )
        ], [SourceNotice(source=self.name, status="ok", message="One item.")]


class FailingAdapter:
    name = "test_failure"

    async def fetch(self, query, *, client, max_items, settings):
        del query, client, max_items, settings
        raise http_error("synthetic source outage")


class EmptyAdapter:
    name = "test_empty"

    async def fetch(self, query, *, client, max_items, settings):
        del query, client, max_items, settings
        return [], [
            SourceNotice(source=self.name, status="empty", message="No records.")
        ]


def http_error(message: str) -> RuntimeError:
    return RuntimeError(message)


def test_parses_trial_space_without_query_planning_llm():
    fields = parse_clinical_space_summary(SPACE_SUMMARY)
    query = build_trial_space_query(_spaces().iloc[0].to_dict())

    assert fields["cancer type allowed"] == "non-small cell lung cancer"
    assert query.disease == "non-small cell lung cancer"
    assert query.histology == "adenocarcinoma"
    assert query.biomarker_query == "EGFR L858R mutation ALK rearrangement"


@pytest.mark.parametrize(
    "column",
    [
        "patient_id",
        "cancer_history_summary",
        "general_exclusion_criteria_evidence",
        "pseudo_mrn",
        "note_text",
    ],
)
def test_contextualization_rejects_patient_bearing_columns(column):
    with pytest.raises(ValueError, match="trial-only input"):
        contextualize_trial_spaces(_spaces(**{column: "PRIVATE_PATIENT_MARKER"}))


def test_partial_sources_are_synthesized_with_notices(monkeypatch):
    evidence_adapter = EvidenceAdapter()
    monkeypatch.setattr(
        "matchminer_ai.contextualization.api.resolve_sources",
        lambda names: [evidence_adapter, FailingAdapter()],
    )
    llm_calls = []

    def fake_llm(messages, *, config, section_name):
        del config
        llm_calls.append((messages, section_name))
        return (
            [
                "## Disease context\nSynthetic context [E1]\n"
                "## Diagnostic considerations\nSynthetic diagnostic [E1]\n"
                "## Therapeutic considerations\nSynthetic treatment [E1]\n"
                "## Evidence limits\nLimited evidence [E1]"
            ],
            {"model_name": "synthetic-model"},
            ["stop"],
        )

    monkeypatch.setattr(
        "matchminer_ai.contextualization.api._run_llm",
        fake_llm,
    )
    result = contextualize_trial_spaces(
        _spaces(),
        sources=("nci_pdq", "pubmed"),
    )

    assert len(evidence_adapter.queries) == 1
    assert evidence_adapter.queries[0].disease == "non-small cell lung cancer"
    assert result.contexts.iloc[0]["contextualization_status"] == "ok"
    assert result.contexts.iloc[0]["evidence_count"] == 1
    assert (
        result.contexts.iloc[0]["diagnostic_evidence_sufficiency"]
        == "insufficient"
    )
    notices = result.metadata["source_notices"]["NCT12345678-1"]
    assert any(notice["status"] == "failed" for notice in notices)
    assert result.evidence.iloc[0]["citation_label"] == "E1"
    assert result.evidence.iloc[0]["evidence_category"] == "general"
    assert len(llm_calls) == 1
    prompt = llm_calls[0][0][0][1]["content"]
    assert "### Workup generally expected before this disease state" in prompt
    assert "### Workup to consider now or at the next decision point" in prompt
    assert "pathologic or histologic confirmation" in prompt
    assert "timing or conditions for repetition" in prompt
    assert "rather than supplying it from intrinsic knowledge" in prompt
    assert '"diagnostic_evidence_signal"' in prompt
    assert "Do not include preclinical or experimental mechanisms" in prompt


def test_no_evidence_skips_llm(monkeypatch):
    monkeypatch.setattr(
        "matchminer_ai.contextualization.api.resolve_sources",
        lambda names: [EmptyAdapter()],
    )

    def should_not_run(*args, **kwargs):
        raise AssertionError("LLM must not run without evidence")

    monkeypatch.setattr(
        "matchminer_ai.contextualization.api._run_llm",
        should_not_run,
    )
    result = contextualize_trial_spaces(_spaces(), sources=("pubmed",))

    row = result.contexts.iloc[0]
    assert row["contextualization_status"] == "no_evidence"
    assert "intrinsic knowledge" in row["contextualization_markdown"]
    assert result.metadata["llm_spaces"] == 0


def test_bad_citation_is_retried_once(monkeypatch):
    monkeypatch.setattr(
        "matchminer_ai.contextualization.api.resolve_sources",
        lambda names: [EvidenceAdapter()],
    )
    outputs = iter(
        [
            "## Disease context\nUnsupported [E99]",
            (
                "## Disease context\nSupported [E1]\n"
                "## Diagnostic considerations\nSupported [E1]\n"
                "## Therapeutic considerations\nSupported [E1]\n"
                "## Evidence limits\nSupported [E1]"
            ),
        ]
    )
    calls = []

    def fake_llm(messages, *, config, section_name):
        del config, section_name
        calls.append(messages)
        return [next(outputs)], {"model_name": "synthetic-model"}, ["stop"]

    monkeypatch.setattr(
        "matchminer_ai.contextualization.api._run_llm",
        fake_llm,
    )
    result = contextualize_trial_spaces(_spaces(), sources=("nci_pdq",))

    assert len(calls) == 2
    assert (
        result.contexts.iloc[0]["contextualization_status"]
        == "ok_after_citation_retry"
    )


def test_trial_space_pseudo_citation_is_retried(monkeypatch):
    monkeypatch.setattr(
        "matchminer_ai.contextualization.api.resolve_sources",
        lambda names: [EvidenceAdapter()],
    )
    outputs = iter(
        [
            "## Disease context\nThe trial space represents disease [Trial Space].",
            (
                "## Disease context\nThe trial space represents disease.\n"
                "## Diagnostic considerations\nSupported [E1]\n"
                "## Therapeutic considerations\nSupported [E1]\n"
                "## Evidence limits\nSupported [E1]"
            ),
        ]
    )
    calls = []

    def fake_llm(messages, *, config, section_name):
        del config, section_name
        calls.append(messages)
        return [next(outputs)], {"model_name": "synthetic-model"}, ["stop"]

    monkeypatch.setattr(
        "matchminer_ai.contextualization.api._run_llm",
        fake_llm,
    )
    result = contextualize_trial_spaces(_spaces(), sources=("nci_pdq",))

    assert len(calls) == 2
    assert "Unsupported pseudo-citation labels" in calls[1][0][-1]["content"]
    assert (
        result.contexts.iloc[0]["contextualization_status"]
        == "ok_after_citation_retry"
    )


def test_evidence_packing_uses_token_budget_and_reserves_diagnostics():
    records = []
    for index, category in enumerate(
        ["diagnostic", "diagnostic", "diagnostic", "therapeutic"], start=1
    ):
        records.append(
            {
                "citation_label": f"E{index}",
                "source": f"synthetic-{index}",
                "evidence_type": f"{category}_guideline_full_text",
                "title": f"Synthetic {category} source {index}",
                "excerpt": "biopsy staging imaging molecular " * 1500,
                "url": f"https://example.org/{index}",
                "source_locator": f"section {index}",
                "published_at": "2026",
                "updated_at": "",
                "jurisdiction": "Synthetic",
                "license": "CC0",
                "attributes": {"evidence_category": category},
            }
        )

    packed, stats = _pack_prompt_evidence(
        pd.DataFrame(records),
        tokenizer=WordTokenizer(),
        max_tokens=12000,
        diagnostic_min_tokens=6000,
        item_max_tokens=4000,
    )

    assert stats["context_evidence_token_budget"] == 12000
    assert stats["packed_evidence_tokens"] == 12000
    assert stats["diagnostic_evidence_tokens"] >= 6000
    assert stats["diagnostic_evidence_sufficiency"] == "broad"
    assert stats["truncated_evidence_count"] >= 1
    assert {"pathology_or_specimen", "staging_or_extent", "imaging"}.issubset(
        stats["diagnostic_coverage"]
    )
    assert all(record["excerpt_tokens"] <= 4000 for record in packed)


def test_therapeutic_packing_prefers_nci_over_experimental_full_text():
    common = {
        "published_at": "2026",
        "updated_at": "",
        "jurisdiction": "Synthetic",
        "license": "CC0",
    }
    records = [
        {
            **common,
            "citation_label": "E1",
            "source": "europe_pmc_open_guidelines",
            "evidence_type": "therapeutic_guideline_full_text",
            "title": "Experimental mechanism discussion",
            "excerpt": "preclinical experimental mechanism " * 1000,
            "url": "https://example.org/experimental",
            "source_locator": "experimental section",
            "attributes": {"evidence_category": "therapeutic"},
        },
        {
            **common,
            "citation_label": "E2",
            "source": "nci_pdq",
            "evidence_type": "treatment_evidence_summary",
            "title": "NCI treatment evidence",
            "excerpt": "established treatment evidence " * 1000,
            "url": "https://example.org/nci",
            "source_locator": "treatment section",
            "attributes": {"evidence_category": "therapeutic"},
        },
    ]

    packed, _ = _pack_prompt_evidence(
        pd.DataFrame(records),
        tokenizer=WordTokenizer(),
        max_tokens=1000,
        diagnostic_min_tokens=0,
        item_max_tokens=1000,
    )

    assert packed[0]["citation_label"] == "E2"
    assert packed[0]["source"] == "NCI PDQ"


def test_patient_personalization_is_separate_and_per_space(monkeypatch):
    contexts = pd.DataFrame(
        [
            {
                "space_trial_id": "NCT12345678-1",
                "trial_id": "NCT12345678",
                "clinical_space_summary": SPACE_SUMMARY,
                "contextualization_markdown": "Grounded context [E1]",
                "contextualization_status": "ok",
            }
        ]
    )
    contextualization = TrialSpaceContextualizationResult(
        contexts=contexts,
        evidence=pd.DataFrame(),
        metadata={},
    )
    captured = []

    def fake_llm(messages, *, config, section_name):
        del config
        captured.extend(messages)
        assert section_name == "patient_contextualization"
        return ["Per-space patient review [E1]"], {}, ["stop"]

    monkeypatch.setattr(
        "matchminer_ai.contextualization.api._run_llm",
        fake_llm,
    )
    with patch(
        "matchminer_ai.contextualization.api.resolve_sources",
        side_effect=AssertionError("personalization must not retrieve"),
    ):
        result = personalize_trial_space_context(
            pd.DataFrame(
                [
                    {
                        "patient_id": "SYNTHETIC-1",
                        "space_trial_id": "NCT12345678-1",
                        "cancer_history_summary": "Fabricated patient summary.",
                        "general_exclusion_criteria_evidence": "No evidence.",
                    }
                ]
            ),
            contextualization,
        )

    assert result.iloc[0]["personalization_status"] == "ok"
    assert "Fabricated patient summary." in captured[0][1]["content"]
    assert "rank" in captured[0][0]["content"].casefold()
    assert "absent test or result" in captured[0][0]["content"]
    assert "diagnostic steps and results explicitly documented" in captured[0][1][
        "content"
    ]
    assert "needed, repeated, updated, or confirmed" in captured[0][1]["content"]
    assert "not documented" in captured[0][1]["content"]
