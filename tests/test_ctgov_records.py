import httpx
import pytest

from matchminer_ai.trials.ctgov_records import (
    extract_trial_eligibility_criteria,
    fetch_trial_ctgov_document,
    fetch_trial_eligibility_criteria,
    normalize_nct_reference,
)


STUDY = {
    "protocolSection": {
        "identificationModule": {
            "briefTitle": "Drug A plus Drug B study",
            "officialTitle": "Official Drug A plus Drug B Study",
        },
        "descriptionModule": {
            "briefSummary": "A combination study.",
            "detailedDescription": "A detailed registry description.",
        },
        "eligibilityModule": {
            "eligibilityCriteria": (
                "Inclusion Criteria:\r\n\r\n* EGFR &amp; ALK testing required.\r\n"
                "\r\nExclusion Criteria:\r\n\r\n* Active brain metastases."
            )
        },
        "statusModule": {
            "overallStatus": "RECRUITING",
            "lastUpdatePostDateStruct": {
                "date": "2026-07-15",
                "type": "ACTUAL",
            },
        },
    }
}


def test_extracts_and_fetches_complete_eligibility_criteria():
    criteria = extract_trial_eligibility_criteria(STUDY)
    assert "EGFR & ALK testing required" in criteria
    assert "\n\n" in criteria

    def respond(request: httpx.Request) -> httpx.Response:
        assert str(request.url).endswith("/api/v2/studies/NCT12345678")
        return httpx.Response(200, json=STUDY)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        result = fetch_trial_eligibility_criteria(
            "nct12345678",
            client=client,
        )

    assert result.nct_id == "NCT12345678"
    assert result.eligibility_criteria == criteria
    assert result.source_url.endswith("/study/NCT12345678")
    assert result.last_update_post_date == "2026-07-15"
    assert result.fetched_at_utc


def test_missing_complete_eligibility_criteria_is_rejected():
    with pytest.raises(ValueError, match="does not provide complete"):
        extract_trial_eligibility_criteria({"protocolSection": {}})


def test_nct_url_fetch_is_wrangled_for_trial_space_extraction():
    assert normalize_nct_reference(
        "https://clinicaltrials.gov/study/NCT12345678?format=json"
    ) == "NCT12345678"
    assert normalize_nct_reference(
        "clinicaltrials.gov/ct2/show/nct12345678"
    ) == "NCT12345678"
    with pytest.raises(ValueError, match="ClinicalTrials.gov"):
        normalize_nct_reference("https://example.test/study/NCT12345678")

    def respond(request: httpx.Request) -> httpx.Response:
        assert str(request.url).endswith("/api/v2/studies/NCT12345678")
        return httpx.Response(200, json=STUDY)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        document = fetch_trial_ctgov_document(
            "https://clinicaltrials.gov/study/NCT12345678",
            client=client,
        )

    assert document.nct_id == "NCT12345678"
    assert document.trial_title == "Official Drug A plus Drug B Study"
    assert document.brief_summary == "A combination study."
    assert document.detailed_description == "A detailed registry description."
    assert "Active brain metastases" in document.eligibility_criteria
