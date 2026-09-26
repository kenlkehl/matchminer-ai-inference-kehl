"""Mechanical QC metrics for source-grounded guideline populations."""

import pandas as pd

from .common import build_qc_artifact, qc_artifact_to_report_row


def guideline_qc_report(spaces: pd.DataFrame, status: dict) -> pd.DataFrame:
    artifacts = [
        build_qc_artifact(
            metric="guideline_uncertain_source_pages",
            ids=status["uncertain_pages"],
            denominator=status["extracted_pages"],
        )
    ]
    for metric, column in (
        ("guideline_spaces_with_uncertainties", "uncertainties"),
        ("guideline_spaces_with_omitted_context", "omitted_source_page_ids"),
        ("guideline_spaces_without_treatment_options", "treatment_options"),
    ):
        missing = column == "treatment_options"
        ids = [
            row["space_trial_id"]
            for row in spaces.to_dict("records")
            if bool(row[column]) != missing
        ]
        artifacts.append(
            build_qc_artifact(metric=metric, ids=ids, denominator=len(spaces))
        )
    return pd.DataFrame([qc_artifact_to_report_row(a) for a in artifacts])
