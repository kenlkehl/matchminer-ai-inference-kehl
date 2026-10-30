"""Internal QC helpers for patient summarization outputs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import pandas as pd

from matchminer_ai._qc.common import (
    build_qc_artifact,
    normalize_series,
    qc_artifact_to_report_row,
)
from matchminer_ai.embedding.inference import count_embedding_tokens
from matchminer_ai.patients.summary_quality import check_patient_summary

if TYPE_CHECKING:
    from matchminer_ai.config import MMAIConfig


DEFAULT_KEYWORDS = [
    "Age",
    "Sex",
    "Cancer type",
    "Histology",
    "Current extent",
    "Biomarkers",
    "Treatment history",
]


def _as_list(value: Iterable[str]) -> list[str]:
    return list(value)


def patient_summary_qc_report(
    patient_summaries: pd.DataFrame,
    *,
    noninformative_summary_qc_artifact: dict[str, object],
    failed_inference_qc_artifact: dict[str, object],
    config: MMAIConfig | None = None,
    expected_keywords: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build a QC report for patient summarization outputs without note inputs.

    Parameters
    ----------
    patient_summaries : pd.DataFrame
        Output from summarize_from_relevant_sentences, one row per patient.
        Required columns: patient_id, patient_summary. Historical split
        summaries are also accepted.
    noninformative_summary_qc_artifact : dict[str, object]
        Counts and patient IDs for summaries removed as non-informative during
        postprocessing.
    failed_inference_qc_artifact : dict[str, object]
        Counts and patient IDs removed because inference errors or rejected
        summaries persisted through every quality retry.
    config : MMAIConfig | None, optional
        Config used to resolve backend and embedding settings when token counts
        are computed inside this QC function.
    expected_keywords : list[str], optional
        Keywords expected in each cancer_history_summary.

    Returns
    -------
    pd.DataFrame
        A QC report with metric name, value, and optional ids.
    """
    expected_keywords = expected_keywords or _as_list(DEFAULT_KEYWORDS)

    summaries = patient_summaries.copy()
    combined = "patient_summary" in summaries
    if combined:
        summaries["cancer_history_summary"] = summaries["patient_summary"].fillna("")
        summaries["general_exclusion_criteria_evidence"] = (
            summaries["patient_summary"].fillna("").str.partition("Other relevant history:")[2].str.strip()
        )
    required = [
        "patient_id",
        "cancer_history_summary",
        "general_exclusion_criteria_evidence",
    ]
    missing = [col for col in required if col not in summaries.columns]
    if missing:
        raise ValueError(f"patient_summaries is missing columns: {', '.join(missing)}")

    summaries["cancer_history_summary"] = normalize_series(
        summaries["cancer_history_summary"]
    )
    summaries["general_exclusion_criteria_evidence"] = normalize_series(
        summaries["general_exclusion_criteria_evidence"]
    )

    metrics: list[dict[str, object]] = []
    total_patients = int(summaries["patient_id"].nunique())

    metrics.append(qc_artifact_to_report_row(noninformative_summary_qc_artifact))
    metrics.append(qc_artifact_to_report_row(failed_inference_qc_artifact))

    quality_text = summaries["cancer_history_summary"]
    if not combined:
        quality_text = quality_text + "\n\n" + summaries["general_exclusion_criteria_evidence"]
    checks = [check_patient_summary(value) for value in quality_text]
    quality_codes = sorted({f["code"] for check in checks for f in check["findings"]})
    for code in [None, *quality_codes]:
        ids = [str(identifier) for identifier, check in zip(summaries["patient_id"], checks)
               if check["flagged"] and (code is None or any(f["code"] == code for f in check["findings"]))]
        metrics.append(qc_artifact_to_report_row(build_qc_artifact(
            metric="patients_summary_quality_flagged" + (f":{code}" if code else ""),
            ids=ids, denominator=total_patients,
        )))

    # QC metric for summaries that exceed embedding model token limit
    if config is not None and config.embedding:
        embedding_config = dict(config.embedding)
        embedding_token_limit = int(embedding_config.get("query_length", embedding_config.get("max_seq_length", 8192)))
        token_series = pd.Series(
            count_embedding_tokens(
                summaries["cancer_history_summary"].fillna("").astype(str).tolist(),
                embedding_config=embedding_config,
            ),
            index=summaries["patient_id"].astype(str).tolist(),
        )
        token_series = pd.to_numeric(token_series, errors="coerce").fillna(0)
        over_limit_ids = token_series[token_series > embedding_token_limit].index
        metrics.append(
            qc_artifact_to_report_row(
                build_qc_artifact(
                    metric="patients_exceed_embedding_token_limit",
                    ids=sorted(over_limit_ids.astype(str).unique().tolist()),
                    denominator=total_patients,
                )
            )
        )

    # QC metric for patients whose exclusion criteria info was not extracted
    missing_history = (summaries["general_exclusion_criteria_evidence"].str.strip().isin(["", "Not documented."])
                       if combined else summaries["cancer_history_summary"].str.strip()
                       == summaries["general_exclusion_criteria_evidence"].str.strip())
    same_text = summaries.loc[missing_history, "patient_id"]
    metrics.append(
        qc_artifact_to_report_row(
            build_qc_artifact(
                metric="patients_exclusion_criteria_not_extracted",
                ids=sorted(same_text.astype(str).unique().tolist()),
                denominator=total_patients,
            )
        )
    )

    # QC metrics for summaries missing expected keywords (per keyword).
    for keyword in expected_keywords:
        missing_patients = summaries.loc[
            ~summaries["cancer_history_summary"].str.contains(keyword, regex=False),
            "patient_id",
        ]
        metrics.append(
            qc_artifact_to_report_row(
                build_qc_artifact(
                    metric=f"patients_missing_keyword:{keyword}",
                    ids=sorted(missing_patients.astype(str).unique().tolist()),
                    denominator=total_patients,
                )
            )
        )

    return pd.DataFrame(metrics)
