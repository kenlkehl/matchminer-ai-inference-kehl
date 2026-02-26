"""Embedding regression tests."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import pandas as pd

from mmai.embedding import embed_for_matching
from mmai.patients import summarize_from_relevant_sentences
from mmai.trials import summarize_trials

if TYPE_CHECKING:
    from mmai.config import MMAIConfig


REGRESSION_DATA_DIR = (
    Path(__file__).resolve().parent / "data" / "embedding_regression" / "mmai-synthetic"
)


def _load_patient_input() -> pd.DataFrame:
    return pd.read_csv(REGRESSION_DATA_DIR / "patient_input.csv")


def _load_patient_output_gold() -> pd.DataFrame:
    return pd.read_csv(REGRESSION_DATA_DIR / "patient_output.csv")


def _load_trial_input() -> pd.DataFrame:
    return pd.read_csv(REGRESSION_DATA_DIR / "trial_input.csv")


def _load_trial_output_gold() -> pd.DataFrame:
    return pd.read_csv(REGRESSION_DATA_DIR / "trial_output.csv")


def _generate_patient_package_embeddings(
    patient_input: pd.DataFrame,
    *,
    config: MMAIConfig | None = None,
) -> pd.DataFrame:
    """Run patient summarize+embed pipeline from post-tagger input."""
    patient_input = patient_input[["patient_id", "patient_long_text"]].copy()
    patient_summaries, _ = cast(
        tuple[pd.DataFrame, dict],
        summarize_from_relevant_sentences(
            patient_input,
            config=config,
            return_qc=False,
        ),
    )
    patient_embedded = embed_for_matching(
        patient_summaries,
        entity_type="patient",
        config=config,
    )
    return patient_embedded[["patient_id", "embedding"]].copy()


def _generate_trial_package_embeddings(
    trial_input: pd.DataFrame,
    *,
    config: MMAIConfig | None = None,
) -> pd.DataFrame:
    """Run trial summarize+embed pipeline from raw trial-level input."""
    trial_input = trial_input[
        ["trial_id", "trial_title", "brief_summary", "eligibility_criteria"]
    ].copy()
    trial_spaces = cast(
        pd.DataFrame,
        summarize_trials(
            trial_input,
            config=config,
            return_qc=False,
        ),
    )
    trial_embedded = embed_for_matching(
        trial_spaces,
        entity_type="trial",
        config=config,
    )
    return trial_embedded[["space_trial_id", "trial_id", "embedding"]].copy()
