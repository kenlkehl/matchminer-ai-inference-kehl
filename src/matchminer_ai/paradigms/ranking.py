"""Roll TrialSpace/TrialChecker patient matches up to space paradigms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

from matchminer_ai.config import load_default_preset
from matchminer_ai.embedding import embed_for_matching
from matchminer_ai.matching import generate_candidate_matches, score_match_quality

if TYPE_CHECKING:
    from matchminer_ai.config import MMAIConfig


RESEARCH_USE_NOTICE = (
    "Space-paradigm rankings are research prioritization signals. They do not "
    "establish diagnosis, trial eligibility, or a treatment recommendation."
)

_PATIENT_COLUMNS = ["patient_id", "cancer_history_summary"]
_TRIAL_COLUMNS = ["space_trial_id", "clinical_space_summary"]
_MEMBERSHIP_COLUMNS = ["space_trial_id", "paradigm_id"]
_CATALOG_COLUMNS = ["paradigm_id", "paradigm_label"]
_NON_SEMANTIC_EMBEDDING_KEYS = {"batch_size", "device"}
_PATIENT_BEARING_TRIAL_COLUMNS = {
    "cancer_history_summary",
    "general_exclusion_criteria_evidence",
    "patient_boilerplate_text",
    "patient_id",
    "patient_summary",
}
_RESERVED_TRIAL_COLUMNS = {
    "embedding",
    "match_quality_pass",
    "match_quality_score",
    "paradigm_count",
    "paradigm_id",
    "similarity_score",
    "space_rank",
    "trialspace_rank",
}
_RESERVED_CATALOG_COLUMNS = {
    "best_match_quality_pass",
    "best_match_quality_score",
    "best_similarity_score",
    "best_space_rank",
    "best_space_trial_id",
    "match_quality_pass",
    "match_quality_score",
    "patient_id",
    "paradigm_rank",
    "similarity_score",
    "space_rank",
    "space_trial_id",
    "supporting_space_count",
    "supporting_space_ids",
    "supporting_space_ranks",
    "trialspace_rank",
}


@dataclass(frozen=True)
class PatientParadigmRankingResult:
    """Outputs from patient-summary-to-space-paradigm matching.

    ``space_matches`` contains the TrialChecker-reranked leading trial spaces.
    ``space_paradigm_matches`` preserves every exact membership, including a
    one-to-many mapping from one trial space to several paradigms.
    ``paradigm_matches`` collapses repeated support for the same paradigm and
    ranks it by its best supporting space, then by support count. It does not
    introduce a new clinical score.

    ``trial_embeddings`` and ``trial_embedding_metadata`` can be cached by a
    caller. Both must be supplied together on reuse so the function can reject
    embeddings made with an incompatible TrialSpace model or prompt contract.
    """

    space_matches: pd.DataFrame
    space_paradigm_matches: pd.DataFrame
    paradigm_matches: pd.DataFrame
    trial_embeddings: pd.DataFrame
    trial_embedding_metadata: dict[str, Any]
    metadata: dict[str, Any]


def _require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{label} must be a pandas DataFrame.")
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def _normalize_string_column(
    frame: pd.DataFrame,
    column: str,
    *,
    label: str,
) -> None:
    if frame[column].isna().any():
        raise ValueError(f"{label}.{column} must not contain null values.")
    frame[column] = frame[column].astype(str).str.strip()
    if frame[column].eq("").any():
        raise ValueError(f"{label}.{column} must not contain empty values.")


def _normalize_inputs(
    patient_summaries: pd.DataFrame,
    trial_spaces: pd.DataFrame,
    space_paradigm_memberships: pd.DataFrame,
    paradigm_catalog: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _require_columns(patient_summaries, _PATIENT_COLUMNS, "patient_summaries")
    _require_columns(trial_spaces, _TRIAL_COLUMNS, "trial_spaces")
    _require_columns(
        space_paradigm_memberships,
        _MEMBERSHIP_COLUMNS,
        "space_paradigm_memberships",
    )
    _require_columns(paradigm_catalog, _CATALOG_COLUMNS, "paradigm_catalog")

    patient_bearing_columns = sorted(
        set(trial_spaces.columns) & _PATIENT_BEARING_TRIAL_COLUMNS
    )
    if patient_bearing_columns:
        raise ValueError(
            "trial_spaces must not contain patient-bearing columns: "
            + ", ".join(patient_bearing_columns)
        )
    reserved_trial_columns = sorted(
        (set(trial_spaces.columns) & _RESERVED_TRIAL_COLUMNS) - set(_TRIAL_COLUMNS)
    )
    if reserved_trial_columns:
        raise ValueError(
            "trial_spaces contains columns reserved for ranking outputs: "
            + ", ".join(reserved_trial_columns)
        )
    reserved_catalog_columns = sorted(
        (set(paradigm_catalog.columns) & _RESERVED_CATALOG_COLUMNS)
        - set(_CATALOG_COLUMNS)
    )
    if reserved_catalog_columns:
        raise ValueError(
            "paradigm_catalog contains columns reserved for ranking outputs: "
            + ", ".join(reserved_catalog_columns)
        )

    patients = patient_summaries.copy()
    spaces = trial_spaces.copy()
    memberships = space_paradigm_memberships[_MEMBERSHIP_COLUMNS].copy()
    catalog = paradigm_catalog.copy()

    for column in _PATIENT_COLUMNS:
        _normalize_string_column(patients, column, label="patient_summaries")
    for column in _TRIAL_COLUMNS:
        _normalize_string_column(spaces, column, label="trial_spaces")
    for column in _MEMBERSHIP_COLUMNS:
        _normalize_string_column(
            memberships,
            column,
            label="space_paradigm_memberships",
        )
    for column in _CATALOG_COLUMNS:
        _normalize_string_column(catalog, column, label="paradigm_catalog")

    if patients.empty:
        raise ValueError("patient_summaries must contain at least one row.")
    if spaces.empty:
        raise ValueError("trial_spaces must contain at least one row.")
    if patients["patient_id"].duplicated().any():
        raise ValueError("patient_summaries must contain unique patient_id values.")
    if spaces["space_trial_id"].duplicated().any():
        raise ValueError("trial_spaces must contain unique space_trial_id values.")
    if memberships.duplicated(_MEMBERSHIP_COLUMNS).any():
        raise ValueError(
            "space_paradigm_memberships must contain unique "
            "space_trial_id/paradigm_id pairs."
        )
    if catalog["paradigm_id"].duplicated().any():
        raise ValueError("paradigm_catalog must contain unique paradigm_id values.")

    unknown_paradigms = sorted(
        set(memberships["paradigm_id"]) - set(catalog["paradigm_id"])
    )
    if unknown_paradigms:
        preview = ", ".join(unknown_paradigms[:5])
        raise ValueError(
            "space_paradigm_memberships references paradigm_id values absent "
            f"from paradigm_catalog: {preview}"
        )

    return patients, spaces, memberships, catalog


def _positive_integer(value: int, *, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer.")
    try:
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive integer.") from exc
    if resolved < 1 or resolved != value:
        raise ValueError(f"{label} must be a positive integer.")
    return resolved


def _embedding_metadata_signature(metadata: dict[str, Any]) -> dict[str, Any]:
    try:
        embedding_config = dict(metadata["config_snapshot"]["embedding"])
        model_metadata = dict(metadata["model_metadata"]["embedding_model"])
        model_name = str(model_metadata["model_name"])
        model_sha = str(model_metadata["model_sha"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "trial_embedding_metadata must be metadata returned by "
            "embed_for_matching(..., return_metadata=True)."
        ) from exc

    semantic_config = {
        key: value
        for key, value in embedding_config.items()
        if key not in _NON_SEMANTIC_EMBEDDING_KEYS
    }
    return {
        "embedding_config": semantic_config,
        "model_name": model_name,
        "model_sha": model_sha,
    }


def _prepare_trial_embeddings(
    trial_embeddings: pd.DataFrame,
    *,
    trial_spaces: pd.DataFrame,
) -> pd.DataFrame:
    _require_columns(
        trial_embeddings,
        ["space_trial_id", "embedding"],
        "trial_embeddings",
    )
    embeddings = trial_embeddings[["space_trial_id", "embedding"]].copy()
    _normalize_string_column(
        embeddings,
        "space_trial_id",
        label="trial_embeddings",
    )
    if embeddings["space_trial_id"].duplicated().any():
        raise ValueError("trial_embeddings must contain unique space_trial_id values.")

    expected_ids = set(trial_spaces["space_trial_id"])
    actual_ids = set(embeddings["space_trial_id"])
    missing_ids = sorted(expected_ids - actual_ids)
    extra_ids = sorted(actual_ids - expected_ids)
    if missing_ids or extra_ids:
        details = []
        if missing_ids:
            details.append(f"missing {len(missing_ids)}")
        if extra_ids:
            details.append(f"containing {len(extra_ids)} unexpected")
        raise ValueError(
            "trial_embeddings space_trial_id set does not match trial_spaces "
            f"({', '.join(details)} IDs)."
        )

    order = {value: index for index, value in enumerate(trial_spaces["space_trial_id"])}
    embeddings["_space_order"] = embeddings["space_trial_id"].map(order)
    return (
        embeddings.sort_values("_space_order", kind="stable")
        .drop(columns="_space_order")
        .reset_index(drop=True)
    )


def _empty_space_matches(trial_spaces: pd.DataFrame) -> pd.DataFrame:
    optional_columns = [
        column
        for column in trial_spaces.columns
        if column not in {"space_trial_id", "clinical_space_summary", "embedding"}
    ]
    return pd.DataFrame(
        columns=[
            "patient_id",
            "space_rank",
            "trialspace_rank",
            "space_trial_id",
            "match_quality_score",
            "match_quality_pass",
            "similarity_score",
            "paradigm_count",
            *optional_columns,
            "clinical_space_summary",
        ]
    )


def _rank_paradigms(
    space_paradigm_matches: pd.DataFrame,
    *,
    catalog_columns: list[str],
    patient_order: dict[str, int],
) -> pd.DataFrame:
    output_columns = [
        "patient_id",
        "paradigm_rank",
        *catalog_columns,
        "best_space_rank",
        "best_space_trial_id",
        "best_match_quality_score",
        "best_match_quality_pass",
        "best_similarity_score",
        "supporting_space_count",
        "supporting_space_ids",
        "supporting_space_ranks",
    ]
    if space_paradigm_matches.empty:
        return pd.DataFrame(columns=output_columns)

    ordered = space_paradigm_matches.sort_values(
        ["patient_id", "space_rank", "paradigm_id", "space_trial_id"],
        kind="stable",
    )
    support = (
        ordered.groupby(["patient_id", "paradigm_id"], sort=False)
        .agg(
            supporting_space_count=("space_trial_id", "nunique"),
            supporting_space_ids=(
                "space_trial_id",
                lambda values: tuple(dict.fromkeys(str(value) for value in values)),
            ),
            supporting_space_ranks=(
                "space_rank",
                lambda values: tuple(dict.fromkeys(int(value) for value in values)),
            ),
        )
        .reset_index()
    )
    best = ordered.drop_duplicates(["patient_id", "paradigm_id"], keep="first")
    best_columns = [
        "patient_id",
        *catalog_columns,
        "space_rank",
        "space_trial_id",
        "match_quality_score",
        "match_quality_pass",
        "similarity_score",
    ]
    paradigms = best[best_columns].rename(
        columns={
            "space_rank": "best_space_rank",
            "space_trial_id": "best_space_trial_id",
            "match_quality_score": "best_match_quality_score",
            "match_quality_pass": "best_match_quality_pass",
            "similarity_score": "best_similarity_score",
        }
    )
    paradigms = paradigms.merge(
        support,
        on=["patient_id", "paradigm_id"],
        how="left",
        validate="one_to_one",
    )
    paradigms["_patient_order"] = paradigms["patient_id"].map(patient_order)
    paradigms = paradigms.sort_values(
        [
            "_patient_order",
            "best_space_rank",
            "supporting_space_count",
            "paradigm_id",
        ],
        ascending=[True, True, False, True],
        kind="stable",
    )
    paradigms["paradigm_rank"] = (
        paradigms.groupby("patient_id", sort=False).cumcount() + 1
    )
    return paradigms.drop(columns="_patient_order")[output_columns].reset_index(
        drop=True
    )


def rank_patient_space_paradigms(
    patient_summaries: pd.DataFrame,
    trial_spaces: pd.DataFrame,
    space_paradigm_memberships: pd.DataFrame,
    paradigm_catalog: pd.DataFrame,
    *,
    config: MMAIConfig | None = None,
    retrieval_k: int = 100,
    top_space_count: int = 10,
    require_match_quality_pass: bool = True,
    trial_embeddings: pd.DataFrame | None = None,
    trial_embedding_metadata: dict[str, Any] | None = None,
) -> PatientParadigmRankingResult:
    """Match patient summaries to trial spaces and roll them up to paradigms.

    TrialSpace first retrieves ``retrieval_k`` candidate spaces for each
    patient. TrialChecker scores that wider pool, and the leading
    ``top_space_count`` spaces after TrialChecker reranking are joined to the
    exact space-to-paradigm membership graph. A broad or disjunctive space can
    therefore support more than one paradigm.

    Parameters
    ----------
    patient_summaries : pandas.DataFrame
        One row per patient with ``patient_id`` and ``cancer_history_summary``.
    trial_spaces : pandas.DataFrame
        One row per searchable space with ``space_trial_id`` and
        ``clinical_space_summary``. Additional trial-only columns are retained
        in ``space_matches``.
    space_paradigm_memberships : pandas.DataFrame
        Exact lineage edges with ``space_trial_id`` and ``paradigm_id``.
    paradigm_catalog : pandas.DataFrame
        One row per paradigm with ``paradigm_id``, the one-line
        ``paradigm_label``, and any optional status, ontology, or report fields.
    config : MMAIConfig, optional
        TrialSpace and TrialChecker configuration. Uses the default preset when
        omitted.
    retrieval_k : int, default 100
        TrialSpace candidates to send to TrialChecker per patient.
    top_space_count : int, default 10
        TrialChecker-reranked spaces retained per patient before roll-up.
    require_match_quality_pass : bool, default True
        When True, retain up to ``top_space_count`` spaces that meet the
        configured TrialChecker cutoff. This can yield fewer than the requested
        count, including zero. Set False only when below-cutoff rows are useful
        for research diagnostics.
    trial_embeddings : pandas.DataFrame, optional
        Cached TrialSpace embeddings for the exact ``trial_spaces`` ID set.
    trial_embedding_metadata : dict, optional
        Metadata returned by ``embed_for_matching`` when creating cached trial
        embeddings. Required with ``trial_embeddings`` and checked against the
        freshly embedded patient model and semantic embedding configuration.

    Returns
    -------
    PatientParadigmRankingResult
        Ranked trial spaces, exact membership rows, unique paradigm rankings,
        reusable trial embeddings, and model/run metadata.

    Notes
    -----
    The paradigm rank is ordered by the best supporting TrialChecker-reranked
    space, then by the number of retained spaces supporting the paradigm. It is
    not a calibrated paradigm-level probability or eligibility judgment.
    """
    retrieval_count = _positive_integer(retrieval_k, label="retrieval_k")
    retained_count = _positive_integer(top_space_count, label="top_space_count")
    if retrieval_count < retained_count:
        raise ValueError(
            "retrieval_k must be greater than or equal to top_space_count."
        )
    if (trial_embeddings is None) != (trial_embedding_metadata is None):
        raise ValueError(
            "trial_embeddings and trial_embedding_metadata must be supplied together."
        )
    if not isinstance(require_match_quality_pass, bool):
        raise ValueError("require_match_quality_pass must be a boolean.")

    patients, spaces, memberships, catalog = _normalize_inputs(
        patient_summaries,
        trial_spaces,
        space_paradigm_memberships,
        paradigm_catalog,
    )
    resolved_config = config or load_default_preset()

    patient_embedding_result = embed_for_matching(
        patients,
        entity_type="patient",
        config=resolved_config,
        return_metadata=True,
    )
    patient_embeddings, patient_embedding_metadata = patient_embedding_result

    reused_trial_embeddings = trial_embeddings is not None
    if trial_embeddings is None:
        trial_embedding_result = embed_for_matching(
            spaces,
            entity_type="trial",
            config=resolved_config,
            return_metadata=True,
        )
        resolved_trial_embeddings, resolved_trial_metadata = trial_embedding_result
    else:
        resolved_trial_embeddings = _prepare_trial_embeddings(
            trial_embeddings,
            trial_spaces=spaces,
        )
        assert trial_embedding_metadata is not None
        resolved_trial_metadata = trial_embedding_metadata

    patient_signature = _embedding_metadata_signature(patient_embedding_metadata)
    trial_signature = _embedding_metadata_signature(resolved_trial_metadata)
    if patient_signature != trial_signature:
        raise ValueError(
            "Patient and trial embeddings are incompatible. Regenerate every trial "
            "embedding with the configured TrialSpace model, revision, prompt, and "
            "maximum sequence length."
        )

    retrieval = generate_candidate_matches(
        patient_embeddings,
        resolved_trial_embeddings,
        k=retrieval_count,
    ).rename(columns={"rank": "trialspace_rank"})
    patient_order = {
        patient_id: index for index, patient_id in enumerate(patients["patient_id"])
    }

    if retrieval.empty:
        space_matches = _empty_space_matches(spaces)
        space_paradigm_matches = pd.DataFrame()
        paradigm_matches = _rank_paradigms(
            space_paradigm_matches,
            catalog_columns=list(catalog.columns),
            patient_order=patient_order,
        )
        checker_metadata: dict[str, Any] = {}
        scored_candidate_count = 0
        passing_candidate_count = 0
    else:
        candidate_pairs = retrieval.merge(
            patients[_PATIENT_COLUMNS],
            on="patient_id",
            how="left",
            validate="many_to_one",
        ).merge(
            spaces[_TRIAL_COLUMNS],
            on="space_trial_id",
            how="left",
            validate="many_to_one",
        )
        checker_result = score_match_quality(
            candidate_pairs,
            config=resolved_config,
            filter_low_quality=False,
            return_metadata=True,
        )
        checker_scores, checker_metadata = checker_result
        scored = retrieval.merge(
            checker_scores,
            on=["patient_id", "space_trial_id"],
            how="left",
            validate="one_to_one",
        ).merge(
            spaces,
            on="space_trial_id",
            how="left",
            validate="many_to_one",
        )
        if scored[["match_quality_score", "match_quality_pass"]].isna().any().any():
            raise ValueError(
                "TrialChecker did not return one score for every candidate."
            )
        scored_candidate_count = len(scored)
        passing_candidate_count = int(scored["match_quality_pass"].sum())
        rankable = (
            scored.loc[scored["match_quality_pass"]].copy()
            if require_match_quality_pass
            else scored
        )

        rankable["_patient_order"] = rankable["patient_id"].map(patient_order)
        rankable = rankable.sort_values(
            [
                "_patient_order",
                "match_quality_score",
                "similarity_score",
                "trialspace_rank",
                "space_trial_id",
            ],
            ascending=[True, False, False, True, True],
            kind="stable",
        )
        space_matches = (
            rankable.groupby("patient_id", sort=False)
            .head(retained_count)
            .copy()
            .reset_index(drop=True)
        )
        space_matches["space_rank"] = (
            space_matches.groupby("patient_id", sort=False).cumcount() + 1
        )
        paradigm_counts = memberships.groupby("space_trial_id")["paradigm_id"].nunique()
        space_matches["paradigm_count"] = (
            space_matches["space_trial_id"].map(paradigm_counts).fillna(0).astype(int)
        )
        optional_trial_columns = [
            column
            for column in spaces.columns
            if column not in {"space_trial_id", "clinical_space_summary", "embedding"}
        ]
        space_columns = [
            "patient_id",
            "space_rank",
            "trialspace_rank",
            "space_trial_id",
            "match_quality_score",
            "match_quality_pass",
            "similarity_score",
            "paradigm_count",
            *optional_trial_columns,
            "clinical_space_summary",
        ]
        space_matches = space_matches.drop(columns="_patient_order")[space_columns]

        space_match_keys = ["patient_id", "space_trial_id"]
        if space_matches.duplicated(space_match_keys).any():
            raise ValueError(
                "Retained space matches must contain unique "
                "patient_id/space_trial_id pairs."
            )

        mapped_memberships = memberships[
            memberships["space_trial_id"].isin(space_matches["space_trial_id"])
        ]
        # The same space can be retained for several patients while also
        # belonging to several paradigms. The join is therefore many-to-many
        # on space_trial_id even though both composite row identities remain
        # unique: patient_id/space_trial_id on the left and
        # space_trial_id/paradigm_id on the right.
        space_paradigm_matches = space_matches.merge(
            mapped_memberships,
            on="space_trial_id",
            how="inner",
            validate="many_to_many",
        ).merge(
            catalog,
            on="paradigm_id",
            how="left",
            validate="many_to_one",
        )
        membership_columns = [
            "patient_id",
            "space_rank",
            "trialspace_rank",
            "space_trial_id",
            "match_quality_score",
            "match_quality_pass",
            "similarity_score",
            *list(catalog.columns),
        ]
        space_paradigm_matches = (
            space_paradigm_matches[membership_columns]
            .sort_values(
                ["patient_id", "space_rank", "paradigm_id"],
                kind="stable",
            )
            .reset_index(drop=True)
        )
        paradigm_matches = _rank_paradigms(
            space_paradigm_matches,
            catalog_columns=list(catalog.columns),
            patient_order=patient_order,
        )

    metadata = {
        "schema_version": "1.0",
        "research_use_notice": RESEARCH_USE_NOTICE,
        "ranking_method": (
            "TrialSpace top-k retrieval; TrialChecker descending rerank; "
            "paradigm rank by best supporting space then supporting-space count"
        ),
        "retrieval_k": retrieval_count,
        "top_space_count": retained_count,
        "require_match_quality_pass": require_match_quality_pass,
        "trial_embeddings_reused": reused_trial_embeddings,
        "input_counts": {
            "patients": len(patients),
            "trial_spaces": len(spaces),
            "space_paradigm_memberships": len(memberships),
            "paradigms": len(catalog),
        },
        "output_counts": {
            "trialchecker_scored_candidates": scored_candidate_count,
            "trialchecker_passing_candidates": passing_candidate_count,
            "space_matches": len(space_matches),
            "space_paradigm_matches": len(space_paradigm_matches),
            "paradigm_matches": len(paradigm_matches),
        },
        "model_metadata": {
            "embedding_model": patient_signature,
            "match_quality_checker": checker_metadata.get("model_metadata", {}).get(
                "match_quality_checker"
            ),
        },
    }
    return PatientParadigmRankingResult(
        space_matches=space_matches,
        space_paradigm_matches=space_paradigm_matches,
        paradigm_matches=paradigm_matches,
        trial_embeddings=resolved_trial_embeddings,
        trial_embedding_metadata=resolved_trial_metadata,
        metadata=metadata,
    )
