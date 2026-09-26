"""Write a local, source-preserving review report from ranked guideline records."""

from __future__ import annotations

import html
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import quote

from matchminer_ai._storage import atomic_text

if TYPE_CHECKING:
    import pandas as pd


def _text(value):
    return re.sub(
        r"([\\`*_\[\]|#])", r"\\\1", html.escape(str(value), quote=False)
    ).replace("\n", " ")


def write_guideline_considerations_report(
    matches: pd.DataFrame,
    patient_summaries: pd.DataFrame,
    output_path: str | Path,
    *,
    metadata: dict | None = None,
    patient_provenance: dict | None = None,
) -> Path:
    """Write ranked diagnostic/treatment menus and evidence to external Markdown.

    ``matches`` is returned by ``retrieve_guideline_considerations``. Supply the
    same patient summary DataFrame to display the actual retrieval input. Optional
    retrieval metadata records model/catalog provenance; ``patient_provenance``
    can record a synthetic source file, source identifier, and selection method.
    Menus, caveats, and quotations are rendered from the catalog without a model
    rewrite. Source PDF links are local and are included only when files exist.

    The required output path must be outside this package/code repository and
    the recorded guideline source collections. The file contains patient text
    and source-derived guideline content; keep it in an appropriate local data
    directory, outside public packages and source control.
    """
    from .guidelines import _patient_frame
    from matchminer_ai.trials._guideline_schema import FIELDS

    patients = _patient_frame(patient_summaries)
    required = {
        "patient_id",
        "space_trial_id",
        "rank",
        "retrieval_rank",
        "similarity_score",
        "match_quality_score",
        "match_quality_pass",
        "name",
        "space",
        "source",
        "diagnostic_workup",
        "treatment_options",
        "evidence",
        "uncertainties",
    }
    if not required.issubset(matches.columns) or matches.empty:
        raise ValueError(
            "matches must contain nonempty ranked guideline consideration records."
        )
    if not set(matches["patient_id"]).issubset(set(patients["patient_id"])):
        raise ValueError("Every returned patient must have a supplied patient summary.")
    output = Path(output_path).resolve()
    if output.suffix.lower() not in {".md", ".markdown"}:
        raise ValueError("output_path must name a Markdown file.")
    package = Path(__file__).resolve().parents[1]
    roots = [package]
    if (package.parent.parent / "pyproject.toml").exists():
        roots.append(package.parent.parent)
    roots += [
        Path(source["input_directory"]).resolve().parent.parent
        for source in matches["source"]
        if source.get("input_directory")
    ]
    if any(output.is_relative_to(root) for root in roots):
        raise ValueError(
            "Report output must stay outside the code repository and guideline source collections."
        )
    lines = [
        "# Patient-specific guideline catalog retrieval",
        "",
        "Research review only. TrialSpace retrieves candidate populations; TrialChecker ranks their fit. "
        "Scores are prioritization signals, not probabilities or treatment recommendations. "
        "The menus below are copied from the stored catalog, with their conditions and uncertainties. "
        "They have not been individually adjudicated for this patient. Source and clinician review are required.",
        "",
    ]
    if metadata:
        retrieval = metadata.get("retrieval", {})
        lines += [
            "## Retrieval provenance",
            "",
            f"- Catalog spaces: {metadata.get('catalog', {}).get('catalog_spaces', 'unavailable')}",
            f"- TrialSpace candidates per patient: {retrieval.get('candidate_k') or 'all catalog spaces'}",
            f"- Requested top N by TrialChecker: {retrieval.get('top_n', 'unavailable')}",
            f"- TrialChecker cutoff flag: {retrieval.get('score_cutoff', 'unavailable')}; no candidates removed by that cutoff.",
            f"- Catalog fingerprint: `{metadata.get('catalog', {}).get('catalog_sha256', 'unavailable')}`",
        ]
        for role, model in metadata.get("model_metadata", {}).items():
            lines.append(
                f"- {_text(role)}: {_text(model.get('model_name', 'unavailable'))}; revision `{model.get('model_sha', 'unavailable')}`"
            )
        for artifact in metadata.get("catalog", {}).get("artifacts", []):
            lines.append(
                f"- Catalog file: {_text(artifact.get('path', 'DataFrame input'))}; stored audit: {_text(artifact.get('recorded_source_audit', 'unavailable'))}"
            )
        lines.append("")
    if patient_provenance:
        lines += ["## Patient source", ""]
        lines += [
            f"- {_text(key)}: {_text(value)}"
            for key, value in patient_provenance.items()
        ]
        lines.append("")
    for patient in patients.to_dict("records"):
        ranked = matches.loc[
            matches["patient_id"] == patient["patient_id"]
        ].sort_values("rank")
        if ranked.empty:
            continue
        lines += [
            f"## Patient {_text(patient['patient_id'])}",
            "",
            "### Summary used for retrieval",
            "",
        ]
        lines += [
            "> " + _text(line) + "  "
            for line in patient["cancer_history_summary"].splitlines()
        ]
        lines += [
            "",
            "### Ranked catalog spaces",
            "",
            "| Rank | Catalog population | TrialChecker score | Cutoff flag | TrialSpace cosine | Retrieval rank |",
            "| --- | --- | ---: | --- | ---: | ---: |",
        ]
        for row in ranked.to_dict("records"):
            lines.append(
                f"| {row['rank']} | {_text(row['name'])} | {row['match_quality_score']:.4f} | "
                f"{'Pass' if row['match_quality_pass'] else 'Below cutoff'} | {row['similarity_score']:.4f} | {row['retrieval_rank']} |"
            )
        lines.append("")
        for row in ranked.to_dict("records"):
            source = row["source"]
            lines += [
                f"### {row['rank']}. {_text(row['name'])}",
                "",
                f"Space ID: `{row['space_trial_id']}`",
                "",
                f"Source: {_text(source['title'])}, {_text(source['version'])}.",
                "",
                "**Catalog population**",
                "",
            ]
            lines += [
                f"- **{label}:** {_text(row['space'][key])}"
                for key, label in FIELDS.items()
            ]
            lines.append("")
            pdf = None
            if source.get("input_directory") and source.get("source_pdf"):
                candidate = (
                    Path(source["input_directory"]).parent.parent / source["source_pdf"]
                )
                if candidate.is_file():
                    pdf = candidate
            evidence_by_key = {}

            def citations(evidence):
                labels = []
                for item in evidence:
                    key = (item["page_id"], tuple(item["line_ids"]), item["quote"])
                    evidence_by_key[key] = item
                    address = (
                        f"{item['page_id']} ({item.get('printed_label', 'unlabeled')}); lines "
                        + ", ".join(map(str, item["line_ids"]))
                    )
                    if pdf is not None and item.get("pdf_page"):
                        link = quote(os.path.relpath(pdf, output.parent), safe="/.-_")
                        labels.append(
                            f"[{_text(address)}]({link}#page={item['pdf_page']})"
                        )
                    else:
                        labels.append(_text(address))
                return "; ".join(labels)

            lines += ["Population evidence: " + citations(row["evidence"]), ""]
            for key, title in (
                ("diagnostic_workup", "Diagnostic / workup considerations"),
                ("treatment_options", "Treatment / management considerations"),
            ):
                lines += [f"**{title}**", ""]
                if not row[key]:
                    lines += [
                        "No items were recorded in this catalog for this population.",
                        "",
                    ]
                for item in row[key]:
                    lines += [
                        f"- **{_text(item['name'])}** — **Conditions:** {_text(item['conditions'])}. "
                        f"**Category:** {_text(item['category'])}. **Sources:** {citations(item['evidence'])}."
                    ]
                lines.append("")
            lines += ["**Catalog uncertainties and context limits**", ""]
            lines += [f"- {_text(value)}" for value in row["uncertainties"]] or [
                "- No uncertainties were recorded; this does not establish completeness."
            ]
            omitted = row.get("omitted_source_page_ids", [])
            if omitted:
                lines += [
                    f"- Extraction detail context omitted {len(omitted)} source pages.",
                    "",
                    "<details>",
                    "<summary>Omitted extraction context pages</summary>",
                    "",
                    ", ".join(map(_text, omitted)),
                    "",
                    "</details>",
                ]
            lines += [
                "",
                "<details>",
                "<summary>Stored source excerpts for this population and its considerations</summary>",
                "",
            ]
            for item in evidence_by_key.values():
                lines += [
                    f"**{_text(item['page_id'])}; lines {', '.join(map(str, item['line_ids']))}**",
                    "",
                ]
                lines += [
                    "> " + _text(line) + "  " for line in item["quote"].splitlines()
                ]
                lines.append("")
            lines += ["</details>", ""]
    atomic_text(output, "\n".join(lines))
    return output
