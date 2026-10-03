"""Opt-in population workup review and explicit, portable cohort archives.

No sampling policy, patient paths, or endpoint defaults are defined here. Source
notes stay attached to their patient IDs. Compression and encoding are separate
caller-controlled stages. Archives contain sensitive data; writes are explicit.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import zipfile
from dataclasses import dataclass, field, replace
from pathlib import Path

import pandas as pd

from matchminer_ai.cancellation import check_cancelled, InferenceCancelled
from matchminer_ai.matching.guidelines import _patient_frame
from matchminer_ai.patients.workup import STATUSES, APPLICABILITY, NOTICE

POPULATION_NOTICE = (
    NOTICE + " Only ColBERT-retrieved excerpts were reviewed. Missing documentation "
    "means not located in those excerpts. TrialChecker scores are ranking signals. "
    "Research use only; confirm applicability, timing and findings with a clinician."
)


def _records(frame):
    return json.loads(
        frame.to_json(orient="records", date_format="iso", force_ascii=False)
    )


def _json(value):
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True
    ).encode()


@dataclass(frozen=True)
class PatientCohort:
    summaries: pd.DataFrame
    notes: pd.DataFrame
    provenance: dict = field(default_factory=dict)
    compressed_notes: pd.DataFrame | None = None
    compression: dict | None = None
    indexes: dict = field(default_factory=dict)
    index_source: str | None = None
    space: dict | None = None
    matches: list[dict] = field(default_factory=list)
    report: dict | None = None

    def selected_notes(self, source="original"):
        if source == "original":
            return self.notes
        if source != "compressed" or self.compressed_notes is None:
            raise ValueError("Compress the cohort first, or select original notes.")
        return self.compressed_notes.drop(columns="note_text").rename(
            columns={"compressed_note_text": "note_text"}
        )


def create_patient_cohort(summaries, notes, *, provenance=None):
    """Validate caller-supplied tables; preserve stable dates/types and original text.

    summaries requires unique patient_id and nonempty cancer_history_summary.
    notes requires patient_id and note_text; note_date/note_type are optional.
    Every summary needs notes; notes with unknown IDs are rejected. No sampling,
    training-column inference, text summarization, or model calls occur here.
    """
    if not isinstance(summaries, pd.DataFrame) or not summaries.columns.is_unique:
        raise ValueError("Supply a summaries DataFrame with unique columns.")
    patients = _patient_frame(summaries)
    for col in ("general_exclusion_criteria_evidence",):
        if col in summaries:
            patients[col] = summaries[col].fillna("").to_numpy()
    if not isinstance(notes, pd.DataFrame) or not notes.columns.is_unique:
        raise ValueError("Supply a notes DataFrame with unique columns.")
    if not {"patient_id", "note_text"} <= set(notes) or notes.patient_id.isna().any():
        raise ValueError("Notes require patient_id and note_text.")
    frame = notes.copy()
    frame["patient_id"] = frame.patient_id.astype(str)
    if set(frame.patient_id) != set(patients.patient_id):
        raise ValueError(
            "Every summary must have notes, and every note must have a summary."
        )
    if not all(isinstance(t, str) and t.strip() for t in frame.note_text):
        raise ValueError("Every note must have nonempty text.")
    if "note_date" not in frame:
        frame["note_date"] = None
    if "note_type" not in frame:
        frame["note_type"] = None
    if not all(pd.isna(t) or isinstance(t, str) for t in frame.note_type):
        raise ValueError("Note types must be text or missing.")
    dates = pd.to_datetime(frame.note_date, errors="coerce", format="mixed", utc=True)
    supplied = frame.note_date.notna() & frame.note_date.astype(str).str.strip().ne("")
    if (dates.isna() & supplied).any():
        raise ValueError("Invalid note dates; leave unavailable dates empty.")
    frame["note_date"] = pd.Series(
        [None if pd.isna(d) else d.isoformat() for d in dates],
        index=frame.index,
        dtype=object,
    )
    frame["note_type"] = frame.note_type.astype(object).where(
        frame.note_type.notna(), None
    )
    frame = (
        frame[["patient_id", "note_text", "note_date", "note_type"]]
        .sort_values(["patient_id", "note_date"], kind="stable", na_position="last")
        .reset_index(drop=True)
    )
    return PatientCohort(
        patients.reset_index(drop=True), frame, copy.deepcopy(provenance or {})
    )


def subset_patient_cohort(
    cohort, patient_ids, *, space=None, matches=None, provenance=None
):
    ids = list(patient_ids)
    if len(ids) != len(set(ids)) or not set(ids) <= set(cohort.summaries.patient_id):
        raise ValueError("Cohort selection contains duplicate or unknown patient IDs.")
    summaries = cohort.summaries.set_index("patient_id").loc[ids].reset_index()
    # An empty retrieval is a valid result; preparation/review reject empty cohorts.
    return PatientCohort(
        summaries,
        cohort.notes.loc[cohort.notes.patient_id.isin(ids)].copy(),
        {**copy.deepcopy(cohort.provenance), **copy.deepcopy(provenance or {})},
        space=copy.deepcopy(space),
        matches=copy.deepcopy(matches or []),
    )


def compress_patient_cohort(cohort, *, config, thinking="off", progress_callback=None):
    from matchminer_ai.patients import compress_patient_notes

    if cohort.notes.empty:
        raise ValueError("Select a nonempty cohort first.")
    progress = progress_callback or (lambda _: None)
    compressed = compress_patient_notes(
        cohort.notes,
        config=config,
        thinking=thinking,
        max_concurrent_requests=8,
        progress_callback=lambda done, total: progress(
            f"Compressing cohort notes: {done}/{total}"
        ),
    )
    check_cancelled()
    metadata = {
        **compressed.attrs.get("note_compression", {}),
        "losslessness_verified": False,
        "input_characters": int(cohort.notes.note_text.str.len().sum()),
        "output_characters": int(compressed.compressed_note_text.str.len().sum()),
        "note_count": len(compressed),
    }
    # Recompression makes compressed vectors and prior results stale.
    return replace(
        cohort,
        compressed_notes=compressed,
        compression=metadata,
        indexes={} if cohort.index_source == "compressed" else cohort.indexes,
        index_source=None
        if cohort.index_source == "compressed"
        else cohort.index_source,
        report=None,
    )


def encode_patient_cohort(
    cohort, *, config, note_source="original", progress_callback=None
):
    from matchminer_ai.patients import encode_patient_notes_colbert

    notes = cohort.selected_notes(note_source)
    if notes.empty:
        raise ValueError("Select a nonempty cohort first.")
    indexes = encode_patient_notes_colbert(
        notes, config=config, progress_callback=progress_callback
    )
    check_cancelled()
    return replace(cohort, indexes=indexes, index_source=note_source, report=None)


def aggregate_population_workup(reviews, recommendations):
    """Count each patient once per item, retaining applicability × status counts.

    completed_applicable / applicable_assessed is a documentation completion
    fraction only. Uncertain/not-applicable/error findings never enter that
    denominator; they remain separately visible, as do all status counts.
    """
    rows = []
    for number, recommendation in enumerate(recommendations):
        counts = {status: 0 for status in (*STATUSES, "error")}
        applicability = {value: 0 for value in APPLICABILITY}
        cross = {value: dict(counts) for value in APPLICABILITY}
        for patient in reviews:
            items = patient["review"]["assessments"]
            if len(items) != len(recommendations):
                raise ValueError("Incomplete population review matrix.")
            row = items[number]
            status, applies = row["status"], row["applicability"]
            if status not in counts or applies not in cross:
                raise ValueError("Invalid population workup finding.")
            counts[status] += 1
            applicability[applies] += 1
            cross[applies][status] += 1
        denominator = sum(cross["applies"][s] for s in STATUSES)
        completed = cross["applies"]["completed"]
        rows.append(
            {
                "recommendation_index": number,
                "recommendation": copy.deepcopy(recommendation),
                "patient_count": len(reviews),
                "statuses": counts,
                "applicability": applicability,
                "status_by_applicability": cross,
                "applicable_assessed": denominator,
                "completed_applicable": completed,
                "documented_completion_fraction": completed / denominator
                if denominator
                else None,
            }
        )
    return rows


def review_population_workup_with_colbert(
    cohort,
    *,
    config,
    top_n=20,
    retrieval_config=None,
    backup_config=None,
    max_consecutive_failures=3,
    progress_callback=None,
):
    """Review every selected patient × workup item, preserving isolated evidence.

    Patients run serially to bound endpoint and memory load. All expected pairs
    remain in the result, including errors; cancellation propagates and never
    publishes a partial run as complete. No implicit compression or encoding.
    """
    from matchminer_ai.patients import review_patient_workup_with_colbert

    if type(top_n) is not int or not 1 <= top_n <= 100:
        raise ValueError("top_n must be from 1 to 100.")
    ids = cohort.summaries.patient_id.tolist()
    if not ids or set(cohort.indexes) != set(ids):
        raise ValueError("Encode every active patient's notes with ColBERT first.")
    recommendations = (cohort.space or {}).get("diagnostic_workup", [])
    if not recommendations:
        raise ValueError("Select a guideline space with workup recommendations.")
    notes = cohort.selected_notes(cohort.index_source)
    grouped = dict(tuple(notes.groupby("patient_id", sort=False)))
    for pid in ids:
        index = cohort.indexes[pid]
        if index.patient_id != pid or not index.matches(grouped[pid], retrieval_config):
            raise ValueError(
                "A patient index is stale or incompatible; encode this cohort again."
            )
    progress = progress_callback or (lambda _: None)
    reviews = []
    for i, pid in enumerate(ids, 1):
        check_cancelled()

        def patient_progress(message):
            progress(f"Patient {i}/{len(ids)} · {message}")

        try:
            result = review_patient_workup_with_colbert(
                cohort.indexes[pid],
                recommendations,
                config=config,
                top_n=top_n,
                retrieval_config=retrieval_config,
                backup_config=backup_config,
                max_consecutive_failures=max_consecutive_failures,
                population_context=cohort.space.get("clinical_space_summary"),
                progress_callback=patient_progress,
            )
        except InferenceCancelled:
            raise
        except Exception:
            # Never expose patient/provider exception text. One failure must not
            # disappear from the population denominator or become not_documented.
            result = {
                "assessments": [
                    dict(
                        recommendation_index=n,
                        recommendation=copy.deepcopy(r),
                        name=r["name"],
                        status="error",
                        applicability="uncertain",
                        evidence=[],
                        bottom_line="No validated assessment is available.",
                        limitations=[
                            "Patient review failed; check the endpoint/index and retry."
                        ],
                    )
                    for n, r in enumerate(recommendations)
                ],
                "metadata": {"method": "colbert", "scope": "retrieved_excerpts"},
                "notice": POPULATION_NOTICE,
            }
        result["metadata"].update(note_source=cohort.index_source)
        if cohort.index_source == "compressed":
            result["metadata"].update(
                scope="retrieved_compressed_excerpts",
                evidence_source="llm_compressed_notes",
                compression=copy.deepcopy(cohort.compression),
            )
            result["notice"] += (
                " Compressed-note excerpts may differ from originals; losslessness is unverified."
            )
        reviews.append({"patient_id": pid, "review": result})
        progress(f"Reviewed {i}/{len(ids)} patients")
    check_cancelled()
    return replace(
        cohort,
        report={
            "version": 1,
            "method": "colbert",
            "note_source": cohort.index_source,
            "top_n": top_n,
            "patient_count": len(ids),
            "space": copy.deepcopy(cohort.space),
            "aggregate": aggregate_population_workup(reviews, recommendations),
            "patients": reviews,
            "notice": POPULATION_NOTICE,
            "provenance": copy.deepcopy(cohort.provenance),
        },
    )


def population_results_table(cohort):
    """One export row per patient × workup item, including failures and provenance."""
    if cohort.report is None:
        raise ValueError("Run the population review before exporting results.")
    matches = {r["patient_id"]: r for r in cohort.matches}
    rows = []
    for patient in cohort.report["patients"]:
        pid = patient["patient_id"]
        for item in patient["review"]["assessments"]:
            rows.append(
                dict(
                    patient_id=pid,
                    space_trial_id=cohort.space["space_trial_id"],
                    recommendation_index=item["recommendation_index"],
                    workup=item["recommendation"]["name"],
                    conditions=item["recommendation"].get("conditions", ""),
                    status=item["status"],
                    applicability=item["applicability"],
                    bottom_line=item["bottom_line"],
                    match_quality_score=matches.get(pid, {}).get("match_quality_score"),
                    note_source=cohort.report["note_source"],
                    method="colbert",
                    evidence_json=json.dumps(
                        item.get("evidence", []), ensure_ascii=False
                    ),
                    limitations_json=json.dumps(
                        item.get("limitations", []), ensure_ascii=False
                    ),
                )
            )
    return pd.DataFrame(rows)


def save_patient_cohort(cohort, path):
    """Explicit atomic ZIP export: source notes, summaries, compression, NPZ indexes,
    selected guideline sources, match provenance, and any completed assessments.
    Index filenames are generated numbers, never patient IDs. No pickle is used.
    """
    from matchminer_ai.patients import save_colbert_patient_index

    path = Path(path)
    payload = dict(
        version=1,
        summaries=_records(cohort.summaries),
        notes=_records(cohort.notes),
        provenance=cohort.provenance,
        compressed_notes=None
        if cohort.compressed_notes is None
        else _records(cohort.compressed_notes),
        compression=cohort.compression,
        index_source=cohort.index_source,
        space=cohort.space,
        matches=cohort.matches,
        report=cohort.report,
        indexes={},
    )
    with tempfile.TemporaryDirectory(prefix=".cohort-", dir=path.parent) as tmp:
        archive = Path(tmp) / "cohort.zip"
        hashes = {}
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as out:
            for i, (pid, index) in enumerate(cohort.indexes.items()):
                check_cancelled()
                name = f"indexes/{i:06}.npz"
                file = save_colbert_patient_index(index, Path(tmp) / "index.npz")
                data = file.read_bytes()
                hashes[name] = hashlib.sha256(data).hexdigest()
                out.writestr(name, data, compress_type=zipfile.ZIP_STORED)
                payload["indexes"][pid] = name
            data = _json(payload)
            out.writestr("cohort.json", data)
            hashes["cohort.json"] = hashlib.sha256(data).hexdigest()
            out.writestr(
                "manifest.json",
                _json(
                    {"format": "mmai-patient-cohort", "version": 1, "sha256": hashes}
                ),
            )
        os.chmod(archive, 0o600)
        check_cancelled()
        os.replace(archive, path)
    return path


def load_patient_cohort(path, *, max_uncompressed_bytes=16 * 1024**3):
    """Restore an explicitly supplied archive, with bounded sizes and checksum checks.

    No archive paths are extracted. Indexes are loaded with pickle disabled,
    expected source fingerprints, and exact patient IDs; retrieval subsequently
    validates the actual model signature. Checksums detect corruption, not trust.
    """
    from matchminer_ai.patients import load_colbert_patient_index

    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        names = [e.filename for e in entries]
        if (
            len(names) != len(set(names))
            or len(names) > 100_003
            or sum(e.file_size for e in entries) > max_uncompressed_bytes
        ):
            raise ValueError("Invalid or oversized cohort archive.")
        manifest = json.loads(archive.read("manifest.json"))
        if (
            manifest.get("format") != "mmai-patient-cohort"
            or manifest.get("version") != 1
        ):
            raise ValueError("Unsupported cohort archive.")
        hashes = manifest["sha256"]
        if set(names) != {"manifest.json", *hashes}:
            raise ValueError("Unexpected cohort archive entries.")

        def read(name):
            check_cancelled()
            value = archive.read(name)
            if hashlib.sha256(value).hexdigest() != hashes[name]:
                raise ValueError("Cohort archive checksum mismatch.")
            return value

        payload = json.loads(read("cohort.json"))
        if payload.get("version") != 1:
            raise ValueError("Unsupported cohort data version.")
        cohort = create_patient_cohort(
            pd.DataFrame(payload["summaries"]),
            pd.DataFrame(payload["notes"]),
            provenance=payload["provenance"],
        )
        compressed = payload.get("compressed_notes")
        if compressed is not None:
            compressed = pd.DataFrame(compressed).reindex(
                columns=[*cohort.notes.columns, "compressed_note_text"]
            )
            for column in ("note_date", "note_type"):
                compressed[column] = (
                    compressed[column]
                    .astype(object)
                    .where(compressed[column].notna(), None)
                )
            if not compressed.drop(columns="compressed_note_text").equals(
                cohort.notes
            ) or not all(
                isinstance(t, str) and t.strip()
                for t in compressed.compressed_note_text
            ):
                raise ValueError("Compressed notes do not match the source table.")
        cohort = replace(
            cohort,
            compressed_notes=compressed,
            compression=payload.get("compression"),
            space=payload.get("space"),
            matches=payload.get("matches", []),
            report=payload.get("report"),
            index_source=payload.get("index_source"),
        )
        indexes = {}
        expected = payload.get("indexes", {})
        if expected and set(expected) != set(cohort.summaries.patient_id):
            raise ValueError("Cohort index membership mismatch.")
        if expected and cohort.index_source not in {"original", "compressed"}:
            raise ValueError("Invalid index note source.")
        if set(hashes) != {"cohort.json", *expected.values()} or len(
            set(expected.values())
        ) != len(expected):
            raise ValueError("Unexpected cohort index entries.")
        expanded_index_bytes = 0
        for pid, name in expected.items():
            data = read(name)
            # Bound the nested NPZ expansion as well as the outer ZIP.
            import io

            with zipfile.ZipFile(io.BytesIO(data)) as inner:
                expanded_index_bytes += sum(e.file_size for e in inner.infolist())
                if expanded_index_bytes > max_uncompressed_bytes:
                    raise ValueError("Oversized patient index.")
            notes = cohort.selected_notes(cohort.index_source)
            indexes[pid] = load_colbert_patient_index(
                io.BytesIO(data),
                patient_id=pid,
                expected_notes=notes.loc[notes.patient_id == pid],
            )
        if cohort.report is not None:
            patients = cohort.report["patients"]
            if [
                p["patient_id"] for p in patients
            ] != cohort.summaries.patient_id.tolist():
                raise ValueError("Review membership mismatch.")
            aggregate = aggregate_population_workup(
                patients, cohort.space["diagnostic_workup"]
            )
            cohort = replace(cohort, report={**cohort.report, "aggregate": aggregate})
    return replace(cohort, indexes=indexes)
