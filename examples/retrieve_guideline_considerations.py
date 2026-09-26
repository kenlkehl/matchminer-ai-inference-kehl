"""Retrieve catalog considerations for one stored patient summary; write local review files.

This example reads user-provided data. No patient summaries or guideline outputs
are bundled with the code. Use an appropriate external output directory.
"""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--patient-summaries", type=Path, required=True)
    parser.add_argument("--patient-id", required=True)
    parser.add_argument("--patient-id-column", default="patient_id")
    parser.add_argument("--summary-column", default="cancer_history_summary")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--model-cache-dir", type=Path)
    parser.add_argument("--metadata-cache-dir", type=Path)
    parser.add_argument(
        "--embedding-cache-dir",
        type=Path,
        help="External directory for reusable guideline vectors (no patient vectors)",
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--synthetic-notes",
        type=Path,
        help="Optional synthetic note table to verify the selected source ID",
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    repository = Path(__file__).resolve().parents[1]
    if output.is_relative_to(repository):
        parser.error("Output must be outside the code repository")
    if args.model_cache_dir:
        os.environ["HF_HUB_CACHE"] = str(args.model_cache_dir)
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch

    from matchminer_ai import load_default_preset
    from matchminer_ai._storage import atomic_json, digest
    from matchminer_ai.matching import (
        retrieve_guideline_considerations,
        write_guideline_considerations_report,
    )

    torch.set_num_threads(args.cpu_threads)
    columns = [args.patient_id_column, args.summary_column]
    # Read just the two summary columns; unrelated raw-note/trial fields never enter inference.
    source = pd.read_parquet(args.patient_summaries, columns=columns)
    selected = source.loc[
        source[args.patient_id_column].astype(str) == args.patient_id, columns
    ]
    if len(selected) != 1:
        parser.error("The source must contain exactly one row for this patient ID")
    patients = selected.rename(
        columns={
            args.patient_id_column: "patient_id",
            args.summary_column: "cancer_history_summary",
        }
    ).copy()
    patients["patient_id"] = patients["patient_id"].astype(str)
    provenance = {
        "source_file": str(args.patient_summaries.resolve()),
        "source_id": args.patient_id,
        "source_id_column": args.patient_id_column,
        "source_summary_column": args.summary_column,
        "summary_sha256": digest(
            patients.iloc[0].cancer_history_summary.encode("utf-8")
        ),
        "summary_changed": False,
    }
    if args.synthetic_notes:
        schema = pq.read_schema(args.synthetic_notes)
        id_type = schema.field(args.patient_id_column).type
        value = (
            int(args.patient_id) if pa.types.is_integer(id_type) else args.patient_id
        )
        note_ids = pq.read_table(
            args.synthetic_notes,
            columns=[args.patient_id_column],
            filters=[(args.patient_id_column, "=", value)],
        )
        if not note_ids.num_rows:
            parser.error(
                "Selected summary ID has no records in the supplied synthetic note table"
            )
        provenance.update(
            data_kind="synthetic",
            synthetic_note_file=str(args.synthetic_notes.resolve()),
            linked_synthetic_note_rows=note_ids.num_rows,
        )
    config = load_default_preset()
    config.embedding["device"] = args.device
    config.raw["match_quality"]["device"] = args.device
    config.model_metadata_cache_dir = str(
        args.metadata_cache_dir or output / "model_metadata"
    )
    result, metadata = retrieve_guideline_considerations(
        patients,
        args.catalog,
        top_n=args.top_n,
        candidate_k=args.candidate_k,
        config=config,
        embedding_cache_dir=args.embedding_cache_dir,
        return_metadata=True,
        progress_callback=lambda text: print(text, flush=True),
    )
    report = write_guideline_considerations_report(
        result,
        patients,
        output / "patient_considerations.md",
        metadata=metadata,
        patient_provenance=provenance,
    )
    result.to_json(
        output / "ranked_spaces.jsonl",
        orient="records",
        lines=True,
        force_ascii=False,
        double_precision=15,
    )
    atomic_json(output / "retrieval_metadata.json", metadata)
    atomic_json(output / "patient_source.json", provenance)
    patients.to_json(
        output / "patient_summary.json", orient="records", indent=2, force_ascii=False
    )
    print(
        json.dumps({"report": str(report), "selected_spaces": len(result)}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
