"""Extract cohorts and eligibility criteria from a synthetic protocol batch with matchminer_ai.trials.extract_protocol_cohorts.

Each protocol goes through the inference package's three-step workflow, with the
same prompts as ctgov listings (``trials.extract_ctgov_cohorts``): a cancer-treatment
screen, the cohort names, and one request per cohort for its criteria, copied verbatim
with boilerplate included. For training, each trial's cohorts whose criteria are
essentially the same are then combined (merge_equivalent_cohorts; --no-merge-cohorts skips
it), and each merged cohort's boilerplate criteria are removed: each criteria line is
classified in its own request and code sorts the unchanged lines (remove_boilerplate_criteria;
--no-deboilerplate skips it). Exports, with the same
provenance: cohorts_unmerged (as extracted), cohorts (merged, all criteria; they seed synthetic
patients) and cohorts_deboilerplated (the merged rows with their key criteria in
cohort_key_criteria and boilerplate criteria in cohort_boilerplate_criteria, which retrieval,
match, boilerplate and per-criterion labels read), each as .parquet and .csv.

Reads a generate_batch.py output directory: every protocols/<NCT>/protocol.md
whose meta.json records a completed generation, plus the ctgov seeds in
seeds.jsonl. Each trial's decision is saved under <output-dir>/trials/ and
reused on rerun, with merge.json and deboilerplate.json beside it (every step keeps its
answers and reasoning). The exports carry each cohort's provenance: the synthetic
protocol (path, SHA-256, generation prompt hash, layout, generating model and
endpoint), the ClinicalTrials.gov record it was seeded from (including the
original ctgov eligibility criteria and when they were retrieved), and the
extraction prompt hash, schema, model and endpoint. Servers come from
<output-dir>/servers.json, written from --server on first run and re-read while
running, as in generate_batch.py. See synthetic_protocols/README.md.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import httpx
import pandas as pd

import matchminer_ai
from matchminer_ai import load_default_preset
from matchminer_ai.trials import extract_protocol_cohorts, merge_equivalent_cohorts, remove_boilerplate_criteria


SEED_FIELDS = ("title", "brief_summary", "detailed_description", "eligibility_criteria")
CTGOV_FIELDS = (*SEED_FIELDS, "last_update_post_date", "retrieved_at")


def served_model(url):
    response = httpx.get(url.rstrip("/") + "/models", timeout=30)
    response.raise_for_status()
    return response.json()["data"][0]["id"]


def git_revision(path):
    try:
        run = lambda *a: subprocess.run(["git", "-C", str(path), *a], capture_output=True, text=True,
                                        check=True).stdout.strip()
        return {"path": str(path), "commit": run("rev-parse", "HEAD"),
                "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}
    except (OSError, subprocess.CalledProcessError):
        return {"path": str(path), "commit": None, "dirty": None}


def text(value):
    return value if isinstance(value, str) else ""


def load_batch(batch_dir):
    """Completed synthetic protocols in seed order, with generation and ctgov provenance."""
    seeds = [json.loads(line) for line in (batch_dir / "seeds.jsonl").read_text().splitlines() if line.strip()]
    trials, not_generated = [], []
    for seed in seeds:
        trial_dir = batch_dir / "protocols" / seed["nct_id"]
        meta = json.loads((trial_dir / "meta.json").read_text()) if (trial_dir / "meta.json").exists() else {}
        if meta.get("finish_reason") != "stop" or not (trial_dir / "protocol.md").exists():
            not_generated.append(seed["nct_id"])
            continue
        # The protocol must have been written from this exact ctgov record.
        changed = [key for key in SEED_FIELDS if text(meta["seed"].get(key)) != text(seed.get(key))]
        if changed:
            raise ValueError(f"{seed['nct_id']}: meta.json seed differs from seeds.jsonl in {changed}")
        response = meta.get("response_metadata") or {}
        trials.append({
            "trial_id": seed["nct_id"], "protocol_path": str(trial_dir / "protocol.md"),
            "synthetic_prompt_sha256": meta["prompt_sha256"], "synthetic_layout_index": meta["layout_index"],
            "synthetic_attempt": meta["attempt"], "synthetic_completed_at": meta["completed_at"],
            "synthetic_endpoint": response.get("endpoint"), "synthetic_model": response.get("request_model"),
            **{f"ctgov_{key}": seed.get(key) for key in CTGOV_FIELDS},
        })
    trials = pd.DataFrame(trials)
    if trials.empty:
        raise ValueError(f"No completed synthetic protocols in {batch_dir / 'protocols'}")
    if trials.synthetic_prompt_sha256.nunique() > 1:
        raise ValueError("The batch mixes synthetic protocol prompt versions; extract each version separately")
    return trials, not_generated


def trial_provenance(trials, metadata, decisions_dir):
    """One row per synthetic protocol: its sources plus the extraction outcome."""
    rows = []
    for trial_id in trials.trial_id:
        decision = metadata["trial_decisions"].get(trial_id)
        steps = metadata["trial_attempts"].get(trial_id) or {}
        # Requests of every step, in order: screen, cohort names, then each cohort's criteria.
        attempts = [*steps.get("screen", []), *steps.get("list", []),
                    *(a for _, log in sorted((steps.get("criteria") or {}).items(), key=lambda kv: int(kv[0])) for a in log)]
        responses = [a.get("response_metadata") or {} for a in attempts]
        status = ("failed" if decision is None else "cohorts" if decision["cancer_treatment_trial"] == "yes"
                  else "not_cancer_treatment")
        rows.append({
            "trial_id": trial_id, "protocol_sha256": metadata["source_documents"][trial_id]["sha256"],
            "extraction_status": status,
            "cohort_count": len(decision["cohorts"]) if status == "cohorts" else 0,
            "extraction_error": metadata["trial_errors"].get(trial_id),
            "extraction_attempts": len(attempts),
            "extraction_endpoint": ";".join(sorted({r["endpoint"] for r in responses if r.get("endpoint")})) or None,
            "extraction_model": ";".join(sorted({r["request_model"] for r in responses if r.get("request_model")})) or None,
            "extraction_prompt_sha256": metadata["prompt_sha256"],
            "extraction_schema_version": metadata["schema_version"],
            "extraction_decision_path": str(decisions_dir / trial_id / "extraction.json") if decision else None,
        })
    return trials.merge(pd.DataFrame(rows), on="trial_id", validate="one_to_one")


def main(args):
    output_dir = args.output_dir or args.batch_dir / "cohort_extraction"
    output_dir.mkdir(parents=True, exist_ok=True)
    # An externally managed registry (``training.pipeline`` --server-urls-file) is used as is.
    registry = args.server_urls_file or output_dir / "servers.json"
    if not registry.exists():
        servers = []
        for spec in args.server:
            url, _, cap = spec.partition("=")
            url = url.rstrip("/") + ("" if url.rstrip("/").endswith("/v1") else "/v1")
            servers.append({"url": url, "model": served_model(url), "upstream_concurrency": int(cap or 64)})
        registry.write_text(json.dumps({"ready": True, "servers": servers}, indent=2))
    servers = json.loads(registry.read_text())["servers"]
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}] servers: " + ", ".join(
        f"{s['url']} ({s.get('model', '?').split('/')[-1]}, cap {s.get('upstream_concurrency', 'adaptive')})" for s in servers), flush=True)

    trials, not_generated = load_batch(args.batch_dir)
    if args.limit:
        trials = trials.head(args.limit)
    print(f"{len(trials)} synthetic protocols to extract; {len(not_generated)} seeds have no completed protocol",
          flush=True)
    config = load_default_preset()
    config.remote.update(enabled=True, server_urls=[], server_urls_file=str(registry), request_timeout=7200)
    config.remote["adaptive_concurrency"] = {**config.remote.get("adaptive_concurrency", {}),
                                             "enabled": True, "metrics_interval": 10}
    settings = config.protocol_cohort_extraction
    # Extraction, merge and boilerplate classification all send these chunks.
    settings["chunk_size"], settings["chunks_in_flight"] = args.chunk_size, args.chunks_in_flight
    # The runtime model selects the sampling profile; each server's own model ID comes from the registry.
    settings["remote"]["model_name"] = args.model_name
    settings["remote"]["request_params"]["max_tokens"] = args.max_tokens
    decisions_dir = output_dir / "trials"
    # Record the code that runs, before the checkout can change during a long run.
    repositories = {"repository": git_revision(Path(matchminer_ai.__file__).resolve().parents[2])}
    started_at, started = datetime.now(timezone.utc).isoformat(timespec="seconds"), time.time()
    cohorts, metadata = extract_protocol_cohorts(
        trials[["trial_id", "protocol_path"]], config=config, output_dir=decisions_dir, return_metadata=True,
        progress_callback=lambda stage, done, total, detail: None if stage == "read" and done < total else print(
            f"[{datetime.now(timezone.utc):%H:%M:%S} +{(time.time() - started) / 60:,.1f} min] {stage} {detail}",
            flush=True),
    )

    # The decisions are also in metadata. pandas deep-copies attrs into every copy and group, which
    # stalled the merge of 5,000 trials on CPU before it sent a request.
    cohorts.attrs = {}
    merging, unmerged = None, cohorts if args.merge_cohorts else None
    if args.merge_cohorts and not cohorts.empty:
        # Saved groupings under trials/<NCT>/merge.json are reused for unchanged cohorts.
        cohorts, merging = merge_equivalent_cohorts(cohorts, config=config, section="protocol_cohort_extraction",
                                                    output_dir=decisions_dir, return_metadata=True)
    cleaning, clean = None, None
    if args.deboilerplate:
        # Saved results under trials/<NCT>/deboilerplate.json are reused for unchanged cohorts.
        clean, cleaning = (remove_boilerplate_criteria(cohorts, config=config, section="protocol_cohort_extraction",
                                                       output_dir=decisions_dir, return_metadata=True)
                           if not cohorts.empty else (cohorts, None))
    provenance = trial_provenance(trials, metadata, decisions_dir)

    def with_provenance(rows):
        counts = rows.groupby("trial_id").size()
        trial_rows = provenance.assign(cohort_count=provenance.trial_id.map(counts).fillna(0).astype(int))
        return rows.assign(protocol_origin="synthetic").merge(trial_rows, on="trial_id", validate="many_to_one")

    cohorts = with_provenance(cohorts)
    provenance["cohort_count"] = provenance.trial_id.map(cohorts.groupby("trial_id").size()).fillna(0).astype(int)
    exports = [("cohorts", cohorts), ("trials", provenance)]
    if unmerged is not None:
        exports.append(("cohorts_unmerged", with_provenance(unmerged)))
    if clean is not None:
        exports.append(("cohorts_deboilerplated", with_provenance(clean)))
    for name, frame in exports:
        frame.to_parquet(output_dir / f"{name}.parquet", index=False)
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    metadata.pop("config_snapshot", None)
    (output_dir / "trial_attempts.json").write_text(json.dumps(metadata["trial_attempts"], default=str))
    (output_dir / "run_metadata.json").write_text(json.dumps({
        "started_at": started_at, "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "batch_dir": str(args.batch_dir.resolve()), "servers": servers, **repositories,
        "settings": {"model_name": args.model_name, "max_tokens": args.max_tokens, "chunk_size": args.chunk_size,
                     "chunks_in_flight": args.chunks_in_flight, "limit": args.limit},
        "synthetic_prompt_sha256": provenance.synthetic_prompt_sha256.iloc[0],
        "trials": len(provenance), "cohort_rows": len(cohorts),
        "cohort_rows_unmerged": None if unmerged is None else len(unmerged),
        "cohort_rows_deboilerplated": None if clean is None else len(clean),
        "status_counts": provenance.extraction_status.value_counts().to_dict(),
        "seeds_without_protocol": not_generated,
        **{key: metadata[key] for key in ("extraction_method", "schema_version", "representation_version",
                                          "prompt_sha256", "structured_outputs", "skipped_completed",
                                          "trial_errors", "model_metadata")},
        "trial_rejections": sorted(metadata["trial_rejections"]),
        "merge": None if merging is None else {key: merging[key] for key in (
            "schema_version", "prompt_sha256", "cohorts_in", "cohorts_out", "merge_errors", "skipped_completed")},
        "deboilerplate": None if cleaning is None else {key: cleaning[key] for key in (
            "schema_version", "prompt_sha256", "cohorts", "items", "items_reused", "deboilerplate_errors",
            "unclassified_items", "skipped_completed")},
    }, indent=2, default=str))
    print(f"finished: {len(cohorts)} cohort rows from {len(provenance)} protocols; "
          f"{len(metadata['trial_errors'])} failed, {len(metadata['trial_rejections'])} not cancer treatment, "
          f"{metadata['skipped_completed']} reused"
          + ("" if cleaning is None else f"; boilerplate not removed from {len(cleaning['deboilerplate_errors'])} "
             "cohorts (criteria kept as extracted)"), flush=True)
    return 0 if not metadata["trial_errors"] or args.allow_failures else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--batch-dir", type=Path, required=True, help="A generate_batch.py output directory")
    parser.add_argument("--output-dir", type=Path, help="Default: <batch-dir>/cohort_extraction")
    parser.add_argument("--server", action="append", default=[], help="BASE_URL[=MAX_CONCURRENCY] for a new servers.json")
    parser.add_argument("--server-urls-file", type=Path, help="Existing live server registry to use instead of --server")
    parser.add_argument("--allow-failures", action="store_true",
                        help="Exit 0 when some trials fail every attempt (recorded in run_metadata.json)")
    parser.add_argument("--model-name", default="nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4")
    parser.add_argument("--max-tokens", type=int, default=50000)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunks-in-flight", type=int, default=2, choices=range(1, 17), metavar="1..16",
                        help="Each step (extraction, merge, boilerplate) keeps --chunk-size x this many requests "
                             "in flight, each sent as another returns; raise it so a large server pool is used")
    parser.add_argument("--limit", type=int, help="Only the first N protocols (smoke tests)")
    parser.add_argument("--no-merge-cohorts", dest="merge_cohorts", action="store_false",
                        help="Keep every extracted cohort instead of combining those with essentially the same criteria")
    parser.add_argument("--no-deboilerplate", dest="deboilerplate", action="store_false",
                        help="Do not write cohorts_deboilerplated (the cohorts with their boilerplate criteria removed)")
    sys.exit(main(parser.parse_args()))
