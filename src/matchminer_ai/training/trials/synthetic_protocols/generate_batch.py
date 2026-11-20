"""Generate ctgov-seeded synthetic protocols with matchminer_ai.trials.generate_synthetic_protocols.

Seeds are ctgov records (nct_id, title, brief_summary, detailed_description,
eligibility_criteria): either an existing JSON-lines file (--seeds) or fetched
from the ClinicalTrials.gov API for the nct_id column of a CSV (--nct-ids-from)
into <output-dir>/seeds.jsonl, with the retrieval time. Requests go through the
inference package's endpoint pool with adaptive concurrency and vLLM load
feedback. The servers come from <output-dir>/servers.json, written from --server
on first run and re-read while running: edit it to add or remove a server
without restarting. Each server entry carries its own served model ID. Reruns
skip completed trials. See synthetic_protocols/README.md.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import httpx
import pandas as pd

from matchminer_ai import load_default_preset
from matchminer_ai.trials import generate_synthetic_protocols


API_URL = "https://clinicaltrials.gov/api/v2/studies"
FIELDS = "NCTId,BriefTitle,OfficialTitle,BriefSummary,DetailedDescription,EligibilityCriteria,LastUpdatePostDate"
IDS_PER_REQUEST = 200


def fetch_seeds(nct_ids, path):
    """Fetch ctgov records in batches and write them, in input order, as JSON lines."""
    retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    seeds = {}
    with httpx.Client(timeout=120) as client:
        for i in range(0, len(nct_ids), IDS_PER_REQUEST):
            params = {"filter.ids": ",".join(nct_ids[i:i + IDS_PER_REQUEST]), "fields": FIELDS, "pageSize": 1000}
            for attempt in range(6):
                response = client.get(API_URL, params=params)
                if response.status_code not in (429, 500, 502, 503, 504):
                    break
                time.sleep(2 ** attempt)
            response.raise_for_status()
            for study in response.json()["studies"]:
                protocol = study["protocolSection"]
                ident, description = protocol["identificationModule"], protocol.get("descriptionModule", {})
                seeds[ident["nctId"]] = dict(
                    nct_id=ident["nctId"], title=ident.get("officialTitle") or ident.get("briefTitle", ""),
                    brief_summary=description.get("briefSummary", ""),
                    detailed_description=description.get("detailedDescription", ""),
                    eligibility_criteria=protocol.get("eligibilityModule", {}).get("eligibilityCriteria", ""),
                    last_update_post_date=protocol.get("statusModule", {}).get("lastUpdatePostDateStruct", {}).get("date", ""),
                    retrieved_at=retrieved_at)
    missing = [n for n in nct_ids if n not in seeds]
    path.write_text("".join(json.dumps(seeds[n]) + "\n" for n in nct_ids if n in seeds))
    print(f"fetched {len(seeds)} seed records; {len(missing)} NCT IDs not returned by the API", flush=True)


def served_model(url):
    response = httpx.get(url.rstrip("/") + "/models", timeout=30)
    response.raise_for_status()
    return response.json()["data"][0]["id"]


def main(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # An externally managed registry (``training.pipeline`` --server-urls-file) is used as is.
    registry = args.server_urls_file or args.output_dir / "servers.json"
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

    seeds_path = args.seeds or args.output_dir / "seeds.jsonl"
    if args.seeds is None and not seeds_path.exists():
        nct_ids = list(dict.fromkeys(pd.read_csv(args.nct_ids_from, usecols=["nct_id"]).nct_id.dropna().astype(str)))
        fetch_seeds(nct_ids, seeds_path)
    seeds = pd.read_json(seeds_path, lines=True).rename(columns={"nct_id": "trial_id"})
    if args.limit:
        seeds = seeds.head(args.limit)
    config = load_default_preset()
    config.remote.update(enabled=True, server_urls=[], server_urls_file=str(registry), request_timeout=7200)
    config.remote["adaptive_concurrency"] = {**config.remote.get("adaptive_concurrency", {}),
                                             "enabled": True, "metrics_interval": 10}
    settings = config.synthetic_protocol_generation
    settings["chunk_size"] = args.chunk_size
    # The runtime model selects the sampling profile; each server's own model ID comes from the registry.
    settings["remote"]["model_name"] = args.model_name
    settings["remote"]["request_params"]["max_tokens"] = args.max_tokens
    started = time.time()
    result, metadata = generate_synthetic_protocols(
        seeds, config=config, output_dir=args.output_dir / "protocols", return_metadata=True,
        progress_callback=lambda stage, done, total, detail: print(
            f"[{datetime.now(timezone.utc):%H:%M:%S} +{(time.time() - started) / 60:,.1f} min] {stage} {detail}",
            flush=True),
    )
    metadata.pop("config_snapshot", None)
    (args.output_dir / "run_metadata.json").write_text(json.dumps(
        {k: v for k, v in metadata.items() if k != "trial_attempts"}, indent=2, default=str))
    (args.output_dir / "trial_attempts.json").write_text(json.dumps(metadata["trial_attempts"], default=str))
    result.drop(columns="protocol").to_csv(args.output_dir / "protocols.csv", index=False)
    print(f"finished: {len(result)} of {len(seeds)} protocols, {len(metadata['trial_errors'])} failed, "
          f"{metadata['skipped_completed']} already complete", flush=True)
    return 0 if not metadata["trial_errors"] or args.allow_failures else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--seeds", type=Path, help="Existing JSON lines of ctgov records")
    source.add_argument("--nct-ids-from", type=Path,
                        help="CSV with an nct_id column; records are fetched into <output-dir>/seeds.jsonl once")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--server", action="append", default=[], help="BASE_URL[=MAX_CONCURRENCY] for a new servers.json")
    parser.add_argument("--server-urls-file", type=Path, help="Existing live server registry to use instead of --server")
    parser.add_argument("--allow-failures", action="store_true",
                        help="Exit 0 when some trials fail every attempt (recorded in run_metadata.json)")
    parser.add_argument("--model-name", default="nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4")
    parser.add_argument("--max-tokens", type=int, default=100000)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--limit", type=int, help="Only the first N seeds (smoke tests)")
    sys.exit(main(parser.parse_args()))
