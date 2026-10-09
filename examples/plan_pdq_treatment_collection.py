"""Snapshot official adult/pediatric PDQ indexes and create an external job plan.

This only inventories/downloads public index pages; extraction is separate.
--existing-summaries, --existing-source and --existing-output optionally reuse an
earlier extract_pdq.py run's original disease names, sources and checkpoints.
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import httpx

from matchminer_ai._storage import atomic_bytes, atomic_json, digest, read_json
from matchminer_ai.trials._pdq_discovery import INDEXES, treatment_inventory
from matchminer_ai.trials.pdq import _sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--catalog-dir", type=Path, required=True)
    parser.add_argument("--existing-summaries", type=Path)
    parser.add_argument("--existing-source", type=Path)
    parser.add_argument("--existing-output", type=Path)
    args = parser.parse_args()
    if any([args.existing_summaries, args.existing_source, args.existing_output]) and not all(
        [args.existing_summaries, args.existing_source, args.existing_output]
    ):
        parser.error("Supply all three existing-run arguments together")
    repo = Path(__file__).resolve().parents[1]
    for path in [args.inventory_dir, args.source_dir, args.catalog_dir]:
        if path.resolve().is_relative_to(repo) or repo.is_relative_to(path.resolve()):
            parser.error("All data directories must be outside the code repository")
    pages, provenance = {}, {}
    with httpx.Client(timeout=60) as client:
        for name, url in INDEXES.items():
            path = args.inventory_dir / f"{name}.html"
            receipt = args.inventory_dir / f"{name}.json"
            if receipt.exists():
                record = read_json(receipt)
                raw = path.read_bytes()
                if record["url"] != url or record["sha256"] != digest(raw):
                    raise ValueError("Index snapshot changed")
            else:
                response = client.get(url, follow_redirects=False)
                response.raise_for_status()
                if response.status_code != 200 or "text/html" not in response.headers.get("content-type", ""):
                    raise ValueError("Official PDQ index did not return HTML")
                raw = response.content
                record = {"url": url, "sha256": digest(raw),
                          "retrieved_utc": datetime.now(timezone.utc).isoformat()}
                atomic_bytes(path, raw)
                atomic_json(receipt, record)
            pages[url], provenance[name] = raw, record
    inventory = treatment_inventory(pages)
    inventory["snapshots"] = provenance
    jobs, reused = {}, set()
    included = {e["url"]: e for e in inventory["entries"] if e["included"]}
    if args.existing_summaries:
        for disease, urls in _sources(read_json(args.existing_summaries)).items():
            if not set(urls) <= set(included) or reused.intersection(urls):
                raise ValueError("Existing sources are absent from the inventory or duplicated")
            jobs[disease] = {"source_directory": str(args.existing_source.resolve()),
                             "output_directory": str((args.existing_output / disease).resolve())}
            reused.update(urls)
            for url in urls:
                included[url]["disease"] = disease
    for url, entry in included.items():
        if url in reused:
            continue
        disease = entry["disease"]
        if disease in jobs:
            raise ValueError(f"Duplicate job name: {disease}")
        jobs[disease] = {"urls": [url],
                         "source_directory": str((args.source_dir / disease).resolve()),
                         "output_directory": str((args.catalog_dir / disease).resolve())}
    atomic_json(args.inventory_dir / "inventory.json", inventory)
    plan = args.inventory_dir / "jobs.json"
    if plan.exists() and read_json(plan) != jobs:
        raise ValueError("Job plan changed; use a new inventory directory")
    atomic_json(plan, jobs)
    print(f"{len(jobs)} treatment catalogs; {len(inventory['entries']) - len(included)} explicitly excluded overview/supporting topics; plan: {plan}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
