"""Snapshot complete NCI PDQ summaries and optionally extract a TrialSpace catalog.

All downloaded sources and generated outputs belong outside the code repository.
The source JSON maps disease slugs to a health-professional PDQ URL or URL list.
An identical source directory reuses its immutable snapshots. Choose a new one
to download an updated edition. No endpoint/model is selected automatically.
"""

import argparse
import json
from pathlib import Path

from matchminer_ai import load_default_preset
from matchminer_ai.trials import prepare_pdq_library, summarize_pdq


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summaries", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--endpoint")
    parser.add_argument("--model", help="Exact served ID; omit for endpoint discovery")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--request-timeout", type=float, default=7200)
    args = parser.parse_args()
    if args.concurrency < 1 or args.request_timeout <= 0:
        parser.error("Concurrency and timeout must be positive")
    if not args.download_only and (not args.endpoint or not args.output_dir):
        parser.error("Extraction requires an explicit --endpoint and --output-dir")
    summaries = json.loads(args.summaries.read_text())
    inventory = prepare_pdq_library(
        summaries,
        output_dir=args.source_dir,
        progress_callback=print,
    )
    print(inventory.to_string(index=False))
    if args.download_only:
        return
    config = load_default_preset()
    config.remote.update(
        enabled=True,
        server_urls=[args.endpoint],
        max_concurrent_requests=args.concurrency,
        request_timeout=args.request_timeout,
    )
    config.guideline["remote"]["model"] = args.model
    for disease in inventory.disease:
        result, metadata = summarize_pdq(
            args.source_dir,
            disease=disease,
            output_dir=args.output_dir / disease,
            config=config,
            return_metadata=True,
            progress_callback=print,
        )
        print(
            json.dumps(
                {
                    "disease": disease,
                    "spaces": len(result),
                    "audit": metadata["validation"]["status"],
                    "output_directory": metadata["output_directory"],
                }
            )
        )


if __name__ == "__main__":
    main()
