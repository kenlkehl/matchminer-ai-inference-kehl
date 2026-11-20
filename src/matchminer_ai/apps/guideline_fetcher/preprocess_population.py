"""Preprocess a MatchMiner-AI DuckDB population into a private, fast-loading cache.

Reads every current patient summary and all of those patients' notes (read-only),
embeds the summaries with the app's TrialSpace configuration, and writes a cache
directory that population management loads in seconds. The cache contains PHI:
it is created 0700 with 0600 files. Logs show counts only.

It always uses PHI-mode settings (the app's `mmai-guideline-fetcher --phi`), including the PHI cache.

Example (GPU 3):
  CUDA_VISIBLE_DEVICES=3 python -m matchminer_ai.apps.guideline_fetcher.preprocess_population
"""

import argparse
import os
import sys

# Summaries of every length in batches of 32 fragment PyTorch's default CUDA allocator: on a
# shared GPU 3 a build grew to 42 GB within 3,500 summaries and had to be stopped; with expandable
# segments (training's setting) the full 184,454-summary build peaked at 21 GB (2026-11-12). They do
# not change the vectors. Set before CUDA starts.
if not (os.environ.get("PYTORCH_ALLOC_CONF") or os.environ.get("PYTORCH_CUDA_ALLOC_CONF")):
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

from .settings import Settings


def main(argv=None):
    settings = Settings(phi=True)
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--database", default=str(settings.population_duckdb),
                        help="DuckDB database (default: %(default)s)")
    parser.add_argument("--summaries",
                        help="a Parquet summaries file or folder of them (such as a summarize_patients run's "
                             "summaries/, with patient_id = MRN) instead of the database's current summaries; "
                             "notes still come from --database")
    parser.add_argument("--output", default=str(settings.population_cache),
                        help="cache directory to create (default: %(default)s)")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace an existing cache after the new one is complete")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="summaries per TrialSpace call (default: %(default)s)")
    parser.add_argument("--memory-limit", default="128GB",
                        help="DuckDB memory before the note sort spills (default: %(default)s)")
    parser.add_argument("--limit", type=int,
                        help="build from a seeded sample of this many patients, for a trial run")
    args = parser.parse_args(argv)
    os.umask(0o077)
    from .population_cache import build_population_cache
    from .services import InputError

    try:
        build_population_cache(
            settings, args.output, database=args.database, summaries=args.summaries,
            batch_size=args.batch_size,
            memory_limit=args.memory_limit, limit=args.limit, overwrite=args.overwrite,
            log=lambda message: print(message, flush=True),
        )
    except InputError as exc:
        sys.exit(f"error: {exc}")


if __name__ == "__main__":
    main()
