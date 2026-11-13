"""Build a GoodOption evidence catalog, including drug classes, from NCT IDs.

The build reads only public ctgov trial records and public literature; no
patient data is involved, so it may run against a shared endpoint. The served
model is discovered from the endpoint and its registered sampling profile is
applied to every catalog LLM stage.

    matchminer-ai-build-good-option-catalog \
        --nct-ids-file data/no_phi/trial_ids.txt \
        --output data/no_phi/good_option_catalog_v12 \
        --server-url http://sn4622130540:8001/v1
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import re
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from matchminer_ai import load_default_preset
from matchminer_ai.config import MMAIConfig, load_config
from matchminer_ai.good_options import ResearchSettings
from matchminer_ai.llm.model_profiles import configure_served_model
from matchminer_ai.trials.drug_catalog import (
    build_good_option_catalog,
    catalog_research_pacing,
    validate_good_option_catalog,
)
from matchminer_ai.trials.drug_research import (
    DEFAULT_HOST_REQUESTS_PER_SECOND,
    NCBI_API_KEY_REQUESTS_PER_SECOND,
)

_NCT_PATTERN = re.compile(r"NCT\d{8}")


def read_nct_ids(path: str | Path) -> list[str]:
    """Read NCT IDs from a text file (one per line, ``#`` comments) or a table."""
    source = Path(path).expanduser()
    if source.suffix.lower() in {".csv", ".parquet"}:
        frame = (
            pd.read_parquet(source)
            if source.suffix.lower() == ".parquet"
            else pd.read_csv(source)
        )
        column = "nct_id" if "nct_id" in frame.columns else "trial_id"
        values = frame[column].dropna().astype(str).tolist()
    else:
        values = [
            line.split("#", 1)[0]
            for line in source.read_text(encoding="utf-8").splitlines()
        ]
    ids = [value.strip().upper() for value in values if value.strip()]
    invalid = [value for value in ids if not _NCT_PATTERN.fullmatch(value)]
    if invalid:
        raise ValueError(f"Invalid NCT IDs in {source}: {invalid[:10]}")
    return list(dict.fromkeys(ids))


def catalog_llm_sections(config: MMAIConfig) -> tuple[str, ...]:
    """The base GoodOption LLM section plus each catalog stage override."""
    stages = sorted(key for key in config.good_option_catalog if key.endswith("_llm"))
    return ("llm_good_option", *(f"good_option_catalog.{key}" for key in stages))


def host_rate(value: str) -> tuple[str, float]:
    """Parse one ``HOST=REQUESTS_PER_SECOND`` override."""
    host, separator, rate = value.partition("=")
    try:
        if separator and host.strip():
            return host.strip(), float(rate)
    except ValueError:
        pass
    raise argparse.ArgumentTypeError(
        f"expected HOST=REQUESTS_PER_SECOND, got {value!r}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--nct-ids-file", default="")
    parser.add_argument("--nct-id", action="append", default=[])
    parser.add_argument("--output", required=True, help="Catalog directory.")
    parser.add_argument(
        "--checkpoint-dir",
        default="",
        help="Resumable checkpoint directory; defaults to <output>_checkpoints.",
    )
    parser.add_argument(
        "--server-url",
        action="append",
        default=[],
        help="OpenAI-compatible endpoint; repeat for several identical servers.",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Expected served model; discovered from the endpoint when omitted.",
    )
    parser.add_argument("--config", default="", help="Config YAML; default preset.")
    parser.add_argument("--max-concurrent-requests", type=int, default=64)
    parser.add_argument(
        "--llm-request-timeout",
        type=float,
        default=86400.0,
        help=(
            "Seconds per LLM request, including time queued on the server. "
            "Multi-pass syntheses on one busy server exceeded 7,200 s and were "
            "silently abandoned and resent, so the default is 24 hours."
        ),
    )
    parser.add_argument(
        "--llm-batch-size",
        type=int,
        default=None,
        help=(
            "Trials screened, and drugs classified, at once (preset: 64). Each "
            "is checkpointed as soon as it finishes and the next starts in its "
            "place, so a slow answer holds one slot rather than a batch; the "
            "endpoint still runs at most --max-concurrent-requests, and the "
            "rest wait for it. Synthesis likewise checkpoints each subject when "
            "its own chain finishes, with up to --max-concurrent-requests chains "
            "in flight. Part of the checkpoint fingerprint."
        ),
    )
    parser.add_argument("--research-concurrency", type=int, default=None)
    parser.add_argument(
        "--subject-concurrency",
        type=int,
        default=8,
        help=(
            "Drugs, then classes, researched at once. Scheduling only: not part "
            "of the checkpoint fingerprint, so it can change on resume."
        ),
    )
    parser.add_argument(
        "--web-search-concurrency",
        type=int,
        default=12,
        help="General-web searches in flight across all subjects.",
    )
    parser.add_argument(
        "--host-requests-per-second",
        type=host_rate,
        action="append",
        default=[],
        metavar="HOST=RATE",
        help=(
            "Pace one host's research requests, shared by every subject; repeat "
            "per host, 0 leaves it unpaced. Overrides "
            "good_option_catalog.host_requests_per_second and the defaults ("
            + ", ".join(
                f"{host} {rate:g}"
                for host, rate in sorted(DEFAULT_HOST_REQUESTS_PER_SECOND.items())
            )
            + f"; E-utilities {NCBI_API_KEY_REQUESTS_PER_SECOND:g} with an NCBI "
            "API key). Scheduling only: not part of the checkpoint fingerprint."
        ),
    )
    parser.add_argument(
        "--ncbi-api-key-file",
        default="",
        help=(
            "File holding an NCBI API key, sent with PubMed requests for NCBI's "
            "higher E-utilities rate. Overrides good_option_catalog."
            "ncbi_api_key_file and the NCBI_API_KEY environment variable. Not "
            "part of the checkpoint fingerprint."
        ),
    )
    parser.add_argument("--research-request-timeout", type=float, default=None)
    parser.add_argument("--reset-checkpoints", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    nct_ids = list(args.nct_id)
    if args.nct_ids_file:
        nct_ids.extend(read_nct_ids(args.nct_ids_file))
    nct_ids = list(dict.fromkeys(value.strip().upper() for value in nct_ids))
    if not nct_ids:
        print("No NCT IDs supplied.", file=sys.stderr)
        return 2
    if not args.server_url:
        print("At least one --server-url is required.", file=sys.stderr)
        return 2

    config = load_config(args.config) if args.config else load_default_preset()
    model_name, profile = configure_served_model(
        config,
        args.server_url,
        sections=catalog_llm_sections(config),
        model_name=args.model or None,
    )
    config.remote["max_concurrent_requests"] = args.max_concurrent_requests
    if args.llm_batch_size is not None:
        for stage in ("screening", "class"):
            config.good_option_catalog[f"{stage}_checkpoint_batch_size"] = (
                args.llm_batch_size
            )
    config.remote["request_timeout"] = args.llm_request_timeout
    remote = config.llm_good_option["remote"]
    print(
        f"Model {model_name}; profile "
        f"{profile.name if profile else 'none (preset sampling)'}; "
        f"request_params={remote.get('request_params')} "
        f"extra_body={remote.get('extra_body')}",
        flush=True,
    )
    for key, stage in config.good_option_catalog.items():
        if key.endswith("_llm"):
            budget = stage.get("remote", {}).get("request_params", {})
            print(f"  {key} max_tokens={budget.get('max_tokens')}", flush=True)

    overrides = {
        key: value
        for key, value in {
            "max_concurrency": args.research_concurrency,
            "request_timeout": args.research_request_timeout,
        }.items()
        if value is not None
    }
    settings = dataclasses.replace(ResearchSettings(), **overrides)
    host_rates = dict(args.host_requests_per_second)
    rates, api_key = catalog_research_pacing(
        config,
        host_requests_per_second=host_rates,
        ncbi_api_key_file=args.ncbi_api_key_file or None,
    )
    print(
        "Research requests per second: "
        + ", ".join(f"{host} {rate:g}" for host, rate in sorted(rates.items()))
        + f"; NCBI API key {'set' if api_key else 'not set'}",
        flush=True,
    )
    started = time.monotonic()

    def progress(stage: str, completed: int, total: int, label: str) -> None:
        elapsed = (time.monotonic() - started) / 60
        print(f"[{elapsed:8.1f}m] [{stage}] {completed}/{total}: {label}", flush=True)

    catalog = asyncio.run(
        build_good_option_catalog(
            nct_ids,
            args.output,
            config=config,
            settings=settings,
            checkpoint_path=args.checkpoint_dir or None,
            reset_checkpoint=args.reset_checkpoints,
            overwrite=args.overwrite,
            progress_callback=progress,
            subject_concurrency=args.subject_concurrency,
            web_search_concurrency=args.web_search_concurrency,
            host_requests_per_second=host_rates,
            ncbi_api_key_file=args.ncbi_api_key_file or None,
        )
    )
    manifest = validate_good_option_catalog(catalog.path)
    print(f"Saved and validated {catalog.path}: counts={manifest['counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
