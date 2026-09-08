"""Interactive terminal client for LLM-based trial-space checking."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TextIO
from urllib import error, request
from urllib.parse import urlparse

import pandas as pd

from matchminer_ai import load_default_preset
from matchminer_ai.config import MMAIConfig

_INPUT_TERMINATOR = ".done"


@dataclass(frozen=True)
class TrialCheckResult:
    """Complete LLM checker output plus its parsed score."""

    reasoning: str
    final_output: str
    score: int
    parse_status: str


def _endpoint_base_url(endpoint: str) -> str:
    """Normalize a server URL, including common full endpoint URLs."""
    value = endpoint.strip().rstrip("/")
    for suffix in ("/chat/completions", "/models"):
        if value.endswith(suffix):
            value = value[: -len(suffix)]
            break
    value = value or "http://localhost:8000/v1"
    if "://" not in value:
        value = f"http://{value}"
    if urlparse(value).path in {"", "/"}:
        value = f"{value.rstrip('/')}/v1"
    return value


def discover_model(
    endpoint: str,
    *,
    api_key: str | None = None,
    timeout: float = 10.0,
) -> tuple[str, list[str]]:
    """Return the normalized base URL and model IDs advertised by vLLM."""
    base_url = _endpoint_base_url(endpoint)
    models_url = f"{base_url.rstrip('/')}/models"
    token = (
        str(api_key or "").strip()
        or str(os.environ.get("OPENAI_API_KEY", "not-needed")).strip()
        or "not-needed"
    )
    req = request.Request(
        models_url,
        headers={"Authorization": f"Bearer {token}"},
        method="GET",
    )
    try:
        with request.urlopen(req, timeout=timeout) as response:
            payload = json.load(response)
    except (OSError, error.URLError, error.HTTPError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"GET {models_url} failed: {exc}") from exc

    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise TypeError(f"GET {models_url} returned an invalid model list.")
    model_ids = [
        str(item.get("id", "")).strip()
        for item in payload["data"]
        if isinstance(item, dict) and str(item.get("id", "")).strip()
    ]
    if not model_ids:
        raise RuntimeError(f"GET {models_url} returned no model IDs.")
    return base_url, model_ids


def build_config(base_url: str, model_name: str) -> MMAIConfig:
    """Configure the package's LLM match checker for one remote vLLM server."""
    config = load_default_preset()
    config.remote["enabled"] = True
    config.remote["provider"] = "openai"
    config.remote["server_urls"] = [base_url]
    config.llm_match_quality["remote"]["model_name"] = model_name
    config.debug_mode = True
    return config


def _score_candidate(candidate: pd.DataFrame, *, config: MMAIConfig):
    """Call the public checker while keeping its heavyweight imports lazy."""
    from matchminer_ai.matching import score_match_quality_with_llm

    return score_match_quality_with_llm(candidate, config=config)


def read_multiline(
    label: str,
    *,
    input_fn: Callable[[], str],
    output: TextIO,
) -> str:
    """Read a non-empty multiline terminal value terminated by ``.done``."""
    while True:
        print(
            f"\nPaste the {label}. On a new line, enter {_INPUT_TERMINATOR}",
            file=output,
        )
        lines: list[str] = []
        while True:
            line = input_fn()
            if line.strip() == _INPUT_TERMINATOR:
                break
            lines.append(line)
        value = "\n".join(lines).strip()
        if value:
            return value
        print(f"The {label} cannot be empty.", file=output)


def check_trial_space(
    patient_summary: str,
    trial_space: str,
    *,
    config: MMAIConfig,
) -> TrialCheckResult:
    """Run the public MatchMiner-AI LLM checker for one interactive pair."""
    candidate = pd.DataFrame(
        [
            {
                "patient_id": "interactive-patient",
                "space_trial_id": "interactive-trial-space",
                "cancer_history_summary": patient_summary,
                "clinical_space_summary": trial_space,
            }
        ]
    )
    result = _score_candidate(candidate, config=config)
    return TrialCheckResult(
        reasoning=str(result.loc[0, "llm_match_quality_reasoning_text"]),
        final_output=str(result.loc[0, "llm_match_quality_answer_text"]),
        score=int(result.loc[0, "llm_match_quality_score"]),
        parse_status=str(result.loc[0, "llm_match_quality_parse_status"]),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Interactively score one patient-summary/trial-space pair with the "
            "MatchMiner-AI LLM checker and a running vLLM server."
        )
    )
    parser.add_argument(
        "endpoint",
        help=(
            "vLLM server URL, such as http://localhost:8000 or http://localhost:8000/v1"
        ),
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    input_fn: Callable[[], str] | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run the interactive LLM trial checker."""
    args = _parser().parse_args(argv)
    input_fn = input if input_fn is None else input_fn
    stdout = sys.stdout if stdout is None else stdout
    stderr = sys.stderr if stderr is None else stderr

    try:
        base_url, model_ids = discover_model(args.endpoint)
    except (RuntimeError, TypeError) as exc:
        print(f"Could not discover a vLLM model: {exc}", file=stderr)
        return 2

    model_name = model_ids[0]
    print(f"Connected to: {base_url}", file=stdout)
    if len(model_ids) > 1:
        print(
            f"Endpoint advertised {len(model_ids)} models; using the first.",
            file=stdout,
        )
    print(f"Model: {model_name}", file=stdout)
    print(
        "Only continue if this endpoint is authorized to receive the patient "
        "summary. This CLI does not send the text to web search.",
        file=stdout,
    )

    try:
        patient_summary = read_multiline(
            "patient summary",
            input_fn=input_fn,
            output=stdout,
        )
        trial_space = read_multiline(
            "trial space",
            input_fn=input_fn,
            output=stdout,
        )
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled before trial checking.", file=stderr)
        return 130

    print("\nRunning MatchMiner-AI LLM trial checking...", file=stdout)
    try:
        result = check_trial_space(
            patient_summary,
            trial_space,
            config=build_config(base_url, model_name),
        )
    # Keep the terminal boundary sanitized: backend exceptions could include
    # request details containing patient text.
    except Exception as exc:  # noqa: BLE001
        print(
            "Trial checking failed "
            f"({type(exc).__name__}). Check the endpoint and its server logs.",
            file=stderr,
        )
        return 1

    print("\nReasoning:", file=stdout)
    print(result.reasoning, file=stdout)
    print("\nFinal output:", file=stdout)
    print(result.final_output, file=stdout)

    if result.score not in range(6):
        print(
            "The final output did not contain a valid 0-5 score "
            f"(parse status: {result.parse_status}).",
            file=stderr,
        )
        return 1

    print(f"\nLLM trial match score: {result.score}/5", file=stdout)
    print(
        "This is a research prioritization signal, not an eligibility "
        "determination or treatment recommendation.",
        file=stdout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
