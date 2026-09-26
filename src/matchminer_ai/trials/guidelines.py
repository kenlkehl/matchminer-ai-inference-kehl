"""Public guideline-to-TrialSpace APIs over user-supplied, local source libraries."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from matchminer_ai._metadata import package_metadata
from matchminer_ai._storage import atomic_json, read_json
from matchminer_ai.config import MMAIConfig, config_snapshot, load_default_preset

if TYPE_CHECKING:
    import pandas as pd


def list_guidelines(source_directory: str | Path) -> pd.DataFrame:
    """List diseases in a local converter library without contacting an endpoint.

    Parameters
    ----------
    source_directory : str or Path
        Collection directory, its ``markdown`` directory, or a parent containing
        exactly one collection. The format is produced by ``convert_guidelines.py``.

    Returns
    -------
    pd.DataFrame
        Disease folder names, titles, editions, and page presence. A status of
        ``present_unverified`` is not a checksum or clinical validation result.
    """
    import pandas as pd

    from ._guideline_sources import inventory

    _, rows = inventory(Path(source_directory))
    return pd.DataFrame(rows)


def _output_directory(source_directory: str | Path, output_dir: str | Path) -> Path:
    from ._guideline_sources import library_root

    output = Path(output_dir).resolve()
    collection = library_root(source_directory).parent.resolve()
    package = Path(__file__).resolve().parents[1]
    checkout = package.parent.parent
    protected = [collection, package]
    if (checkout / "pyproject.toml").exists():
        protected.append(checkout)
    for root in protected:
        if output.is_relative_to(root) or root.is_relative_to(output):
            raise ValueError(
                "output_dir must be outside the source collection and package/code "
                "repository, and must not contain either directory."
            )
    return output


def summarize_guidelines(
    source_directory: str | Path,
    *,
    disease: str,
    output_dir: str | Path,
    config: MMAIConfig | None = None,
    return_metadata: bool = False,
    return_qc: bool = False,
    progress_callback: Callable[[str], None] | None = None,
) -> (
    pd.DataFrame
    | tuple[pd.DataFrame, dict]
    | tuple[pd.DataFrame, pd.DataFrame]
    | tuple[pd.DataFrame, dict, pd.DataFrame]
):
    """Extract canonical disease states, workup, and treatment menus with an LLM.

    Parameters
    ----------
    source_directory : str or Path
        A locally supplied page-preserving Markdown library with converter
        manifests and the corresponding original PDFs. No sources are downloaded.
    disease : str
        One disease folder name returned by :func:`list_guidelines`.
    output_dir : str or Path
        Required external directory for this disease/edition/run. Checkpoints,
        raw model responses, exact source quotations, JSONL, CSV, and a Markdown
        report are saved here. Treat all these artifacts as source-derived data.
        Reusing the directory resumes identical requests; changing the source,
        model, prompts, or generation settings requires a new directory.
    config : MMAIConfig, optional
        Uses ``config.guideline`` and the shared ``config.remote`` settings.
        Remote mode must be explicitly enabled. A single OpenAI-compatible
        endpoint is supported; this function never starts a model server.
        Defaults reserve 100,000 output tokens, discover the full context window,
        enable reasoning, and resolve vendor sampling from the served model
        (Gemma 4 or Qwen 3.8; xhigh effort for Qwen).
    return_metadata : bool, optional
        Also return package/config/model provenance, resolved run settings, source
        fingerprint, and the mechanical audit result.
    return_qc : bool, optional
        Also return a QC DataFrame with the package's metric/value/denominator/
        percent/ids columns. These checks do not establish clinical correctness.
    progress_callback : callable, optional
        Receives progress strings from the orchestration thread. Exceptions in
        the callback are logged and do not invalidate completed model work.

    Returns
    -------
    pd.DataFrame or tuple
        One row per canonical population. Standard trial-space columns are
        ``space_trial_id``, ``trial_id`` (a guideline namespace, not an NCT ID),
        ``clinical_space_number``, ``clinical_space_summary``, and
        ``general_exclusion_criteria`` (NA; no trial protocol is being summarized).
        Additional columns retain ``space``, ``diagnostic_workup``,
        ``treatment_options``, exact ``evidence``, ``source``, ``uncertainties``,
        context omissions, and a research-use notice. Return flag combinations
        follow :func:`summarize_trials`: data, optional metadata, optional QC.

    Notes
    -----
    The endpoint performs all clinical extraction. Code assigns source ownership,
    record IDs, and exact quotations, and validates structure and provenance.
    Failures raise and retain resumable checkpoints; partial catalogs are not
    returned as complete. PDF layout extraction cannot reconstruct every graphical
    branch. Human source review is required; outputs are not treatment advice.
    """
    import pandas as pd

    from matchminer_ai._qc.guidelines import guideline_qc_report
    from matchminer_ai.llm.backends import build_llm_runtime_config
    from matchminer_ai.llm.structured import resolve_structured_config

    from ._guideline_generation import Client
    from ._guideline_pipeline import run_guideline
    from ._guideline_sources import load_guideline

    resolved = load_default_preset() if config is None else config
    if not isinstance(resolved, MMAIConfig):
        raise TypeError("config must be an MMAIConfig instance or None.")
    if not resolved.remote.get("enabled", False):
        raise ValueError(
            "summarize_guidelines requires config.remote['enabled'] = True."
        )
    stage = resolved.guideline
    if not stage or "remote" not in stage:
        raise ValueError("Configure the guideline stage (see load_default_preset()).")
    if not isinstance(disease, str) or not disease.strip():
        raise ValueError("disease must be a nonempty library folder name.")
    workers = resolved.remote.get("max_concurrent_requests", 16)
    packet_pages = stage.get("packet_pages", 8)
    for name, value in (
        ("max_concurrent_requests", workers),
        ("packet_pages", packet_pages),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable or None.")
    output = _output_directory(source_directory, output_dir)
    guideline = load_guideline(Path(source_directory), disease)
    runtime = build_llm_runtime_config("guideline", dict(stage), config=resolved)
    previous = (
        read_json(output / "run_config.json")
        if (output / "run_config.json").exists()
        else None
    )
    llm, model_metadata = resolve_structured_config(
        runtime, cache_dir=output / "checkpoints", previous=previous, client_type=Client
    )
    status = run_guideline(
        guideline,
        output,
        llm,
        workers=workers,
        packet_pages=packet_pages,
        progress_callback=progress_callback,
        finalize=lambda: audit_guideline_catalog(
            source_directory, disease=disease, output_dir=output
        ),
    )
    validation = read_json(output / "validation.json")
    rows = [
        json.loads(line)
        for line in (output / "paradigms.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    result = pd.DataFrame(rows)
    qc = guideline_qc_report(result, status)
    metadata = {
        "package": package_metadata(),
        "config_snapshot": config_snapshot(resolved),
        "model_metadata": {"guideline_summarizer": model_metadata},
        "source": read_json(output / "sources.json"),
        "run_config": read_json(output / "run_config.json"),
        "validation": validation,
        "output_directory": str(output),
    }
    if return_metadata and return_qc:
        return result, metadata, qc
    if return_metadata:
        return result, metadata
    if return_qc:
        return result, qc
    return result


def audit_guideline_catalog(
    source_directory: str | Path, *, disease: str, output_dir: str | Path
) -> dict:
    """Verify a completed catalog against its local sources and raw checkpoints.

    Makes no model/network calls. Revalidates source hashes, branch ownership,
    evidence lines, immutable clinical fields, sampling/context settings, and
    exports. Saves ``validation.json`` in the external run directory and returns
    its audit dictionary. A pass establishes mechanical provenance, not clinical
    correctness, source applicability, or completeness.
    """
    from ._guideline_audit import audit_catalog
    from ._guideline_sources import load_guideline

    output = _output_directory(source_directory, output_dir)
    result: dict = audit_catalog(
        load_guideline(Path(source_directory), disease), output
    )
    atomic_json(output / "validation.json", result)
    return result
