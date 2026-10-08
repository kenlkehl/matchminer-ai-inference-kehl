"""Public PDQ snapshot and TrialSpace catalog APIs."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from matchminer_ai._storage import (
    atomic_json,
    atomic_text,
    digest,
    directory_lock,
    read_json,
)

from ._pdq_sources import FORMAT, GENERATOR, SOURCE_NOTICE, parse_pdq_html, pdq_url

if TYPE_CHECKING:
    import pandas as pd
    from matchminer_ai.config import MMAIConfig


def _sources(summaries):
    import re

    if not isinstance(summaries, Mapping) or not summaries:
        raise ValueError("summaries must map disease folder names to PDQ URL(s)")
    result = {}
    for disease, urls in summaries.items():
        if not isinstance(disease, str) or not re.fullmatch(
            r"[a-z0-9]+(?:-[a-z0-9]+)*", disease
        ):
            raise ValueError("PDQ disease names must be lowercase alphanumeric slugs")
        if isinstance(urls, str):
            urls = [urls]
        if not isinstance(urls, Sequence) or not urls:
            raise ValueError(
                "Each disease requires one or more complete PDQ summary URLs"
            )
        result[disease] = list(dict.fromkeys(pdq_url(url) for url in urls))
    return result


def _fetch(client, url):
    # Validate every redirect before following it. No page-controlled additional
    # links, embedded assets, search queries or patient information are fetched.
    for _ in range(6):
        url = pdq_url(url)
        response = client.get(url, follow_redirects=False)
        if response.status_code in {301, 302, 303, 307, 308}:
            from urllib.parse import urljoin

            url = pdq_url(urljoin(url, response.headers["location"]))
            continue
        response.raise_for_status()
        if "text/html" not in response.headers.get("content-type", "").lower():
            raise ValueError("PDQ endpoint did not return HTML")
        if len(response.content) > 32 * 1024 * 1024:
            raise ValueError("PDQ summary exceeds the 32 MiB download limit")
        return response.content, url
    raise ValueError("Too many PDQ redirects")


def prepare_pdq_library(
    summaries: Mapping[str, str | Sequence[str]],
    *,
    output_dir: str | Path,
    request_timeout: float = 60,
    progress_callback: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Download complete NCI PDQ health-professional summaries as a local library.

    ``summaries`` maps caller-chosen disease slugs to a URL or ordered URL list.
    Each supplied document is fully captured, including tables, references,
    hierarchy, visible update dates, source links and reuse notices. Only text is
    interpreted; omitted visual assets are recorded as source limitations.

    ``output_dir`` must be outside this code repository. It contains original
    HTML and a ``markdown`` library accepted by ``list_guidelines``,
    ``summarize_pdq`` and ``summarize_guidelines``. Identical invocations verify
    and reuse snapshots offline; use a NEW directory to fetch a newer edition.
    A partial download is resumable but is never published as a complete library.
    Clinical extraction is a separate, explicitly configured LLM call.

    Returns the same inventory DataFrame as ``list_guidelines``. Source text is
    subject to NCI's page-specific reuse exceptions; no images are downloaded.
    """
    import httpx

    from ._guideline_sources import load_guideline
    from .guidelines import list_guidelines

    requested = _sources(summaries)
    if (
        isinstance(request_timeout, bool)
        or not isinstance(request_timeout, (int, float))
        or request_timeout <= 0
    ):
        raise ValueError("request_timeout must be positive")
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable or None")
    output = Path(output_dir).resolve()
    package = Path(__file__).resolve().parents[1]
    checkout = package.parent.parent
    for protected in [
        package,
        *([checkout] if (checkout / "pyproject.toml").exists() else []),
    ]:
        if output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError(
                "PDQ snapshots must be outside the package/code repository"
            )

    def notify(message):
        if progress_callback is not None:
            try:
                progress_callback(message)
            except Exception:
                import logging

                logging.getLogger(__name__).exception("PDQ progress callback failed")

    output.mkdir(parents=True, exist_ok=True)
    library = output / "markdown"
    with directory_lock(output):
        plan = output / "snapshot_plan.json"
        identity = {"source_format": FORMAT, "summaries": requested}
        if plan.exists() and read_json(plan) != identity:
            raise ValueError("PDQ source selection changed; use a new output directory")
        if not plan.exists() and any(p.name != ".run.lock" for p in output.iterdir()):
            raise ValueError(
                "PDQ snapshot output must be empty or an identical resumable library"
            )
        atomic_json(plan, identity)
        entries = []
        with httpx.Client(timeout=request_timeout) as client:
            for disease, urls in requested.items():
                folder = library / disease
                manifest = folder / "manifest.json"
                if manifest.exists():
                    saved = load_guideline(library, disease)
                    if (
                        saved.metadata.get("requested_urls") != urls
                        or saved.metadata.get("source_kind") != "nci_pdq"
                    ):
                        raise ValueError(
                            "Existing PDQ snapshot has a different source selection"
                        )
                    metadata = saved.metadata
                    notify(f"{disease}: reused verified PDQ snapshot")
                else:
                    documents, sections = [], []
                    for number, url in enumerate(urls, 1):
                        notify(
                            f"{disease}: downloading PDQ summary {number}/{len(urls)}"
                        )
                        raw, resolved_url = _fetch(client, url)
                        parsed, info = parse_pdq_html(raw, resolved_url)
                        raw_path = f"html/document-{number:03d}.html"
                        # Lossless bytes, not a text re-encoding of the response.
                        from matchminer_ai._storage import atomic_bytes

                        atomic_bytes(folder / raw_path, raw)
                        documents.append(
                            {
                                **info,
                                "requested_url": url,
                                "path": raw_path,
                                "sha256": digest(raw),
                                "retrieved_at": datetime.now(UTC).isoformat(),
                            }
                        )
                        sections.extend(parsed)
                    records, generated = [], {}
                    for number, section in enumerate(sections, 1):
                        path = f"sections/s{number:04d}.md"
                        text = section["text"] + "\n"
                        atomic_text(folder / path, text)
                        generated[path] = digest(text.encode("utf-8"))
                        records.append(
                            {k: v for k, v in section.items() if k != "text"}
                            | {"path": path, "section_number": number}
                        )
                    source_hash = digest([d["sha256"] for d in documents])
                    metadata = {
                        "generator": GENERATOR,
                        "source_format": FORMAT,
                        "source_kind": "nci_pdq",
                        "is_clinical_practice_guideline": False,
                        "title": " / ".join(d["title"] for d in documents),
                        "version": "snapshot-" + source_hash[:16],
                        "source_sha256": source_hash,
                        "requested_urls": urls,
                        "documents": documents,
                        "page_count": len(records),
                        "pages": records,
                        "generated_files": generated,
                        "warnings": list(
                            dict.fromkeys(w for s in sections for w in s["warnings"])
                        ),
                        "notice": SOURCE_NOTICE,
                    }
                    atomic_json(manifest, metadata)
                    load_guideline(library, disease)
                    notify(f"{disease}: verified {len(records)} PDQ source sections")
                entries.append(
                    {
                        "folder": disease,
                        "title": metadata["title"],
                        "version": metadata["version"],
                        "pages": metadata["page_count"],
                        "source_sha256": metadata["source_sha256"],
                    }
                )
        atomic_json(
            library / "manifest.json",
            {"generator": GENERATOR, "source_format": FORMAT, "guidelines": entries},
        )
    return list_guidelines(output)


def summarize_pdq(
    source_directory: str | Path,
    *,
    disease: str,
    output_dir: str | Path,
    config: MMAIConfig | None = None,
    return_metadata: bool = False,
    return_qc: bool = False,
    progress_callback: Callable[[str], None] | None = None,
) -> pd.DataFrame | tuple:
    """Extract a PDQ catalog with the shared guideline pipeline and return conventions.

    Prepare an immutable source snapshot with ``prepare_pdq_library`` first.
    Uses ``config.guideline`` and ``config.remote`` just like
    ``summarize_guidelines``. Source format selects PDQ-specific prompting,
    identifiers, provenance and section citations. All clinical work is performed
    by the configured LLM, with full context packing, normal output reserve,
    reasoning, vendor sampling, branch ownership, exact quotes, coverage checks
    and offline final provenance audit. Existing retrieval/workup APIs consume
    the returned records without conversion. No server is selected implicitly.
    """
    from ._guideline_sources import load_guideline
    from .guidelines import summarize_guidelines

    if (
        load_guideline(source_directory, disease).metadata.get("source_kind")
        != "nci_pdq"
    ):
        raise ValueError("summarize_pdq requires a prepared NCI PDQ source library")
    return summarize_guidelines(
        source_directory,
        disease=disease,
        output_dir=output_dir,
        config=config,
        return_metadata=return_metadata,
        return_qc=return_qc,
        progress_callback=progress_callback,
    )
