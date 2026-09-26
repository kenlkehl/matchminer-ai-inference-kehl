"""Read the existing page-preserving converter format; never scrape or rewrite inputs."""

import re
from dataclasses import dataclass
from pathlib import Path

from matchminer_ai._storage import digest, read_json


class InputNotReady(ValueError):
    """A copied collection is incomplete, changed, or incompatible."""


@dataclass(frozen=True)
class Page:
    id: str
    number: int
    label: str
    path: str
    text: str
    sha256: str
    links: tuple[int, ...]


@dataclass
class Guideline:
    directory: Path
    metadata: dict
    pages: dict[str, Page]
    fingerprint: str

    @property
    def disease(self):
        return self.directory.name

    @property
    def primary(self):
        # Front matter contains change logs with *removed* recommendations.
        # Retain and hash it, but don't mine old/deleted recommendations from it.
        return [
            p for p in self.pages.values() if not p.path.startswith("00-front-matter/")
        ]


def safe_child(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise InputNotReady(f"Manifest path escapes input folder: {relative}")
    return path


def library_root(path):
    path = Path(path).resolve()
    if (path / "markdown").is_dir():
        return path / "markdown"
    # Accept the enclosing nccn folder when it contains one copied collection.
    children = sorted(path.glob("*/markdown"))
    if len(children) == 1:
        return children[0]
    return path


def inventory(root, check_presence=True):
    root = library_root(root)
    manifest = read_json(root / "manifest.json")
    if "guidelines" not in manifest:
        raise InputNotReady(
            "Supply the collection or markdown root, not one disease folder"
        )
    result = []
    for entry in manifest["guidelines"]:
        disease = entry["folder"]
        folder = safe_child(root, disease)
        try:
            if not check_presence:
                result.append(
                    {
                        "disease": disease,
                        "title": entry["title"],
                        "version": entry["version"],
                        "pages": entry["pages"],
                        "status": "not_checked",
                    }
                )
                continue
            metadata = read_json(folder / "manifest.json")
            present = sum(
                safe_child(folder, p["path"]).is_file() for p in metadata["pages"]
            )
            ready = present == metadata["page_count"] == len(metadata["pages"])
        except (OSError, ValueError, KeyError):
            ready, present = False, 0
        result.append(
            {
                "disease": disease,
                "title": entry["title"],
                "version": entry["version"],
                "pages": entry["pages"],
                "present": present,
                "status": "present_unverified" if ready else "pending_copy",
            }
        )
    return root, result


def load_guideline(root, disease):
    root = library_root(root)
    directory = safe_child(root, disease)
    try:
        manifest_bytes = (directory / "manifest.json").read_bytes()
        import json

        metadata = json.loads(manifest_bytes)
        if metadata.get("generator") != "nccn-pdf-to-markdown":
            raise InputNotReady("Expected the nccn-pdf-to-markdown manifest format")
        records = metadata["pages"]
        if sorted(p["pdf_page"] for p in records) != list(
            range(1, metadata["page_count"] + 1)
        ):
            raise InputNotReady(
                "Manifest must cover each physical PDF page exactly once"
            )
        generated = metadata["generated_files"]
        pages = {}
        for record in sorted(records, key=lambda r: r["pdf_page"]):
            path = safe_child(directory, record["path"])
            raw = path.read_bytes()
            if digest(raw) != generated[record["path"]]:
                raise InputNotReady(
                    f"Page not fully copied or changed: {record['path']}"
                )
            markdown = raw.decode("utf-8")
            block = re.search(
                r"^(`{3,})text\n(.*?)\n\1\s*$", markdown, re.MULTILINE | re.DOTALL
            )
            if not block:
                raise InputNotReady(f"No extracted text block in {record['path']}")
            number = record["pdf_page"]
            page_id = f"p{number:04d}"
            pages[page_id] = Page(
                page_id,
                number,
                record["label"],
                record["path"],
                block.group(2),
                digest(raw),
                tuple(record["internal_link_pages"]),
            )
        # A successfully copied Markdown set must match the accompanying PDF.
        pdf = safe_child(root.parent, metadata["source_pdf"])
        import hashlib

        with pdf.open("rb") as stream:
            pdf_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        if pdf_hash != metadata["source_sha256"]:
            raise InputNotReady(f"Source PDF not fully copied or changed: {pdf.name}")
        if (directory / "manifest.json").read_bytes() != manifest_bytes:
            raise InputNotReady("Manifest changed during input verification")
    except (OSError, KeyError, ValueError) as exc:
        raise InputNotReady(f"{disease}: {exc}") from exc
    fingerprint = digest(
        {
            "pdf": pdf_hash,
            "manifest": digest(manifest_bytes),
            "pages": {key: page.sha256 for key, page in pages.items()},
        }
    )
    return Guideline(directory, metadata, pages, fingerprint)


def packets(guideline, max_chars=None, max_pages=8):
    batch, size = [], 0
    for page in guideline.primary:
        if max_chars is not None and len(page.text) > max_chars:
            raise ValueError(
                f"{page.id} exceeds packet_chars; increase the budget (pages are never cut)"
            )
        if batch and (
            (max_chars is not None and size + len(page.text) > max_chars)
            or len(batch) >= max_pages
        ):
            yield batch
            batch, size = [], 0
        batch.append(page)
        size += len(page.text)
    if batch:
        yield batch


def select_context(
    guideline, required_ids, query, max_chars=None, *, priority_order=False
):
    """Required pages, PDF-linked neighbors, then lexical context; report every omitted page."""
    required_ids = set(required_ids)
    selected = [p for p in guideline.pages.values() if p.id in required_ids]
    if len(selected) != len(required_ids):
        raise ValueError("Unknown required page")
    size = sum(len(p.text) for p in selected)
    if max_chars is not None and size > max_chars:
        raise ValueError(
            "Required evidence exceeds context_chars; increase the budget"
        )
    linked = {n for p in selected for n in p.links}
    terms = {
        t
        for t in re.findall(r"[a-z0-9-]{4,}", query.lower())
        if t
        not in {"allowed", "required", "excluded", "cancer", "treatment", "patients"}
    }

    def rank(page):
        words = set(re.findall(r"[a-z0-9-]{4,}", page.text.lower()))
        return (page.number in linked, len(terms & words), -page.number)

    remaining = sorted(
        (p for p in guideline.primary if p.id not in required_ids),
        key=rank,
        reverse=True,
    )
    for page in remaining:
        if max_chars is None or size + len(page.text) <= max_chars:
            selected.append(page)
            size += len(page.text)
    if not priority_order:
        selected.sort(key=lambda p: p.number)
    ids = {p.id for p in selected}
    return selected, [p.id for p in guideline.primary if p.id not in ids]


def render_pages(pages, primary_ids=()):
    primary_ids = set(primary_ids)
    return "\n\n".join(
        f"<source_page id={p.id} label={p.label!r} "
        f"role={'PRIMARY' if p.id in primary_ids else 'CONTEXT'}>\n"
        + "\n".join(
            f"L{i:04d} | {line}"
            for i, line in enumerate(p.text.splitlines(), 1)
            if line.strip()
        )
        + "\n</source_page>"
        for p in pages
    )
