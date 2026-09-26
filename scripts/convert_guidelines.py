#!/usr/bin/env python3
"""Convert local guideline PDFs into a page-complete, navigable Markdown library.

Requires pypdf and Poppler's pdftotext. No network calls or model calls are made.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

from pypdf import PdfReader

VERSION = "1.0.3"
GENERATOR = "nccn-pdf-to-markdown"
CODE_PATTERN = r"([A-Z][A-Z0-9]*(?:[-/ ][A-Z][A-Z0-9]*)*-(?:\d+[A-Z]?|[A-Z]+))"
CODE = re.compile(r"\b" + CODE_PATTERN + r"\b")
FOOTER_CODE = re.compile(CODE_PATTERN + r"\s*(?:(\d+)\s+OF\s+(\d+))?")
NOTE = (
    "This is extracted source text, not a summary. Spacing is retained in a text block. "
    "Arrows, colors, diagrams, superscripts, and multi-column reading order are not "
    "fully represented; consult the linked original PDF page for visual relationships. "
    "All source notices and footnotes are retained. No OCR is performed."
)


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:100] or "untitled"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def relative_link(source: Path, target: Path, fragment: str = "") -> str:
    return (
        quote(os.path.relpath(target, source.parent).replace(os.sep, "/"), safe="/.-_")
        + fragment
    )


def md_label(text: str) -> str:
    return (
        re.sub(r"\s+", " ", text)
        .replace("\\", "\\\\")
        .replace("[", "\\[")
        .replace("]", "\\]")
    )


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, data: object) -> None:
    write(path, json.dumps(data, ensure_ascii=False, indent=2))


def normalize_text(text: str) -> str:
    # Do not collapse spaces, dehyphenate, reorder columns, or strip source notices.
    text = re.sub(
        r"[\x00-\x08\x0b-\x1f\x7f]",
        lambda m: f"⟦U+{ord(m[0]):04X}⟧",
        text.replace("\r\n", "\n"),
    )
    return "\n".join(line.rstrip() for line in text.split("\n")).strip("\n")


def printed_code(text: str) -> tuple[str | None, str | None]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    # Only standalone footer lines are eligible: references in body text are not labels.
    for pos in range(len(lines) - 1, max(-1, len(lines) - 9), -1):
        line = re.sub(r"[‐‑‒–−]", "-", lines[pos])
        if line == "UPDATES":
            return "UPDATES", None
        # Some PDFs place MS-### on the same line as the copyright footer.
        if "All rights reserved" in line and "permission of NCCN" in line:
            line = re.split(r"permission of NCCN\.?", line)[-1].strip()
        # Footer codes can occupy the right column beside a copyright/source line.
        line = re.split(r"\s{3,}", line)[-1]
        match = FOOTER_CODE.fullmatch(line)
        if not match and re.fullmatch(r"[A-Z][A-Z0-9]{1,15}", line):
            for following in lines[pos + 1 :]:
                subpage = re.fullmatch(r"(\d+)\s+OF\s+(\d+)", following, re.I)
                if subpage:
                    return line, f"{subpage[1]} of {subpage[2]}"
        if match:
            code = match[1].upper()
            suffix = f"{match[2]} of {match[3]}" if match[2] else None
            if not suffix:
                for following in lines[pos + 1 :]:
                    subpage = re.fullmatch(r"(\d+)\s+OF\s+(\d+)", following, re.I)
                    if subpage:
                        suffix = f"{subpage[1]} of {subpage[2]}"
            return code, suffix
    return None, None


def page_group(code: str | None, page: int) -> tuple[str, str]:
    if code is None:
        return ("00-front-matter" if page <= 3 else "90-unclassified"), "pages"
    if code == "UPDATES":
        return "00-front-matter", "updates"
    if "-" not in code:
        return "10-guideline-sections", slug(code)
    family, ending = code.rsplit("-", 1)
    if family == "MS":
        return "20-discussion-and-references", "ms"
    if family in {"ST", "ABBR", "REF"}:
        return "30-appendices", slug(family)
    return "10-guideline-sections", slug(family if ending[0].isdigit() else code)


def document_info(first_page: str, fallback: str) -> tuple[str, str | None]:
    lines = [line.strip() for line in first_page.splitlines() if line.strip()]
    title_parts = []
    collecting = False
    for line in lines:
        if "Clinical Practice Guidelines in Oncology" in line:
            collecting = True
            continue
        if collecting:
            if re.search(r"\bVersion\s+\d", line):
                before = re.split(r"\bVersion\s+\d", line)[0].strip()
                if before:
                    title_parts.append(before)
                break
            if "NCCN Guidelines" not in line:
                title_parts.append(line)
    title = " ".join(title_parts).strip() or fallback
    match = re.search(r"\bVersion\s+(\d+\.\d{4})", first_page)
    return title, match[1] if match else None


def extract_pages(pdf: Path, binary: str, expected: int) -> list[str]:
    result = subprocess.run(
        [binary, "-layout", "-enc", "UTF-8", str(pdf), "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    parts = result.stdout.decode("utf-8", errors="strict").split("\f")
    if parts and not parts[-1].strip():
        parts.pop()
    if len(parts) != expected:
        raise ValueError(
            f"Text extraction returned {len(parts)} pages; PDF has {expected}. No output committed."
        )
    return [normalize_text(part) for part in parts]


def bookmarks(reader: PdfReader, pages: list[dict]) -> tuple[list[dict], list[str]]:
    rows, warnings = [], []
    by_code = defaultdict(list)
    for page in pages:
        if page["printed_code"]:
            by_code[page["printed_code"]].append(page["pdf_page"])

    def walk(items: list, depth: int = 0) -> None:
        for item in items:
            if isinstance(item, list):
                walk(item, depth + 1)
                continue
            title = re.sub(r"\s+", " ", str(item.title)).strip()
            try:
                index = reader.get_destination_page_number(item)
                page = (
                    index + 1 if index is not None and 0 <= index < len(pages) else None
                )
            except Exception as exc:
                page = None
                warnings.append(f"Bookmark {title!r}: {exc}")
            codes = CODE.findall(re.sub(r"[‐‑‒–−]", "-", title))
            expected_code = codes[-1] if codes else None
            matches = by_code.get(expected_code, [])
            mismatch = bool(
                page and matches and pages[page - 1]["printed_code"] != expected_code
            )
            if mismatch:
                warnings.append(
                    f"Bookmark {title!r} points to PDF page {page}; printed code {expected_code} occurs on pages {matches}."
                )
            elif page is None:
                warnings.append(f"Bookmark {title!r} has no usable local destination.")
            rows.append(
                dict(
                    title=title,
                    depth=depth,
                    pdf_page=page,
                    printed_code_matches=matches if mismatch else [],
                    mismatch=mismatch,
                )
            )

    walk(reader.outline)
    return rows, warnings


def resolve_destination(reader: PdfReader, dest: object) -> int | None:
    if isinstance(dest, str):
        named = reader.named_destinations.get(dest)
        return reader.get_destination_page_number(named) if named is not None else None
    if hasattr(dest, "get_object"):
        dest = dest.get_object()
    if isinstance(dest, (list, tuple)) and dest:
        obj = dest[0].get_object() if hasattr(dest[0], "get_object") else dest[0]
        if isinstance(obj, int):
            return obj
        return reader.get_page_number(obj)
    return None


def page_links(reader: PdfReader, index: int) -> tuple[list[int], list[str], int]:
    internal, external, unresolved = set(), set(), 0
    for ref in reader.pages[index].get("/Annots", []):
        annotation = ref.get_object()
        if annotation.get("/Subtype") != "/Link":
            continue
        try:
            action = annotation.get("/A", {})
            action = action.get_object() if hasattr(action, "get_object") else action
            if action.get("/S") == "/URI":
                uri = str(action.get("/URI", ""))
                if urlsplit(uri).scheme in {"http", "https", "mailto"}:
                    external.add(uri)
                else:
                    unresolved += 1
            elif "/Dest" in annotation or action.get("/S") == "/GoTo":
                dest = annotation.get("/Dest", action.get("/D"))
                page = resolve_destination(reader, dest)
                if page is not None and 0 <= page < len(reader.pages):
                    internal.add(page + 1)
                else:
                    unresolved += 1
            else:
                unresolved += 1
        except Exception:
            unresolved += 1
    return sorted(internal), sorted(external), unresolved


def convert(pdf: Path, output: Path, binary: str, overwrite: bool) -> dict:
    target = output / slug(pdf.stem)
    source_hash = sha256(pdf)
    if target.exists():
        marker = target / "manifest.json"
        old = json.loads(marker.read_text()) if marker.is_file() else {}
        if old.get("generator") != GENERATOR:
            raise ValueError(f"Refusing to replace unrecognized folder: {target}")
        if not overwrite:
            if (
                old.get("source_sha256") == source_hash
                and old.get("converter_version") == VERSION
            ):
                problems = validate_document(target)
                if not problems:
                    return old
            raise ValueError(
                f"Existing output changed or is incomplete: {target}. Use --overwrite to rebuild generated output."
            )

    reader = PdfReader(pdf)
    if reader.is_encrypted and not reader.decrypt(""):
        raise ValueError("Password-protected PDF; no password supplied.")
    texts = extract_pages(pdf, binary, len(reader.pages))
    title, version = document_info(texts[0], pdf.stem)
    pages, warnings = [], []
    for number, text in enumerate(texts, 1):
        code, subpage = printed_code(text)
        group, section = page_group(code, number)
        label = (
            " ".join(part for part in (code, subpage) if part) or f"PDF page {number}"
        )
        filename = f"page-{number:04d}-{slug(label)}.md"
        pages.append(
            dict(
                pdf_page=number,
                printed_code=code,
                printed_subpage=subpage,
                label=label,
                path=f"{group}/{section}/{filename}",
                characters=len(text),
                text_sha256=hashlib.sha256(text.encode()).hexdigest(),
            )
        )
        if not text.strip():
            warnings.append(
                f"PDF page {number} has no extractable text; consult the original page. OCR was not performed."
            )
        if "\ufffd" in text:
            warnings.append(
                f"PDF page {number} contains Unicode replacement characters."
            )
        controls = sorted(set(re.findall(r"⟦U\+[0-9A-F]{4}⟧", text)))
        if controls:
            warnings.append(
                f"PDF page {number}: unmapped control glyphs are shown explicitly as {', '.join(controls)}; verify their appearance in the PDF."
            )
    outline, outline_warnings = bookmarks(reader, pages)
    warnings.extend(outline_warnings)
    groups = defaultdict(list)
    for page in pages:
        groups[str(Path(page["path"]).parent)].append(page)

    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=output))
    try:
        for page, text in zip(pages, texts):
            path = stage / page["path"]
            number = page["pdf_page"]
            internal, external, unresolved = page_links(reader, number - 1)
            page["internal_link_pages"] = internal
            page["external_links"] = external
            if unresolved:
                warnings.append(
                    f"PDF page {number}: {unresolved} link annotation(s) could not be represented."
                )
            front = dict(
                guideline=title,
                version=version,
                source_pdf=pdf.name,
                source_sha256=source_hash,
                pdf_page=number,
                printed_code=page["printed_code"],
                extraction="pdftotext -layout; no OCR",
            )
            lines = [
                "---",
                *[
                    f"{key}: {json.dumps(value, ensure_ascii=False)}"
                    for key, value in front.items()
                ],
                "---",
                "",
                f"# {md_label(page['label'])}",
                "",
                f"[Guideline index]({relative_link(path, stage / 'README.md')}) · "
                f"[Section index](README.md) · "
                f"[Original PDF, page {number}]({relative_link(path, pdf, f'#page={number}')})",
                "",
            ]
            neighbors = []
            for offset, label in [(-1, "Previous page"), (1, "Next page")]:
                index = number - 1 + offset
                if 0 <= index < len(pages):
                    neighbors.append(
                        f"[{label}]({relative_link(path, stage / pages[index]['path'])})"
                    )
            lines += [
                " · ".join(neighbors),
                "",
                "> " + NOTE,
                "",
                "## Extracted page text",
                "",
            ]
            fence = "`" * max(
                3, max((len(run) + 1 for run in re.findall(r"`+", text)), default=3)
            )
            lines += [fence + "text", text, fence, ""]
            if not text.strip():
                lines += ["**No extractable text. Open the original PDF page.**", ""]
            if internal:
                lines += [
                    "## Links embedded in this PDF page",
                    "",
                    "These preserve PDF link destinations; they do not infer flowchart relationships.",
                    "",
                ]
                for destination in internal:
                    linked = pages[destination - 1]
                    lines.append(
                        f"- [{md_label(linked['label'])} (PDF {destination})]({relative_link(path, stage / linked['path'])})"
                    )
            if external:
                lines += ["", "## External links embedded in this page", ""]
                for uri in external:
                    lines.append(
                        f"- [{md_label(uri)}]({quote(uri, safe=':/?&=#%+@,;~_-')})"
                    )
            write(path, "\n".join(lines))

        # Every page has exactly one canonical content file. Indexes only link to it.
        for group, members in sorted(groups.items()):
            index = stage / group / "README.md"
            lines = [
                f"# {md_label(title)}: {group.split('/')[-1].upper()}",
                "",
                f"[Guideline index]({relative_link(index, stage / 'README.md')})",
                "",
            ]
            lines += [
                f"- [{md_label(p['label'])} (PDF {p['pdf_page']})]({Path(p['path']).name})"
                for p in members
            ]
            write(index, "\n".join(lines))

        bookmark_path = stage / "bookmarks.md"
        lines = [
            f"# {md_label(title)}: PDF bookmark outline",
            "",
            "[Guideline index](README.md)",
            "",
            "The original bookmark hierarchy is retained. Destinations can be inaccurate in the source PDF. "
            "Where a bookmark's page code conflicts with its destination, both links are provided. "
            "Page files are grouped by their printed footer codes, independently of bookmarks.",
            "",
        ]
        if not outline:
            lines.append("This PDF has no bookmarks. Use the guideline's page index.")
        for bookmark in outline:
            prefix = "  " * bookmark["depth"] + "- "
            if bookmark["pdf_page"]:
                dest = pages[bookmark["pdf_page"] - 1]
                entry = f"[{md_label(bookmark['title'])}]({dest['path']}) (PDF {bookmark['pdf_page']})"
            else:
                entry = md_label(bookmark["title"]) + " — unresolved source destination"
            if bookmark["mismatch"]:
                alternatives = [
                    f"[PDF {n}]({pages[n - 1]['path']})"
                    for n in bookmark["printed_code_matches"]
                ]
                entry += (
                    "; **page-code mismatch**; matching printed code: "
                    + ", ".join(alternatives)
                )
            lines.append(prefix + entry)
        write(bookmark_path, "\n".join(lines))
        index = stage / "README.md"
        lines = [
            f"# {md_label(title)}",
            "",
            f"Version: {version or 'not detected'} · PDF pages: {len(pages)}",
            "",
            f"[Original PDF]({relative_link(index, pdf)}) · [Bookmark outline](bookmarks.md) · [Conversion report](conversion-report.md)",
            "",
            NOTE,
            "",
            "## Sections",
            "",
        ]
        lines += [
            f"- [{group}]({group}/README.md) — {len(members)} pages"
            for group, members in sorted(groups.items())
        ]
        lines += ["", "## All pages in source order", ""]
        lines += [
            f"- [{md_label(p['label'])} (PDF {p['pdf_page']})]({p['path']})"
            for p in pages
        ]
        write(index, "\n".join(lines))
        write(
            stage / "conversion-report.md",
            "\n".join(
                [
                    f"# Conversion report: {md_label(title)}",
                    "",
                    "[Guideline index](README.md)",
                    "",
                    f"Source SHA-256: `{source_hash}`",
                    "",
                    f"All {len(pages)} physical PDF pages have canonical Markdown files.",
                    "",
                    "Pages without recognized footer codes are retained in front matter or unclassified folders. "
                    "Discussion and references remain together under MS because their shared printed numbering does not reliably distinguish them.",
                    "",
                    "## Source/extraction observations",
                    "",
                    *(
                        [f"- {w}" for w in warnings]
                        or ["No extraction or bookmark issues detected."]
                    ),
                ]
            ),
        )
        manifest = dict(
            generator=GENERATOR,
            converter_version=VERSION,
            source_pdf=pdf.name,
            source_path=str(pdf),
            source_sha256=source_hash,
            title=title,
            version=version,
            page_count=len(pages),
            pages=pages,
            bookmarks=outline,
            warnings=warnings,
            created_utc=datetime.now(timezone.utc).isoformat(),
        )
        manifest["generated_files"] = {
            str(p.relative_to(stage)): sha256(p) for p in sorted(stage.rglob("*.md"))
        }
        write_json(stage / "manifest.json", manifest)
        problems = validate_document(stage)
        if problems:
            raise ValueError("Output validation failed: " + "; ".join(problems[:5]))
        if target.exists():
            # Only reached for explicitly requested --overwrite of marked generated output.
            backup = Path(
                tempfile.mkdtemp(prefix=f".{target.name}-backup-", dir=output)
            )
            backup.rmdir()
            target.rename(backup)
            try:
                stage.rename(target)
            except BaseException:
                backup.rename(target)
                raise
            shutil.rmtree(backup)
        else:
            stage.rename(target)
        return manifest
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def validate_document(folder: Path) -> list[str]:
    manifest = json.loads((folder / "manifest.json").read_text())
    errors = []
    numbers = [p["pdf_page"] for p in manifest["pages"]]
    if numbers != list(range(1, manifest["page_count"] + 1)):
        errors.append("Page inventory is incomplete or out of order")
    paths = [p["path"] for p in manifest["pages"]]
    if len(set(paths)) != len(paths):
        errors.append("Duplicate canonical page paths")
    for page in manifest["pages"]:
        if page["path"] not in manifest["generated_files"]:
            errors.append(
                f"Canonical page missing from generated-file inventory: {page['path']}"
            )
        path = folder / page["path"]
        if path.is_file():
            match = re.search(r"(?ms)^(`{3,})text\n(.*?)\n\1(?:\n|$)", path.read_text())
            if (
                not match
                or hashlib.sha256(match[2].encode()).hexdigest() != page["text_sha256"]
            ):
                errors.append(f"Extracted text integrity mismatch: {page['path']}")
    for name, digest in manifest["generated_files"].items():
        path = folder / name
        if not path.is_file() or sha256(path) != digest:
            errors.append(f"Missing/modified generated file: {name}")
            continue
        # Validate generated navigation, never interpret links inside extracted source text.
        outside_fence = re.sub(r"(?ms)^(`{3,})text\n.*?^\1\n?", "", path.read_text())
        for link in re.findall(r"\]\(([^\s)]+)\)", outside_fence):
            if urlsplit(link).scheme or link.startswith("#"):
                continue
            target = path.parent / unquote(link.split("#", 1)[0])
            if not target.is_file():
                errors.append(f"Broken local link in {name}: {link}")
    return errors


def library_index(output: Path) -> list[dict]:
    docs = []
    for path in sorted(output.glob("*/manifest.json")):
        data = json.loads(path.read_text())
        if data.get("generator") == GENERATOR:
            docs.append(
                dict(
                    folder=path.parent.name,
                    title=data["title"],
                    version=data["version"],
                    source_pdf=data["source_pdf"],
                    source_sha256=data["source_sha256"],
                    pages=data["page_count"],
                    observations=len(data["warnings"]),
                )
            )
    docs.sort(key=lambda d: d["title"].casefold())
    lines = [
        "# Guideline Markdown library",
        "",
        f"{len(docs)} guidelines · {sum(d['pages'] for d in docs)} PDF pages",
        "",
        "Start here, choose a guideline, then open its section index or bookmark outline. "
        "Each physical PDF page appears exactly once as a canonical Markdown file. "
        "Use the printed code and physical page number when citing a source.",
        "",
        NOTE,
        "",
        "Bookmarks and embedded links preserve the source PDF's destinations; check flagged bookmark mismatches. "
        "Search page text with a text-search tool when no heading or bookmark covers your topic. "
        "The JSON manifests provide page paths, source hashes, link destinations, and conversion observations.",
        "",
        "## Guidelines",
        "",
    ]
    for doc in docs:
        lines.append(
            f"- [{md_label(doc['title'])}]({doc['folder']}/README.md) — version {doc['version'] or 'unknown'}; {doc['pages']} pages"
        )
    write(output / "README.md", "\n".join(lines))
    write_json(
        output / "manifest.json",
        dict(generator=GENERATOR, converter_version=VERSION, guidelines=docs),
    )
    return docs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="External collection Markdown directory",
    )
    parser.add_argument(
        "--pdf", action="append", help="Convert a specific PDF filename; repeatable"
    )
    parser.add_argument(
        "--pdftotext",
        default=shutil.which("pdftotext"),
        help="Path to Poppler pdftotext",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild recognized generated guideline folders",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Check existing page inventories, file hashes, and local links",
    )
    args = parser.parse_args(argv)
    source = args.input_dir.resolve()
    output = args.output_dir.resolve()
    checkout = Path(__file__).resolve().parent.parent
    if (checkout / "pyproject.toml").is_file() and any(
        path.is_relative_to(checkout) or checkout.is_relative_to(path)
        for path in (source, output)
    ):
        parser.error(
            "Guideline sources and generated Markdown must stay outside the code repository"
        )
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    if args.validate_only:
        manifests = sorted(output.glob("*/manifest.json"))
        if not manifests:
            parser.error("No generated guidelines found")
        errors = []
        for marker in manifests:
            errors.extend(
                f"{marker.parent.name}: {e}" for e in validate_document(marker.parent)
            )
        print(
            "\n".join(errors)
            if errors
            else f"Validated {len(manifests)} guidelines: page coverage, file hashes, and local links OK."
        )
        return bool(errors)
    if not args.pdftotext:
        parser.error(
            "pdftotext is required. Install Poppler or supply --pdftotext /path/to/pdftotext"
        )
    pdfs = (
        [source / name for name in args.pdf]
        if args.pdf
        else sorted(source.glob("*.pdf"))
    )
    pdfs = [p.resolve() for p in pdfs]
    if not pdfs or any(not p.is_file() for p in pdfs):
        parser.error("No PDFs found, or a requested PDF does not exist")
    if len({slug(p.stem) for p in pdfs}) != len(pdfs):
        parser.error(
            "PDF filenames collide after normalization; rename the conflicting files"
        )
    output.mkdir(parents=True, exist_ok=True)
    root_manifest = output / "manifest.json"
    if root_manifest.exists():
        if json.loads(root_manifest.read_text()).get("generator") != GENERATOR:
            parser.error("Output folder contains an unrecognized manifest")
    elif any((output / name).exists() for name in ["README.md", "conversion-run.json"]):
        parser.error(
            "Output folder contains existing files that this converter would replace"
        )
    failures = []
    for i, pdf in enumerate(pdfs, 1):
        try:
            result = convert(pdf, output, args.pdftotext, args.overwrite)
            print(
                f"[{i}/{len(pdfs)}] {pdf.name}: {result['page_count']} pages, {len(result['warnings'])} observations",
                flush=True,
            )
        except Exception as exc:
            failures.append(dict(pdf=pdf.name, error=str(exc)))
            print(
                f"[{i}/{len(pdfs)}] FAILED {pdf.name}: {exc}",
                file=sys.stderr,
                flush=True,
            )
    docs = library_index(output)
    write_json(
        output / "conversion-run.json",
        dict(
            failed=failures, requested=len(pdfs), successful=len(pdfs) - len(failures)
        ),
    )
    print(
        f"Library: {len(docs)} guidelines, {sum(d['pages'] for d in docs)} pages. Index: {output / 'README.md'}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
