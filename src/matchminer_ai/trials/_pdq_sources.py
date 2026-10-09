"""Complete, reproducible NCI PDQ HTML snapshots for the shared catalog pipeline.

This adapter selects no clinical recommendations. It preserves section hierarchy,
lists, table cells/spans, references and source links; the configured LLM performs
clinical extraction. Live contextualization's ranked excerpts are deliberately
not used as a substitute for a complete source document.
"""

from __future__ import annotations

import re
from urllib.parse import urljoin, urlsplit, urlunsplit

from matchminer_ai._storage import digest

FORMAT = "nci-pdq-html-v1"
GENERATOR = "nci-pdq-html-to-markdown"
REUSE_URL = "https://www.cancer.gov/policies/copyright-reuse"
SOURCE_NOTICE = (
    "Derived from NCI PDQ health-professional evidence summaries. PDQ summaries "
    "are not clinical practice guidelines. Research extraction requiring human "
    "review; not an NCI-endorsed product, a treatment recommendation, or an "
    "eligibility determination. Exact citations establish source occurrence, "
    "not clinical correctness or completeness. Listed alternatives are not a "
    "checklist of universally required care."
)


def pdq_url(value):
    """Only public health-professional summary URLs, never arbitrary fetch targets."""
    if not isinstance(value, str):
        raise ValueError("PDQ source URL must be a string")
    p = urlsplit(value)
    path = p.path.rstrip("/")
    hp_path = path.startswith("/types/") and (
        "/hp/" in path or "/hp-" in path or path.endswith("-hp-pdq")
    )
    treatment_topic = path.startswith("/about-cancer/treatment/types/") and path.endswith("-hp-pdq")
    # Two official treatment summaries currently omit the -pdq suffix. Their
    # HTML must still pass the same PDQ metadata/title/body checks below.
    legacy_treatment = path in {
        "/types/lymphoma/hp/mantle-cell-lymphoma-treatment",
        "/types/myeloproliferative/hp/myeloproliferative-neoplasms-treatment",
    }
    if (
        p.scheme != "https"
        or p.netloc not in {"www.cancer.gov", "cancer.gov"}
        or not ((hp_path and path.endswith("-pdq")) or treatment_topic or legacy_treatment)
        or p.query
        or p.fragment
    ):
        raise ValueError("Use an HTTPS cancer.gov health-professional PDQ summary URL")
    return urlunsplit(("https", "www.cancer.gov", p.path.rstrip("/"), "", ""))


def _text(element):
    # Concatenate inline markup without breaking words, but keep block boundaries
    # even when minified HTML has no whitespace between adjacent paragraphs/items.
    blocks = {"p", "div", "br", "li", "ul", "ol", "dt", "dd"}

    def parts(node):
        if node.text:
            yield node.text
        for child in node:
            if child.tag in blocks:
                yield " "
            yield from parts(child)
            if child.tag in blocks:
                yield " "
            if child.tail:
                yield child.tail

    return " ".join("".join(parts(element)).split())


def _anchor(element, root):
    for node in [element, *element.iterancestors()]:
        if node.get("id"):
            return node.get("id")
        if node is root:
            break
    return ""


def _table_lines(table):
    """Keep explicit cell boundaries and span attributes instead of guessing a grid."""
    from html import escape

    def content(node):
        # Retain structural markup inside a cell. Flattening adjacent nested
        # items can concatenate drug names or erase alternatives/combination logic.
        text = escape(node.text or "")
        retained = {
            "p",
            "div",
            "br",
            "ul",
            "ol",
            "li",
            "sup",
            "sub",
            "table",
            "caption",
            "thead",
            "tbody",
            "tfoot",
            "tr",
            "th",
            "td",
        }
        for child in node:
            if child.tag in retained:
                attrs = "".join(
                    f' {key}="{escape(child.get(key), quote=True)}"'
                    for key in ("rowspan", "colspan", "scope", "start", "type", "value")
                    if child.get(key)
                )
                text += (
                    "<br/>"
                    if child.tag == "br"
                    else f"<{child.tag}{attrs}>" + content(child) + f"</{child.tag}>"
                )
            else:
                text += content(child)
            text += escape(child.tail or "")
        return text

    lines = ["<table>"]
    for node in table:
        if node.tag == "caption":
            lines.append("<caption>" + escape(_text(node)) + "</caption>")
    for row in table.xpath(".//tr"):
        if next(row.iterancestors("table"), None) is not table:
            continue
        cells = []
        for cell in row:
            if cell.tag not in {"th", "td"}:
                continue
            attrs = "".join(
                f' {key}="{escape(cell.get(key), quote=True)}"'
                for key in ("rowspan", "colspan", "scope")
                if cell.get(key)
            )
            rendered = " ".join(content(cell).split())
            cells.append(f"<{cell.tag}{attrs}>{rendered}</{cell.tag}>")
        lines.append("<tr>" + "".join(cells) + "</tr>")
    return [*lines, "</table>"]


def parse_pdq_html(raw: bytes, url: str):
    """Render every article section without a relevance filter or text-length cap."""
    from lxml import html

    url = pdq_url(url)
    root = html.fromstring(raw)
    metas = {
        e.get("name", e.get("property", "")): e.get("content", "")
        for e in root.xpath("//meta[@name or @property]")
    }
    titles = root.xpath("//h1")
    if (
        len(titles) != 1
        or "Health Professional" not in _text(titles[0])
        or metas.get("dcterms.type") != "pdqCancerInfoSummary"
    ):
        raise ValueError("Response is not an NCI PDQ health-professional summary")
    bodies = root.xpath(
        '//*[@id="cgvBody"]/div[contains(concat(" ", normalize-space(@class), " "), " summary-sections ")]'
    )
    if len(bodies) != 1 or not bodies[0].xpath(".//h2"):
        raise ValueError(
            "PDQ article structure missing; refusing partial fallback text"
        )
    body = bodies[0]
    for node in body.xpath(".//script|.//style|.//nav|.//noscript|.//button"):
        node.drop_tree()
    title = _text(titles[0])
    dates = [e.get("datetime") for e in root.xpath("//time[@datetime]")]
    canonical = root.xpath('//link[@rel="canonical"]/@href')
    if canonical and pdq_url(canonical[0]) != url:
        raise ValueError("PDQ canonical URL differs from fetched summary")
    sections, headings, current = [], [], None

    def start(heading, level, anchor):
        nonlocal headings, current
        headings = [(n, t) for n, t in headings if n < level]
        headings.append((level, heading))
        path = [text for _, text in headings]
        editorial = bool(
            re.match(
                r"Latest Updates|Changes to This Summary|About This PDQ", path[0], re.I
            )
        )
        current = {
            "label": " > ".join(path),
            "section_path": path,
            "source_url": url + ("#" + anchor if anchor else ""),
            "primary": not editorial,
            "lines": ["#" * n + " " + text for n, text in headings],
            "links": [],
            "anchors": [anchor] if anchor else [],
            "warnings": [],
        }
        sections.append(current)

    def links(node):
        if current is None:
            return
        for target in [node, *node.xpath(".//*[@id]")]:
            if target.get("id") and target.get("id") not in current["anchors"]:
                current["anchors"].append(target.get("id"))
        for asset in node.xpath(".//img|.//svg|.//iframe|.//video|.//object"):
            current["warnings"].append(
                "Non-text content not interpreted: " + (asset.get("alt") or asset.tag)
            )
        for a in ([node] if node.tag == "a" and node.get("href") else []) + node.xpath(
            ".//a[@href]"
        ):
            target = urljoin(url, a.get("href"))
            if urlsplit(target).scheme in {"http", "https"}:
                item = {"text": _text(a), "url": target}
                if item not in current["links"]:
                    current["links"].append(item)

    def walk(node, list_depth=0, prefix=""):
        nonlocal current
        tag = node.tag
        if not isinstance(tag, str):
            return
        if re.fullmatch("h[2-6]", tag):
            start(_text(node), int(tag[1]), _anchor(node, body))
            links(node)
            return
        if tag in {"script", "style", "nav", "noscript", "button"}:
            return
        if current is None and tag in {"p", "table", "ul", "ol"}:
            start(title, 1, "")
        if (
            current is not None
            and node.get("id")
            and node.get("id") not in current["anchors"]
        ):
            current["anchors"].append(node.get("id"))
        if tag == "table":
            current["lines"].extend(_table_lines(node))
            links(node)
            return
        if tag in {"ul", "ol"}:
            items = [c for c in node if c.tag == "li"]
            step = -1 if tag == "ol" and "reversed" in node.attrib else 1
            index = int(node.get("start", len(items) if step < 0 else 1))
            for child in node:
                if child.tag == "li" and child.get("value"):
                    index = int(child.get("value"))
                walk(child, list_depth + 1, f"{index}. " if tag == "ol" else "- ")
                if child.tag == "li":
                    index += step
            return
        if tag == "li":
            # Keep nested list relationships, not one flattened paragraph.
            import copy

            own = copy.deepcopy(node)
            for nested in own.xpath("./ul|./ol"):
                nested.drop_tree()
            if _text(own):
                current["lines"].append(
                    "  " * max(0, list_depth - 1) + prefix + _text(own)
                )
            links(own)
            for child in node:
                if child.tag in {"ul", "ol"}:
                    walk(child, list_depth)
            return
        if tag in {"p", "blockquote", "pre", "figcaption", "dt", "dd"}:
            if _text(node):
                current["lines"].append(_text(node))
            links(node)
            return
        if tag in {"img", "svg", "iframe", "video", "object"}:
            if current is not None:
                current["warnings"].append(
                    "Non-text content not interpreted: " + (node.get("alt") or tag)
                )
            return
        # Preserve otherwise unwrapped article text; do not silently discard it.
        if tag == "a":
            links(node)
        if current is not None and node.text and node.text.strip():
            current["lines"].append(" ".join(node.text.split()))
        for child in node:
            walk(child, list_depth)
            if current is not None and child.tail and child.tail.strip():
                current["lines"].append(" ".join(child.tail.split()))

    walk(body)
    if not any(
        len(s["lines"]) > len(s["section_path"]) for s in sections if s["primary"]
    ):
        raise ValueError("PDQ summary has no substantive clinical text")
    for section in sections:
        section["text"] = "\n".join(section.pop("lines"))
    return sections, {
        "title": title,
        "url": url,
        "canonical_url": canonical[0] if canonical else url,
        "published_at": metas.get("dcterms.issued") or None,
        # Preserve the visible updated date independently of potentially different
        # content-system metadata dates. Do not invent an edition from fetch time.
        "updated_at": metas.get("dcterms.modified") or (dates[-1] if dates else None),
        "document_dates": dates,
        "attribution": "National Cancer Institute, " + title,
        "reuse_url": REUSE_URL,
        "reuse_notice": "NCI text reuse is subject to page-specific third-party material exceptions. No image assets downloaded.",
    }


def load_pdq_guideline(directory, metadata, manifest_bytes):
    """Verify raw HTML, deterministic rendering, links and metadata on every load."""
    from ._guideline_sources import Guideline, InputNotReady, Page, safe_child

    if metadata.get("source_format") != FORMAT:
        raise InputNotReady("Unsupported PDQ snapshot format")
    expected, documents = [], []
    for document in metadata["documents"]:
        raw = safe_child(directory, document["path"]).read_bytes()
        if digest(raw) != document["sha256"]:
            raise InputNotReady("PDQ HTML snapshot changed")
        sections, info = parse_pdq_html(raw, document["url"])
        for key, value in info.items():
            if document.get(key) != value:
                raise InputNotReady("PDQ document metadata differs from raw HTML")
        expected.extend(sections)
        documents.append(document["sha256"])
    records = metadata["pages"]
    if len(expected) != len(records) or len(records) != metadata["page_count"]:
        raise InputNotReady("PDQ section inventory differs from full HTML snapshot")
    pages = {}
    urls = {
        s["source_url"].split("#")[0] + "#" + anchor: i
        for i, s in enumerate(expected, 1)
        for anchor in s["anchors"]
    }
    for number, (section, record) in enumerate(zip(expected, records), 1):
        raw = safe_child(directory, record["path"]).read_bytes()
        if (
            record["section_number"] != number
            or any(
                record.get(k) != section[k]
                for k in (
                    "label",
                    "section_path",
                    "source_url",
                    "primary",
                    "links",
                    "anchors",
                    "warnings",
                )
            )
            or raw.decode("utf-8") != section["text"] + "\n"
            or digest(raw) != metadata["generated_files"][record["path"]]
        ):
            raise InputNotReady("PDQ section content/provenance differs from raw HTML")
        page_id = f"p{number:04d}"
        pages[page_id] = Page(
            page_id,
            number,
            record["label"],
            record["path"],
            section["text"],
            digest(raw),
            tuple(
                sorted({urls[x["url"]] for x in section["links"] if x["url"] in urls})
            ),
            source_url=section["source_url"],
            section_path=tuple(section["section_path"]),
        )
    if metadata["source_sha256"] != digest(documents):
        raise InputNotReady("PDQ aggregate source hash differs")
    if (
        metadata.get("source_kind") != "nci_pdq"
        or metadata.get("is_clinical_practice_guideline") is not False
        or metadata.get("title")
        != " / ".join(d["title"] for d in metadata["documents"])
        or metadata.get("version") != "snapshot-" + digest(documents)[:16]
        or metadata.get("notice") != SOURCE_NOTICE
        or metadata.get("warnings")
        != list(dict.fromkeys(w for s in expected for w in s["warnings"]))
    ):
        raise InputNotReady(
            "PDQ source identity/notice differs from verified documents"
        )
    if (directory / "manifest.json").read_bytes() != manifest_bytes:
        raise InputNotReady("PDQ manifest changed while verifying")
    fingerprint = digest(
        {
            "html": documents,
            "manifest": digest(manifest_bytes),
            "pages": {k: p.sha256 for k, p in pages.items()},
        }
    )
    return Guideline(directory, metadata, pages, fingerprint)
