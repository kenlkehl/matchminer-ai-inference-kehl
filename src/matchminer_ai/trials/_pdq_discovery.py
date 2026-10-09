"""Inventory official PDQ treatment indexes without selecting clinical content."""

import re
from urllib.parse import urljoin, urlsplit

from ._pdq_sources import pdq_url

INDEXES = {
    "adult": "https://www.cancer.gov/publications/pdq/information-summaries/adult-treatment",
    "pediatric": "https://www.cancer.gov/publications/pdq/information-summaries/pediatric-treatment",
}

# Explicit collection scope, retained in the inventory as exclusions. Never
# silently discard an unfamiliar URL: it must be reviewed before plan creation.
NON_DISEASE_TOPICS = {
    "/about-cancer/managing-care/track-care-costs/financial-toxicity-hp-pdq": "Financial toxicity topic, not a cancer treatment catalog",
    "/types/brain/hp/child-brain-treatment-pdq": "Overview/index of separately cataloged childhood brain tumor summaries",
    "/types/head-and-neck/hp": "Index of separately cataloged head and neck treatment summaries",
    "/types/childhood-cancers/pediatric-genomics-hp-pdq": "Cross-disease genomics overview, not a cancer treatment catalog",
    "/types/childhood-cancers/late-effects-hp-pdq": "Survivorship and late effects topic, not a cancer treatment catalog",
    "/types/childhood-cancers/hp-stem-cell-transplant": "Cross-disease transplant overview, not a cancer treatment catalog",
    "/types/childhood-cancers/hp/rare-childhood-cancers-pdq": "Overview/index of separately cataloged rare childhood cancer summaries",
}


def treatment_inventory(pages):
    """Return included/excluded URLs with every label and source-index membership.

    ``pages`` maps the two official index URLs to original HTML bytes. Only
    health-professional links in each index article are selected. Duplicate
    alphabetical entries and cross-index appearances become aliases, not jobs.
    """
    from lxml import html

    if set(pages) != set(INDEXES.values()):
        raise ValueError("Provide both official PDQ treatment indexes")
    entries = {}
    for index, raw in pages.items():
        root = html.fromstring(raw)
        articles = root.xpath('//main[@id="main-content"]//article')
        if len(articles) != 1:
            raise ValueError("PDQ index must have one main-content article")
        found = 0
        for anchor in articles[0].xpath('.//a[@href]'):
            if " ".join(anchor.text_content().lower().split()) != "health professional":
                continue
            url = urljoin(index, anchor.get("href"))
            parsed = urlsplit(url)
            if (parsed.scheme != "https" or parsed.netloc not in {"www.cancer.gov", "cancer.gov"}
                or parsed.query or parsed.fragment):
                raise ValueError(f"Unrecognized PDQ index target: {url}")
            url = "https://www.cancer.gov" + parsed.path.rstrip("/")
            reason = NON_DISEASE_TOPICS.get(parsed.path.rstrip("/"))
            if reason is None:
                url = pdq_url(url)
            labels = [" ".join((li.text or "").split()) for li in anchor.iterancestors("li")]
            label = next((label for label in labels if label), "")
            if not label:
                raise ValueError(f"PDQ index link missing disease label: {url}")
            entry = entries.setdefault(url, {"url": url, "labels": [], "indexes": [],
                                            "included": reason is None})
            if reason:
                entry["exclusion_reason"] = reason
            for key, value in [("labels", label), ("indexes", index)]:
                if value not in entry[key]:
                    entry[key].append(value)
            found += 1
        if not found:
            raise ValueError("PDQ index contains no health-professional links")
    slugs = {}
    for url, entry in entries.items():
        if not entry["included"]:
            continue
        slug = re.sub(r"[^a-z0-9]+", "-", urlsplit(url).path.rsplit("/", 1)[-1].lower()).strip("-")
        slug = re.sub(r"(?:-hp)?-pdq$", "", slug)
        entry["disease"] = slug
        slugs.setdefault(slug, []).append(entry)
    # Adult/child summaries can share the final URL component. Use their full
    # URL paths for both names rather than merging independent source documents.
    for collisions in slugs.values():
        if len(collisions) > 1:
            for entry in collisions:
                path = urlsplit(entry["url"]).path.removeprefix("/types/").replace("/hp/", "/")
                entry["disease"] = re.sub(r"[^a-z0-9]+", "-", path.lower()).strip("-").removesuffix("-pdq")
    names = [entry["disease"] for entry in entries.values() if entry["included"]]
    if len(names) != len(set(names)):
        raise ValueError("Ambiguous PDQ source paths; choose explicit job names")
    return {"indexes": INDEXES, "entries": list(entries.values())}
