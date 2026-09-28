"""Model-selected literal passages for quotations that resist free-text copying."""

import json
import re
from dataclasses import replace

from matchminer_ai._storage import digest
from matchminer_ai.llm.structured import EndpointError

from ._guideline_context import pack_messages
from ._guideline_quotes import resolve_excerpt
from ._guideline_schema import STRINGS, arr, obj, validate_shape
from .prompt_builder import load_prompt_text

PROMPT = "guideline.quote_select.txt"
SELECTION = obj(
    {
        "selected_passages": arr({"type": "integer", "minimum": 1}),
        "unresolved": STRINGS,
    }
)
MARKER = "LITERAL_PASSAGE_SELECTION_JSON:\n"


def _words(text):
    stop = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "for",
        "to",
        "in",
        "with",
        "is",
        "are",
        "was",
        "be",
        "as",
        "by",
        "if",
        "on",
        "at",
    }
    return set(re.findall(r"\w+", text.lower())) - stop


def literal_choices(pages, assertion, limit=32):
    """Rank original text only; never repair words or choose clinical evidence."""
    rejected = assertion["rejected_evidence"]["source_text"]
    needle = _words(rejected)
    query = _words(assertion["assertion"]["name"])
    candidates = {}
    for page in pages.values():
        lines = page.text.splitlines(keepends=True)
        for i, line in enumerate(lines):
            # Whitespace-separated PDF columns stay separate initial fragments.
            # Ambiguous fragments gain adjacent *literal* text, not joined columns.
            for cell in re.finditer(r"\S(?:.*?\S)?(?= {3,}|\s*$)", line.rstrip("\r\n")):
                text = cell.group()
                words = _words(text)
                overlap = len(words & needle)
                if not overlap:
                    continue
                score = 5 * overlap / max(1, len(needle)) + len(words & query) / max(
                    1, len(query)
                )
                if " ".join(rejected.split()) in " ".join(text.split()):
                    score += 10
                options = [text, line.strip()]
                for radius in (1, 2):
                    options.extend(
                        [
                            "".join(lines[max(0, i - radius) : i + 1]).strip(),
                            "".join(lines[i : min(len(lines), i + radius + 1)]).strip(),
                        ]
                    )
                excerpt = None
                for candidate in options:
                    try:
                        resolve_excerpt(page, candidate)
                    except ValueError:
                        continue
                    excerpt = candidate
                    break
                if excerpt is None:
                    continue
                key = (page.id, excerpt)
                item = {
                    "page_id": page.id,
                    "source_text": excerpt,
                    "nearby_source_text": "".join(lines[max(0, i - 10) : i + 6]),
                }
                previous = candidates.get(key)
                if previous is None or score > previous[0]:
                    candidates[key] = (score, page.number, i, item)
    ranked = sorted(candidates.values(), key=lambda x: (-x[0], x[1], x[2]))
    return [{"choice": i, **row[3]} for i, row in enumerate(ranked[:limit], 1)]


def selected_patch(value, candidates, pages):
    validate_shape(value, SELECTION)
    selected = value["selected_passages"]
    if value["unresolved"] or not selected:
        raise ValueError(
            "No supporting literal passage selected: " + "; ".join(value["unresolved"])
        )
    if len(set(selected)) != len(selected) or any(
        i > len(candidates) for i in selected
    ):
        raise ValueError("Select distinct displayed passage choices only")
    evidence = []
    for i in selected:
        item = candidates[i - 1]
        if item["choice"] != i or item["page_id"] not in pages:
            raise ValueError("Passage choice references an unsupplied source")
        resolve_excerpt(pages[item["page_id"]], item["source_text"])
        evidence.append({k: item[k] for k in ("page_id", "source_text")})
    return {"evidence": evidence, "unresolved": []}


def select_literal_passages(client, guideline, pages, assertion):
    candidates = literal_choices(pages, assertion)
    if not candidates:
        raise EndpointError("No literal source passages available for citation repair")
    payload = {"assertion": assertion, "candidates": candidates}
    source_ids = {item["page_id"] for item in candidates}
    source_ids.update(
        r["page_id"] for r in assertion["existing_evidence"] if r["page_id"] in pages
    )
    context, _, messages = pack_messages(
        client,
        replace(guideline, pages={k: p for k, p in pages.items() if k in source_ids}),
        source_ids,
        assertion["assertion"]["name"],
        load_prompt_text(PROMPT),
        {},
        SELECTION,
        tail=MARKER + json.dumps(payload, ensure_ascii=False),
        population_guidance=False,
        system_message=load_prompt_text("guideline.citation_system.txt"),
    )
    supplied = {p.id: p for p in context}
    job = "quote-select-" + digest(messages)[:20]
    result = client.complete(
        job,
        messages,
        SELECTION,
        lambda v: selected_patch(v, candidates, supplied),
        reuse_exhausted=True,
    )
    return selected_patch(result, candidates, supplied), {
        "job": job,
        "result": result,
        "result_sha256": digest(result),
        "payload": payload,
    }


def audit_selection(proof, patch, assertion, accepted, pages):
    key = (proof["job"], proof["result_sha256"])
    if not isinstance(accepted, dict) or key not in accepted:
        raise ValueError("Passage selection lacks an accepted model response")
    record = accepted[key]
    payload = proof["payload"]
    if (
        payload["assertion"] != assertion
        or digest(proof["result"]) != proof["result_sha256"]
    ):
        raise ValueError("Passage selection assertion or response changed")
    rendered = MARKER + json.dumps(payload, ensure_ascii=False)
    if not any(rendered in m["content"] for m in record["request"]["body"]["messages"]):
        raise ValueError("Passage choices differ from the model request")
    if (
        proof["result"] != record["result"]
        or selected_patch(proof["result"], payload["candidates"], pages) != patch
    ):
        raise ValueError("Selected passages differ from the audited repair")
