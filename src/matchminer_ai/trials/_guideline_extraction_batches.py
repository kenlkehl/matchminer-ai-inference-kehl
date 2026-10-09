"""Split exhausted extraction tasks while preserving one owner per source page."""

from matchminer_ai._storage import atomic_json, read_json
from ._guideline_ownership import page_owners

VERSION = "adaptive-extraction-packets-v1"


def can_split(error, size):
    return size > 1 and (
        "exhausted" in error or "Required evidence and instructions exceed input budget" in error
    ) and any(marker in error for marker in (
        "page_coverage must account", "states_extracted requires", "mixes AND and OR",
        "Unknown or unsupplied evidence page:", "nonexistent source line IDs",
        "Retry prompt exceeds input budget", "Required evidence and instructions exceed input budget",
        "Incomplete response: finish_reason='length'",
    ))


def validate_ledger(ledger, guideline):
    if ledger.get("version") != VERSION or ledger.get("source_fingerprint") != guideline.fingerprint:
        raise ValueError("Extraction split ledger differs from source")
    for row in ledger["splits"].values():
        ids = row["primary_page_ids"]
        children = row["children"]
        if (len(ids) != len(set(ids)) or set(ids) - guideline.pages.keys()
            or not can_split(row["reason"], len(ids))
            or not children or any(not x or len(x) >= len(ids) for x in children)
            or [p for child in children for p in child] != ids):
            raise ValueError("Extraction split changed page ownership")


def run_extractions(jobs, guideline, output, workers, run_jobs, extract, notify):
    path = output / "extraction_packet_splits.json"
    ledger = read_json(path) if path.exists() else {
        "version": VERSION, "source_fingerprint": guideline.fingerprint, "splits": {},
    }
    validate_ledger(ledger, guideline)
    saved = output / "extraction.json"
    previous = read_json(saved).get("failures", {}) if saved.exists() else {}

    def mark(key, pages, error):
        ids = [p.id for p in pages]
        middle = len(ids) // 2
        ledger["splits"][key] = {"primary_page_ids": ids, "children": [ids[:middle], ids[middle:]], "reason": error}
        atomic_json(path, ledger)
        notify(f"{guideline.disease}: splitting exhausted {key} into smaller owned extraction packets")

    def expand(key, pages):
        if key not in ledger["splits"] and can_split(previous.get(key, ""), len(pages)):
            mark(key, pages, previous[key])
        row = ledger["splits"].get(key)
        if row is None:
            return [(key, pages)]
        if row["primary_page_ids"] != [p.id for p in pages]:
            raise ValueError("Extraction split input pages changed")
        return [job for i, child in enumerate(row["children"], 1)
                for job in expand(f"{key}-part{i:04d}", [guideline.pages[p] for p in child])]

    pending = [job for key, pages in jobs for job in expand(key, pages)]
    values, failures = {}, {}
    while pending:
        results, errors = run_jobs(pending, workers, extract)
        values.update(results)
        failures.update(errors)
        atomic_json(saved, {"packets": values, "failures": failures})
        following = []
        for key, pages in pending:
            if can_split(errors.get(key, ""), len(pages)):
                mark(key, pages, errors[key])
                failures.pop(key)
                following.extend(expand(key, pages))
        pending = following
    if not failures:
        page_owners({key: v["primary_page_ids"] for key, v in values.items()},
                    [p.id for p in guideline.primary])
    return values, failures
