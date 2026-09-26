"""Assign source branches to disjoint page owners in code, never by model IDs."""

VERSION = "primary-branch-ownership-v1"


def page_owners(assignments, expected_pages):
    owners = {}
    for job, pages in assignments.items():
        for page in pages:
            if page in owners:
                raise ValueError(
                    f"Extraction ownership: {page} assigned to multiple calls"
                )
            owners[page] = job
    if set(owners) != set(expected_pages):
        raise ValueError(
            "Extraction ownership: page assignments do not cover the guideline exactly"
        )
    return owners


def branch_ledger(packets, expected_pages):
    owners = page_owners(
        {job: packet["primary_page_ids"] for job, packet in packets.items()},
        expected_pages,
    )
    branches = {}
    for job, packet in sorted(packets.items()):
        for index, candidate in enumerate(packet["result"]["candidates"], 1):
            anchor = candidate["defining_branch"]
            page = anchor["page_id"]
            if owners.get(page) != job:
                raise ValueError(
                    "Extraction ownership: candidate came from another page's owner"
                )
            lines = tuple(sorted(anchor["line_ids"]))
            key = (page, lines)
            branch = branches.setdefault(
                key,
                {
                    "defining_branch": {"page_id": page, "line_ids": list(lines)},
                    "owner_job": job,
                    "candidate_ids": [],
                },
            )
            # One branch may yield multiple spaces when alternatives cross fields.
            branch["candidate_ids"].append(f"{job}-c{index:03d}")
    return {
        "version": VERSION,
        "page_owners": owners,
        "branches": [branches[key] for key in sorted(branches)],
    }
