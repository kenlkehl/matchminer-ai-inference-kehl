"""Add cohort-style key criteria to a completed guideline catalog as a derived edition (2026-11-14).

Catalogs extracted before the cohort-criteria step keep their audited records. This writes a new directory whose
records are the base catalog's, unchanged, plus ``key_criteria`` and ``key_criteria_items`` from one checkpointed
request per space (the pipeline's own step, :mod:`._guideline_cohort_criteria`). The base directory is only read.
The derived audit (``validation.json``) checks that:
- the base export is unchanged since the run started;
- every record equals its base record apart from the two fields;
- every accepted response matches its raw provider response;
- the criteria are those responses, still validate against their spaces and render to ``key_criteria``.
``status.json`` names the base catalog. Catalog discovery takes the most recently completed edition per disease, so a
derived edition supersedes its base without deleting it.
"""

import json
import shutil
from datetime import UTC, datetime
from functools import partial
from importlib import resources
from pathlib import Path

from matchminer_ai import __version__
from matchminer_ai._storage import atomic_json, digest, directory_lock, read_json

from ._guideline_cohort_criteria import JOB_PREFIX, PROMPT_FILES, VERSION as CRITERIA_VERSION
from ._guideline_cohort_criteria import audit_cohort_criteria, write_cohort_criteria

VERSION = "cohort-criteria-backfill-v1"
IMPLEMENTATION = "matchminer_ai.trials.add_guideline_cohort_criteria-v1"
ADDED_FIELDS = ("key_criteria", "key_criteria_items")
# Copied unchanged from the base so discovery and provenance work on the derived edition.
COPIED_FILES = ("sources.json",)


def now():
    return datetime.now(UTC).isoformat()


def _rows(data):
    return [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]


def _base_export(base):
    status, audit = read_json(base / "status.json"), read_json(base / "validation.json")
    data = (base / "paradigms.jsonl").read_bytes()
    if status.get("status") != "complete" or audit.get("status") != "passed":
        raise ValueError(f"{base}: the base catalog is not complete with a passing audit")
    if audit.get("paradigms_sha256") != digest(data):
        raise ValueError(f"{base}: the base export differs from its audited hash")
    rows = _rows(data)
    if not rows:
        raise ValueError(f"{base}: the base catalog is empty")
    if any(field in row for row in rows for field in ADDED_FIELDS):
        raise ValueError(f"{base}: the base catalog already has cohort-style key criteria")
    return status, audit, data, rows


def _identity_llm(llm):
    settings = json.loads(json.dumps(llm.public_dict()))
    for key in ("timeout", "attempts", "api_key_env", "stream", "max_concurrent_requests", "transport_base_url"):
        settings.pop(key, None)
    return settings


def backfill_cohort_criteria(catalog_dir, output, llm, client_type, *, workers, notify):
    """Write the derived edition in ``output``; return (rows, audit). Rerunning resumes from checkpoints."""
    from ._guideline_pipeline import export, parallel_jobs

    base, output = Path(catalog_dir).resolve(), Path(output).resolve()
    if base == output:
        raise ValueError("Write the derived edition to a new directory; the base catalog is only read")
    status, audit, data, rows = _base_export(base)
    prompt_files = resources.files("matchminer_ai.prompts")
    base_config = read_json(base / "run_config.json") if (base / "run_config.json").exists() else {}
    config = {
        "implementation": IMPLEMENTATION,
        "backfill_version": VERSION,
        "cohort_criteria": CRITERIA_VERSION,
        "derived_from": {
            "catalog": str(base),
            "paradigms_sha256": digest(data),
            "validation_sha256": digest(audit),
            "config_sha256": base_config.get("config_sha256"),
        },
        "prompt_resources_sha256": {name: digest(prompt_files.joinpath(name).read_bytes()) for name in PROMPT_FILES},
        "llm": _identity_llm(llm),
    }
    config_id = digest(config)
    output.mkdir(parents=True, exist_ok=True)
    with directory_lock(output):
        config_path = output / "run_config.json"
        if config_path.exists() and read_json(config_path).get("config_sha256") != config_id:
            raise ValueError("Base catalog, model or cohort-criteria prompts changed; choose a new output_dir")
        atomic_json(config_path, {**config, "config_sha256": config_id, "package_version": __version__,
                                  "llm_runtime": llm.public_dict(), "runtime": {"workers": workers}})
        state = {"status": "running", "stage": "cohort_criteria", "derived_from": str(base),
                 "paradigms": len(rows), "updated_utc": now()}
        atomic_json(output / "status.json", state)
        try:
            client = client_type(llm, output / "checkpoints")
            title = rows[0]["source"]["title"]
            run_jobs = partial(parallel_jobs, notify=notify)
            derived = write_cohort_criteria(client, rows, title, run_jobs, workers, notify=notify)
            export(output, derived)
            for name in COPIED_FILES:
                if (base / name).exists():
                    shutil.copy2(base / name, output / name)
            result = audit_backfill(output)
            atomic_json(output / "validation.json", result)
            state.update(
                status="complete", stage="export", updated_utc=now(),
                **{key: status[key] for key in ("source_pages", "candidates", "review_required",
                                                "clinical_completeness") if key in status},
            )
            atomic_json(output / "status.json", state)
        except BaseException as exc:
            state.update(status="failed", updated_utc=now(), error=f"{type(exc).__name__}: {exc}")
            atomic_json(output / "status.json", state)
            raise
    notify(f"{rows[0]['source']['disease']}: derived edition with cohort-style key criteria in {output}")
    return derived, result


def audit_backfill(output):
    """Check a derived edition against its base catalog and its own checkpoints (no LLM calls)."""
    from ._guideline_audit import audit_accepted_response, require

    output = Path(output)
    config = read_json(output / "run_config.json")
    require(config.get("implementation") == IMPLEMENTATION, "Not a cohort-criteria derived edition")
    base = Path(config["derived_from"]["catalog"])
    base_status, base_audit, base_data, base_rows = _base_export(base)
    require(digest(base_data) == config["derived_from"]["paradigms_sha256"], "The base export changed")
    data = (output / "paradigms.jsonl").read_bytes()
    rows = _rows(data)
    require(len(rows) == len(base_rows), "The derived edition has a different number of records")
    for row, base_row in zip(rows, base_rows):
        require({k: v for k, v in row.items() if k not in ADDED_FIELDS} == base_row,
                f"{base_row.get('paradigm_id')}: a record differs from its base apart from the key criteria")
    results, accepted_count, raw_checked = [], 0, 0
    for path in (output / "checkpoints").glob("*/accepted.json"):
        accepted = read_json(path)
        require(accepted["request_sha256"] == path.parent.name, "Checkpoint identity differs")
        require(accepted["result_sha256"] == digest(accepted["result"]), "Checkpoint is corrupted")
        accepted_count += 1
        audit_accepted_response(path.parent, accepted)
        raw_checked += 1
        if accepted["job"].startswith(JOB_PREFIX):
            results.append(accepted["result"])
    audit_cohort_criteria(rows, results)
    return {
        "status": "passed",
        "paradigms": len(rows),
        "paradigms_sha256": digest(data),
        "cohort_criteria": CRITERIA_VERSION,
        "backfill_version": VERSION,
        "derived_from": {
            "catalog": str(base),
            "paradigms_sha256": digest(base_data),
            "audit_status": base_audit.get("status"),
            "audit_paradigms": base_audit.get("paradigms"),
            "status_updated_utc": base_status.get("updated_utc"),
        },
        "source_fingerprint": base_audit.get("source_fingerprint"),
        "model": config["llm"].get("model"),
        "accepted_requests": accepted_count,
        "verified_raw_provider_responses": raw_checked,
        "clinical_correctness": "not established by these structural and provenance checks",
    }
