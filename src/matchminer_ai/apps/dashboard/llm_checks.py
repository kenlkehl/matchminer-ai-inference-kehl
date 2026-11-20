"""The dashboard's LLM checks, through the inference package and the configured endpoint (``llm.py``):

- the **GoodOption check** of a patient and a trial (``matchminer_ai.matching.check_good_options``, the package's
  on-demand entry point around ``score_good_options_with_llm``: its four-criterion rubric for each of the trial's
  scoreable drugs, with the catalog's drug and class evidence, the output cap, the endpoint's context and the parse
  retries of the ``good_option_check`` preset section);
- the **per-criterion LLM check** of a patient and a cohort (``check_eligibility_criteria_with_llm``, one request per
  tagged line with the per-criterion teacher prompt). It is new in the inference package (branch
  ``criteria-llm-check``); with a package without it the check is unavailable, and the dashboard says so.

Each runs as a background job (its own pool, so the decision model's queue is not held up). Answers are cached in the
app's SQLite cache (``llm_checks.sqlite``), keyed by the patient's text, the trial or cohort, the catalog's file stamps
(GoodOption), the endpoint and model, the prompts, the effective request settings and the package version, so the
same inputs answer at once; "Run again" asks again. The GoodOption catalog is loaded on the first check (about 10 s
and 1.2 GB, validated by the package, then kept until its files change).
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import threading
import time

from .core.cache import ScoreCache, text_key
from .core.jobs import JobManager
from .drugs import catalog_stamps
from .llm import (DEFAULT_BACKEND, NEMOTRON, NOTICE, VERTEX_LABELS, VERTEX_PROFILES, Connection, default_endpoint,
                  vertex_credentials_error, vertex_project)

CACHE_VERSION = "mmai-dash-llm-checks-v1"
GOOD_OPTION_PROMPTS = ("llm_good_option.system.txt", "llm_good_option.user.txt", "llm_good_option.rubric.txt",
                       "llm_good_option.retry.txt")
GOOD_OPTION_STATUSES = {
    "ok": "Scored against the rubric.",
    "parse_failed": "The LLM's answer never passed the package's validation, after its retries.",
    "no_scoreable_drug": "The GoodOption catalog lists no investigational drug to score for this trial.",
    "missing_catalog_trial": "The trial is not in the GoodOption catalog.",
    "trial_ctgov_blocked": "The catalog could not read the trial's ClinicalTrials.gov record.",
    "missing_drug_summary": "One of the trial's scoreable drugs has no catalog summary.",
    "drug_research_blocked": "The catalog's research on one of the trial's scoreable drugs did not complete.",
}
# Protocol-derived criteria are confidential documents. Since 2026-11-17 (user: "ours is covered for confidential
# data") they may go to the site's Vertex endpoint as well as its on-prem server. The patient summaries are synthetic.
# The package's own words on what the score is (check_good_options' docstring and the GoodOption API docs).
SCORE_MEANING = ("The score is an evidence fraction for human review, not an eligibility decision or treatment "
                 "recommendation: each scored drug answers four yes-or-no questions, a point each, and the trial's "
                 "score is its points over four per drug.")


def _number(value):
    """A JSON-safe number (NaN and pandas' NA become None)."""
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else (int(number) if number.is_integer() else number)


def _scalars(value: dict | None) -> dict:
    """The small scalar entries of a metadata mapping (model, server, revision ...)."""
    return {k: v for k, v in (value or {}).items() if isinstance(v, (str, int, float, bool)) and len(str(v)) < 300}


def rubric() -> dict:
    """The GoodOption rubric in the inference package's own words: its introduction and the four criteria."""
    from matchminer_ai.llm.prompts import load_prompt_text

    text = load_prompt_text("llm_good_option.rubric.txt").strip()
    parts = re.split(r"(?m)^\d\. (\w+) — (.+)$", text)
    criteria = [{"criterion": parts[i], "question": parts[i + 1].strip(), "text": parts[i + 2].strip()}
                for i in range(1, len(parts) - 2, 3)]
    return {"intro": parts[0].strip(), "criteria": criteria, "score_meaning": SCORE_MEANING}


class LLMChecks:
    def __init__(self, settings, connection: Connection, cache: ScoreCache, workers: int = 4):
        self.settings = settings
        self.connection = connection
        self.cache = cache
        self.jobs = JobManager(workers=workers)
        self.lock = threading.Lock()
        self.running: dict[str, str] = {}  # cache key -> job id, so a double click joins the running job
        self._base = None

    # ------------------------------------------------------------------ configuration

    def base_config(self):
        from matchminer_ai import load_default_preset

        if self._base is None:
            self._base = load_default_preset()
        return self._base

    def set_connection(self, connection: Connection) -> None:
        self.connection = connection

    @staticmethod
    def package() -> dict:
        import matchminer_ai

        return {"version": getattr(matchminer_ai, "__version__", None), "path": str(Path(matchminer_ai.__file__).parent)}

    def connection_unavailable(self) -> str | None:
        if not self.connection.configured:
            return "No LLM endpoint is configured (Data and Models)."
        return vertex_credentials_error() if self.connection.vertex else None

    def good_option_unavailable(self) -> str | None:
        if reason := self.connection_unavailable():
            return reason
        if not (Path(self.settings.good_options_catalog) / "manifest.json").is_file():
            return f"No GoodOption catalog at {self.settings.good_options_catalog}."
        return None

    def criteria_unavailable(self) -> str | None:
        if reason := self.connection_unavailable():
            return reason
        import matchminer_ai.matching as matching

        if not hasattr(matching, "check_eligibility_criteria_with_llm"):
            return (f"The installed inference package ({self.package()['path']}) has no "
                    "check_eligibility_criteria_with_llm; it is on the branch criteria-llm-check, not merged yet.")
        if "llm_criteria_check" not in self.base_config().raw:
            return "The installed inference package's preset has no llm_criteria_check section."
        return None

    def describe(self) -> dict:
        return {"connection": self.connection.describe(), "notice": NOTICE,
                "defaults": {"backend": DEFAULT_BACKEND, "endpoint": default_endpoint(), "model": NEMOTRON},
                "vertex_profiles": {k: {**v, "label": VERTEX_LABELS[k], "project": vertex_project()}
                                    for k, v in VERTEX_PROFILES.items()},
                "good_option": {"available": self.good_option_unavailable() is None,
                                "reason": self.good_option_unavailable()},
                "criteria": {"available": self.criteria_unavailable() is None, "reason": self.criteria_unavailable()},
                "package": self.package(), "rubric": rubric(), "jobs": self.jobs.recent(10)}

    # ------------------------------------------------------------------ keys

    def _contract(self, kind: str, config) -> str:
        """The prompts and effective request settings of a check, with the package version."""
        from matchminer_ai.llm.prompts import load_prompt_text

        if kind == "good_option":
            prompts = {name: load_prompt_text(name) for name in GOOD_OPTION_PROMPTS}
            settings = {name: config.raw.get(name) for name in ("good_option_check", "good_option_prompt")}
            settings["llm_good_option"] = config.llm_good_option
        else:
            from matchminer_ai.labeling.tasks import contract_hash

            prompts = {"criteria_contract": contract_hash("criteria")}
            settings = {"llm_criteria_check": config.raw.get("llm_criteria_check")}
        return hashlib.sha256(json.dumps([prompts, settings, self.package()["version"]], sort_keys=True,
                                         default=str).encode()).hexdigest()

    def good_option_key(self, patient_text: str, trial_id: str, connection: Connection, config) -> str:
        return text_key(CACHE_VERSION, "good_option", connection.identity(), trial_id, patient_text.strip(),
                        catalog_stamps(self.settings.good_options_catalog), self._contract("good_option", config))

    def criteria_key(self, pair, connection: Connection, config) -> str:
        return text_key(CACHE_VERSION, "criteria", connection.identity(), pair.cohort_id, pair.patient_key.strip(),
                        pair.patient_boilerplate.strip(), pair.items, self._contract("criteria", config))

    # ------------------------------------------------------------------ requests

    def _start(self, key: str, kind: str, label: str, run, rerun: bool) -> dict:
        """The cached answer, else the running job for the same inputs, else a new job."""
        if not rerun:
            cached = self.cache.get_many([key]).get(key)
            if cached:
                return {"result": {**cached, "cached": True}}
        with self.lock:
            job = self.jobs.get(self.running.get(key, ""))
            if job is not None and job.status in ("queued", "running"):
                return {"job": job.describe(with_result=False)}

            def body(progress):
                try:
                    value = run(progress)
                    self.cache.put_many({key: value})
                    return value
                finally:
                    with self.lock:
                        self.running.pop(key, None)

            job = self.jobs.submit(kind, label, body, device="endpoint")
            self.running[key] = job.id
        return {"job": job.describe(with_result=False)}

    def good_option(self, patient: dict, trial_id: str, drugs: list[dict], rerun: bool = False) -> dict:
        """Start (or return the cached) GoodOption check of a patient (``id`` and ``key``, the cancer history the
        dashboard scores with) and a trial; ``drugs`` are the store's drug entries of the trial (class links)."""
        reason = self.good_option_unavailable()
        if reason:
            raise ValueError(reason)
        connection = self.connection
        config = connection.configure(self.base_config(), ("llm_good_option",))
        key = self.good_option_key(patient["key"], trial_id, connection, config)
        return self._start(key, "good_option", f"GoodOption check {patient['id']} x {trial_id}",
                           lambda progress: self._run_good_option(patient, trial_id, drugs, connection, config,
                                                                  progress), rerun)

    def criteria(self, pair, rerun: bool = False) -> dict:
        """Start (or return the cached) per-criterion LLM check of a patient and a cohort (a ``PairText``)."""
        reason = self.criteria_unavailable()
        if reason:
            raise ValueError(reason)
        if not pair.items:
            raise ValueError("this cohort has no tagged criteria lines (per-criterion checks need them)")
        connection = self.connection
        config = connection.configure(self.base_config(), ("llm_criteria_check",))
        key = self.criteria_key(pair, connection, config)
        return self._start(key, "criteria_llm", f"LLM criteria check {pair.patient_id} x {pair.cohort_id}",
                           lambda progress: self._run_criteria(pair, connection, config, progress), rerun)

    # ------------------------------------------------------------------ the work

    def _run_good_option(self, patient, trial_id, drugs, connection, config, progress) -> dict:
        import pandas as pd
        from matchminer_ai.matching import check_good_options
        from matchminer_ai.trials import load_good_option_catalog
        from matchminer_ai.trials.drug_evidence import RUBRIC_CRITERIA

        progress(0, 2, "loading the GoodOption catalog (about 10 s, once)")
        catalog = load_good_option_catalog(self.settings.good_options_catalog, cache=True)
        summaries = (catalog.scoreable_summaries_for_trial(trial_id) if catalog.trial_status(trial_id) == "ok"
                     else ())
        progress(1, 2, f"asking {connection.label} about {len(summaries)} "
                       f"{'drug' if len(summaries) == 1 else 'drugs'}")
        started = time.monotonic()
        frame = pd.DataFrame([{"patient_id": patient["id"], "trial_id": trial_id,
                               "cancer_history_summary": patient["key"]}])
        out, metadata = check_good_options(frame, catalog=catalog, config=config, return_metadata=True)
        row = out.to_dict("records")[0]
        by_name = {s.preferred_name.casefold(): s for s in summaries}
        store_drugs = {d.get("drug_id"): d for d in drugs}
        assessments = []
        for found in row.get("good_option_drug_assessments") or []:
            summary = by_name.get(str(found.get("drug_name", "")).casefold())
            drug_id = summary.drug_id if summary else None
            entry = store_drugs.get(drug_id) or {}
            criteria = [{"criterion": c, "point": int(found[c]["point"]), "rationale": found[c]["rationale"]}
                        for c in RUBRIC_CRITERIA if isinstance(found.get(c), dict)]
            assessments.append({"drug_name": found.get("drug_name"), "drug_id": drug_id,
                                "role": entry.get("role"), "mechanism_types": entry.get("mechanism_types") or [],
                                "class_ids": entry.get("class_ids") or {},
                                "targeted_biomarkers": found.get("targeted_biomarkers") or [],
                                "points": sum(c["point"] for c in criteria), "criteria": criteria})
        status = str(row.get("good_option_status") or "")
        settings = config.raw.get("good_option_check") or {}
        progress(2, 2, "done")
        return {"kind": "good_option", "patient_id": patient["id"], "trial_id": trial_id, "status": status,
                "status_text": GOOD_OPTION_STATUSES.get(status, status), "score": _number(row.get("good_option_score")),
                "points": _number(row.get("good_option_points")),
                "max_points": _number(row.get("good_option_max_points")),
                "drug_count": _number(row.get("good_option_drug_count")),
                "patient_disease_type": row.get("good_option_patient_disease_type") or "",
                "assessments": assessments, "uncertainties": list(row.get("good_option_uncertainties") or []),
                "reasoning": row.get("good_option_reasoning_text") or "",
                "finish_reason": row.get("good_option_finish_reason") or "",
                "parse_error": row.get("good_option_parse_error") or "",
                "parse_attempts": _number(row.get("good_option_parse_attempts")),
                "reasoning_off_fallback_used": row.get("good_option_reasoning_off_fallback_used"),
                "max_parse_attempts": settings.get("max_parse_attempts"),
                "reasoning_off_fallback": settings.get("reasoning_off_fallback"),
                "evidence_packing": metadata.get("evidence_packing") or {},
                "context": metadata.get("good_option_check") or {},
                "model_metadata": _scalars((metadata.get("model_metadata") or {}).get("llm_good_option")),
                "connection": connection.describe(), "seconds": round(time.monotonic() - started, 1),
                "checked_at": time.time(), "catalog_compatibility_id": metadata.get("catalog_compatibility_id"),
                "package": self.package()}

    def _run_criteria(self, pair, connection, config, progress) -> dict:
        import pandas as pd
        from matchminer_ai.matching import check_eligibility_criteria_with_llm

        progress(0, len(pair.items), f"asking {connection.label}")
        started = time.monotonic()
        frame = pd.DataFrame([{"patient_id": pair.patient_id, "cohort_trial_id": pair.cohort_id,
                               "cancer_history_summary": pair.patient_key,
                               "general_exclusion_criteria_evidence": pair.patient_boilerplate,
                               "cohort_criteria_items": json.dumps(pair.items, ensure_ascii=False)}])
        out, metadata = check_eligibility_criteria_with_llm(
            frame, config=config, return_metadata=True,
            progress=lambda done, total: progress(done, total, f"{done} of {total} lines answered"))
        details = metadata["model_metadata"]["check_eligibility_criteria_with_llm"]
        lines = [{"number": int(r["criterion_number"]), "type": r["criterion_type"], "boilerplate": bool(r["boilerplate"]),
                  "text": r["criterion_text"], "answer": r["llm_criterion_answer"],
                  "fails": r["llm_criterion_clearly_fails"], "status": r["llm_criterion_status"],
                  "reason": r["llm_criterion_reason"], "finish_reason": r["llm_criterion_finish_reason"],
                  "attempts": int(r["llm_criterion_attempts"]), "parse_status": r["llm_criterion_parse_status"],
                  "reasoning": r.get("llm_criterion_reasoning_text") or ""} for r in out.to_dict("records")]
        return {"kind": "criteria_llm", "patient_id": pair.patient_id, "cohort_id": pair.cohort_id, "lines": lines,
                "counts": {"lines": len(lines), "clearly_fail": sum(1 for line in lines if line["fails"] is True),
                           "unanswered": sum(1 for line in lines if line["answer"] is None),
                           **{k: details.get(k) for k in ("requests", "retries", "parse_failures", "unchecked_lines")}},
                "structured_outputs": details.get("structured_outputs"), "contract": metadata.get("criteria_contract"),
                "model_metadata": _scalars(details), "connection": connection.describe(),
                "seconds": round(time.monotonic() - started, 1), "checked_at": time.time(), "package": self.package()}
