"""The decision model (LiquidAI d1, in-process with Transformers) through the inference package's public stages.

- ``score_match_quality``: the five match questions on the patient's key section and the cohort's key criteria
  (the 0-1 match score is the package's expected-score formula);
- ``exclusion_criteria_check``: the boilerplate question on the two boilerplate sections;
- ``check_eligibility_criteria``: one question per tagged criteria line (yes = does not clearly fail it);
- ``interpret_match_quality`` / ``interpret_exclusion_criteria``: gradient-times-input token attribution.

The app builds no states, questions or scores of its own. Answers are cached in the app's SQLite score cache,
keyed by the model's fingerprint, the package's question digest and the exact input texts.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import threading
import time
from typing import Callable

from .cache import ScoreCache, text_key

MATCH_LABELS = ("reasonableness", "cancer_type", "burden", "treatment_history", "biomarker")
FAMILY_NAMES = {"match": "match questions 1-5", "boilerplate": "boilerplate question 6",
                "criterion": "per-criterion question 7"}
Progress = Callable[[int, int, str], None]


@dataclass
class PairText:
    patient_id: str
    cohort_id: str
    patient_key: str
    patient_boilerplate: str
    cohort_key: str
    cohort_boilerplate: str
    items: list | None = None


class DecisionService:
    def __init__(self, settings, score_cache: ScoreCache, cache_dir: Path):
        self.settings = settings
        self.cache = score_cache
        self.model, self.revision, self.why = settings.resolved_decision_model()
        self.local = Path(self.model).is_dir()
        self.device = str(settings.device)  # the loaded weights' device once loaded (e.g. "cuda" becomes "cuda:0")
        self.contract = self._contract()
        self.fingerprint = (self.contract or {}).get("fingerprint") or (f"{self.model}@{self.revision}"
                                                                         if not self.local else None)
        if self.fingerprint is None:  # a local checkpoint without a contract: name it by its files' stamps
            from .cache import file_stamp
            self.fingerprint = text_key({p.name: file_stamp(p) for p in sorted(Path(self.model).glob("*"))
                                         if p.is_file()})
        from matchminer_ai.matching.decision_model import questions_digest

        self.questions = questions_digest()
        self.config = self._config()
        self.lock = threading.Lock()
        self.loaded_at: float | None = None
        self.load_seconds: float | None = None
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        (Path(cache_dir) / "effective_decision_model.json").write_text(
            json.dumps(self.config.raw["decision_model"], indent=2) + "\n")

    # ------------------------------------------------------------------ configuration

    def _contract(self) -> dict | None:
        path = Path(self.model) / "decision_contract.json"
        return json.loads(path.read_text()) if self.local and path.exists() else None

    def _config(self):
        from matchminer_ai import load_default_preset

        config = load_default_preset()
        section = dict(config.raw.get("decision_model") or {})
        section.update(model_name=self.model, device=self.settings.device, dtype=self.settings.dtype,
                       batch_size=int(self.settings.decision_batch_size),
                       token_budget=int(self.settings.decision_token_budget))
        section["revision"] = self.revision or section.get("revision")
        config.raw["decision_model"] = section
        return config

    def trained_states(self) -> dict | None:
        """States per question family in the checkpoint's training contract (None for the base model)."""
        return (self.contract or {}).get("states")

    def untrained_families(self) -> list[str]:
        states = self.trained_states()
        if states is None:
            return []
        return [family for family in ("match", "boilerplate", "criterion") if not states.get(family)]

    def warnings(self) -> list[str]:
        notes = []
        if self.contract is None and not self.local:
            notes.append("Base d1-2B, zero-shot: no question family was fine-tuned.")
        for family in self.untrained_families():
            notes.append(f"This checkpoint was not trained on the {FAMILY_NAMES[family]} (0 training states); "
                         "treat those answers as untrained.")
        contract_questions = (self.contract or {}).get("questions_sha256")
        if contract_questions and contract_questions != self.questions:
            notes.append("The checkpoint was trained on different question texts than this inference package asks "
                         f"({contract_questions[:12]}... vs {self.questions[:12]}...).")
        return notes

    def describe(self) -> dict:
        return {"model": self.model, "revision": self.revision, "why": self.why, "fingerprint": self.fingerprint,
                "questions_sha256": self.questions, "contract": self.contract, "trained_states": self.trained_states(),
                "untrained_families": self.untrained_families(), "warnings": self.warnings(),
                "device": self.device, "dtype": self.settings.dtype, "threads": self.settings.threads,
                "loaded": self.loaded_at is not None, "load_seconds": self.load_seconds,
                # match_quality: P(reasonable) for a pass (inference reasonableness_cutoff, since 2026-11-17).
                "cutoffs": {"match_quality": self.config.raw.get("match_quality", {}).get("reasonableness_cutoff", 0.5),
                            "exclusion_criteria": self.config.raw.get("exclusion_criteria", {}).get("pass_cutoff"),
                            "criteria_check": self.config.raw.get("criteria_check", {}).get("fail_cutoff")}}

    def _prepare(self):
        """Thread limits and the model load (once); the package caches the loaded model."""
        if self.loaded_at is not None:
            return
        import torch

        if str(self.settings.device) == "cpu":
            torch.set_num_threads(max(1, int(self.settings.threads)))
        from matchminer_ai.matching.decision_model import decision_settings, load_decision_model

        started = time.monotonic()
        model = load_decision_model(decision_settings(self.config))
        self.device = str(model.get_input_embeddings().weight.device)
        self.load_seconds = round(time.monotonic() - started, 1)
        self.loaded_at = time.time()

    def preload(self, progress: Progress | None = None) -> dict:
        """Load the model before the first request (a startup job), so the first scores do not wait for it."""
        if progress:
            progress(0, 1, f"loading {Path(self.model).name if self.local else self.model}")
        with self.lock:
            self._prepare()
        if progress:
            progress(1, 1, f"loaded in {self.load_seconds} s")
        return {"load_seconds": self.load_seconds}

    # ------------------------------------------------------------------ keys

    def _key(self, family: str, *texts) -> str:
        return text_key(family, self.fingerprint, self.questions, *texts)

    def match_key(self, pair: PairText) -> str:
        return self._key("match", pair.patient_key.strip(), pair.cohort_key.strip())

    def boilerplate_key(self, pair: PairText) -> str:
        return self._key("boilerplate", pair.patient_boilerplate.strip(), pair.cohort_boilerplate.strip())

    def criteria_key(self, pair: PairText) -> str:
        return self._key("criterion", pair.patient_key.strip(), pair.patient_boilerplate.strip(), pair.items)

    def cached_scores(self, pairs: list[PairText]) -> list[dict]:
        """Whatever is already cached for these pairs (no model work)."""
        keys = [(self.match_key(p), self.boilerplate_key(p)) for p in pairs]
        found = self.cache.get_many([k for pair in keys for k in pair])
        return [{"match": found.get(m), "boilerplate": found.get(b) if p.cohort_boilerplate.strip()
                 else {"skipped": "the cohort has no boilerplate criteria"}} for p, (m, b) in zip(pairs, keys)]

    # ------------------------------------------------------------------ scoring

    def score(self, pairs: list[PairText], families=("match", "boilerplate"), progress: Progress | None = None,
              chunk: int = 2) -> list[dict]:
        import pandas as pd
        from matchminer_ai.matching import exclusion_criteria_check, score_match_quality

        match_keys = [self.match_key(p) for p in pairs]
        bp_keys = [self.boilerplate_key(p) for p in pairs]
        found = self.cache.get_many(match_keys + bp_keys)
        todo_match = list({k: p for k, p in zip(match_keys, pairs) if "match" in families and k not in found}.items())
        todo_bp = list({k: p for k, p in zip(bp_keys, pairs) if "boilerplate" in families and k not in found
                        and p.cohort_boilerplate.strip()}.items())
        total, done = len(todo_match) + len(todo_bp), 0
        if progress:
            progress(0, total, "loading the decision model" if total and self.loaded_at is None else "starting")
        with self.lock:
            if total:
                self._prepare()
            for start in range(0, len(todo_match), chunk):
                part = todo_match[start:start + chunk]
                frame = pd.DataFrame([{"patient_id": p.patient_id, "cohort_trial_id": p.cohort_id,
                                       "cancer_history_summary": p.patient_key, "cohort_key_criteria": p.cohort_key}
                                      for _, p in part])
                out = score_match_quality(frame, config=self.config, filter_low_quality=False)
                values = {}
                for (key, _), row in zip(part, out.to_dict("records")):
                    values[key] = {"score": float(row["match_quality_score"]), "pass": bool(row["match_quality_pass"]),
                                   **{label: float(row[f"match_quality_{label}"]) for label in MATCH_LABELS}}
                self.cache.put_many(values)
                found.update(values)
                done += len(part)
                if progress:
                    progress(done, total, "match questions")
            for start in range(0, len(todo_bp), chunk):
                part = todo_bp[start:start + chunk]
                frame = pd.DataFrame([{"patient_id": p.patient_id, "cohort_trial_id": p.cohort_id,
                                       "general_exclusion_criteria_evidence": p.patient_boilerplate,
                                       "cohort_boilerplate_criteria": p.cohort_boilerplate} for _, p in part])
                out = exclusion_criteria_check(frame, config=self.config)
                values = {key: {"p": float(row["exclusion_criteria_pass_probability"]),
                                "pass": bool(row["exclusion_criteria_pass"])}
                          for (key, _), row in zip(part, out.to_dict("records"))}
                self.cache.put_many(values)
                found.update(values)
                done += len(part)
                if progress:
                    progress(done, total, "boilerplate question")
        results = []
        for pair, mk, bk in zip(pairs, match_keys, bp_keys):
            boilerplate = (found.get(bk) if pair.cohort_boilerplate.strip()
                           else {"skipped": "the cohort has no boilerplate criteria"})
            results.append({"patient_id": pair.patient_id, "cohort_id": pair.cohort_id, "match": found.get(mk),
                            "boilerplate": boilerplate})
        return results

    def criteria(self, pair: PairText, progress: Progress | None = None) -> dict:
        """Every tagged line of the cohort, asked about the patient's complete summary."""
        import pandas as pd
        from matchminer_ai.matching import check_eligibility_criteria

        if not pair.items:
            raise ValueError("this cohort has no tagged criteria lines (per-criterion checks need them)")
        key = self.criteria_key(pair)
        cached = self.cache.get_many([key]).get(key)
        if cached:
            return {**cached, "cached": True}
        if progress:
            progress(0, 1, "loading the decision model" if self.loaded_at is None else f"{len(pair.items)} lines")
        with self.lock:
            self._prepare()
            frame = pd.DataFrame([{"patient_id": pair.patient_id, "cohort_trial_id": pair.cohort_id,
                                   "cancer_history_summary": pair.patient_key,
                                   "general_exclusion_criteria_evidence": pair.patient_boilerplate,
                                   "cohort_criteria_items": json.dumps(pair.items, ensure_ascii=False)}])
            out, metadata = check_eligibility_criteria(frame, config=self.config, return_metadata=True)
        model = metadata["model_metadata"]["check_eligibility_criteria"]
        value = {"lines": [{"number": int(r["criterion_number"]), "type": r["criterion_type"],
                            "boilerplate": bool(r["boilerplate"]), "text": r["criterion_text"],
                            "p": float(r["criterion_pass_probability"]), "fails": bool(r["criterion_clearly_fails"])}
                           for r in out.to_dict("records")],
                 "unchecked_lines": model.get("unchecked_lines", 0), "input_tokens": model.get("input_tokens"),
                 "over_max_length": model.get("over_max_length")}
        self.cache.put_many({key: value})
        if progress:
            progress(1, 1, "done")
        return value

    def attribution(self, pair: PairText, target: str, progress: Progress | None = None) -> dict:
        """Token attribution of one answer: a match label, or ``boilerplate``. Not cached (on demand)."""
        import pandas as pd
        from matchminer_ai.matching import interpret_exclusion_criteria, interpret_match_quality

        if progress:
            progress(0, 1, "loading the decision model" if self.loaded_at is None else "gradient pass")
        with self.lock:
            self._prepare()
            if target == "boilerplate":
                if not pair.cohort_boilerplate.strip():
                    raise ValueError("the cohort has no boilerplate criteria")
                frame = pd.DataFrame([{"patient_id": pair.patient_id, "cohort_trial_id": pair.cohort_id,
                                       "general_exclusion_criteria_evidence": pair.patient_boilerplate,
                                       "cohort_boilerplate_criteria": pair.cohort_boilerplate}])
                row = interpret_exclusion_criteria(frame, config=self.config, top_k=None).to_dict("records")[0]
                sources = {"patient_boilerplate_summary": pair.patient_boilerplate.strip(),
                           "cohort_boilerplate_criteria": pair.cohort_boilerplate.strip()}
                decision = {"p_yes": float(row["exclusion_criteria_pass_probability"])}
            elif target in MATCH_LABELS:
                frame = pd.DataFrame([{"patient_id": pair.patient_id, "cohort_trial_id": pair.cohort_id,
                                       "cancer_history_summary": pair.patient_key, "cohort_key_criteria": pair.cohort_key}])
                row = interpret_match_quality(frame, config=self.config, top_k=None, head=target).to_dict("records")[0]
                sources = {"patient_key_summary": pair.patient_key.strip(), "cohort_key_criteria": pair.cohort_key.strip()}
                decision = {"p_yes": float(row[f"match_quality_{target}"])}
            else:
                raise ValueError(f"unknown attribution target {target!r}")
        if progress:
            progress(1, 1, "done")
        return {"target": target, "attribution_target": row["attribution_target"], **decision, "sources": sources,
                "tokens": row["token_attributions"], "coverage": row["attribution_coverage"],
                "caveat": row["attribution_caveat"]}
