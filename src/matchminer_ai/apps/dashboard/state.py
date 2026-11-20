"""What the dashboard server holds: the trial store (rebuilt as extractions land), the synthetic patients and the
round's vectors, the real cohorts' vectors, the encoder, the decision model and background jobs."""
from __future__ import annotations

import threading
import time

import numpy as np

from .core import cancer_types
from .core.cache import JSONCache, ScoreCache, file_stamp, text_key
from .core.corpus import Corpus
from .core.decisions import DecisionService, PairText
from .core.encoder import EncoderClient
from .core.jobs import JobManager
from .core.notes import NotesIndex
from .core.vectors import VectorStore, select_round

from .evidence import CatalogEvidence
from .llm import Connection
from .llm_checks import LLMChecks
from .pipeline import Builder, Sources, read_store, write_store
from .vectors import CohortVectors


class NotFound(KeyError):
    pass


class DashState:
    def __init__(self, settings, *, decisions=None, encoder=None, verify_digests: bool = True, log=print):
        started = time.monotonic()
        self.settings, self.log = settings, log
        settings.store_dir.mkdir(parents=True, exist_ok=True)
        settings.store_dir.chmod(0o700)
        settings.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache = JSONCache(settings.cache_dir)
        self.jobs = JobManager()  # the decision model: one job at a time
        self.background = JobManager()  # store rebuilds and cohort embedding
        # ---- trials
        self.builder = Builder(Sources(settings.trial_data, settings.oa_db, settings.grouping,
                                       settings.good_options_catalog))
        self.store_lock = threading.RLock()
        self.store = read_store(settings.store_path)
        self.store_error: str | None = None
        self.last_check: float | None = None
        if self.store is None:
            log(f"Building the dashboard from {settings.trial_data} ...")
            self.store = self.builder.build()
            write_store(self.store, settings.store_path)
        log(f"  {self.store['counts']['trials']} trials, {self.store['counts']['dashboard_cohorts']} dashboard cohorts "
            f"in {len(self.store['centers'])} disease centers")
        # ---- synthetic patients and the round
        log(f"Loading the synthetic corpus from {settings.path('corpus')} ...")
        self.corpus = Corpus(settings.path("corpus"), settings.path("patient_metadata"), self.cache, verify=verify_digests)
        log(f"  {len(self.corpus.patients):,} patients ({self.corpus.load_seconds} s)")
        self.round_lock = threading.RLock()
        self.round, self.skipped_rounds = select_round(settings.path("trialspace"), self.corpus)
        self.round_switched_at: float | None = None
        self.vectors = VectorStore(self.round) if self.round else None
        log(f"  vectors: round {self.round.number if self.round else None}")
        self._make_encoder = ((lambda info: encoder) if encoder else
                              (lambda info: EncoderClient(info, settings, settings.cache_dir / "encoder_worker.log")))
        self.encoder = self._make_encoder(self.round)
        self.cohort_vectors = CohortVectors(settings.store_dir / "vectors", self.round.encoder if self.round else None)
        self.patient_cancer_types = self._patient_cancer_types()
        self.patient_classes = np.array([self.corpus.meta.get(r["id"], {}).get("cls") or "" for r in self.corpus.patients])
        self._masks: dict[str, np.ndarray] = {}
        self.decisions = decisions or DecisionService(settings, ScoreCache(settings.cache_dir / "scores.sqlite"),
                                                      settings.cache_dir)
        self.notes = NotesIndex(settings.path("notes"), self.cache)
        self.evidence = CatalogEvidence(settings.good_options_catalog)  # read on demand (evidence dialogs)
        self.llm = LLMChecks(settings, Connection.make(settings.llm_backend, settings.llm_endpoint, settings.llm_model,
                                                       settings.llm_tokenizer),
                             ScoreCache(settings.cache_dir / "llm_checks.sqlite"))
        self.custom: dict[str, dict] = {}
        self.startup_seconds = round(time.monotonic() - started, 1)

    # ------------------------------------------------------------------ the trial store

    def check_store(self) -> bool:
        """Rebuild the store when its sources changed (a trial finished extracting, a new run's files, the rules);
        True when it was replaced. Cohorts new to the store are then embedded in the background."""
        self.last_check = time.time()
        if self.builder.sources.signature() == self.store.get("signature"):
            return False
        store = self.builder.build()
        write_store(store, self.settings.store_path)
        with self.store_lock:
            old, self.store, self.store_error = self.store, store, None
        added = len(set(store["cohorts"]) - set(old["cohorts"]))
        self.log(f"Dashboard rebuilt: {store['counts']['trials']} trials, {store['counts']['dashboard_cohorts']} "
                 f"dashboard cohorts ({added} cohorts new) in {store['build_seconds']} s")
        self.embed_cohorts()
        return True

    def watch(self, interval: float, stop: threading.Event):
        while not stop.wait(interval):
            try:
                self.check_store()
            except Exception as exc:  # a half-written source; tried again next time
                self.store_error = f"{type(exc).__name__}: {exc}"
                self.log(f"Dashboard rebuild failed: {self.store_error}")
            try:
                self.refresh_round()
            except Exception as exc:
                self.log(f"TrialSpace round check failed: {type(exc).__name__}: {exc}")

    def embed_cohorts(self):
        """Embed the store's cohorts that have no vector for the round's encoder (a background job)."""
        if not self.round or not self.encoder.available:
            return None
        if not self.cohort_vectors.missing(self.store):
            self.cohort_vectors.state = "ready"
            return None
        return self.background.submit("embed", "trial cohorts (TrialSpace)",
                                      lambda progress: self.cohort_vectors.embed(self.store, self.encoder, progress),
                                      device=self.settings.embedding_device)

    def refresh_round(self) -> bool:
        """Switch to a higher TrialSpace round once it validates; the trial cohorts are then re-embedded."""
        current = self.round.number if self.round else 0
        info, skipped = select_round(self.settings.path("trialspace"), self.corpus, above=current)
        if info is None:
            self.skipped_rounds = skipped
            return False
        vectors, encoder = VectorStore(info), self._make_encoder(info)
        cohort_vectors = CohortVectors(self.settings.store_dir / "vectors", info.encoder)
        with self.round_lock:
            old = self.encoder
            self.round, self.vectors, self.encoder, self.skipped_rounds = info, vectors, encoder, skipped
            self.cohort_vectors, self.round_switched_at = cohort_vectors, time.time()
            self.custom.clear()
        if old is not encoder:
            old.close()
        self.log(f"Switched to TrialSpace round {info.number}; re-embedding the trial cohorts")
        self.embed_cohorts()
        return True

    # ------------------------------------------------------------------ store views

    def trial(self, trial_id: str) -> dict:
        trial = self.store["trials"].get(trial_id)
        if trial is None:
            raise NotFound(trial_id)
        return trial

    def cohort(self, cohort_id: str) -> dict:
        cohort = self.store["cohorts"].get(cohort_id)
        if cohort is None:
            raise NotFound(cohort_id)
        return cohort

    # ------------------------------------------------------------------ patients

    def _patient_cancer_types(self) -> list[list[str]]:
        key = {"patients": file_stamp(self.settings.path("corpus") / "patients.jsonl"), "version": cancer_types.VERSION}
        cached = self.cache.get("patient_cancer_types")
        if cached and cached.get("key") == key and len(cached["value"]) == len(self.corpus.patients):
            return cached["value"]
        value = [cancer_types.patient_types(header) for header in self.corpus.headers]
        self.cache.put("patient_cancer_types", {"key": key, "value": value})
        return value

    def patient_mask(self, cancer_type: str = "", sex: str = "", cls: str = "") -> np.ndarray | None:
        masks = []
        if cancer_type:
            if cancer_type not in cancer_types.TYPE_LABELS:
                raise ValueError(f"unknown cancer type {cancer_type!r}")
            if cancer_type not in self._masks:
                self._masks[cancer_type] = np.array([cancer_type in t for t in self.patient_cancer_types], dtype=bool)
            masks.append(self._masks[cancer_type])
        if sex:
            sexes = np.array([(h.get("sex") or "").lower() for h in self.corpus.headers])
            masks.append(sexes == sex.lower())
        if cls:
            masks.append(self.patient_classes == cls)
        if not masks:
            return None
        mask = masks[0].copy()
        for m in masks[1:]:
            mask &= m
        return mask

    def patient_entity(self, pid: str) -> dict:
        if pid in self.custom:
            return self.custom[pid]
        row = self.corpus.patient(pid)
        if row is None:
            raise NotFound(pid)
        return {"id": pid, "custom": False, "key": row["text"], "boilerplate": row.get("boilerplate") or "",
                "row": self.corpus.patient_index[pid]}

    def patient_summary(self, pid: str) -> dict:
        entity = self.patient_entity(pid)
        if entity["custom"]:
            from .core.corpus import parse_header

            return {"id": pid, "custom": True, "label": entity.get("label") or "Entered summary", **parse_header(entity["key"])}
        summary = self.corpus.patient_summary(pid)
        return {"id": pid, "custom": False, **{k: summary.get(k) for k in ("age", "sex", "cancer_type", "histology", "cls")},
                "cancer_types": self.patient_cancer_types[entity["row"]]}

    def add_custom_patient(self, key: str, boilerplate: str, label: str = "") -> dict:
        from matchminer_ai.embedding.trialspace_contract import PATIENT_PROMPT

        key, boilerplate = key.strip(), (boilerplate or "").strip()
        if not key:
            raise ValueError("the cancer summary is empty")
        pid = f"entered-{text_key('patient', key, boilerplate)[:12]}"
        if pid not in self.custom:
            vector = self.encoder.encode([key], PATIENT_PROMPT)[0]
            self.custom[pid] = {"id": pid, "custom": True, "key": key, "boilerplate": boilerplate,
                                "label": label or "Entered summary", "vector": np.asarray(vector, dtype=np.float32)}
        return self.custom[pid]

    def patient_vector(self, entity: dict) -> np.ndarray:
        return entity["vector"] if entity.get("custom") else self.vectors.patients[entity["row"]]

    def pair(self, patient_id: str, cohort_id: str) -> PairText:
        p, c = self.patient_entity(patient_id), self.cohort(cohort_id)
        return PairText(p["id"], c["id"], p["key"], p["boilerplate"], c["key"], c["boilerplate"], c["items"])

    # ------------------------------------------------------------------ status

    def describe(self) -> dict:
        s = self.store
        return {"store": {"built_at": s["built_at"], "build_seconds": s["build_seconds"], "counts": s["counts"],
                          "signature": s["signature"], "rules": s["rules"], "error": self.store_error,
                          "last_check": self.last_check, "path": str(self.settings.store_path),
                          "poll_seconds": self.settings.rebuild_poll_seconds},
                "sources": {"trial_data": str(self.settings.trial_data), "oa_db": s.get("oa_db") or str(self.settings.oa_db),
                            "grouping": str(self.settings.grouping),
                            "good_options": str(self.settings.good_options_catalog)},
                "good_options": s.get("good_options"),
                "patients": {"corpus": str(self.settings.path("corpus")), "count": len(self.corpus.patients)},
                "round": self.round.describe() if self.round else None, "skipped_rounds": self.skipped_rounds,
                "round_switched_at": self.round_switched_at, "encoder": self.encoder.describe(),
                "cohort_vectors": self.cohort_vectors.describe(s), "decision_model": self.decisions.describe(),
                "notes": self.notes.describe(),
                "jobs": self.jobs.recent(10) + self.background.recent(10) + self.llm.jobs.recent(10),
                "startup_seconds": self.startup_seconds, "score_cache": self.decisions.cache.count(),
                "llm": {"connection": self.llm.connection.describe(), "cached": self.llm.cache.count()}}
