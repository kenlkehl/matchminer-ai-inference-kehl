"""Workup question answering over locally retrieved ColBERT note chunks."""

from __future__ import annotations

import copy
import hashlib
import json
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from importlib import resources
from threading import Lock

from matchminer_ai.cancellation import check_cancelled, submit_cancellable
from matchminer_ai.config import MMAIConfig, load_default_preset
from matchminer_ai.llm.backends import build_llm_runtime_config, remote_enabled
from matchminer_ai.llm.structured import (
    EndpointError,
    StructuredClient,
    resolve_structured_config,
)

from ._workup_backup import WorkupBackup, backup_event
from .colbert import ColBERTPatientIndex, retrieve_patient_chunks_colbert
from .workup import NOTICE, _schema, _validate


class _Answerer:
    """One run's endpoints; each question owns its client and patient-only prompt."""

    def __init__(self, config, backup_config, threshold, concurrency, context):
        self.backup = WorkupBackup(backup_config, threshold)
        self.concurrency = concurrency
        self.population_context = context
        self.primary = self.resolve(
            config, self.backup.threshold if self.backup.enabled else 3
        )
        self.selected = self.primary
        self.events = []
        self.lock = Lock()
        self.system = (
            resources.files("matchminer_ai.prompts")
            .joinpath("patient.workup_colbert.system.txt")
            .read_text()
        )

    def resolve(self, selected, attempts):
        runtime = build_llm_runtime_config("patient", selected.patient, config=selected)
        runtime["max_retries"] = attempts
        # The same cap reaches the HTTP limiter; a UI config default of one must
        # not silently serialize a caller's explicitly requested batch.
        runtime["max_concurrent_requests"] = self.concurrency
        resolved, _ = resolve_structured_config(runtime, cache_dir=None)
        return resolved

    @staticmethod
    def client(selected):
        class ReviewClient(StructuredClient):
            def __init__(self):
                super().__init__(selected, None)
                self.requests = 0

            def _http(self, endpoint, body=None, **kwargs):
                if endpoint == "/chat/completions":
                    self.requests += 1
                return super()._http(endpoint, body, **kwargs)

        return ReviewClient()

    def switch(self, number):
        # Pending questions use the backup after one switch. Already dispatched
        # primary requests may finish; their own model provenance stays primary.
        with self.lock:
            selected = self.backup.get(lambda c: self.resolve(c, 3))
            if not self.events:
                self.events.append(
                    backup_event(
                        self.primary,
                        selected,
                        "request_or_validation_failures",
                        self.backup.threshold,
                        recommendation_start=number,
                    )
                )
                self.selected = selected
            return selected, copy.deepcopy(self.events)

    def answer(self, number, recommendation, hits):
        check_cancelled()
        with self.lock:
            selected = self.selected
            events = copy.deepcopy(self.events)
        using_backup = bool(events)
        client = self.client(selected)
        clients = [client]
        backup_failed = False
        # Retrieval ranks remain in metadata; chronological presentation helps
        # distinguish a prior plan from later completion/conflicting updates.
        ordered = sorted(hits, key=lambda h: (h["note_number"], h["start"]))
        payload = dict(
            guideline_population=self.population_context,
            recommendations=[
                dict(
                    name=recommendation["name"],
                    conditions=recommendation.get("conditions", ""),
                )
            ],
            retrieved_note_fragments=[
                {k: h[k] for k in ("note_number", "note_date", "note_type", "text")}
                for h in ordered
            ],
        )
        messages = [
            dict(role="system", content=self.system),
            dict(role="user", content=json.dumps(payload, ensure_ascii=False)),
        ]
        prior = [dict(status="not_documented")]
        reviewed = False
        error = None
        while True:
            check_cancelled()
            if not client.fits(messages):
                error = "Retrieved chunks exceed this model's input budget. Reduce top_n or increase the configured context budget."
                break
            reviewed = True
            try:
                value = client.complete(
                    "patient-workup-colbert",
                    messages,
                    _schema(1),
                    lambda v: _validate(v, [recommendation], prior),
                )
                row = value["assessments"][0]
                break
            except EndpointError:
                if not self.backup.enabled or using_backup:
                    error = (
                        "The configured model could not return a validated assessment."
                    )
                    break
                try:
                    selected, events = self.switch(number)
                except EndpointError:
                    backup_failed = True
                    error = "The configured backup endpoint could not be prepared."
                    break
                client = self.client(selected)
                clients.append(client)
                using_backup = True
        if error:
            row = dict(
                name=recommendation["name"],
                status="error",
                applicability="uncertain",
                bottom_line="No validated documentation assessment is available.",
                limitations=[error],
            )
        else:
            row["limitations"] = [
                "Only the retrieved chunks were reviewed; relevant or later documentation may be absent."
            ]
        row.update(
            recommendation_index=number,
            recommendation=copy.deepcopy(recommendation),
            evidence=[
                dict(
                    note_number=h["note_number"],
                    note_date=h["note_date"],
                    note_type=h["note_type"],
                    quote=h["text"],
                    start=h["start"],
                    end=h["end"],
                    retrieval_rank=h["rank"],
                    retrieval_score=h["score"],
                )
                for h in ordered
            ]
            if reviewed
            else [],
            metadata=dict(
                model=client.config.model,
                retrieved_chunk_count=len(hits),
                evidence_selection="automatic_reviewed_excerpts",
                backup_used=using_backup,
                backup_events=copy.deepcopy(events),
                backup_resolution_failed=backup_failed,
            ),
        )
        row["metadata"]["requests"] = sum(c.requests for c in clients)
        return row


def _failure(recommendation, number, message):
    return dict(
        recommendation_index=number,
        recommendation=copy.deepcopy(recommendation),
        name=recommendation["name"],
        status="error",
        applicability="uncertain",
        evidence=[],
        bottom_line="No validated documentation assessment is available.",
        limitations=[message],
        metadata={},
    )


def _finish(index, assessments, answerer, top_n, concurrency):
    events = []
    for row in assessments:
        for event in row["metadata"].get("backup_events", []):
            if event not in events:
                events.append(event)
    return dict(
        assessments=assessments,
        notice=NOTICE
        + " ColBERT retrieved only a subset of the record for each question. Relevant or later evidence may be missed. Research use only.",
        metadata=dict(
            method="colbert",
            scope="retrieved_excerpts",
            **{k: v for k, v in index.summary().items() if k != "model"},
            top_n=top_n,
            retrieval_model=index.config.model_name,
            model=assessments[-1]["metadata"].get("model", answerer.primary.model),
            models_used=list(
                dict.fromkeys(
                    r["metadata"]["model"]
                    for r in assessments
                    if "model" in r["metadata"]
                )
            ),
            primary_model=answerer.primary.model,
            model_signature=copy.deepcopy(index.model_signature),
            requests=sum(r["metadata"].get("requests", 0) for r in assessments),
            backup_used=any(r["metadata"].get("backup_used") for r in assessments),
            backup_events=events,
            backup_resolution_failed=any(
                r["metadata"].get("backup_resolution_failed") for r in assessments
            ),
            max_concurrent_requests=concurrency,
            patient_summary_used=False,
            evidence_selection="automatic_reviewed_excerpts",
            prompt_sha256=hashlib.sha256(answerer.system.encode()).hexdigest(),
        ),
    )


def review_patient_workup_with_colbert(
    index: ColBERTPatientIndex,
    recommendations: list[dict],
    *,
    config: MMAIConfig | None = None,
    backup_config: MMAIConfig | None = None,
    max_consecutive_failures: int = 3,
    max_concurrent_requests: int = 8,
    top_n: int = 20,
    retrieval_config=None,
    population_context: str | None = None,
    progress_callback=None,
) -> dict:
    """Review one patient with a bounded batch of independent workup questions.

    Source notes must already be encoded. Every request contains one question
    and only this patient's retrieved chunks, retained verbatim by code. No
    full-record pass, patient summary, agent tools or public search is used.
    max_concurrent_requests=1 restores sequential dispatch. The configured LLM,
    retry limits and optional backup are retained; no data is checkpointed.
    """
    if not isinstance(index, ColBERTPatientIndex):
        raise TypeError("Encode or load a ColBERTPatientIndex before reviewing workup.")
    return review_patients_workup_with_colbert(
        {index.patient_id: index},
        recommendations,
        config=config,
        backup_config=backup_config,
        max_consecutive_failures=max_consecutive_failures,
        max_concurrent_requests=max_concurrent_requests,
        top_n=top_n,
        retrieval_config=retrieval_config,
        population_context=population_context,
        progress_callback=progress_callback,
    )[index.patient_id]


def review_patients_workup_with_colbert(
    indexes: dict[str, ColBERTPatientIndex],
    recommendations: list[dict],
    *,
    config: MMAIConfig | None = None,
    backup_config: MMAIConfig | None = None,
    max_consecutive_failures: int = 3,
    max_concurrent_requests: int = 8,
    top_n: int = 20,
    retrieval_config=None,
    population_context: str | None = None,
    progress_callback=None,
) -> dict[str, dict]:
    """One bounded queue across all patient × workup pairs, in input result order.

    Local retrieval prepares one patient's questions at a time while LLM requests
    run concurrently. At most max_concurrent_requests pairs are submitted at once;
    no per-patient pools multiply the limit. Prompts and clients remain isolated.
    The explicit backup switch applies to this run's failed and pending requests;
    already in-flight primary responses retain their actual model provenance.
    Cancellation propagates to workers and prevents publishing partial results.
    """
    if (
        type(max_concurrent_requests) is not int
        or not 1 <= max_concurrent_requests <= 32
    ):
        raise ValueError("max_concurrent_requests must be an integer from 1 to 32.")
    if type(top_n) is not int or not 1 <= top_n <= 1000:
        raise ValueError("top_n must be an integer from 1 to 1000.")
    if (
        not isinstance(indexes, dict)
        or not indexes
        or any(
            not isinstance(index, ColBERTPatientIndex) or index.patient_id != pid
            for pid, index in indexes.items()
        )
    ):
        raise ValueError("Supply patient IDs mapped to their own ColBERT indexes.")
    config = config or load_default_preset()
    if not isinstance(config, MMAIConfig) or not remote_enabled(config):
        raise ValueError("ColBERT review requires a configured remote patient LLM.")
    if population_context is not None and not isinstance(population_context, str):
        raise TypeError("population_context must be guideline population text or None.")
    if (
        not isinstance(recommendations, list)
        or not recommendations
        or any(
            not isinstance(r, dict)
            or not isinstance(r.get("name"), str)
            or not r["name"].strip()
            or not isinstance(r.get("conditions", ""), str)
            for r in recommendations
        )
    ):
        raise ValueError(
            "Supply workup recommendations with names and text conditions."
        )
    check_cancelled()
    progress = progress_callback or (lambda _: None)
    progress(
        f"Preparing ColBERT review · up to {max_concurrent_requests} concurrent checks"
    )
    answerer = _Answerer(
        config,
        backup_config,
        max_consecutive_failures,
        max_concurrent_requests,
        population_context,
    )
    questions = [
        f"{r['name']}\n{r.get('conditions', '')}".strip() for r in recommendations
    ]
    rows = {pid: [None] * len(recommendations) for pid in indexes}
    remaining = dict.fromkeys(indexes, len(recommendations))
    completed = 0
    patients_done = 0
    total = len(indexes) * len(recommendations)
    pending = {}
    prepared = 0

    def status():
        progress(
            f"ColBERT review · {completed}/{total} patient–question checks complete · "
            f"{patients_done}/{len(indexes)} patients complete · "
            f"{len(pending)} in flight (limit {max_concurrent_requests}) · "
            f"{prepared}/{len(indexes)} patients retrieved"
        )

    def collect():
        nonlocal completed, patients_done
        check_cancelled()
        done, _ = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
        for future in done:
            pid, number = pending.pop(future)
            # InferenceCancelled is BaseException; it must escape safe errors.
            try:
                rows[pid][number] = future.result()
            except Exception:
                rows[pid][number] = _failure(
                    recommendations[number],
                    number,
                    "The workup question failed; check the endpoint and retry.",
                )
            completed += 1
            remaining[pid] -= 1
            if remaining[pid] == 0:
                patients_done += 1
        if done:
            status()

    pool = ThreadPoolExecutor(
        max_workers=max_concurrent_requests, thread_name_prefix="mmai-colbert-workup"
    )
    try:
        status()
        for pid, index in indexes.items():
            check_cancelled()
            progress(
                f"Retrieving ColBERT evidence for patient {prepared + 1}/{len(indexes)} · "
                f"{completed}/{total} checks complete · {len(pending)} in flight"
            )
            try:
                retrieved = retrieve_patient_chunks_colbert(
                    index,
                    questions,
                    top_n=top_n,
                    config=retrieval_config,
                )
                if len(retrieved) != len(recommendations):
                    raise ValueError("Incomplete retrieval")
            except Exception:
                rows[pid] = [
                    _failure(
                        r, n, "Patient retrieval failed; check the index and retry."
                    )
                    for n, r in enumerate(recommendations)
                ]
                completed += len(recommendations)
                patients_done += 1
                remaining[pid] = 0
                prepared += 1
                status()
                continue
            prepared += 1
            for number, (recommendation, hits) in enumerate(
                zip(recommendations, retrieved, strict=True)
            ):
                while len(pending) >= max_concurrent_requests:
                    collect()
                check_cancelled()
                future = submit_cancellable(
                    pool, answerer.answer, number, recommendation, hits
                )
                pending[future] = (pid, number)
                status()
        while pending:
            collect()
        check_cancelled()
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    return {
        pid: _finish(index, rows[pid], answerer, top_n, max_concurrent_requests)
        for pid, index in indexes.items()
    }
