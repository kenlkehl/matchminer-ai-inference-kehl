"""Workup question answering over locally retrieved ColBERT note chunks."""

from __future__ import annotations

import copy
import hashlib
import json
from importlib import resources

from matchminer_ai.cancellation import check_cancelled
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


def review_patient_workup_with_colbert(
    index: ColBERTPatientIndex,
    recommendations: list[dict],
    *,
    config: MMAIConfig | None = None,
    backup_config: MMAIConfig | None = None,
    max_consecutive_failures: int = 3,
    top_n: int = 20,
    retrieval_config=None,
    population_context: str | None = None,
    progress_callback=None,
) -> dict:
    """Retrieve top_n chunks for each workup question and request one assessment.

    Encode notes separately with encode_patient_notes_colbert (possibly batched
    across patients) or explicitly load an index from disk. No full-record pass,
    agent tools, patient summary, or public search is used. Every retrieved
    fragment is retained verbatim by code as review context, not as a selected
    supporting citation. Missing evidence describes only the retrieved subset.

    Configured patient LLM settings and an optional explicit backup are honored.
    After repeated request/validation failures the failed and remaining items
    use the backup, with its own settings. No clinical data is checkpointed.
    Per-item failures are errors, never missing-documentation findings.
    """
    if not isinstance(index, ColBERTPatientIndex):
        raise TypeError("Encode or load a ColBERTPatientIndex before reviewing workup.")
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
    backup = WorkupBackup(backup_config, max_consecutive_failures)
    progress = progress_callback or (lambda _: None)
    questions = [
        f"{r['name']}\n{r.get('conditions', '')}".strip() for r in recommendations
    ]
    retrieved = retrieve_patient_chunks_colbert(
        index,
        questions,
        top_n=top_n,
        config=retrieval_config,
        progress_callback=progress,
    )
    system = (
        resources.files("matchminer_ai.prompts")
        .joinpath("patient.workup_colbert.system.txt")
        .read_text()
    )

    def resolve(selected, attempts):
        runtime = build_llm_runtime_config("patient", selected.patient, config=selected)
        runtime["max_retries"] = attempts
        resolved, _ = resolve_structured_config(runtime, cache_dir=None)
        return resolved

    class ReviewClient(StructuredClient):
        def __init__(self, selected):
            super().__init__(selected, None)
            self.requests = 0

        def _http(self, endpoint, body=None, **kwargs):
            if endpoint == "/chat/completions":
                self.requests += 1
            return super()._http(endpoint, body, **kwargs)

    primary = resolve(config, backup.threshold if backup.enabled else 3)
    client = ReviewClient(primary)
    clients = [client]
    events, assessments = [], []
    backup_failed = False
    for number, (recommendation, hits) in enumerate(
        zip(recommendations, retrieved, strict=True)
    ):
        check_cancelled()
        progress(
            f"Answering ColBERT workup question {number + 1}/{len(recommendations)}"
        )
        # Retrieval ranks remain in metadata; chronological presentation helps
        # distinguish a prior plan from later completion/conflicting updates.
        ordered = sorted(hits, key=lambda h: (h["note_number"], h["start"]))
        payload = dict(
            guideline_population=population_context,
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
            dict(role="system", content=system),
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
                if not backup.enabled or events or backup_failed:
                    error = (
                        "The configured model could not return a validated assessment."
                    )
                    break
                try:
                    selected = backup.get(lambda c: resolve(c, 3))
                except EndpointError:
                    backup_failed = True
                    error = "The configured backup endpoint could not be prepared."
                    break
                events.append(
                    backup_event(
                        primary,
                        selected,
                        "request_or_validation_failures",
                        backup.threshold,
                        recommendation_start=number,
                    )
                )
                client = ReviewClient(selected)
                clients.append(client)
                progress(
                    "Primary ColBERT answer model failed repeatedly; using the configured backup"
                )
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
                backup_used=bool(events),
                backup_events=copy.deepcopy(events),
                backup_resolution_failed=backup_failed,
            ),
        )
        assessments.append(row)
    progress("ColBERT workup documentation review complete")
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
            model=client.config.model,
            primary_model=primary.model,
            model_signature=copy.deepcopy(index.model_signature),
            requests=sum(c.requests for c in clients),
            backup_used=bool(events),
            backup_events=events,
            backup_resolution_failed=backup_failed,
            patient_summary_used=False,
            evidence_selection="automatic_reviewed_excerpts",
            prompt_sha256=hashlib.sha256(system.encode()).hexdigest(),
        ),
    )
