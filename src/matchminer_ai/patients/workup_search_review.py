"""Coverage-driven follow-up of provisional guideline workup assessments.

Adapted from OCI's per-variable note-search review. The shared vocabulary cache
contains only guideline questions, endpoint identities and literal search terms.
Patient passages, answers and coverage remain local to one invocation.
"""

from __future__ import annotations

import copy
import hashlib
import heapq
import json
import re
import threading
import time
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from importlib import resources

from matchminer_ai.llm.structured import EndpointError

from .note_search_qa import _MeasuredClient, _object, _remember_excerpts

VERSION = "workup-targeted-review-v1"
_VOCABULARY = OrderedDict()
_CACHE_LOCK = threading.Lock()
_PLAN_LOCKS = [threading.Lock() for _ in range(32)]


@dataclass(frozen=True)
class WorkupSearchReviewConfig:
    """Additional budgets after the initial note-search assessment.

    Set max_review_passes=0 to disable all follow-up. Request attempts, including
    vocabulary generation and retries, share max_calls_per_item. Full-record
    fallback visits each source note in order, splitting to fit the model context.
    It stops explicitly at the configured chunk/request budgets.
    """

    max_review_passes: int = 2
    retry_zero_match_missing: bool = True
    review_hits_per_item: int = 4
    review_context_chars: int = 360
    max_evidence_chars: int = 16000
    max_calls_per_item: int = 12
    max_full_record_fallback_items: int = 5
    max_full_record_chunks: int = 8

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name == "retry_zero_match_missing":
                if type(value) is not bool:
                    raise ValueError("retry_zero_match_missing must be boolean.")
            else:
                minimum = (
                    0
                    if name in {"max_review_passes", "max_full_record_fallback_items"}
                    else 1
                )
                if type(value) is not int or value < minimum:
                    raise ValueError(f"{name} must be an integer >= {minimum}.")


def _prompt(name):
    return (
        resources.files("matchminer_ai.prompts")
        .joinpath(name)
        .read_text(encoding="utf-8")
    )


def _term_key(term):
    return re.sub(r"[\s_\-‐‑–—]+", " ", term.strip()).casefold()


def literal_matches(history, terms):
    """Exact deduplicated counts, with bounded first/last match positions."""
    iterators = []
    for term in terms:
        parts = re.split(r"[\s_\-‐‑–—]+", term.strip())
        pattern = r"[\s_\-‐‑–—]+".join(re.escape(p) for p in parts if p)
        if not pattern:
            continue
        if term[0].isalnum():
            pattern = r"(?<!\w)" + pattern
        if term[-1].isalnum():
            pattern += r"(?!\w)"
        iterators.append(re.finditer(pattern, history, re.IGNORECASE))
    count, previous, first, last = 0, None, [], deque(maxlen=256)
    for match in heapq.merge(*iterators, key=lambda m: (m.start(), m.end())):
        position = match.span()
        if position == previous:
            continue
        previous = position
        count += 1
        if len(first) < 256:
            first.append(position)
        else:
            last.append(position)
    positions = sorted(set(first).union(last))
    return {
        "positions": positions,
        "match_count": count,
        "positions_omitted": count > len(positions),
    }


def _covered(position, excerpts):
    return any(e["start"] <= position[0] and position[1] <= e["end"] for e in excerpts)


def _spread(positions, count):
    if len(positions) <= count:
        return positions
    if count == 1:
        return positions[-1:]
    return [positions[i * (len(positions) - 1) // (count - 1)] for i in range(count)]


def _excerpt(history, note, start, end):
    return {
        "start": start,
        "end": end,
        "quote": history[start:end],
        "note_number": note["note_number"],
        "note_date": note["note_date"],
    }


class _ReviewLimit(RuntimeError):
    pass


class _ReviewValidation(RuntimeError):
    pass


class _ReviewClient(_MeasuredClient):
    """Distinguish exhausted invalid replies from a transport failure."""

    def __init__(self, config):
        super().__init__(config)
        self.transport_failures = 0

    def _http(self, *args, **kwargs):
        try:
            return super()._http(*args, **kwargs)
        except EndpointError:
            self.transport_failures += 1
            raise


class _ItemReview:
    def __init__(self, answer, *, history, spans, llm, config, contract):
        self.started = time.monotonic()
        self.answer = copy.deepcopy(answer)
        self.history, self.spans = history, spans
        self.llm, self.config, self.contract = llm, config, contract
        self.client = _ReviewClient(llm)
        self.client.max_calls = config.max_calls_per_item
        self.seen = list(self.answer["evidence"])
        self.unresolved = (
            bool(answer.get("needs_review")) or answer["answer"]["status"] == "unclear"
        )
        self.remaining = False
        self.stopped = False
        self.matches = {"positions": [], "match_count": 0, "positions_omitted": False}
        self.audit = {
            "version": VERSION,
            "status": "complete",
            "rounds": [],
            "zero_match_research": None,
            "full_record": None,
            "failures": [],
        }

    def request(self, phase, prompt, payload, schema, validator):
        remaining = self.config.max_calls_per_item - self.client.requests
        if remaining <= 0:
            raise _ReviewLimit("request_budget")
        self.client.config = replace(
            self.llm, attempts=min(self.llm.attempts, remaining)
        )
        messages = [
            {
                "role": "system",
                "content": prompt + "\n\nResponse JSON schema:\n" + json.dumps(schema),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        if not self.client.fits(messages):
            raise _ReviewLimit("context_budget")
        before_errors = len(self.client.validation_errors)
        before_transport = self.client.transport_failures
        try:
            return self.client.complete(
                "workup-note-search-" + phase, messages, schema, validator
            )
        except EndpointError:
            if (
                self.client.transport_failures == before_transport
                and len(self.client.validation_errors) > before_errors
            ):
                raise _ReviewValidation from None
            raise

    def vocabulary(self, unsuccessful=None):
        prompt = _prompt("patient.workup_search_terms.system.txt")
        # This question contains only the catalog item and population, constructed
        # by the workup adapter. Never include a provisional patient assessment.
        payload = {
            "workup_item": json.loads(self.answer["question"]),
            "unsuccessful_terms": unsuccessful,
        }
        key = hashlib.sha256(
            json.dumps(
                {
                    "version": VERSION,
                    "prompt": prompt,
                    "payload": payload,
                    "llm": self.llm.public_dict(),
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        schema = _object(
            {
                "terms": {
                    "type": "array",
                    "minItems": 0 if unsuccessful is not None else 1,
                    "maxItems": 8,
                    "items": {"type": "string", "minLength": 1, "maxLength": 100},
                }
            }
        )

        def validate(value):
            if not isinstance(value, dict) or set(value) != {"terms"}:
                raise ValueError("Return a terms list for this workup item.")
            terms = value["terms"]
            if (
                not isinstance(terms, list)
                or not (0 if unsuccessful is not None else 1) <= len(terms) <= 8
                or any(
                    not isinstance(t, str) or not t.strip() or len(t) > 100
                    for t in terms
                )
            ):
                raise ValueError(
                    "Supply up to eight nonempty literal terms of at most 100 characters."
                )

        with _PLAN_LOCKS[int(key[:8], 16) % len(_PLAN_LOCKS)]:
            with _CACHE_LOCK:
                saved = _VOCABULARY.get(key)
                if saved is not None:
                    _VOCABULARY.move_to_end(key)
                    return list(saved)
            result = self.request(
                "alternative-terms" if unsuccessful is not None else "terms",
                prompt,
                payload,
                schema,
                validate,
            )
            seen = {_term_key(t) for t in (unsuccessful or [])}
            terms = []
            for term in result["terms"]:
                if _term_key(term) not in seen:
                    terms.append(term.strip())
                    seen.add(_term_key(term))
            with _CACHE_LOCK:
                _VOCABULARY[key] = tuple(terms)
                while len(_VOCABULARY) > 256:
                    _VOCABULARY.popitem(last=False)
            return terms

    def passages(self, positions, pass_index):
        context = self.config.review_context_chars * (pass_index + 1)
        budget = self.config.max_evidence_chars // max(1, len(positions))
        result = []
        for a, b in positions:
            note = next(
                (n for n in self.spans if n["start"] <= a < b <= n["end"]), None
            )
            if note is None or b - a > budget:
                continue
            left = max(note["start"], a - min(context, (budget - (b - a)) // 2))
            right = min(note["end"], b + context, left + budget)
            _remember_excerpts(result, [_excerpt(self.history, note, left, right)])
        return result

    def review_payload(self, excerpts, reasons):
        return {
            "workup_item": json.loads(self.answer["question"]),
            "provisional_assessment": {
                k: self.answer[k] for k in ("status", "answer", "limitations")
            },
            "review_reasons": reasons,
            "source_excerpts": excerpts,
            "record_characters": len(self.history),
        }

    def assess(self, excerpts, reasons, phase):
        schema = _object(
            {
                "status": {"type": "string", "enum": ["answered", "unknown"]},
                "answer": self.contract.schema,
                "limitations": {
                    "type": "array",
                    "maxItems": 12,
                    "items": {"type": "string", "maxLength": 1000},
                },
                "needs_more_evidence": {"type": "boolean"},
            }
        )
        evidence = list(self.answer["evidence"])
        _remember_excerpts(evidence, excerpts)

        def validate(value):
            if not isinstance(value, dict) or set(value) != set(schema["properties"]):
                raise ValueError(
                    "Return the assessment, limitations and needs_more_evidence."
                )
            if (
                value["status"] not in {"answered", "unknown"}
                or type(value["needs_more_evidence"]) is not bool
            ):
                raise ValueError(
                    "Supply a valid answer status and boolean evidence flag."
                )
            limits = value["limitations"]
            if (
                not isinstance(limits, list)
                or len(limits) > 12
                or any(not isinstance(s, str) or len(s) > 1000 for s in limits)
                or (value["status"] == "unknown" and not limits)
            ):
                raise ValueError(
                    "Supply bounded limitations, including an explanation for unknown answers."
                )
            self.contract.validate({**value, "evidence": evidence})

        result = self.request(
            phase,
            self.contract.instructions
            + "\n\n"
            + _prompt("patient.workup_search_review.system.txt"),
            self.review_payload(excerpts, reasons),
            schema,
            validate,
        )
        changed = result["answer"] != self.answer["answer"]
        self.answer.update({k: result[k] for k in ("status", "answer", "limitations")})
        self.answer["evidence"] = evidence
        _remember_excerpts(self.seen, excerpts)
        self.unresolved = (
            result["needs_more_evidence"] or result["answer"]["status"] == "unclear"
        )
        return changed

    def failure(self, phase, exc):
        # Never expose endpoint output or patient content through error strings.
        kind = (
            "invalid_response"
            if isinstance(exc, _ReviewValidation)
            else (str(exc) if isinstance(exc, _ReviewLimit) else "endpoint_failure")
        )
        self.audit["failures"].append({"phase": phase, "kind": kind})
        self.unresolved = True
        if not isinstance(exc, _ReviewValidation):
            self.stopped = True

    def targeted(self):
        try:
            terms = self.vocabulary()
            # Search source text only: generated note headers cannot create hits.
            self.matches = self.find_matches(terms)
            self.audit["terms"] = terms
            if (
                self.config.retry_zero_match_missing
                and self.answer["status"] == "unknown"
                and self.matches["match_count"] == 0
            ):
                alternatives = self.vocabulary(terms)
                self.audit["zero_match_research"] = {
                    "initial_terms": list(terms),
                    "alternative_terms": alternatives,
                }
                terms = terms + alternatives
                self.matches = self.find_matches(terms)
                self.audit["zero_match_research"]["match_count_after_research"] = (
                    self.matches["match_count"]
                )
            for index in range(self.config.max_review_passes):
                unseen = [
                    p for p in self.matches["positions"] if not _covered(p, self.seen)
                ]
                reasons = []
                if self.unresolved:
                    reasons.append("Unresolved evidence or conflicting mentions.")
                if (
                    index == 0
                    and self.answer["status"] == "unknown"
                    and self.matches["positions"]
                ):
                    reasons.append(
                        "Uncertain provisional assessment with matching passages."
                    )
                if unseen:
                    reasons.append(
                        "Additional matching passages have not been supplied."
                    )
                if not reasons or not self.matches["positions"]:
                    break
                prior = [p for p in self.matches["positions"] if _covered(p, self.seen)]
                chosen = (
                    _spread(prior, 1)
                    if unseen and prior and self.config.review_hits_per_item > 1
                    else []
                )
                chosen += _spread(
                    unseen or self.matches["positions"],
                    self.config.review_hits_per_item - len(chosen),
                )
                excerpts = self.passages(chosen, index)
                if not excerpts:
                    self.unresolved = True
                    break
                changed = self.assess(excerpts, reasons, "targeted-review")
                self.audit["rounds"].append(
                    {
                        "pass": index + 1,
                        "changed": changed,
                        "needs_more_evidence": self.unresolved,
                        "supplied_ranges": [[e["start"], e["end"]] for e in excerpts],
                    }
                )
        except (_ReviewValidation, _ReviewLimit, EndpointError) as exc:
            self.failure("targeted_review", exc)
        self.remaining = self.matches["positions_omitted"] or any(
            not _covered(p, self.seen) for p in self.matches["positions"]
        )

    def find_matches(self, terms):
        positions, count = [], 0
        for note in self.spans:
            found = literal_matches(note["text"], terms)
            count += found["match_count"]
            positions.extend(
                (a + note["start"], b + note["start"]) for a, b in found["positions"]
            )
            # Keep a bounded first/last reservoir even for thousands of notes.
            if len(positions) > 512:
                positions = positions[:256] + positions[-256:]
        return {
            "positions": positions,
            "match_count": count,
            "positions_omitted": count > len(positions),
        }

    def full_record(self):
        audit = {"chunks": 0, "complete": False}
        self.audit["full_record"] = audit
        try:
            for note in self.spans:
                cursor = note["start"]
                while cursor < note["end"]:
                    if audit["chunks"] >= self.config.max_full_record_chunks:
                        raise _ReviewLimit("full_record_chunk_budget")
                    end = min(note["end"], cursor + self.config.max_evidence_chars)
                    while True:
                        excerpt = _excerpt(self.history, note, cursor, end)
                        try:
                            self.assess(
                                [excerpt],
                                [
                                    (
                                        "Serial full-record fallback. Reassess with this next source passage. "
                                        "Earlier passages are represented by the provisional assessment; more may follow."
                                    )
                                ],
                                "full-record",
                            )
                            break
                        except _ReviewLimit as exc:
                            if str(exc) != "context_budget" or end - cursor <= 1:
                                raise
                            end = cursor + (end - cursor) // 2
                    audit["chunks"] += 1
                    # Overlap adjacent chunks so a boundary doesn't discard context.
                    cursor = (
                        end
                        if end == note["end"]
                        else max(cursor + 1, end - min(360, (end - cursor) // 4))
                    )
            audit["complete"] = True
            self.remaining = False
        except (_ReviewValidation, _ReviewLimit, EndpointError) as exc:
            self.failure("full_record", exc)

    def finish(self):
        self.audit["coverage"] = {
            "match_count": self.matches["match_count"],
            "positions_omitted": self.matches["positions_omitted"],
            "unreviewed_stored_positions": sum(
                not _covered(p, self.seen) for p in self.matches["positions"]
            ),
        }
        incomplete = self.unresolved or self.remaining
        self.audit["unresolved_or_coverage_limited"] = incomplete
        if self.audit["failures"] and incomplete:
            self.audit["status"] = "incomplete"
            self.answer["limitations"].append(
                "Follow-up review could not finish; the last validated assessment is retained."
            )
        elif incomplete:
            self.audit["status"] = "incomplete"
            self.answer["limitations"].append(
                "Follow-up review left unresolved evidence or unreviewed matches within its budgets."
            )
        self.audit["elapsed_seconds"] = round(time.monotonic() - self.started, 4)
        self.audit["requests"] = self.client.requests
        self.audit["max_calls"] = self.config.max_calls_per_item
        metadata = self.answer["metadata"]
        metadata["elapsed_seconds"] += self.audit["elapsed_seconds"]
        metadata["initial_search_requests"] = metadata["requests"]
        metadata["requests"] += self.client.requests
        metadata["followup_review"] = self.audit
        for key in ("request_metrics", "finish_reasons", "validation_errors"):
            metadata.setdefault(key, []).extend(getattr(self.client, key))
        complete_usage = (
            metadata.get("usage_complete", False)
            and self.client.requests == self.client.usage_reports
        )
        metadata["usage_complete"] = complete_usage
        for key in ("prompt_tokens", "completion_tokens"):
            metadata[key] = (
                (metadata[key] + getattr(self.client, key)) if complete_usage else None
            )
        self.answer.pop("needs_review", None)
        return self.answer


def review_answers(
    answers, *, history, spans, llm, config, contract, concurrency, progress
):
    """Parallel focused review, then deterministic bounded fallback by input order."""
    items = [
        _ItemReview(
            a, history=history, spans=spans, llm=llm, config=config, contract=contract
        )
        if a["status"] != "error" and isinstance(a["answer"], dict)
        else None
        for a in answers
    ]

    def run(indices, method, label):
        if not indices:
            return
        with ThreadPoolExecutor(max_workers=min(concurrency, len(indices))) as pool:
            futures = {pool.submit(getattr(items[i], method)): i for i in indices}
            for count, future in enumerate(as_completed(futures), 1):
                future.result()
                progress(f"Agentic {label}: {count}/{len(indices)} items finished")

    run([i for i, item in enumerate(items) if item], "targeted", "coverage review")
    eligible = [
        i
        for i, item in enumerate(items)
        if item and not item.stopped and (item.unresolved or item.remaining)
    ]
    # Prioritize unresolved findings, preserving input order within each tier.
    eligible.sort(key=lambda i: not items[i].unresolved)
    run(
        eligible[: config.max_full_record_fallback_items],
        "full_record",
        "full-record fallback",
    )
    return [
        item.finish() if item else answer
        for item, answer in zip(items, answers, strict=True)
    ]
