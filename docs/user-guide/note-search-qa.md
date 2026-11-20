# Note-search question answering

`answer_patient_questions` answers questions about one patient's original notes by letting an LLM search them in an
isolated Python REPL, instead of reading every note chunk for every question. The notes can be a DataFrame,
concatenated text, or both, with an optional existing summary as navigation context.
`answer_patient_question_batch` runs the same loop across patients and questions concurrently, and
`review_patient_workup_with_note_search` applies it to guideline workup items; the guideline fetcher offers it as its
agentic workup method, beside the default full-note review.

The feature is experimental. Answers are research documentation-review results for human review, not guideline
concordance or eligibility determinations. All three functions are in `matchminer_ai.patients`; their docstrings are
on the [Patients API](../api/patients.md) page.

## One patient

```python
import pandas as pd

from matchminer_ai.patients import NoteSearchLLMConfig, NoteSearchLimits, answer_patient_questions

llm = NoteSearchLLMConfig(
    base_url="http://localhost:8000/v1",
    model="served-model-id",   # the ID your server reports
    context_window=65536,      # within the model's actual capacity
)

notes = pd.DataFrame([
    {"note_date": "2026-01-01", "note_type": "Oncology", "note_text": "Fabricated note. CT chest ordered."},
    {"note_date": "2026-02-01", "note_type": "Radiology", "note_text": "Fabricated note. CT chest completed."},
])
result = answer_patient_questions(
    notes=notes,
    questions=["Was the CT completed after January?", "Is the CT result documented?"],
    patient_summary=None,      # optional navigation context, never evidence
    llm=llm,
    limits=NoteSearchLimits(max_cells=12, max_calls=16),
    max_parallel_questions=4,
)
for answer in result["answers"]:
    print(answer["status"], answer["answer"], answer["evidence"])
```

The first positional argument, `history`, takes concatenated text (or a notes DataFrame); `notes=` takes a
single-patient DataFrame. The result is a dict with `answers` (one per question, in order), `notice`, and
`metadata`.

- **DataFrame input:** `note_text` is required; `note_date` and `note_type` may be absent or missing. Dates are
  validated (an unparseable date is an error; leave unavailable dates empty), converted to UTC, and sorted stably,
  with undated rows last. Note numbers are assigned in code after sorting. The DataFrame is copied, and unrelated
  columns never reach the worker.
- **Text input:** becomes one note with unavailable date and type. Offsets refer to the text exactly as supplied.
- **Both:** when text and a DataFrame are given, only the DataFrame supplies evidence; this avoids duplicate text and
  stale dates. Pass text alone when it is the intended authoritative record.
- **Summary:** an existing `patient_summary` may orient searches but is never appended to the notes or accepted in
  place of note excerpts.

## Endpoint configuration

`NoteSearchLLMConfig` names the endpoint explicitly; this workflow never discovers or falls back to another server.

| Field | Default | Meaning |
| --- | --- | --- |
| `base_url`, `model` | required | OpenAI-compatible `/v1` URL and the model ID it serves. |
| `api_key_env` | `OPENAI_API_KEY` | Environment variable holding the bearer token. |
| `provider`, `google_project_id` | `openai`, `""` | Structured-client provider settings. |
| `context_window` | 65,536 | Prompt plus output budget in tokens; set within the model's real capacity. |
| `safety_tokens` | 2,048 | Margin kept beyond the output reserve. |
| `max_tokens` | `None` | Output reserve; `None` uses the model profile's floor (32,768 for Qwen 3.8 Flash Next) or 8,192. |
| `max_concurrent_requests` | 8 | Requests in flight to this endpoint, shared by every caller in the process. |
| `timeout`, `attempts` | 120 s, 2 | Per-request timeout and attempts, retries included. |
| `tokenizer_mode` | `bytes` | `bytes` budgets UTF-8 bytes conservatively; `endpoint` counts exactly with vLLM's `/tokenize`. |
| `response_format` | `json_schema` | Structured-output mode for each turn. |
| `sampling_profile` | `auto` | Model-aware sampling from the shared resolver (see [LLM runtime](../reference/llm-runtime.md)). |
| `thinking` | `off` | `off` or `on`, sent as `chat_template_kwargs.enable_thinking`; an explicit switch in `extra_body` takes precedence. |
| `reasoning_effort` | `xhigh` | Graded effort for models that support one (Qwen 3.8: `xhigh`, `medium`, `low`), when thinking is on. |
| `search_reasoning_effort` | `low` | Effort for the first, search-only turn where graded effort applies; `None` uses `reasoning_effort` throughout. |
| `temperature`, `top_p`, `top_k` | `None` | Explicit sampling overrides of the profile. |
| `request_params`, `extra_body` | `{}` | Extra request fields; they cannot override the model, messages, streaming, response format, tools, credentials, or output reserve. |

Request pacing fields (`request_start_interval_seconds`, `capacity_retry_*`) are described under
[Request pacing](../reference/llm-runtime.md#request-pacing).

- **Effort:** models with only an on/off switch (Gemma 4, Nemotron 3.5 Super) and unknown models get no effort
  parameter. Later turns keep `reasoning_effort` because each can answer.
- **No replay:** each turn is a fresh system/user request. Provider reasoning fields and earlier assistant messages
  are never sent again, including on validation retries, and `preserve_thinking=False` is set for the chat template
  (this does not turn thinking off for the current response).
- **Context:** the output reserve is fixed and never reduced to fit a prompt. A search state that no longer fits
  returns `unknown` with `termination_reason="context_limit"`; reduce the memory and output limits or raise a
  validated `context_window`.

Patient questions, note excerpts, and the model-written memory reach the endpoint, so use an endpoint authorized
for those data. The harness writes no patient checkpoints or logs; the returned answers and evidence are themselves
patient data.

## Multiple patients and questions

```python
from matchminer_ai.patients import answer_patient_question_batch

batch = answer_patient_question_batch(
    [
        {"patient_id": "P001", "history": "Fabricated record. Assay completed.",
         "questions": ["Was the assay completed?", "Was imaging documented?"]},
        {"patient_id": "P002", "history": "Fabricated record. Assay ordered; not yet performed.",
         "questions": ["Was the assay completed?"]},
    ],
    llm=llm,
    max_parallel_patients=4,
    max_parallel_questions=4,
    max_active_questions=8,
    progress_callback=lambda event: print(event),
)
for patient in batch["patients"]:
    print(patient["patient_id"], [answer["status"] for answer in patient["answers"]])
```

Each patient dictionary has `patient_id` (a unique, nonempty string), `questions`, and `notes`, `history`, or both,
plus an optional `patient_summary`. Results keep patient and question order even when requests finish out of order,
and duplicate questions keep separate positions.

- **Scheduling:** up to `max_parallel_patients` patients are admitted, at most `max_parallel_questions` questions
  run per admitted patient, and at most `max_active_questions` question workers run in total.
- **Endpoint cap:** `llm.max_concurrent_requests` applies across all callers of the same endpoint in one Python
  process; separate processes do not share it.
- **Progress:** callbacks run on the calling thread once per completed question and receive only
  `patient_index`, `question_index`, `stage`, and `status`.
- **Failures:** an endpoint or worker failure returns `status="error"` for that question while the others continue.
  Invalid inputs, and unavailable isolation, fail before any endpoint request.

## The loop

Each patient-question pair gets a fresh, isolated Python worker holding the full record.

1. **Search first.** The first request is search-only: the model must answer with a Python cell, at the lower
   `search_reasoning_effort` where supported. The prompt asks it to search names, abbreviations, and synonyms
   together in one cell, with word boundaries around abbreviations.
2. **Read and decide.** Each request carries the question, the record's size and note metadata, the model's compact
   memory, and the last cell's code, output, original `source_excerpts`, and bounded error details. The initial
   prompt contains no note text. After a successful search the model should answer, or run one more cell for a
   specific gap, omitted passage, conflict, or code error. Missing documentation may finish as `unknown` after a
   successful multi-synonym search.
3. **Responses.** The model returns a JSON envelope: a Python cell as an array of source lines (joined with newlines,
   indentation kept) or a final answer, plus an updated memory string of discoveries, unsuccessful queries, and
   open questions. Python variables persist between cells; earlier turns and reasoning do not.
4. **Budgets.** After `max_cells` cells, one final-only turn must answer or declare unknown. `max_calls` caps
   generation requests per question, final answer and retries included. One request is always reserved for the final
   answer, so retries reduce the cells available; if an early answer fails after a successful search, the reserved
   request can still answer from that evidence. Token-counting requests do not count.

There is one model-controlled action (Python execution): no agent framework or tool router, no recursive model calls,
no embeddings, and no web search. Each active question loads the full record into its own worker, so memory scales
with the number of active questions, and evidence is not shared between questions.

### REPL helpers

| Name | What it gives the model |
| --- | --- |
| `history` | The full original text; DataFrame input is joined with a generated header (number, date, type) per note. |
| `notes` | DataFrame with `note_number`, `note_date` (UTC, `NaT` when missing), `note_type`, `note_text`; one row for text input. |
| `patient_summary` | The optional summary, or `None`. |
| `scan(patterns, context=160, limit=12)` | Searches 1 to `max_scan_patterns` regexes over the whole history. Returns `hits` (each with `start`, `end`, `quote`, `match_start`, `match_end`, `truncated`), the exact `match_count`, and `omitted` and `truncated` flags. |
| `search(pattern, start=0, context=250, limit=6)` | One regex with pagination: the same hit shape plus `has_more` and `next_start`. |
| `read(start, end)` | An original character slice. |
| `show_notes(selection, start=0, limit=6, chars=1600, pattern=None, context=250)` | Bounded original text of selected note rows, optionally centered on a pattern's first match; returns `notes`, `has_more`, `next_start`. |
| `pd`, `re`, `json`, `math`, `collections` | Preloaded modules. |

`scan` merges matches across all patterns, deduplicates identical spans, and samples hits spread across the record,
earliest and latest included when the output budget allows; sparse results within `limit` are all kept. Character
positions do not establish event chronology. Each regex is at most 2,000 characters, `context` 0-2,000, and `limit`
2-20 for `scan` and 1-20 for `search` and `show_notes`. The default allowance of 128 patterns lets the model pass its
whole synonym list in one call; `NoteSearchLimits(max_scan_patterns=...)` changes it for the worker and the prompt.

A cell can filter `notes` with pandas, by text, note type, or date:

```python
selected = notes.loc[
    notes.note_text.str.contains(r"\bCT\b|computed tomography", case=False, na=False)
    & notes.note_date.between(pd.Timestamp("2026-02-01", tz="UTC"), pd.Timestamp("2026-03-01", tz="UTC"))
]
selected
```

A final DataFrame, row, or note-text Series expression, or a `print` of one, displays bounded original passages
through `show_notes`. Date and type filters can hide undated notes and earlier or later evidence, so the prompt asks
the model to broaden them when needed; a note date is not necessarily an event date.

### Evidence capture

The search, read, and display helpers register the original spans they return, and the parent process rebuilds
each excerpt from its own copy of the record, so model code cannot alter the evidence. These `source_excerpts` are
sent on the next turn even if the cell did not print them. Printed calculations, counts, metadata, and modified
strings do not count as evidence, and the model never copies quotes, chooses evidence IDs, or computes offsets.
Excerpts and stdout share the cell output budget; omitted excerpts are flagged, and an excerpt that never reached
the model is not reported as reviewed. Excerpts from earlier turns are retained by the parent.

## Results

Each answer contains:

- **`question_index`, `question`, `status`:** `answered`, `unknown`, or `error`.
- **`answer`:** the answer or an explanation of the uncertainty; `null` for errors.
- **`evidence`:** the excerpts supplied to the model, each with verbatim `quote`, zero-based `start`, exclusive
  `end` (Python characters, not bytes), and `note_date`. DataFrame input adds `note_number` and any `note_type`,
  and its excerpts stay within one note, excluding generated headers; text input has `note_date` null. Contained
  and duplicate excerpts are merged. These are automatically retained context, possibly irrelevant or
  contradictory, not individually selected citations.
- **`limitations`:** unresolved clinical or documentation gaps, or a sanitized failure. An `unknown` answer always
  has one.
- **`metadata`:** cells run and succeeded, cell error types, generation requests and failed attempts, per-request
  timing and requested effort, worker startup and per-cell seconds, endpoint-reported prompt and completion tokens
  (`null` when any request lacked usage), finish reasons, fixed validation messages, output size and truncation
  counts, `max_cells`, `max_calls`, `max_scan_patterns`, `scope="searched_excerpts"`,
  `evidence_selection="automatic_reviewed_excerpts"`, and `termination_reason` (`answered`, `model_unknown`,
  `search_budget`, `call_budget`, `context_limit`, `endpoint_failure`, `worker_failure`, `cell_execution_failure`,
  or `consecutive_python_errors` when a workup backup endpoint is configured).

If every executed cell fails, the result is `error`, not missing documentation. The batch result has `patients`
(each with `patient_id` and `answers`), `notice`, and `metadata` recording the concurrency, budgets, model, thinking,
efforts, sampling, and output reserve used.

Automatic capture establishes text provenance, not clinical entailment; review whether the attached context
supports the answer. Search can miss synonyms, dispersed evidence, later updates, and contradictions. The prompt
distinguishes planned or ordered from performed or completed and asks for alternate searches and contradictory
evidence, but unknown or missing documentation does not establish that care was omitted.

## Limits and isolation

`NoteSearchLimits` holds every budget:

| Field | Default | Bound |
| --- | --- | --- |
| `max_cells` | 12 | Python cells per question before the final-only turn. |
| `max_calls` | 16 | Generation requests per question, retries included; `None` or at least 2. |
| `max_scan_patterns` | 128 | Patterns per `scan` call. |
| `max_output_chars` | 12,000 | Cell output, excerpts included. |
| `max_memory_chars` | 6,000 | Model memory between turns. |
| `max_code_chars`, `max_question_chars` | 6,000 each | One cell's code; one question. |
| `max_history_bytes` | 32,000,000 | UTF-8 size of the record. |
| `max_summary_chars` | 100,000 | The optional summary. |
| `cell_timeout_seconds` | 5 | Wall time per cell, enforced by the parent; also ends catastrophic regexes. |
| `worker_startup_timeout_seconds` | 20 | Worker start and dependency loading. |
| `worker_memory_mb` | 512 | Worker address space; at least 512 with pandas. |

The REPL requires Linux with `libseccomp.so.2` and fails closed: if isolation cannot be installed, no code runs and
there is no unsandboxed fallback.

- **Process:** Python starts with `-I -S`, an environment holding only numeric-library thread limits, working
  directory `/`, and no extra file descriptors. pandas and its dependencies load from the installed package paths
  before isolation.
- **Syscalls:** before any model code runs, a default-deny seccomp allowlist, applied to every thread, permits only
  in-memory Python and the existing pipes. Opening files, sockets, creating processes or threads, executing
  programs, and changing resource limits are blocked, so imports that need new files fail.
- **Resources:** the address space is capped at `worker_memory_mb`, core dumps and file writes are disabled, and
  note-type columns use Python-backed strings to avoid Arrow's large virtual reservations.

Python namespace restrictions are not the isolation boundary. Model code may change its own variables but not the
parent's copy of the record used to rebuild excerpts.

## Guideline workup review

```python
from matchminer_ai.patients import WorkupSearchReviewConfig, review_patient_workup_with_note_search

result = review_patient_workup_with_note_search(
    notes,                               # notes DataFrame or concatenated text
    [{"name": "CT chest", "conditions": "Stage II or higher"}],
    config=config,                       # MMAIConfig with a remote patient endpoint
    population_context="Fabricated guideline population text.",
    review=WorkupSearchReviewConfig(),   # max_review_passes=0 disables follow-up
)
```

The function runs one search question per workup item on the config's `patient` endpoint and sampling settings,
which must be remote. Each recommendation needs a `name` and may have text `conditions`.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `notes`, `history` | | A notes DataFrame or text, or both (DataFrame first, text as `history=`). |
| `patient_summary` | `None` | Navigation context only. |
| `population_context` | `None` | The guideline population's text. |
| `limits` | `NoteSearchLimits()` | Initial-search budgets: 12 cells and 16 requests per item. |
| `max_parallel_questions` | `None` | Items searched at once; `None` runs all. The endpoint's request cap still applies. |
| `thinking` | `"off"` | `"on"`, `"off"`, or `None` to keep the patient section's switch. |
| `search_reasoning_effort` | `"low"` | As in `NoteSearchLLMConfig`. |
| `review` | `WorkupSearchReviewConfig()` | Follow-up review budgets (below). |
| `backup_config`, `max_consecutive_failures` | `None`, 3 | An optional backup endpoint for items that fail repeatedly; see the [Patients API](../api/patients.md). |
| `progress_callback` | `None` | Receives progress strings. |

Models served without a thinking switch keep their own reasoning behavior whatever `thinking` says.

The search uses its own prompts: `patient.workup_search.system.txt` for the initial search and
`patient.workup_assessment.txt` for the shared care-status definitions. Each item's answer is the workup assessment
shape: `applicability` (`applies`, `not_applicable`, `uncertain`), `status` (`completed`, `partially_completed`,
`planned`, `not_done`, `not_documented`, `unclear`, or `error` when the search failed), and `bottom_line`. Missing or
unclear documentation is an `unknown` answer, and a documented status needs note evidence.

The result's `assessments` add `name`, `recommendation_index`, `recommendation`, `evidence` (with code-derived
`note_number` and `note_date`), `review_status` (`answered`, `unknown`, `error`), `limitations`, and per-item
`metadata`. Top-level `metadata` has `method="agentic"`, `scope="searched_excerpts"`,
`evidence_selection="automatic_reviewed_excerpts"`, the follow-up settings, `incomplete_review_items`,
`failed_items`, `requests`, note and undated-note counts, and which input supplied the notes. Endpoint or worker
failures are `status="error"` and never imply missing documentation. "Not documented" means nothing was located in
the searched excerpts, not that every note was clinically reviewed. The guideline fetcher runs every item at once (it
raises the endpoint's request cap to the item count) and labels the attached text "Reviewed patient-note excerpts".

### Follow-up review

By default each provisional assessment gets a coverage check and up to two focused reviews; the generic
question-answering functions do not.

1. **Vocabulary.** A patient-free request writes up to eight literal search terms for the item (names, synonyms,
   abbreviations, components) with `patient.workup_search_terms.system.txt`. An in-memory cache of up to 256 plans,
   keyed by item, guideline population, prompt, and endpoint settings, shares them across patients; it holds no
   patient text or answers, and concurrent identical requests share one plan.
2. **Literal matching.** Python searches every original note, excluding generated headers: escaped literals,
   case-insensitive, with flexible spaces, hyphens, and underscores. It counts every match, keeps at most 512
   positions (the first and last 256), and tracks which are already covered by supplied excerpts.
3. **Zero matches.** For an `unknown` assessment with no matches, one more request asks for alternative wording and
   Python searches again (`retry_zero_match_missing`). Zero matches still leave documentation uncertain.
4. **Focused review.** Unseen matches, uncertain answers with matches, and conflicts the model flagged get up to
   `max_review_passes` reviews with `patient.workup_search_review.system.txt`. Each request carries only the item, its
   provisional assessment, the review reasons, and original passages: up to `review_hits_per_item` positions,
   including one already-seen position when there are unseen ones, with `review_context_chars` (360) around each hit
   on the first pass and twice that on the second. Excerpts stay inside one note, with code-derived positions,
   numbers, and dates.
5. **Full-record fallback.** Items still unresolved or with unreviewed matches (unresolved first, then input order),
   up to `max_full_record_fallback_items` per call, are reviewed serially over every note in overlapping chunks of
   at most `max_evidence_chars`, halved as needed to fit the context, carrying the assessment forward. At most
   `max_full_record_chunks` chunks run per item; reaching a chunk, request, or context limit is reported as
   incomplete, never as a complete review.

| `WorkupSearchReviewConfig` field | Default | Meaning |
| --- | --- | --- |
| `max_review_passes` | 2 | Focused reviews per item; 0 disables all follow-up and fallback. |
| `retry_zero_match_missing` | `True` | Ask once for alternative terms when nothing matched. |
| `review_hits_per_item` | 4 | Match positions sampled per review. |
| `review_context_chars` | 360 | Context around each hit on the first pass. |
| `max_evidence_chars` | 16,000 | Passage text per focused request or fallback chunk. |
| `max_calls_per_item` | 12 | Follow-up requests per item, vocabulary and retries included. |
| `max_full_record_fallback_items` | 5 | Items per call that may use full-record fallback. |
| `max_full_record_chunks` | 8 | Fallback chunks per item. |

Follow-up shares the endpoint's concurrency, provider, sampling, thinking, and effort settings and adds no tool calls
or reasoning replay. A failed follow-up keeps the last validated assessment with an explicit limitation and
`metadata.followup_review.status="incomplete"`. An invalid review reply may still use full-record fallback; an
endpoint failure stops further requests for that item.

Per-item `metadata.requests` counts initial and follow-up requests; `metadata.initial_search_requests` and
`metadata.followup_review.requests` separate them. The follow-up audit records the terms, zero-match research,
changes per pass, supplied ranges, match coverage, fallback completion, and failures. The checks improve retrieval
coverage; they do not show that all relevant evidence was found, and populated answers can still be wrong.

## Evaluating efficiency and recall

The harness provides measurement hooks, not an established speedup or accuracy result. Compare it with full-record
review on the same authorized records and questions. Measure prompt and completion tokens and wall time, plus missed
positive evidence, unsupported answers, planned-versus-completed errors, contradiction handling, and appropriate
unknowns. Include abbreviation variants, negative statements, duplicate notes, and evidence far apart in time. Small
records may cost more, because search adds model turns. Full-note workup review batches recommendations, so compare
equivalent item sets.

## Fabricated smoke test

After checking the server's health and the model it reports, run:

```bash
python examples/note_search_qa_smoke.py \
  --endpoint http://localhost:8000/v1 \
  --model served-model-id
```

The script builds two fabricated histories of about 127,000 characters and four questions: a completed test, a
pending result, an explicitly canceled procedure, and a missing test. It runs patients and questions concurrently
with four cells per question and prints the answers and usage as JSON. `--context-window` (default 262,144) and
`--reasoning-effort` (`xhigh`, `medium`, or `low`) are optional.
