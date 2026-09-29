# Experimental note-search question answering

`answer_patient_questions` answers arbitrary questions about one full-text
patient history. `answer_patient_question_batch` runs the same workflow across
patients and questions concurrently. This is a standalone, opt-in alternative to
reading every note chunk for every question. The guideline fetcher also offers
this harness as an optional workup-review method; full-note review stays the default.
Embedding-based raw-note QA and summarization are unchanged.

## Guideline workup review

`review_patient_workup_with_note_search(notes, recommendations, config=config)`
uses the patient endpoint in `MMAIConfig` and the same isolated search loop, with
one structured assessment per catalog workup item. It accepts raw text or a
single-patient note DataFrame, with optional `population_context`, `limits`,
`max_parallel_questions` (default `None`, meaning all supplied items) and a
string-valued `progress_callback`. Default workup limits are
`NoteSearchLimits(max_cells=2, max_calls=3)`: up to two search cells plus the final
answer, with validation and transport retries sharing the same three-call budget.
The normal path is one multi-pattern search cell followed by a final assessment.
A further cell is reserved for a concrete gap, conflict, or execution error;
the prompt explicitly asks the model to stop when it has sufficient evidence.
`search_reasoning_effort="low"` applies to the initial search-only request where
graded effort is supported. Subsequent turns retain the configured assessment
effort because each can answer. Set this option to `None` to inherit the assessment
effort on every request. Thinking remains enabled and traces are never replayed.
The configured remote request cap also applies. The guideline fetcher sets that
cap to the number of workup items; other jobs retain their existing settings.

It returns the existing workup `assessments` shape: recommendation, applicability,
status, bottom_line, automatically retained original excerpts and code-derived note numbers/dates. Structured
dates come only from DataFrame input; pasted text remains undated. Quotes cannot
include generated headers or cross note boundaries. Each item additionally retains
`review_status` (answered/unknown/error), limitations and search metrics. Endpoint
or worker failures become `status="error"`; they never imply missing documentation.
No extra synthesis agent is used. The existing full-note API remains available.

Agentic results have `metadata.method="agentic"` and `scope="searched_excerpts"`.
They also have `evidence_selection="automatic_reviewed_excerpts"`. The model
returns no citation IDs, quotes or offsets. The frontend/export labels the
automatically attached text **Reviewed patient-note excerpts**; these are context
supplied to the model, not individually selected supporting citations.
“Not documented” means no documentation was located in the searched excerpts,
not that all notes were clinically reviewed. The frontend and exports display
this distinction, as well as limitations and the actual method used. Both modes
use the browser session's saved endpoint/provider, model and reasoning settings.

## One patient

```python
from matchminer_ai.patients import (
    NoteSearchLLMConfig, NoteSearchLimits, answer_patient_questions,
)

llm = NoteSearchLLMConfig(
    base_url="http://your-authorized-server:8001/v1",
    model="the-model-id-reported-by-your-server",
    context_window=65536,  # Set within the model's actual capacity.
    safety_tokens=1024,
    max_concurrent_requests=8,
    timeout=120,
    attempts=2,
    tokenizer_mode="bytes",
    response_format="json_schema",
)

result = answer_patient_questions(
    "Fabricated record. Day 1: assay ordered. Day 8: assay completed, result pending.",
    ["Was the assay performed?", "Is the assay result documented?"],
    llm=llm,
    limits=NoteSearchLimits(max_cells=8, max_scan_patterns=128),
    max_parallel_questions=4,
)
for answer in result["answers"]:
    print(answer["status"], answer["answer"], answer["evidence"])
```

`NoteSearchLLMConfig` defaults to `sampling_profile="auto"` and thinking on.
It calls the repository's shared sampling resolver: Gemma 4 and Qwen 3.8 settings
are inherited from that implementation, not copied into this harness. Qwen uses
`xhigh` assessment reasoning effort by default (configurable to `medium` or `low`).
The separate `search_reasoning_effort` defaults to `"low"` for the first,
search-only turn, or `None` to inherit the assessment effort. The shared resolver
keeps Qwen's API and chat-template effort synchronized. Gemma 4 uses an on/off
switch, so it keeps thinking enabled without an invented graded effort setting.
Unknown models receive no new effort parameter unless one was already configured.
An omitted
`max_tokens` uses the existing model-profile budget floor when present (currently
32,768 for Qwen3.8-Flash-Next), otherwise 8,192. Explicit sampling fields and
`request_params`/`extra_body` overrides follow the shared resolver's behavior.

Each turn is a fresh system/user request. Provider reasoning fields and prior
assistant messages are never inserted into subsequent requests, including validation
retries. `preserve_thinking=False` is set for the chat template; this does not turn
thinking off for the current response. Memory holds bounded factual evidence and
search state, with explicit instructions not to copy reasoning traces.

The endpoint and model are explicit; this workflow never selects a fallback
server. `OPENAI_API_KEY` supplies the bearer token by default; use `api_key_env`
to name a different environment variable. The existing structured client's
provider settings can also be used. Patient questions, extracted note excerpts,
and accumulated model-written memory reach the endpoint. Use an endpoint
authorized for those data. The harness writes no patient checkpoints or logs;
returned answers and evidence are themselves patient data.

`tokenizer_mode="bytes"` conservatively budgets UTF-8 bytes instead of claiming
exact model token counts and avoids a tokenizer request per turn. Use
`"endpoint"` with a vLLM server's `/tokenize` route for exact accounting. The
output reserve is fixed and never reduced to fit a prompt. An oversized search
state returns `unknown` with `termination_reason="context_limit"`. Reduce the
configured memory/output sizes or increase the validated context budget.

## Multiple patients and questions

```python
from matchminer_ai.patients import answer_patient_question_batch

batch = answer_patient_question_batch(
    [
        {"patient_id": "synthetic-a", "history": "Assay completed.",
         "questions": ["Was the assay completed?", "Was imaging documented?"]},
        {"patient_id": "synthetic-b", "history": "Assay ordered; not yet performed.",
         "questions": ["Was the assay completed?"]},
    ],
    llm=llm,
    max_parallel_patients=4,
    max_parallel_questions=4,
    max_active_questions=8,
    progress_callback=lambda event: print(event),
)
```

Patient IDs must be unique strings. Each patient has its own question list.
Results preserve both patient and question order, even when requests finish out
of order. Duplicate questions retain separate positions. The scheduler admits
up to `max_parallel_patients` patients, at most `max_parallel_questions` questions
per admitted patient, and `max_active_questions` total live question workers.
The structured client separately enforces `llm.max_concurrent_requests` across
all callers of the same endpoint in one Python process. Multiple application
processes do not share that cap.

Progress callbacks run on the calling thread once per completed question and
receive only patient/question indices, stage, and status. Individual worker or
endpoint failures return an `error` for that question; other questions continue.
Invalid batch inputs and unavailable isolation fail before endpoint requests.

## The loop

1. Start a fresh Python worker for each patient/question pair. The full history
   is available as the local `history` string, with `re` and bounded
   `scan(patterns, context=160, limit=12)`,
   `search(pattern, start=..., context=..., limit=...)` and `read(start, end)` helpers.
   `scan` searches every supplied regex throughout the history and returns
   `hits`, `match_count`, `omitted`, and `truncated`. It retains bounded samples
   from different character positions, including earliest/latest matches when
   the output budget permits. Sparse results within the limit are all retained;
   identical match spans are deduplicated. Counts include distinct overlapping spans;
   positions do not establish event chronology. Each hit has `start`, `end`,
   `quote`, `match_start`, `match_end`, and `truncated`. `search` remains available
   for targeted pagination, with the same hit shape plus `has_more`/`next_start`.
2. Send the model the question, history size, compact working memory, and the latest
   cell code, output, original `source_excerpts`, and bounded error details. The initial prompt contains no
   note excerpts.
3. The initial request is search-only with lower effort where supported. The
   model is instructed to search synonyms/components in one cell, using word
   boundaries around abbreviations. After a successful search, it should answer
   immediately unless a specific evidence gap warrants another cell. Missing
   documentation may finish as unknown after a successful multi-synonym search.
   The complete response schema is included in the prompt as well as the request
   format. The model emits a Python cell as an array of source lines, or a final
   answer,
   in a JSON envelope. The harness joins source lines with newlines, preserving
   indentation. Cells can use normal Python variables, string processing, regex, and loops. Their
   variables persist. Search hits include original character offsets and a
   pagination cursor. Oversized output is truncated with an explicit flag.
4. Search/read helpers automatically register the source spans they retrieve.
   The parent rebuilds these from the original notes and sends them as
   `source_excerpts`, even if the cell did not print its return value. Use `read()`
   for arbitrary slices; printed calculations or modified strings alone cannot
   establish source provenance. Source excerpts and stdout share the cell output
   budget. Omitted excerpts are flagged explicitly. The parent retains excerpts
   sent on earlier turns without asking the model to remember IDs or copy quotes.
   Between cells, the model updates a bounded memory string containing discoveries,
   unsuccessful queries, and remaining uncertainties.
   Earlier conversation turns are not replayed, but Python variables remain. The
   latest source cell is retained to make syntax and name errors repairable;
   reasoning traces are not retained.
5. After at most `max_cells` cells, one final-only model turn must answer or
   declare unknown. Optional `max_calls` (at least two) additionally caps total
   generation requests per question, including the final answer and retries.
   The harness reserves one request for the final answer and reduces the available
   search cells when retries consume calls. A failed final response cannot retry
   past that cap. If an early answer/follow-up fails after a successful search,
   the reserved last request can still produce a final answer from that evidence.
   Model discovery and token-counting requests are not generation
   calls. The general-purpose QA API defaults to `max_calls=None`; the workup
   wrapper defaults to three. Endpoint retry limits apply within the total budget.

There is one model-controlled action (Python execution), no agent framework or
multi-tool router, no recursive submodel calls, no embeddings, and no web search.
Full histories are loaded independently into active question processes; memory
use therefore scales with the active question count. There is no cross-question
evidence cache in this first implementation.

## Results and interpretation

Each answer contains:

- `question_index`, `question`, and `status`: `answered`, `unknown`, or `error`.
- `answer`: the grounded answer or explanation of uncertainty; `null` for errors.
- `evidence`: verbatim `quote`, zero-based `start`, exclusive `end`, and `note_date`.
  These are automatically retained excerpts supplied to the model, including
  potentially irrelevant or contradictory context. The parent reconstructs each
  quote from the original history independently of mutable worker strings. No
  model-generated quote, offset or citation ID is accepted. Repeated text, Unicode,
  line breaks and source offsets are preserved. Contained/duplicate excerpts are
  deduplicated; earlier distinct excerpts survive subsequent turns. Excerpts that
  could not be supplied because of the context limit are not labeled reviewed.
  Offsets count Python characters, not UTF-8 bytes.
  `note_date` is always `null` for this full-text API; dates are not inferred.
- `limitations`: unresolved clinical/documentation gaps or a sanitized failure.
- `metadata`: executed cell count, actual generation request attempts, elapsed
  time per generation request (including failures, with requested effort), worker
  startup time and execution time per completed cell,
  seconds, successful-cell count, cell error types, provider finish reasons,
  endpoint-reported prompt/completion tokens, output character count,
  truncation count, and termination reason and fixed validation error messages. Token totals are `null` when any
  request's usage is unavailable; failed attempts are still counted as requests.
  If every executed cell fails, the result is `error`, not missing documentation.

Automatic capture establishes text provenance, not clinical entailment or selection
of supporting citations. Review whether the attached context supports the answer.
Search can miss synonyms, dispersed evidence, later updates, and contradictions.
Unknown/missing documentation does not establish that care was omitted. The
prompt distinguishes planned/ordered from performed/completed and asks for
alternate searches and contradictory evidence. A generated answer is a research
documentation-review result for human review, not definitive guideline concordance.

## Search-pattern allowance

`scan` accepts up to 128 regular expressions in one call by default. Set
`NoteSearchLimits(max_scan_patterns=...)` to another positive integer; the
same setting reaches the worker and the mini-agent system prompt. The model
can supply its complete synonym list without splitting it into small calls.
This also applies to the optional workup-review mini-agents.

The scanner merges match iterators across the supplied patterns and retains
exact counts, deduplicating identical spans even across many patterns.
`limit` still caps returned hits at 2-20, with omission/truncation flags;
each regex remains limited to 2000 characters. Output, source-excerpt,
memory, and cell-time budgets remain in force. A larger pattern allowance
does not establish that every relevant finding was reviewed. The configured
allowance is recorded in answer and batch metadata.

## Execution limits

The REPL requires Linux with `libseccomp.so.2`. It launches Python with `-I -S`,
an empty inherited environment, closed extra file descriptors, and no package
site initialization. Before any model code executes, a default-deny syscall
allowlist blocks filesystem opens, network sockets, process/thread creation,
execution of programs, and changes to resource limits. Only operations needed
for in-memory Python and its existing communication pipes are allowed. Imports
requiring new files consequently fail; `re`, `json`, `math`, and `collections`
are preloaded. Python namespace restrictions are not used as the isolation boundary.

Each cell has a parent-enforced wall deadline (default 5 seconds); this also
terminates catastrophic regex searches. The process has a default 512 MiB address
space limit and cannot write core dumps. Cell output defaults to 12,000 characters,
memory to 6,000 characters, and history to 32 MB of UTF-8 text. Limits are adjustable
with `NoteSearchLimits`. If isolation cannot be installed, the harness fails closed;
there is no unsandboxed fallback. Model code may alter its own Python variables,
but it cannot alter the parent's original history used to reconstruct excerpts.

## Evaluating efficiency and recall

This implementation provides measurement hooks, not an established speedup or
clinical accuracy result. Compare the same authorized records and questions
against the full-record method. Measure total prompt/completion tokens and wall
time, plus missed positive evidence, unsupported answers, planned-versus-completed
errors, contradiction handling, and appropriate unknowns. Include abbreviation
variants, negative statements, duplicate notes, and evidence far apart in time.
Small records may cost more because search introduces additional model turns.
The existing workup review batches recommendations, so compare equivalent item
sets and account for its batching when measuring the baseline.

## Repeating the fabricated smoke test

After verifying the server health and reported model, run:

```bash
python examples/note_search_qa_smoke.py \
  --endpoint http://sn4622130540:8001/v1 \
  --model Inferact/Qwen3.8-Flash-Next-NVFP4
```

The example creates two 127,000-character fabricated histories and four questions
covering a completed procedure, a pending result, an explicitly canceled procedure,
and a missing test. It exercises concurrency across patients and questions.
It uses the default thinking and sampling profiles and prints the answer/usage JSON.
The model and server in this command are examples from the local development test,
not a service bundled with the package.

### Development smoke result, 2026-09-29

On the verified `sn4622130540:8001` endpoint serving
`Inferact/Qwen3.8-Flash-Next-NVFP4`, the example returned the expected answers for
all four fabricated questions: HER2 testing completed, its result not yet
returned, CT chest explicitly not performed, and ECG documentation unknown.
Both patient and question concurrency were enabled. Thinking was on at the
inherited `xhigh` effort, with `preserve_thinking=False`.

The run used 17 generation requests, 31,214 prompt tokens, and 35,303 completion
tokens, taking 522.68 seconds. All completions finished normally. Two recoverable
Python cell errors were exposed to the model and corrected or worked around.
The repository suite passed 634 tests and 71 subtests, with four GPU/resource
tests deselected.

For input-volume accounting, the same server counted 114,471 tokens for the
four question-plus-full-history prompts, excluding an answering system prompt
or schema. The REPL run therefore used about 72.7% fewer input tokens in this
small synthetic example. No full-record answers were generated for this
comparison, so it does **not** establish relative latency, total generation cost,
or clinical accuracy. Server load was substantial during testing. The large
completion-token count makes reasoning effort a useful next evaluation variable;
this earlier run used uniform xhigh effort. The current implementation uses the
separate low initial-search effort described above.
