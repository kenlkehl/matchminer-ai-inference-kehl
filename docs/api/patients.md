# Patients

::: matchminer_ai.patients
    options:
      members:
        - compress_patient_note
        - compress_patient_notes
        - NoteCompressionError
        - answer_question_with_raw_patient_notes
        - concatenate_patient_note_pdfs
        - full_patient_screen
        - summarize_patients
        - review_patient_workup
        - review_patient_workup_with_note_search
        - WorkupSearchReviewConfig
        - answer_patient_questions
        - answer_patient_question_batch
        - NoteSearchLimits
        - NoteSearchLLMConfig
        - structure_patient_summaries
        - structure_patient_summary

## Note compression

`compress_patient_notes` sends one independent request per nonblank `note_text`
row to the configured remote patient LLM. Before dispatch, every run of
whitespace (including spaces, tabs and newlines) becomes one ASCII space, and
leading/trailing whitespace is removed. Original source text stays intact in the
returned DataFrame; `compressed_note_text` holds only the final compressed note.
Input row order, index, patient IDs, dates and other columns are preserved, and
only note text is sent to the endpoint. The API accepts multiple patients without
mixing their notes or adding patient metadata to the prompt.

```python
from matchminer_ai import load_default_preset
from matchminer_ai.patients import compress_patient_note, compress_patient_notes

config = load_default_preset()
config.remote["enabled"] = True
config.remote["server_urls"] = ["http://localhost:8002/v1"]
config.patient["remote"]["model_name"] = "my-endpoint-model"

compressed = compress_patient_notes(
    notes_dataframe,
    config=config,
    max_concurrent_requests=8,
    progress_callback=lambda completed, total: print(completed, total),
)
one_note = compress_patient_note("Fabricated note text.", config=config)
```

Concurrency defaults to `config.remote["max_concurrent_requests"]`. Requests
share the existing process-wide endpoint cap, authentication, model sampling,
streaming, timeout, bounded retries, optional dispatch pacing and capacity
cooldowns. The existing cancellation scope also stops compression workers. The
module requires one endpoint, so model discovery and context accounting describe
the same server. If the endpoint supplies no model context limit, set
`config.remote["context_window"]` explicitly. For providers without vLLM's
`/tokenize` route, explicitly set `config.remote["tokenizer_mode"] = "bytes"` to
use conservative UTF-8 byte budgeting. Output-token reserves follow the patient
LLM request parameters.

Reasoning defaults to off for both APIs where supported, even if the patient LLM
configuration enables thinking. The request sends
`chat_template_kwargs.enable_thinking=false`, with inherited reasoning-effort
fields removed. An adapter that cannot disable reasoning, including Gemini,
omits this switch and retains its configured reasoning settings. If an endpoint
explicitly rejects the switch with HTTP 400 or 422, the request is retried once
at the same endpoint without the switch. Other failures keep the normal bounded
retry behavior. Set `thinking="on"` to opt in. The prompt stays the same;
reasoning mode is controlled only by the API argument. These overrides do not
mutate the shared configuration. The returned table's `attrs["note_compression"]`
records `thinking_requested` and dispatched `thinking` (`off`, `on`, `default`,
`mixed`, or `not_requested` for entirely blank input). `default` means that no
thinking switch was sent; it does not establish whether the model reasoned.

The prompt is: “Compress this note to extreme but lossless token density while
remaining understandable. Return only the compressed note, with no explanatory
text, commentary, or preamble.” No preservation checklist is appended. Separate
reasoning is discarded; blank, refused, token-limited, malformed and recognizable
explanatory/inline-reasoning responses are rejected with bounded retries. No
patient checkpoints are created and no clinical text is logged. A failure raises
instead of returning a partial table or inserting an error as a compressed note.
Oversized notes are rejected, never truncated or automatically split.

Blank strings produce `""` without a request. Null/non-string text and an existing
`compressed_note_text` column are rejected. Compression is experimental: asking
for losslessness does not establish it. Keep original notes for human review and
exact evidence quotations. Use only an endpoint authorized for the input data.
This is a standalone research workflow; existing summarization and note-question
answering do not automatically use compressed notes.

## PDF patient records

`summarize_patients` accepts its existing note-level DataFrame or one local PDF
path (or an ordered sequence of PDF paths) for a single patient. PDF input uses
the same local, page-aware embedded-text extraction and RapidOCR fallback as
`ocr_pdf`, combines the documents into one long note string, and then runs the
existing serial summarization workflow.

```python
from matchminer_ai.patients import summarize_patients

summary = summarize_patients(
    ["record-part-1.pdf", "record-part-2.pdf"],
    patient_id="research-patient-1",
    config=config,
)
```

To inspect or edit the extracted long note before summarization, call the
preparation function directly:

```python
from matchminer_ai.patients import concatenate_patient_note_pdfs

long_note = concatenate_patient_note_pdfs(
    ["record-part-1.pdf", "record-part-2.pdf"]
)
```

Document order is the order supplied by the caller. Numbered boundary markers
are inserted, but filenames and filesystem timestamps are not copied into the
patient text and no clinical dates are inferred. OCR is probabilistic; compare
important content with the original PDFs. OCR runs locally, while subsequent
summarization sends the extracted patient text to the configured LLM backend.
Use only an endpoint authorized for the records' sensitivity.

## Raw-note question answering

`answer_question_with_raw_patient_notes` accepts either a pre-concatenated EHR
string or a DataFrame for one patient. For DataFrame input, `note_text` and
`note_date` are the default columns. The function parses and stably sorts note
dates, adds dated note headers, and concatenates every non-empty note without
deduplicating the record. It chunks each note independently, so a chunk never
crosses a note boundary and carries exactly one source `note_date`.

The selected Hugging Face SentenceTransformer model supplies both the chunking
tokenizer and embeddings. The LLM agent starts with semantic retrieval for the
original question, then can request focused `pull_relevant_input_text` searches
or ask narrower related questions. These tools only query the in-memory note
index; they do not call web search. A final evidence item is accepted only when
its chunk was retrieved and its quote is an exact substring of that chunk.
For DataFrame input, each accepted evidence object also receives the chunk's
code-derived `note_date`. This field is not trusted from LLM output.
Pre-concatenated string input has no structured source-date mapping and therefore
returns `note_date: null`.

```python
from matchminer_ai import load_default_preset
from matchminer_ai.patients import answer_question_with_raw_patient_notes

config = load_default_preset()
config.remote["enabled"] = True
config.remote["server_urls"] = ["http://localhost:8002/v1"]
# Set this to the model ID reported by the configured endpoint.
config.raw_patient_note_qa["remote"]["model_name"] = "my-vllm-model"
# Qwen raw-note retrieval uses CUDA by default.
config.raw_patient_note_qa["embedding_device"] = "cuda"

answer = answer_question_with_raw_patient_notes(
    "What treatment response is documented, and when?",
    notes_dataframe,
    embedding_model_name="Qwen/Qwen3-Embedding-0.6B",
    config=config,
)
```

The returned JSON-compatible dictionary contains the original `question`, a
grounded `answer`, exact-quote `evidence`, and `limitations`. Embedding retrieval
runs locally, but retrieved patient excerpts reach the configured LLM endpoint.
Use only an endpoint authorized for the data's sensitivity. The output is for
research review and does not establish diagnosis, treatment recommendations,
or trial eligibility. This standalone workflow does not add retrieval or note
tagging to `summarize_patients`, whose serial summarization behavior is
unchanged.

## Full patient screen

`full_patient_screen` accepts the same concatenated-note string or dated,
one-patient DataFrame as raw-note question answering plus either the complete
eligibility-criteria text for one trial or the structured mapping returned by
`extract_trial_space_eligibility_criteria`. An LLM first converts every
independently assessable protocol requirement into a focused raw-note question.
Each exact non-empty criteria line receives a code-assigned source ID before the
LLM call. The LLM references that ID, and the package attaches the original text
after validation, so harmless model rewriting cannot break source grounding.
The package then calls `answer_question_with_raw_patient_notes` for every
question and retains each grounded answer, exact-quote evidence item, limitation,
and code-derived `note_date`.

With a remote backend enabled, the package embeds the raw-note chunks once on
the configured device (CUDA by default), transfers the normalized index to CPU,
and runs independent questions in spawned processes. A single parent-owned
embedding model supplies the small dynamic query vectors, while each worker
performs cosine retrieval against the prepared CPU index and issues its own LLM
requests. Workers do not re-embed the note record or load duplicate embedding
models. The configured worker count is capped by the question count and
available CPUs. For an in-process local vLLM backend, set `max_workers=1`; the
package will not duplicate a local GPU engine across workers.

```python
from matchminer_ai.patients import full_patient_screen

screen = full_patient_screen(
    notes_dataframe,
    complete_eligibility_criteria,
    config=config,
    max_workers=4,
)
```

For an uploaded eligibility PDF, applications can keep all reusable behavior
inside the package by composing `ocr_pdf`,
`extract_trial_space_eligibility_criteria`, and `full_patient_screen`. When the
structured mapping is supplied, criterion type and exact criterion text are
attached from code-assigned sources rather than copied from model output.

The final JSON-compatible object includes an overall research signal, a concise
summary, one result per criterion with the complete raw-note QA response, known
limitations, execution counts, and a code-generated research-use notice. It is
not an eligibility determination. Retrieved excerpts and the aggregated
patient-specific results reach the configured LLM endpoint; no web search is
used. Use only an endpoint authorized for the sensitivity of the patient data.

## Structured patient summary schema

`structure_patient_summary` accepts a free-text cancer-history summary and
returns a JSON-compatible dictionary. `age` and `sex` are patient-level. The
`cancers` array contains one entry per active cancer; each entry contains:

- `cancer_type` and `histology` objects grounded to OncoTree codes;
- one biomarker object per marker, with `marker`, `type`, and `result`;
- one treatment object per source treatment-history line, retaining dates and
  response;
- a `drugs` array with source name, normalized NCIt preferred name and code,
  NCIt-definition-grounded target and mechanism, and normalization status; and
- `cancer_burden`, constrained to `early_or_curative_intent` or
  `advanced_or_palliative_intent`.

The OncoTree agent sees only one hierarchy level at a time. The NCIt agent sees
only bounded search candidates and pulls full definitions only for candidate
indices it selects. A complete ontology is never placed in LLM context.

For large-scale conversion, `structure_patient_summaries` accepts an ordered
sequence of summary strings and returns the corresponding dictionaries in the
same order. It batches all prompts whose dependencies are ready: initial fact
extraction across patients, each active OncoTree traversal level, and each NCIt
candidate-selection or definition-resolution step. Repeated normalized drug
mentions are resolved once per batch and retain each patient's original source
name in the final records.

```python
from matchminer_ai import load_default_preset
from matchminer_ai.patients import structure_patient_summaries

config = load_default_preset()
config.remote["enabled"] = True
config.remote["server_urls"] = ["http://localhost:8002/v1"]
config.patient_structuring["remote"]["model_name"] = "my-vllm-model"

structured = structure_patient_summaries(
    patient_table["cancer_history_summary"].tolist(),
    config=config,
)
patient_table["structured_patient_summary"] = structured
```

Remote mode uses `remote.max_concurrent_requests` per server,
`remote.batch_size`, and round-robin distribution across `remote.server_urls`.
These settings bound concurrent OpenAI-compatible chat requests; the endpoint
may then apply its own continuous batching. Local mode passes each ready prompt
wave directly to the in-process vLLM engine as one prompt list. Later ontology
steps remain dependent on earlier model outputs, so the workflow uses several
batched waves rather than one flat request.

## Workup documentation review

```python
from matchminer_ai.patients import review_patient_workup

review = review_patient_workup(
    raw_notes,                       # single-patient note DataFrame or raw string
    matched_space["diagnostic_workup"],
    config=config,                  # patient LLM config; remote.enabled=True
    population_context=matched_space["clinical_space_summary"],  # guideline target, not patient facts
    progress_callback=print,
)
```

This opt-in workflow reviews **all** raw notes serially, carrying forward a running
JSON assessment for every recommendation. It is independent of patient summarization;
no patient summary or vector retrieval is used as evidence. DataFrames accept
`note_text`, optional `note_date`, and optional single-valued `patient_id`. Dates
are validated and stably sorted; undated notes follow dated notes without assuming
that their events happened later. String input has no structured date provenance.
Repeated sentences are retained so dated evidence and contradictions are not lost.

The result contains `assessments`, `notice`, and `metadata`. Each assessment retains
the original recommendation and its zero-based input index, plus `status`
(`completed`, `partially_completed`, `planned`, `not_done`, `not_documented`, `unclear`),
`applicability` (`applies`, `not_applicable`, `uncertain`), `bottom_line`, and
`evidence` (`note_number`, code-assigned `note_date`, verbatim `quote`). Note numbers
are one-based chronological positions. Evidence is validated against the current
fragments or previously accepted quotes; prior evidence survives later updates.
Quotation validation verifies provenance, not clinical interpretation.

Uses patient model, vendor sampling and reasoning settings (xhigh by default).
Requires one OpenAI-compatible remote endpoint with `/tokenize` and advertised
`max_model_len`, or explicit `patient.context_window`; `patient.tokenizer_mode=bytes`
is an explicit conservative fallback for providers without `/tokenize`.
The full advertised context is available, reserving the configured patient
`remote.request_params.max_tokens` (20,000 by default). Chunk size/overlap default
to patient settings (50,000/500); optional function overrides and
`recommendation_batch_size=6` bound each serial request. Oversized prompts split
note packets further, never silently truncate notes or reduce output capacity.
The model returns ordered natural-language names, not catalog IDs.

Requests and responses stay in memory; no patient checkpoints or reasoning traces
are returned or written. Three bounded attempts reject incomplete, malformed or
ungrounded output. Provider and validation errors are not echoed with patient text.
The configured endpoint receives raw notes and must be authorized for the input.

Both this function and `review_patient_workup_with_note_search` accept
`backup_config=None` and `max_consecutive_failures=3` (an integer from 1 to 10).
Pass a separate `MMAIConfig` with the backup's endpoint, model, sampling and reasoning
to opt in. No implicit backup or extra endpoint probe occurs on the successful path.
After repeated failed request/validation attempts, full-note review switches once,
retries the failed packet against the backup's context capacity, and retains all
validated prior assessments and evidence. Remaining packets use the backup.
Metadata includes `primary_model`, `backup_used`, `backup_events`, actual request
attempts, and `validated_packets`.

Agent review also switches after consecutive Python errors within a question,
resetting that streak after a successful cell. Only the failed item restarts in a
fresh isolated REPL; the backup gets the same bounded per-attempt cell/request
budget and source record. Successful items and clinical unknown findings are
retained. Follow-up uses the producing model, can switch once after repeated
failures, and retains its existing total call budget. A failed backup never
triggers another model switch. Errors remain errors and failed follow-up retains
the last validated finding with an explicit incomplete status. Per-item metadata
records each switch, primary/backup models, reasons, combined costs and caps;
request metrics identify their actual models. Backups never bypass worker isolation,
cancellation, output/context reserves or exact-evidence validation.

This is a documentation review for human review, **not an overall guideline
concordance determination**. Not documented does not mean not done. Conditional
indications, compound items, alternatives, historical tests, timing and conflicting
evidence need clinical review. Treatment-concordance assessment is not included.
