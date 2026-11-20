# Patients

`summarize_patients` returns `patient_id`, the complete `patient_summary`, and
its two sections. The complete text contains all cancer-history sections and
ends with:

```text
Other relevant history:
ECOG 1. Chronic kidney disease. Remote inactive prostate cancer.
```

| Column | Content | Read by |
| --- | --- | --- |
| `patient_summary` | The complete summary. | Serial updates, summary-quality checks |
| `cancer_history_summary` | The **key** section: the text before `Other relevant history:`. | TrialSpace retrieval, the decision model's match questions |
| `general_exclusion_criteria_evidence` | The **boilerplate** section: the text after the header (`Not documented.` when absent). | The decision model's boilerplate question |

The key section's treatment history lists anti-cancer treatment only, one line per regimen (agents, dates, best response), and each serial update carries every earlier regimen forward (`patient.serial.user.primer.txt`, since 2026-11-12).

The boilerplate section records the evidence for the general criteria most trials share, with dates: performance status, measurable disease, brain and other CNS metastases and their treatment, the most recent routine labs (blood counts, coagulation, kidney and liver function), ejection fraction and QTc if known, toxicities of earlier cancer treatment, conditions other than cancer (including transplants and drug allergies), recent major surgery other than cancer surgery, other and inactive cancers, participation in any clinical trial, and hospice enrollment or a decision to stop anti-cancer therapy (`prompts/patient.serial.user.primer.txt`).

Structured, evidence-linked records from the same note table (document-level
and patient-level extraction) are described in
[Structured Clinical Extraction](extraction.md).

Historical boilerplate headings are normalized on output. `embed_for_matching`
and the match-quality and exclusion checkers also accept a complete
`patient_summary` in place of a section column and take the section they need. `patients.postprocess.split_patient_summary`
returns the two sections of one text, and `add_summary_sections` adds both
columns to a table with `patient_summary`. Raw answers and reasoning remain
separate debug/trace data.

::: matchminer_ai.patients.postprocess
    options:
      members:
        - split_patient_summary
        - add_summary_sections

::: matchminer_ai.patients
    options:
      members:
        - compress_patient_note
        - compress_patient_notes
        - NoteCompressionError
        - concatenate_patient_note_pdfs
        - summarize_patients
        - review_patient_workup
        - review_patient_workup_with_note_search
        - WorkupSearchReviewConfig
        - answer_patient_questions
        - answer_patient_question_batch
        - NoteSearchLimits
        - NoteSearchLLMConfig

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

## Workup documentation review

```python
from matchminer_ai.patients import review_patient_workup

review = review_patient_workup(
    raw_notes,                       # single-patient note DataFrame or raw string
    matched_space["diagnostic_workup"],
    config=config,                  # patient LLM config; remote.enabled=True
    population_context=matched_space["cohort_key_criteria"],  # guideline key criteria, not patient facts
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
`evidence` (`note_number`, code-assigned `note_date`/`note_type`, verbatim `quote`).
The LLM returns only `name`, `status`, `applicability`, and `bottom_line`; it does not
generate quotes, note numbers or citation IDs. Code retains every reviewed original
note fragment in `evidence`, including across packets and backup switches. Note
numbers are one-based chronological positions. These fragments are review context,
including potentially irrelevant or conflicting passages, rather than individually
selected supporting citations. `metadata.evidence_selection` is
`automatic_reviewed_excerpts`; verbatim provenance does not validate the model's
clinical interpretation. The same distinction applies to compressed-note inputs.

Uses patient model, vendor sampling and reasoning settings (xhigh by default).
Requires one OpenAI-compatible remote endpoint with `/tokenize` and advertised
`max_model_len`, or explicit `patient.context_window`; `patient.tokenizer_mode=bytes`
is an explicit conservative fallback for providers without `/tokenize`.
The full advertised context is available, reserving the configured patient
`remote.request_params.max_tokens` (50,000 by default). Chunk size/overlap default
to patient settings (50,000/500); optional function overrides and
`recommendation_batch_size=6` bound each serial request. Oversized prompts split
note packets further, never silently truncate notes or reduce output capacity.
The model returns ordered natural-language names, not catalog IDs.

Requests and responses stay in memory; no patient checkpoints or reasoning traces
are returned or written. Three bounded attempts reject incomplete, malformed or
misaligned findings. Later silence cannot erase a previously documented finding.
Provider and validation errors are not echoed with patient text.
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

## Summary quality checks

`patients.check_patient_summary(text)` runs read-only checks on one summary. It looks for:
- self-editing language ("I should", "Wait,");
- prompt or formatting commentary;
- raw reasoning delimiters;
- repeated cancer-field values;
- repeated long prose.

It returns versioned findings with character offsets, line numbers and a `high` or `review` confidence. The
diagnostics contain no patient text. These are review signals, not clinical validation, and the checker never
rewrites a summary.
- Multiple primary cancers and repeated headings are allowed.
- Matching age, cancer type, histology and biomarkers alone asks for review.
- Matching extent and treatment history as well identifies a repeated cancer section.

`summarize_patients(..., return_qc=True)` adds aggregate flags to its QC report.

Serial summarization checks each chunk's answer before it becomes the next chunk's prior summary.
- **Regenerated:** drafting text, repeated full sections, empty output and incomplete completions. The chunk is
  regenerated from the original notes and the last accepted prior summary, up to
  `patient.summary_quality_max_attempts` (3) attempts.
- **Not regenerated:** matching only the core cancer fields, which stays a review signal.
- **Trace:** `patient.summary_quality_trace_dir` keeps every prompt, answer and reasoning trace in a local directory,
  rejected attempts included. Debug output includes `patient_summary_attempts`, and the metadata reports attempt and
  rejection counts.

## Updating existing summaries

Pass earlier summaries as `existing_summaries`, together with the patients' notes,
to continue each summary instead of starting over. Each patient's first new chunk
is summarized onto the existing summary.

The summary can be given in any of these layouts; the first one present is used:
- `patient_summary_with_boilerplate`: a complete summary;
- `cancer_history_summary` and `general_exclusion_criteria_evidence`: this
  function's own output, so a result can be passed back with new notes;
- `patient_summary` and `patient_boilerplate_text`: the training pipeline's output;
- `patient_summary` alone: a complete summary.

**`last_note_date`** (optional) is the date of the last note an existing summary
includes. Every output row has it, so the output can be passed straight back.
- When it is given (it must be readable on every row), only notes dated after it
  are summarized onto the summary. Undated notes can't be placed after it and are
  left out.
- When that last note is within 10 days of today (`RECENT_SUMMARY_LOOKBACK_DAYS`),
  notes from 10 days before it are summarized again, so notes that arrive late with
  earlier dates are not missed.
- Without `last_note_date`, every supplied note is summarized onto the summary, so
  pass only new notes.

A patient with no new usable notes keeps their existing summary unchanged. It is
normalized, but never dropped as non-informative.

```python
first = summarize_patients(notes_until_june)
update = summarize_patients(notes_until_july, existing_summaries=first)  # newer notes only
```

## Scheduling patient summarization

Each patient's chunks are summarized in order, each prompt carrying the summary
accepted for that patient's previous chunk. `patient.scheduling` chooses how
patients share the backend:

- `continuous` (the default): with a remote backend, each patient's next chunk
  is sent as soon as the previous one is final (after its quality retries), so
  the endpoints keep their full request capacity in flight and a long record
  does not hold back the rest of its batch. Waiting patients start longest
  remaining chain first. A local engine, which batches prompts itself, runs in
  rounds instead.
- `rounds`: every patient's first chunk is sent as one batch, then every
  patient's second chunk, and so on; each round waits for its slowest patient.

Both send the same prompts with the same quality checks and retries, and give
the same output rows, metadata and QC counts. See
[`patient.scheduling`](../reference/configuration.md#patientscheduling).

## Resuming patient summarization

Pass a run-specific filesystem directory as `checkpoint_dir` to save progress
during patient summarization. Each chunk's result is saved as soon as it is
final, so an interrupted run loses only the requests it had in flight. Call
`summarize_patients` again with the same directory to continue each patient
from their last saved chunk, under either scheduling.

Omit `checkpoint_dir` to run without writing checkpoint files.

### Checkpoint files

When `checkpoint_dir` is supplied, patient summarization writes these files to
that directory:

- `prepared_chunks.parquet` — the patient note chunks prepared for
  summarization, saved once before the first chunk is summarized.
- `chunk_journal.jsonl` — one JSON line per finished (patient, chunk) result,
  appended and synced to disk as each result is final (a round's results
  together, under `rounds`).

Runs before the journal wrote `round_0000.parquet`, `round_0001.parquet`, ...
instead, one Parquet file per round with every patient's result for that round.
Those files are still read, so such a directory resumes too, and a directory
with both round files and a journal resumes from both. New runs write only the
journal.

A journal line and a round file row have the following columns; the journal adds
`chunk_index`, which a round file has in its name:

| Column | Description |
| --- | --- |
| `patient_id` | Patient identifier. |
| `chunk_index` | The patient's chunk, counting from 0 (journal only). |
| `prior_summary` | The summary carried into this chunk, before it was incorporated. |
| `patient_reasoning_text` | The model's reasoning trace for this chunk. |
| `patient_answer_text` | The model's summary output for this chunk. |
| `finish_reason` | Generation finish reason, e.g. `"stop"` or `"error"`. |

On resume, round files are read in round order and then the journal in the
order it was written. A result is used only when it is the patient's next chunk
and, after the first chunk, its `prior_summary` is the summary the patient's
saved chain has reached, so every patient continues one consistent chain. A
last journal line that an interrupted write left unfinished is skipped and then
removed.

`"error"` marks a patient whose summary still failed after
`patient.summary_quality_max_attempts` attempts; that patient's later chunks are
skipped, including after resuming. Checkpoint files contain patient note
text and summaries, so store them under the same controls as the input notes.
