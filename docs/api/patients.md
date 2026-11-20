# Patients

`summarize_patients` turns each patient's longitudinal notes into one cancer-history summary, which TrialSpace and
the decision model compare with cohorts ([How matching works](../user-guide/patient-centric-matching.md)). The same
module has helpers for patient PDFs, note compression and workup documentation review. Structured, evidence-linked
records from the same notes are described in [Structured Clinical Extraction](extraction.md).

## Serial summarization

```python
import pandas as pd
from matchminer_ai import load_preset
from matchminer_ai.patients import summarize_patients

config = load_preset("default")
notes = pd.DataFrame([
    {"patient_id": "P001", "note_date": "2025-01-05", "note_text": "Fabricated note text ..."},
    {"patient_id": "P001", "note_date": "2025-03-12", "note_text": "Fabricated note text ..."},
])
patients, qc = summarize_patients(notes, config=config, checkpoint_dir="runs/patients", return_qc=True)
```

The input has one row per note with `patient_id`, `note_text` and `note_date`. Each patient's record is prepared,
then summarized chunk by chunk:

1. **Order:** notes are sorted by date, undated notes last; notes without text are dropped.
2. **Repeated sentences:** a sentence repeated across the patient's notes is kept only where it first appears.
3. **Chunks:** the notes are joined under `=== Clinical Note dated <date> ===` headers and split into overlapping
   token chunks (`patient.chunk_size` 50,000 and `patient.chunk_overlap` 500, counted with the patient model's
   tokenizer).
4. **Updates:** each chunk's prompt (`patient.serial.user.primer.txt`) carries the summary accepted for the
   previous chunk, and its answer becomes the summary the next chunk updates.
5. **Sections:** the final summary is split into its key and boilerplate sections.

The notes reach the configured LLM endpoint, so use only an endpoint authorized for them.

## Output

One row per patient:

| Column | Content | Read by |
| --- | --- | --- |
| `patient_id` | The patient. | |
| `last_note_date` | Date (YYYY-MM-DD) of the latest dated note the summary includes. | [Updates](#updating-existing-summaries) |
| `patient_summary` | The complete summary. | The per-criterion check, serial updates, summary-quality checks |
| `cancer_history_summary` | The **key** section: the text before `Other relevant history:`. | TrialSpace retrieval, the decision model's match questions |
| `general_exclusion_criteria_evidence` | The **boilerplate** section: the text after that heading (`Not documented.` when absent). | The decision model's boilerplate question |

The complete summary holds the cancer-history fields (age, sex, cancer type, histology, current extent, biomarkers,
treatment history; one block per active cancer) and ends with the boilerplate section:

```text
Other relevant history:
ECOG 1. Chronic kidney disease. Remote inactive prostate cancer.
```

- **Treatment history** lists anti-cancer treatment only, one line per regimen (dates, agents, cycles, best
  response). Each update carries every earlier regimen forward.
- **Other relevant history** records, with dates, the evidence for the general criteria most trials share:
  performance status, measurable disease, brain and other CNS metastases and their treatment, the most recent
  routine labs (blood counts, coagulation, kidney and liver function), ejection fraction and QTc, toxicities of
  earlier cancer treatment, conditions other than cancer (including transplants and drug allergies), recent major
  surgery other than cancer surgery, other and inactive cancers, participation in any clinical trial, and hospice
  enrollment or a decision to stop anti-cancer therapy.
- **Left out:** a patient whose summary is empty or starts with "No information", or still fails the
  [quality checks](#summary-quality-checks), has no row. The QC report (`return_qc=True`) counts them
  (`patients_dropped_noninformative_summary`, `patients_failed_inference`); see
  [QC metrics](../reference/qc-metrics.md).
- **Debug columns:** with `debug_mode`, `patient_answer_text` and `patient_reasoning_text` (the last chunk's answer
  and reasoning) and `patient_summary_attempts` (every attempt's trace).
- **Metadata:** `return_metadata=True` adds the package version, config snapshot and model metadata.

A `Boilerplate:` heading in a summary is read as `Other relevant history:`. `embed_for_matching`, the decision-model
checks and their LLM versions also accept a complete `patient_summary` in place of a section column and take the
section they need. `patients.postprocess.split_patient_summary` returns the two sections of one text, and
`add_summary_sections` adds both columns to a table with `patient_summary`.

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

`summarize_patients(..., return_qc=True)` adds aggregate flags to its QC report (`patients_summary_quality_flagged`,
overall and per finding).

Serial summarization checks each chunk's answer before it becomes the next chunk's prior summary.

- **Regenerated:** every finding except matching core cancer fields (drafting or commentary text, reasoning
  delimiters, repeated sections or prose), plus empty output and incomplete completions. The chunk is regenerated
  from the original notes and the last accepted prior summary, up to `patient.summary_quality_max_attempts` (3)
  attempts.
- **Not regenerated:** matching only the core cancer fields, which stays a review signal.
- **Trace:** `patient.summary_quality_trace_dir` keeps every prompt, answer and reasoning trace in a local directory,
  rejected attempts included. Debug output includes `patient_summary_attempts`, and the metadata
  (`return_metadata=True`) reports attempt and rejection counts (`summary_quality`).

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

Pass a run-specific directory as `checkpoint_dir` to save progress. Each chunk's result is saved as soon as it is
final, so an interrupted run loses only the requests it had in flight. Call `summarize_patients` again with the same
directory to continue each patient from their last saved chunk, under either scheduling. Without `checkpoint_dir`,
nothing is written.

The directory holds:

- `prepared_chunks.parquet`: the patients' note chunks, saved once before the first chunk is summarized;
- `chunk_journal.jsonl`: one JSON line per finished (patient, chunk) result, appended and synced to disk as each
  result is final (a round's results together, under `rounds`).

| Column | Description |
| --- | --- |
| `patient_id` | Patient identifier. |
| `chunk_index` | The patient's chunk, counting from 0. |
| `prior_summary` | The summary carried into this chunk, before it was incorporated. |
| `patient_reasoning_text` | The model's reasoning trace for this chunk. |
| `patient_answer_text` | The model's summary output for this chunk. |
| `finish_reason` | Generation finish reason, e.g. `"stop"` or `"error"`. |

On resume, a saved result is used only when it is the patient's next chunk and, after the first chunk, its
`prior_summary` is the summary the patient's saved chain has reached, so every patient continues one consistent
chain. A last journal line that an interrupted write left unfinished is skipped and removed. Older directories with
one Parquet file per round (`round_0000.parquet`, ...) resume too.

`"error"` marks a patient whose summary still failed after `patient.summary_quality_max_attempts` attempts; that
patient's later chunks are skipped, including after resuming. Checkpoint files contain patient note text and
summaries, so store them under the same controls as the input notes.

### Inspecting a run

`patients.serial_trace(checkpoint_dir)` reads a checkpoint directory the way resuming does and returns one row per
prepared chunk, in each patient's chain order: `patient_id`, `chunk_index`, `first_date`, `last_date` and, where the
chunk has a result, `prior_summary`, `reasoning`, `summary` and `finish_reason`. Nothing is sent to a model.

## PDF patient records

`summarize_patients` also accepts one local PDF path, or an ordered sequence of PDF paths, for a single patient.
Each page uses embedded text when it has enough, otherwise local RapidOCR (as in [`ocr_pdf`](documents.md)). The
documents are combined into one long note, which is then summarized like any other record.

```python
summary = summarize_patients(
    ["record-part-1.pdf", "record-part-2.pdf"],
    patient_id="P001",
    config=config,
)
```

To inspect or edit the extracted text first, call the preparation step directly:

```python
from matchminer_ai.patients import concatenate_patient_note_pdfs

long_note = concatenate_patient_note_pdfs(["record-part-1.pdf", "record-part-2.pdf"])
```

- **Order and boundaries:** documents keep the order given, with numbered boundary markers. Filenames and file
  timestamps are not copied into the text, and no clinical dates are inferred: the combined note is undated.
- **Accuracy:** OCR is probabilistic; compare important content with the original PDFs.
- **Privacy:** OCR runs locally, but summarization sends the extracted text to the configured LLM endpoint.

## Note compression

`compress_patient_notes` rewrites each nonblank `note_text` row in one independent request to the configured
remote patient LLM, and `compress_patient_note` does the same for one string.

```python
from matchminer_ai.patients import compress_patient_note, compress_patient_notes

config.remote["enabled"] = True
config.remote["server_urls"] = ["http://localhost:8000/v1"]
config.patient["remote"]["model_name"] = "my-endpoint-model"

compressed = compress_patient_notes(
    notes,
    config=config,
    max_concurrent_requests=8,
    progress_callback=lambda completed, total: print(completed, total),
)
one_note = compress_patient_note("Fabricated note text.", config=config)
```

- **Input and output:** every run of whitespace becomes one space before sending, and only that note's text is
  sent. The returned copy keeps every input row, column and index unchanged and adds `compressed_note_text`. Blank
  strings give `""` without a request; null or non-string text and an existing `compressed_note_text` column are
  rejected.
- **Prompt:** `patient.note_compression.system.txt` asks for extreme but lossless token density and only the
  compressed note. Blank, refused, token-limited, malformed and explanatory answers are retried, then raise; a
  failure never returns a partial table. Oversized notes are rejected, never truncated or split.
- **Endpoint:** exactly one remote server URL, so model discovery and token counting describe the server that
  answers. Requests share the endpoint cap, authentication, sampling, timeout, retries and cancellation;
  `max_concurrent_requests` defaults to `remote.max_concurrent_requests`. Set `remote.context_window` when the
  endpoint does not advertise one, and `remote.tokenizer_mode = "bytes"` for servers without vLLM's `/tokenize`.
- **Reasoning:** `thinking="off"` (the default) sends `chat_template_kwargs.enable_thinking=false` without changing
  the shared configuration. An adapter that cannot disable reasoning keeps its settings, and an endpoint that
  rejects the switch with HTTP 400 or 422 is retried once without it. `thinking="on"` opts in.
  `attrs["note_compression"]` records `thinking_requested` and the `thinking` actually dispatched (`off`, `on`,
  `default` when no switch was sent, `mixed`, or `not_requested` for all-blank input).
- **Privacy:** reasoning is discarded, and no notes, prompts or outputs are logged or written to disk.

Compression is experimental: asking for losslessness does not establish it. Keep the original notes for human review
and exact quotations. Summarization and note-search question answering do not use compressed notes.

## Workup documentation review

```python
from matchminer_ai.patients import review_patient_workup

review = review_patient_workup(
    patient_notes,                   # one patient's note DataFrame or concatenated text
    space["diagnostic_workup"],      # a guideline catalog record's workup items
    config=config,                   # patient LLM, remote.enabled=True
    population_context=space["cohort_key_criteria"],  # the guideline population, not patient facts
    progress_callback=print,
)
```

This opt-in workflow reviews **all** of one patient's notes serially, carrying forward a JSON assessment of every
workup item. It does not use the patient summary or any retrieval as evidence.

- **Input:** a DataFrame with nonempty `note_text`, optional `note_date` and `note_type`, and an optional
  `patient_id` with a single value; or a string, which has no date provenance. Dated notes are sorted, undated notes
  follow them, and repeated sentences are kept so dated evidence and contradictions are not lost.
- **Result:** `assessments`, `notice` and `metadata`. Each assessment keeps the original recommendation and its
  zero-based `recommendation_index`, plus `status` (`completed`, `partially_completed`, `planned`, `not_done`,
  `not_documented`, `unclear`), `applicability` (`applies`, `not_applicable`, `uncertain`), `bottom_line` and
  `evidence`.
- **Evidence:** the model returns only `name`, `status`, `applicability` and `bottom_line`, never quotes or note
  numbers. Code attaches every reviewed note fragment verbatim (`note_number`, one-based in chronological order;
  `note_date`; `note_type`; `quote`), including irrelevant or conflicting passages. They are review context, not
  selected citations (`metadata.evidence_selection` is `automatic_reviewed_excerpts`), and verbatim provenance does
  not validate the model's interpretation.
- **Requests:** the patient model, sampling and reasoning settings (xhigh by default) on exactly one remote endpoint
  with `/tokenize` and an advertised `max_model_len`, or an explicit `patient.context_window` (with
  `patient.tokenizer_mode: bytes` for servers without `/tokenize`). The configured output reserve
  (`patient.remote.request_params.max_tokens`, 50,000) is kept. Chunks follow `patient.chunk_size` and
  `chunk_overlap` unless overridden, with at most `recommendation_batch_size` (6) items per request. Oversized
  prompts split note packets further and never truncate notes.
- **Failures:** three bounded attempts reject incomplete, malformed or misaligned findings, and a later chunk's
  silence cannot erase a documented finding. Errors are not echoed with patient text. Nothing is written to disk,
  and no reasoning is returned.

The configured endpoint receives the raw notes and must be authorized for them.

**Backup model.** `review_patient_workup` and `review_patient_workup_with_note_search` accept `backup_config` (a
separate `MMAIConfig` with the backup's endpoint, model, sampling and reasoning) and `max_consecutive_failures`
(default 3, from 1 to 10). Nothing changes on the successful path.

- **Full-note review** switches once after that many failed attempts on a packet, retries the packet within the
  backup's context and keeps every validated assessment; the remaining packets use the backup. Metadata reports
  `primary_model`, `backup_used`, `backup_events`, request attempts and `validated_packets`.
- **Note-search review** restarts only the failed item, in a fresh isolated REPL with the same budget, after
  repeated Python errors or failed attempts. Follow-up can switch once within its call budget; a failed backup never
  triggers another switch. Per-item metadata records each switch, the models, the reasons and the combined costs.
  Backups keep worker isolation, cancellation, output and context reserves, and exact-evidence validation.

This is a documentation review for human review, **not an overall guideline concordance determination**. Not
documented does not mean not done. Conditional indications, compound items, alternatives, historical tests, timing
and conflicting evidence need clinical review. Treatment concordance is not assessed.

## Note-search question answering

`answer_patient_questions`, `answer_patient_question_batch` and `review_patient_workup_with_note_search` answer
questions and review workup items by searching the original notes in an isolated Python REPL instead of reading
every chunk. See [note-search question answering](../user-guide/note-search-qa.md).

## API reference

::: matchminer_ai.patients
    options:
      members:
        - summarize_patients
        - check_patient_summary
        - serial_trace
        - concatenate_patient_note_pdfs
        - compress_patient_note
        - compress_patient_notes
        - NoteCompressionError
        - review_patient_workup
        - review_patient_workup_with_note_search
        - WorkupSearchReviewConfig
        - answer_patient_questions
        - answer_patient_question_batch
        - NoteSearchLimits
        - NoteSearchLLMConfig

::: matchminer_ai.patients.postprocess
    options:
      members:
        - split_patient_summary
        - add_summary_sections
