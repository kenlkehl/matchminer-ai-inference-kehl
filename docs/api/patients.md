# Patients

::: matchminer_ai.patients
    options:
      members:
        - answer_question_with_raw_patient_notes
        - full_patient_screen
        - summarize_patients
        - structure_patient_summaries
        - structure_patient_summary

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
one-patient DataFrame as raw-note question answering plus the complete
eligibility-criteria text for one trial. An LLM first converts every
independently assessable protocol requirement into a focused raw-note question.
Each exact non-empty criteria line receives a code-assigned source ID before the
LLM call. The LLM references that ID, and the package attaches the original text
after validation, so harmless model rewriting cannot break source grounding.
The package then calls `answer_question_with_raw_patient_notes` for every
question and retains each grounded answer, exact-quote evidence item, limitation,
and code-derived `note_date`.

With the remote/OpenAI-compatible backend enabled and raw-note embeddings on
CPU, independent questions run in spawned CPU processes. This lets the bounded
question agents issue simultaneous requests to a vLLM server or another
configured endpoint. The configured worker count is capped by the question
count and available CPUs. For an in-process local vLLM backend, set
`max_workers=1`; the package will not duplicate a local GPU engine across
workers.

```python
from matchminer_ai.patients import full_patient_screen

screen = full_patient_screen(
    notes_dataframe,
    complete_eligibility_criteria,
    config=config,
    max_workers=4,
)
```

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
