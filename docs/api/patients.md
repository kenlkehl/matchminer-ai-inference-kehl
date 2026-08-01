# Patients

::: matchminer_ai.patients
    options:
      members:
        - answer_question_with_raw_patient_notes
        - summarize_patients
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
