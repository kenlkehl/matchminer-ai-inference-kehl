# ColBERT workup documentation review

`matchminer_ai.patients` provides a third, opt-in workup review method alongside
full-note review and agentic note search. It locally encodes each patient's notes
into token vectors, encodes each workup question, retrieves that patient's top
chunks with exact ColBERT MaxSim (sum of each query token's maximum cosine
similarity), then asks the configured patient LLM for one structured assessment
per question. It does not change patient summarization or trial matching.

The default is [LightOn GTE-ModernColBERT-v1](https://huggingface.co/lightonai/GTE-ModernColBERT-v1)
(Apache-2.0), with **64-token chunks**, **zero overlap**, **32 chunks per encoding
batch**, and **20 retrieved chunks per question**. Chunk boundaries use the
retrieval model's tokenizer. Chunks never cross note or patient boundaries;
DataFrame notes are stably ordered by date. Every retained excerpt has exact
original character offsets and a code-derived note date/type. Free text has no
structured date provenance.

## Encode once, review repeatedly

```python
import pandas as pd
from matchminer_ai import load_default_preset
from matchminer_ai.patients import (
    ColBERTConfig,
    encode_patient_notes_colbert,
    review_patient_workup_with_colbert,
)

# Minimal fabricated examples. Supply authorized records in actual use.
notes = pd.DataFrame([
    {"patient_id": "example-a", "note_date": "2026-01-01",
     "note_type": "Imaging", "note_text": "Chest CT completed."},
    {"patient_id": "example-b", "note_date": None,
     "note_text": "Imaging is planned."},
])
retrieval = ColBERTConfig(device="cpu", chunk_size=64, batch_size=32)
indexes = encode_patient_notes_colbert(notes, config=retrieval,
                                       progress_callback=print)
# Alternatively supply {patient_id: text_or_single_patient_dataframe}.
# Encoding batches span the cohort; each result is a separate patient index.

llm = load_default_preset()
llm.remote.update(enabled=True, server_urls=["http://your-authorized-server:8000/v1"])
llm.patient["remote"].update(model_name="your-served-model")
result = review_patient_workup_with_colbert(
    indexes["example-a"], [{"name": "Chest CT", "conditions": "If indicated"}],
    config=llm, top_n=20, population_context="A supplied guideline population",
    progress_callback=print,
)
```

Use `retrieve_patient_chunks_colbert(index, questions, top_n=20)` to retrieve
chunks without calling an LLM. Questions are encoded in batches. Each question
returns a ranked list with text, note number/date/type, character `start`/`end`,
chunk index, rank, and score. Retrieval is restricted to the supplied patient's
index and ties retain source order. Changing top N or the answering LLM reuses
the patient vectors.

The query combines each recommendation's name and conditions; the guideline
population is included in the LLM assessment prompt. The default query capacity
is 512 tokens including special tokens. Oversized queries raise a clear error
instead of being truncated; increase `query_length` and re-encode if needed.
Retrieved fragments are presented chronologically to the LLM with their source
dates. Prompts exceeding the LLM's input capacity produce an explicit item error;
no chunks are silently dropped and the configured output reserve is preserved.

## Optional explicit disk save/load

The encoding/review functions write no patient files. The package separately
supports caller-controlled storage for offline or cohort workflows:

```python
from dataclasses import replace
from matchminer_ai.patients import save_colbert_patient_index, load_colbert_patient_index

save_colbert_patient_index(indexes["example-a"], "/authorized/private/example-a.npz")
loaded = load_colbert_patient_index(
    "/authorized/private/example-a.npz",
    patient_id="example-a",
    expected_notes=notes[notes.patient_id == "example-a"],
    expected_config=retrieval,
)
# The loader does not load the encoder or re-encode notes.
# To query on a different device, use retrieval_config=replace(loaded.config, device="cpu")
# with review_patient_workup_with_colbert, or config=... with the retrieval API.
```

Files contain sensitive original text, provenance, and token vectors. The caller
must choose an appropriate location and retention policy. Saves are atomic with
owner-only file permissions; NPZ/JSON loads disable pickle. Source fingerprints,
vector checksums, and span validation detect stale or corrupted indexes.
At retrieval, the loaded model/configuration fingerprint must match the index.
Changing model weights, tokenizer, chunking, query capacity, or encoder version
requires re-encoding. Batch size and device can change without invalidating it.
The demo application does **not** invoke these persistence APIs.

## Model loading and compatibility

The loader uses the existing PyTorch, Transformers, Hugging Face Hub, and
safetensors stack; PyLate is not a runtime dependency. Published PyLate releases
currently pin dependency versions older than this package's requirements.
The encoder follows the model's [PyLate contract](https://github.com/lightonai/pylate):
trained query/document marker tokens, learned token projection, punctuation and
padding masks, query expansion settings, and unit-normalized token vectors.
It retains token vectors instead of pooling to a single embedding.

`ColBERTConfig(model_name=..., revision=...)` supports compatible saved
Transformer plus `1_Dense` identity-activation projection checkpoints, including
local safetensors folders. Arbitrary Sentence Transformer or original Stanford
checkpoint layouts are rejected. Remote custom code is not enabled. Model files
may be downloaded to the normal Hugging Face cache, but no patient text is sent
for downloading or encoding. Pin `revision` for reproducible runs. Results retain
model revision, artifact checksum, encoder settings, and prompt checksum.

## Assessment scope, failures and backups

Statuses and applicability use the same contract as the other workup reviewers.
An absent finding means **not located in the retrieved excerpts**, not that care
was omitted or the complete record was reviewed. Results preserve every retrieved
fragment that was sent for review as code-owned review context. These are not
individually selected supporting citations or model-written quotations. Patient
summaries are not used as evidence, and no agent actions or web search occur.

Pass an explicit `backup_config` and `max_consecutive_failures` (default 3, range
1–10) to allow a single switch after bounded request/validation failures. Unknown
clinical findings do not trigger fallback. The failed and remaining questions
use the backup's own sampling/reasoning settings with the same retrieved chunks.
Unanswered items have `status="error"`; failures never become missing-documentation
findings. Backup events and failed backup preparation are visible in metadata.
Cancellation scopes stop further encoding batches, retrieval, and LLM requests.

Retrieved patient excerpts reach the configured LLM and, if enabled and needed,
its explicit backup. Use endpoints authorized for the input's sensitivity. This
is a research documentation-review workflow requiring human review, not an
overall guideline-concordance determination, diagnosis, or treatment advice.
