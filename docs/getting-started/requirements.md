# Requirements

What you need before running the matching workflow: software, compute, models and input data.

## Software

- **Python:** 3.12 or later, on Linux. CI tests 3.12, 3.13 and 3.14.
- **Package:** installed from a checkout; see [installation](installation.md). The core dependencies include PyTorch,
  vLLM, Transformers and sentence-transformers; their versions are in the
  [GPU recommendations](gpu-recommendations.md#software-stack).

## Compute

The workflow uses three kinds of model. Only the LLM can run on another machine.

| Model | Used by | Where it runs |
| --- | --- | --- |
| LLM | `process_trials`, `summarize_patients`, the `*_with_llm` checks | In-process with vLLM, or on any OpenAI-compatible server |
| TrialSpace | `embed_for_matching` | In-process with sentence-transformers on `embedding.device` (default `cuda`) |
| Decision model | `score_match_quality`, `exclusion_criteria_check`, `check_eligibility_criteria` | In-process with Transformers on `decision_model.device` (default `cuda`) |

- **LLM on a server:** the machine running `matchminer-ai` needs no GPU for the LLM steps. See
  [choosing an inference setup](../user-guide/local-vs-remote.md).
- **LLM in-process:** the default model, `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4`, needs a GPU with
  native FP4 (Blackwell).
- **TrialSpace and the decision model:** both are small and run on one GPU. They can run on CPU (`device: cpu`), but
  CPU speed has not been evaluated and may be impractical.
- **Retrieval:** `generate_candidate_matches` computes cosine similarities on CPU.

See the [GPU recommendations](gpu-recommendations.md) for memory sizes and the hardware we use.

!!! warning
    Patient text goes to the LLM endpoint you configure. Before sending clinical text to an endpoint, make sure it is
    approved for your data and institution.

## Models

- **TrialSpace checkpoint:** you supply it. The default preset's `embedding.model_path` names a local checkpoint that
  is not published, and embedding fails rather than fall back to another model. Set it to your own TrialSpace model
  directory, and embed patients and cohorts with the same model.
- **Decision model:** `LiquidAI/d1-2B` at a pinned revision is downloaded from Hugging Face on first use. It runs its
  own model code (`trust_remote_code`). Set `decision_model.model_name` to a fine-tuned checkpoint to use one.
- **LLM:** downloaded from Hugging Face when it runs in-process. In either setup, `summarize_patients` loads the LLM's
  tokenizer (`patient.remote.tokenizer_name` or the model name) to split notes into chunks.

## Data

The workflow starts from two tables. Later steps also accept intermediate results (summaries, embeddings, candidate
pairs); their inputs are documented on the [Package API](../api/trials.md) pages.

### Trials

`process_trials` takes one row per trial with a unique `trial_id`. With ClinicalTrials.gov listings (the default),
each row also has the listing's title, brief summary and eligibility criteria. From the ClinicalTrials.gov study
JSON:

| Column | ClinicalTrials.gov JSON field |
| --- | --- |
| `trial_id` | `protocolSection.identificationModule.nctId` |
| `trial_title` | `protocolSection.identificationModule.briefTitle` |
| `brief_summary` | `protocolSection.descriptionModule.briefSummary` |
| `eligibility_criteria` | `protocolSection.eligibilityModule.eligibilityCriteria` |

For full protocols (`source="protocol"`), each row has `trial_id` and `protocol_path`, a PDF, `.txt` or `.md` file.
PDFs are converted to text locally ([PDF document OCR](../api/documents.md)).

### Patients

`summarize_patients` takes one row per clinical note:

| Column | Content |
| --- | --- |
| `patient_id` | Patient identifier |
| `note_text` | Full note text |
| `note_date` | Note date |

It also accepts one patient's record as one or more PDF files. You are responsible for preparing note data that is
approved for your environment.

The sample files in
[`examples/data/`](https://github.com/kenlkehl/matchminer-ai-kehl/tree/main/examples/data) show both table shapes
with synthetic data. See [`process_trials`](../api/trials.md) and [`summarize_patients`](../api/patients.md) for the
complete column requirements.
