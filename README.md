# matchminer-ai

## Overview

LLM stages with `sampling_profile: auto` recognize Qwen 3.5 model names, including
`Qwen/Qwen3.5-4B`, and apply Alibaba's
[general-task sampling defaults](https://huggingface.co/Qwen/Qwen3.5-4B#best-practices).
With `chat_template_kwargs.enable_thinking=false`, temperature is 0.7 and top-p 0.8;
with thinking enabled, they are 1.0 and 0.95. Both modes use top-k 20, min-p 0,
presence penalty 1.5 and repetition penalty 1.0. Qwen 3.5 uses a boolean thinking
switch rather than Qwen 3.8's graded effort; no graded effort is inferred for it.
Explicit caller sampling parameters still override these defaults. An opaque served
alias can use `sampling_profile: qwen3.5` to select the same profile explicitly.

`matchminer-ai` is a Python package for running the clinical trial matching inference workflow described in [Altreuter et al., MatchMiner-AI: An Open-Source Solution for Cancer Clinical Trial Matching](https://doi.org/10.48550/arXiv.2412.17228). The package provides modular functions for the core MatchMiner-AI workflow: summarizing trials and patient histories, generating embeddings of each, retrieving candidate matches, scoring match quality, and assessing exclusion criteria.

For trial-centric matching, `find_trial_centric_cutoff` can start from a
complete TrialSpace-ranked patient list and use cached, boundary-centered
TrialChecker or LLM probes to estimate how many leading patients should proceed
to both full checker stages. Its default pass thresholds are 0.20 on
TrialChecker's 0-1 sigmoid scale and 1 on the LLM checker's 0-5 scale. This is a
compute-allocation heuristic, not an eligibility threshold.
The first probe defaults to 50% down the ranking and can be moved with
`initial_cutoff_proportion` when the expected qualifying fraction is known to
be much smaller or larger.
For offline QA, `assess_trial_centric_cutoff_stability` scores a complete
authorized ranking once with TrialChecker and replays the cutoff search from
10%, 25%, 50%, 75%, and 90%. It reports the spread in selected cutoffs and in
threshold-passing patients retained by those cutoffs. The full-corpus scoring
cost makes this a stability diagnostic rather than the production allocation
path.

For a specific TrialChecker or BoilerplateChecker prediction,
`interpret_match_quality` and `interpret_exclusion_criteria` provide on-demand
gradient-times-input token attribution mapped back to the original patient and
trial fields. These local sensitivity scores are intended for model debugging
and human review. They are not a clinical rationale, eligibility evidence, or
proof that a highlighted token caused the prediction.

Optional source-grounded extensions can also:

- extract text from local PDFs with page-aware embedded-text preservation and
  OCR fallback, producing a UTF-8 text file without sending document content to
  an external service;
- accept one or more patient-record PDFs, combine their locally extracted text
  in caller-supplied order, and pass that long note through the existing serial
  patient summarization workflow;
- extract complete, source-grounded inclusion and exclusion criteria relevant
  to one trial space from an OCR eligibility-checklist text file using the
  configured LLM backend;
- answer a focused question over one patient's raw notes through local
  embedding retrieval and a bounded, evidence-citing LLM agent, with one
  code-derived source date per chunk when note-level dated input is available;
- screen complete trial eligibility criteria by decomposing them into grounded
  raw-note questions, embedding the note index once on CUDA, reusing its CPU
  vectors across concurrent question processes against an authorized endpoint,
  and synthesizing a human-reviewable JSON result;
- transform one or a batch of free-text cancer histories into JSON with one
  record per active cancer, dependency-ready LLM batching, hierarchical
  OncoTree coding, and locally searched NCIt drug normalization;
- transform a clinical-space summary into JSON while preserving age, sex,
  disease burden, treatment, response, and biomarker requirements;
- roll a patient summary's TrialSpace/TrialChecker-ranked spaces up through an
  exact caller-supplied space-to-paradigm membership graph, while preserving
  one-to-many mappings and paradigm caveat metadata;
- build a versioned patient-free evidence catalog from a list of NCT IDs,
  LLM-screen registry entries for concrete agents with direct anticancer
  treatment intent, deduplicate NCIt-normalized drugs across trials, synthesize
  mechanism, efficacy, biomarker prevalence, biomarker-directed benefit, and
  safety, then resume compatible atomic screening, research, and synthesis
  checkpoints, and
  estimate patient-specific option quality with a four-point-per-drug LLM
  rubric over drug and class evidence packed to the teacher's context (the
  four-logit GoodOptionChecker is deprecated); and
- contextualize a trial space using heading-aware NCI PDQ, FDA
  companion-diagnostic and DailyMed material, accepted CIViC evidence, focused
  PubMed searches, and permissively licensed Europe PMC guideline/consensus
  full text.

Trial-space retrieval rejects patient-bearing columns. Patient personalization
is a separate API that sends patient context only to the configured LLM
backend. These extensions produce research considerations, not treatment
recommendations, guideline compliance, or eligibility determinations.

Good Option APIs follow the same stage boundaries as the other matching
components: import `build_good_option_catalog`, `load_good_option_catalog`, and
`validate_good_option_catalog` from `matchminer_ai.trials`, and import
`check_good_options` (the on-demand check, configured by `good_option_check`),
`score_good_options_with_llm`, or `evaluate_good_options` from
`matchminer_ai.matching` (the classifier `score_good_options` is deprecated). Screening, synthesis, scoring, retry, and checker
input templates live in `src/matchminer_ai/prompts/`. The former
`matchminer_ai.good_options` imports remain compatibility aliases. This source
refactor preserves prompt text, score semantics, and existing artifact and
checkpoint compatibility. Scoring prompts now re-render drug and class
evidence from stored facts at a trial-sized budget (`good_option_prompt`);
the template, rubric, and catalog compatibility are unchanged.

The optional space-paradigm roll-up does not bundle a canonical paradigm
catalog. It consumes caller-supplied trial spaces, exact membership edges, and
one-line paradigm descriptors. Its ranking is a transparent roll-up of matching
signals, not a new eligibility or treatment score.

For detailed instructions, please see the
[documentation website](https://dfci.github.io/matchminer-ai-inference/).

The opt-in `matchminer_ai.trials.summarize_guidelines` API extracts canonical
TrialSpace populations, diagnostic workup, and treatment menus from a locally
supplied guideline library using a configured remote LLM. See the
[guideline extraction guide](docs/user-guide/guideline-extraction.md).
The package contains conversion/extraction code and synthetic tests only; it
does not include NCCN PDFs, guideline text, or generated catalogs. Source-derived
outputs and checkpoints must be written outside the code repository.
Existing catalogs can also be queried by exact space using
`trials.get_guideline_considerations`, or ranked for patient summaries using
`matching.retrieve_guideline_considerations` (TrialSpace retrieval followed by
TrialChecker ranking). These APIs return the stored diagnostic and treatment
menus with their conditions, citations, and uncertainties.
Pass `embedding_cache_dir` to the retrieval API to persist and reuse compatible
guideline embeddings across requests and restarts; patient vectors are not cached.
Validated guideline files also reuse an in-memory cache until their catalog,
completion status, or audit files change, avoiding repeated reads and validation.

`patients.review_patient_workup` reviews a retrieved diagnostic/workup list against
one patient's raw notes in chronological token chunks. It returns JSON-compatible
per-item statuses, applicability, bottom lines and validated verbatim note evidence.
This opt-in documentation review uses the patient remote LLM configuration, not
the patient summary, and writes no patient checkpoints. See the
[patient API guide](docs/api/patients.md#workup-documentation-review).

`patients.compress_patient_notes` is an opt-in note-compression module using the
configured remote patient LLM. It normalizes whitespace runs to one space,
compresses independent notes concurrently, and adds `compressed_note_text` to a
copy of the input table while preserving original notes, metadata and row order.
`compress_patient_note` returns only compressed text for a single string. The
short prompt requests extreme but lossless token density and forbids explanatory
text; it contains no preservation checklist. Reasoning defaults to off through
the endpoint request argument where supported. Otherwise the switch is omitted
and the endpoint's configured reasoning settings are retained.
`thinking="on"` explicitly enables it for compression.
This experimental compression does not certify losslessness or change serial
summarization or note-search defaults.
See [note compression](docs/api/patients.md#note-compression).

For an experimental alternative that explores selected excerpts,
`patients.answer_patient_questions` accepts a notes DataFrame, a full-text history,
or both, arbitrary questions, an optional existing patient summary and an explicit
LLM endpoint configuration. DataFrame columns are `note_text`, optional `note_date`
and `note_type`; the DataFrame is authoritative when both forms exist. A persistent
isolated Python REPL navigates the actual pandas table and searches the local text
while the model carries a compact evidence
notebook. `answer_patient_question_batch` adds bounded concurrency across patients
and within each patient's questions. Both automatically retain original excerpts
shown to the model, with exact character spans; the model supplies no citation IDs
or copied quotes. These are review context, not selected supporting citations.
The `scan` helper accepts up to 128 patterns per call by default, configurable
with `NoteSearchLimits(max_scan_patterns=...)`, while keeping results bounded.
Pandas selections automatically retain bounded original excerpts with dates and
types; summaries guide navigation but do not supply evidence. Defaults allow twelve
Python cells and sixteen initial request attempts per question, including retries.
Both return answered/unknown/error statuses and cost metadata. The optional
`review_patient_workup_with_note_search` adapter uses this harness for workup
items, then checks retrieval coverage with shared patient-free vocabulary,
alternative wording after zero-match searches, and up to two focused review
passes. Unresolved or incompletely retrieved items can use bounded serial
full-record fallback (at most five items). `WorkupSearchReviewConfig` controls
these follow-ups, including a separate twelve-call budget per item; set
`max_review_passes=0` to disable them. Source quotes and dates remain code-owned.
Full-note review and patient summarization remain available. See the
[note-search REPL guide](docs/user-guide/note-search-qa.md).

`patients.encode_patient_notes_colbert` batches token-level encoding across a
cohort and returns separate patient indexes. `review_patient_workup_with_colbert`
adds an opt-in third workup reviewer: GTE-ModernColBERT-v1 retrieves each question's
top 20 patient chunks (64 tokens each by default), then the configured LLM answers
from those excerpts. Indexes remain in memory unless callers explicitly use the
package's save/load APIs. Source dates/spans, model compatibility, and explicit
answer-model backups are retained. This reviews retrieved excerpts and may miss
relevant documentation. See the [ColBERT workup guide](docs/user-guide/colbert-workup.md).

Full-note `review_patient_workup` also asks for findings without model-written
quotes or citation IDs. It retains every reviewed original note fragment and its
date/type in code, including across serial updates and backup switches. Both
workup methods label these excerpts as automatically retained review context,
not individually selected supporting citations. Full-note checks still visit all
supplied chunks and validate item order, findings, and longitudinal continuity.

Both workup APIs accept an optional explicit `backup_config` and
`max_consecutive_failures=3` (1–10). Without a backup their normal behavior is
unchanged. The backup is resolved lazily after repeated request/validation failures;
agent search also switches after consecutive Python errors, resetting the streak
after a successful cell. Unknown clinical findings do not trigger a backup.
Full-note review retains validated state and switches once for the failed and
remaining packets, repacking for the backup's context budget. Agent search retries
only a failed item with a fresh isolated REPL and the same bounded per-attempt
budget. Follow-up uses that model and can independently switch after repeated
failures within its existing call budget, retaining validated findings. Backups use
their own sampling/reasoning configuration and the same selected notes; no primary
endpoint settings or reasoning traces are replayed. Results record safe backup
provenance and combined per-item costs. No backup model is chosen implicitly.

Structured note/workup requests can pace starts without reducing question-worker
concurrency. Configure `request_start_interval_seconds`,
`capacity_retry_initial_seconds` and `capacity_retry_max_seconds` on
`NoteSearchLLMConfig` or the shared remote runtime. Defaults leave pacing disabled.
Clients share dispatch spacing and capacity cooldowns by endpoint/model within
one Python process; different processes need their own coordination. HTTP 429/503
retries use randomized exponential backoff when enabled. Numeric or HTTP-date
`Retry-After` delays are applied up to the configured ceiling. Capacity failures
preserve the original request, count toward existing attempt/call limits and do
not add model-validation feedback. Preparation/tokenizer routes use their
separate pool. No patient text, credentials or provider error bodies enter pacing
state; accepted-request metadata records dispatch wait time.

Patient applications can opt into cooperative cancellation using
`CancellationToken` and `cancellation_scope` from `matchminer_ai.cancellation`.
Wrap the patient workflow in the scope and call `token.cancel()` from the Stop
handler. Question and coverage thread pools inherit the scope; endpoint-capacity
and cooldown waits, retry delays and Python cells check it. Cancellation raises
`InferenceCancelled`, a control-flow `BaseException`, rather than returning a
clinical finding or retrying it as model failure. Isolated REPL processes close
on cancellation. Blocking HTTP calls stop being awaited promptly; their transport
thread retains the concurrency slot until the response closes or times out.
Requests already accepted by the remote model may continue there, with late
responses discarded locally. Remote async summarization cancels its pending tasks
and closes clients; native local model operations finish their current operation.
Without a scope, existing workflow behavior is unchanged.

## Ontology attribution

The optional structured patient-summary and trial-space workflows bundle and
use the following ontology snapshots locally:

- **OncoTree**, developed at Memorial Sloan Kettering Cancer Center, stable
  hierarchy snapshot downloaded 2026-07-31. OncoTree is licensed under the
  [Creative Commons Attribution 4.0 International License](https://github.com/cBioPortal/oncotree/blob/master/LICENSE.md).
  Project and source: [cBioPortal/oncotree](https://github.com/cBioPortal/oncotree).
- **NCI Thesaurus (NCIt) 26.07d**, produced by the National Cancer Institute
  Enterprise Vocabulary Services group, Center for Biomedical Informatics and
  Information Technology, National Cancer Institute, Maryland, USA. NCIt is
  licensed under the
  [Creative Commons Attribution 4.0 International License](https://evs.nci.nih.gov/ftp1/NCI_Thesaurus/ThesaurusTermsofUse.pdf).
  Source and terms of use:
  [NCI Enterprise Vocabulary Services](https://evs.nci.nih.gov/ftp1/NCI_Thesaurus/).

OncoTree codes and NCIt records remain attributable to their respective
creators. The bundled snapshots are unmodified; MatchMiner-AI adds local search,
LLM-guided selection, and output formatting around them. NCI Thesaurus is a
trademark of the National Cancer Institute. MatchMiner-AI does not imply
endorsement by MSK or NCI.

> [!WARNING]
> This package is currently pre-v1 and under active development.

## Compute requirements

A GPU is recommended for the clinical trial matching inference workflow. Please see the
[requirements documentation](https://dfci.github.io/matchminer-ai-inference/getting-started/requirements/)
for more information on compute expectations and GPU recommendations.

## Installation

This package requires Python 3.12+.

The package has been tested in Linux environments.

We recommend using [`uv`](https://docs.astral.sh/uv/) to create the Python
environment and install the package:

```shell
uv venv --python 3.12
source .venv/bin/activate
uv pip install matchminer-ai
```

## Quickstart
See the example notebook for a full walkthrough using sample input data:
[example notebook](https://github.com/dfci/matchminer-ai-inference/blob/main/examples/run_examples.ipynb)

### Terminal LLM trial check

After installing the package, point the interactive checker at a running vLLM
OpenAI-compatible endpoint:

```shell
matchminer-ai-llm-trial-check http://localhost:8000/v1
```

From a source checkout, the equivalent direct script invocation is:

```shell
python llm_trial_check.py http://localhost:8000/v1
```

The command discovers the endpoint's model through `/v1/models`, then prompts
for a multiline patient summary and trial space. End each paste with a line
containing only `.done`. It prints the endpoint's complete separate reasoning
trace, complete final answer, and the parsed 0-5 MatchMiner-AI score. If the
server requires an API key, set `OPENAI_API_KEY` before running the command.
The patient summary is sent to the specified endpoint, so use only an endpoint
authorized for the sensitivity of the input data.

## Citation

If you use `matchminer-ai`, please cite:
>Altreuter J, Trukhanov P, Paul MA, Hassett MJ, Riaz IB, Afzal MU, Mohammed AA, Sammons S, Lindsay J, Mallaber E, Klein HR, Gungor G, Galvin M, Deletto M, Van Nostrand SC, Provencher J, Yu J, Tahir N, Wischhusen J, Kozyreva O, Ortiz T, Tuncer H, Masri JE, Malcolm A, Mazor T, Cerami E, Kehl KL. MatchMiner-AI: An Open-Source Solution for Cancer Clinical Trial Matching. *arXiv*. 2026. doi: [10.48550/arXiv.2412.17228](https://doi.org/10.48550/arXiv.2412.17228)

## Contributing

Contributions are welcome! Please follow our
[contribution instructions][contributing] if you are interested in contributing
to this project.

[contributing]: https://dfci.github.io/matchminer-ai-inference/development/contributing/

Existing local guideline catalogs can be reviewed with
`trials.review_guideline_citations`. The configured guideline LLM checks source
support; code locates exact excerpts and preserves clinical fields. Unsupported
or partially supported items retain explicit per-item issues for display and
human review. See [citation review and collection repair](docs/user-guide/guideline-extraction.md#reviewing-existing-catalog-citations).
