# Training

`matchminer_ai.training` produces the models the package uses at runtime. It turns ClinicalTrials.gov records into
trial cohorts, generates synthetic patients (event histories and clinical notes) for those cohorts, summarizes the
patients, labels patient-cohort pairs with an LLM teacher, and trains the TrialSpace retriever (dense EmbeddingGemma 2)
and the decision model (LiquidAI d1-2B fine-tuned on seven yes/no question families). OncoReasoning
([its page](oncoreasoning.md)) distills the teacher's outputs, reasoning included, into a small generative model.

Training data is synthetic or public. Only the [evaluations](../evaluation/index.md) touch real patient data.

Runtime behavior is the package's, not a copy: the cohort extraction prompts and parser, the merge and boilerplate
steps, the patient summarization prompts and the split of a summary into its key and boilerplate sections, the
TrialSpace prompts (`matchminer_ai.embedding.trialspace_contract`), the match-label prompt and 0-6 score
(`matchminer_ai.matching.match_labels`), the boilerplate and per-criterion teacher prompts, the decision model's
states and questions (`matchminer_ai.matching.decision_model`), sampling profiles, the endpoint pool and the teacher
labeler (`matchminer_ai.labeling`).

**Key criteria** are a cohort's criteria that are not boilerplate. Retrieval, match labels and the decision model's
five match questions read the cohort's key criteria and the patient summary's key section (its cancer history); its
boilerplate question reads the cohort's boilerplate criteria and the summary's boilerplate section
(`Other relevant history:`); its per-criterion question reads the complete summary and every criteria line, each
tagged inclusion or exclusion and boilerplate or not (`cohort_criteria_items`).

## Setup

Install the package with the training extra, and point it at a data root (where `phi/` and `no_phi/` live; never
inside the repository) and, for site-specific endpoints and paths, a [site config](../reference/site-config.md):

```bash
pip install -e '.[training]'
export MMAI_DATA_ROOT=/path/to/data        # or --data-root / the site config's paths.data_root
export MMAI_SITE_CONFIG=/path/to/site.yaml # optional
```

Every stage is a module (`python -m matchminer_ai.training.<stage>`); `mmai-train` runs the pipeline. Stage names
below are relative to `matchminer_ai.training`. LLM stages never start or stop a shared vLLM fleet: they use an
existing server registry (`--server-urls-file` / `--teacher-servers-file`, a JSON file re-read while running) or fixed
URLs; `training.from_summaries` can also start one vLLM teacher per local GPU.

## Pipeline

`mmai-train` (`python -m matchminer_ai.training.pipeline`) runs
a contiguous range of stages (`--start-at` to `--stop-after`, inclusive). The
default range is `train` only.

| Stage | Command | Main outputs (defaults under `<data root>/no_phi`) |
| --- | --- | --- |
| `parse-trials` | `trials.parse_ctgov` (requires `--ctgov-json`, a CT.gov API v2 export) | `ctgov_trials.csv` |
| `screen-trials` | `trials.screen_trials` | `screened_trials.csv` + `.screening.json`: every trial meeting the ctgov rule, with its DFCI holdout flag and screen decision; the counts and removed IDs of each step. Every screen answer and its reasoning goes to `upstream/screen/screen.jsonl`. |
| `sample-trials` | `trials.sample_trials` --screened-trials` | `ctgov_sample_trials.csv` + `.sampling.json`: the seeded draw of `--sample-trials` (5,000) accepted trials, which seed the protocols and give the source splits |
| `synthetic-protocols` | `trials.synthetic_protocols.generate_batch` | `synthetic_protocols/` (`seeds.jsonl`, `protocols/<NCT>/protocol.md`, `protocols.csv`) |
| `protocol-cohorts` | `trials.synthetic_protocols.extract_cohorts` | `synthetic_protocols/cohort_extraction/` or `--protocol-cohort-dir` (`cohorts.parquet` merged, `cohorts_unmerged.parquet`, `cohorts_deboilerplated.parquet` merged with key and boilerplate criteria, `trials.parquet`) |
| `ctgov-cohorts` | `trials.extract_ctgov_cohorts` --trials-jsonl` (the protocol seeds) | `ctgov_cohorts/` (`trial_cohort_lineitems.csv` merged, `trial_cohort_lineitems_unmerged.csv`, `trial_cohort_lineitems_deboilerplated.csv` merged with key and boilerplate criteria, `trials_with_cohorts.csv`, `trial_cohorts.json`) |
| `generation-cohorts` | `trials.synthetic_protocols.patient_cohorts` | `synthetic_generation/sample_trial_cohort_lineitems.csv` (merged protocol cohorts with all their criteria, for patient generation) and `sample_trial_cohort_lineitems_deboilerplated.csv` (the same rows, for retrieval: `cohort_eligibility_criteria` keeps all the criteria, `cohort_key_criteria` holds the key criteria and `cohort_boilerplate_criteria` the boilerplate), each + `.provenance.json` |
| `retrieval-cohorts` | `trials.synthetic_protocols.retrieval_cohorts` | `retrieval_cohorts.csv` (both versions of every trial) + `.provenance.json` |
| `generate-synthetic` | `synthetic.generate` | `synthetic_generation/all_synthetic_events.parquet`, `all_synthetic_notes.parquet`, `manifest.json` |
| `prepare-notes` | `synthetic.prepare` notes` | `all_synthetic_notes.parquet`, `synthetic_patient_metadata.parquet` |
| `summarize-patients` | `summaries.summarize` | `patient_summaries.parquet`, `patient_serial_summaries.parquet` |
| `prepare-training` | `synthetic.prepare` summaries` | `patient_summaries_with_spaces.parquet` (key section in `patient_summary`, boilerplate section in `patient_boilerplate_text`) |
| `train` | `training.from_summaries` (corpus from `retrieval_cohorts.csv`) | TrialSpace rounds and the decision model under `<models root>/embeddinggemma2_pipeline_v1` |

Patients are generated only from the protocol cohorts (`generation-cohorts`),
with all their criteria; retrieval, labeling and training use
`retrieval_cohorts.csv`, which holds each trial's protocol and ctgov versions
(see Mining below) with their key and boilerplate criteria. The three LLM
extraction/generation stages after sampling share `--server-urls-file`; a trial
that fails every attempt in one of them keeps whichever version it has.

Inspect a plan, then remove `--dry-run` to execute:

```bash
mmai-train --start-at parse-trials --stop-after prepare-training \
  --ctgov-json <data root>/no_phi/ctgov_export.json \
  --data-dir <data root>/no_phi/new_dataset \
  --run-dir <data root>/no_phi/new_training_run \
  --models-dir <models root>/new_training_run \
  --server-urls-file <data root>/no_phi/live_pool/servers.json \
  --dry-run
```

**Resume and safety.** All data-production paths must be under `<data root>/no_phi`.
Each stage's fingerprint covers its command, input size/mtime stamps and source
hashes. A stage with a matching `completed/<stage>.json` is skipped; a changed
fingerprint is an error (use a fresh `--pipeline-dir`). A lock in the pipeline
directory prevents concurrent runs, and logs go to `<pipeline-dir>/logs/`.
`--wait-for-generation` attaches to a generation job that is already running
(it polls `status.json` and never starts or stops it), and is implied when
`generation.lock` is held.

The default LLM for data production and labeling is
`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4` (`train_from_summaries.DEFAULT_TEACHER`; also
`trials.ctgov_llm`'s `--model` for 0b and 0d, and the real-data evaluations' `GOLD_LLM`), overridden
with `--generation-model` / `--generation-tokenizer` (upstream) and
`--teacher-model` / `--teacher-served-model` (training; an endpoint's model root
must match the teacher model or be an approved quantization of it, listed in
`EQUIVALENT_TEACHER_ROOTS` in the inference package's `matchminer_ai.labeling.runner`: the
Gemma 4 31B builds, and Nemotron 3.5 Super VL NVFP4 and FP8, whose tokenizers and chat
templates are identical). Scripts that start their own vLLM servers for Nemotron (the
teacher pool of `training.from_summaries` and `summaries.summarize` in local mode) add `--trust-remote-code` and its hybrid-Mamba flags
(`vllm_reasoning.nemotron_serve_flags()`) with the `nemotron_v3` reasoning parser. The
NVFP4 build needs native FP4 (Blackwell); on H100s, override with the FP8 build,
`RedHatAI/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-FP8`.


## Data stages

- **0a — parse.** Flattens the CT.gov JSON to one row per trial: NCT ID, study
  type, phases, ctgov status and dates (with original precision and
  ACTUAL/ESTIMATED types, `trials.ctgov_metadata`), title, brief summary,
  eligibility criteria and `trial_text`. No filtering and no LLM.
- **0b — screen** (`trials.screen_trials`). The ctgov rule (interventional
  trials first posted in `--start-year` (2020) or later with early phase 1
  through phase 4; 80/10/10 splits with seed 42), the DFCI holdout, then the
  cohort workflow's cancer-treatment screen on every remaining listing, with
  every answer and its reasoning saved (see Sampling above). Its LLM settings and
  endpoint options are shared with 0d (`trials.ctgov_llm`).
- **0c — sample.** Production (`--screened-trials`) draws 5,000 whole trials
  (seed 42) from those `trials.screen_trials` accepted, before any cohort
  extraction. The older modes keep cancer-treatment trials from an extracted
  cohort table, drop trials with invalid or incomplete cohorts, exclude DFCI
  trials open during 2025–2026 (`dfci_trials_summary_10-18-26.csv`), and sample
  5,000 whole trials with all their cohorts (seed 42); `--extraction-dir`
  samples a completed extraction export without new LLM calls, and
  `repair_single_cohort_criteria.py` fixes exports made before the single-cohort
  correction.
- **Synthetic protocols and their cohorts** (`trials.synthetic_protocols.generate_batch`,
  `trials.synthetic_protocols.extract_cohorts`): one synthetic protocol per sampled
  trial, then its cohorts; see `synthetic_protocols/README.md`.
- **0d — ctgov cohorts** (`trials.extract_ctgov_cohorts` --trials-jsonl`,
  formerly `0b_extract_trial_cohorts.py`). Extracts every sampled ctgov record
  (the protocol batch's `seeds.jsonl`) with `matchminer_ai.trials.extract_cohorts`
  and the package's three-step workflow (screen, cohort names, one request per
  cohort for its criteria, copied verbatim). For training, equivalent cohorts are
  then merged and each merged cohort's boilerplate separated
  (`--output-deboilerplated-cohorts`: key criteria in `cohort_key_criteria`,
  boilerplate in `cohort_boilerplate_criteria`, `cohort_eligibility_criteria`
  unchanged). Remote only; per-trial gzip checkpoints and a run fingerprint make
  it resumable, and failed trials leave `.partial` outputs and an error.
- **Synthetic patients** (`synthetic.generate`, prompts in
  `synthetic.event_prompts`, note prompt and masking in
  `synthetic.note_prompts`). For every cohort: five positive seeds
  (meet all criteria) and five near-miss negative seeds; each seed becomes a
  validated event chronology (demographics, diagnoses, treatments, adverse
  events, notes, imaging, pathology and NGS reports); each report event becomes a
  note written with the full history and the target event marked. Validation
  failures are fed back and retried (up to 16 attempts). A patient whose history
  fails validation on every attempt gets a new seed, drawn with the unchanged
  seed prompt and distinct from the cohort's other seeds, and is expanded again
  (`--max-seed-replacements`, default 3); replacements are recorded in
  `patient_seed_replacements/` and as `seed_replacement` in patient provenance,
  and seed batches are never rewritten. Transport failures keep their seed. Runs
  against a remote pool with adaptive concurrency; `--stop-after-events` stops
  before notes.
- **Prepare notes** (`synthetic.prepare` notes`). Assigns explicitly
  fabricated calendar dates from event ages, ending at `--synthetic-date-anchor`
  (2025-01-01), and writes patient metadata. All synthetic splits are pooled for
  training (`synthetic.pool`); the original split is kept as `source_split`.
- **Summarize** (`summaries.summarize`, with `summaries.pool` and
  `summaries.workspace`). Serial, chronological summarization with the inference
  package's prompts: each patient's dated notes are concatenated, split into
  overlapping token chunks (the pipeline uses 50,000 tokens with 500 overlap), and
  each chunk updates the running summary. Invalid chunks are rewound and redone.
  There is no note tagging or retrieval step.
- **Prepare training** (`synthetic.prepare` summaries`). Joins final
  summaries to patient and cohort IDs, class, source trial and split, and keeps
  each summary's two sections apart (2026-11-04): `patient_summary` is the key
  section (the cancer history, before `Other relevant history:`) and
  `patient_boilerplate_text` the boilerplate section.

CSV inputs share `cohort_format.read_clinical_csv`, which keeps literal text
such as `None` or `NA`; only empty cells are missing.


## Training (`training.from_summaries`)

The integrated runner trains the TrialSpace retriever (EmbeddingGemma 2) and the
decision model. Since 2026-11-06 (user instruction) every stage is its own script,
and the runner only orders them: it asks each stage script whether its output is
complete (`--check`), skips it if so, and otherwise runs it and checks again;
`--start-at` and `--stop-after` choose a range of stages. Each script writes
`stage_receipt.json` into its output directory last (`trialspace/stage.py`): an
identity with no paths, hosts or devices (its settings and the receipts of the
directories it read), its outputs by name and byte size, and provenance. A stage
run on another machine (for example `trialspace.mine` on a cloud VM's GPUs) is
adopted once its directory is copied back. `--check` exits 0 when complete, 10 when
not yet, and 20 when the settings or outputs differ (use a fresh directory). The
teacher labeler, the encoder and the exact search are the inference package's
(`matchminer_ai.labeling`, `matchminer_ai.embedding.encoder`,
`matchminer_ai.matching.retrieval`). The stages, in order:

1. `prepare-trialspace` (`trialspace.prepare`): the corpus every
   round shares (below).
2. Round 1: mine with the base model (`mine-1`), label (`label-trialspace-1`,
   `--task match`; `label-boilerplate-1`; `label-criteria-1`), project the
   match labels into a pair table (`project-trialspace-1`), fit
   (`train-trialspace-1`).
3. Round 2: the same with the round-1 model; labeling imports round-1
   judgments of pairs mined again (`--reuse-labels`); the boilerplate and
   criteria labels skip what round 1 labeled (`--skip-labeled`).
4. Round 3: mine with the round-2 model and label (`mine-3`,
   `label-trialspace-3`, importing round-2 judgments; `label-boilerplate-3`,
   `label-criteria-3`). There is no third fit.
5. `prepare-decision-data`, `train-decision-model` (`training.decision_model`
   prepare` / `train`): the decision model on every round's labels of all seven
   question families.

The other stage scripts: `trialspace.mine` (`mine-N`), `trialspace.label`
(`label-{trialspace,boilerplate,criteria}-N`), `trialspace.project`
(`project-trialspace-N`) and `trialspace.finetune` (`train-trialspace-N`);
`trialspace.evaluate` gives the in-sample diagnostics outside the plan.

**Corpus.** `trialspace.prepare` (`trialspace/corpus.py`, `trialspace/data.py`) gives each
patient `text` = the summary's key section and `boilerplate` = its boilerplate
section (a complete summary is split at `Other relevant history:`), and each
cohort `text` = `cohort_key_criteria` and `boilerplate` =
`cohort_boilerplate_criteria`. A cohort table without `cohort_key_criteria`
falls back to `cohort_eligibility_criteria` (all criteria), so pass the
de-boilerplated `retrieval_cohorts.csv`, not the patient-generation table.

**Retriever.** TrialSpace is the only retriever. Base `google/embeddinggemma-2`
at revision `914f7f89`, mined by exhaustive cosine similarity over the key texts
only, with 8,192-token inputs (`trialspace.mine`; one GPU or several with
`--devices cuda:0,cuda:1,...`, whose vectors equal a one-GPU run's, or vLLM >= 0.32
embedding servers with `--embedding-backend openai`), and fit with
`trialspace.finetune` --prompt-profile trialspace` (the patient and cohort
prompts are saved under the inference names `patient_summary` and
`this_space`): CoSENT over every labeled pair plus MNRL over positives with
known positives masked, 96 pairs per GPU (gradient-cache chunks of 8 texts,
`--trialspace-loss-mini-batch-size`; 32 fits on larger-memory GPUs), BF16,
learning rate 2e-5 with 1% linear warmup and decay over 1 epoch
(`--trialspace-epochs`), 4,096-token training cap (key texts only). Round 2
starts from the round-1 model. The fit runs one rank per GPU in `--gpus`
(`torch.distributed.run`). Each rank's ranking losses see only its own 96
pairs, and gradients are averaged, so an optimizer step covers 96 times the
GPU count. Rank zero stages inputs to local SSD and publishes the model.
Models are saved to `trialspace_round{1,2}/` under
`--models-dir`. Dense stages run with `--embedding-python` (an environment
with EmbeddingGemma 2 support).

**Match labels** (`trialspace.label` --task match`; 2026-11-04, user
instruction). One teacher request per pair answers all five match labels,
`reasonableness`, `cancer_type`, `burden`, `treatment_history` and `biomarker`,
each yes/no with an `evidence_status` and a `reason`, using the inference
package's `match_quality.labels.system.txt` (`matchminer_ai.matching.match_labels`)
on the patient's key section and the cohort's key criteria. The score is 0–6:
0 unless reasonable, else 1 plus one each for cancer type, burden and treatment
history and two for biomarker (`trialspace/regression.py`, `contracts.py`).
`scores.jsonl` rows keep all five answers (never null) and `question_keys`
`{"match": key}`. Since 2026-11-05 (user instruction) the answer is decoded under
the inference package's `MATCH_LABELS_JSON_SCHEMA` (vLLM `structured_outputs`,
applied after the reasoning), part of the label contract: label directories made
before then do not match it. The earlier gated design (`potentially_eligible` first,
specificity questions only after a yes, one question per request, prompts in
`colbert_experiments/prompts/`) is gone; its label directories do not match the
current contract.

**Boilerplate labels** (`label-boilerplate-N`: `trialspace.label`
--task boilerplate --skip-labeled <earlier rounds' boilerplate_teacher>`; user,
2026-11-04, per round since 2026-11-05). Every unique combination of patient
boilerplate section and cohort boilerplate criteria among the round's candidate
pairs not labeled in an earlier round (pairs whose cohort
has boilerplate criteria; one representative pair each, counts in
`combinations.json`) is asked the inference
package's `llm_exclusion_criteria.user.txt` (Yes!/No!: does the patient clearly
fail any boilerplate criterion) on the patient's boilerplate section and the
cohort's boilerplate criteria. Pairs whose cohort has no boilerplate are
skipped. Answers go to `labels.jsonl` as `excluded`, with their reasoning.

**Per-criterion labels** (`label-criteria-N`: `trialspace.label`
--task criteria --criteria-pair-fraction 0.01 --skip-labeled <earlier rounds'
criteria_teacher>`; user, 2026-11-05). A seeded hash of each candidate pair
selects `--criteria-pair-fraction` of the round's pairs (the same pairs in every
round), and every criteria line of a selected pair that has an inclusion or
exclusion tag gets one teacher request: the inference package's
`criterion_check.labels.user.txt` around the decision model's own state (the
complete patient summary and the cohort's complete numbered criteria) and
question (the line restated with its number and type). Lines already labeled
for the patient (an identical sibling cohort, or an earlier round) are skipped.
The answer is `{"answer", "criterion_status", "reason"}`: **`answer` yes means
the patient does not clearly fail the line, no means the patient clearly fails
it (an inclusion line clearly not met, or an exclusion line clearly met)**;
`criterion_status` (`met`, `not_met`, `insufficient_information`,
`not_applicable`) records the line as written, and a no must name the failing
status. The answer is decoded under its JSON schema (vLLM `structured_outputs`,
applied after the reasoning): unconstrained, Nemotron 3.5 Super dropped the
opening quote of `reason` in about one answer in six (2026-11-05 test). At 0.01, about 29,000 pairs × about 30 lines is roughly 0.9 million
requests per round (match labels: about 2.9 million).

**Decision model** (`training.decision_model`; 2026-11-05, user instruction).
Base [`LiquidAI/d1-2B`](https://huggingface.co/LiquidAI/d1-2B) at revision
`652f920e` (`--decision-model`, `--decision-revision`; LFM Open License v1.0),
a System One model that answers yes/no (`noul`) questions about a state in one
forward pass. Seven question families, every one asked as yes/no:

| Family | State | Questions (yes means) |
| --- | --- | --- |
| `match` (1-5) | the patient's key section and the cohort's key criteria | `reasonableness` (potentially eligible; no = clearly ineligible), `cancer_type`, `burden`, `treatment_history`, `biomarker` (the cohort specifically requires that dimension and the patient documentedly satisfies it), from each pair's match labels |
| `boilerplate` (6) | the patient's boilerplate section and the cohort's boilerplate criteria | potentially eligible: the boilerplate teacher's `excluded`, inverted |
| `criterion` (7) | the complete summary and the cohort's complete numbered criteria | one per labeled line: **yes = does not clearly fail the line, no = clearly fails it** |

`prepare` writes one row per state (`examples.jsonl`: family, ids, each
question's answer; a later round wins a conflict), `prepared.json` (counts,
answer balance, question digest) and `preview.jsonl` (rendered prompts with
their answers, for inspection). States and questions are rendered from the
corpus by the inference package, so training reads exactly what inference
reads. `train` fully fine-tunes the model with transformers' `Trainer` (no
LoRA): each state's questions are shuffled (reseeded per epoch) and packed after
the state by the checkpoint's own serving code, at most 16 per sequence and
10,000 tokens (`--decision-max-length`). d1-2B reads its prompt with `set`
attention (the state both ways with its questions on full-attention layers), so
the answer is never appended to the input: **the loss is the cross-entropy of the
answer at each question's answer slot only**, never on prompt tokens, read as
inference reads it: the best `yes` form against the best `no` form. Attention
runs in the autocast dtype on d1's flash kernels, as at inference.
Defaults: 1 epoch, 8 states per GPU × 4 accumulation steps, learning rate 1e-5
with cosine decay and 3% warmup, BF16, gradient checkpointing, the vision tower
frozen, every synthetic pair (no holdout). The output (`decision_model/` under
`--models-dir`) holds the weights, the d1 code, tokenizer, LICENSE, a
modification notice, `calibration.json` (temperature 1.0) and
`decision_contract.json`; point the inference package's
`decision_model.model_name` at it. vLLM 0.33 cannot serve it (no LFM3
architecture, causal d1-3B reader only); the inference package serves it with
Transformers.

**Mining.** EmbeddingGemma 2 (`trialspace.mine`, `trialspace/mining.py`) produces
this candidate set by default, with no flags needed:

- Each patient searches only the cohorts of 500 randomly sampled trials
  (`--patient-trial-sample`; 0 searches all trials). Each patient gets its own
  sample, drawn reproducibly from `--seed` (42), the patient ID and the
  mining round: every round draws a fresh sample (`--trial-sample-round`,
  which the runner sets to 2 and 3; round 1 keeps the original draws).
- A trial with both a synthetic-protocol version and a ctgov
  (ClinicalTrials.gov) version of its cohorts (`trials.synthetic_protocols.retrieval_cohorts`)
  is represented by one of them: in each patient's search, a version chosen at
  random per patient and trial; as a cohort query, a version chosen at random
  per trial (`trialspace/retrieval.py`). Trials with one version use it.
  Each cohort's version is `source` in the prepared corpus and `cohort_source`
  in the exported pair tables.
- Per patient: the top 20 cohorts by the retriever, plus 5 TF-IDF and 5 random
  cohorts, all from that patient's sampled trials.
- Per cohort: the top 40 patients over the whole corpus, plus 5 TF-IDF and 5
  random patients.
- Every patient's pairs with the cohorts of its source trial, in both versions.

The TF-IDF and random extras (`--lexical-k`, `--random-k`, default 5 each)
supply teacher-labeled pairs the retriever would not rank highly. Keep them
unless an experiment says otherwise. Each round's `mining/` also holds
per-direction tables (`patient_to_cohort.parquet`, `cohort_to_patient.parquet`)
with the key texts as `patient_summary` / `this_space` (also
`cohort_key_criteria`) and the boilerplate texts as `patient_boilerplate_text` /
`trial_boilerplate_text`.

```bash
python -m matchminer_ai.training.from_summaries --dry-run
python -m matchminer_ai.training.from_summaries \
  --patient-summaries <data root>/no_phi/patient_summaries_with_spaces.parquet \
  --trial-cohorts <data root>/no_phi/retrieval_cohorts.csv \
  --teacher-servers-file <data root>/no_phi/live_pool/servers.json \
  --teacher-served-model nemotron-3.5-super-vl-120b \
  --embedding-python /path/to/embeddinggemma2-env/bin/python --gpus 2,3
```

Defaults: `--run-dir <data root>/no_phi/embeddinggemma2_pipeline_v1`,
`--models-dir <models root>/embeddinggemma2_pipeline_v1` (the models root is the site config's
`paths.models_root`, else `<data root>/../models`). Teachers come from `--teacher-servers-file`,
repeated `--teacher-url`, or one vLLM server started per GPU in `--gpus`
(default 0–7). Each stage owns its completion: its output directory's path-free `stage_receipt.json`, which the
stage's `--check` compares with the identity it would compute (0 complete, 10 not yet, 20 begun under other
settings). A stage done on another machine and copied back is complete here when its receipt matches. Decision-model settings:
`--decision-model`, `--decision-revision`, `--decision-max-length`,
`--decision-epochs`, `--decision-batch-size`,
`--decision-gradient-accumulation-steps`, `--decision-learning-rate`,
`--criteria-pair-fraction`, `--criteria-seed`.

### Labeling and retrieval storage

- **Labeling** (`trialspace.label` --task match|boilerplate|criteria`; the
  labeler and its local workspace are `matchminer_ai.labeling`) runs on local SSD by default
  (`/tmp/mmai-labels/<hash>`, or `MMAI_LABEL_SCRATCH_DIR`; 16 GiB reserve;
  `MMAI_LOCAL_LABEL_STORAGE=0` disables). Concurrency per server starts at 96 and
  can reach 768. Status is mirrored to the output directory every 10 seconds. On
  completion, `scores.jsonl` (match labels only), `labels.jsonl`, the response
  database (prompts, answers and reasoning) and manifests are published with
  `storage_complete.json`; failed runs keep their scratch for resume.
- **Retrieval** (`trialspace.mine`, `trialspace.retrieval_workspace`)
  stages work in `/tmp/mmai-retrieval` (`--retrieval-scratch-dir`; network
  filesystems are rejected), publishes candidates, rankings, encodings and
  `complete.json` to `--output`, then removes the scratch. Unchanged dense
  encodings can be reused across merged corpora (`retrieval_cache_import.py`);
  rankings never are.

Models and data are published only by size-checked copies to shared `no_phi`
storage. There is no Hugging Face upload code in this repository.

## Synthetic trial protocols

`synthetic_protocols/` builds a corpus of synthetic full protocol documents
seeded by ctgov records, with realistic, often messily specified eligibility
covering one or more cohorts. Generation runs through
`matchminer_ai.trials.generate_synthetic_protocols`; see
[Synthetic protocols](synthetic-protocols.md). Training uses
these synthetic protocols, not real ones. Real protocol PDFs posted
on ClinicalTrials.gov (check licensing before redistributing) are for comparison and evaluation only.
`trials.synthetic_protocols.extract_cohorts` extracts each synthetic protocol's
cohorts and eligibility criteria with
`matchminer_ai.trials.extract_protocol_cohorts`, keeping every cohort's
provenance: the synthetic protocol and its generation, and the original
ClinicalTrials.gov record and eligibility criteria it was seeded from. In the
pipeline above, these cohorts seed the synthetic patients (`generation-cohorts`)
and, beside the ctgov versions, form the retrieval cohort table
(`retrieval-cohorts`).


## Module map

| Module | Role |
| --- | --- |
| `training.pipeline`, `training.from_summaries` | Orchestration (`mmai-train`): stage ranges, fingerprints, resume |
| `training.trials` | ctgov parsing, screening, sampling, ctgov cohort extraction, synthetic protocols and their cohorts |
| `training.synthetic` | Synthetic patients: seeds, event histories, notes, note preparation, the pooled corpus |
| `training.summaries` | Serial summarization of the synthetic notes |
| `training.trialspace` | TrialSpace corpus, mining, labeling, projection, fits, stage receipts |
| `training.decision_model` | Decision-model data and fine-tuning |
| `training.oncoreasoning` | OncoReasoning data, training and evaluation |
| `training.cohort_format`, `training.paths`, `training.vllm_reasoning` | Shared cohort table rules, data paths and run identities, vLLM helpers |

## Tests

`tests/training` covers synthetic generation and validation, cohort extraction with a fake LLM, pipeline planning,
resume and wiring, note preparation, CSV literals, trial versions, decision data, label contracts, packing and the
embedding losses; `finetune_embedder` tests train a tiny random model.

## Data and clinical safety

Training stages read and write only under the data root's `no_phi` tree and refuse other directories. Do not point
them at real clinical data.
