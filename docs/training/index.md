# Training

`matchminer_ai.training` produces the models the package uses at runtime. It turns ClinicalTrials.gov (ctgov) records
into trial cohorts, generates synthetic patients (event histories and clinical notes) for those cohorts, summarizes the
patients, labels patient-cohort pairs with an LLM teacher, and trains the TrialSpace retriever (dense EmbeddingGemma 2)
and the decision model (LiquidAI d1-2B fine-tuned on seven yes/no question families). OncoReasoning
([its page](oncoreasoning.md)) distills the teacher's outputs, reasoning included, into a small generative model.

Training data is synthetic or public. Only the [evaluations](../evaluation/index.md) touch real patient data.

Training runs the package's own code, not a copy: the cohort extraction prompts and parser, the merge and boilerplate
steps, the patient summarization prompts and the split of a summary into its key and boilerplate sections, the
TrialSpace prompts (`matchminer_ai.embedding.trialspace_contract`), the match-label prompt and 0-6 score
(`matchminer_ai.matching.match_labels`), the boilerplate and per-criterion teacher prompts, the decision model's
states and questions (`matchminer_ai.matching.decision_model`), sampling profiles, the endpoint pool and the teacher
labeler ([`matchminer_ai.labeling`](../api/labeling.md)).

**Key criteria** are a cohort's criteria that are not boilerplate (see
[How matching works](../user-guide/patient-centric-matching.md)). Retrieval, the match labels and the decision model's
five match questions read the cohort's key criteria and the patient summary's key section (its cancer history). The
boilerplate question reads the cohort's boilerplate criteria and the summary's boilerplate section
(`Other relevant history:`). The per-criterion question reads the complete summary and every criteria line, each
tagged inclusion or exclusion and boilerplate or not (`cohort_criteria_items`).

## Setup

Install the package with the training extra, and point it at a data root (where `phi/` and `no_phi/` live; never
inside the repository) and, for site-specific endpoints and paths, a [site config](../reference/site-config.md):

```bash
uv pip install -e ".[training]"
export MMAI_DATA_ROOT=/path/to/data        # or --data-root / the site config's paths.data_root
export MMAI_SITE_CONFIG=/path/to/site.yaml # optional
```

Every stage is a module (`python -m matchminer_ai.training.<stage>`); `mmai-train` runs the pipeline. Stage modules
below are named relative to `matchminer_ai.training`. LLM stages never start or stop a shared vLLM fleet: they use an
existing server registry (`--server-urls-file` / `--teacher-servers-file`, a JSON file re-read while running) or fixed
URLs. `training.from_summaries` can also start one vLLM teacher per local GPU.

## Pipeline

`mmai-train` (`python -m matchminer_ai.training.pipeline`) runs a contiguous range of stages, `--start-at` to
`--stop-after` (inclusive). The default range is `train` only.

| Stage | Module | Main outputs (defaults under `--data-dir`, which defaults to `<data root>/no_phi`) |
| --- | --- | --- |
| `parse-trials` | `trials.parse_ctgov` (requires `--ctgov-json`, a ctgov API v2 export) | `ctgov_trials.csv` |
| `screen-trials` | `trials.screen_trials` | `screened_trials.csv` (every trial meeting the ctgov rule, with its DFCI holdout flag and screen decision) and `.screening.json` (the counts and IDs removed at each step) |
| `sample-trials` | `trials.sample_trials --screened-trials` | `ctgov_sample_trials.csv` and `.sampling.json`: the seeded draw of `--sample-trials` (5,000) accepted trials, which seed the protocols and give the source splits |
| `synthetic-protocols` | `trials.synthetic_protocols.generate_batch` | `synthetic_protocols/` (`seeds.jsonl`, `protocols/<NCT>/protocol.md`, `protocols.csv`) |
| `protocol-cohorts` | `trials.synthetic_protocols.extract_cohorts` | `synthetic_protocols/cohort_extraction/` or `--protocol-cohort-dir`: `cohorts.parquet` (merged), `cohorts_unmerged.parquet`, `cohorts_deboilerplated.parquet` (merged, with key and boilerplate criteria), `trials.parquet` |
| `ctgov-cohorts` | `trials.extract_ctgov_cohorts --trials-jsonl` (the protocol seeds) | `ctgov_cohorts/`: `trial_cohort_lineitems.csv` (merged), `trial_cohort_lineitems_unmerged.csv`, `trial_cohort_lineitems_deboilerplated.csv` (merged, with key and boilerplate criteria), `trials_with_cohorts.csv`, `trial_cohorts.json` |
| `generation-cohorts` | `trials.synthetic_protocols.patient_cohorts` | `synthetic_generation/sample_trial_cohort_lineitems.csv` (merged protocol cohorts with all their criteria, for patient generation) and `sample_trial_cohort_lineitems_deboilerplated.csv` (the same rows with `cohort_key_criteria` and `cohort_boilerplate_criteria`, for retrieval), each with `.provenance.json` |
| `retrieval-cohorts` | `trials.synthetic_protocols.retrieval_cohorts` | `retrieval_cohorts.csv` (both versions of every trial) and `.provenance.json` |
| `generate-synthetic` | `synthetic.generate` | `synthetic_generation/all_synthetic_events.parquet`, `all_synthetic_notes.parquet`, `manifest.json` |
| `prepare-notes` | `synthetic.prepare notes` | `all_synthetic_notes.parquet`, `synthetic_patient_metadata.parquet` |
| `summarize-patients` | `summaries.summarize` | `patient_summaries.parquet`, `patient_serial_summaries.parquet` |
| `prepare-training` | `synthetic.prepare summaries` | `patient_summaries_with_spaces.parquet` (key section in `patient_summary`, boilerplate section in `patient_boilerplate_text`) |
| `train` | `from_summaries` (cohorts from `retrieval_cohorts.csv`) | TrialSpace rounds and the decision model (see below) |

Patients are generated only from the protocol cohorts (`generation-cohorts`), with all their criteria. Retrieval,
labeling and training use `retrieval_cohorts.csv`, which holds each trial's protocol and ctgov versions with their key
and boilerplate criteria (see [Synthetic protocols](synthetic-protocols.md)). Every LLM stage uses the pipeline's
`--server-urls-file`; a trial that fails every attempt in one of the extraction or generation stages keeps whichever
version it has.

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

**Resume and safety.** All data-production directories must be under `<data root>/no_phi`. State and logs go to
`--pipeline-dir` (default `<run dir>/upstream`). Each stage's fingerprint covers its command, the size and mtime of
its inputs and the hashes of its source files. A stage with a matching `completed/<stage>.json` is skipped; a changed
fingerprint or output is an error (use a fresh `--pipeline-dir`). A lock in the pipeline directory prevents concurrent
runs, and each stage logs to `<pipeline-dir>/logs/<stage>.log`. `--wait-for-generation` attaches to a generation job
that is already running (it polls `status.json` and never starts or stops it), and is implied when `generation.lock`
is held.

**Models.** The default teacher and generation model of every LLM stage is
`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4` (`from_summaries.DEFAULT_TEACHER`). Override it with
`--generation-model` / `--generation-tokenizer` (data production) and `--teacher-model` / `--teacher-served-model`
(labeling). An endpoint's model root must match the teacher model or be an approved build of it
(`matchminer_ai.labeling.runner.EQUIVALENT_TEACHER_ROOTS`: the Gemma 4 31B builds, and Nemotron 3.5 Super VL NVFP4 and
FP8, whose tokenizers and chat templates are identical). When `training.from_summaries` starts its own vLLM teachers
for Nemotron, it adds the hybrid-Mamba engine flags (`vllm_reasoning.nemotron_serve_flags()`) and the `nemotron_v3`
reasoning parser. The NVFP4 build needs native FP4 (Blackwell); on H100s, use the FP8 build,
`RedHatAI/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-FP8`.

## Data stages

- **Parse** (`trials.parse_ctgov`). Flattens the ctgov JSON to one row per trial: NCT ID, study type, phases, ctgov
  status and dates (with their original precision and ACTUAL/ESTIMATED types, `trials.ctgov_metadata`), title, brief
  summary, eligibility criteria and `trial_text`. No filtering and no LLM.
- **Screen** (`trials.screen_trials`). In order:
    1. The ctgov rule: interventional trials first posted in `--start-year` (2020) or later, every listed phase early
       phase 1 through phase 4. Source splits are assigned 80/10/10 with seed 42.
    2. The DFCI holdout: trials open at DFCI from `--dfci-first-year` to `--dfci-last-year` (2025-2026, from
       `--dfci-trials-csv`) are excluded.
    3. The cohort workflow's cancer-treatment screen on every remaining listing (title, brief summary and eligibility
       criteria as one document). Every answer and its reasoning is appended to `<pipeline-dir>/screen/screen.jsonl`
       and reused on rerun.

    Its LLM settings and endpoint options are shared with ctgov cohort extraction (`trials.ctgov_llm`).
- **Sample** (`trials.sample_trials --screened-trials`). Draws 5,000 whole trials (seed 42) from those the screen
  accepted, before any cohort extraction.
- **Synthetic protocols and their cohorts** (`trials.synthetic_protocols.generate_batch`,
  `trials.synthetic_protocols.extract_cohorts`): one synthetic protocol per sampled trial, then its cohorts; see
  [Synthetic protocols](synthetic-protocols.md).
- **ctgov cohorts** (`trials.extract_ctgov_cohorts --trials-jsonl`). Extracts every sampled ctgov record (the
  protocol batch's `seeds.jsonl`) with `matchminer_ai.trials.extract_cohorts`: a cancer-treatment screen, the cohort
  names, then one request per cohort for its criteria, copied verbatim. Equivalent cohorts are then merged and each
  merged cohort's lines classified (`--output-deboilerplated-cohorts`: key criteria in `cohort_key_criteria`,
  boilerplate in `cohort_boilerplate_criteria`, the tagged lines in `cohort_criteria_items`,
  `cohort_eligibility_criteria` unchanged). Remote endpoints only. Per-trial gzip checkpoints and a run fingerprint
  make it resumable; a trial that fails leaves `.partial` outputs and an error unless `--allow-failures` is given.
- **Synthetic patients** (`synthetic.generate`; prompts in `synthetic.event_prompts`, note prompt and masking in
  `synthetic.note_prompts`). For every cohort: five positive seeds (meet all criteria) and five near-miss negative
  seeds. Each seed becomes a validated event chronology (demographics, diagnoses, treatments, adverse events, notes,
  imaging, pathology and NGS reports), and each report event becomes a note written with the full history and the
  target event marked. Validation failures are fed back and retried (up to 16 attempts). A patient whose history
  fails validation on every attempt gets a new seed, drawn with the unchanged seed prompt and distinct from the
  cohort's other seeds, and is expanded again (`--max-seed-replacements`, default 3). Replacements are recorded in
  `patient_seed_replacements/` and as `seed_replacement` in patient provenance; seed batches are never rewritten, and
  transport failures keep their seed. Runs against a remote pool with adaptive concurrency; `--stop-after-events`
  stops before notes.
- **Prepare notes** (`synthetic.prepare notes`). Assigns explicitly fabricated calendar dates from event ages, ending
  at `--synthetic-date-anchor` (2025-01-01), and writes patient metadata. All synthetic splits are pooled for training
  (`synthetic.pool`); the original split is kept as `source_split`.
- **Summarize** (`summaries.summarize`). The package's own serial summarizer
  (`matchminer_ai.patients.summarize_patients`, the same code and prompts as at runtime): each patient's dated notes
  are deduplicated and concatenated, split into overlapping token chunks (`--summary-chunk-size` 50,000 tokens,
  `--summary-chunk-overlap` 500), and each chunk updates the running summary; a chunk that fails the quality checks is
  asked again. The stage writes every chunk's prior summary, reasoning and answer (`patients.serial_trace`, the table
  OncoReasoning trains on) and each patient's final summary. There is no note tagging or retrieval step.
- **Prepare training** (`synthetic.prepare summaries`). Joins the final summaries to patient and cohort IDs, class,
  source trial and split, and keeps each summary's two sections apart: `patient_summary` is the key section (the
  cancer history, before `Other relevant history:`) and `patient_boilerplate_text` the boilerplate section.

CSV inputs share `cohort_format.read_clinical_csv`, which keeps literal text such as `None` or `NA`; only empty cells
are missing.

## TrialSpace and the decision model (`training.from_summaries`)

The `train` stage runs `training.from_summaries`, which orders the TrialSpace and decision-model stages. Each stage is
its own module and owns its completion: it writes a `stage_receipt.json` into its output directory last
(`trialspace.stage`). The receipt holds an identity with no paths, hosts or devices (the stage's settings and the
receipts of the directories it read), its outputs by name and byte size, and provenance. `--check` compares the
receipt with the identity the stage would compute and exits 0 when complete, 10 when not yet, and 20 when it was begun
under other settings or its outputs changed (use a fresh directory). The runner checks each stage, skips it when
complete, and otherwise runs it and checks again. A stage run on another machine (for example `trialspace.mine` on a
cloud VM's GPUs) is complete here once its directory is copied back with a matching receipt.

The stages, in order (`--start-at` and `--stop-after` of `python -m matchminer_ai.training.from_summaries` choose a
range; every earlier stage must already be complete):

1. `prepare-trialspace` (`trialspace.prepare`): the corpus every round shares.
2. Round 1: mine with the base model (`mine-1`), label (`label-trialspace-1`, the match labels;
   `label-boilerplate-1`; `label-criteria-1`), project the match labels into a pair table (`project-trialspace-1`),
   fit (`train-trialspace-1`).
3. Round 2: the same with the round-1 model. Match labeling imports round-1 judgments of pairs mined again
   (`--reuse-labels`); the boilerplate and criteria labels skip what earlier rounds labeled (`--skip-labeled`).
4. Round 3: mine with the round-2 model and label (`mine-3`, `label-trialspace-3` importing round-2 judgments,
   `label-boilerplate-3`, `label-criteria-3`). There is no third fit.
5. `prepare-decision-data`, `train-decision-model` (`training.decision_model prepare` / `train`): the decision model
   on every round's labels of all seven question families.

The stage modules are `trialspace.mine` (`mine-N`), `trialspace.label` (`label-{trialspace,boilerplate,criteria}-N`),
`trialspace.project` (`project-trialspace-N`) and `trialspace.finetune` (`train-trialspace-N`). `trialspace.evaluate`
compares a fitted model with the base model on its labeled pairs (in-sample diagnostics, not a pipeline stage). The
teacher labeler, the encoder and the exact search are the package's (`matchminer_ai.labeling`,
`matchminer_ai.embedding.encoder`, `matchminer_ai.matching.retrieval`).

**Corpus.** `trialspace.prepare` (`trialspace.corpus`, `trialspace.data`) gives each patient `text` = the summary's
key section and `boilerplate` = its boilerplate section (a complete summary is split at `Other relevant history:`),
and each cohort `text` = `cohort_key_criteria`, `boilerplate` = `cohort_boilerplate_criteria` and `items` = its tagged
lines (`cohort_criteria_items`). Each cohort's trial version is its `source` (`protocol` or `ctgov`). A cohort table
without `cohort_key_criteria` falls back to `cohort_eligibility_criteria` (all criteria), so pass the de-boilerplated
`retrieval_cohorts.csv`, not the patient-generation table. All synthetic patients and cohorts form one pool; their
original splits are kept as provenance.

**Mining.** `trialspace.mine` (`trialspace.mining`) produces each round's candidate pairs:

- Each patient searches only the cohorts of 500 randomly sampled trials (`--trialspace-patient-trial-sample`; 0
  searches all trials), approximating one center's open trials. Each patient's sample is drawn reproducibly from the
  seed (42), the patient ID and the mining round, so every round draws a fresh sample.
- A trial with both a synthetic-protocol and a ctgov version of its cohorts is represented by one of them: in each
  patient's search, a version chosen at random per patient and trial; as a cohort query, a version chosen at random
  per trial (`trialspace.retrieval`). Trials with one version use it. The version is `cohort_source` in the exported
  pair tables.
- Per patient: the top 20 cohorts by the retriever, plus 5 TF-IDF and 5 random cohorts, all from that patient's
  sampled trials.
- Per cohort: the top 40 patients over the whole corpus, plus 5 TF-IDF and 5 random patients.
- Every patient's pairs with the cohorts of its source trial, in both versions.

The TF-IDF and random extras (`--trialspace-lexical-k`, `--trialspace-random-k`) supply teacher-labeled pairs the
retriever would not rank highly. Each round's `mining/` holds `candidates.jsonl`, the rankings, the encodings and
per-direction tables (`patient_to_cohort.parquet`, `cohort_to_patient.parquet`) with the key texts as
`patient_summary` / `this_space` (also `cohort_key_criteria`) and the boilerplate texts as `patient_boilerplate_text` /
`trial_boilerplate_text`.

**Retriever.** Base `google/embeddinggemma-2` at revision `914f7f89`, mined by exhaustive cosine similarity over the
key texts with 8,192-token inputs. Mining runs on one GPU or several (`--mining-devices cuda:0,cuda:1,...`, whose
vectors equal a one-GPU run's), or over vLLM >= 0.32 embedding servers
(`trialspace.mine --embedding-backend openai`). The fit (`trialspace.finetune --prompt-profile trialspace`) saves the
patient and cohort prompts under the names inference uses, `patient_summary` and `this_space`:

- **Losses:** CoSENT over every labeled pair (the 0-6 score scaled to [-1, 1]) plus MultipleNegativesRanking over
  pairs scoring at least 1, with other known positive pairs masked out of the in-batch negatives.
- **Batches:** 96 pairs per GPU (`--trialspace-batch-size`), encoded in gradient-cache chunks of 8 texts
  (`--trialspace-loss-mini-batch-size`; 32 fits on larger-memory GPUs). One rank runs per GPU in `--gpus`
  (`torch.distributed.run`); each rank's ranking losses see only its own pairs and gradients are averaged, so an
  optimizer step covers 96 times the GPU count.
- **Schedule:** BF16, learning rate 2e-5 with 1% linear warmup and decay over 1 epoch (`--trialspace-epochs`), a
  4,096-token training cap.
- **Output:** `trialspace_round1/` and `trialspace_round2/` under `--models-dir`; round 2 starts from the round-1
  model. Rank zero stages inputs to local disk and publishes the model.

Mining, projection and fits run with `--embedding-python` (an environment with EmbeddingGemma 2 support).

**Match labels** (`trialspace.label --task match`). One teacher request per pair answers all five match labels,
`reasonableness`, `cancer_type`, `burden`, `treatment_history` and `biomarker`, each yes/no with an `evidence_status`
and a `reason`. It uses `match_quality.labels.system.txt` (`matchminer_ai.matching.match_labels`) on the patient's
key section and the cohort's key criteria, decoded under `MATCH_LABELS_JSON_SCHEMA` (vLLM structured outputs, applied
after the reasoning). The score is 0-6: 0 unless reasonable, else 1 plus one each for cancer type, burden and
treatment history and two for biomarker (`match_labels.match_score`). `scores.jsonl` rows keep all five answers. The
prompt, schema and scoring form the label contract (`matchminer_ai.labeling.score_contract`), which is part of the
corpus identity, so a contract change needs fresh stage directories.

**Boilerplate labels** (`trialspace.label --task boilerplate`). Among the round's candidate pairs whose cohort has
boilerplate criteria, every unique combination of patient boilerplate section and cohort boilerplate criteria not
labeled in an earlier round is asked once (one representative pair each; counts in `combinations.json`). The prompt
is `llm_exclusion_criteria.user.txt` (the prompt of `exclusion_criteria_check_with_llm`, ending `Yes!` or `No!`: does
the patient clearly fail any boilerplate criterion). Answers go to `labels.jsonl` as `excluded`, with their reasoning.

**Per-criterion labels** (`trialspace.label --task criteria`). A seeded hash of each candidate pair selects
`--criteria-pair-fraction` (0.01) of the round's pairs, the same pairs in every round. Every criteria line of a
selected pair that has an inclusion or exclusion tag gets one teacher request: `criterion_check.labels.user.txt`
around the decision model's own state (the complete patient summary and the cohort's complete numbered criteria) and
question (the line restated with its number and type). Lines already labeled for the patient (in an identical sibling
cohort, or in an earlier round) are skipped. The answer is `{"answer", "criterion_status", "reason"}`:

- **`answer`:** yes means the patient does not clearly fail the line; no means the patient clearly fails it (an
  inclusion line clearly not met, or an exclusion line clearly met).
- **`criterion_status`:** `met`, `not_met`, `insufficient_information` or `not_applicable`, for the line as it binds
  the patient; a no must name the failing status.

The answer is decoded under its JSON schema (vLLM structured outputs after the reasoning); unconstrained, Nemotron 3.5
Super drops the opening quote of free-text strings in a sizable share of answers. On the 5,000-trial corpus the
default fraction gives about 0.9 million per-criterion requests per round, against about 2.9 million match-label
requests.

**Decision model** (`training.decision_model`). Base [`LiquidAI/d1-2B`](https://huggingface.co/LiquidAI/d1-2B) at
revision `652f920e` (`--decision-model`, `--decision-revision`; LFM Open License v1.0), a System One model that
answers yes/no (`noul`) questions about a state in one forward pass. Seven question families, each asked as yes/no:

| Family | State | Questions (yes means) |
| --- | --- | --- |
| `match` (1-5) | the patient's key section and the cohort's key criteria | `reasonableness` (potentially eligible; no = clearly ineligible), `cancer_type`, `burden`, `treatment_history`, `biomarker` (the cohort specifically requires that dimension and the patient documentedly satisfies it), from each pair's match labels |
| `boilerplate` (6) | the patient's boilerplate section and the cohort's boilerplate criteria | potentially eligible: the boilerplate teacher's `excluded`, inverted |
| `criterion` (7) | the complete summary and the cohort's complete numbered criteria | one per labeled line: yes = does not clearly fail the line, no = clearly fails it |

- **`prepare`** writes one row per state (`examples.jsonl`: family, IDs, each question's answer; a later round wins
  a conflict), `prepared.json` (counts, answer balance, question digest) and `preview.jsonl` (rendered prompts with
  their answers, for inspection). States and questions are rendered from the corpus by the package's builders, so
  training reads exactly what inference reads.
- **`train`** fully fine-tunes the model with transformers' `Trainer` (no LoRA). Each state's questions are shuffled
  (reseeded per epoch) and packed after the state by the checkpoint's own serving code, at most 16 per sequence and
  10,000 tokens (`--decision-max-length`). d1-2B reads its prompt with `set` attention (the state both ways with its
  questions on full-attention layers), so the answer is never appended to the input: the loss is the cross-entropy
  of the answer at each question's answer slot only, read as inference reads it (the best `yes` form against the best
  `no` form). Attention runs in the autocast dtype on d1's flash kernels, as at inference.
- **Defaults:** 1 epoch, 8 states per GPU x 4 accumulation steps, learning rate 1e-5 with cosine decay and 3% warmup,
  BF16, gradient checkpointing, the vision tower frozen, every synthetic pair (no holdout).
- **Output:** `decision_model/` under `--models-dir` holds the weights, the d1 code, tokenizer, `LICENSE`, a
  modification notice (`NOTICE.md`), `calibration.json` (temperature 1.0) and `decision_contract.json`. Point the
  preset's `decision_model.model_name` at it ([Configuration](../reference/configuration.md)). vLLM cannot serve it;
  the package runs it in-process with Transformers.

**Running it.**

```bash
python -m matchminer_ai.training.from_summaries --dry-run
python -m matchminer_ai.training.from_summaries \
  --patient-summaries <data root>/no_phi/patient_summaries_with_spaces.parquet \
  --trial-cohorts <data root>/no_phi/retrieval_cohorts.csv \
  --teacher-servers-file <data root>/no_phi/live_pool/servers.json \
  --teacher-served-model nemotron-3.5-super-vl-120b \
  --embedding-python /path/to/embeddinggemma2-env/bin/python --gpus 2,3
```

- **Directories:** `--run-dir` defaults to `<data root>/no_phi/embeddinggemma2_pipeline_v1` and `--models-dir` to
  `<models root>/embeddinggemma2_pipeline_v1` (the models root is the site config's `paths.models_root`, else
  `<data root>/../models`). Inputs must be under `<data root>/no_phi`.
- **Teachers:** `--teacher-servers-file` (an external registry, never rewritten), repeated `--teacher-url`, or one
  vLLM server started per GPU in `--gpus` (default 0-7).
- **Decision-model options:** `--decision-model`, `--decision-revision`, `--decision-max-length`,
  `--decision-epochs`, `--decision-batch-size`, `--decision-gradient-accumulation-steps`, `--decision-learning-rate`,
  `--criteria-pair-fraction`, `--criteria-seed`.

### Labeling and retrieval storage

- **Labeling** (`trialspace.label`; the labeler and its local workspace are `matchminer_ai.labeling`) runs on local
  disk by default (`/tmp/mmai-labels/<hash>`, or `MMAI_LABEL_SCRATCH_DIR`; 16 GiB reserve;
  `MMAI_LOCAL_LABEL_STORAGE=0` disables it). Concurrency per server starts at 96 and can reach 768
  (`--label-initial-concurrency-per-server`, `--label-concurrency-per-server`). Status is mirrored to the output
  directory every 10 seconds. On completion, `scores.jsonl` (match labels only), `labels.jsonl`, the response database
  (prompts, answers and reasoning) and manifests are published with `storage_complete.json`; a failed run keeps its
  scratch for resume.
- **Retrieval** (`trialspace.mine`, `trialspace.retrieval_workspace`) works in `/tmp/mmai-retrieval`
  (`--retrieval-scratch-dir`; network filesystems are rejected), publishes candidates, rankings, encodings and
  `complete.json` to `--output`, then removes the scratch. Unchanged dense encodings can be reused across merged
  corpora (`trialspace.retrieval_cache_import`); rankings never are.

Models and data are published only by size-checked copies to shared `no_phi` storage. The package has no Hugging Face
upload code.

## Module map

| Module | Role |
| --- | --- |
| `training.pipeline`, `training.from_summaries` | Orchestration (`mmai-train`): stage ranges, fingerprints, resume |
| `training.trials` | ctgov parsing, screening, sampling, ctgov cohort extraction, synthetic protocols and their cohorts |
| `training.synthetic` | Synthetic patients: seeds, event histories, notes, note preparation, the pooled corpus |
| `training.summaries` | Serial summarization of the synthetic notes |
| `training.trialspace` | TrialSpace corpus, mining, labeling, projection, fits, stage receipts (`trialspace.stage`) |
| `training.decision_model` | Decision-model data and fine-tuning |
| `training.oncoreasoning` | OncoReasoning data, training and evaluation |
| `training.cohort_format`, `training.paths`, `training.vllm_reasoning` | Shared cohort table rules, data paths and run identities, vLLM helpers |

## Tests

`tests/training` covers synthetic generation and validation, cohort extraction with a fake LLM, pipeline planning,
resume and wiring, note preparation, CSV literals, trial versions, decision data, label contracts, OncoReasoning
packing and the embedding losses; the TrialSpace fit tests train a tiny random model.

## Data and clinical safety

Training stages read and write only under the data root's `no_phi` tree and refuse other directories. Do not point
them at real clinical data.
