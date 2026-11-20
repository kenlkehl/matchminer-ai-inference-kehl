# Synthetic trial protocols

ClinicalTrials.gov eligibility text is much less granular than the eligibility
sections of full protocols. This workflow generates a corpus of synthetic full
protocol documents, one per ctgov seed, whose eligibility reads like real
protocols: often messily specified, with one or more protocol-defined cohorts,
parts or arms. The corpus is training material for cohort and criteria
extraction and matching; it is **not** a set of real trial criteria.

## Components

| Piece | Location |
| --- | --- |
| Generation API | `matchminer_ai.trials.generate_synthetic_protocols` (inference package) |
| Prompt (v11) | `matchminer_ai/prompts/trial.synthetic_protocol.system.txt`, `trial.synthetic_protocol.user.txt` |
| Eligibility layouts | `matchminer_ai/data/synthetic_protocol_layouts.json` |
| Config | `synthetic_protocol_generation` section of the inference preset |
| Batch driver | `synthetic_protocols/`trials.synthetic_protocols.generate_batch` (this repository) |
| Cohort extraction API | `matchminer_ai.trials.extract_protocol_cohorts` (inference package) |
| Cohort extraction driver | `synthetic_protocols/`trials.synthetic_protocols.extract_cohorts` (this repository) |
| ctgov-version cohorts | `trials.extract_ctgov_cohorts` --trials-jsonl` on the batch's `seeds.jsonl` (this repository) |
| Retrieval cohort table (both versions) | `synthetic_protocols/`trials.synthetic_protocols.retrieval_cohorts` (this repository) |
| Real protocol PDFs (comparison data) | `scripts/download_ctgov_protocols.py` |

Requires the inference package at commit `acb37a0` or later (overlapping
chunks); cohort extraction needs `cbd7aff` or later (saved, resumable decisions
and Markdown input).

## What the prompt does

The ctgov record (official title, brief summary, detailed description and
eligibility criteria) is a **seed, not a specification**: the protocol may refine
or differ from it, as a real protocol does from its ctgov entry. The prompt
asks for a complete sponsor-style protocol in Markdown, written prospectively
(no enrollment status or results), with cohorts defined in the study design,
eligibility detail scaled to the trial's kind and sponsor, sponsor texture
(introductory sentence, NOTE lines, cross-references to sections that exist),
label-derived criteria for every study agent, and no statement that the document
is synthetic. Eligibility details from the record stay in the study population
section and synopsis.

Each trial's eligibility layout is chosen by a stable hash of its NCT ID with
weights from a survey of 59 real multi-cohort protocols:

| Weight | Layout |
| --- | --- |
| 20 (~65%) | One inclusion and one exclusion list; differences between parts/cohorts inside criteria (disease-keyed sub-items, part-specific thresholds, NOTEs) |
| 9 (~29%) | One list each; only criteria limited to some parts/cohorts carry a short qualifier |
| 1 (~3%) | A "disease-specific inclusion criteria" subsection, then other criteria |
| 1 (~3%) | Core criteria plus additional criteria in later part/module sections |

Prompt headings never sort criteria into "common" and "cohort-specific" sets,
which would mirror the extraction schema. The archived training repository's `synthetic_protocols/PROMPT_HISTORY.md` records the
v1–v11 iterations and the checks behind them.

## Running a batch

```bash
B=<data root>/no_phi/synthetic_protocols_from_registry/batch_v11_sample5000_20261029
python -m matchminer_ai.training.trials.synthetic_protocols.generate_batch \
  --nct-ids-from <data root>/no_phi/sample_trial_cohort_lineitems.csv \
  --output-dir "$B" --chunk-size 500 \
  --server http://teacher-a:8000/v1=256 --server http://teacher-b:8000/v1=32
```

- **Seeds:** `--nct-ids-from` fetches ctgov records for the CSV's `nct_id`
  column once into `<output-dir>/seeds.jsonl` (with `retrieved_at`);
  `--seeds FILE` uses an existing JSON-lines file instead. `--limit N` runs the
  first N seeds for a smoke test.
- **Servers:** on the first run `--server URL=CAP` entries are written to
  `<output-dir>/servers.json`, each with the model ID the server reports, so
  servers serving different builds (for example NVFP4 and FP8) can be mixed. The
  file is re-read about every second: remove or add an entry, or change its
  `upstream_concurrency` cap, without restarting. Later runs reuse the file and
  ignore `--server`.
- **Concurrency:** the inference package's endpoint pool runs adaptive
  concurrency per server (starting at 32, growing in saturated successful
  windows up to the cap, halving under vLLM load pressure read from `/metrics`).
  The inference package sends each request on its own and starts another as
  soon as one returns, keeping `--chunk-size` x `chunks_in_flight` (500 x 2 by
  default) requests in flight, so one slow protocol holds up only itself.
- **Sampling:** the runtime model name (`--model-name`, default Nemotron 3.5
  Super VL NVFP4) selects the package's sampling profile (temperature 1.0,
  top-p 0.95, thinking on); `--max-tokens` defaults to 100,000.
- **Stopping and resuming:** stop with `kill <pid>`; rerun the same command. A
  trial is skipped when its `meta.json` records `finish_reason` `stop` and the
  current prompt hash, so a prompt or layout change regenerates everything.
  Requests still in flight when a run stops are lost and repeated.

## Outputs

```text
<output-dir>/
  seeds.jsonl            ctgov seeds and their retrieval time
  servers.json           live endpoint registry
  protocols/<NCT>/
    protocol.md          the synthetic protocol
    reasoning.txt        the model's reasoning trace
    meta.json            prompt hash, layout index, attempt, finish reason,
                         serving endpoint and model, seed fields, completion time
  protocols.csv          one row per generated trial (written at the end)
  run_metadata.json      errors, skipped count, model metadata
  trial_attempts.json    every attempt with its finish reason and endpoint
```

Empty or truncated outputs (Nemotron occasionally returns reasoning with no
answer) are retried up to `max_attempts` (3); trials that still fail are listed
in `run_metadata.json`.

## Extracting cohorts and eligibility criteria

```bash
python -m matchminer_ai.training.trials.synthetic_protocols.extract_cohorts --batch-dir "$B" \
  --server http://teacher-a:8000/v1=256 --server http://teacher-b:8000/v1=32
```

The driver takes every protocol in the batch whose `meta.json` records a
completed generation, checks that its seed matches `seeds.jsonl`, and sends the
whole Markdown document to `extract_protocol_cohorts` with the full-protocol
prompt and JSON-schema structured outputs (default model Nemotron 3.5 Super VL
NVFP4, 50,000 completion tokens). Each step (extraction, merge, boilerplate) keeps
`--chunk-size` x `--chunks-in-flight` requests in flight (500 x 2 = 1,000 by default;
`training.pipeline` --protocol-cohort-chunks-in-flight` defaults to 16, 8,000 requests, for the
GPU fleet), sending each request on its own as another returns, so one slow answer
never holds up a batch. It refuses a batch that mixes generation prompt versions. Servers, adaptive concurrency,
stopping and resuming work as for generation: each accepted decision is saved
when its trial is decided and reused on rerun when the extraction prompt, schema and
protocol bytes are unchanged. Output goes to `<batch-dir>/cohort_extraction/`
(or `--output-dir`):

```text
cohort_extraction/
  servers.json           live endpoint registry
  trials/<NCT>/
    extraction.json      raw answer, parsed decision, attempts with endpoint and
                         model, prompt hash, schema, protocol SHA-256
    reasoning.json       each step's reasoning
    merge.json           a multi-cohort trial's merge grouping, with the answer and
                         reasoning (reused on rerun)
    deboilerplate.json   each merged cohort's criteria lines, each classified as
                         boilerplate or not in its own request (answer and reasoning
                         kept; reused per unchanged cohort), and the two lists code
                         sorted them into (key_criteria, boilerplate_criteria; earlier
                         runs named the key list eligibility_criteria)
  cohorts_unmerged.parquet, .csv
                         one row per extracted cohort (criteria verbatim, boilerplate
                         included), with provenance
  cohorts.parquet, .csv  one row per cohort after merging those with essentially the
                         same criteria (merged_from: the extracted cohort numbers),
                         boilerplate included (they seed patients), with provenance
  cohorts_deboilerplated.parquet, .csv
                         the merged cohorts with their criteria separated (retrieval's
                         protocol version): cohort_eligibility_criteria keeps all the
                         criteria, cohort_key_criteria holds the key criteria (those that
                         are not boilerplate; retrieval, match labels and the decision model's match questions
                         read them) and cohort_boilerplate_criteria the boilerplate
                         (the boilerplate labels read it: consent, compliance, samples/tissue/biopsy, measurable
                         or evaluable disease, prognosis, performance status, organ
                         function, brain metastases, toxicities of earlier treatment,
                         washout periods of 30 days or less from radiation, systemic or trial
                         therapy, recovery from recent major surgery, non-cancer conditions,
                         unrelated other cancers, participation in
                         another trial at the same time, contraception/pregnancy/lactation, vaccines,
                         staff and catch-all exclusions, each unless central to the trial)
  trials.parquet, .csv   one row per protocol: provenance and extraction status
  run_metadata.json      git commits of both repositories, settings, servers,
                         status counts, failures, seeds without a protocol
  trial_attempts.json    every attempt, including failed trials
```

Each row in `cohorts` and `trials` carries its provenance:

| Columns | Source |
| --- | --- |
| `protocol_origin` (`synthetic`), `protocol_path`, `protocol_sha256` | The synthetic protocol |
| `synthetic_prompt_sha256`, `synthetic_layout_index`, `synthetic_attempt`, `synthetic_completed_at`, `synthetic_endpoint`, `synthetic_model` | Its generation (`meta.json`) |
| `ctgov_title`, `ctgov_brief_summary`, `ctgov_detailed_description`, `ctgov_eligibility_criteria`, `ctgov_last_update_post_date`, `ctgov_retrieved_at` (`ctgov_*` before 2026-11-05) | The ClinicalTrials.gov seed record, including its original eligibility criteria |
| `extraction_status`, `cohort_count`, `extraction_error`, `extraction_attempts`, `extraction_endpoint`, `extraction_model`, `extraction_prompt_sha256`, `extraction_schema_version`, `extraction_decision_path` | The extraction |

`cohorts` also has the inference package's cohort columns (`COHORT_COLUMNS`,
`cohort_criteria_source` = `protocol`). `extraction_status` is `cohorts`,
`not_cancer_treatment` or `failed`; only `cohorts` trials have cohort rows.

## Patient-generation input

```bash
python -m matchminer_ai.training.trials.synthetic_protocols.patient_cohorts --batch-dir "$B" \
  --extraction-dir "$B/cohort_extraction_v2" \
  --output <data root>/no_phi/<generation-run>/sample_trial_cohort_lineitems.csv \
  --retrieval-output <data root>/no_phi/<generation-run>/sample_trial_cohort_lineitems_deboilerplated.csv
```

`trials.synthetic_protocols.patient_cohorts` turns an extraction's `cohorts.parquet` (the merged cohorts with all
their criteria, boilerplate included, since 2026-11-04 by user instruction; `--cohorts-file`;
`--extraction-dir`, default `<batch-dir>/cohort_extraction/`) into the cohort
table that `generate_synthetic_data.py` (`training.pipeline` --sampled-cohorts`)
reads: accepted cohorts in seed order with a contiguous `cohort_index`, each
trial's source `split` copied from the ctgov sample it was seeded from
(`--splits-from`, default `<data root>/no_phi/sample_trial_cohort_lineitems.csv`),
and every provenance column. Single-cohort rows keep `cohort_criteria_source`
`protocol`, so the generator uses the protocol's criteria; the table has no
`eligibility_criteria` column, and the ctgov text stays in
`ctgov_eligibility_criteria`. The script checks that the generator's own
adapter reproduces the extracted criteria exactly, and writes
`<output>.provenance.json` (source stamps, counts, failures, prompt hashes).
`--retrieval-output` also writes the same rows, in the same order with the same
`cohort_index`, from `cohorts_deboilerplated.parquet` (`--deboilerplated-file`; with
`cohort_key_criteria` and `cohort_boilerplate_criteria`), with its own receipt: the
protocol version that `trials.synthetic_protocols.retrieval_cohorts` reads.

## In the training pipeline

`training.pipeline` (and the identical `train_all.sh`, `train_all_gcp.sh` and
`temp_train_gcp.sh` wrappers) runs this track as stages after `sample-trials`:
`synthetic-protocols` (`trials.synthetic_protocols.generate_batch` --nct-ids-from` the ctgov sample),
`protocol-cohorts` (`trials.synthetic_protocols.extract_cohorts`), `ctgov-cohorts`
(`trials.extract_ctgov_cohorts` --trials-jsonl` on the batch's `seeds.jsonl`),
`generation-cohorts` (`trials.synthetic_protocols.patient_cohorts`, the patient-generation input) and
`retrieval-cohorts` (`trials.synthetic_protocols.retrieval_cohorts`, the `train` stage's cohort table).
There `trials.synthetic_protocols.generate_batch` and `trials.synthetic_protocols.extract_cohorts` use the pipeline's
`--server-urls-file` instead of writing their own `servers.json`, and
`--allow-failures` lets a stage finish when a few trials fail every attempt;
those trials keep whichever version they have.

## ctgov versions and the retrieval cohort table

Users may be allowed to send some trials' full protocols to an LLM and only the
ClinicalTrials.gov text of others, so TrialSpace and the downstream models are
trained on both representations. Every sampled trial is therefore also
extracted from the ctgov record its protocol was written from. Since
2026-11-03 both extractions run the inference package's three-step workflow with
the same prompts (`trial.cohorts.screen.user.txt`, `trial.cohorts.list.user.txt`,
`trial.cohorts.criteria.user.txt`; schema `trial-cohorts-stepwise-v1`): a
cancer-treatment screen, the cohort names, and one request per cohort for its
complete criteria, with the listing's title, brief summary and eligibility
criteria as one document. Since 2026-11-04 (user instruction) the criteria are
copied verbatim with boilerplate included, and training gives ctgov listings the
same two follow-on steps as protocols: the merge, then the boilerplate step, which
classifies each line of a merged cohort's criteria in its own request and sorts the
unchanged lines in code (since 2026-11-04, user instruction; the model never copies
the criteria). Retrieval, match labels and the decision model read both versions' key
criteria (the criteria that are not boilerplate) and its boilerplate question their
boilerplate criteria (`--output-deboilerplated-cohorts`, and
`trials.synthetic_protocols.patient_cohorts` --retrieval-output` beside the generation table, from
`cohorts_deboilerplated.parquet`: `cohort_key_criteria`, `cohort_boilerplate_criteria`),
while patients are seeded from the merged cohorts with all their criteria (user,
2026-11-04). Earlier runs used
the single-request ctgov prompt (`trial.cohorts.user.txt`, schema
`trial-registry-cohorts-v1`):

```bash
python -m matchminer_ai.training.trials.extract_ctgov_cohorts --trials-jsonl "$B/seeds.jsonl" \
  --work-dir <run>/ctgov_cohorts/work --output-trials <run>/ctgov_cohorts/trials_with_cohorts.csv \
  --output-cohorts <run>/ctgov_cohorts/trial_cohort_lineitems.csv --output-json <run>/ctgov_cohorts/trial_cohorts.json \
  --output-unmerged-cohorts <run>/ctgov_cohorts/trial_cohort_lineitems_unmerged.csv \
  --output-deboilerplated-cohorts <run>/ctgov_cohorts/trial_cohort_lineitems_deboilerplated.csv \
  --model <served alias> --tokenizer-model <model or tokenizer path> --server-urls-file <servers.json> \
  --max-attempts 5 --allow-failures
python -m matchminer_ai.training.trials.synthetic_protocols.retrieval_cohorts \
  --protocol-cohorts <generation-run>/sample_trial_cohort_lineitems_deboilerplated.csv \
  --ctgov-cohorts <run>/ctgov_cohorts/trial_cohort_lineitems_deboilerplated.csv \
  --ctgov-trials <run>/ctgov_cohorts/trials_with_cohorts.csv --output <run>/retrieval_cohorts.csv
```

`--trials-jsonl` extracts every record, without `0b`'s date and phase selection
or split assignment; `--allow-failures` exports even when a trial fails every
attempt (that trial keeps only its protocol version). `trials.synthetic_protocols.retrieval_cohorts`
keeps the protocol rows unchanged and adds the ctgov rows with IDs
`<NCT>-ctgov-cohort-<n>`, `cohort_criteria_source` `ctgov`, their ctgov
fields as `ctgov_*`, and the trial's split from `--splits-from`; a trial
accepted in only one version keeps that one. This table, not the
patient-generation table, is the `--cohorts` input to `trialspace_prepare.py`,
which gives each cohort its key criteria as `text` and its boilerplate
criteria as `boilerplate`. Patients are still generated from the protocol cohorts only.

Mining then represents each trial by one version (main README, Mining): each
patient searches one version of each of its sampled trials, chosen at random
per patient and trial, and each trial queries for patients with one version,
chosen at random per trial. A patient's pairs with its own source trial
include both versions.

## Caveats

- These are fabricated documents. Label any dataset built from them as
  synthetic; the documents themselves deliberately carry no disclaimer.
- Seeds sent to the generation endpoints are public ctgov text. Do not add
  patient data to seeds.
- A batch can mix model builds (NVFP4 and FP8); each protocol's `meta.json`
  records the endpoint and model that produced it.
- Single-cohort seeds sometimes gain dose-escalation or expansion parts, which
  shifts the cohort-count distribution relative to the seeds.
- Training is planned on this synthetic corpus, not on real protocols; real
  protocol PDFs are for comparison and evaluation only. The extracted cohorts
  reach patient generation through `trials.synthetic_protocols.patient_cohorts` and
  `training.pipeline` --sampled-cohorts`, and retrieval together with the
  ctgov versions through `trials.synthetic_protocols.retrieval_cohorts`.
- Extracted criteria come from the synthetic protocol, not the ctgov; the
  `ctgov_*` columns are the seed, which the protocol may refine or differ from.

## Batches

| Directory under `<data root>/no_phi/synthetic_protocols_from_registry/` | Prompt | Status |
| --- | --- | --- |
| `batch_v5_sample5000_20261028/` | v5 | Stopped at 3,861 of 4,981; eligibility structure mirrored the extraction schema (superseded) |
| `batch_v11_sample5000_20261029/` | v11 via the inference package | Current: 4,981 protocols (2026-10-29). `cohort_extraction_v2/` (schema `trial-protocol-cohorts-v2`, `--output-dir`) is current: 7,721 cohorts from 4,848 trials, 132 judged not cancer treatment, 1 failed (duplicate cohort criteria). `cohort_extraction/` (schema v1: 7,716 cohorts from 4,850 trials) is superseded: its `minLength` strings kept vLLM structured outputs from writing any JSON escape, so cohort-specific and single-cohort criteria have no line breaks or double quotes |
