# Synthetic trial protocols

ClinicalTrials.gov (ctgov) eligibility text is much less granular than the eligibility sections of full protocols.
This workflow generates a corpus of synthetic full protocol documents, one per ctgov seed, whose eligibility reads like
real protocols: often messily specified, with one or more protocol-defined cohorts, parts or arms. The corpus is
training material for cohort and criteria extraction and matching; it is **not** a set of real trial criteria.

## Components

| Piece | Location |
| --- | --- |
| Generation API | `matchminer_ai.trials.generate_synthetic_protocols` |
| Prompts | `matchminer_ai/prompts/trial.synthetic_protocol.system.txt`, `trial.synthetic_protocol.user.txt` |
| Eligibility layouts | `matchminer_ai/data/synthetic_protocol_layouts.json` |
| Config | `synthetic_protocol_generation` section of the default preset |
| Batch driver | `matchminer_ai.training.trials.synthetic_protocols.generate_batch` |
| Cohort extraction API | `matchminer_ai.trials.extract_protocol_cohorts`, `merge_equivalent_cohorts`, `remove_boilerplate_criteria` |
| Cohort extraction driver | `matchminer_ai.training.trials.synthetic_protocols.extract_cohorts` |
| Patient-generation cohort table | `matchminer_ai.training.trials.synthetic_protocols.patient_cohorts` |
| ctgov-version cohorts | `matchminer_ai.training.trials.extract_ctgov_cohorts --trials-jsonl` on the batch's `seeds.jsonl` |
| Retrieval cohort table (both versions) | `matchminer_ai.training.trials.synthetic_protocols.retrieval_cohorts` |

## What the prompt does

The ctgov record (official title, brief summary, detailed description and eligibility criteria) is a **seed, not a
specification**: the protocol may refine or differ from it, as a real protocol does from its ctgov entry. The prompt
asks for a complete sponsor-style protocol in Markdown, written prospectively (no enrollment status or results), with
cohorts defined in the study design, eligibility detail scaled to the trial's kind and sponsor, sponsor texture
(introductory sentence, NOTE lines, cross-references to sections that exist), label-derived criteria for every study
agent, and no statement that the document is synthetic. Eligibility details from the record stay in the study
population section and synopsis.

Each trial's eligibility layout is chosen by a stable hash of its NCT ID, with weights from a survey of 59 real
multi-cohort protocols:

| Weight | Layout |
| --- | --- |
| 20 (~65%) | One inclusion and one exclusion list; differences between parts/cohorts inside criteria (disease-keyed sub-items, part-specific thresholds, NOTEs) |
| 9 (~29%) | One list each; only criteria limited to some parts/cohorts carry a short qualifier |
| 1 (~3%) | A "disease-specific inclusion criteria" subsection, then other criteria |
| 1 (~3%) | Core criteria plus additional criteria in later part/module sections |

Prompt headings never sort criteria into "common" and "cohort-specific" sets, which would mirror the extraction
schema.

## Running a batch

```bash
B=<data root>/no_phi/synthetic_protocols/my_batch
python -m matchminer_ai.training.trials.synthetic_protocols.generate_batch \
  --nct-ids-from <data root>/no_phi/ctgov_sample_trials.csv \
  --output-dir "$B" --chunk-size 500 \
  --server http://localhost:8000/v1=256 --server http://localhost:8001/v1=32
```

- **Seeds:** `--nct-ids-from` fetches ctgov records for the CSV's `nct_id` column once into `<output-dir>/seeds.jsonl`
  (with `retrieved_at`); `--seeds FILE` uses an existing JSON-lines file instead. `--limit N` runs the first N seeds
  for a smoke test.
- **Servers:** on the first run, `--server URL=CAP` entries (cap 64 when omitted) are written to
  `<output-dir>/servers.json`, each with the model ID the server reports, so servers serving different builds (for
  example NVFP4 and FP8) can be mixed. The file is re-read about every second: remove or add an entry, or change its
  `upstream_concurrency` cap, without restarting. Later runs reuse the file and ignore `--server`;
  `--server-urls-file` uses an existing registry instead.
- **Concurrency:** the package's endpoint pool runs adaptive concurrency per server (starting at 32, growing in
  saturated successful windows up to the cap, halving under vLLM load pressure read from `/metrics`). Each request is
  sent on its own, and another starts as soon as one returns, keeping `--chunk-size` x `chunks_in_flight` (500 x 2 by
  default) requests in flight, so one slow protocol holds up only itself.
- **Sampling:** the runtime model name (`--model-name`, default Nemotron 3.5 Super VL NVFP4) selects the package's
  sampling profile (temperature 1.0, top-p 0.95, thinking on); `--max-tokens` defaults to 100,000.
- **Stopping and resuming:** stop with `kill <pid>`; rerun the same command. A trial is skipped when its `meta.json`
  records `finish_reason` `stop` and the current prompt hash, so a prompt or layout change regenerates everything.
  Requests still in flight when a run stops are lost and repeated. `--allow-failures` exits 0 when some trials fail
  every attempt.

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

Empty or truncated outputs (Nemotron occasionally returns reasoning with no answer) are retried up to `max_attempts`
(3); trials that still fail are listed in `run_metadata.json`.

## Extracting cohorts and eligibility criteria

```bash
python -m matchminer_ai.training.trials.synthetic_protocols.extract_cohorts --batch-dir "$B" \
  --server http://localhost:8000/v1=256 --server http://localhost:8001/v1=32
```

The driver takes every protocol in the batch whose `meta.json` records a completed generation, checks that its seed
matches `seeds.jsonl`, and refuses a batch that mixes generation prompt versions. Each whole Markdown document goes to
`extract_protocol_cohorts`, which runs the same three steps as ctgov listings, with the same prompts
(`trial.cohorts.screen.user.txt`, `trial.cohorts.list.user.txt`, `trial.cohorts.criteria.user.txt`): a
cancer-treatment screen, the cohort names, and one request per cohort for its complete criteria, copied verbatim with
boilerplate included. Answers are constrained to each step's JSON schema (default model Nemotron 3.5 Super VL NVFP4,
50,000 completion tokens). Then, for training:

1. **Merge** (`merge_equivalent_cohorts`; `--no-merge-cohorts` skips it): a trial's cohorts whose criteria are
   essentially the same are combined.
2. **Boilerplate** (`remove_boilerplate_criteria`; `--no-deboilerplate` skips it): each line of a merged cohort's
   criteria is classified in its own request (inclusion or exclusion, boilerplate or not;
   `trial.cohorts.deboilerplate.user.txt`), and code sorts the unchanged lines into key and boilerplate criteria. See
   [How matching works](../user-guide/patient-centric-matching.md#key-and-boilerplate-criteria).

Each step keeps `--chunk-size` x `--chunks-in-flight` requests in flight (500 x 2 = 1,000 by default;
`mmai-train --protocol-cohort-chunks-in-flight` defaults to 16, 8,000 requests, for a GPU fleet), sending each request
on its own as another returns. Servers, adaptive concurrency, stopping and resuming work as for generation: each
step's result is saved when its trial is decided and reused on rerun when the prompt, schema and input are unchanged.
Output goes to `<batch-dir>/cohort_extraction/` (or `--output-dir`):

```text
cohort_extraction/
  servers.json           live endpoint registry
  trials/<NCT>/
    extraction.json      parsed decision, cohort names, attempts with endpoint and
                         model, prompt hash, schema, protocol SHA-256
    reasoning.json       each step's reasoning
    merge.json           a multi-cohort trial's merge grouping, with the answer and
                         reasoning
    deboilerplate.json   each merged cohort's criteria lines with their tags (answer and
                         reasoning kept per line) and the two lists code sorted them
                         into (key_criteria, boilerplate_criteria)
  cohorts_unmerged.parquet, .csv
                         one row per extracted cohort (criteria verbatim, boilerplate
                         included), with provenance
  cohorts.parquet, .csv  one row per cohort after merging (merged_from: the extracted
                         cohort numbers), boilerplate included (they seed patients),
                         with provenance
  cohorts_deboilerplated.parquet, .csv
                         the merged cohorts with their criteria separated (retrieval's
                         protocol version): cohort_eligibility_criteria keeps all the
                         criteria, cohort_key_criteria holds the key criteria,
                         cohort_boilerplate_criteria the boilerplate and
                         cohort_criteria_items every line with its tags
  trials.parquet, .csv   one row per protocol: provenance and extraction status
  run_metadata.json      the checkout's git commit, settings, servers, status counts,
                         failures, seeds without a protocol, merge and boilerplate
                         summaries
  trial_attempts.json    every attempt, including failed trials
```

Each row in `cohorts` and `trials` carries its provenance:

| Columns | Source |
| --- | --- |
| `protocol_origin` (`synthetic`), `protocol_path`, `protocol_sha256` | The synthetic protocol |
| `synthetic_prompt_sha256`, `synthetic_layout_index`, `synthetic_attempt`, `synthetic_completed_at`, `synthetic_endpoint`, `synthetic_model` | Its generation (`meta.json`) |
| `ctgov_title`, `ctgov_brief_summary`, `ctgov_detailed_description`, `ctgov_eligibility_criteria`, `ctgov_last_update_post_date`, `ctgov_retrieved_at` | The ctgov seed record, including its original eligibility criteria |
| `extraction_status`, `cohort_count`, `extraction_error`, `extraction_attempts`, `extraction_endpoint`, `extraction_model`, `extraction_prompt_sha256`, `extraction_schema_version`, `extraction_decision_path` | The extraction |

`cohorts` also has the package's cohort columns (`COHORT_COLUMNS`, with `cohort_criteria_source` = `protocol`).
`extraction_status` is `cohorts`, `not_cancer_treatment` or `failed`; only `cohorts` trials have cohort rows.

## Patient-generation input

```bash
python -m matchminer_ai.training.trials.synthetic_protocols.patient_cohorts --batch-dir "$B" \
  --output <data root>/no_phi/<generation-run>/sample_trial_cohort_lineitems.csv \
  --retrieval-output <data root>/no_phi/<generation-run>/sample_trial_cohort_lineitems_deboilerplated.csv \
  --splits-from <data root>/no_phi/ctgov_sample_trials.csv
```

`trials.synthetic_protocols.patient_cohorts` turns an extraction's `cohorts.parquet` (the merged cohorts with all their
criteria, boilerplate included; `--cohorts-file`; `--extraction-dir`, default `<batch-dir>/cohort_extraction/`) into
the cohort table that `synthetic.generate` reads (`mmai-train --sampled-cohorts`):

- accepted cohorts in seed order with a contiguous `cohort_index`;
- each trial's source `split`, copied from the ctgov sample it was seeded from (`--splits-from`);
- every provenance column.

Rows keep `cohort_criteria_source` `protocol`, so the generator uses the protocol's criteria; the table has no
`eligibility_criteria` column, and the ctgov text stays in `ctgov_eligibility_criteria`. The module checks that the
generator's own adapter reproduces the extracted criteria exactly, and writes `<output>.provenance.json` (source
stamps, counts, failures, prompt hashes). `--retrieval-output` also writes the same rows, in the same order with the
same `cohort_index`, from `cohorts_deboilerplated.parquet` (`--deboilerplated-file`), with its own receipt: the
protocol version that `trials.synthetic_protocols.retrieval_cohorts` reads.

## In the training pipeline

`mmai-train` runs this track as stages after `sample-trials`:

| Stage | Module |
| --- | --- |
| `synthetic-protocols` | `trials.synthetic_protocols.generate_batch --nct-ids-from` the ctgov sample |
| `protocol-cohorts` | `trials.synthetic_protocols.extract_cohorts` |
| `ctgov-cohorts` | `trials.extract_ctgov_cohorts --trials-jsonl` on the batch's `seeds.jsonl` |
| `generation-cohorts` | `trials.synthetic_protocols.patient_cohorts` (the patient-generation input) |
| `retrieval-cohorts` | `trials.synthetic_protocols.retrieval_cohorts` (the `train` stage's cohort table) |

There, generation and protocol extraction use the pipeline's `--server-urls-file` instead of writing their own
`servers.json`, and `--allow-failures` lets a stage finish when a few trials fail every attempt; those trials keep
whichever version they have.

## ctgov versions and the retrieval cohort table

Users may be allowed to send some trials' full protocols to an LLM and only the ctgov text of others, so TrialSpace
and the downstream models are trained on both representations. Every sampled trial is therefore also extracted from
the ctgov record its protocol was written from, with the same three-step workflow and prompts (schema
`trial-cohorts-stepwise-v1`), the listing's title, brief summary and eligibility criteria forming one document. The
ctgov cohorts get the same merge and boilerplate steps as the protocol cohorts. Retrieval, the match labels and the
decision model read both versions' key criteria, and the boilerplate question their boilerplate criteria
(`--output-deboilerplated-cohorts`, and `patient_cohorts --retrieval-output` for the protocol version). Patients are
generated from the merged protocol cohorts only, with all their criteria.

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
  --ctgov-trials <run>/ctgov_cohorts/trials_with_cohorts.csv --output <run>/retrieval_cohorts.csv \
  --splits-from <data root>/no_phi/ctgov_sample_trials.csv
```

`--trials-jsonl` extracts every record, without the screen's ctgov rule or split assignment. `--allow-failures`
exports even when a trial fails every attempt (that trial keeps only its protocol version).
`trials.synthetic_protocols.retrieval_cohorts` keeps the protocol rows unchanged (IDs `<NCT>-cohort-<n>`) and adds the
ctgov rows with IDs `<NCT>-ctgov-cohort-<n>`, `cohort_criteria_source` `ctgov`, their ctgov fields as `ctgov_*`,
`cohort_index` continuing after the protocol rows, and the trial's split from `--splits-from`. A trial accepted in only
one version keeps that one. This table, not the patient-generation table, is the `train` stage's cohort input
(`training.from_summaries --trial-cohorts`), where `trialspace.prepare` gives each cohort its key criteria as `text`
and its boilerplate criteria as `boilerplate`.

Mining then represents each trial by one version (see [Training](index.md)): each patient searches one version of
each of its sampled trials, chosen at random per patient and trial, and each trial queries for patients with one
version, chosen at random per trial. A patient's pairs with its own source trial include both versions.

## Caveats

- These are fabricated documents. Label any dataset built from them as synthetic; the documents themselves
  deliberately carry no disclaimer.
- Seeds sent to the generation endpoints are public ctgov text. Do not add patient data to seeds.
- A batch can mix model builds (NVFP4 and FP8); each protocol's `meta.json` records the endpoint and model that
  produced it.
- Single-cohort seeds sometimes gain dose-escalation or expansion parts, which shifts the cohort-count distribution
  relative to the seeds.
- Training uses this synthetic corpus, not real protocols. Real protocol PDFs posted on ClinicalTrials.gov are for
  comparison and evaluation only; check their licensing before redistributing them.
- Extracted criteria come from the synthetic protocol, not the ctgov record; the `ctgov_*` columns are the seed,
  which the protocol may refine or differ from.
