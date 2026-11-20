# Teacher Labeling

`matchminer_ai.labeling` produces resumable teacher labels of patient-cohort pairs for training TrialSpace and the
decision model. The [training pipeline](../training/index.md)'s label stages (`label-trialspace-N`,
`label-boilerplate-N`, `label-criteria-N`; `matchminer_ai.training.trialspace.label`) call it.

Three tasks (`matchminer_ai.labeling.tasks`):

- **`match`:** the five [match labels](match-quality.md#match-labels) of a pair in one request
  (`match_quality.labels.system.txt` on the patient's key summary section and the cohort's key criteria), decoded
  under the match JSON schema and scored 0-6 into a resumable score journal.
- **`boilerplate`:** whether the patient clearly fails the cohort's boilerplate criteria
  (`llm_exclusion_criteria.user.txt`, ending in `Yes!` or `No!`). Pairs whose cohort has no boilerplate criteria are
  skipped. The decision model's boilerplate question learns the inverse: its yes means not excluded.
- **`criteria`:** one request per tagged criteria line of a pair (`criterion_check.labels.user.txt` around the
  decision model's per-criterion state and question), decoded under its schema; yes means the patient does not
  clearly fail the line.

## Inputs and outputs

- **Corpus (`--corpus`):** a directory with `patients.jsonl` and `cohorts.jsonl`, written by the training pipeline.
  Each entity has an `id`, a `split` (`train`, `val` or `test`), its key `text` and its `boilerplate` text; cohorts
  also carry their tagged criteria `items`.
- **Candidates (`--candidates`):** a JSONL file of pairs (`patient_id`, `cohort_id`, `split`; for `criteria`, one
  row per line with its `number`, `criterion_type` and the cohort's `criteria_digest`), built by the training
  pipeline. Pairs stay within their split.
- **Output directory (`--output`):** every answer is kept with its reasoning and complete prompt in `responses/`,
  and answers that fail validation in `invalid_responses/`. `labels.jsonl` is the flattened export. For `match`,
  `scores.jsonl` with `scores_ready.json` holds the complete training scores. `status.json` reports progress.

A run resumes where it stopped. `labeling_manifest.json` records the task, label contract, inputs and teacher
settings, and a run whose teacher identity differs is refused: use a new directory.

Requests go through the adaptive teacher pool (`matchminer_ai.llm.teacher_pool`). Prompts are rendered on the
client with the teacher's chat template (`--teacher-tokenizer`) and sent to `/v1/completions` on a live server
registry, with per-server concurrency that follows vLLM's `/metrics`. Each server must advertise `--teacher-model`
with a root in the teacher's model family (`EQUIVALENT_TEACHER_ROOTS` lists approved quantizations of one model).
The prompts carry patient text to the configured servers: use synthetic or authorized data and endpoints only.

The full list of flags:

```bash
python -c "import argparse; from matchminer_ai.labeling import add_label_cli_args; add_label_cli_args(argparse.ArgumentParser()).print_help()"
```

::: matchminer_ai.labeling.runner
    options:
      members:
        - add_label_cli_args
        - run_label
        - questions

::: matchminer_ai.labeling.tasks
    options:
      members:
        - contract_hash
        - judge_messages
        - parse_judgment
