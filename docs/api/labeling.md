# Teacher Labeling

`matchminer_ai.labeling` produces resumable teacher labels of patient-cohort pairs for training TrialSpace and
the decision model. It moved here from the training repository on 2026-11-06, unchanged in behaviour: label
directories written before the move keep resuming, because the label contracts hash to the same values. The
training repository's `trialspace_label.py` stage calls it.

Three tasks (`matchminer_ai.labeling.tasks`):

- `match`: the five match labels of a pair in one request (`match_quality.labels.system.txt` on the patient's
  key summary section and the cohort's key criteria), decoded under the match JSON schema and scored 0-6 into
  a resumable score journal;
- `boilerplate`: whether the patient clearly fails the cohort's boilerplate criteria
  (`llm_exclusion_criteria.user.txt`, ending in `Yes!` or `No!`);
- `criteria`: one request per tagged criteria line of a pair (`criterion_check.labels.user.txt`), decoded
  under its schema; yes means the patient does not clearly fail the line.

Requests go through the adaptive teacher pool (`matchminer_ai.llm.teacher_pool`): prompts rendered on the
client with the teacher's chat template and sent to `/v1/completions` on a live server registry, with
per-server concurrency that follows vLLM's `/metrics`. Every answer is kept with its reasoning and complete
prompt (`responses/`); `labels.jsonl` is the flattened export and, for `match`, `scores.jsonl` with
`scores_ready.json` the complete training scores. The prompts carry patient text to the configured servers:
use synthetic or authorized data and endpoints only.

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
