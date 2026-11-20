"""Resumable teacher labels of patient-cohort pairs, for training TrialSpace and the decision model.

The training pipeline's labeling stages (:mod:`matchminer_ai.training.trialspace.label`) run it. Three tasks
(:mod:`.tasks`):

- ``match``: the five match labels of a pair in one request (``match_quality.labels.system.txt``),
  scored 0-6 into a resumable score journal (:mod:`.journal`, :mod:`.scoring`);
- ``boilerplate``: whether the patient clearly fails the cohort's boilerplate criteria;
- ``criteria``: one request per criteria line of a pair (yes = does not clearly fail it).

Requests go through the adaptive teacher pool (:mod:`matchminer_ai.llm.teacher_pool`), rendered on the
client with the teacher's chat template and decoded under each task's JSON schema where it has one.
Every answer is kept with its reasoning and complete prompt; ``labels.jsonl`` is the flattened export.
Synthetic or authorized data only: the prompts carry patient text to the configured servers.
"""
from .corpus import full_summary, load_corpus
from .scoring import aggregate_pair, score_contract
from .tasks import TASKS, answer_schema, contract_hash, judge_messages, parse_judgment, prompt_sources

# The runner needs the teacher pool's HTTP clients; contracts, scoring and files do not, so environments
# without them (for example an embedding-only one) can still use those.
_RUNNER = ('EQUIVALENT_TEACHER_ROOTS', 'add_label_cli_args', 'label', 'label_key', 'questions', 'run_label',
           'validate_row', 'validate_served_teacher')


def __getattr__(name):
    if name in _RUNNER:
        from . import runner
        return getattr(runner, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


__all__ = ['EQUIVALENT_TEACHER_ROOTS', 'TASKS', 'add_label_cli_args', 'aggregate_pair', 'answer_schema',
           'contract_hash', 'full_summary', 'judge_messages', 'label', 'label_key', 'load_corpus', 'parse_judgment',
           'prompt_sources', 'questions', 'run_label', 'score_contract', 'validate_row', 'validate_served_teacher']
