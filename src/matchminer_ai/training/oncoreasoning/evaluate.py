#!/usr/bin/env python3
"""Evaluate OncoReasoning against the teacher on examples it was not trained on (user, 2026-11-16).

The evaluation set is ``build_data.py --save-text --exclude-from <training data dirs>``: the same verified adapters,
a different seed, and none of the training examples (no synthetic partition is reserved; these are simply examples
the budget did not sample). Every metric compares the student with the teacher's accepted answer, so they are
in-distribution agreement diagnostics, not held-out clinical validation.

    generate  one process per GPU: each example's prompt -> the student's completion (reasoning, </think>, answer)
    score     parse student and teacher answers per task; write report.json

Per task: ``finished`` (the completion closed its reasoning and ended), ``valid`` (the answer parses as that task's
answer), ``agree`` (the decision fields equal the teacher's: screen yes/no; cohort count; merge groups; both
per-line tags; the five match answers, all five and the 0-6 score; boilerplate excluded; per-criterion yes/no),
``token_f1`` (bag-of-words F1 with the teacher's answer: criteria text, cohort names, summaries), and lengths.

    python -m matchminer_ai.training.oncoreasoning.evaluate generate --eval-data EVAL --model MODEL --output OUT [--gpus 8]
    python -m matchminer_ai.training.oncoreasoning.evaluate score --output OUT
"""
import argparse
import collections
import json
import os
from pathlib import Path
import re

MATCH = ('reasonableness', 'cancer_type', 'burden', 'treatment_history', 'biomarker')


def load_eval(eval_dirs):
    rows = []
    for d in eval_dirs:
        for path in sorted(Path(d).glob('parts/*/examples.jsonl')):
            rows.extend(json.loads(line) for line in path.open())
    return rows


def generate_worker(gpu, rows, args, out_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=True)
    tok.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, trust_remote_code=True,
                                                 dtype=torch.bfloat16, attn_implementation=args.attention).cuda().eval()
    done = set()
    if out_path.exists():
        done = {json.loads(line)['id'] for line in out_path.open()}
    rows = [r for r in rows if r['id'] not in done]
    rows.sort(key=lambda r: len(r['prompt']))
    with out_path.open('a') as out:
        i = 0
        while i < len(rows):
            longest = len(tok(rows[i]['prompt'], add_special_tokens=False)['input_ids'])
            budget = max(1, args.batch_tokens // max(1, longest + args.max_new_tokens))
            batch = rows[i:i + min(budget, args.max_batch)]
            i += len(batch)
            enc = tok([r['prompt'] for r in batch], add_special_tokens=False, return_tensors='pt', padding=True).to('cuda')
            new = min(args.max_new_tokens, max(2 * r['completion_tokens'] + 512 for r in batch))
            with torch.no_grad():
                output = model.generate(**enc, max_new_tokens=new, do_sample=args.temperature > 0,
                                        temperature=args.temperature or None, top_k=args.top_k or None,
                                        repetition_penalty=args.repetition_penalty, pad_token_id=tok.pad_token_id,
                                        eos_token_id=tok.convert_tokens_to_ids('<|im_end|>'))
            for r, ids in zip(batch, output[:, enc['input_ids'].shape[1]:]):
                text = tok.decode(ids, skip_special_tokens=False)
                ended = '<|im_end|>' in text
                text = text.split('<|im_end|>')[0].replace('<|pad|>', '')
                out.write(json.dumps(dict(id=r['id'], task=r['task'], completion=text, ended=ended,
                                          new_tokens=int((ids != tok.pad_token_id).sum()))) + '\n')
            out.flush()


def generate(args):
    import multiprocessing
    rows = load_eval(args.eval_data)
    if args.per_task:
        by_task = collections.defaultdict(list)
        for r in rows:
            by_task[r['task']].append(r)
        rows = [r for task_rows in by_task.values() for r in task_rows[:args.per_task]]
    args.output.mkdir(parents=True, exist_ok=True)
    ctx = multiprocessing.get_context('spawn')
    procs = [ctx.Process(target=generate_worker, args=(g, rows[g::args.gpus], args, args.output / f'outputs-{g}.jsonl'))
             for g in range(args.gpus)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    if any(p.exitcode for p in procs):
        raise SystemExit('a generation worker failed')


def split(completion):
    """(reasoning, answer, finished): the student's text before and after </think>."""
    if '</think>' not in completion:
        return completion, '', False
    reasoning, answer = completion.split('</think>', 1)
    return reasoning, answer.strip(), True


def load_json(text):
    text = text.strip()
    fence = re.fullmatch(r'```(?:json)?\s*(.*?)\s*```', text, re.S)
    return json.loads(fence.group(1) if fence else text)


def decision(task, answer):
    """The task's decision fields from one answer; raises on an invalid answer."""
    if task == 'label_boilerplate':
        tail = answer.strip()[-12:]
        if tail.endswith('Yes!'):
            return dict(excluded=True)
        if tail.endswith('No!'):
            return dict(excluded=False)
        raise ValueError('no final Yes!/No!')
    if task == 'summary':
        if not answer.strip():
            raise ValueError('empty summary')
        return dict(text=answer)
    obj = load_json(answer)
    if task == 'label_match':
        labels = obj.get('labels', obj)
        answers = {q: str(labels[q]['answer']).lower() for q in MATCH}
        if not all(a in ('yes', 'no') for a in answers.values()):
            raise ValueError('a match answer is not yes/no')
        return answers
    if task == 'label_criteria':
        a = str(obj.get('answer', obj.get('criterion', {}).get('answer'))).lower()
        if a not in ('yes', 'no'):
            raise ValueError('criterion answer is not yes/no')
        return dict(answer=a)
    if task == 'cohort_screen':
        a = obj['cancer_treatment_trial']
        if a not in ('yes', 'no'):
            raise ValueError('screen is not yes/no')
        return dict(answer=a)
    if task == 'cohort_list':
        names = obj['cohorts']
        if not names or not all(isinstance(n, str) and n.strip() for n in names):
            raise ValueError('empty cohort list')
        return dict(count=len(names), text=' '.join(names))
    if task == 'cohort_criteria':
        text = obj['eligibility_criteria']
        if not isinstance(text, str) or not text.strip():
            raise ValueError('empty criteria')
        return dict(text=text)
    if task == 'cohort_merge':
        return dict(groups=sorted(sorted(int(i) for i in g) for g in obj['groups']))
    if task == 'cohort_deboilerplate':
        if obj['criterion_type'] not in ('inclusion', 'exclusion') or not isinstance(obj['boilerplate'], bool):
            raise ValueError('bad per-line tags')
        return dict(criterion_type=obj['criterion_type'], boilerplate=obj['boilerplate'])
    raise ValueError(f'unknown task {task}')


def score_match(answers):
    if answers['reasonableness'] != 'yes':
        return 0
    return 1 + sum(answers[q] == 'yes' for q in ('cancer_type', 'burden', 'treatment_history')) + 2 * (
        answers['biomarker'] == 'yes')


def token_f1(a, b):
    ta, tb = collections.Counter(re.findall(r'\w+', a.lower())), collections.Counter(re.findall(r'\w+', b.lower()))
    common = sum((ta & tb).values())
    if not common:
        return 0.0
    p, r = common / sum(ta.values()), common / sum(tb.values())
    return 2 * p * r / (p + r)


def score(args):
    references = {r['id']: r for r in load_eval(args.eval_data)} if args.eval_data else {}
    outputs = [json.loads(line) for p in sorted(args.output.glob('outputs-*.jsonl')) for line in p.open()]
    tasks = collections.defaultdict(lambda: collections.defaultdict(list))
    for o in outputs:
        ref = references[o['id']]
        m = tasks[o['task']]
        reasoning, answer, closed = split(o['completion'])
        m['finished'].append(closed and o['ended'])
        m['new_tokens'].append(o['new_tokens'])
        m['teacher_tokens'].append(ref['completion_tokens'])
        teacher = decision(o['task'], ref['answer'])
        try:
            student = decision(o['task'], answer)
        except (ValueError, KeyError, TypeError, AttributeError, json.JSONDecodeError):
            m['valid'].append(False)
            continue
        m['valid'].append(True)
        if o['task'] == 'label_match':
            for q in MATCH:
                m[f'agree_{q}'].append(student[q] == teacher[q])
            m['agree_all5'].append(student == teacher)
            m['agree_score'].append(score_match(student) == score_match(teacher))
        elif o['task'] in ('summary', 'cohort_criteria'):
            m['token_f1'].append(token_f1(student['text'], teacher['text']))
        elif o['task'] == 'cohort_list':
            m['agree_count'].append(student['count'] == teacher['count'])
            m['token_f1'].append(token_f1(student['text'], teacher['text']))
        else:
            m['agree'].append(student == teacher)
    report = {}
    for task, m in sorted(tasks.items()):
        n = len(m['finished'])
        report[task] = dict(n=n, **{k: round(sum(v) / len(v), 4) for k, v in m.items() if v and k not in (
            'new_tokens', 'teacher_tokens')}, mean_new_tokens=round(sum(m['new_tokens']) / n),
            mean_teacher_tokens=round(sum(m['teacher_tokens']) / n))
        report[task]['valid_of_all'] = round(sum(m['valid']) / n, 4)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    g = sub.add_parser('generate')
    g.add_argument('--eval-data', type=Path, nargs='+', required=True)
    g.add_argument('--model', required=True)
    g.add_argument('--revision')
    g.add_argument('--output', type=Path, required=True)
    g.add_argument('--gpus', type=int, default=1)
    g.add_argument('--per-task', type=int, help='At most N examples per task')
    g.add_argument('--attention', default='sdpa')
    g.add_argument('--max-new-tokens', type=int, default=32768)
    g.add_argument('--batch-tokens', type=int, default=1_500_000, help='Prompt + new tokens per batch')
    g.add_argument('--max-batch', type=int, default=64)
    g.add_argument('--temperature', type=float, default=0.0)
    g.add_argument('--top-k', type=int, default=0)
    g.add_argument('--repetition-penalty', type=float, default=1.0)
    s = sub.add_parser('score')
    s.add_argument('--eval-data', type=Path, nargs='+', required=True)
    s.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    (generate if args.command == 'generate' else score)(args)


if __name__ == '__main__':
    main()
