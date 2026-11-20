#!/usr/bin/env python3
"""Reduce both judges' answers to units, compare them and build the adjudication queue (standard library only).

Rerun it whenever new answers arrive (for example when the Azure judge finishes); adjudications already made are
kept (they are keyed by unit id) and remember which judge values they were made against.

A unit is one categorical judgment (a summary statement's error level, a criteria line's boilerplate tag, one
decision question's yes/no). Its status:
  agreed     both judges gave the same compared value (some kinds compare a collapse, e.g. major vs not major)
  disputed   both answered and differ                      -> queued
  single     only one judge answered; the other refused, was truncated or was invalid twice  -> queued
  awaiting   only one judge answered and the other has not run yet (not queued until it has)
  missing    neither answered
A seeded ``--audit-fraction`` of agreed units is queued too, to estimate how often the judges agree and are both
wrong. Agreed "nothing missing" units (omissions per section, missing criteria per cohort) are not audited: they
cannot be checked without rereading the whole source.
"""
import argparse
from pathlib import Path
import random

from matchminer_ai.evaluation.dual_judge.judging import (COMPONENTS, JUDGES, component_module, compared, log, private, read_jsonl, unit_status,  # noqa: E402
                     write_json, write_jsonl)

NOT_AUDITED = {('omission', 'none'), ('missing', 'none')}


def current_requests(run_dir, component):
    """{request_id: item_id} of the component's current requests (a rebuilt component supersedes earlier ones)."""
    return {r['request_id']: r['item_id'] for r in read_jsonl(Path(run_dir) / 'requests' / f'{component}.jsonl')}


def final_answers(run_dir, judge, component):
    """{item_id: latest final row} of one judge's journal, for the items' current requests only (rows that are not
    final are retried transport errors; answers to superseded requests are kept in the journal but not used)."""
    current, latest = current_requests(run_dir, component), {}
    for row in read_jsonl(Path(run_dir) / 'responses' / judge / f'{component}.jsonl'):
        if row.get('final') and current.get(row['request_id']) == row['item_id']:
            latest[row['item_id']] = row
    return latest


def compare(run_dir, audit_fraction=0.10, seed=42):
    run_dir = Path(run_dir)
    units, summary = [], dict(components={})
    for component in COMPONENTS:
        items = read_jsonl(run_dir / 'items' / f'{component}.jsonl')
        if not items:
            continue
        module = component_module(component)
        answers = {judge: final_answers(run_dir, judge, component) for judge in JUDGES}
        statuses = {judge: {} for judge in JUDGES}
        for item in items:
            per_judge, item_status = {}, {}
            for judge in JUDGES:
                row = answers[judge].get(item['item_id'])
                status = item_status[judge] = row['status'] if row else 'not_run'
                statuses[judge][status] = statuses[judge].get(status, 0) + 1
                per_judge[judge] = module.units(item, row['answer']) if row and row['status'] == 'ok' else {}
            keys = list(dict.fromkeys(key for judge in JUDGES for key in per_judge[judge]))
            for key in keys:
                kind = next(per_judge[j][key][0] for j in JUDGES if key in per_judge[j])
                values = {j: per_judge[j][key][1] if key in per_judge[j] else None for j in JUDGES}
                comp = {j: None if v is None else compared(module.KINDS, kind, v) for j, v in values.items()}
                status = unit_status(comp)
                if status == 'single' and any(comp[j] is None and item_status[j] == 'not_run' for j in JUDGES):
                    status = 'awaiting'
                units.append(dict(
                    unit_id=f'{component}:{item["item_id"]}:{key}', component=component, item_id=item['item_id'],
                    key=key, kind=kind, values=values, compared=comp, status=status,
                    details={j: per_judge[j][key][2] if key in per_judge[j] else None for j in JUDGES},
                    stratum=item.get('stratum'), weight=item.get('weight', 1.0)))
        summary['components'][component] = dict(items=len(items), judge_status=statuses)
    rng = random.Random(f'{seed}:audit')
    for unit in units:
        if unit['status'] in ('disputed', 'single'):
            unit['queued'], unit['queue_reason'] = True, unit['status']
        elif unit['status'] == 'agreed' and (unit['kind'], next(iter(v for v in unit['compared'].values() if v))) \
                not in NOT_AUDITED and rng.random() < audit_fraction:
            unit['queued'], unit['queue_reason'] = True, 'audit'
        else:
            unit['queued'], unit['queue_reason'] = False, None
    for component, entry in summary['components'].items():
        mine = [u for u in units if u['component'] == component]
        entry['units'] = {status: sum(u['status'] == status for u in mine)
                          for status in ('agreed', 'disputed', 'single', 'awaiting', 'missing')}
        entry['queued'] = {reason: sum(u['queue_reason'] == reason for u in mine)
                           for reason in ('disputed', 'single', 'audit')}
        entry['queued_items'] = len({u['item_id'] for u in mine if u['queued']})
    summary.update(audit_fraction=audit_fraction, seed=seed, units=len(units), queued=sum(u['queued'] for u in units))
    write_jsonl(run_dir / 'compare' / 'units.jsonl', units)
    write_json(run_dir / 'compare' / 'summary.json', summary)
    return units, summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-dir', required=True, type=Path)
    p.add_argument('--audit-fraction', type=float, default=0.10)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args(argv)
    private()
    _, summary = compare(args.run_dir, args.audit_fraction, args.seed)
    for component, entry in summary['components'].items():
        log(f'{component}: {entry["items"]} items; judges {entry["judge_status"]}; units {entry["units"]}; '
            f'queued {entry["queued"]} in {entry["queued_items"]} items')
    log(f'{summary["queued"]} of {summary["units"]} units queued for adjudication')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
