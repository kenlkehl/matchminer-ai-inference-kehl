#!/usr/bin/env python3
"""Metrics of a dual-judge run: judge agreement and accuracy, and each component's results against the reference
(standard library only). Writes ``metrics/metrics.json`` and ``metrics/report.md``; rerun any time, including on
partial adjudication (pending units are counted and left out).

Reference (gold) per unit: the user's latest adjudication, else the judges' agreed value; disputed or single units
not yet adjudicated are pending. Estimates weight each item by its sampling weight (stratum size / sampled), so
oversampled strata (for example teacher-model disagreements in the decision sample) do not distort rates; 95%
intervals come from a bootstrap over items (one item = one trial, one index event, or one teacher request).
"""
import argparse
from collections import defaultdict
from pathlib import Path
import random
import statistics

from matchminer_ai.evaluation.dual_judge.judging import (COMPONENTS, JUDGES, PRICES, cohen_kappa, gold, latest_decisions, log, private,  # noqa: E402
                     read_jsonl, usage_cost, write_json)
from matchminer_ai.evaluation.dual_judge.compare import current_requests, final_answers  # noqa: E402

DRAWS = 1000


# --- statistics ----------------------------------------------------------------------------------------------------

def weighted_rate(rows):
    """rows: (cluster, weight, 0/1)."""
    total = sum(w for _, w, _ in rows)
    return None if total == 0 else sum(w * y for _, w, y in rows) / total


def cluster_bootstrap(rows, statistic, draws=DRAWS, seed=42):
    """95% interval of ``statistic(rows)`` resampling clusters (the first field of each row)."""
    clusters = defaultdict(list)
    for row in rows:
        clusters[row[0]].append(row)
    keys = sorted(clusters, key=str)
    if len(keys) < 2:
        return None
    rng, values = random.Random(seed), []
    for _ in range(draws):
        sample = [r for k in rng.choices(keys, k=len(keys)) for r in clusters[k]]
        value = statistic(sample)
        if value is not None:
            values.append(value)
    if len(values) < draws // 2:
        return None
    values.sort()
    return [values[int(0.025 * len(values))], values[int(0.975 * len(values)) - 1]]


def rate(rows):
    point = weighted_rate(rows)
    return dict(n=len(rows), items=len({r[0] for r in rows}), rate=point,
                ci95=cluster_bootstrap(rows, weighted_rate) if point is not None else None)


def auroc(rows):
    """Weighted AUROC of rows (cluster, weight, y, p); ties count half."""
    pos = [(w, p) for _, w, y, p in rows if y == 1]
    neg = [(w, p) for _, w, y, p in rows if y == 0]
    total = sum(a for a, _ in pos) * sum(b for b, _ in neg)
    if total == 0:
        return None
    return sum(a * b * (1.0 if p > q else 0.5 if p == q else 0.0) for a, p in pos for b, q in neg) / total


def average_precision(rows):
    """Weighted average precision of rows (cluster, weight, y, p), ranking by p (ties as one group)."""
    positives = sum(w for _, w, y, _ in rows if y == 1)
    if positives == 0:
        return None
    ordered = sorted(rows, key=lambda r: -r[3])
    ap, tp, seen, i = 0.0, 0.0, 0.0, 0
    while i < len(ordered):
        j = i
        while j < len(ordered) and ordered[j][3] == ordered[i][3]:
            j += 1
        group = ordered[i:j]
        tp += sum(w for _, w, y, _ in group if y == 1)
        seen += sum(w for _, w, _, _ in group)
        ap += sum(w for _, w, y, _ in group if y == 1) / positives * (tp / seen)
        i = j
    return ap


def brier(rows):
    total = sum(w for _, w, _, _ in rows)
    return None if total == 0 else sum(w * (p - y) ** 2 for _, w, y, p in rows) / total


def scored(rows, function):
    point = function(rows)
    return dict(value=point, ci95=cluster_bootstrap(rows, function) if point is not None else None)


# --- the run -------------------------------------------------------------------------------------------------------

def judge_costs(run_dir, component):
    out = {}
    for judge in JUDGES:
        current = current_requests(run_dir, component)
        every = read_jsonl(Path(run_dir) / 'responses' / judge / f'{component}.jsonl')
        rows = [row for row in every if row['request_id'] in current]
        superseded = sum(usage_cost(row.get('usage'), row.get('prices') or PRICES[judge])
                         for row in every if row['request_id'] not in current)
        finals = final_answers(run_dir, judge, component)
        if not rows:
            out[judge] = dict(ran=False, superseded_cost_usd=round(superseded, 2))
            continue
        statuses = defaultdict(int)
        for row in finals.values():
            statuses[row['status']] += 1
        seconds = [row['seconds'] for row in rows if row.get('seconds') is not None and row['status'] != 'error']
        cost = sum(usage_cost(row.get('usage'), row.get('prices') or PRICES[judge]) for row in rows)
        out[judge] = dict(ran=True, final=dict(statuses), attempts=len(rows), cost_usd=round(cost, 2),
                          superseded_cost_usd=round(superseded, 2),
                          cost_per_item_usd=round(cost / max(1, len(finals)), 3),
                          median_seconds=statistics.median(seconds) if seconds else None,
                          retried=sum(row['attempt'] == 2 for row in rows),
                          input_tokens=sum((row.get('usage') or {}).get('input_tokens', 0)
                                           + (row.get('usage') or {}).get('cache_read_tokens', 0)
                                           + (row.get('usage') or {}).get('cache_write_tokens', 0) for row in rows),
                          output_tokens=sum((row.get('usage') or {}).get('output_tokens', 0) for row in rows))
    return out


def _count(values):
    out = defaultdict(int)
    for value in values:
        out[value] += 1
    return out


def agreement(units, decisions):
    """Per kind: unit counts by status, agreement and kappa where both answered, each judge's accuracy against the
    reference (all reference units, and adjudicated units only), and the audit's both-wrong rate."""
    out = {}
    for kind in sorted({u['kind'] for u in units}):
        mine = [u for u in units if u['kind'] == kind]
        both = [u for u in mine if all(u['compared'][j] is not None for j in JUDGES)]
        pairs = [(u['compared'][JUDGES[0]], u['compared'][JUDGES[1]]) for u in both]
        entry = dict(units=len(mine), **{s: sum(u['status'] == s for u in mine)
                                         for s in ('agreed', 'disputed', 'single', 'awaiting', 'missing')},
                     queued=sum(u['queued'] for u in mine),
                     adjudicated=sum(u['unit_id'] in decisions for u in mine),
                     pending=sum(u['queued'] and u['unit_id'] not in decisions for u in mine),
                     agreement=(sum(a == b for a, b in pairs) / len(pairs)) if pairs else None,
                     kappa=cohen_kappa(pairs), judges={})
        for judge in JUDGES:
            with_gold, adjudicated = [], []
            for u in mine:
                value, source = gold(u, decisions)
                if value is None or u['compared'][judge] is None:
                    continue
                with_gold.append(u['compared'][judge] == value)
                if source == 'adjudicated':
                    adjudicated.append(u['compared'][judge] == value)
            entry['judges'][judge] = dict(
                accuracy=(sum(with_gold) / len(with_gold)) if with_gold else None, n=len(with_gold),
                accuracy_adjudicated=(sum(adjudicated) / len(adjudicated)) if adjudicated else None,
                n_adjudicated=len(adjudicated))
        entry['distribution'] = {judge: dict(sorted(_count(u['compared'][judge] for u in mine
                                                          if u['compared'][judge] is not None).items()))
                                 for judge in JUDGES}
        audited = [u for u in mine if u['queue_reason'] == 'audit' and u['unit_id'] in decisions]
        entry['audit'] = dict(decided=len(audited), both_wrong=sum(
            decisions[u['unit_id']]['value'] != next(v for v in u['compared'].values() if v is not None) for u in audited))
        out[kind] = entry
    return out


def reference(units, decisions):
    """{unit_id: gold value} for units with a reference."""
    out = {}
    for u in units:
        value, _ = gold(u, decisions)
        if value is not None:
            out[u['unit_id']] = value
    return out


def extraction_results(units, ref):
    def rows(kind, positive):
        return [(u['item_id'], u['weight'], int(ref[u['unit_id']] in positive))
                for u in units if u['kind'] == kind and u['unit_id'] in ref]
    return {
        'screen correct': rate(rows('screen', {'correct'})),
        'cohort set acceptable (correct or defensible)': rate(rows('cohort_set', {'acceptable'})),
        'lines with a fidelity error': rate(rows('fidelity', {'minor', 'major'})),
        'lines with a major fidelity error': rate(rows('fidelity', {'major'})),
        'lines with a wrong key/boilerplate tag': rate(rows('boilerplate_tag', {'should_be_key', 'should_be_boilerplate'})),
        'lines tagged boilerplate that should be key': rate(rows('boilerplate_tag', {'should_be_key'})),
        'lines with a wrong inclusion/exclusion tag': rate(rows('type_tag', {'wrong'})),
        'cohorts missing a major criterion': rate(rows('missing', {'major'})),
        'trials with major issues overall': rate(rows('overall', {'major'})),
    }


def summary_results(units, ref):
    def rows(kind, positive, key_prefix=None):
        return [(u['item_id'], u['weight'], int(ref[u['unit_id']] in positive)) for u in units
                if u['kind'] == kind and u['unit_id'] in ref and (key_prefix is None or u['key'] == key_prefix)]
    out = {'statements with an error': rate(rows('statement', {'minor', 'major'})),
           'statements with a major error': rate(rows('statement', {'major'})),
           'summaries with major issues overall': rate(rows('overall', {'major'})),
           'summaries with a major omission in any section': None}
    by_item = defaultdict(list)
    for u in units:
        if u['kind'] == 'omission' and u['unit_id'] in ref:
            by_item[(u['item_id'], u['weight'])].append(ref[u['unit_id']] == 'major')
    out['summaries with a major omission in any section'] = rate(
        [(item, weight, int(any(flags))) for (item, weight), flags in by_item.items()])
    sections = sorted({u['key'].split('/', 1)[1] for u in units if u['kind'] == 'omission'})
    out['major omissions by section'] = {s: rate(rows('omission', {'major'}, f'omission/{s}')) for s in sections}
    return out


def decision_results(units, ref, items):
    by_id = {i['item_id']: i for i in items}
    out = {}
    for family in sorted({u['kind'] for u in units}):
        teacher, model = [], []
        for u in units:
            if u['kind'] != family or u['unit_id'] not in ref:
                continue
            item, y = by_id[u['item_id']], int(ref[u['unit_id']] == 'yes')
            teacher.append((u['item_id'], u['weight'], int(item['teacher'][family] == ref[u['unit_id']])))
            p = item['decision_p_yes'].get(family)
            if p is not None:
                model.append((u['item_id'], u['weight'], y, p))
        out[family] = dict(n=len(teacher), reference_yes=sum(y for _, _, y, _ in model),
                           teacher_accuracy=rate(teacher), model_auroc=scored(model, auroc),
                           model_auprc_yes=scored(model, average_precision),
                           model_auprc_no=scored([(c, w, 1 - y, 1 - p) for c, w, y, p in model], average_precision),
                           model_brier=scored(model, brier))
    return out


def metrics(run_dir):
    run_dir = Path(run_dir)
    units = read_jsonl(run_dir / 'compare' / 'units.jsonl')
    decisions = latest_decisions(read_jsonl(run_dir / 'adjudications.jsonl'))
    ref = reference(units, decisions)
    result = dict(units=len(units), with_reference=len(ref), adjudications=len(decisions), components={})
    for component in COMPONENTS:
        mine = [u for u in units if u['component'] == component]
        if not mine:
            continue
        items = read_jsonl(run_dir / 'items' / f'{component}.jsonl')
        entry = dict(items=len(items), judges=judge_costs(run_dir, component), agreement=agreement(mine, decisions))
        if component == 'extraction':
            entry['results'] = extraction_results(mine, ref)
        elif component == 'summary':
            entry['results'] = summary_results(mine, ref)
        else:
            entry['results'] = decision_results(mine, ref, items)
        result['components'][component] = entry
    return result


# --- report --------------------------------------------------------------------------------------------------------

def fmt(value, digits=3):
    if value is None:
        return '-'
    return f'{value:.{digits}f}' if isinstance(value, float) else str(value)


def ci(entry, key='rate'):
    if not entry or entry.get(key) is None:
        return '-'
    interval = entry.get('ci95')
    return fmt(entry[key]) + (f' ({fmt(interval[0])}-{fmt(interval[1])})' if interval else '')


def report(result):
    lines = ['# Dual-judge evaluation', '',
             f'{result["units"]} units, {result["with_reference"]} with a reference (agreed or adjudicated), '
             f'{result["adjudications"]} adjudicated. Rates are weighted by sampling weight; 95% intervals resample '
             'items. Small samples give wide intervals.', '']
    for component, entry in result['components'].items():
        lines += [f'## {component} ({entry["items"]} items)', '', '### Judges', '',
                  '| Judge | Final statuses | Retried | Cost | Cost/item | Median s/request | Input tokens | Output tokens | '
                  'Spent on superseded requests |', '|---|---|---|---|---|---|---|---|---|']
        for judge, j in entry['judges'].items():
            if not j['ran']:
                lines.append(f'| {judge} | not run | | | | | | | ${j["superseded_cost_usd"]} |')
                continue
            lines.append(f'| {judge} | {j["final"]} | {j["retried"]} | ${j["cost_usd"]} | ${j["cost_per_item_usd"]} | '
                         f'{fmt(j["median_seconds"], 1)} | {j["input_tokens"]:,} | {j["output_tokens"]:,} | '
                         f'${j["superseded_cost_usd"]} |')
        lines += ['', '### Agreement and adjudication', '',
                  '| Unit | Units | Agreed | Disputed | Single | Awaiting | Agreement | Kappa | Queued | Adjudicated | '
                  'Pending | ' + ' | '.join(f'{j} accuracy (adjudicated)' for j in JUDGES) + ' | Audit both wrong |',
                  '|---' * (12 + len(JUDGES)) + '|']
        for kind, a in entry['agreement'].items():
            accuracy = ' | '.join(
                f'{fmt(a["judges"][j]["accuracy"])} ({fmt(a["judges"][j]["accuracy_adjudicated"])}, '
                f'n={a["judges"][j]["n_adjudicated"]})' for j in JUDGES)
            lines.append(f'| {kind} | {a["units"]} | {a["agreed"]} | {a["disputed"]} | {a["single"]} | {a["awaiting"]} | '
                         f'{fmt(a["agreement"])} | {fmt(a["kappa"])} | {a["queued"]} | {a["adjudicated"]} | '
                         f'{a["pending"]} | {accuracy} | {a["audit"]["both_wrong"]}/{a["audit"]["decided"]} |')
        lines += ['', "### Each judge's answers (compared values; before any adjudication)", '',
                  '| Unit | ' + ' | '.join(JUDGES) + ' |', '|---' * (1 + len(JUDGES)) + '|']
        for kind, a in entry['agreement'].items():
            lines.append(f'| {kind} | ' + ' | '.join(
                ', '.join(f'{v} {n}' for v, n in a['distribution'][j].items()) or '-' for j in JUDGES) + ' |')
        lines += ['', '### Results against the reference', '']
        results = entry['results']
        if component == 'decision':
            lines += ['| Question | n | Reference yes | Teacher accuracy | Model AUROC | AUPRC (yes) | AUPRC (no) | Brier |',
                      '|---|---|---|---|---|---|---|---|']
            for family, r in results.items():
                lines.append(f'| {family} | {r["n"]} | {r["reference_yes"]} | {ci(r["teacher_accuracy"])} | '
                             f'{ci(r["model_auroc"], "value")} | {ci(r["model_auprc_yes"], "value")} | '
                             f'{ci(r["model_auprc_no"], "value")} | {ci(r["model_brier"], "value")} |')
        else:
            lines += ['| Measure | Units | Items | Rate (95% CI) |', '|---|---|---|---|']
            for name, r in results.items():
                if name == 'major omissions by section':
                    continue
                lines.append(f'| {name} | {r["n"]} | {r["items"]} | {ci(r)} |' if r else f'| {name} | 0 | 0 | - |')
            if 'major omissions by section' in results:
                lines += ['', '| Section: major omission | Summaries | Rate (95% CI) |', '|---|---|---|']
                for section, r in results['major omissions by section'].items():
                    lines.append(f'| {section} | {r["n"]} | {ci(r)} |')
        lines.append('')
    return '\n'.join(lines) + '\n'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-dir', required=True, type=Path)
    args = p.parse_args(argv)
    private()
    result = metrics(args.run_dir)
    write_json(args.run_dir / 'metrics' / 'metrics.json', result)
    (args.run_dir / 'metrics' / 'report.md').write_text(report(result))
    log(f'metrics: {result["units"]} units, {result["with_reference"]} with a reference, '
        f'{result["adjudications"]} adjudicated -> {args.run_dir / "metrics"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
