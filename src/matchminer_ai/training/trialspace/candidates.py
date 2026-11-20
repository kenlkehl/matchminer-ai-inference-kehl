"""Which pairs and criteria lines each round asks the boilerplate and per-criterion teachers, and label reuse plans.

``unique_boilerplate_pairs``: one pair per unique combination of patient boilerplate section and cohort
boilerplate criteria not labeled in an earlier round. ``criteria_line_candidates``: every tagged criteria line of
a seeded fraction of the pairs (yes = the patient does not clearly fail the line), skipping lines labeled for
the patient before. ``write_reuse_plan`` records an earlier round's match labels for import.
"""
import json
from pathlib import Path

from matchminer_ai.training.cohort_format import file_stamp
from .data import digest, load_corpus, read_jsonl, write_json, write_jsonl


def skipped(directories, field):
    """The values of ``field`` in earlier label directories' candidates.jsonl."""
    return {row[field] for directory in directories for row in read_jsonl(Path(directory) / 'candidates.jsonl')}


def selected(seed, patient_id, cohort_id, fraction):
    """Whether a pair is among the seeded ``fraction`` whose criteria lines are labeled."""
    return int(digest([seed, patient_id, cohort_id])[:15], 16) / 16 ** 15 < fraction


def criteria_digest(items):
    """A cohort's tagged lines (heading, text, type): sibling cohorts with the same lines share labels."""
    return digest([[i.get('heading') or '', i['text'], i['criterion_type']] for i in items])


def criteria_line_candidates(corpus, candidate_files, output, *, fraction, seed, skip=()):
    """One candidate row per tagged criteria line of each selected pair (``selected``), skipping
    lines already labeled for the patient (the same patient and criteria in an earlier round's
    directory, or a sibling cohort here). Saved to <output>/candidates.jsonl with a receipt."""
    from matchminer_ai.matching.decision_model import checkable_items
    if not 0 < fraction <= 1:
        raise ValueError('--criteria-pair-fraction must be in (0, 1]')
    patients, cohorts = load_corpus(corpus)
    cmap = {c['id']: c for c in cohorts}
    done = skipped(skip, 'line_key')
    rows, pairs, seen_pairs = [], 0, set()
    for path in candidate_files:
        for pair in read_jsonl(path):
            key = (pair['patient_id'], pair['cohort_id'])
            if key in seen_pairs or not selected(seed, *key, fraction):
                continue
            seen_pairs.add(key)
            cohort = cmap[pair['cohort_id']]
            items = checkable_items(cohort.get('items') or [])
            if not items:
                continue
            pairs += 1
            digest_ = criteria_digest(cohort['items'])
            for item in items:
                line = digest([pair['patient_id'], digest_, item['number']])
                if line in done:
                    continue
                done.add(line)
                rows.append(dict(patient_id=pair['patient_id'], cohort_id=pair['cohort_id'], trial_id=cohort['trial_id'],
                                 split=pair['split'], criteria_digest=digest_, number=int(item['number']),
                                 criterion_type=item['criterion_type'], boilerplate=bool(item.get('boilerplate')),
                                 line_key=line))
    path = Path(output) / 'candidates.jsonl'
    if path.exists():
        if read_jsonl(path) != rows:
            raise ValueError('This label directory was built from different candidates; use a new directory')
    else:
        write_jsonl(path, rows)
    write_json(Path(output) / 'criteria_sample.json', dict(
        sources=[dict(path=str(Path(f).resolve()), stamp=file_stamp(f)) for f in candidate_files],
        skipped_directories=[str(Path(d).resolve()) for d in skip], fraction=fraction, seed=seed,
        selected_pairs_with_lines=pairs, lines=len(rows),
        exclusion_lines=sum(r['criterion_type'] == 'exclusion' for r in rows),
        boilerplate_lines=sum(r['boilerplate'] for r in rows)))
    return path


def unique_boilerplate_pairs(corpus, candidate_files, output, skip=()):
    """One pair per unique (patient boilerplate section, cohort boilerplate criteria) combination.

    Reads every candidate file (one TrialSpace round's), keeps pairs whose cohort has boilerplate
    criteria, and takes the first pair in (patient_id, cohort_id) order for each combination of
    the two texts not already in an earlier round's label directory (``skip``). Saved to
    <output>/candidates.jsonl with a receipt; a rerun must find the same pairs.
    """
    patients, cohorts = load_corpus(corpus)
    pmap, cmap = {p['id']: p for p in patients}, {c['id']: c for c in cohorts}
    pairs, seen_pairs = [], set()
    for path in candidate_files:
        for pair in read_jsonl(path):
            key = (pair['patient_id'], pair['cohort_id'])
            if key not in seen_pairs and (cmap[pair['cohort_id']].get('boilerplate') or '').strip():
                seen_pairs.add(key)
                pairs.append(dict(patient_id=pair['patient_id'], cohort_id=pair['cohort_id'],
                                  trial_id=pair.get('trial_id', cmap[pair['cohort_id']]['trial_id']), split=pair['split']))
    chosen, combinations, done = [], {}, skipped(skip, 'boilerplate_combination')
    for pair in sorted(pairs, key=lambda pair: (pair['patient_id'], pair['cohort_id'])):
        combination = digest([pmap[pair['patient_id']].get('boilerplate') or '', cmap[pair['cohort_id']]['boilerplate']])
        if combination in done:
            continue
        if combination not in combinations:
            combinations[combination] = 0
            chosen.append(dict(pair, boilerplate_combination=combination))
        combinations[combination] += 1
    path = Path(output) / 'candidates.jsonl'
    if path.exists():
        if read_jsonl(path) != chosen:
            raise ValueError('This label directory was built from different candidates; use a new directory')
    else:
        write_jsonl(path, chosen)
    write_json(Path(output) / 'combinations.json', dict(
        sources=[dict(path=str(Path(f).resolve()), stamp=file_stamp(f)) for f in candidate_files],
        skipped_directories=[str(Path(d).resolve()) for d in skip], skipped_combinations=len(done),
        candidate_pairs_with_boilerplate=len(pairs), unique_combinations=len(chosen),
        pairs_per_combination_max=max(combinations.values(), default=0)))
    return path


def write_reuse_plan(args):
    """Record the earlier round to import from (matchminer_ai.labeling.full_pool_reuse verifies it)."""
    if not (args.reuse_labels and args.reuse_candidates):
        raise ValueError('--reuse-labels and --reuse-candidates go together')
    plan = dict(sources=[dict(labels=str(args.reuse_labels), candidates=str(args.reuse_candidates),
                              mode='candidate_intersection', corpus=str(args.corpus))])
    path = Path(args.output) / 'reuse_sources.json'
    if path.exists():
        if json.loads(path.read_text()) != plan:
            raise ValueError('This label directory already imports from a different source')
        return
    write_json(path, plan)
