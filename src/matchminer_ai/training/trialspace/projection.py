"""Self-contained pair tables (key texts, 0-6 scores and the five answers) of a labeled candidate set, for fitting."""
import json
from pathlib import Path

import pandas as pd

from matchminer_ai.embedding.trialspace_contract import DIRECTIONS
from matchminer_ai.labeling import contract_hash, score_contract
from matchminer_ai.labeling.publication import ready_scores
from matchminer_ai.matching.match_labels import MAX_SCORE

from matchminer_ai.training.cohort_format import file_stamp
from .data import digest, load_corpus, read_jsonl, write_json, write_jsonl
from .mining import completed, finish, freeze

VERSION = 'trialspace-projection-v3'


def project_labels(corpus, union, labels, sources, output, *, reasoning_source=None, source_origins=None):
    """Self-contained pair tables (key texts, scores and the five answers) for fitting."""
    patients, cohorts = load_corpus(corpus)
    union, labels, output = Path(union), Path(labels), Path(output)
    reasoning_source = reasoning_source or str(labels / 'responses')
    source_origins = source_origins or {}
    count = json.loads((union / 'complete.json').read_text())['pairs']
    if ready_scores(labels, corpus=corpus, candidates=union / 'candidates.jsonl', expected_pairs=count) is None:
        raise ValueError('The candidate set must be completely labeled before projection or training')
    scores = {(s['patient_id'], s['cohort_id']): s for s in read_jsonl(labels / 'scores.jsonl')}
    pmap, cmap = {p['id']: p for p in patients}, {c['id']: c for c in cohorts}
    contract, question_contract = score_contract(), contract_hash('match')
    result = {}
    for name, source in sources.items():
        target = output / name
        pairs = read_jsonl(Path(source) / 'candidates.jsonl')
        freeze(target / 'manifest.json', dict(version=VERSION, source=str(source_origins.get(name, source)),
            candidates_sha256=digest(pairs), union_scores_stamp=file_stamp(labels / 'scores.jsonl'),
            contract=contract, shared_reasoning=reasoning_source))
        if completed(target):
            result[name] = json.loads((target / 'complete.json').read_text())
            continue
        projected, rows = [], []
        for pair in pairs:
            p, c = pmap[pair['patient_id']], cmap[pair['cohort_id']]
            score = scores[p['id'], c['id']]
            if (score['contract'] != contract or score['question_contract'] != question_contract
                    or score['split'] != p['split'] or p['split'] != c['split']
                    or pair['split'] != p['split'] or type(score['score']) is not int
                    or not 0 <= score['score'] <= MAX_SCORE):
                raise ValueError('Invalid/cross-split shared label')
            projected.append(score)
            rows.append(dict(patient_id=p['id'], cohort_id=c['id'], cohort_trial_id=c['id'],
                nct_id=c['trial_id'], patient_summary=p['text'], this_space=c['text'], split=p['split'],
                patient_source_split=p.get('source_split', p['split']),
                cohort_source_split=c.get('source_split', c['split']), cohort_source=c.get('source', 'ctgov'),
                eligibility_result=score['score'], normalized_score=score['normalized_score'],
                answers=json.dumps(score['answers']), question_keys=json.dumps(score['question_keys']),
                question_contract=score['question_contract'], score_contract=score['contract'],
                retrieval_sources=json.dumps(pair['sources']),
                patient_to_cohort=any(h['direction'] in ('patient_to_cohort', 'both') for h in pair['sources']),
                cohort_to_patient=any(h['direction'] in ('cohort_to_patient', 'both') for h in pair['sources'])))
        frame = pd.DataFrame(rows)
        for key in ['pairs', *DIRECTIONS]:
            chosen = frame if key == 'pairs' else frame.loc[frame[key]]
            path = target / (key + '.parquet')
            chosen.to_parquet(path.with_suffix('.tmp'), index=False, compression='zstd',
                              dictionary_pagesize_limit=256 * 1024**2)
            path.with_suffix('.tmp').replace(path)
        write_jsonl(target / 'scores.jsonl', projected)
        finish(target, ['pairs.parquet', 'patient_to_cohort.parquet', 'cohort_to_patient.parquet', 'scores.jsonl'],
               pairs=len(rows), by_split=frame.split.value_counts().to_dict())
        result[name] = json.loads((target / 'complete.json').read_text())
    write_json(output / 'complete.json', dict(models=result, overlapping_pairs_duplicated=len(sources) > 1,
        shared_label_contract=contract, shared_reasoning=reasoning_source))
