"""Recover final outputs from a complete journal using CPUs only, without teachers."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path

from .tasks import contract_hash
from .corpus import load_corpus
from .files import digest, read_jsonl, write_json
from .export import export_labels
from .scoring import score_contract
from .journal import ScoreJournal, pending_candidates
from .publication import input_fingerprints, publish_scores


def finalize(corpus, candidates, output, *, workers=8, reader_threads=8, scores_only=False):
    output = Path(output)
    with (output / 'labeling.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = json.loads((output / 'status.json').read_text())
        status.update(pid=os.getpid(), state='validating_completed_scores', finalization_only=True)
        status.pop('exported_questions', None)
        def progress(state, **values):
            status.update(state=state, updated_at=datetime.now(timezone.utc).isoformat(), **values)
            write_json(output / 'status.json', status)
        progress('validating_completed_scores')
        inputs = input_fingerprints(corpus, candidates)
        patients, cohorts = load_corpus(corpus)
        pairs = read_jsonl(candidates)
        signature = json.loads((output / 'labeling_manifest.json').read_text())
        expected = dict(contract=contract_hash(), aggregation_contract=score_contract(),
                        patients=digest(patients), cohorts=digest(cohorts), pairs=digest(pairs))
        if any(signature.get(key) != value for key, value in expected.items()):
            raise ValueError('Journal finalization inputs differ from the original teacher manifest')
        scores = ScoreJournal(output, signature).load({p['id']: p for p in patients}, {c['id']: c for c in cohorts})
        if pending_candidates(pairs, scores):
            raise ValueError('Journal is incomplete; finalization cannot generate missing labels')
        if inputs != input_fingerprints(corpus, candidates):
            raise ValueError('Finalization inputs changed while being validated')
        progress('publishing_scores', inference_complete=True, failures={}, scored_pairs=len(scores))
        receipt = publish_scores(output, signature, scores, pairs, inputs=inputs)
        print(json.dumps(dict(training_scores_ready=True, pairs=receipt['pairs'], questions=receipt['questions'])), flush=True)
        progress('training_scores_ready', training_scores_ready=True)
        if scores_only:
            return receipt
        def keys():
            for key in sorted(scores):
                yield from scores[key]['question_keys'].values()
        export_labels(output / 'responses', output / 'labels.jsonl', keys(), total=receipt['questions'],
                      workers=workers, reader_threads=reader_threads,
                      progress=lambda values: progress('exporting_labels', **values))
        progress('complete')
        return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--candidates', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--reader-threads', type=int, default=8)
    p.add_argument('--scores-only', action='store_true')
    args = p.parse_args()
    finalize(args.corpus, args.candidates, args.output, workers=args.workers,
             reader_threads=args.reader_threads, scores_only=args.scores_only)


if __name__ == '__main__':
    main()
