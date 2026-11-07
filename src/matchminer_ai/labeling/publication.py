"""Publish complete training scores independently of the large reasoning export."""
import json
import os
from pathlib import Path


FORMAT = 'complete-training-scores-v1'


def fingerprint(path):
    stat = Path(path).stat()
    return dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def input_fingerprint(path):
    """Byte size, timestamp and change stamp, without reading file content.

    Copies on other mounts get new timestamps, so ready_scores compares them by
    'bytes' only.
    """
    from .files import file_stamp
    return dict(**fingerprint(path), stamp=file_stamp(path))


def input_fingerprints(corpus, candidates):
    return dict(patients=input_fingerprint(Path(corpus) / 'patients.jsonl'),
                cohorts=input_fingerprint(Path(corpus) / 'cohorts.jsonl'),
                candidates=input_fingerprint(candidates))


def publish_scores(output, signature, scores, pairs, *, inputs):
    """Called with validated journal/aggregate records while holding labeling.lock."""
    from .files import write_json
    from .scoring import score_balance
    from .journal import pending_candidates
    if pending_candidates(pairs, scores):
        raise ValueError('Cannot publish training scores: candidate labels are incomplete')
    output = Path(output)
    target = output / 'scores.jsonl'
    temporary = target.with_suffix('.jsonl.tmp')
    questions = 0
    with temporary.open('wb') as stream:
        for key in sorted(scores):
            row = scores[key]
            raw = (json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n').encode()
            stream.write(raw)
            questions += len(row['question_keys'])
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(target)
    write_json(output / 'score_balance.json', score_balance(list(scores.values())))
    receipt = dict(format=FORMAT, complete=True, pairs=len(scores), questions=questions,
                   teacher_manifest=signature, inputs=inputs, scores_file='scores.jsonl',
                   scores_stat=fingerprint(target),
                   reasoning_source='responses', candidate_coverage_verified=True)
    write_json(output / 'scores_ready.json', receipt)
    return receipt


def ready_scores(output, *, corpus, candidates, expected_pairs=None):
    """Check published byte sizes across mounts without reading content or reasoning traces."""
    output = Path(output)
    path = output / 'scores_ready.json'
    if not path.exists():
        return None
    receipt = json.loads(path.read_text())
    signature = json.loads((output / 'labeling_manifest.json').read_text())
    if (receipt.get('format') != FORMAT or receipt.get('complete') is not True
            or receipt.get('candidate_coverage_verified') is not True
            or receipt.get('scores_file') != 'scores.jsonl'
            or receipt.get('teacher_manifest') != signature):
        raise ValueError('Training-score publication has an invalid or changed teacher identity')
    if expected_pairs is not None and receipt.get('pairs') != expected_pairs:
        raise ValueError('Published training-score count differs from the expected complete candidate set')
    paths = dict(patients=Path(corpus) / 'patients.jsonl', cohorts=Path(corpus) / 'cohorts.jsonl',
                 candidates=Path(candidates), scores=output / 'scores.jsonl')
    recorded = {**receipt.get('inputs', {}), 'scores': receipt.get('scores_stat', {})}
    for name, filename in paths.items():
        expected = recorded.get(name, {}).get('bytes')
        if expected is None:
            raise ValueError('Training-score receipt needs recorded byte sizes: ' + name)
        actual = Path(filename).stat().st_size
        if actual != expected:
            raise ValueError(f'Training data size differs from the validated publication: {filename}; '
                             f'expected bytes={expected}; observed bytes={actual}')
    return receipt
