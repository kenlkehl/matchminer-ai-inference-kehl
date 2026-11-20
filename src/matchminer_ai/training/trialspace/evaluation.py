"""Before/after in-sample diagnostics of a fitted TrialSpace on its labeled pairs; no synthetic holdout exists."""
from pathlib import Path

import numpy as np

from matchminer_ai.embedding.trialspace_contract import model_prompts
from matchminer_ai.labeling import score_contract

from matchminer_ai.training.cohort_format import file_stamp
from .data import digest, load_corpus, read_jsonl, write_json
from .mining import encode_corpus, freeze

VERSION = 'trialspace-evaluation-v3'


def evaluate_dense(args):
    """Before/after in-sample diagnostics; no synthetic holdout exists."""
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    patients, cohorts = load_corpus(args.corpus)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model)
    freeze(output / 'manifest.json', dict(version=VERSION, patients=digest(patients), cohorts=digest(cohorts),
        scores=file_stamp(Path(args.labels) / 'scores.jsonl'), max_tokens=args.max_tokens,
        prompts=model_prompts('trialspace'),
        trained_model={str(p.relative_to(model_path)): file_stamp(p) for p in sorted(model_path.rglob('*'))
                       if p.is_file() and p.suffix in ('.json', '.safetensors')}))
    vectors = encode_corpus(args, patients, cohorts)
    base = {kind: np.load(Path(args.embeddinggemma) / (kind + '.npy'), mmap_mode='r')
            for kind in ('patients', 'cohorts')}
    pi, ci = {p['id']: i for i, p in enumerate(patients)}, {c['id']: i for i, c in enumerate(cohorts)}
    from matchminer_ai.training.synthetic.pool import require_full_pool
    require_full_pool(patients, cohorts)
    groups = {'all_synthetic': read_jsonl(Path(args.labels) / 'scores.jsonl')}
    results = {}
    for split, rows in groups.items():
        p = np.array([pi[row['patient_id']] for row in rows])
        c = np.array([ci[row['cohort_id']] for row in rows])
        y = np.array([row['score'] for row in rows])
        results[split] = dict(pairs=len(rows), models={})
        for name, values in [('base', base), ('trained', vectors)]:
            # Bound intermediate allocations for large candidate unions.
            prediction = np.concatenate([np.einsum('ij,ij->i', values['patients'][p[i:i+8192]],
                values['cohorts'][c[i:i+8192]]) for i in range(0, len(p), 8192)])
            rho = float(spearmanr(y, prediction).statistic)
            aucs = {str(t): (float(roc_auc_score(y >= t, prediction)) if len(np.unique(y >= t)) == 2 else None)
                    for t in (1, 3)}
            results[split]['models'][name] = dict(spearman=rho if np.isfinite(rho) else None,
                auroc_score_at_least=aucs)
    write_json(output / 'metrics.json', dict(results=results, contract=score_contract(),
        held_out=False, labels=str(args.labels), interpretation='In-sample synthetic labeled-pair diagnostics; selection-conditioned AUROC, not generalization or full-corpus recall.'))
