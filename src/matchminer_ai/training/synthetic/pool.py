"""Production synthetic data form one retrieval and fitting pool.

Historical train/val/test assignments are provenance, never holdouts. Real-data
evaluation and explicitly experimental modules are separate from this policy.
"""

POLICY = 'all_synthetic_training'
SOURCE_SPLITS = ('train', 'val', 'test')

# Pooled-record checks shared with the inference labeler's earlier-round reuse.
from matchminer_ai.labeling.corpus import pooled_records, require_full_pool  # noqa: E402,F401


def require_all_sources(splits):
    normalized = {'val' if value == 'validation' else value for value in splits}
    if normalized != set(SOURCE_SPLITS):
        raise ValueError('Production uses all synthetic data; source-split holdouts/subsets are disabled')


def pooled_frame(frame):
    frame = frame.copy()
    if 'source_split' not in frame:
        frame['source_split'] = frame['split'] if 'split' in frame else 'train'
    frame['split'] = 'train'
    return frame


def require_full_corpus(path):
    from matchminer_ai.training.trialspace.data import load_corpus
    patients, cohorts = load_corpus(path)
    require_full_pool(patients, cohorts)
    return patients, cohorts
