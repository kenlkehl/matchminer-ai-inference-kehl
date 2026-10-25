"""Batched matching encodings with explicit query/document roles and provenance."""
from functools import lru_cache
import json

import torch

from .colbert import BASE_MODEL, ColBERT
from .colbert_contract import INSTRUCTION, regression_query


def token_series(vectors, index):
    """Row-array views keep float16 buffers and permit ordinary Parquet I/O.

    Plain two-dimensional ndarray cells cannot be written by Arrow, while
    parameterized Arrow extension dtypes cannot be read by all supported Pandas
    versions. A list of 1-D array views avoids both and avoids Python floats.
    """
    import pandas as pd
    return pd.Series([list(vector) for vector in vectors], index=index, dtype=object)


@lru_cache(maxsize=1)
def _model(path, revision, device, query_length, document_length, cache_dir):
    # A new checkpoint must have a new path/revision; do not overwrite a loaded
    # checkpoint in place. Fingerprints bind saved embeddings to actual weights.
    model = ColBERT(path, revision=revision, query_length=query_length,
                   document_length=document_length, cache_dir=cache_dir)
    scoring = model.folder / 'regression_scoring.json'
    if scoring.exists():
        saved = json.loads(scoring.read_text())
        if saved.get('score') != 'mean_maxsim' or saved.get('query_instruction') != INSTRUCTION:
            raise ValueError('Checkpoint retrieval instruction/scoring differs from this runtime')
    return model.to(device).eval()


def encode_matching(texts, *, entity_type, settings):
    device = settings.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    model = _model(str(settings.get('model_path', BASE_MODEL)), settings.get('revision'),
                   device, int(settings.get('query_length', 8192)),
                   int(settings.get('document_length', 8192)), settings.get('cache_dir'))
    direction = 'patient_to_cohort' if entity_type == 'patient' else 'cohort_to_patient'
    batch_size = int(settings.get('batch_size', 16))
    queries = model.encode([regression_query(direction, text) for text in texts],
                           is_query=True, batch_size=batch_size)
    documents = model.encode(texts, is_query=False, batch_size=batch_size)
    contract = dict(version='cohort-colbert-encoding-v1', **model.fingerprint(),
                    query_instruction=INSTRUCTION, storage_dtype='float16',
                    roles='Q_instruction_full_text/D_full_text')
    signature = json.dumps(contract, sort_keys=True)
    metadata = dict(model_name=str(settings.get('model_path', BASE_MODEL)),
                    model_sha=model.artifact_sha256, encoding_contract=contract)
    return ([q.numpy() for q in queries], [d.numpy() for d in documents], signature, metadata)
