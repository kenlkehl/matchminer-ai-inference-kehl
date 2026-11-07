"""Exact similarity search in query batches over one or more devices, with a stable top-k.

For normalized embeddings, ``score_batches`` yields cosine similarities of consecutive query batches
against every document, in query order. Each device holds a copy of the documents and scores whole
batches, so the scores do not depend on how many devices share the work. ``stable_top_k`` breaks ties
by document index (training's mining has always selected this way). Selection policies on top of the
scores (which documents a query may see, lexical or random extras) belong to the caller.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Iterator, Sequence

import numpy as np


def stable_top_k(scores, k: int) -> list[int]:
    """Indices of the ``k`` highest scores, highest first; ties keep index order."""
    return np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")[:k].tolist()


def score_batches(queries, documents, *, batch_size: int = 128,
                  devices: Sequence[str] = ("cpu",)) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(start, scores)`` for query rows ``start:start + batch_size``: float32 ``queries @ documents.T``.

    Batches are spread over ``devices`` (for example ``("cuda:0", "cuda:1")``); a few run ahead of the
    consumer per device.
    """
    import torch

    if batch_size < 1 or not devices:
        raise ValueError("score_batches needs a positive batch size and at least one device")
    queries = np.asarray(queries, dtype=np.float32)
    documents = np.asarray(documents, dtype=np.float32)
    on = {device: torch.as_tensor(documents, device=device) for device in dict.fromkeys(devices)}
    starts = list(range(0, len(queries), batch_size))

    def score(number):
        device = devices[number % len(devices)]
        start = starts[number]
        q = torch.as_tensor(queries[start:start + batch_size], device=device)
        return start, (q @ on[device].T).cpu().numpy()

    window = 2 * len(devices)
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        running = [pool.submit(score, number) for number in range(min(window, len(starts)))]
        for number in range(len(starts)):
            yield running[number].result()
            running[number] = None  # release the batch once yielded
            if number + window < len(starts):
                running.append(pool.submit(score, number + window))


__all__ = ["score_batches", "stable_top_k"]
