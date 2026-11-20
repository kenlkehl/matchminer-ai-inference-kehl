"""OncoReasoning packing masks: only completion tokens are targets, and every packed document restarts its positions."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from matchminer_ai.training.oncoreasoning import train


def test_targets_are_exactly_the_completions_and_positions_restart():
    rng = np.random.default_rng(0)
    lengths, prompts = [7, 12, 5, 9, 20], [3, 8, 2, 4, 15]
    tokens = rng.integers(10, 1000, size=sum(lengths)).astype(np.int32)
    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    frame = pd.DataFrame(dict(task=['a', 'b', 'a', 'b', 'a'], offset=offsets, n_tokens=lengths, prompt_tokens=prompts,
                              part=0))
    bins, dropped = train.pack(frame.n_tokens.to_numpy(), 24, seed=1)
    assert dropped == 0 and sorted(i for b in bins for i in b) == list(range(5))
    assert all(frame.n_tokens.to_numpy()[b].sum() <= 24 for b in bins)
    data = train.PackedBins(frame, [tokens], bins, 24, pad_id=0, tasks=['a', 'b'])
    for b, members in enumerate(bins):
        item = {k: v.numpy() for k, v in data[b].items()}
        at, targets = 0, []
        for e in members:
            n, p = lengths[e], prompts[e]
            doc = tokens[offsets[e]:offsets[e] + n]
            assert (item['input_ids'][at:at + n] == doc).all()
            assert (item['position_ids'][at:at + n] == np.arange(n)).all()
            # position t predicts t+1: targets are the completion tokens, from the last prompt position
            assert (item['shift_labels'][at:at + p - 1] == train.IGNORE).all()
            assert (item['shift_labels'][at + p - 1:at + n - 1] == doc[p:]).all()
            assert item['shift_labels'][at + n - 1] == train.IGNORE  # never predicts across documents
            targets.extend(doc[p:])
            at += n
        assert (item['shift_labels'][at:] == train.IGNORE).all() and int(item['real_tokens']) == at
        assert (item['position_ids'][at:] == np.arange(24 - at)).all()  # the padding is its own document
        assert (item['shift_labels'] != train.IGNORE).sum() == len(targets) == (item['labels'] != train.IGNORE).sum()
        assert ((item['target_task'] >= 0) == (item['shift_labels'] != train.IGNORE)).all()
