#!/usr/bin/env python3
"""In-sample diagnostics of a fitted TrialSpace against the base model on its labeled pairs (not a pipeline stage).

Encodes the corpus with ``--model`` (as ``trialspace.mine`` does), and compares its and the base mining's
(``--embeddinggemma``) similarities with the teacher's 0-6 scores: Spearman correlation and AUROC at scores >= 1
and >= 3. No synthetic holdout exists; these are selection-conditioned in-sample numbers.
"""
from pathlib import Path

from matchminer_ai.training.trialspace import evaluation
from matchminer_ai.training.trialspace.mine import parser as mine_parser


def main():
    parser = mine_parser()
    parser.description = __doc__
    parser.add_argument('--embeddinggemma', type=Path, required=True, help='The base model\'s mining directory')
    parser.add_argument('--labels', type=Path, required=True, help='The match label directory (scores.jsonl)')
    evaluation.evaluate_dense(parser.parse_args())


if __name__ == '__main__':
    main()
