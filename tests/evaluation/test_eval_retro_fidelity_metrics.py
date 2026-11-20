"""Scoring contract of the retrospective distillation-fidelity evaluation, on fabricated data: P@20 and the 0-6
score, the (c) rerank, the merged union and the per-criterion sample, label polarity, and the decision-model
metrics (AUROC, AUPRC, Brier, ECE, calibration intercept and slope, clustered bootstrap)."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

from matchminer_ai.evaluation.retro_fidelity import lists as rf_lists
from matchminer_ai.evaluation.retro_fidelity import metrics as rf_metrics

YES = dict(reasonableness='yes', cancer_type='yes', burden='no', treatment_history='yes', biomarker='yes')
NO = dict(reasonableness='no', cancer_type='yes', burden='yes', treatment_history='yes', biomarker='yes')


def frame(rows):
    return pd.DataFrame(rows, columns=['workflow', 'query_id', 'ranker', 'rank', 'patient_id', 'cohort_id', 'score'])


def test_match_score_is_zero_unless_reasonable_and_weights_biomarker_twice():
    assert rf_metrics.match_score(YES) == 1 + 1 + 0 + 1 + 2
    assert rf_metrics.match_score(NO) == 0
    from matchminer_ai.matching.match_labels import match_score
    assert match_score(YES) == rf_metrics.match_score(YES) and match_score(NO) == 0


def test_precision_at_20_counts_labeled_pairs_and_short_lists():
    rows = [('patient_centric', 'q1', 'embeddinggemma2', r, 'q1', f'c{r}', 0.0) for r in range(1, 21)]
    rows += [('patient_centric', 'q2', 'embeddinggemma2', r, 'q2', f'c{r}', 0.0) for r in range(1, 6)]  # 5 only
    teacher = {('q1', f'c{r}'): dict(answers=YES if r <= 5 else NO, score=rf_metrics.match_score(YES if r <= 5 else NO))
               for r in range(1, 20)}  # c20 of q1 unlabeled
    teacher.update({('q2', f'c{r}'): dict(answers=YES, score=5) for r in range(1, 6)})
    cells = rf_metrics.per_query(frame(rows), teacher)
    q1 = cells[('patient_centric', 'embeddinggemma2', 'q1')]
    assert (q1['n'], q1['labeled'], q1['unlabeled']) == (20, 19, 1)
    assert q1['precision'] == pytest.approx(5 / 19) and q1['mean_score'] == pytest.approx(25 / 19)
    table = rf_metrics.retrieval_metrics(frame(rows), teacher, draws=200)['table']['patient_centric/embeddinggemma2']
    assert table['queries'] == 2 and table['queries_with_fewer_than_top'] == 1 and table['unlabeled_pairs'] == 1
    assert table['precision_at_top']['mean'] == pytest.approx((5 / 19 + 1) / 2)


def test_rerank_orders_by_probability_with_retriever_rank_ties_and_keeps_top():
    candidates = pd.DataFrame(dict(workflow='trial_centric', query_id='c', rank=range(1, 26),
                                   patient_id=[f'p{i}' for i in range(1, 26)], cohort_id='c'))
    probability = {(f'p{i}', 'c'): (0.9 if i in (25, 3) else 0.1) for i in range(1, 26)}
    out = rf_lists.rerank(candidates, probability)
    assert len(out) == 20 and list(out.patient_id[:3]) == ['p3', 'p25', 'p1']
    assert list(out['rank']) == list(range(1, 21)) and set(out.ranker) == {'trialspace_round2_decision'}
    assert out.retriever_rank.iloc[1] == 25


def test_union_keeps_each_pair_once_with_every_list_position():
    lists = frame([('patient_centric', 'p1', 'embeddinggemma2', 1, 'p1', 'c1', 0.5),
                   ('patient_centric', 'p1', 'trialspace_round2', 3, 'p1', 'c1', 0.4),
                   ('trial_centric', 'c1', 'trialspace_round2', 2, 'p1', 'c1', 0.4),
                   ('trial_centric', 'c1', 'trialspace_round2', 1, 'p2', 'c1', 0.6)])
    pairs = rf_lists.union(lists, {'c1': 'NCT0'})
    assert [(p['patient_id'], p['cohort_id']) for p in pairs] == [('p1', 'c1'), ('p2', 'c1')]
    assert len(pairs[0]['sources']) == 3 and {s['direction'] for s in pairs[0]['sources']} == {
        'patient_to_cohort', 'cohort_to_patient'}
    assert pairs[0]['split'] == 'val' and pairs[0]['trial_id'] == 'NCT0'


def test_criteria_sample_reaches_each_stratum_target_once_per_line():
    items = [dict(number=n, heading='', text=f'line {n}', criterion_type='inclusion' if n % 2 else 'exclusion',
                  boilerplate=False) for n in range(1, 4)]
    cohorts = {f'c{i}': dict(trial_id='NCT0', items=items) for i in range(6)}  # siblings with the same criteria
    lists = frame([('patient_centric', 'p1', r, k, 'p1', f'c{k}', 0.0)
                   for r in ('embeddinggemma2', 'trialspace_round2') for k in range(6)])
    rows, counts = rf_lists.criteria_sample(lists, cohorts, [('patient_centric', 'embeddinggemma2'),
                                                             ('patient_centric', 'trialspace_round2')], 3, seed=1)
    assert len(rows) == 3 and len({r['line_key'] for r in rows}) == 3  # p1's lines once, whichever sibling holds them
    assert counts['patient_centric/embeddinggemma2']['lines'] == 3 and counts['patient_centric/trialspace_round2']['lines'] == 0
    again, _ = rf_lists.criteria_sample(lists, cohorts, [('patient_centric', 'embeddinggemma2')], 3, seed=1,
                                        taken=[r['line_key'] for r in rows])
    assert again == []


def test_label_polarity_of_boilerplate_and_criteria():
    assert rf_metrics.boilerplate_yes(False) is True and rf_metrics.boilerplate_yes(True) is False
    assert rf_metrics.criterion_yes('yes') is True and rf_metrics.criterion_yes('no') is False
    with pytest.raises(ValueError):
        rf_metrics.boilerplate_yes('no')
    with pytest.raises(ValueError):
        rf_metrics.criterion_yes('maybe')


def test_auroc_and_average_precision_match_sklearn_with_ties():
    from sklearn.metrics import average_precision_score, roc_auc_score
    rng = np.random.default_rng(0)
    y = rng.random(500) < 0.3
    p = np.round(np.clip(0.3 * y + rng.random(500) * 0.7, 0, 1), 2)  # ties
    assert rf_metrics.auroc(y, p) == pytest.approx(roc_auc_score(y, p))
    assert rf_metrics.average_precision(y, p) == pytest.approx(average_precision_score(y, p))
    assert rf_metrics.average_precision(~y, 1 - p) == pytest.approx(average_precision_score(~y, 1 - p))
    assert np.isnan(rf_metrics.auroc(np.ones(5, bool), np.arange(5)))


def test_calibration_of_a_calibrated_and_an_overconfident_model():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.02, 0.98, 40000)
    y = rng.random(40000) < p
    intercept, slope = rf_metrics.calibration(y, p)
    assert abs(intercept) < 0.05 and abs(slope - 1) < 0.05
    table, ece = rf_metrics.reliability(y, p, 10)
    assert len(table) == 10 and sum(b['n'] for b in table) == 40000 and ece < 0.02
    sharp = 1 / (1 + np.exp(-3 * rf_metrics.logit(p)))  # overconfident: recalibration slope about 1/3
    assert rf_metrics.calibration(y, sharp)[1] == pytest.approx(1 / 3, abs=0.03)
    shifted = 1 / (1 + np.exp(-(rf_metrics.logit(p) + 1)))  # over-predicts yes: negative intercept
    assert rf_metrics.calibration(y, shifted)[0] == pytest.approx(-1, abs=0.05)
    point = rf_metrics.point_metrics(np.array([1, 0, 1, 0], bool), np.array([0.9, 0.1, 0.6, 0.4]))
    assert point['brier'] == pytest.approx((0.01 + 0.01 + 0.16 + 0.16) / 4) and point['auroc'] == 1.0


def test_clustered_bootstrap_resamples_whole_clusters():
    rng = np.random.default_rng(2)
    clusters = np.repeat(np.arange(50), 20)
    p = rng.random(1000)
    y = rng.random(1000) < p
    out = rf_metrics.clustered_bootstrap(y, p, clusters, draws=200, seed=3)
    assert out['n'] == 1000 and out['clusters'] == 50 and out['prevalence_yes'] == pytest.approx(y.mean())
    for name in ('auroc', 'auprc_yes', 'auprc_no', 'brier', 'ece', 'calibration_intercept', 'calibration_slope'):
        cell = out[name]
        assert cell['low'] <= cell['value'] <= cell['high'], name
    assert out['mean_predicted'] == pytest.approx(p.mean()) and len(out['reliability']) == 10
