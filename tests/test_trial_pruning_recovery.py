"""Semantic pruning repairs must be bounded, resumable and parent-preserving."""
import json
from pathlib import Path

import pytest

from matchminer_ai.trials.hierarchy import (
    Completion, Hierarchy, ReadyHierarchy, ResponseCache, load_prompts,
)

PROMPTS = load_prompts(Path(__file__).parents[1] / 'src/matchminer_ai/prompts')
SOURCE = 'Afatinib treats HER2-mutant cancers. Non-small cell lung cancer is excluded.'
REVIEW = dict(decision='review', reason='This is a therapeutic basket study; enumerate compatible sites.',
              evidence='Afatinib treats HER2-mutant cancers.')


def branches(*values, scope='all eligible patients'):
    return Completion(json.dumps(dict(branches=[dict(value=v, scope=scope) for v in values], prune_reason='')))


def default_answer(request):
    if request.field_name == 'study_relevance':
        return Completion(json.dumps(REVIEW))
    if request.field_name == 'boilerplate':
        return Completion('{"exclusions":["Active infection"]}')
    value = dict(age='18+', cancer_primary_site='breast', sex='female', histology='adenocarcinoma',
                 disease_state='metastatic', prior_treatment_requirements='NA',
                 biomarker_requirements='HER2 mutation')[request.field_name]
    return branches(value)


def drive(state, cache, policy, calls, *, stop_after_review=False):
    for _ in range(100):
        requests = state.requests()
        if not requests:
            return
        completions = []
        for request in requests:
            answer = cache.get(request)
            if answer is None:
                calls.append(request)
                answer = policy(request)
                cache.put(request, answer)
            completions.append(answer)
        state.accept(requests, completions)
        if stop_after_review and any(r.field_name == 'study_relevance' for r in requests):
            return
    pytest.fail('Recovery failed to terminate')


@pytest.mark.parametrize('state_type', [Hierarchy, ReadyHierarchy])
def test_afatinib_recovery_survives_restart_and_preserves_old_cache(state_type, tmp_path):
    cache = ResponseCache(tmp_path, {'existing_runtime': 'unchanged'})
    calls = []

    def policy(r):
        if r.field_name == 'cancer_primary_site' and not r.recovery_round:
            return Completion('{"branches":[],"prune_reason":"No single primary site named"}')
        if r.field_name == 'cancer_primary_site':
            assert r.parent == {'age': '18+'}
            assert REVIEW['reason'] in r.messages[-1]['content']
            assert PROMPTS['cancer_primary_site'].strip() in r.messages[1]['content']
            assert PROMPTS['anti_boilerplate'] in r.messages[1]['content']
            assert 'INSTRUCTIONS FOR THIS STEP' not in r.messages[-1]['content']
            return branches('breast', 'colon')
        return default_answer(r)

    interrupted = state_type(0, SOURCE, PROMPTS)
    drive(interrupted, cache, policy, calls, stop_after_review=True)
    assert interrupted.recovery_rounds == 1 and not interrupted.error
    original = next(r for r in calls if r.field_name == 'cancer_primary_site')
    original_bytes = cache.path(original).read_bytes()
    resumed = state_type(0, SOURCE, PROMPTS)
    drive(resumed, cache, policy, calls)
    assert not resumed.error
    output = resumed.output()
    assert len(output.split('\nBoilerplate')[0].splitlines()) == 2
    assert 'Cancer primary site allowed: breast.' in output
    assert 'Cancer primary site allowed: colon.' in output
    repaired = [r for r in calls if r.recovery_round]
    assert len(repaired) == 1 and repaired[0].key != original.key
    assert repaired[0].recovery_of == original.key
    assert cache.path(original).read_bytes() == original_bytes
    assert sum(r.field_name == 'age' for r in calls) == 1
    assert sum(r.field_name == 'boilerplate' for r in calls) == 1
    assert any(n.get('recovery_of') == original.key for n in resumed.trace)
    count = len(calls)
    again = state_type(0, SOURCE, PROMPTS)
    drive(again, cache, policy, calls)
    assert again.output() == output and len(calls) == count


@pytest.mark.parametrize('state_type', [Hierarchy, ReadyHierarchy])
def test_repeated_pruning_stops_after_two_recovery_rounds(state_type, tmp_path):
    cache = ResponseCache(tmp_path, {})
    calls = []

    def policy(r):
        if r.field_name == 'cancer_primary_site':
            return Completion('{"branches":[],"prune_reason":"Still uncertain"}')
        return default_answer(r)

    state = state_type(0, SOURCE, PROMPTS)
    drive(state, cache, policy, calls)
    assert 'after 2 automatic recovery rounds' in state.error
    assert [r.recovery_round for r in calls if r.field_name == 'cancer_primary_site'] == [0, 1, 2]
    assert sum(r.field_name == 'study_relevance' for r in calls) == 3
    with pytest.raises(ValueError, match='after 2 automatic'):
        state.output()
    count = len(calls)
    again = state_type(0, SOURCE, PROMPTS)
    drive(again, cache, policy, calls)
    assert again.error == state.error and len(calls) == count


@pytest.mark.parametrize('state_type', [Hierarchy, ReadyHierarchy])
def test_recovery_reconnects_pruned_fields_at_different_depths(state_type, tmp_path):
    cache = ResponseCache(tmp_path, {})
    calls = []

    def policy(r):
        if r.field_name == 'cancer_primary_site':
            return Completion(json.dumps(dict(branches=[dict(value='breast', scope=s) for s in ['A', 'B']], prune_reason='')))
        if (r.field_name == 'sex' and r.scope == 'A' or r.field_name == 'histology') and not r.recovery_round:
            return Completion('{"branches":[],"prune_reason":"Uncertain branch"}')
        return default_answer(r)

    state = state_type(0, SOURCE, PROMPTS)
    drive(state, cache, policy, calls)
    assert not state.error
    # Two source branches converge on one leaf; a restored edge must point to
    # the repaired histology node, not its old empty cached response.
    assert len(state.output().split('\nBoilerplate')[0].splitlines()) == 1
    assert {r.field_name for r in calls if r.recovery_round} == {'sex', 'histology'}
    assert sum(r.field_name == 'histology' and r.recovery_round > 0 for r in calls) == 1


@pytest.mark.parametrize('state_type', [Hierarchy, ReadyHierarchy])
def test_semantic_recovery_respects_total_call_budget(state_type, tmp_path):
    def policy(r):
        if r.field_name == 'cancer_primary_site':
            return Completion('{"branches":[],"prune_reason":"No single named site"}')
        return default_answer(r)
    state = state_type(0, SOURCE, PROMPTS, max_calls=4)
    drive(state, ResponseCache(tmp_path, {}), policy, [])
    assert 'max_calls' in state.error
