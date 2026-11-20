import gzip
import importlib
import json
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from matchminer_ai.training.synthetic.generate import Generator
from matchminer_ai.training.cohort_format import training_rows, validate_cohort_rows, complete_cohort_criteria, file_stamp
from test_training_synthetic_generation import Tokenizer, args_for

extract = importlib.import_module('matchminer_ai.training.trials.extract_ctgov_cohorts')
screen = importlib.import_module('matchminer_ai.training.trials.screen_trials')
sample = importlib.import_module('matchminer_ai.training.trials.sample_trials')


def source(number=1, **changes):
    return dict(nct_id=f'NCT{number:08}', study_first_post_date='2020-01-01',
        study_type='INTERVENTIONAL', phases='["PHASE1", "PHASE2"]',
        title=f'Trial {number}', brief_summary='Public trial summary.',
        eligibility_criteria='Adults with cancer.', last_update_post_date='2026-10', **changes)


def single(criteria, name='All participants'):
    return dict(cancer_treatment_trial='yes', multiple_cohorts='no',
                cohorts=[dict(cohort_name=name, eligibility_criteria=criteria)])


def quoted(text, prompt):
    """Whether a rendered prompt holds the text (its string form escapes line breaks)."""
    return text in prompt or json.dumps(text)[1:-1] in prompt


def step_answer(prompt, decision, fence, groups=None):
    """One step's answer derived from a trial's stepwise decision; a string answers every step as is."""
    if isinstance(decision, str):
        return decision
    if 'Group the cohorts whose eligibility criteria are essentially the same' in prompt:
        answer = dict(groups=groups or [[i + 1] for i in range(len(decision['cohorts']))])
    elif 'Decide whether this trial evaluates a cancer treatment' in prompt:
        answer = dict(cancer_treatment_trial=decision['cancer_treatment_trial'])
    elif 'List the cohorts it defines' in prompt:
        answer = dict(cohorts=[c['cohort_name'] for c in decision['cohorts']])
    else:
        target = prompt.rsplit('this cohort only: ', 1)[1]
        answer = dict(eligibility_criteria=next(c['eligibility_criteria'] for c in decision['cohorts']
                                                if target.startswith(c['cohort_name'])))
    return f'```json\n{json.dumps(answer)}\n```' if fence else json.dumps(answer)


@pytest.fixture
def backend(monkeypatch):
    """Answers each prompt of a call from the decision keyed by a title it contains (``fence``: fenced JSON).

    Merge prompts hold only cohort criteria, so their trial is found by a cohort's criteria;
    ``outputs['merge:<title>']`` gives its groups (default: every cohort kept). The boilerplate step
    classifies one criteria line per request: a line that mentions consent is boilerplate.
    """
    import matchminer_ai.trials.cohorts as api
    import matchminer_ai.llm.backends as backends
    monkeypatch.setattr(backends, 'get_endpoint_model_metadata',
        lambda model, **kwargs: {'model_name': model, 'model_sha': 'test-checkpoint'})
    outputs, calls = {}, []
    def generate(**kwargs):
        calls.append(kwargs)
        def trial(prompt):
            titles = [title for title in outputs if title != 'fence' and not title.startswith('merge:')]
            found = next((title for title in titles if title in prompt), None)
            return found or next(title for title in titles if isinstance(outputs[title], dict) and any(
                quoted(c['eligibility_criteria'], prompt) for c in outputs[title].get('cohorts', [])))
        def answer(prompt):
            if 'ITEM TO CLASSIFY' in prompt:
                return json.dumps(dict(criterion_type='inclusion',
                                       boilerplate='consent' in prompt.rsplit('ITEM TO CLASSIFY', 1)[1].lower()))
            return step_answer(prompt, outputs[trial(prompt)], outputs.get('fence', False), outputs.get(f'merge:{trial(prompt)}'))
        answers = [answer(str(prompt)) for prompt in kwargs['prompt_list']]
        return SimpleNamespace(final_outputs=answers, reasoning_outputs=['Reasoning trace'] * len(answers),
            finish_reasons=['stop'] * len(answers), model_metadata={'responses': [
                dict(row_idx=i, endpoint='http://fake/v1', model_root='test-checkpoint') for i in range(len(answers))]})
    import matchminer_ai.trials.cohort_merge as merge_api
    import matchminer_ai.trials.cohort_boilerplate as boilerplate_api
    for module in (api, merge_api, boilerplate_api):
        monkeypatch.setattr(module, 'get_llm_backend', lambda config: SimpleNamespace(generate_llm_outputs=generate))
    return outputs, calls


def arguments(tmp_path):
    return extract.parser().parse_args(['--server_urls', 'http://fake/v1',
        '--model', 'gemma4-31b', '--tokenizer-model', 'nvidia/Gemma-4-31B-IT-NVFP4',
        '--trials-jsonl', str(tmp_path / 'source.jsonl'), '--work-dir', str(tmp_path), '--workers', '2',
        '--output-trials', str(tmp_path / 'audit.csv'), '--output-cohorts', str(tmp_path / 'cohorts.csv'),
        '--output-json', str(tmp_path / 'combined.json')])


def write_trials(args, rows):
    """Sampled ctgov records, as a synthetic protocol batch's seeds.jsonl holds them."""
    args.trials_jsonl.write_text(''.join(json.dumps(row) + '\n' for row in rows))


def test_extraction_csv_preserves_literal_none_as_original_eligibility(tmp_path, backend):
    from matchminer_ai.training.cohort_format import read_clinical_csv
    answers, calls = backend
    answers['Trial 1'] = single('None')
    args = arguments(tmp_path)
    row = source()
    row['eligibility_criteria'] = 'None'
    write_trials(args, [row])
    extract.run(args, extract.make_config(args))
    assert len(calls) == 4  # screen, cohort names, the one cohort's criteria, its boilerplate removal
    assert read_clinical_csv(args.output_cohorts).cohort_eligibility_criteria.tolist() == ['None']
    assert read_clinical_csv(args.output_trials).eligibility_criteria.tolist() == ['None']


def test_selection_includes_phase4_and_boundary_preserves_metadata_and_splits():
    rows = [source(i) for i in range(1, 9)]
    rows[1]['phases'] = '["PHASE4"]'
    rows[2]['phases'] = '["EARLY_PHASE1"]'
    rows[3]['study_first_post_date'] = '2019-12-31'
    rows[4]['study_type'] = 'OBSERVATIONAL'
    rows[5]['phases'] = '["NA"]'
    rows[6]['study_first_post_date'] = None
    rows[7]['phases'] = '"PHASE1"'
    frame = screen.select_trials(pd.DataFrame(rows))
    assert frame.nct_id.tolist() == [f'NCT{i:08}' for i in [1, 2, 3]]
    assert frame.last_update_post_date.eq('2026-10').all()
    pd.testing.assert_frame_equal(frame, screen.select_trials(pd.DataFrame(rows[::-1])))
    frame['split'] = ['train', 'test', 'val']
    assert screen.select_trials(frame, seed=99).split.tolist() == frame.split.tolist()
    with pytest.raises(ValueError, match='duplicate'):
        screen.select_trials(pd.concat([frame, frame.assign(title='Conflicting title')]))


def test_public_api_stepwise_decisions_complete_rows_and_resume(tmp_path, backend, monkeypatch):
    answers, calls = backend
    shared = 'Age >= 18; exclude active infection.'
    decisions = [dict(cancer_treatment_trial='no'),
                 single('Inclusion criteria:\nAdults with treated cancer.'),
                 dict(cancer_treatment_trial='yes', multiple_cohorts='yes',
                      cohorts=[dict(cohort_name='Cohort A', eligibility_criteria=f'Untreated lung cancer.\n{shared}'),
                               dict(cohort_name='Cohort B', eligibility_criteria=f'Previously treated breast cancer.\n{shared}')])]
    answers.update({f'Trial {i}': d for i, d in enumerate(decisions, 1)}, fence=True)
    args = arguments(tmp_path)
    write_trials(args, [source(i) for i in range(1, 4)])
    config = extract.make_config(args)
    assert config.remote['adaptive_concurrency']['enabled'] is True
    assert config.trial['remote']['request_params']['temperature'] == 1
    assert config.trial['remote']['extra_body']['top_k'] == 64
    extract.run(args, config)
    # Screens for all three; cohort names for the two accepted; one criteria call each (both cohorts in one);
    # one merge call for the two-cohort trial; one boilerplate call for each accepted trial.
    assert len(calls) == 10
    rows, audit = pd.read_csv(args.output_cohorts), pd.read_csv(args.output_trials)
    validate_cohort_rows(rows)
    assert len(rows) == 3 and len(audit) == 3
    assert rows.nct_id.tolist() == ['NCT00000002', 'NCT00000003', 'NCT00000003']
    assert rows.cohort_count.tolist() == [1, 2, 2]
    assert rows.cohort_name.tolist() == ['All participants', 'Cohort A', 'Cohort B']
    assert not rows.cohort_eligibility_criteria.str.contains('Cohort A|Cohort B').any()
    # The model's extracted single-cohort criteria, not the ctgov field.
    assert rows.cohort_eligibility_criteria.iloc[0] == 'Inclusion criteria:\nAdults with treated cancer.'
    assert rows.cohort_criteria_source.eq('ctgov').all()
    assert training_rows(rows).cohort_eligibility_criteria.tolist() == rows.cohort_eligibility_criteria.tolist()
    assert rows.cohort_eligibility_criteria.iloc[1:].str.contains('Age >= 18; exclude active infection.', regex=False).all()
    assert rows.last_update_post_date.eq('2026-10').all()
    assert json.loads(args.output_json.read_text()) == dict(zip([f'NCT{i:08}' for i in range(1, 4)], decisions))
    with gzip.open(tmp_path / 'checkpoints/NCT00000003.json.gz', 'rt') as stream:
        record = json.load(stream)
    attempts = record['metadata']['trial_attempts']['NCT00000003']
    assert attempts['screen'][0]['reasoning'] == 'Reasoning trace'
    assert attempts['criteria']['1'][0]['response_metadata']['model_root'] == 'test-checkpoint'
    assert record['metadata']['trial_cohort_names']['NCT00000003'] == ['Cohort A', 'Cohort B']
    assert record['merge']['trial_groups'] == {'NCT00000003': [[0], [1]]}
    extract.run(args, config)
    assert len(calls) == 10
    config.trial['remote']['request_params']['temperature'] = .5
    with pytest.raises(ValueError, match='fresh --work-dir'):
        extract.run(args, config)


def test_failure_is_not_rejection_or_sampleable_output_and_only_failed_trial_retries(tmp_path, backend):
    answers, calls = backend
    answers.update({'Trial 1': '{"cancer_treatment_trial":"no"}', 'Trial 2': 'invalid'})
    args = arguments(tmp_path)
    write_trials(args, [source(1), source(2)])
    config = extract.make_config(args)
    with pytest.raises(RuntimeError, match='1 trials failed'):
        extract.run(args, config)
    assert not args.output_cohorts.exists()
    assert args.output_cohorts.with_suffix('.partial.csv').exists()
    assert len(json.loads((tmp_path / 'errors.json').read_text())) == 1
    assert len(calls) == 4  # Valid trial's screen once; three screening attempts for the invalid one.
    answers['Trial 2'] = single('Adults.')
    extract.run(args, config)
    assert len(calls) == 8  # Only the failed trial reruns: screen, names, criteria, boilerplate removal.
    assert pd.read_csv(args.output_cohorts).nct_id.tolist() == ['NCT00000002']



def test_cohorts_are_combined_then_cleared_of_boilerplate_for_training(tmp_path, backend):
    answers, calls = backend
    decision = dict(cancer_treatment_trial='yes', multiple_cohorts='yes', cohorts=[
        dict(cohort_name='Arm A', eligibility_criteria='Adults with treated lung cancer.\nSigned informed consent.'),
        dict(cohort_name='Arm B', eligibility_criteria='Adults with treated lung cancer; same population.'),
        dict(cohort_name='Arm C', eligibility_criteria='Children with untreated sarcoma.')])
    answers.update({'Trial 1': decision, 'merge:Trial 1': [[1, 2], [3]]})
    args = arguments(tmp_path)
    write_trials(args, [source(1)])
    extract.run(args, extract.make_config(args))
    rows = pd.read_csv(args.output_cohorts)
    assert rows.cohort_name.tolist() == ['Arm A; Arm B', 'Arm C'] and rows.cohort_count.tolist() == [2, 2]
    assert rows.cohort_eligibility_criteria.tolist() == ['Adults with treated lung cancer.\nSigned informed consent.',
                                                         'Children with untreated sarcoma.']
    assert rows.merged_from.tolist() == ['[0, 1]', '[2]']
    validate_cohort_rows(rows)
    # The merged rows again with boilerplate removed: what training's retrieval reads.
    clean = pd.read_csv(tmp_path / 'cohorts_deboilerplated.csv', keep_default_na=False)
    separated = ['cohort_key_criteria', 'cohort_boilerplate_criteria', 'cohort_criteria_items']
    # The merged rows unchanged (all their criteria), plus the key and boilerplate criteria.
    assert clean.drop(columns=separated).equals(pd.read_csv(args.output_cohorts, keep_default_na=False))
    assert clean.cohort_key_criteria.tolist() == ['Adults with treated lung cancer.', 'Children with untreated sarcoma.']
    assert clean.cohort_boilerplate_criteria.tolist() == ['Signed informed consent.', '']
    items = json.loads(clean.cohort_criteria_items[0])  # every line, with both tags, survives the CSV
    assert [(i['criterion_type'], i['boilerplate']) for i in items][-1] == ('inclusion', True)
    validate_cohort_rows(clean)
    with gzip.open(tmp_path / 'checkpoints/NCT00000001.json.gz', 'rt') as stream:
        record = json.load(stream)
    assert len(record['unmerged_cohort_rows']) == 3 and record['result']['cohorts'] == decision['cohorts']
    saved = record['deboilerplate']['cohorts']
    assert [c['cohort_number'] for c in saved] == [0, 1] and saved[0]['items'][1]['boilerplate'] is True
    assert saved[0]['items'][1]['attempts'][0]['reasoning'] == 'Reasoning trace'
    before = pd.read_csv(tmp_path / 'cohorts_unmerged.csv')
    assert before.cohort_name.tolist() == ['Arm A', 'Arm B', 'Arm C'] and before.cohort_count.tolist() == [3, 3, 3]
    assert 'merged_from' not in before
    validate_cohort_rows(before)
    args.merge_cohorts = False
    with pytest.raises(ValueError, match='fresh --work-dir'):
        extract.run(args, extract.make_config(args))

def cohort_rows(trial, count=2):
    return [dict(nct_id=trial, trial_id=trial, cohort_trial_id=f'{trial}-cohort-{i}', cohort_number=i,
        cohort_count=count, cohort_index=i, cohort_eligibility_criteria=f'Specific {i}. Common: age >= 18; no infection.',
        eligibility_criteria=f'Specific {i}. Common: age >= 18; no infection.',
        cohort_criteria_source='eligibility_criteria' if count == 1 else 'llm',
        cancer_treatment_trial='yes', multiple_cohorts='yes' if count > 1 else 'no', split='train') for i in range(count)]


def test_sampling_keeps_all_cohorts_and_excludes_dfci_overlap_and_incomplete_trials(tmp_path):
    rows = sum((cohort_rows(f'NCT{i:08}') for i in range(1, 10)), [])
    # An incomplete set and an invalid text must remove the entire trial.
    rows.pop()
    rows[-3]['cohort_eligibility_criteria'] = ''
    inputs, output, dfci = tmp_path / 'input.csv', tmp_path / 'sample.csv', tmp_path / 'dfci.csv'
    pd.DataFrame(rows).to_csv(inputs, index=False)
    pd.DataFrame([
        ('NCT00000001', '2024-01', '2025-01', 'CLOSED'),
        ('NCT00000002', '2026-12', '2027-01', 'CLOSED'),
        ('NCT00000003', '2024-01', '2024-12', 'CLOSED'),
        ('NCT00000004', '2027-01', None, 'CLOSED'),
        ('NCT00000005', None, None, 'CLOSED'),
        ('NCT00000006', '2020-01', '2024-12', 'OPEN TO ACCRUAL'),
        ('NCT00000006', '2027-01', '2028-01', 'CLOSED'),
    ], columns=['nct_id', 'open_date', 'close_date', 'status']).to_csv(dfci, index=False)
    first = sample.sample_trial_spaces(inputs, output, dfci, n_trials=3, chunksize=1)
    assert set(first.nct_id) == {'NCT00000003', 'NCT00000004', 'NCT00000007'}
    assert len(first) == 6 and first.cohort_index.tolist() == list(range(6))
    second = sample.sample_trial_spaces(inputs, output, dfci, n_trials=3, chunksize=5)
    pd.testing.assert_frame_equal(first, second)
    receipt = json.loads(output.with_suffix('.sampling.json').read_text())
    assert receipt['sampled_trials'] == 3 and receipt['sampled_cohorts'] == 6
    assert receipt['all_cohorts_retained']
    assert receipt['invalid_trials'] == ['NCT00000008', 'NCT00000009']
    with pytest.raises(ValueError, match='need 5000'):
        sample.sample_trial_spaces(inputs, output, dfci)


@pytest.mark.parametrize('kind', ['positive', 'negative'])
def test_native_cohorts_reach_synthetic_seed_prompts_without_inverting_common_inclusions(tmp_path, kind):
    frame = training_rows(pd.DataFrame(cohort_rows('NCT00000001')))
    assert frame.trial_boilerplate_text.eq('').all()
    generator = Generator(args_for(tmp_path), Tokenizer())
    item = dict(frame.iloc[0], key=kind + ':0', patient_class=kind)
    prompt = generator.build_payload(item, 'seeds')['prompt']
    assert item['cohort_eligibility_criteria'] in prompt
    assert '\nNo history of:\n' not in prompt
    conflict = frame.copy()
    conflict['trial_boilerplate_text'] = 'Age >= 18'
    with pytest.raises(ValueError, match='separately extracted boilerplate'):
        training_rows(conflict)


def test_native_cohort_inputs_work_in_patient_preparation(tmp_path):
    from matchminer_ai.training.synthetic.prepare import load_spaces
    cohorts, patients = tmp_path / 'cohorts.csv', tmp_path / 'patients.parquet'
    pd.DataFrame(cohort_rows('NCT00000001')).to_csv(cohorts, index=False)
    pd.DataFrame([dict(pseudo_mrn='1', patient_summary='Synthetic history', split='train')]).to_parquet(patients)
    prepared = load_spaces(cohorts)
    assert prepared.space_index.tolist() == [0, 1]


def test_all_rejected_run_exports_empty_cohort_table_with_schema(tmp_path, backend):
    answers, calls = backend
    answers['Trial 1'] = '{"cancer_treatment_trial":"no"}'
    args = arguments(tmp_path)
    write_trials(args, [source()])
    extract.run(args, extract.make_config(args))
    empty = pd.read_csv(args.output_cohorts)
    assert empty.empty and 'cohort_eligibility_criteria' in empty
    assert 'last_update_post_date' in empty  # the ctgov records' fields are kept (splits come from the sample)
    assert pd.read_csv(args.output_trials).cancer_treatment_trial.tolist() == ['no']


def test_new_run_does_not_overwrite_outputs_from_another_run(tmp_path, backend):
    args = arguments(tmp_path)
    write_trials(args, [source()])
    args.output_cohorts.write_text('previous output')
    with pytest.raises(ValueError, match='fresh output paths'):
        extract.run(args, extract.make_config(args))
    assert args.output_cohorts.read_text() == 'previous output'


def test_dfci_missing_date_columns_are_unbounded():
    frame = pd.DataFrame(dict(nct_id=['NCT00000001'], open_date=[float('nan')],
                             close_date=[float('nan')], status=['CLOSED']))
    assert sample.dfci_nct_ids_open_during(frame, 2025, 2026) == {'NCT00000001'}


def test_prepared_synthetic_patient_metadata_retains_native_cohort_provenance(tmp_path):
    from matchminer_ai.training.synthetic.prepare import prepare_notes
    from test_training_pipeline import synthetic_files
    notes, cohorts = synthetic_files(tmp_path)
    rows = [dict(cohort_rows(f'NCT{i}', 1)[0], cohort_index=i, split=split)
            for i, split in enumerate(['train', 'val'])]
    pd.DataFrame(rows).to_csv(cohorts, index=False)
    output, patients = tmp_path / 'notes.parquet', tmp_path / 'patients.parquet'
    prepare_notes(notes, cohorts, output, patients)
    metadata = pd.read_parquet(patients)
    assert metadata.cohort_trial_id.tolist() == [f'{trial}-cohort-0' for trial in metadata.nct_id]
    assert metadata.this_space.eq(metadata.cohort_eligibility_criteria).all()
    assert metadata.trial_boilerplate_text.eq('').all()


def production_export(tmp_path):
    run = tmp_path / 'production'
    run.mkdir()
    source_rows = [source(i) for i in range(1, 5)]
    for row in source_rows:
        row['trial_text'] = '\n'.join(row[k] for k in ('title', 'brief_summary', 'eligibility_criteria'))
    pd.DataFrame(source_rows).to_parquet(run / 'inputs.parquet')
    decisions, cohorts = {}, []
    for src in source_rows:
        trial = src['nct_id']
        if trial == 'NCT00000001':
            decisions[trial] = {'cancer_treatment_trial': 'no'}
            continue
        multiple = trial == 'NCT00000003'
        decision = dict(cancer_treatment_trial='yes', multiple_cohorts='yes' if multiple else 'no')
        if multiple:
            decision.update(cohorts=['Untreated lung cancer.', 'Previously treated breast cancer.'],
                            common_eligibility_criteria='Adults; no active infection.')
        decisions[trial] = decision
        for i in range(2 if multiple else 1):
            cohorts.append(dict(trial_id=trial, cohort_trial_id=f'{trial}-cohort-{i}', cohort_number=i,
                cancer_treatment_trial='yes', multiple_cohorts=decision['multiple_cohorts'],
                cohort_specific_eligibility_criteria=decision['cohorts'][i] if multiple else None,
                common_eligibility_criteria=decision.get('common_eligibility_criteria'),
                cohort_eligibility_criteria=(f'Cohort-specific eligibility criteria:\n{decision["cohorts"][i]}\n\n'
                    f'Common eligibility criteria:\n{decision["common_eligibility_criteria"]}') if multiple else src['eligibility_criteria'],
                cohort_criteria_source='llm' if multiple else 'eligibility_criteria'))
    pd.DataFrame(cohorts).to_json(run / 'cohorts.jsonl', orient='records', lines=True)
    (run / 'combined_outputs.json').write_text(json.dumps(decisions))
    (run / 'status.json').write_text(json.dumps(dict(state='complete', errors=0, total=4, rejected=1, cohort_rows=4)))
    (run / 'manifest.json').write_text(json.dumps(dict(run_identity='fixture-run')))
    dfci = tmp_path / 'dfci.csv'
    pd.DataFrame([dict(nct_id='NCT00000004', open_date='2026-01', close_date='2027-01', status='CLOSED')]).to_csv(dfci, index=False)
    return run, dfci


def test_sample_completed_production_export(tmp_path):
    run, dfci = production_export(tmp_path)
    output = tmp_path / 'sample.csv'
    first = sample.sample_trial_spaces(None, output, dfci, extraction_dir=run, n_trials=2, chunksize=1)
    assert set(first.nct_id) == {'NCT00000002', 'NCT00000003'}
    assert len(first) == 3 and first.cancer_treatment_trial.eq('yes').all()
    single = first.loc[first.multiple_cohorts.eq('no')].iloc[0]
    assert single.cohort_eligibility_criteria == single.eligibility_criteria
    assert single.trial_text != single.eligibility_criteria
    for row in first.loc[first.multiple_cohorts.eq('yes')].itertuples():
        assert row.cohort_specific_eligibility_criteria in row.cohort_eligibility_criteria
        assert row.common_eligibility_criteria in row.cohort_eligibility_criteria
    receipt = json.loads(output.with_suffix('.sampling.json').read_text())
    assert receipt['source_trials'] == 4 and receipt['cancer_treatment_trials'] == 3
    assert receipt['excluded_non_cancer_trials'] == ['NCT00000001']
    assert receipt['excluded_dfci_trials'] == ['NCT00000004']
    assert receipt['extraction_run_identity'] == 'fixture-run'
    assert receipt['source_stamp']['cohorts.jsonl'] == file_stamp(run / 'cohorts.jsonl')
    second = sample.sample_trial_spaces(None, output, dfci, extraction_dir=run, n_trials=2, chunksize=3)
    pd.testing.assert_frame_equal(first, second)


def test_production_import_rejects_missing_cohort_or_mixed_source_text(tmp_path):
    from matchminer_ai.training.cohort_format import load_production_cohorts
    run, _ = production_export(tmp_path)
    cohorts = pd.read_json(run / 'cohorts.jsonl', lines=True)
    cohorts.loc[~cohorts.cohort_trial_id.eq('NCT00000003-cohort-1')].to_json(run / 'cohorts.jsonl', orient='records', lines=True)
    with pytest.raises(ValueError, match='Incomplete'):
        load_production_cohorts(run)
    cohorts.loc[cohorts.multiple_cohorts.eq('no'), 'cohort_eligibility_criteria'] = 'Different source'
    cohorts.to_json(run / 'cohorts.jsonl', orient='records', lines=True)
    with pytest.raises(ValueError, match='original eligibility_criteria'):
        load_production_cohorts(run)


def test_csv_sampler_explicitly_filters_cancer_and_rebuilds_combined_text(tmp_path):
    run, dfci = production_export(tmp_path)
    from matchminer_ai.training.cohort_format import load_production_cohorts
    frame, _ = load_production_cohorts(run)
    frame['cohort_eligibility_criteria'] = 'Stale combined text'
    rejected = {name: None for name in frame}
    rejected.update(nct_id='NCT00000001', trial_id='NCT00000001', cancer_treatment_trial='no')
    frame = pd.concat([frame, pd.DataFrame([rejected])], ignore_index=True)
    inputs, output = tmp_path / 'input.csv', tmp_path / 'sample.csv'
    frame.to_csv(inputs, index=False)
    result = sample.sample_trial_spaces(inputs, output, dfci, n_trials=2, chunksize=1)
    assert len(result) == 3 and set(result.nct_id) == {'NCT00000002', 'NCT00000003'}
    assert not result.cohort_eligibility_criteria.eq('Stale combined text').any()
    assert result.loc[result.multiple_cohorts.eq('no'), 'cohort_eligibility_criteria'].iloc[0] == 'Adults with cancer.'
    assert json.loads(output.with_suffix('.sampling.json').read_text())['excluded_non_cancer_trials'] == ['NCT00000001']


def test_legacy_sample_also_honors_explicit_cancer_treatment_decisions(tmp_path):
    from test_training_synthetic_generation import SPACE
    inputs, output, dfci = tmp_path / 'old.csv', tmp_path / 'sample.csv', tmp_path / 'dfci.csv'
    pd.DataFrame([dict(nct_id=f'NCT{i:08}', this_space=SPACE, cancer_treatment_trial=decision)
                  for i, decision in enumerate(['no', 'yes', 'unknown'], 1)]).to_csv(inputs, index=False)
    pd.DataFrame(columns=['nct_id', 'open_date', 'close_date', 'status']).to_csv(dfci, index=False)
    result = sample.sample_trial_spaces(inputs, output, dfci, n_trials=1)
    assert result.nct_id.tolist() == ['NCT00000002']


def test_single_cohort_requires_eligibility_not_title_summary_fallback():
    from matchminer_ai.training.cohort_format import REPRESENTATION
    from matchminer_ai.trials.cohorts import COHORT_REPRESENTATION_VERSION
    assert REPRESENTATION == COHORT_REPRESENTATION_VERSION
    frame = pd.DataFrame(cohort_rows('NCT00000001', 1))
    frame['trial_text'] = 'Title and summary plus criteria'
    frame['cohort_eligibility_criteria'] = frame.trial_text
    fixed = complete_cohort_criteria(frame)
    assert fixed.cohort_eligibility_criteria.iloc[0] == frame.eligibility_criteria.iloc[0]
    assert fixed.cohort_criteria_source.iloc[0] == 'eligibility_criteria'
    frame = frame.drop(columns='eligibility_criteria')
    frame['cohort_criteria_source'] = 'trial_text'
    with pytest.raises(ValueError, match='requires original eligibility_criteria'):
        complete_cohort_criteria(frame)


def test_protocol_single_cohort_keeps_protocol_criteria_over_registry_text():
    frame = pd.DataFrame(cohort_rows('NCT00000001', 1))
    frame['cohort_eligibility_criteria'] = 'Inclusion criteria:\n1. Protocol-defined criterion.'
    frame['cohort_criteria_source'] = 'protocol'
    fixed = complete_cohort_criteria(frame)
    assert fixed.cohort_eligibility_criteria.iloc[0] == frame.cohort_eligibility_criteria.iloc[0]
    assert fixed.cohort_eligibility_criteria.iloc[0] != frame.eligibility_criteria.iloc[0]
    assert fixed.cohort_criteria_source.iloc[0] == 'protocol'
    assert training_rows(frame).this_space.iloc[0] == frame.cohort_eligibility_criteria.iloc[0]


def test_generation_calls_five_positive_and_five_negative_seeds_separately_per_cohort(tmp_path, monkeypatch):
    import matchminer_ai.training.synthetic.generate as generation
    from matchminer_ai.training.cohort_format import load_production_cohorts
    from test_training_synthetic_generation import fake_session, SEEDS, expand_seed
    run, _ = production_export(tmp_path)
    frame, _ = load_production_cohorts(run)
    frame = frame[frame.nct_id.isin(['NCT00000002', 'NCT00000003'])].reset_index(drop=True)
    frame['cohort_index'] = frame.index
    frame = training_rows(frame)
    instance = Generator(args_for(tmp_path), Tokenizer())
    calls = []
    async def complete(client, payload):
        calls.append(payload)
        answer = SEEDS if payload['stage'] == 'seeds' else expand_seed(payload['seed_text'])
        return 'reasoning', answer, dict(finish_reason='stop', raw_text=answer)
    fake_session(monkeypatch, complete)
    asyncio.run(instance.generate_patient_events(frame))
    seeds = [p for p in calls if p['stage'] == 'seeds']
    assert len(seeds) == 2 * len(frame) == 6
    for row in frame.to_dict('records'):
        for kind in ('positive', 'negative'):
            payload = next(p for p in seeds if p['key'] == f'{kind}:{row["space_index"]}')
            assert row['cohort_eligibility_criteria'] in payload['prompt']
            assert 'exactly five patients' in payload['prompt']
            assert 'Trial 2' not in payload['prompt'] and 'Public trial summary.' not in payload['prompt']
            assert ('meet all of these criteria' if kind == 'positive' else 'would narrowly miss at least one') in payload['prompt']
            saved = json.loads(instance.seed_path(dict(row, patient_class=kind)).read_text())
            assert len(generation.parse_seeds(saved['answer'])) == 5
            assert saved['cohort_trial_id'] == row['cohort_trial_id']
    histories = pd.concat([pd.read_parquet(p) for p in generation.checkpoint_paths(tmp_path / 'patient_events')])
    assert len(histories) == 30
    assert histories.groupby(['cohort_trial_id', 'patient_class']).size().eq(5).all()


def test_named_production_export_preserves_names_through_sampling_and_the_corpus(tmp_path):
    from matchminer_ai.training.trialspace.data import prepare, read_jsonl
    from matchminer_ai.training.cohort_format import NAMED_SPLIT_REPRESENTATION, LEGACY_REPRESENTATION, load_production_cohorts
    run, dfci = production_export(tmp_path)
    historical, _ = load_production_cohorts(run)
    assert training_rows(historical).trial_representation.eq(LEGACY_REPRESENTATION).all()
    decisions = json.loads((run / 'combined_outputs.json').read_text())
    decision = decisions['NCT00000003']
    decision['cohorts'] = [dict(cohort_name=f'Arm {i + 1}', cohort_specific_eligibility_criteria=text)
                           for i, text in enumerate(decision['cohorts'])]
    (run / 'combined_outputs.json').write_text(json.dumps(decisions))
    rows = pd.read_json(run / 'cohorts.jsonl', lines=True)
    rows['cohort_name'] = [f'Arm {number + 1}' if multi == 'yes' else None
                          for number, multi in zip(rows.cohort_number, rows.multiple_cohorts)]
    rows.to_json(run / 'cohorts.jsonl', orient='records', lines=True)
    output = tmp_path / 'sample.csv'
    sampled = sample.sample_trial_spaces(None, output, dfci, extraction_dir=run, n_trials=2)
    assert sampled.trial_representation.eq(NAMED_SPLIT_REPRESENTATION).all()
    assert sampled.loc[sampled.multiple_cohorts.eq('yes'), 'cohort_name'].tolist() == ['Arm 1', 'Arm 2']
    assert not sampled.cohort_eligibility_criteria.str.contains('Arm 1|Arm 2').any()
    patients = tmp_path / 'patients.jsonl'
    patients.write_text(json.dumps(dict(patient_id='synthetic', patient_summary='Cancer history.',
                                       split=sampled.split.iloc[0])) + '\n')
    prepare(patients, output, tmp_path / 'corpus')
    corpus = read_jsonl(tmp_path / 'corpus/cohorts.jsonl')
    assert [c['cohort_name'] for c in corpus if c['trial_id'] == 'NCT00000003'] == ['Arm 1', 'Arm 2']
    assert all('Arm 1' not in c['text'] and 'Arm 2' not in c['text'] for c in corpus)
    rows.loc[rows.multiple_cohorts.eq('yes'), 'cohort_name'] = 'Wrong name'
    rows.to_json(run / 'cohorts.jsonl', orient='records', lines=True)
    with pytest.raises(ValueError, match='name disagree'):
        load_production_cohorts(run)


def test_new_representation_requires_names_without_relabeling_historical_rows():
    from matchminer_ai.training.cohort_format import REPRESENTATION, LEGACY_REPRESENTATION
    frame = pd.DataFrame(cohort_rows('NCT00000001'))
    assert training_rows(frame).trial_representation.eq(LEGACY_REPRESENTATION).all()
    frame['trial_representation'] = REPRESENTATION
    with pytest.raises(ValueError, match='requires cohort_name'):
        training_rows(frame)
    frame['cohort_name'] = ['Arm A', None]
    with pytest.raises(ValueError, match='nonempty cohort_name'):
        training_rows(frame)
