#!/usr/bin/env python3
"""Extract cancer-treatment cohorts through the package's cohort API.

extract_cohorts runs the inference package's three-step workflow on each
ctgov listing (title, brief summary and eligibility criteria as one document):
a cancer-treatment screen, the cohort names, and one request per cohort for its
criteria, copied verbatim with boilerplate included, with the same prompts as
synthetic protocols. No training-owned prompt. For training, the same steps as
protocols (synthetic_protocols/extract_cohorts.py) follow: a trial's cohorts whose
criteria are essentially the same are combined (merge_equivalent_cohorts,
trial.cohorts.merge.user.txt; --no-merge-cohorts keeps every extracted cohort), then each
cohort's boilerplate criteria are separated from the rest: each criteria line is classified
in its own request and code sorts the unchanged lines (remove_boilerplate_criteria,
trial.cohorts.deboilerplate.user.txt;
--no-deboilerplate-cohorts skips it). Exports: the merged rows (--output-cohorts,
boilerplate kept), the rows before merging (--output-unmerged-cohorts) and the merged
rows with their key criteria (the criteria that are not boilerplate) in cohort_key_criteria
and boilerplate criteria in cohort_boilerplate_criteria (--output-deboilerplated-cohorts;
cohort_eligibility_criteria stays the complete criteria; cohort_criteria_items holds every line with its
inclusion/exclusion and boilerplate tags). Training's retrieval and match labels read the key criteria, the
boilerplate labels the boilerplate criteria, and the per-criterion labels the tagged lines. Checkpoints retain each trial's decision, cohort names, unmerged,
merged and de-boilerplated rows, reasoning and answers of every step, validation
attempts per step and endpoint provenance.

Input: an already-sampled set of ctgov records (--trials-jsonl: the seeds.jsonl of a
synthetic protocol batch, drawn by ``trials.screen_trials`` and ``trials.sample_trials``), every one of
which is extracted. This gives each sampled trial its ctgov-derived cohorts beside the cohorts
extracted from its synthetic protocol (synthetic_protocols/README.md); in ``training.pipeline`` it is
the ctgov-cohorts stage. (Until 2026-11-05 this was 0b_extract_trial_cohorts.py, which could
also select trials from the parsed export itself; the ctgov rule now lives in the screen. Until 2026-11-05 "ctgov" was called
"registry", and this script 0d_extract_registry_cohorts.py.)
"""
import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import ExitStack
import csv
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import json
from pathlib import Path

import pandas as pd

from matchminer_ai.training.trials.ctgov_llm import add_llm_arguments, make_config, prepare_runtime
from matchminer_ai.training.cohort_format import REPRESENTATION, validate_cohort_rows, file_stamp
from matchminer_ai.training.paths import core_sources, module_file, no_phi


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(temporary, 'wt') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    temporary.replace(path)


def load_sampled_trials(path):
    """ctgov records of an already-sampled trial set; no selection or split assignment."""
    frame = pd.DataFrame([json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()])
    required = {'nct_id', 'title', 'brief_summary', 'eligibility_criteria'}
    if frame.empty or required - set(frame):
        raise ValueError(f'Sampled trials need columns {sorted(required)}')
    if not frame.nct_id.str.fullmatch(r'NCT\d{8}', na=False).all():
        raise ValueError('Expected ClinicalTrials.gov NCT identifiers')
    if frame.nct_id.duplicated().any():
        raise ValueError('Duplicate NCT IDs in the sampled trials')
    for column in ('title', 'brief_summary', 'eligibility_criteria'):
        frame[column] = frame[column].fillna('')
    return frame.sort_values('nct_id').reset_index(drop=True)


def extract_one(row, config, merge=True, deboilerplate=True):
    from matchminer_ai.trials import merge_equivalent_cohorts, remove_boilerplate_criteria, extract_cohorts
    from matchminer_ai.trials.cohorts import COHORT_COLUMNS
    trial = row['nct_id']
    inputs = pd.DataFrame([dict(trial_id=trial, trial_title=row['title'],
        brief_summary=row['brief_summary'], eligibility_criteria=row['eligibility_criteria'])])
    record = dict(trial_id=trial, result=None, cohort_rows=[])
    try:
        cohorts, metadata = extract_cohorts(inputs, config=config, return_metadata=True)
        record.update(result=metadata['trial_decisions'].get(trial),
            error=metadata['trial_errors'].get(trial), metadata=metadata,
            cohort_rows=json.loads(cohorts[COHORT_COLUMNS].to_json(orient='records')))
        if record['result'] is None:
            record['error'] = record['error'] or 'No validated decision returned by extract_cohorts'
        elif record['result']['cancer_treatment_trial'] == 'yes':
            if merge and len(cohorts) > 1:
                # Training keeps one row per distinct population; the extraction stays in the checkpoint.
                merged, merging = merge_equivalent_cohorts(cohorts[COHORT_COLUMNS], config=config, return_metadata=True)
                record.update(unmerged_cohort_rows=record['cohort_rows'],
                              cohort_rows=json.loads(merged.to_json(orient='records')),
                              merge={key: merging[key] for key in ('schema_version', 'prompt_sha256', 'trial_groups',
                                                                   'merge_errors', 'trial_attempts')})
            if deboilerplate:
                rows = pd.DataFrame(record['cohort_rows'])
                clean, cleaning = remove_boilerplate_criteria(rows, config=config, return_metadata=True)
                record.update(deboilerplated_cohort_rows=json.loads(clean.to_json(orient='records')),
                              deboilerplate={'schema_version': cleaning['schema_version'],
                                             'prompt_sha256': cleaning['prompt_sha256'],
                                             'cohorts': cleaning['trial_cohorts'][trial],
                                             'errors': cleaning['deboilerplate_errors'],
                                             'unclassified_items': cleaning['unclassified_items']})
            for key in ('cohort_rows', 'deboilerplated_cohort_rows') if deboilerplate else ('cohort_rows',):
                rows = pd.DataFrame(record[key])
                rows['cohort_count'] = len(rows)
                validate_cohort_rows(rows)
        elif record['cohort_rows']:
            raise ValueError('Rejected trial unexpectedly returned cohort rows')
    except Exception as exc:
        record.update(result=None, cohort_rows=[], error=f'{type(exc).__name__}: {exc}')
    record['completed_at'] = datetime.now(timezone.utc).isoformat()
    return record


def read_checkpoint(path, fingerprint):
    with gzip.open(path, 'rt') as stream:
        record = json.load(stream)
    if record['run_identity'] != fingerprint:
        raise ValueError(f'Checkpoint identity mismatch: {path}')
    return record


def output_paths(args):
    """Cohort rows, trial audit, decisions JSON, then (when those steps run) the rows before merging
    and the rows with boilerplate removed."""
    paths = [args.output_cohorts, args.output_trials, args.output_json]
    if args.merge_cohorts:
        paths.append(args.output_unmerged_cohorts
                     or args.output_cohorts.with_name(f'{args.output_cohorts.stem}_unmerged{args.output_cohorts.suffix}'))
    if args.deboilerplate_cohorts:
        paths.append(args.output_deboilerplated_cohorts
                     or args.output_cohorts.with_name(f'{args.output_cohorts.stem}_deboilerplated{args.output_cohorts.suffix}'))
    return paths


def export_results(frame, args, fingerprint):
    from matchminer_ai.trials.cohorts import COHORT_COLUMNS
    # Stream exports; long reasoning traces stay once per trial in checkpoints.
    from matchminer_ai.trials.cohort_merge import MERGED_FROM
    from matchminer_ai.trials.cohort_boilerplate import BOILERPLATE_COLUMN, ITEMS_COLUMN, KEY_COLUMN
    row_fields = ['cohort_count', 'cohort_index', 'trial_representation']
    cohort_fields = list(frame.columns) + COHORT_COLUMNS + [MERGED_FROM, *row_fields]
    audit_fields = list(frame.columns) + ['cancer_treatment_trial', 'multiple_cohorts',
                                        'cohort_count', 'cohort_result_json', 'extraction_error']
    errors, index, unmerged_index, clean_index, unremoved = [], 0, 0, 0, 0
    outputs = output_paths(args)
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    temps = [p.with_suffix(p.suffix + '.tmp') for p in outputs]
    with ExitStack() as stack:
        streams = [stack.enter_context(p.open('w', newline='')) for p in temps]
        rows_writer = csv.DictWriter(streams[0], fieldnames=cohort_fields)
        audit_writer = csv.DictWriter(streams[1], fieldnames=audit_fields)
        json_out = streams[2]
        extra = iter(streams[3:])
        # Before merging: every extracted cohort (a trial with one cohort, or not merged, is unchanged).
        unmerged_writer = csv.DictWriter(next(extra), fieldnames=list(frame.columns) + COHORT_COLUMNS + row_fields) \
            if args.merge_cohorts else None
        # The merged rows (same rows and order) with their key and boilerplate criteria in their own columns,
        # and every criteria line with its inclusion/exclusion and boilerplate tags (a JSON string).
        clean_writer = csv.DictWriter(next(extra), fieldnames=cohort_fields + [KEY_COLUMN, BOILERPLATE_COLUMN, ITEMS_COLUMN]) \
            if args.deboilerplate_cohorts else None
        for writer in (rows_writer, audit_writer, unmerged_writer, clean_writer):
            if writer:
                writer.writeheader()
        json_out.write('{\n')
        separator = ''
        for source in frame.astype(object).where(frame.notna(), None).to_dict('records'):
            record = read_checkpoint(args.work_dir / 'checkpoints' / (source['nct_id'] + '.json.gz'), fingerprint)
            result = record['result']
            count = len(record['cohort_rows'])
            audit_writer.writerow({**source, 'cancer_treatment_trial': (result or {}).get('cancer_treatment_trial'),
                'multiple_cohorts': (result or {}).get('multiple_cohorts'), 'cohort_count': count,
                'cohort_result_json': json.dumps(result), 'extraction_error': record.get('error')})
            if result is None:
                errors.append({'trial_id': source['nct_id'], 'error': record.get('error')})
                continue
            json_out.write(separator + json.dumps(source['nct_id']) + ': ' + json.dumps(result, ensure_ascii=False))
            separator = ',\n'
            for cohort in record['cohort_rows']:
                rows_writer.writerow({**source, **cohort, 'cohort_count': count,
                                      'cohort_index': index, 'trial_representation': REPRESENTATION})
                index += 1
            if unmerged_writer:
                extracted = record.get('unmerged_cohort_rows', record['cohort_rows'])
                for cohort in extracted:
                    unmerged_writer.writerow({**source, **{k: v for k, v in cohort.items() if k != MERGED_FROM},
                                              'cohort_count': len(extracted), 'cohort_index': unmerged_index,
                                              'trial_representation': REPRESENTATION})
                    unmerged_index += 1
            if clean_writer:
                unremoved += len((record.get('deboilerplate') or {}).get('errors') or {})
                merged = {row['cohort_number']: row for row in record['cohort_rows']}
                for cohort in record.get('deboilerplated_cohort_rows', []):  # none for rejected trials
                    if KEY_COLUMN not in cohort:  # checkpoints before the key column kept key text here
                        cohort = {**cohort, KEY_COLUMN: cohort['cohort_eligibility_criteria'],
                                  'cohort_eligibility_criteria': merged[cohort['cohort_number']]['cohort_eligibility_criteria']}
                    clean_writer.writerow({**source, **cohort, 'cohort_count': count,
                                           'cohort_index': clean_index, 'trial_representation': REPRESENTATION})
                    clean_index += 1
        json_out.write('\n}\n')
    partial = errors and not args.allow_failures
    for temporary, path in zip(temps, outputs):
        destination = path.with_suffix('.partial' + path.suffix) if partial else path
        temporary.replace(destination)
    atomic_json(args.work_dir / 'errors.json', errors)
    if partial:
        raise RuntimeError(f'{len(errors)} trials failed; partial exports saved. Rerun to retry failures before sampling.')
    if errors:
        print(f'{len(errors)} trials failed every attempt and have no cohorts (errors.json)', flush=True)
    print(f'Exported {len(frame)} decisions and {index} cohort rows'
          + (f' ({unmerged_index} before merging)' if args.merge_cohorts else '')
          + (f'; {clean_index} with boilerplate removed' if args.deboilerplate_cohorts else ''), flush=True)
    if unremoved:
        print(f'{unremoved} cohorts kept their boilerplate: the boilerplate step failed every attempt', flush=True)


def run(args, config):
    from matchminer_ai.config import config_snapshot
    from matchminer_ai.trials.cohort_boilerplate import DEBOILERPLATE_PROMPT, DEBOILERPLATE_SCHEMA_VERSION
    from matchminer_ai.trials.cohort_merge import MERGE_PROMPT, MERGE_SCHEMA_VERSION
    from matchminer_ai.trials.cohort_workflow import SCHEMA_VERSION, load_prompts, prompt_sha256
    from matchminer_ai.llm.prompts import load_prompt_text
    frame = load_sampled_trials(args.trials_jsonl)
    source = args.trials_jsonl
    identity = dict(input=str(source.resolve()), input_stamp=file_stamp(source), selection='sampled_trials',
        schema=SCHEMA_VERSION, prompt_sha256=prompt_sha256(load_prompts(config.trial['cohorts'].get('prompt_files'))),
        merge=dict(schema=MERGE_SCHEMA_VERSION, prompt_sha256=hashlib.sha256(load_prompt_text(MERGE_PROMPT).encode()).hexdigest())
        if args.merge_cohorts else None,
        deboilerplate=dict(schema=DEBOILERPLATE_SCHEMA_VERSION,
                           prompt_sha256=hashlib.sha256(load_prompt_text(DEBOILERPLATE_PROMPT).encode()).hexdigest())
        if args.deboilerplate_cohorts else None,
        config=config_snapshot(config), source_sha256=core_sources(),
        runner_sha256=sha_file(__file__), adapter_sha256=sha_file(module_file('cohort_format')),
        outputs=[str(p.resolve()) for p in output_paths(args)])
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    manifest = args.work_dir / 'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['run_identity'] != fingerprint:
        raise ValueError('Extraction inputs, prompt, code or settings changed; use a fresh --work-dir')
    if not manifest.exists() and any(p.exists() for p in output_paths(args)):
        raise ValueError('Outputs already exist without this run identity; use fresh output paths')
    atomic_json(manifest, dict(run_identity=fingerprint, identity=identity,
        selected_trials=len(frame), api='matchminer_ai.trials.extract_cohorts'))
    directory = args.work_dir / 'checkpoints'
    directory.mkdir(exist_ok=True)
    pending, completed = [], 0
    for row in frame.to_dict('records'):
        path = directory / (row['nct_id'] + '.json.gz')
        if path.exists() and read_checkpoint(path, fingerprint)['result'] is not None:
            completed += 1
        else:
            pending.append(row)
    print(f'Selected {len(frame)} trials; resuming {completed} validated decisions; extracting {len(pending)}', flush=True)
    if pending:
        # Resolve once before launching workers, including an honest fallback for
        # served aliases. Avoid thousands of simultaneous Hub metadata lookups.
        from matchminer_ai.llm.backends import get_endpoint_model_metadata
        metadata = get_endpoint_model_metadata(args.model, cache_dir=config.model_metadata_cache_dir)
        atomic_json(Path(config.model_metadata_cache_dir) / (args.model.replace('/', '_') + '.json'), metadata)
    iterator = iter(pending)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        while True:
            while len(futures) < args.workers:
                row = next(iterator, None)
                if row is None:
                    break
                futures[executor.submit(extract_one, row, config, args.merge_cohorts, args.deboilerplate_cohorts)] = row['nct_id']
            if not futures:
                break
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                trial = futures.pop(future)
                record = future.result()
                record['run_identity'] = fingerprint
                atomic_json(directory / (trial + '.json.gz'), record)
                completed += 1
            atomic_json(args.work_dir / 'status.json', dict(processed=completed, total=len(frame)))
    export_results(frame, args, fingerprint)
    atomic_json(args.work_dir / 'status.json', dict(processed=completed, total=len(frame), status='completed'))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trials-jsonl', type=Path, required=True,
                   help='Already-sampled ctgov records (nct_id, title, brief_summary, eligibility_criteria), '
                        'such as a synthetic protocol batch seeds.jsonl; every one is extracted')
    p.add_argument('--output-trials', type=Path, help='Default: <data root>/no_phi/trials_with_cohorts.csv')
    p.add_argument('--output-cohorts', type=Path, help='Default: <data root>/no_phi/trial_cohort_lineitems.csv')
    p.add_argument('--output-json', type=Path, help='Default: <data root>/no_phi/trial_cohorts.json')
    p.add_argument('--output-unmerged-cohorts', type=Path,
                   help='Cohort rows before merging; default <output-cohorts stem>_unmerged.csv (not with --no-merge-cohorts)')
    p.add_argument('--output-deboilerplated-cohorts', type=Path,
                   help='Merged cohort rows with boilerplate removed; default <output-cohorts stem>_deboilerplated.csv '
                        '(not with --no-deboilerplate-cohorts)')
    p.add_argument('--work-dir', type=Path, help='Default: <data root>/no_phi/trial_cohort_extraction')
    p.add_argument('--allow-failures', action='store_true',
                   help='Export complete files even if some trials fail every attempt (listed in errors.json); '
                        'for ctgov versions of sampled trials, whose protocol versions remain')
    p.add_argument('--no-merge-cohorts', dest='merge_cohorts', action='store_false',
                   help='Keep every extracted cohort instead of combining those with essentially the same criteria')
    p.add_argument('--no-deboilerplate-cohorts', dest='deboilerplate_cohorts', action='store_false',
                   help='Do not remove boilerplate criteria (no --output-deboilerplated-cohorts export)')
    return add_llm_arguments(p)


def main(argv=None):
    args = parser().parse_args(argv)
    for key, name in (('output_trials', 'trials_with_cohorts.csv'), ('output_cohorts', 'trial_cohort_lineitems.csv'),
                      ('output_json', 'trial_cohorts.json'), ('work_dir', 'trial_cohort_extraction')):
        if getattr(args, key) is None:
            setattr(args, key, no_phi(name))
    destinations = [p.resolve() for p in output_paths(args)]
    if len(set(destinations)) != len(destinations) or args.trials_jsonl.resolve() in destinations:
        raise ValueError('Input and all output paths must be distinct')
    prepare_runtime(args)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    with (args.work_dir / 'extraction.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args, make_config(args))


if __name__ == '__main__':
    main()
