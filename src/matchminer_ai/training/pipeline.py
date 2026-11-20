#!/usr/bin/env python3
"""Resumable CT.gov → synthetic protocols → synthetic notes → summaries → model training orchestration.

The trial sample is drawn from the CT.gov export in this order (user instruction,
2026-11-04): the ctgov rule (first posted since --start-year, interventional,
early phase I-IV), the DFCI holdout (trials open at DFCI in --dfci-first-year to
--dfci-last-year), the cohort workflow's cancer-treatment screen (screen-trials,
every answer and its reasoning saved), then a seeded draw of --sample-trials
(sample-trials). Each sampled trial gets a synthetic full protocol. Patients are
generated from the protocol cohorts; retrieval, labeling and training see every
trial in both versions, its protocol cohorts and its ctgov cohorts
(retrieval-cohorts), so the models work for users holding either source.

The default starts at training (``mmai-train`` with no stage options).
Select an earlier --start-at and optional --stop-after to run data production.
Serving/VM ownership stays with the external supervisor for upstream stages.
"""
import argparse
from dataclasses import dataclass, asdict
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import matchminer_ai.training.from_summaries as training
from matchminer_ai.training.cohort_format import file_stamp
from matchminer_ai.training.paths import PACKAGE, child_env, core_files, module_command, module_file
# Every sampled trial gets a synthetic protocol; patients are generated from the
# protocol's cohorts, and retrieval/labeling/training see both the protocol and
# the ctgov (ClinicalTrials.gov) version of each trial's cohorts. Until 2026-11-05
# "ctgov" was called "registry"; the old stage name and options are still accepted.
STAGES = ('parse-trials', 'screen-trials', 'sample-trials', 'synthetic-protocols', 'protocol-cohorts',
          'ctgov-cohorts', 'generation-cohorts', 'retrieval-cohorts', 'generate-synthetic',
          'prepare-notes', 'summarize-patients', 'prepare-training', 'train')


def stage_name(value):
    """A stage name; 'registry-cohorts' (its name before 2026-11-05) is ctgov-cohorts."""
    return 'ctgov-cohorts' if value == 'registry-cohorts' else value


LLM_STAGES = {'screen-trials', 'synthetic-protocols', 'protocol-cohorts', 'ctgov-cohorts',
              'generate-synthetic', 'summarize-patients'}


@dataclass
class Stage:
    name: str
    command: list[str]
    inputs: list[Path]
    outputs: list[Path]
    sources: list[Path]


def parser():
    p = training.parser()
    p.description = __doc__
    p.add_argument('--start-at', type=stage_name, choices=STAGES, default='train')
    p.add_argument('--stop-after', type=stage_name, choices=STAGES, default='train')
    p.add_argument('--pipeline-dir', type=Path, help='Upstream state/log directory; default RUN_DIR/upstream')
    p.add_argument('--ctgov-json', type=Path)
    p.add_argument('--parsed-trials', type=Path)
    p.add_argument('--start-year', type=int, default=2020)
    p.add_argument('--ctgov-sample', '--registry-sample', type=Path,
                   help='sample-trials output: the sampled trials (protocol seeds, source splits); '
                        'default DATA_DIR/ctgov_sample_trials.csv')
    p.add_argument('--screened-trials', type=Path,
                   help='screen-trials output: every trial meeting the ctgov rule, with its DFCI holdout '
                        'flag and screen decision; default DATA_DIR/screened_trials.csv')
    p.add_argument('--protocol-dir', type=Path, help='Synthetic protocol batch; default DATA_DIR/synthetic_protocols')
    p.add_argument('--protocol-cohort-dir', type=Path,
                   help='protocol-cohorts output, e.g. to re-extract an existing batch into a new run; '
                        'default PROTOCOL_DIR/cohort_extraction')
    p.add_argument('--protocol-max-tokens', type=int, default=100000)
    p.add_argument('--protocol-chunk-size', type=int, default=500)
    p.add_argument('--sampled-spaces', '--sampled-cohorts', dest='sampled_spaces', type=Path,
                   help='Patient-generation cohorts (the protocol cohorts); default GENERATION_DIR/sample_trial_cohort_lineitems.csv')
    p.add_argument('--retrieval-cohorts', type=Path,
                   help='Both versions of every trial for retrieval and training; default DATA_DIR/retrieval_cohorts.csv')
    p.add_argument('--sample-trials', type=int, default=5000)
    p.add_argument('--cohort-workers', type=int, default=1536)
    p.add_argument('--cohort-max-attempts', type=int, choices=range(1, 11), default=3)
    p.add_argument('--ctgov-cohort-max-attempts', '--registry-cohort-max-attempts', type=int, choices=range(1, 11), default=5,
                   help='Attempts per record for ctgov-cohorts (cohorts from the protocol seeds\' ctgov records)')
    p.add_argument('--ctgov-cohort-concurrency-per-server', '--registry-cohort-concurrency-per-server', type=int, choices=range(1, 1025), metavar='1..1024',
                   default=512, help='Per-server request ceiling for ctgov-cohorts')
    p.add_argument('--protocol-cohort-chunks-in-flight', type=int, choices=range(1, 17), metavar='1..16', default=16,
                   help='protocol-cohorts keeps 500 x this many requests in flight in each step (extraction, merge, '
                        'boilerplate), each sent as another returns; the default 8,000 fills the GPU fleet '
                        '(ctgov-cohorts runs per listing)')
    p.add_argument('--cohort-max-tokens', type=int, default=50000)
    p.add_argument('--data-seed', type=int, default=42)
    p.add_argument('--dfci-trials-csv', type=Path, help='Default: DATA_DIR/dfci_trials_summary_10-18-26.csv')
    p.add_argument('--dfci-first-year', type=int, default=2025)
    p.add_argument('--dfci-last-year', type=int, default=2026)
    p.add_argument('--generation-dir', type=Path)
    p.add_argument('--synthetic-notes', type=Path,
                   help='Prepared synthetic notes (prepare-notes output); default: DATA_DIR/all_synthetic_notes.parquet')
    p.add_argument('--raw-synthetic-notes', type=Path)
    p.add_argument('--server-urls-file', type=Path, help='Externally managed live pool for upstream LLM stages')
    p.add_argument('--concurrency-control-file', type=Path,
                   help='Live per-server limits for the upstream generation and summary stages; '
                        'labeling uses --teacher-concurrency-file')
    p.add_argument('--generation-model', help='API model ID/alias; defaults to the teacher model/alias')
    p.add_argument('--generation-tokenizer', help='HF tokenizer/model-family ID; defaults to --teacher-model')
    p.add_argument('--download-dir', type=Path)
    p.add_argument('--generation-context', type=int, default=250000)
    p.add_argument('--generation-concurrency-per-server', type=int, default=1024)
    p.add_argument('--generation-scheduler-adaptive-concurrency', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--generation-initial-concurrency-per-server', type=int, default=64)
    p.add_argument('--generation-pending-requests', type=int, default=8192)
    p.add_argument('--generation-request-timeout', type=float, default=7200)
    p.add_argument('--generation-max-attempts', type=int, default=16)
    p.add_argument('--generation-seed-max-tokens', type=int, default=16000)
    p.add_argument('--generation-event-max-tokens', type=int, default=20000)
    p.add_argument('--generation-note-preparation-workers', type=int, default=0,
                   help='CPU processes for note inputs: 0 uses up to 8 available CPUs; 1 is serial')
    p.add_argument('--api-key-env', default='OPENAI_API_KEY')
    p.add_argument('--wait-for-generation', action='store_true',
                   help='Observe the existing generation job until complete; never launch or interrupt it')
    p.add_argument('--generation-wait-timeout', type=float, default=0, help='Seconds; 0 waits indefinitely')
    p.add_argument('--synthetic-date-anchor', default='2025-01-01')
    p.add_argument('--note-preparation-scratch-dir', type=Path,
                   help='Local SSD scratch for SQLite and temporary prepared notes; final products stay in --data-dir')
    p.add_argument('--stage-note-preparation-input', action='store_true',
                   help='Copy source notes to local scratch before preparation')
    p.add_argument('--summary-resume-dir', type=Path)
    p.add_argument('--summary-start-gate', type=Path)
    p.add_argument('--summary-scratch-dir', type=Path)
    p.add_argument('--summary-chunk-size', type=int, default=50000)
    p.add_argument('--summary-chunk-overlap', type=int, default=500)
    p.add_argument('--summary-max-tokens', type=int, default=20000)
    return p


def resolve_paths(args):
    training.resolve_defaults(args)
    args.dfci_trials_csv = args.dfci_trials_csv or args.data_dir / 'dfci_trials_summary_10-18-26.csv'
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser().resolve())
    defaults = dict(pipeline_dir=args.run_dir / 'upstream', parsed_trials=args.data_dir / 'ctgov_trials.csv',
                    trial_spaces=args.data_dir / 'trial_cohort_lineitems.csv',
                    generation_dir=args.data_dir / 'synthetic_generation',
                    synthetic_notes=args.data_dir / 'all_synthetic_notes.parquet',
                    patient_summaries=args.data_dir / 'patient_summaries_with_spaces.parquet')
    for key, value in defaults.items():
        if args.start_at == 'train' and key in ('trial_spaces', 'synthetic_notes', 'patient_summaries'):
            continue
        if getattr(args, key) is None:
            setattr(args, key, value)
    args.ctgov_sample = args.ctgov_sample or args.data_dir / 'ctgov_sample_trials.csv'
    args.screened_trials = args.screened_trials or args.data_dir / 'screened_trials.csv'
    args.protocol_dir = args.protocol_dir or args.data_dir / 'synthetic_protocols'
    args.protocol_cohort_dir = args.protocol_cohort_dir or args.protocol_dir / 'cohort_extraction'
    args.retrieval_cohorts = args.retrieval_cohorts or args.data_dir / 'retrieval_cohorts.csv'
    args.sampled_spaces = args.sampled_spaces or args.generation_dir / 'sample_trial_cohort_lineitems.csv'
    args.raw_synthetic_notes = args.raw_synthetic_notes or args.generation_dir / 'all_synthetic_notes.parquet'
    args.generation_model = args.generation_model or args.teacher_served_model or args.teacher_model
    args.generation_tokenizer = args.generation_tokenizer or args.teacher_model
    if args.start_at != 'train' and args.server_urls_file and not args.teacher_url and not args.teacher_servers_file:
        args.teacher_servers_file = args.server_urls_file
        args.teacher_served_model = args.teacher_served_model or args.generation_model
        # Labeling keeps its own limits (--teacher-concurrency-file, --label-concurrency-per-server):
        # its short gate/follow-up requests (~4k tokens) let a server run several times more of
        # them than the long generation and summary requests --concurrency-control-file is tuned for.
    return args


def training_arguments(args):
    # The upstream extraction contains the full denominator. Training retrieves
    # over the sampled trials in both versions: the protocol cohorts that seeded
    # synthetic generation and the same trials' ctgov cohorts.
    if args.start_at == 'train':
        return args
    return argparse.Namespace(**{**vars(args), 'trial_spaces': args.retrieval_cohorts})


def training_command(args):
    args = training_arguments(args)
    command = module_command('from_summaries')
    for action in training.parser()._actions:
        if action.dest in ('help', 'dry_run'):
            continue
        value = getattr(args, action.dest)
        if value is None:
            continue
        option = action.option_strings[0]
        if isinstance(action, argparse._StoreTrueAction):
            if value:
                command.append(option)
        elif isinstance(action, argparse._AppendAction):
            for item in value:
                command += [option, str(item)]
        elif isinstance(value, (list, tuple)):
            command += [option, *map(str, value)]
        else:
            command += [option, str(value)]
    return command


def unmerged(cohorts_csv):
    """trials.extract_ctgov_cohorts's export of the cohort rows before merging (None when training starts later)."""
    return cohorts_csv and cohorts_csv.with_name(f'{cohorts_csv.stem}_unmerged{cohorts_csv.suffix}')


def deboilerplated(cohorts_csv):
    """The merged cohort rows with boilerplate removed, beside a cohort table (ctgov extraction's or patient_cohorts')."""
    return cohorts_csv and cohorts_csv.with_name(f'{cohorts_csv.stem}_deboilerplated{cohorts_csv.suffix}')


def build_plan(args):
    if STAGES.index(args.start_at) > STAGES.index(args.stop_after):
        raise ValueError('--start-at must precede --stop-after')
    selected = STAGES[STAGES.index(args.start_at):STAGES.index(args.stop_after) + 1]
    if args.wait_for_generation and 'generate-synthetic' not in selected:
        raise ValueError('--wait-for-generation requires the generate-synthetic stage in the selected range')
    d, g, work = args.data_dir, args.generation_dir, args.pipeline_dir
    py = sys.executable
    registry = args.server_urls_file or args.teacher_servers_file
    remote = ['--server_urls_file', str(registry)] if registry else []
    remote += ['--max_concurrent_per_server', str(args.generation_concurrency_per_server),
               '--request_timeout', str(args.generation_request_timeout),
               '--max_attempts', str(args.generation_max_attempts), '--api-key-env', args.api_key_env]
    if args.concurrency_control_file:
        remote += ['--concurrency-control-file', str(args.concurrency_control_file)]
    model = ['--model', args.generation_model, '--tokenizer-model', args.generation_tokenizer]
    cache = ['--download-dir', str(args.download_dir)] if args.download_dir else []
    source = lambda *names: [module_file(name) for name in names]
    screening = [*module_command('trials.screen_trials', py), '--input', str(args.parsed_trials),
                 '--start-year', str(args.start_year), '--seed', str(args.data_seed),
                 '--dfci-trials-csv', str(args.dfci_trials_csv), '--dfci-first-year', str(args.dfci_first_year),
                 '--dfci-last-year', str(args.dfci_last_year), '--output', str(args.screened_trials),
                 '--work-dir', str(work / 'screen'), '--workers', str(args.cohort_workers),
                 '--max-attempts', str(args.cohort_max_attempts), '--max-tokens', str(args.cohort_max_tokens),
                 '--max_concurrent_per_server', str(args.generation_concurrency_per_server),
                 '--request_timeout', str(args.generation_request_timeout), '--api-key-env', args.api_key_env,
                 *model, *(['--server_urls_file', str(registry)] if registry else [])]
    inference_sources = PACKAGE
    # The adaptive teacher pool (training's remote_vllm_pool.py before 2026-11-06).
    pool_sources = [inference_sources / 'llm' / name for name in ('teacher_pool.py', 'scheduler_control.py', 'teacher_runtime.py')]
    package = core_files()  # the inference core, so edits to training, evaluation or the apps don't invalidate
    pool = ['--server-urls-file', str(registry)] if registry else []
    batch, extracted = args.protocol_dir, args.protocol_cohort_dir
    ctgov_cohorts = d / 'ctgov_cohorts'
    # Merged rows (used downstream), audit, decisions, then the rows before merging.
    ctgov_rows = [ctgov_cohorts / 'trial_cohort_lineitems.csv', ctgov_cohorts / 'trials_with_cohorts.csv',
                     ctgov_cohorts / 'trial_cohorts.json', unmerged(ctgov_cohorts / 'trial_cohort_lineitems.csv'),
                     deboilerplated(ctgov_cohorts / 'trial_cohort_lineitems.csv')]
    ctgov_extraction = [*module_command('trials.extract_ctgov_cohorts', py), '--trials-jsonl', str(batch / 'seeds.jsonl'),
                  '--output-trials', str(ctgov_rows[1]), '--output-cohorts', str(ctgov_rows[0]),
                  '--output-json', str(ctgov_rows[2]), '--output-unmerged-cohorts', str(ctgov_rows[3]),
                  '--output-deboilerplated-cohorts', str(ctgov_rows[4]),
                  '--work-dir', str(work / 'ctgov_extraction'), '--workers', str(args.cohort_workers),
                  '--max-attempts', str(args.ctgov_cohort_max_attempts), '--max-tokens', str(args.cohort_max_tokens),
                  '--max_concurrent_per_server', str(args.ctgov_cohort_concurrency_per_server),
                  '--request_timeout', str(args.generation_request_timeout), '--api-key-env', args.api_key_env,
                  '--allow-failures', *model, *(['--server_urls_file', str(registry)] if registry else [])]
    generated = [g / name for name in ('all_synthetic_notes.parquet', 'all_synthetic_events.parquet', 'manifest.json')]
    patient_map = d / 'synthetic_patient_metadata.parquet'
    raw_summaries = d / 'patient_summaries.parquet'
    candidates = [
        Stage('parse-trials', [*module_command('trials.parse_ctgov', py), '--input', str(args.ctgov_json),
                              '--output', str(args.parsed_trials)],
              [args.ctgov_json] if args.ctgov_json else [], [args.parsed_trials], source('trials.parse_ctgov', 'trials.ctgov_metadata')),
        # ctgov rule, DFCI holdout and the cancer-treatment screen; every screen answer is saved.
        Stage('screen-trials', screening, [args.parsed_trials, args.dfci_trials_csv],
              [args.screened_trials, Path(f'{args.screened_trials}.screening.json')],
              source('trials.screen_trials', 'trials.ctgov_llm', 'trials.sample_trials',
                     'cohort_format') + package),
        Stage('sample-trials', [*module_command('trials.sample_trials', py), '--screened-trials', str(args.screened_trials),
              '--output-csv', str(args.ctgov_sample), '--n-trials', str(args.sample_trials), '--seed', str(args.data_seed)],
              [args.screened_trials, Path(f'{args.screened_trials}.screening.json')],
              [args.ctgov_sample, args.ctgov_sample.with_suffix('.sampling.json')],
              source('trials.sample_trials', 'cohort_format')),
        # One synthetic full protocol per sampled trial, seeded from its ctgov record.
        Stage('synthetic-protocols', [*module_command('trials.synthetic_protocols.generate_batch', py),
              '--nct-ids-from', str(args.ctgov_sample), '--output-dir', str(batch),
              '--chunk-size', str(args.protocol_chunk_size), '--model-name', args.generation_tokenizer,
              '--max-tokens', str(args.protocol_max_tokens), '--allow-failures', *pool],
              [args.ctgov_sample], [batch / 'seeds.jsonl', batch / 'protocols.csv', batch / 'run_metadata.json'],
              source('trials.synthetic_protocols.generate_batch') + package),
        Stage('protocol-cohorts', [*module_command('trials.synthetic_protocols.extract_cohorts', py), '--batch-dir', str(batch),
              '--output-dir', str(extracted), '--model-name', args.generation_tokenizer,
              '--max-tokens', str(args.cohort_max_tokens), '--chunks-in-flight', str(args.protocol_cohort_chunks_in_flight),
              '--allow-failures', *pool],
              [batch / 'seeds.jsonl', batch / 'protocols.csv'],
              [extracted / 'cohorts.parquet', extracted / 'trials.parquet', extracted / 'run_metadata.json',
               extracted / 'cohorts_unmerged.parquet', extracted / 'cohorts_deboilerplated.parquet'],
              source('trials.synthetic_protocols.extract_cohorts') + package),
        # The same trials from the ctgov records their protocols were written
        # from, with the ctgov prompts that follow the protocol prompts' rules.
        Stage('ctgov-cohorts', ctgov_extraction, [batch / 'seeds.jsonl'], ctgov_rows,
              source('trials.extract_ctgov_cohorts', 'trials.ctgov_llm', 'cohort_format') + package),
        # Patients are seeded from the merged cohorts with all their criteria, boilerplate included
        # (user, 2026-11-04); retrieval reads the same cohorts with the boilerplate removed.
        Stage('generation-cohorts', [*module_command('trials.synthetic_protocols.patient_cohorts', py), '--batch-dir', str(batch),
              '--extraction-dir', str(extracted), '--output', str(args.sampled_spaces),
              '--retrieval-output', str(deboilerplated(args.sampled_spaces)),
              '--splits-from', str(args.ctgov_sample)],
              [extracted / 'cohorts.parquet', extracted / 'cohorts_deboilerplated.parquet', extracted / 'trials.parquet',
               args.ctgov_sample],
              [args.sampled_spaces, Path(f'{args.sampled_spaces}.provenance.json'), deboilerplated(args.sampled_spaces),
               Path(f'{deboilerplated(args.sampled_spaces)}.provenance.json')],
              source('trials.synthetic_protocols.patient_cohorts', 'cohort_format')),
        Stage('retrieval-cohorts', [*module_command('trials.synthetic_protocols.retrieval_cohorts', py),
              '--protocol-cohorts', str(deboilerplated(args.sampled_spaces)), '--ctgov-cohorts', str(ctgov_rows[4]),
              '--ctgov-trials', str(ctgov_rows[1]), '--output', str(args.retrieval_cohorts),
              '--splits-from', str(args.ctgov_sample)],
              [deboilerplated(args.sampled_spaces), ctgov_rows[4], ctgov_rows[1], args.ctgov_sample],
              [args.retrieval_cohorts, Path(f'{args.retrieval_cohorts}.provenance.json')],
              source('trials.synthetic_protocols.retrieval_cohorts', 'cohort_format')),
        Stage('generate-synthetic', [*module_command('synthetic.generate', py), '--input-csv', str(args.sampled_spaces),
              '--output-dir', str(g), '--seed', str(args.data_seed), '--max-model-len', str(args.generation_context),
              '--seed-max-tokens', str(args.generation_seed_max_tokens),
              '--event-max-tokens', str(args.generation_event_max_tokens),
              '--note-preparation-workers', str(args.generation_note_preparation_workers),
              '--scheduler-adaptive-concurrency' if args.generation_scheduler_adaptive_concurrency else '--no-scheduler-adaptive-concurrency',
              '--initial-concurrent-per-server', str(args.generation_initial_concurrency_per_server),
              '--pending-requests', str(args.generation_pending_requests), *model, *cache, *remote],
              [args.sampled_spaces], generated,
              [*source('synthetic.generate', 'synthetic.note_prompts',
                       'synthetic.event_prompts', 'vllm_reasoning', 'cohort_format'),
               *pool_sources]),
        Stage('prepare-notes', [*module_command('synthetic.prepare', py), 'notes',
              '--input', str(args.raw_synthetic_notes), '--sampled-spaces', str(args.sampled_spaces),
              '--output', str(args.synthetic_notes), '--patients', str(patient_map),
              '--date-anchor', args.synthetic_date_anchor,
              *(['--scratch-dir', str(args.note_preparation_scratch_dir)] if args.note_preparation_scratch_dir else []),
              *(['--stage-input'] if args.stage_note_preparation_input else [])],
              [args.raw_synthetic_notes, args.sampled_spaces], [args.synthetic_notes, patient_map], source('synthetic.prepare', 'cohort_format')),
        Stage('summarize-patients', [*module_command('summaries.summarize', py),
              '--input_parquet', str(args.synthetic_notes), '--output_parquet', str(d / 'patient_serial_summaries.parquet'),
              '--patient_summaries_parquet', str(raw_summaries), '--shard_dir', str(work / 'summaries'),
              '--max_model_len', str(args.generation_context), '--chunk_size', str(args.summary_chunk_size),
              '--chunk_overlap', str(args.summary_chunk_overlap), '--max_tokens', str(args.summary_max_tokens),
              '--scheduler-adaptive-concurrency' if args.generation_scheduler_adaptive_concurrency else '--no-scheduler-adaptive-concurrency',
              '--initial-concurrent-per-server', str(args.generation_initial_concurrency_per_server),
              '--pending-requests', str(args.generation_pending_requests),
              *(['--summary-resume-dir', str(args.summary_resume_dir)] if args.summary_resume_dir else []),
              *(['--summary-start-gate', str(args.summary_start_gate)] if args.summary_start_gate else []),
              *(['--summary-scratch-dir', str(args.summary_scratch_dir)] if args.summary_scratch_dir else []),
              *model, *(['--download_dir', str(args.download_dir)] if args.download_dir else []), *remote],
              [args.synthetic_notes], [raw_summaries, d / 'patient_serial_summaries.parquet'],
              [*source('summaries.summarize', 'summaries.workspace', 'trialspace.retrieval_workspace'),
               *(inference_sources / 'patients' / name for name in ('summarize.py', 'prepare.py', 'checkpoints.py')),
               inference_sources / 'patients/postprocess.py', inference_sources / 'patients/prompt_builder.py',
               inference_sources / 'patients/summary_quality.py',
               inference_sources / 'prompts/patient.serial.user.primer.txt',
               inference_sources / 'prompts/patient.serial.user.question.txt']),
        Stage('prepare-training', [*module_command('synthetic.prepare', py), 'summaries',
              '--input', str(raw_summaries), '--patients', str(patient_map), '--output', str(args.patient_summaries)],
              [raw_summaries, patient_map], [args.patient_summaries],
              [*source('synthetic.prepare'), inference_sources / 'patients/postprocess.py']),
        Stage('train', training_command(args), list(training.input_paths(training_arguments(args)).values()),
              training.stages(training_arguments(args))[-1].outputs, source('from_summaries')),
    ]
    return candidates[STAGES.index(args.start_at):STAGES.index(args.stop_after) + 1]


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, default=str, indent=2) + '\n')
    temporary.replace(path)


def file_record(path):
    """Stage inputs/outputs are large data files: identify them by size and mtime, never by reading them."""
    path = Path(path).resolve()
    return dict(path=str(path), stamp=file_stamp(path))


def source_record(path):
    """Source files are small; content hashes survive checkouts that only touch mtimes."""
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def generation_active(directory):
    path = Path(directory) / 'generation.lock'
    if not path.exists():
        return False
    with path.open('r') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
    return False


def wait_generation(args):
    """Read only: the existing supervisor remains responsible for restart/recovery."""
    manifest = json.loads((args.generation_dir / 'manifest.json').read_text())
    if manifest['input_stamp'] != file_stamp(args.sampled_spaces):
        raise ValueError('Running generation uses different sampled trial spaces')
    deadline = time.monotonic() + args.generation_wait_timeout if args.generation_wait_timeout else None
    while True:
        status = json.loads((args.generation_dir / 'status.json').read_text())
        if status['stage'] == 'completed':
            return
        if status['stage'] == 'failed':
            raise RuntimeError('Existing synthetic generation failed; see its status/log before continuing')
        if deadline and time.monotonic() >= deadline:
            raise TimeoutError('Timed out waiting for existing synthetic generation; it was left running')
        print(f"[wait] synthetic generation: {status['stage']}, saved={status.get('stage_completed', 0)}", flush=True)
        time.sleep(30)


def execute(args, plan):
    # Upstream outputs must remain in the explicitly non-PHI workspace.
    for path in (args.data_dir, args.generation_dir, args.pipeline_dir, args.protocol_dir, args.protocol_cohort_dir):
        if not training.in_no_phi(path):
            raise ValueError("Data-production directories must be inside the data root's no_phi tree")
    if args.synthetic_notes == args.raw_synthetic_notes:
        raise ValueError('Prepared notes must use a different path from raw generated notes')
    if any(stage.name == 'generate-synthetic' for stage in plan) and args.raw_synthetic_notes != args.generation_dir / 'all_synthetic_notes.parquet':
        raise ValueError('Generation writes raw notes inside --generation-dir; use --raw-synthetic-notes only when starting later')
    if any(stage.name == 'parse-trials' for stage in plan) and args.ctgov_json is None:
        raise ValueError('--ctgov-json is required when starting at parse-trials')
    llm_stages = {stage.name for stage in plan} & LLM_STAGES
    if llm_stages - ({'generate-synthetic'} if args.wait_for_generation else set()) and not (args.server_urls_file or args.teacher_servers_file):
        raise ValueError('Upstream LLM stages require --server-urls-file for an existing serving pool')
    args.pipeline_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    with (args.pipeline_dir / 'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        env = child_env(TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')
        for stage in plan:
            status_path = args.pipeline_dir / 'status.json'
            try:
                identity = dict(command=(training.training_identity_command(stage.command)
                                         if stage.name == 'train' else stage.command),
                                inputs=[file_record(p) for p in stage.inputs],
                                sources=[source_record(p) for p in [Path(__file__), *stage.sources]])
                fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
                request = args.pipeline_dir / 'requests' / f'{stage.name}.json'
                receipt = args.pipeline_dir / 'completed' / f'{stage.name}.json'
                if request.exists() and json.loads(request.read_text())['fingerprint'] != fingerprint:
                    raise ValueError(f'Inputs, code, or options changed for {stage.name}; use a fresh pipeline directory')
                if receipt.exists():
                    saved = json.loads(receipt.read_text())
                    if saved['fingerprint'] != fingerprint or saved['outputs'] != [file_record(p) for p in stage.outputs]:
                        raise ValueError(f'Changed or missing completed output for {stage.name}')
                    print(f'[skip] {stage.name}', flush=True)
                    continue
                atomic_json(request, dict(fingerprint=fingerprint, **identity, execution_command=stage.command))
                atomic_json(status_path, dict(stage=stage.name, status='running'))
                if stage.name == 'sample-trials' and generation_active(args.generation_dir):
                    raise RuntimeError('Cannot resample an active generation run; start at generate-synthetic with --wait-for-generation')
                print(f'[start] {stage.name}', flush=True)
                if stage.name == 'generate-synthetic' and (args.wait_for_generation or generation_active(args.generation_dir)):
                    wait_generation(args)
                else:
                    log = args.pipeline_dir / 'logs' / f'{stage.name}.log'
                    log.parent.mkdir(exist_ok=True)
                    for output in stage.outputs:
                        output.parent.mkdir(parents=True, exist_ok=True)
                    with log.open('a') as stream:
                        stage_env = dict(env, MMAI_LLM_TRACE_DIR=str(args.pipeline_dir / 'llm_traces' / stage.name),
                                         MMAI_LLM_TRACE_STAGE=stage.name)
                        subprocess.run(stage.command, cwd=args.pipeline_dir, env=stage_env, check=True, stdout=stream, stderr=subprocess.STDOUT)
                outputs = [file_record(p) for p in stage.outputs]
                atomic_json(receipt, dict(fingerprint=fingerprint, outputs=outputs))
                print(f'[done] {stage.name}', flush=True)
            except BaseException as exc:
                atomic_json(status_path, dict(stage=stage.name, status='failed', error=str(exc)))
                raise
        atomic_json(args.pipeline_dir / 'status.json', dict(status='completed', stop_after=args.stop_after))


def main(argv=None):
    raw = sys.argv[1:] if argv is None else argv
    args = resolve_paths(parser().parse_args(raw))
    plan = build_plan(args)
    if args.dry_run:
        display = []
        for stage in plan:
            display.extend([asdict(s) for s in training.stages(training_arguments(args))]
                           if stage.name == 'train' else [asdict(stage)])
        print(json.dumps(display, default=str, indent=2))
    elif args.start_at == args.stop_after == 'train':
        # Preserve the established trainer's own locking, identities and resume.
        subprocess.run(training_command(args), env=child_env(), check=True)
    else:
        execute(args, plan)


if __name__ == '__main__':
    main()
