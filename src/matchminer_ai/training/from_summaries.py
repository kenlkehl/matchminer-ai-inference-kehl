#!/usr/bin/env python3
"""Resumable TrialSpace → decision model pipeline.

Existing non-PHI summaries seed mining; the NVIDIA teacher provides all fresh labels.
TrialSpace (EmbeddingGemma 2) is mined and refit over
three rounds on the patients' key summary sections and the cohorts' key criteria. Every round
the teacher labels all seven decision-model question families: the five match labels of each
candidate pair (also TrialSpace's labels), the boilerplate label of each new unique boilerplate
combination, and each criteria line of a seeded fraction of the pairs (yes = the patient does
not clearly fail the line). Every round's labels fine-tune the decision model (LiquidAI d1-2B,
``training.decision_model``), which replaces TrialChecker and BoilerplateChecker. OncoReasoning
distillation and the training GoodOption catalog were removed on 2026-11-05 (user instruction);
GoodOption stays in the inference package, scored with LLMs.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

from matchminer_ai.training.trialspace.data import write_json as atomic_json
from matchminer_ai.training.vllm_reasoning import NEMOTRON_SERVE_FLAGS, resolve_parser_name, resolve_tool_call_parser
from matchminer_ai.labeling.concurrency import DEFAULT_CEILING, DEFAULT_INITIAL, DEFAULT_AUTO_WORKERS_MAX
from matchminer_ai.training.paths import child_env, models_root, module_command, no_phi
DEFAULT_TEACHER = 'nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4'
DEFAULT_EMBEDDER = 'google/embeddinggemma-2'
DEFAULT_DECISION_MODEL = 'LiquidAI/d1-2B'
DECISION_MODEL_REVISION = '652f920e544ea37f81e4b0ce89b81d4ed0da90b1'
EMBEDDINGGEMMA2_REVISION = '914f7f89142e33e77833254d9c9b90c3cef7303b'

TRIALSPACE_OPERATIONAL_OPTIONS = {
    'trialspace_save_steps', 'trialspace_save_total_limit', 'trialspace_stop_at', 'trialspace_pause_file',
}


def training_identity_command(command):
    """Exclude only checkpoint cadence and pause controls from upstream resume identity."""
    flags = {'--' + key.replace('_', '-') for key in TRIALSPACE_OPERATIONAL_OPTIONS}
    result, index = [], 0
    while index < len(command):
        if command[index] in flags:
            index += 2
        else:
            result.append(command[index])
            index += 1
    return result


@dataclass
class Stage:
    name: str
    command: list[str]
    outputs: list[Path]
    teacher: bool = False

    def check(self):
        """The stage script's ``--check`` (exit 0 complete, 10 not yet, 20 changed) and its environment.

        A distributed stage is checked without its launcher, with the world size it will run at."""
        if 'torch.distributed.run' not in self.command:
            return [*self.command, '--check'], {}
        module = self.command.index('-m', self.command.index('torch.distributed.run'))  # the stage's own -m
        ranks = next(part.split('=', 1)[1] for part in self.command if part.startswith('--nproc_per_node='))
        return [self.command[0], *self.command[module:], '--check'], {'WORLD_SIZE': ranks}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, help="Default: <data root>/no_phi")
    p.add_argument("--trial-spaces", "--trial-cohorts", dest="trial_spaces", type=Path,
                   help="Complete accepted cohort CSV/parquet; default: DATA_DIR/trial_cohort_lineitems.csv")
    p.add_argument("--patient-summaries", type=Path,
                   help="Summarized synthetic patients; default: DATA_DIR/patient_summaries_with_spaces.parquet")
    p.add_argument("--run-dir", type=Path, help="Default: <data root>/no_phi/embeddinggemma2_pipeline_v1")
    p.add_argument("--models-dir", type=Path, help="Default: <models root>/embeddinggemma2_pipeline_v1")
    p.add_argument("--teacher-model", default=DEFAULT_TEACHER)
    p.add_argument("--teacher-served-model", help="API model alias; --teacher-model remains the underlying HF model")
    p.add_argument("--teacher-url", action="append", default=[],
                   help="Existing OpenAI-compatible teacher URL; repeat for a pool. Skips local vLLM startup.")
    p.add_argument("--teacher-servers-file", type=Path,
                   help="Externally managed live endpoint registry. Never rewritten or shut down by this runner.")
    p.add_argument("--teacher-concurrency-file", type=Path,
                   help="Live per-server limits for TrialSpace labeling only; its short requests "
                        "need higher limits than generation or summary stages.")
    p.add_argument("--embedding-model", default=DEFAULT_EMBEDDER, help="Round-1 TrialSpace base model")
    p.add_argument('--embedding-revision', help='Base model revision; default: the pinned EmbeddingGemma 2 revision')
    p.add_argument('--embedding-python', type=Path,
                   help='Python for dense mining, projection and fits (needs EmbeddingGemma 2 support); default: this one')
    p.add_argument('--synthetic-training-splits', nargs='+', choices=['train', 'val', 'test'],
                   default=['train', 'val', 'test'], help='Synthetic source partitions used by every student fit')
    p.add_argument('--trialspace-context', type=int, default=8192)
    p.add_argument('--trialspace-steps', type=int, help='Explicit optimizer updates per fit; otherwise derive from complete epochs')
    p.add_argument('--trialspace-epochs', type=float, default=1, help='Complete passes over the labeled pairs per fit')
    p.add_argument('--trialspace-batch-size', type=int, default=96, help='EmbeddingGemma 2 pairs per GPU (one rank per --gpus entry); each rank ranks its own pairs, '
                        'and an optimizer step covers this times the GPU count')
    p.add_argument('--trialspace-loss-mini-batch-size', type=int, default=8,
                   help='EmbeddingGemma 2 gradient-cache encoder chunk in texts; the loss still spans each rank\'s full '
                        'batch. 32 and 16 ran out of memory on 96 GB GPUs at the 4,096-token cap and 8 fit (2026-11-07); '
                        'the GB300 takes 32. Every rank must run the same number of chunks, so chunks count texts, '
                        'not tokens (token-packed chunks deadlocked DDP)')
    p.add_argument('--trialspace-save-steps', type=int, default=200)
    p.add_argument('--trialspace-save-total-limit', type=int, default=2)
    p.add_argument('--trialspace-stop-at', help='Pause student training at an ISO timestamp with timezone offset')
    p.add_argument('--trialspace-pause-file', type=Path, help='Pause student training when this file exists')
    p.add_argument('--trialspace-top-k', type=int, default=20)
    p.add_argument('--trialspace-top-k-patient-to-cohort', type=int)
    p.add_argument('--trialspace-top-k-cohort-to-patient', type=int, default=40)
    p.add_argument('--trialspace-lexical-k', type=int, default=5)
    p.add_argument('--trialspace-random-k', type=int, default=5)
    p.add_argument('--trialspace-patient-trial-sample', type=int, default=500,
                   help="Trials sampled per patient for patient-to-cohort mining, like one center's open trials (0 = all); "
                        'redrawn for each mining round')
    p.add_argument('--trialspace-label-workers', type=int, default=0,
                   help='Concurrent pair chains; 0 scales automatically with the live teacher pool')
    p.add_argument('--trialspace-label-max-auto-workers', type=int, default=DEFAULT_AUTO_WORKERS_MAX)
    p.add_argument('--decision-model', default=DEFAULT_DECISION_MODEL, help='Base decision model (System One)')
    p.add_argument('--decision-revision', help='Base decision-model revision; default: the pinned d1-2B commit')
    p.add_argument('--decision-max-length', type=int, default=10000,
                   help='Decision-model training length: a state with its packed questions')
    p.add_argument('--decision-epochs', type=float, default=1)
    p.add_argument('--decision-batch-size', type=int, default=8, help='States per GPU per micro-batch')
    p.add_argument('--decision-gradient-accumulation-steps', type=int, default=4)
    p.add_argument('--decision-learning-rate', type=float, default=1e-5)
    p.add_argument('--criteria-pair-fraction', type=float, default=0.01,
                   help='Share of each round\'s candidate pairs whose every criteria line gets a teacher label')
    p.add_argument('--criteria-seed', type=int, default=42)
    p.add_argument('--trialspace-cache-dir', type=Path, help='Local Hugging Face model cache')
    p.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    p.add_argument('--mining-devices', default='cuda:0',
                   help='Comma-separated mining encoders and scorers, one worker each (e.g. cuda:0,cuda:1); '
                        'a mine-N directory made elsewhere (trialspace.mine) is adopted by its receipt')
    p.add_argument("--vllm", default=str(Path(sys.executable).parent / "vllm"))
    p.add_argument("--teacher-max-context", type=int, default=250000)
    p.add_argument('--teacher-max-num-seqs', type=int, choices=range(1, 1025),
                   metavar='1..1024', default=256,
                   help='Sequence capacity for locally launched vLLM servers; request concurrency adapts below it')
    p.add_argument("--teacher-port", type=int, default=8100)
    p.add_argument("--label-concurrency-per-server", type=int, choices=range(1, 1025),
                   metavar="1..1024", default=DEFAULT_CEILING,
                   help="Adaptive TrialSpace labeling ceiling per endpoint; live per-server limits can lower it.")
    p.add_argument('--label-initial-concurrency-per-server', type=int, choices=range(1, 1025),
                   metavar='1..1024', default=DEFAULT_INITIAL,
                   help='Starting target before automatic queue/cache/throughput adjustment')
    p.add_argument("--teacher-startup-timeout", type=int, default=1800)
    p.add_argument("--dry-run", action="store_true")
    return p


def resolve_defaults(args):
    """Fill --data-dir, --run-dir and --models-dir from the data root (matchminer_ai.site) when not given."""
    args.data_dir = args.data_dir or no_phi()
    args.run_dir = args.run_dir or no_phi('embeddinggemma2_pipeline_v1')
    args.models_dir = args.models_dir or models_root() / 'embeddinggemma2_pipeline_v1'
    return args


def input_paths(args):
    """Use the same explicit filenames in validation, manifests, and every stage."""
    defaults = dict(patient_summaries="patient_summaries_with_spaces.parquet", trial_spaces="trial_cohort_lineitems.csv")
    return {key: (getattr(args, key, None) or args.data_dir / filename).expanduser().resolve()
            for key, filename in defaults.items()}


def teacher_args(args, servers):
    """Teacher endpoint, model and concurrency options shared by every labeling stage."""
    teacher_model = args.teacher_served_model or args.teacher_model
    command = ['--server_urls_file', str(servers), '--teacher-model', teacher_model,
        '--teacher-tokenizer', args.teacher_model, '--teacher-context', str(args.teacher_max_context),
        '--max_concurrent_per_server', str(args.label_concurrency_per_server),
        '--initial-concurrent-per-server', str(min(args.label_initial_concurrency_per_server, args.label_concurrency_per_server)),
        '--pending-requests', str(args.trialspace_label_workers),
        '--max-auto-pending-requests', str(args.trialspace_label_max_auto_workers), '--max_attempts', '16',
        '--request_timeout', '7200', '--scheduler-adaptive-concurrency']
    if args.trialspace_cache_dir:
        command += ['--cache-dir', str(args.trialspace_cache_dir)]
    if args.teacher_concurrency_file:
        command += ['--concurrency-control-file', str(args.teacher_concurrency_file)]
    return command


def stages(args):
    run, models = args.run_dir, args.models_dir
    inputs = input_paths(args)
    py = sys.executable
    servers = getattr(args, 'teacher_servers_file', None) or run / "teacher_servers.json"
    gpu_count = len(args.gpus.split(','))
    teacher_model = args.teacher_served_model or args.teacher_model
    launch = [py, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={gpu_count}"]
    corpus = run / 'trialspace_corpus'
    result = [Stage('prepare-trialspace', [*module_command('trialspace.prepare', py),
        '--patients', str(inputs['patient_summaries']), '--cohorts', str(inputs['trial_spaces']),
        '--output', str(corpus),
        '--merge-source-splits'], [corpus / 'prepared.json'])]
    from matchminer_ai.training.synthetic.pool import require_all_sources
    require_all_sources(args.synthetic_training_splits)
    training_splits = args.synthetic_training_splits
    mining_splits = ['train', 'val', 'test'] if 'test' in training_splits else ['train', 'val']
    dense_py = str(args.embedding_python or py)
    dense_launch = [dense_py, *launch[1:]]  # One trialspace.finetune rank per GPU in --gpus.
    embedding = args.embedding_model
    revision = args.embedding_revision or (EMBEDDINGGEMMA2_REVISION if embedding == DEFAULT_EMBEDDER else None)
    # Three mining and labeling rounds; TrialSpace is refit after rounds 1 and 2, and every round's
    # labels of all seven question families train the decision model.
    teacher_dirs, boilerplate_dirs, criteria_dirs = [], [], []
    for iteration in (1, 2, 3):
        round_dir = run / f'trialspace_round{iteration}'
        mining = round_dir / 'mining'
        candidates = [mining / f'{direction}.parquet' for direction in ('patient_to_cohort', 'cohort_to_patient')]
        mine = [*module_command('trialspace.mine', dense_py), '--corpus', str(corpus), '--devices', args.mining_devices,
            '--output', str(mining), '--model', str(embedding), '--max-tokens', str(args.trialspace_context),
            '--patient-k', str(args.trialspace_top_k_patient_to_cohort or args.trialspace_top_k),
            '--cohort-k', str(args.trialspace_top_k_cohort_to_patient or args.trialspace_top_k),
            '--lexical-k', str(args.trialspace_lexical_k), '--random-k', str(args.trialspace_random_k),
            '--patient-trial-sample', str(args.trialspace_patient_trial_sample)]
        if iteration == 1 and revision:
            mine += ['--revision', revision]
        if args.trialspace_cache_dir:
            mine += ['--cache-dir', str(args.trialspace_cache_dir)]
        if iteration > 1:  # A fresh trial sample per patient each round; round 1 keeps the original draws.
            mine += ['--trial-sample-round', str(iteration)]
        result.append(Stage(f'mine-{iteration}', mine, [mining / 'complete.json', mining / 'candidates.jsonl', *candidates]))
        directory = round_dir / 'teacher'
        teacher_dirs.append(directory)
        scores = directory / 'scores.jsonl'
        label = [*module_command('trialspace.label', py), '--task', 'match', '--corpus', str(corpus),
            '--candidates', str(mining / 'candidates.jsonl'), '--output', str(directory), *teacher_args(args, servers)]
        if iteration > 1:  # Pairs judged in the previous round are imported, not judged again.
            previous = run / f'trialspace_round{iteration - 1}'
            label += ['--reuse-labels', str(previous / 'teacher'),
                      '--reuse-candidates', str(previous / 'mining' / 'candidates.jsonl')]
        result.append(Stage(f'label-trialspace-{iteration}', label,
                            [scores, directory / 'labels.jsonl', directory / 'labeling_manifest.json'], True))
        # The boilerplate label of each unique (patient boilerplate, cohort boilerplate) combination
        # among this round's candidates, and every criteria line of a seeded fraction of its pairs,
        # each skipping what earlier rounds labeled.
        for task, dirs, extra in (
                ('boilerplate', boilerplate_dirs, []),
                ('criteria', criteria_dirs, ['--criteria-pair-fraction', str(args.criteria_pair_fraction),
                                             '--criteria-seed', str(args.criteria_seed)])):
            out = round_dir / f'{task}_teacher'
            command = [*module_command('trialspace.label', py), '--task', task, '--corpus', str(corpus),
                       '--candidates', str(mining / 'candidates.jsonl'), '--output', str(out), *extra,
                       *(['--skip-labeled', *map(str, dirs)] if dirs else []), *teacher_args(args, servers)]
            result.append(Stage(f'label-{task}-{iteration}', command,
                                [out / 'labels.jsonl', out / 'labeling_manifest.json'], True))
            dirs.append(out)
        if iteration == 3:
            break
        previous_embedding = embedding
        embedding = models / f"trialspace_round{iteration}"
        pairs = round_dir / 'pairs'
        result.append(Stage(f'project-trialspace-{iteration}', [*module_command('trialspace.project', dense_py),
            '--corpus', str(corpus), '--mining', str(mining),
            '--labels', str(directory), '--output', str(pairs)],
            [pairs / 'embeddinggemma2' / 'complete.json', pairs / 'embeddinggemma2' / 'pairs.parquet']))
        command = dense_launch + [*module_command('trialspace.finetune')[1:], '-i', str(pairs / 'embeddinggemma2' / 'pairs.parquet'),
            '-c', str(round_dir / 'fit'), '-o', str(embedding), '--base-model', str(previous_embedding),
            '--prompt-profile', 'trialspace', '--model-dtype', 'bfloat16',
            '--batch-size', str(args.trialspace_batch_size),
            '--loss-mini-batch-size', str(args.trialspace_loss_mini_batch_size), '--mask-known-positives',
            '--logging-steps', '25', '--save-steps', str(args.trialspace_save_steps)]
        command += (['--max-steps', str(args.trialspace_steps)] if args.trialspace_steps is not None
                    else ['--epochs', str(args.trialspace_epochs)])
        if iteration == 1 and revision:
            command += ['--model-revision', revision]
        if args.trialspace_cache_dir:
            command += ['--model-cache-dir', str(args.trialspace_cache_dir)]
        result.append(Stage(f'train-trialspace-{iteration}', command,
                            [embedding / 'modules.json', embedding / 'model.safetensors']))
    # The decision model: every round's labels of the seven families, one example per state.
    decision_data, decision_model = run / 'decision_data', models / 'decision_model'
    decision_revision = args.decision_revision or (DECISION_MODEL_REVISION if args.decision_model == DEFAULT_DECISION_MODEL
                                                   else None)
    base = ['--base-model', args.decision_model, *(['--base-revision', decision_revision] if decision_revision else []),
            '--max-length', str(args.decision_max_length)]
    result.append(Stage('prepare-decision-data', [*module_command('decision_model', py), 'prepare', '--corpus', str(corpus),
        '--match-labels', *map(str, teacher_dirs), '--boilerplate-labels', *map(str, boilerplate_dirs),
        '--criteria-labels', *map(str, criteria_dirs), '--output-dir', str(decision_data), *base],
        [decision_data / 'prepared.json', decision_data / 'examples.jsonl']))
    result.append(Stage('train-decision-model', [*launch, *module_command('decision_model')[1:], 'train',
        '--data-dir', str(decision_data), '--output-model', str(decision_model), *base,
        '--epochs', str(args.decision_epochs), '--batch-size', str(args.decision_batch_size),
        '--gradient-accumulation-steps', str(args.decision_gradient_accumulation_steps),
        '--learning-rate', str(args.decision_learning_rate)],
        [decision_model / 'decision_contract.json', decision_model / 'model.safetensors']))
    return result


def teacher_command(args, index):
    nemotron = "nemotron" in args.teacher_model.lower()
    command = [args.vllm, "serve", args.teacher_model, "--host", "127.0.0.1",
        "--port", str(args.teacher_port + index), "--tensor-parallel-size", "1",
        *([] if nemotron else ["--quantization", "modelopt"]),
        "--reasoning-parser", resolve_parser_name(args.teacher_model),
        "--language-model-only", "--max-model-len", str(args.teacher_max_context),
        "--max-num-seqs", str(args.teacher_max_num_seqs), "--gpu-memory-utilization", "0.90",
        "--no-enable-log-requests", *(NEMOTRON_SERVE_FLAGS if nemotron else [])]
    tool_parser = resolve_tool_call_parser(args.teacher_model)
    if tool_parser:
        command += ["--enable-auto-tool-choice", "--tool-call-parser", tool_parser]
    return command


def teacher_environment(args):
    """Expose the pip CUDA toolkit when using a driver-only accelerator image."""
    env = dict(os.environ)
    executable = Path(shutil.which(args.vllm) or args.vllm).expanduser()
    env["PATH"] = str(executable.parent) + os.pathsep + env.get("PATH", "")
    if not env.get("CUDA_HOME") and not shutil.which("nvcc"):
        python = executable.parent / "python"
        probe = (
            "import sysconfig; from pathlib import Path; "
            "root=Path(sysconfig.get_paths()['purelib'])/'nvidia'; "
            "print(next((str(p) for p in sorted(root.glob('cu*'), reverse=True) "
            "if (p/'bin/nvcc').is_file()), ''))"
        )
        cuda_home = subprocess.check_output([str(python), "-c", probe], text=True).strip()
        if cuda_home:
            env["CUDA_HOME"] = cuda_home
            env["PATH"] = str(Path(cuda_home) / "bin") + os.pathsep + env.get("PATH", "")
    return env


@contextmanager
def teacher_pool(args):
    processes, handles, servers = [], [], []
    server_file = args.run_dir / "teacher_servers.json"
    atomic_json(server_file, {"ready": False, "servers": []})
    try:
        environment = teacher_environment(args)
        for index, gpu in enumerate(args.gpus.split(',')):
            port = args.teacher_port + index
            handle = (args.run_dir / "logs" / f"teacher-{index}.log").open("a")
            handles.append(handle)
            env = {**environment, "CUDA_VISIBLE_DEVICES": gpu}
            processes.append(subprocess.Popen(teacher_command(args, index), env=env,
                             stdout=handle, stderr=subprocess.STDOUT, start_new_session=True))
            servers.append({"url": f"http://127.0.0.1:{port}/v1", "gpus": [int(gpu)], "kind": "local"})
        deadline = time.monotonic() + args.teacher_startup_timeout
        ready = set()
        while len(ready) != len(processes):
            for index, process in enumerate(processes):
                if process.poll() is not None:
                    raise RuntimeError(f"Teacher {index} exited; see its log")
                if index in ready:
                    continue
                try:
                    with urllib.request.urlopen(servers[index]["url"] + "/models", timeout=3) as response:
                        models = json.load(response)["data"]
                    if args.teacher_model not in [item["id"] for item in models]:
                        raise RuntimeError("Teacher endpoint reports an unexpected model")
                    ready.add(index)
                except (OSError, TimeoutError):
                    pass
            if time.monotonic() > deadline:
                raise TimeoutError("Teacher pool startup timed out; see teacher logs")
            if len(ready) < len(processes):
                print(f"Waiting for teacher pool: {len(ready)}/{len(processes)} ready", flush=True)
                time.sleep(15)
        atomic_json(server_file, {"ready": True, "servers": servers})
        yield
    finally:
        atomic_json(server_file, {"ready": False, "servers": []})
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        deadline = time.monotonic() + 45
        for process in processes:
            try:
                process.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        for handle in handles:
            handle.close()


def server_urls(url=None, file=None):
    """Teacher endpoint URLs from one URL or a server registry file."""
    if file:
        data = json.loads(Path(file).read_text())
        entries = data.get("servers", data.get("server_urls", [])) if isinstance(data, dict) else data
        urls = [entry["url"] if isinstance(entry, dict) else entry for entry in entries]
    else:
        urls = [url] if url else []
    if not urls or any(not value.startswith(("http://", "https://")) for value in urls):
        raise ValueError("Supply an explicit teacher endpoint or ready server pool")
    return list(dict.fromkeys(value.rstrip('/') for value in urls))


@contextmanager
def managed_teacher_pool(args):
    """Read an external registry without taking ownership of it or its servers."""
    path = args.teacher_servers_file
    registry = json.loads(path.read_text())
    entries = registry.get('servers', [])
    if not registry.get('ready', True) or not entries:
        raise ValueError('The external teacher registry has no ready servers')
    for entry in entries:
        url = entry if isinstance(entry, str) else entry['url']
        url = server_urls(url=url)[0].rstrip('/')
        with urllib.request.urlopen(url + '/models', timeout=15) as response:
            models = json.load(response)['data']
        served = args.teacher_served_model or args.teacher_model
        matches = [model for model in models if model['id'] == served]
        if not matches:
            raise ValueError(f'Teacher {url} does not serve {served}')
        root = entry.get('model_root', args.teacher_model) if isinstance(entry, dict) else args.teacher_model
        if args.teacher_served_model and matches[0].get('root') != root:
            raise ValueError(f'Teacher {url} does not report the declared model root {root}')
        if int(matches[0].get('max_model_len') or args.teacher_max_context) < args.teacher_max_context:
            raise ValueError(f'Teacher {url} has insufficient context')
    yield


@contextmanager
def external_teacher_pool(args):
    """Publish validated external endpoints without owning their processes."""
    registry = args.run_dir / "teacher_servers.json"
    atomic_json(registry, {"ready": False, "servers": []})
    servers = []
    try:
        for url in dict.fromkeys(args.teacher_url):
            url = url.rstrip('/')
            if not url.endswith('/v1'):
                url += '/v1'
            url = server_urls(url=url)[0]
            with urllib.request.urlopen(url + '/models', timeout=15) as response:
                models = json.load(response)['data']
            served = args.teacher_served_model or args.teacher_model
            matches = [model for model in models if model['id'] == served]
            if not matches:
                raise ValueError(f"Teacher {url} does not serve {served}")
            if args.teacher_served_model and matches[0].get('root') != args.teacher_model:
                raise ValueError(f"Teacher alias {served} does not identify {args.teacher_model} as its root")
            context = matches[0].get('max_model_len')
            if context and int(context) < args.teacher_max_context:
                raise ValueError(f"Teacher {url} supports {context} tokens; lower --teacher-max-context")
            servers.append({"url": url, "kind": "remote",
                            # Other teacher stages use this as a fixed demand
                            # hint; only the adaptive labeler uses the ceiling.
                            "upstream_concurrency": min(args.label_initial_concurrency_per_server,
                                                        args.label_concurrency_per_server)})
        if not servers:
            raise ValueError("Supply at least one --teacher-url")
        atomic_json(registry, {"ready": True, "servers": servers})
        yield
    finally:
        atomic_json(registry, {"ready": False, "servers": []})


def in_no_phi(path):
    """Synthetic training data stays in the data root's no_phi tree (matchminer_ai.site)."""
    return Path(path).resolve().is_relative_to(no_phi().resolve())


def run_parser():
    """parser() plus the stage range; training.pipeline extends parser() and has its own --start-at/--stop-after."""
    p = parser()
    p.add_argument('--start-at', help='First stage to run; every earlier stage must already be complete')
    p.add_argument('--stop-after', help='Last stage to run (e.g. label-trialspace-1)')
    return p


def main():
    args = resolve_defaults(run_parser().parse_args())
    for key in ("data_dir", "run_dir", "models_dir"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    for key in ("trial_spaces", "patient_summaries", "teacher_servers_file", "teacher_concurrency_file",
                'trialspace_cache_dir'):
        if getattr(args, key) is not None:
            setattr(args, key, getattr(args, key).expanduser().resolve())
    if args.teacher_servers_file and args.teacher_url:
        raise ValueError('Use either --teacher-servers-file or --teacher-url')
    plan = stages(args)
    names = [stage.name for stage in plan]
    for option in ('start_at', 'stop_after'):
        if getattr(args, option) and getattr(args, option) not in names:
            raise ValueError(f'--{option.replace("_", "-")} must be one of: ' + ', '.join(names))
    first = names.index(args.start_at) if args.start_at else 0
    last = names.index(args.stop_after) if args.stop_after else len(names) - 1
    if args.dry_run:
        print(json.dumps([dict(vars(stage), check=stage.check()[0]) for stage in plan[first:last + 1]],
                         default=str, indent=2))
        return
    if not in_no_phi(args.data_dir) or not all(in_no_phi(path) for path in input_paths(args).values()):
        raise ValueError("The integrated runner requires inputs in an authorized data/no_phi directory")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "logs").mkdir(exist_ok=True)
    lock = (args.run_dir / "pipeline.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    atomic_json(args.run_dir / 'reasoning_archives.json', dict(
        policy='Preserve returned reasoning and complete responses; answer-only targets remain separate.',
        match_label_traces=[str(args.run_dir / f'trialspace_round{i}/teacher/responses') for i in (1, 2, 3)],
        boilerplate_label_traces=[str(args.run_dir / f'trialspace_round{i}/boilerplate_teacher/responses') for i in (1, 2, 3)],
        criteria_label_traces=[str(args.run_dir / f'trialspace_round{i}/criteria_teacher/responses') for i in (1, 2, 3)]))
    os.environ.update(child_env())
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    os.environ["PYTHONUNBUFFERED"] = "1"

    def check(stage):
        command, extra = stage.check()
        result = subprocess.run(command, cwd=args.run_dir, env={**os.environ, **extra}, capture_output=True, text=True)
        if result.returncode not in (0, 10):
            raise ValueError(f'{stage.name}: {(result.stdout + result.stderr).strip()[-2000:]}')
        return result.returncode == 0

    # Each stage script owns its completion: its output directory's stage receipt. A stage done elsewhere
    # (for example mine-N on a cloud VM, copied back) is complete here when its receipt matches.
    for stage in plan[:first]:
        if not check(stage):
            raise ValueError(f'--start-at {args.start_at}: the earlier stage {stage.name} is not complete')
    pool, stage = None, None
    try:
        for stage in plan[first:last + 1]:
            if check(stage):
                print(f"[skip] {stage.name}", flush=True)
                continue
            if pool and not stage.teacher:
                pool.__exit__(None, None, None)
                pool = None
            if stage.teacher and not pool:
                pool = (managed_teacher_pool(args) if args.teacher_servers_file else
                        external_teacher_pool(args) if args.teacher_url else teacher_pool(args))
                pool.__enter__()
            atomic_json(args.run_dir / "status.json", {"stage": stage.name, "status": "running"})
            print(f"[start] {stage.name}", flush=True)
            # Each command emits progress but no clinical rows to its own log.
            with (args.run_dir / "logs" / (stage.name + ".log")).open("a") as log:
                env = dict(os.environ)
                if "torch.distributed.run" in stage.command:
                    env["CUDA_VISIBLE_DEVICES"] = args.gpus
                    # Variable-length batches fragment the caching allocator: the decision model's last step once
                    # failed with 4.2 GiB reserved but unusable.
                    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
                subprocess.run(stage.command, cwd=args.run_dir, env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
            if not check(stage):
                raise RuntimeError(f"{stage.name} finished without a complete stage receipt")
            print(f"[done] {stage.name}", flush=True)
        atomic_json(args.run_dir / "status.json", {"status": "complete" if last == len(plan) - 1 else "stopped",
                                                   "last_stage": plan[last].name})
    except BaseException:
        atomic_json(args.run_dir / "status.json", {"stage": stage.name if stage else None, "status": "failed"})
        raise
    finally:
        if pool:
            pool.__exit__(None, None, None)


if __name__ == "__main__":
    main()
