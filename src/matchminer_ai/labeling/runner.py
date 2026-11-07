"""Resumable teacher labels using the existing adaptive vLLM pool.

Task ``match`` (the default): one request per pair answers the five match labels on the patient's
key section and the cohort's key criteria, and each pair's 0-6 score is committed to the score
journal. Task ``boilerplate``: one request per pair whose cohort has boilerplate criteria decides
whether the patient clearly fails them, from the patient's boilerplate section. Task ``criteria``:
one request per candidate line (training builds them: trialspace/candidates.py) answers whether
the patient clearly fails that criteria line, from the complete summary and the cohort's numbered
criteria (yes = does not clearly fail it). Every response is kept with its reasoning and complete
prompt.
"""
import asyncio
from contextlib import AsyncExitStack
from dataclasses import asdict, replace
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import time
import uuid

from matchminer_ai.llm.teacher_pool import (CompletionSampling, OutputValidationError, RequestSession,
                                            build_registry_from_args, make_completion_work_fn)
from matchminer_ai.llm.teacher_runtime import (load_tokenizer, resolve_chat_template_kwargs,
                                               resolve_parser_name, resolve_sampling_defaults)
from .tasks import TASKS, answer_schema, contract_hash, judge_messages, parse_judgment, prompt_sources
from .corpus import load_corpus
from .files import digest, read_jsonl, write_json, write_jsonl
from .concurrency import DEFAULT_AUTO_WORKERS_MAX, DEFAULT_CEILING, DEFAULT_INITIAL, run_workers, worker_target
from matchminer_ai.llm.async_processing import BackgroundWork, PromptPreparer
from .responses import ResponseStore


# Quantizations of one model approved to serve the same teacher (identical tokenizers and templates).
EQUIVALENT_TEACHER_ROOTS = (
    {'google/gemma-4-31B-it', 'nvidia/Gemma-4-31B-IT-NVFP4',
     'RedHatAI/Gemma-4-31B-IT-FP8-Dynamic', 'RedHatAI/gemma-4-31B-it-FP8-Dynamic'},
    {'nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4', 'RedHatAI/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-FP8'},
)


def validate_served_teacher(models, *, requested, tokenizer):
    """Check the advertised root, including the approved equivalent quantizations."""
    matches = [model for model in models if model['id'] == requested]
    if len(matches) != 1:
        raise ValueError('Requested teacher model is not uniquely advertised: ' + requested)
    served = matches[0]
    allowed = next((roots for roots in EQUIVALENT_TEACHER_ROOTS if tokenizer in roots), {tokenizer})
    if served.get('root', served['id']) not in allowed:
        raise ValueError('Served teacher root differs from the configured model family')
    return served


def label_key(row):
    if row['task'] == 'criteria':  # one line of one patient's criteria, whichever sibling cohort holds it
        return digest([row['patient_id'], row['criteria_digest'], row['number'], row['task']])
    return digest([row['patient_id'], row['cohort_id'], row['task']])


def questions(pairs, patients, cohorts, *, task='match', sources=None):
    """One item per pair: key texts for ``match``; boilerplate texts for ``boilerplate``, whose
    pairs without cohort boilerplate criteria have nothing to judge and are skipped."""
    patients = patients if isinstance(patients, dict) else {r['id']: r for r in patients}
    cohorts = cohorts if isinstance(cohorts, dict) else {r['id']: r for r in cohorts}
    if task == 'criteria':
        yield from _criteria_questions(pairs, patients, cohorts, sources)
        return
    field = 'text' if task == 'match' else 'boilerplate'
    seen = set()
    for pair in pairs:
        p, c = patients[pair['patient_id']], cohorts[pair['cohort_id']]
        if p['split'] != c['split'] or pair['split'] != p['split']:
            raise ValueError('Teacher pairs must stay within their corpus split')
        if task == 'boilerplate' and not (c.get(field) or '').strip():
            continue
        item = {**pair, 'task': task}
        key = label_key(item)
        if key in seen:
            continue
        seen.add(key)
        yield {**item, 'key': key, 'messages': judge_messages(task, p.get(field) or '', c[field], sources=sources)}


def _criteria_questions(pairs, patients, cohorts, sources):
    """One item per candidate line: the complete patient summary, the cohort's tagged items, the line."""
    from .corpus import full_summary
    seen = set()
    for pair in pairs:
        p, c = patients[pair['patient_id']], cohorts[pair['cohort_id']]
        if p['split'] != c['split'] or pair['split'] != p['split']:
            raise ValueError('Teacher pairs must stay within their corpus split')
        item = {**pair, 'task': 'criteria'}
        if c['items'][pair['number'] - 1]['criterion_type'] != pair['criterion_type']:
            raise ValueError('Candidate line type differs from the corpus')
        key = label_key(item)
        if key in seen:
            continue
        seen.add(key)
        yield {**item, 'key': key, 'messages': judge_messages('criteria', full_summary(p), c['items'], sources=sources,
                                                              number=pair['number'])}


def validate_row(task, row):
    """A saved response still satisfies its task's answer contract."""
    if task == 'match':
        parse_judgment('match', json.dumps(row['labels']))
    elif task == 'criteria':
        parse_judgment('criteria', json.dumps(row['criterion']), row['criterion_type'])
    elif type(row.get('excluded')) is not bool:
        raise ValueError('A boilerplate label needs a boolean excluded answer')


async def label(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'labeling.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        from .workspace import run_local_label
        return await run_local_label(args, _label)


async def _label(args, output):
    auto_maximum = getattr(args, 'max_auto_pending_requests', DEFAULT_AUTO_WORKERS_MAX)
    worker_target(args.pending_requests, 0, auto_maximum)  # Validate before loading the corpus.
    timing = dict(pid=os.getpid(), started_at=datetime.now(timezone.utc).isoformat(), first_request_at=None,
                  attempts_started=0)
    from .publication import input_fingerprints, publish_scores
    source_inputs = input_fingerprints(args.corpus, args.candidates)
    patients, cohorts = load_corpus(args.corpus)
    pairs = read_jsonl(args.candidates)
    sources = prompt_sources()
    task = getattr(args, 'task', 'match')
    if task not in TASKS:
        raise ValueError(f'task must be one of {TASKS}')
    tokenizer = load_tokenizer(args.teacher_tokenizer, cache_dir=args.cache_dir)
    defaults = resolve_sampling_defaults(args.teacher_tokenizer)
    sampling = CompletionSampling(model=args.teacher_model, request_timeout=args.request_timeout, **defaults)
    signature = dict(task=task, contract=contract_hash(task, sources), patients=digest(patients), cohorts=digest(cohorts),
                     pairs=digest(pairs), sampling=asdict(sampling), tokenizer=args.teacher_tokenizer,
                     template=digest(tokenizer.chat_template), max_tokens=args.max_tokens,
                     context_length=args.teacher_context)
    gated = task == 'match'  # match labels are scored and journaled per pair
    if gated:
        from .scoring import score_contract
        signature['aggregation_contract'] = score_contract(signature['contract'])
    manifest = output / 'labeling_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != signature:
        raise ValueError('Teacher identity changed; use a new label directory')
    write_json(manifest, signature)
    write_json(output / 'prompt_sources.json', sources)
    checkpoint_dir = output / 'responses'
    checkpoint_dir.mkdir(exist_ok=True)
    served_name = getattr(args, 'teacher_served_model', None) or args.teacher_model
    base = make_completion_work_fn(replace(sampling, model=served_name),
                                   resolve_parser_name(args.teacher_tokenizer, 'auto'), tokenizer)
    server_models = {}
    model_locks = {}

    async def work(client, payload):
        endpoint = str(client.base_url)
        if endpoint not in server_models:
            # Hundreds of first requests can reach a newly admitted endpoint at
            # once. Validate its catalog once, retrying if that lookup fails.
            async with model_locks.setdefault(endpoint, asyncio.Lock()):
                if endpoint not in server_models:
                    catalog = await client.models.list()
                    served = validate_served_teacher([m.model_dump() for m in catalog.data],
                                                     requested=served_name, tokenizer=args.teacher_tokenizer)
                    if served.get('max_model_len', args.teacher_context) < args.teacher_context:
                        raise ValueError('Teacher server context is smaller than the configured budget')
                    server_models[endpoint] = served
        if timing['first_request_at'] is None:
            timing['first_request_at'] = datetime.now(timezone.utc).isoformat()
        timing['attempts_started'] += 1
        reasoning, answer, metadata = await base(client, payload)
        trace = dict(endpoint=endpoint, served_model=server_models[endpoint], reasoning=reasoning,
                     raw_answer=answer, rendered_prompt=payload['prompt'], **metadata)
        try:
            if metadata['finish_reason'] != 'stop':
                raise ValueError('Incomplete teacher output: ' + metadata['finish_reason'])
            result = parse_judgment(task, answer, payload.get('criterion_type'))
        except (ValueError, TypeError, IndexError) as error:
            await storage.run(write_json, output / 'invalid_responses' / (str(uuid.uuid4()) + '.json'),
                              dict(key=payload['key'], error=str(error), **trace))
            raise OutputValidationError(str(error)) from error
        return dict(**result, **trace, generated_at=datetime.now(timezone.utc).isoformat())

    if getattr(args, 'scheduler_adaptive_concurrency', False) and not getattr(args, 'scheduler_metrics_file', None):
        args.scheduler_metrics_file = str(output / 'concurrency_status.json')
    registry = build_registry_from_args(args)
    pmap, cmap = {p['id']: p for p in patients}, {c['id']: c for c in cohorts}
    if (output / 'reuse_sources.json').exists():
        if not gated:
            raise ValueError('Candidate-subset reuse requires match labels')
        from .reuse import import_completed_sources
        write_json(output / 'status.json', dict(state='waiting_for_source_labels', **timing))
        imported = await asyncio.to_thread(import_completed_sources, output, signature, pmap, cmap, pairs,
            compact=getattr(args, 'compact_response_store', False))
        print(f'[labels] imported {imported} completed candidate-subset pairs with their reasoning traces', flush=True)
    restored_pairs = set()
    scheduled_pairs = pairs
    def scheduled():
        # Read lazily: a resumed journal narrows scheduled_pairs before the first item is drawn.
        yield from questions(scheduled_pairs, pmap, cmap, task=task, sources=sources)
    iterator = iter(scheduled())
    completed, failures, scores = {}, {}, {}
    restored_questions = 0
    from .scoring import MAX_SCORE
    score_counts = {str(i): 0 for i in range(MAX_SCORE + 1)}
    loop_lag = dict(latest_seconds=0.0, max_seconds=0.0)
    client_stats = {}
    def snapshot(state='labeling'):
        # Copy mutable counters on the event loop, then serialize/write outside
        # it. Never send patient text into operational telemetry.
        return dict(state=state, **timing,
            updated_at=datetime.now(timezone.utc).isoformat(), total_pairs=len(pairs),
            completed=restored_questions + len(completed), failures=dict(failures), client=dict(client_stats),
            event_loop_lag=dict(loop_lag),
            **(dict(scored_pairs=len(scores), score_distribution=dict(score_counts)) if gated else {}))

    async def monitor_loop():
        while True:
            started = time.monotonic()
            await asyncio.sleep(1)
            lag = max(0.0, time.monotonic() - started - 1)
            loop_lag['latest_seconds'] = lag
            loop_lag['max_seconds'] = max(loop_lag['max_seconds'], lag)

    async def report_progress():
        while True:
            client_stats.update(preprocessing=dict(preparer.stats),
                                preprocessing_queue=preparer.queue.qsize(),
                                request_queue=session.queue.qsize() if hasattr(session, 'queue') else None)
            await telemetry.run(write_json, output / 'status.json', snapshot())
            await asyncio.sleep(10)

    async with (AsyncExitStack() as stack, BackgroundWork() as storage,
                BackgroundWork(name='label-checkpoint-read') as checkpoint_reads,
                BackgroundWork(1, 1, 'label-progress') as telemetry,
                PromptPreparer(tokenizer, resolve_chat_template_kwargs(args.teacher_tokenizer),
                               context=args.teacher_context, max_tokens=args.max_tokens) as preparer,
                RequestSession(registry, work, max_attempts=args.max_attempts) as session):
        journal = None
        if gated:
            from .journal import ScoreJournal, pending_candidates
            journal = ScoreJournal(output, signature)
            write_json(output / 'status.json', snapshot('loading_completed_scores'))
            resume_started = time.monotonic()
            scores = await storage.run(journal.load, pmap, cmap)
            if scores:
                scheduled_pairs = await storage.run(pending_candidates, pairs, scores)
            restored_pairs.update(scores)
            for score in scores.values():
                score_counts[str(score['score'])] += 1
                restored_questions += len(score['question_keys'])
            timing.update(restored_pairs=len(restored_pairs), restored_questions=restored_questions,
                          resume_index_seconds=time.monotonic() - resume_started)
            await stack.enter_async_context(journal)
            print(f'[labels] restored {len(restored_pairs)} completed pairs from compact journal; '
                  'individual reasoning traces were not replayed', flush=True)
        def load_checkpoint(item):
            # Network-filesystem reads must not block every active HTTP request
            # or the scheduler while a production run resumes saved answers.
            path, row = response_store.read(item['key'])
            if row is not None:
                if row['key'] != item['key'] or row['prompt_sha256'] != digest(item['messages']):
                    raise ValueError('Checkpoint key or clinical text mismatch')
                validate_row(task, row)
                if 'messages' not in row:
                    row['messages'] = item['messages']
                    row['sampling'] = asdict(sampling)
                    response_store.write(path, row)
                return path, row
            return path, None

        response_store = await storage.run(ResponseStore, checkpoint_dir,
            compact=getattr(args, 'compact_response_store', False))
        stack.callback(response_store.close)

        async def ask(item):
            path, row = await checkpoint_reads.run(load_checkpoint, item)
            if row is not None:
                completed[item['key']] = path
                return row
            prepared = await preparer.prepare(item['messages'])
            prompt = prepared['prompt']
            result = await session.submit(dict(prompt=prompt, key=item['key'], max_tokens=args.max_tokens,
                enable_thinking=True, include_completion_metadata=True, include_raw_text=True,
                criterion_type=item.get('criterion_type'), json_schema=answer_schema(task)))
            if not isinstance(result, dict):
                failures[item['key']] = str(result)
                return None
            # Keep the exact question and complete clinical input alongside the
            # reasoning, rather than requiring corpus reconstruction to distill it.
            row = dict(item)
            row.update(result, prompt_sha256=prepared['prompt_sha256'], contract=signature['contract'],
                       teacher_model=args.teacher_model, rendered_prompt=prompt, sampling=asdict(sampling))
            # Await durable response storage before accepting the answer, while
            # other request chains continue. Reasoning archives remain complete.
            await storage.run(response_store.write, path, row)
            completed[item['key']] = path
            return row

        async def process(item):
            if not gated:
                await ask(item)
                return
            from .scoring import aggregate_pair
            row = await ask(item)
            if row is not None:
                score = aggregate_pair(item, row, question_contract=signature['contract'])
                # All question checkpoints (including reasoning) already exist.
                # Commit the small score before counting this pair as complete.
                await journal.commit(score)
                scores[digest([score['patient_id'], score['cohort_id']])] = score
                score_counts[str(score['score'])] += 1

        def resized(target):
            timing.update(pair_worker_limit=target,
                          pair_worker_mode='fixed' if args.pending_requests else 'automatic')

        reporter = asyncio.create_task(report_progress())
        loop_monitor = asyncio.create_task(monitor_loop())
        try:
            await run_workers(iterator, process, lambda: registry.request_capacity(),
                              workers=args.pending_requests, maximum=auto_maximum, on_resize=resized)
        finally:
            reporter.cancel()
            loop_monitor.cancel()
            await asyncio.gather(reporter, loop_monitor, return_exceptions=True)
            await telemetry.run(write_json, output / 'status.json', snapshot('incomplete'))
    def export_keys():
        for pair_key in sorted(restored_pairs):
            for key in scores[pair_key]['question_keys'].values():
                yield key
        for key in sorted(completed):
            yield key
    # This is a final flattened distillation export, after LLM work finishes.
    # It is not used to reconstruct progress or dispatch requests on resume.
    from .export import export_labels
    timing['inference_complete'] = not failures
    timing['inference_finished_at'] = datetime.now(timezone.utc).isoformat()
    if gated and not failures:
        if source_inputs != input_fingerprints(args.corpus, args.candidates):
            raise ValueError('Corpus or candidates changed during labeling')
        write_json(output / 'status.json', snapshot('publishing_scores'))
        publish_scores(output, signature, scores, pairs, inputs=source_inputs)
        timing['training_scores_ready'] = True
    def export_progress(values):
        write_json(output / 'status.json', dict(snapshot('exporting_labels'), **values))
    export_path = Path(getattr(args, 'labels_export_path', output / 'labels.jsonl'))
    export_labels(checkpoint_dir, export_path, export_keys(),
                  total=restored_questions + len(completed), progress=export_progress)
    if gated and failures:
        from .scoring import score_balance
        write_jsonl(output / 'scores.jsonl', [scores[k] for k in sorted(scores)])
        write_json(output / 'score_balance.json', score_balance(list(scores.values())))
    write_json(output / 'status.json', snapshot('incomplete' if failures else 'complete'))
    if failures:
        raise RuntimeError(f'{len(failures)} unresolved teacher labels; resume this command to retry')
    return (read_jsonl(export_path)
            if getattr(args, 'return_records', True) else None)


def add_label_cli_args(parser):
    """The labeler's command-line flags, with the teacher pool's (as training's label stage passes them)."""
    from matchminer_ai.llm.teacher_pool import add_remote_cli_args
    add_remote_cli_args(parser)
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--candidates', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--task', choices=list(TASKS), default='match',
                        help='match: the five match labels in one request per pair, scored 0-6 into scores.jsonl; '
                             'boilerplate: whether the patient clearly fails the cohort\'s boilerplate criteria; '
                             'criteria: one request per criteria line of sampled pairs (yes = does not clearly fail it)')
    parser.add_argument('--teacher-model', required=True, help='Served model ID')
    parser.add_argument('--teacher-served-model', help='Endpoint model name when different from the recorded teacher identity; advertised root is verified')
    parser.add_argument('--teacher-tokenizer', required=True, help='Underlying model for tokenizer and vendor sampling')
    parser.add_argument('--cache-dir')
    parser.add_argument('--max-tokens', type=int, default=32768)
    parser.add_argument('--teacher-context', type=int, default=250000)
    parser.add_argument('--pending-requests', type=int, default=0,
                        help='Pairs in flight; 0 automatically follows live endpoint capacity')
    parser.add_argument('--max-auto-pending-requests', type=int, default=DEFAULT_AUTO_WORKERS_MAX,
                        help='Upper bound on automatic pair workers; explicit --pending-requests overrides it')
    parser.set_defaults(max_concurrent_per_server=DEFAULT_CEILING, initial_concurrent_per_server=DEFAULT_INITIAL,
                        scheduler_adaptive_concurrency=True, request_timeout=7200, max_attempts=4)
    return parser


def run_label(args):
    """Label ``args.candidates`` (flags of :func:`add_label_cli_args`) to completion or a resumable failure."""
    if not args.scheduler_metrics_file:
        args.scheduler_metrics_file = str(Path(args.output) / 'concurrency_status.json')
    return asyncio.run(label(args))
