"""Checkpointed serial patient summaries on the shared adaptive endpoint pool."""
import argparse
import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import shutil
import uuid

from matchminer_ai.llm.teacher_pool import (CompletionSampling, OutputValidationError, RequestSession,
                                            build_registry_from_args, make_completion_work_fn)
from matchminer_ai.patients.postprocess import normalize_patient_summary
from matchminer_ai.patients.summary_quality import check_patient_summary, summary_retry_codes


def rewind_invalid_summaries(rounds, completed_rounds, all_results, shard_dir):
    """Archive bad chunks and descendants; replay from the last valid serial state.

    Unaffected chunk checkpoints remain reusable. Round-level files are archived
    from the earliest invalid round, because a completed round otherwise bypasses
    request validation entirely on resume.
    """
    earliest = None
    bad_patients = {}
    for ri, rows in enumerate(rounds[:completed_rounds]):
        for pid, idx, *_ in rows:
            if idx in all_results and summary_retry_codes(all_results[idx][1]):
                bad_patients[pid] = min(ri, bad_patients.get(pid, ri))
                earliest = ri if earliest is None else min(earliest, ri)
    if earliest is None:
        return completed_rounds, all_results
    root = Path(shard_dir)
    archive = root / 'quality_rejected' / uuid.uuid4().hex
    save(archive/'repair.json', dict(earliest_round=earliest, patients=bad_patients,
         state='archiving', reason='Summary quality checks require regeneration from original notes'))
    # Preserve the invalid round as the recovery marker until every descendant
    # checkpoint is gone. An interrupted archive can then safely resume.
    for ri, rows in enumerate(rounds):
        for pid, idx, *_ in rows:
            if pid in bad_patients and ri >= bad_patients[pid]:
                source = root / 'chunks' / f'{idx // 256:05d}' / f'{idx:09d}.json'
                if source.exists():
                    destination = archive / source.relative_to(root)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(source, destination)
    for ri in reversed(range(earliest, len(rounds))):
        source = root / f'round_{ri:04d}.parquet'
        if source.exists():
            shutil.move(source, archive / source.name)
    keep = {idx for rows in rounds[:earliest] for _, idx, *_ in rows}
    save(archive/'repair.json', dict(earliest_round=earliest, patients=bad_patients, state='archived',
         reason='Summary quality checks require regeneration from original notes'))
    return earliest, {idx: value for idx, value in all_results.items() if idx in keep}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def add_arguments(parser):
    parser.add_argument('--scheduler-adaptive-concurrency', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--initial-concurrent-per-server', type=int, default=16)
    parser.add_argument('--scheduler-metrics-file')
    parser.add_argument('--pending-requests', type=int, default=8192)


def checkpoint_key(prompt, sampling, parser_name, template):
    value = dict(prompt=prompt, sampling=asdict(sampling), parser=parser_name, template=str(template))
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def trace_reference(path, root, archive=None):
    member = str(path.relative_to(root))
    if archive:
        return dict(trace_path=str(archive) + '#' + member, trace_archive=str(archive), trace_member=member)
    return dict(trace_path=str(path))


def attempt_index(root):
    result = {}
    for path in (root / 'attempts').glob('*/*.json'):
        first = path.name.split('_', 1)[0]
        if first.isdigit():
            result.setdefault(int(first), []).append(path)
    return result


def recover_attempt(paths, *, idx, prompt, sampling, identity, root, archive=None, allow_legacy=False):
    for path in sorted(paths, reverse=True):
        try:
            trace = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        saved_identity = trace.get('checkpoint_identity')
        if saved_identity != identity and (saved_identity is not None or not allow_legacy):
            continue
        if (trace.get('chunk_index') != idx or trace.get('prompt') != prompt
                or trace.get('sampling') != asdict(sampling)
                or summary_retry_codes(trace.get('answer', ''), trace.get('finish_reason'))):
            continue
        return dict(reasoning=trace['reasoning'], summary=normalize_patient_summary(trace['answer']),
                    **trace_reference(path, root, archive))
    return None


def traced_work(base, root, sampling, archive=None):
    async def work(client, payload):
        reasoning, answer, metadata = await base(client, payload)
        trace = dict(chunk_index=payload['chunk_index'], endpoint=str(client.base_url),
                     prompt=payload['prompt'], reasoning=reasoning, answer=answer,
                     sampling=asdict(sampling), generated_at=datetime.now(timezone.utc).isoformat(),
                     checkpoint_identity=payload.get('checkpoint_identity'),
                     request_max_tokens=payload.get('max_tokens'), **metadata)
        trace['summary_quality'] = check_patient_summary(answer)
        path = root / 'attempts' / f'{payload["chunk_index"] // 256:05d}' / f'{payload["chunk_index"]}_{uuid.uuid4()}.json'
        await asyncio.to_thread(save, path, trace)
        codes = summary_retry_codes(answer, metadata['finish_reason'])
        if codes:
            raise OutputValidationError('Patient summary requires regeneration: ' + ', '.join(codes))
        return dict(reasoning=reasoning, summary=normalize_patient_summary(answer), **trace_reference(path, root, archive))
    return work


async def run_rounds(*, rounds, patient_chunk_order, patient_summaries, all_results,
                     completed_rounds, args, prompt_pool, parser_name, tokenizer,
                     build_prompt, save_round):
    root = Path(args.shard_dir)
    pending = getattr(args, 'pending_requests', 8192)
    if pending < 1:
        raise ValueError('Summary pending requests must be positive')
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    needed = 2 * pending + 2048
    if hard != resource.RLIM_INFINITY and hard < needed:
        raise RuntimeError('Open-file hard limit is too small for the summary pool')
    if soft != resource.RLIM_INFINITY and soft < needed:
        resource.setrlimit(resource.RLIMIT_NOFILE, (needed, hard))
    args.scheduler_metrics_file = getattr(args, 'scheduler_metrics_file', None) or str(root / 'concurrency_status.json')
    if args.max_concurrent_per_server is None:
        args.max_concurrent_per_server = int(args.max_concurrent_requests)
    sampling = CompletionSampling(model=args.model, request_timeout=float(args.request_timeout),
        **{k: getattr(args, k) for k in ('temperature', 'top_k', 'top_p', 'presence_penalty', 'min_p', 'repetition_penalty')})
    archive = getattr(args, 'summary_trace_archive', None)
    work = traced_work(make_completion_work_fn(sampling, parser_name, tokenizer), root, sampling, archive)
    attempts = await asyncio.to_thread(attempt_index, root)
    contract = dict(parser=parser_name, template_sha256=hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest(),
                    max_model_len=args.max_model_len, max_tokens=args.max_tokens)
    proof_path = root / 'resume_contract.json'
    proof = await asyncio.to_thread(lambda: json.loads(proof_path.read_text()) if proof_path.exists() else {})
    allow_legacy = proof.get('attempt_identity') == contract
    reused_chunks = recovered_attempts = 0
    status_lock = asyncio.Lock()
    async def update_status(value):
        async with status_lock:
            await asyncio.to_thread(save, root / 'status.json', value)
    chunk_to_patient = {cidx: pid for pid, chunks in patient_chunk_order.items() for cidx in chunks}
    total_chunks = sum(map(len, rounds))
    for round_idx in range(completed_rounds, len(rounds)):
        items = rounds[round_idx]
        priors = {idx: patient_summaries.get(pid) or 'None - this is the first segment for this patient'
                  for pid, idx, first, last, text in items}
        if any(value.lstrip().startswith('ERROR:') for value in priors.values()):
            raise RuntimeError('A failed prior summary must be repaired before continuing')
        await update_status(dict(stage='preparing_prompts', round=round_idx + 1,
             rounds=len(rounds), completed_chunks=len(all_results), expected_chunks=total_chunks))
        inputs = [(idx, patient_summaries.get(pid), first, last, text, args.max_model_len)
                  for pid, idx, first, last, text in items]
        prompts = prompt_pool.map(build_prompt, inputs, chunksize=max(1, len(inputs) // (prompt_pool._processes * 4)))
        iterator, completed = iter(prompts), {}
        gate = getattr(args, 'summary_start_gate', None)
        if gate and not Path(gate).exists():
            print(f'Waiting for summary recovery readiness gate: {gate}', flush=True)
            while not Path(gate).exists():
                await asyncio.sleep(1)
        registry = build_registry_from_args(args)
        async with RequestSession(registry, work, max_attempts=int(args.max_attempts)) as session:
            async def worker():
                nonlocal reused_chunks, recovered_attempts
                while (item := next(iterator, None)) is not None:
                    idx, prompt, tokens = item
                    available = args.max_model_len - tokens - 256
                    if available <= 0:
                        raise ValueError('Full summary prompt exceeds context')
                    max_tokens = min(available, args.max_tokens) if args.max_tokens is not None else available
                    identity = checkpoint_key(prompt, sampling, parser_name, tokenizer.chat_template)
                    # Output budget and the prior summary are part of checkpoint identity.
                    identity = hashlib.sha256((identity + str(max_tokens) + priors[idx]).encode()).hexdigest()
                    path = root / 'chunks' / f'{idx // 256:05d}' / f'{idx:09d}.json'
                    if await asyncio.to_thread(path.exists):
                        result = await asyncio.to_thread(lambda: json.loads(path.read_text()))
                        if result['identity'] != identity:
                            raise ValueError('Saved summary prompt or sampling changed')
                        if summary_retry_codes(result['summary']):
                            rejected = root/'quality_rejected'/uuid.uuid4().hex/path.name
                            rejected.parent.mkdir(parents=True, exist_ok=True)
                            await asyncio.to_thread(path.replace, rejected)
                            result = None
                        else:
                            reused_chunks += 1
                    else:
                        result = None
                    if result is None:
                        result = await asyncio.to_thread(recover_attempt, attempts.get(idx, []), idx=idx,
                            prompt=prompt, sampling=sampling, identity=identity, root=root, archive=archive,
                            allow_legacy=allow_legacy)
                        if result is not None:
                            recovered_attempts += 1
                        else:
                            result = await session.submit(dict(chunk_index=idx, prompt=prompt, max_tokens=max_tokens,
                                checkpoint_identity=identity, include_completion_metadata=True, include_raw_text=True))
                        if not isinstance(result, dict):
                            raise RuntimeError(f'Patient summary failed for chunk {idx}: {result}')
                        result.update(identity=identity, prior_summary=priors[idx], patient_id=chunk_to_patient[idx])
                        await asyncio.to_thread(save, path, result)
                    completed[idx] = (result['reasoning'], result['summary'], priors[idx])
                    if len(completed) % 32 == 0 or len(completed) == len(items):
                        await update_status(dict(stage='summarizing', round=round_idx + 1,
                             rounds=len(rounds), completed_chunks=len(all_results) + len(completed), expected_chunks=total_chunks,
                             reused_chunks=reused_chunks, recovered_attempts=recovered_attempts))
            tasks = [asyncio.create_task(worker()) for _ in range(min(pending, len(items)))]
            try:
                await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
        if set(completed) != {item[1] for item in items}:
            raise RuntimeError('Incomplete summary round')
        save_round(args.shard_dir, round_idx, [(idx, *completed[idx]) for idx in sorted(completed)])
        all_results.update(completed)
        for idx, (_, summary, _) in completed.items():
            patient_summaries[chunk_to_patient[idx]] = summary
    await update_status(dict(stage='completed', completed_chunks=len(all_results), expected_chunks=total_chunks,
                             reused_chunks=reused_chunks, recovered_attempts=recovered_attempts))
