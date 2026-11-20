#!/usr/bin/env python3
"""Build OncoReasoning's training set: teacher requests, reasoning and answers rendered in the student's chat format.

User, 2026-11-16: bring back OncoReasoning as a text-generation distillation of every teacher task with its
reasoning (patient summarization, trial cohort ETL, labeling), starting from LiquidAI/LFM3-1.7B. Sources and
checks: ``sources.py``. Each example becomes ``prompt`` + ``completion`` in the student's own template:

    <|startoftext|><|im_start|>user\\n{request}<|im_end|>\\n<|im_start|>assistant\\n<think>   <- prompt (no loss)
    {teacher reasoning}</think>{teacher answer}<|im_end|>                                  <- completion (loss)

which is exactly what LFM3's chat template renders for an assistant turn with ``reasoning``; a system message is
rendered when the teacher had one. Every task runs in its own process (each imports its frozen inference snapshot)
and samples in a seeded random order until its example or token budget is met. Output:

    parts/<task>-<shard>/tokens.bin   int32 token ids, examples back to back
    parts/<task>-<shard>/index.parquet  one row per example: task, source, id, offset, n_tokens, prompt_tokens
    manifest.json   counts, tokens, supervised tokens and length percentiles per task; dropped and skipped counts
    preview.jsonl   the first two rendered examples per task (synthetic text), for review
    stage_receipt.json (trialspace/stage.py; kind oncoreasoning-data)

Synthetic (no_phi) data only.

    python -m matchminer_ai.training.oncoreasoning.build_data --output-dir DIR --examples summary=200 label_match=300 ... [--check]
"""
import argparse
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
DATASET = 'synthetic_dataset_n5000_screened_20261106'  # under <data root>/no_phi
BASE_MODEL, BASE_REVISION = 'LiquidAI/LFM3-1.7B', '1da06bec6a819da10b54c67e5c90972b46d81871'
FORMAT = 'oncoreasoning-lfm3-think-v1'


def budgets(pairs, cast=int):
    out = {}
    for pair in pairs or []:
        task, value = pair.split('=', 1)
        out[task] = cast(float(value))
    return out


def run_task(task, shard, shards, args, out_dir, queue):
    """One task's sampling, rendering and tokenization, in a fresh process whose matchminer_ai is the task's snapshot.

    The spawned process imported this package to unpickle the task; it is dropped from sys.modules so that the
    task's frozen snapshot (put first on sys.path) is the matchminer_ai that ``sources`` imports."""
    for name in [m for m in sys.modules if m == 'matchminer_ai' or m.startswith('matchminer_ai.')]:
        del sys.modules[name]
    sys.path.insert(0, str(HERE))
    import sources
    if sources.SNAPSHOTS[task]:
        sys.path.insert(0, str(Path(args.dataset) / sources.SNAPSHOTS[task]))
    import numpy as np
    import pandas as pd
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base_model, revision=args.base_revision, trust_remote_code=True)
    end = tok.convert_tokens_to_ids('<|im_end|>')
    stats = dict(task=task, shard=shard, examples=0, tokens=0, supervised=0, too_long=0, rendered=0)
    # the label samplers read the big shared files; everything else reads little (summaries read a local cache)
    weight = 4 if task in sources.LABEL_DIRS else 1
    pacer = sources.Pacer(args.mb_per_s * weight / args.weight_total)
    want_examples = args.examples.get(task)
    want_tokens = args.tokens.get(task)
    if want_examples is not None:
        want_examples = -(-want_examples // shards)
    if want_tokens is not None:
        want_tokens = -(-want_tokens // shards)
    excluded = set()
    for previous in args.exclude_from:
        for index in sorted(Path(previous).glob(f'parts/{task}-*/index.parquet')):
            excluded.update(pd.read_parquet(index, columns=['id']).id)
    part = out_dir / 'parts' / f'{task}-{shard:03d}'
    part.mkdir(parents=True, exist_ok=True)
    rows, preview, started = [], [], time.time()
    # summary shards split one seeded patient order; the other samplers draw independently per shard
    n_hint = None if task not in args.examples else int(args.examples[task] * 1.3) + 20
    # seek-sampled label shards draw independently; every other sampler's shards split one seeded order
    seed = f'{args.seed}:{shard}' if task in sources.LABEL_DIRS and not args.label_block_mb else str(args.seed)
    try:
        text_out = (part / 'examples.jsonl').open('w') if args.save_text else None
        with (part / 'tokens.bin').open('wb') as tokens_out:
            blocks = args.label_block_mb if task in sources.LABEL_DIRS else None
            for ex in sources.iterate(task, args.dataset, seed, pacer, stats, n=n_hint, shard=shard, shards=shards,
                                      block_mb=blocks, lines_per_cohort=args.deboilerplate_lines, skip=excluded,
                                      partitions=args.label_partitions, earliest_round=args.match_earliest_round):
                if ex['id'] in excluded:
                    stats['excluded'] = stats.get('excluded', 0) + 1
                    continue
                prompt = tok.apply_chat_template(ex['messages'], add_generation_prompt=True, tokenize=False,
                                                 enable_thinking=True)
                completion = ex['reasoning'] + '</think>' + ex['answer'] + '<|im_end|>'
                p = tok(prompt, add_special_tokens=False)['input_ids']
                c = tok(completion, add_special_tokens=False)['input_ids']
                stats['rendered'] += 1
                if c[-1] != end:
                    raise RuntimeError(f'{task}: completion does not end with <|im_end|>')
                n = len(p) + len(c)
                if n > args.max_length:
                    stats['too_long'] += 1
                    continue
                ids = np.asarray(p + c, dtype=np.int32)
                rows.append(dict(task=task, source=ex['source'], id=ex['id'], offset=stats['tokens'], n_tokens=n,
                                 prompt_tokens=len(p)))
                tokens_out.write(ids.tobytes())
                if text_out is not None:  # an evaluation set: the request and the teacher's completion as text
                    text_out.write(json.dumps(dict(task=task, source=ex['source'], id=ex['id'], prompt=prompt,
                                                   reasoning=ex['reasoning'], answer=ex['answer'],
                                                   completion_tokens=len(c))) + '\n')
                stats['examples'] += 1
                stats['tokens'] += n
                stats['supervised'] += len(c)
                if len(preview) < 2:
                    preview.append(dict(task=task, id=ex['id'], prompt=prompt, completion=completion))
                if stats['examples'] % 200 == 0:
                    queue.put(('progress', task, shard, dict(stats, seconds=round(time.time() - started))))
                if (want_examples is not None and stats['examples'] >= want_examples) or (
                        want_tokens is not None and stats['tokens'] >= want_tokens):
                    break
        if text_out is not None:
            text_out.close()
        pd.DataFrame(rows, columns=['task', 'source', 'id', 'offset', 'n_tokens', 'prompt_tokens']).to_parquet(
            part / 'index.parquet', index=False)
        stats['seconds'] = round(time.time() - started)
        (part / 'stats.json').write_text(json.dumps(stats, indent=2) + '\n')
        (part / 'preview.json').write_text(json.dumps(preview) + '\n')
        queue.put(('done', task, shard, stats))
    except BaseException as error:  # report, then fail the build
        queue.put(('error', task, shard, dict(stats, error=repr(error)[-2000:])))
        raise


def summarize(out_dir):
    import numpy as np
    import pandas as pd
    parts = sorted((out_dir / 'parts').glob('*/index.parquet'))
    index = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True) if parts else pd.DataFrame()
    tasks = {}
    for task, g in index.groupby('task'):
        n = g.n_tokens.to_numpy()
        tasks[task] = dict(examples=int(len(g)), tokens=int(n.sum()), supervised=int((g.n_tokens - g.prompt_tokens).sum()),
                           sources={k: int(v) for k, v in g.source.value_counts().items()},
                           length={f'p{q}': int(np.percentile(n, q)) for q in (50, 90, 99, 100)})
    stats = {}
    for p in sorted((out_dir / 'parts').glob('*/stats.json')):
        s = json.loads(p.read_text())
        stats[f"{s['task']}-{s['shard']:03d}"] = s
    return dict(tasks=tasks, totals=dict(examples=int(len(index)), tokens=int(index.n_tokens.sum()) if len(index) else 0,
                                         supervised=int((index.n_tokens - index.prompt_tokens).sum()) if len(index) else 0),
                parts=stats)


def identity(args):
    from matchminer_ai.training.trialspace.stage import receipt_id
    return dict(version=FORMAT, base_model=args.base_model, base_revision=args.base_revision, seed=args.seed,
                max_length=args.max_length, examples=args.examples, tokens=args.tokens, shards=args.shards,
                label_block_mb=args.label_block_mb, deboilerplate_lines=args.deboilerplate_lines, save_text=args.save_text,
                label_partitions=args.label_partitions, match_earliest_round=args.match_earliest_round,
                excluded=[receipt_id(d) for d in args.exclude_from])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--dataset', type=Path, help=f'Default: <data root>/no_phi/{DATASET}')
    p.add_argument('--base-model', default=BASE_MODEL)
    p.add_argument('--base-revision', default=BASE_REVISION)
    p.add_argument('--examples', nargs='*', default=[], help='task=N examples (per task, all shards together)')
    p.add_argument('--tokens', nargs='*', default=[], help='task=N tokens (per task, all shards together)')
    p.add_argument('--shards', nargs='*', default=[], help='task=K parallel samplers: label shards sample '
                   'independently; the other tasks split one seeded order')
    p.add_argument('--max-length', type=int, default=131072, help='Drop longer examples (counted in the manifest)')
    p.add_argument('--mb-per-s', type=float, default=60, help='All workers\' NFS reads together')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--exclude-from', nargs='*', default=[], type=Path,
                   help='Earlier data directories whose examples (by task and id) this build skips')
    p.add_argument('--deboilerplate-lines', type=int, default=4, help='Per-line examples per cohort, at most')
    p.add_argument('--save-text', action='store_true', help='Also write each example as text (examples.jsonl per '
                   'part), for an evaluation set sampled with --exclude-from the training sets')
    p.add_argument('--label-partitions', nargs='*', default=[], help='task=k: keep only label pairs in partitions '
                   '0..k-1 of 100 (sources.partition; block sampling), so a cap of a task is its first k partitions')
    p.add_argument('--match-earliest-round', action='store_true', help='Keep a match pair only from the earliest '
                   'round that labeled it (later rounds re-judged 1.34M of 9.82M pairs)')
    p.add_argument('--label-block-mb', type=float, help='Sample label files in sequential blocks of this size '
                   '(MiB), keeping every record in a block, instead of one random seek per record')
    from matchminer_ai.training.paths import no_phi
    from matchminer_ai.training.trialspace import stage
    stage.add_check_arg(p)
    args = p.parse_args()
    args.dataset = args.dataset or no_phi(DATASET)
    args.examples, args.tokens, args.shards = budgets(args.examples), budgets(args.tokens), budgets(args.shards)
    args.label_partitions = budgets(args.label_partitions)
    if args.label_partitions and not args.label_block_mb:
        raise SystemExit('--label-partitions needs --label-block-mb (every block is read and filtered)')
    sys.path.insert(0, str(HERE))
    import sources
    unknown = (set(args.examples) | set(args.tokens) | set(args.shards)) - set(sources.TASKS)
    if unknown:
        raise SystemExit(f'unknown tasks {sorted(unknown)}; tasks are {list(sources.TASKS)}')
    ident = identity(args)
    if args.check:
        stage.exit_check(args.output_dir, 'oncoreasoning-data', ident)
    if stage.require_new_or_same(args.output_dir, 'oncoreasoning-data', ident):
        print('[skip] oncoreasoning-data already complete', flush=True)
        return
    if (args.output_dir / 'parts').exists():
        raise SystemExit(f'{args.output_dir}/parts exists without a receipt; use a fresh --output-dir')
    tasks = [t for t in sources.TASKS if t in args.examples or t in args.tokens]
    jobs = [(t, k, args.shards.get(t, 1)) for t in tasks for k in range(args.shards.get(t, 1))]
    args.workers = len(jobs)
    args.weight_total = sum(4 if t in sources.LABEL_DIRS else 1 for t, _, _ in jobs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if 'summary' in tasks:  # the summary shards' local caches, read from the shared copies once at the full pace
        pacer = sources.Pacer(args.mb_per_s)
        for name, columns in (('all_synthetic_notes.parquet', sources.NOTE_COLUMNS),
                              ('patient_serial_summaries.parquet', sources.SUMMARY_COLUMNS)):
            print(json.dumps(dict(event='local cache', path=str(sources.local_cache(
                Path(args.dataset) / 'patient_summaries' / name, columns, pacer)))), flush=True)
    if args.match_earliest_round and 'label_match' in tasks:  # one read of the compact scores for every match shard
        import numpy as np
        pacer = sources.Pacer(args.mb_per_s)
        os.environ['ONCOREASONING_EARLIER_PAIRS'] = str(args.output_dir)
        for r in (2, 3):
            path = args.output_dir / f'match_pairs_before_round{r}.npy'
            if not path.exists():
                pairs = sources.earlier_round_pairs(Path(args.dataset), (1, 2, 3), r, pacer)
                np.save(path, np.fromiter(pairs, dtype=np.uint64, count=len(pairs)))
            print(json.dumps(dict(event='earlier-round match pairs', before_round=r, path=str(path))), flush=True)
    ctx = multiprocessing.get_context('spawn')
    queue = ctx.Queue()
    procs = [ctx.Process(target=run_task, args=(t, k, n, args, args.output_dir, queue), name=f'{t}-{k}') for t, k, n in jobs]
    for proc in procs:
        proc.start()
    pending, failed = {(t, k) for t, k, _ in jobs}, []
    log = (args.output_dir / 'build.log.jsonl').open('a')
    while pending:
        kind, task, shard, stats = queue.get()
        extra = {k: v for k, v in stats.items() if k not in ('task', 'shard')}
        log.write(json.dumps(dict(at=time.strftime('%Y-%m-%dT%H:%M:%S'), event=kind, task=task, shard=shard, **extra)) + '\n')
        log.flush()
        if kind in ('done', 'error'):
            pending.discard((task, shard))
            if kind == 'error':
                failed.append((task, shard, stats.get('error')))
            print(json.dumps(dict(event=kind, task=task, shard=shard, examples=stats.get('examples'),
                                  tokens=stats.get('tokens'), seconds=stats.get('seconds'),
                                  error=(stats.get('error') or '')[-300:] or None)), flush=True)
    for proc in procs:
        proc.join()
    if failed:
        raise SystemExit(f'{len(failed)} task(s) failed: {[(t, k) for t, k, _ in failed]}')
    manifest = dict(format=FORMAT, base_model=args.base_model, base_revision=args.base_revision, **summarize(args.output_dir))
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    with (args.output_dir / 'preview.jsonl').open('w') as f:
        for path in sorted((args.output_dir / 'parts').glob('*/preview.json')):
            for item in json.loads(path.read_text()):
                f.write(json.dumps(item) + '\n')
    files = sorted(str(p.relative_to(args.output_dir)) for p in args.output_dir.rglob('*')
                   if p.is_file() and p.name != stage.NAME)
    stage.write_receipt(args.output_dir, 'oncoreasoning-data', ident, files)
    print(json.dumps(dict(event='built', **manifest['totals'])), flush=True)


if __name__ == '__main__':
    main()
