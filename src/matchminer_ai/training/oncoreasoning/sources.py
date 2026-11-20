"""Teacher examples for OncoReasoning: every pipeline task's stored request, reasoning and answer (user, 2026-11-16).

Each adapter yields dicts ``{task, source, id, messages, reasoning, answer}`` in a seeded random order: ``messages``
is the request exactly as the teacher (Nemotron 3.5 Super) received it, rebuilt with the frozen inference snapshot
that sent it and checked against the hash the run stored; ``reasoning`` and ``answer`` are the teacher's accepted
response (finish_reason "stop", no validation error, first attempt, so the request carries no retry notice).

The adapters import ``matchminer_ai`` from the snapshot their task ran with, so each runs in its own process
(``build_data.py`` sets ``sys.path`` first). Reads of the shared NFS copies are paced. Synthetic data only.
"""
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import time

SNAPSHOTS = {  # task -> inference snapshot (relative to the dataset) that rendered its requests
    'summary': 'source/matchminer-ai-inference-kehl/src',
    'cohort_screen': 'as_generated/source_cohorts_fast/matchminer-ai-inference-kehl/src',
    'cohort_list': 'as_generated/source_cohorts_fast/matchminer-ai-inference-kehl/src',
    'cohort_criteria': 'as_generated/source_cohorts_fast/matchminer-ai-inference-kehl/src',
    'cohort_merge': 'as_generated/source_cohorts_fast/matchminer-ai-inference-kehl/src',
    'cohort_deboilerplate': 'source/matchminer-ai-inference-kehl/src',
    'label_match': None, 'label_boilerplate': None, 'label_criteria': None,
}
TASKS = tuple(SNAPSHOTS)
LABEL_DIRS = {'label_match': 'teacher', 'label_boilerplate': 'boilerplate_teacher', 'label_criteria': 'criteria_teacher'}
CRITERIA_CONTRACT = '7a7eb230af1f502010a5e1707d6bb46a547d1a4618d4859efc1983bb22590afc'
CTGOV_EXPORT = 'ctgov_trials.csv'  # the parsed ClinicalTrials.gov export, beside the dataset in no_phi/


class Pacer:
    """Keep this process's reads of the shared copies under ``mb_per_s``."""

    def __init__(self, mb_per_s):
        self.rate, self.start, self.bytes = mb_per_s * 1e6, time.monotonic(), 0

    def __call__(self, n):
        self.bytes += n
        ahead = self.bytes / self.rate - (time.monotonic() - self.start)
        if ahead > 0:
            time.sleep(ahead)


def sha256(text):
    return hashlib.sha256(text.encode()).hexdigest()


def accepted(attempts):
    """The accepted first attempt of one request, or None (a retry's request carries the failure notice)."""
    if not attempts:
        return None
    last = attempts[-1]
    if (last.get('attempt') != 1 or last.get('finish_reason') != 'stop' or 'validation_error' in last
            or not str(last.get('answer') or '').strip() or not str(last.get('reasoning') or '').strip()):
        return None
    return last


def example(task, source, ident, messages, reasoning, answer):
    return dict(task=task, source=source, id=ident, messages=messages, reasoning=str(reasoning).strip(),
                answer=str(answer).strip())


# ---------------------------------------------------------------------------------------------------- labels
def iter_labels(task, root, rng, pacer, stats, rounds=(1, 2, 3), block_mb=None, shard=0, shards=1, partitions=None,
                earliest_round=False):
    """Sample label records. Seek sampling (the default): a random byte offset, then the next whole line; a record's
    chance depends on the length of the record before it, not its own, so the sample is close to uniform without an
    index, but every record costs a random NFS read. Block sampling (``block_mb``): random, non-overlapping blocks
    read sequentially, every whole line inside kept (in shuffled order); records are labeled in completion order, so
    a block mixes patients, and NFS reads stay sequential."""
    if block_mb:
        yield from iter_label_blocks(task, root, rng, pacer, stats, rounds, block_mb, shard, shards, partitions,
                                     earliest_round)
        return
    paths = [root / 'trialspace' / f'trialspace_round{r}' / LABEL_DIRS[task] / 'labels.jsonl' for r in rounds]
    if task == 'label_criteria':
        for path in paths:
            manifest = json.loads((path.parent / 'labeling_manifest.json').read_text())
            if CRITERIA_CONTRACT not in json.dumps(manifest):
                raise RuntimeError(f'{path.parent} is not the relabeled criteria contract {CRITERIA_CONTRACT[:8]}')
    handles = [path.open('rb') for path in paths]
    sizes = [path.stat().st_size for path in paths]
    seen = set()
    try:
        while True:
            r = rng.randrange(len(paths))
            f = handles[r]
            f.seek(rng.randrange(sizes[r]))
            skipped = f.readline()
            start = f.tell()
            line = f.readline()
            pacer(len(skipped) + len(line))
            if not line or (r, start) in seen:
                continue
            seen.add((r, start))
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                stats['unparsed'] = stats.get('unparsed', 0) + 1
                continue
            if not valid_label(task, record):
                stats['rejected'] = stats.get('rejected', 0) + 1
                continue
            ident = f"r{rounds[r]}:{record['patient_id']}:{record['cohort_id']}" + (
                f":{record['criteria_digest'][:12]}:{record['number']}" if task == 'label_criteria' else '')
            yield example(task, f'round{rounds[r]}', ident, record['messages'], record['reasoning'], record['raw_answer'])
    finally:
        for f in handles:
            f.close()


PARTITION_SALT = 'oncoreasoning-20261116'


def partition(task, patient_id, cohort_id, parts=100):
    """A pair's fixed partition (0..parts-1) for a task: any cap of a task's pool is its first k partitions."""
    return int(hashlib.sha256(f'{PARTITION_SALT}:{task}:{patient_id}:{cohort_id}'.encode()).hexdigest()[:8], 16) % parts


def pair_hash(patient_id, cohort_id):
    return int(hashlib.sha256(f'{patient_id}:{cohort_id}'.encode()).hexdigest()[:16], 16)


def earlier_round_pairs(root, rounds, before, pacer):
    """Hashes of the match pairs labeled in rounds earlier than ``before`` (from the compact scores.jsonl): a pair
    re-judged in a later round keeps only its earliest round's record."""
    seen = set()
    for r in rounds:
        if r >= before:
            break
        with (root / 'trialspace' / f'trialspace_round{r}' / 'teacher' / 'scores.jsonl').open('rb') as f:
            for line in f:
                pacer(len(line))
                rec = json.loads(line)
                seen.add(pair_hash(rec['patient_id'], rec['cohort_id']))
    return seen


def iter_label_blocks(task, root, rng, pacer, stats, rounds, block_mb, shard=0, shards=1, partitions=None,
                      earliest_round=False):
    paths = [root / 'trialspace' / f'trialspace_round{r}' / LABEL_DIRS[task] / 'labels.jsonl' for r in rounds]
    if task == 'label_criteria':
        for path in paths:
            if CRITERIA_CONTRACT not in (path.parent / 'labeling_manifest.json').read_text():
                raise RuntimeError(f'{path.parent} is not the relabeled criteria contract {CRITERIA_CONTRACT[:8]}')
    size = int(block_mb * 2**20)
    jobs = [(r, start) for r, path in enumerate(paths) for start in range(0, path.stat().st_size, size)]
    rng.shuffle(jobs)
    jobs = jobs[shard::shards]  # the shards split one block order, so no block is read twice
    earlier = {}
    if earliest_round and task == 'label_match':  # built once by build_data.py's main process
        import numpy as np
        cache = Path(os.environ['ONCOREASONING_EARLIER_PAIRS'])
        earlier = {r: set(np.load(cache / f'match_pairs_before_round{r}.npy').tolist()) for r in rounds if r > rounds[0]}
    handles = [path.open('rb') for path in paths]
    try:
        for r, start in jobs:
            f = handles[r]
            f.seek(start)
            data = f.read(size)
            pacer(len(data))
            lines = data.split(b'\n')
            lines = lines[1:-1] if start else lines[:-1]  # drop the partial lines at the block's edges
            rng.shuffle(lines)
            for line in lines:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    stats['unparsed'] = stats.get('unparsed', 0) + 1
                    continue
                if partitions is not None and partition(task, record['patient_id'], record['cohort_id']) >= partitions:
                    continue
                if earlier.get(rounds[r]) and pair_hash(record['patient_id'], record['cohort_id']) in earlier[rounds[r]]:
                    stats['later_round_repeat'] = stats.get('later_round_repeat', 0) + 1
                    continue
                if not valid_label(task, record):
                    stats['rejected'] = stats.get('rejected', 0) + 1
                    continue
                ident = f"r{rounds[r]}:{record['patient_id']}:{record['cohort_id']}" + (
                    f":{record['criteria_digest'][:12]}:{record['number']}" if task == 'label_criteria' else '')
                yield example(task, f'round{rounds[r]}', ident, record['messages'], record['reasoning'],
                              record['raw_answer'])
    finally:
        for f in handles:
            f.close()


def valid_label(task, record):
    if record.get('finish_reason') != 'stop' or not str(record.get('reasoning') or '').strip():
        return False
    if not str(record.get('raw_answer') or '').strip() or not record.get('messages'):
        return False
    if task == 'label_match':
        labels = record.get('labels') or {}
        return len(labels) == 5 and all(str((v or {}).get('answer')).lower() in ('yes', 'no') for v in labels.values())
    if task == 'label_boilerplate':
        return isinstance(record.get('excluded'), bool)
    return record.get('contract') == CRITERIA_CONTRACT and str((record.get('criterion') or {}).get('answer')).lower() in (
        'yes', 'no')


# --------------------------------------------------------------------------------------------------- summary
NOTE_COLUMNS = ['pseudo_mrn', 'date', 'synthetic_note']


SUMMARY_COLUMNS = ['pseudo_mrn', 'chunk_index', 'first_date', 'last_date', 'prior_summary', 'reasoning_trace',
                   'new_summary']


def local_cache(source, columns, pacer):
    """A local copy of some columns of a shared parquet, in their original order, read from the shared copy once
    (paced) for every summary shard; keyed by the source's name, size and mtime. The notes parquet (14.5 GB, unsorted
    by patient) and the serial summaries both go through it."""
    import fcntl
    import pyarrow.parquet as pq
    st = source.stat()
    cache = Path(os.environ.get('ONCOREASONING_CACHE', '/tmp')) / f'oncoreasoning_{source.stem}_{st.st_size}_{int(st.st_mtime)}.parquet'
    legacy = cache.parent / f'oncoreasoning_notes_{st.st_size}_{int(st.st_mtime)}.parquet'  # the first build's name
    if source.stem == 'all_synthetic_notes' and legacy.exists():
        return legacy
    with open(str(cache) + '.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not cache.exists():
            parquet = pq.ParquetFile(source)
            positions = [parquet.schema_arrow.names.index(c) for c in columns]
            partial = Path(str(cache) + '.partial')
            writer = None
            for g in range(parquet.metadata.num_row_groups):
                pacer(sum(parquet.metadata.row_group(g).column(c).total_compressed_size for c in positions))
                table = parquet.read_row_group(g, columns=columns)
                if writer is None:
                    writer = pq.ParquetWriter(partial, table.schema, compression='zstd')
                writer.write_table(table)
            writer.close()
            partial.replace(cache)
    return cache


def iter_summary(task, root, rng, pacer, stats, n=None, shard=0, shards=1, skip=()):
    """Serial summaries: every n=5000 patient's notes fit one 50,000-token chunk, so each request is the first
    segment (no prior summary); notes are deduplicated and concatenated exactly as summarize_patients.py did."""
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from matchminer_ai.patients.prompt_builder import get_serial_patient_prompt

    sys_root = root / 'source' / 'matchminer-ai-training-kehl'
    import importlib.util
    import sys
    sys.path.insert(0, str(sys_root))  # its sibling modules (vllm_reasoning_utils) from the same frozen snapshot
    for name in [m for m in sys.modules if m == 'vllm_reasoning_utils']:
        del sys.modules[name]
    spec = importlib.util.spec_from_file_location('frozen_summarize_patients', sys_root / 'summarize_patients.py')
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)
    teacher_tokenizer = AutoTokenizer.from_pretrained(
        str(root.parent / 'synthetic_protocolcohorts_v11_cohortsv2_nemotron35_20261029' / 'tokenizer'
            / 'NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4'))
    summaries = pq.read_table(local_cache(root / 'patient_summaries' / 'patient_serial_summaries.parquet', SUMMARY_COLUMNS,
                                          pacer)).to_pandas()
    if (summaries.chunk_index != 0).any() or summaries.pseudo_mrn.duplicated().any():
        raise RuntimeError('expected one first-segment request per patient')
    order = list(range(len(summaries)))
    rng.shuffle(order)
    if skip:  # patients an earlier build used, dropped before their requests are built
        ids = summaries.pseudo_mrn.astype(str).to_numpy()
        order = [i for i in order if ids[i] not in skip]
    wanted = order[: n or len(order)][shard::shards]
    patients = set(summaries.pseudo_mrn.iloc[wanted].astype(str))
    import pandas as pd
    cache = local_cache(root / 'patient_summaries' / 'all_synthetic_notes.parquet', NOTE_COLUMNS, pacer)
    parts = []
    parquet = pq.ParquetFile(cache)
    for g in range(parquet.metadata.num_row_groups):
        frame = parquet.read_row_group(g).to_pandas()
        frame = frame[frame.pseudo_mrn.astype(str).isin(patients)]
        if len(frame):
            parts.append(frame)
    notes = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=NOTE_COLUMNS)
    notes['pseudo_mrn'] = notes.pseudo_mrn.astype(str)
    notes = notes.sort_values(['pseudo_mrn', 'date']).reset_index(drop=True)
    grouped = {pid: g for pid, g in notes.groupby('pseudo_mrn', sort=False)}
    for i in wanted:
        row = summaries.iloc[i]
        pid = str(row.pseudo_mrn)
        group = grouped.get(pid)
        if group is None:
            stats['no_notes'] = stats.get('no_notes', 0) + 1
            continue
        pairs = [(str(d) if pd.notna(d) else 'unknown date', str(t)) for d, t in zip(group.date, group.synthetic_note)]
        chunks = frozen.concatenate_and_chunk_notes(frozen.deduplicate_patient_notes(pairs), teacher_tokenizer,
                                                    50000, 500)
        if len(chunks) != 1 or (chunks[0][1], chunks[0][2]) != (str(row.first_date), str(row.last_date)):
            stats['chunk_mismatch'] = stats.get('chunk_mismatch', 0) + 1
            continue
        messages = get_serial_patient_prompt(
            prior_summary=None, first_date=chunks[0][1], last_date=chunks[0][2], chunk_text=chunks[0][0],
            tokenizer=teacher_tokenizer, max_model_len=250000, margin_tokens=5000,
            model_name='nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4',
            primer_filename='patient.serial.user.primer.txt', question_filename='patient.serial.user.question.txt')
        if str(row.prior_summary) not in messages[-1]['content']:
            stats['prior_mismatch'] = stats.get('prior_mismatch', 0) + 1
            continue
        if not str(row.reasoning_trace).strip() or not str(row.new_summary).strip():
            stats['empty'] = stats.get('empty', 0) + 1
            continue
        yield example(task, 'serial', pid, messages, row.reasoning_trace, row.new_summary)


# ------------------------------------------------------------------------------------------------ cohort ETL
def _workflow():
    from matchminer_ai.trials import cohort_workflow as w
    prompts = w.load_prompts()
    return w, prompts, w.prompt_sha256(prompts)


def _ctgov_seeds(root):
    import importlib.util
    path = root / 'as_generated' / 'source_cohorts_fast' / 'matchminer-ai-training-kehl' / '0b_extract_trial_cohorts.py'
    spec = importlib.util.spec_from_file_location('frozen_extract_trial_cohorts', path)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.path.insert(1, str(path.parent))
    spec.loader.exec_module(module)
    frame = module.load_sampled_trials(root / 'as_generated' / 'synthetic_protocols' / 'seeds.jsonl')
    return {r.nct_id: r for r in frame.itertuples(index=False)}


def iter_extraction(task, root, rng, pacer, stats, shard=0, shards=1):
    """Screen, cohort names and per-cohort criteria for both trial versions: synthetic protocols (extraction.json)
    and the 5,000 sampled ctgov listings (registry_extraction checkpoints). The ctgov screens of all 29,005 screened
    listings come from screen.jsonl instead (the 5,000 are among them)."""
    w, prompts, hashes = _workflow()
    step = task.split('_', 1)[1]
    protocol_dir = root / 'as_generated' / 'synthetic_protocols' / 'cohort_extraction' / 'trials'
    ctgov_dir = root / 'as_generated' / 'upstream' / 'registry_extraction' / 'checkpoints'
    jobs = [('protocol', t) for t in sorted(os.listdir(protocol_dir))]
    screens = []
    if step == 'screen':
        screens = load_ctgov_screens(root, rng, pacer, stats, w, hashes)
        jobs += [('ctgov_screen', i) for i in range(len(screens))]
    else:
        jobs += [('ctgov', p[:-len('.json.gz')]) for p in sorted(os.listdir(ctgov_dir)) if p.endswith('.json.gz')]
    rng.shuffle(jobs)
    jobs = jobs[shard::shards]
    seeds = None
    for version, trial in jobs:
        if version == 'ctgov_screen':
            entry, text = screens[trial]
            attempt = accepted(entry.get('attempts'))
            if attempt is None:
                stats['not_accepted'] = stats.get('not_accepted', 0) + 1
                continue
            yield example(task, 'ctgov', f"ctgov:{entry['trial_id']}", w.build_messages('screen', text, prompts),
                          attempt['reasoning'], attempt['answer'])
            continue
        if version == 'protocol':
            path = protocol_dir / trial / 'extraction.json'
            if not path.exists():
                continue
            raw = path.read_bytes()
            pacer(len(raw))
            record = json.loads(raw)
            if record.get('prompt_sha256_by_step', {}).get(step) != hashes[step]:
                stats['prompt_mismatch'] = stats.get('prompt_mismatch', 0) + 1
                continue
            text_path = Path(record['source_document']['text_path'])
            text = text_path.read_text()
            pacer(len(text))
            if sha256(text) != record['source_document']['sha256'] and sha256(text) != record.get('source_sha256'):
                stats['source_mismatch'] = stats.get('source_mismatch', 0) + 1
                continue
            attempts, names = record['attempts'], record.get('cohort_names') or []
        else:
            raw = (ctgov_dir / f'{trial}.json.gz').read_bytes()
            pacer(len(raw))
            metadata = json.loads(gzip.decompress(raw))['metadata']
            if metadata.get('prompt_sha256_by_step', {}).get(step) != hashes[step]:
                stats['prompt_mismatch'] = stats.get('prompt_mismatch', 0) + 1
                continue
            if seeds is None:
                seeds = _ctgov_seeds(root)
            seed = seeds[trial]
            text = w.registry_trial_text(seed.title, seed.brief_summary, seed.eligibility_criteria)
            attempts = metadata['trial_attempts'].get(trial) or {}
            names = metadata.get('trial_cohort_names', {}).get(trial) or []
        if step in ('screen', 'list'):
            entry = accepted(attempts.get(step))
            if entry is None:
                stats['not_accepted'] = stats.get('not_accepted', 0) + 1
                continue
            yield example(task, version, f'{version}:{trial}', w.build_messages(step, text, prompts),
                          entry['reasoning'], entry['answer'])
        else:
            for i in sorted((attempts.get('criteria') or {}), key=int):
                entry = accepted(attempts['criteria'][i])
                if entry is None or int(i) >= len(names):
                    stats['not_accepted'] = stats.get('not_accepted', 0) + 1
                    continue
                messages = w.build_messages('criteria', text, prompts, cohort_names=names, cohort_name=names[int(i)])
                yield example(task, version, f'{version}:{trial}:{i}', messages, entry['reasoning'], entry['answer'])


def load_ctgov_screens(root, rng, pacer, stats, w, hashes):
    """(entry, listing text) for every screen.jsonl decision, the text checked against the stored input hash."""
    import pandas as pd
    path = root / 'as_generated' / 'upstream' / 'screen' / 'screen.jsonl'
    raw = path.read_text()
    pacer(len(raw))
    entries = [e for e in map(json.loads, raw.splitlines())
               if e.get('prompt_sha256') == hashes['screen'] and e.get('decision') in ('yes', 'no')]
    wanted = {e['trial_id'] for e in entries}
    texts = {}
    for chunk in pd.read_csv(root.parent / CTGOV_EXPORT, keep_default_na=False, na_values=[''], chunksize=20000,
                             usecols=['nct_id', 'title', 'brief_summary', 'eligibility_criteria', 'trial_text']):
        pacer(int(chunk.memory_usage(deep=True).sum()))
        for r in chunk[chunk.nct_id.isin(wanted)].itertuples(index=False):
            texts[r.nct_id] = [r.trial_text if isinstance(r.trial_text, str) else '',
                               w.registry_trial_text(*(v if isinstance(v, str) else '' for v in
                                                       (r.title, r.brief_summary, r.eligibility_criteria)))]
    out = []
    for entry in entries:
        match = [t for t in texts.get(entry['trial_id'], []) if sha256(t) == entry['input_sha256']]
        if not match:
            stats['ctgov_screen_input_mismatch'] = stats.get('ctgov_screen_input_mismatch', 0) + 1
            continue
        out.append((entry, match[0]))
    return out


def iter_merge(task, root, rng, pacer, stats, shard=0, shards=1):
    """Merges of each trial's equivalent cohorts (trials with two or more cohorts), both versions: synthetic protocols
    (trials/*/merge.json, input checked by its stored hash) and ctgov listings (the registry checkpoint's merge
    attempts, on its own unmerged rows)."""
    import pandas as pd
    from matchminer_ai.llm.prompts import load_prompt_text
    from matchminer_ai.trials.cohort_merge import MERGE_PROMPT, build_merge_messages
    prompt = load_prompt_text(MERGE_PROMPT)
    prompt_sha = sha256(prompt)
    base = root / 'as_generated' / 'synthetic_protocols' / 'cohort_extraction'
    ctgov_dir = root / 'as_generated' / 'upstream' / 'registry_extraction' / 'checkpoints'
    unmerged = pd.read_parquet(base / 'cohorts_unmerged.parquet')
    pacer((base / 'cohorts_unmerged.parquet').stat().st_size)
    groups = {t: g.sort_values('cohort_number') for t, g in unmerged.groupby('trial_id', sort=False)}
    jobs = [('protocol', t) for t, g in groups.items() if len(g) >= 2]
    jobs += [('ctgov', p[:-len('.json.gz')]) for p in sorted(os.listdir(ctgov_dir)) if p.endswith('.json.gz')]
    rng.shuffle(jobs)
    jobs = jobs[shard::shards]
    for version, trial in jobs:
        if version == 'protocol':
            path = base / 'trials' / str(trial) / 'merge.json'
            if not path.exists():
                stats['missing'] = stats.get('missing', 0) + 1
                continue
            raw = path.read_bytes()
            pacer(len(raw))
            record = json.loads(raw)
            group = groups[trial]
            names, criteria = list(group.cohort_name), list(group.cohort_eligibility_criteria)
            content = json.dumps([[n, c] for n, c in zip(names, criteria)])
            if record.get('prompt_sha256') != prompt_sha or record.get('input_sha256') != sha256(content):
                stats['input_mismatch'] = stats.get('input_mismatch', 0) + 1
                continue
            attempts = record.get('attempts')
        else:
            raw = (ctgov_dir / f'{trial}.json.gz').read_bytes()
            pacer(len(raw))
            record = json.loads(gzip.decompress(raw))
            rows = sorted(record.get('unmerged_cohort_rows') or [], key=lambda r: r['cohort_number'])
            merge = record.get('merge') or {}
            if len(rows) < 2:
                continue
            if merge.get('prompt_sha256') != prompt_sha:
                stats['prompt_mismatch'] = stats.get('prompt_mismatch', 0) + 1
                continue
            names, criteria = [r['cohort_name'] for r in rows], [r['cohort_eligibility_criteria'] for r in rows]
            attempts = (merge.get('trial_attempts') or {}).get(trial)
        entry = accepted(attempts)
        if entry is None:
            stats['not_accepted'] = stats.get('not_accepted', 0) + 1
            continue
        yield example(task, version, f'{version}:{trial}', build_merge_messages(names, criteria, prompt),
                      entry['reasoning'], entry['answer'])


def iter_deboilerplate(task, root, rng, pacer, stats, lines_per_cohort=4, shard=0, shards=1):
    """The v4 per-line boilerplate + inclusion/exclusion classification of both versions' merged cohorts; up to
    ``lines_per_cohort`` random lines per cohort, so the sample spans many cohorts."""
    import pandas as pd
    from matchminer_ai.llm.prompts import load_prompt_text
    from matchminer_ai.trials import cohort_boilerplate as cb
    prompt = load_prompt_text('trial.cohorts.deboilerplate.user.txt')
    prompt_sha = sha256(prompt)
    v4 = root / 'trials_deboilerplate_v4'
    import sys
    sys.path.insert(1, str(root / 'source' / 'matchminer-ai-training-kehl'))
    from trial_cohort_format import read_clinical_csv  # the frozen training snapshot's (on sys.path above)
    versions = {
        'protocol': (pd.read_parquet(v4 / 'synthetic_protocols' / 'cohort_extraction' / 'cohorts.parquet'),
                     v4 / 'synthetic_protocols' / 'cohort_extraction' / 'trials'),
        'ctgov': (read_clinical_csv(v4 / 'ctgov_cohorts' / 'trial_cohort_lineitems.csv'),
                  v4 / 'upstream' / 'ctgov_deboilerplate'),
    }
    jobs = []
    for version, (frame, directory) in versions.items():
        for trial, group in frame.groupby('trial_id', sort=False):
            jobs.append((version, str(trial), group, directory))
    rng.shuffle(jobs)
    jobs = jobs[shard::shards]
    for version, trial, group, directory in jobs:
        path = directory / trial / 'deboilerplate.json'
        if not path.exists():
            stats['missing'] = stats.get('missing', 0) + 1
            continue
        raw = path.read_bytes()
        pacer(len(raw))
        record = json.loads(raw)
        if record.get('prompt_sha256') != prompt_sha:
            stats['prompt_mismatch'] = stats.get('prompt_mismatch', 0) + 1
            continue
        saved = {c['input_sha256']: c for c in record.get('cohorts', [])}
        for name, text in zip(group.cohort_name.astype(str), group.cohort_eligibility_criteria.fillna('').astype(str)):
            cohort = saved.get(hashlib.sha256(json.dumps([name, text]).encode()).hexdigest())
            if cohort is None:
                stats['input_mismatch'] = stats.get('input_mismatch', 0) + 1
                continue
            items = cb.split_criteria(text)
            done = {i['number']: i for i in cohort.get('items', [])}
            numbers = [n for n in range(1, len(items) + 1) if n in done]
            rng.shuffle(numbers)
            for number in numbers[:lines_per_cohort]:
                entry = accepted(done[number].get('attempts'))
                if entry is None:
                    stats['not_accepted'] = stats.get('not_accepted', 0) + 1
                    continue
                messages = cb.build_deboilerplate_messages(name, items, number, prompt)
                yield example(task, version, f'{version}:{trial}:{sha256(name)[:8]}:{number}', messages,
                              entry['reasoning'], entry['answer'])


ADAPTERS = {
    'summary': iter_summary, 'cohort_screen': iter_extraction, 'cohort_list': iter_extraction,
    'cohort_criteria': iter_extraction, 'cohort_merge': iter_merge, 'cohort_deboilerplate': iter_deboilerplate,
    'label_match': iter_labels, 'label_boilerplate': iter_labels, 'label_criteria': iter_labels,
}


def iterate(task, root, seed, pacer, stats, n=None, shard=0, shards=1, block_mb=None, lines_per_cohort=4, skip=(),
            partitions=None, earliest_round=False):
    rng = random.Random(f'{seed}:{task}')
    if task == 'summary':
        return iter_summary(task, Path(root), rng, pacer, stats, n=n, shard=shard, shards=shards, skip=skip)
    if task in LABEL_DIRS:
        return iter_labels(task, Path(root), rng, pacer, stats, block_mb=block_mb, shard=shard, shards=shards,
                           partitions=partitions.get(task) if partitions else None, earliest_round=earliest_round)
    kwargs = dict(lines_per_cohort=lines_per_cohort) if task == 'cohort_deboilerplate' else {}
    return ADAPTERS[task](task, Path(root), rng, pacer, stats, shard=shard, shards=shards, **kwargs)
