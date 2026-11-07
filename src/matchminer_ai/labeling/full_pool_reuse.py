"""Verified candidate-intersection reuse and source-splits → full-pool migration."""
import fcntl
import json
from pathlib import Path

from .corpus import pooled_records, require_full_pool
from .tasks import judge_messages, prompt_sources
from .corpus import load_corpus
from .files import digest, read_jsonl, write_json
from .runner import validate_row
from .responses import ResponseStore
from .journal import ScoreJournal, pending_candidates


def import_full_pool_source(output, signature, patients, cohorts, pairs, entry, *, compact):
    intersection = entry.get('mode') == 'candidate_intersection'
    source = Path(entry['labels']).resolve()
    if source == Path(output).resolve():
        raise ValueError('Cannot reuse the destination as a source')
    require_full_pool(list(patients.values()), list(cohorts.values()))
    with (source/'labeling.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        original = json.loads((source/'labeling_manifest.json').read_text())
        old_patients, old_cohorts = load_corpus(entry['corpus'])
        if intersection:
            require_full_pool(old_patients, old_cohorts)
        for kind, old_rows, target in [('patients', old_patients, patients), ('cohorts', old_cohorts, cohorts)]:
            if intersection and original[kind] != signature[kind]:
                raise ValueError('Intersection reuse requires identical corpus and provenance: '+kind)
            if (digest(old_rows) != original[kind] or digest(pooled_records(old_rows)) != signature[kind]
                    or digest(list(target.values())) != signature[kind]):
                raise ValueError('Full-pool reuse requires identical IDs, clinical text and provenance: '+kind)
        ignored = {'patients', 'cohorts', 'pairs'}
        if ({k:v for k,v in original.items() if k not in ignored}
                != {k:v for k,v in signature.items() if k not in ignored}):
            raise ValueError('Full-pool reuse requires identical prompts, teacher, tokenizer and sampling')
        old_candidates = read_jsonl(entry['candidates'])
        if digest(old_candidates) != original['pairs']:
            raise ValueError('Original candidates changed')
        old_p = {r['id']:r for r in old_patients}
        old_c = {r['id']:r for r in old_cohorts}
        scores = ScoreJournal(source, original).load(old_p, old_c)
        pending_candidates(old_candidates, scores)
        wanted = {(r['patient_id'], r['cohort_id']) for r in pairs}
        journal = ScoreJournal(output, signature, allow_import=True)
        existing = journal.load(patients, cohorts)
        fresh = [dict(row, split='train') for key,row in scores.items()
                 if key not in existing and (row['patient_id'],row['cohort_id']) in wanted]
        required = {key for row in fresh for key in row['question_keys'].values()}
        src, dst = ResponseStore(source/'responses', read_only=True), ResponseStore(Path(output)/'responses', compact=compact)
        sources = prompt_sources()
        imported = 0
        try:
            keys = (src.database.keys() if src.database is not None else
                    {p.stem for p in (source/'responses').rglob('*.json')})
            batch = []
            for key in sorted(keys):
                _, row = src.read(key)
                pair = (row['patient_id'], row['cohort_id'])
                if pair not in wanted:
                    continue
                pid, cid = pair
                messages = judge_messages(row['task'], patients[pid]['text'], cohorts[cid]['text'], sources=sources)
                if (key != digest([pid,cid,row['task']]) or row.get('key') != key
                        or row.get('contract') != original['contract']
                        or row.get('prompt_sha256') != digest(messages)
                        or row.get('messages') != messages or row.get('finish_reason') != 'stop'
                        or row['split'] != old_p[pid]['split'] or row['split'] != old_c[cid]['split']):
                    raise ValueError('Reused question identity, clinical prompt, or completion changed: '+key)
                validate_row(row['task'], row)
                migrated = row if intersection else dict(row, source_split=row.get('source_split', row['split']), split='train')
                path, saved = dst.read(key)
                if saved is not None and saved != migrated:
                    raise ValueError('Conflicting full-pool response: '+key)
                if saved is None:
                    if dst.database is not None:
                        batch.append((key,json.dumps(migrated,ensure_ascii=False).encode()))
                    else:
                        dst.write(path,migrated)
                required.discard(key)
                imported += 1
                if len(batch) >= 256:
                    dst.database.put_many(batch);batch.clear()
            if batch:
                dst.database.put_many(batch)
            if required:
                raise ValueError('Completed source pairs are missing reasoning responses')
        finally:
            src.close();dst.close()
        if fresh:
            journal.append(fresh)
        write_json(Path(output)/'reuse_receipt.json', dict(mode='candidate_intersection' if intersection else 'merge_source_splits', target_manifest=signature,
            source_manifest=original, source=entry.get('public_labels',str(source)),
            source_corpus=entry.get('public_corpus',entry['corpus']),
            source_candidates=entry.get('public_candidates',entry['candidates']),
            completed_source_pairs=len(scores), imported_pairs=len(fresh), imported_questions=imported,
            outside_target_pairs=sum((r['patient_id'],r['cohort_id']) not in wanted for r in scores.values()),
            trace_metadata_changes=[] if intersection else ['split','source_split'], clinical_prompt_and_raw_reasoning_unchanged=True))
        return len(fresh)
