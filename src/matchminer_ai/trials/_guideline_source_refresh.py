"""Re-extract a source owner after irreconcilable singleton coverage failure.

No clinical fields are patched in Python. The endpoint rereads the whole owned
packet; new extraction, citation, ownership and coverage checks still run.
"""

from matchminer_ai._storage import atomic_json, digest, read_json
from ._guideline_completeness import population

VERSION = 'source-owner-review-v1'
MAX_REFRESHED_PACKETS = 2


def load_refreshes(output, guideline):
    path = output / 'extraction_source_reviews.json'
    ledger = read_json(path) if path.exists() else {
        'version': VERSION, 'source_fingerprint': guideline.fingerprint, 'packets': {},
    }
    if ledger.get('version') != VERSION or ledger.get('source_fingerprint') != guideline.fingerprint:
        raise ValueError('Source-review ledger differs from source')
    for row in ledger['packets'].values():
        if (row['original_result_sha256'] != digest(row['original_packet']['result'])
            or row['candidate'] not in row['original_packet']['result']['candidates']
            or row['coverage']['input_sha256'] != digest([population(row['candidate'])])
            or len(row['coverage']['reviews']) != 1
            or row['coverage']['reviews'][0]['status'] != 'missing'):
            raise ValueError('Source-review ledger has invalid original population provenance')
    return ledger


def prepare_refresh(output, guideline):
    """A transport, syntax, or multi-population failure cannot trigger this fallback."""
    paths = [output / n for n in ('status.json', 'canonical_batches.json', 'candidates.json', 'extraction.json')]
    if not all(p.exists() for p in paths) or read_json(paths[0]).get('status') != 'failed':
        return False
    ledger = load_refreshes(output, guideline)
    if len(ledger['packets']) >= MAX_REFRESHED_PACKETS:
        return False
    candidates = read_json(paths[2])
    packets = read_json(paths[3])['packets']
    for key, error in read_json(paths[1]).get('failures', {}).items():
        if 'exhausted' not in error or 'Catalog omitted or broadened 1 of 1' not in error:
            continue
        reports = sorted((output/'population_coverage').glob(key+'*.json'), key=lambda p:p.stat().st_mtime, reverse=True)
        for path in reports:
            report = read_json(path)
            if len(report.get('reviews', [])) != 1 or report['reviews'][0]['status'] != 'missing':
                continue
            matches = [c for c in candidates if digest([population(c)]) == report.get('input_sha256')]
            if len(matches) != 1:
                continue
            candidate = matches[0]
            owner = candidate['candidate_id'].rsplit('-c', 1)[0]
            if owner in ledger['packets'] or owner not in packets:
                continue
            original = {k:v for k,v in candidate.items() if k!='candidate_id'}
            packet = packets[owner]
            if original not in packet['result']['candidates']:
                raise ValueError('Source-review candidate differs from its original owner')
            ledger['packets'][owner] = {
                'candidate': original, 'coverage': report, 'trigger_error': error,
                'original_packet': packet, 'original_result_sha256': digest(packet['result']),
            }
            atomic_json(output/'extraction_source_reviews.json', ledger)
            return True
    return False


def review_job(key, record):
    return key + '-source-review-' + digest(record)[:16]


def archive_changed_candidates(output, candidates):
    """Preserve derived artifacts before rebuilding for changed source extraction."""
    old = output/'candidates.json'
    pending = output/'candidate_archive_pending.json'
    if pending.exists():
        transaction = read_json(pending)
        if transaction['new_sha256'] != digest(candidates):
            raise ValueError('Interrupted candidate archive differs from replacement extraction')
    else:
        if not old.exists() or digest(read_json(old)) == digest(candidates):
            return
        transaction = {'old_sha256': digest(read_json(old)), 'new_sha256': digest(candidates)}
        atomic_json(pending, transaction)
    if not isinstance(transaction['old_sha256'], str) or len(transaction['old_sha256']) != 64 or any(c not in '0123456789abcdef' for c in transaction['old_sha256']):
        raise ValueError('Invalid candidate archive identity')
    folder = output/'superseded_candidate_artifacts'/transaction['old_sha256']
    folder.mkdir(parents=True, exist_ok=True)
    names = ['candidates.json', 'canonical_batches.json', 'canonical_batch_splits.json',
             'canonical_coverage.json', 'canonical_selection.json', 'canonical_groups.json',
             'canonical_context.json', 'population_coverage', 'details.json',
             'paradigms.jsonl', 'trial_spaces.csv', 'report.md', 'validation.json']
    for name in names:
        source, target = output/name, folder/name
        if source.exists():
            if target.exists():
                raise ValueError('Refusing to overwrite a previous candidate archive')
            source.rename(target)
    pending.unlink()


def audit_refreshes(output, guideline, packets):
    ledger = load_refreshes(output, guideline)
    if not ledger['packets']:
        return
    accepted = [read_json(p) for p in (output/'checkpoints').glob('*/accepted.json')]
    for key, record in ledger['packets'].items():
        if key not in packets or packets[key]['primary_page_ids'] != record['original_packet']['primary_page_ids']:
            raise ValueError('Source review changed primary page ownership')
        expected = digest(packets[key]['result'])
        jobs = {review_job(key, record), review_job(key, record)+'-retry-headroom'}
        if not any(a['job'] in jobs and a['result_sha256'] == expected for a in accepted):
            raise ValueError('Source review has no accepted replacement extraction')
        if not any(a['job'] in {key, key+'-retry-headroom'} and a['result_sha256'] == record['original_result_sha256'] for a in accepted):
            raise ValueError('Source review has no accepted original extraction')
