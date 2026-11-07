"""SSD-backed label generation with compact traces and verified NFS publication."""
import asyncio
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time

from .files import atomic_json, copy_verified, require_local_disk, require_free_space
from .files import file_stamp


def enabled(args):
    return getattr(args, 'local_label_storage', os.environ.get('MMAI_LOCAL_LABEL_STORAGE', '1') != '0')


class LabelWorkspace:
    def __init__(self, args):
        self.original = args
        self.public = Path(args.output).resolve()
        scratch = Path(getattr(args, 'label_scratch_dir', None) or os.environ.get('MMAI_LABEL_SCRATCH_DIR')
                       or Path(tempfile.gettempdir())/'mmai-labels').resolve()
        self.work = scratch/hashlib.sha256(str(self.public).encode()).hexdigest()[:24]
        if self.work.is_relative_to(self.public) or self.public.is_relative_to(self.work):
            raise ValueError('Label scratch and public output must be separate')
        self.state = self.work/'state'
        self.inputs = self.work/'inputs'
        self.local = copy.copy(args)
        self.local.output = self.state
        self.local.compact_response_store = True
        self.local.labels_export_path = self.public/'labels.jsonl'
        self.public_metrics = Path(getattr(args, 'scheduler_metrics_file', None) or self.public/'concurrency_status.json')
        self.local.scheduler_metrics_file = str(self.state/'concurrency_status.json')
        self.stop = threading.Event()
        self.thread = None
        self.lock = None
        self.phase = 'staging'

    def status(self, phase, **values):
        self.phase = phase
        atomic_json(self.public/'storage_status.json', dict(phase=phase, at=time.time(),
            workspace=str(self.work), local_status=str(self.state/'status.json'), **values))
        print(f'[label storage] {phase}: {self.work}', flush=True)

    def mirror(self):
        while not self.stop.is_set():
            try:
                for name, target in [('status.json', self.public/'status.json'),
                                     ('concurrency_status.json', self.public_metrics)]:
                    path = self.state/name
                    if path.exists():
                        value = json.loads(path.read_text())
                        value.update(local_workspace=str(self.work), storage_phase=self.phase)
                        atomic_json(target, value)
            except Exception as error:
                print('[label storage] status mirror delayed: '+repr(error), flush=True)
            self.stop.wait(10)

    def stage_file(self, source, destination, cache):
        source, destination = Path(source), Path(destination)
        st = source.stat()
        key = str(source.resolve())
        prior = cache.get(key)
        if (prior and prior['bytes'] == st.st_size and prior['mtime_ns'] == st.st_mtime_ns
                and destination.exists() and destination.stat().st_size == st.st_size):
            return prior
        record = dict(copy_verified(source, destination), mtime_ns=st.st_mtime_ns)
        cache[key] = record
        return record

    def stage_reuse(self, cache):
        plan_path = self.public/'reuse_sources.json'
        if not plan_path.exists():
            return
        plan = json.loads(plan_path.read_text())
        staged = []
        for index, entry in enumerate(plan['sources']):
            source = Path(entry['labels']).resolve()
            target = self.inputs/'reuse'/str(index)
            target.mkdir(parents=True, exist_ok=True)
            with (source/'labeling.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                receipt = source/'storage_complete.json'
                if not receipt.exists():
                    receipt = source/'checkpoint_snapshot.json'
                published = json.loads(receipt.read_text()) if receipt.exists() else None
                products = published['products'] if published and published.get('complete') else None
                # Stage just resume inputs; the large flattened JSONL export is
                # already durable and is unnecessary for candidate-subset reuse.
                for name in ['labeling_manifest.json', 'prompt_sources.json',
                             'score_journal', 'responses']:
                    path = source/name
                    if products is not None:
                        # Older immutable JSON checkpoints can remain at the
                        # destination after SQLite publication. The verified
                        # product manifest defines the active response set.
                        for relative, expected in products.items():
                            if relative == name or relative.startswith(name+'/'):
                                record = self.stage_file(source/relative, target/relative, cache)
                                if any(record[k] != expected[k] for k in ('bytes', 'stamp')):
                                    raise ValueError('Published reuse input changed: '+relative)
                    elif path.is_dir():
                        for item in path.rglob('*'):
                            if item.is_file() and not item.name.endswith(('.tmp','-wal','-shm')):
                                self.stage_file(item, target/item.relative_to(source), cache)
                    elif path.exists():
                        self.stage_file(path, target/name, cache)
                (target/'labeling.lock').touch()
            candidate = target/'source_candidates.jsonl'
            self.stage_file(entry['candidates'], candidate, cache)
            copied_entry = dict(entry, labels=str(target), candidates=str(candidate),
                public_labels=str(source), public_candidates=entry['candidates'])
            if entry.get('mode') in ('merge_source_splits', 'candidate_intersection'):
                for name in ['patients.jsonl','cohorts.jsonl']:
                    self.stage_file(Path(entry['corpus'])/name, target/'corpus'/name, cache)
                copied_entry.update(corpus=str(target/'corpus'), public_corpus=entry['corpus'])
            staged.append(copied_entry)
        atomic_json(self.state/'reuse_sources.json', dict(sources=staged))

    def prepare(self):
        self.work.parent.mkdir(parents=True, exist_ok=True)
        require_local_disk(self.work.parent)
        require_free_space(self.work.parent, reserve_gb=16)
        self.lock=(self.work.parent/(self.work.name+'.lock')).open('a')
        fcntl.flock(self.lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.state.mkdir(parents=True, exist_ok=True)
        self.inputs.mkdir(exist_ok=True)
        self.status('staging')
        cache_path = self.work/'input_cache.json'
        cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        identities = {}
        corpus = self.inputs/'corpus'
        for item in sorted(Path(self.original.corpus).iterdir()):
            if item.is_file():
                record=self.stage_file(item, corpus/item.name, cache)
                identities['corpus/'+item.name] = dict(bytes=record['bytes'], stamp=record['stamp'])
        candidate = self.inputs/'candidates.jsonl'
        record=self.stage_file(self.original.candidates, candidate, cache)
        identities['candidates'] = dict(bytes=record['bytes'], stamp=record['stamp'])
        identity=dict(version=1, public=str(self.public), inputs=identities)
        path=self.work/'identity.json'
        if path.exists() and json.loads(path.read_text()) != identity:
            raise ValueError('Label inputs changed; local work retained: '+str(self.work))
        atomic_json(path, identity)
        resume = getattr(self.original, 'label_resume_dir', None) or os.environ.get('MMAI_LABEL_RESUME_DIR')
        source = Path(resume) if resume else self.public
        if not (self.state/'labeling_manifest.json').exists():
            for directory, dirs, names in os.walk(source):
                for name in names:
                    if name in {'labeling.lock','storage_status.json','storage_complete.json','labels.jsonl'} or name.endswith(('.tmp','-wal','-shm')):
                        continue
                    path=Path(directory)/name;target=self.state/path.relative_to(source)
                    if not target.exists():copy_verified(path,target)
        self.local.corpus = corpus
        self.local.candidates = candidate
        self.stage_reuse(cache)
        atomic_json(cache_path, cache)
        self.status('labeling')
        self.thread=threading.Thread(target=self.mirror, daemon=True)
        self.thread.start()
        return self.local

    def finish(self):
        self.stop.set()
        if self.thread:self.thread.join()
        self.status('publishing')
        # The flattened export streams from local SQLite to the requested NFS
        # destination. It never consumes a second uncompressed SSD copy.
        export=self.public/'labels.jsonl'
        if not export.exists():raise ValueError('Missing complete reasoning export')
        products={'labels.jsonl':dict(bytes=export.stat().st_size, stamp=file_stamp(export))}
        paths=[p for p in self.state.rglob('*') if p.is_file() and
               p.name not in {'labeling.lock','reuse_sources.json'} and not p.name.endswith(('.tmp','-wal','-shm'))]
        late={'scores_ready.json','status.json'}
        for path in sorted(paths, key=lambda p:(p.name in late, p.name=='status.json', str(p))):
            relative=path.relative_to(self.state)
            products[str(relative)]=copy_verified(path,self.public/relative)
        atomic_json(self.public/'storage_complete.json',dict(version=1, verified_at=time.time(),
            identity=json.loads((self.work/'identity.json').read_text()), products=products,
            response_format='sqlite-zlib-v1', complete=True))
        self.status('complete', files=len(products))
        shutil.rmtree(self.work)

    def close(self):
        self.stop.set()
        if self.thread:self.thread.join()
        if self.lock:self.lock.close()


async def run_local_label(args, callback):
    if not enabled(args):return await callback(args, Path(args.output))
    workspace=LabelWorkspace(args)
    try:
        local=await asyncio.to_thread(workspace.prepare)
        gate=os.environ.get('MMAI_LABEL_START_GATE')
        while gate and not Path(gate).exists():await asyncio.sleep(1)
        result=await callback(local, workspace.state)
        await asyncio.to_thread(workspace.finish)
        return result
    except BaseException as error:
        await asyncio.to_thread(workspace.status, 'failed', error=repr(error), scratch_retained=True)
        raise
    finally:await asyncio.to_thread(workspace.close)
