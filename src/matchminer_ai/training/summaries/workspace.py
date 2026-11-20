"""Local patient-summary checkpoints, with bulk archival to the shared destination."""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
import threading
import time

from matchminer_ai.training.trialspace.retrieval_workspace import atomic_json, copy_verified, require_free_space, require_local_disk
from matchminer_ai.training.cohort_format import file_stamp


def add_arguments(parser):
    parser.add_argument('--local-summary-storage', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--summary-scratch-dir', type=Path, default=Path(tempfile.gettempdir())/'mmai-summaries')
    parser.add_argument('--summary-resume-dir', type=Path, help='Verified local checkpoint snapshot for recovery')
    parser.add_argument('--summary-start-gate', type=Path, help='Wait for this local readiness file before dispatch')


def extract_archive(archive, destination):
    with tarfile.open(archive, 'r:gz') as stream:
        members=stream.getmembers()
        for member in members:
            p=Path(member.name)
            if p.is_absolute() or '..' in p.parts or not (member.isfile() or member.isdir()):
                raise ValueError('Unsafe summary archive member: '+member.name)
        stream.extractall(destination, members=members, filter='data')


def link_checkpoint_tree(source, destination):
    for directory, dirs, names in os.walk(source, followlinks=False):
        for name in dirs+names:
            if (Path(directory)/name).is_symlink():raise ValueError('Summary snapshot contains a symlink')
        for name in names:
            if name.endswith('.tmp'):continue
            path=Path(directory)/name;target=destination/path.relative_to(source)
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():
                try:os.link(path,target)
                except OSError:copy_verified(path,target)


class SummaryWorkspace:
    def __init__(self,args):
        self.original=args
        self.public=Path(args.shard_dir).resolve()
        root=Path(getattr(args,'summary_scratch_dir',None) or Path(tempfile.gettempdir())/'mmai-summaries').resolve()
        self.work=root/hashlib.sha256(str(self.public).encode()).hexdigest()[:24]
        self.checkpoints=self.work/'checkpoints'
        self.outputs=self.work/'outputs'
        self.archive=self.public/'checkpoints.tar.gz'
        self.stop=threading.Event();self.thread=None;self.lock=None
        self.local=copy.copy(args)
        self.local.shard_dir=str(self.checkpoints)
        self.local.output_parquet=str(self.outputs/'patient_serial_summaries.parquet')
        self.local.patient_summaries_parquet=str(self.outputs/'patient_summaries.parquet')
        self.local.scheduler_metrics_file=str(self.checkpoints/'concurrency_status.json')
        self.local.summary_trace_archive=str(self.archive)

    def publish_status(self,phase,**values):
        atomic_json(self.public/'storage_status.json',dict(phase=phase,at=time.time(),
                    local_workspace=str(self.work),local_status=str(self.checkpoints/'status.json'),**values))
        print(f'[summary storage] {phase}: {self.work}',flush=True)

    def mirror(self):
        # A stalled shared mount can delay this daemon, never the event loop.
        while not self.stop.is_set():
            try:
                for name in ['status.json','concurrency_status.json']:
                    source=self.checkpoints/name
                    if source.exists():
                        value=json.loads(source.read_text());value['local_workspace']=str(self.work)
                        atomic_json(self.public/name,value)
            except Exception as error:
                print('[summary storage] status mirror delayed: '+repr(error),flush=True)
            self.stop.wait(20)

    def prepare(self):
        self.work.parent.mkdir(parents=True,exist_ok=True)
        require_local_disk(self.work.parent);require_free_space(self.work.parent,reserve_gb=16)
        self.lock=(self.work.parent/(self.work.name+'.lock')).open('a')
        fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.work.mkdir(exist_ok=True);self.checkpoints.mkdir(exist_ok=True);self.outputs.mkdir(exist_ok=True)
        self.public.mkdir(parents=True,exist_ok=True)
        options={k:str(v) if isinstance(v,Path) else v for k,v in vars(self.original).items()
                 if k not in {'summary_scratch_dir','summary_resume_dir','summary_start_gate'}}
        self.publish_status('verifying_input')
        source=Path(self.original.input_parquet).resolve();stat=source.stat()
        prior_path=self.work/'identity.json';prior=json.loads(prior_path.read_text()) if prior_path.exists() else None
        # Path, size and modification time identify the input without reading it.
        record={'path':str(source),'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}
        identity={'version':1,'input':record,'options':options}
        if prior and prior!=identity:raise ValueError('Summary input/settings changed; retained workspace: '+str(self.work))
        atomic_json(prior_path,identity)
        resume=getattr(self.original,'summary_resume_dir',None)
        if resume:
            require_local_disk(resume)
            link_checkpoint_tree(Path(resume),self.checkpoints)
        elif not (self.checkpoints/'prepared_chunks.parquet').exists():
            if self.archive.exists():
                receipt=json.loads((self.public/'storage_complete.json').read_text())
                if receipt['identity']!=identity or file_stamp(self.archive)!=receipt['archive']['stamp']:
                    raise ValueError('Summary archive identity/stamp changed')
                extract_archive(self.archive,self.checkpoints)
            else:
                # Legacy direct-storage runs can resume too. Current recovery
                # supplies a parallel, verified local snapshot via --summary-resume-dir.
                for directory,dirs,names in os.walk(self.public):
                    for name in names:
                        if name.endswith('.tmp') or name in ['storage_status.json','storage_complete.json','checkpoints.tar.gz']:continue
                        path=Path(directory)/name;target=self.checkpoints/path.relative_to(self.public)
                        if not target.exists():copy_verified(path,target)
        self.publish_status('summarizing')
        self.thread=threading.Thread(target=self.mirror,daemon=True);self.thread.start()
        return self.local

    def finish(self):
        import pyarrow.parquet as pq
        self.stop.set()
        if self.thread:self.thread.join()
        self.publish_status('archiving_checkpoints')
        archive=self.work/'checkpoints.tar.gz'
        with tarfile.open(archive.with_suffix('.tmp'),'w:gz',compresslevel=1) as stream:
            for directory,dirs,names in os.walk(self.checkpoints):
                for name in dirs+names:
                    if (Path(directory)/name).is_symlink():raise ValueError('Checkpoint archive contains a symlink')
                for name in sorted(names):
                    if name.endswith('.tmp'):continue
                    path=Path(directory)/name
                    stream.add(path,arcname=str(path.relative_to(self.checkpoints)),recursive=False)
        archive.with_suffix('.tmp').replace(archive)
        self.publish_status('publishing_archive',bytes=archive.stat().st_size)
        archive_record=copy_verified(archive,self.archive)
        products={}
        for key in ['output_parquet','patient_summaries_parquet']:
            local=Path(getattr(self.local,key));public=Path(getattr(self.original,key))
            rows=pq.ParquetFile(local).metadata.num_rows
            record=copy_verified(local,public)
            assert pq.ParquetFile(public).metadata.num_rows==rows
            products[key]={'path':str(public.resolve()),'rows':rows,**record}
        status_path=self.checkpoints/'status.json'
        if status_path.exists():atomic_json(self.public/'status.json',json.loads(status_path.read_text()))
        atomic_json(self.public/'storage_complete.json',dict(identity=json.loads((self.work/'identity.json').read_text()),
                    archive={'path':str(self.archive),**archive_record},products=products,verified_at=time.time()))
        self.publish_status('complete',products=products)
        shutil.rmtree(self.work)

    def close(self):
        self.stop.set()
        if self.lock:self.lock.close()


def run_summary(args,callback):
    if not getattr(args,'local_summary_storage',True):return callback(args)
    workspace=SummaryWorkspace(args)
    try:
        local=workspace.prepare()
        callback(local)
        workspace.finish()
    except BaseException as error:
        workspace.publish_status('failed',error=repr(error),scratch_retained=True)
        raise
    finally:workspace.close()
