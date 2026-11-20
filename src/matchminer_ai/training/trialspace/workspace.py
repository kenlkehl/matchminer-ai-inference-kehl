"""SSD training inputs/checkpoints with resumable, verified final publication."""
import argparse
import copy
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

from matchminer_ai.training.trialspace.retrieval_workspace import atomic_json, copy_verified, files, require_free_space, require_local_disk
from matchminer_ai.training.cohort_format import file_stamp


def add_training_storage_args(parser):
    parser.add_argument('--local-training-storage', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--training-scratch-dir', type=Path,
                        default=Path(tempfile.gettempdir())/'mmai-training')
    parser.add_argument('--training-min-free-gb', type=float, default=32)


def tree_manifest(root):
    return {str(p.relative_to(root)): dict(bytes=p.stat().st_size, stamp=file_stamp(p))
            for p in files(root)}


def verify_tree(root, manifest):
    for name, item in manifest.items():
        path = Path(root)/name
        if not path.resolve().is_relative_to(Path(root).resolve()) or file_stamp(path) != item['stamp']:
            raise ValueError('Training artifact changed: '+str(path))


def publish_tree(source, target, manifest=None):
    """Publish completion markers last; verify destination sizes before rename.

    copy_verified keeps source mtimes, so an already-published file has its
    source's stamp. Size alone could keep an older same-size model file.
    """
    source, target = Path(source), Path(target)
    manifest = tree_manifest(source) if manifest is None else manifest
    verify_tree(source, manifest)
    for name in sorted(manifest, key=lambda n: (Path(n).name == 'complete.json', n.count('/') == 0, n)):
        expected = manifest[name]
        destination = target/name
        if destination.exists() and file_stamp(destination) == expected['stamp']:
            continue
        copy_verified(source/name, destination, expected['stamp'])
    return manifest


class TrainingWorkspace:
    def __init__(self, args):
        self.original = args
        self.model = Path(args.output_model).resolve()
        self.checkpoints = Path(args.ckpt_dir).resolve()
        scratch = Path(args.training_scratch_dir).resolve()
        key = hashlib.sha256(json.dumps([str(self.model), str(self.checkpoints)]).encode()).hexdigest()[:24]
        self.work = scratch/key
        for public in [self.model, self.checkpoints]:
            if public.is_relative_to(self.work) or self.work.is_relative_to(public):
                raise ValueError('Training scratch and public destinations must be separate')
        self.lock = self.public_lock = None
        self.owns_lock = False

    def status(self, phase, **extra):
        value = dict(phase=phase, at=time.time(), workspace=str(self.work),
                     local_checkpoints=str(self.work/'checkpoints'), local_model=str(self.work/'model'), **extra)
        atomic_json(self.checkpoints/'training_storage.json', value)
        print('[training storage] '+phase+': '+str(self.work), flush=True)

    def prepare(self):
        from matchminer_ai.training.trialspace.finetune import expand_inputs
        self.work.parent.mkdir(parents=True, exist_ok=True)
        require_local_disk(self.work.parent)
        require_free_space(self.work.parent, reserve_gb=self.original.training_min_free_gb)
        self.lock = (self.work.parent/(self.work.name+'.lock')).open('a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.checkpoints.mkdir(parents=True, exist_ok=True)
        self.public_lock = (self.checkpoints/'training_storage.lock').open('a')
        fcntl.flock(self.public_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.owns_lock = True
        inputs = [Path(p).resolve() for p in expand_inputs(self.original.input_parquet)]
        # Encoder chunk sizes only trade memory for speed, so a resumed run may lower them.
        settings = {k: str(v) if isinstance(v, Path) else v for k,v in vars(self.original).items()
                    if k not in {'training_scratch_dir', 'training_min_free_gb', 'local_training_storage',
                                 'loss_mini_batch_size', 'encoder_mini_batch_size', 'encoder_token_budget'}}
        model_path = Path(self.original.base_model)
        self.signature = dict(settings=settings, inputs={str(p): file_stamp(p) for p in inputs},
                              local_base_model=tree_manifest(model_path) if model_path.is_dir() else None)
        identity = self.work/'workspace.json'
        if identity.exists() and json.loads(identity.read_text()) != self.signature:
            raise ValueError('Training inputs/settings changed; retained SSD workspace: '+str(self.work))
        self.work.mkdir(exist_ok=True)
        atomic_json(identity, self.signature)
        self.status('staging')
        local = self.local_args()
        for source, target in zip(inputs, local.input_parquet):
            target, expected = Path(target), self.signature['inputs'][str(source)]
            # The signature pins source stamps; a staged copy is checked by size.
            if not target.exists() or target.stat().st_size != source.stat().st_size:
                require_free_space(self.work, source.stat().st_size, self.original.training_min_free_gb)
                copy_verified(source, target, expected)
        if model_path.is_dir():
            publish_tree(model_path, Path(local.base_model), self.signature['local_base_model'])
        # A surviving SSD workspace owns its checkpoints. Restore shared copies
        # only when local state was lost; a new/incomplete checkpoint is retained.
        if not Path(local.ckpt_dir).exists():
            for source in files(self.checkpoints):
                if source.name.startswith('training_storage'):
                    continue
                copy_verified(source, Path(local.ckpt_dir)/source.relative_to(self.checkpoints))
        return local

    def local_args(self):
        """The staged SSD paths (the same for every rank of one training launch)."""
        from matchminer_ai.training.trialspace.finetune import expand_inputs
        local = copy.copy(self.original)
        local.input_parquet = [str(self.work/'inputs'/f'{index:04d}.parquet')
                               for index, _ in enumerate(expand_inputs(self.original.input_parquet))]
        local.ckpt_dir, local.output_model = str(self.work/'checkpoints'), str(self.work/'model')
        local.model_cache_dir = self.original.model_cache_dir or str(self.work/'model_cache')
        if Path(self.original.base_model).is_dir():
            local.base_model = str(self.work/'base_model')
        return local

    def publish(self):
        receipt = json.loads((self.work/'trained_complete.json').read_text())
        self.status('publishing')
        products = {}
        for key, destination in [('checkpoints', self.checkpoints), ('model', self.model)]:
            products[key] = publish_tree(self.work/key, destination, receipt[key])
        atomic_json(self.checkpoints/'training_storage_complete.json', dict(
            complete=True, verified_at=time.time(), workspace=str(self.work),
            signature=self.signature, products=products, local_checkpoint_retained=True))
        self.status('complete')

    def close(self):
        if self.public_lock: self.public_lock.close()
        if self.lock: self.lock.close()


def distributed_state():
    """The process group of a torchrun/accelerate launch with several ranks, else None."""
    if int(os.environ.get('WORLD_SIZE', '1')) == 1:
        return None
    from datetime import timedelta
    from accelerate import PartialState
    # Also selects this rank's GPU. The other ranks wait while rank zero stages inputs.
    state = PartialState(timeout=timedelta(hours=1))
    if state.num_processes != int(os.environ.get('LOCAL_WORLD_SIZE', state.num_processes)):
        raise ValueError('Local training storage supports a single node')
    return state


def run_training(args, callback):
    state = distributed_state()
    if not args.local_training_storage:
        return callback(args)
    workspace = TrainingWorkspace(args)
    if state is not None and not state.is_main_process:
        # Rank zero owns the SSD workspace, its locks and publication; train from its staged copy.
        state.wait_for_everyone()
        if not (workspace.work/'trained_complete.json').exists():
            callback(workspace.local_args())
        return
    try:
        local = workspace.prepare()
        if state is not None:
            state.wait_for_everyone()
        complete = workspace.work/'trained_complete.json'
        if not complete.exists():
            workspace.status('training')
            callback(local)
            atomic_json(complete, dict(model=tree_manifest(workspace.work/'model'),
                                      checkpoints=tree_manifest(workspace.work/'checkpoints')))
        gc.collect()
        import torch
        if torch.cuda.is_initialized(): torch.cuda.empty_cache()
        workspace.publish()
    except BaseException as error:
        if workspace.owns_lock and workspace.work.exists():
            workspace.status('failed', error=repr(error), scratch_retained=True)
        raise
    finally:
        workspace.close()
