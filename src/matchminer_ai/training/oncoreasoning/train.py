#!/usr/bin/env python3
"""Fine-tune OncoReasoning: a full fine-tune of LiquidAI/LFM3-1.7B on build_data.py's teacher examples (user, 2026-11-16).

- Full fine-tune with transformers' Trainer (no LoRA/peft/TRL), fp32 master weights with bf16 autocast, gradient
  checkpointing, fused AdamW, cosine schedule.
- Packing: examples are packed into bins of exactly ``--bin-tokens`` tokens (first-fit over length-sorted windows of a
  seeded shuffle; the tail of a bin is padding). ``position_ids`` restart at every example, which LFM3 turns into
  per-document attention (its packed-sequence masks) and per-document short-conv windows (``seq_idx``), so examples
  never see each other. Every bin has the same length, so flex attention compiles once.
- Loss: next-token cross-entropy on the completion tokens only (the teacher's reasoning, ``</think>``, its answer and
  ``<|im_end|>``), never on the request. It is summed over the global batch's completion tokens and divided by their
  count (Trainer's ``num_items_in_batch``). The vocabulary projection runs in checkpointed chunks over the completion
  positions only, so no full-length logits are kept.
- Logs, besides Trainer's: completion-token loss per task, real (non-padding) tokens per second across ranks, and peak
  CUDA memory.

Stage receipt kind ``oncoreasoning-model`` (trialspace/stage.py). Synthetic data only.

    torchrun --nproc_per_node 8 -m matchminer_ai.training.oncoreasoning.train --data-dir DATA --output-model OUT [--max-steps N]
"""
import argparse
import json
import os
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
BASE_MODEL, BASE_REVISION = 'LiquidAI/LFM3-1.7B', '1da06bec6a819da10b54c67e5c90972b46d81871'
IGNORE = -100


def load_examples(data_dirs):
    """Index rows of every part of every data directory (task, offset, n_tokens, prompt_tokens) and a memory map
    of each part's tokens."""
    import numpy as np
    import pandas as pd
    parts, frames = [], []
    dirs = [data_dirs] if isinstance(data_dirs, (str, Path)) else data_dirs
    indexes = []
    for d in map(str, dirs):  # DIR or DIR@task,task: only those tasks' parts of DIR
        root, _, only = d.partition('@')
        keep = set(only.split(',')) if only else None
        indexes += [i for i in sorted((Path(root) / 'parts').glob('*/index.parquet'))
                    if keep is None or i.parent.name.rsplit('-', 1)[0] in keep]
    for k, index in enumerate(indexes):
        frame = pd.read_parquet(index)
        frame['part'] = k
        frames.append(frame)
        parts.append(np.memmap(index.parent / 'tokens.bin', dtype=np.int32, mode='r'))
    frame = pd.concat(frames, ignore_index=True)
    duplicated = frame.duplicated(['task', 'id'])
    if duplicated.any():  # independent samplers (seek sampling, separate builds) can draw one record twice
        print(json.dumps(dict(event='duplicates dropped', examples=int(duplicated.sum()))), flush=True)
        frame = frame[~duplicated].reset_index(drop=True)
    return frame, parts


def pack(lengths, capacity, seed, window=4096):
    """Bins of example positions, each at most ``capacity`` tokens; the same on every rank for one seed."""
    import numpy as np
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(lengths))
    bins, dropped = [], 0
    for start in range(0, len(order), window):
        chunk = sorted(order[start:start + window], key=lambda i: -lengths[i])
        open_bins = []  # [free, members]
        for i in chunk:
            n = int(lengths[i])
            if n > capacity:
                dropped += 1
                continue
            for b in open_bins:
                if b[0] >= n:
                    b[0] -= n
                    b[1].append(int(i))
                    break
            else:
                open_bins.append([capacity - n, [int(i)]])
        bins.extend(b[1] for b in open_bins)
    rng.shuffle(bins)
    return bins, dropped


class PackedBins:
    """One fixed-length packed row per item: input_ids, position_ids, labels/shift_labels, target task ids."""

    def __init__(self, frame, parts, bins, capacity, pad_id, tasks):
        self.frame, self.parts, self.bins, self.capacity, self.pad_id = frame, parts, bins, capacity, pad_id
        self.task_code = {t: i for i, t in enumerate(tasks)}
        self.cols = (frame.part.to_numpy(), frame.offset.to_numpy(), frame.n_tokens.to_numpy(),
                     frame.prompt_tokens.to_numpy(), frame.task.map(self.task_code).to_numpy())

    def __len__(self):
        return len(self.bins)

    def __getitem__(self, i):
        import numpy as np
        import torch
        part, offset, n_tokens, prompt_tokens, task = self.cols
        T = self.capacity
        ids = np.full(T, self.pad_id, dtype=np.int64)
        pos = np.zeros(T, dtype=np.int64)
        shift = np.full(T, IGNORE, dtype=np.int64)
        target_task = np.full(T, -1, dtype=np.int64)
        at = 0
        for e in self.bins[i]:
            n, p = int(n_tokens[e]), int(prompt_tokens[e])
            ids[at:at + n] = self.parts[part[e]][offset[e]:offset[e] + n]
            pos[at:at + n] = np.arange(n)
            # position t predicts token t+1; targets are the completion tokens p..n-1
            shift[at + p - 1:at + n - 1] = ids[at + p:at + n]
            target_task[at + p - 1:at + n - 1] = task[e]
            at += n
        if at < T:
            pos[at:] = np.arange(T - at)  # the padding is its own document, never a target
        labels = np.full(T, IGNORE, dtype=np.int64)
        labels[1:] = shift[:-1]  # Trainer counts labels[..., 1:] or shift_labels; both give the target count
        return dict(input_ids=torch.from_numpy(ids), position_ids=torch.from_numpy(pos),
                    shift_labels=torch.from_numpy(shift), labels=torch.from_numpy(labels),
                    target_task=torch.from_numpy(target_task), real_tokens=torch.tensor(at))


def collate(items):
    import torch
    return {k: torch.stack([item[k] for item in items]) for k in items[0]}


def make_trainer_class():
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint
    from transformers import Trainer

    def chunk_ce(hidden, weight, targets):
        return F.cross_entropy(F.linear(hidden, weight).float(), targets, reduction='none')

    class CompletionTrainer(Trainer):
        """Completion-token cross-entropy over the vocabulary in checkpointed chunks."""
        ce_chunk = 4096
        task_names = ()

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            shift = inputs['shift_labels']
            target_task = inputs['target_task']
            real = inputs['real_tokens']
            captured = {}
            core = self.accelerator.unwrap_model(model)
            handle = core.model.register_forward_hook(lambda m, a, out: captured.__setitem__('h', out.last_hidden_state))
            try:
                model(input_ids=inputs['input_ids'], position_ids=inputs['position_ids'], logits_to_keep=1,
                      use_cache=False)
            finally:
                handle.remove()
            hidden = captured['h']
            keep = shift.ne(IGNORE)
            h, t, k = hidden[keep], shift[keep], target_task[keep]
            weight = core.get_output_embeddings().weight
            losses = [checkpoint(chunk_ce, h[i:i + self.ce_chunk], weight, t[i:i + self.ce_chunk], use_reentrant=False)
                      for i in range(0, h.shape[0], self.ce_chunk)]
            per_token = torch.cat(losses) if losses else hidden.new_zeros(0, dtype=torch.float32)
            total = per_token.sum()
            if num_items_in_batch is not None:
                loss = total / num_items_in_batch
                if self.args.average_tokens_across_devices and self.accelerator.num_processes > 1:
                    loss = loss * self.accelerator.num_processes
            else:
                loss = total / max(1, int(keep.sum()))
            with torch.no_grad():
                stats = getattr(self, '_task_stats', None)
                if stats is None:
                    stats = self._task_stats = torch.zeros(len(self.task_names), 2, device=per_token.device,
                                                           dtype=torch.float64)
                if per_token.numel():
                    stats[:, 0].index_add_(0, k, per_token.detach().double())
                    stats[:, 1].index_add_(0, k, torch.ones_like(per_token, dtype=torch.float64))
                self._real_tokens = getattr(self, '_real_tokens', 0) + int(real.sum())
                self._padded_tokens = getattr(self, '_padded_tokens', 0) + int(inputs['input_ids'].numel())
            return (loss, None) if return_outputs else loss

        def log(self, logs, *args, **kwargs):
            if 'loss' in logs and getattr(self, '_task_stats', None) is not None:
                import torch.distributed as dist
                stats = self._task_stats
                counts = torch.tensor([getattr(self, '_real_tokens', 0), getattr(self, '_padded_tokens', 0)],
                                      device=stats.device, dtype=torch.float64)
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(stats)
                    dist.all_reduce(counts)
                now = time.time()
                elapsed = now - getattr(self, '_last_log_time', getattr(self, '_start_time', now))
                for name, (s, n) in zip(self.task_names, stats.tolist()):
                    if n:
                        logs[f'loss/{name}'] = round(s / n, 4)
                if elapsed > 0:
                    logs['tokens_per_s'] = round(counts[0].item() / elapsed)
                    logs['padded_tokens_per_s'] = round(counts[1].item() / elapsed)
                logs['peak_mem_gb'] = round(torch.cuda.max_memory_allocated() / 2**30, 1) if torch.cuda.is_available() else 0
                logs['peak_reserved_gb'] = round(torch.cuda.max_memory_reserved() / 2**30, 1) if torch.cuda.is_available() else 0
                stats.zero_()
                self._real_tokens = self._padded_tokens = 0
                self._last_log_time = now
            super().log(logs, *args, **kwargs)

    return CompletionTrainer


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-dir', nargs='+', required=True, help='build_data.py outputs: DIR, or DIR@task,task for '
                   'only those tasks of DIR')
    p.add_argument('--output-model', type=Path, required=True)
    p.add_argument('--work-dir', type=Path, help='Checkpoints (default <output-model>_training)')
    p.add_argument('--base-model', default=BASE_MODEL)
    p.add_argument('--base-revision', default=BASE_REVISION)
    p.add_argument('--attention', default='flex_attention', choices=['flex_attention', 'sdpa', 'flash_attention_2'])
    p.add_argument('--bin-tokens', type=int, default=65536, help='Tokens per packed row (one row per GPU micro-batch)')
    p.add_argument('--gradient-accumulation-steps', type=int, default=2)
    p.add_argument('--learning-rate', type=float, default=1e-5)
    p.add_argument('--warmup-ratio', type=float, default=0.02)
    p.add_argument('--weight-decay', type=float, default=0.01)
    p.add_argument('--epochs', type=float, default=1)
    p.add_argument('--max-steps', type=int, default=-1, help='Stop after N optimizer steps (a pilot)')
    p.add_argument('--logging-steps', type=int, default=10)
    p.add_argument('--save-steps', type=int, default=500)
    p.add_argument('--no-gradient-checkpointing', action='store_true',
                   help='Keep all activations (faster when they fit; a memory setting, not part of the identity)')
    p.add_argument('--ce-chunk', type=int, default=4096, help='Completion positions per vocabulary chunk (memory only)')
    p.add_argument('--loader-workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no-receipt', action='store_true', help='A pilot: write no stage receipt')
    p.add_argument('--no-save', action='store_true', help='A pilot: save no model (implies --no-receipt)')
    from matchminer_ai.training.trialspace.stage import add_check_arg
    add_check_arg(p)
    return p


def identity(args):
    from matchminer_ai.training.trialspace.stage import receipt_id
    return dict(version='oncoreasoning-model-v1', data=[[receipt_id(d.partition('@')[0]), d.partition('@')[2]] for d in args.data_dir], base_model=args.base_model,
                base_revision=args.base_revision, attention=args.attention, bin_tokens=args.bin_tokens,
                gradient_accumulation_steps=args.gradient_accumulation_steps, learning_rate=args.learning_rate,
                warmup_ratio=args.warmup_ratio, weight_decay=args.weight_decay, epochs=args.epochs,
                max_steps=args.max_steps, seed=args.seed, world_size=int(os.environ.get('WORLD_SIZE', '1')))


def train(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from transformers.trainer_utils import get_last_checkpoint

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, revision=args.base_revision, trust_remote_code=True)
    frame, parts = load_examples(args.data_dir)
    tasks = sorted(frame.task.unique())
    bins, dropped = pack(frame.n_tokens.to_numpy(), args.bin_tokens, args.seed)
    rank0 = int(os.environ.get('RANK', '0')) == 0
    if rank0:
        lengths = frame.n_tokens.to_numpy()
        fill = sum(int(lengths[b].sum()) for b in bins) / (len(bins) * args.bin_tokens) if bins else 0
        print(json.dumps(dict(event='packed', examples=len(frame), bins=len(bins), dropped_longer_than_bin=dropped,
                              fill=round(float(fill), 4), tasks=tasks)), flush=True)
    dataset = PackedBins(frame, parts, bins, args.bin_tokens, tokenizer.pad_token_id, tasks)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, revision=args.base_revision, dtype=torch.float32,
                                                 attn_implementation=args.attention, trust_remote_code=True)
    model.config.use_cache = False
    work = args.work_dir or Path(str(args.output_model) + '_training')
    Trainer = make_trainer_class()
    Trainer.ce_chunk, Trainer.task_names = args.ce_chunk, tasks
    trainer = Trainer(model=model, train_dataset=dataset, data_collator=collate, args=TrainingArguments(
        output_dir=str(work), learning_rate=args.learning_rate, num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=1, gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_ratio, lr_scheduler_type='cosine',  # a float below 1 is a ratio of the steps weight_decay=args.weight_decay,
        optim='adamw_torch_fused', bf16=True, gradient_checkpointing=not args.no_gradient_checkpointing,
        gradient_checkpointing_kwargs=dict(use_reentrant=False), remove_unused_columns=False,
        logging_steps=args.logging_steps, save_steps=args.save_steps, save_total_limit=2, report_to='none',
        seed=args.seed, dataloader_num_workers=args.loader_workers, dataloader_pin_memory=True,
        ddp_find_unused_parameters=False, average_tokens_across_devices=True, logging_first_step=True,
        use_cpu=not torch.cuda.is_available()))  # CPU only for a smoke test
    trainer.model_accepts_loss_kwargs = True  # compute_loss divides by the global completion-token count
    trainer._start_time = time.time()
    trainer.train(resume_from_checkpoint=get_last_checkpoint(str(work)) if work.exists() else None)
    if trainer.is_world_process_zero():
        summary = dict(examples=len(frame), bins=len(bins), dropped=dropped, tasks=tasks)
        if args.no_save:
            print(json.dumps(dict(event='pilot finished', global_step=trainer.state.global_step, **summary,
                                  log_history=trainer.state.log_history[-5:])), flush=True)
        else:
            save(trainer, model, tokenizer, args, summary)


def save(trainer, model, tokenizer, args, packed):
    import torch
    from huggingface_hub import snapshot_download
    output = args.output_model
    model.to(torch.bfloat16)
    trainer.save_model(str(output))
    tokenizer.save_pretrained(str(output))
    base = Path(args.base_model) if Path(args.base_model).is_dir() else Path(
        snapshot_download(args.base_model, revision=args.base_revision))
    config = json.loads((output / 'config.json').read_text())
    config['auto_map'] = json.loads((base / 'config.json').read_text())['auto_map']
    (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    for path in base.glob('*.py'):
        (output / path.name).write_bytes(path.read_bytes())
    for name in ('chat_template.jinja', 'generation_config.json', 'LICENSE'):
        if (base / name).exists() and not (output / name).exists():
            (output / name).write_bytes((base / name).read_bytes())
    (output / 'oncoreasoning_training.json').write_text(json.dumps(dict(
        base_model=args.base_model, base_revision=args.base_revision, packed=packed,
        log_history=trainer.state.log_history[-50:], global_step=trainer.state.global_step), indent=2) + '\n')


def main():
    args = parser().parse_args()
    args.no_receipt = args.no_receipt or args.no_save
    from matchminer_ai.training.trialspace import stage
    ident = identity(args)
    if args.check:
        stage.exit_check(args.output_model, 'oncoreasoning-model', ident)
    if not args.no_receipt and stage.require_new_or_same(args.output_model, 'oncoreasoning-model', ident):
        print('[skip] oncoreasoning-model already complete', flush=True)
        return
    train(args)
    if int(os.environ.get('RANK', '0')) == 0 and not args.no_receipt:
        files = [str(p.relative_to(args.output_model)) for p in sorted(args.output_model.rglob('*'))
                 if p.is_file() and p.name != stage.NAME]
        stage.write_receipt(args.output_model, 'oncoreasoning-model', ident, files)


if __name__ == '__main__':
    main()
