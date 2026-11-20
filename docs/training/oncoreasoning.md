# OncoReasoning: multi-task distillation of the teacher, with its reasoning

User instruction, 2026-11-16: bring back OncoReasoning and train it, as text generation, on every teacher task
(patient summarization, trial cohort ETL, labeling) using the teacher's outputs **including its reasoning**, starting
from the fine-tuned reasoning model `LiquidAI/LFM3-1.7B` (revision `1da06bec6a819da10b54c67e5c90972b46d81871`, LFM Open
License v1.0, the same license as the decision model's d1-2B). This replaces the November 5 code (Gemma student, LoRA
adapters, freshly distilled answers without reasoning), which was removed and is not restored.

## Tasks and sources

Every example is a request exactly as the teacher (Nemotron 3.5 Super) received it, and the teacher's accepted
response: finish reason `stop`, no validation error, first attempt (a retry's request carries a failure notice).
Requests not stored verbatim are rebuilt with the frozen inference snapshot that sent them and checked against the
hash the run stored. Paths are relative to `<data root>/no_phi/synthetic_dataset_n5000_screened_20261106/`.

| Task | Source | Request rebuilt from / checked by |
|---|---|---|
| `summary` | `patient_summaries/patient_serial_summaries.parquet` (`reasoning_trace`, `new_summary`) | notes in `all_synthetic_notes.parquet`, deduplicated and concatenated by the frozen `summaries.summarize`, prompt by `source/` inference (`get_serial_patient_prompt`); first/last dates and the prior-summary text must match |
| `cohort_screen` | protocols: `as_generated/synthetic_protocols/cohort_extraction/trials/*/extraction.json`; ctgov: `as_generated/upstream/screen/screen.jsonl` (all 29,005 screened listings) | `source_cohorts_fast` prompts (per-step hashes); protocol text sha256; ctgov listing sha256 (`input_sha256`) |
| `cohort_list`, `cohort_criteria` | protocols as above; ctgov: `as_generated/upstream/registry_extraction/checkpoints/*.json.gz` | same prompts; ctgov listing from `seeds.jsonl` via the frozen `load_sampled_trials` |
| `cohort_merge` | protocols: `.../cohort_extraction/trials/*/merge.json`; ctgov: the registry checkpoints' `merge` attempts (trials with 2+ cohorts) | protocols: `cohorts_unmerged.parquet`, `input_sha256`; ctgov: the checkpoint's own `unmerged_cohort_rows`; prompt hash 0e17e31f |
| `cohort_deboilerplate` | v4 per-line tags: `trials_deboilerplate_v4/.../trials/*/deboilerplate.json` (protocols), `trials_deboilerplate_v4/upstream/ctgov_deboilerplate/*/deboilerplate.json` | `source/` prompt (hash 211bc8a1), each cohort's `input_sha256`; up to 4 random lines per cohort |
| `label_match`, `label_boilerplate`, `label_criteria` | `trialspace/trialspace_round{1,2,3}/{teacher,boilerplate_teacher,criteria_teacher}/labels.jsonl` (stored `messages`, `reasoning`, `raw_answer`) | stored verbatim; question 7 only from the relabeled contract `7a7eb230` (never `criteria_teacher_v0_20261110`) |

Label files are sampled by seeking to a random byte offset and taking the next whole line: a record's chance depends
on the length of the record before it, not its own, so the sample is close to uniform without an index. Reads of the
shared copies are paced.

## Format

LFM3's chat template renders an assistant turn with `reasoning` as `<think>{reasoning}</think>{answer}<|im_end|>`.
Each example is stored as token ids: the prompt (the template with `add_generation_prompt=True, enable_thinking=True`,
ending `<|im_start|>assistant\n<think>`) and the completion (`{reasoning}</think>{answer}<|im_end|>`). The loss is on
the completion only. The teacher's system prompts are kept where it had one (the match labels); the serial-summary
reasoning prompt is empty for Nemotron.

## Stages

```bash
# data (CPU; one process per task shard, each importing its frozen snapshot)
python -m matchminer_ai.training.oncoreasoning.build_data --output-dir DATA \
    --tokens summary=... label_match=... ... --examples summary=N --shards summary=8 label_match=16 ...
# model (8 GPUs)
torchrun --nproc_per_node 8 -m matchminer_ai.training.oncoreasoning.train --data-dir DATA --output-model MODEL
```

- `build_data` writes `parts/<task>-<shard>/{tokens.bin,index.parquet}`, `manifest.json` (examples, tokens,
  supervised tokens and length percentiles per task), `preview.jsonl` and `stage_receipt.json` (`oncoreasoning-data`).
- `train.py`: full fine-tune with transformers' Trainer (fp32 master weights, bf16 autocast, gradient checkpointing,
  fused AdamW, cosine). Examples are packed into fixed rows of `--bin-tokens`; `position_ids` restart per example,
  which LFM3 turns into per-document attention and per-document short-conv windows (checked: a packed document's
  logits equal its own run's to 1e-6). The vocabulary projection runs in checkpointed chunks over the completion
  positions only. Logs add per-task completion loss, real tokens per second and peak memory. `--no-save` for pilots.
  Stage receipt `oncoreasoning-model`.
  - `--data-dir` takes several directories, and `DIR@task,task` uses only those tasks of `DIR`.
  - Duplicate `(task, id)` examples are dropped.
- More `build_data` options:
  - `--exclude-from` skips examples in earlier builds.
  - `--label-block-mb` samples label files in sequential blocks, split across shards.
  - `--label-partitions` and `--match-earliest-round` cover the full run (the archived training repository's `oncoreasoning_training/FULL_RUN_PLAN.md`).
  - `--save-text` writes evaluation sets.
  - Every summary shard reads one local cache of the notes and serial summaries (`ONCOREASONING_CACHE`).
- `evaluate` generate|score` compares student completions with the teacher's answers per task:
  - validity;
  - agreement on the decision fields;
  - token F1 for free text.

## Full run

The archived training repository's `oncoreasoning_training/FULL_RUN_PLAN.md` covers:
- the 2026-11-16 B200 pilot: throughput, memory, per-task loss and the validated settings;
- the full data pool and options A–D;
- what is already built and where;
- the match/boilerplate partitions (`--label-partitions`, `--match-earliest-round`);
- the full-run driver and its VM contract;
- the steps to launch.

No full run has been started (user, 2026-11-16: "too expensive for now").

## Evaluation

The examples not sampled are the evaluation pool (no synthetic partition is reserved, per the training policy):
teacher-answer agreement for the labels (parsed yes/no per question), schema validity and agreement for the cohort
steps, and summary checks. These are in-distribution diagnostics, not held-out validation.
