# OncoReasoning: multi-task distillation of the teacher, with its reasoning

OncoReasoning is a small generative model trained, as text generation, on every teacher task (patient summarization,
trial cohort extraction, labeling) using the teacher's outputs **including its reasoning**. It is a full fine-tune of
the reasoning model `LiquidAI/LFM3-1.7B` (revision `1da06bec6a819da10b54c67e5c90972b46d81871`, LFM Open License v1.0,
the same license as the decision model's d1-2B).

The stages live in `matchminer_ai.training.oncoreasoning`. They are standalone: `mmai-train` does not run them, and the
package does not use the resulting model at runtime. They read synthetic (`no_phi`) data only.

## Tasks and sources

Every example is a request exactly as the teacher (Nemotron 3.5 Super) received it, and the teacher's accepted
response: finish reason `stop`, no validation error, first attempt (a retry's request carries a failure notice).
Requests not stored verbatim are rebuilt with the frozen code snapshot that sent them, which the dataset keeps, and
checked against the hash the run stored. Each task runs in its own process that imports its snapshot
(`oncoreasoning.sources.SNAPSHOTS`). Paths are relative to the dataset (`--dataset`, default
`<data root>/no_phi/synthetic_dataset_n5000_screened_20261106/`).

| Task | Source | Request rebuilt from / checked by |
|---|---|---|
| `summary` | `patient_summaries/patient_serial_summaries.parquet` (`reasoning_trace`, `new_summary`) | the notes in `patient_summaries/all_synthetic_notes.parquet`, deduplicated and concatenated by the frozen summarizer, and the frozen serial prompt (`get_serial_patient_prompt`); first and last dates and the prior-summary text must match |
| `cohort_screen` | protocols: `as_generated/synthetic_protocols/cohort_extraction/trials/*/extraction.json`; ctgov: `as_generated/upstream/screen/screen.jsonl` (every screened listing) | the frozen cohort prompts (per-step hashes); protocol text SHA-256; ctgov listing SHA-256 (`input_sha256`) |
| `cohort_list`, `cohort_criteria` | protocols as above; ctgov: `as_generated/upstream/registry_extraction/checkpoints/*.json.gz` | the same prompts; the ctgov listing from `seeds.jsonl` |
| `cohort_merge` | protocols: `.../cohort_extraction/trials/*/merge.json`; ctgov: the checkpoints' `merge` attempts (trials with 2+ cohorts) | protocols: `cohorts_unmerged.parquet`, `input_sha256`; ctgov: the checkpoint's own `unmerged_cohort_rows`; the merge prompt hash |
| `cohort_deboilerplate` | per-line tags: `trials_deboilerplate_v4/synthetic_protocols/cohort_extraction/trials/*/deboilerplate.json` (protocols), `trials_deboilerplate_v4/upstream/ctgov_deboilerplate/*/deboilerplate.json` (ctgov) | the frozen prompt's hash and each cohort's `input_sha256`; up to `--deboilerplate-lines` (4) random lines per cohort |
| `label_match`, `label_boilerplate`, `label_criteria` | `trialspace/trialspace_round{1,2,3}/{teacher,boilerplate_teacher,criteria_teacher}/labels.jsonl` (stored `messages`, `reasoning`, `raw_answer`) | stored verbatim; per-criterion labels only from the label contract `7a7eb230` (`sources.CRITERIA_CONTRACT`) |

Label files are sampled by seeking to a random byte offset and taking the next whole line: a record's chance depends
on the length of the record before it, not its own, so the sample is close to uniform without an index.
`--label-block-mb` instead reads random non-overlapping blocks sequentially and keeps every record in them. Reads of
the shared copies are paced (`--mb-per-s`).

## Format

LFM3's chat template renders an assistant turn with `reasoning` as `<think>{reasoning}</think>{answer}<|im_end|>`.
Each example is stored as token IDs: the prompt (the template with `add_generation_prompt=True, enable_thinking=True`,
ending `<|im_start|>assistant\n<think>`) and the completion (`{reasoning}</think>{answer}<|im_end|>`). The loss is on
the completion only. The teacher's system prompt is kept where it had one (the match labels).

## Stages

```bash
# data (CPU; one process per task shard)
python -m matchminer_ai.training.oncoreasoning.build_data --output-dir DATA \
    --examples summary=200 label_match=300 ... --shards summary=8 label_match=16 ...
# model (8 GPUs)
torchrun --nproc_per_node 8 -m matchminer_ai.training.oncoreasoning.train --data-dir DATA --output-model MODEL
```

**`build_data`** samples each task in a seeded random order until its budget (`--examples task=N` or
`--tokens task=N`) is met, and writes:

- `parts/<task>-<shard>/tokens.bin` (int32 token IDs, examples back to back) and `index.parquet` (one row per
  example: task, source, ID, offset, length, prompt length);
- `manifest.json`: examples, tokens, supervised tokens and length percentiles per task, and dropped and skipped counts;
- `preview.jsonl`: the first two rendered examples per task shard, for review;
- `stage_receipt.json` (kind `oncoreasoning-data`; see [Training](index.md) for receipts and `--check`).

More `build_data` options:

- `--exclude-from` skips the examples of earlier builds; `--save-text` also writes each example as text
  (`examples.jsonl` per part), for an evaluation set.
- `--label-partitions task=k` keeps only label pairs in partitions 0 to k-1 of 100 (requires `--label-block-mb`), and
  `--match-earliest-round` keeps a match pair only from the earliest round that labeled it.
- `--max-length` (131,072 tokens) drops longer examples, counted in the manifest.
- Every summary shard reads one local cache of the notes and serial summaries (in `ONCOREASONING_CACHE`, default
  `/tmp`).

**`train`** fully fine-tunes the model with transformers' `Trainer` (no LoRA): fp32 master weights, bf16 autocast,
gradient checkpointing, fused AdamW, cosine schedule (defaults: learning rate 1e-5, 2% warmup, 1 epoch).

- **Packing:** examples are packed into fixed rows of `--bin-tokens` (65,536). `position_ids` restart per example,
  which LFM3 turns into per-document attention and per-document short-conv windows, so examples never see each other
  (a packed document's logits equal its own run's to 1e-6).
- **Loss:** next-token cross-entropy over the completion tokens only, with the vocabulary projection in checkpointed
  chunks over those positions (`--ce-chunk`).
- **Inputs:** `--data-dir` takes several directories, and `DIR@task,task` uses only those tasks of `DIR`. Duplicate
  `(task, id)` examples are dropped.
- **Logs:** per-task completion loss, real tokens per second and peak memory, besides `Trainer`'s.
- **Pilots:** `--max-steps N`; `--no-save` saves no model and writes no receipt. Otherwise the output gets a stage
  receipt of kind `oncoreasoning-model`.

**`evaluate`** compares student completions with the teacher's accepted answers:

```bash
python -m matchminer_ai.training.oncoreasoning.evaluate generate --eval-data EVAL --model MODEL --output OUT --gpus 8
python -m matchminer_ai.training.oncoreasoning.evaluate score --eval-data EVAL --output OUT
```

Per task it reports `finished` (the completion closed its reasoning and ended), `valid` (the answer parses as that
task's answer), `agree` (the decision fields equal the teacher's: screen yes/no, cohort count, merge groups, both
per-line tags, the five match answers and the 0-6 score, boilerplate `excluded`, per-criterion yes/no), `token_f1`
(bag-of-words F1 with the teacher's free text: criteria text, cohort names, summaries) and lengths.

## Evaluation data

No synthetic partition is reserved. An evaluation set is a
`build_data --save-text --exclude-from <training data dirs>` build: the same verified adapters, a different `--seed`,
and none of the training examples. Its metrics are in-distribution agreement with the teacher, not held-out or
clinical validation.
