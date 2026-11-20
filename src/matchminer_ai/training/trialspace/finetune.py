#!/usr/bin/env python3
import argparse
import os
import glob
import json
import pandas as pd
import torch
from matchminer_ai.training.trialspace.prompt_profiles import PROFILES, SYMMETRIC_PREFIX, model_prompts, training_prompts

# Google's symmetric similarity format: either side can be the retrieval query.
# https://huggingface.co/google/embeddinggemma-2#best-practices
PROMPT_PREFIX = SYMMETRIC_PREFIX
DEFAULT_BASE_MODEL = "google/embeddinggemma-2"
DEFAULT_INPUT = "space_specific_eligibility_checks.parquet"
MAX_TOKENS = 4096  # key texts only; DFCI key criteria reach ~2,100 tokens


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune a text embedding model for trial–patient retrieval."
    )
    # Allow multiple -i flags; also supports glob patterns
    parser.add_argument(
        "-i", "--input-parquet",
        action="append",
        help="Path(s) to input parquet file(s). May be provided multiple times "
             "and may include glob patterns. If omitted, defaults to "
             f"'{DEFAULT_INPUT}'."
    )
    parser.add_argument(
        "-c", "--ckpt-dir",
        default="./initial_embeddinggemma2_training",
        help="Checkpoint/output directory for the SentenceTransformer trainer."
    )
    parser.add_argument(
        "-o", "--output-model",
        default="pt_trial_summary_pertrial_embeddinggemma2_finetuned.model",
        help="Directory name to save the final fine-tuned model (directory will be created)."
    )
    parser.add_argument(
        "-m", "--base-model",
        default=DEFAULT_BASE_MODEL,
        help="EmbeddingGemma 2 model ID or a local TrialSpace checkpoint from this model family."
    )
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--dataloader-num-workers', type=int, default=4,
                        help='Background tokenization/data-loading workers per training process; 0 runs inline.')
    parser.add_argument('--dataloader-prefetch-factor', type=int, default=2,
                        help='Batches prepared ahead per worker (ignored with zero workers).')
    parser.add_argument('--loss-mini-batch-size', type=int, default=0,
                        help='Encoder chunk size for gradient caching of both full-batch losses; 0 disables caching.')
    parser.add_argument('--encoder-mini-batch-size', type=int, default=0,
                        help='Encode length-sorted chunks once and retain their graphs for the full loss; 0 disables chunking.')
    parser.add_argument('--encoder-token-budget', type=int, default=8192,
                        help='Padded-token limit per retained encoder chunk; a longer single text runs alone.')
    parser.add_argument('--gradient-checkpointing', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--mask-known-positives', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--save-steps', type=int, default=2500,
                        help='Checkpoint interval; a shorter interval is useful on Spot VMs.')
    parser.add_argument('--logging-steps', type=int, default=100)
    parser.add_argument('--epochs', type=float, default=1,
                        help='Passes over the training pairs; the linear warmup/decay schedule spans them')
    parser.add_argument('--max-steps', type=int,
                        help='Stop after this many optimizer steps (learning-rate schedule over them) instead of --epochs')
    parser.add_argument('--prompt-profile', choices=PROFILES, default='sentence_similarity')
    parser.add_argument('--max-tokens', type=int, default=MAX_TOKENS)
    parser.add_argument('--model-revision')
    parser.add_argument('--model-cache-dir')
    parser.add_argument('--model-dtype', choices=('float32', 'bfloat16'), default='float32',
                        help='Model/optimizer parameter dtype; BF16 autocast is enabled separately on CUDA.')
    from matchminer_ai.training.trialspace.workspace import add_training_storage_args
    add_training_storage_args(parser)
    from matchminer_ai.training.trialspace.stage import add_check_arg
    add_check_arg(parser)
    return parser.parse_args()


def expand_inputs(input_args):
    """Expand a list of input paths (which may include globs) to a concrete file list."""
    if not input_args:
        input_args = [DEFAULT_INPUT]
    files = []
    for pattern in input_args:
        expanded = sorted(glob.glob(pattern)) if any(ch in pattern for ch in "*?[]") else [pattern]
        files.extend(expanded)
    # Deduplicate while preserving order
    seen = set()
    unique_files = []
    for f in files:
        if f not in seen:
            unique_files.append(f)
            seen.add(f)
    if not unique_files:
        raise FileNotFoundError("No input parquet files found after expanding patterns.")
    return unique_files


def load_and_concat(parquet_files):
    """Load multiple parquet files and concatenate them."""
    frames = []
    for p in parquet_files:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Input parquet not found: {p}")
        df = pd.read_parquet(p)
        print(f"[INFO] Loaded {p} with shape {df.shape}")
        from matchminer_ai.training.synthetic.pool import pooled_frame
        frames.append(pooled_frame(df))
    if len(frames) == 1:
        return frames[0]
    concat_df = pd.concat(frames, axis=0, ignore_index=True)
    print(f"[INFO] Concatenated {len(frames)} files -> shape {concat_df.shape}")
    return concat_df


def require_embeddinggemma2(base_model, *, revision=None, cache_dir=None):
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(base_model, revision=revision, cache_dir=cache_dir)
    if config.model_type != "embedding_gemma2":
        raise ValueError(
            "TrialSpace now requires google/embeddinggemma-2 or a checkpoint "
            "fine-tuned from it. Use fresh model/checkpoint directories for this run."
        )


def load_text_model(base_model, device, *, prompt_profile='sentence_similarity',
                    max_tokens=MAX_TOKENS, revision=None, cache_dir=None, model_dtype='float32'):
    """Keep Google's mean pooling/normalization, loading only the text encoder.

    The loader is the inference package's (the same one mining encodes with). The trainer uses BF16 autocast
    independently of parameter precision; BF16 parameters reproduce v22's native loading for the
    memory-limited larger-encoder-batch configuration. FP16 is unsupported.
    """
    from matchminer_ai.embedding.encoder import EncoderSettings, load_sentence_transformer

    if not 1 <= max_tokens <= 8192:
        raise ValueError('EmbeddingGemma 2 requires a token limit between 1 and 8192')
    if model_dtype not in ('float32', 'bfloat16'):
        raise ValueError('EmbeddingGemma 2 model dtype must be float32 or bfloat16')
    require_embeddinggemma2(base_model, revision=revision, cache_dir=cache_dir)
    return load_sentence_transformer(EncoderSettings(
        base_model, revision=revision, cache_dir=cache_dir, max_seq_length=max_tokens, dtype=model_dtype,
        prompts=model_prompts(prompt_profile), default_prompt_name='SentenceSimilarity',
        include_prompt_in_pooling=True, text_only=True), device)


def validate_resume_checkpoint(checkpoint, *, prompt_profile='sentence_similarity'):
    """Do not silently resume old Qwen or differently formatted training runs."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(checkpoint)
    with open(os.path.join(checkpoint, "config_sentence_transformers.json")) as handle:
        saved = json.load(handle)
    if (
        config.model_type != "embedding_gemma2"
        or config.vision_config is not None
        or config.audio_config is not None
        or saved.get("default_prompt_name") != "SentenceSimilarity"
        or any(saved.get("prompts", {}).get(name) != prefix
               for name, prefix in model_prompts(prompt_profile).items())
    ):
        raise ValueError(
            f"Incompatible TrialSpace checkpoint: {checkpoint}. "
            "Choose a fresh --ckpt-dir for text-only EmbeddingGemma 2 training."
        )


def per_loss_logging(losses_map):
    """Callback logging each dataset's loss over its own batches (loss_<dataset>, batches_<dataset>).

    The proportional sampler gives each batch to one dataset's loss, so the logged loss mixes them by their
    batch counts (about one ranking batch in nine). Each window's per-dataset sums and batch counts are summed
    over ranks; every rank logs at the same steps, so the all-reduce is collective.
    """
    from transformers import TrainerCallback
    names = list(losses_map)
    recorded = {name: [] for name in names}

    def recorder(name):
        def hook(module, inputs, output):
            if torch.is_grad_enabled():
                recorded[name].append(output.detach().float())
        return hook

    for name, loss in losses_map.items():
        loss.register_forward_hook(recorder(name))

    class PerLossLogger(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs or 'loss' not in logs:
                return
            totals = torch.zeros(len(names), 2, device=args.device)
            for i, name in enumerate(names):
                if recorded[name]:
                    totals[i, 0] = torch.stack(recorded[name]).sum()
                    totals[i, 1] = len(recorded[name])
                recorded[name].clear()
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(totals)
            extra = {}
            for (total, count), name in zip(totals.tolist(), names):
                if count:
                    extra[f'loss_{name}'] = round(total / count, 4)
                extra[f'batches_{name}'] = int(count)
            logs.update(extra)  # Before the progress printer, which runs after user callbacks.
            if state.log_history and state.log_history[-1].get('step') == state.global_step:
                state.log_history[-1].update(extra)  # Saved with each checkpoint's trainer_state.json.

    return PerLossLogger()


def stage_identity(args):
    """The fit stage's identity: its projected pairs and base model by their receipts, and the settings."""
    from pathlib import Path
    from matchminer_ai.training.trialspace import stage

    from matchminer_ai.training.cohort_format import file_stamp

    def located(path):  # the stage directory holding an input file and its name there, or (standalone) its stamp
        path = Path(path).resolve()
        for parent in path.parents:
            if (parent / stage.NAME).exists():
                return dict(stage=stage.receipt_id(parent), name=str(path.relative_to(parent)))
        return dict(file=path.name, stamp=file_stamp(path))

    def model(path):
        base = Path(path)
        if not base.is_dir():
            return path
        if stage.read_receipt(base):
            return stage.receipt_id(base)
        return stage.digest({str(p.relative_to(base)): file_stamp(p) for p in sorted(base.rglob('*')) if p.is_file()})
    return dict(version='trialspace-fit-v2', inputs=[located(p) for p in expand_inputs(args.input_parquet)],
                base_model=model(args.base_model),
                revision=args.model_revision, max_tokens=args.max_tokens, model_dtype=args.model_dtype,
                prompts=model_prompts(args.prompt_profile), batch_size=args.batch_size,
                world_size=int(os.environ.get('WORLD_SIZE', '1')), gradient_checkpointing=args.gradient_checkpointing,
                encoder_chunks=encoder_chunks(args), mask_known_positives=args.mask_known_positives,
                epochs=args.epochs, max_steps=args.max_steps)


def encoder_chunks(args):
    """How the encoder is chunked for the full-batch losses. The chunk sizes only trade memory for speed (both ways
    give the unchunked loss and gradients), so they stay out of the identities: a run that ran out of memory resumes
    from its checkpoint with smaller chunks."""
    return 'gradient_cache' if args.loss_mini_batch_size else 'retained' if args.encoder_mini_batch_size else None


def main():
    args = parse_args()
    from pathlib import Path
    from matchminer_ai.training.trialspace import stage
    identity = stage_identity(args)
    if args.check:
        stage.exit_check(args.output_model, 'fit', identity)
    if stage.require_new_or_same(args.output_model, 'fit', identity):
        print('[skip] model already fitted', flush=True)
        return
    from matchminer_ai.training.trialspace.workspace import run_training
    run_training(args, train)
    if int(os.environ.get('RANK', '0')) == 0:
        model = Path(args.output_model)
        files = [str(p.relative_to(model)) for p in sorted(model.rglob('*')) if p.is_file() and p.name != stage.NAME]
        stage.write_receipt(model, 'fit', identity, files,
                            upstream={f'input{i}': item['stage'] for i, item in enumerate(identity['inputs'])
                                      if 'stage' in item})


def train(args):
    from pathlib import Path
    if args.batch_size < 2:
        raise ValueError('Ranking losses require batch size at least two')
    if args.dataloader_num_workers < 0 or args.dataloader_prefetch_factor < 1:
        raise ValueError('Data-loader workers must be nonnegative and prefetch factor positive')
    if args.dataloader_num_workers:
        # Parallelize whole batches in native DataLoader workers. Avoid nesting
        # a tokenizer thread pool inside every worker (and before Linux fork).
        os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    if args.loss_mini_batch_size < 0:
        raise ValueError('Loss mini-batch size cannot be negative')
    if args.encoder_mini_batch_size < 0 or args.encoder_token_budget < 1:
        raise ValueError('Encoder chunk size cannot be negative and token budget must be positive')
    if args.loss_mini_batch_size and args.encoder_mini_batch_size:
        raise ValueError('Choose gradient caching or retained encoder chunks, not both')
    input_files = expand_inputs(args.input_parquet)
    from matchminer_ai.training.trialspace.data import write_json
    from matchminer_ai.matching.match_labels import MAX_SCORE
    from matchminer_ai.training.cohort_format import file_stamp
    identity = dict(inputs={str(Path(p).resolve()): file_stamp(p) for p in input_files},
        base_model=args.base_model, revision=args.model_revision, max_tokens=args.max_tokens,
        model_dtype=args.model_dtype,
        prompts=model_prompts(args.prompt_profile), loss=['MultipleNegativesRankingLoss', 'CoSENTLoss'],
        positive_score_minimum=1, normalization=f'score / {MAX_SCORE} * 2 - 1',
        batch_size=args.batch_size, gradient_checkpointing=args.gradient_checkpointing,
        encoder_chunks=encoder_chunks(args),
        sort_encoder_chunks_by_length=bool(args.loss_mini_batch_size or args.encoder_mini_batch_size),
        negative_mask='known_positive_pairs' if args.mask_known_positives else None, epochs=args.epochs, learning_rate=2e-5, warmup_ratio=.01,
        multi_dataset_batch_sampler='proportional', seed=42, training_pool='all_synthetic_training', diagnostics_held_out=False)
    if args.epochs <= 0:
        raise ValueError('--epochs must be positive')
    if args.max_steps is not None:
        if args.max_steps < 1:
            raise ValueError('--max-steps must be positive')
        identity['max_steps'] = args.max_steps
    manifest = Path(args.ckpt_dir) / 'training_identity.json'
    if int(os.environ.get('RANK', '0')) == 0:  # One writer per launch; a mismatch stops every rank.
        if manifest.exists() and json.loads(manifest.read_text()) != identity:
            raise ValueError('Training data/settings changed; use a fresh checkpoint directory')
        write_json(manifest, identity)
    print(f"[INFO] Using {len(input_files)} input file(s):")
    for f in input_files:
        print(f"       - {f}")
    print(f"[INFO] Checkpoint dir: {args.ckpt_dir}")
    print(f"[INFO] Output model dir: {args.output_model}")
    print(f"[INFO] Base model: {args.base_model}")

    # Deferred imports
    from sentence_transformers import (
        losses, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    )
    from datasets import Dataset
    from matchminer_ai.training.trialspace.checkpoints import latest_complete_checkpoint

    checkpoint = latest_complete_checkpoint(args.ckpt_dir)
    if checkpoint:
        validate_resume_checkpoint(checkpoint, prompt_profile=args.prompt_profile)

    # --- Load & filter data ---
    trial_checks = load_and_concat(input_files)
    # Source assignments are provenance only; load_and_concat retains every row.

    # Optional filter if present
    if "patient_long_text" in trial_checks.columns:
        trial_checks = trial_checks[trial_checks.patient_long_text.fillna("") != ""]

    req_cols = {"patient_summary", "this_space", "eligibility_result"}
    missing = req_cols - set(trial_checks.columns)
    if missing:
        raise ValueError(f"Input parquet missing required columns: {missing}")

    trial_checks = trial_checks[
        ~trial_checks.patient_summary.fillna("").str.contains(
            "No cancer|no cancer|No primary|No evidence of malignancy", na=False
        )
    ]
    trial_checks = trial_checks[
        ~trial_checks.patient_summary.fillna("").str.startswith("No information")
    ]

    # remove leading digit and period and space from trial space if present
    if args.prompt_profile == 'sentence_similarity':
        trial_checks["this_space"] = trial_checks['this_space'].str.replace(r'^\s*\d+\.', '', regex=True)

    # Drop rows where the LLM could not parse a score
    trial_checks = trial_checks[trial_checks.eligibility_result >= 0].copy()

    # Normalize eligibility_result (raw gated score 0 to MAX_SCORE) to [-1, 1] for cosine-similarity-scale labels
    trial_checks["eligibility_label"] = (trial_checks["eligibility_result"] / MAX_SCORE) * 2 - 1

    print("\n[INFO] Dataframe info after filtering:")
    trial_checks.info()
    print("\n[INFO] eligibility_result (raw score) counts:")
    print(trial_checks.eligibility_result.value_counts(dropna=False))
    print("\n[INFO] eligibility_label (normalized) distribution:")
    print(trial_checks.eligibility_label.describe())

    # Keep raw text in the datasets. The trainer applies the prefix once, then
    # the model tokenizer truncates the complete input, including special tokens.
    for column in ("patient_summary", "this_space"):
        trial_checks[column] = trial_checks[column].fillna("").astype(str)

    eligible = trial_checks[trial_checks.eligibility_result >= 1]
    print("\n[INFO] Eligible subset info:")
    eligible.info()

    # --- Model (from the base model) ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[INFO] Using device: {device}")
    model = load_text_model(args.base_model, device, prompt_profile=args.prompt_profile,
                            max_tokens=args.max_tokens, revision=args.model_revision,
                            cache_dir=args.model_cache_dir, model_dtype=args.model_dtype)
    print(f'[INFO] Model parameter dtype: {next(model.parameters()).dtype}')
    if args.gradient_checkpointing:
        # Enable through ST directly: Transformers 5.19's Trainer passes new
        # keyword arguments that ST 6.1's wrapper does not yet accept.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

    # --- Datasets ---
    mnri_frame = eligible[['patient_summary', 'this_space']].copy()
    if args.mask_known_positives:
        import numpy as np
        patient_codes, patient_texts = pd.factorize(trial_checks['patient_summary'], sort=False)
        cohort_codes, cohort_texts = pd.factorize(trial_checks['this_space'], sort=False)
        positive = (trial_checks.eligibility_result >= 1).to_numpy()
        identities = np.column_stack([patient_codes[positive], cohort_codes[positive]])
        mnri_frame['label'] = identities.tolist()
        positive_keys = identities[:, 0]*len(cohort_texts) + identities[:, 1]
    mnri_dataset = Dataset.from_pandas(mnri_frame, preserve_index=False)
    contrastive_dataset = Dataset.from_pandas(
        trial_checks[["patient_summary", "this_space", "eligibility_label"]]
        .rename(columns={"eligibility_label": "label"}),
        preserve_index=False
    )
    train_dataset = {
        "mnri_dataset": mnri_dataset,
        "contrastive_dataset": contrastive_dataset,
    }

    # --- Losses ---
    if args.mask_known_positives:
        from matchminer_ai.training.trialspace.losses import KnownPositiveRankingLoss
        mll_train_loss = KnownPositiveRankingLoss(model, positive_keys, len(cohort_texts))
    else:
        mll_train_loss = losses.MultipleNegativesRankingLoss(model=model)
    contrastive_train_loss = losses.CoSENTLoss(model=model)
    losses_map = {
        "mnri_dataset": mll_train_loss,
        "contrastive_dataset": contrastive_train_loss,
    }
    if args.loss_mini_batch_size:
        from matchminer_ai.training.trialspace.losses import CachedEmbeddingLoss
        losses_map = {name: CachedEmbeddingLoss(loss, args.loss_mini_batch_size)
                      for name, loss in losses_map.items()}
    if args.encoder_mini_batch_size:
        from matchminer_ai.training.trialspace.losses import RetainedEmbeddingLoss
        losses_map = {name: RetainedEmbeddingLoss(loss, args.encoder_mini_batch_size, args.encoder_token_budget)
                      for name, loss in losses_map.items()}

    # --- Trainer ---
    args_st = SentenceTransformerTrainingArguments(
        output_dir=args.ckpt_dir,
        per_device_train_batch_size=args.batch_size,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=bool(args.dataloader_num_workers),
        dataloader_prefetch_factor=(args.dataloader_prefetch_factor
                                   if args.dataloader_num_workers else None),
        accelerator_config={'non_blocking': torch.cuda.is_available()},
        gradient_checkpointing=False,  # Already enabled on the model above when requested.
        learning_rate=2e-5,
        lr_scheduler_type="linear",
        warmup_ratio=0.01,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        logging_steps=args.logging_steps,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps is not None else -1,
        prompts=training_prompts(args.prompt_profile),
        batch_sampler='batch_sampler',
        multi_dataset_batch_sampler='proportional',
        seed=42,
        report_to=[],
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=False,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args_st,
        train_dataset=train_dataset,
        loss=losses_map,
        callbacks=[per_loss_logging(losses_map)],
    )

    print(f'[INFO] Data loader: {args_st.dataloader_num_workers} workers, '
          f'prefetch={args_st.dataloader_prefetch_factor}, '
          f'persistent={args_st.dataloader_persistent_workers}, '
          f'pin_memory={args_st.dataloader_pin_memory}, '
          f'non_blocking={args_st.accelerator_config.non_blocking}')

    print("\n[INFO] Starting training...")
    if checkpoint:
        print(f"[INFO] Resuming from {checkpoint}")
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        print("[INFO] Starting fresh training")
        trainer.train()
    print("\n[INFO] Training complete.")

    # --- Save final model ---
    if trainer.is_world_process_zero():
        os.makedirs(os.path.dirname(os.path.abspath(args.output_model)), exist_ok=True)
        model.save(args.output_model)
        print(f"[INFO] Saved fine-tuned model to: {args.output_model}")


if __name__ == "__main__":
    main()
