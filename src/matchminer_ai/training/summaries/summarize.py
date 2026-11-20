#!/usr/bin/env python3
"""Serial summaries of the synthetic patients' notes, by the package's summarizer, written as training's tables.

``matchminer_ai.patients.summarize_patients`` does the work, exactly as at runtime: each patient's dated notes are
deduplicated, concatenated with date headers and split into overlapping token chunks; every chunk's prompt carries the
summary accepted for the patient's previous chunk; a chunk whose answer fails the quality checks is asked again
(``--max_attempts`` times). There is no note tagging or retrieval step. ``--shard_dir`` holds its checkpoints (a rerun
resumes there; with ``--local-summary-storage`` they are kept on local scratch and archived to ``--shard_dir`` at the
end, ``summaries.workspace``). Requests go to an existing OpenAI-compatible pool: ``--server_urls_file`` (a registry
re-read while running; each row's ``upstream_concurrency`` is its live limit) or ``--server_urls``.

Outputs, read by ``synthetic.prepare summaries`` and OncoReasoning:
- ``--output_parquet``: one row per chunk (``patients.serial_trace``): the patient column, ``chunk_index``,
  ``first_date``, ``last_date``, ``prior_summary``, ``reasoning_trace`` (= ``new_summary_reasoning``),
  ``new_summary`` (the answer), ``patient_summary`` (the answer normalized) and ``summary_generation_date``;
- ``--patient_summaries_parquet``: one row per patient: ``patient_summary`` (the last chunk's answer, normalized),
  ``last_note_date`` and ``summary_generation_date``. A patient whose chain failed every attempt keeps the failed
  chunk's ``ERROR:`` answer (or an empty summary), which ``synthetic.prepare`` refuses.

Example::

    python -m matchminer_ai.training.summaries.summarize \\
      --input_parquet DATA/no_phi/all_synthetic_notes.parquet \\
      --output_parquet DATA/no_phi/patient_serial_summaries.parquet \\
      --patient_summaries_parquet DATA/no_phi/patient_summaries.parquet --shard_dir DATA/no_phi/summary_shards \\
      --model nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 --server_urls_file servers.json
"""

import argparse
from datetime import date
import os
from pathlib import Path

import pandas as pd


def load_summary_notes(path, patient_id_col, date_col, text_col):
    """Read only the columns summarization uses; note-generation reasoning and event histories stay on disk."""
    columns = list(dict.fromkeys([patient_id_col, date_col, text_col]))
    return pd.read_parquet(path, columns=columns)


def make_config(args):
    """The default preset with this stage's model, chunking, limits and server pool."""
    from matchminer_ai.config import load_default_preset
    from matchminer_ai.llm.model_profiles import apply_model_profile

    config = load_default_preset()
    family = args.tokenizer_model or args.model
    if apply_model_profile(config, family, sections=['patient']) is None:
        raise ValueError('No vendor sampling profile; supply a recognized --tokenizer-model')
    config.patient.update(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap,
                          summary_quality_max_attempts=args.max_attempts,
                          prompt_build_workers=args.preparation_workers)
    config.patient['remote'].update(model_name=args.model, tokenizer_name=family)
    config.patient['remote']['request_params']['max_tokens'] = args.max_tokens
    config.remote.update(enabled=True, server_urls=args.server_urls or [], server_urls_file=args.server_urls_file,
                         max_concurrent_requests=args.max_concurrent_per_server, request_timeout=args.request_timeout,
                         max_retries=args.max_retries)
    config.remote['adaptive_concurrency'].update(enabled=args.scheduler_adaptive_concurrency,
                                                  initial=args.initial_concurrent_per_server, metrics_interval=10)
    config.model_metadata_cache_dir = str(Path(args.shard_dir) / 'model_metadata')
    return config


def _text(value):
    return '' if value is None or (isinstance(value, float) and pd.isna(value)) else str(value)


def training_tables(trace, patient_col, generated_on):
    """``serial_trace`` rows as the serial and per-patient tables training has always written."""
    from matchminer_ai.patients.postprocess import normalize_patient_summary

    serial = pd.DataFrame({
        patient_col: trace.patient_id, 'chunk_index': trace.chunk_index.astype(int),
        'first_date': trace.first_date.map(_text), 'last_date': trace.last_date.map(_text),
        'prior_summary': trace.prior_summary.map(_text), 'reasoning_trace': trace.reasoning.map(_text),
        'new_summary_reasoning': trace.reasoning.map(_text), 'new_summary': trace.summary.map(_text),
    })
    serial['patient_summary'] = serial.new_summary.map(normalize_patient_summary)
    serial['summary_generation_date'] = generated_on
    last = serial.groupby(patient_col, sort=False).tail(1)
    patients = pd.DataFrame({patient_col: last[patient_col].tolist(),
                             'patient_summary': last.patient_summary.tolist(),
                             'last_note_date': last.last_date.tolist(),
                             'summary_generation_date': generated_on})
    return serial, patients


def summarize(args):
    from matchminer_ai.patients import serial_trace, summarize_patients

    notes = load_summary_notes(args.input_parquet, args.patient_id_col, args.date_col, args.text_col)
    if args.max_patients is not None:
        notes = notes[notes[args.patient_id_col].isin(notes[args.patient_id_col].unique()[:args.max_patients])]
    frame = pd.DataFrame({'patient_id': notes[args.patient_id_col].astype(str),
                          'note_text': notes[args.text_col], 'note_date': notes[args.date_col]})
    print(f'Summarizing {frame.patient_id.nunique()} patients ({len(frame)} notes)', flush=True)
    _, metadata = summarize_patients(frame, config=make_config(args), checkpoint_dir=args.shard_dir,
                                     return_metadata=True)
    serial, patients = training_tables(serial_trace(args.shard_dir), args.patient_id_col, date.today().isoformat())
    for table, path in ((serial, args.output_parquet), (patients, args.patient_summaries_parquet)):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(path, index=False)
    empty = int(patients.patient_summary.str.strip().eq('').sum())
    print(f'Wrote {len(serial)} chunk rows and {len(patients)} patients ({empty} without a summary); '
          f'quality: {metadata.get("summary_quality")}', flush=True)


def parser():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--input_parquet', required=True)
    ap.add_argument('--output_parquet', required=True)
    ap.add_argument('--patient_summaries_parquet',
                    help='One row per patient (default: <data root>/no_phi/patient_summaries.parquet)')
    ap.add_argument('--shard_dir', required=True, help='Checkpoint directory (resumes there)')
    ap.add_argument('--patient_id_col', default='pseudo_mrn')
    ap.add_argument('--date_col', default='date')
    ap.add_argument('--text_col', default='synthetic_note')
    ap.add_argument('--chunk_size', type=int, default=50000, help='Tokens per chunk')
    ap.add_argument('--chunk_overlap', type=int, default=500)
    ap.add_argument('--max_tokens', type=int, default=20000, help='Output tokens per request')
    ap.add_argument('--max_model_len', type=int, default=250000, help='Server context (the servers enforce it)')
    ap.add_argument('--preparation_workers', type=int, default=min(os.cpu_count() or 4, 32),
                    help='Processes that build the prompts')
    ap.add_argument('--model', required=True, help='Served model ID or alias')
    ap.add_argument('--tokenizer-model', help='Tokenizer/model-family ID when --model is a served alias')
    ap.add_argument('--download_dir', help='Hugging Face cache for the tokenizer')
    servers = ap.add_mutually_exclusive_group(required=True)
    servers.add_argument('--server_urls_file', help='Live server registry (rows with url and upstream_concurrency)')
    servers.add_argument('--server_urls', nargs='+', help='Fixed OpenAI-compatible /v1 URLs')
    ap.add_argument('--max_concurrent_per_server', type=int, default=64,
                    help='Per-server limit when a registry row gives no upstream_concurrency')
    ap.add_argument('--scheduler-adaptive-concurrency', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--initial-concurrent-per-server', type=int, default=16)
    ap.add_argument('--pending-requests', type=int, default=8192, help='Accepted for the pipeline; unused')
    ap.add_argument('--concurrency-control-file', help="Accepted for the pipeline; unused (the registry's "
                    'upstream_concurrency is the live per-server limit)')
    ap.add_argument('--request_timeout', type=float, default=7200)
    ap.add_argument('--max_retries', type=int, default=6, help='Transport retries per request')
    ap.add_argument('--max_attempts', type=int, default=16, help='Attempts per chunk before its patient fails')
    ap.add_argument('--api-key-env', default='OPENAI_API_KEY')
    ap.add_argument('--max_patients', type=int, help='The first N patients only (tests)')
    from matchminer_ai.training.summaries.workspace import add_arguments as add_storage_arguments
    add_storage_arguments(ap)
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    if not args.patient_summaries_parquet:
        from matchminer_ai.training.paths import no_phi
        args.patient_summaries_parquet = str(no_phi('patient_summaries.parquet'))
    if args.api_key_env != 'OPENAI_API_KEY':
        os.environ['OPENAI_API_KEY'] = os.environ.get(args.api_key_env, '')
    os.environ.setdefault('OPENAI_API_KEY', 'not-needed')
    if args.download_dir:
        os.environ.setdefault('HF_HUB_CACHE', str(args.download_dir))
    from matchminer_ai.training.summaries.workspace import run_summary
    return run_summary(args, summarize)


if __name__ == '__main__':
    main()
