"""The training summarize stage runs the package's summarizer and writes training's serial and per-patient tables."""
import json

import pandas as pd

from matchminer_ai.llm.backends import LLMGenerationResult
from matchminer_ai.llm.prompt_rendering import Prompt
from matchminer_ai.patients.checkpoints import save_prepared_chunks
import matchminer_ai.training.summaries.summarize as stage


class _PromptPool:
    def map(self, func, work_items, chunksize=1):
        return [Prompt(row_idx=item.row_idx, max_tokens=7, prompt_text=f'{item.prior_summary_text}=>{item.chunk_text}',
                       messages=[{'role': 'user', 'content': f'{item.prior_summary_text}=>{item.chunk_text}'}])
                for item in work_items]


class _Backend:
    """Batch-style (rounds) fake: answers depend only on the prompt; 'x' chunks fail."""

    def generate_llm_outputs(self, *, prompt_list, llm_config, model_metadata_cache_dir=None):
        finals, reasons, finishes = [], [], []
        for prompt in prompt_list:
            chunk = prompt.prompt_text.rsplit('=>', 1)[1]
            failed = chunk.startswith('x')
            finals.append('ERROR: request failed' if failed else f'History through {chunk}.\nOther relevant history:\nNone')
            reasons.append('' if failed else f'reasoning for {chunk}')
            finishes.append('error' if failed else 'stop')
        return LLMGenerationResult(final_outputs=finals, reasoning_outputs=reasons, finish_reasons=finishes,
                                   raw_outputs=[], model_metadata={'model_name': 'fake', 'model_sha': 'sha'})


def test_summarize_stage_writes_the_serial_trace_and_final_summaries(tmp_path, monkeypatch):
    monkeypatch.setattr('matchminer_ai.patients.summarize.prep_prompt_pool', lambda patient_config, n_workers: _PromptPool())
    monkeypatch.setattr('matchminer_ai.patients.summarize.shutdown_prompt_pool', lambda pool: None)
    monkeypatch.setattr('matchminer_ai.patients.summarize.get_llm_backend', lambda config: _Backend())
    notes = tmp_path / 'notes.parquet'
    pd.DataFrame([dict(pseudo_mrn=p, date='2025-01-01', synthetic_note='A fabricated note.') for p in ('1', '2')]
                 ).to_parquet(notes, index=False)
    shards = tmp_path / 'shards'
    save_prepared_chunks(shards, pd.DataFrame([
        dict(patient_id=p, chunk_index=i, first_date=f'2025-01-0{i + 1}', last_date=f'2025-01-0{i + 2}', chunk_text=t)
        for p, texts in (('1', ['a', 'b']), ('2', ['x'])) for i, t in enumerate(texts)]))
    registry = tmp_path / 'servers.json'
    registry.write_text(json.dumps({'servers': [{'url': 'http://teacher.example:8000/v1'}]}))
    serial_path, patients_path = tmp_path / 'serial.parquet', tmp_path / 'patients.parquet'
    stage.main(['--input_parquet', str(notes), '--output_parquet', str(serial_path), '--patient_summaries_parquet',
                str(patients_path), '--shard_dir', str(shards), '--server_urls_file', str(registry),
                '--model', 'nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4', '--no-local-summary-storage'])
    serial, patients = pd.read_parquet(serial_path), pd.read_parquet(patients_path)
    assert serial.columns.tolist() == ['pseudo_mrn', 'chunk_index', 'first_date', 'last_date', 'prior_summary',
                                       'reasoning_trace', 'new_summary_reasoning', 'new_summary', 'patient_summary',
                                       'summary_generation_date']
    one = serial[serial.pseudo_mrn == '1']
    assert one.chunk_index.tolist() == [0, 1] and one.prior_summary.iloc[1] == one.new_summary.iloc[0]
    assert one.reasoning_trace.tolist() == ['reasoning for a', 'reasoning for b']
    by_patient = patients.set_index('pseudo_mrn')
    assert by_patient.loc['1', 'patient_summary'].startswith('History through b.')
    assert by_patient.loc['1', 'last_note_date'] == '2025-01-03'
    assert by_patient.loc['2', 'patient_summary'].startswith('ERROR')  # synthetic.prepare refuses it
