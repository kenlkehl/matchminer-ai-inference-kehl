import importlib
import json
from types import SimpleNamespace

import pandas as pd
import pytest

import matchminer_ai.training.synthetic.generate as generation
from test_training_synthetic_generation import HISTORIES


def reference_masking(frame):
    """Pre-optimization behavior, including first-equal duplicate selection."""
    result = frame.loc[frame.event_type.isin(generation.REPORT_TYPES)].copy()
    histories = frame.groupby('patient_id').event_text.apply(list).to_dict()
    masked = []
    for row in result.itertuples():
        history, found = [], False
        for text in histories.get(row.patient_id, []):
            if text == row.event_text and not found:
                text = ('<BEGIN EVENT CORRESPONDING TO SYNTHETIC NOTE> '
                        + f'<{row.event_type}>{text}'
                        + ' <END EVENT CORRESPONDING TO SYNTHETIC NOTE>')
                found = True
            history.append(text)
        masked.append('\n'.join(history))
    result['masked_text'] = masked
    return result


def test_masking_preserves_duplicate_events_interleaving_indexes_and_text():
    frame = pd.DataFrame([
        ('p1', 'demographics', 'Jos\u00e9, age 50.'),
        ('p2', 'diagnosis', 'Different history'),
        ('p1', 'diagnosis', 'Duplicate finding'),
        ('p1', 'pathology_report', 'Duplicate finding'),
        ('p2', 'clinical_note', 'Margins < 1 mm.\nSecond line.'),
        ('p1', 'imaging_report', 'Duplicate finding'),
        ('p1', 'clinical_note', 'Event at the end'),
        (None, 'ngs_report', 'Missing patient ID'),
    ], columns=['patient_id', 'event_type', 'event_text'], index=[3, 1, 3, 8, 1, 9, 2, 7])
    frame['extra_provenance'] = range(len(frame))
    original = frame.copy(deep=True)
    module = importlib.import_module('matchminer_ai.training.synthetic.note_prompts')
    pd.testing.assert_frame_equal(module.create_masked_dataset(frame), reference_masking(frame))
    pd.testing.assert_frame_equal(frame, original)


def test_masking_empty_report_selection_retains_columns():
    frame = pd.DataFrame([dict(patient_id='p', event_type='diagnosis', event_text='Cancer')])
    result = importlib.import_module('matchminer_ai.training.synthetic.note_prompts').create_masked_dataset(frame)
    assert result.empty and list(result.columns) == [*frame.columns, 'masked_text']


def fixture_events():
    spaces = pd.DataFrame({'space_index': [0, 1, 2]})
    events = pd.DataFrame([
        dict(space_index=space, patient_class=kind, nct_id=f'NCT{space}', answer=HISTORIES)
        for space in spaces.space_index for kind in ('negative', 'positive')])
    # Preserve dataframe order within a chunk even when it is not grouped by space.
    events = events.iloc[[2, 5, 0, 3, 1, 4]].reset_index(drop=True)
    return spaces, events


def preparer(directory, workers):
    directory.mkdir(parents=True, exist_ok=True)
    instance = object.__new__(generation.Generator)
    instance.out = directory
    instance.args = SimpleNamespace(spaces_per_note_chunk=2, note_preparation_workers=workers)
    instance.status = {}
    return instance


@pytest.mark.parametrize('workers', [1, 2])
def test_parallel_note_inputs_equal_legacy_and_resume_partial_chunks(tmp_path, workers):
    spaces, events = fixture_events()
    instance = preparer(tmp_path, workers)
    chunks = instance.prepare_notes(spaces, events, 'fixed-run-identity')
    offset = 0
    for index, chunk in enumerate(chunks):
        selected = spaces.iloc[2 * index:2 * index + 2].space_index
        patients = generation.event_rows(events.loc[events.space_index.isin(selected)].to_dict('records'),
                                         'fixed-run-identity')
        expected = reference_masking(patients).reset_index(drop=True)
        expected['row_id'] = range(offset, offset + len(expected))
        expected['key'] = expected.row_id.map(lambda n: f'note:{n}')
        pd.testing.assert_frame_equal(pd.read_parquet(chunk['events']), patients)
        pd.testing.assert_frame_equal(pd.read_parquet(chunk['path']), expected)
        assert chunk['first_row_id'] == offset and chunk['last_row_id'] == offset + len(expected) - 1
        offset += len(expected)
    assert offset == 60
    manifest = json.loads((tmp_path / 'note_input_manifest.json').read_text())
    assert manifest == dict(chunks=chunks, notes=offset)
    from pathlib import Path
    completed = Path(chunks[0]['path'])
    before = completed.stat().st_mtime_ns, completed.read_bytes()
    # Simulate interruption after writing events but before saving note inputs.
    Path(chunks[1]['path']).unlink()
    assert instance.prepare_notes(spaces, events, 'fixed-run-identity') == chunks
    assert (completed.stat().st_mtime_ns, completed.read_bytes()) == before
    status = json.loads((tmp_path / 'status.json').read_text())
    assert status['note_preparation_completed_chunks'] == status['note_preparation_expected_chunks'] == 2
    assert status['prepared_note_inputs'] == offset
    # Changing only worker count does not change saved IDs or rewrite complete chunks.
    instance.args.note_preparation_workers = 3 - workers
    assert instance.prepare_notes(spaces, events, 'fixed-run-identity') == chunks
    assert (completed.stat().st_mtime_ns, completed.read_bytes()) == before


def test_worker_validation_failure_does_not_publish_manifest(tmp_path):
    spaces, events = fixture_events()
    events.loc[events.space_index.eq(2), 'answer'] = '<unknown> Invalid event'
    instance = preparer(tmp_path, 2)
    with pytest.raises(generation.OutputValidationError):
        instance.prepare_notes(spaces, events, 'fixed-run-identity')
    assert not (tmp_path / 'note_input_manifest.json').exists()
    # Other completed chunks are durable and can be reused after a retry.
    assert (tmp_path / 'note_inputs/promptframe_000000.parquet').exists()


def test_resume_rejects_changed_note_count(tmp_path):
    spaces, events = fixture_events()
    instance = preparer(tmp_path, 1)
    chunks = instance.prepare_notes(spaces, events, 'fixed-run-identity')
    frame = pd.read_parquet(chunks[0]['path'])
    frame.iloc[:-1].to_parquet(chunks[0]['path'], index=False)
    with pytest.raises(ValueError, match='Saved note input count changed'):
        instance.prepare_notes(spaces, events, 'fixed-run-identity')


def test_preparation_worker_auto_limit(monkeypatch):
    monkeypatch.setattr(generation.os, 'process_cpu_count', lambda: 72)
    assert generation.note_preparation_workers() == 8
    assert generation.note_preparation_workers(3) == 3
    monkeypatch.setattr(generation.os, 'process_cpu_count', lambda: 2)
    assert generation.note_preparation_workers() == 2
    with pytest.raises(ValueError, match='nonnegative'):
        generation.note_preparation_workers(-1)
