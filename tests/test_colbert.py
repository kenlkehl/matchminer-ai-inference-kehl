"""Patient isolation, exact late interaction, provenance and explicit persistence."""

import copy
import json
import re
from dataclasses import replace
from threading import RLock

import numpy as np
import pandas as pd
import pytest

from matchminer_ai.cancellation import (
    CancellationToken,
    InferenceCancelled,
    cancellation_scope,
)
from matchminer_ai.patients import (
    ColBERTConfig,
    encode_patient_notes_colbert,
    retrieve_patient_chunks_colbert,
    save_colbert_patient_index,
    load_colbert_patient_index,
)
from matchminer_ai.patients import colbert


class Tokenizer:
    def __call__(self, text, **kwargs):
        spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        return {"offset_mapping": spans, "input_ids": list(range(len(spans)))}


class Encoder:
    tokenizer = Tokenizer()

    def __init__(self):
        self.calls = []

    def encode(self, texts, *, is_query, **kwargs):
        self.calls.append((list(texts), is_query))
        return [
            np.asarray([[1.0, 0.0], [0.0, 1.0]])
            if "scan" in text.lower()
            else np.asarray([[1.0, 1.0]])
            for text in texts
        ]


@pytest.fixture
def encoder(monkeypatch):
    model = Encoder()
    monkeypatch.setattr(
        colbert,
        "_encoder",
        lambda config: (
            model,
            {
                "model": config.model_name,
                "revision": "fixed",
                "settings": config.encoding_settings(),
            },
            RLock(),
        ),
    )
    return model


def sample(config=None):
    return encode_patient_notes_colbert(
        {
            "a": pd.DataFrame(
                [
                    {
                        "note_text": "Unrelated later note.",
                        "note_date": "2026-03-01",
                        "note_type": "Follow-up",
                    },
                    {
                        "note_text": "scan done\nα & β",
                        "note_date": "2026-01-01",
                        "note_type": "Imaging",
                    },
                ]
            ),
            "b": "Other patient private material.",
        },
        config=config or ColBERTConfig(chunk_size=2, batch_size=3),
    )


def test_batched_cohort_encoding_note_spans_and_no_cross_patient_results(encoder):
    config = ColBERTConfig(chunk_size=2, batch_size=3)
    indexes = sample(config)
    a, b = indexes.values()
    assert all(len(texts) <= 3 and not query for texts, query in encoder.calls)
    assert any(
        any("Other patient" in t for t in texts) and any("note." in t for t in texts)
        for texts, _ in encoder.calls
    )  # a batch can span patients, chunks cannot
    assert [note["note_date"][:10] for note in a.notes] == ["2026-01-01", "2026-03-01"]
    for index in indexes.values():
        for chunk in index.chunks:
            note = index.notes[chunk["note_number"] - 1]
            assert note["text"][chunk["start"] : chunk["end"]] == chunk["text"]
            assert chunk["note_date"] == note["note_date"]
        for note in index.notes:
            assert (
                "".join(
                    c["text"]
                    for c in index.chunks
                    if c["note_number"] == note["note_number"]
                )
                == note["text"]
            )
    assert all(c["note_date"] is None for c in b.chunks)
    encoded_calls = len(encoder.calls)
    results = retrieve_patient_chunks_colbert(
        a, ["scan completed?", "unrelated"], top_n=10
    )
    assert len(results) == 2 and len(results[0]) == len(a.chunks)
    assert results[0][0]["text"].startswith("scan")
    assert all("Other patient" not in h["text"] for rows in results for h in rows)
    assert all(query for _, query in encoder.calls[encoded_calls:])
    assert len(encoder.calls[encoded_calls:]) == 1  # questions also batched


def test_true_token_maxsim_not_pooled_cosine_and_negative_scores(encoder):
    index = encode_patient_notes_colbert({"a": "scan done"})["a"]
    chunk = index.chunks[0]
    index = replace(
        index,
        chunks=(chunk, chunk, chunk),
        embeddings=(
            np.eye(2, dtype=np.float32),
            np.asarray([[2**-0.5, 2**-0.5]], dtype=np.float32),
            np.asarray([[-1.0, -1.0]], dtype=np.float32),
        ),
    )
    hits = retrieve_patient_chunks_colbert(index, ["scan"], top_n=3)[0]
    assert [h["chunk_index"] for h in hits] == [0, 1, 2]
    assert [h["score"] for h in hits] == pytest.approx([2, 2**0.5, -2])


def test_table_and_mapping_patient_identity_validation(encoder):
    table = pd.DataFrame(
        [
            {"patient_id": "a", "note_text": "one"},
            {"patient_id": "b", "note_text": "two"},
        ]
    )
    assert list(encode_patient_notes_colbert(table)) == ["a", "b"]
    with pytest.raises(ValueError, match="keys"):
        encode_patient_notes_colbert({"c": table.iloc[:1]})
    with pytest.raises(ValueError, match="patient_id"):
        encode_patient_notes_colbert(table.drop(columns="patient_id"))
    with pytest.raises(ValueError, match="keys"):
        encode_patient_notes_colbert(
            {"a": table.drop(columns="patient_id").assign(patient_id=["a", "b"])}
        )


def test_save_load_without_model_and_stale_source_configuration_detection(
    encoder, tmp_path, monkeypatch
):
    index = sample()["a"]
    path = save_colbert_patient_index(index, tmp_path / "saved.npz")
    assert path.stat().st_mode & 0o777 == 0o600
    monkeypatch.setattr(
        colbert,
        "_encoder",
        lambda c: pytest.fail("disk load must not encode or load a model"),
    )
    loaded = load_colbert_patient_index(
        path,
        patient_id="a",
        expected_config=replace(index.config, device="cpu", batch_size=1),
    )
    assert loaded.chunks == index.chunks and loaded.notes == index.notes
    for a, b in zip(loaded.embeddings, index.embeddings):
        assert np.array_equal(a, b)
    with pytest.raises(ValueError, match="different patient"):
        load_colbert_patient_index(path, patient_id="b")
    with pytest.raises(ValueError, match="does not match"):
        load_colbert_patient_index(path, expected_notes="changed")
    with pytest.raises(ValueError, match="settings differ"):
        load_colbert_patient_index(
            path, expected_config=replace(index.config, chunk_size=3)
        )


def test_saved_index_corruption_is_rejected(encoder, tmp_path):
    path = save_colbert_patient_index(sample()["a"], tmp_path / "saved.npz")
    with np.load(path, allow_pickle=False) as saved:
        arrays = {k: saved[k] for k in saved.files}
    original = copy.deepcopy(arrays)
    arrays["vectors"][0, 0] += 1
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="corrupted"):
        load_colbert_patient_index(path)
    arrays = original
    manifest = json.loads(arrays["manifest"].item())
    manifest["chunks"][0]["note_date"] = "invented"
    arrays["manifest"] = np.asarray(json.dumps(manifest))
    arrays["manifest_sha256"] = np.asarray(colbert._digest(manifest))
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="provenance"):
        load_colbert_patient_index(path)


def test_revision_and_long_questions_are_not_silently_mixed_or_truncated(
    encoder, monkeypatch
):
    index = sample()["a"]
    with pytest.raises(ValueError, match="settings changed"):
        retrieve_patient_chunks_colbert(
            index, ["scan"], config=replace(index.config, model_name="different")
        )
    with pytest.raises(ValueError, match="query_length"):
        retrieve_patient_chunks_colbert(index, ["x " * 600])
    actual = colbert._encoder
    monkeypatch.setattr(
        colbert, "_encoder", lambda c: (encoder, {"revision": "changed"}, RLock())
    )
    with pytest.raises(ValueError, match="revision changed"):
        retrieve_patient_chunks_colbert(index, ["scan"])
    monkeypatch.setattr(colbert, "_encoder", actual)
    with pytest.raises(ValueError, match="top_n"):
        retrieve_patient_chunks_colbert(index, ["scan"], top_n=0)


def test_cancellation_stops_encoding_before_more_batches(encoder):
    token = CancellationToken()
    original = encoder.encode

    def cancelling(*args, **kwargs):
        value = original(*args, **kwargs)
        token.cancel()
        return value

    encoder.encode = cancelling
    with cancellation_scope(token), pytest.raises(InferenceCancelled):
        sample(ColBERTConfig(chunk_size=2, batch_size=1))
    assert len(encoder.calls) == 1


def test_overlap_and_note_metadata_affect_index_identity(encoder):
    notes = pd.DataFrame(
        [{"note_text": "one two three four five", "note_date": "2026-01-01"}]
    )
    index = encode_patient_notes_colbert(
        {"a": notes}, config=ColBERTConfig(chunk_size=3, chunk_overlap=1)
    )["a"]
    assert [c["text"] for c in index.chunks] == ["one two three ", "three four five"]
    assert index.matches(notes)
    assert not index.matches(notes.assign(note_date="2026-01-02"))
    assert not index.matches(notes.assign(note_type="Changed"))


@pytest.mark.parametrize(
    "values",
    [
        {"chunk_size": 0},
        {"chunk_overlap": 64},
        {"batch_size": False},
        {"query_length": 2},
    ],
)
def test_invalid_encoding_settings(values):
    with pytest.raises(ValueError):
        ColBERTConfig(**values)


def test_byte_token_offsets_keep_unicode_characters_whole():
    class ByteTokenizer:
        def __call__(self, text, **kwargs):
            offsets = []
            for i, character in enumerate(text):
                offsets.extend([(i, i + 1)] * len(character.encode("utf-8")))
            return {"offset_mapping": offsets, "input_ids": list(range(len(offsets)))}

    text = "Aα🙂B"
    chunks = colbert._chunks(
        [{"note_number": 1, "note_date": None, "note_type": None, "text": text}],
        ByteTokenizer(),
        ColBERTConfig(chunk_size=2),
    )
    assert all(c["text"] for c in chunks)
    assert "".join(c["text"] for c in chunks) == text
    assert [c["text"] for c in chunks] == ["Aα", "🙂", "B"]
