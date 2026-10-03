"""Local, batched ColBERT note encoding and patient-isolated MaxSim retrieval.

Patient indexes stay in memory unless the caller explicitly
uses save_colbert_patient_index. Model downloads never include patient content.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from threading import RLock

import numpy as np
import pandas as pd

from matchminer_ai.cancellation import check_cancelled

from .workup import _notes


@dataclass(frozen=True)
class ColBERTConfig:
    model_name: str = "lightonai/GTE-ModernColBERT-v1"
    revision: str | None = None
    device: str | None = None
    chunk_size: int = 64
    chunk_overlap: int = 0
    batch_size: int = 32
    query_length: int = 512

    def __post_init__(self):
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("Supply a ColBERT model name or local path.")
        for name, lower, upper in (
            ("chunk_size", 1, 2048),
            ("batch_size", 1, 512),
            ("query_length", 8, 8192),
        ):
            if (
                type(getattr(self, name)) is not int
                or not lower <= getattr(self, name) <= upper
            ):
                raise ValueError(f"{name} must be an integer from {lower} to {upper}.")
        if (
            type(self.chunk_overlap) is not int
            or not 0 <= self.chunk_overlap < self.chunk_size
        ):
            raise ValueError("chunk_overlap must be smaller than chunk_size.")

    def encoding_settings(self):
        return {
            k: v for k, v in asdict(self).items() if k not in {"device", "batch_size"}
        }


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def colbert_notes_fingerprint(notes):
    """Fingerprint normalized note content, order, dates and types (no model load)."""
    return _digest(_notes(notes))


@dataclass(frozen=True)
class ColBERTPatientIndex:
    patient_id: str
    config: ColBERTConfig
    model_signature: dict
    notes: tuple[dict, ...]
    chunks: tuple[dict, ...]
    embeddings: tuple[np.ndarray, ...]
    source_sha256: str

    def matches(self, notes, config=None):
        return self.source_sha256 == colbert_notes_fingerprint(notes) and (
            config is None
            or config.encoding_settings() == self.config.encoding_settings()
        )

    def summary(self):
        return dict(
            model=self.config.model_name,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            chunk_count=len(self.chunks),
            note_count=len(self.notes),
            source_sha256=self.source_sha256,
        )


_model_lock = RLock()


@lru_cache(maxsize=2)
def _cached_encoder(config):
    from ._colbert_encoder import TokenEncoder

    model = TokenEncoder(config)
    return model, model.signature, RLock()


def _encoder(config):
    with _model_lock:
        check_cancelled()
        return _cached_encoder(config)


def _chunks(source, tokenizer, config):
    chunks = []
    for note in source:
        check_cancelled()
        offsets = tokenizer(
            note["text"], add_special_tokens=False, return_offsets_mapping=True
        )["offset_mapping"]
        if not offsets:
            raise ValueError("A note contains no encodable tokens.")
        start = 0
        while start < len(offsets):
            end = min(start + config.chunk_size, len(offsets))
            # Byte-level tokenizers can map several tokens to one Unicode
            # character. Keep that character whole at either chunk boundary.
            while end < len(offsets) and offsets[end][0] < offsets[end - 1][1]:
                end += 1
            left = 0 if start == 0 else offsets[start][0]
            right = len(note["text"]) if end == len(offsets) else offsets[end][0]
            text = note["text"][left:right]
            # Splitting can alter tokenization at a boundary. Never silently truncate.
            if (
                len(tokenizer(text, add_special_tokens=False)["input_ids"])
                > config.chunk_size + 5
            ):
                raise ValueError(
                    "A note boundary exceeds the encoder's document capacity."
                )
            chunks.append(
                dict(
                    note_number=note["note_number"],
                    note_date=note["note_date"],
                    note_type=note["note_type"],
                    start=left,
                    end=right,
                    text=text,
                )
            )
            if end == len(offsets):
                break
            start = max(start + 1, end - config.chunk_overlap)
            while start < end and offsets[start][0] < offsets[start - 1][1]:
                start += 1
    return chunks


def _vectors(values):
    result = []
    for value in values:
        array = np.array(value, dtype=np.float32, copy=True)
        if array.ndim != 2 or not all(array.shape) or not np.isfinite(array).all():
            raise ValueError("ColBERT returned invalid token embeddings.")
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        array /= np.maximum(norms, 1e-12)
        array.setflags(write=False)
        result.append(array)
    if result and len({a.shape[1] for a in result}) != 1:
        raise ValueError("ColBERT embedding dimensions are inconsistent.")
    return result


def encode_patient_notes_colbert(
    patients: Mapping[str, str | pd.DataFrame] | pd.DataFrame,
    *,
    config: ColBERTConfig | None = None,
    progress_callback=None,
) -> dict[str, ColBERTPatientIndex]:
    """Encode a cohort in shared batches, returning a separate index per patient.

    Supply a patient-ID mapping to text/single-patient tables, or a note table
    with a patient_id column. Notes never cross patient/note boundaries. Dates
    are unavailable for free text. No patient files or patient-bearing remote
    requests are made. The public model may be downloaded to the model cache.
    """
    config = config or ColBERTConfig()
    if not isinstance(config, ColBERTConfig):
        raise TypeError("config must be a ColBERTConfig.")
    if isinstance(patients, pd.DataFrame):
        if "patient_id" not in patients or patients.patient_id.isna().any():
            raise ValueError("A cohort table requires nonmissing patient_id values.")
        patients = dict(tuple(patients.groupby("patient_id", sort=False)))
    if not isinstance(patients, Mapping) or not patients:
        raise ValueError("Supply a nonempty mapping or cohort note table.")
    if any(not isinstance(key, str) or not key.strip() for key in patients):
        raise ValueError("Patient IDs must be nonempty strings.")
    sources = {}
    for key, value in patients.items():
        if isinstance(value, pd.DataFrame) and "patient_id" in value:
            if not value.patient_id.eq(key).all():
                raise ValueError("Mapping keys must match the notes' patient_id.")
        sources[key] = _notes(value)
    progress = progress_callback or (lambda _: None)
    progress("Loading local ColBERT encoder")
    model, signature, lock = _encoder(config)
    with lock:
        chunks = {
            key: _chunks(source, model.tokenizer, config)
            for key, source in sources.items()
        }
        flattened = [chunk["text"] for rows in chunks.values() for chunk in rows]
        encoded = []
        for start in range(0, len(flattened), config.batch_size):
            check_cancelled()
            batch = flattened[start : start + config.batch_size]
            values = model.encode(
                batch,
                is_query=False,
                batch_size=config.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            if len(values) != len(batch):
                raise ValueError(
                    "ColBERT returned the wrong number of chunk embeddings."
                )
            encoded.extend(_vectors(values))
            progress(
                f"Encoded {len(encoded)}/{len(flattened)} patient chunks with ColBERT"
            )
    check_cancelled()
    result, offset = {}, 0
    for key, rows in chunks.items():
        result[key] = ColBERTPatientIndex(
            key,
            config,
            copy.deepcopy(signature),
            tuple(sources[key]),
            tuple(rows),
            tuple(encoded[offset : offset + len(rows)]),
            _digest(sources[key]),
        )
        offset += len(rows)
    return result


def retrieve_patient_chunks_colbert(
    index: ColBERTPatientIndex,
    questions: list[str],
    *,
    top_n: int = 20,
    config: ColBERTConfig | None = None,
    progress_callback=None,
) -> list[list[dict]]:
    """Exact token-level sum-of-maxima cosine retrieval within one patient's index.

    Returns one ranked list per question with original text, spans, note dates,
    chunk_index, rank and score. Ties retain source order. Questions exceeding
    query_length are rejected rather than silently truncated.
    """
    if not isinstance(index, ColBERTPatientIndex):
        raise TypeError("index must be a ColBERTPatientIndex.")
    if type(top_n) is not int or not 1 <= top_n <= 1000:
        raise ValueError("top_n must be an integer from 1 to 1000.")
    if (
        not isinstance(questions, list)
        or not questions
        or any(not isinstance(q, str) or not q.strip() for q in questions)
    ):
        raise ValueError("Supply a nonempty list of questions.")
    config = config or index.config
    if config.encoding_settings() != index.config.encoding_settings():
        raise ValueError("ColBERT settings changed; re-encode the notes.")
    model, signature, lock = _encoder(config)
    if signature != index.model_signature:
        raise ValueError(
            "ColBERT model/tokenizer revision changed; re-encode the notes."
        )
    progress = progress_callback or (lambda _: None)
    progress(f"Encoding {len(questions)} workup questions with ColBERT")
    queries = []
    with lock:
        for question in questions:
            if (
                len(model.tokenizer(question, add_special_tokens=False)["input_ids"])
                + 3
                > config.query_length
            ):
                raise ValueError(
                    "Question exceeds ColBERT query_length; increase it and re-encode."
                )
        for start in range(0, len(questions), config.batch_size):
            check_cancelled()
            batch = questions[start : start + config.batch_size]
            values = model.encode(
                batch,
                is_query=True,
                batch_size=config.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            if len(values) != len(batch):
                raise ValueError(
                    "ColBERT returned the wrong number of question embeddings."
                )
            queries.extend(_vectors(values))
    results = []
    for number, query in enumerate(queries, 1):
        scores = []
        first = 0
        while first < len(index.embeddings):
            check_cancelled()
            last, tokens = first, 0
            # Bound the temporary similarity matrix, while combining short
            # chunks into efficient matrix multiplications. Never pad maxima.
            while last < len(index.embeddings):
                length = len(index.embeddings[last])
                if last > first and (tokens + length) * len(query) > 1_048_576:
                    break
                tokens += length
                last += 1
            documents = index.embeddings[first:last]
            if any(query.shape[1] != document.shape[1] for document in documents):
                raise ValueError("ColBERT query/document embedding dimensions differ.")
            similarities = query @ np.concatenate(documents).T
            offset = 0
            for document in documents:
                scores.append(
                    float(
                        similarities[:, offset : offset + len(document)]
                        .max(axis=1)
                        .sum()
                    )
                )
                offset += len(document)
            first = last
        selected = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:top_n]
        results.append(
            [
                {**index.chunks[i], "chunk_index": i, "rank": rank, "score": scores[i]}
                for rank, i in enumerate(selected, 1)
            ]
        )
        progress(f"Retrieved ColBERT chunks for question {number}/{len(questions)}")
    return results


def save_colbert_patient_index(index: ColBERTPatientIndex, path) -> Path:
    """Explicitly save sensitive text + vectors locally; never called by the demo.

    Atomic, owner-readable NPZ storage, with JSON metadata and no pickle objects.
    The caller controls the destination and its retention policy.
    """
    path = Path(path)
    manifest = dict(
        version=1,
        patient_id=index.patient_id,
        config=asdict(index.config),
        model_signature=index.model_signature,
        notes=index.notes,
        chunks=index.chunks,
        source_sha256=index.source_sha256,
    )
    vectors = np.concatenate(index.embeddings)
    offsets = np.cumsum(
        [0] + [len(value) for value in index.embeddings], dtype=np.int64
    )
    manifest["vectors_sha256"] = hashlib.sha256(
        vectors.tobytes() + offsets.tobytes()
    ).hexdigest()
    descriptor, temp = tempfile.mkstemp(
        prefix=".colbert-", suffix=".npz", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(
                handle,
                manifest=np.asarray(json.dumps(manifest)),
                manifest_sha256=np.asarray(_digest(manifest)),
                vectors=vectors,
                offsets=offsets,
            )
        check_cancelled()
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    return path


def load_colbert_patient_index(
    path, *, expected_config=None, expected_notes=None, patient_id=None
) -> ColBERTPatientIndex:
    """Load without a model or re-encoding; check model compatibility at retrieval.

    Optional expected_notes/config/patient_id reject a stale or wrong index.
    Never loads Python pickle objects.
    """
    with np.load(path, allow_pickle=False) as saved:
        manifest = json.loads(str(saved["manifest"].item()))
        if _digest(manifest) != str(saved["manifest_sha256"].item()):
            raise ValueError("Invalid or corrupted ColBERT index metadata.")
        vectors, offsets = saved["vectors"], saved["offsets"]
    if manifest.get("version") != 1:
        raise ValueError("Unsupported ColBERT index format.")
    config = ColBERTConfig(**manifest["config"])
    notes, chunks = manifest["notes"], manifest["chunks"]
    if (
        vectors.dtype != np.float32
        or vectors.ndim != 2
        or not all(vectors.shape)
        or not np.isfinite(vectors).all()
        or offsets.dtype != np.int64
        or offsets.ndim != 1
        or len(offsets) != len(chunks) + 1
        or offsets[0] != 0
        or offsets[-1] != len(vectors)
        or (np.diff(offsets) <= 0).any()
        or not chunks
        or not notes
        or _digest(notes) != manifest["source_sha256"]
        or hashlib.sha256(vectors.tobytes() + offsets.tobytes()).hexdigest()
        != manifest["vectors_sha256"]
    ):
        raise ValueError("Invalid or corrupted ColBERT index.")
    for chunk in chunks:
        number, start, end = chunk["note_number"], chunk["start"], chunk["end"]
        if (
            type(number) is not int
            or not 1 <= number <= len(notes)
            or type(start) is not int
            or type(end) is not int
            or not 0 <= start < end <= len(notes[number - 1]["text"])
        ):
            raise ValueError("Invalid ColBERT note span.")
        note = notes[number - 1]
        if chunk["text"] != note["text"][start:end] or any(
            chunk[k] != note[k] for k in ("note_number", "note_date", "note_type")
        ):
            raise ValueError(
                "ColBERT chunk provenance does not match the original note."
            )
    vectors.setflags(write=False)
    index = ColBERTPatientIndex(
        manifest["patient_id"],
        config,
        manifest["model_signature"],
        tuple(notes),
        tuple(chunks),
        tuple(
            vectors[left:right]
            for left, right in zip(offsets[:-1], offsets[1:], strict=True)
        ),
        manifest["source_sha256"],
    )
    if patient_id is not None and index.patient_id != patient_id:
        raise ValueError("The ColBERT index belongs to a different patient.")
    if (
        expected_config is not None
        and config.encoding_settings() != expected_config.encoding_settings()
    ):
        raise ValueError("The saved ColBERT encoding settings differ.")
    if expected_notes is not None and not index.matches(expected_notes):
        raise ValueError("The saved ColBERT index does not match these notes.")
    return index
