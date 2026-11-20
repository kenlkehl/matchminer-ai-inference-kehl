"""TrialSpace vectors: pick the latest valid mining round, prove its rows are the corpus rows, and search.

A round ``trialspace_round<N>/mining/`` (training's ``trialspace_mine.py``) holds ``patients.npy`` and
``cohorts.npy`` in corpus order, a ``manifest.json`` naming the encoder (``encoder_identity``) and corpus,
and a ``stage_receipt.json`` written last. Rounds are tried from the highest number down; a round is used
only when every check in :func:`validate_round` passes, including that every output its receipt lists (and
every file of the fitted TrialSpace that made it) is present at its recorded size: a round being copied back is
skipped until the copy is whole. Vectors from different rounds are never combined: a query is always the round's
own row, or a text embedded by an encoder whose identity equals the round's.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re

import numpy as np

RECEIPT_FORMAT = "mmai-stage-receipt-v1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class RoundRejected(Exception):
    pass


@dataclass
class ModelRef:
    """Where the encoder of a round can be loaded: a Hub id at a revision, or a fitted model directory."""
    kind: str  # "hub" or "local"
    model: str  # Hub id or directory
    revision: str | None
    identity_model: str  # the manifest's encoder "model" (Hub id or the fitted model's receipt id)


@dataclass
class RoundInfo:
    number: int
    directory: Path
    manifest: dict
    receipt: dict
    model: ModelRef
    shapes: dict = field(default_factory=dict)

    @property
    def encoder(self) -> dict:
        return self.manifest["encoder"]

    def describe(self) -> dict:
        return {"round": self.number, "directory": str(self.directory), "receipt_id": self.receipt.get("receipt_id"),
                "finished_at": (self.receipt.get("provenance") or {}).get("finished_at"),
                "encoder": self.encoder, "model": self.model.__dict__, "shapes": self.shapes,
                "dimension": self.manifest.get("dimension"), "corpus": self.manifest.get("corpus")}


def discover_rounds(trialspace_dir: Path) -> list[int]:
    """Round numbers whose mining directory has a stage receipt, highest first."""
    numbers = []
    for path in Path(trialspace_dir).glob("trialspace_round*"):
        match = re.fullmatch(r"trialspace_round(\d+)", path.name)
        if match and (path / "mining" / "stage_receipt.json").exists():
            numbers.append(int(match.group(1)))
    return sorted(numbers, reverse=True)


def npy_shape(path: Path) -> tuple[tuple[int, ...], str]:
    """A .npy file's shape and dtype from its header alone."""
    with open(path, "rb") as stream:
        version = np.lib.format.read_magic(stream)
        shape, _, dtype = (np.lib.format.read_array_header_1_0(stream) if version == (1, 0)
                           else np.lib.format.read_array_header_2_0(stream))
    return tuple(shape), str(dtype)


def incomplete_outputs(directory: Path, receipt: dict) -> list[str]:
    """The outputs a stage receipt lists that are missing or of another size (e.g. still being copied)."""
    return [name for name, size in (receipt.get("outputs") or {}).items()
            if not (Path(directory) / name).is_file() or (Path(directory) / name).stat().st_size != size]


def resolve_model(manifest_encoder: dict, trialspace_dir: Path, number: int) -> ModelRef:
    """The encoder that made a round's vectors: the base Hub model (round 1) or a fitted TrialSpace whose
    stage receipt id the manifest names (later rounds, ``models/trialspace_round<N-1>``)."""
    model = str(manifest_encoder.get("model"))
    if not HEX64.match(model):
        return ModelRef("hub", model, manifest_encoder.get("revision"), model)
    models = Path(trialspace_dir) / "models"
    candidates = [models / f"trialspace_round{number - 1}"] + sorted(models.glob("trialspace_round*"))
    for directory in candidates:
        path = directory / "stage_receipt.json"
        receipt = json.loads(path.read_text()) if path.exists() else {}
        if receipt.get("receipt_id") == model:
            missing = incomplete_outputs(directory, receipt)
            if missing:
                raise RoundRejected(f"fitted TrialSpace {directory.name} is incomplete (still being copied?): "
                                    f"{len(missing)} file(s) missing or a different size from its stage receipt, "
                                    f"e.g. {missing[0]}")
            return ModelRef("local", str(directory), None, model)
    raise RoundRejected(f"encoder model {model[:12]}... is a fitted TrialSpace whose directory (stage receipt) "
                        f"was not found under {models}")


def validate_round(number: int, trialspace_dir: Path, corpus) -> RoundInfo:
    directory = Path(trialspace_dir) / f"trialspace_round{number}" / "mining"
    receipt = json.loads((directory / "stage_receipt.json").read_text())
    if receipt.get("format") != RECEIPT_FORMAT or receipt.get("kind") != "mine":
        raise RoundRejected("stage receipt is not a complete mine receipt")
    missing = incomplete_outputs(directory, receipt)
    if missing:
        raise RoundRejected(f"{len(missing)} output(s) missing or a different size from the stage receipt (still "
                            f"being copied?), e.g. {missing[0]}")
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise RoundRejected("manifest.json missing")
    manifest = json.loads(manifest_path.read_text())
    identity = receipt.get("identity") or {}
    if manifest.get("encoder") != identity.get("encoder"):
        raise RoundRejected("manifest encoder differs from the stage receipt's encoder")
    if manifest.get("corpus") != identity.get("corpus") or manifest.get("corpus") != corpus.receipt_id:
        raise RoundRejected(f"made from corpus {str(manifest.get('corpus'))[:12]}..., but the loaded corpus is "
                            f"{str(corpus.receipt_id)[:12]}...")
    shapes = {}
    for name, rows in (("patients.npy", len(corpus.patients)), ("cohorts.npy", len(corpus.cohorts))):
        path = directory / name
        if not path.exists():
            raise RoundRejected(f"{name} missing")
        shape, dtype = npy_shape(path)
        if len(shape) != 2 or shape[0] != rows or shape[1] != manifest.get("dimension"):
            raise RoundRejected(f"{name} has shape {shape}; expected ({rows}, {manifest.get('dimension')})")
        if dtype not in ("<f4", "float32"):
            raise RoundRejected(f"{name} has dtype {dtype}; expected float32")
        shapes[name] = list(shape)
    if corpus.digests:
        for side in ("patients", "cohorts"):
            if manifest.get(side) and manifest[side] != corpus.digests[side]:
                raise RoundRejected(f"the corpus {side} rows differ from the rows this round encoded (digest)")
    model = resolve_model(manifest["encoder"], trialspace_dir, number)
    return RoundInfo(number, directory, manifest, receipt, model, shapes)


def select_round(trialspace_dir: Path, corpus, above: int = 0) -> tuple[RoundInfo | None, list[dict]]:
    """The highest valid round numbered above ``above``, and the reasons any higher rounds were skipped."""
    skipped = []
    for number in (n for n in discover_rounds(trialspace_dir) if n > above):
        try:
            return validate_round(number, trialspace_dir, corpus), skipped
        except (RoundRejected, OSError, ValueError, KeyError) as exc:
            skipped.append({"round": number, "reason": f"{type(exc).__name__}: {exc}" if not isinstance(
                exc, RoundRejected) else str(exc)})
    return None, skipped


def encoder_settings(info: RoundInfo, *, cache_dir: str | None, device: str = "cpu"):
    """``EncoderSettings`` rebuilt from the round's manifest, and whether its identity equals the manifest's.

    Every field of ``encoder_identity`` is copied from the manifest except the model, which is the Hub id or
    the fitted model's directory (named in the identity by its receipt id). The identity also names the
    installed package's encoder implementation, so a changed implementation is refused too.
    """
    from matchminer_ai.embedding.encoder import EncoderSettings, encoder_identity

    enc = info.encoder
    settings = EncoderSettings(
        info.model.model, revision=enc.get("revision"), cache_dir=cache_dir, devices=(device,),
        batch_size=int(enc.get("batch_size") or 32), max_seq_length=enc.get("max_seq_length"),
        dtype=enc.get("dtype"), autocast=enc.get("autocast"), prompts=dict(enc.get("prompts") or {}),
        default_prompt_name=enc.get("default_prompt_name"), include_prompt_in_pooling=enc.get("include_prompt_in_pooling"),
        text_only=bool(enc.get("text_only")), model_identity=info.model.identity_model if info.model.kind == "local" else None,
        backend=enc.get("backend", "sentence_transformers"))
    return settings, encoder_identity(settings) == enc


def find_hub_cache(caches, model: str, revision: str | None) -> Path | None:
    """The first Hugging Face cache holding the model's weights at this revision."""
    folder = "models--" + model.replace("/", "--")
    for cache in caches:
        snapshots = Path(cache) / folder / "snapshots"
        targets = [snapshots / revision] if revision else sorted(snapshots.glob("*"))
        for snapshot in targets:
            if any((snapshot / name).exists() for name in ("model.safetensors", "pytorch_model.bin",
                                                            "model.safetensors.index.json")):
                return Path(cache)
    return None


class VectorStore:
    """The round's normalized vectors in memory, searched exactly (cosine = dot product)."""

    def __init__(self, info: RoundInfo):
        self.info = info
        self.patients = np.load(info.directory / "patients.npy", allow_pickle=False)
        self.cohorts = np.load(info.directory / "cohorts.npy", allow_pickle=False)

    def side(self, name: str) -> np.ndarray:
        return self.patients if name == "patients" else self.cohorts

    def scores(self, query: np.ndarray, side: str) -> np.ndarray:
        from matchminer_ai.matching.retrieval import score_batches

        query = np.asarray(query, dtype=np.float32).reshape(1, -1)
        documents = self.side(side)
        if query.shape[1] != documents.shape[1]:
            raise ValueError(f"query dimension {query.shape[1]} differs from the round's {documents.shape[1]}")
        _, scores = next(score_batches(query, documents, batch_size=1, devices=("cpu",)))
        return scores[0]

    def search(self, query: np.ndarray, side: str, k: int, allowed: np.ndarray | None = None,
               group: list | None = None) -> list[tuple[int, float]]:
        """Top-k rows (index, cosine), ties in row order; ``allowed`` masks rows, ``group`` keeps the best row
        per group key (e.g. one cohort per trial)."""
        from matchminer_ai.matching.retrieval import stable_top_k

        scores = self.scores(query, side).astype(np.float64)
        if allowed is not None:
            scores = np.where(allowed, scores, -np.inf)
        candidates = int(np.isfinite(scores).sum())
        if group is None:
            order = stable_top_k(scores, min(k, candidates))
        else:
            order, seen = [], set()
            for i in stable_top_k(scores, candidates):
                if group[i] in seen:
                    continue
                seen.add(group[i])
                order.append(i)
                if len(order) == k:
                    break
        return [(int(i), float(scores[i])) for i in order]
