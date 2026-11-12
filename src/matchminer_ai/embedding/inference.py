"""Embedding-model inference helpers.

``embed_for_matching`` loads and encodes TrialSpace with the functions training mines with
(:mod:`matchminer_ai.embedding.encoder`): the ``embedding`` configuration section maps onto
:class:`~matchminer_ai.embedding.encoder.EncoderSettings`, so the default preset encodes exactly
as training's mining encoded its TrialSpace.
"""

from __future__ import annotations

from dataclasses import asdict
import logging
import os
from functools import lru_cache
from importlib import resources
from importlib.metadata import version
import json
from pathlib import Path
from typing import Any, Dict, cast

import numpy as np
import torch

from matchminer_ai.llm.backends import get_model_metadata
from .encoder import EncoderSettings, _encode_on, encoder_identity, load_sentence_transformer
from .trialspace_contract import COHORT_PROMPT, PATIENT_PROMPT, model_prompts


logger = logging.getLogger(__name__)
_CUDNN_SDPA_FALLBACK_MODELS: set[tuple[str, str]] = set()


def _load_prompt_text(filename: str) -> str:
    """Load a prompt text asset bundled with the package."""
    prompt_path = resources.files("matchminer_ai.prompts").joinpath(filename)
    with prompt_path.open("r", encoding="utf-8") as handle:
        return handle.read()


def _model_path(embedding_config: Dict[str, Any]) -> str:
    """The configured model. A local path must be a readable directory; it never falls back.

    The default is a local, unpublished TrialSpace on purpose: where that directory is
    missing or unreadable, embedding fails instead of using another model's vectors.
    """
    model_path = os.path.expanduser(str(embedding_config.get("model_path") or "").strip())
    if not model_path:
        raise ValueError("embedding.model_path is not set.")
    local = Path(model_path)
    if model_path.startswith(("/", ".")) or local.exists():
        if not (local.is_dir() and os.access(local, os.R_OK | os.X_OK)):
            raise FileNotFoundError(
                f"The TrialSpace model directory {model_path} is missing or unreadable. "
                "The default embedding.model_path is a local, unpublished TrialSpace on "
                "purpose, so embedding stops here; nothing falls back to another model. "
                "Run where that directory is readable, or set embedding.model_path to "
                "another TrialSpace and embed both sides again with it."
            )
    return model_path


def _embedding_runtime(
    embedding_config: Dict[str, Any],
) -> tuple[EncoderSettings, str]:
    """The encoder settings an ``embedding`` section describes, and its fallback query prompt.

    Keys absent from a configuration keep their earlier meaning: the model's own dtype,
    saved prompts and pooling, no autocast, batches of 12. The prompt file is the query
    prompt only for a model that saved none (and no prompt profile gives one). It is
    prepended exactly as written (only the file's final newline is removed), as
    sentence-transformers prepends a model's saved prompts.
    """
    model_path = _model_path(embedding_config)
    device = str(embedding_config.get("device", "cpu")).strip() or "cpu"
    profile = embedding_config.get("prompt_profile")
    settings = EncoderSettings(
        model_path,
        devices=(device,),
        batch_size=int(embedding_config.get("batch_size", 12)),
        max_seq_length=int(embedding_config["max_seq_length"]),
        dtype=embedding_config.get("dtype"),
        autocast=embedding_config.get("autocast"),
        prompts=model_prompts(profile) if profile else {},
        default_prompt_name=embedding_config.get("default_prompt_name"),
        include_prompt_in_pooling=embedding_config.get("include_prompt_in_pooling"),
        text_only=bool(embedding_config.get("text_only", False)),
    )
    prompt_filename = str(embedding_config.get("prompt_file", "")).strip()
    query_prompt = _load_prompt_text(prompt_filename).removesuffix("\n")
    return settings, query_prompt


@lru_cache(maxsize=4)
def _get_embedding_model(settings_key: str, prompt: str):
    """Load and cache a SentenceTransformer exactly as training's encoder loads it.

    The model's saved prompts are the ones it was trained with. A configured prompt profile
    is applied over them as mining applies it (for a TrialSpace from matchminer-ai-training
    it restates them); ``prompt`` (the configured prompt file) only fills in a missing
    ``query`` prompt, as the training evaluation scripts do.
    """
    settings = EncoderSettings(**json.loads(settings_key))
    model = load_sentence_transformer(settings, settings.devices[0])
    model.prompts.setdefault("query", prompt)
    return model


def _model_for(settings: EncoderSettings, prompt: str):
    return _get_embedding_model(json.dumps(asdict(settings), sort_keys=True), prompt)


@lru_cache(maxsize=4)
def _get_embedding_tokenizer(model_path: str):
    """Load and cache the tokenizer for embedding token counts."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def _has_side_prompts(prompts: dict) -> bool:
    return PATIENT_PROMPT in prompts and COHORT_PROMPT in prompts


def _prompt_name(prompts: dict, entity_type: str | None) -> str:
    """The model's prompt for one side, or the shared query prompt."""
    if entity_type is not None and entity_type not in ("patient", "trial"):
        raise ValueError("entity_type must be 'patient', 'trial' or None")
    if entity_type is None or not _has_side_prompts(prompts):
        return "query"
    return PATIENT_PROMPT if entity_type == "patient" else COHORT_PROMPT


@lru_cache(maxsize=8)
def _saved_prompts(model_path: str) -> dict:
    """The prompts a sentence-transformers model saved with its weights, without loading it.

    Reads config_sentence_transformers.json from a local directory or the Hugging Face
    cache (downloading only that file); empty when it cannot be read.
    """
    path = Path(model_path) / "config_sentence_transformers.json"
    try:
        if not path.is_file():
            from huggingface_hub import hf_hub_download, try_to_load_from_cache

            cached = try_to_load_from_cache(model_path, "config_sentence_transformers.json")
            path = Path(cached if isinstance(cached, str) else
                        hf_hub_download(model_path, "config_sentence_transformers.json"))
        return dict(json.loads(path.read_text(encoding="utf-8")).get("prompts") or {})
    except Exception:  # noqa: BLE001 - token counting then uses the configured prompt
        return {}


def _encode(model: Any, texts: list[str], settings: EncoderSettings, prompt_name: str = "query") -> np.ndarray:
    """Normalized float32 vectors as training's encoder makes them (its batches, autocast and
    normalization), prepending the named prompt exactly."""
    return _encode_on(model, texts, prompt_name, settings, settings.devices[0], settings.batch_size)


def _encode_without_cudnn_sdpa(
    model: Any, texts: list[str], settings: EncoderSettings, prompt_name: str = "query"
) -> np.ndarray:
    """Encode with CUDA SDPA implementations that do not use cuDNN."""
    backends = [
        torch.nn.attention.SDPBackend.FLASH_ATTENTION,
        torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
        torch.nn.attention.SDPBackend.MATH,
    ]
    with torch.nn.attention.sdpa_kernel(backends):
        return _encode(model, texts, settings, prompt_name)


def _is_cudnn_sdpa_plan_error(exc: RuntimeError, device: str) -> bool:
    """Return whether a CUDA embedding failed in cuDNN attention planning."""
    message = str(exc).lower()
    return (
        device.partition(":")[0].lower() == "cuda"
        and "cudnn frontend error" in message
        and "no valid execution plans built" in message
    )


def _encode_with_cudnn_sdpa_fallback(
    model: Any,
    texts: list[str],
    settings: EncoderSettings,
    prompt_name: str = "query",
) -> np.ndarray:
    """Prefer the default attention dispatcher, then remember a safe fallback."""
    model_path, device = settings.model, settings.devices[0]
    fallback_key = (model_path, device)
    if fallback_key in _CUDNN_SDPA_FALLBACK_MODELS:
        return _encode_without_cudnn_sdpa(model, texts, settings, prompt_name)

    try:
        return _encode(model, texts, settings, prompt_name)
    except RuntimeError as exc:
        if not _is_cudnn_sdpa_plan_error(exc, device):
            raise
        _CUDNN_SDPA_FALLBACK_MODELS.add(fallback_key)
        logger.warning(
            "cuDNN could not build an attention execution plan for "
            f"{model_path} on {device}; retrying with non-cuDNN CUDA attention "
            "backends.",
        )
        return _encode_without_cudnn_sdpa(model, texts, settings, prompt_name)


def generate_embeddings(
    texts: list[str],
    *,
    embedding_config: Dict[str, Any],
    model_metadata_cache_dir: str | None = None,
    entity_type: str | None = None,
) -> tuple[list[list[float]], Dict[str, Any]]:
    """Generate sentence-transformer embeddings and model metadata.

    ``entity_type`` ("patient" or "trial") selects the model's prompt for that
    side when the model has one; otherwise the shared query prompt is used.
    """
    settings, query_prompt = _embedding_runtime(embedding_config)
    model = _model_for(settings, query_prompt)
    model_metadata = get_model_metadata(
        settings.model,
        cache_dir=model_metadata_cache_dir,
    )
    vectors = _encode_with_cudnn_sdpa_fallback(
        model,
        texts,
        settings,
        prompt_name=_prompt_name(getattr(model, "prompts", None) or {}, entity_type),
    )
    embedding_list = np.asarray(vectors, dtype=np.float32).tolist()
    return cast(list[list[float]], embedding_list), model_metadata


def count_embedding_tokens(
    texts: list[str],
    *,
    embedding_config: Dict[str, Any],
    entity_type: str | None = None,
) -> list[int]:
    """Count embedding-model input tokens of exactly the text the encoder sees: the side's
    prompt (as :func:`generate_embeddings` resolves it) followed directly by the text."""
    settings, query_prompt = _embedding_runtime(embedding_config)
    tokenizer = _get_embedding_tokenizer(settings.model)
    prompts = {"query": query_prompt, **_saved_prompts(settings.model), **settings.prompts}
    prompt = prompts[_prompt_name(prompts, entity_type)]
    prepared = [prompt + text for text in texts]
    encoded = tokenizer(prepared, add_special_tokens=True, truncation=False)
    input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    return [len(ids) for ids in input_ids]


def _loaded_weights_sha256(model):
    """Digest the loaded parameters once per model object."""
    import hashlib

    import torch

    if getattr(model, "_mmai_weights_sha256", None) is None:
        hasher = hashlib.sha256()
        for name, tensor in sorted(model.state_dict().items()):
            data = tensor.detach().to("cpu").contiguous().reshape(-1)
            hasher.update(f"{name}:{data.dtype}:{tuple(tensor.shape)}".encode())
            hasher.update(data.view(torch.uint8).numpy())
        model._mmai_weights_sha256 = hasher.hexdigest()
    return model._mmai_weights_sha256


def _stage_receipt_id(model_path):
    """The training stage receipt that wrote a local model directory, if it has one."""
    try:
        return json.loads((Path(model_path) / "stage_receipt.json").read_text())["receipt_id"]
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _embedding_cache_identity(embedding_config, model_metadata):
    """Identify the already-loaded encoder and how it encodes, not just a model name.

    A local directory can change in place and a Hub model may report no revision; both
    are then named by a digest of the weights actually loaded.
    """
    from matchminer_ai._storage import digest

    settings, prompt = _embedding_runtime(embedding_config)
    model = _model_for(settings, prompt)
    first = model._first_module()
    revision = getattr(
        getattr(getattr(first, "auto_model", None), "config", None),
        "_commit_hash",
        None,
    )
    tokenizer = getattr(first, "tokenizer", None)
    if not hasattr(tokenizer, "backend_tokenizer"):
        return None
    local = Path(settings.model).is_dir()
    # transformers 5 no longer records the resolved commit on the config; the
    # loaded weights themselves then identify the encoder.
    weights = _loaded_weights_sha256(model) if local or not revision else None
    tokenization = json.loads(tokenizer.backend_tokenizer.to_str())
    # These fields are modified by each encode batch; the cutoff is captured below.
    tokenization.pop("padding", None)
    tokenization.pop("truncation", None)
    cuda = settings.devices[0].partition(":")[0].lower() == "cuda"
    return {
        "schema": 2,
        "model_path": settings.model,
        **({"model_stage_receipt_id": _stage_receipt_id(settings.model)} if local else {}),
        "loaded_revision": revision,
        **({"loaded_weights_sha256": weights} if weights else {}),
        "reported_model_metadata": model_metadata,
        "tokenizer_sha256": digest(tokenization),
        "special_tokens": tokenizer.special_tokens_map,
        "prompt": prompt,
        "model_prompts": dict(getattr(model, "prompts", None) or {}),
        "max_seq_length": settings.max_seq_length,
        # Precision, prompts, pooling, batching (the same fields training's mining records).
        "encoder": encoder_identity(settings),
        # Autocast runs on CUDA only: CPU vectors are float32 throughout.
        "effective_autocast": settings.autocast if cuda else None,
        "modules": [
            {"class": type(module).__name__, "config": module.get_config_dict()}
            for module in model._modules.values()
            if hasattr(module, "get_config_dict")
        ],
        "implementation_sha256": digest(Path(__file__).read_bytes()),
        "libraries": {
            name: version(name)
            for name in ("sentence-transformers", "transformers", "torch")
        },
    }


__all__ = [
    "count_embedding_tokens",
    "generate_embeddings",
]
