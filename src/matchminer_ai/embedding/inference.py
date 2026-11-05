"""Embedding-model inference helpers."""

from __future__ import annotations

import logging
from functools import lru_cache
from importlib import resources
from importlib.metadata import version
import json
from pathlib import Path
from typing import Any, Dict, cast

import torch

from matchminer_ai.llm.backends import get_model_metadata
from .trialspace_contract import COHORT_PROMPT, PATIENT_PROMPT


logger = logging.getLogger(__name__)
_CUDNN_SDPA_FALLBACK_MODELS: set[tuple[str, str]] = set()


def _load_prompt_text(filename: str) -> str:
    """Load a prompt text asset bundled with the package."""
    prompt_path = resources.files("matchminer_ai.prompts").joinpath(filename)
    with prompt_path.open("r", encoding="utf-8") as handle:
        return handle.read()


def _resolve_embedding_runtime(
    embedding_config: Dict[str, Any],
) -> tuple[str, str, str, int]:
    """Resolve embedding model path, device, query prompt text, and length."""
    model_path = str(embedding_config.get("model_path", "")).strip()
    device = str(embedding_config.get("device", "cpu")).strip() or "cpu"
    prompt_filename = str(embedding_config.get("prompt_file", "")).strip()
    max_seq_length = int(embedding_config["max_seq_length"])
    query_prompt = _load_prompt_text(prompt_filename).strip()
    return model_path, device, query_prompt, max_seq_length


@lru_cache(maxsize=4)
def _get_embedding_model(
    model_path: str,
    device: str,
    prompt: str,
    max_seq_length: int,
):
    """Load and cache a SentenceTransformer embedding model.

    A model that saves its own patient and cohort prompts (TrialSpace trained by
    matchminer-ai-training) keeps them; otherwise ``prompt`` (the configured
    prompt file) is the query prompt for both sides, as TrialSpace-0526 expects.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_path, device=device)
    if not _has_side_prompts(model):
        model.prompts["query"] = prompt
    # SentenceTransformer uses this as the truncation cutoff during encode().
    model.max_seq_length = max_seq_length
    return model


@lru_cache(maxsize=4)
def _get_embedding_tokenizer(model_path: str):
    """Load and cache the tokenizer for embedding token counts."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def _has_side_prompts(model: Any) -> bool:
    prompts = getattr(model, "prompts", None) or {}
    return PATIENT_PROMPT in prompts and COHORT_PROMPT in prompts


def _prompt_name(model: Any, entity_type: str | None) -> str:
    """The model's prompt for one side, or the shared query prompt."""
    if entity_type is not None and entity_type not in ("patient", "trial"):
        raise ValueError("entity_type must be 'patient', 'trial' or None")
    if entity_type is None or not _has_side_prompts(model):
        return "query"
    return PATIENT_PROMPT if entity_type == "patient" else COHORT_PROMPT


def _encode(model: Any, texts: list[str], prompt_name: str = "query") -> Any:
    """Encode normalized embeddings as one tensor.

    A model with side prompts uses the named prompt. Other models (TrialSpace-0526)
    have always been encoded with ``prompt="query"``, which prepends the literal
    text "query" rather than the configured prompt; that is kept so embeddings
    stored by earlier releases stay comparable.
    """
    prompt = dict(prompt_name=prompt_name) if _has_side_prompts(model) else dict(prompt="query")
    return model.encode(
        texts,
        **prompt,
        batch_size=12,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )


def _encode_without_cudnn_sdpa(model: Any, texts: list[str], prompt_name: str = "query") -> Any:
    """Encode with CUDA SDPA implementations that do not use cuDNN."""
    backends = [
        torch.nn.attention.SDPBackend.FLASH_ATTENTION,
        torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
        torch.nn.attention.SDPBackend.MATH,
    ]
    with torch.nn.attention.sdpa_kernel(backends):
        return _encode(model, texts, prompt_name)


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
    *,
    model_path: str,
    device: str,
    prompt_name: str = "query",
) -> Any:
    """Prefer the default attention dispatcher, then remember a safe fallback."""
    fallback_key = (model_path, device)
    if fallback_key in _CUDNN_SDPA_FALLBACK_MODELS:
        return _encode_without_cudnn_sdpa(model, texts, prompt_name)

    try:
        return _encode(model, texts, prompt_name)
    except RuntimeError as exc:
        if not _is_cudnn_sdpa_plan_error(exc, device):
            raise
        _CUDNN_SDPA_FALLBACK_MODELS.add(fallback_key)
        logger.warning(
            "cuDNN could not build an attention execution plan for "
            f"{model_path} on {device}; retrying with non-cuDNN CUDA attention "
            "backends.",
        )
        return _encode_without_cudnn_sdpa(model, texts, prompt_name)


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
    model_path, device, query_prompt, max_seq_length = _resolve_embedding_runtime(
        embedding_config
    )
    model = _get_embedding_model(model_path, device, query_prompt, max_seq_length)
    model_metadata = get_model_metadata(
        model_path,
        cache_dir=model_metadata_cache_dir,
    )
    with torch.no_grad():
        embeddings_tensor = _encode_with_cudnn_sdpa_fallback(
            model,
            texts,
            model_path=model_path,
            device=device,
            prompt_name=_prompt_name(model, entity_type),
        )
    embedding_list = embeddings_tensor.cpu().to(dtype=torch.float32).tolist()
    return cast(list[list[float]], embedding_list), model_metadata


def count_embedding_tokens(
    texts: list[str],
    *,
    embedding_config: Dict[str, Any],
    entity_type: str | None = None,
) -> list[int]:
    """Count embedding-model input tokens after applying the side's prompt."""
    model_path, device, query_prompt, max_seq_length = _resolve_embedding_runtime(
        embedding_config
    )
    tokenizer = _get_embedding_tokenizer(model_path)
    if entity_type is not None:
        model = _get_embedding_model(model_path, device, query_prompt, max_seq_length)
        if _has_side_prompts(model):
            prefix = model.prompts[_prompt_name(model, entity_type)]
            encoded = tokenizer([prefix + text for text in texts], add_special_tokens=True, truncation=False)
            input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
            return [len(ids) for ids in input_ids]
    prepared = [f"{query_prompt} {text}".strip() for text in texts]
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


def _embedding_cache_identity(embedding_config, model_metadata):
    """Identify the already-loaded encoder, not just a mutable Hub model name."""
    from matchminer_ai._storage import digest

    model_path, device, prompt, length = _resolve_embedding_runtime(embedding_config)
    # A local directory can be changed in place without an immutable revision.
    if Path(model_path).exists():
        return None
    model = _get_embedding_model(model_path, device, prompt, length)
    first = model._first_module()
    revision = getattr(
        getattr(getattr(first, "auto_model", None), "config", None),
        "_commit_hash",
        None,
    )
    tokenizer = getattr(first, "tokenizer", None)
    if not hasattr(tokenizer, "backend_tokenizer"):
        return None
    # transformers 5 no longer records the resolved commit on the config; the
    # loaded weights themselves then identify the encoder.
    weights = None if revision else _loaded_weights_sha256(model)
    tokenization = json.loads(tokenizer.backend_tokenizer.to_str())
    # These fields are modified by each encode batch; the cutoff is captured below.
    tokenization.pop("padding", None)
    tokenization.pop("truncation", None)
    return {
        "schema": 1,
        "model_path": model_path,
        "loaded_revision": revision,
        **({"loaded_weights_sha256": weights} if weights else {}),
        "reported_model_metadata": model_metadata,
        "tokenizer_sha256": digest(tokenization),
        "special_tokens": tokenizer.special_tokens_map,
        "prompt": prompt,
        "model_prompts": dict(getattr(model, "prompts", None) or {}),
        "max_seq_length": length,
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
