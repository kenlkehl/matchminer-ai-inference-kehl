"""Embedding-model inference helpers."""

from __future__ import annotations

import logging
from functools import lru_cache
from importlib import resources
from typing import Any, Dict, cast

from matchminer_ai.llm.backends import get_model_metadata


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
    """Load and cache a SentenceTransformer embedding model."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_path, device=device)
    model.prompts["query"] = prompt
    # SentenceTransformer uses this as the truncation cutoff during encode().
    model.max_seq_length = max_seq_length
    return model


@lru_cache(maxsize=4)
def _get_embedding_tokenizer(model_path: str):
    """Load and cache the tokenizer for embedding token counts."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def _encode_without_cudnn_sdpa(model: Any, texts: list[str]) -> Any:
    """Encode with CUDA SDPA implementations that do not use cuDNN."""
    import torch

    backends = [
        torch.nn.attention.SDPBackend.FLASH_ATTENTION,
        torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
        torch.nn.attention.SDPBackend.MATH,
    ]
    with torch.nn.attention.sdpa_kernel(backends):
        return model.encode(texts, prompt="query")


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
) -> Any:
    """Prefer the default attention dispatcher, then remember a safe fallback."""
    fallback_key = (model_path, device)
    if fallback_key in _CUDNN_SDPA_FALLBACK_MODELS:
        return _encode_without_cudnn_sdpa(model, texts)

    try:
        return model.encode(texts, prompt="query")
    except RuntimeError as exc:
        if not _is_cudnn_sdpa_plan_error(exc, device):
            raise
        _CUDNN_SDPA_FALLBACK_MODELS.add(fallback_key)
        logger.warning(
            "cuDNN could not build an attention execution plan for "
            f"{model_path} on {device}; retrying with non-cuDNN CUDA attention "
            "backends.",
        )
        return _encode_without_cudnn_sdpa(model, texts)


def generate_embeddings(
    texts: list[str],
    *,
    embedding_config: Dict[str, Any],
    model_metadata_cache_dir: str | None = None,
) -> tuple[list[list[float]], Dict[str, Any]]:
    """Generate sentence-transformer embeddings and model metadata."""
    model_path, device, query_prompt, max_seq_length = _resolve_embedding_runtime(
        embedding_config
    )
    model = _get_embedding_model(model_path, device, query_prompt, max_seq_length)
    model_metadata = get_model_metadata(
        model_path,
        cache_dir=model_metadata_cache_dir,
    )
    embeddings = _encode_with_cudnn_sdpa_fallback(
        model,
        texts,
        model_path=model_path,
        device=device,
    )
    embedding_list = (
        embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings
    )
    return cast(list[list[float]], embedding_list), model_metadata


def count_embedding_tokens(
    texts: list[str],
    *,
    embedding_config: Dict[str, Any],
) -> list[int]:
    """Count embedding-model input tokens after applying the query prompt."""
    model_path, _device, query_prompt, _max_seq_length = _resolve_embedding_runtime(
        embedding_config
    )
    tokenizer = _get_embedding_tokenizer(model_path)
    prepared = [f"{query_prompt} {text}".strip() for text in texts]
    encoded = tokenizer(prepared, add_special_tokens=True, truncation=False)
    input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    return [len(ids) for ids in input_ids]


__all__ = [
    "count_embedding_tokens",
    "generate_embeddings",
]
