"""Text-classification checker helpers and token-attribution utilities.

``run_checker`` serves only the deprecated GoodOptionChecker classifier
(:func:`~matchminer_ai.matching.good_options.score_good_options`). The match-quality,
boilerplate and per-criterion checks use the decision model
(:mod:`matchminer_ai.matching.decision_model`), whose token attributions reuse
:data:`TOKEN_ATTRIBUTION_CAVEAT` and :func:`_component_token_attributions`.
"""

from __future__ import annotations

import gc
from functools import lru_cache
from typing import Any, Dict, Mapping, Sequence, cast

from matchminer_ai.llm.backends import get_model_metadata

TOKEN_ATTRIBUTION_CAVEAT = (
    "Local gradient attribution measures the model output's sensitivity to input "
    "token embeddings. It is not a clinical rationale, eligibility evidence, or "
    "proof of causation, and correlated tokens can receive unstable scores."
)


@lru_cache(maxsize=2)
def _get_checker_pipeline(
    model_name: str,
    device: str,
    max_length: int,
    model_metadata_cache_dir: str | None,
):
    """Load and cache a Transformers text-classification checker pipeline."""
    from transformers import AutoTokenizer, pipeline

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=model_metadata_cache_dir,
        trust_remote_code=True,
    )
    return pipeline(
        "text-classification",
        model_name,
        tokenizer=tokenizer,
        truncation=True,
        padding=True,  # to the longest input of each batch, not to max_length
        max_length=max_length,
        device=device,
    )


def clear_checker_pipeline_cache() -> None:
    """Release cached checker pipeline handles and clear Python/GPU caches."""
    _get_checker_pipeline.cache_clear()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def run_checker(
    prompts: list[str],
    *,
    checker_config: Dict[str, Any],
    model_metadata_cache_dir: str | None = None,
    return_all_scores: bool = False,
) -> tuple[list[Any], Dict[str, Any]]:
    """Run a text-classification checker model on prompts."""
    model_name = checker_config["model_name"]
    device = checker_config["device"]
    max_length = int(checker_config.get("max_length", 4096))
    checker_pipeline = _get_checker_pipeline(
        model_name,
        device,
        max_length,
        model_metadata_cache_dir,
    )
    model_metadata = get_model_metadata(
        model_name,
        cache_dir=model_metadata_cache_dir,
    )
    pipeline_kwargs = {"top_k": None} if return_all_scores else {}
    pipeline_kwargs["batch_size"] = int(checker_config.get("batch_size", 8))
    outputs = cast(list[Any], checker_pipeline(prompts, **pipeline_kwargs))
    return outputs, model_metadata


def _component_token_attributions(
    *,
    prompt: str,
    offsets: list[tuple[int, int]],
    token_scores: list[float],
    component_ranges: Mapping[str, tuple[int, int]],
    top_k: int | None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    """Map prompt-token attribution values back to original input fields."""
    component_tokens: dict[str, list[dict[str, Any]]] = {
        name: [] for name in component_ranges
    }
    coverage: dict[str, dict[str, Any]] = {}

    for name, (component_start, component_end) in component_ranges.items():
        source_text = prompt[component_start:component_end]
        for (token_start, token_end), raw_score in zip(
            offsets,
            token_scores,
            strict=True,
        ):
            overlap_start = max(token_start, component_start)
            overlap_end = min(token_end, component_end)
            if overlap_start >= overlap_end:
                continue
            relative_start = overlap_start - component_start
            relative_end = overlap_end - component_start
            component_tokens[name].append(
                {
                    "text": source_text[relative_start:relative_end],
                    "start": relative_start,
                    "end": relative_end,
                    "raw_score": float(raw_score),
                }
            )

        tokens = component_tokens[name]
        covered_character_end = max(
            (int(token["end"]) for token in tokens),
            default=0,
        )
        non_whitespace_end = max(
            (index + 1 for index, char in enumerate(source_text) if not char.isspace()),
            default=0,
        )
        coverage[name] = {
            "source_character_count": len(source_text),
            "retained_token_count": len(tokens),
            "covered_character_end": covered_character_end,
            "truncated": covered_character_end < non_whitespace_end,
        }

    all_scores = [
        abs(float(token["raw_score"]))
        for tokens in component_tokens.values()
        for token in tokens
    ]
    scale = max(all_scores, default=0.0)
    for name, tokens in component_tokens.items():
        for token in tokens:
            normalized_score = (
                float(token["raw_score"]) / scale if scale > 0.0 else 0.0
            )
            token["normalized_score"] = normalized_score
            token["importance"] = abs(normalized_score)
            token["direction"] = (
                "supports_prediction"
                if normalized_score > 0.0
                else "opposes_prediction"
                if normalized_score < 0.0
                else "neutral"
            )
        if top_k is not None and len(tokens) > top_k:
            tokens = sorted(
                tokens,
                key=lambda item: abs(float(item["raw_score"])),
                reverse=True,
            )[:top_k]
        component_tokens[name] = sorted(
            tokens,
            key=lambda item: (int(item["start"]), int(item["end"])),
        )

    return component_tokens, coverage


__all__ = [
    "TOKEN_ATTRIBUTION_CAVEAT",
    "clear_checker_pipeline_cache",
    "run_checker",
]
