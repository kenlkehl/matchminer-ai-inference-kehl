"""Checker-model inference helpers."""

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
        padding="max_length",
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
) -> tuple[list[dict[str, Any]], Dict[str, Any]]:
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
    outputs = cast(
        list[dict[str, Any]],
        checker_pipeline(prompts),
    )
    return outputs, model_metadata


def format_checker_prompt_with_ranges(
    template: str,
    components: Sequence[tuple[str, str]],
) -> tuple[str, dict[str, tuple[int, int]]]:
    """Render a checker prompt and retain each source component's char range."""
    markers = [
        f"\x00MMAI_CHECKER_COMPONENT_{index}\x00"
        for index in range(len(components))
    ]
    marked_prompt = template.format(*markers)
    prompt = marked_prompt
    component_ranges: dict[str, tuple[int, int]] = {}

    for marker, (name, text) in zip(markers, components, strict=True):
        if prompt.count(marker) != 1:
            raise ValueError(
                "Checker prompt templates must include each input placeholder "
                "exactly once to support token attribution."
            )
        start = prompt.index(marker)
        prompt = prompt.replace(marker, text, 1)
        component_ranges[name] = (start, start + len(text))

    return prompt, component_ranges


def _checker_model_device(checker_pipeline: Any, embedding_layer: Any) -> Any:
    """Resolve the input device used by a cached Transformers pipeline."""
    weight = getattr(embedding_layer, "weight", None)
    if weight is not None and getattr(weight, "device", None) is not None:
        return weight.device
    return checker_pipeline.device


def _target_logit_index(
    logits: Any,
    *,
    model: Any,
    target_label: str | None,
) -> int:
    """Map a pipeline label back to its logit index, with argmax fallback."""
    if logits.shape[-1] == 1:
        return 0

    normalized_target = str(target_label or "").strip().upper()
    id_to_label = getattr(getattr(model, "config", None), "id2label", {}) or {}
    for raw_index, raw_label in dict(id_to_label).items():
        if str(raw_label).strip().upper() == normalized_target:
            return int(raw_index)
    if normalized_target.startswith("LABEL_"):
        try:
            index = int(normalized_target.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            pass
        else:
            if 0 <= index < int(logits.shape[-1]):
                return index
    return int(logits[0].argmax().item())


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


def attribute_checker_tokens(
    prompts: Sequence[str],
    *,
    component_ranges: Sequence[Mapping[str, tuple[int, int]]],
    checker_config: Dict[str, Any],
    target_labels: Sequence[str | None] | None = None,
    target_directions: Sequence[float] | None = None,
    top_k: int | None = 30,
    model_metadata_cache_dir: str | None = None,
) -> list[dict[str, Any]]:
    """Calculate local gradient-times-input token attributions for checker inputs.

    Attribution requires a fast tokenizer with offset mappings. For a
    single-logit model, ``target_directions`` controls whether the positive or
    negative decision is interpreted. For a multi-class model, ``target_labels``
    selects the prediction logit.
    """
    if len(prompts) != len(component_ranges):
        raise ValueError("component_ranges must contain one mapping per prompt.")
    if target_labels is not None and len(target_labels) != len(prompts):
        raise ValueError("target_labels must contain one value per prompt.")
    if target_directions is not None and len(target_directions) != len(prompts):
        raise ValueError("target_directions must contain one value per prompt.")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be at least 1 or None.")

    import torch

    model_name = checker_config["model_name"]
    device = checker_config["device"]
    max_length = int(checker_config.get("max_length", 4096))
    checker_pipeline = _get_checker_pipeline(
        model_name,
        device,
        max_length,
        model_metadata_cache_dir,
    )
    tokenizer = checker_pipeline.tokenizer
    model = checker_pipeline.model
    embedding_layer = model.get_input_embeddings()
    model_device = _checker_model_device(checker_pipeline, embedding_layer)
    model.eval()

    explanations: list[dict[str, Any]] = []
    for index, (prompt, ranges) in enumerate(
        zip(prompts, component_ranges, strict=True)
    ):
        encoded = tokenizer(
            prompt,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        if "offset_mapping" not in encoded:
            raise ValueError(
                "Checker token attribution requires a fast tokenizer that "
                "returns character offsets."
            )
        raw_offsets = encoded.pop("offset_mapping")[0].tolist()
        offsets = [(int(start), int(end)) for start, end in raw_offsets]
        model_inputs = {
            name: value.to(model_device) for name, value in encoded.items()
        }
        captured_embeddings: list[Any] = []

        def capture_embeddings(_module: Any, _inputs: Any, output: Any) -> None:
            captured_embeddings.append(
                output[0] if isinstance(output, tuple) else output
            )

        hook = embedding_layer.register_forward_hook(capture_embeddings)
        try:
            with torch.enable_grad():
                model_outputs = model(**model_inputs)
                if not captured_embeddings:
                    raise RuntimeError(
                        "Could not capture checker input embeddings for attribution."
                    )
                logits = model_outputs.logits
                if logits.ndim == 1:
                    logits = logits.unsqueeze(0)
                label = target_labels[index] if target_labels is not None else None
                target_index = _target_logit_index(
                    logits,
                    model=model,
                    target_label=label,
                )
                direction = (
                    float(target_directions[index])
                    if target_directions is not None
                    else 1.0
                )
                target_logit = logits[0, target_index]
                if logits.shape[-1] == 1:
                    target_logit = target_logit * direction
                else:
                    competing_logits = torch.cat(
                        [
                            logits[0, :target_index],
                            logits[0, target_index + 1 :],
                        ]
                    )
                    target_logit = target_logit - torch.logsumexp(
                        competing_logits,
                        dim=0,
                    )
                embedding_output = captured_embeddings[-1]
                gradients = torch.autograd.grad(
                    target_logit,
                    embedding_output,
                    retain_graph=False,
                    create_graph=False,
                )[0]
                token_scores_tensor = (gradients * embedding_output).sum(dim=-1)[0]
        finally:
            hook.remove()

        token_scores = [
            float(value)
            for value in token_scores_tensor.detach().float().cpu().tolist()
        ]
        tokens, coverage = _component_token_attributions(
            prompt=prompt,
            offsets=offsets,
            token_scores=token_scores,
            component_ranges=ranges,
            top_k=top_k,
        )
        explanations.append(
            {
                "method": "gradient_x_input",
                "model_target_label": str(label or ""),
                "model_target_index": target_index,
                "token_attributions": tokens,
                "coverage": coverage,
            }
        )

    return explanations


__all__ = [
    "TOKEN_ATTRIBUTION_CAVEAT",
    "attribute_checker_tokens",
    "clear_checker_pipeline_cache",
    "format_checker_prompt_with_ranges",
    "run_checker",
]
