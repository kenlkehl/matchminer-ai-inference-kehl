from __future__ import annotations

import httpx
import pytest

from matchminer_ai import load_default_preset
from matchminer_ai.cli.build_good_option_catalog import (
    catalog_llm_sections,
    read_nct_ids,
)
from matchminer_ai.llm import model_profiles
from matchminer_ai.llm.backends import build_llm_runtime_config
from matchminer_ai.llm.model_profiles import (
    QWEN3_8_FLASH_NEXT_THINKING,
    apply_model_profile,
    configure_served_model,
    resolve_model_profile,
)


def _serve(monkeypatch: pytest.MonkeyPatch, models_by_url: dict[str, list[str]]):
    def fake_get(url: str, **_: object) -> httpx.Response:
        base = url.removesuffix("/models")
        payload = {"data": [{"id": model} for model in models_by_url[base]]}
        return httpx.Response(200, json=payload, request=httpx.Request("GET", url))

    monkeypatch.setattr(model_profiles.httpx, "get", fake_get)


@pytest.mark.parametrize(
    "model_name",
    ["Inferact/Qwen3.8-Flash-Next-NVFP4", "Qwen/Qwen3.8-Flash-Next"],
)
def test_qwen3_8_flash_next_variants_resolve_to_thinking_profile(model_name):
    assert resolve_model_profile(model_name) is QWEN3_8_FLASH_NEXT_THINKING


def test_unregistered_model_has_no_profile():
    assert resolve_model_profile("google/gemma-4-31B-it") is None


def test_profile_replaces_sampling_but_keeps_output_budget():
    config = load_default_preset()
    apply_model_profile(
        config, "Inferact/Qwen3.8-Flash-Next-NVFP4", sections=("llm_good_option",)
    )
    remote = config.llm_good_option["remote"]
    assert remote["model_name"] == "Inferact/Qwen3.8-Flash-Next-NVFP4"
    assert remote["request_params"] == {
        "max_tokens": 100000,
        "temperature": 1.0,
        "top_p": 0.95,
        "presence_penalty": 0.0,
    }
    assert remote["extra_body"]["top_k"] == 20
    assert remote["extra_body"]["min_p"] == 0.0
    assert remote["extra_body"]["repetition_penalty"] == 1.0
    assert remote["extra_body"]["chat_template_kwargs"] == {"enable_thinking": True}
    assert config.llm_good_option["reasoning_parser"] == "qwen3"


def test_unregistered_model_keeps_preset_sampling():
    config = load_default_preset()
    before = dict(config.llm_good_option["remote"]["request_params"])
    assert (
        apply_model_profile(config, "acme/unknown-7b", sections=("llm_good_option",))
        is None
    )
    assert config.llm_good_option["remote"]["model_name"] == "acme/unknown-7b"
    assert config.llm_good_option["remote"]["request_params"] == before


def test_configure_served_model_discovers_and_enables_remote(monkeypatch):
    _serve(monkeypatch, {"http://gpu-host:8001/v1": ["Qwen/Qwen3.8-Flash-Next"]})
    config = load_default_preset()
    model, profile = configure_served_model(
        config, ["gpu-host:8001"], sections=("llm_good_option",)
    )
    assert model == "Qwen/Qwen3.8-Flash-Next"
    assert profile is QWEN3_8_FLASH_NEXT_THINKING
    assert config.remote["enabled"] is True
    assert config.remote["server_urls"] == ["http://gpu-host:8001/v1"]
    runtime = build_llm_runtime_config(
        "llm_good_option", config.llm_good_option, config=config
    )
    assert runtime["model_name"] == "Qwen/Qwen3.8-Flash-Next"
    assert runtime["sampling_params"]["temperature"] == 1.0


def test_configure_served_model_rejects_mismatched_servers(monkeypatch):
    _serve(
        monkeypatch,
        {"http://a:1/v1": ["Qwen/Qwen3.8-Flash-Next"], "http://b:1/v1": ["other"]},
    )
    with pytest.raises(ValueError, match="different models"):
        configure_served_model(
            load_default_preset(), ["a:1", "b:1"], sections=("llm_good_option",)
        )


def test_configure_served_model_rejects_unexpected_model(monkeypatch):
    _serve(monkeypatch, {"http://a:1/v1": ["Qwen/Qwen3.8-Flash-Next"]})
    with pytest.raises(ValueError, match="Requested model"):
        configure_served_model(
            load_default_preset(),
            ["a:1"],
            sections=("llm_good_option",),
            model_name="google/gemma-4-31B-it",
        )


def test_read_nct_ids_skips_comments_and_duplicates(tmp_path):
    path = tmp_path / "ids.txt"
    path.write_text("# header\nNCT00000001\nnct00000002  # note\n\nNCT00000001\n")
    assert read_nct_ids(path) == ["NCT00000001", "NCT00000002"]


def test_thinking_floor_raises_catalog_stage_budgets_only_upward():
    config = load_default_preset()
    sections = catalog_llm_sections(config)
    assert "good_option_catalog.screening_llm" in sections
    assert "good_option_catalog.class_llm" in sections
    apply_model_profile(config, "Qwen/Qwen3.8-Flash-Next", sections=sections)
    floor = QWEN3_8_FLASH_NEXT_THINKING.min_max_tokens
    catalog = config.good_option_catalog
    for stage in ("screening_llm", "class_llm"):
        assert catalog[stage]["remote"]["request_params"]["max_tokens"] == floor
    assert catalog["synthesis_llm"]["remote"]["request_params"]["max_tokens"] == 100000
    assert config.llm_good_option["remote"]["request_params"]["max_tokens"] == 100000
    assert "reasoning_parser" not in catalog["screening_llm"]


def test_discover_served_context_tokens_reads_vllm_max_model_len(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads = {
        "http://vllm:8000/v1/models": {
            "data": [{"id": "served", "max_model_len": 131072}]
        },
        "http://other:8000/v1/models": {"data": [{"id": "served"}]},
    }

    def fake_get(url: str, **_: object) -> httpx.Response:
        return httpx.Response(
            200, json=payloads[url], request=httpx.Request("GET", url)
        )

    monkeypatch.setattr(model_profiles.httpx, "get", fake_get)
    assert model_profiles.discover_served_context_tokens("vllm:8000") == 131072
    assert model_profiles.discover_served_context_tokens("other:8000") is None
