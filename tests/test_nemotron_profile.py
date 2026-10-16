import pytest

from matchminer_ai import load_default_preset
from matchminer_ai.llm.model_profiles import apply_model_profile, resolve_model_profile
from matchminer_ai.llm.reasoning import (
    resolve_reasoning_parser,
    resolve_tool_call_parser,
)
from matchminer_ai.llm.sampling import remote_sampling, vendor_defaults
from matchminer_ai.llm.vllm_server import build_vllm_server_command

MODEL = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4"


@pytest.mark.parametrize(
    "model", [MODEL, MODEL.replace("NVFP4", "BF16"), "nemotron-3.5-super-vl"]
)
def test_nemotron_profile_and_parsers(model):
    profile = resolve_model_profile(model)
    assert profile.name == "nemotron-3.5-super-vl"
    assert resolve_reasoning_parser(model) == "nemotron_v3"
    assert resolve_tool_call_parser(model) == "qwen3_coder"
    assert resolve_reasoning_parser(model, "custom") == "custom"
    assert resolve_tool_call_parser(model, "none") is None
    defaults, template = vendor_defaults(model)
    assert defaults == {
        "temperature": 1.0,
        "top_p": 0.95,
        "presence_penalty": 0.0,
        "top_k": -1,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
    }
    assert template == {"enable_thinking": True}


@pytest.mark.parametrize("thinking", [True, False])
def test_explicit_sampling_and_thinking_overrides(thinking):
    params, extra = remote_sampling(
        {
            "model_name": "opaque",
            "sampling_profile": "nemotron-3.5-super-vl",
            "remote": {
                "request_params": {"temperature": 0.4, "max_tokens": 123},
                "extra_body": {
                    "top_k": 42,
                    "chat_template_kwargs": {
                        "enable_thinking": thinking,
                        "medium_effort": True,
                    },
                },
            },
        }
    )
    assert params["temperature"] == 0.4 and params["top_p"] == 0.95
    assert params["max_tokens"] == 123
    assert extra["top_k"] == 42
    assert extra["chat_template_kwargs"] == {
        "enable_thinking": thinking,
        "medium_effort": True,
    }
    assert "reasoning_effort" not in params


def test_profile_replaces_old_filters_and_server_builds_both_parsers():
    config = load_default_preset()
    budget = config.trial["remote"]["request_params"]["max_tokens"]
    apply_model_profile(config, MODEL, sections=["trial"])
    assert config.trial["remote"]["request_params"]["max_tokens"] == budget
    assert config.trial["remote"]["extra_body"]["top_k"] == -1
    config.trial["local"]["model_name"] = MODEL
    command = build_vllm_server_command(config=config, task="trial").command
    assert command[command.index("--reasoning-parser") + 1] == "nemotron_v3"
    assert command[command.index("--tool-call-parser") + 1] == "qwen3_coder"
    assert "--enable-auto-tool-choice" in command
    overridden = build_vllm_server_command(
        config=config, task="trial", extra_args=["--tool-call-parser", "custom"]
    ).command
    assert overridden.count("--tool-call-parser") == 1
    assert overridden[-1] == "custom"
    config.trial["tool_call_parser"] = "none"
    disabled = build_vllm_server_command(config=config, task="trial").command
    assert "--tool-call-parser" not in disabled
    assert "--enable-auto-tool-choice" not in disabled


def test_other_nemotron_generations_are_not_guessed():
    assert resolve_model_profile("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B") is None
