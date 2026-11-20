"""Task runtime configuration and the local backend."""
from unittest.mock import MagicMock

from matchminer_ai.llm.backends import LocalBackend, build_llm_runtime_config
from matchminer_ai.llm.prompt_rendering import Prompt


def test_build_llm_runtime_config_uses_remote_task_model_name(default_config):
    """Use the task-level remote model name for endpoint requests."""
    default_config.remote["enabled"] = True
    default_config.trial["remote"] = {
        "model_name": "task-alias",
        "request_params": {"max_tokens": 10},
        "extra_body": {},
    }

    runtime_config = build_llm_runtime_config(
        "trial",
        default_config.trial,
        config=default_config,
    )

    assert runtime_config["model_name"] == "task-alias"
    assert runtime_config["backend_mode"] == "remote"


def test_build_llm_runtime_config_flattens_task_local_config(default_config):
    """Task-local runtime config drives local engine and generation settings."""
    default_config.local = {}
    default_config.trial["local"] = {
        "model_name": "local-model",
        "engine": {
            "max_model_len": 1234,
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.8,
        },
        "generation": {
            "max_tokens": 99,
            "temperature": 0.2,
        },
        "chat_template_kwargs": {"enable_thinking": True},
    }

    runtime_config = build_llm_runtime_config(
        "trial",
        default_config.trial,
        config=default_config,
    )

    assert runtime_config["max_model_len"] == 1234
    assert runtime_config["model_name"] == "local-model"
    assert runtime_config["sampling_params"] == {
        "max_tokens": 99,
        "temperature": 0.2,
    }
    assert runtime_config["chat_template_kwargs"] == {"enable_thinking": True}
    assert runtime_config["backend_mode"] == "local"


def test_local_backend_generate_llm_outputs(monkeypatch, default_config):
    """Ensure our function running vLLM locally return both raw LLM outputs and model metadata."""
    mock_llm = MagicMock()
    mock_tokenizer = MagicMock()
    mock_llm.get_tokenizer.return_value = mock_tokenizer
    mock_tokenizer.apply_chat_template.side_effect = ["p1", "p2"]

    mock_outputs = [MagicMock(), MagicMock()]
    for n, output in enumerate(mock_outputs):
        output.outputs[0].text = f"SUM{n}"
        output.outputs[0].finish_reason = "stop"
    mock_llm.generate.return_value = mock_outputs

    mock_vllm = MagicMock()
    mock_vllm.LLM.return_value = mock_llm
    mock_vllm.SamplingParams = MagicMock()
    monkeypatch.setitem(__import__("sys").modules, "vllm", mock_vllm)
    monkeypatch.setattr(
        "matchminer_ai.llm.backends.get_model_metadata",
        lambda model_name, cache_dir=None: {
            "model_sha": "sha",
            "created_at": "now",
            "last_modified": "now",
        },
    )
    monkeypatch.setattr(
        "matchminer_ai.llm.backends.parse_reasoning_output",
        lambda text, parser_name, tokenizer: (f"REASON {text}", f"FINAL {text}"),
    )

    backend = LocalBackend()
    default_config.trial.setdefault("local", {}).setdefault("engine", {})[
        "trust_remote_code"
    ] = True
    default_config.trial["local"]["engine"]["speculative_config"] = {
        "num_speculative_tokens": 4,
    }
    default_config.trial["reasoning_parser"] = "gemma4"
    default_config.trial["tool_call_parser"] = "qwen3_coder"
    default_config.trial["prompt_file"] = "match_quality.labels.system.txt"
    default_config.trial.setdefault("local", {}).setdefault("generation", {})[
        "seed"
    ] = 123
    llm_config = build_llm_runtime_config(
        "trial",
        default_config.trial,
        config=default_config,
    )
    generation = backend.generate_llm_outputs(
        prompt_list=[
            Prompt(row_idx=0, prompt_text="p1", max_tokens=10),
            Prompt(row_idx=1, prompt_text="p2", max_tokens=10),
        ],
        llm_config=llm_config,
    )

    assert generation.final_outputs == ["FINAL SUM0", "FINAL SUM1"]
    assert generation.raw_outputs == ["SUM0", "SUM1"]
    assert generation.reasoning_outputs == ["REASON SUM0", "REASON SUM1"]
    assert generation.model_metadata["model_sha"] == "sha"
    assert generation.finish_reasons == ["stop", "stop"]
    assert mock_vllm.LLM.call_args.kwargs["trust_remote_code"] is True
    assert mock_vllm.LLM.call_args.kwargs["speculative_config"] == {
        "num_speculative_tokens": 4,
    }
    assert "prompt_file" not in mock_vllm.LLM.call_args.kwargs
    assert "tool_call_parser" not in mock_vllm.LLM.call_args.kwargs
    assert mock_vllm.SamplingParams.call_args.kwargs["seed"] == 123
