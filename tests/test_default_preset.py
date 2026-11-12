from matchminer_ai.config import load_default_preset

MODEL = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4"
LLM_SECTIONS = (
    "trial", "patient", "raw_patient_note_qa", "full_patient_screen",
    "trial_space_criteria_extraction", "protocol_cohort_extraction",
    "synthetic_protocol_generation", "llm_match_quality", "llm_exclusion_criteria",
    "llm_good_option",
)
# Production vLLM settings for Nemotron 3.5 Super, as vllm.LLM keyword arguments.
NEMOTRON_ENGINE = {
    "language_model_only": True,
    "trust_remote_code": True,
    "enable_prefix_caching": True,
    "mamba_backend": "flashinfer",
    "mamba_ssm_cache_dtype": "float16",
    "enable_mamba_cache_stochastic_rounding": True,
    "mamba_cache_philox_rounds": 5,
    "mamba_cache_mode": "align",
}


def test_default_preset_uses_vendor_profiles_and_compatible_matching_models():
    """Resolve vendor sampling without changing the dated matching model family."""
    config = load_default_preset()

    assert config.local == {}
    assert config.trial["local"]["model_name"] == MODEL
    assert config.trial["local"]["engine"]["max_model_len"] == 100000
    from matchminer_ai.llm.backends import build_llm_runtime_config

    for name in ("trial", "patient", "llm_match_quality", "llm_exclusion_criteria"):
        stage = getattr(config, name)
        local = build_llm_runtime_config(name, stage, config=config)
        assert local["sampling_params"]["temperature"] == 1.0
        assert local["sampling_params"]["top_p"] == 0.95
        assert local["sampling_params"]["top_k"] == -1
        assert local["sampling_params"]["min_p"] == 0.0
        assert local["sampling_params"]["presence_penalty"] == 0.0
        assert local["sampling_params"]["repetition_penalty"] == 1.0
        assert local["chat_template_kwargs"] == {"enable_thinking": True}
        assert stage["reasoning_effort"] == "xhigh"
    assert config.remote["max_concurrent_requests"] == 32

    assert config.patient["local"]["model_name"] == MODEL
    assert config.patient["remote"]["model_name"] == MODEL
    assert config.patient["remote"]["tokenizer_name"] == MODEL
    assert config.patient["local"]["engine"]["max_model_len"] == 200000
    assert config.patient["chunk_size"] == 50000
    assert config.patient["chunk_overlap"] == 500
    assert config.patient["local"]["generation"]["max_tokens"] == 50000
    assert config.patient["remote"]["request_params"]["max_tokens"] == 50000
    assert config.raw_patient_note_qa["embedding_model_name"] == (
        "Qwen/Qwen3-Embedding-0.6B"
    )
    assert config.raw_patient_note_qa["embedding_device"] == "cuda"
    assert config.raw_patient_note_qa["chunk_size"] == 220
    assert config.raw_patient_note_qa["chunk_overlap"] == 32
    assert config.raw_patient_note_qa["query_prefix"].startswith("Instruct:")
    assert config.raw_patient_note_qa["query_prefix"].endswith("\nQuery:")
    assert config.raw_patient_note_qa["max_agent_steps"] == 6
    assert config.raw_patient_note_qa["local"]["generation"]["max_tokens"] == 16000
    assert config.full_patient_screen["max_workers"] == 4
    assert config.full_patient_screen["process_start_method"] == "spawn"
    assert config.full_patient_screen["max_questions"] == 128
    assert (
        config.full_patient_screen["remote"]["request_params"]["max_tokens"]
        == 12000
    )
    assert config.full_patient_screen["local"]["chat_template_kwargs"] == {
        "enable_thinking": False
    }
    assert config.full_patient_screen["remote"]["request_params"][
        "response_format"
    ] == {"type": "json_object"}
    assert config.full_patient_screen["remote"]["extra_body"][
        "chat_template_kwargs"
    ] == {"enable_thinking": False}
    assert config.trial_space_criteria_extraction["max_document_characters"] == 120000
    assert config.trial_space_criteria_extraction["max_criteria_per_type"] == 256
    assert config.trial_space_criteria_extraction["response_retry_limit"] == 2
    assert (
        config.trial_space_criteria_extraction["local"]["generation"]["max_tokens"]
        == 50000
    )
    assert config.trial_space_criteria_extraction["local"]["engine"][
        "max_model_len"
    ] == 100000
    assert config.trial_space_criteria_extraction["local"]["chat_template_kwargs"] == {
        "enable_thinking": True
    }
    assert config.trial_space_criteria_extraction["remote"]["request_params"][
        "max_tokens"
    ] == 50000
    assert config.trial_space_criteria_extraction["remote"]["request_params"][
        "response_format"
    ] == {"type": "json_object"}
    assert config.trial_space_criteria_extraction["remote"]["extra_body"][
        "chat_template_kwargs"
    ] == {"enable_thinking": True}

    assert config.embedding.get("backend", "sequence") == "sequence"
    assert config.embedding["model_path"] == "ksg-dfci/TrialSpace-0526"
    assert config.embedding["max_seq_length"] == 4096
    from matchminer_ai.matching.decision_model import decision_settings
    decision = decision_settings(config)
    assert (decision["model_name"], decision["revision"]) == ("LiquidAI/d1-2B", "652f920e544ea37f81e4b0ce89b81d4ed0da90b1")
    assert decision["max_length"] == 10000
    assert config.raw["match_quality"] == {"score_cutoff": 0.2}

    assert config.llm_match_quality["local"]["engine"]["max_model_len"] == 100000
    assert config.llm_match_quality["local"]["model_name"] == MODEL
    assert config.llm_match_quality["remote"]["model_name"] == MODEL
    assert config.llm_match_quality["local"]["generation"]["max_tokens"] == 32768
    assert config.llm_exclusion_criteria["local"]["engine"]["max_model_len"] == 100000
    assert config.llm_exclusion_criteria["local"]["model_name"] == MODEL
    assert config.llm_exclusion_criteria["remote"]["model_name"] == MODEL
    assert config.llm_exclusion_criteria["local"]["generation"]["max_tokens"] == 32768
    assert config.llm_good_option["local"]["engine"]["max_model_len"] == 262144
    assert config.llm_good_option["local"]["generation"]["max_tokens"] == 100000
    assert config.llm_good_option["local"]["generation"][
        "repetition_penalty"
    ] == 1.1
    assert config.llm_good_option["local"]["chat_template_kwargs"] == {
        "enable_thinking": True
    }
    assert config.llm_good_option["remote"]["request_params"]["max_tokens"] == 100000
    assert config.llm_good_option["remote"]["extra_body"][
        "repetition_penalty"
    ] == 1.1
    assert config.llm_good_option["remote"]["extra_body"][
        "chat_template_kwargs"
    ] == {"enable_thinking": True}
    assert (
        config.good_option_catalog["synthesis_llm"]["local"]["generation"]["max_tokens"]
        == 100000
    )
    assert (
        config.good_option_catalog["synthesis_llm"]["remote"]["request_params"][
            "max_tokens"
        ]
        == 100000
    )
    # A 60k-token chunk, existing facts, and this budget fit a 262,144 context;
    # 100k chunks lost ~25% of facts in a Gemma 4 comparison.
    assert config.good_option_catalog["synthesis_chunk_max_tokens"] == 60000
    assert config.good_option_catalog["class_evidence_max_tokens"] == 60000
    assert config.raw["good_option_checker"] == {
        "model_name": "",
        "device": "cuda",
        "max_length": 8192,
    }
    config.remote["enabled"] = True
    remote = build_llm_runtime_config(
        "llm_match_quality", config.llm_match_quality, config=config
    )
    assert remote["remote"]["request_params"] == {
        "max_tokens": 32768,
        "temperature": 1.0,
        "top_p": 0.95,
        "presence_penalty": 0.0,
    }
    assert remote["remote"]["extra_body"]["top_k"] == -1
    assert remote["remote"]["extra_body"]["repetition_penalty"] == 1.0


def test_default_preset_serves_nemotron_in_every_llm_section():
    """Every LLM section defaults to Nemotron 3.5 Super, locally and remotely."""
    config = load_default_preset()
    for name in LLM_SECTIONS:
        section = getattr(config, name)
        assert section["local"]["model_name"] == MODEL, name
        assert section["remote"]["model_name"] == MODEL, name
        engine = section["local"]["engine"]
        assert {key: engine[key] for key in NEMOTRON_ENGINE} == NEMOTRON_ENGINE, name
        assert engine["tensor_parallel_size"] == 1, name
    assert config.patient["remote"]["tokenizer_name"] == MODEL


def test_default_local_engine_settings_reach_vllm(monkeypatch):
    """The local backend passes the preset's Nemotron engine keys to vllm.LLM."""
    import sys
    from unittest.mock import MagicMock

    from matchminer_ai.llm import backends
    from matchminer_ai.llm.prompt_rendering import Prompt

    mock_llm = MagicMock()
    output = MagicMock()
    output.outputs[0].text = "answer"
    output.outputs[0].finish_reason = "stop"
    mock_llm.generate.return_value = [output]
    mock_vllm = MagicMock()
    mock_vllm.LLM.return_value = mock_llm
    monkeypatch.setitem(sys.modules, "vllm", mock_vllm)
    monkeypatch.setattr(backends, "get_model_metadata", lambda name, cache_dir=None: {})
    monkeypatch.setattr(
        backends, "parse_reasoning_output", lambda text, parser_name, tokenizer, **kw: ("", text)
    )
    config = load_default_preset()
    runtime = backends.build_llm_runtime_config("patient", config.patient, config=config)
    backends._get_local_llm.cache_clear()
    try:
        backends.LocalBackend().generate_llm_outputs(
            prompt_list=[Prompt(row_idx=0, prompt_text="synthetic prompt", max_tokens=8)],
            llm_config=runtime,
        )
    finally:
        backends._get_local_llm.cache_clear()
    kwargs = mock_vllm.LLM.call_args.kwargs
    assert kwargs["model"] == MODEL
    assert {key: kwargs[key] for key in NEMOTRON_ENGINE} == NEMOTRON_ENGINE
    assert kwargs["max_model_len"] == 200000
    assert mock_vllm.SamplingParams.call_args.kwargs["max_tokens"] == 50000
    assert mock_vllm.SamplingParams.call_args.kwargs["top_k"] == -1


def test_default_server_command_serves_nemotron_like_production():
    from matchminer_ai.llm.vllm_server import build_vllm_server_command

    command = build_vllm_server_command(config=load_default_preset(), task="patient").command
    assert command[:3] == ["vllm", "serve", MODEL]
    for flag in ("--trust-remote-code", "--language-model-only", "--enable-prefix-caching",
                 "--enable-mamba-cache-stochastic-rounding"):
        assert flag in command
    for flag, value in (("--mamba-backend", "flashinfer"), ("--mamba-ssm-cache-dtype", "float16"),
                        ("--mamba-cache-philox-rounds", "5"), ("--mamba-cache-mode", "align"),
                        ("--max-model-len", "200000"), ("--reasoning-parser", "nemotron_v3"),
                        ("--tool-call-parser", "qwen3_coder")):
        assert command[command.index(flag) + 1] == value
