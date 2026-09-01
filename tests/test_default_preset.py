from matchminer_ai.config import load_default_preset


def test_default_preset_matches_training_runtime_defaults():
    """Keep public inference defaults aligned with the training scripts."""
    config = load_default_preset()

    assert config.local == {}
    assert config.trial["local"]["model_name"] == "google/gemma-4-31B-it"
    assert config.trial["local"]["engine"]["max_model_len"] == 30000
    assert config.trial["local"]["generation"] == {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "max_tokens": 20000,
        "repetition_penalty": 1.0,
        "skip_special_tokens": False,
    }

    assert config.trial["remote"] == {
        "model_name": "google/gemma-4-31B-it",
        "request_params": {
            "max_tokens": 20000,
            "temperature": 1.0,
            "top_p": 0.95,
            "presence_penalty": 1.5,
        },
        "extra_body": {
            "top_k": 20,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "skip_special_tokens": False,
            "chat_template_kwargs": {"enable_thinking": True},
        },
    }

    assert config.patient["local"]["model_name"] == "google/gemma-4-31B-it"
    assert config.patient["remote"]["model_name"] == "google/gemma-4-31B-it"
    assert config.patient["remote"]["tokenizer_name"] == "google/gemma-4-31B-it"
    assert config.patient["local"]["engine"]["max_model_len"] == 100000
    assert config.patient["chunk_size"] == 50000
    assert config.patient["chunk_overlap"] == 500
    assert config.patient["local"]["generation"]["temperature"] == 0.0
    assert config.patient["local"]["generation"]["top_k"] == 1
    assert config.patient["local"]["generation"]["max_tokens"] == 20000
    assert config.raw_patient_note_qa["embedding_model_name"] == (
        "Qwen/Qwen3-Embedding-0.6B"
    )
    assert config.raw_patient_note_qa["embedding_device"] == "cuda"
    assert config.raw_patient_note_qa["chunk_size"] == 220
    assert config.raw_patient_note_qa["chunk_overlap"] == 32
    assert config.raw_patient_note_qa["query_prefix"].startswith("Instruct:")
    assert config.raw_patient_note_qa["query_prefix"].endswith("\nQuery:")
    assert config.raw_patient_note_qa["max_agent_steps"] == 6
    assert config.raw_patient_note_qa["local"]["generation"]["max_tokens"] == 4000
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
    assert config.patient_structuring["oncotree_version"] == "stable-2026-07-31"
    assert config.patient_structuring["ncit_version"] == "26.07d"
    assert config.patient_structuring["ncit_candidate_limit"] == 8
    assert config.patient_structuring["ncit_max_agent_steps"] == 3
    assert config.patient_structuring["local"]["generation"]["max_tokens"] == 30000
    assert config.patient_structuring["remote"]["request_params"]["max_tokens"] == 30000
    assert config.trial_space_structuring["oncotree_version"] == (
        "stable-2026-07-31"
    )
    assert config.trial_space_structuring["ncit_version"] == "26.07d"
    assert config.trial_space_structuring["ncit_candidate_limit"] == 8
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

    assert config.embedding["model_path"] == "ksg-dfci/TrialSpace-0526"
    assert config.embedding["max_seq_length"] == 2500
    assert config.raw["match_quality"]["model_name"] == "ksg-dfci/TrialChecker-0526"
    assert config.raw["match_quality"]["max_length"] == 4096
    assert config.raw["exclusion_criteria"]["model_name"] == (
        "ksg-dfci/BoilerplateChecker-0526"
    )
    assert config.raw["exclusion_criteria"]["max_length"] == 3192

    assert config.llm_match_quality["local"]["engine"]["max_model_len"] == 50000
    assert config.llm_match_quality["local"]["model_name"] == "google/gemma-4-31B-it"
    assert config.llm_match_quality["remote"]["model_name"] == "google/gemma-4-31B-it"
    assert config.llm_match_quality["local"]["generation"]["temperature"] == 0.0
    assert config.llm_match_quality["local"]["generation"]["max_tokens"] == (15000)
    assert config.llm_exclusion_criteria["local"]["engine"]["max_model_len"] == 50000
    assert config.llm_exclusion_criteria["local"]["model_name"] == (
        "google/gemma-4-31B-it"
    )
    assert config.llm_exclusion_criteria["remote"]["model_name"] == (
        "google/gemma-4-31B-it"
    )
    assert config.llm_exclusion_criteria["local"]["generation"]["temperature"] == 0.0
    assert config.llm_exclusion_criteria["local"]["generation"]["max_tokens"] == 20000
    assert config.llm_good_option["local"]["engine"]["max_model_len"] == 131072
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
        == 32000
    )
    assert (
        config.good_option_catalog["synthesis_llm"]["remote"]["request_params"][
            "max_tokens"
        ]
        == 32000
    )
    assert config.raw["good_option_checker"] == {
        "model_name": "",
        "device": "cuda",
        "max_length": 8192,
    }
    assert config.llm_match_quality["remote"]["request_params"] == {
        "max_tokens": 15000,
        "temperature": 0.0,
        "top_p": 1.0,
        "presence_penalty": 0.0,
    }
    assert config.llm_match_quality["remote"]["extra_body"]["top_k"] == 1
    assert config.help_me_choose["local"]["generation"]["max_tokens"] == 6000
    assert config.help_me_choose["remote"]["request_params"]["max_tokens"] == 6000
    assert config.help_me_choose["local"]["generation"]["repetition_penalty"] == 1.1
    assert config.help_me_choose["remote"]["extra_body"]["repetition_penalty"] == 1.1
    assert config.trial_space_contextualization["sources"] == [
        "nci_pdq",
        "fda",
        "civic",
        "pubmed",
        "europe_pmc_open_guidelines",
    ]
    assert (
        config.trial_space_contextualization["evidence_context_max_tokens"]
        == 12000
    )
    assert (
        config.trial_space_contextualization["diagnostic_context_min_tokens"]
        == 8000
    )
    assert (
        config.trial_space_contextualization["local"]["generation"]["temperature"]
        == 0.0
    )
    assert (
        config.patient_contextualization["remote"]["request_params"]["max_tokens"]
        == 5000
    )
