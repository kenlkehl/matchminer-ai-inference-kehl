"""Model-specific text controls must reach rendered and remote requests safely."""
from types import SimpleNamespace

import pytest

from matchminer_ai.llm.reasoning import reasoning_system_prompt
from matchminer_ai.trials.summarize import summarize_trials_multi_cohort


MODELS = [
    ("openai/gpt-oss-120b", True),
    ("openai/gpt-oss-20b", True),
    ("GPTOSS-120B", True),
    ("google/gemma-4-31B-it", False),
    ("nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4", False),
    ("Qwen/Qwen3.6-27B", False),
    ("unknown/model", False),
]


@pytest.mark.parametrize("model,expected", MODELS)
def test_task_instruction_is_preserved_without_cross_family_header(model, expected):
    text = reasoning_system_prompt(model, "Follow the requested JSON contract exactly.")
    assert ("Reasoning: high" in text) is expected
    assert text.endswith("Follow the requested JSON contract exactly.")


@pytest.mark.parametrize("model,expected", MODELS)
def test_one_shot_trial_remote_messages_use_actual_model(model, expected):
    captured = []
    backend = SimpleNamespace(generate_llm_outputs=lambda **kwargs: captured.extend(kwargs['prompt_list']))
    summarize_trials_multi_cohort(
        ["Synthetic trial document."], backend,
        trial_config=dict(extraction_method="one_shot", backend_mode="remote",
                          model_name=model, sampling_params={"max_tokens": 1000}),
        primer_filename="trial.user.primer.txt", question_filename="trial.user.question.txt",
    )
    assert ("Reasoning: high" in captured[0].messages[0]['content']) is expected
    assert "Synthetic trial document." in captured[0].messages[1]['content']
