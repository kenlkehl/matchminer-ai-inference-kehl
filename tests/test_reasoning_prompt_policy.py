"""Model-specific text controls must reach rendered and remote requests safely."""
import pytest

from matchminer_ai.llm.reasoning import reasoning_system_prompt


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
