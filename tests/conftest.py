import pandas as pd
import pytest

from matchminer_ai.config import MMAIConfig
from matchminer_ai.trials.postprocess import _strip_numerical_prefix


TRIAL_SPACE_1 = (
    "1. Age: 18+. Sex: Any. Cancer type allowed: A. Histology allowed: Any. "
    "Cancer burden allowed: Any. Prior treatment required: None. "
    "Prior treatment excluded: None. Biomarkers required: None. "
    "Biomarkers excluded: None."
)
TRIAL_SPACE_2 = (
    "2. Age: 18+. Sex: Any. Cancer type allowed: B. Histology allowed: Any. "
    "Cancer burden allowed: Any. Prior treatment required: None. "
    "Prior treatment excluded: None. Biomarkers required: None. "
    "Biomarkers excluded: None."
)
TRIAL_SPACE_3 = (
    "3. Age: 18+. Sex: Any. Cancer type allowed: C. Histology allowed: Any. "
    "Cancer burden allowed: Any. Prior treatment required: None. "
    "Prior treatment excluded: None. Biomarkers required: None. "
    "Biomarkers excluded: None."
)
TRIAL_SPACE_4 = (
    "1. Age: 18+. Sex: Any. Cancer type allowed: D. Histology allowed: Any. "
    "Cancer burden allowed: Any. Prior treatment required: None. "
    "Prior treatment excluded: None. Biomarkers required: None. "
    "Biomarkers excluded: None."
)
BOILERPLATE_1 = "Uncontrolled brain metastases."
BOILERPLATE_2 = "History of pneumonitis."


@pytest.fixture
def default_trial_config() -> dict:
    return {
        # Fixtures below exercise the explicitly retained legacy space path.
        "extraction_method": "one_shot",
        "local": {
            "model_name": "model",
            "engine": {
                "max_model_len": 100,
                "tensor_parallel_size": 1,
                "gpu_memory_utilization": 0.9,
            },
            "generation": {
                "temperature": 0.0,
                "top_k": 1,
                "max_tokens": 10,
                "repetition_penalty": 1.0,
            },
            "chat_template_kwargs": {},
        },
        "remote": {
            "model_name": "model",
            "request_params": {"max_tokens": 10, "temperature": 0.0},
            "extra_body": {
                "top_k": 1,
                "repetition_penalty": 1.0,
            },
        },
        "prompt_files": {
            "primer": "trial.user.primer.txt",
            "question": "trial.user.question.txt",
        },
        "boilerplate_marker": "Boilerplate exclusions",
        "model_metadata_cache_dir": ".mmai_cache/model_metadata",
    }


@pytest.fixture
def default_config(default_trial_config: dict) -> MMAIConfig:
    return MMAIConfig(
        preset_name="default",
        debug_mode=False,
        trial=default_trial_config,
        patient={},
        local={},
        remote={},
        model_metadata_cache_dir=None,
        llm_match_quality={},
        llm_exclusion_criteria={},
        raw={},
        embedding={},
    )


@pytest.fixture
def mock_summarized_data() -> pd.DataFrame:
    # post llm summarization, raw LLM output
    summarized = pd.DataFrame(
        [
            {
                "trial_id": "T1",
                "trial_title": "Title 1",
                "brief_summary": "Brief 1",
                "eligibility_criteria": "Criteria 1",
                "trial_text": "Text 1",
                "trial_answer_text": (
                    f"{TRIAL_SPACE_1}\n"
                    f"{TRIAL_SPACE_2}\n"
                    f"{TRIAL_SPACE_3}\n"
                    "Boilerplate exclusions:\n"
                    f"{BOILERPLATE_1}"
                ),
            },
            {
                "trial_id": "T2",
                "trial_title": "Title 2",
                "brief_summary": "Brief 2",
                "eligibility_criteria": "Criteria 2",
                "trial_text": "Text 2",
                "trial_answer_text": (
                    f"{TRIAL_SPACE_4}\n" "Boilerplate exclusions:\n" f"{BOILERPLATE_2}"
                ),
            },
        ]
    )
    return summarized


@pytest.fixture
def expected_flattened_spaces(mock_summarized_data: pd.DataFrame) -> pd.DataFrame:
    with_summaries = mock_summarized_data.copy()
    with_summaries["space_text"] = [
        (TRIAL_SPACE_1 + "\n" + TRIAL_SPACE_2 + "\n" + TRIAL_SPACE_3).strip(),
        TRIAL_SPACE_4.strip(),
    ]
    with_summaries["boilerplate_text"] = [
        BOILERPLATE_1.strip(),
        BOILERPLATE_2.strip(),
    ]

    for_embed = with_summaries.iloc[[0, 0, 0, 1]].copy()
    for_embed["clinical_space_summary"] = [
        _strip_numerical_prefix(TRIAL_SPACE_1),
        _strip_numerical_prefix(TRIAL_SPACE_2),
        _strip_numerical_prefix(TRIAL_SPACE_3),
        _strip_numerical_prefix(TRIAL_SPACE_4),
    ]
    for_embed["clinical_space_number"] = [0, 1, 2, 0]

    return for_embed.reset_index(drop=True)


def prompt_text(prompt):
    """All message text of a rendered prompt (the original request plus any retry note)."""
    messages = getattr(prompt, "messages", None)
    return "\n".join(m["content"] for m in messages) if messages else prompt.prompt_text


class ScriptedCohortBackend:
    """Fake LLM backend for the three-step cohort workflow.

    ``script`` maps a text that identifies a trial's document to its answers by
    step: ``{"screen": [...], "list": [...], "criteria": {cohort_name: [...]}}``, and the
    boilerplate step by item text: ``{"deboilerplate": {item: [...]}}``.
    Answers are consumed in order (the last one repeats); a dict is sent as JSON
    and an ``(answer, finish_reason)`` pair sets the finish reason.
    """

    def __init__(self):
        self.script, self.calls = {}, []

    @staticmethod
    def step_of(text):
        if "Decide whether this trial evaluates a cancer treatment" in text:
            return "screen", None
        if "List the cohorts it defines" in text:
            return "list", None
        if "Group the cohorts whose eligibility criteria are essentially the same" in text:
            return "merge", None
        if "is a boilerplate criterion. Boilerplate criteria are general requirements" in text:
            return "deboilerplate", text.rsplit("ITEM TO CLASSIFY\n", 1)[1].split("] ", 1)[1].split("\n")[0].strip()
        return "criteria", text.rsplit("Extract the complete eligibility criteria of this cohort only: ", 1)[1].split("\n")[0].strip()

    def answer(self, prompt):
        import json
        text = prompt_text(prompt)
        step, target = self.step_of(text)
        marker = next(m for m in self.script if m in text)
        queue = self.script[marker][step]
        if step == "criteria":
            queue = queue[target]
        elif step == "deboilerplate":  # answers by item; an item not listed is not boilerplate
            queue = queue.get(target, [{"boilerplate": False}])
        value = queue.pop(0) if len(queue) > 1 else queue[0]
        answer, finish = value if isinstance(value, tuple) else (value, "stop")
        return json.dumps(answer) if isinstance(answer, dict) else answer, finish

    def generate_llm_outputs(self, *, prompt_list, llm_config, **kwargs):
        from copy import deepcopy
        from matchminer_ai.llm.backends import LLMGenerationResult
        self.calls.append((prompt_list, deepcopy(llm_config)))
        answers = [self.answer(prompt) for prompt in prompt_list]
        return LLMGenerationResult(
            final_outputs=[a for a, _ in answers], finish_reasons=[f for _, f in answers],
            reasoning_outputs=["test reasoning"] * len(answers), raw_outputs=[],
            model_metadata={"model_sha": "test", "responses": [
                {"row_idx": i, "endpoint": "http://fake/v1"} for i in range(len(answers))]},
        )


@pytest.fixture
def cohort_backend(monkeypatch):
    backend = ScriptedCohortBackend()
    for module in ("matchminer_ai.trials.cohorts", "matchminer_ai.trials.protocol_cohorts",
                   "matchminer_ai.trials.cohort_merge", "matchminer_ai.trials.cohort_boilerplate",
                   "matchminer_ai.trials.cohort_screen"):
        monkeypatch.setattr(f"{module}.get_llm_backend", lambda _config: backend)
    return backend
