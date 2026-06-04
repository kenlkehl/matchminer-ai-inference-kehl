from matchminer_ai.llm.prompt_rendering import build_prompt_list


class MockTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, **kwargs):
        self.calls.append(kwargs)
        return "rendered prompt"


def test_build_prompt_list_passes_chat_template_kwargs(monkeypatch):
    tokenizer = MockTokenizer()
    monkeypatch.setattr(
        "matchminer_ai.llm.prompt_rendering._get_chat_template_tokenizer",
        lambda model_name: tokenizer,
    )

    messages = [[{"role": "user", "content": "hello"}]]
    prompts = build_prompt_list(
        messages,
        llm_config={
            "model_name": "google/gemma-4-31B-it",
            "sampling_params": {"max_tokens": 5},
            "chat_template_kwargs": {"enable_thinking": True},
        },
    )

    assert tokenizer.calls[0]["enable_thinking"] is True
    assert prompts[0].prompt_text == "rendered prompt"
    assert prompts[0].messages == messages[0]
