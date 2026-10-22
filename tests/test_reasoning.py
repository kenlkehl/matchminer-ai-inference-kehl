import sys
from types import ModuleType

import pytest

from matchminer_ai.llm.reasoning import parse_reasoning_output, resolve_reasoning_parser


def _module(name, **attrs):
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def test_resolve_reasoning_parser_auto_for_default_model():
    assert resolve_reasoning_parser("google/gemma-4-31B-it", "auto") == "gemma4"


def test_resolve_reasoning_parser_auto_uses_known_model_mapping():
    assert resolve_reasoning_parser("openai/gpt-oss-120b", "auto") == "openai_gptoss"


def test_resolve_reasoning_parser_allows_explicit_override():
    assert resolve_reasoning_parser("unknown/model", "qwen3") == "qwen3"


def test_resolve_reasoning_parser_can_disable():
    assert resolve_reasoning_parser("google/gemma-4-31B-it", "none") is None


def test_resolve_reasoning_parser_unknown_auto_raises():
    with pytest.raises(ValueError, match="Cannot infer"):
        resolve_reasoning_parser("unknown/model", "auto")


def test_parse_reasoning_output_uses_gemma4_utility(monkeypatch):
    """Gemma 4 local parsing uses vLLM's offline thinking parser."""

    def parse_thinking_output(text):
        assert text == "raw output"
        return {"thinking": " model reasoning ", "answer": " final text "}

    monkeypatch.setitem(sys.modules, "vllm", _module("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.reasoning", _module("vllm.reasoning"))
    monkeypatch.setitem(
        sys.modules,
        "vllm.reasoning.gemma4_utils",
        _module(
            "vllm.reasoning.gemma4_utils",
            parse_thinking_output=parse_thinking_output,
        ),
    )

    reasoning, content = parse_reasoning_output(
        "raw output",
        parser_name="gemma4",
        tokenizer=object(),
    )

    assert reasoning == "model reasoning"
    assert content == "final text"


def test_parse_reasoning_output_uses_reasoning_parser_manager(monkeypatch):
    """Other parsers go through vLLM's class-based parser registry."""

    class FakeParser:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

        def extract_reasoning(self, text, request):
            assert text == "raw output"
            assert request is None
            return " parsed reasoning ", " parsed final "

    class FakeReasoningParserManager:
        @classmethod
        def get_reasoning_parser(cls, parser_name):
            assert parser_name == "qwen3"
            return FakeParser

    monkeypatch.setitem(sys.modules, "vllm", _module("vllm"))
    monkeypatch.setitem(
        sys.modules,
        "vllm.reasoning",
        _module(
            "vllm.reasoning",
            ReasoningParserManager=FakeReasoningParserManager,
        ),
    )

    reasoning, content = parse_reasoning_output(
        "raw output",
        parser_name="qwen3",
        tokenizer=object(),
    )

    assert reasoning == "parsed reasoning"
    assert content == "parsed final"


@pytest.mark.parametrize("prefix", ["", "<|start|>assistant"])
@pytest.mark.parametrize("transition", [
    "<|end|><|start|>assistant", "<|constrain|><|start|>assistant", "<|start|>assistant",
])
@pytest.mark.parametrize("ending", ["", "<|return|>", "<|end|>", "<|call|>"])
def test_beam_native_channels_without_local_vllm(prefix, transition, ending, monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm", None)
    raw = (prefix + "<|channel|>analysis<|message|> Reason about the seed. "
           + transition + "<|channel|>final<|message|> <ngs_report> NRG1 <1%. " + ending)
    assert parse_reasoning_output(raw, parser_name="beam", tokenizer=None) == (
        "Reason about the seed.", "<ngs_report> NRG1 <1%.")


@pytest.mark.parametrize("raw,expected", [
    ("<|channel|>analysis<|message|>Unfinished reasoning", ("Unfinished reasoning", "")),
    ("<|channel|>analysis<|message|>Think<|end|><|start|>assistant<|channel|>final",
     ("Think", "")),
    ("<|channel|>final<|message|>Final only<|return|>", ("", "Final only")),
    ("<|start|>assistant to=search<|channel|>analysis <|constrain|>xml<|message|>"
     "<rfl:parameter name=\"query\">\nnot an answer\n</rfl:parameter><|call|>", ("", "")),
    ("unframed output", ("", "")),
    ("", ("", "")),
])
def test_beam_does_not_promote_reasoning_or_tool_output_to_final(raw, expected):
    assert parse_reasoning_output(raw, parser_name="beam", tokenizer=None) == expected


def test_nemotron_direct_answer_parser_receives_disabled_thinking(monkeypatch):
    class Parser:
        def __init__(self, tokenizer, **options):
            assert options == {"chat_template_kwargs": {"enable_thinking": False}}

        def extract_reasoning(self, text, request):
            assert request.chat_template_kwargs == {"enable_thinking": False}
            return "", '{"branches": []}<|im_end|>'

    manager = type('Manager', (), {'get_reasoning_parser': staticmethod(lambda name: Parser)})
    monkeypatch.setitem(sys.modules, 'vllm.reasoning', _module('vllm.reasoning', ReasoningParserManager=manager))
    assert parse_reasoning_output('raw', parser_name='nemotron_v3', tokenizer=object(),
                                  enable_thinking=False) == ('', '{"branches": []}')


def test_nemotron_parser_reuse_is_isolated_and_serializes_concurrent_calls(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    import time

    constructed = []

    class Parser:
        def __init__(self, tokenizer, **options):
            self.tokenizer = tokenizer
            self.thinking = options.get('chat_template_kwargs', {}).get('enable_thinking')
            self.active = False
            constructed.append(self)

        def extract_reasoning(self, text, request):
            assert not self.active, 'Mutable parser used concurrently'
            self.active = True
            try:
                time.sleep(0.002)
                mode = None if request is None else request.chat_template_kwargs['enable_thinking']
                assert mode is self.thinking
                return str(mode), text
            finally:
                self.active = False

    manager = type('Manager', (), {'get_reasoning_parser': staticmethod(lambda name: Parser)})
    monkeypatch.setitem(sys.modules, 'vllm.reasoning', _module('vllm.reasoning', ReasoningParserManager=manager))
    tokenizers = [object(), object()]
    barrier = Barrier(8)

    def parse(i):
        barrier.wait()
        return parse_reasoning_output(str(i), parser_name='nemotron_v3',
                                      tokenizer=tokenizers[0], enable_thinking=True)

    with ThreadPoolExecutor(max_workers=8) as workers:
        assert list(workers.map(parse, range(8))) == [('True', str(i)) for i in range(8)]
    assert len(constructed) == 1
    for tokenizer, mode in [(tokenizers[0], False), (tokenizers[0], None), (tokenizers[1], True)]:
        assert parse_reasoning_output('answer', parser_name='nemotron_v3',
                                      tokenizer=tokenizer, enable_thinking=mode) == (str(mode), 'answer')
    assert len(constructed) == 4


def test_nemotron_failed_parser_is_rebuilt_and_cache_is_bounded(monkeypatch):
    from matchminer_ai.llm import reasoning

    constructed = []

    class Parser:
        def __init__(self, tokenizer, **options):
            constructed.append(self)

        def extract_reasoning(self, text, request):
            if text == 'bad':
                raise ValueError('parse failure')
            return '', text

    manager = type('Manager', (), {'get_reasoning_parser': staticmethod(lambda name: Parser)})
    monkeypatch.setitem(sys.modules, 'vllm.reasoning', _module('vllm.reasoning', ReasoningParserManager=manager))
    tokenizer = object()
    for text in ['ok', 'bad', 'ok']:
        if text == 'bad':
            with pytest.raises(ValueError, match='parse failure'):
                parse_reasoning_output(text, parser_name='nemotron_v3', tokenizer=tokenizer)
        else:
            assert parse_reasoning_output(text, parser_name='nemotron_v3', tokenizer=tokenizer) == ('', text)
    assert len(constructed) == 2
    for _ in range(reasoning._PARSER_CACHE_SIZE + 2):
        parse_reasoning_output('ok', parser_name='nemotron_v3', tokenizer=object())
    assert len(reasoning._PARSER_CACHE) <= reasoning._PARSER_CACHE_SIZE
