import sys
from unittest.mock import MagicMock

from matchminer_ai.llm import backends


def test_one_local_engine_at_a_time(monkeypatch):
    """A section with other engine settings shuts the loaded engine down before loading its own."""
    engines = []

    def make_engine(model, **kwargs):
        engine = MagicMock(name=f"engine-{kwargs['max_model_len']}")
        engines.append(engine)
        return engine

    mock_vllm = MagicMock()
    mock_vllm.LLM.side_effect = make_engine
    monkeypatch.setitem(sys.modules, "vllm", mock_vllm)
    backends.clear_local_llm_cache()
    try:
        first = backends._get_local_llm("m", '{"max_model_len": 100000}')
        assert backends._get_local_llm("m", '{"max_model_len": 100000}') is first
        second = backends._get_local_llm("m", '{"max_model_len": 200000}')
        assert second is not first
        first.llm_engine.engine_core.shutdown.assert_called_once()
        second.llm_engine.engine_core.shutdown.assert_not_called()
        assert mock_vllm.LLM.call_count == 2
    finally:
        backends.clear_local_llm_cache()
    second.llm_engine.engine_core.shutdown.assert_called_once()
