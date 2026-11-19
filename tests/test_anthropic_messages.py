"""Claude through a Databricks AI Gateway behind the structured client (no network)."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

anthropic = pytest.importorskip("anthropic")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from matchminer_ai.config import load_default_preset  # noqa: E402
from matchminer_ai.llm import anthropic_messages as am  # noqa: E402
from matchminer_ai.llm import remote_auth  # noqa: E402
from matchminer_ai.llm.structured import (  # noqa: E402
    CredentialsError,
    EndpointError,
    StructuredClient,
    StructuredConfig,
)
from matchminer_ai.patients import note_search_qa as qa  # noqa: E402
from matchminer_ai.patients.workup_sql import (  # noqa: E402
    agent_contract,
    build_extraction_database,
    compile_workup_sql,
    resolve_agent_llm,
)

GATEWAY = "https://adb-1234567890123456.7.azuredatabricks.net/ai-gateway/anthropic"
SONNET = "ai_gateway_model_services_prod.team_project.team_project-claude-sonnet-5-5"
HAIKU = "ai_gateway_model_services_prod.team_project.team_project-claude-haiku-5-5"


def message(text, *, stop="end_turn", model="claude-sonnet-5-5", fallback=False):
    content = [SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)]
    if fallback:
        content.insert(0, SimpleNamespace(type="fallback"))
    return SimpleNamespace(id="msg_1", model=model, stop_reason=stop, content=content,
                           usage=SimpleNamespace(input_tokens=100, output_tokens=40,
                                                 cache_read_input_tokens=900,
                                                 cache_creation_input_tokens=0))


class FakeClaude:
    """Stands in for anthropic.Anthropic: records requests, answers from a script."""

    def __init__(self, answers):
        self.answers, self.requests = list(answers), []
        self.messages = self.beta = self
        self.beta.messages = self

    def with_options(self, **_):
        return self

    def _next(self, request):
        self.requests.append(request)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer(request) if callable(answer) else answer

    def create(self, **request):
        return self._next(request)

    def stream(self, **request):
        final = self._next(request)

        class Stream:
            current_message_snapshot = final

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def __iter__(self):
                return iter([SimpleNamespace(type="text", text=final.content[-1].text)])

            def get_final_message(self):
                return final

        return Stream()


@pytest.fixture
def claude(monkeypatch):
    def install(answers):
        fake = FakeClaude(answers)
        monkeypatch.setattr(am, "_client", lambda base_url, profile: fake)
        return fake
    return install


def claude_config(**values):
    return StructuredConfig(**{
        "base_url": GATEWAY, "model": SONNET, "provider": "databricks_anthropic",
        "databricks_profile": "unity-gateway", "max_tokens": 2048, "context_window": 1_000_000,
        "tokenizer_mode": "bytes", "response_format": "json_schema", "attempts": 1,
        "timeout": 5, **values})


SCHEMA = {"type": "object", "additionalProperties": False, "required": ["terms"],
          "properties": {"terms": {"type": "array", "items": {"type": "string"}, "maxItems": 8,
                                   "description": "Words."}}}


def test_requests_become_cached_messages_with_effort_and_schema():
    body = {"model": SONNET, "max_tokens": 4096, "temperature": 1.0, "top_p": 0.95,
            "reasoning_effort": "medium", "stream": True, "stream_options": {},
            "response_format": {"type": "json_schema", "json_schema": {"schema": SCHEMA}},
            "messages": [{"role": "system", "content": "Rules."},
                         {"role": "user", "content": "Turn."}]}
    request = am.messages_request(body)
    assert request["system"] == [{"type": "text", "text": "Rules.",
                                  "cache_control": {"type": "ephemeral"}}]
    assert request["messages"] == [{"role": "user", "content": "Turn."}]
    assert request["output_config"]["effort"] == "medium"
    terms = request["output_config"]["format"]["schema"]["properties"]["terms"]
    assert "maxItems" not in terms and "8" in terms["description"]  # Moved, still enforced in code.
    assert "temperature" not in request and "top_p" not in request
    assert request["fallbacks"] == "default" and request["betas"] == [am.FALLBACK_BETA]
    assert "fallbacks" not in am.messages_request({**body, "model": HAIKU})
    with pytest.raises(ValueError, match="end with a user message"):
        am.messages_request({**body, "messages": body["messages"] + [
            {"role": "assistant", "content": "{"}]})
    with pytest.raises(ValueError, match="does not take these request fields: seed"):
        am.messages_request({**body, "seed": 3})
    with pytest.raises(ValueError, match="effort"):
        am.messages_request({**body, "reasoning_effort": "extreme"})


def test_responses_become_chat_completions():
    result = am.chat_completion(message('{"a": 1}', fallback=True), streamed=True)
    assert result["choices"][0] == {"index": 0, "finish_reason": "stop",
                                    "message": {"role": "assistant", "content": '{"a": 1}'}}
    assert result["usage"]["prompt_tokens"] == 1000 and result["usage"]["completion_tokens"] == 40
    assert result["usage"]["cache_read_input_tokens"] == 900
    assert result["transport"]["fallback_used"] is True
    for stop, finish in (("max_tokens", "length"), ("refusal", "refusal")):
        assert am.chat_completion(message("", stop=stop), streamed=False)[
            "choices"][0]["finish_reason"] == finish


def test_gateway_urls_and_fields_are_checked_before_any_request():
    assert am.gateway_url_error(GATEWAY) is None
    for url in ("http://adb-1.2.azuredatabricks.net/ai-gateway/anthropic",
                "https://adb-1.2.azuredatabricks.net/aigateway/anthropic",
                "https://evil.example.com/ai-gateway/anthropic",
                "https://adb-1.2.azuredatabricks.net:8443/ai-gateway/anthropic"):
        assert am.gateway_url_error(url), url
    with pytest.raises(ValueError, match="gateway URL"):
        StructuredClient(claude_config(base_url="https://evil.example.com/ai-gateway/anthropic"),
                         None)
    with pytest.raises(ValueError, match="tokenizer_mode='bytes'"):
        StructuredClient(claude_config(tokenizer_mode="endpoint"), None)
    with pytest.raises(ValueError, match="seed"):
        StructuredClient(claude_config(request_params={"seed": 1}), None)
    with pytest.raises(ValueError, match="structured requests only"):
        remote_auth.remote_api_key({"provider": "databricks"})


def test_structured_requests_go_through_claude(claude):
    fake = claude([message('{"terms": ["ct"]}')])
    client = qa._MeasuredClient(claude_config(request_params={"reasoning_effort": "low"}))
    value = client.complete("job", [{"role": "system", "content": "Rules."},
                                    {"role": "user", "content": "Turn."}], SCHEMA, lambda v: v)
    assert value == {"terms": ["ct"]}
    request = fake.requests[0]
    assert request["model"] == SONNET and request["output_config"]["effort"] == "low"
    assert client.prompt_tokens == 1000 and client.finish_reasons == ["stop"]
    with pytest.raises(EndpointError, match="no /models route"):
        client.discover()


def test_gateway_errors_keep_only_the_status(claude):
    response = SimpleNamespace(status_code=429, headers={"retry-after": "3"}, request=None)
    claude([anthropic.RateLimitError("provider text with patient words", response=response,
                                     body={"message": "provider text"})])
    client = qa._MeasuredClient(claude_config())
    with pytest.raises(EndpointError) as caught:
        client.complete("job", [{"role": "user", "content": "Turn."}], SCHEMA, lambda v: v)
    assert "provider text" not in str(caught.value)
    assert client.request_metrics[0]["http_status"] == 429
    assert client.request_metrics[0]["transport_error"] is True


def test_missing_login_is_a_clear_endpoint_error(monkeypatch):
    class Broken:
        def __init__(self, profile=None):
            raise ValueError("no ~/.databrickscfg profile")

    databricks_sdk = pytest.importorskip("databricks.sdk")
    monkeypatch.setattr(databricks_sdk, "WorkspaceClient", Broken)
    remote_auth.databricks_token_provider.cache_clear()
    am._client.cache_clear()
    client = StructuredClient(claude_config(), None)
    with pytest.raises(CredentialsError, match="databricks auth login --host"):
        client.complete("job", [{"role": "user", "content": "Turn."}], SCHEMA, lambda v: v)
    assert am.check_gateway(GATEWAY, SONNET, profile="unity-gateway")["error"].startswith(
        "Could not get a Databricks access token")


def test_token_provider_reads_the_profile_and_checks_the_workspace(monkeypatch):
    class Workspace:
        def __init__(self, profile=None):
            self.config = SimpleNamespace(
                host="https://adb-1234567890123456.7.azuredatabricks.net",
                authenticate=lambda: {"Authorization": "Bearer tok"})

    databricks_sdk = pytest.importorskip("databricks.sdk")
    monkeypatch.setattr(databricks_sdk, "WorkspaceClient", Workspace)
    remote_auth.databricks_token_provider.cache_clear()
    am._client.cache_clear()
    assert remote_auth.databricks_token_provider("unity-gateway").token() == "tok"
    assert am._client(GATEWAY, "unity-gateway") is am._client(GATEWAY, "unity-gateway")
    with pytest.raises(remote_auth.DatabricksCredentialsError, match="different workspace"):
        am._client("https://adb-9.9.azuredatabricks.net/ai-gateway/anthropic", "unity-gateway")
    am._client.cache_clear()


def sql_agent_config(provider_remote=True):
    config = load_default_preset()
    config.remote.update(enabled=True, provider="google_agent_platform",
                         google_project_id="fabricated-project",
                         server_urls=["https://aiplatform.googleapis.com/v1/x/endpoints/openapi"])
    section = config.workup_sql_agent
    section["remote"].update(model_name=SONNET, request_params={"max_tokens": 32000,
                                                                "reasoning_effort": "medium"},
                             extra_body={})
    if provider_remote:
        section["remote"].update(provider="databricks_anthropic", server_urls=[GATEWAY],
                                 databricks_profile="unity-gateway")
    section.update(sampling_profile="none", context_window=1_000_000)
    return config


def test_the_sql_agent_section_may_use_its_own_connection():
    resolved = resolve_agent_llm(sql_agent_config())
    assert resolved.provider == "databricks_anthropic" and resolved.base_url == GATEWAY
    assert resolved.databricks_profile == "unity-gateway" and resolved.google_project_id == ""
    assert resolved.model == SONNET and resolved.context_window == 1_000_000
    assert resolved.request_params == {"reasoning_effort": "medium"}
    # Each model's queries are cached apart.
    gemini = sql_agent_config(provider_remote=False)
    gemini.workup_sql_agent["remote"]["model_name"] = "google/gemini-4-hydrogen"
    assert agent_contract(sql_agent_config())["llm"] == {
        "provider": "databricks_anthropic", "model": SONNET, "effort": "medium"}
    assert agent_contract(gemini)["llm"] == {"provider": "google_agent_platform",
                                            "model": "google/gemini-4-hydrogen",
                                            "effort": "medium"}
    higher = sql_agent_config()
    higher.workup_sql_agent["remote"]["request_params"]["reasoning_effort"] = "high"
    assert agent_contract(higher)["llm"]["effort"] == "high"


def test_the_sql_agent_runs_on_claude(claude):
    from test_workup_sql import CT_SQL, PID, REQUIREMENT, action, rows

    database = build_extraction_database(rows(), patient_ids=[PID, "p2", "p3", "p4"])
    turns = [action("search", terms=["ct"]), action("submit", CT_SQL, summary="A chest CT."),
             action("accept")]
    fake = claude([message(json.dumps(turn)) for turn in turns])
    entries, metadata = compile_workup_sql(
        [REQUIREMENT], database=database, cache={}, llm=resolve_agent_llm(sql_agent_config()),
        return_metadata=True)
    assert entries[0]["termination"] == "accepted" and entries[0]["model"] == SONNET
    assert metadata["requests"] == 3 and metadata["prompt_tokens"] == 3000
    # The cached instructions block is the same every turn; the schema block follows.
    systems = {r["system"][0]["text"] for r in fake.requests}
    assert len(systems) == 1 and all(r["system"][0]["cache_control"] for r in fake.requests)
    assert all(r["system"][1]["text"].startswith("Response JSON schema") and
               "cache_control" not in r["system"][1] for r in fake.requests)
    assert all(r["output_config"]["format"]["type"] == "json_schema" for r in fake.requests)
    assert all(PID not in json.dumps(r) for r in fake.requests)
