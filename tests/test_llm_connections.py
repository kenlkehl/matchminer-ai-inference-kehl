import pytest

from matchminer_ai.llm import connections


def test_normalize_endpoint_accepts_addresses_and_rejects_credentials():
    assert connections.normalize_endpoint("gpu-server:8000") == "http://gpu-server:8000/v1"
    assert connections.normalize_endpoint("https://host.example/v1/") == "https://host.example/v1"
    for bad in ("", "http://user:pw@host:8000", "http://host/v1?x=1", "ftp://host", "http://host:99999"):
        with pytest.raises(ValueError):
            connections.normalize_endpoint(bad)


def test_vertex_profiles_follow_the_site_project(tmp_path, monkeypatch):
    site = tmp_path / "site.yaml"
    site.write_text("vertex:\n  project: example-project\n")
    monkeypatch.setenv("MMAI_SITE_CONFIG", str(site))
    profile = connections.vertex_profile("gemma_vertex")
    assert "/projects/example-project/" in profile["endpoint"]
    gemma, gemini = connections.vertex_remote("gemma_vertex"), connections.vertex_remote("gemini4_vertex")
    assert gemma["google_project_id"] == "example-project" and gemma["request_start_interval_seconds"] == 2.0
    assert "request_start_interval_seconds" not in gemini
    assert connections.vertex_request_params("gemini4_vertex", 99999) == {"max_tokens": 65536,
                                                                          "reasoning_effort": "high"}
    assert connections.vertex_extra_body("gemini4_vertex", True) == {}
    assert connections.vertex_extra_body("gemma_vertex", False) == {"chat_template_kwargs": {"enable_thinking": False}}
