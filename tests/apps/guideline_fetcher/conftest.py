"""Shared test setup. PHI-mode tests use fabricated tokens and records only."""

import os

import pytest

pytest.importorskip("fastapi")
# The Databricks gateway host is a site value; tests use a placeholder (read when settings is imported).
os.environ.setdefault("GUIDELINE_DATABRICKS_HOST", "https://workspace.example.cloud.databricks.com")
from fastapi.testclient import TestClient  # noqa: E402

from matchminer_ai.apps.guideline_fetcher.settings import Settings

ACCESS_TOKEN = "fabricated-test-access-token-0123456789abcdef"
# Starlette's TestClient sends Host: testserver; PHI mode allows loopback names only.
TEST_HOSTS = ["testserver", "127.0.0.1", "localhost"]


def phi_settings(**values):
    """PHI-mode settings that also accept the test client's Host header."""
    return Settings(phi=True, **{"allowed_hosts": TEST_HOSTS, **values})


def mode_settings(phi, **values):
    return phi_settings(**values) if phi else Settings(**values)


@pytest.fixture(autouse=True)
def scratch_data_root(tmp_path, monkeypatch):
    """Default paths resolve under a scratch data root, never a real one."""
    monkeypatch.setenv("MMAI_DATA_ROOT", str(tmp_path / "data"))


@pytest.fixture(autouse=True)
def fabricated_access_token(monkeypatch):
    # PHI-mode apps load the token when created; tests never create or read the
    # operator's token file.
    monkeypatch.setenv("GUIDELINE_ACCESS_TOKEN", ACCESS_TOKEN)


@pytest.fixture(autouse=True)
def authenticated_test_clients(monkeypatch):
    """Clients act as a signed-in PHI-mode user (the non-PHI app ignores the header).

    Tests of the sign-in boundary remove the header.
    """
    original = TestClient.__init__

    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.headers["Authorization"] = f"Bearer {ACCESS_TOKEN}"

    monkeypatch.setattr(TestClient, "__init__", init)
