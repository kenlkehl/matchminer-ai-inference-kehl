import pytest


@pytest.fixture(autouse=True)
def data_root(tmp_path, monkeypatch):
    """Training stages resolve their default paths under the data root (matchminer_ai.site); tests use tmp_path/data."""
    monkeypatch.setenv('MMAI_DATA_ROOT', str(tmp_path / 'data'))
    monkeypatch.delenv('MMAI_SITE_CONFIG', raising=False)
    return tmp_path / 'data'
