import pytest

from matchminer_ai import site


def test_site_config_dotted_keys_and_require(tmp_path, monkeypatch):
    path = tmp_path / "site.yaml"
    path.write_text("endpoints:\n  teacher:\n    url: http://teacher.example:8000/v1\n")
    monkeypatch.setenv(site.SITE_CONFIG_ENV, str(path))
    config = site.load_site_config()
    assert config.get("endpoints.teacher.url") == "http://teacher.example:8000/v1"
    assert config.get("endpoints.judge.url") is None
    with pytest.raises(KeyError, match="endpoints.judge.url"):
        config.require("endpoints.judge.url")


def test_no_site_config_is_empty(monkeypatch):
    monkeypatch.delenv(site.SITE_CONFIG_ENV, raising=False)
    assert not site.load_site_config()


def test_data_root_has_no_default_and_refuses_the_checkout(tmp_path, monkeypatch):
    monkeypatch.delenv(site.DATA_ROOT_ENV, raising=False)
    monkeypatch.delenv(site.SITE_CONFIG_ENV, raising=False)
    with pytest.raises(ValueError, match="No data root"):
        site.data_root()
    assert site.data_root(tmp_path) == tmp_path.resolve()
    checkout = site.source_checkout()
    if checkout is not None:
        with pytest.raises(ValueError, match="inside the source checkout"):
            site.data_root(checkout / "data")
