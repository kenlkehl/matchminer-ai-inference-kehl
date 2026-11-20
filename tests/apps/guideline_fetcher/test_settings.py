import sys
from types import SimpleNamespace

import pytest

from matchminer_ai.apps.guideline_fetcher.settings import Settings


@pytest.mark.parametrize("available, expected", [(True, "cuda"), (False, "cpu")])
@pytest.mark.parametrize("requested", [None, "auto"])
def test_automatic_device_reaches_both_local_models(monkeypatch, available, expected, requested):
    if requested is None:
        monkeypatch.delenv("MMAI_DEVICE", raising=False)
    else:
        monkeypatch.setenv("MMAI_DEVICE", requested)
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: available))
    )
    config = SimpleNamespace(
        embedding={}, raw={"decision_model": {}}, remote={}, patient={"remote": {}}
    )
    monkeypatch.setitem(
        sys.modules, "matchminer_ai", SimpleNamespace(load_default_preset=lambda: config)
    )
    settings = Settings(llm_backend="local")
    resolved = settings.inference_config()
    assert settings.device == expected
    assert resolved.embedding["device"] == expected
    assert resolved.raw["decision_model"]["device"] == expected


@pytest.mark.parametrize("requested", ["cpu", "cuda:1"])
def test_explicit_device_does_not_probe_or_override_choice(monkeypatch, requested):
    monkeypatch.setenv("MMAI_DEVICE", requested)
    monkeypatch.setitem(sys.modules, "torch", None)
    assert Settings().device == requested


def test_each_mode_keeps_its_own_cache(tmp_path, monkeypatch):
    from matchminer_ai.apps.guideline_fetcher import settings

    NO_PHI_CACHE = settings.data_path(settings.NO_PHI_CACHE)
    PHI_CACHE = settings.data_path(settings.PHI_CACHE)
    monkeypatch.setenv("MMAI_DEVICE", "cpu")
    monkeypatch.delenv("GUIDELINE_CACHE", raising=False)
    assert Settings().cache == NO_PHI_CACHE and "no_phi" in NO_PHI_CACHE.parts
    assert Settings(phi=True).cache == PHI_CACHE and "phi" in PHI_CACHE.parts
    assert Settings(phi=True).inference_config().model_metadata_cache_dir == str(PHI_CACHE)
    monkeypatch.setenv("GUIDELINE_CACHE", str(tmp_path / "data/no_phi/cache"))
    assert Settings().cache == tmp_path / "data/no_phi/cache"
    with pytest.raises(ValueError, match="no_phi"):
        Settings(phi=True)
    monkeypatch.setenv("GUIDELINE_CACHE", str(tmp_path / "data/phi/cache"))
    assert Settings(phi=True).cache == tmp_path / "data/phi/cache"
    with pytest.raises(ValueError, match="phi folder"):
        Settings()
