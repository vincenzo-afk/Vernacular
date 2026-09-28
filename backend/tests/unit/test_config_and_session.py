"""
Tests for:
1. app/config.py — settings must load lazily (on first real use), not
   eagerly at import time. See config.py's get_settings() docstring
   for why: every pipeline module reaches this transitively via
   app/ws/session.py, so an eager Settings() at module scope would
   crash on plain import in any environment without a populated .env
   (including this very test run, before this was fixed).
2. app/ws/session.py::_build_llm_client — provider selection and the
   NotImplementedError for unimplemented providers (Groq, Google).
"""

import pytest

from app import config
from app.ws.session import _build_llm_client


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """
    get_settings() is lru_cache'd (a real singleton), which would leak
    across tests that set different env vars via monkeypatch. Clear it
    before and after every test in this module.
    """
    config.get_settings.cache_clear()
    yield
    config.get_settings.cache_clear()


def test_get_settings_raises_clearly_when_required_env_vars_missing(monkeypatch):
    from pydantic import ValidationError

    for var in (
        "ASSEMBLYAI_API_KEY",
        "ELEVENLABS_API_KEY",
        "LLM_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)

    with pytest.raises(ValidationError):
        config.get_settings()


def test_get_settings_succeeds_with_required_env_vars(monkeypatch):
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "fake-aai-key")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "fake-el-key")
    monkeypatch.setenv("LLM_API_KEY", "fake-llm-key")

    settings = config.get_settings()

    assert settings.assemblyai_api_key == "fake-aai-key"
    assert settings.llm_provider == "openai"  # default


def test_get_settings_is_cached_returns_same_instance(monkeypatch):
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "fake-aai-key")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "fake-el-key")
    monkeypatch.setenv("LLM_API_KEY", "fake-llm-key")

    first = config.get_settings()
    second = config.get_settings()

    assert first is second


class _FakeSettings:
    """Minimal stand-in for config.Settings, for _build_llm_client tests."""

    def __init__(self, llm_provider: str, llm_api_key: str = "k", llm_model: str = "m"):
        self.llm_provider = llm_provider
        self.llm_api_key = llm_api_key
        self.llm_model = llm_model


def test_build_llm_client_raises_for_unimplemented_provider():
    settings = _FakeSettings(llm_provider="groq")
    with pytest.raises(NotImplementedError, match="groq"):
        _build_llm_client(settings)


def test_build_llm_client_raises_for_unknown_provider():
    settings = _FakeSettings(llm_provider="not-a-real-provider")
    with pytest.raises(NotImplementedError):
        _build_llm_client(settings)


def test_build_llm_client_openai_import_boundary(monkeypatch):
    """
    Simulate an unavailable live SDK explicitly so this test remains
    deterministic whether the sandbox has optional provider packages
    installed or not. The important contract is that adapter creation
    crosses the real import boundary rather than silently no-op'ing.
    """
    settings = _FakeSettings(llm_provider="openai")
    real_import = __import__

    def blocked_openai_import(name, *args, **kwargs):
        if name == "openai":
            raise ModuleNotFoundError("simulated optional dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", blocked_openai_import)
    with pytest.raises(ModuleNotFoundError):
        _build_llm_client(settings)


def test_build_llm_client_anthropic_requires_anthropic_package_installed():
    settings = _FakeSettings(llm_provider="anthropic")
    with pytest.raises(ModuleNotFoundError):
        _build_llm_client(settings)
