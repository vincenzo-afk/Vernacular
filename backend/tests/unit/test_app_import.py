"""
Confirms the whole app import chain (main.py -> ws/session.py ->
config.py) doesn't eagerly require settings at import time, and that
the basic FastAPI app + health endpoint work with no API keys
configured. Settings are only actually needed once a WebSocket session
starts, which is a live-API-touching path this test deliberately does
not exercise -- see TESTING.md.
"""

from fastapi.testclient import TestClient

from app import config
from app.main import app


def test_app_imports_and_health_endpoint_works_without_env_vars(monkeypatch):
    for var in ("ASSEMBLYAI_API_KEY", "ELEVENLABS_API_KEY", "LLM_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    config.get_settings.cache_clear()

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
