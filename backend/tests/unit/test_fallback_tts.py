"""
Tests for FallbackTTS. These do NOT import the real `openai` package
(it may not even be installed in an offline test environment — see
TESTING.md) — instead they monkeypatch FallbackTTS._get_client() to
return a small fake object shaped like the bits of the OpenAI SDK this
class actually calls.

The specific thing under test: FallbackTTS must never pass a
provider-foreign voice_id (e.g. an ElevenLabs voice/clone ID) through
to OpenAI's API, which only accepts its own fixed voice name enum. See
the FallbackTTS class docstring in app/pipeline/tts.py for why this
matters -- passing an incompatible ID through would make the fallback
path fail exactly when it's needed most (mid-demo, primary provider
already down).
"""

import pytest

from app.pipeline.tts import FallbackTTS
from app.schemas.style_metadata import Pace, StyleMetadata


class _FakeStreamingResponse:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def iter_bytes(self):
        for chunk in self._chunks:
            yield chunk


class _FakeSpeechEndpoint:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self._chunks = [b"\x01\x02"]

    class _StreamingResponseNamespace:
        def __init__(self, parent: "_FakeSpeechEndpoint") -> None:
            self._parent = parent

        def create(self, **kwargs):
            self._parent.calls.append(kwargs)
            return _FakeStreamingResponse(self._parent._chunks)

    @property
    def with_streaming_response(self):
        return self._StreamingResponseNamespace(self)


class _FakeAudioNamespace:
    def __init__(self, speech: _FakeSpeechEndpoint) -> None:
        self.speech = speech


class _FakeOpenAIClient:
    def __init__(self) -> None:
        self.speech = _FakeSpeechEndpoint()
        self.audio = _FakeAudioNamespace(self.speech)


@pytest.fixture
def fallback_tts_with_fake_client(monkeypatch):
    tts = FallbackTTS(provider="openai", api_key="fake-key")
    fake_client = _FakeOpenAIClient()
    monkeypatch.setattr(tts, "_get_client", lambda: fake_client)
    return tts, fake_client


@pytest.mark.asyncio
async def test_ignores_elevenlabs_style_voice_id_and_uses_default(
    fallback_tts_with_fake_client,
):
    tts, fake_client = fallback_tts_with_fake_client

    chunks = [
        chunk
        async for chunk in tts.synthesize(
            text="hola",
            style=StyleMetadata(pace=Pace.NORMAL),
            voice_id="21m00Tcm4TlvDq8ikWAM",  # realistic ElevenLabs-shaped ID
        )
    ]

    assert chunks == [b"\x01\x02"]
    assert len(fake_client.speech.calls) == 1
    call = fake_client.speech.calls[0]
    assert call["voice"] == FallbackTTS.DEFAULT_VOICE
    assert call["voice"] != "21m00Tcm4TlvDq8ikWAM"


@pytest.mark.asyncio
async def test_default_voice_id_also_maps_to_default_voice(
    fallback_tts_with_fake_client,
):
    tts, fake_client = fallback_tts_with_fake_client

    async for _ in tts.synthesize(
        text="hola", style=StyleMetadata(), voice_id="default"
    ):
        pass

    assert fake_client.speech.calls[0]["voice"] == FallbackTTS.DEFAULT_VOICE


@pytest.mark.asyncio
async def test_pace_maps_to_speed_within_openai_supported_range(
    fallback_tts_with_fake_client,
):
    tts, fake_client = fallback_tts_with_fake_client

    async for _ in tts.synthesize(
        text="hola", style=StyleMetadata(pace=Pace.RUSHED), voice_id="default"
    ):
        pass

    speed = fake_client.speech.calls[0]["speed"]
    # OpenAI's documented supported range is 0.25-4.0 -- our pace
    # mapping should always stay well inside it.
    assert 0.25 <= speed <= 4.0


def test_unsupported_provider_raises_not_implemented():
    tts = FallbackTTS(provider="cartesia", api_key="fake-key")
    with pytest.raises(NotImplementedError):
        tts._get_client()
