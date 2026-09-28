"""
Both TTS providers must be explicitly asked for the SAME raw PCM16
format that the wire protocol declares to the client (see
pipeline/tts.py AUDIO_* constants). Before this was pinned, neither
call specified a format, so both silently returned MP3 -- which the
browser cannot decode chunk-by-chunk and the client had no way to
identify.
"""

import pytest

from app.pipeline import tts
from app.pipeline.tts import ElevenLabsTTS, FallbackTTS
from app.schemas.style_metadata import StyleMetadata
from tests.unit.test_fallback_tts import _FakeOpenAIClient


def test_constants_are_consistent_with_provider_definitions():
    # OpenAI defines response_format="pcm" as 24 kHz mono 16-bit LE;
    # ElevenLabs' pcm_24000 is 24 kHz. They must agree with our
    # declared contract or playback pitch/speed would be wrong.
    assert tts.AUDIO_SAMPLE_RATE == 24000
    assert tts.ELEVENLABS_OUTPUT_FORMAT == f"pcm_{tts.AUDIO_SAMPLE_RATE}"
    assert tts.OPENAI_RESPONSE_FORMAT == "pcm"


@pytest.mark.asyncio
async def test_fallback_requests_pcm_from_openai(monkeypatch):
    fb = FallbackTTS(provider="openai", api_key="k")
    client = _FakeOpenAIClient()
    monkeypatch.setattr(fb, "_get_client", lambda: client)

    async for _ in fb.synthesize("hola", StyleMetadata(), "default"):
        pass

    assert client.speech.calls[0]["response_format"] == "pcm"


class _FakeElevenStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        async def gen():
            for c in self._chunks:
                yield c

        return gen()


class _FakeElevenTTSNamespace:
    def __init__(self):
        self.calls = []

    def convert_as_stream(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeElevenStream([b"\x00\x01"])


class _FakeElevenClient:
    def __init__(self):
        self.text_to_speech = _FakeElevenTTSNamespace()


@pytest.mark.asyncio
async def test_elevenlabs_requests_pcm_24000(monkeypatch):
    el = ElevenLabsTTS(api_key="k")
    client = _FakeElevenClient()
    monkeypatch.setattr(el, "_get_client", lambda: client)

    chunks = [c async for c in el.synthesize("hola", StyleMetadata(), "voice123")]

    assert chunks == [b"\x00\x01"]
    call = client.text_to_speech.calls[0]
    assert call["output_format"] == "pcm_24000"
    assert call["voice_id"] == "voice123"
