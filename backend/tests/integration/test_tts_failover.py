"""
Sticky TTS provider failover, per the tts.py failure-handling contract
in backend/app/pipeline/AGENTS.md:

- On primary failure, switch to the fallback for the REST of the
  session (no flapping back -- jarring voice changes).
- The segment that triggered the failure is retried on the fallback,
  not lost.
- Fallback-produced segments are flagged degraded so the UI never
  presents reduced fidelity as full quality (ARCHITECTURE.md §5).
"""

import json

import pytest

from app.pipeline.orchestrator import SessionOrchestrator
from app.pipeline.register_detector import RegisterDetector
from app.pipeline.stt import TranscriptEvent
from app.pipeline.translator import Translator
from tests.fakes.fake_llm import FakeLLMClient, ScriptedResponse
from tests.fakes.fake_tts import FakeTTS

REGISTER = json.dumps(
    {
        "tone": "neutral", "pace": "normal", "formality": "neutral",
        "emotion": "neutral", "sarcasm_score": 0.0, "emphasis_words": [],
        "pause_pattern": "natural", "confidence": 0.9,
    }
)
TRANSLATION = json.dumps(
    {
        "translated_text": "hola",
        "style": {
            "tone": "neutral", "pace": "normal", "formality": "neutral",
            "emotion": "neutral", "sarcasm_score": 0.0, "emphasis_words": [],
            "pause_pattern": "natural", "confidence": 0.9,
        },
    }
)


class ScriptedSTT:
    def __init__(self, texts):
        self._events = [TranscriptEvent(text=t, is_final=True) for t in texts]

    async def stream(self, audio_chunks):
        for e in self._events:
            yield e


async def _audio():
    yield b"\x00"


class FlakyThenGoodTTS(FakeTTS):
    """Fails on the first call, would succeed afterwards -- proves the
    orchestrator does NOT flap back to it after failing over."""

    def __init__(self):
        super().__init__(chunk=b"PRIMARY!")
        self._n = 0

    async def synthesize(self, text, style, voice_id):
        self._n += 1
        if self._n == 1:
            self.calls.append(None)
            raise RuntimeError("primary down")
        async for c in super().synthesize(text, style, voice_id):
            yield c


def _orch(texts, primary, fallback):
    n = len(texts)
    return SessionOrchestrator(
        stt=ScriptedSTT(texts),
        register_detector=RegisterDetector(
            llm_client=FakeLLMClient([ScriptedResponse(REGISTER)] * n)
        ),
        translator=Translator(
            llm_client=FakeLLMClient([ScriptedResponse(TRANSLATION)] * n)
        ),
        tts=primary,
        fallback_tts=fallback,
        target_language="es",
        voice_id="v",
    )


@pytest.mark.asyncio
async def test_healthy_primary_is_not_degraded_and_fallback_unused():
    primary, fallback = FakeTTS(chunk=b"AAAA"), FakeTTS(chunk=b"BBBB")
    segs = [s async for s in _orch(["one"], primary, fallback).run(_audio())]

    assert len(segs) == 1
    assert segs[0].degraded is False
    assert fallback.calls == []


@pytest.mark.asyncio
async def test_primary_failure_retries_same_segment_on_fallback_and_flags_degraded():
    primary = FlakyThenGoodTTS()
    fallback = FakeTTS(chunk=b"FALLBACK")
    segs = [s async for s in _orch(["one"], primary, fallback).run(_audio())]

    # The segment that hit the failure is NOT lost.
    assert len(segs) == 1
    assert b"".join(segs[0].audio_chunks) == b"FALLBACK"
    assert segs[0].degraded is True


@pytest.mark.asyncio
async def test_failover_is_sticky_does_not_flap_back_to_recovered_primary():
    primary = FlakyThenGoodTTS()  # fails once, then would work fine
    fallback = FakeTTS(chunk=b"FALLBACK")
    segs = [
        s async for s in _orch(["one", "two", "three"], primary, fallback).run(_audio())
    ]

    assert len(segs) == 3
    assert all(s.degraded for s in segs)
    assert all(b"".join(s.audio_chunks) == b"FALLBACK" for s in segs)
    # Primary was tried exactly once (the failure), never again.
    assert primary._n == 1


@pytest.mark.asyncio
async def test_no_fallback_configured_drops_segment_without_killing_session():
    primary = FakeTTS(fail=True)
    segs = [s async for s in _orch(["one", "two"], primary, None).run(_audio())]

    assert segs == []  # both dropped, but run() completed normally


@pytest.mark.asyncio
async def test_fallback_also_failing_drops_segment_without_killing_session():
    primary, fallback = FakeTTS(fail=True), FakeTTS(fail=True)
    segs = [s async for s in _orch(["one", "two"], primary, fallback).run(_audio())]

    assert segs == []
