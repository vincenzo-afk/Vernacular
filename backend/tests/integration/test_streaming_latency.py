"""
Proves the pipeline STREAMS audio instead of buffering the clip.

The earlier tests would all pass against a buffered implementation
(collect everything, then emit). These do not: each uses a TTS that
yields one chunk and then BLOCKS on an event. If the orchestrator
streams, the first chunk reaches the consumer while the TTS is still
blocked; if it buffers, the consumer times out waiting. That is the
observable difference the ~2s latency budget depends on
(ARCHITECTURE.md §3, CLAUDE.md constraint #2).
"""

import asyncio
import json

import pytest

from app.pipeline.orchestrator import (
    AudioChunk,
    SegmentEnd,
    SegmentStart,
    SessionOrchestrator,
)
from app.pipeline.register_detector import RegisterDetector
from app.pipeline.stt import TranscriptEvent
from app.pipeline.translator import Translator
from tests.fakes.fake_llm import FakeLLMClient, ScriptedResponse

STYLE = {
    "tone": "neutral", "pace": "normal", "formality": "neutral",
    "emotion": "neutral", "sarcasm_score": 0.0, "emphasis_words": [],
    "pause_pattern": "natural", "confidence": 0.9,
}
REGISTER = json.dumps(STYLE)
TRANSLATION = json.dumps({"translated_text": "hola", "style": STYLE})


class ScriptedSTT:
    def __init__(self, texts):
        self._events = [TranscriptEvent(text=t, is_final=True) for t in texts]

    async def stream(self, audio_chunks):
        for e in self._events:
            yield e


async def _audio():
    yield b"\x00"


class GatedTTS:
    """Yields FIRST, then blocks until released, then yields SECOND."""

    def __init__(self):
        self.release = asyncio.Event()
        self.calls = 0

    async def synthesize(self, text, style, voice_id):
        self.calls += 1
        yield b"FIRST"
        await self.release.wait()
        yield b"SECOND"


class FailsMidStreamTTS:
    async def synthesize(self, text, style, voice_id):
        yield b"PARTIAL"
        raise RuntimeError("provider died mid-clip")


class GoodTTS:
    async def synthesize(self, text, style, voice_id):
        yield b"FALLBACK"


def _orch(tts, fallback=None, texts=("one",)):
    n = len(texts)
    return SessionOrchestrator(
        stt=ScriptedSTT(list(texts)),
        register_detector=RegisterDetector(
            llm_client=FakeLLMClient([ScriptedResponse(REGISTER)] * n)
        ),
        translator=Translator(
            llm_client=FakeLLMClient([ScriptedResponse(TRANSLATION)] * n)
        ),
        tts=tts,
        fallback_tts=fallback,
        target_language="es",
        voice_id="v",
    )


@pytest.mark.asyncio
async def test_first_audio_chunk_is_delivered_while_tts_is_still_blocked():
    tts = GatedTTS()
    stream = _orch(tts).run_stream(_audio())

    async def first_events():
        got = []
        async for ev in stream:
            got.append(ev)
            if isinstance(ev, AudioChunk):
                return got  # stop as soon as audio appears

    # TTS is blocked and NOT released. A buffering pipeline would hang
    # here until timeout; a streaming one yields Start + FIRST at once.
    got = await asyncio.wait_for(first_events(), timeout=1.0)

    assert isinstance(got[0], SegmentStart)
    assert isinstance(got[1], AudioChunk) and got[1].data == b"FIRST"
    assert not tts.release.is_set()  # proves TTS had not finished
    tts.release.set()
    await stream.aclose()


@pytest.mark.asyncio
async def test_remaining_chunks_and_end_follow_once_tts_resumes():
    tts = GatedTTS()
    tts.release.set()
    events = [e async for e in _orch(tts).run_stream(_audio())]

    kinds = [type(e).__name__ for e in events]
    assert kinds == ["SegmentStart", "AudioChunk", "AudioChunk", "SegmentEnd"]
    assert [e.data for e in events if isinstance(e, AudioChunk)] == [
        b"FIRST", b"SECOND",
    ]


@pytest.mark.asyncio
async def test_time_to_first_byte_is_recorded_separately_from_total_tts():
    orch = _orch(GatedTTS())
    orch._tts.release.set()
    _ = [e async for e in orch.run_stream(_audio())]

    stages = [t.stage for t in orch.timings]
    assert "tts_first_byte" in stages
    assert "tts" in stages
    ttfb = next(t for t in orch.timings if t.stage == "tts_first_byte")
    total = next(t for t in orch.timings if t.stage == "tts")
    assert ttfb.duration_ms <= total.duration_ms


@pytest.mark.asyncio
async def test_failure_before_any_audio_fails_over_and_flags_degraded():
    class FailsImmediately:
        async def synthesize(self, text, style, voice_id):
            raise RuntimeError("down")
            yield b""  # pragma: no cover - makes this an async generator

    events = [
        e async for e in _orch(FailsImmediately(), GoodTTS()).run_stream(_audio())
    ]
    start = events[0]
    assert isinstance(start, SegmentStart) and start.degraded is True
    assert [e.data for e in events if isinstance(e, AudioChunk)] == [b"FALLBACK"]


@pytest.mark.asyncio
async def test_mid_stream_failure_ends_segment_without_splicing_a_second_voice():
    """
    Once PARTIAL has been sent it can't be taken back. The fallback
    must NOT be spliced into the same utterance; the segment ends with
    what was delivered, and the session (and later segments) survive.
    """
    events = [
        e
        async for e in _orch(
            FailsMidStreamTTS(), GoodTTS(), texts=("one", "two")
        ).run_stream(_audio())
    ]

    chunks = [e.data for e in events if isinstance(e, AudioChunk)]
    # Segment one: only the partial audio, no fallback voice spliced in.
    # Segment two: provider now marked failed -> clean fallback audio.
    assert chunks == [b"PARTIAL", b"FALLBACK"]
    ends = [e for e in events if isinstance(e, SegmentEnd)]
    assert len(ends) == 2  # both segments terminated cleanly
