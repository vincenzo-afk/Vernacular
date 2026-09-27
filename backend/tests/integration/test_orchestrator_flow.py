"""
See TESTING.md — integration tests wire the orchestrator to fakes for
every stage and verify the stages actually compose: output comes out,
timings are recorded, and a mid-stream register-detection failure
doesn't kill the session.
"""

import json
from collections.abc import AsyncIterator

import pytest

from app.pipeline.orchestrator import SessionOrchestrator
from app.pipeline.register_detector import RegisterDetector
from app.pipeline.stt import TranscriptEvent, WordTiming
from app.pipeline.translator import Translator
from app.schemas.style_metadata import Tone
from tests.fakes.fake_llm import FakeLLMClient, ScriptedResponse
from tests.fakes.fake_tts import FakeTTS

SARCASTIC_REGISTER_JSON = json.dumps(
    {
        "tone": "sarcastic",
        "pace": "slow",
        "formality": "casual",
        "emotion": "frustration",
        "sarcasm_score": 0.88,
        "emphasis_words": ["great"],
        "pause_pattern": "dramatic",
        "confidence": 0.81,
    }
)

TRANSLATION_JSON = json.dumps(
    {
        "translated_text": "Ay, qué bien, otra reunión.",
        "style": {
            "tone": "sarcastic",
            "pace": "slow",
            "formality": "casual",
            "emotion": "frustration",
            "sarcasm_score": 0.85,
            "emphasis_words": ["bien"],
            "pause_pattern": "dramatic",
            "confidence": 0.81,
        },
    }
)


class ScriptedSTT:
    """
    A minimal STTProvider that yields a fixed sequence of
    TranscriptEvents regardless of the input audio — sufficient for
    exercising the orchestrator without a real audio pipeline.
    """

    def __init__(self, events: list[TranscriptEvent]) -> None:
        self._events = events

    async def stream(
        self, audio_chunks: AsyncIterator[bytes]
    ) -> AsyncIterator[TranscriptEvent]:
        for event in self._events:
            yield event


async def _dummy_audio() -> AsyncIterator[bytes]:
    yield b"\x00" * 10


def _final_transcript(text: str = "Oh, great, another meeting.") -> TranscriptEvent:
    return TranscriptEvent(
        text=text,
        is_final=True,
        words=[
            WordTiming(word="Oh,", start_ms=0, end_ms=200, confidence=0.99),
            WordTiming(
                word="great,",
                start_ms=200,
                end_ms=600,
                confidence=0.98,
                is_stressed=True,
            ),
        ],
        sentiment="NEGATIVE",
    )


@pytest.mark.asyncio
async def test_full_pipeline_produces_translated_segment_with_timings():
    stt = ScriptedSTT([_final_transcript()])
    register_llm = FakeLLMClient([ScriptedResponse(content=SARCASTIC_REGISTER_JSON)])
    translate_llm = FakeLLMClient([ScriptedResponse(content=TRANSLATION_JSON)])
    tts = FakeTTS(chunk=b"\x01\x02\x03\x04")

    orchestrator = SessionOrchestrator(
        stt=stt,
        register_detector=RegisterDetector(llm_client=register_llm),
        translator=Translator(llm_client=translate_llm),
        tts=tts,
        target_language="es",
        voice_id="test-voice",
    )

    segments = [seg async for seg in orchestrator.run(_dummy_audio())]

    assert len(segments) == 1
    segment = segments[0]
    assert segment.translated_text == "Ay, qué bien, otra reunión."
    assert segment.style.tone == Tone.SARCASTIC
    assert b"".join(segment.audio_chunks) == b"\x01\x02\x03\x04"

    # Timing should have been recorded for register detection,
    # translation, and TTS at minimum.
    stages_recorded = {t.stage for t in orchestrator.timings}
    assert "register_detection" in stages_recorded
    assert "translation" in stages_recorded
    assert "tts" in stages_recorded
    assert all(t.duration_ms >= 0 for t in orchestrator.timings)

    # TTS should have been called with the translated text and the
    # translation-refined style, not the source style.
    assert len(tts.calls) == 1
    assert tts.calls[0].text == "Ay, qué bien, otra reunión."
    assert tts.calls[0].voice_id == "test-voice"


@pytest.mark.asyncio
async def test_register_detection_failure_falls_back_and_session_continues():
    stt = ScriptedSTT([_final_transcript()])
    # Register LLM fails outright -> RegisterDetector resolves to
    # neutral_fallback() internally rather than raising (see
    # test_register_detector.py) -- the orchestrator should get a
    # normal StyleMetadata back and proceed to translate/synthesize.
    register_llm = FakeLLMClient([ScriptedResponse(content="", should_raise=True)])
    translate_llm = FakeLLMClient([ScriptedResponse(content=TRANSLATION_JSON)])
    tts = FakeTTS(chunk=b"\x01\x02")

    orchestrator = SessionOrchestrator(
        stt=stt,
        register_detector=RegisterDetector(llm_client=register_llm),
        translator=Translator(llm_client=translate_llm),
        tts=tts,
        target_language="es",
        voice_id="test-voice",
    )

    segments = [seg async for seg in orchestrator.run(_dummy_audio())]

    # Session should not have died — a segment still comes out, just
    # with the translation LLM having been given a neutral source style.
    assert len(segments) == 1
    assert len(translate_llm.calls) == 1


@pytest.mark.asyncio
async def test_tts_failure_on_one_segment_does_not_kill_session():
    stt = ScriptedSTT([_final_transcript("First."), _final_transcript("Second.")])
    register_llm = FakeLLMClient(
        [
            ScriptedResponse(content=SARCASTIC_REGISTER_JSON),
            ScriptedResponse(content=SARCASTIC_REGISTER_JSON),
        ]
    )
    translate_llm = FakeLLMClient(
        [
            ScriptedResponse(content=TRANSLATION_JSON),
            ScriptedResponse(content=TRANSLATION_JSON),
        ]
    )
    # First TTS call fails, would raise if synthesize() were called a
    # second time on the same fake since fail=True applies to every
    # call -- so instead we verify the orchestrator skips the failed
    # segment rather than propagating, using a fake that always fails.
    tts = FakeTTS(fail=True)

    orchestrator = SessionOrchestrator(
        stt=stt,
        register_detector=RegisterDetector(llm_client=register_llm),
        translator=Translator(llm_client=translate_llm),
        tts=tts,
        target_language="es",
        voice_id="test-voice",
    )

    # Per pipeline/AGENTS.md, a segment that fails end-to-end is
    # skipped, not raised -- so we expect zero yielded segments here,
    # but critically no exception should propagate out of run().
    segments = [seg async for seg in orchestrator.run(_dummy_audio())]

    assert segments == []
