"""
See TESTING.md and backend/app/pipeline/AGENTS.md — RegisterDetector
must never raise past its own boundary, and must fall back to
StyleMetadata.neutral_fallback() on both LLM failure and low confidence.
"""

import json

import pytest

from app.pipeline.register_detector import RegisterDetector
from app.pipeline.stt import TranscriptEvent, WordTiming
from app.schemas.style_metadata import StyleMetadata, Tone
from tests.fakes.fake_llm import FakeLLMClient, ScriptedResponse


def _sarcastic_transcript() -> TranscriptEvent:
    return TranscriptEvent(
        text="Oh, great, another meeting.",
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
            WordTiming(word="another", start_ms=1200, end_ms=1500, confidence=0.97),
            WordTiming(word="meeting.", start_ms=1500, end_ms=1900, confidence=0.98),
        ],
        sentiment="NEGATIVE",
    )


HIGH_CONFIDENCE_SARCASTIC_JSON = json.dumps(
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

LOW_CONFIDENCE_JSON = json.dumps(
    {
        "tone": "neutral",
        "pace": "normal",
        "formality": "formal",
        "emotion": "neutral",
        "sarcasm_score": 0.1,
        "emphasis_words": [],
        "pause_pattern": "natural",
        "confidence": 0.35,
    }
)


@pytest.mark.asyncio
async def test_high_confidence_result_is_returned_as_is():
    llm = FakeLLMClient([ScriptedResponse(content=HIGH_CONFIDENCE_SARCASTIC_JSON)])
    detector = RegisterDetector(llm_client=llm)

    result = await detector.detect(_sarcastic_transcript())

    assert result.tone == Tone.SARCASTIC
    assert result.sarcasm_score == 0.88
    assert result.emphasis_words == ["great"]
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_low_confidence_result_falls_back_to_neutral():
    llm = FakeLLMClient([ScriptedResponse(content=LOW_CONFIDENCE_JSON)])
    detector = RegisterDetector(llm_client=llm)

    result = await detector.detect(_sarcastic_transcript())

    assert result == StyleMetadata.neutral_fallback()


@pytest.mark.asyncio
async def test_llm_exception_falls_back_to_neutral_without_raising():
    llm = FakeLLMClient([ScriptedResponse(content="", should_raise=True)])
    detector = RegisterDetector(llm_client=llm)

    # Must not raise — this is the core contract in pipeline/AGENTS.md.
    result = await detector.detect(_sarcastic_transcript())

    assert result == StyleMetadata.neutral_fallback()


@pytest.mark.asyncio
async def test_malformed_json_falls_back_to_neutral_without_raising():
    llm = FakeLLMClient([ScriptedResponse(content="not valid json at all")])
    detector = RegisterDetector(llm_client=llm)

    result = await detector.detect(_sarcastic_transcript())

    assert result == StyleMetadata.neutral_fallback()


@pytest.mark.asyncio
async def test_markdown_fenced_json_is_parsed():
    fenced = f"```json\n{HIGH_CONFIDENCE_SARCASTIC_JSON}\n```"
    llm = FakeLLMClient([ScriptedResponse(content=fenced)])
    detector = RegisterDetector(llm_client=llm)

    result = await detector.detect(_sarcastic_transcript())

    assert result.tone == Tone.SARCASTIC


@pytest.mark.asyncio
async def test_empty_transcript_short_circuits_without_calling_llm():
    llm = FakeLLMClient([])  # no scripted responses — would raise if called
    detector = RegisterDetector(llm_client=llm)

    result = await detector.detect(
        TranscriptEvent(text="   ", is_final=True, words=[])
    )

    assert result == StyleMetadata.neutral_fallback()
    assert len(llm.calls) == 0


class _HangingLLM:
    async def complete(self, **kwargs):
        import asyncio

        await asyncio.sleep(3600)  # a provider that never answers


@pytest.mark.asyncio
async def test_hung_llm_times_out_and_falls_back_instead_of_stalling():
    """
    Regression: the docs promised a timeout fallback but detect() had
    no timeout, so a hung provider stalled the segment (and, since the
    translator waits on the style, the whole live session) forever.
    """
    import asyncio
    import time

    detector = RegisterDetector(llm_client=_HangingLLM())
    detector.TIMEOUT_S = 0.05  # keep the test fast

    start = time.monotonic()
    result = await asyncio.wait_for(detector.detect(_sarcastic_transcript()), timeout=2.0)

    assert result == StyleMetadata.neutral_fallback()
    assert time.monotonic() - start < 1.0
