"""Latency-report math, plus an end-to-end check that the orchestrator's
real StageTiming output feeds it and that injected slowness shows up in
the right stage (not just that the arithmetic is right in isolation)."""

import asyncio
import json

import pytest

from app.latency import (
    critical_path_p90_ms,
    format_report,
    percentile,
    summarize,
)
from app.pipeline.orchestrator import SessionOrchestrator, StageTiming
from app.pipeline.register_detector import RegisterDetector
from app.pipeline.stt import TranscriptEvent
from app.pipeline.translator import Translator
from tests.fakes.fake_llm import FakeLLMClient, ScriptedResponse


def test_percentile_basics():
    assert percentile([10], 90) == 10
    assert percentile([0, 100], 50) == 50
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert percentile([1, 2, 3, 4, 5], 100) == 5


def test_percentile_is_order_independent_and_rejects_empty():
    assert percentile([5, 1, 3], 50) == percentile([1, 3, 5], 50)
    with pytest.raises(ValueError):
        percentile([], 50)


def _t(stage, ms):
    return StageTiming(stage=stage, start_ms=0.0, end_ms=ms)


def test_over_budget_is_judged_on_p90_not_the_mean():
    # Mostly fast, one slow outlier: p90 must catch it.
    timings = [_t("translation", 100)] * 8 + [_t("translation", 2000)] * 2
    (s,) = summarize(timings)
    assert s.budget_ms == 600.0
    assert s.over_budget is True


def test_critical_path_excludes_speculative_register_detection():
    timings = [
        _t("register_detection", 5000),  # huge, but off the critical path
        _t("translation", 300),
        _t("tts_first_byte", 200),
    ]
    assert critical_path_p90_ms(summarize(timings)) == 500


def test_report_renders_verdict():
    within = format_report(summarize([_t("translation", 300), _t("tts_first_byte", 200)]))
    over = format_report(summarize([_t("translation", 1500), _t("tts_first_byte", 900)]))
    assert "WITHIN" in within
    assert "OVER" in over


STYLE = {
    "tone": "neutral", "pace": "normal", "formality": "neutral",
    "emotion": "neutral", "sarcasm_score": 0.0, "emphasis_words": [],
    "pause_pattern": "natural", "confidence": 0.9,
}


class SlowTTS:
    def __init__(self, first_byte_delay):
        self._delay = first_byte_delay

    async def synthesize(self, text, style, voice_id):
        await asyncio.sleep(self._delay)
        yield b"x"


class ScriptedSTT:
    def __init__(self, n):
        self._events = [TranscriptEvent(text=f"u{i}", is_final=True) for i in range(n)]

    async def stream(self, audio_chunks):
        for e in self._events:
            yield e


async def _audio():
    yield b"\x00"


@pytest.mark.asyncio
async def test_injected_tts_delay_shows_up_in_first_byte_stage_from_real_orchestrator():
    n = 3
    orch = SessionOrchestrator(
        stt=ScriptedSTT(n),
        register_detector=RegisterDetector(
            llm_client=FakeLLMClient([ScriptedResponse(json.dumps(STYLE))] * n)
        ),
        translator=Translator(
            llm_client=FakeLLMClient(
                [ScriptedResponse(json.dumps({"translated_text": "h", "style": STYLE}))] * n
            )
        ),
        tts=SlowTTS(first_byte_delay=0.12),
        target_language="es",
        voice_id="v",
    )
    _ = [e async for e in orch.run_stream(_audio())]

    stats = {s.stage: s for s in summarize(orch.timings)}
    assert stats["tts_first_byte"].count == n
    # The injected 120ms must land in tts_first_byte, and NOT in translation.
    assert stats["tts_first_byte"].p50_ms >= 100
    assert stats["translation"].p50_ms < 50
