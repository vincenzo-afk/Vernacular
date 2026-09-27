"""
Targeted tests for SessionOrchestrator._resolve_style_for_final — the
speculative-vs-fresh register detection logic described in
pipeline/orchestrator.py and pipeline/AGENTS.md. These are narrower
than the full-pipeline tests in tests/integration/test_orchestrator_flow.py:
they exercise _resolve_style_for_final directly so the prefix-validity
heuristic and cancellation behavior are covered precisely, not just as
a side effect of a full run.
"""

import asyncio
import json

import pytest

from app.pipeline.orchestrator import SessionOrchestrator
from app.pipeline.register_detector import RegisterDetector
from app.pipeline.stt import TranscriptEvent
from app.pipeline.translator import Translator
from app.schemas.style_metadata import StyleMetadata, Tone
from tests.fakes.fake_llm import FakeLLMClient, ScriptedResponse
from tests.fakes.fake_tts import FakeTTS

SARCASTIC_JSON = json.dumps(
    {
        "tone": "sarcastic",
        "pace": "slow",
        "formality": "casual",
        "emotion": "frustration",
        "sarcasm_score": 0.88,
        "emphasis_words": [],
        "pause_pattern": "dramatic",
        "confidence": 0.81,
    }
)

URGENT_JSON = json.dumps(
    {
        "tone": "urgent",
        "pace": "rushed",
        "formality": "casual",
        "emotion": "concern",
        "sarcasm_score": 0.02,
        "emphasis_words": [],
        "pause_pattern": "clipped",
        "confidence": 0.9,
    }
)


def _make_orchestrator(register_llm: FakeLLMClient) -> SessionOrchestrator:
    return SessionOrchestrator(
        stt=None,  # not exercised directly in these tests
        register_detector=RegisterDetector(llm_client=register_llm),
        translator=Translator(llm_client=FakeLLMClient([])),
        tts=FakeTTS(),
        target_language="es",
        voice_id="test-voice",
    )


@pytest.mark.asyncio
async def test_reuses_completed_speculative_result_when_text_is_prefix():
    register_llm = FakeLLMClient([ScriptedResponse(content=SARCASTIC_JSON)])
    orchestrator = _make_orchestrator(register_llm)

    speculative_task = asyncio.create_task(
        orchestrator._run_register_detection(
            TranscriptEvent(text="Oh, great", is_final=False)
        )
    )
    await speculative_task  # let it finish before resolving

    final_event = TranscriptEvent(
        text="Oh, great, another meeting.", is_final=True
    )
    style = await orchestrator._resolve_style_for_final(
        final_event, speculative_task, "Oh, great"
    )

    assert style.tone == Tone.SARCASTIC
    # Only the speculative call should have hit the LLM -- no second
    # (fresh) call should have been made.
    assert len(register_llm.calls) == 1


@pytest.mark.asyncio
async def test_runs_fresh_detection_when_final_text_diverges_from_speculative():
    # Two scripted responses: the speculative call gets one result,
    # the fresh call (because text diverged) gets a different one --
    # if the orchestrator incorrectly reused the speculative result,
    # this test would observe the wrong tone.
    register_llm = FakeLLMClient(
        [
            ScriptedResponse(content=SARCASTIC_JSON),
            ScriptedResponse(content=URGENT_JSON),
        ]
    )
    orchestrator = _make_orchestrator(register_llm)

    speculative_task = asyncio.create_task(
        orchestrator._run_register_detection(
            TranscriptEvent(text="Watch out the", is_final=False)
        )
    )
    await speculative_task

    # Final transcript is NOT a continuation of "Watch out the" -- STT
    # revised its interpretation of the utterance entirely.
    final_event = TranscriptEvent(text="What about the car?", is_final=True)
    style = await orchestrator._resolve_style_for_final(
        final_event, speculative_task, "Watch out the"
    )

    assert style.tone == Tone.URGENT  # from the fresh call, not sarcastic
    assert len(register_llm.calls) == 2


@pytest.mark.asyncio
async def test_no_speculative_task_runs_fresh_detection():
    register_llm = FakeLLMClient([ScriptedResponse(content=SARCASTIC_JSON)])
    orchestrator = _make_orchestrator(register_llm)

    final_event = TranscriptEvent(text="Oh, great, another meeting.", is_final=True)
    style = await orchestrator._resolve_style_for_final(final_event, None, "")

    assert style.tone == Tone.SARCASTIC
    assert len(register_llm.calls) == 1


@pytest.mark.asyncio
async def test_slow_speculative_task_is_cancelled_and_fresh_result_used():
    """
    If the speculative task hasn't finished within the short grace
    window, it should be cancelled (not abandoned running) and a fresh
    detection performed instead.
    """
    register_llm = FakeLLMClient(
        [
            ScriptedResponse(content=SARCASTIC_JSON),  # for the fresh call
        ]
    )
    orchestrator = _make_orchestrator(register_llm)

    async def never_finishes(event: TranscriptEvent) -> StyleMetadata:
        await asyncio.sleep(10)  # much longer than the 0.05s grace window
        return StyleMetadata()  # pragma: no cover -- should be cancelled first

    slow_task = asyncio.create_task(never_finishes(TranscriptEvent(text="Oh", is_final=False)))

    final_event = TranscriptEvent(text="Oh, great, another meeting.", is_final=True)
    style = await orchestrator._resolve_style_for_final(final_event, slow_task, "Oh")

    assert slow_task.cancelled()
    assert style.tone == Tone.SARCASTIC  # came from the fresh call
    assert len(register_llm.calls) == 1


@pytest.mark.asyncio
async def test_speculative_task_that_raised_falls_back_to_fresh_detection():
    """
    RegisterDetector.detect() is documented to never raise, but this
    exercises the orchestrator's own defensive handling in case that
    contract is ever violated (e.g. a future refactor introduces a bug
    in RegisterDetector) -- the orchestrator must not let an unexpected
    exception from the speculative task kill the session.
    """
    register_llm = FakeLLMClient([ScriptedResponse(content=URGENT_JSON)])
    orchestrator = _make_orchestrator(register_llm)

    async def broken_detection(event: TranscriptEvent) -> StyleMetadata:
        raise RuntimeError("simulated contract violation")

    broken_task = asyncio.create_task(
        broken_detection(TranscriptEvent(text="Watch out", is_final=False))
    )
    with pytest.raises(RuntimeError):
        await broken_task  # confirm it actually raised, for test sanity

    final_event = TranscriptEvent(text="Watch out, the car!", is_final=True)
    style = await orchestrator._resolve_style_for_final(
        final_event, broken_task, "Watch out"
    )

    assert style.tone == Tone.URGENT  # from the fresh fallback call
    assert len(register_llm.calls) == 1
