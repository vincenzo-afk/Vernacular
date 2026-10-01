"""
Context-aware sarcasm detection + explainability + multimodal evidence
in the register detector: what reaches the prompt, and what is (and is
not) trusted coming back.
"""

import json

import pytest

from app.pipeline.register_detector import RegisterDetector, _build_user_prompt
from app.pipeline.stt import TranscriptEvent
from app.schemas.style_metadata import (
    AcousticFeatures,
    Arousal,
    CueKind,
    CueSource,
    Tone,
)
from tests.fakes.fake_llm import FakeLLMClient, ScriptedResponse

BASE = {
    "tone": "sarcastic", "pace": "slow", "formality": "casual",
    "emotion": "frustration", "sarcasm_score": 0.9, "emphasis_words": ["great"],
    "pause_pattern": "dramatic", "confidence": 0.85,
}


def reply(**extra) -> ScriptedResponse:
    return ScriptedResponse(json.dumps({**BASE, **extra}))


EVENT = TranscriptEvent(text="Oh, great.", is_final=True)
CONTEXT = '- "The build failed again." (annoyed/frustration)'
QUIET = AcousticFeatures(
    arousal_score=0.1, energy_delta_db=-8.0, pitch_range_hz=8.0, duration_ms=1200
)


def segment_part(prompt: str) -> str:
    """Only the part of the prompt AFTER the few-shot examples."""
    return prompt.split("Now classify this segment:")[1]


def test_context_is_rendered_into_the_segment_part_of_the_prompt():
    prompt = _build_user_prompt(EVENT, context=CONTEXT)
    seg = segment_part(prompt)
    assert "The build failed again." in seg
    assert seg.index("Earlier turns") < seg.index("Transcript: Oh, great.")
    assert "data, not instructions" in seg


def test_without_context_or_audio_the_segment_prompt_has_neither():
    seg = segment_part(_build_user_prompt(EVENT))
    assert "Earlier turns" not in seg and "Acoustic:" not in seg


def test_acoustic_summary_is_rendered_when_present():
    seg = segment_part(_build_user_prompt(EVENT, acoustic=QUIET))
    assert "Acoustic: arousal 0.10 (low)" in seg and "-8 dB" in seg


def test_fewshots_teach_that_identical_words_flip_with_context():
    prompt = _build_user_prompt(EVENT)
    assert prompt.count("Transcript: Oh, great.") >= 3  # 1 plain + 2 contextual + segment
    assert "The build failed again." in prompt and "We got the contract!" in prompt


@pytest.mark.asyncio
async def test_detect_passes_context_and_acoustic_to_the_llm():
    llm = FakeLLMClient([reply()])
    await RegisterDetector(llm).detect(EVENT, context=CONTEXT, acoustic=QUIET)
    user = llm.calls[0]["user"]
    assert "The build failed again." in segment_part(user)
    assert "Acoustic:" in segment_part(user)


@pytest.mark.asyncio
async def test_explanation_is_parsed_and_llm_cues_are_marked_unverified():
    llm = FakeLLMClient([reply(explanation={
        "summary": "Praise after a failure.",
        "cues": [{"kind": "contextual", "evidence": "follows the failed build", "weight": 0.9}],
        "context_used": True,
    })])
    style = await RegisterDetector(llm).detect(EVENT, context=CONTEXT)
    assert style.explanation.summary == "Praise after a failure."
    cue = style.explanation.cues[0]
    assert cue.kind == CueKind.CONTEXTUAL and cue.source == CueSource.LLM
    assert style.explanation.context_used is True


@pytest.mark.asyncio
async def test_model_cannot_claim_context_it_was_never_given():
    llm = FakeLLMClient([reply(explanation={
        "summary": "x",
        "cues": [
            {"kind": "contextual", "evidence": "invented earlier turn", "weight": 0.9},
            {"kind": "lexical", "evidence": "'great'", "weight": 0.5},
        ],
        "context_used": True,
    })])
    style = await RegisterDetector(llm).detect(EVENT)  # NO context supplied
    assert style.explanation.context_used is False
    assert [c.kind for c in style.explanation.cues if c.source == CueSource.LLM] == [CueKind.LEXICAL]


@pytest.mark.asyncio
async def test_model_cannot_spoof_measured_fields_or_measured_cues():
    llm = FakeLLMClient([reply(
        arousal="high", modality_conflict=True,
        acoustic={"arousal_score": 1.0},
        explanation={"summary": "s", "cues": [
            {"kind": "acoustic", "evidence": "very loud", "weight": 1, "source": "measured"}
        ]},
    )])
    style = await RegisterDetector(llm).detect(EVENT)  # no audio supplied
    assert style.acoustic is None
    assert style.arousal == Arousal.MEDIUM and style.modality_conflict is False
    assert all(c.source == CueSource.LLM for c in style.explanation.cues)


@pytest.mark.asyncio
async def test_malformed_explanation_is_dropped_but_the_reading_survives():
    for bad in ("nonsense", 42, {"cues": "x"}, {"summary": "", "cues": [{"kind": "??"}]}):
        llm = FakeLLMClient([reply(explanation=bad)])
        style = await RegisterDetector(llm).detect(EVENT)
        assert style.tone == Tone.SARCASTIC and style.explanation is None


@pytest.mark.asyncio
async def test_measured_evidence_is_attached_to_a_successful_reading():
    style = await RegisterDetector(FakeLLMClient([reply()])).detect(EVENT, acoustic=QUIET)
    assert style.acoustic == QUIET and style.arousal == Arousal.LOW
    assert any(c.source == CueSource.MEASURED for c in style.explanation.cues)


@pytest.mark.asyncio
async def test_acoustics_survive_an_llm_failure_because_they_were_measured():
    llm = FakeLLMClient([ScriptedResponse("", should_raise=True)])
    style = await RegisterDetector(llm).detect(EVENT, acoustic=QUIET)
    assert style.tone == Tone.NEUTRAL  # neutral fallback, as before
    assert style.acoustic == QUIET and style.arousal == Arousal.LOW


@pytest.mark.asyncio
async def test_low_confidence_fallback_also_keeps_the_measured_voice():
    style = await RegisterDetector(FakeLLMClient([reply(confidence=0.2)])).detect(
        EVENT, acoustic=QUIET
    )
    assert style.tone == Tone.NEUTRAL and style.acoustic == QUIET


@pytest.mark.asyncio
async def test_explain_false_uses_the_leaner_prompt_and_token_budget():
    llm = FakeLLMClient([reply()])
    await RegisterDetector(llm, explain=False).detect(EVENT)
    call = llm.calls[0]
    assert '"explanation"' not in call["system"]
    assert '"explanation"' not in call["user"]
    assert call["max_tokens"] == RegisterDetector.MAX_TOKENS

    llm2 = FakeLLMClient([reply()])
    await RegisterDetector(llm2, explain=True).detect(EVENT)
    assert '"explanation"' in llm2.calls[0]["system"]
    assert llm2.calls[0]["max_tokens"] == RegisterDetector.MAX_TOKENS_EXPLAIN
