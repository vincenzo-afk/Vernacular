"""Translator: conversation context in, measured delivery fields carried through."""

import json

import pytest

from app.pipeline.translator import Translator, _build_user_prompt
from app.schemas.style_metadata import (
    AcousticFeatures,
    Arousal,
    Cue,
    CueKind,
    StyleExplanation,
    StyleMetadata,
    Tone,
)
from tests.fakes.fake_llm import FakeLLMClient, ScriptedResponse

STYLE = {
    "tone": "sarcastic", "pace": "slow", "formality": "casual",
    "emotion": "frustration", "sarcasm_score": 0.8, "emphasis_words": ["bien"],
    "pause_pattern": "dramatic", "confidence": 0.8,
}
OK = ScriptedResponse(json.dumps({"translated_text": "Ay, qué bien.", "style": STYLE}))

MEASURED = StyleMetadata(
    tone=Tone.SARCASTIC, sarcasm_score=0.9,
    arousal=Arousal.LOW, modality_conflict=True,
    acoustic=AcousticFeatures(arousal_score=0.1, duration_ms=900),
    explanation=StyleExplanation(summary="why", cues=[Cue(kind=CueKind.LEXICAL, evidence="great")]),
)


def test_prompt_contains_context_only_when_given():
    with_ctx = _build_user_prompt("Oh, great.", MEASURED, "es", '- "Hi" -> "Hola"')
    assert '- "Hi" -> "Hola"' in with_ctx and "Earlier turns" in with_ctx
    assert with_ctx.index("Earlier turns") < with_ctx.index("Source text:")
    assert "Earlier turns" not in _build_user_prompt("Oh, great.", MEASURED, "es")


def test_only_the_eight_register_fields_reach_the_translator_prompt():
    prompt = _build_user_prompt("Oh, great.", MEASURED, "es")
    assert '"sarcasm_score": 0.9' in prompt
    for leaked in ("acoustic", "arousal", "modality_conflict", "explanation", "arousal_score"):
        assert leaked not in prompt


@pytest.mark.asyncio
async def test_translate_passes_context_through_to_the_llm():
    llm = FakeLLMClient([OK])
    await Translator(llm).translate("Oh, great.", MEASURED, "es", context='- "Hi" -> "Hola"')
    assert '- "Hi" -> "Hola"' in llm.calls[0]["user"]


@pytest.mark.asyncio
async def test_measured_fields_survive_translation():
    """The LLM never sees or returns them; without carry-through they
    would silently vanish at this stage."""
    result = await Translator(FakeLLMClient([OK])).translate("Oh, great.", MEASURED, "es")
    assert result.translated_text == "Ay, qué bien."
    s = result.style
    assert s.acoustic == MEASURED.acoustic
    assert s.arousal == Arousal.LOW and s.modality_conflict is True
    assert s.explanation == MEASURED.explanation


@pytest.mark.asyncio
async def test_measured_fields_survive_the_literal_fallback_too():
    llm = FakeLLMClient([
        ScriptedResponse("", should_raise=True),
        ScriptedResponse("", should_raise=True),
        ScriptedResponse("Ay, qué bien."),  # the literal-fallback call
    ])
    result = await Translator(llm).translate("Oh, great.", MEASURED, "es")
    assert result.was_literal_fallback
    assert result.style.tone == Tone.NEUTRAL  # honest downgrade, as before
    assert result.style.acoustic == MEASURED.acoustic


@pytest.mark.asyncio
async def test_translator_without_context_still_works_as_before():
    result = await Translator(FakeLLMClient([OK])).translate("Oh, great.", StyleMetadata(), "es")
    assert result.translated_text == "Ay, qué bien."
