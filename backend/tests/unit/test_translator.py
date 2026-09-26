"""
See TESTING.md and backend/app/pipeline/AGENTS.md — Translator retries
once on failure, then falls back to a literal translation rather than
raising.
"""

import json

import pytest

from app.pipeline.translator import Translator
from app.schemas.style_metadata import StyleMetadata, Tone
from tests.fakes.fake_llm import FakeLLMClient, ScriptedResponse

GOOD_RESPONSE = json.dumps(
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


def _sarcastic_style() -> StyleMetadata:
    return StyleMetadata(
        tone=Tone.SARCASTIC,
        sarcasm_score=0.88,
        emphasis_words=["great"],
        confidence=0.81,
    )


@pytest.mark.asyncio
async def test_successful_translation_preserves_register():
    llm = FakeLLMClient([ScriptedResponse(content=GOOD_RESPONSE)])
    translator = Translator(llm_client=llm)

    result = await translator.translate(
        source_text="Oh, great, another meeting.",
        source_style=_sarcastic_style(),
        target_language="es",
    )

    assert result.translated_text == "Ay, qué bien, otra reunión."
    assert result.style.tone == Tone.SARCASTIC
    assert result.was_literal_fallback is False
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_retries_once_on_first_failure_then_succeeds():
    llm = FakeLLMClient(
        [
            ScriptedResponse(content="", should_raise=True),
            ScriptedResponse(content=GOOD_RESPONSE),
        ]
    )
    translator = Translator(llm_client=llm)

    result = await translator.translate(
        source_text="Oh, great, another meeting.",
        source_style=_sarcastic_style(),
        target_language="es",
    )

    assert result.translated_text == "Ay, qué bien, otra reunión."
    assert result.was_literal_fallback is False
    assert len(llm.calls) == 2


@pytest.mark.asyncio
async def test_falls_back_to_literal_after_two_failures():
    llm = FakeLLMClient(
        [
            ScriptedResponse(content="", should_raise=True),
            ScriptedResponse(content="", should_raise=True),
            ScriptedResponse(content="Oh, great, another meeting. (literal)"),
        ]
    )
    translator = Translator(llm_client=llm)

    result = await translator.translate(
        source_text="Oh, great, another meeting.",
        source_style=_sarcastic_style(),
        target_language="es",
    )

    assert result.was_literal_fallback is True
    assert result.style == StyleMetadata.neutral_fallback()
    # two register-aware attempts + one literal-fallback call
    assert len(llm.calls) == 3


@pytest.mark.asyncio
async def test_falls_back_to_source_text_if_literal_fallback_also_fails():
    llm = FakeLLMClient(
        [
            ScriptedResponse(content="", should_raise=True),
            ScriptedResponse(content="", should_raise=True),
            ScriptedResponse(content="", should_raise=True),
        ]
    )
    translator = Translator(llm_client=llm)

    result = await translator.translate(
        source_text="Oh, great, another meeting.",
        source_style=_sarcastic_style(),
        target_language="es",
    )

    assert result.was_literal_fallback is True
    assert result.translated_text == "Oh, great, another meeting."
    assert result.style == StyleMetadata.neutral_fallback()


@pytest.mark.asyncio
async def test_empty_source_text_short_circuits_without_calling_llm():
    llm = FakeLLMClient([])
    translator = Translator(llm_client=llm)

    result = await translator.translate(
        source_text="   ",
        source_style=_sarcastic_style(),
        target_language="es",
    )

    assert result.translated_text == ""
    assert len(llm.calls) == 0
