"""
See TESTING.md — unit tests for the StyleMetadata schema itself:
bounds validation, defaults, and the neutral_fallback() factory.
"""

import pytest
from pydantic import ValidationError

from app.schemas.style_metadata import (
    Emotion,
    Formality,
    Pace,
    PausePattern,
    StyleMetadata,
    Tone,
)


def test_defaults_are_neutral():
    style = StyleMetadata()
    assert style.tone == Tone.NEUTRAL
    assert style.pace == Pace.NORMAL
    assert style.formality == Formality.NEUTRAL
    assert style.emotion == Emotion.NEUTRAL
    assert style.sarcasm_score == 0.0
    assert style.emphasis_words == []
    assert style.pause_pattern == PausePattern.NATURAL
    assert style.confidence == 1.0


def test_neutral_fallback_matches_defaults():
    fallback = StyleMetadata.neutral_fallback()
    assert fallback == StyleMetadata()


@pytest.mark.parametrize("value", [-0.1, 1.1, 2.0, -5])
def test_sarcasm_score_out_of_bounds_rejected(value):
    with pytest.raises(ValidationError):
        StyleMetadata(sarcasm_score=value)


@pytest.mark.parametrize("value", [-0.1, 1.1, 2.0, -5])
def test_confidence_out_of_bounds_rejected(value):
    with pytest.raises(ValidationError):
        StyleMetadata(confidence=value)


@pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
def test_sarcasm_score_in_bounds_accepted(value):
    style = StyleMetadata(sarcasm_score=value)
    assert style.sarcasm_score == value


def test_invalid_tone_rejected():
    with pytest.raises(ValidationError):
        StyleMetadata(tone="mischievous")  # type: ignore[arg-type]


def test_valid_full_construction():
    style = StyleMetadata(
        tone=Tone.SARCASTIC,
        pace=Pace.SLOW,
        formality=Formality.CASUAL,
        emotion=Emotion.FRUSTRATION,
        sarcasm_score=0.88,
        emphasis_words=["great"],
        pause_pattern=PausePattern.DRAMATIC,
        confidence=0.81,
    )
    assert style.tone == Tone.SARCASTIC
    assert style.emphasis_words == ["great"]


def test_model_dump_json_roundtrip():
    style = StyleMetadata(tone=Tone.URGENT, pace=Pace.RUSHED, sarcasm_score=0.02)
    dumped = style.model_dump_json()
    restored = StyleMetadata.model_validate_json(dumped)
    assert restored == style
