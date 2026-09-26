"""
See TESTING.md — pure mapping tests for style_to_voice_settings(), no
network calls involved.
"""

import pytest

from app.pipeline.tts import style_to_voice_settings
from app.schemas.style_metadata import Pace, StyleMetadata, Tone


def test_neutral_style_maps_to_stable_low_exaggeration():
    settings = style_to_voice_settings(StyleMetadata())
    assert settings.stability > 0.6  # neutral should be relatively stable
    assert settings.style < 0.3  # and not exaggerated
    assert settings.speed == pytest.approx(1.0)


def test_sarcastic_tone_lowers_stability_and_raises_style():
    neutral = style_to_voice_settings(StyleMetadata(tone=Tone.NEUTRAL))
    sarcastic = style_to_voice_settings(
        StyleMetadata(tone=Tone.SARCASTIC, sarcasm_score=0.9, confidence=0.9)
    )
    assert sarcastic.stability < neutral.stability
    assert sarcastic.style > neutral.style


def test_high_sarcasm_score_pulls_toward_sarcastic_settings_even_with_other_tone():
    # A remark classified primarily as "annoyed" but with a high
    # sarcasm_score should still shift toward sarcastic-style delivery.
    annoyed_low_sarcasm = style_to_voice_settings(
        StyleMetadata(tone=Tone.ANNOYED, sarcasm_score=0.1, confidence=0.9)
    )
    annoyed_high_sarcasm = style_to_voice_settings(
        StyleMetadata(tone=Tone.ANNOYED, sarcasm_score=0.95, confidence=0.9)
    )
    assert annoyed_high_sarcasm.stability < annoyed_low_sarcasm.stability
    assert annoyed_high_sarcasm.style > annoyed_low_sarcasm.style


@pytest.mark.parametrize(
    "pace,expected_speed",
    [
        (Pace.SLOW, 0.85),
        (Pace.NORMAL, 1.0),
        (Pace.FAST, 1.15),
        (Pace.RUSHED, 1.3),
    ],
)
def test_pace_maps_to_expected_speed(pace, expected_speed):
    settings = style_to_voice_settings(StyleMetadata(pace=pace, confidence=0.9))
    assert settings.speed == pytest.approx(expected_speed)


def test_low_confidence_forces_neutral_settings_regardless_of_tone():
    settings = style_to_voice_settings(
        StyleMetadata(
            tone=Tone.EXCITED, pace=Pace.RUSHED, sarcasm_score=0.9, confidence=0.2
        )
    )
    neutral = style_to_voice_settings(StyleMetadata())
    assert settings.stability == neutral.stability
    assert settings.style == neutral.style
    assert settings.speed == pytest.approx(1.0)


def test_similarity_boost_is_high_by_default():
    settings = style_to_voice_settings(StyleMetadata())
    assert settings.similarity_boost >= 0.7


def test_as_dict_contains_all_elevenlabs_fields():
    settings = style_to_voice_settings(StyleMetadata())
    d = settings.as_dict()
    assert set(d.keys()) == {"stability", "similarity_boost", "style", "speed"}
