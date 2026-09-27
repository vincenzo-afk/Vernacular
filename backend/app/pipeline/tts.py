"""
Stage 4: Expressive text-to-speech.

Provider-agnostic interface (TTSProvider) + ElevenLabs implementation.
Fallback providers (Cartesia, OpenAI TTS) implement the same interface
— see ARCHITECTURE.md §5 for what to do when the primary provider
fails or rate-limits mid-demo.
"""

import logging
from collections.abc import AsyncIterator
from typing import Protocol

from app.schemas.style_metadata import Pace, StyleMetadata, Tone

logger = logging.getLogger("vernacular.tts")


class TTSProvider(Protocol):
    async def synthesize(
        self, text: str, style: StyleMetadata, voice_id: str
    ) -> AsyncIterator[bytes]:
        """
        Streams synthesized audio bytes. Must begin yielding audio
        before the full clip is generated where the provider supports
        it (time-to-first-byte matters more than total generation time
        — see ARCHITECTURE.md §3).
        """
        ...


# --- StyleMetadata -> ElevenLabs parameter mapping -------------------------
#
# ElevenLabs' voice_settings API exposes: stability (0-1, lower = more
# expressive/variable), similarity_boost (0-1, higher = closer to the
# cloned voice), style (0-1, exaggeration of the style embedded in the
# voice), and a separate speed multiplier. This mapping is a pure
# function of StyleMetadata so it's unit-testable without any network
# call — see backend/tests/unit/test_tts_mapping.py.

_TONE_TO_STABILITY_STYLE: dict[Tone, tuple[float, float]] = {
    # tone -> (stability, style_exaggeration)
    # Lower stability + higher style = more expressive/variable delivery.
    Tone.NEUTRAL: (0.75, 0.15),
    Tone.SARCASTIC: (0.35, 0.75),
    Tone.URGENT: (0.30, 0.60),
    Tone.WARM: (0.55, 0.45),
    Tone.FORMAL: (0.80, 0.10),
    Tone.HESITANT: (0.45, 0.30),
    Tone.EXCITED: (0.30, 0.80),
    Tone.ANNOYED: (0.40, 0.65),
}

_PACE_TO_SPEED: dict[Pace, float] = {
    Pace.SLOW: 0.85,
    Pace.NORMAL: 1.0,
    Pace.FAST: 1.15,
    Pace.RUSHED: 1.3,
}


class ElevenLabsVoiceSettings:
    """Plain data holder for the parameters we send to ElevenLabs."""

    def __init__(
        self, stability: float, similarity_boost: float, style: float, speed: float
    ) -> None:
        self.stability = stability
        self.similarity_boost = similarity_boost
        self.style = style
        self.speed = speed

    def as_dict(self) -> dict:
        return {
            "stability": self.stability,
            "similarity_boost": self.similarity_boost,
            "style": self.style,
            "speed": self.speed,
        }


def style_to_voice_settings(style: StyleMetadata) -> ElevenLabsVoiceSettings:
    """
    Pure mapping from StyleMetadata to ElevenLabs voice settings. Kept
    as a standalone function (not a method on ElevenLabsTTS) so it's
    testable without constructing a real client — see
    backend/tests/unit/test_tts_mapping.py.

    Sarcasm is layered on top of the tone-based stability/style values:
    a high sarcasm_score pushes toward more exaggerated, less stable
    delivery even if the base tone alone wouldn't call for it — this
    mirrors how sarcastic delivery in speech tends to be more
    performative than a "sarcastic" label alone implies.
    """
    stability, style_exaggeration = _TONE_TO_STABILITY_STYLE[style.tone]

    # Blend continuously toward the sarcastic-tone values, weighted by
    # sarcasm_score, even if the classified tone was something else
    # (e.g. a sarcastic remark classified primarily as "annoyed").
    # Applied at every score (not gated behind a threshold) so this is
    # a genuine smooth ramp rather than a step function — a gate here
    # would mean e.g. sarcasm_score=0.49 and sarcasm_score=0.51 produce
    # wildly different settings despite being nearly identical inputs,
    # which is exactly the kind of discontinuity a continuous blend is
    # supposed to avoid. At sarcasm_score=0 the blend weight is 0, so
    # this is a no-op for genuinely non-sarcastic readings.
    sarcastic_stability, sarcastic_style = _TONE_TO_STABILITY_STYLE[Tone.SARCASTIC]
    weight = min(1.0, max(0.0, style.sarcasm_score))
    stability = stability * (1 - weight) + sarcastic_stability * weight
    style_exaggeration = style_exaggeration * (1 - weight) + sarcastic_style * weight

    similarity_boost = 0.75  # kept high by default to preserve cloned-voice identity

    speed = _PACE_TO_SPEED[style.pace]
    # Confidence below the fallback threshold should already have been
    # normalized to neutral upstream (RegisterDetector), but clamp
    # defensively here too since tts.py has no visibility into whether
    # its caller applied that fallback correctly.
    if style.confidence < 0.5:
        stability, style_exaggeration = _TONE_TO_STABILITY_STYLE[Tone.NEUTRAL]
        speed = 1.0

    return ElevenLabsVoiceSettings(
        stability=round(stability, 3),
        similarity_boost=similarity_boost,
        style=round(style_exaggeration, 3),
        speed=round(speed, 3),
    )


class ElevenLabsTTS:
    """
    ElevenLabs Multilingual v2 implementation of TTSProvider, with
    optional voice cloning. Maps StyleMetadata fields onto ElevenLabs'
    style / stability / similarity_boost / speed parameters via
    style_to_voice_settings() above.

    Voice cloning: the clone is generated once per session from a short
    calibration clip at session start (see ARCHITECTURE.md §3) — do not
    regenerate the clone per utterance. See CONTRIBUTING.md "Known
    gaps" — the consent flow for voice cloning is not yet implemented;
    do not wire clone_voice() into a live demo path without one.
    """

    MODEL_ID = "eleven_multilingual_v2"

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._client = None  # lazily constructed real SDK client

    def _get_client(self):
        if self._client is None:
            from elevenlabs.client import AsyncElevenLabs

            self._client = AsyncElevenLabs(api_key=self._api_key)
        return self._client

    async def synthesize(
        self, text: str, style: StyleMetadata, voice_id: str
    ) -> AsyncIterator[bytes]:
        settings = style_to_voice_settings(style)
        client = self._get_client()

        logger.debug(
            "tts.synthesize voice_id=%s tone=%s settings=%s",
            voice_id,
            style.tone,
            settings.as_dict(),
        )

        audio_stream = client.text_to_speech.convert_as_stream(
            voice_id=voice_id,
            model_id=self.MODEL_ID,
            text=text,
            voice_settings=settings.as_dict(),
        )
        async for chunk in audio_stream:
            yield chunk

    async def clone_voice(self, calibration_audio: bytes) -> str:
        """
        Returns a voice_id for use in synthesize(). One call per
        session. Requires explicit user consent captured before this
        is invoked — see CONTRIBUTING.md "Known gaps".
        """
        client = self._get_client()
        voice = await client.voices.add(
            name="vernacular-session-voice",
            files=[calibration_audio],
        )
        return voice.voice_id


class FallbackTTS:
    """
    Cartesia / OpenAI TTS fallback, used when ElevenLabs is unavailable
    or rate-limited. Has coarser style control than ElevenLabs — the
    frontend should indicate degraded fidelity to the user rather than
    silently presenting it as full-quality (ARCHITECTURE.md §5).

    Style mapping here is intentionally simpler: these providers don't
    expose the same stability/style/similarity controls, so we only
    map pace -> speed and leave tone/formality to influence word choice
    upstream (already baked into the translated text) rather than
    trying to fake fine-grained delivery control this provider doesn't
    support.

    Voice identity: ElevenLabs voice IDs (including cloned-voice IDs)
    are meaningless to OpenAI's TTS API, which only accepts a small
    fixed set of stock voice names (alloy, echo, fable, onyx, nova,
    shimmer). TTSProvider's interface takes a single `voice_id` string
    shared across providers, but that ID is provider-specific — this
    class deliberately ignores whatever voice_id it's given and always
    uses DEFAULT_VOICE, rather than passing an ElevenLabs ID through to
    an API that would reject it. This means a cloned voice's identity
    is lost during fallback, which is exactly the kind of degradation
    ARCHITECTURE.md §5 says must be surfaced to the user, not hidden.
    """

    DEFAULT_VOICE = "alloy"

    def __init__(self, provider: str, api_key: str) -> None:
        self._provider = provider
        self._api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            if self._provider == "openai":
                from openai import AsyncOpenAI

                self._client = AsyncOpenAI(api_key=self._api_key)
            else:
                raise NotImplementedError(
                    f"FallbackTTS provider '{self._provider}' not yet "
                    "implemented — openai is currently the only supported "
                    "fallback. Add a branch here for cartesia when needed."
                )
        return self._client

    async def synthesize(
        self, text: str, style: StyleMetadata, voice_id: str
    ) -> AsyncIterator[bytes]:
        speed = _PACE_TO_SPEED[style.pace]
        client = self._get_client()

        if voice_id and voice_id != "default":
            logger.warning(
                "fallback_tts: caller passed voice_id=%s, but this "
                "provider cannot honor a primary-provider voice ID "
                "(see class docstring) — using %s instead",
                voice_id,
                self.DEFAULT_VOICE,
            )

        logger.debug(
            "fallback_tts.synthesize provider=%s speed=%s", self._provider, speed
        )

        async with client.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice=self.DEFAULT_VOICE,
            input=text,
            speed=speed,
        ) as response:
            async for chunk in response.iter_bytes():
                yield chunk
