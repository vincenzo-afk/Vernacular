"""
Stage 4: Expressive text-to-speech.

Provider-agnostic interface (TTSProvider) + ElevenLabs implementation.
Fallback providers (Cartesia, OpenAI TTS) implement the same interface
— see ARCHITECTURE.md §5 for what to do when the primary provider
fails or rate-limits mid-demo.
"""

from collections.abc import AsyncIterator
from typing import Protocol

from app.schemas.style_metadata import StyleMetadata


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


class ElevenLabsTTS:
    """
    ElevenLabs Multilingual v2 implementation of TTSProvider, with
    optional voice cloning. Maps StyleMetadata fields onto ElevenLabs'
    style / stability / similarity_boost / speed parameters.

    Voice cloning: the clone is generated once per session from a short
    calibration clip at session start (see ARCHITECTURE.md §3) — do not
    regenerate the clone per utterance.
    """

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    async def synthesize(
        self, text: str, style: StyleMetadata, voice_id: str
    ) -> AsyncIterator[bytes]:
        raise NotImplementedError("ElevenLabs TTS integration pending")

    async def clone_voice(self, calibration_audio: bytes) -> str:
        """Returns a voice_id for use in synthesize(). One call per session."""
        raise NotImplementedError("ElevenLabs voice cloning integration pending")


class FallbackTTS:
    """
    Cartesia / OpenAI TTS fallback, used when ElevenLabs is unavailable
    or rate-limited. Has coarser style control than ElevenLabs — the
    frontend should indicate degraded fidelity to the user rather than
    silently presenting it as full-quality (ARCHITECTURE.md §5).
    """

    def __init__(self, provider: str, api_key: str) -> None:
        self._provider = provider
        self._api_key = api_key

    async def synthesize(
        self, text: str, style: StyleMetadata, voice_id: str
    ) -> AsyncIterator[bytes]:
        raise NotImplementedError("Fallback TTS integration pending")
