"""
Stage 1: Speech-to-text.

Provider-agnostic interface (STTProvider) + AssemblyAI implementation.
See ARCHITECTURE.md §4 — no AssemblyAI SDK calls should exist outside
this file. Anything that needs a transcript should depend on
STTProvider, not on AssemblyAI directly.
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class WordTiming:
    word: str
    start_ms: int
    end_ms: int
    confidence: float
    is_stressed: bool = False  # informs StyleMetadata.emphasis_words


@dataclass
class TranscriptEvent:
    text: str
    is_final: bool
    words: list[WordTiming] = field(default_factory=list)
    sentiment: str | None = None  # raw sentiment label from provider, if available


class STTProvider(Protocol):
    """
    Implement this for any speech-to-text backend. AssemblyAI is the
    required provider for this project (hackathon constraint), but the
    interface exists so the register detector and orchestrator never
    depend on AssemblyAI-specific types.
    """

    async def stream(
        self, audio_chunks: AsyncIterator[bytes]
    ) -> AsyncIterator[TranscriptEvent]:
        """
        Consume raw audio chunks, yield TranscriptEvent as partial and
        final transcripts become available. Must yield partials
        frequently enough to support the pipelining strategy in
        ARCHITECTURE.md §3 (register detection starts on partials, not
        just finals).
        """
        ...


class AssemblyAISTT:
    """
    AssemblyAI Realtime STT + Sentiment Analysis implementation of
    STTProvider. TODO: implement using AssemblyAI's realtime websocket
    API. Word-level timestamps and sentiment scores from this provider
    feed directly into StyleMetadata construction in
    pipeline/register_detector.py.
    """

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    async def stream(
        self, audio_chunks: AsyncIterator[bytes]
    ) -> AsyncIterator[TranscriptEvent]:
        raise NotImplementedError("AssemblyAI realtime STT integration pending")
