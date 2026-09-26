"""
Fake STTProvider for tests that don't need live AssemblyAI calls.
See CLAUDE.md / AGENTS.md — pipeline tests should not require live
API keys to run in CI.
"""

from collections.abc import AsyncIterator

from app.pipeline.stt import TranscriptEvent, WordTiming


class FakeSTT:
    def __init__(self, scripted_events: list[TranscriptEvent]) -> None:
        self._events = scripted_events

    async def stream(
        self, audio_chunks: AsyncIterator[bytes]
    ) -> AsyncIterator[TranscriptEvent]:
        for event in self._events:
            yield event


def sarcastic_example() -> TranscriptEvent:
    return TranscriptEvent(
        text="Oh, great, another meeting.",
        is_final=True,
        words=[
            WordTiming(word="Oh,", start_ms=0, end_ms=200, confidence=0.99),
            WordTiming(
                word="great,",
                start_ms=200,
                end_ms=600,
                confidence=0.98,
                is_stressed=True,
            ),
            WordTiming(word="another", start_ms=1200, end_ms=1500, confidence=0.97),
            WordTiming(word="meeting.", start_ms=1500, end_ms=1900, confidence=0.98),
        ],
        sentiment="NEGATIVE",
    )
