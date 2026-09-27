"""
Fake TTSProvider for tests. Records calls, yields dummy audio bytes.
See TESTING.md.
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass

from app.schemas.style_metadata import StyleMetadata


@dataclass
class TTSCall:
    text: str
    style: StyleMetadata
    voice_id: str


class FakeTTS:
    def __init__(self, chunk: bytes = b"\x00\x01", fail: bool = False) -> None:
        self.calls: list[TTSCall] = []
        self._chunk = chunk
        self._fail = fail

    async def synthesize(
        self, text: str, style: StyleMetadata, voice_id: str
    ) -> AsyncIterator[bytes]:
        self.calls.append(TTSCall(text=text, style=style, voice_id=voice_id))
        if self._fail:
            raise RuntimeError("Scripted TTS failure for test")
        # Yield in two chunks to simulate streaming rather than one blob.
        half = max(1, len(self._chunk) // 2)
        yield self._chunk[:half]
        yield self._chunk[half:]
