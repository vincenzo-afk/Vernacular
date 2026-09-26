"""
Ties the four pipeline stages together for one streaming session.

See ARCHITECTURE.md §1 and §3 for the data flow diagram and latency
budget this orchestration is designed to hit. Key behaviors this module
is responsible for:

- Running register detection on partial transcripts concurrently with
  STT finalization, not sequentially after it.
- Starting TTS synthesis on the first translated clause rather than
  waiting for a full translated sentence.
- Applying fallback behavior (ARCHITECTURE.md §5) at each stage
  boundary so a single stage failure never kills the session.
- Logging stage-level entry/exit timestamps for latency visibility
  (see CLAUDE.md / AGENTS.md — this is a functional requirement, not
  optional instrumentation).
"""

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

from app.pipeline.register_detector import RegisterDetector
from app.pipeline.stt import STTProvider
from app.pipeline.translator import Translator
from app.pipeline.tts import TTSProvider


@dataclass
class StageTiming:
    stage: str
    start_ms: float
    end_ms: float


class SessionOrchestrator:
    def __init__(
        self,
        stt: STTProvider,
        register_detector: RegisterDetector,
        translator: Translator,
        tts: TTSProvider,
        target_language: str,
        voice_id: str,
    ) -> None:
        self._stt = stt
        self._register_detector = register_detector
        self._translator = translator
        self._tts = tts
        self._target_language = target_language
        self._voice_id = voice_id
        self.timings: list[StageTiming] = []

    async def run(self, audio_chunks: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
        """
        Consumes source audio, yields translated+synthesized audio.
        See ARCHITECTURE.md §1 for the full four-stage flow this
        method coordinates.
        """
        raise NotImplementedError(
            "Orchestration wiring pending — see ARCHITECTURE.md §3 for "
            "the concurrency strategy this method must implement "
            "(partial-transcript pipelining, streaming at every boundary)."
        )

    def _record(self, stage: str, start_ms: float) -> None:
        self.timings.append(
            StageTiming(stage=stage, start_ms=start_ms, end_ms=time.monotonic() * 1000)
        )
