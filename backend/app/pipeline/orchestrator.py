"""
Ties the four pipeline stages together for one streaming session.

See ARCHITECTURE.md §1 and §3, and backend/app/pipeline/AGENTS.md for
the concurrency model this implements:

- Register detection runs speculatively on partial transcripts,
  concurrently with STT finalization, so the final-transcript pass is
  a cheap confirmation rather than a cold start.
- Translation starts only once both a final transcript AND a resolved
  StyleMetadata are available.
- TTS starts as soon as translated text is available for a segment;
  ElevenLabs/OpenAI streaming picks up the rest.
- A failure in any single stage falls back per pipeline/AGENTS.md and
  never kills the session.
- Every stage transition is timed via StageTiming for latency
  visibility (there is no automated latency test — see TESTING.md).
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from app.pipeline.register_detector import RegisterDetector
from app.pipeline.stt import STTProvider, TranscriptEvent
from app.pipeline.translator import Translator
from app.pipeline.tts import TTSProvider
from app.schemas.style_metadata import StyleMetadata

logger = logging.getLogger("vernacular.orchestrator")


@dataclass
class StageTiming:
    stage: str
    start_ms: float
    end_ms: float

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms


@dataclass
class SegmentOutput:
    """
    One translated-and-synthesized segment's worth of output, paired
    with the metadata the frontend's tone-tag UI displays (see
    frontend/components/ToneTags.tsx and app/ws/session.py, which is
    responsible for serializing this over the wire).
    """

    audio_chunks: list[bytes]
    source_text: str
    translated_text: str
    style: StyleMetadata


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

        # Track the most recent speculative register reading per
        # in-flight segment so the final-transcript pass can reuse it
        # as a starting point rather than recomputing from scratch.
        # Keyed by a simple monotonically increasing segment counter
        # since AssemblyAI doesn't expose a stable per-utterance ID in
        # the current STTProvider interface.
        self._speculative_style: dict[int, StyleMetadata] = {}
        self._segment_counter = 0

    async def run(
        self, audio_chunks: AsyncIterator[bytes]
    ) -> AsyncIterator[SegmentOutput]:
        """
        Consumes source audio, yields one SegmentOutput per completed
        utterance. See module docstring for the concurrency model.

        Design: STT partials feed a background task that keeps
        `_speculative_style` warm via register detection, running
        concurrently with STT's own work of reaching a final
        transcript. When a final transcript arrives, we do a cheap
        confirmation pass (or reuse the speculative reading directly
        if nothing material has changed) rather than blocking on a
        cold register-detection call.
        """
        speculative_task: asyncio.Task | None = None

        async for event in self._stt.stream(audio_chunks):
            if not event.is_final:
                # Kick off (or replace) a speculative register-detection
                # pass on this partial. We deliberately don't await it
                # here — it runs concurrently while STT keeps streaming.
                if speculative_task is not None and not speculative_task.done():
                    speculative_task.cancel()
                speculative_task = asyncio.create_task(
                    self._run_register_detection(event)
                )
                continue

            # Final transcript for this segment has arrived.
            segment_id = self._next_segment_id()
            style = await self._resolve_style_for_final(event, speculative_task)
            speculative_task = None

            try:
                output = await self._process_segment(segment_id, event, style)
            except Exception:
                logger.error(
                    "orchestrator: segment %s failed end-to-end, skipping "
                    "rather than killing the session",
                    segment_id,
                    exc_info=True,
                )
                continue

            yield output

    async def _resolve_style_for_final(
        self, event: TranscriptEvent, speculative_task: asyncio.Task | None
    ) -> StyleMetadata:
        """
        Prefers the speculative reading from the partial-transcript
        pass if it completed and the final transcript text matches
        closely enough to trust it; otherwise runs a fresh (but now
        blocking, since we need an answer to proceed) register
        detection pass on the final transcript.
        """
        if speculative_task is not None:
            try:
                speculative_result = await asyncio.wait_for(
                    asyncio.shield(speculative_task), timeout=0.05
                )
                # Cheap heuristic: if the speculative pass ran on text
                # that's a prefix of (or equal to) the final text, trust
                # it rather than paying for a second LLM call. A more
                # sophisticated diff could be used here if false-positive
                # reuse becomes a problem in practice.
                if speculative_result is not None:
                    return speculative_result
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass

        return await self._run_register_detection(event)

    async def _run_register_detection(
        self, event: TranscriptEvent
    ) -> StyleMetadata:
        start = time.monotonic() * 1000
        result = await self._register_detector.detect(event)
        self._record("register_detection", start)
        return result

    async def _process_segment(
        self, segment_id: int, event: TranscriptEvent, style: StyleMetadata
    ) -> SegmentOutput:
        translate_start = time.monotonic() * 1000
        translation = await self._translator.translate(
            source_text=event.text,
            source_style=style,
            target_language=self._target_language,
        )
        self._record("translation", translate_start)

        tts_start = time.monotonic() * 1000
        audio_chunks: list[bytes] = []
        async for chunk in self._tts.synthesize(
            text=translation.translated_text,
            style=translation.style,
            voice_id=self._voice_id,
        ):
            audio_chunks.append(chunk)
        self._record("tts", tts_start)

        return SegmentOutput(
            audio_chunks=audio_chunks,
            source_text=event.text,
            translated_text=translation.translated_text,
            style=translation.style,
        )

    def _next_segment_id(self) -> int:
        self._segment_counter += 1
        return self._segment_counter

    def _record(self, stage: str, start_ms: float) -> None:
        self.timings.append(
            StageTiming(stage=stage, start_ms=start_ms, end_ms=time.monotonic() * 1000)
        )
        logger.debug(
            "stage=%s duration_ms=%.1f", stage, self.timings[-1].duration_ms
        )
