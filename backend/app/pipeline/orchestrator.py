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
from dataclasses import dataclass

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
    # True when this segment was synthesized by the fallback TTS
    # provider (reduced style control, stock voice). Surfaced to the
    # frontend so degraded fidelity is never presented as full quality
    # -- see ARCHITECTURE.md §5.
    degraded: bool = False


@dataclass
class SegmentStart:
    """Emitted once per segment, BEFORE any audio, as soon as
    translation is done -- so the UI can show text and tone tags while
    the voice is still being synthesized."""

    source_text: str
    translated_text: str
    style: StyleMetadata
    # Whether this segment's audio is (or will be) from the fallback
    # provider. Known before the first byte because the failover
    # decision is made before any audio is emitted for the segment.
    degraded: bool = False


@dataclass
class AudioChunk:
    data: bytes


@dataclass
class SegmentEnd:
    pass


SegmentEvent = SegmentStart | AudioChunk | SegmentEnd


class SessionOrchestrator:
    def __init__(
        self,
        stt: STTProvider,
        register_detector: RegisterDetector,
        translator: Translator,
        tts: TTSProvider,
        target_language: str,
        voice_id: str,
        fallback_tts: TTSProvider | None = None,
    ) -> None:
        self._stt = stt
        self._register_detector = register_detector
        self._translator = translator
        self._tts = tts
        self._target_language = target_language
        self._voice_id = voice_id
        self._fallback_tts = fallback_tts
        # Sticky: once the primary TTS provider fails, every remaining
        # segment in this session uses the fallback. Flapping between
        # providers mid-session produces jarring voice changes (see
        # pipeline/AGENTS.md), so we never switch back.
        self._using_fallback = False
        self.timings: list[StageTiming] = []
        self._segment_counter = 0

    async def run(
        self, audio_chunks: AsyncIterator[bytes]
    ) -> AsyncIterator[SegmentOutput]:
        """
        Buffered convenience wrapper over run_stream(): reassembles
        each segment's events into one SegmentOutput. Prefer
        run_stream() on the latency-critical path (the WebSocket
        session does) -- this method necessarily waits for the whole
        clip, which is exactly what the streaming design avoids.
        """
        current: SegmentStart | None = None
        chunks: list[bytes] = []
        async for event in self.run_stream(audio_chunks):
            if isinstance(event, SegmentStart):
                current, chunks = event, []
            elif isinstance(event, AudioChunk):
                chunks.append(event.data)
            elif current is not None:  # SegmentEnd
                yield SegmentOutput(
                    audio_chunks=chunks,
                    source_text=current.source_text,
                    translated_text=current.translated_text,
                    style=current.style,
                    degraded=current.degraded,
                )
                current = None

    async def run_stream(
        self, audio_chunks: AsyncIterator[bytes]
    ) -> AsyncIterator[SegmentEvent]:
        """
        Consumes source audio and yields a stream of events per
        completed utterance: SegmentStart, then AudioChunk* as the TTS
        provider emits bytes (NOT after the clip is finished), then
        SegmentEnd. See module docstring for the concurrency model.

        Design: each STT partial launches a speculative register-
        detection task running concurrently with STT's own work of
        reaching a final transcript. When a final transcript arrives,
        we reuse that speculative result if it's still valid (see
        _resolve_style_for_final) rather than blocking on a cold
        register-detection call.
        """
        # Tracks the in-flight speculative register-detection task
        # together with the exact partial-transcript text it was
        # launched against, so _resolve_style_for_final can actually
        # verify the "is this still valid" heuristic described below
        # instead of blindly trusting whatever finished first.
        speculative_task: asyncio.Task | None = None
        speculative_source_text: str = ""

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
                speculative_source_text = event.text
                continue

            # Final transcript for this segment has arrived.
            segment_id = self._next_segment_id()
            style = await self._resolve_style_for_final(
                event, speculative_task, speculative_source_text
            )
            speculative_task = None
            speculative_source_text = ""

            # Isolation contract: a failure inside one segment must
            # never kill the session. We can't wrap a `yield` in
            # try/except and keep streaming, so the segment generator
            # handles its own errors and always terminates cleanly.
            async for segment_event in self._process_segment(segment_id, event, style):
                yield segment_event

    async def _resolve_style_for_final(
        self,
        event: TranscriptEvent,
        speculative_task: asyncio.Task | None,
        speculative_source_text: str,
    ) -> StyleMetadata:
        """
        Prefers the speculative reading from the partial-transcript
        pass if it completed AND the text it ran on is a prefix of (or
        equal to) the final transcript — meaning nothing material
        changed between the partial and the final, so the reading is
        still trustworthy. Otherwise runs a fresh (blocking, since we
        need an answer to proceed) register detection pass on the
        final transcript.

        A speculative task that hasn't finished, or whose source text
        diverged from the final transcript (e.g. STT revised an early
        word), is cancelled rather than abandoned — an unreferenced
        background task left running is a resource leak and makes
        failures in it silently invisible.
        """
        text_still_valid = event.text.startswith(speculative_source_text)

        if speculative_task is not None and text_still_valid:
            try:
                if speculative_task.done():
                    # No race here, so no timeout needed — the result
                    # is already available.
                    return speculative_task.result()
                return await asyncio.wait_for(speculative_task, timeout=0.05)
            except asyncio.TimeoutError:
                # Didn't finish in time to be worth waiting further —
                # cancel it explicitly rather than shielding-and-
                # forgetting, so it doesn't keep running unobserved.
                speculative_task.cancel()
                logger.debug(
                    "orchestrator: speculative register detection for "
                    "segment did not finish in time, cancelling and "
                    "running fresh"
                )
            except Exception:
                # RegisterDetector.detect() is documented to never
                # raise (it resolves failures to neutral_fallback()
                # internally — see pipeline/AGENTS.md), so reaching
                # this branch means that contract was violated
                # somewhere. Defend anyway rather than trusting a
                # cross-module guarantee blindly: log it and fall
                # through to a fresh detection call instead of letting
                # an unexpected exception here kill the whole session.
                logger.exception(
                    "orchestrator: speculative register-detection task "
                    "raised unexpectedly (RegisterDetector should never "
                    "raise) — running a fresh detection instead"
                )

        elif speculative_task is not None and not speculative_task.done():
            # Text diverged from what the speculative pass was run on
            # — its result would no longer be a reliable reading for
            # this final transcript. Cancel rather than let it run to
            # completion for no purpose.
            speculative_task.cancel()

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
    ) -> AsyncIterator[SegmentEvent]:
        try:
            translate_start = time.monotonic() * 1000
            translation = await self._translator.translate(
                source_text=event.text,
                source_style=style,
                target_language=self._target_language,
            )
            self._record("translation", translate_start)
        except Exception:
            logger.exception(
                "orchestrator: segment %s translation failed, skipping "
                "rather than killing the session",
                segment_id,
            )
            return

        tts_start = time.monotonic() * 1000
        first_byte_recorded = False
        started = False
        try:
            async for chunk, degraded in self._synthesize_stream(
                translation.translated_text, translation.style
            ):
                if not started:
                    # First byte for this segment: announce it (with
                    # the now-final `degraded` flag) and record TTFB --
                    # the metric the ~2s budget in ARCHITECTURE.md §3
                    # actually cares about, not total synthesis time.
                    yield SegmentStart(
                        source_text=event.text,
                        translated_text=translation.translated_text,
                        style=translation.style,
                        degraded=degraded,
                    )
                    started = True
                if not first_byte_recorded:
                    self._record("tts_first_byte", tts_start)
                    first_byte_recorded = True
                yield AudioChunk(chunk)
        except Exception:
            logger.exception(
                "orchestrator: segment %s TTS failed, %s",
                segment_id,
                "ending the segment with the audio already sent"
                if started
                else "skipping it",
            )
            if not started:
                return
        else:
            self._record("tts", tts_start)

        if started:
            yield SegmentEnd()

    async def _synthesize_stream(
        self, text: str, style: StyleMetadata
    ) -> AsyncIterator[tuple[bytes, bool]]:
        """
        Streams (chunk, degraded) from the primary provider, or from
        the fallback once we've failed over.

        Failover rules (pipeline/AGENTS.md, refined for streaming):
        - Sticky: after the first primary failure every later segment
          uses the fallback -- flapping between voices mid-session is
          jarring.
        - The failing segment is retried on the fallback ONLY if the
          primary had emitted no audio yet. Once bytes have been sent
          we cannot take them back, and splicing a second voice into
          the middle of one utterance is worse than ending it early,
          so in that case the error propagates and the segment ends
          with what was already delivered (the provider is still
          marked failed for future segments).
        """
        if not self._using_fallback:
            emitted = False
            try:
                async for chunk in self._tts.synthesize(
                    text=text, style=style, voice_id=self._voice_id
                ):
                    emitted = True
                    yield chunk, False
                return
            except Exception:
                if self._fallback_tts is None:
                    raise
                logger.exception(
                    "orchestrator: primary TTS failed, switching to fallback "
                    "provider for the remainder of the session"
                )
                self._using_fallback = True
                if emitted:
                    raise

        if self._fallback_tts is None:  # pragma: no cover - defensive
            raise RuntimeError("fallback requested but none configured")
        async for chunk in self._fallback_tts.synthesize(
            text=text, style=style, voice_id=self._voice_id
        ):
            yield chunk, True

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
