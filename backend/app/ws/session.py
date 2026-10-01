"""
WebSocket session handling — one connection per translation session.

Receives raw audio chunks from the frontend, feeds them into a
SessionOrchestrator, and streams back translated audio plus live
StyleMetadata tags (for the frontend's tone-tag UI — see
frontend/components/ToneTags.tsx and frontend/lib/ws-client.ts, which
this wire format must stay in sync with — see CONTRIBUTING.md "Known
gaps" on why that sync is currently manual).

Wire format (backend -> frontend). First, once per session:

    {"type": "ready", "audio": {"encoding": "pcm_s16le",
                                "sample_rate": 24000, "channels": 1}}

declaring the format of every binary frame that follows. Then one JSON
text message per completed segment, followed by binary audio-chunk
messages for that segment:

    {"type": "segment", "segment_id": 1, "source_text": "...",
     "translated_text": "...", "style": {...StyleMetadata fields...},
     "degraded": false, "key_moment": {...}, "timings": {...},
     "speech_start_ms": 120, "speech_end_ms": 1900, "context_turns": 2}
    <binary audio chunk>
    <binary audio chunk>
    ...

Other backend -> frontend messages (all JSON text):

    {"type": "caption", "text": "...", "is_final": false, "segment_id": null}
        live source-language caption; finals carry the segment_id the
        segment message will use, and arrive BEFORE translation is done.
    {"type": "network", "tier": "good|fair|poor", "send_latency_ms": 12.0,
     "client_rtt_ms": 40.0}      -- the adaptive-streaming tier changed
    {"type": "pong", "id": "..."}     -- reply to a client ping (RTT probe)
    {"type": "summary", ...}          -- reply to a client summarize request

Frontend -> backend: binary frames are mic audio (PCM16 16 kHz mono);
JSON text frames are control messages:

    {"type": "network", "tier": "good|fair|poor", "rtt_ms": 80}
    {"type": "ping", "id": "..."}
    {"type": "summarize", "language": "target" | "source"}
"""

import asyncio
import json
import logging
import time

from fastapi import WebSocket, WebSocketDisconnect

from app.adaptive import (
    AdaptiveController,
    AudioCoalescer,
    CaptionThrottle,
    parse_network_report,
)
from app.config import get_settings
from app.pipeline.orchestrator import (
    AudioChunk,
    CaptionEvent,
    SegmentEnd,
    SegmentOutput,
    SegmentStart,
    SessionOrchestrator,
)
from app.pipeline.register_detector import RegisterDetector
from app.pipeline.stt import AssemblyAISTT
from app.pipeline.summarizer import ConversationSummarizer, extractive_summary
from app.pipeline.translator import Translator
from app.pipeline.tts import (
    AUDIO_CHANNELS,
    AUDIO_ENCODING,
    AUDIO_SAMPLE_RATE,
    ElevenLabsTTS,
    FallbackTTS,
)

logger = logging.getLogger("vernacular.ws.session")


def ready_to_wire() -> dict:
    """
    First message on every session. Declares the audio format of all
    binary frames that follow, so the client never hardcodes it: the
    server is the single source of truth (pipeline/tts.py). Mirrors
    `ReadyMessage` in frontend/lib/ws-client.ts.
    """
    return {
        "type": "ready",
        "audio": {
            "encoding": AUDIO_ENCODING,
            "sample_rate": AUDIO_SAMPLE_RATE,
            "channels": AUDIO_CHANNELS,
        },
    }


def segment_start_to_wire(segment: SegmentStart | SegmentOutput) -> dict:
    """
    The JSON control message sent ahead of each segment's binary audio
    chunks. Extracted from handle_session so the wire contract is a
    pure, testable function: its keys must match `SegmentMessage` in
    frontend/lib/ws-client.ts exactly (there is no schema codegen --
    tests/unit/test_wire_contract.py cross-checks the two).

    Accepts either the streaming SegmentStart or the buffered
    SegmentOutput; they carry the same fields.
    """
    return {
        "type": "segment",
        "segment_id": segment.segment_id,
        "source_text": segment.source_text,
        "translated_text": segment.translated_text,
        "style": segment.style.model_dump(mode="json"),
        "degraded": segment.degraded,
        "key_moment": segment.key_moment.model_dump(mode="json"),
        "timings": segment.timings.to_wire(),
        "speech_start_ms": segment.speech_start_ms,
        "speech_end_ms": segment.speech_end_ms,
        "context_turns": segment.context_turns,
    }


def caption_to_wire(event: CaptionEvent) -> dict:
    """Live caption message. Mirrors `CaptionMessage` in ws-client.ts."""
    return {
        "type": "caption",
        "text": event.text,
        "is_final": event.is_final,
        "segment_id": event.segment_id,
    }


def network_to_wire(controller: AdaptiveController) -> dict:
    """Mirrors `NetworkMessage` in ws-client.ts."""
    send = controller.send_latency_ms
    return {
        "type": "network",
        "tier": controller.policy.tier.value,
        "send_latency_ms": None if send is None else round(send, 1),
        "client_rtt_ms": controller.client_rtt_ms,
    }


def summary_to_wire(summary) -> dict:
    """Mirrors `SummaryMessage` in ws-client.ts."""
    return {"type": "summary", **summary.to_wire()}


# Backwards-compatible name used by the wire-contract tests.
segment_to_wire = segment_start_to_wire


def _build_llm_client(settings):
    """
    Constructs the LLM client used by both RegisterDetector and
    Translator, based on settings.llm_provider. Kept here rather than
    in config.py so pipeline modules stay decoupled from any one
    provider SDK — see CLAUDE.md constraint #4.
    """
    if settings.llm_provider == "openai":
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=settings.llm_api_key)

        class _OpenAIAdapter:
            async def complete(self, *, system: str, user: str, **kwargs) -> str:
                response = await client.chat.completions.create(
                    model=settings.llm_model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=kwargs.get("temperature", 0.3),
                    max_tokens=kwargs.get("max_tokens", 500),
                )
                return response.choices[0].message.content or ""

        return _OpenAIAdapter()

    if settings.llm_provider == "anthropic":
        from anthropic import AsyncAnthropic

        client = AsyncAnthropic(api_key=settings.llm_api_key)

        class _AnthropicAdapter:
            async def complete(self, *, system: str, user: str, **kwargs) -> str:
                response = await client.messages.create(
                    model=settings.llm_model,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                    temperature=kwargs.get("temperature", 0.3),
                    max_tokens=kwargs.get("max_tokens", 500),
                )
                return "".join(
                    block.text for block in response.content if block.type == "text"
                )

        return _AnthropicAdapter()

    raise NotImplementedError(
        f"LLM provider '{settings.llm_provider}' not wired yet — add an "
        "adapter above following the _OpenAIAdapter / _AnthropicAdapter "
        "pattern. Groq and Google are listed as options in README.md but "
        "not yet implemented."
    )


async def handle_session(websocket: WebSocket, target_language: str) -> None:
    await websocket.accept()

    settings = get_settings()

    llm_client = _build_llm_client(settings)
    stt = AssemblyAISTT(api_key=settings.assemblyai_api_key)
    register_detector = RegisterDetector(
        llm_client=llm_client, explain=settings.register_explain
    )
    translator = Translator(llm_client=llm_client)
    summarizer = ConversationSummarizer(llm_client=llm_client)

    tts = ElevenLabsTTS(api_key=settings.elevenlabs_api_key)
    fallback_tts = None
    if settings.fallback_tts_api_key:
        # Sticky failover target: the orchestrator switches to this on
        # the first ElevenLabs failure and stays on it for the rest of
        # the session (see pipeline/AGENTS.md). Segments it produces
        # are flagged `degraded` on the wire.
        fallback_tts = FallbackTTS(
            provider=settings.fallback_tts_provider,
            api_key=settings.fallback_tts_api_key,
        )
    else:
        logger.warning(
            "ws.session: no FALLBACK_TTS_API_KEY configured — an "
            "ElevenLabs failure will drop segments instead of failing "
            "over to a fallback voice."
        )

    orchestrator = SessionOrchestrator(
        stt=stt,
        register_detector=register_detector,
        translator=translator,
        tts=tts,
        fallback_tts=fallback_tts,
        target_language=target_language,
        voice_id="default",  # TODO: voice cloning calibration step, gated
        # on explicit consent — see CONTRIBUTING.md "Known gaps"
        emit_captions=True,
    )

    await run_session(
        websocket,
        orchestrator,
        summarizer=summarizer,
        target_language=target_language,
    )


class _Sender:
    """
    Serializes sends on one websocket (the audio loop, the control
    reader and summary tasks all send) and feeds each send's duration
    to the adaptive controller: `send` only stalls when the client's
    receive path is backed up, which makes it a genuine congestion
    signal that needs no client cooperation.
    """

    def __init__(self, websocket: WebSocket, controller: AdaptiveController) -> None:
        self._ws = websocket
        self._controller = controller
        self._lock = asyncio.Lock()

    async def text(self, payload: dict) -> None:
        changed = await self._send(self._ws.send_text, json.dumps(payload))
        if changed:
            await self.announce_tier()

    async def data(self, payload: bytes) -> None:
        changed = await self._send(self._ws.send_bytes, payload)
        if changed:
            await self.announce_tier()

    async def announce_tier(self) -> None:
        # Raw send: announcing a tier change must not feed the
        # controller that just changed tier.
        async with self._lock:
            await self._ws.send_text(json.dumps(network_to_wire(self._controller)))

    async def _send(self, fn, arg) -> bool:
        async with self._lock:
            start = time.monotonic()
            await fn(arg)
            elapsed_ms = (time.monotonic() - start) * 1000
        return self._controller.observe_send(elapsed_ms)


async def run_session(
    websocket: WebSocket,
    orchestrator: SessionOrchestrator,
    summarizer: ConversationSummarizer | None = None,
    target_language: str | None = None,
) -> None:
    """
    The wire-protocol half of a session, separated from provider
    construction (handle_session) so it can be exercised end-to-end
    with fake providers over a real socket -- see
    tests/integration/test_ws_roundtrip.py.

    Sends the `ready` message first, then streams `caption` messages
    (if the orchestrator emits them) and `segment` messages each
    followed by that segment's binary audio frames. Adapts framing to
    the network tier (app/adaptive.py) and answers client control
    messages (network report, ping, summarize).
    """
    controller = AdaptiveController()
    sender = _Sender(websocket, controller)
    throttle = CaptionThrottle()
    coalescer = AudioCoalescer()
    background: set[asyncio.Task] = set()

    await websocket.send_text(json.dumps(ready_to_wire()))

    async def send_summary(language: str) -> None:
        try:
            if summarizer is None:
                summary = extractive_summary(orchestrator.memory)
            else:
                summary = await summarizer.summarize(
                    orchestrator.memory,
                    target_language=target_language if language == "target" else None,
                )
            await sender.text(summary_to_wire(summary))
        except Exception:
            logger.exception("ws.session: summary request failed")

    async def handle_control(text: str) -> None:
        try:
            message = json.loads(text)
        except json.JSONDecodeError:
            logger.debug("ws.session: ignoring non-JSON control frame")
            return
        if not isinstance(message, dict):
            return
        kind = message.get("type")
        if kind == "network":
            report = parse_network_report(message)
            if report is None:
                logger.debug("ws.session: ignoring malformed network report")
            elif controller.report_client(*report):
                await sender.announce_tier()
        elif kind == "ping":
            await sender.text({"type": "pong", "id": message.get("id")})
        elif kind == "summarize":
            language = "source" if message.get("language") == "source" else "target"
            # Off the audio path: an LLM call must never block mic intake.
            task = asyncio.create_task(send_summary(language))
            background.add(task)
            task.add_done_callback(background.discard)
        else:
            logger.debug("ws.session: ignoring unknown control message %r", kind)

    async def audio_in():
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect(message.get("code", 1000))
            if message.get("bytes") is not None:
                yield message["bytes"]
            elif message.get("text") is not None:
                await handle_control(message["text"])

    try:
        # Stream, don't buffer: each audio frame is forwarded the
        # moment the TTS provider emits it, so the listener hears the
        # start of a sentence while the rest is still being
        # synthesized (CLAUDE.md constraint #2, ARCHITECTURE.md §3).
        # On a degraded link small frames are coalesced (never a whole
        # clip; flushed at every segment boundary).
        async for event in orchestrator.run_stream(audio_in()):
            policy = controller.policy
            if isinstance(event, CaptionEvent):
                if throttle.allow(
                    time.monotonic() * 1000, event.is_final, policy.caption_min_interval_ms
                ):
                    await sender.text(caption_to_wire(event))
            elif isinstance(event, SegmentStart):
                leftover = coalescer.flush()
                if leftover:
                    await sender.data(leftover)
                await sender.text(segment_start_to_wire(event))
            elif isinstance(event, AudioChunk):
                frame = coalescer.push(event.data, policy.audio_coalesce_bytes)
                if frame:
                    await sender.data(frame)
            elif isinstance(event, SegmentEnd):
                frame = coalescer.flush()
                if frame:
                    await sender.data(frame)
        frame = coalescer.flush()
        if frame:
            await sender.data(frame)
    except WebSocketDisconnect:
        logger.info("ws.session: client disconnected")
    except Exception:
        logger.exception("ws.session: unhandled error, closing session")
        # Per pipeline/AGENTS.md, individual segment failures are
        # already caught inside the orchestrator and skipped rather
        # than propagating. Reaching here means something outside that
        # contract broke (e.g. the STT stream itself) — surface it to
        # the client rather than hanging.
        try:
            await websocket.send_text(
                json.dumps({"type": "error", "message": "session_failed"})
            )
        except Exception:
            # The socket is very likely already dead at this point
            # (that's usually why the send above failed) — there's
            # nothing further we can do, but log rather than silently
            # swallowing in case this ever fires for a different
            # reason (e.g. a serialization bug in the payload above).
            logger.warning(
                "ws.session: failed to send error notification to client, "
                "socket likely already closed",
                exc_info=True,
            )
    finally:
        for task in background:
            task.cancel()
        await asyncio.gather(*background, return_exceptions=True)
