"""
Stage 1: Speech-to-text.

Provider-agnostic interface (STTProvider) + AssemblyAI v3 streaming
implementation. See ARCHITECTURE.md §4 — no AssemblyAI SDK/protocol
details should leak outside this file. Anything that needs a
transcript should depend on STTProvider, not on AssemblyAI directly.

AssemblyAI v3 streaming protocol notes (see backend/app/pipeline/AGENTS.md
and README.md for why this matters to get right):

- Endpoint: wss://streaming.assemblyai.com/v3/ws
- Auth: `Authorization` header carrying the raw API key (not "Bearer ...").
- Connection query params include sample_rate (required) and
  format_turns (bool, whether finals come back punctuated/formatted).
- Audio is sent as raw binary WebSocket frames — PCM16 little-endian,
  the sample rate declared in the query params.
- Server sends JSON text messages of several `type`s:
    "Begin"        - session started, includes a session id
    "Turn"         - a transcript update. `end_of_turn: true` means this
                     is the finalized version of this turn (our
                     TranscriptEvent.is_final=True); otherwise it's a
                     partial we should treat as an interim update.
                     Carries `transcript` (str) and `words` (list of
                     {text, start, end, confidence}, ms-based timings).
    "Termination"  - server closed the session (e.g. after we sent a
                     terminate message, or on timeout).
    "Error"        - a protocol-level error occurred.
- IMPORTANT: AssemblyAI's v3 streaming API does not include sentiment
  analysis in its message payload — sentiment analysis in AssemblyAI's
  product is an *async* (post-hoc, full-file) transcription feature,
  not part of the realtime websocket stream. TranscriptEvent.sentiment
  is therefore always None from this provider. Register detection
  (pipeline/register_detector.py) already treats sentiment as optional
  and derives pacing/stress signals from word timings instead — see
  its docstring. Don't quietly "restore" a fake sentiment field here;
  if sentiment ends up being a hard requirement later, it needs a
  separate async AssemblyAI call or a different provider, which is a
  real architectural change, not a one-line fix.
"""

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Protocol
from urllib.parse import urlencode

import websockets

logger = logging.getLogger("vernacular.stt")

ASSEMBLYAI_WS_BASE_URL = "wss://streaming.assemblyai.com/v3/ws"

# Reconnect policy for dropped connections mid-session — see
# pipeline/AGENTS.md failure-handling contract for stt.py.
RECONNECT_INITIAL_DELAY_S = 0.2
RECONNECT_MAX_DELAY_S = 5.0
MAX_BUFFER_SECONDS = 3.0


@dataclass
class WordTiming:
    word: str
    start_ms: int
    end_ms: int
    confidence: float
    is_stressed: bool = False  # informs StyleMetadata.emphasis_words;
    # AssemblyAI's v3 stream doesn't label stress directly, so this is
    # derived heuristically in _stress_heuristic() below rather than
    # taken verbatim from the provider.


@dataclass
class TranscriptEvent:
    text: str
    is_final: bool
    words: list[WordTiming] = field(default_factory=list)
    sentiment: str | None = None  # see module docstring — always None
    # for AssemblyAI's realtime stream; kept on the dataclass because
    # STTProvider is meant to support future providers that DO supply
    # a realtime sentiment signal.


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


def _stress_heuristic(words: list[WordTiming]) -> None:
    """
    Mutates `words` in place, setting is_stressed=True on words whose
    duration is notably longer than the segment's median word
    duration. This is a coarse proxy for prosodic stress in the
    absence of pitch/energy data from the v3 streaming API (which
    doesn't expose that) — see register_detector.py's few-shot
    examples, which describe emphasis in terms of held/elongated words.

    This is explicitly a heuristic, not a claim of acoustic accuracy.
    If a future provider or AssemblyAI feature exposes real prosodic
    stress data, prefer that over this durations-only proxy.
    """
    if len(words) < 2:
        return
    durations = sorted(w.end_ms - w.start_ms for w in words)
    mid = len(durations) // 2
    median = (
        durations[mid]
        if len(durations) % 2
        else (durations[mid - 1] + durations[mid]) / 2
    )
    if median <= 0:
        return
    for w in words:
        duration = w.end_ms - w.start_ms
        if duration >= median * 1.6:
            w.is_stressed = True


def _parse_turn_message(payload: dict) -> TranscriptEvent:
    words = [
        WordTiming(
            word=w.get("text", ""),
            start_ms=int(w.get("start", 0)),
            end_ms=int(w.get("end", 0)),
            confidence=float(w.get("confidence", 0.0)),
        )
        for w in payload.get("words", [])
    ]
    _stress_heuristic(words)
    return TranscriptEvent(
        text=payload.get("transcript", ""),
        is_final=bool(payload.get("end_of_turn", False)),
        words=words,
        sentiment=None,
    )


class AssemblyAISTT:
    """
    AssemblyAI v3 realtime streaming implementation of STTProvider.

    Connects once per `stream()` call, forwards audio chunks as binary
    frames, and yields a TranscriptEvent per "Turn" message received.
    Reconnects with exponential backoff on a dropped connection,
    buffering up to MAX_BUFFER_SECONDS of audio so an in-flight
    utterance isn't silently lost — per the failure-handling contract
    in backend/app/pipeline/AGENTS.md.
    """

    def __init__(
        self,
        api_key: str,
        sample_rate: int = 16000,
        format_turns: bool = True,
        base_url: str = ASSEMBLYAI_WS_BASE_URL,
    ) -> None:
        self._api_key = api_key
        self._sample_rate = sample_rate
        self._format_turns = format_turns
        # Overridable so tests can point this at a local fake server
        # implementing the same wire protocol, instead of mocking
        # internals. Defaults to the real AssemblyAI endpoint.
        self._base_url = base_url
        # Rough estimate for buffer sizing: PCM16 mono => 2 bytes/sample.
        self._bytes_per_second = sample_rate * 2

    def _build_url(self) -> str:
        params = {
            "sample_rate": self._sample_rate,
            "encoding": "pcm_s16le",
            "format_turns": str(self._format_turns).lower(),
        }
        return f"{self._base_url}?{urlencode(params)}"

    async def stream(
        self, audio_chunks: AsyncIterator[bytes]
    ) -> AsyncIterator[TranscriptEvent]:
        # A bounded ring-style buffer of recently-sent audio, so that if
        # the connection drops we can replay what the server likely
        # hadn't finished processing yet, rather than losing it. This
        # is a best-effort recovery, not a guarantee — AssemblyAI does
        # not support resuming a session with prior context, so a
        # reconnect always starts a fresh session server-side.
        recent_audio: list[bytes] = []
        max_buffer_bytes = int(MAX_BUFFER_SECONDS * self._bytes_per_second)

        audio_iter = audio_chunks.__aiter__()
        delay = RECONNECT_INITIAL_DELAY_S

        while True:
            try:
                async with websockets.connect(
                    self._build_url(),
                    additional_headers={"Authorization": self._api_key},
                    max_size=None,
                ) as ws:
                    delay = RECONNECT_INITIAL_DELAY_S  # reset backoff on success

                    # Confirm the session actually began before treating
                    # the connection as usable.
                    begin_raw = await ws.recv()
                    begin_msg = json.loads(begin_raw)
                    if begin_msg.get("type") != "Begin":
                        raise RuntimeError(
                            f"Expected 'Begin' message, got: {begin_msg.get('type')}"
                        )
                    logger.info(
                        "assemblyai_stt: session started id=%s",
                        begin_msg.get("id"),
                    )

                    # Replay whatever we'd buffered from a prior dropped
                    # connection before resuming live audio.
                    for chunk in recent_audio:
                        await ws.send(chunk)
                    recent_audio.clear()

                    send_task = asyncio.create_task(
                        self._pump_audio(
                            ws, audio_iter, recent_audio, max_buffer_bytes
                        )
                    )

                    try:
                        async for raw in ws:
                            try:
                                payload = json.loads(raw)
                            except json.JSONDecodeError:
                                logger.warning(
                                    "assemblyai_stt: non-JSON message, skipping"
                                )
                                continue

                            msg_type = payload.get("type")
                            if msg_type == "Turn":
                                yield _parse_turn_message(payload)
                            elif msg_type == "Termination":
                                logger.info(
                                    "assemblyai_stt: session terminated by server"
                                )
                                return
                            elif msg_type == "Error":
                                logger.error(
                                    "assemblyai_stt: server error: %s",
                                    payload.get("error"),
                                )
                            # "Begin" already handled above; unknown
                            # types are ignored rather than raising, so
                            # a future protocol addition doesn't break
                            # this integration outright.
                    finally:
                        send_task.cancel()
                        try:
                            await send_task
                        except asyncio.CancelledError:
                            pass  # expected — we just cancelled it above
                        except Exception:
                            # A real failure in _pump_audio (as opposed
                            # to our own cancellation) shouldn't be
                            # silently dropped — log it, but don't let
                            # it mask whatever caused us to reach this
                            # `finally` in the first place.
                            logger.warning(
                                "assemblyai_stt: audio pump task raised "
                                "during shutdown",
                                exc_info=True,
                            )

                    # audio_iter exhausted with no server-side
                    # termination — the session ended normally.
                    return

            except StopAsyncIteration:
                # Caller's audio source ended cleanly.
                return
            except Exception:
                logger.warning(
                    "assemblyai_stt: connection error, reconnecting in %.1fs",
                    delay,
                    exc_info=True,
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, RECONNECT_MAX_DELAY_S)
                # Loop back around and reconnect; recent_audio still
                # holds whatever hadn't been confirmed sent.
                continue

    @staticmethod
    async def _pump_audio(
        ws,
        audio_iter,
        recent_audio: list[bytes],
        max_buffer_bytes: int,
    ) -> None:
        """
        Forwards audio chunks from the caller's async iterator to the
        websocket as binary frames, while keeping a bounded trailing
        buffer for reconnect replay. Runs as its own task so receiving
        Turn messages is never blocked waiting on the next audio chunk.
        """
        buffered_bytes = sum(len(c) for c in recent_audio)
        async for chunk in audio_iter:
            await ws.send(chunk)
            recent_audio.append(chunk)
            buffered_bytes += len(chunk)
            while buffered_bytes > max_buffer_bytes and recent_audio:
                dropped = recent_audio.pop(0)
                buffered_bytes -= len(dropped)
