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
import collections
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
        """
        Owns ONE long-lived reader over the caller's audio iterator and
        feeds an asyncio.Queue; each connection's sender reads from that
        queue.

        Why not iterate `audio_chunks` inside the per-connection sender
        (which is what this used to do): cancelling a sender that is
        suspended inside `async for chunk in audio_chunks` throws
        CancelledError into the source async generator and CLOSES it.
        The next connection's sender then iterated an already-finished
        generator and got nothing -- so the first network blip
        permanently killed the microphone feed while the session looked
        alive (reconnected, replayed a little audio, then silence).
        Confirmed by tests/unit/test_stt_reconnect.py.

        The queue also fixes a second loss: a chunk pulled from the
        source but not yet delivered when a connection died used to
        vanish. Now a chunk is only removed from the pipeline once a
        send has actually succeeded.
        """
        queue: asyncio.Queue[bytes | None] = asyncio.Queue()

        async def read_source() -> None:
            try:
                async for chunk in audio_chunks:
                    await queue.put(chunk)
            finally:
                await queue.put(None)  # end-of-audio sentinel

        reader = asyncio.create_task(read_source())

        # Recently *delivered* audio, kept so a drop can replay what the
        # server may not have finished processing. Bounded so a long
        # outage can't grow it without limit (oldest is dropped).
        replay: collections.deque[bytes] = collections.deque()
        max_replay_bytes = int(MAX_BUFFER_SECONDS * self._bytes_per_second)
        pending: bytes | None = None  # pulled from queue, not yet delivered
        source_done = False
        delay = RECONNECT_INITIAL_DELAY_S

        try:
            while True:
                try:
                    async with websockets.connect(
                        self._build_url(),
                        additional_headers={"Authorization": self._api_key},
                        max_size=None,
                    ) as ws:
                        begin = json.loads(await ws.recv())
                        if begin.get("type") != "Begin":
                            raise RuntimeError(
                                f"Expected 'Begin' message, got: {begin.get('type')}"
                            )
                        delay = RECONNECT_INITIAL_DELAY_S
                        logger.info(
                            "assemblyai_stt: session started id=%s", begin.get("id")
                        )

                        # Replay what the previous connection may not have
                        # finished. Kept in `replay` (not cleared) so a
                        # second drop can replay it again.
                        for chunk in list(replay):
                            await ws.send(chunk)

                        state = {"pending": pending, "done": source_done}
                        sender = asyncio.create_task(
                            self._send_loop(
                                ws, queue, state, replay, max_replay_bytes
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
                                # Unknown types are ignored so a future
                                # protocol addition doesn't break us.
                        finally:
                            sender.cancel()
                            try:
                                await sender
                            except asyncio.CancelledError:
                                pass  # we just cancelled it
                            except Exception:
                                logger.warning(
                                    "assemblyai_stt: sender raised during shutdown",
                                    exc_info=True,
                                )
                            # Carry whatever the sender hadn't delivered
                            # into the next connection.
                            pending = state["pending"]
                            source_done = state["done"]

                        if source_done and pending is None and queue.empty():
                            return  # audio ended and everything was sent
                        raise ConnectionError("server closed the connection")

                except Exception:
                    if source_done and pending is None and queue.empty():
                        return
                    logger.warning(
                        "assemblyai_stt: connection error, reconnecting in %.1fs",
                        delay,
                        exc_info=True,
                    )
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, RECONNECT_MAX_DELAY_S)
        finally:
            reader.cancel()
            try:
                await reader
            except asyncio.CancelledError:
                pass  # we just cancelled it
            except Exception:
                # The caller's audio source itself failed. That is a
                # real error, not cleanup noise -- surface it.
                logger.warning(
                    "assemblyai_stt: audio source raised", exc_info=True
                )

    @staticmethod
    async def _send_loop(
        ws,
        queue: "asyncio.Queue[bytes | None]",
        state: dict,
        replay: "collections.deque[bytes]",
        max_replay_bytes: int,
    ) -> None:
        """
        Delivers queued audio to one connection. A chunk is held in
        `state["pending"]` from the moment it leaves the queue until
        `ws.send` succeeds, so if this task is cancelled or the socket
        dies mid-send, the chunk is not lost -- stream() carries it to
        the next connection.
        """
        while True:
            if state["pending"] is None:
                item = await queue.get()
                if item is None:
                    state["done"] = True
                    return
                state["pending"] = item
            await ws.send(state["pending"])
            chunk = state["pending"]
            state["pending"] = None

            replay.append(chunk)
            total = sum(len(c) for c in replay)
            while total > max_replay_bytes and replay:
                total -= len(replay.popleft())
