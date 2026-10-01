"""
Wire-level behaviour of run_session against an in-memory WebSocket:
live captions, network-aware adaptation, RTT ping, summary requests,
and resilience to junk control frames.

Uses a hand-rolled fake socket (not Starlette's TestClient) so timing
is fully controlled -- e.g. slow sends can be simulated to prove the
SERVER notices congestion by itself.
"""

import asyncio
import json

import pytest
from fastapi import WebSocketDisconnect

from app.pipeline.orchestrator import SessionOrchestrator
from app.pipeline.register_detector import RegisterDetector
from app.pipeline.stt import TranscriptEvent
from app.pipeline.summarizer import ConversationSummarizer
from app.pipeline.translator import Translator
from app.ws.session import run_session
from tests.fakes.fake_llm import FakeLLMClient, ScriptedResponse

STYLE = {
    "tone": "sarcastic", "pace": "slow", "formality": "casual",
    "emotion": "frustration", "sarcasm_score": 0.9, "emphasis_words": [],
    "pause_pattern": "dramatic", "confidence": 0.9,
}
TRANSLATION = json.dumps({"translated_text": "Ay, qué bien.", "style": STYLE})


class FakeWebSocket:
    def __init__(self, text_delay: float = 0.0):
        self.incoming: asyncio.Queue = asyncio.Queue()
        self.sent: list[tuple[str, object]] = []
        self._delay = text_delay

    async def receive(self):
        return await self.incoming.get()

    async def send_text(self, text: str):
        if self._delay:
            await asyncio.sleep(self._delay)
        self.sent.append(("text", json.loads(text)))

    async def send_bytes(self, data: bytes):
        self.sent.append(("bytes", data))

    # -- test helpers
    def client_sends_audio(self, data: bytes = b"GO"):
        self.incoming.put_nowait({"type": "websocket.receive", "bytes": data})

    def client_sends_json(self, obj):
        text = obj if isinstance(obj, str) else json.dumps(obj)
        self.incoming.put_nowait({"type": "websocket.receive", "text": text})

    def client_disconnects(self):
        self.incoming.put_nowait({"type": "websocket.disconnect", "code": 1000})

    def messages(self, kind: str) -> list[dict]:
        return [m for t, m in self.sent if t == "text" and m.get("type") == kind]

    def audio(self) -> bytes:
        return b"".join(m for t, m in self.sent if t == "bytes")

    def frames(self) -> list[bytes]:
        return [m for t, m in self.sent if t == "bytes"]


class ScriptedSTT:
    """Waits for the client's b'GO' audio frame, then plays the script.
    Consumes audio afterwards so control frames keep being processed."""

    def __init__(self, events):
        self._events = events

    async def stream(self, audio_chunks):
        started = False
        async for chunk in audio_chunks:
            if not started and chunk == b"GO":
                started = True
                for e in self._events:
                    yield e


class ChunkyTTS:
    def __init__(self, n: int = 10, size: int = 100):
        self._n, self._size = n, size

    async def synthesize(self, text, style, voice_id):
        for i in range(self._n):
            yield bytes([65 + i]) * self._size


def partials_then_final():
    return [
        TranscriptEvent(text="Oh", is_final=False),
        TranscriptEvent(text="Oh gr", is_final=False),
        TranscriptEvent(text="Oh, great", is_final=False),
        TranscriptEvent(text="Oh, great.", is_final=True),
    ]


def make(events, tts=None, ws=None, summarizer=None):
    ws = ws or FakeWebSocket()
    orch = SessionOrchestrator(
        stt=ScriptedSTT(events),
        register_detector=RegisterDetector(
            FakeLLMClient([ScriptedResponse(json.dumps(STYLE))] * 8), explain=False
        ),
        translator=Translator(FakeLLMClient([ScriptedResponse(TRANSLATION)] * 8)),
        tts=tts or ChunkyTTS(),
        target_language="es",
        voice_id="v",
        emit_captions=True,
    )
    return ws, orch, summarizer


async def run(ws, orch, summarizer=None, *, until, timeout=3.0):
    """Runs the session; disconnects the client once `until()` is true."""
    task = asyncio.create_task(
        run_session(ws, orch, summarizer=summarizer, target_language="es")
    )
    deadline = asyncio.get_event_loop().time() + timeout
    while not until():
        assert asyncio.get_event_loop().time() < deadline, "condition never met"
        await asyncio.sleep(0.01)
    ws.client_disconnects()
    await asyncio.wait_for(task, timeout=2)


def segment_done(ws):
    return lambda: len(ws.messages("segment")) >= 1 and len(ws.audio()) >= 1000


# ---------------------------------------------------------------- basics


@pytest.mark.asyncio
async def test_ready_first_then_captions_then_segment_then_audio():
    ws, orch, _ = make(partials_then_final())
    ws.client_sends_audio()
    await run(ws, orch, until=segment_done(ws))

    assert ws.sent[0][1]["type"] == "ready"
    types = [m["type"] for t, m in ws.sent if t == "text"]
    assert types.index("caption") < types.index("segment")
    seg = ws.messages("segment")[0]
    assert seg["segment_id"] == 1 and seg["key_moment"]["is_key"] is True
    assert "server_total_ms" in seg["timings"] and seg["context_turns"] == 0
    final_caption = [c for c in ws.messages("caption") if c["is_final"]][0]
    assert final_caption["segment_id"] == seg["segment_id"]
    assert len(ws.audio()) == 1000  # GOOD tier: every byte, none dropped


@pytest.mark.asyncio
async def test_good_network_forwards_every_partial_and_every_frame_uncoalesced():
    ws, orch, _ = make(partials_then_final())
    ws.client_sends_audio()
    await run(ws, orch, until=segment_done(ws))
    assert len([c for c in ws.messages("caption") if not c["is_final"]]) == 3
    assert len(ws.frames()) == 10  # 10 TTS frames -> 10 wire frames


# -------------------------------------------------------- adaptive: client


@pytest.mark.asyncio
async def test_poor_client_report_throttles_partials_but_never_the_final():
    ws, orch, _ = make(partials_then_final())
    ws.client_sends_json({"type": "network", "tier": "poor", "rtt_ms": 900})
    ws.client_sends_audio()
    await run(ws, orch, until=segment_done(ws))

    announced = ws.messages("network")
    assert announced and announced[0]["tier"] == "poor" and announced[0]["client_rtt_ms"] == 900
    captions = ws.messages("caption")
    assert len([c for c in captions if not c["is_final"]]) == 1  # 3 partials -> 1
    assert captions[-1]["is_final"] is True and captions[-1]["text"] == "Oh, great."


@pytest.mark.asyncio
async def test_poor_tier_coalesces_audio_without_changing_a_single_byte():
    expected = b"".join(bytes([65 + i]) * 100 for i in range(10))
    ws, orch, _ = make(partials_then_final())
    ws.client_sends_json({"type": "network", "tier": "poor"})
    ws.client_sends_audio()
    await run(ws, orch, until=segment_done(ws))
    assert ws.audio() == expected
    assert len(ws.frames()) < 10  # fewer, larger packets


@pytest.mark.asyncio
async def test_coalesced_audio_is_flushed_at_the_segment_boundary_not_held():
    """A short clip below the coalesce target must still be delivered."""
    ws, orch, _ = make(partials_then_final(), tts=ChunkyTTS(n=2, size=10))
    ws.client_sends_json({"type": "network", "tier": "poor"})
    ws.client_sends_audio()
    await run(ws, orch, until=lambda: len(ws.audio()) >= 20)
    assert ws.audio() == b"A" * 10 + b"B" * 10


# -------------------------------------------------------- adaptive: server


@pytest.mark.asyncio
async def test_server_detects_congestion_itself_from_slow_sends():
    ws = FakeWebSocket(text_delay=0.25)  # every JSON send takes 250 ms
    ws, orch, _ = make(partials_then_final(), ws=ws)
    ws.client_sends_audio()  # NOTE: the client never reports anything
    await run(ws, orch, until=lambda: any(m["tier"] == "poor" for m in ws.messages("network")))
    assert ws.messages("network")[0]["send_latency_ms"] >= 200


# -------------------------------------------------------------- controls


@pytest.mark.asyncio
async def test_ping_is_answered_with_the_same_id():
    ws, orch, _ = make([])
    ws.client_sends_json({"type": "ping", "id": "p7"})
    ws.client_sends_audio(b"x")
    await run(ws, orch, until=lambda: bool(ws.messages("pong")))
    assert ws.messages("pong") == [{"type": "pong", "id": "p7"}]


@pytest.mark.asyncio
async def test_summarize_returns_an_llm_summary_of_the_conversation_so_far():
    ws, orch, _ = make(partials_then_final())
    summary_llm = FakeLLMClient([ScriptedResponse(json.dumps({
        "overview": "A sarcastic remark about meetings.", "key_points": ["Oh, great."],
        "action_items": [], "overall_tone": "sarcastic",
    }))])
    ws.client_sends_audio()
    task = asyncio.create_task(
        run_session(ws, orch, summarizer=ConversationSummarizer(summary_llm), target_language="es")
    )
    for _ in range(200):  # wait for the segment, THEN ask for a summary
        if ws.messages("segment") and len(ws.audio()) >= 1000:
            break
        await asyncio.sleep(0.01)
    ws.client_sends_json({"type": "summarize", "language": "target"})
    for _ in range(200):
        if ws.messages("summary"):
            break
        await asyncio.sleep(0.01)
    ws.client_disconnects()
    await asyncio.wait_for(task, timeout=2)

    summary = ws.messages("summary")[0]
    assert summary["generated_by"] == "llm" and summary["turn_count"] == 1
    assert "Oh, great." in summary_llm.calls[0]["user"]
    assert "'es'" in summary_llm.calls[0]["system"]


@pytest.mark.asyncio
async def test_summarize_without_a_summarizer_still_answers_extractively():
    ws, orch, _ = make(partials_then_final())
    ws.client_sends_audio()
    task = asyncio.create_task(run_session(ws, orch))
    for _ in range(200):
        if ws.messages("segment") and len(ws.audio()) >= 1000:
            break
        await asyncio.sleep(0.01)
    ws.client_sends_json({"type": "summarize"})
    for _ in range(200):
        if ws.messages("summary"):
            break
        await asyncio.sleep(0.01)
    ws.client_disconnects()
    await asyncio.wait_for(task, timeout=2)
    assert ws.messages("summary")[0]["generated_by"] == "extractive"


@pytest.mark.asyncio
async def test_junk_control_frames_are_ignored_and_the_session_survives():
    ws, orch, _ = make(partials_then_final())
    for junk in ("not json", "[1,2]", '"str"', '{"type":"warp"}', '{"type":"network","tier":"banana"}', "{}"):
        ws.client_sends_json(junk)
    ws.client_sends_audio()
    await run(ws, orch, until=segment_done(ws))
    assert ws.messages("segment") and not ws.messages("error") and not ws.messages("network")


@pytest.mark.asyncio
async def test_client_disconnect_ends_the_session_cleanly():
    ws, orch, _ = make([])
    ws.client_disconnects()
    await asyncio.wait_for(run_session(ws, orch), timeout=2)
    assert not ws.messages("error")
