"""
End-to-end over a REAL WebSocket: FastAPI's TestClient drives
run_session() with a fake STT/LLM/TTS orchestrator. This is the
closest offline stand-in for what the browser sees, and it pins the
protocol ordering the frontend depends on:

    ready  ->  segment(JSON)  ->  binary audio frames  ->  segment ...

It also verifies the audio survives the trip byte-exact and decodes,
using the same little-endian int16 -> float rule as
frontend/lib/pcm.ts, to the samples the TTS layer produced.
"""

import json
import struct

from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient

from app.pipeline.orchestrator import SessionOrchestrator
from app.pipeline.register_detector import RegisterDetector
from app.pipeline.stt import TranscriptEvent
from app.pipeline.translator import Translator
from app.ws.session import run_session
from tests.fakes.fake_llm import FakeLLMClient, ScriptedResponse
from tests.fakes.fake_tts import FakeTTS

REGISTER = json.dumps(
    {
        "tone": "sarcastic", "pace": "slow", "formality": "casual",
        "emotion": "frustration", "sarcasm_score": 0.9, "emphasis_words": [],
        "pause_pattern": "dramatic", "confidence": 0.9,
    }
)
TRANSLATION = json.dumps(
    {
        "translated_text": "Ay, qué bien.",
        "style": {
            "tone": "sarcastic", "pace": "slow", "formality": "casual",
            "emotion": "frustration", "sarcasm_score": 0.9, "emphasis_words": [],
            "pause_pattern": "dramatic", "confidence": 0.9,
        },
    }
)

# Known PCM16 LE samples the fake "TTS" will emit.
SAMPLES = [0, 16384, -16384, 1000, -2000, 32767, -32768, 5]
PCM = struct.pack("<" + "h" * len(SAMPLES), *SAMPLES)


class ScriptedSTT:
    def __init__(self, texts):
        self._events = [TranscriptEvent(text=t, is_final=True) for t in texts]

    async def stream(self, audio_chunks):
        # Consume one audio frame first, as the real STT would.
        async for _ in audio_chunks:
            break
        for e in self._events:
            yield e


def _app(texts, tts, fallback=None) -> FastAPI:
    n = len(texts)

    async def ws_endpoint(websocket: WebSocket):
        await websocket.accept()
        orch = SessionOrchestrator(
            stt=ScriptedSTT(texts),
            register_detector=RegisterDetector(
                llm_client=FakeLLMClient([ScriptedResponse(REGISTER)] * n)
            ),
            translator=Translator(
                llm_client=FakeLLMClient([ScriptedResponse(TRANSLATION)] * n)
            ),
            tts=tts,
            fallback_tts=fallback,
            target_language="es",
            voice_id="v",
        )
        await run_session(websocket, orch)

    app = FastAPI()
    app.add_api_websocket_route("/ws", ws_endpoint)
    return app


def _decode(pcm: bytes) -> list[int]:
    return list(struct.unpack("<" + "h" * (len(pcm) // 2), pcm))


def test_ready_first_then_segment_then_byte_exact_audio():
    client = TestClient(_app(["Oh, great."], FakeTTS(chunk=PCM)))
    with client.websocket_connect("/ws") as ws:
        ws.send_bytes(b"\x00\x00")  # one mic frame to kick the fake STT

        ready = json.loads(ws.receive_text())
        assert ready["type"] == "ready"
        assert ready["audio"] == {
            "encoding": "pcm_s16le", "sample_rate": 24000, "channels": 1,
        }

        seg = json.loads(ws.receive_text())
        assert seg["type"] == "segment"
        assert seg["translated_text"] == "Ay, qué bien."
        assert seg["degraded"] is False
        assert seg["style"]["tone"] == "sarcastic"

        # FakeTTS yields the clip as two frames; concatenated they must
        # be byte-identical to what the "TTS" produced.
        audio = ws.receive_bytes() + ws.receive_bytes()
        assert audio == PCM
        assert _decode(audio) == SAMPLES


def test_wire_preserves_order_so_odd_reframing_reassembles_the_samples():
    """
    A real network may re-frame the stream so a frame ends mid-sample.
    The wire must therefore preserve byte ORDER and CONTENT exactly;
    the client then carries the dangling byte (frontend PcmAligner).
    Here we re-split the received stream at an odd offset and apply
    the same carry rule, and require the original samples back.
    """
    client = TestClient(_app(["x"], FakeTTS(chunk=PCM)))
    with client.websocket_connect("/ws") as ws:
        ws.send_bytes(b"\x00\x00")
        ws.receive_text()  # ready
        ws.receive_text()  # segment
        stream = ws.receive_bytes() + ws.receive_bytes()

    odd_split = 5  # lands inside the third sample
    carry, whole_parts = b"", []
    for chunk in (stream[:odd_split], stream[odd_split:]):
        data = carry + chunk
        usable = len(data) - (len(data) % 2)
        whole_parts.append(data[:usable])
        carry = data[usable:]

    assert carry == b""
    assert _decode(b"".join(whole_parts)) == SAMPLES


def test_degraded_flag_reaches_the_client_after_primary_failure():
    primary = FakeTTS(fail=True)
    fallback = FakeTTS(chunk=PCM)
    client = TestClient(_app(["one"], primary, fallback))
    with client.websocket_connect("/ws") as ws:
        ws.send_bytes(b"\x00\x00")
        ws.receive_text()  # ready
        seg = json.loads(ws.receive_text())

    assert seg["degraded"] is True
