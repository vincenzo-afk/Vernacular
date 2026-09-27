"""
Tests for AssemblyAISTT against a real local WebSocket server that
speaks the same wire protocol AssemblyAI's v3 streaming API uses
(Begin/Turn/Termination/Error JSON messages, binary audio frames).

This deliberately does NOT mock `websockets.connect` or any internals
of AssemblyAISTT — it runs a real websocket server on localhost and
points AssemblyAISTT at it via the `base_url` override, so these tests
exercise the actual connection, send, and parsing code paths. This is
still an offline test (no real AssemblyAI account or network egress
needed) and belongs in tests/unit per TESTING.md, not tests/live,
since "live" in this repo specifically means "talks to the real
third-party API."
"""

import asyncio
import json

import pytest
import websockets
from websockets.asyncio.server import serve

from app.pipeline.stt import (
    AssemblyAISTT,
    WordTiming,
    _parse_turn_message,
    _stress_heuristic,
)


class FakeAssemblyAIServer:
    """
    Minimal local server that mimics enough of AssemblyAI's v3
    streaming protocol to exercise AssemblyAISTT's connection and
    message-parsing logic: sends a Begin message on connect, then
    plays back a scripted sequence of server->client messages
    (Turn/Termination/Error) after receiving the first audio frame,
    on a short delay to simulate realistic async behavior.
    """

    def __init__(self, scripted_messages: list[dict]) -> None:
        self._scripted_messages = scripted_messages
        self.received_audio: list[bytes] = []
        self._server = None

    async def _handler(self, ws):
        await ws.send(
            json.dumps(
                {
                    "type": "Begin",
                    "id": "test-session-id",
                    "expires_at": 0,
                }
            )
        )
        try:
            async for message in ws:
                if isinstance(message, bytes):
                    self.received_audio.append(message)
                    # Once we've seen at least one audio frame, play
                    # back the scripted server messages.
                    if len(self.received_audio) == 1:
                        for msg in self._scripted_messages:
                            await ws.send(json.dumps(msg))
        except websockets.exceptions.ConnectionClosed:
            pass

    async def __aenter__(self):
        self._server = await serve(self._handler, "localhost", 0)
        port = self._server.sockets[0].getsockname()[1]
        self.url = f"ws://localhost:{port}"
        return self

    async def __aexit__(self, *exc):
        self._server.close()
        await self._server.wait_closed()


async def _single_chunk_audio():
    yield b"\x00\x01" * 100


# --- Pure parsing helper tests (no server needed) --------------------------


def test_stress_heuristic_flags_notably_longer_words():
    words = [
        WordTiming(word="Oh,", start_ms=0, end_ms=200, confidence=0.99),
        WordTiming(word="great,", start_ms=200, end_ms=900, confidence=0.98),
        WordTiming(word="another", start_ms=1200, end_ms=1400, confidence=0.97),
        WordTiming(word="meeting.", start_ms=1400, end_ms=1600, confidence=0.98),
    ]
    _stress_heuristic(words)
    assert words[1].is_stressed is True  # "great," held much longer
    assert words[0].is_stressed is False
    assert words[2].is_stressed is False


def test_stress_heuristic_noop_on_single_word():
    words = [WordTiming(word="Understood.", start_ms=0, end_ms=500, confidence=0.9)]
    _stress_heuristic(words)  # should not raise
    assert words[0].is_stressed is False


def test_parse_turn_message_final():
    payload = {
        "type": "Turn",
        "end_of_turn": True,
        "transcript": "Hello world.",
        "words": [
            {"text": "Hello", "start": 0, "end": 500, "confidence": 0.99},
            {"text": "world.", "start": 500, "end": 1000, "confidence": 0.98},
        ],
    }
    event = _parse_turn_message(payload)
    assert event.is_final is True
    assert event.text == "Hello world."
    assert len(event.words) == 2
    assert event.words[0].word == "Hello"
    assert event.sentiment is None  # see stt.py module docstring


def test_parse_turn_message_partial():
    payload = {"type": "Turn", "end_of_turn": False, "transcript": "Hello wor", "words": []}
    event = _parse_turn_message(payload)
    assert event.is_final is False


# --- Real connection tests against the local fake server -------------------


@pytest.mark.asyncio
async def test_yields_final_transcript_event_from_real_connection():
    scripted = [
        {
            "type": "Turn",
            "end_of_turn": False,
            "transcript": "Oh, great",
            "words": [],
        },
        {
            "type": "Turn",
            "end_of_turn": True,
            "transcript": "Oh, great, another meeting.",
            "words": [
                {"text": "Oh,", "start": 0, "end": 200, "confidence": 0.99},
                {"text": "great,", "start": 200, "end": 900, "confidence": 0.98},
            ],
        },
        {"type": "Termination"},
    ]

    async with FakeAssemblyAIServer(scripted) as server:
        stt = AssemblyAISTT(api_key="fake-key", base_url=server.url.replace("ws://", "ws://"))
        # AssemblyAISTT builds URLs assuming a wss:// AssemblyAI-style
        # base; our local test server is plain ws://, which is fine —
        # _build_url only appends query params, it doesn't care about
        # scheme.
        events = []
        async for event in stt.stream(_single_chunk_audio()):
            events.append(event)

        assert len(events) == 2
        assert events[0].is_final is False
        assert events[0].text == "Oh, great"
        assert events[1].is_final is True
        assert events[1].text == "Oh, great, another meeting."
        assert len(events[1].words) == 2
        assert server.received_audio  # confirms audio was actually sent


@pytest.mark.asyncio
async def test_ignores_error_message_and_continues():
    scripted = [
        {"type": "Error", "error": "something non-fatal"},
        {
            "type": "Turn",
            "end_of_turn": True,
            "transcript": "Still works.",
            "words": [],
        },
        {"type": "Termination"},
    ]

    async with FakeAssemblyAIServer(scripted) as server:
        stt = AssemblyAISTT(api_key="fake-key", base_url=server.url)
        events = [event async for event in stt.stream(_single_chunk_audio())]

        assert len(events) == 1
        assert events[0].text == "Still works."


@pytest.mark.asyncio
async def test_raises_if_server_does_not_send_begin_first():
    async def bad_handler(ws):
        # Send a Turn message before any Begin -- protocol violation.
        await ws.send(json.dumps({"type": "Turn", "end_of_turn": True, "transcript": "x", "words": []}))
        await asyncio.sleep(0.05)

    server = await serve(bad_handler, "localhost", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        stt = AssemblyAISTT(
            api_key="fake-key",
            base_url=f"ws://localhost:{port}",
        )
        # The connection loop catches the RuntimeError from the
        # missing-Begin check and retries with backoff rather than
        # propagating -- so to observe the failure within a bounded
        # test time, we run with a short overall timeout and expect no
        # events to have been yielded yet.
        events = []
        with pytest.raises(asyncio.TimeoutError):
            async with asyncio.timeout(0.3):
                async for event in stt.stream(_single_chunk_audio()):
                    events.append(event)
        assert events == []
    finally:
        server.close()
        await server.wait_closed()
