"""
Reconnect behavior of AssemblyAISTT against a real local server that
kills connections on demand.

Regression for a real, previously-shipped bug: after the first dropped
connection the session reconnected and replayed a little buffered
audio, then went permanently silent. Cause: the per-connection sender
iterated the caller's async generator directly; cancelling it mid
`async for` closed that generator, so every later connection got an
already-finished source. A single network blip therefore killed the
microphone feed while the session looked alive.
"""

import asyncio
import json

import pytest
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

from app.pipeline.stt import AssemblyAISTT


class DroppingServer:
    """Kills the first `drop_first_after` frames' connection abruptly,
    then behaves normally, recording what each connection received."""

    def __init__(self, drop_first_after: int, answer_after: int):
        self.sessions: list[list[bytes]] = []
        self._drop_after = drop_first_after
        self._answer_after = answer_after

    async def handler(self, ws):
        idx = len(self.sessions)
        got: list[bytes] = []
        self.sessions.append(got)
        await ws.send(json.dumps({"type": "Begin", "id": f"s{idx}"}))
        try:
            async for msg in ws:
                if not isinstance(msg, bytes):
                    continue
                got.append(msg)
                if idx == 0 and len(got) == self._drop_after:
                    await ws.close(code=1011)
                    return
                if idx >= 1 and len(got) >= self._answer_after:
                    await ws.send(
                        json.dumps(
                            {"type": "Turn", "end_of_turn": True,
                             "transcript": "recovered", "words": []}
                        )
                    )
                    await ws.send(json.dumps({"type": "Termination"}))
        except ConnectionClosed:  # torn down by the test or the client
            return


async def _audio(n: int, gap: float = 0.03):
    for i in range(n):
        yield bytes([i]) * 4
        await asyncio.sleep(gap)


async def _run(server: DroppingServer, audio, timeout: float = 8.0):
    async with serve(server.handler, "localhost", 0) as srv:
        port = srv.sockets[0].getsockname()[1]
        stt = AssemblyAISTT(api_key="k", base_url=f"ws://localhost:{port}")
        texts = []
        async with asyncio.timeout(timeout):
            async for ev in stt.stream(audio):
                texts.append(ev.text)
        return texts


@pytest.mark.asyncio
async def test_live_audio_resumes_after_a_dropped_connection():
    server = DroppingServer(drop_first_after=3, answer_after=6)

    texts = await _run(server, _audio(10))

    assert texts == ["recovered"]
    second = [f[0] for f in server.sessions[1]]
    # The second connection must receive audio that was produced AFTER
    # the drop (frames 3+), not merely a replay of what came before.
    assert max(second) >= 5, f"live audio never resumed; saw only {second}"


@pytest.mark.asyncio
async def test_replays_recent_audio_on_the_new_connection():
    server = DroppingServer(drop_first_after=3, answer_after=6)

    await _run(server, _audio(10))

    first = [f[0] for f in server.sessions[0]]
    second = [f[0] for f in server.sessions[1]]
    assert first == [0, 1, 2]
    assert second[:3] == [0, 1, 2]  # replayed, in order, before live audio


@pytest.mark.asyncio
async def test_no_frame_is_lost_or_reordered_across_the_drop():
    server = DroppingServer(drop_first_after=3, answer_after=8)

    await _run(server, _audio(10))

    second = [f[0] for f in server.sessions[1]]
    live = [x for x in second if x >= 3]
    assert live == sorted(live), f"frames reordered: {live}"
    assert live[:4] == [3, 4, 5, 6], f"gap in live audio after reconnect: {live}"


@pytest.mark.asyncio
async def test_source_iterator_is_not_closed_by_a_reconnect():
    """
    The direct cause of the original bug, asserted precisely.

    The source is legitimately cancelled once at the very end, when
    stream() returns and cleans up its reader -- so "was it ever
    closed" is the wrong question. The right one: was it closed
    BEFORE the reconnected session finished receiving live audio? We
    record how many frames the second connection had by the time the
    source was closed; if the reconnect had closed it, that count
    would be stuck at the replay size (3) instead of the full stream.
    """
    server = DroppingServer(drop_first_after=3, answer_after=6)
    frames_on_second_when_closed: list[int] = []

    async def source():
        try:
            for i in range(10):
                yield bytes([i]) * 4
                await asyncio.sleep(0.03)
        except (GeneratorExit, asyncio.CancelledError):
            second = server.sessions[1] if len(server.sessions) > 1 else []
            frames_on_second_when_closed.append(len(second))
            raise

    texts = await _run(server, source())

    assert texts == ["recovered"]
    assert frames_on_second_when_closed, "source was never closed at all"
    # 3 replayed frames + live frames. If the reconnect had closed the
    # source this would still be 3 (replay only), as in the old bug.
    assert frames_on_second_when_closed[0] > 3, (
        "source was closed before the reconnected session got live "
        f"audio (second connection had {frames_on_second_when_closed[0]} frames)"
    )
