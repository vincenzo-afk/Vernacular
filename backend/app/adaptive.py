"""
Network-aware adaptive streaming policy.

The pipeline's ~2 s budget assumes a decent link. On a weak one (mobile
data, congested Wi-Fi) the bottleneck stops being the providers and
becomes the number of small messages we push through it, so the server
adapts what it sends -- without ever dropping content:

- **Live captions** (partial transcripts) are throttled. Partials are
  superseded by the next partial anyway, so dropping intermediate ones
  loses nothing; finals are ALWAYS sent.
- **Translated audio frames** are coalesced into larger frames (fewer
  packets, less per-message overhead). This can delay the first byte by
  at most one coalesce window (100-200 ms of audio), only on degraded
  links, and it never waits for a whole clip -- CLAUDE.md constraint #2
  still holds. Frames are flushed at every segment boundary.

Two independent signals choose the tier, and the WORSE one wins:

1. What the client reports (`{"type":"network","tier":...}`): it sees
   round-trip time, the browser's connection type and its own socket
   send buffer (lib/network.ts).
2. What the server observes: how long `websocket.send` takes. That call
   only stalls when the client's receive path is backed up, so it is a
   real congestion signal that needs no client cooperation.

Degrading is immediate; recovering needs several consecutive better
observations (hysteresis), so a flapping link does not flap the
stream's framing.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.pipeline.tts import AUDIO_SAMPLE_RATE


class NetworkTier(str, Enum):
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


_RANK = {NetworkTier.GOOD: 0, NetworkTier.FAIR: 1, NetworkTier.POOR: 2}
_BY_RANK = {v: k for k, v in _RANK.items()}


def _pcm_bytes(ms: int) -> int:
    return AUDIO_SAMPLE_RATE * 2 * ms // 1000  # PCM16 mono


@dataclass(frozen=True)
class StreamPolicy:
    tier: NetworkTier
    caption_min_interval_ms: int  # 0 = no throttling
    audio_coalesce_bytes: int  # 0 = forward every TTS frame as-is


POLICIES: dict[NetworkTier, StreamPolicy] = {
    NetworkTier.GOOD: StreamPolicy(NetworkTier.GOOD, 0, 0),
    NetworkTier.FAIR: StreamPolicy(NetworkTier.FAIR, 250, _pcm_bytes(100)),
    NetworkTier.POOR: StreamPolicy(NetworkTier.POOR, 700, _pcm_bytes(200)),
}

# Server-observed send-latency thresholds (ms, smoothed).
SEND_FAIR_MS = 60.0
SEND_POOR_MS = 200.0
_EWMA_ALPHA = 0.3
RECOVERY_OBSERVATIONS = 8


def _worse(a: NetworkTier, b: NetworkTier) -> NetworkTier:
    return a if _RANK[a] >= _RANK[b] else b


def tier_for_send_latency(ms: float) -> NetworkTier:
    if ms >= SEND_POOR_MS:
        return NetworkTier.POOR
    if ms >= SEND_FAIR_MS:
        return NetworkTier.FAIR
    return NetworkTier.GOOD


class AdaptiveController:
    """Holds the current tier; both `report_*` methods return True iff it changed."""

    def __init__(self) -> None:
        self._client = NetworkTier.GOOD
        self._send_ewma: float | None = None
        self._tier = NetworkTier.GOOD
        self._better_streak = 0
        self.client_rtt_ms: float | None = None

    @property
    def policy(self) -> StreamPolicy:
        return POLICIES[self._tier]

    @property
    def send_latency_ms(self) -> float | None:
        return self._send_ewma

    def report_client(self, tier: NetworkTier, rtt_ms: float | None = None) -> bool:
        self._client = tier
        if rtt_ms is not None:
            self.client_rtt_ms = rtt_ms
        return self._recompute()

    def observe_send(self, duration_ms: float) -> bool:
        if self._send_ewma is None:
            self._send_ewma = duration_ms
        else:
            self._send_ewma += _EWMA_ALPHA * (duration_ms - self._send_ewma)
        return self._recompute()

    def _recompute(self) -> bool:
        server = (
            tier_for_send_latency(self._send_ewma)
            if self._send_ewma is not None
            else NetworkTier.GOOD
        )
        target = _worse(self._client, server)
        if _RANK[target] > _RANK[self._tier]:
            self._tier, self._better_streak = target, 0
            return True
        if _RANK[target] < _RANK[self._tier]:
            self._better_streak += 1
            if self._better_streak >= RECOVERY_OBSERVATIONS:
                # Recover one step at a time, never straight to GOOD.
                self._tier = _BY_RANK[_RANK[self._tier] - 1]
                self._better_streak = 0
                return True
            return False
        self._better_streak = 0
        return False


class CaptionThrottle:
    """Drops intermediate partial captions faster than the policy allows."""

    def __init__(self) -> None:
        self._last_sent_ms: float | None = None

    def allow(self, now_ms: float, is_final: bool, min_interval_ms: int) -> bool:
        if (
            is_final
            or min_interval_ms <= 0
            or self._last_sent_ms is None
            or now_ms - self._last_sent_ms >= min_interval_ms
        ):
            self._last_sent_ms = now_ms
            return True
        return False


class AudioCoalescer:
    """
    Merges small TTS frames into frames of at least `target_bytes`.
    Byte order and content are preserved exactly (the wire test in
    tests/integration/test_ws_roundtrip.py depends on that).
    """

    def __init__(self) -> None:
        self._buf = bytearray()

    def push(self, data: bytes, target_bytes: int) -> bytes | None:
        self._buf += data
        if target_bytes <= 0 or len(self._buf) >= target_bytes:
            return self.flush()
        return None

    def flush(self) -> bytes | None:
        if not self._buf:
            return None
        out = bytes(self._buf)
        self._buf.clear()
        return out


def parse_network_report(message: dict) -> tuple[NetworkTier, float | None] | None:
    """Validates a client `network` control message; None if malformed."""
    try:
        tier = NetworkTier(message.get("tier"))
    except ValueError:
        return None
    rtt = message.get("rtt_ms")
    if isinstance(rtt, bool) or not isinstance(rtt, (int, float)) or rtt < 0:
        rtt = None
    return tier, (float(rtt) if rtt is not None else None)
