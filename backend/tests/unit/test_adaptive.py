"""Network-aware adaptive streaming policy."""

from app.adaptive import (
    POLICIES,
    RECOVERY_OBSERVATIONS,
    AdaptiveController,
    AudioCoalescer,
    CaptionThrottle,
    NetworkTier,
    parse_network_report,
    tier_for_send_latency,
)


def test_send_latency_thresholds():
    assert tier_for_send_latency(5) == NetworkTier.GOOD
    assert tier_for_send_latency(80) == NetworkTier.FAIR
    assert tier_for_send_latency(500) == NetworkTier.POOR


def test_good_tier_changes_nothing():
    p = POLICIES[NetworkTier.GOOD]
    assert p.caption_min_interval_ms == 0 and p.audio_coalesce_bytes == 0


def test_client_report_degrades_immediately():
    c = AdaptiveController()
    assert c.report_client(NetworkTier.POOR, rtt_ms=900) is True
    assert c.policy.tier == NetworkTier.POOR and c.client_rtt_ms == 900


def test_server_observed_congestion_degrades_without_client_help():
    c = AdaptiveController()
    changed = [c.observe_send(400) for _ in range(3)]
    assert any(changed) and c.policy.tier == NetworkTier.POOR


def test_the_worse_of_the_two_signals_wins():
    c = AdaptiveController()
    c.report_client(NetworkTier.FAIR)
    for _ in range(5):
        c.observe_send(500)
    assert c.policy.tier == NetworkTier.POOR
    c.report_client(NetworkTier.GOOD)  # client says fine, server still sees stalls
    assert c.policy.tier == NetworkTier.POOR


def test_recovery_needs_a_streak_and_steps_down_one_tier_at_a_time():
    c = AdaptiveController()
    c.report_client(NetworkTier.POOR)
    for _ in range(RECOVERY_OBSERVATIONS - 1):
        assert c.report_client(NetworkTier.GOOD) is False
    assert c.policy.tier == NetworkTier.POOR
    assert c.report_client(NetworkTier.GOOD) is True
    assert c.policy.tier == NetworkTier.FAIR  # not straight to GOOD


def test_a_bad_sample_resets_the_recovery_streak():
    c = AdaptiveController()
    c.report_client(NetworkTier.POOR)
    for _ in range(RECOVERY_OBSERVATIONS - 1):
        c.report_client(NetworkTier.GOOD)
    c.report_client(NetworkTier.POOR)
    for _ in range(RECOVERY_OBSERVATIONS - 1):
        c.report_client(NetworkTier.GOOD)
    assert c.policy.tier == NetworkTier.POOR


def test_caption_throttle_drops_partials_but_never_finals():
    t = CaptionThrottle()
    assert t.allow(0, False, 700) is True  # first always goes out
    assert t.allow(100, False, 700) is False
    assert t.allow(300, True, 700) is True  # final: always
    assert t.allow(400, False, 700) is False  # measured from the final
    assert t.allow(1100, False, 700) is True


def test_caption_throttle_is_a_noop_when_interval_is_zero():
    t = CaptionThrottle()
    assert all(t.allow(i, False, 0) for i in range(5))


def test_coalescer_merges_small_frames_preserving_bytes_and_order():
    c = AudioCoalescer()
    assert c.push(b"AA", 6) is None
    assert c.push(b"BB", 6) is None
    assert c.push(b"CC", 6) == b"AABBCC"
    assert c.push(b"D", 6) is None
    assert c.flush() == b"D"
    assert c.flush() is None


def test_coalescer_passthrough_when_target_is_zero():
    assert AudioCoalescer().push(b"xyz", 0) == b"xyz"


def test_parse_network_report_validates():
    assert parse_network_report({"tier": "poor", "rtt_ms": 300}) == (NetworkTier.POOR, 300.0)
    assert parse_network_report({"tier": "fair"}) == (NetworkTier.FAIR, None)
    assert parse_network_report({"tier": "banana"}) is None
    assert parse_network_report({}) is None
    assert parse_network_report({"tier": "good", "rtt_ms": -5}) == (NetworkTier.GOOD, None)
    assert parse_network_report({"tier": "good", "rtt_ms": True}) == (NetworkTier.GOOD, None)
