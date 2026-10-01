"""
Acoustic analysis + multimodal fusion, verified on SYNTHETIC signals
(sine + harmonics with known pitch/level). This proves the measurement
math; it does not prove the arousal heuristic matches real emotional
speech -- see the prosody.py module docstring.
"""

import math
import random
import struct
import time

import pytest

from app.pipeline.prosody import (
    FRAME_BYTES,
    ProsodyAnalyzer,
    arousal_level,
    describe_acoustic,
    fuse_modalities,
    measured_cues,
)
from app.pipeline.stt import WordTiming
from app.schemas.style_metadata import (
    AcousticFeatures,
    Arousal,
    CueKind,
    CueSource,
    Emotion,
    StyleExplanation,
    StyleMetadata,
    Cue,
    Tone,
)

RATE = 16000


def voice(hz: float, secs: float, amp: float) -> bytes:
    n = int(RATE * secs)
    out = []
    for i in range(n):
        t = i / RATE
        v = (
            math.sin(2 * math.pi * hz * t)
            + 0.5 * math.sin(2 * math.pi * 2 * hz * t)
            + 0.25 * math.sin(2 * math.pi * 3 * hz * t)
        )
        out.append(int(max(-32768, min(32767, v * amp))))
    return struct.pack(f"<{n}h", *out)


def silence(secs: float) -> bytes:
    return b"\x00\x00" * int(RATE * secs)


def noise(secs: float, amp: float) -> bytes:
    rng = random.Random(1)
    n = int(RATE * secs)
    return struct.pack(f"<{n}h", *[int(rng.gauss(0, amp)) for _ in range(n)])


def analyze(audio: bytes, chunk: int = 4096) -> AcousticFeatures | None:
    a = ProsodyAnalyzer()
    for i in range(0, len(audio), chunk):
        a.feed(audio[i : i + chunk])
    return a.snapshot()


@pytest.mark.parametrize("hz", [110, 150, 220, 320])
def test_pitch_is_recovered_within_two_percent(hz):
    f = analyze(voice(hz, 1.0, 6000))
    assert f is not None and f.pitch_mean_hz is not None
    assert abs(f.pitch_mean_hz - hz) / hz < 0.02


def test_silence_and_faint_noise_yield_no_speech_and_no_pitch():
    assert analyze(silence(1.0)) is None
    f = analyze(noise(1.0, 3000))
    # loud broadband noise counts as sound but must not be given a pitch
    assert f is None or f.pitch_mean_hz is None


def test_louder_voice_measures_louder_and_more_aroused_against_baseline():
    a = ProsodyAnalyzer()
    # Establish this speaker's baseline with a normal-level utterance.
    for _ in range(3):
        a.feed(voice(150, 1.0, 4000))
        a.finish_utterance()
    a.feed(voice(150, 1.0, 4000))
    normal = a.snapshot()
    a.finish_utterance()
    a.feed(voice(150, 1.0, 16000))
    loud = a.snapshot()
    assert loud.energy_delta_db > normal.energy_delta_db + 8
    # A real margin, not just '>': a broken loudness term must fail this.
    assert loud.arousal_score - normal.arousal_score > 0.2
    assert abs(normal.arousal_score - 0.5) < 0.1  # ~ baseline => ~ neutral
    assert abs(normal.energy_delta_db) < 1.5  # ~ baseline


def test_higher_pitch_than_baseline_raises_arousal():
    a = ProsodyAnalyzer()
    for _ in range(3):
        a.feed(voice(130, 1.0, 5000))
        a.finish_utterance()
    a.feed(voice(130, 1.0, 5000))
    base = a.snapshot()
    a.finish_utterance()
    a.feed(voice(200, 1.0, 5000))
    high = a.snapshot()
    assert high.pitch_delta_pct > 40
    assert high.arousal_score - base.arousal_score > 0.15


def test_chunking_does_not_change_the_result():
    audio = voice(180, 1.0, 6000)
    whole = analyze(audio, chunk=len(audio))
    odd = analyze(audio, chunk=173)  # never frame-aligned, odd sizes
    assert odd.pitch_mean_hz == pytest.approx(whole.pitch_mean_hz, rel=0.01)
    assert odd.energy_db == pytest.approx(whole.energy_db, abs=0.5)


def test_too_little_speech_reports_nothing():
    a = ProsodyAnalyzer()
    a.feed(voice(150, 0.06, 6000))  # 2 frames < MIN_SPEECH_FRAMES
    assert a.snapshot() is None


def test_finish_utterance_reports_speech_span_and_resets():
    a = ProsodyAnalyzer()
    a.feed(silence(0.5) + voice(150, 1.0, 6000) + silence(0.5))
    u = a.finish_utterance()
    assert u.features is not None
    assert 450 <= u.speech_start_ms <= 600
    assert 1450 <= u.speech_end_ms <= 1600
    again = a.finish_utterance()
    assert again.features is None and again.speech_start_ms is None


def test_memory_is_bounded_on_a_long_monologue():
    a = ProsodyAnalyzer()
    for _ in range(60):  # 60 s of speech, never finished
        a.feed(voice(150, 1.0, 6000))
    assert len(a._u_energies) <= 1500
    assert len(a._u_pitches) <= 600
    assert a.snapshot() is not None


def test_malformed_input_is_tolerated():
    a = ProsodyAnalyzer()
    a.feed(b"")
    a.feed(b"\x01")  # a single stray byte
    a.feed(b"\x01" * 3)
    assert a.snapshot() is None


def test_analysis_is_far_faster_than_real_time():
    """The hot-path guard: feeding audio must cost a tiny fraction of
    its own duration. 20% is ~40x looser than measured, so this only
    trips on a real regression (e.g. an accidental O(n^2))."""
    audio = voice(180, 10.0, 6000)
    a = ProsodyAnalyzer()
    start = time.perf_counter()
    for i in range(0, len(audio), 1024):
        a.feed(audio[i : i + 1024])
    elapsed = time.perf_counter() - start
    assert elapsed < 0.20 * 10.0


# ------------------------------------------------------------------ fusion


def acoustic(arousal: float, **kw) -> AcousticFeatures:
    return AcousticFeatures(arousal_score=arousal, duration_ms=1000, **kw)


def test_arousal_levels():
    assert arousal_level(0.1) == Arousal.LOW
    assert arousal_level(0.5) == Arousal.MEDIUM
    assert arousal_level(0.9) == Arousal.HIGH


def test_fuse_without_audio_returns_style_unchanged():
    style = StyleMetadata(tone=Tone.WARM)
    assert fuse_modalities(style, None) is style


def test_positive_words_in_a_quiet_flat_voice_is_a_modality_conflict():
    style = StyleMetadata(tone=Tone.WARM, emotion=Emotion.POSITIVE)
    out = fuse_modalities(style, acoustic(0.1, energy_delta_db=-8))
    assert out.modality_conflict is True
    assert out.arousal == Arousal.LOW
    kinds = {(c.kind, c.source) for c in out.explanation.cues}
    assert (CueKind.INCONGRUENCE, CueSource.MEASURED) in kinds


def test_matching_words_and_voice_is_not_a_conflict():
    style = StyleMetadata(tone=Tone.URGENT)
    assert fuse_modalities(style, acoustic(0.9)).modality_conflict is False
    # quiet + neutral words: nothing to contradict
    assert fuse_modalities(StyleMetadata(), acoustic(0.1)).modality_conflict is False


def test_very_short_speech_never_claims_a_conflict():
    style = StyleMetadata(tone=Tone.EXCITED)
    short = AcousticFeatures(arousal_score=0.05, duration_ms=150)
    assert fuse_modalities(style, short).modality_conflict is False


def test_fusion_is_idempotent_and_keeps_llm_cues():
    style = StyleMetadata(
        tone=Tone.WARM,
        explanation=StyleExplanation(
            summary="s", cues=[Cue(kind=CueKind.LEXICAL, evidence="great")]
        ),
    )
    a = acoustic(0.1, energy_delta_db=-9)
    once = fuse_modalities(style, a)
    twice = fuse_modalities(once, a)
    assert once.explanation.cues == twice.explanation.cues
    assert [c.source for c in twice.explanation.cues].count(CueSource.LLM) == 1
    assert twice.explanation.summary == "s"


def test_fusion_never_rewrites_the_llm_tone_or_emotion():
    style = StyleMetadata(tone=Tone.WARM, emotion=Emotion.AFFECTION)
    out = fuse_modalities(style, acoustic(0.05))
    assert (out.tone, out.emotion) == (Tone.WARM, Emotion.AFFECTION)


def test_measured_cues_from_pauses_and_stress():
    words = [
        WordTiming("Oh,", 0, 200, 0.9),
        WordTiming("great,", 200, 700, 0.9, is_stressed=True),
        WordTiming("another", 1500, 1800, 0.9),
    ]
    cues = measured_cues(StyleMetadata(), None, words)
    text = " ".join(c.evidence for c in cues)
    assert "pause" in text and "great" in text
    assert all(c.source == CueSource.MEASURED for c in cues)


def test_describe_acoustic_is_one_compact_line():
    line = describe_acoustic(
        acoustic(0.8, energy_delta_db=6.0, pitch_delta_pct=20.0, pitch_range_hz=45.0)
    )
    assert "\n" not in line
    assert "high" in line and "+6 dB" in line and "+20%" in line
