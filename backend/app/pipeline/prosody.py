"""
Acoustic analysis: the audio half of multimodal emotion detection.

Until now every register signal came from the transcript text plus word
timings -- the actual voice was never looked at. This module measures
the mic audio itself (loudness, pitch, pitch movement) per utterance and
turns it into `AcousticFeatures`, and `fuse_modalities()` combines that
with the LLM's text-based reading into a fused `arousal`, a
`modality_conflict` flag, and *measured* explanation cues.

Design constraints (see CLAUDE.md / pipeline/AGENTS.md):

- **Hot path, so it must be cheap and non-blocking.** `feed()` is called
  inline from the audio tap in the orchestrator for every mic chunk. It
  is pure Python (no numpy dependency), works on 30 ms frames, and does
  pitch tracking on a 4 kHz decimation using prefix sums, so the
  per-frame cost is one dot product per candidate lag. tests/unit/
  test_prosody.py asserts a real-time-factor ceiling so a regression
  here is caught.
- **Never raises into the audio path.** Callers wrap `feed()`; malformed
  input (odd byte counts, tiny chunks) is tolerated.
- **Relative, not absolute.** Loudness and pitch depend on the mic and
  the speaker, so features are reported against the speaker's own
  running baseline (`energy_delta_db`, `pitch_delta_pct`).

What this is NOT: an emotion classifier trained on labeled speech. The
arousal score is a transparent heuristic (louder / higher / more varied
than this speaker's norm => more activated). It has been validated only
on synthetic signals, not on real speakers -- treat it as one signal
that the LLM and the fusion rules weigh, not as ground truth.
"""

from __future__ import annotations

import logging
import math
import operator
import sys
from array import array
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

from app.schemas.style_metadata import (
    AcousticFeatures,
    Arousal,
    Cue,
    CueKind,
    CueSource,
    Emotion,
    StyleExplanation,
    StyleMetadata,
    Tone,
)

logger = logging.getLogger("vernacular.prosody")

SAMPLE_RATE = 16000  # must match the mic capture / AssemblyAISTT (page.tsx)
FRAME_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 480
FRAME_BYTES = FRAME_SAMPLES * 2

_DECIMATE = 4
_PITCH_RATE = SAMPLE_RATE // _DECIMATE  # 4000 Hz
MIN_PITCH_HZ = 75
MAX_PITCH_HZ = 400
_MIN_LAG = _PITCH_RATE // MAX_PITCH_HZ  # 10
_MAX_LAG = _PITCH_RATE // MIN_PITCH_HZ  # 53
_VOICING_THRESHOLD = 0.5  # normalized autocorrelation needed to call a frame voiced

_MIN_SPEECH_RMS = 250.0  # ~ -42 dBFS: below this is never speech
_SPEECH_OVER_NOISE = 3.0
_INITIAL_NOISE_RMS = 150.0
_NOISE_ALPHA = 0.05

MIN_SPEECH_FRAMES = 5  # 150 ms of speech before we report anything
_BASELINE_MIN_FRAMES = 8
_BASELINE_ALPHA = 0.15
# Typical mean voiced-frame level of conversational speech; only used
# until this speaker's own baseline has been learned.
_DEFAULT_BASE_ENERGY_DB = -28.0

_MAX_ENERGY_SAMPLES = 1500  # ~45 s of speech; bounds memory on a monologue
_MAX_PITCH_SAMPLES = 600

# Fusion thresholds.
LOW_AROUSAL_BELOW = 0.35
HIGH_AROUSAL_ABOVE = 0.65
_CONFLICT_AROUSAL_BELOW = 0.30
_CONFLICT_MIN_SPEECH_MS = 300

_HIGH_ENERGY_TONES = {Tone.URGENT, Tone.EXCITED, Tone.ANNOYED}
_HIGH_ENERGY_EMOTIONS = {Emotion.EXCITEMENT, Emotion.FRUSTRATION, Emotion.ANNOYANCE}
_POSITIVE_TONES = {Tone.WARM, Tone.EXCITED}
_POSITIVE_EMOTIONS = {Emotion.POSITIVE, Emotion.EXCITEMENT, Emotion.AFFECTION}

try:  # Python 3.12+: C-speed dot product
    _dot = math.sumprod  # type: ignore[attr-defined]
except AttributeError:  # pragma: no cover - 3.11 fallback

    def _dot(a: Sequence[float], b: Sequence[float]) -> float:
        return sum(map(operator.mul, a, b))


@dataclass
class UtteranceAudio:
    """What the analyzer knows about one finished utterance."""

    features: AcousticFeatures | None
    # Position of the utterance on the mic stream's own clock (ms since
    # the first sample this analyzer saw). None when no speech was heard.
    speech_start_ms: int | None
    speech_end_ms: int | None


class ProsodyAnalyzer:
    """
    Incremental per-utterance voice analysis.

    Usage: `feed()` every mic chunk as it arrives; `snapshot()` any time
    for the utterance-so-far (used by speculative register detection on
    partial transcripts); `finish_utterance()` when a final transcript
    lands, which returns the utterance's features and resets the
    accumulators (the speaker baseline persists across utterances).
    """

    def __init__(self) -> None:
        self._pending = bytearray()
        self._frames_total = 0
        self._noise_rms = _INITIAL_NOISE_RMS
        self._base_energy_db: float | None = None
        self._base_pitch_hz: float | None = None
        self._reset_utterance()

    # ------------------------------------------------------------------ input

    def feed(self, chunk: bytes) -> None:
        """Accepts raw PCM16 little-endian mono at SAMPLE_RATE, any size."""
        if not chunk:
            return
        self._pending += chunk
        usable = (len(self._pending) // FRAME_BYTES) * FRAME_BYTES
        if not usable:
            return
        data = bytes(self._pending[:usable])
        del self._pending[:usable]
        samples = array("h")
        samples.frombytes(data)
        if sys.byteorder == "big":  # pragma: no cover
            samples.byteswap()
        for i in range(0, len(samples), FRAME_SAMPLES):
            self._process_frame(samples[i : i + FRAME_SAMPLES])

    # ----------------------------------------------------------------- output

    def snapshot(self) -> AcousticFeatures | None:
        """Features for the utterance so far; None if too little speech."""
        if self._u_speech < MIN_SPEECH_FRAMES:
            return None
        return self._features()

    def finish_utterance(self) -> UtteranceAudio:
        features = self.snapshot()
        if self._u_first_speech is None or self._u_last_speech is None:
            start_ms = end_ms = None
        else:
            start_ms = self._u_first_speech * FRAME_MS
            end_ms = (self._u_last_speech + 1) * FRAME_MS
        if features is not None and self._u_speech >= _BASELINE_MIN_FRAMES:
            self._update_baseline(features)
        self._reset_utterance()
        return UtteranceAudio(features, start_ms, end_ms)

    # --------------------------------------------------------------- internals

    def _reset_utterance(self) -> None:
        self._u_frames = 0
        self._u_speech = 0
        self._u_energies: list[float] = []
        self._u_pitches: deque[float] = deque(maxlen=_MAX_PITCH_SAMPLES)
        self._u_first_speech: int | None = None
        self._u_last_speech: int | None = None

    def _process_frame(self, frame: array) -> None:
        index = self._frames_total
        self._frames_total += 1
        self._u_frames += 1

        rms = math.sqrt(_dot(frame, frame) / len(frame))
        if rms <= max(_MIN_SPEECH_RMS, self._noise_rms * _SPEECH_OVER_NOISE):
            self._noise_rms += _NOISE_ALPHA * (rms - self._noise_rms)
            return

        self._u_speech += 1
        if self._u_first_speech is None:
            self._u_first_speech = index
        self._u_last_speech = index
        if len(self._u_energies) < _MAX_ENERGY_SAMPLES:
            self._u_energies.append(20.0 * math.log10(max(rms, 1.0) / 32768.0))
        pitch = _estimate_pitch(frame)
        if pitch is not None:
            self._u_pitches.append(pitch)

    def _features(self) -> AcousticFeatures:
        energies = self._u_energies
        mean_db = sum(energies) / len(energies)
        variability = math.sqrt(sum((e - mean_db) ** 2 for e in energies) / len(energies))

        pitches = sorted(self._u_pitches)
        pitch_mean = sum(pitches) / len(pitches) if len(pitches) >= 3 else None
        pitch_range = None
        if len(pitches) >= 5:
            lo = pitches[int(0.1 * (len(pitches) - 1))]
            hi = pitches[int(0.9 * (len(pitches) - 1))]
            pitch_range = hi - lo

        base_e = self._base_energy_db if self._base_energy_db is not None else _DEFAULT_BASE_ENERGY_DB
        delta_e = mean_db - base_e
        delta_p = None
        if pitch_mean is not None and self._base_pitch_hz:
            delta_p = (pitch_mean / self._base_pitch_hz - 1.0) * 100.0

        return AcousticFeatures(
            arousal_score=_arousal_score(delta_e, delta_p, variability),
            energy_db=min(mean_db, 0.0),
            energy_delta_db=delta_e,
            energy_variability_db=variability,
            pitch_mean_hz=pitch_mean,
            pitch_delta_pct=delta_p,
            pitch_range_hz=pitch_range,
            voiced_ratio=self._u_speech / max(self._u_frames, 1),
            duration_ms=self._u_speech * FRAME_MS,
        )

    def _update_baseline(self, features: AcousticFeatures) -> None:
        if self._base_energy_db is None:
            self._base_energy_db = features.energy_db
        else:
            self._base_energy_db += _BASELINE_ALPHA * (features.energy_db - self._base_energy_db)
        if features.pitch_mean_hz is not None:
            if self._base_pitch_hz is None:
                self._base_pitch_hz = features.pitch_mean_hz
            else:
                self._base_pitch_hz += _BASELINE_ALPHA * (
                    features.pitch_mean_hz - self._base_pitch_hz
                )


def _arousal_score(delta_e_db: float, delta_pitch_pct: float | None, variability_db: float) -> float:
    """
    Louder, higher-pitched and more dynamically varied than this
    speaker's norm => more activated. 6 dB, 15 % pitch and 4 dB of
    extra variability each count as "one unit". A transparent heuristic,
    not a trained model (see module docstring).
    """
    e = delta_e_db / 6.0
    v = (variability_db - 4.0) / 4.0
    if delta_pitch_pct is None:
        x = 0.8 * e + 0.2 * v
    else:
        p = (delta_pitch_pct / 100.0) / 0.15
        x = 0.55 * e + 0.35 * p + 0.10 * v
    return 0.5 + 0.5 * math.tanh(x)


def _estimate_pitch(frame: array) -> float | None:
    """
    Autocorrelation pitch on a 4 kHz decimation of one 30 ms frame.
    Returns Hz, or None if the frame is unvoiced. Picks the *smallest*
    lag whose correlation is within 10 % of the best, to avoid the
    classic octave error (locking onto 2x the period).
    """
    m = len(frame) // _DECIMATE
    if m <= _MAX_LAG + 8:
        return None
    x = [
        float(frame[4 * i] + frame[4 * i + 1] + frame[4 * i + 2] + frame[4 * i + 3])
        for i in range(m)
    ]
    mean = sum(x) / m
    x = [v - mean for v in x]

    # Prefix sums of x^2 give each overlap window's energy in O(1).
    prefix = [0.0] * (m + 1)
    acc = 0.0
    for i, v in enumerate(x):
        acc += v * v
        prefix[i + 1] = acc
    if acc < 1e-6:
        return None

    r = [0.0] * (_MAX_LAG + 2)
    for lag in range(_MIN_LAG - 1, _MAX_LAG + 2):
        n = m - lag
        num = _dot(x[:n], x[lag:])
        den = math.sqrt(prefix[n] * (prefix[m] - prefix[lag]))
        r[lag] = num / den if den > 0 else 0.0

    best = max(r[_MIN_LAG : _MAX_LAG + 1])
    if best < _VOICING_THRESHOLD:
        return None
    candidates = [
        L
        for L in range(_MIN_LAG, _MAX_LAG + 1)
        if r[L] >= 0.9 * best and r[L] >= r[L - 1] and r[L] >= r[L + 1]
    ]
    lag = candidates[0] if candidates else max(range(_MIN_LAG, _MAX_LAG + 1), key=lambda L: r[L])
    # Parabolic interpolation around the peak for sub-sample lag.
    denom = r[lag - 1] - 2.0 * r[lag] + r[lag + 1]
    refined = lag + (0.5 * (r[lag - 1] - r[lag + 1]) / denom if denom else 0.0)
    hz = _PITCH_RATE / refined
    return hz if MIN_PITCH_HZ <= hz <= MAX_PITCH_HZ else None


# ------------------------------------------------------------------ fusion


def arousal_level(score: float) -> Arousal:
    if score < LOW_AROUSAL_BELOW:
        return Arousal.LOW
    if score > HIGH_AROUSAL_ABOVE:
        return Arousal.HIGH
    return Arousal.MEDIUM


def _modality_conflict(style: StyleMetadata, acoustic: AcousticFeatures) -> bool:
    """
    True when the words and the voice disagree about how activated the
    speaker is: wording that implies high energy (urgent / excited /
    annoyed) or warmth/positivity, delivered quietly and flatly. Can be
    deliberate (deadpan) -- it is a signal, not a verdict, and is one of
    the classic cues for sarcasm.
    """
    if acoustic.duration_ms < _CONFLICT_MIN_SPEECH_MS:
        return False
    if acoustic.arousal_score >= _CONFLICT_AROUSAL_BELOW:
        return False
    expects_high = style.tone in _HIGH_ENERGY_TONES or style.emotion in _HIGH_ENERGY_EMOTIONS
    positive = style.tone in _POSITIVE_TONES or style.emotion in _POSITIVE_EMOTIONS
    return expects_high or positive


def measured_cues(
    style: StyleMetadata,
    acoustic: AcousticFeatures | None,
    words: Sequence | None = None,
    *,
    conflict: bool | None = None,
) -> list[Cue]:
    """
    Evidence computed deterministically from the audio and word timings
    (`source=measured`) -- unlike LLM-stated cues these can be checked.
    `words` are `WordTiming`-like objects (word/start_ms/end_ms/
    is_stressed); duck-typed to keep this module free of an STT import.
    """
    cues: list[Cue] = []
    if acoustic is not None:
        d = acoustic.energy_delta_db
        if abs(d) >= 5.0:
            cues.append(
                Cue(
                    kind=CueKind.ACOUSTIC,
                    evidence=f"voice {abs(d):.0f} dB {'louder' if d > 0 else 'quieter'} than this speaker's usual",
                    weight=min(0.9, 0.4 + abs(d) / 20.0),
                    source=CueSource.MEASURED,
                )
            )
        p = acoustic.pitch_delta_pct
        if p is not None and abs(p) >= 12.0:
            cues.append(
                Cue(
                    kind=CueKind.ACOUSTIC,
                    evidence=f"pitch {abs(p):.0f}% {'higher' if p > 0 else 'lower'} than usual",
                    weight=min(0.9, 0.4 + abs(p) / 100.0),
                    source=CueSource.MEASURED,
                )
            )
        if (
            acoustic.pitch_range_hz is not None
            and acoustic.pitch_range_hz < 20.0
            and acoustic.duration_ms >= 500
        ):
            cues.append(
                Cue(
                    kind=CueKind.ACOUSTIC,
                    evidence=f"flat pitch (only {acoustic.pitch_range_hz:.0f} Hz of movement)",
                    weight=0.5,
                    source=CueSource.MEASURED,
                )
            )
    if words:
        gaps = [b.start_ms - a.end_ms for a, b in zip(words, words[1:])]
        long_gaps = [g for g in gaps if g > 400]
        if long_gaps:
            cues.append(
                Cue(
                    kind=CueKind.PROSODIC,
                    evidence=f"{len(long_gaps)} pause(s) over 400 ms (longest {max(long_gaps)} ms)",
                    weight=min(0.9, 0.3 + 0.2 * len(long_gaps)),
                    source=CueSource.MEASURED,
                )
            )
        stressed = [w.word for w in words if getattr(w, "is_stressed", False)]
        if stressed:
            cues.append(
                Cue(
                    kind=CueKind.PROSODIC,
                    evidence=f"held/stressed: {', '.join(stressed[:3])}",
                    weight=0.4,
                    source=CueSource.MEASURED,
                )
            )
    if conflict is None and acoustic is not None:
        conflict = _modality_conflict(style, acoustic)
    if conflict:
        cues.append(
            Cue(
                kind=CueKind.INCONGRUENCE,
                evidence="delivery is quieter and flatter than the words suggest",
                weight=0.7,
                source=CueSource.MEASURED,
            )
        )
    return cues[:5]


def fuse_modalities(
    style: StyleMetadata,
    acoustic: AcousticFeatures | None,
    words: Sequence | None = None,
) -> StyleMetadata:
    """
    Combine the text-derived register with the measured voice.

    Sets `acoustic`, the fused `arousal`, `modality_conflict`, and
    replaces any earlier *measured* cues in the explanation (LLM cues
    are kept untouched). Idempotent, so it is safe to call once in the
    detector and again in the orchestrator with the final utterance's
    audio. Deliberately does NOT rewrite `emotion`/`tone`: the LLM has
    already seen the acoustic summary in its prompt, and overriding its
    label with a heuristic would be less trustworthy than the label.
    """
    if acoustic is None:
        return style
    conflict = _modality_conflict(style, acoustic)
    cues = measured_cues(style, acoustic, words, conflict=conflict)
    explanation = style.explanation
    if explanation is not None:
        kept = [c for c in explanation.cues if c.source != CueSource.MEASURED]
        explanation = explanation.model_copy(update={"cues": kept + cues})
    elif cues:
        explanation = StyleExplanation(summary="", cues=cues)
    return style.model_copy(
        update={
            "acoustic": acoustic,
            "arousal": arousal_level(acoustic.arousal_score),
            "modality_conflict": conflict,
            "explanation": explanation,
        }
    )


def describe_acoustic(acoustic: AcousticFeatures) -> str:
    """One-line summary for the register-detection prompt."""
    bits = [
        f"arousal {acoustic.arousal_score:.2f} "
        f"({arousal_level(acoustic.arousal_score).value})",
        f"loudness {acoustic.energy_delta_db:+.0f} dB vs speaker baseline",
    ]
    if acoustic.pitch_delta_pct is not None:
        bits.append(f"pitch {acoustic.pitch_delta_pct:+.0f}% vs baseline")
    if acoustic.pitch_range_hz is not None:
        bits.append(f"pitch movement {acoustic.pitch_range_hz:.0f} Hz")
    return "; ".join(bits)
