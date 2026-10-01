"""
Key-moment detection.

A "key moment" is an utterance worth a second look when reviewing a
conversation afterwards: a strongly sarcastic remark, an urgent
warning, an emotional spike, a sudden change of tone, or a case where
the voice contradicts the words.

Deliberately deterministic and LLM-free: it is computed from the
`StyleMetadata` the pipeline already produced, so it adds no latency
and no extra provider call, and every reason it gives is checkable.
It is a *derived* insight, not a new register dimension, so it lives
outside `StyleMetadata` (CLAUDE.md constraint #3 is about style
information flowing between pipeline stages).
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from app.schemas.style_metadata import Arousal, Emotion, StyleMetadata, Tone

KEY_MOMENT_THRESHOLD = 0.6

_STRONG_EMOTIONS = {
    Emotion.FRUSTRATION,
    Emotion.ANNOYANCE,
    Emotion.EXCITEMENT,
    Emotion.CONCERN,
    Emotion.AFFECTION,
}


class KeyMomentReason(str, Enum):
    HIGH_SARCASM = "high_sarcasm"
    URGENT = "urgent"
    STRONG_EMOTION = "strong_emotion"
    HIGH_AROUSAL = "high_arousal"
    EMOTION_SHIFT = "emotion_shift"
    MODALITY_CONFLICT = "modality_conflict"
    EMPHASIS = "emphasis"


class KeyMoment(BaseModel):
    is_key: bool = False
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    reasons: list[KeyMomentReason] = Field(default_factory=list)


def detect_key_moment(
    style: StyleMetadata,
    previous: StyleMetadata | None = None,
    threshold: float = KEY_MOMENT_THRESHOLD,
) -> KeyMoment:
    """
    Scores one utterance. Contributions are additive and capped at 1.0;
    a single strong signal (clear sarcasm, urgency) is enough on its own,
    weaker ones (mild emphasis, arousal) only matter in combination.
    The total is discounted by the detector's own confidence.
    """
    score = 0.0
    reasons: list[KeyMomentReason] = []

    def add(amount: float, reason: KeyMomentReason) -> None:
        nonlocal score
        score += amount
        reasons.append(reason)

    if style.sarcasm_score >= 0.6:
        add(0.8 * style.sarcasm_score, KeyMomentReason.HIGH_SARCASM)
    if style.tone == Tone.URGENT:
        add(0.6, KeyMomentReason.URGENT)
    if style.emotion in _STRONG_EMOTIONS and style.tone != Tone.NEUTRAL:
        add(0.3, KeyMomentReason.STRONG_EMOTION)
    if style.acoustic is not None and style.arousal == Arousal.HIGH:
        add(0.2, KeyMomentReason.HIGH_AROUSAL)
    if previous is not None and _is_shift(previous, style):
        add(0.25, KeyMomentReason.EMOTION_SHIFT)
    if style.modality_conflict:
        add(0.3, KeyMomentReason.MODALITY_CONFLICT)
    if len(style.emphasis_words) >= 2:
        add(0.1, KeyMomentReason.EMPHASIS)

    score = min(1.0, score) * (0.5 + 0.5 * style.confidence)
    return KeyMoment(is_key=score >= threshold, score=round(score, 3), reasons=reasons)


def _is_shift(previous: StyleMetadata, current: StyleMetadata) -> bool:
    if previous.tone == current.tone:
        return False
    # A shift into or out of neutral is ordinary turn-taking, not drama.
    return previous.tone != Tone.NEUTRAL and current.tone != Tone.NEUTRAL
