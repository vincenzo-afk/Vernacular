"""
StyleMetadata — the structured contract for register information flowing
through the Vernacular pipeline.

See ARCHITECTURE.md §2 and docs/style-metadata-schema.md for the full
field reference and design rationale. Do not replace this with free-text
style descriptions — every pipeline stage binds to these typed fields.
"""

from enum import Enum
from typing import ClassVar

from pydantic import BaseModel, Field


class Tone(str, Enum):
    NEUTRAL = "neutral"
    SARCASTIC = "sarcastic"
    URGENT = "urgent"
    WARM = "warm"
    FORMAL = "formal"
    HESITANT = "hesitant"
    EXCITED = "excited"
    ANNOYED = "annoyed"


class Pace(str, Enum):
    SLOW = "slow"
    NORMAL = "normal"
    FAST = "fast"
    RUSHED = "rushed"


class Formality(str, Enum):
    CASUAL = "casual"
    NEUTRAL = "neutral"
    FORMAL = "formal"


class Emotion(str, Enum):
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    NEGATIVE = "negative"
    ANNOYANCE = "annoyance"
    EXCITEMENT = "excitement"
    CONCERN = "concern"
    AFFECTION = "affection"
    FRUSTRATION = "frustration"


class PausePattern(str, Enum):
    NATURAL = "natural"
    CLIPPED = "clipped"
    HALTING = "halting"
    DRAMATIC = "dramatic"


class Arousal(str, Enum):
    """Fused (text + voice) activation level. See `fuse_modalities` in
    pipeline/prosody.py for how it is derived."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class CueKind(str, Enum):
    """What kind of evidence a Cue in a StyleExplanation points at."""

    LEXICAL = "lexical"  # the words themselves
    PROSODIC = "prosodic"  # pauses, pacing, held words (from word timings)
    ACOUSTIC = "acoustic"  # loudness / pitch measured from the audio
    CONTEXTUAL = "contextual"  # what was said earlier in the conversation
    INCONGRUENCE = "incongruence"  # words and delivery disagree


class CueSource(str, Enum):
    LLM = "llm"  # stated by the register-detection model (unverified)
    MEASURED = "measured"  # computed deterministically from signals


class AcousticFeatures(BaseModel):
    """
    Voice measurements for one utterance, computed from the raw mic
    audio (pipeline/prosody.py) -- the audio half of multimodal
    emotion detection. Measured, not guessed: it survives even when the
    LLM register reading falls back to neutral.

    Loudness and pitch are reported relative to the speaker's own
    running baseline (`energy_delta_db`, `pitch_delta_pct`), because
    absolute loudness depends on the microphone rather than the
    speaker.
    """

    arousal_score: float = Field(default=0.5, ge=0.0, le=1.0)
    energy_db: float = Field(default=-60.0, le=0.0)  # mean dBFS, voiced frames
    energy_delta_db: float = 0.0  # vs. this speaker's baseline
    energy_variability_db: float = Field(default=0.0, ge=0.0)
    pitch_mean_hz: float | None = Field(default=None, ge=0.0)
    pitch_delta_pct: float | None = None  # vs. this speaker's baseline
    pitch_range_hz: float | None = Field(default=None, ge=0.0)
    voiced_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    duration_ms: int = Field(default=0, ge=0)


class Cue(BaseModel):
    """One piece of evidence behind a tone/sarcasm reading."""

    kind: CueKind
    evidence: str
    weight: float = Field(default=0.5, ge=0.0, le=1.0)
    source: CueSource = CueSource.LLM


class StyleExplanation(BaseModel):
    """
    Why the detector read the register the way it did. LLM-stated cues
    (`source=llm`) are the model's own account and can be wrong;
    `source=measured` cues are computed from audio/word timings and can
    be checked. The UI must keep the two visually distinct.
    """

    summary: str = ""
    cues: list[Cue] = Field(default_factory=list)
    # True only when the reading actually leaned on earlier turns of
    # the conversation (and some were supplied to the model).
    context_used: bool = False


class StyleMetadata(BaseModel):
    """
    The register signal passed between pipeline stages.

    Produced by: pipeline/register_detector.py (Stage 2)
    Refined by: pipeline/translator.py (Stage 3) — may adjust fields to
        reflect target-language register expression (e.g. formality via
        honorifics rather than word choice).
    Consumed by: pipeline/tts.py (Stage 4) — maps fields onto TTS
        provider style/stability/speed controls.
    """

    tone: Tone = Tone.NEUTRAL
    pace: Pace = Pace.NORMAL
    formality: Formality = Formality.NEUTRAL
    emotion: Emotion = Emotion.NEUTRAL
    sarcasm_score: float = Field(default=0.0, ge=0.0, le=1.0)
    emphasis_words: list[str] = Field(default_factory=list)
    pause_pattern: PausePattern = PausePattern.NATURAL
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    # --- multimodal + explainability fields (all optional/defaulted so
    # existing code constructing StyleMetadata without them still works).
    # `arousal`, `acoustic` and `modality_conflict` are filled by
    # pipeline/prosody.py::fuse_modalities from the measured audio; the
    # translator carries them through unchanged (they describe the
    # speaker's delivery, not the target-language text).
    arousal: Arousal = Arousal.MEDIUM
    acoustic: AcousticFeatures | None = None
    modality_conflict: bool = False
    explanation: StyleExplanation | None = None

    REGISTER_FIELDS: ClassVar[tuple[str, ...]] = (
        "tone",
        "pace",
        "formality",
        "emotion",
        "sarcasm_score",
        "emphasis_words",
        "pause_pattern",
        "confidence",
    )

    def register_dict(self) -> dict:
        """
        Only the eight register fields the LLM stages reason about --
        excludes the measured/explanatory extras so they are not echoed
        into the translator prompt (wasted tokens on the latency path,
        and the model has no business rewriting measurements).
        """
        dumped = self.model_dump(mode="json")
        return {k: dumped[k] for k in self.REGISTER_FIELDS}

    @classmethod
    def neutral_fallback(cls) -> "StyleMetadata":
        """
        The fallback used when register detection fails or returns
        confidence below the configured threshold. See ARCHITECTURE.md
        §5 — never block the pipeline waiting for a confident read.
        """
        return cls(
            tone=Tone.NEUTRAL,
            pace=Pace.NORMAL,
            formality=Formality.NEUTRAL,
            emotion=Emotion.NEUTRAL,
            sarcasm_score=0.0,
            emphasis_words=[],
            pause_pattern=PausePattern.NATURAL,
            confidence=1.0,
        )
