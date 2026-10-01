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
