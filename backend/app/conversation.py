"""
Conversation memory: what has been said so far, kept in a bounded
rolling window and rendered into compact context for the LLM stages.

Two consumers on the hot path:

- Register detection uses earlier turns for **context-aware sarcasm
  detection**: "Oh, great." after "The server crashed again." is
  sarcastic; the same words after "You got the promotion!" are not.
- Translation uses earlier turns for pronoun, terminology and
  formality consistency across sentences.

And one off the hot path: the summarizer reads the whole memory.

Two-phase turns: a turn is *begun* the moment its final transcript
arrives (so the very next utterance's detection can already see the
words), and *completed* once translation finishes (adding the
translation and the detected style). Context rendering copes with
both states.

Everything rendered into a prompt is capped in turns AND characters --
context is latency (input tokens) and the speaker's earlier words are
untrusted text, so the prompt fences it as data, not instructions.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field

from app.insights import KeyMoment
from app.schemas.style_metadata import StyleMetadata, Tone

MAX_STORED_TURNS = 500
DETECTION_CONTEXT_TURNS = 4
TRANSLATION_CONTEXT_TURNS = 3
CONTEXT_MAX_CHARS = 600
_TURN_TEXT_CHARS = 160


@dataclass
class Turn:
    segment_id: int
    source_text: str
    translated_text: str | None = None
    style: StyleMetadata | None = None  # the SOURCE-language reading
    key_moment: KeyMoment | None = None
    speech_start_ms: int | None = None
    speech_end_ms: int | None = None
    created_at: float = field(default_factory=time.time)

    @property
    def complete(self) -> bool:
        return self.translated_text is not None


def _clip(text: str, limit: int = _TURN_TEXT_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


class ConversationMemory:
    def __init__(self, max_turns: int = MAX_STORED_TURNS) -> None:
        self._max_turns = max_turns
        self._turns: list[Turn] = []

    # ---------------------------------------------------------------- writing

    def begin_turn(self, segment_id: int, source_text: str) -> Turn:
        turn = Turn(segment_id=segment_id, source_text=source_text)
        self._turns.append(turn)
        if len(self._turns) > self._max_turns:
            del self._turns[: len(self._turns) - self._max_turns]
        return turn

    def complete_turn(
        self,
        segment_id: int,
        translated_text: str,
        style: StyleMetadata,
        key_moment: KeyMoment | None = None,
        speech_start_ms: int | None = None,
        speech_end_ms: int | None = None,
    ) -> None:
        turn = self.get(segment_id)
        if turn is None:  # evicted by the size cap, or never begun
            turn = self.begin_turn(segment_id, "")
        turn.translated_text = translated_text
        turn.style = style
        turn.key_moment = key_moment
        turn.speech_start_ms = speech_start_ms
        turn.speech_end_ms = speech_end_ms

    def reset(self) -> None:
        self._turns.clear()

    # ---------------------------------------------------------------- reading

    @property
    def turns(self) -> list[Turn]:
        return list(self._turns)

    def __len__(self) -> int:
        return len(self._turns)

    def get(self, segment_id: int) -> Turn | None:
        for turn in reversed(self._turns):
            if turn.segment_id == segment_id:
                return turn
        return None

    def previous_style(self, before_segment_id: int) -> StyleMetadata | None:
        """Source style of the most recent completed turn before this one."""
        for turn in reversed(self._turns):
            if turn.segment_id < before_segment_id and turn.style is not None:
                return turn.style
        return None

    def _recent(self, before: int | None, limit: int) -> list[Turn]:
        # `before` matters because the orchestrator's STT pump can run
        # AHEAD of the worker: utterance 2 may already be in memory
        # while utterance 1 is still being classified, and 1 must never
        # be given 2 as "earlier" context.
        candidates = [
            t
            for t in self._turns
            if (before is None or t.segment_id < before) and t.source_text.strip()
        ]
        return candidates[-limit:]

    def detection_context(
        self,
        before: int | None = None,
        max_turns: int = DETECTION_CONTEXT_TURNS,
        max_chars: int = CONTEXT_MAX_CHARS,
    ) -> tuple[str, int]:
        """
        (rendered context, number of turns it covers) for the register
        detector: only turns with segment_id < `before` (all turns when
        None). Empty string / 0 when there is nothing earlier.
        Includes each completed turn's detected tone so the model can
        see the emotional trajectory, not just the words.
        """
        lines: list[str] = []
        for turn in self._recent(before, max_turns):
            label = ""
            if turn.style is not None:
                label = f" ({turn.style.tone.value}/{turn.style.emotion.value})"
            lines.append(f'- "{_clip(turn.source_text)}"{label}')
        return _fit(lines, max_chars)

    def translation_context(
        self,
        before: int | None = None,
        max_turns: int = TRANSLATION_CONTEXT_TURNS,
        max_chars: int = CONTEXT_MAX_CHARS,
    ) -> tuple[str, int]:
        """
        (rendered context, turn count) for the translator: earlier
        source -> translation pairs, so pronouns, names, terminology and
        formality level stay consistent from one sentence to the next.
        """
        lines: list[str] = []
        for turn in self._recent(before, max_turns):
            if turn.translated_text is None:
                lines.append(f'- "{_clip(turn.source_text)}"')
            else:
                lines.append(
                    f'- "{_clip(turn.source_text)}" -> "{_clip(turn.translated_text)}"'
                )
        return _fit(lines, max_chars)

    def stats(self) -> dict:
        done = [t for t in self._turns if t.style is not None]
        tones = Counter(t.style.tone.value for t in done if t.style.tone != Tone.NEUTRAL)
        return {
            "turn_count": len(self._turns),
            "dominant_tone": tones.most_common(1)[0][0] if tones else Tone.NEUTRAL.value,
            "mean_sarcasm": (
                sum(t.style.sarcasm_score for t in done) / len(done) if done else 0.0
            ),
            "key_moment_count": sum(
                1 for t in self._turns if t.key_moment is not None and t.key_moment.is_key
            ),
        }


def _fit(lines: list[str], max_chars: int) -> tuple[str, int]:
    """Keep the NEWEST lines that fit in max_chars (recent context matters most)."""
    kept: list[str] = []
    total = 0
    for line in reversed(lines):
        if kept and total + len(line) + 1 > max_chars:
            break
        kept.append(line[:max_chars])
        total += len(line) + 1
    kept.reverse()
    return "\n".join(kept), len(kept)
