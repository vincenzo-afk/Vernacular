"""
Stage 2: Register detection.

Takes a transcript (partial or final) plus prosody signals (word
timings, sentiment) and produces StyleMetadata. This is one of the two
genuinely custom components of this project (see README.md) — the
prompt/few-shot design here is what makes sarcasm/formality/urgency
detection work, not a generic LLM call.

Design notes:
- Runs on partial transcripts as they stream in (see ARCHITECTURE.md
  §3) so the final-transcript pass is a cheap confirmation, not a cold
  start.
- Must return StyleMetadata.neutral_fallback() rather than raising or
  blocking if the LLM call fails or times out, or if it returns
  confidence below RegisterDetector.CONFIDENCE_THRESHOLD.

See backend/app/pipeline/AGENTS.md for the exact failure-handling
contract this module must satisfy.
"""

import asyncio
import json
import logging

from app.pipeline.llm_response_utils import strip_markdown_fence
from app.pipeline.prosody import describe_acoustic, fuse_modalities
from app.pipeline.stt import TranscriptEvent
from app.schemas.style_metadata import (
    AcousticFeatures,
    Cue,
    CueKind,
    CueSource,
    StyleExplanation,
    StyleMetadata,
)

logger = logging.getLogger("vernacular.register_detector")

# Any object implementing:
#     async def complete(self, *, system: str, user: str, **kwargs) -> str
# satisfies this. Kept untyped deliberately — see CONTRIBUTING.md
# "Known gaps" on why the LLM provider isn't locked to one SDK type yet.
LLMClient = object


_OUTPUT_SPEC = """\
Output ONLY a single JSON object with exactly these fields, no other text:

{
  "tone": one of ["neutral","sarcastic","urgent","warm","formal","hesitant","excited","annoyed"],
  "pace": one of ["slow","normal","fast","rushed"],
  "formality": one of ["casual","neutral","formal"],
  "emotion": one of ["neutral","positive","negative","annoyance","excitement","concern","affection","frustration"],
  "sarcasm_score": float between 0.0 and 1.0,
  "emphasis_words": array of strings, words from the transcript that carried \
stress or emphasis,
  "pause_pattern": one of ["natural","clipped","halting","dramatic"],
  "confidence": float between 0.0 and 1.0, your own confidence in this reading. \
Use low confidence (below 0.5) for very short utterances, ambiguous text with \
no strong prosodic signal, or fragments that could plausibly be several \
different registers.
}

Sarcasm is graded, not binary — reserve high sarcasm_score for cases where \
the literal meaning and the evident intent clearly diverge (e.g. flat praise \
paired with a negative sentiment signal or a dramatic pause pattern).

A short acknowledgment like "okay" or "understood" with no strong prosody \
signal should get LOW confidence and neutral/default values across the board \
— do not invent a confident reading from insufficient evidence.
"""

# Few-shot examples included in every prompt. See
# docs/style-metadata-schema.md for these same worked examples with
# explanation of *why* each field takes the value it does.
FEW_SHOTS = [
    {
        "transcript": "Oh, great, another meeting.",
        "prosody": "long pause before 'another'; flat/negative sentiment; word 'great' held longer than surrounding words",
        "output": {
            "tone": "sarcastic",
            "pace": "slow",
            "formality": "casual",
            "emotion": "frustration",
            "sarcasm_score": 0.88,
            "emphasis_words": ["great"],
            "pause_pattern": "dramatic",
            "confidence": 0.81,
        },
    },
    {
        "transcript": "Watch out, the car's not stopping!",
        "prosody": "fast delivery, minimal pausing, high-intensity positive-arousal sentiment",
        "output": {
            "tone": "urgent",
            "pace": "rushed",
            "formality": "casual",
            "emotion": "concern",
            "sarcasm_score": 0.02,
            "emphasis_words": ["watch", "not"],
            "pause_pattern": "clipped",
            "confidence": 0.93,
        },
    },
    {
        "transcript": "Understood.",
        "prosody": "single word, neutral sentiment, no notable stress",
        "output": {
            "tone": "neutral",
            "pace": "normal",
            "formality": "formal",
            "emotion": "neutral",
            "sarcasm_score": 0.1,
            "emphasis_words": [],
            "pause_pattern": "natural",
            "confidence": 0.35,
        },
    },
]


def _build_user_prompt(transcript: TranscriptEvent) -> str:
    """
    Assembles the transcript + prosody description + few-shots into the
    user-turn prompt. Kept as a standalone function (not a method) so
    it's independently testable and reusable if register detection is
    ever batched.
    """
    prosody_bits = []
    if transcript.sentiment:
        prosody_bits.append(f"sentiment: {transcript.sentiment}")
    stressed = [w.word for w in transcript.words if w.is_stressed]
    if stressed:
        prosody_bits.append(f"stressed words: {', '.join(stressed)}")
    if transcript.words:
        total_ms = transcript.words[-1].end_ms - transcript.words[0].start_ms
        if total_ms > 0:
            wps = len(transcript.words) / (total_ms / 1000)
            prosody_bits.append(f"speech rate: {wps:.1f} words/sec")
        gaps = [
            b.start_ms - a.end_ms
            for a, b in zip(transcript.words, transcript.words[1:])
        ]
        long_gaps = [g for g in gaps if g > 400]
        if long_gaps:
            prosody_bits.append(f"{len(long_gaps)} notable pause(s) >400ms")

    prosody_description = "; ".join(prosody_bits) or "no notable prosody signal"

    examples_text = "\n\n".join(
        f"Transcript: {ex['transcript']}\n"
        f"Prosody: {ex['prosody']}\n"
        f"Output: {json.dumps(ex['output'])}"
        for ex in FEW_SHOTS
    )

    return (
        f"Examples:\n\n{examples_text}\n\n"
        f"---\n\n"
        f"Now classify this segment:\n\n"
        f"Transcript: {transcript.text}\n"
        f"Prosody: {prosody_description}\n"
        f"Output:"
    )


class RegisterDetector:
    CONFIDENCE_THRESHOLD = 0.5

    # Hard upper bound on one detection. ARCHITECTURE.md §3 budgets
    # ~400ms for this stage and §5 promises a timeout fallback; without
    # a bound, one slow LLM call stalls the translator (which waits on
    # this result) and freezes the live session. Deliberately generous
    # relative to the budget: a *late* reading is still better than a
    # neutral one when we can afford it, but an unbounded wait never
    # is. Tune against real latency numbers (tests/live/).
    TIMEOUT_S = 1.5

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

    async def detect(self, transcript: TranscriptEvent) -> StyleMetadata:
        """
        Returns StyleMetadata for the given transcript event. Falls
        back to StyleMetadata.neutral_fallback() on any failure or
        low-confidence result — never raises to the orchestrator for
        a detection failure, since losing register fidelity gracefully
        is far better than blocking translation (ARCHITECTURE.md §5).
        """
        if not transcript.text.strip():
            return StyleMetadata.neutral_fallback()

        try:
            result = await asyncio.wait_for(
                self._call_llm(transcript), timeout=self.TIMEOUT_S
            )
        except asyncio.TimeoutError:
            logger.warning(
                "register_detector: no answer within %.1fs, falling back "
                "to neutral rather than stalling the segment",
                self.TIMEOUT_S,
            )
            return StyleMetadata.neutral_fallback()
        except Exception:
            logger.warning(
                "register_detector: LLM call failed, falling back to neutral",
                exc_info=True,
            )
            return StyleMetadata.neutral_fallback()

        if result.confidence < self.CONFIDENCE_THRESHOLD:
            logger.info(
                "register_detector: confidence %.2f below threshold %.2f, "
                "falling back to neutral",
                result.confidence,
                self.CONFIDENCE_THRESHOLD,
            )
            return StyleMetadata.neutral_fallback()

        return result

    async def _call_llm(self, transcript: TranscriptEvent) -> StyleMetadata:
        user_prompt = _build_user_prompt(transcript)
        raw = await self._llm_client.complete(
            system=SYSTEM_PROMPT,
            user=user_prompt,
            temperature=0.2,
            max_tokens=300,
        )
        return self._parse_response(raw)

    @staticmethod
    def _parse_response(raw: str) -> StyleMetadata:
        """
        Parses the LLM's JSON output into StyleMetadata. Strips common
        wrapping (markdown code fences) defensively, since not every
        model reliably honors "output only JSON" instructions.
        """
        text = strip_markdown_fence(raw)
        data = json.loads(text)
        return StyleMetadata(**data)
