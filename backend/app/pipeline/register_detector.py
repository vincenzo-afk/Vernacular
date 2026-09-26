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
"""

from app.pipeline.stt import TranscriptEvent
from app.schemas.style_metadata import StyleMetadata


class RegisterDetector:
    CONFIDENCE_THRESHOLD = 0.5

    # TODO: system prompt + few-shot examples go here. This is the core
    # IP of the register-detection stage — see docs/style-metadata-schema.md
    # for target output format and worked examples (sarcastic, urgent,
    # low-confidence cases).
    SYSTEM_PROMPT = """\
You are a register-detection system. Given a transcript segment and
prosody signals (pace, pauses, sentiment), classify the speaker's tone,
pace, formality, emotion, sarcasm level, emphasized words, and pause
pattern. Output must conform to the StyleMetadata schema exactly.
"""

    def __init__(self, llm_client) -> None:
        self._llm_client = llm_client

    async def detect(self, transcript: TranscriptEvent) -> StyleMetadata:
        """
        Returns StyleMetadata for the given transcript event. Falls
        back to StyleMetadata.neutral_fallback() on any failure or
        low-confidence result — never raises to the orchestrator for
        a detection failure, since losing register fidelity gracefully
        is far better than blocking translation (ARCHITECTURE.md §5).
        """
        try:
            result = await self._call_llm(transcript)
        except Exception:
            return StyleMetadata.neutral_fallback()

        if result.confidence < self.CONFIDENCE_THRESHOLD:
            return StyleMetadata.neutral_fallback()

        return result

    async def _call_llm(self, transcript: TranscriptEvent) -> StyleMetadata:
        raise NotImplementedError("Register detection LLM call pending")
