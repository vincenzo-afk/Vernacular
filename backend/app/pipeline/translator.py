"""
Stage 3: Translation + register mapping.

Deliberately a single LLM call that does translation AND register
preservation together — not two sequential steps. A literal translation
of a sarcastic or formal sentence frequently isn't sarcastic or formal
once translated literally (idiom, sentence structure, and honorific/
formality markers differ across languages), so the rewrite needs full
context of both the source meaning and the target StyleMetadata.

See ARCHITECTURE.md §2, CLAUDE.md constraint #6, and
backend/app/pipeline/AGENTS.md for the exact failure-handling contract.
"""

import asyncio
import json
import logging
from dataclasses import dataclass

from app.schemas.style_metadata import StyleMetadata

logger = logging.getLogger("vernacular.translator")

LLMClient = object  # see register_detector.py — kept provider-agnostic

FIRST_CALL_TIMEOUT_S = 3.0
RETRY_TIMEOUT_S = 1.5


@dataclass
class TranslationResult:
    translated_text: str
    style: StyleMetadata  # may differ from input style — e.g. formality
    # re-expressed via target-language honorifics rather than tone alone
    was_literal_fallback: bool = False


SYSTEM_PROMPT = """\
You are a translator that preserves speaking register, not just literal \
meaning. You will be given source text, the target language, and a \
structured description of the speaker's register (tone, pace, formality, \
emotion, sarcasm level, emphasized words, pause pattern).

Your job has two parts, done together in one pass:

1. Translate the text into the target language.
2. Rewrite the translation, if needed, so that a native speaker of the \
target language would perceive the SAME register the original speaker \
intended — not necessarily the same literal words. This matters most for:
   - Sarcasm: a literal translation of a sarcastic remark is often read as \
sincere in the target language. Prefer idiom or phrasing that a native \
speaker would actually recognize as sarcastic.
   - Formality: many languages mark formality grammatically (honorifics, \
verb conjugation, pronoun choice — e.g. Japanese keigo, French tu/vous, \
Korean speech levels) rather than through word choice alone. Apply the \
correct grammatical formality marker for the target language rather than \
just choosing "formal-sounding" vocabulary.
   - Urgency/emphasis: preserve which words carry stress, mapped to the \
corresponding concept in the target language — emphasis position may need \
to move since word order differs across languages.

Output ONLY a single JSON object, no other text:

{
  "translated_text": the translated (and register-adjusted) text,
  "style": {
    "tone": same enum as input,
    "pace": same enum as input,
    "formality": same enum as input, but re-evaluate for the target language \
if the original formality signal is expressed differently there,
    "emotion": same enum as input,
    "sarcasm_score": float 0.0-1.0, re-evaluate for how well the TRANSLATED \
text carries the sarcasm — may differ from the input if idiom substitution \
changed the delivery,
    "emphasis_words": array of strings, the TARGET LANGUAGE words that \
should carry stress (translate the concept, not necessarily the source word),
    "pause_pattern": same enum as input,
    "confidence": same as input confidence, passed through unchanged
  }
}
"""


def _build_user_prompt(
    source_text: str, source_style: StyleMetadata, target_language: str
) -> str:
    return (
        f"Target language: {target_language}\n\n"
        f"Source text: {source_text}\n\n"
        f"Source register: {source_style.model_dump_json()}\n\n"
        f"Output:"
    )


class Translator:
    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

    async def translate(
        self,
        source_text: str,
        source_style: StyleMetadata,
        target_language: str,
    ) -> TranslationResult:
        """
        On failure: retry once with a shorter timeout. If that also
        fails, fall back to a literal (register-naive) translation
        rather than dropping the utterance — see ARCHITECTURE.md §5
        and pipeline/AGENTS.md.
        """
        if not source_text.strip():
            return TranslationResult(translated_text="", style=source_style)

        try:
            return await asyncio.wait_for(
                self._call_llm(source_text, source_style, target_language),
                timeout=FIRST_CALL_TIMEOUT_S,
            )
        except Exception:
            logger.warning(
                "translator: first attempt failed, retrying with shorter timeout",
                exc_info=True,
            )

        try:
            return await asyncio.wait_for(
                self._call_llm(source_text, source_style, target_language),
                timeout=RETRY_TIMEOUT_S,
            )
        except Exception:
            logger.error(
                "translator: retry also failed, falling back to literal "
                "translation",
                exc_info=True,
            )
            return await self._literal_fallback(
                source_text, source_style, target_language
            )

    async def _call_llm(
        self, source_text: str, source_style: StyleMetadata, target_language: str
    ) -> TranslationResult:
        user_prompt = _build_user_prompt(source_text, source_style, target_language)
        raw = await self._llm_client.complete(
            system=SYSTEM_PROMPT,
            user=user_prompt,
            temperature=0.3,
            max_tokens=500,
        )
        return self._parse_response(raw)

    async def _literal_fallback(
        self, source_text: str, source_style: StyleMetadata, target_language: str
    ) -> TranslationResult:
        """
        Register-naive translation used only when the full
        register-aware call has failed twice. Still attempts an actual
        translation (better than passing through the source language
        untranslated), but does not try to preserve register — the
        output style is downgraded to neutral to avoid claiming a
        register fidelity we didn't achieve.
        """
        try:
            raw = await asyncio.wait_for(
                self._llm_client.complete(
                    system=(
                        "Translate the following text into "
                        f"{target_language}. Output ONLY the translated "
                        "text, nothing else."
                    ),
                    user=source_text,
                    temperature=0.0,
                    max_tokens=300,
                ),
                timeout=RETRY_TIMEOUT_S,
            )
            return TranslationResult(
                translated_text=raw.strip(),
                style=StyleMetadata.neutral_fallback(),
                was_literal_fallback=True,
            )
        except Exception:
            logger.error(
                "translator: literal fallback also failed — returning "
                "source text untranslated as last resort",
                exc_info=True,
            )
            return TranslationResult(
                translated_text=source_text,
                style=StyleMetadata.neutral_fallback(),
                was_literal_fallback=True,
            )

    @staticmethod
    def _parse_response(raw: str) -> TranslationResult:
        text = raw.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
            text = text.strip()

        data = json.loads(text)
        return TranslationResult(
            translated_text=data["translated_text"],
            style=StyleMetadata(**data["style"]),
        )
