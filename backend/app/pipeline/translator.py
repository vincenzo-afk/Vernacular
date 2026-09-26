"""
Stage 3: Translation + register mapping.

Deliberately a single LLM call that does translation AND register
preservation together — not two sequential steps. A literal translation
of a sarcastic or formal sentence frequently isn't sarcastic or formal
once translated literally (idiom, sentence structure, and honorific/
formality markers differ across languages), so the rewrite needs full
context of both the source meaning and the target StyleMetadata.

See ARCHITECTURE.md §2 and CLAUDE.md constraint #6.
"""

from dataclasses import dataclass

from app.schemas.style_metadata import StyleMetadata


@dataclass
class TranslationResult:
    translated_text: str
    style: StyleMetadata  # may differ from input style — e.g. formality
    # re-expressed via target-language honorifics rather than tone alone


class Translator:
    def __init__(self, llm_client) -> None:
        self._llm_client = llm_client

    async def translate(
        self,
        source_text: str,
        source_style: StyleMetadata,
        target_language: str,
    ) -> TranslationResult:
        """
        Streams translated text as it's generated (see
        ARCHITECTURE.md §3 — TTS should be able to start on the first
        translated clause, not wait for the full sentence). Returns the
        final TranslationResult once the full segment is translated.

        On failure: retry once with a shorter timeout. If that also
        fails, fall back to a literal (register-naive) translation
        rather than dropping the utterance — see ARCHITECTURE.md §5.
        """
        raise NotImplementedError("Translation LLM call pending")
