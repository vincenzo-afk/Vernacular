# Style Metadata Schema Reference

This is the full reference for `StyleMetadata`, the structured contract that carries register information between Vernacular's pipeline stages. See `ARCHITECTURE.md` §2 for how it fits into the pipeline, and `backend/app/schemas/style_metadata.py` for the pydantic implementation this document should stay in sync with.

---

## Purpose

`StyleMetadata` is what makes register a first-class, testable, bindable signal instead of a vague instruction buried in a prompt. It is:

- **Produced** by the register detector (Stage 2) from the source transcript + prosody data
- **Refined** by the translator (Stage 3), which may adjust fields to reflect how the target language expresses the same register (e.g., formality expressed via honorific suffixes rather than word choice)
- **Consumed** by the TTS stage (Stage 4), which maps fields directly onto ElevenLabs' style/stability/similarity/speed controls

---

## Fields

### `tone`
**Type:** enum — `neutral | sarcastic | urgent | warm | formal | hesitant | excited | annoyed`
**Source:** LLM classification from transcript content + sentiment signals
**Consumed by:** TTS style parameter selection; also informs word choice during translation (Stage 3 may pick different target-language phrasing for `sarcastic` vs `warm` tone carrying similar literal meaning)

### `pace`
**Type:** enum — `slow | normal | fast | rushed`
**Source:** Derived from AssemblyAI word-level timestamps (words per second in the utterance, pause durations)
**Consumed by:** TTS speed control

### `formality`
**Type:** enum — `casual | neutral | formal`
**Source:** LLM classification from word choice, sentence structure, and address forms in the source transcript
**Consumed by:** Translation stage (drives register-appropriate word choice / honorifics in target language); TTS voice style selection

### `emotion`
**Type:** string, constrained to a fixed set mirroring AssemblyAI's sentiment categories plus project-specific extensions: `neutral | positive | negative | annoyance | excitement | concern | affection | frustration`
**Source:** AssemblyAI Sentiment Analysis, refined by the register detector LLM using transcript context
**Consumed by:** TTS emotional style parameter

### `sarcasm_score`
**Type:** float, `0.0`–`1.0`
**Source:** LLM classification — sarcasm is treated as graded rather than binary, since confidence varies significantly with context and text-only signal is inherently ambiguous
**Consumed by:** Translation stage (high sarcasm_score may trigger idiom substitution rather than literal translation, since sarcasm often doesn't survive literal translation); TTS style intensity

### `emphasis_words`
**Type:** array of strings (subset of words from the source transcript)
**Source:** AssemblyAI word-level timestamp/stress data — words with notable duration or pitch deviation relative to the utterance's baseline
**Consumed by:** Translation stage should track which *concept* carried emphasis so Stage 4 can apply emphasis to the corresponding target-language word(s), which may not be a direct translation of the same source word (word order and emphasis placement differ across languages)

### `pause_pattern`
**Type:** enum — `natural | clipped | halting | dramatic`
**Source:** Derived from pause durations and positions in AssemblyAI's word-level timing data
**Consumed by:** TTS pacing/pause insertion

### `confidence`
**Type:** float, `0.0`–`1.0`
**Source:** The register detector's self-reported confidence in the overall reading
**Consumed by:** Orchestrator — below a configured threshold, the pipeline falls back to a neutral-register default rather than acting on a low-confidence read (see `ARCHITECTURE.md` §5)

---

## Example: sarcastic remark

Source utterance (English): *"Oh, great, another meeting."* (said flatly, with a long pause before "another")

```json
{
  "tone": "sarcastic",
  "pace": "slow",
  "formality": "casual",
  "emotion": "frustration",
  "sarcasm_score": 0.88,
  "emphasis_words": ["great"],
  "pause_pattern": "dramatic",
  "confidence": 0.81
}
```

## Example: urgent warning

Source utterance (English): *"Watch out, the car's not stopping!"* (fast, clipped)

```json
{
  "tone": "urgent",
  "pace": "rushed",
  "formality": "casual",
  "emotion": "concern",
  "sarcasm_score": 0.02,
  "emphasis_words": ["watch", "not"],
  "pause_pattern": "clipped",
  "confidence": 0.93
}
```

## Example: formal, low-confidence read

Source utterance, ambiguous/short: *"Understood."*

```json
{
  "tone": "neutral",
  "pace": "normal",
  "formality": "formal",
  "emotion": "neutral",
  "sarcasm_score": 0.1,
  "emphasis_words": [],
  "pause_pattern": "natural",
  "confidence": 0.35
}
```

This last example is short enough that the detector can't be confident — `confidence: 0.35` would fall below the fallback threshold in most configurations, so the orchestrator would treat this as neutral register regardless of the specific field values above.

---

## Adding a new field

If a new register dimension needs to be tracked:

1. Add the typed field to the pydantic model in `backend/app/schemas/style_metadata.py` with a sensible default (so existing code that constructs `StyleMetadata` without it doesn't break).
2. Document it here, following the format above: type, source, what consumes it.
3. Update the register-detection prompt/few-shots (`pipeline/register_detector.py`) to actually populate it.
4. Update the TTS mapping (`pipeline/tts.py`) to consume it, or explicitly note if it's translation-only (like `formality` influencing word choice) and not meant to reach TTS directly.

Do not add a new field as a loose/untyped dict entry or a free-text note appended to an existing field — see `CLAUDE.md` and `AGENTS.md` constraint #2/#3 on why the schema needs to stay fully typed.
