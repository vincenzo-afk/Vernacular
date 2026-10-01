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

### Multimodal and explanatory fields

These extend the eight register fields above. All are optional/defaulted, so code that builds a `StyleMetadata` without them still works. Unlike the eight register fields they are **not** rewritten by the translator — they describe the speaker's delivery, not the target-language text — and the translator prompt does not contain them (`StyleMetadata.register_dict()` is what the LLM stages see). The translator carries them through to its output unchanged.

### `acoustic`
**Type:** object or `null` — `arousal_score`, `energy_db`, `energy_delta_db`, `energy_variability_db`, `pitch_mean_hz`, `pitch_delta_pct`, `pitch_range_hz`, `voiced_ratio`, `duration_ms`
**Source:** measured from the raw mic audio by `pipeline/prosody.py` (loudness and pitch of voiced frames), reported relative to the speaker's own running baseline. `null` when too little speech was heard. **Never accepted from the LLM.**
**Consumed by:** register detection (as an `Acoustic:` line in the prompt), `arousal`/`modality_conflict` fusion, the explanation's measured cues, key-moment scoring, the UI.

### `arousal`
**Type:** enum — `low | medium | high`
**Source:** `acoustic.arousal_score` thresholded (below 0.35 / above 0.65). Defaults to `medium` when there is no audio measurement. A transparent heuristic (louder / higher / more varied than this speaker's norm), validated only on synthetic signals — not a trained emotion model.
**Consumed by:** key-moment scoring, the UI's "voice energy" tag.

### `modality_conflict`
**Type:** bool
**Source:** `pipeline/prosody.py::fuse_modalities` — true when the wording implies high energy or warmth (urgent/excited/annoyed, or positive emotion) but the voice is quiet and flat (arousal < 0.30 over at least 300 ms of speech). It is a signal, not a verdict: deadpan delivery is a classic sarcasm cue but can also just be a calm speaker.
**Consumed by:** key-moment scoring, the UI's "words ≠ voice" tag, and the explanation.

### `explanation`
**Type:** object or `null` — `summary` (one sentence), `cues[]` (`kind`, `evidence`, `weight`, `source`), `context_used`
**Source:** `cues` with `source: "llm"` are the register-detection model's own account and are **unverified**; cues with `source: "measured"` are computed deterministically from the audio and word timings. `kind` is one of `lexical | prosodic | acoustic | contextual | incongruence`. A `contextual` cue and `context_used: true` are discarded unless earlier turns were actually supplied to the model. The UI keeps the two sources visually distinct.
**Consumed by:** the "Why this tone?" panel only — nothing downstream depends on it.

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

## Example: context-aware, multimodal, explained reading

Source utterance: *"Oh, great."* said right after *"The build failed again."*, quietly and flatly.

```json
{
  "tone": "sarcastic",
  "pace": "slow",
  "formality": "casual",
  "emotion": "frustration",
  "sarcasm_score": 0.9,
  "emphasis_words": ["great"],
  "pause_pattern": "dramatic",
  "confidence": 0.86,
  "arousal": "low",
  "acoustic": {
    "arousal_score": 0.22,
    "energy_db": -31.0,
    "energy_delta_db": -6.0,
    "energy_variability_db": 2.1,
    "pitch_mean_hz": 142.0,
    "pitch_delta_pct": -4.0,
    "pitch_range_hz": 9.0,
    "voiced_ratio": 0.7,
    "duration_ms": 900
  },
  "modality_conflict": false,
  "explanation": {
    "summary": "Praise right after a failure, said quietly and flat: sarcasm.",
    "cues": [
      {"kind": "contextual", "evidence": "follows 'The build failed again.'", "weight": 0.9, "source": "llm"},
      {"kind": "acoustic", "evidence": "voice 6 dB quieter than this speaker's usual", "weight": 0.7, "source": "measured"}
    ],
    "context_used": true
  }
}
```

---

## Adding a new field

If a new register dimension needs to be tracked:

1. Add the typed field to the pydantic model in `backend/app/schemas/style_metadata.py` with a sensible default (so existing code that constructs `StyleMetadata` without it doesn't break).
2. Document it here, following the format above: type, source, what consumes it.
3. Update the register-detection prompt/few-shots (`pipeline/register_detector.py`) to actually populate it.
4. Update the TTS mapping (`pipeline/tts.py`) to consume it, or explicitly note if it's translation-only (like `formality` influencing word choice) and not meant to reach TTS directly.

Do not add a new field as a loose/untyped dict entry or a free-text note appended to an existing field — see `CLAUDE.md` and `AGENTS.md` constraint #2/#3 on why the schema needs to stay fully typed.
