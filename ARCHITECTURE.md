# Vernacular — Architecture

This document describes how Vernacular actually works under the hood: the data flow, module boundaries, the style metadata contract, and the latency budget. Read this before modifying `backend/app/pipeline/`.

---

## 1. System overview

Vernacular is a streaming pipeline. Audio comes in continuously from a microphone; translated, style-matched audio goes out continuously to a speaker. There is no "record then process" step — every stage operates on partial data as it arrives and refines its output as more context becomes available.

```
┌────────────┐   audio chunks (WS/LiveKit)   ┌──────────────────┐
│  Frontend  │ ─────────────────────────────▶ │  FastAPI backend │
│ (Next.js)  │ ◀───────────────────────────── │  (orchestrator)  │
└────────────┘   translated audio + tags      └──────────────────┘
                                                        │
                        ┌───────────────────────────────┼───────────────────────────────┐
                        ▼                                ▼                                ▼
              ┌───────────────────┐          ┌────────────────────┐          ┌────────────────────┐
              │ 1. STT             │          │ 2. Register        │          │ 4. TTS              │
              │ AssemblyAI          │─────────▶│    Detector         │─────────▶│ ElevenLabs           │
              │ Realtime STT +      │  text +  │ (LLM)                │  style   │ Multilingual v2       │
              │ Sentiment Analysis  │  prosody │                      │  meta-   │ + Voice Cloning       │
              └───────────────────┘  signals  └────────────────────┘  data     └────────────────────┘
                        │                                │                                ▲
                        │ transcript                     │                                │
                        └───────────────┬────────────────┘                                │
                                        ▼                                                  │
                              ┌────────────────────┐                                       │
                              │ 3. Translator        │  translated text + preserved  │
                              │ (LLM, register-aware) │───────────────────────────────┘
                              └────────────────────┘  register
```

### The four pipeline stages

| # | Stage | Module | Input | Output |
|---|---|---|---|---|
| 1 | Speech-to-text | `pipeline/stt.py` | Raw audio stream | Partial/final transcript + word-level timestamps + sentiment scores |
| 2 | Register detection | `pipeline/register_detector.py` | Transcript + prosody signals (pace, pauses, sentiment) | `StyleMetadata` object |
| 3 | Translation + register mapping | `pipeline/translator.py` | Final transcript segment + `StyleMetadata` | Translated text, register-preserving, + updated `StyleMetadata` for target language idiom |
| 4 | Expressive TTS | `pipeline/tts.py` | Translated text + `StyleMetadata` | Synthesized audio stream |

Stages 1 and 2 run **concurrently** where possible: sentiment analysis on partial transcripts starts as soon as AssemblyAI emits them, rather than waiting for the final transcript of a full utterance. Stage 3 requires a finalized transcript segment (a completed phrase/sentence, determined by AssemblyAI's endpointing) plus the register metadata. Stage 4 begins streaming synthesis as soon as the first translated clause is available — it does not wait for the entire translated sentence.

---

## 2. The Style Metadata contract

This is the single most important interface in the system. It is the JSON structure that Stage 2 produces, Stage 3 refines, and Stage 4 consumes. Every module boundary in this pipeline is defined in terms of this schema — see `backend/app/schemas/style_metadata.py` for the pydantic model and `docs/style-metadata-schema.md` for the full field reference.

```json
{
  "tone": "sarcastic",
  "pace": "fast",
  "formality": "informal",
  "emotion": "annoyance",
  "sarcasm_score": 0.82,
  "emphasis_words": ["totally", "fine"],
  "pause_pattern": "clipped",
  "confidence": 0.76
}
```

- **`tone`**: enum — `neutral | sarcastic | urgent | warm | formal | hesitant | excited | annoyed`
- **`pace`**: enum — `slow | normal | fast | rushed`
- **`formality`**: enum — `casual | neutral | formal`
- **`emotion`**: free-text label from a constrained set (mirrors AssemblyAI sentiment categories plus our extensions)
- **`sarcasm_score`**: float 0.0–1.0, since sarcasm is graded, not binary
- **`emphasis_words`**: which source words carried prosodic stress (from AssemblyAI word-level data), used to guide which target-language words should carry stress in TTS
- **`pause_pattern`**: enum — `natural | clipped | halting | dramatic`
- **`confidence`**: the register detector's own confidence in this reading — low confidence triggers the neutral-register fallback (see §5)

**Why a structured schema instead of a free-text style description passed to the TTS prompt:** structured fields are testable, cacheable, and bindable directly to ElevenLabs' style/stability/similarity controls without another LLM call to interpret free text. It also makes Stage 3's job well-defined: translate the words *and* decide how each metadata field should be re-expressed in the target language (e.g., a formality marker that's a suffix in Japanese vs. word choice in English).

---

## 3. Latency budget

Target: **~2 seconds end-to-end**, from the speaker finishing a phrase to translated audio starting to play.

| Stage | Budget | Notes |
|---|---|---|
| STT finalization (endpointing) | ~300ms | AssemblyAI's endpointing latency for a completed utterance |
| Register detection | ~400ms | Runs partially in parallel with STT finalization using partial transcripts; only needs to confirm/refine once final transcript lands |
| Translation + register mapping | ~600ms | Single LLM call, streamed; first-token latency matters more than total completion time since TTS can start on partial output |
| TTS synthesis (first audio chunk) | ~500ms | ElevenLabs streaming synthesis; time-to-first-byte is the relevant metric, not full-clip generation |
| Network + orchestration overhead | ~200ms | WebSocket round trips, buffering |

**Key latency techniques used in this codebase:**

1. **Partial-transcript pipelining.** Register detection begins on AssemblyAI's partial transcripts, not just finals. By the time the final transcript arrives, register detection often only needs a cheap confirmation pass rather than a cold start.
2. **Streaming at every boundary.** The translator streams tokens; TTS begins synthesizing as soon as a translated clause boundary is available rather than waiting for the full sentence. STT, translation, and TTS are all consumed as streams end-to-end — never buffer-then-process a whole stage if it can be avoided.
3. **Speculative execution with cheap rollback.** Register detection runs on the best-available partial transcript. If the final transcript changes meaning materially (rare, but happens with corrections/false starts), the translation step re-runs. This is a deliberate latency/correctness tradeoff: optimize for the common case.
4. **Voice cloning is precomputed, not per-utterance.** The speaker's voice clone is generated once per session (from a short calibration clip at session start), not regenerated per phrase.

See `backend/app/pipeline/orchestrator.py` for how these stages are actually wired together with `asyncio` tasks.

---

## 4. Module boundaries and provider abstraction

No provider-specific SDK calls exist outside `pipeline/*.py`. Each pipeline module exposes a plain async interface; swapping AssemblyAI for a different STT provider, or ElevenLabs for Cartesia, means implementing that interface in a new file — nothing else in the codebase changes.

```python
# pipeline/stt.py
class STTProvider(Protocol):
    async def stream(self, audio_chunks: AsyncIterator[bytes]) -> AsyncIterator[TranscriptEvent]: ...

# pipeline/tts.py
class TTSProvider(Protocol):
    async def synthesize(self, text: str, style: StyleMetadata, voice_id: str) -> AsyncIterator[bytes]: ...
```

This matters for two reasons specific to this project: (1) the tech stack table lists explicit fallbacks (Cartesia/OpenAI TTS if ElevenLabs rate-limits during a demo), and (2) the register-detection LLM and translation LLM may reasonably be different models chosen for different strengths (e.g., a fast/cheap model for register detection, a stronger model for translation quality) — the interface should not assume they're the same provider or even the same call.

---

## 5. Failure modes and fallbacks

| Failure | Fallback behavior |
|---|---|
| Register detection times out or returns low `confidence` | Fall back to neutral-register `StyleMetadata` (tone: neutral, pace: normal, formality: neutral) rather than blocking the pipeline |
| Translation LLM call fails | Retry once with a shorter timeout; if it fails again, pass through a literal (register-naive) translation rather than dropping the utterance entirely |
| ElevenLabs rate-limited or errors | Fall back to Cartesia or OpenAI TTS with best-effort style mapping (these providers have less granular style control — document this degradation to the user in the UI, don't silently pretend it's full-fidelity) |
| AssemblyAI connection drops mid-session | Reconnect with exponential backoff; buffer up to N seconds of audio client-side to avoid losing an in-flight utterance |
| Voice cloning unavailable/not consented | Fall back to a stock expressive voice in the target language rather than cloning |

The general principle: **never let a non-critical stage block the pipeline**. Losing register fidelity gracefully (falling back to neutral) is far better than losing translation entirely.

---

## 6. What's built vs. planned

This is a living section — update it as the implementation progresses. As of this document's creation, the repo scaffold exists but pipeline stages are stubs pending hackathon implementation. Track actual status in `backend/app/pipeline/` docstrings and update this table alongside code changes.

| Component | Status |
|---|---|
| Repo scaffold, docs | Done |
| STT integration (AssemblyAI) | Planned |
| Register detector (LLM + prompt) | Planned |
| Translator (LLM, register-aware) | Planned |
| TTS integration (ElevenLabs) | Planned |
| Voice cloning flow | Planned |
| Frontend (waveform, live transcript, tone tags) | Planned |
| Latency instrumentation/dashboard | Planned |
