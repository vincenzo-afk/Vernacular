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
              │ v3 Realtime         │  text +  │ (LLM)                │  style   │ Multilingual v2       │
              │ Streaming            │  word    │                      │  meta-   │ + Voice Cloning       │
              └───────────────────┘  timings   └────────────────────┘  data     └────────────────────┘
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
| 1 | Speech-to-text | `pipeline/stt.py` | Raw audio stream | Partial/final transcript + word-level timestamps. AssemblyAI's v3 realtime streaming API does **not** provide sentiment analysis (that's a separate, async-only AssemblyAI feature) — `TranscriptEvent.sentiment` is always `None` from this provider; see `stt.py`'s module docstring. |
| 2 | Register detection | `pipeline/register_detector.py` | Transcript + prosody signals derived from word timings (pace, pause positions, word duration as a stress proxy) | `StyleMetadata` object |
| 3 | Translation + register mapping | `pipeline/translator.py` | Final transcript segment + `StyleMetadata` | Translated text, register-preserving, + updated `StyleMetadata` for target language idiom |
| 4 | Expressive TTS | `pipeline/tts.py` | Translated text + `StyleMetadata` | Synthesized audio stream |

Stages 1 and 2 run **concurrently** where possible: register detection on partial transcripts starts as soon as AssemblyAI emits them, rather than waiting for the final transcript of a full utterance (see `orchestrator.py`'s speculative-detection logic and `pipeline/AGENTS.md`). This concurrency is driven by transcript text and word-timing-derived prosody signals, not by a separate sentiment-analysis pass — there isn't one, per the note in row 1 above. Stage 3 requires a finalized transcript segment (a completed phrase/sentence, determined by AssemblyAI's endpointing) plus the register metadata. Stage 4 begins streaming synthesis as soon as the first translated clause is available — it does not wait for the entire translated sentence.

---

## 2. The Style Metadata contract

This is the single most important interface in the system. It is the JSON structure that Stage 2 produces, Stage 3 refines, and Stage 4 consumes. Every module boundary in this pipeline is defined in terms of this schema — see `backend/app/schemas/style_metadata.py` for the pydantic model and `docs/style-metadata-schema.md` for the full field reference.

```json
{
  "tone": "sarcastic",
  "pace": "fast",
  "formality": "casual",
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
| Translation + register mapping | ~600ms | Single LLM call, **not streamed** (see "Not yet built" below): TTS starts only after the full translation returns, so this is total completion time, not first-token latency |
| TTS synthesis (first audio chunk) | ~500ms | ElevenLabs streaming synthesis; time-to-first-byte is the relevant metric, not full-clip generation |
| Network + orchestration overhead | ~200ms | WebSocket round trips, buffering |

**Latency techniques that are built and tested:**

1. **Partial-transcript pipelining.** Register detection begins on AssemblyAI's partial transcripts, not just finals. When the final arrives, the speculative result is reused if the final text is a continuation of the partial it ran on (otherwise it is cancelled and re-run), so detection is usually already done. `tests/unit/test_orchestrator_speculative_resolution.py`.
2. **Streaming on the output side.** TTS audio is forwarded to the client frame-by-frame as the provider emits it (`run_stream()`), and `tts_first_byte` is timed separately from total synthesis. `tests/integration/test_streaming_latency.py` fails if a change makes the pipeline buffer a clip.
3. **Bounded stages.** Register detection has a 1.5s timeout and translation has 3.0s/1.5s, so one slow provider can degrade fidelity but cannot freeze the session.

**Latency accounting for the feature layer (§7).** Every addition that touches the hot path, and where the budget absorbs it. None of these numbers has been measured against live providers — the first two are reasoned estimates, the third was measured offline:

| Addition | Where it lands | Cost |
|---|---|---|
| Conversation context in the detection + translation prompts | Input tokens on two existing LLM calls | Capped at 600 chars (~150 tokens) per call — small next to the few-shot block already in the detection prompt. Not measured live. |
| Explanation field in the detection output | Output tokens on the register-detection call | ~60–100 extra output tokens. This is real generation time in a stage budgeted at ~400 ms, but detection is speculative (usually hidden behind STT finalization). `REGISTER_EXPLAIN=false` removes it. **Must be checked with `python -m app.benchmark --live`.** |
| Acoustic analysis (`pipeline/prosody.py`) | Inline in the audio tap, per mic chunk | Pure Python, ~0.5% of real time on the dev sandbox (≈44 ms of CPU per 10 s of audio, i.e. ~0.6 ms per 128 ms chunk). It is synchronous CPU work in the event loop rather than I/O, so the "no blocking calls" rule is met in spirit: it is bounded, and a test fails if it exceeds 20% of real time. |
| STT pump / worker split (orchestrator) | Structure only | Adds no latency to a single utterance. It stops a segment's translation/TTS from blocking the reading of the *next* utterance's partials. Cost: a segment can now wait behind the previous one; that wait is reported as `queue_wait_ms` instead of being hidden. |
| Adaptive audio coalescing (POOR/FAIR network only) | Output side | Up to one coalesce window (100 ms FAIR / 200 ms POOR) of extra delay on the *first* frame, only on degraded links, never a whole clip; flushed at every segment end. Off on a good link. |
| Key-moment detection, memory, HUD timings | — | Deterministic arithmetic on data already in hand; no I/O, no LLM. |

**Not yet built (do not assume these exist):**

- **Streaming translation → TTS.** Translation is a single blocking `complete()`; TTS starts only after the whole translation returns. Moving to clause-level streaming is the largest remaining latency win but is not a small change: the translator's output is one JSON object containing both `translated_text` and the refined `style`, and JSON cannot be consumed until it closes. It would need a different output shape (e.g. a style header line first, then plain text streamed), which touches the prompt, both LLM adapters, the parser and the retry/fallback path — and its effect on translation quality can only be judged against a live model. Scope it as its own change with a live evaluation.
- **Translation re-run on a revised final.** Only register detection is re-run when the final transcript diverges from the partial; translation always starts from the final, so there is nothing to roll back.
- **Per-session voice-clone precompute.** Cloning itself is not implemented (see `CONTRIBUTING.md` "Known gaps").

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
| Register detection errors, returns low `confidence`, or **exceeds `RegisterDetector.TIMEOUT_S` (1.5s)** | Fall back to neutral-register `StyleMetadata` rather than blocking. The timeout matters: the translator waits on this result, so an unbounded LLM call would freeze the whole live session. |
| Translation LLM call fails or times out (3.0s, then 1.5s) | Retry once; if it fails again, a literal (register-naive) translation, and if even that fails the source text is passed through — never a dropped utterance |
| ElevenLabs errors or rate-limits | **Sticky** failover to `FallbackTTS` for the rest of the session (no flapping between voices). Retried on the fallback only if no audio had been emitted for that segment; a mid-clip failure ends the segment instead of splicing a second voice into it. Fallback segments carry `degraded: true` and the UI shows a banner. Fallback uses a stock OpenAI voice — Cartesia is not implemented. |
| AssemblyAI connection drops mid-session | Reconnect with exponential backoff (0.2s doubling to 5s), replaying up to 3s of recently *delivered* audio, then resuming live audio. Implemented server-side in `stt.py` (not client-side) around a queue owned by a single reader task; see the note below. |
| Voice analysis (`prosody.py`) raises, or too little speech to measure | The chunk still goes to STT; `acoustic` stays `null` for that utterance and detection runs on text/timings alone. Logged once, never interrupts audio. |
| Register detection fails or is low-confidence *but audio was measured* | Neutral-register fallback as before, **with the measured `acoustic`/`arousal` still attached** — they were measured, not guessed. |
| Explanation missing or malformed in the model's reply | Dropped; the register reading is used as if none was requested. A `contextual` cue is also dropped if no context was supplied. |
| Conversation summary LLM call fails, times out (8 s) or returns unusable JSON | Deterministic extractive summary, labelled `generated_by: "extractive"` so the UI never presents it as the real thing. |
| Poor network (client-reported or server-observed slow sends) | Partial captions throttled, audio frames coalesced — nothing is dropped; finals and audio bytes are always delivered. Recovers one tier at a time after 8 good observations. |
| Voice cloning unavailable/not consented | Stock voice rather than cloning. **Cloning itself is not implemented**, so this is currently the only path. |

> **STT reconnect note (a real bug this design fixes).** The first implementation iterated the caller's audio generator inside the per-connection sender task. Cancelling that task on a dropped connection *closed the generator*, so the session reconnected, replayed a little audio, and then received nothing — one network blip silently killed the microphone feed. Audio now flows source → `asyncio.Queue` → per-connection sender, so cancelling a sender never touches the source, and a chunk is removed from the pipeline only after a send succeeds. Guarded by `tests/unit/test_stt_reconnect.py` (verified to fail against the old design).

The general principle: **never let a non-critical stage block the pipeline**. Losing register fidelity gracefully (falling back to neutral) is far better than losing translation entirely.

---

## 6. What's built vs. planned

This is a living section — update it as the implementation progresses. Track actual status in `backend/app/pipeline/` docstrings and update this table alongside code changes.

| Component | Status |
|---|---|
| Repo scaffold, docs | Done |
| `StyleMetadata` schema | Done — see `backend/app/schemas/style_metadata.py` |
| Register detector (LLM + prompt + few-shots, fallback logic) | Done — real prompt, JSON parsing, confidence-threshold fallback. Not yet validated against a live LLM's actual output quality. |
| Translator (LLM, register-aware, retry + literal fallback) | Done — real prompt, retry-then-literal-fallback chain implemented and tested, including markdown-fence stripping shared with the register detector via `pipeline/llm_response_utils.py`. Not yet validated against a live LLM. |
| TTS style mapping (`StyleMetadata` → ElevenLabs voice settings) | Done — pure mapping function, unit tested. |
| STT integration (AssemblyAI v3 realtime streaming) | Done — real protocol implementation (`Begin`/`Turn`/`Termination`/`Error` messages, correct auth, reconnect-with-backoff, bounded replay buffer). Verified against a real local WebSocket server speaking the same protocol (`tests/unit/test_stt.py`, and `test_stt_reconnect.py` for drop-and-resume), not yet against a live AssemblyAI account. Note: AssemblyAI's realtime API does not provide sentiment analysis — `TranscriptEvent.sentiment` is always `None` from this provider; register detection derives signal from word timings instead. |
| ElevenLabs API integration | Partially wired — `ElevenLabsTTS.synthesize()`/`clone_voice()` call the real SDK shape, but untested against a live API key |
| Fallback TTS (OpenAI) | Wired and failover-tested against fakes; untested against a live key. Ignores the ElevenLabs `voice_id` and always uses a stock OpenAI voice (OpenAI only accepts a fixed voice-name enum) — a surfaced degradation, not a bug. Cartesia branch not implemented. |
| Orchestrator concurrency (speculative register detection on partials) | Done — see `backend/app/pipeline/orchestrator.py`. The partial-vs-final reuse logic actually checks whether the final transcript is a continuation of the text the speculative pass ran on (not just "did it finish"), and cancels rather than abandons a stale or slow speculative task. Tested in both `tests/integration/test_orchestrator_flow.py` and `tests/unit/test_orchestrator_speculative_resolution.py`. |
| LLM provider adapters (OpenAI, Anthropic) | Done for OpenAI and Anthropic — see `backend/app/ws/session.py::_build_llm_client()`. Groq and Google not yet implemented (raises `NotImplementedError` with a clear message). |
| Settings loading | Done — lazy (`get_settings()`, `lru_cache`'d), so importing any module never crashes without a populated `.env`; only calling `get_settings()` does, with a clear pydantic validation error. Verified in `tests/unit/test_config_and_session.py` and `tests/unit/test_app_import.py`. |
| TTS provider failover | Done — `SessionOrchestrator` switches to `fallback_tts` on the first primary failure and **stays there for the session** (no flapping between voices). The failing segment is retried on the fallback only if no audio had been emitted yet; once bytes are sent, a second voice is never spliced into the same utterance. Segments from the fallback carry `degraded: true` end-to-end, and the UI shows a banner. `tests/integration/test_tts_failover.py`, `test_streaming_latency.py`. |
| Streaming audio (no clip buffering) | Done — `SessionOrchestrator.run_stream()` yields `SegmentStart`, `AudioChunk*`, `SegmentEnd`; the WebSocket forwards each frame as the TTS provider emits it. `tts_first_byte` is timed separately from total `tts` (TTFB is what the ~2s budget is about). `test_streaming_latency.py` uses a TTS that blocks mid-clip and fails if the pipeline buffers. `run()` remains as a buffered convenience wrapper. |
| Audio format contract | Done — both TTS providers are explicitly asked for raw PCM16 mono 24 kHz (`pcm_24000` / OpenAI `pcm`); previously neither call specified a format and both silently returned MP3, which can't be decoded chunk-by-chunk in a browser. The server declares the format in a `ready` message at session start; the client never hardcodes it. |
| Wire-contract drift guard | Done — `tests/unit/test_wire_contract.py` parses `frontend/lib/ws-client.ts` and asserts its `ReadyMessage`/`AudioFormat`/`SegmentMessage`/`StyleTag` fields exactly match what the backend serializes. Verified to fail when a field is removed from one side. |
| Docs-vs-schema drift guard | Done — `tests/unit/test_docs_examples.py` validates every `StyleMetadata` JSON example in the markdown docs against the real pydantic model (this once caught a doc example with an invalid `"informal"` value). |
| Latency measurement | Done as a tool, **not yet run against real providers** — `app/latency.py` (p50/p90/max per stage, p90-based budget verdict, critical-path sum) and `python -m app.benchmark` (`--dry-run` offline, `--live` explicit and billed). The dry run only echoes injected delays; there is currently no real latency number for this system. STT endpointing and browser network overhead are outside what it can see. See `TESTING.md`. |
| Voice cloning consent flow | **Not implemented** — see `CONTRIBUTING.md` "Known gaps"; do not wire into a live demo without one |
| Frontend: WebSocket client (`lib/ws-client.ts`) | Done — matches the backend's actual wire format exactly (segment/error JSON messages + binary audio frames), with defensive handling for a message that doesn't match the expected shape (`onProtocolError`) |
| Frontend: mic capture + live session (`app/page.tsx`) | Done — real `AudioWorklet`-based PCM16 capture and downsampling to 16kHz (matching `AssemblyAISTT`'s expected encoding), start/stop controls, live transcript and tone-tag rendering as segments arrive. Verified with `tsc --noEmit --strict` and `eslint` (including `react-hooks/exhaustive-deps`), not yet tested against a live backend session. |
| Frontend: translated audio playback | Implemented, not yet heard on real hardware — `lib/player.ts` (`PcmPlayer`) decodes the server-declared PCM16 format, carries odd trailing bytes across frames (`lib/pcm.ts`, unit-tested), and schedules buffers gaplessly on the `AudioContext` clock, resyncing after underruns. Pure logic covered by `npm test` (6 tests); the `AudioContext` path itself is unverified in a browser (see `CONTRIBUTING.md` "Known gaps"). |
| Frontend: waveform visualization | Stub — `Waveform.tsx` renders a static placeholder, not real amplitude data |
| Frontend type-checking / linting | Done — `npm run typecheck` (`tsc --noEmit --strict`, clean) and `npm run lint` (ESLint flat config with `typescript-eslint` + `react-hooks` rules, clean) |
| Test suite (backend, unit + integration, against fakes) | Done — 114 backend tests + 6 frontend tests passing, see `TESTING.md` for the full breakdown by file |
| Live/integration tests against real provider APIs | Planned — `tests/live/` is currently an empty scaffold, see `TESTING.md` |
| Latency benchmarking against live APIs | Tooling exists (`--live`); has never been run — see the "Latency measurement" row |

**What an agent picking this up next should prioritize:** the whole path from microphone to speaker is now implemented and tested against fakes: mic capture, AssemblyAI streaming STT, register detection, register-aware translation, streaming TTS with sticky failover, and gapless PCM playback. What has **not** happened is any contact with reality. In order of value: (1) run it once against real AssemblyAI / LLM / ElevenLabs accounts and listen to it — every provider call shape here comes from documentation, not from a successful call; (2) run `python -m app.benchmark --live` to get the first real latency numbers against the ~2s budget; (3) hear `PcmPlayer` on real hardware, with headphones (echo handling is not implemented, see `CONTRIBUTING.md` "Known gaps"); (4) the voice-cloning consent flow, which must exist before cloning is enabled. `tests/live/` remains an empty scaffold and is the natural home for (1) and (2).
