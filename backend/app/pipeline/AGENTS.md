# AGENTS.md — `backend/app/pipeline/`

This directory is the hot path: audio in, translated audio out, ~2 seconds end-to-end. Everything here is held to a stricter standard than the rest of the repo. Read the repo-level `AGENTS.md` first — this file adds constraints specific to this directory; it doesn't replace anything there.

---

## Files in this directory and their one job each

| File | One job | Must never |
|---|---|---|
| `stt.py` | Turn an audio stream into a transcript event stream | Buffer a full utterance before yielding anything |
| `register_detector.py` | Turn a transcript event into `StyleMetadata` | Raise an exception to its caller — always resolve to either a real reading or `StyleMetadata.neutral_fallback()` |
| `translator.py` | Turn (source text, source style) into (translated text, target style) | Split translation and register-mapping into two separate LLM calls |
| `tts.py` | Turn (translated text, style) into an audio byte stream | Wait for full text before starting synthesis, when the provider supports streaming input |
| `orchestrator.py` | Wire the above four into one session, with correct concurrency and fallback | Let a single stage's failure kill the session, or collect a whole audio clip before yielding it (use `run_stream()`; `run()` is a buffered convenience wrapper and must not be used on the latency path) |

If you're adding code to one of these files, check that it's still doing only its one job. If a change to `translator.py` needs TTS-specific logic, that logic belongs in `tts.py`, reached via `TranslationResult`/`StyleMetadata`, not inlined into the translator.

---

## The concurrency model (read before touching `orchestrator.py`)

```
time ──────────────────────────────────────────────────────────▶

STT:        [partial][partial][partial]......[FINAL]
Register:          [--- speculative pass on partials ---][confirm on FINAL]
Translate:                                                       [start after FINAL + register confirmed]
TTS:                                                                    [start on first translated clause]
```

- Register detection is **speculative**: it runs on partial transcripts before the final transcript exists. This is correct because most partials stabilize before the utterance ends — the "confirm on FINAL" pass is usually cheap (the LLM sees it already reasoned about very similar text).
- Translation **cannot** start until both (a) STT has emitted a final transcript for the segment, and (b) register detection has resolved (either a real reading or the neutral fallback) — it needs both source text and style as input, by design (see `CLAUDE.md` constraint #6, translation + register mapping is atomic).
- TTS starts on the **first translated clause**, not the full translated sentence, wherever the TTS provider supports streaming text input. If a provider only accepts a full string, `tts.py` should buffer only as much as that provider strictly requires — do not add unnecessary buffering in `orchestrator.py` "to be safe."

If you're implementing `SessionOrchestrator.run()`, this means: use `asyncio.Queue` or async generators to let register detection consume STT's partial stream independently of the main STT→translate→TTS chain, and use `asyncio.create_task` (not sequential `await`) for anything that can genuinely run concurrently. Sequential `await` on independent operations is the most common way this pipeline's latency budget gets silently blown.

---

## Failure handling — the exact behavior expected per stage

This is the concrete version of `ARCHITECTURE.md` §5, scoped to what code in this directory must actually do:

- **`stt.py`**: on a dropped connection, reconnect with exponential backoff (0.2s doubling to a 5s cap) and replay up to `MAX_BUFFER_SECONDS` (3s) of recently *delivered* audio, then resume live audio; if the replay buffer overflows, drop oldest rather than grow unbounded. **Never iterate the caller's audio generator inside a per-connection task**: cancelling that task closes the generator and permanently kills the feed. Audio must flow source → queue → sender (see `ARCHITECTURE.md` §5 and `tests/unit/test_stt_reconnect.py`).
- **`register_detector.py`**: any exception from the LLM call, a result with `confidence < CONFIDENCE_THRESHOLD`, **or no answer within `TIMEOUT_S` (1.5s)** resolves to `StyleMetadata.neutral_fallback()`. This method must never raise past its own boundary and must never wait unboundedly — the translator blocks on it, so a hung call would freeze the live session.
- **`translator.py`**: one retry with a shorter timeout on the first failure. On a second failure, fall back to a literal/register-naive translation (still produce *some* output) rather than propagating an exception — a wrong-register translation is a better outcome than no translation.
- **`tts.py` / `orchestrator.py` (TTS failover)**: on primary provider (ElevenLabs) failure or rate-limit, the orchestrator switches to `FallbackTTS` for the **remainder of the session** — flapping between providers mid-session produces jarring voice changes, so it never switches back. The segment that hit the failure is retried on the fallback **only if the primary had emitted no audio yet**. Once bytes have gone out they cannot be taken back, and splicing a second voice into the middle of one utterance is worse than ending it early, so a mid-clip failure ends that segment with the audio already sent (the provider is still marked failed for later segments). Every segment produced by the fallback carries `degraded: true` on the wire and the UI must show it — never present reduced fidelity as full quality (`ARCHITECTURE.md` §5). Both providers must be asked for raw PCM16 mono 24 kHz (`AUDIO_*` constants in `tts.py`); do not rely on a provider default format.
- **`orchestrator.py`**: no unhandled exception from any stage should propagate out of `run()` and kill the WebSocket session. Catch at each stage boundary, apply the fallback above, log it (see logging note below), and continue the session.

---

## Logging requirement (not optional)

Every stage transition in this directory must record a `StageTiming` (see `orchestrator.py`) with real monotonic timestamps — not estimates, not "should be fast." This is how latency regressions get caught, since there's no automated latency test in CI (see `TESTING.md` — latency is benchmarked manually against live APIs). If you add a new stage or split an existing one, add corresponding timing capture in the same change.

---

## Before submitting a change to this directory

In addition to the repo-level checklist in `AGENTS.md`:

- [ ] Does my change preserve the concurrency model above, or does it introduce an unnecessary sequential `await` on the hot path?
- [ ] If I touched a stage's failure handling, does it match the exact fallback behavior specified above (not just "something reasonable")?
- [ ] Did I add/update `StageTiming` capture for any new or modified stage boundary?
- [ ] If I touched anything on the audio path, does `tests/integration/test_streaming_latency.py` still pass? (It fails if a change makes the pipeline buffer a clip.)
- [ ] Did I add a unit test using the fakes in `backend/tests/fakes/` covering both the happy path and the fallback path (see `TESTING.md`)?
