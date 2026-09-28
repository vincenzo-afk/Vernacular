# TESTING.md

How to test Vernacular — for both human contributors and AI agents. Read this before adding new tests or before assuming existing ones "probably pass."

---

## Philosophy

This is a real-time streaming system built on three external APIs (AssemblyAI, an LLM provider, ElevenLabs). Tests must not require live API keys to run in CI — every pipeline stage is tested against a fake implementation of its provider interface (`STTProvider`, `TTSProvider`, the LLM client). Live-API integration tests exist separately, are opt-in, and are never part of the default test run.

This split matters for agents specifically: if you're asked to "add a test" or "make sure this works," the default expectation is a fast, deterministic, offline test using the fakes in `backend/tests/fakes/`. Don't reach for a live API call unless explicitly asked for an integration test.

---

## Test layout

```
backend/tests/
├── fakes/
│   ├── fake_stt.py           # FakeSTT + scripted TranscriptEvents
│   ├── fake_llm.py           # FakeLLMClient — scripted responses for
│   │                         #   register detection and translation
│   └── fake_tts.py           # FakeTTS — records calls, returns dummy bytes
├── unit/
│   ├── test_style_metadata.py
│   ├── test_register_detector.py
│   ├── test_translator.py
│   ├── test_tts_mapping.py               # pure StyleMetadata -> ElevenLabs mapping
│   ├── test_fallback_tts.py              # FallbackTTS voice-identity handling
│   ├── test_stt.py                       # AssemblyAISTT against a real local
│   │                                     #   fake server speaking its wire protocol
│   ├── test_stt_reconnect.py             # drop-and-resume: live audio must survive a reconnect
│   ├── test_orchestrator_speculative_resolution.py  # partial-vs-final
│   │                                     #   register-detection reuse logic
│   ├── test_config_and_session.py        # lazy settings loading, LLM adapter selection
│   ├── test_app_import.py                # whole app import chain + /health endpoint
│   ├── test_tts_audio_format.py          # both providers are asked for raw PCM16 24 kHz
│   ├── test_wire_contract.py             # backend wire JSON == frontend ws-client.ts types
│   ├── test_docs_examples.py             # every StyleMetadata example in the docs validates
│   ├── test_latency.py                   # percentile/budget math + real-orchestrator timing
│   └── test_benchmark.py                 # benchmark CLI: dry-run correctness, live guard rails
├── integration/
│   ├── test_orchestrator_flow.py         # full pipeline, all fakes wired together
│   ├── test_tts_failover.py              # sticky primary -> fallback TTS failover
│   ├── test_streaming_latency.py         # audio streams; it is NOT buffered per clip
│   └── test_ws_roundtrip.py              # real WebSocket: ready -> segment -> audio frames
└── live/                            # opt-in, requires real API keys, skipped by default
    └── (empty scaffold — see "Live tests" below; nothing implemented yet)
```

This listing reflects the actual files in the repo as of the last update — if you add a test file, add it here too, in the same change, so this doesn't drift the way it once did (see git history / CONTRIBUTING.md on why stale docs cost more than they save).

Note `test_stt.py` specifically: it runs a real local WebSocket server on localhost speaking AssemblyAI's v3 wire protocol and points `AssemblyAISTT` at it via a `base_url` override. This is still an offline test — no real AssemblyAI account, no network egress — so it lives in `tests/unit/`, not `tests/live/`. "Live" in this repo means "talks to the actual third-party API," not "uses a real network socket."

Run the default (fast, offline) suite:

```bash
cd backend
PYTHONPATH=. pytest tests/unit tests/integration
```

Run live tests (requires `.env` populated with real keys):

```bash
pytest tests/live -m live
```

As of the last update, `tests/live/` is an empty scaffold (just `__init__.py`) — no live tests have been written yet, so this command currently collects zero tests. Add real live tests here as the provider integrations in `stt.py`/`tts.py` get validated against actual accounts; see "Live tests" below for what belongs here vs. in `tests/unit/`.

`pytest.ini` marks live tests with `@pytest.mark.live` and excludes that marker from the default run (`addopts = -m "not live"`) — see `backend/pytest.ini`.

---

## What each layer of tests is responsible for

### Unit tests (`tests/unit/`)

Test one pipeline stage in isolation, with every dependency faked.

- **`test_style_metadata.py`**: schema validation — enum bounds, `sarcasm_score`/`confidence` clamped to `[0,1]`, `neutral_fallback()` produces a valid, genuinely neutral object.
- **`test_register_detector.py`**: given a scripted LLM response, does `RegisterDetector.detect()` return the expected `StyleMetadata`? Given an LLM call that raises, does it return `neutral_fallback()` instead of propagating? Given a low-confidence LLM response, does it also fall back?
- **`test_translator.py`**: given a scripted LLM response, does `Translator.translate()` return the expected `TranslationResult`? Given a failing first call, does it retry once? Given two consecutive failures, does it fall back to literal translation rather than raising? Does the literal-fallback path strip markdown fences the same as the main JSON path?
- **`test_tts_mapping.py`**: does `StyleMetadata` map to the correct ElevenLabs style parameters? (This is a pure mapping test — no network calls.)
- **`test_fallback_tts.py`**: does `FallbackTTS` correctly refuse to pass a primary-provider (ElevenLabs-shaped) voice ID through to OpenAI's incompatible fixed voice enum, using a fake OpenAI-shaped client rather than the real SDK (which isn't an offline-test dependency)?
- **`test_stt.py`**: `AssemblyAISTT` against a real local WebSocket server speaking the same Begin/Turn/Termination/Error protocol AssemblyAI's v3 streaming API uses — exercises the actual connection and parsing code, not a mock of it. Also covers the stress-detection heuristic and turn-message parsing as pure functions.
- **`test_stt_reconnect.py`**: a real local server kills the first connection mid-stream. Asserts live audio resumes on the new connection (not just a replay), replayed frames arrive in order before live ones, nothing is lost or reordered, and the caller's audio source is not closed by the reconnect. All four fail against the original design (verified), which closed the source on the first drop and silently killed the mic feed.
- **`test_orchestrator_speculative_resolution.py`**: targeted coverage of `_resolve_style_for_final`'s prefix-validity check — does it correctly reuse a completed speculative register-detection result when the final transcript is a continuation of the partial it ran on, and correctly discard/cancel it and run fresh when the text diverged?
- **`test_config_and_session.py`**: does `get_settings()` load lazily (not crash on import) and raise clearly when required env vars are missing? Does `_build_llm_client` select the right adapter per `llm_provider` and raise `NotImplementedError` for unsupported ones (Groq, Google)?
- **`test_app_import.py`**: does the whole app import chain (`main.py` → `ws/session.py` → `config.py`) import cleanly with zero env vars configured, and does `/health` respond correctly?
- **`test_tts_audio_format.py`**: both `ElevenLabsTTS` and `FallbackTTS` explicitly request raw PCM16 at 24 kHz. Before this, neither call named a format and both returned MP3, which a browser cannot decode chunk-by-chunk.
- **`test_wire_contract.py`**: *drift guard.* Parses `frontend/lib/ws-client.ts` and asserts its `ReadyMessage`, `AudioFormat`, `SegmentMessage` and `StyleTag` fields exactly equal what the backend serializes. There is no schema codegen, so this is what makes the manual sync enforceable. Verified to fail when a field is removed from either side.
- **`test_docs_examples.py`**: *drift guard.* Extracts every `StyleMetadata`-shaped JSON block from the markdown docs and validates it against the real pydantic model. It once caught a flagship example using an invalid `"informal"` formality value.
- **`test_latency.py`**: percentile math, that over-budget is judged on p90 rather than the mean, that register detection is excluded from the critical path, and — end to end — that a delay injected into the TTS lands in the `tts_first_byte` stage of the real orchestrator and not in `translation`.
- **`test_benchmark.py`**: dry-run exits 0, labels its numbers as not real, and reports the injected delays in the right stages; a mode is required and the two are mutually exclusive; `--live` without credentials, or without a voice ID, exits 1 and never runs (it will not guess a voice). These never touch the network.

### Integration tests (`tests/integration/`)

Test the orchestrator wiring multiple fakes together — this is where you catch "the stages don't actually compose correctly" bugs that unit tests miss.

- **`test_orchestrator_flow.py`**: feed a fake audio stream through `SessionOrchestrator.run()` with a scripted STT + `FakeLLMClient` + `FakeTTS`, assert that translated audio comes out, that `StageTiming` entries were recorded for every stage, and that a register-detector failure or a TTS failure mid-stream doesn't kill the session.
- **`test_tts_failover.py`**: on primary TTS failure the orchestrator switches to the fallback for the rest of the session and never flaps back; the failing segment is retried on the fallback; fallback segments are flagged `degraded`; with no fallback (or a failing one) segments are dropped without killing the session.
- **`test_streaming_latency.py`**: the tests that distinguish *streaming* from *buffering*. The TTS fake yields one chunk and then blocks; the first chunk must reach the consumer while the TTS is still blocked. Also covers `tts_first_byte` timing and the rule that a mid-clip failure ends the segment rather than splicing a second voice into it. Verified to fail (timeout) if the orchestrator is changed to buffer the clip.
- **`test_ws_roundtrip.py`**: drives `run_session()` over a real FastAPI WebSocket and pins the protocol order the browser depends on (`ready`, then `segment`, then binary frames), byte-exact audio, and that re-framing at an odd byte offset still reassembles the original samples.

### Live tests (`tests/live/`)

Real API calls. Skipped by default. Use sparingly — these exist to catch actual provider integration bugs (wrong auth header, wrong websocket message format, wrong response parsing) that no fake can catch. Do not add business logic tests here; if a live test is testing something the fakes could test, move that assertion into `unit/` or `integration/`.

---

## Rules for agents adding or modifying tests

1. **New pipeline logic needs a unit test using the existing fakes**, not a new live-API test. If the fakes in `tests/fakes/` don't support the scenario you need, extend the fake — don't skip the test or reach for a live call.
2. **Never make the default test run depend on network access or real credentials.** If you add a test that needs a live key, mark it `@pytest.mark.live` and put it in `tests/live/`.
3. **Test the fallback path, not just the happy path.** Every stage in this pipeline has a documented failure/fallback behavior (`ARCHITECTURE.md` §5). If you touch a pipeline stage, check whether its fallback behavior still has test coverage — this codebase treats untested fallback logic as equivalent to broken fallback logic, since fallback paths only run in production during a live demo failure.
4. **Timing/latency assertions should be structural, not wall-clock.** Don't assert `elapsed < 500ms` in a unit test using fakes — that's testing your test environment's speed, not the pipeline's design. Instead assert *that* `StageTiming` was recorded, that stages that should run concurrently don't block each other in the fake's execution order, etc. Actual latency budget verification belongs in a manual/live benchmark, documented separately (see "Latency benchmarking" below), not in the automated suite.
5. **If you add a new `StyleMetadata` field** (see `AGENTS.md` checklist), add a test case to `test_style_metadata.py` covering its bounds/defaults, and update `test_tts_mapping.py` if it should affect TTS output.

---

## Latency benchmarking (manual, not CI)

The ~2s end-to-end target (`ARCHITECTURE.md` §3) is a property of real providers, so the offline suite can only prove the *measurement* is correct, not the number. Two entry points, deliberately separate:

```bash
cd backend
python -m app.benchmark --dry-run                       # offline; proves the harness works
python -m app.benchmark --live --language es --repeats 2  # REAL providers; bills your accounts
```

- **`--dry-run`** uses fake providers with injected delays (50ms detection, 150ms translation, 100ms TTS first byte) and prints a banner saying so. The numbers it shows are the delays we injected. They say nothing about any real service; the run only shows the harness and report work.
- **`--live`** runs the fixed utterance set (`app.benchmark.UTTERANCES`: sarcasm, urgency, formal, a one-word fragment, a longer neutral line) through the real LLM and ElevenLabs. It needs API keys and `BENCHMARK_VOICE_ID` in `.env`, refuses to run without them, and exits 1. It requires an explicit flag because it costs money. It loads the same settings as the server, so `ASSEMBLYAI_API_KEY` must be set (a placeholder is fine — STT is not called).

The report gives p50 / p90 / max per stage against the budgets in `app/latency.py`, judged on **p90** (a good mean can hide a slow tail), plus a critical-path sum of translation + `tts_first_byte`. Register detection is excluded from that sum on purpose: it runs speculatively during STT and is normally finished when the final transcript arrives.

**What this does not measure:** STT endpointing latency (needs a live AssemblyAI audio stream) and network overhead to the browser. The report says so; add those from a real session before comparing anything to 2000ms. Until `--live` has been run, **there is no real latency number for this system.**

Run it after changing the orchestrator's concurrency, the register-detection or translation prompts, or provider/model selection. Those are the changes most likely to regress latency without failing any test.

---

## Frontend checks (`frontend/`)

There's no browser test framework (no Playwright) — only Node's built-in `node:test` for pure logic. "Tested" for frontend code means passing all three of the following, which are fast enough to run on every change and catch the two most common classes of bug in this codebase's frontend:

```bash
cd frontend
npm run typecheck   # tsc --noEmit — catches wire-format drift between
                     #   ws-client.ts and what components actually consume
npm test             # node:test unit tests for lib/pcm.ts (odd-byte alignment,
                     #   gapless scheduling) — no browser needed
npm run lint         # eslint with react-hooks/exhaustive-deps active —
                     #   catches missing dependencies in useCallback/useEffect,
                     #   which is the single easiest mistake to make in
                     #   app/page.tsx's mic-capture and session wiring
```

Both are checked into CI expectations the same way the backend's offline pytest suite is — neither requires a browser, a mic, or a live backend to run.

If a future change adds real interactivity tests (e.g. Playwright driving the mic-permission flow), they belong in a new `frontend/tests/` directory, offline-by-default per the same philosophy as the backend: fake the WebSocket server (there's already a real-local-server pattern to follow — see `backend/tests/unit/test_stt.py`'s `FakeAssemblyAIServer` for the equivalent idea applied to a different protocol) rather than requiring a live backend and live provider APIs to run the frontend's own test suite.

**What `npm run typecheck` would have caught, historically:** the wire format documented in `backend/app/ws/session.py`'s module docstring and the shape `frontend/lib/ws-client.ts` actually parses have to match exactly (segment/error message shapes, field names). There's no shared schema codegen between them (see `CONTRIBUTING.md` "Known gaps") — `tsc --noEmit` won't catch a backend field rename by itself, but it *will* catch a frontend consumer (e.g. a component) using a field that no longer exists on `StyleTag`/`SegmentMessage` if `ws-client.ts` is updated correctly and a consumer isn't. Keep that manual-sync discipline in mind: updating one side without checking the other is exactly the kind of change `npm run typecheck` is meant to catch, but only if you actually run it.
