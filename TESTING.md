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
│   └── test_tts_mapping.py
├── integration/
│   ├── test_orchestrator_flow.py   # full pipeline, all fakes wired together
│   └── conftest.py
└── live/                            # opt-in, requires real API keys, skipped by default
    ├── test_assemblyai_live.py
    └── test_elevenlabs_live.py
```

Run the default (fast, offline) suite:

```bash
cd backend
pytest tests/unit tests/integration
```

Run live tests (requires `.env` populated with real keys):

```bash
pytest tests/live -m live
```

`pytest.ini` / `pyproject.toml` should mark live tests with `@pytest.mark.live` and exclude that marker from the default run — see `backend/pytest.ini`.

---

## What each layer of tests is responsible for

### Unit tests (`tests/unit/`)

Test one pipeline stage in isolation, with every dependency faked.

- **`test_style_metadata.py`**: schema validation — enum bounds, `sarcasm_score`/`confidence` clamped to `[0,1]`, `neutral_fallback()` produces a valid, genuinely neutral object.
- **`test_register_detector.py`**: given a scripted LLM response, does `RegisterDetector.detect()` return the expected `StyleMetadata`? Given an LLM call that raises, does it return `neutral_fallback()` instead of propagating? Given a low-confidence LLM response, does it also fall back?
- **`test_translator.py`**: given a scripted LLM response, does `Translator.translate()` return the expected `TranslationResult`? Given a failing first call, does it retry once? Given two consecutive failures, does it fall back to literal translation rather than raising?
- **`test_tts_mapping.py`**: does `StyleMetadata` map to the correct ElevenLabs style parameters? (This is a pure mapping test — no network calls, fake the ElevenLabs client and assert on the parameters it was called with.)

### Integration tests (`tests/integration/`)

Test the orchestrator wiring multiple fakes together — this is where you catch "the stages don't actually compose correctly" bugs that unit tests miss.

- **`test_orchestrator_flow.py`**: feed a fake audio stream through `SessionOrchestrator.run()` with `FakeSTT` + `FakeLLMClient` + `FakeTTS`, assert that translated audio comes out, that `StageTiming` entries were recorded for every stage, and that a register-detector failure mid-stream doesn't kill the session (it should fall back to neutral and continue).

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

The ~2s end-to-end latency target (`ARCHITECTURE.md` §3) is a real-world property measured against live APIs, not something the offline test suite can verify. Benchmark it manually:

```bash
cd backend
python -m tests.live.benchmark_latency --target-language es --utterance-count 10
```

This script (to be implemented — see `backend/tests/live/` scaffold) should record stage-level timings for real utterances against real provider APIs and report p50/p90 for each stage plus end-to-end. Run this after any change to the orchestrator's concurrency strategy, the register-detection prompt, or provider selection — these are the areas most likely to silently regress latency without breaking correctness.
