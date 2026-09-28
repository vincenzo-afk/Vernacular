# CLAUDE.md

Instructions for Claude (via Claude Code or any Claude-based agent) working in this repository.

Read [README.md](./README.md) and [ARCHITECTURE.md](./ARCHITECTURE.md) first if you haven't already — this file assumes you know what Vernacular is and how the pipeline is structured. This file is about *how to work in this codebase*, not what it does.

If you're about to edit anything in `backend/app/pipeline/`, also read `backend/app/pipeline/AGENTS.md` — it has stricter, more specific rules for that directory (concurrency model, exact per-stage failure-handling contract) than what's summarized here. For process conventions (how to structure a change) see `CONTRIBUTING.md`; for what to test and how, see `TESTING.md`.

Check `ARCHITECTURE.md` §6 before assuming a module is a stub or a real implementation — as of the last update, every backend pipeline stage (`stt.py`, `register_detector.py`, `translator.py`, `tts.py`, `orchestrator.py`) is a real, tested implementation, and the frontend's WebSocket client and mic capture (`lib/ws-client.ts`, `app/page.tsx`) are wired end-to-end. The main remaining gap is validation against live provider accounts and real audio hardware — see `ARCHITECTURE.md` §6 for specifics.

---

## Project in one paragraph

Vernacular is a real-time voice translation pipeline that preserves speaking register (tone, pace, formality, sarcasm, emotion) across languages, not just literal word meaning. Audio flows through four streaming stages — STT (AssemblyAI) → register detection (LLM) → register-aware translation (LLM) → expressive TTS (ElevenLabs) — glued together by a FastAPI backend and consumed by a Next.js frontend over WebSockets. The full data contract between stages is the `StyleMetadata` schema in `backend/app/schemas/style_metadata.py`. The system has a hard ~2-second end-to-end latency target.

---

## Non-negotiable constraints

These are not style preferences — violating them breaks the project's core value proposition or its demo-day viability.

1. **The ~2s latency budget is real, not aspirational.** Before adding any synchronous step, a new LLM call, or a blocking I/O operation to the hot path (the path from audio-in to audio-out), check the latency budget in `ARCHITECTURE.md` §3. If your change adds latency, say so explicitly and show where the budget absorbs it — don't add unaccounted-for latency silently.
2. **Never buffer a full utterance when streaming is possible.** STT, translation, and TTS are all consumed as streams. If you write code that waits for a complete transcript, a complete translation, or a complete audio clip before starting the next stage, you have likely broken the pipeline's core design — check if a streaming alternative exists first.
3. **All style information flows through `StyleMetadata`, never free text.** Do not pass style as a prose string ("say this sarcastically") between pipeline stages. If you need a new style dimension, add a typed field to the schema in `backend/app/schemas/style_metadata.py` and update `docs/style-metadata-schema.md` — don't bolt on an untyped string field.
4. **No provider SDK calls outside `pipeline/*.py`.** `STTProvider` and `TTSProvider` (and the equivalent for the translation LLM) are the abstraction boundary. Route calls through them so providers stay swappable (see the fallback providers listed in the tech stack).
5. **Degrade gracefully, never silently drop output.** If register detection fails, fall back to neutral register (§5 of ARCHITECTURE.md) — don't skip the utterance. If a preferred provider fails, fall back per the table in ARCHITECTURE.md §5 — don't crash the session.
6. **Translation and register mapping are one step, not two.** Do not implement "translate literally, then apply tone as a post-process" — a literal translation frequently cannot carry the same register in the target language (idiom, sentence structure, and honorifics/formality markers differ by language). The single LLM call in `translator.py` must do both simultaneously with full context.

---

## Code standards for this repo

- **Backend**: Python 3.11+, FastAPI, fully async (`async def` / `asyncio`) for anything in the pipeline hot path. Type-hint everything; pipeline modules use `Protocol` classes for provider interfaces (see `pipeline/stt.py`, `pipeline/tts.py`).
- **Style metadata is a pydantic model.** Validate at every boundary — don't pass raw dicts between stages.
- **Frontend**: Next.js (App Router), TypeScript, Tailwind. No CSS-in-JS libraries beyond Tailwind utilities.
- **Tests**: pipeline stages should be testable independently of live provider APIs — mock `STTProvider`/`TTSProvider` implementations belong in `backend/tests/fakes/`. Don't write tests that require live API keys to run in CI.
- **Config/secrets**: all API keys via environment variables, loaded through `backend/app/config.py`. Never hardcode a key, never commit `.env`. `.env.example` files document required vars without values.
- **Logging**: log stage-level timing (start/end timestamps per pipeline stage) so latency regressions are visible — this is a latency-sensitive system and timing visibility isn't optional instrumentation, it's core to the project.

---

## When you're asked to add a feature

Before writing code, check:

1. Does this touch the pipeline hot path (audio-in to audio-out)? If yes, re-read ARCHITECTURE.md §3 and account for latency impact explicitly in your response.
2. Does this need a new style dimension? If yes, it goes in the `StyleMetadata` schema, not as a side channel.
3. Does this add a new external provider/API? If yes, it needs a `Protocol` interface in the relevant `pipeline/` module, following the existing pattern — not a direct SDK call from `orchestrator.py` or anywhere else.
4. Does this change fallback/error behavior? If yes, update the fallback table in ARCHITECTURE.md §5 to match.

## When you're asked to fix a bug

Check whether the bug is a latency regression before assuming it's a correctness bug — in a streaming pipeline, "wrong output" and "output arrived too late to matter" often look similar from the outside (e.g., translated audio for the wrong utterance because the previous one hadn't finished streaming). Check stage timing logs first.

## What NOT to do

- Don't add a "simple mode" that translates without register detection as the default path — that's a different, less interesting product. A fast neutral-register fallback for failure cases (§5) is fine; making it the primary path is not.
- Don't hardcode language-specific formality logic (e.g., Japanese honorifics, French tu/vous) as special-case string manipulation in `translator.py`. That logic belongs in the LLM prompt/few-shots for the translation step, where it can generalize — see the register-detection prompt design note in `pipeline/register_detector.py`.
- Don't introduce a synchronous/blocking call anywhere in `pipeline/` or `orchestrator.py`. If a library only offers a sync client, wrap it (`asyncio.to_thread` or equivalent) rather than calling it directly in an async context.
