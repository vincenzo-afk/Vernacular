# CONTRIBUTING.md

Workflow conventions for anyone — human or AI agent — making changes to this repository. This is about *process*; for coding constraints see `CLAUDE.md` / `AGENTS.md`, and for test expectations see `TESTING.md`.

---

## Before you start

1. Read `README.md` and `ARCHITECTURE.md` if you haven't. Most bad changes in this repo come from not knowing the latency budget or the `StyleMetadata` contract exists, not from writing bad code per se.
2. Check `ARCHITECTURE.md` §6 ("What's built vs. planned") to see what's actually implemented vs. stubbed. Don't assume a module is production-ready just because a file exists — check for `NotImplementedError` or a `# TODO` before building on top of it.
3. If your change touches `backend/app/pipeline/`, also read `backend/app/pipeline/AGENTS.md` — the pipeline directory has stricter, more specific rules than the repo-level docs because it's the latency- and correctness-critical hot path.

---

## Making a change

### Small fixes / stub implementations
If you're filling in a `NotImplementedError` or fixing a bug in existing logic:
1. Check the module's docstring for the design constraints it's already documenting — don't re-derive the design, follow what's written.
2. Write or update the corresponding unit test in `backend/tests/unit/` using the fakes in `backend/tests/fakes/` (see `TESTING.md`).
3. Update `ARCHITECTURE.md` §6's status table if you've moved something from "Planned" to a real implementation.

### New features
1. Work out which pipeline stage(s) it touches, if any. If it's a new external API integration, it needs a `Protocol` interface (see `CLAUDE.md` constraint #4) before any provider-specific code.
2. If it introduces a new register/style dimension, it's a typed field on `StyleMetadata` (`backend/app/schemas/style_metadata.py`) — update `docs/style-metadata-schema.md` in the same change, not as a follow-up.
3. If it touches the hot path (audio-in to audio-out), explicitly note the latency impact against the budget in `ARCHITECTURE.md` §3 in your PR description / summary. "I didn't think about latency" is not an acceptable state to leave a hot-path change in.
4. Add tests per `TESTING.md` before considering the change done — specifically, cover both the happy path and the fallback/failure path.

### Changing an existing contract
`StyleMetadata`, the `STTProvider`/`TTSProvider` protocols, and the WebSocket wire format between frontend and backend are the three load-bearing interfaces in this repo. Changing any of them means also updating:
- `StyleMetadata` → `docs/style-metadata-schema.md`, and every pipeline stage that constructs or consumes it
- `STTProvider`/`TTSProvider` → every implementation (`AssemblyAISTT`, `ElevenLabsTTS`, `FallbackTTS`, and the fakes)
- WebSocket wire format → both `backend/app/ws/session.py` and `frontend/lib/ws-client.ts`, kept in sync manually since there's no shared schema codegen yet (this is a known gap — see "Known gaps" below)

---

## Commit / PR hygiene

- Keep changes scoped to one pipeline stage or one clear concern where possible — this pipeline's stages are intentionally decoupled behind interfaces, and a PR that touches STT, translation, and TTS all at once is hard to review for latency and correctness regressions independently.
- If a change is a stub-to-real-implementation fill-in, say so explicitly rather than describing it the same way as new functionality — reviewers (human or agent) treat "connects to a live API for the first time" as higher-risk than "adds a helper function."
- Don't commit `.env` files. `.env.example` files should be updated whenever a new environment variable is introduced.

---

## Known gaps (things intentionally left unresolved — don't "fix" without discussion)

- **No shared schema codegen between backend and frontend.** The WebSocket wire format (audio chunks, style-tag JSON messages, transcript JSON messages) is currently kept in sync by hand between `backend/app/ws/session.py` and `frontend/lib/ws-client.ts`. This works at hackathon scale; introducing a schema tool (e.g. generating TS types from the pydantic models) is reasonable future work but is a bigger structural change than most tasks call for — flag it rather than silently adding a codegen dependency.
- **`llm_client` is untyped (`Any`) in `RegisterDetector` and `Translator` constructors.** This is deliberate short-term flexibility while the LLM provider choice is still being finalized (see tech stack table in `README.md` — OpenAI/Anthropic/Groq/Google are all live options). Don't lock this down to one provider's SDK type without confirming which provider is actually being used for the demo.
- **Voice cloning consent flow is not implemented.** `ElevenLabsTTS.clone_voice()` exists as an interface method but there's no UI consent step yet. Don't wire this into the main demo flow without adding one — using someone's cloned voice without explicit consent captured in the flow is a real (not just technical) problem.
- **Translated audio playback is not implemented in the frontend.** `frontend/app/page.tsx`'s `onAudioChunk` callback currently discards every audio chunk it receives (see the `// TODO` there) — the WebSocket connection, mic capture, transcript display, and tone tags are all real and wired end-to-end, but the person speaking cannot yet hear the translated output. This needs a jitter-buffer/scheduling strategy since segments arrive as a burst of chunks rather than a smooth stream (see `ARCHITECTURE.md` §3 on why TTS streams per-clause) — treat this as its own focused change, not a quick addition.

---

## When in doubt

Prefer asking (or, for an agent, stating your assumption explicitly and proceeding with the most conservative interpretation) over guessing on:
- Which LLM provider to hardcode a default to
- Whether a new field belongs in `StyleMetadata` or is out of scope for the register-detection system
- Whether a latency tradeoff is acceptable (the ~2s budget is a hackathon demo target, not a formally negotiated SLA — but treat it as a hard constraint unless told otherwise)
