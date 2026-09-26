# AGENTS.md

This file gives any AI coding agent (Claude, GPT-based agents, Cursor, Copilot Workspace, etc.) the context needed to work correctly in this repository. If you are an AI agent and this is the first file you're reading, also read [README.md](./README.md) and [ARCHITECTURE.md](./ARCHITECTURE.md) before making changes — the constraints below assume that context.

If you are Claude specifically (Claude Code, claude.ai, or the Claude API), see [CLAUDE.md](./CLAUDE.md) for the same information in Claude-specific conventions. This file is the provider-agnostic version.

---

## What Vernacular is

A real-time voice translation system. It does not just translate words — it detects the *register* of speech (sarcasm, urgency, formality, emotion, pacing) and reconstructs that register in the translated output, spoken in a voice that can optionally be a clone of the original speaker.

The pipeline: **audio in → speech-to-text (AssemblyAI) → register detection (LLM) → register-aware translation (LLM) → expressive text-to-speech (ElevenLabs) → audio out**, streamed end-to-end with a ~2 second latency target.

---

## The one thing to internalize before editing anything

**Register is carried as structured data (`StyleMetadata`), and the whole system is a streaming pipeline with a hard latency budget.** Nearly every design constraint in this repo follows from those two facts. If a change you're making would (a) pass style information as free text instead of the typed schema, or (b) require buffering a complete utterance/translation/audio-clip before the next stage can start, stop and reconsider the approach — it likely conflicts with the project's core design.

---

## Required reading map

| If you need to know... | Read |
|---|---|
| What the product does and why | `README.md` |
| How data flows between stages, the latency budget, fallback behavior, what's actually implemented vs. stubbed | `ARCHITECTURE.md` (§6 has current build status) |
| The exact fields in the style metadata contract | `docs/style-metadata-schema.md` and `backend/app/schemas/style_metadata.py` |
| Claude-specific coding conventions for this repo | `CLAUDE.md` |
| Stricter rules specific to the pipeline hot path | `backend/app/pipeline/AGENTS.md` — read this before touching any file in that directory |
| Process/workflow conventions for making a change | `CONTRIBUTING.md` |
| What to test and how (fakes vs. live) | `TESTING.md` |

---

## Hard constraints (apply regardless of which agent/model you are)

1. **Don't break streaming.** Every pipeline stage (`pipeline/stt.py`, `pipeline/register_detector.py`, `pipeline/translator.py`, `pipeline/tts.py`) is designed to consume and produce streams/async iterators, not complete objects. New code in these modules must preserve that.
2. **Don't bypass `StyleMetadata`.** It's the pydantic schema in `backend/app/schemas/style_metadata.py` — the single contract for how register information moves through the system. Any new "style" or "tone" information belongs there as a typed field, not as a comment, a prose string, or a separate ad hoc parameter.
3. **Don't call provider SDKs directly outside `pipeline/*.py`.** STT, translation LLM, and TTS providers are each behind a `Protocol`/interface class. This is what lets the project swap ElevenLabs → Cartesia or AssemblyAI → an alternative without touching orchestration or app logic. Respect the existing boundary; extend it for new providers rather than routing around it.
4. **Respect the latency budget.** See `ARCHITECTURE.md` §3 for the per-stage budget (~2s total). Any change to the hot path (audio-in to audio-out) needs an explicit accounting of its latency impact, not a silent assumption that "it'll probably be fine."
5. **Fail soft, not hard.** Register detection failing should degrade to neutral register, not drop the utterance. A provider failing should trigger the documented fallback (`ARCHITECTURE.md` §5), not crash the session. If you're writing new failure-handling code, follow this pattern — errors in a non-critical stage should never take down the whole pipeline.
6. **Translation + register mapping is atomic.** One LLM call does both the translation and the register-preserving rewrite together, because a literal translation of a sarcastic or formal sentence frequently isn't sarcastic or formal in the target language once translated literally — the rewrite needs full context. Don't split this into "translate, then adjust tone" as two sequential, context-poor steps.

---

## Working conventions

- **Backend** (`backend/`): Python, FastAPI, async-first. Provider clients live in `pipeline/`. Pydantic for all cross-module data (especially `StyleMetadata`). Config via environment variables through `config.py` — never hardcoded secrets.
- **Frontend** (`frontend/`): Next.js App Router, TypeScript, Tailwind CSS.
- **Tests**: pipeline logic should be testable against fake/mock provider implementations (see `backend/tests/fakes/`), not live API calls.
- **Instrumentation**: log stage entry/exit timestamps. This is a latency-sensitive real-time system — timing visibility is a functional requirement, not optional polish.

---

## Before submitting a change, verify

- [ ] Did I preserve streaming behavior in every stage I touched?
- [ ] If I added a new style signal, is it a typed field in `StyleMetadata`, documented in `docs/style-metadata-schema.md`?
- [ ] If I added a new external API call, is it behind a `Protocol` interface in `pipeline/`?
- [ ] Did I account for the latency impact of my change against the ~2s budget?
- [ ] Does my error handling degrade gracefully per `ARCHITECTURE.md` §5, rather than propagating a hard failure?
- [ ] Did I avoid hardcoding language-specific grammar/formality rules as string manipulation, instead relying on the LLM prompt design for generalization?

If any answer is "no" or "not sure," re-read the relevant section of `ARCHITECTURE.md` before proceeding.
