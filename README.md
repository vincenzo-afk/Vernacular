# Vernacular

**Real-time voice translation that preserves *how* you speak, not just *what* you say.**

Vernacular is a real-time voice translation system that goes beyond literal word-for-word translation. It captures the speaker's *register* — sarcasm, urgency, formality, warmth, hesitation — and reconstructs that same register in the target language, spoken in the target voice, with matching rhythm and emphasis.

If you say something sarcastic in English, the Spanish output should sound sarcastic too. If you're speaking fast and clipped because you're annoyed, the translated audio should carry that same tension — not a flat, robotic reading of the translated words.

---

## Why this exists

Most real-time translation tools solve half the problem: they get the words right and throw away everything else. Tone, pacing, emphasis, sarcasm, hesitation — all of it gets flattened into a neutral TTS voice. The result is technically correct and emotionally wrong. A joke lands as a statement. An urgent warning sounds like a weather report. A biting remark sounds polite.

Vernacular treats **register as a first-class signal**, not an afterthought. The pipeline explicitly detects paralinguistic and pragmatic cues from the source audio, maps them into a structured style representation, and uses that representation to drive expressive, style-controlled speech synthesis in the target language.

---

## How it works (high level)

```
Microphone
    │
    ▼
┌─────────────────────────┐
│ AssemblyAI Realtime STT │  → partial + final transcripts
│ + Sentiment Analysis    │  → word-level timestamps, pacing data
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│ Register Detection      │  → LLM classifies tone/pace/formality/
│ (LLM, our prompt)        │    emotion/sarcasm from transcript +
└─────────────────────────┘    prosody signals
    │
    ▼
┌─────────────────────────┐
│ Style Metadata           │  → {tone, pace, formality, emotion,
│ (structured JSON)         │    sarcasm_score, emphasis_words}
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│ Translation + Register  │  → LLM translates AND rewrites the
│ Mapping (LLM)            │    translation to preserve register in
└─────────────────────────┘    the target language's idiom
    │
    ▼
┌─────────────────────────┐
│ Expressive TTS            │  → ElevenLabs Multilingual v2,
│ (ElevenLabs)               │    style-controlled, optionally
└─────────────────────────┘    voice-cloned to the original speaker
    │
    ▼
Speaker output (translated audio in ~2s end-to-end)
```

See [ARCHITECTURE.md](./ARCHITECTURE.md) for the full data flow, module boundaries, and latency budget breakdown.

---

## Tech stack

### Voice AI core

| Component | Tool |
|---|---|
| Real-time speech-to-text | AssemblyAI Realtime STT API |
| Sentiment/emotion analysis | AssemblyAI Sentiment Analysis |
| Word-level timestamps / pacing data | AssemblyAI |
| Translation + register mapping | LLM API (GPT-4o / Claude / Groq / Gemini) |
| Expressive TTS with style control | ElevenLabs Multilingual v2 (fallback: Cartesia / OpenAI TTS) |
| Voice cloning | ElevenLabs Voice Cloning |

### App layer

| Component | Tool |
|---|---|
| Real-time audio streaming | LiveKit or plain WebSockets |
| Backend / pipeline orchestration | FastAPI (Python) |
| Frontend | Next.js + Tailwind CSS |
| Demo UI extras | Live transcript, waveform viz, real-time tone tags |

### The custom part (what's actually ours)

1. **Register-detection system prompt + few-shots** — detects sarcasm/formality/urgency/emotion from transcript + prosody and emits structured style metadata.
2. **Style metadata schema** — the JSON contract between detection and synthesis: `{tone, pace, formality, emotion, sarcasm_score}`.
3. **Latency pipeline** — parallel partial-transcript processing engineered to hit a ~2s end-to-end budget.

---

## Repository layout

```
vernacular/
├── README.md              ← you are here
├── CLAUDE.md               ← instructions for Claude Code / AI coding agents
├── AGENTS.md                ← instructions for any AI agent working on this repo
├── ARCHITECTURE.md          ← system design, data flow, module contracts
├── backend/
│   ├── app/
│   │   ├── main.py                    # FastAPI app entrypoint
│   │   ├── pipeline/
│   │   │   ├── stt.py                 # AssemblyAI Realtime STT client
│   │   │   ├── register_detector.py   # LLM-based register/style detection
│   │   │   ├── translator.py          # LLM translation + register mapping
│   │   │   ├── tts.py                 # ElevenLabs TTS client
│   │   │   └── orchestrator.py        # Ties the pipeline stages together
│   │   ├── schemas/
│   │   │   └── style_metadata.py      # StyleMetadata pydantic model
│   │   ├── ws/
│   │   │   └── session.py             # WebSocket session handling
│   │   └── config.py                  # Settings / env var loading
│   ├── tests/
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── app/
│   │   ├── page.tsx                   # Main demo UI
│   │   └── layout.tsx
│   ├── components/
│   │   ├── Waveform.tsx
│   │   ├── LiveTranscript.tsx
│   │   └── ToneTags.tsx
│   ├── lib/
│   │   └── ws-client.ts
│   ├── package.json
│   └── .env.example
└── docs/
    └── style-metadata-schema.md      # Full schema reference + examples
```

---

## Quick start

### Prerequisites

- Python 3.11+
- Node.js 18+
- API keys for: AssemblyAI, an LLM provider (OpenAI/Anthropic/Groq/Google), ElevenLabs

### Backend

```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in API keys
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
cp .env.example .env.local   # set NEXT_PUBLIC_WS_URL=ws://localhost:8000
npm run dev
```

Open `http://localhost:3000`, grant mic access, pick a target language, and speak.

---

## Design principles

These hold across every module in this repo — see [CLAUDE.md](./CLAUDE.md) and [AGENTS.md](./AGENTS.md) for the enforcement version of these rules.

1. **Register is data, not vibes.** Every stage passes structured `StyleMetadata`, not prose descriptions. This makes the pipeline testable and lets TTS style controls bind directly to detected signals.
2. **Latency is a hard constraint, not an optimization target.** The ~2s end-to-end budget is a design input from day one, not something to fix after the fact. Partial transcripts are processed as they stream in; stages run in parallel wherever the data dependency allows it.
3. **Translation and register-mapping happen together, not in sequence.** A literal translation of a sarcastic sentence is often not sarcastic in the target language. The LLM step must rewrite for equivalent effect, not just equivalent meaning.
4. **Degrade gracefully.** If sentiment/register detection fails or times out, fall back to neutral-register translation rather than blocking the pipeline.
5. **Every external API call is behind an interface.** STT, LLM, and TTS providers are swappable (see fallbacks in the tech stack table) — no provider-specific code outside the client modules in `pipeline/`.

---

## Status

Early-stage / hackathon project. See [ARCHITECTURE.md](./ARCHITECTURE.md) for what's built vs. planned.
