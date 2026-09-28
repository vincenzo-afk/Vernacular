"""
Latency benchmark CLI.

    python -m app.benchmark --dry-run                 # offline, fake providers
    python -m app.benchmark --live --language es      # REAL providers, costs money

Two modes, deliberately separated:

--dry-run   Fake providers with injected delays. Proves the harness and
            the report work. The numbers it prints are the delays we
            injected -- they say NOTHING about real provider latency,
            and the output says so.

--live      Real LLM + ElevenLabs calls for the translation and TTS
            stages, over a fixed set of utterances (below). Requires
            explicit --live and API keys; it will not run by accident,
            because it bills the configured accounts.

Note: --live loads the same Settings as the server, which requires all
three API keys (including AssemblyAI) even though this benchmark never
calls STT. Set a placeholder for ASSEMBLYAI_API_KEY if you only have
the other two.

What --live does NOT cover: STT endpointing (needs a real microphone
stream into AssemblyAI) and network overhead to a browser. Those sit
outside the orchestrator's stages and must be measured in a real
session. The report states this rather than implying a full
end-to-end number.
"""

import argparse
import asyncio
import json
import sys

from app.latency import format_report, summarize
from app.pipeline.orchestrator import SessionOrchestrator
from app.pipeline.register_detector import RegisterDetector
from app.pipeline.stt import TranscriptEvent, WordTiming
from app.pipeline.translator import Translator

# Fixed so runs are comparable over time. Chosen to span the register
# cases the product exists for: sarcasm, urgency, formality, a short
# low-information fragment, and a longer neutral sentence.
UTTERANCES = [
    "Oh, great, another meeting.",
    "Watch out, the car is not stopping!",
    "Good afternoon, I would like to confirm our appointment for Thursday.",
    "Okay.",
    "We reviewed the quarterly numbers and everything looks broadly on track.",
]

# Live runs are fewer because each repeat makes real, billed calls.
DEFAULT_DRY_REPEATS = 3
DEFAULT_LIVE_REPEATS = 2

_DRY_STYLE = {
    "tone": "neutral", "pace": "normal", "formality": "neutral",
    "emotion": "neutral", "sarcasm_score": 0.0, "emphasis_words": [],
    "pause_pattern": "natural", "confidence": 0.9,
}


class _ScriptedSTT:
    def __init__(self, texts: list[str]) -> None:
        self._events = [
            TranscriptEvent(
                text=t,
                is_final=True,
                words=[WordTiming(word=w, start_ms=i * 300, end_ms=i * 300 + 250,
                                  confidence=0.95)
                       for i, w in enumerate(t.split())],
            )
            for t in texts
        ]

    async def stream(self, audio_chunks):
        async for _ in audio_chunks:
            break
        for e in self._events:
            yield e


async def _one_frame():
    yield b"\x00\x00"


class _DelayedLLM:
    """Dry-run only: answers correctly after a fixed delay."""

    def __init__(self, delay_s: float, translated: bool) -> None:
        self._delay = delay_s
        self._translated = translated

    async def complete(self, *, system: str, user: str, **kwargs) -> str:
        await asyncio.sleep(self._delay)
        if self._translated:
            return json.dumps({"translated_text": "hola", "style": _DRY_STYLE})
        return json.dumps(_DRY_STYLE)


class _DelayedTTS:
    def __init__(self, first_byte_s: float) -> None:
        self._d = first_byte_s

    async def synthesize(self, text, style, voice_id):
        await asyncio.sleep(self._d)
        yield b"\x00\x01" * 100


async def run_dry(n_repeats: int = DEFAULT_DRY_REPEATS) -> SessionOrchestrator:
    texts = UTTERANCES * n_repeats
    return await _drive(
        SessionOrchestrator(
            stt=_ScriptedSTT(texts),
            register_detector=RegisterDetector(llm_client=_DelayedLLM(0.05, False)),
            translator=Translator(llm_client=_DelayedLLM(0.15, True)),
            tts=_DelayedTTS(0.10),
            target_language="es",
            voice_id="dry-run",
        )
    )


async def run_live(
    language: str, n_repeats: int = DEFAULT_LIVE_REPEATS
) -> SessionOrchestrator:
    # Imported here so --dry-run never touches settings or provider SDKs.
    from app.config import get_settings
    from app.pipeline.tts import ElevenLabsTTS
    from app.ws.session import _build_llm_client

    settings = get_settings()
    if not settings.benchmark_voice_id:
        raise RuntimeError(
            "BENCHMARK_VOICE_ID is not set. Set it to an ElevenLabs voice "
            "ID in .env; the live benchmark won't guess a voice."
        )
    llm = _build_llm_client(settings)
    return await _drive(
        SessionOrchestrator(
            stt=_ScriptedSTT(UTTERANCES * n_repeats),
            register_detector=RegisterDetector(llm_client=llm),
            translator=Translator(llm_client=llm),
            tts=ElevenLabsTTS(api_key=settings.elevenlabs_api_key),
            target_language=language,
            voice_id=settings.benchmark_voice_id,
        )
    )


async def _drive(orch: SessionOrchestrator) -> SessionOrchestrator:
    async for _ in orch.run_stream(_one_frame()):
        pass
    return orch


DRY_BANNER = (
    "DRY RUN — fake providers with injected delays. These numbers are the\n"
    "delays we injected, NOT measurements of any real service. They only\n"
    "show that the harness and report work.\n"
)
LIVE_FOOTER = (
    "\nNot measured here: STT endpointing (needs a live AssemblyAI audio\n"
    "stream) and network overhead to the browser. Add them from a real\n"
    "session before comparing to the 2000ms end-to-end budget.\n"
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m app.benchmark")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="offline, fake providers")
    mode.add_argument("--live", action="store_true",
                      help="REAL provider calls; bills your accounts")
    ap.add_argument("--language", default="es")
    ap.add_argument(
        "--repeats", type=int, default=None,
        help="passes over the fixed utterance set (default: "
             f"{DEFAULT_DRY_REPEATS} dry, {DEFAULT_LIVE_REPEATS} live; "
             "live repeats are billed)",
    )
    args = ap.parse_args(argv)
    if args.repeats is not None and args.repeats < 1:
        ap.error("--repeats must be >= 1")

    if args.dry_run:
        orch = asyncio.run(run_dry(args.repeats or DEFAULT_DRY_REPEATS))
        print(DRY_BANNER)
        print(format_report(summarize(orch.timings)))
        return 0

    try:
        orch = asyncio.run(
            run_live(args.language, args.repeats or DEFAULT_LIVE_REPEATS)
        )
    # A CLI's outermost boundary is the one place a broad except is right:
    # missing config, absent SDK, bad key and network errors are an open
    # set, and every one should become a readable message + exit code 1
    # rather than a traceback.
    except Exception as exc:  # noqa: BLE001
        print(f"live benchmark failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(format_report(summarize(orch.timings)))
    print(LIVE_FOOTER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
