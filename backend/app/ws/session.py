"""
WebSocket session handling — one connection per translation session.

Receives raw audio chunks from the frontend, feeds them into a
SessionOrchestrator, and streams back translated audio plus live
StyleMetadata tags (for the frontend's tone-tag UI — see
frontend/components/ToneTags.tsx and frontend/lib/ws-client.ts, which
this wire format must stay in sync with — see CONTRIBUTING.md "Known
gaps" on why that sync is currently manual).

Wire format (backend -> frontend), one JSON text message per completed
segment, followed by binary audio-chunk messages for that segment:

    {"type": "segment", "source_text": "...", "translated_text": "...",
     "style": {...StyleMetadata fields...}}
    <binary audio chunk>
    <binary audio chunk>
    ...
"""

import json
import logging

from fastapi import WebSocket, WebSocketDisconnect

from app.config import settings
from app.pipeline.orchestrator import SessionOrchestrator
from app.pipeline.register_detector import RegisterDetector
from app.pipeline.stt import AssemblyAISTT
from app.pipeline.translator import Translator
from app.pipeline.tts import ElevenLabsTTS, FallbackTTS

logger = logging.getLogger("vernacular.ws.session")


def _build_llm_client():
    """
    Constructs the LLM client used by both RegisterDetector and
    Translator, based on settings.llm_provider. Kept here rather than
    in config.py so pipeline modules stay decoupled from any one
    provider SDK — see CLAUDE.md constraint #4.
    """
    if settings.llm_provider == "openai":
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=settings.llm_api_key)

        class _OpenAIAdapter:
            async def complete(self, *, system: str, user: str, **kwargs) -> str:
                response = await client.chat.completions.create(
                    model=settings.llm_model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=kwargs.get("temperature", 0.3),
                    max_tokens=kwargs.get("max_tokens", 500),
                )
                return response.choices[0].message.content or ""

        return _OpenAIAdapter()

    if settings.llm_provider == "anthropic":
        from anthropic import AsyncAnthropic

        client = AsyncAnthropic(api_key=settings.llm_api_key)

        class _AnthropicAdapter:
            async def complete(self, *, system: str, user: str, **kwargs) -> str:
                response = await client.messages.create(
                    model=settings.llm_model,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                    temperature=kwargs.get("temperature", 0.3),
                    max_tokens=kwargs.get("max_tokens", 500),
                )
                return "".join(
                    block.text for block in response.content if block.type == "text"
                )

        return _AnthropicAdapter()

    raise NotImplementedError(
        f"LLM provider '{settings.llm_provider}' not wired yet — add an "
        "adapter above following the _OpenAIAdapter / _AnthropicAdapter "
        "pattern. Groq and Google are listed as options in README.md but "
        "not yet implemented."
    )


async def handle_session(websocket: WebSocket, target_language: str) -> None:
    await websocket.accept()

    llm_client = _build_llm_client()
    stt = AssemblyAISTT(api_key=settings.assemblyai_api_key)
    register_detector = RegisterDetector(llm_client=llm_client)
    translator = Translator(llm_client=llm_client)

    tts = ElevenLabsTTS(api_key=settings.elevenlabs_api_key)
    fallback_tts = None
    if settings.fallback_tts_api_key:
        fallback_tts = FallbackTTS(
            provider=settings.fallback_tts_provider,
            api_key=settings.fallback_tts_api_key,
        )

    orchestrator = SessionOrchestrator(
        stt=stt,
        register_detector=register_detector,
        translator=translator,
        tts=tts,
        target_language=target_language,
        voice_id="default",  # TODO: voice cloning calibration step, gated
        # on explicit consent — see CONTRIBUTING.md "Known gaps"
    )

    async def audio_in():
        while True:
            chunk = await websocket.receive_bytes()
            yield chunk

    try:
        async for segment in orchestrator.run(audio_in()):
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "segment",
                        "source_text": segment.source_text,
                        "translated_text": segment.translated_text,
                        "style": segment.style.model_dump(mode="json"),
                    }
                )
            )
            for chunk in segment.audio_chunks:
                await websocket.send_bytes(chunk)
    except WebSocketDisconnect:
        logger.info("ws.session: client disconnected")
    except Exception:
        logger.error(
            "ws.session: unhandled error, closing session", exc_info=True
        )
        # Per pipeline/AGENTS.md, individual segment failures are
        # already caught inside the orchestrator and skipped rather
        # than propagating. Reaching here means something outside that
        # contract broke (e.g. the STT stream itself) — surface it to
        # the client rather than hanging.
        try:
            await websocket.send_text(
                json.dumps({"type": "error", "message": "session_failed"})
            )
        except Exception:
            pass
        finally:
            if fallback_tts is not None:
                logger.info(
                    "ws.session: fallback_tts was configured but not yet "
                    "wired into orchestrator provider-switch logic — see "
                    "pipeline/AGENTS.md failure-handling contract for "
                    "tts.py, this is a known gap to close before relying "
                    "on it in a live demo."
                )
