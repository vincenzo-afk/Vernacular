"""
WebSocket session handling — one connection per translation session.

Receives raw audio chunks from the frontend, feeds them into a
SessionOrchestrator, and streams back translated audio plus live
StyleMetadata tags (for the frontend's tone-tag UI — see
frontend/components/ToneTags.tsx).
"""

from fastapi import WebSocket

from app.config import settings
from app.pipeline.orchestrator import SessionOrchestrator
from app.pipeline.register_detector import RegisterDetector
from app.pipeline.stt import AssemblyAISTT
from app.pipeline.translator import Translator
from app.pipeline.tts import ElevenLabsTTS


async def handle_session(websocket: WebSocket, target_language: str) -> None:
    await websocket.accept()

    stt = AssemblyAISTT(api_key=settings.assemblyai_api_key)
    register_detector = RegisterDetector(llm_client=None)  # TODO: wire real client
    translator = Translator(llm_client=None)  # TODO: wire real client
    tts = ElevenLabsTTS(api_key=settings.elevenlabs_api_key)

    orchestrator = SessionOrchestrator(
        stt=stt,
        register_detector=register_detector,
        translator=translator,
        tts=tts,
        target_language=target_language,
        voice_id="default",  # TODO: voice cloning calibration step
    )

    async def audio_in():
        while True:
            chunk = await websocket.receive_bytes()
            yield chunk

    async for audio_out in orchestrator.run(audio_in()):
        await websocket.send_bytes(audio_out)
