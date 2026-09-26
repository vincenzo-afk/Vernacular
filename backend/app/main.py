"""
FastAPI entrypoint. See README.md Quick Start for how to run this.
"""

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from app.ws.session import handle_session

app = FastAPI(title="Vernacular")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten before any real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.websocket("/ws/session")
async def ws_session(websocket: WebSocket, target_language: str = "es") -> None:
    await handle_session(websocket, target_language)
