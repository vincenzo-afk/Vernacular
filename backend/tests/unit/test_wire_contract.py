"""
Backend <-> frontend wire-contract drift guard.

The WebSocket message shape is maintained by hand on both sides (see
CONTRIBUTING.md "Known gaps": no schema codegen). This test makes the
manual sync enforceable: it serializes a real SegmentOutput through
segment_to_wire() and asserts that its keys exactly match the fields
declared on `SegmentMessage` / `StyleTag` in frontend/lib/ws-client.ts.
Add a field on one side only and this fails.
"""

import re
from pathlib import Path

from app.pipeline.orchestrator import SegmentOutput
from app.schemas.style_metadata import StyleMetadata
from app.ws.session import ready_to_wire, segment_to_wire

TS_FILE = Path(__file__).resolve().parents[3] / "frontend" / "lib" / "ws-client.ts"


def _ts_interface_fields(name: str) -> set[str]:
    src = TS_FILE.read_text(encoding="utf-8")
    body = re.search(rf"export interface {name} \{{(.*?)\n\}}", src, re.DOTALL)
    assert body, f"interface {name} not found in ws-client.ts"
    # Strip comments so words inside docs/JSDoc aren't read as fields.
    text = re.sub(r"/\*.*?\*/", "", body.group(1), flags=re.DOTALL)
    text = re.sub(r"//.*", "", text)
    return set(re.findall(r"^\s*(\w+)\??:", text, re.MULTILINE))


def _wire() -> dict:
    return segment_to_wire(
        SegmentOutput(
            audio_chunks=[b"x"],
            source_text="hi",
            translated_text="hola",
            style=StyleMetadata(),
            degraded=True,
        )
    )


def test_segment_message_keys_match_frontend_interface():
    assert set(_wire().keys()) == _ts_interface_fields("SegmentMessage")


def test_style_keys_match_frontend_stytag_interface():
    assert set(_wire()["style"].keys()) == _ts_interface_fields("StyleTag")


def test_wire_values_are_json_serializable_and_typed():
    import json

    wire = json.loads(json.dumps(_wire()))
    assert wire["type"] == "segment"
    assert wire["degraded"] is True
    assert isinstance(wire["style"]["sarcasm_score"], float)


def test_ready_message_keys_match_frontend_interface():
    wire = ready_to_wire()
    assert set(wire.keys()) == _ts_interface_fields("ReadyMessage")
    assert set(wire["audio"].keys()) == _ts_interface_fields("AudioFormat")


def test_ready_message_declares_the_format_the_tts_layer_actually_emits():
    from app.pipeline import tts

    audio = ready_to_wire()["audio"]
    assert audio["encoding"] == tts.AUDIO_ENCODING == "pcm_s16le"
    assert audio["sample_rate"] == tts.AUDIO_SAMPLE_RATE
    assert audio["channels"] == tts.AUDIO_CHANNELS == 1
