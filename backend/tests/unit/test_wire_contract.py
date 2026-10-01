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

from app.adaptive import AdaptiveController
from app.pipeline.orchestrator import CaptionEvent, SegmentOutput
from app.pipeline.summarizer import ConversationSummary
from app.schemas.style_metadata import (
    AcousticFeatures,
    Cue,
    CueKind,
    StyleExplanation,
    StyleMetadata,
)
from app.ws.session import (
    caption_to_wire,
    network_to_wire,
    ready_to_wire,
    segment_to_wire,
    summary_to_wire,
)

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


# ----------------------------------------------------------------------
# Messages and nested types added with captions / multimodal / explain /
# key moments / latency HUD / adaptive streaming / summary. Each backend
# payload is built with EVERY optional field populated (a None-valued
# nested object would hide a missing key), then compared to the TS
# interface. Add a field on one side only and these fail.
# ----------------------------------------------------------------------


def _full_style() -> StyleMetadata:
    return StyleMetadata(
        acoustic=AcousticFeatures(pitch_mean_hz=150.0, pitch_delta_pct=5.0, pitch_range_hz=30.0),
        explanation=StyleExplanation(
            summary="why", cues=[Cue(kind=CueKind.LEXICAL, evidence="great")]
        ),
    )


def _full_wire() -> dict:
    return segment_to_wire(
        SegmentOutput(
            audio_chunks=[], source_text="a", translated_text="b",
            style=_full_style(), segment_id=3,
        )
    )


def test_full_style_keys_match_frontend_including_new_fields():
    assert set(_full_wire()["style"].keys()) == _ts_interface_fields("StyleTag")


def test_acoustic_keys_match_frontend():
    assert set(_full_wire()["style"]["acoustic"].keys()) == _ts_interface_fields("AcousticTag")


def test_explanation_and_cue_keys_match_frontend():
    explanation = _full_wire()["style"]["explanation"]
    assert set(explanation.keys()) == _ts_interface_fields("ExplanationTag")
    assert set(explanation["cues"][0].keys()) == _ts_interface_fields("CueTag")


def test_key_moment_and_timings_keys_match_frontend():
    wire = _full_wire()
    assert set(wire["key_moment"].keys()) == _ts_interface_fields("KeyMomentTag")
    assert set(wire["timings"].keys()) == _ts_interface_fields("TimingsTag")


def test_cue_kind_values_match_the_frontend_union():
    src = TS_FILE.read_text(encoding="utf-8")
    union = re.search(r"export type CueKind =(.*?);", src, re.DOTALL).group(1)
    assert set(re.findall(r'"(\w+)"', union)) == {k.value for k in CueKind}


def test_caption_message_keys_match_frontend():
    wire = caption_to_wire(CaptionEvent(text="hi", is_final=True, segment_id=1))
    assert set(wire.keys()) == _ts_interface_fields("CaptionMessage")


def test_network_message_keys_match_frontend():
    c = AdaptiveController()
    c.report_client(__import__("app.adaptive", fromlist=["NetworkTier"]).NetworkTier.FAIR, 120.0)
    c.observe_send(10)
    assert set(network_to_wire(c).keys()) == _ts_interface_fields("NetworkMessage")


def test_summary_message_keys_match_frontend():
    wire = summary_to_wire(ConversationSummary(overview="o"))
    assert set(wire.keys()) == _ts_interface_fields("SummaryMessage")


def test_pong_message_keys_match_frontend():
    assert set({"type": "pong", "id": "p1"}.keys()) == _ts_interface_fields("PongMessage")


def test_every_backend_message_type_is_recognised_by_the_frontend():
    src = TS_FILE.read_text(encoding="utf-8")
    known = set(re.findall(r'"(\w+)",', re.search(r"CONTROL_TYPES = new Set\(\[(.*?)\]\)", src, re.DOTALL).group(1)))
    for message in (
        ready_to_wire(), _full_wire(),
        caption_to_wire(CaptionEvent(text="", is_final=False)),
        network_to_wire(AdaptiveController()),
        summary_to_wire(ConversationSummary(overview="o")),
        {"type": "pong", "id": None}, {"type": "error", "message": "x"},
    ):
        assert message["type"] in known
