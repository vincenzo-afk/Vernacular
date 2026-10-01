"""
Orchestrator behaviours added with live captions, conversation memory,
context-aware detection, multimodal audio and per-segment timings.

The headline property: captions stay LIVE while a previous segment is
still being translated/spoken. The old design processed segments inline
inside the STT loop, so a partial transcript for the next utterance sat
unread until the previous segment's audio had finished. The gated-TTS
test below deadlocks (times out) against that design.
"""

import asyncio
import json

import pytest

from app.pipeline.orchestrator import (
    AudioChunk,
    CaptionEvent,
    SegmentEnd,
    SegmentStart,
    SessionOrchestrator,
)
from app.pipeline.prosody import UtteranceAudio
from app.pipeline.register_detector import RegisterDetector
from app.pipeline.stt import TranscriptEvent
from app.pipeline.translator import Translator
from app.schemas.style_metadata import AcousticFeatures, Arousal, Tone
from tests.fakes.fake_llm import FakeLLMClient, ScriptedResponse

SARCASTIC = {
    "tone": "sarcastic", "pace": "slow", "formality": "casual",
    "emotion": "frustration", "sarcasm_score": 0.9, "emphasis_words": [],
    "pause_pattern": "dramatic", "confidence": 0.9,
}
NEUTRAL = {**SARCASTIC, "tone": "neutral", "emotion": "neutral", "sarcasm_score": 0.0}
TRANSLATION = json.dumps({"translated_text": "Ay, qué bien.", "style": SARCASTIC})


def reg(style=SARCASTIC) -> ScriptedResponse:
    return ScriptedResponse(json.dumps(style))


class ScriptedSTT:
    """Yields items in order. An asyncio.Event item means 'wait for it'."""

    def __init__(self, items, consume_audio=False):
        self._items = items
        self._consume = consume_audio
        self.audio_bytes = 0

    async def stream(self, audio_chunks):
        if self._consume:
            async for chunk in audio_chunks:
                self.audio_bytes += len(chunk)
        for item in self._items:
            if isinstance(item, asyncio.Event):
                await item.wait()
            elif isinstance(item, Exception):
                raise item
            else:
                yield item


class GatedTTS:
    def __init__(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def synthesize(self, text, style, voice_id):
        self.started.set()
        yield b"AA"
        await self.release.wait()
        yield b"BB"


class QuickTTS:
    async def synthesize(self, text, style, voice_id):
        yield b"AA"


async def _audio():
    yield b"\x00\x00"


def final(text):
    return TranscriptEvent(text=text, is_final=True)


def partial(text):
    return TranscriptEvent(text=text, is_final=False)


def build(stt, tts, *, register, translate, **kw):
    return SessionOrchestrator(
        stt=stt,
        register_detector=RegisterDetector(FakeLLMClient(register), explain=False),
        translator=Translator(FakeLLMClient(translate)),
        tts=tts,
        target_language="es",
        voice_id="v",
        **kw,
    )


async def collect(orch):
    return [e async for e in orch.run_stream(_audio())]


# --------------------------------------------------------- live captions


@pytest.mark.asyncio
async def test_captions_keep_flowing_while_the_previous_segment_is_still_speaking():
    tts = GatedTTS()
    stt = ScriptedSTT([final("first."), tts.started, partial("second utt"), final("second utterance.")])
    orch = build(
        stt, tts, emit_captions=True,
        register=[reg(), reg(), reg()], translate=[ScriptedResponse(TRANSLATION)] * 2,
    )
    seen: list = []

    async def consume():
        async for ev in orch.run_stream(_audio()):
            seen.append(ev)

    task = asyncio.create_task(consume())
    try:
        # TTS is BLOCKED (release never set). The partial caption for the
        # next utterance must still arrive.
        for _ in range(100):
            await asyncio.sleep(0.01)
            if any(isinstance(e, CaptionEvent) and not e.is_final and e.text == "second utt" for e in seen):
                break
        else:
            pytest.fail("partial caption never arrived while TTS was blocked")
        assert not tts.release.is_set()
        assert any(isinstance(e, AudioChunk) for e in seen)  # first clip audio did stream
    finally:
        tts.release.set()
        await asyncio.wait_for(task, timeout=2)


@pytest.mark.asyncio
async def test_final_caption_precedes_its_segment_and_shares_its_id():
    orch = build(
        ScriptedSTT([partial("Oh"), final("Oh, great.")]), QuickTTS(), emit_captions=True,
        register=[reg(), reg()], translate=[ScriptedResponse(TRANSLATION)],
    )
    events = await collect(orch)
    kinds = [type(e).__name__ for e in events]
    assert kinds[0] == "CaptionEvent" and kinds.index("CaptionEvent") < kinds.index("SegmentStart")
    final_caption = next(e for e in events if isinstance(e, CaptionEvent) and e.is_final)
    start = next(e for e in events if isinstance(e, SegmentStart))
    assert final_caption.segment_id == start.segment_id == 1
    assert final_caption.text == "Oh, great." and not events[0].is_final


@pytest.mark.asyncio
async def test_no_captions_unless_asked_so_the_event_stream_is_unchanged():
    orch = build(
        ScriptedSTT([partial("Oh"), final("Oh, great.")]), QuickTTS(),
        register=[reg(), reg()], translate=[ScriptedResponse(TRANSLATION)],
    )
    kinds = [type(e).__name__ for e in await collect(orch)]
    assert kinds == ["SegmentStart", "AudioChunk", "SegmentEnd"]


# ------------------------------------------- memory / context / key moments


@pytest.mark.asyncio
async def test_second_utterance_is_detected_and_translated_in_context():
    detect_llm = FakeLLMClient([reg(NEUTRAL), reg()])
    translate_llm = FakeLLMClient([ScriptedResponse(TRANSLATION)] * 2)
    orch = SessionOrchestrator(
        stt=ScriptedSTT([final("The build failed again."), final("Oh, great.")]),
        register_detector=RegisterDetector(detect_llm, explain=False),
        translator=Translator(translate_llm),
        tts=QuickTTS(), target_language="es", voice_id="v",
    )
    await collect(orch)

    second_detect = detect_llm.calls[1]["user"].split("Now classify this segment:")[1]
    assert "The build failed again." in second_detect          # earlier words
    assert "neutral/neutral" in second_detect                   # earlier tone
    first_detect = detect_llm.calls[0]["user"].split("Now classify this segment:")[1]
    assert "Earlier turns" not in first_detect                  # nothing to see yet

    assert "The build failed again." in translate_llm.calls[1]["user"]
    assert '-> "Ay, qué bien."' in translate_llm.calls[1]["user"]
    assert "Earlier turns" not in translate_llm.calls[0]["user"]


@pytest.mark.asyncio
async def test_context_can_be_disabled():
    detect_llm = FakeLLMClient([reg(NEUTRAL), reg()])
    orch = SessionOrchestrator(
        stt=ScriptedSTT([final("The build failed again."), final("Oh, great.")]),
        register_detector=RegisterDetector(detect_llm, explain=False),
        translator=Translator(FakeLLMClient([ScriptedResponse(TRANSLATION)] * 2)),
        tts=QuickTTS(), target_language="es", voice_id="v", use_context=False,
    )
    await collect(orch)
    assert "Earlier turns" not in detect_llm.calls[1]["user"].split("Now classify this segment:")[1]


@pytest.mark.asyncio
async def test_memory_records_turns_with_style_and_key_moment():
    orch = build(
        ScriptedSTT([final("Oh, great.")]), QuickTTS(),
        register=[reg()], translate=[ScriptedResponse(TRANSLATION)],
    )
    events = await collect(orch)
    start = next(e for e in events if isinstance(e, SegmentStart))
    turn = orch.memory.get(1)
    assert turn.source_text == "Oh, great." and turn.translated_text == "Ay, qué bien."
    assert turn.style.tone == Tone.SARCASTIC          # the SOURCE reading
    assert start.key_moment.is_key and turn.key_moment.is_key
    assert start.segment_id == 1


@pytest.mark.asyncio
async def test_key_moment_uses_the_previous_turn_for_shift_detection():
    warm = {**SARCASTIC, "tone": "warm", "emotion": "affection", "sarcasm_score": 0.0}
    annoyed = {**SARCASTIC, "tone": "annoyed", "emotion": "annoyance", "sarcasm_score": 0.0}
    orch = build(
        ScriptedSTT([final("Thanks so much."), final("Why is it late?")]), QuickTTS(),
        register=[reg(warm), reg(annoyed)], translate=[ScriptedResponse(TRANSLATION)] * 2,
    )
    starts = [e for e in await collect(orch) if isinstance(e, SegmentStart)]
    assert "emotion_shift" in [r.value for r in starts[1].key_moment.reasons]
    assert "emotion_shift" not in [r.value for r in starts[0].key_moment.reasons]


# ----------------------------------------------------------- timings


@pytest.mark.asyncio
async def test_segment_timings_break_down_the_critical_path():
    orch = build(
        ScriptedSTT([final("Oh, great.")]), QuickTTS(),
        register=[reg()], translate=[ScriptedResponse(TRANSLATION)],
    )
    t = next(e for e in await collect(orch) if isinstance(e, SegmentStart)).timings
    assert t.translation_ms >= 0 and t.tts_first_byte_ms is not None
    parts = t.queue_wait_ms + t.style_wait_ms + t.translation_ms + t.tts_first_byte_ms
    assert t.server_total_ms >= parts - 1.0          # total covers every stage
    assert t.server_total_ms - parts < 250           # ...with only scheduling slack


@pytest.mark.asyncio
async def test_a_backed_up_pipeline_shows_up_as_queue_wait():
    tts = GatedTTS()
    orch = build(
        ScriptedSTT([final("one."), final("two.")]), tts,
        register=[reg(), reg()], translate=[ScriptedResponse(TRANSLATION)] * 2,
    )
    starts: list = []

    async def consume():
        async for ev in orch.run_stream(_audio()):
            if isinstance(ev, SegmentStart):
                starts.append(ev)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.15)  # segment 1 is mid-clip; segment 2 waits behind it
    tts.release.set()
    await asyncio.wait_for(task, timeout=2)
    assert starts[0].timings.queue_wait_ms < 50
    assert starts[1].timings.queue_wait_ms >= 100


# ------------------------------------------------- multimodal audio path


FEATURES = AcousticFeatures(
    arousal_score=0.1, energy_delta_db=-8.0, pitch_range_hz=6.0, duration_ms=1500
)


class FakeProsody:
    def __init__(self):
        self.fed = 0

    def feed(self, chunk):
        self.fed += len(chunk)

    def snapshot(self):
        return FEATURES

    def finish_utterance(self):
        return UtteranceAudio(FEATURES, 250, 1750)


@pytest.mark.asyncio
async def test_measured_voice_reaches_the_llm_the_segment_and_the_timing_fields():
    detect_llm = FakeLLMClient([reg()])
    prosody = FakeProsody()
    warm = {**SARCASTIC, "tone": "warm", "emotion": "positive", "sarcasm_score": 0.1}
    detect_llm = FakeLLMClient([reg(warm)])
    orch = SessionOrchestrator(
        stt=ScriptedSTT([final("I love it.")], consume_audio=True),
        register_detector=RegisterDetector(detect_llm, explain=False),
        translator=Translator(FakeLLMClient([ScriptedResponse(TRANSLATION)])),
        tts=QuickTTS(), target_language="es", voice_id="v", prosody=prosody,
    )
    start = next(e for e in await collect(orch) if isinstance(e, SegmentStart))
    assert "Acoustic: arousal 0.10 (low)" in detect_llm.calls[0]["user"]
    assert start.style.acoustic == FEATURES and start.style.arousal == Arousal.LOW
    assert start.style.modality_conflict is True       # warm words, quiet voice
    assert (start.speech_start_ms, start.speech_end_ms) == (250, 1750)
    assert prosody.fed == 2                            # the audio was tapped


class ExplodingProsody(FakeProsody):
    def feed(self, chunk):
        raise RuntimeError("analyzer bug")

    def snapshot(self):
        raise RuntimeError("analyzer bug")

    def finish_utterance(self):
        raise RuntimeError("analyzer bug")


@pytest.mark.asyncio
async def test_a_broken_voice_analyzer_never_interrupts_audio_or_translation():
    stt = ScriptedSTT([partial("Oh"), final("Oh, great.")], consume_audio=True)
    orch = build(
        stt, QuickTTS(), prosody=ExplodingProsody(),
        register=[reg(), reg()], translate=[ScriptedResponse(TRANSLATION)],
    )
    events = await collect(orch)
    assert stt.audio_bytes == 2                        # mic audio still reached STT
    start = next(e for e in events if isinstance(e, SegmentStart))
    assert start.translated_text == "Ay, qué bien." and start.style.acoustic is None


# --------------------------------------------------------- failure paths


@pytest.mark.asyncio
async def test_stt_failure_is_raised_after_already_queued_segments_are_delivered():
    orch = build(
        ScriptedSTT([final("Oh, great."), RuntimeError("stt died")]), QuickTTS(),
        register=[reg()], translate=[ScriptedResponse(TRANSLATION)],
    )
    got = []
    with pytest.raises(RuntimeError, match="stt died"):
        async for ev in orch.run_stream(_audio()):
            got.append(ev)
    assert any(isinstance(e, SegmentStart) for e in got)
    assert any(isinstance(e, SegmentEnd) for e in got)


@pytest.mark.asyncio
async def test_closing_the_stream_early_cancels_background_tasks():
    tts = GatedTTS()
    orch = build(
        ScriptedSTT([final("one.")]), tts,
        register=[reg()], translate=[ScriptedResponse(TRANSLATION)],
    )
    before = len(asyncio.all_tasks())
    stream = orch.run_stream(_audio())
    async for ev in stream:
        if isinstance(ev, AudioChunk):
            break
    await stream.aclose()
    await asyncio.sleep(0.05)
    assert len(asyncio.all_tasks()) <= before
