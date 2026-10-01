"""Conversation memory (bounded, two-phase turns) and key-moment detection."""

from app.conversation import ConversationMemory
from app.insights import KeyMomentReason, detect_key_moment
from app.schemas.style_metadata import (
    AcousticFeatures,
    Arousal,
    Emotion,
    StyleMetadata,
    Tone,
)


def done(mem, sid, src, tr="t", style=None, key=None):
    mem.begin_turn(sid, src)
    mem.complete_turn(sid, tr, style or StyleMetadata(), key_moment=key)


def test_two_phase_turn_lifecycle():
    mem = ConversationMemory()
    turn = mem.begin_turn(1, "hello")
    assert not turn.complete
    mem.complete_turn(1, "hola", StyleMetadata(tone=Tone.WARM))
    assert mem.get(1).complete and mem.get(1).translated_text == "hola"


def test_detection_context_excludes_current_and_shows_tone():
    mem = ConversationMemory()
    done(mem, 1, "The build failed again.", style=StyleMetadata(tone=Tone.ANNOYED, emotion=Emotion.FRUSTRATION))
    mem.begin_turn(2, "Oh, great.")  # the one being classified
    ctx, n = mem.detection_context(before=2)
    assert n == 1
    assert "The build failed again." in ctx and "annoyed/frustration" in ctx
    assert "Oh, great." not in ctx


def test_context_never_includes_turns_that_came_after_the_one_being_classified():
    """The STT pump can run ahead of the worker, so later turns may
    already be in memory; they must not leak into an earlier turn."""
    mem = ConversationMemory()
    mem.begin_turn(1, "first")
    mem.begin_turn(2, "second")
    mem.begin_turn(3, "third")
    ctx1, n1 = mem.detection_context(before=1)
    ctx2, _ = mem.detection_context(before=2)
    assert (ctx1, n1) == ("", 0)
    assert "first" in ctx2 and "second" not in ctx2 and "third" not in ctx2
    assert "third" not in mem.translation_context(before=2)[0]


def test_context_is_bounded_in_turns_and_characters_and_keeps_newest():
    mem = ConversationMemory()
    for i in range(1, 30):
        done(mem, i, f"sentence number {i} " + "x" * 100)
    ctx, n = mem.detection_context(max_turns=4, max_chars=300)
    assert n <= 4 and len(ctx) <= 300 + 10
    assert "number 29" in ctx and "number 1 " not in ctx


def test_translation_context_pairs_source_and_translation():
    mem = ConversationMemory()
    done(mem, 1, "Hello Maria", "Hola María")
    ctx, n = mem.translation_context()
    assert n == 1 and '"Hello Maria" -> "Hola María"' in ctx


def test_store_is_capped():
    mem = ConversationMemory(max_turns=3)
    for i in range(1, 8):
        done(mem, i, f"s{i}")
    assert [t.segment_id for t in mem.turns] == [5, 6, 7]


def test_completing_an_evicted_turn_does_not_crash():
    mem = ConversationMemory(max_turns=1)
    mem.begin_turn(1, "a")
    mem.begin_turn(2, "b")  # evicts 1
    mem.complete_turn(1, "x", StyleMetadata())  # must not raise
    assert mem.get(1) is not None


def test_previous_style_and_stats():
    mem = ConversationMemory()
    done(mem, 1, "a", style=StyleMetadata(tone=Tone.SARCASTIC, sarcasm_score=0.8))
    done(mem, 2, "b", style=StyleMetadata(tone=Tone.SARCASTIC, sarcasm_score=0.4))
    assert mem.previous_style(2).tone == Tone.SARCASTIC
    assert mem.previous_style(1) is None
    st = mem.stats()
    assert st["turn_count"] == 2 and st["dominant_tone"] == "sarcastic"
    assert abs(st["mean_sarcasm"] - 0.6) < 1e-9


def test_reset_clears_memory():
    mem = ConversationMemory()
    done(mem, 1, "a")
    mem.reset()
    assert len(mem) == 0


# ------------------------------------------------------------ key moments


def test_plain_neutral_speech_is_not_a_key_moment():
    km = detect_key_moment(StyleMetadata())
    assert not km.is_key and km.reasons == []


def test_strong_sarcasm_is_a_key_moment_with_a_reason():
    km = detect_key_moment(StyleMetadata(tone=Tone.SARCASTIC, sarcasm_score=0.9))
    assert km.is_key and KeyMomentReason.HIGH_SARCASM in km.reasons


def test_urgent_tone_is_a_key_moment():
    km = detect_key_moment(StyleMetadata(tone=Tone.URGENT, emotion=Emotion.CONCERN))
    assert km.is_key and KeyMomentReason.URGENT in km.reasons


def test_low_confidence_discounts_the_score():
    hi = detect_key_moment(StyleMetadata(tone=Tone.URGENT, confidence=1.0))
    lo = detect_key_moment(StyleMetadata(tone=Tone.URGENT, confidence=0.2))
    assert lo.score < hi.score


def test_shift_between_two_non_neutral_tones_counts_but_neutral_transition_does_not():
    warm = StyleMetadata(tone=Tone.WARM)
    annoyed = StyleMetadata(tone=Tone.ANNOYED, emotion=Emotion.ANNOYANCE)
    assert KeyMomentReason.EMOTION_SHIFT in detect_key_moment(annoyed, previous=warm).reasons
    assert KeyMomentReason.EMOTION_SHIFT not in detect_key_moment(
        annoyed, previous=StyleMetadata()
    ).reasons


def test_modality_conflict_and_arousal_contribute():
    style = StyleMetadata(
        tone=Tone.WARM,
        modality_conflict=True,
        arousal=Arousal.HIGH,
        acoustic=AcousticFeatures(arousal_score=0.9),
    )
    reasons = detect_key_moment(style).reasons
    assert KeyMomentReason.MODALITY_CONFLICT in reasons
    assert KeyMomentReason.HIGH_AROUSAL in reasons


def test_score_is_capped_at_one():
    style = StyleMetadata(
        tone=Tone.URGENT, sarcasm_score=1.0, emotion=Emotion.FRUSTRATION,
        modality_conflict=True, emphasis_words=["a", "b"],
    )
    assert detect_key_moment(style).score <= 1.0
