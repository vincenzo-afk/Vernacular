import assert from "node:assert/strict";
import { test } from "node:test";
import {
  buildCues,
  concatChunks,
  formatTimestamp,
  pcmToWav,
  toSrt,
  toTranscriptJson,
  toTranscriptMarkdown,
  toTranscriptText,
  toVtt,
  type ExportSegment,
} from "../lib/export";
import type { StyleTag } from "../lib/ws-client";

const style = (over: Partial<StyleTag> = {}): StyleTag => ({
  tone: "neutral",
  pace: "normal",
  formality: "neutral",
  emotion: "neutral",
  sarcasm_score: 0,
  emphasis_words: [],
  pause_pattern: "natural",
  confidence: 1,
  arousal: "medium",
  acoustic: null,
  modality_conflict: false,
  explanation: null,
  ...over,
});

const seg = (over: Partial<ExportSegment> = {}): ExportSegment => ({
  segmentId: 1,
  sourceText: "Oh, great.",
  translatedText: "Ay, qué bien.",
  style: style(),
  keyMoment: { is_key: false, score: 0, reasons: [] },
  degraded: false,
  speechStartMs: 1000,
  speechEndMs: 2500,
  ...over,
});

test("formatTimestamp uses comma for SRT and dot for VTT", () => {
  assert.equal(formatTimestamp(3_723_456, ","), "01:02:03,456");
  assert.equal(formatTimestamp(3_723_456, "."), "01:02:03.456");
  assert.equal(formatTimestamp(-5, ","), "00:00:00,000");
});

test("SRT output is numbered with arrow timings", () => {
  const srt = toSrt(buildCues([seg(), seg({ segmentId: 2, speechStartMs: 3000, speechEndMs: 4000 })]));
  assert.equal(
    srt,
    "1\n00:00:01,000 --> 00:00:02,500\nAy, qué bien.\n\n" +
      "2\n00:00:03,000 --> 00:00:04,000\nAy, qué bien.\n"
  );
});

test("VTT has header, dot timings and neutralizes '-->' in text", () => {
  const vtt = toVtt(buildCues([seg({ translatedText: "a --> b" })]));
  assert.ok(vtt.startsWith("WEBVTT\n\n1\n00:00:01.000 --> 00:00:02.500\n"));
  assert.ok(vtt.includes("a -> b"));
});

test("cues never overlap and have a minimum duration", () => {
  const cues = buildCues([
    seg({ speechStartMs: 0, speechEndMs: 2000 }),
    seg({ segmentId: 2, speechStartMs: 1500, speechEndMs: 1600 }), // overlaps + too short
  ]);
  assert.equal(cues[1].startMs, 2000);
  assert.ok(cues[1].endMs - cues[1].startMs >= 800);
});

test("segments without measured timing get sequential estimated cues", () => {
  const cues = buildCues([
    seg({ speechStartMs: null, speechEndMs: null }),
    seg({ segmentId: 2, speechStartMs: null, speechEndMs: null }),
  ]);
  assert.equal(cues[0].startMs, 0);
  assert.equal(cues[1].startMs, cues[0].endMs);
});

test("empty text segments are skipped and cues renumbered", () => {
  const cues = buildCues([seg({ translatedText: "  " }), seg({ segmentId: 2 })]);
  assert.equal(cues.length, 1);
  assert.equal(cues[0].index, 1);
});

test("'both' puts translation first, original second", () => {
  const [cue] = buildCues([seg()], "both");
  assert.equal(cue.text, "Ay, qué bien.\nOh, great.");
});

test("transcript text/markdown/json carry both languages and key flags", () => {
  const key = seg({
    keyMoment: { is_key: true, score: 0.8, reasons: ["high_sarcasm"] },
    style: style({ tone: "sarcastic", sarcasm_score: 0.9 }),
  });
  const opts = { targetLanguage: "es" };
  const text = toTranscriptText([key], opts);
  assert.ok(text.includes("★") && text.includes("sarcasm 90%"));
  assert.ok(text.includes("Original:   Oh, great."));
  assert.ok(toTranscriptMarkdown([key], opts).includes("key moment"));
  const json = JSON.parse(toTranscriptJson([key], opts));
  assert.equal(json.segments[0].translated_text, "Ay, qué bien.");
  assert.equal(json.segments[0].key_moment.is_key, true);
});

test("onlyKeyMoments filters the transcript", () => {
  const plain = seg({ segmentId: 1 });
  const key = seg({ segmentId: 2, keyMoment: { is_key: true, score: 0.9, reasons: [] } });
  const text = toTranscriptText([plain, key], { targetLanguage: "es", onlyKeyMoments: true });
  assert.ok(text.includes("[2]") && !text.includes("[1]"));
});

test("pcmToWav writes a valid 44-byte header and drops a half sample", () => {
  const pcm = new Uint8Array([1, 0, 2, 0, 3]); // 2 samples + 1 stray byte
  const wav = pcmToWav(pcm, { encoding: "pcm_s16le", sample_rate: 24000, channels: 1 });
  const view = new DataView(wav.buffer);
  const tag = (o: number) => String.fromCharCode(...wav.slice(o, o + 4));
  assert.equal(tag(0), "RIFF");
  assert.equal(tag(8), "WAVE");
  assert.equal(tag(36), "data");
  assert.equal(view.getUint32(24, true), 24000); // sample rate
  assert.equal(view.getUint32(28, true), 48000); // byte rate
  assert.equal(view.getUint32(40, true), 4); // data length: stray byte dropped
  assert.equal(view.getUint32(4, true), 36 + 4);
  assert.equal(wav.length, 48);
  assert.deepEqual(Array.from(wav.slice(44)), [1, 0, 2, 0]);
});

test("concatChunks preserves byte order across frames", () => {
  const a = new Uint8Array([1, 2]).buffer;
  const b = new Uint8Array([3]).buffer;
  assert.deepEqual(Array.from(concatChunks([a, b])), [1, 2, 3]);
});
