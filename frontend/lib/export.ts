/**
 * Pure export helpers (no DOM dependency, so they run under plain
 * Node in tests/export.test.ts): transcript, subtitle and audio files
 * built from the segments a session produced.
 *
 * Timing for subtitles comes from the server's voice-activity gate
 * (`speech_start_ms` / `speech_end_ms`, on the mic stream's clock, which
 * starts when the Start button is pressed). When a segment has no
 * measured timing we estimate from text length and say so nowhere in
 * the file -- subtitles are a convenience artifact, not evidence.
 */

import type {
  AudioFormat,
  KeyMomentTag,
  SegmentMessage,
  StyleTag,
} from "./ws-client";

export interface ExportSegment {
  segmentId: number;
  sourceText: string;
  translatedText: string;
  style: StyleTag;
  keyMoment: KeyMomentTag;
  degraded: boolean;
  speechStartMs: number | null;
  speechEndMs: number | null;
}

export function toExportSegment(msg: SegmentMessage): ExportSegment {
  return {
    segmentId: msg.segment_id,
    sourceText: msg.source_text,
    translatedText: msg.translated_text,
    style: msg.style,
    keyMoment: msg.key_moment,
    degraded: msg.degraded,
    speechStartMs: msg.speech_start_ms,
    speechEndMs: msg.speech_end_ms,
  };
}

export type SubtitleText = "translated" | "source" | "both";

export interface SubtitleCue {
  index: number;
  startMs: number;
  endMs: number;
  text: string;
}

const MIN_CUE_MS = 800;
const MS_PER_CHAR = 60;
const MIN_ESTIMATED_CUE_MS = 1500;

function pick(seg: ExportSegment, which: SubtitleText): string {
  if (which === "source") return seg.sourceText.trim();
  if (which === "translated") return seg.translatedText.trim();
  return `${seg.translatedText.trim()}\n${seg.sourceText.trim()}`;
}

/** Builds ordered, non-overlapping cues; skips empty text. */
export function buildCues(
  segments: ExportSegment[],
  which: SubtitleText = "translated"
): SubtitleCue[] {
  const cues: SubtitleCue[] = [];
  let cursor = 0;
  for (const seg of segments) {
    const text = pick(seg, which);
    if (!text.replace(/\s/g, "")) continue;

    let start: number;
    let end: number;
    if (seg.speechStartMs != null && seg.speechEndMs != null) {
      start = seg.speechStartMs;
      end = seg.speechEndMs;
    } else {
      start = cursor;
      end =
        start + Math.max(MIN_ESTIMATED_CUE_MS, seg.sourceText.length * MS_PER_CHAR);
    }
    start = Math.max(start, cursor); // never overlap the previous cue
    end = Math.max(end, start + MIN_CUE_MS);
    cues.push({ index: cues.length + 1, startMs: start, endMs: end, text });
    cursor = end;
  }
  return cues;
}

function pad(n: number, width: number): string {
  return String(Math.floor(n)).padStart(width, "0");
}

export function formatTimestamp(ms: number, separator: "," | "."): string {
  const total = Math.max(0, Math.round(ms));
  const h = total / 3_600_000;
  const m = (total % 3_600_000) / 60_000;
  const s = (total % 60_000) / 1000;
  const milli = total % 1000;
  return `${pad(h, 2)}:${pad(m, 2)}:${pad(s, 2)}${separator}${pad(milli, 3)}`;
}

export function toSrt(cues: SubtitleCue[]): string {
  return (
    cues
      .map(
        (c) =>
          `${c.index}\n${formatTimestamp(c.startMs, ",")} --> ${formatTimestamp(
            c.endMs,
            ","
          )}\n${c.text}`
      )
      .join("\n\n") + (cues.length ? "\n" : "")
  );
}

export function toVtt(cues: SubtitleCue[]): string {
  // "-->" inside cue text would end the cue early in WebVTT.
  const safe = (t: string) => t.replace(/-->/g, "->");
  return (
    "WEBVTT\n\n" +
    cues
      .map(
        (c) =>
          `${c.index}\n${formatTimestamp(c.startMs, ".")} --> ${formatTimestamp(
            c.endMs,
            "."
          )}\n${safe(c.text)}`
      )
      .join("\n\n") +
    (cues.length ? "\n" : "")
  );
}

function describeStyle(style: StyleTag): string {
  const bits = [style.tone, style.emotion];
  if (style.sarcasm_score >= 0.5) {
    bits.push(`sarcasm ${Math.round(style.sarcasm_score * 100)}%`);
  }
  return bits.join(", ");
}

export interface TranscriptOptions {
  targetLanguage: string;
  includeStyle?: boolean;
  onlyKeyMoments?: boolean;
}

export function toTranscriptText(
  segments: ExportSegment[],
  opts: TranscriptOptions
): string {
  const rows = opts.onlyKeyMoments
    ? segments.filter((s) => s.keyMoment.is_key)
    : segments;
  const lines: string[] = [`Vernacular transcript (target language: ${opts.targetLanguage})`, ""];
  for (const seg of rows) {
    const flag = seg.keyMoment.is_key ? " ★" : "";
    lines.push(`[${seg.segmentId}]${flag}${opts.includeStyle === false ? "" : ` (${describeStyle(seg.style)})`}`);
    lines.push(`  Original:   ${seg.sourceText}`);
    lines.push(`  Translated: ${seg.translatedText}`);
    lines.push("");
  }
  return lines.join("\n");
}

export function toTranscriptMarkdown(
  segments: ExportSegment[],
  opts: TranscriptOptions
): string {
  const rows = opts.onlyKeyMoments
    ? segments.filter((s) => s.keyMoment.is_key)
    : segments;
  const out: string[] = [`# Vernacular transcript`, "", `Target language: \`${opts.targetLanguage}\``, ""];
  for (const seg of rows) {
    out.push(`## Turn ${seg.segmentId}${seg.keyMoment.is_key ? " ★ key moment" : ""}`);
    if (opts.includeStyle !== false) out.push(`*${describeStyle(seg.style)}*`);
    out.push("", `> ${seg.sourceText.replace(/\n/g, " ")}`, "", seg.translatedText, "");
  }
  return out.join("\n");
}

export function toTranscriptJson(
  segments: ExportSegment[],
  opts: TranscriptOptions
): string {
  return JSON.stringify(
    {
      target_language: opts.targetLanguage,
      segments: segments.map((s) => ({
        segment_id: s.segmentId,
        source_text: s.sourceText,
        translated_text: s.translatedText,
        style: s.style,
        key_moment: s.keyMoment,
        degraded: s.degraded,
        speech_start_ms: s.speechStartMs,
        speech_end_ms: s.speechEndMs,
      })),
    },
    null,
    2
  );
}

/** Wraps PCM16 little-endian samples in a RIFF/WAVE container. */
export function pcmToWav(pcm: Uint8Array, format: AudioFormat): Uint8Array {
  const bytesPerSample = 2;
  const blockAlign = format.channels * bytesPerSample;
  // An odd trailing byte is a half-sample; drop it rather than emit a
  // file whose data length disagrees with its sample size.
  const dataLength = pcm.length - (pcm.length % blockAlign);
  const out = new Uint8Array(44 + dataLength);
  const view = new DataView(out.buffer);
  const ascii = (offset: number, text: string) => {
    for (let i = 0; i < text.length; i++) out[offset + i] = text.charCodeAt(i);
  };
  ascii(0, "RIFF");
  view.setUint32(4, 36 + dataLength, true);
  ascii(8, "WAVE");
  ascii(12, "fmt ");
  view.setUint32(16, 16, true); // PCM fmt chunk size
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, format.channels, true);
  view.setUint32(24, format.sample_rate, true);
  view.setUint32(28, format.sample_rate * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true);
  ascii(36, "data");
  view.setUint32(40, dataLength, true);
  out.set(pcm.subarray(0, dataLength), 44);
  return out;
}

/** Concatenates the binary frames of a session into one buffer. */
export function concatChunks(chunks: ArrayBuffer[]): Uint8Array {
  const total = chunks.reduce((n, c) => n + c.byteLength, 0);
  const out = new Uint8Array(total);
  let offset = 0;
  for (const c of chunks) {
    out.set(new Uint8Array(c), offset);
    offset += c.byteLength;
  }
  return out;
}

/** Hard cap on retained translated audio (bytes): ~30 min of 24 kHz
 * mono PCM16 is ~86 MB, which is more than a tab should hold. */
export const MAX_AUDIO_BYTES = 64 * 1024 * 1024;
