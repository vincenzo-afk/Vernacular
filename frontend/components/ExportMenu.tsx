"use client";

import { useState } from "react";
import {
  buildCues,
  pcmToWav,
  toSrt,
  toTranscriptJson,
  toTranscriptMarkdown,
  toTranscriptText,
  toVtt,
  type ExportSegment,
  type SubtitleText,
} from "@/lib/export";
import { downloadFile } from "@/lib/download";
import type { AudioFormat } from "@/lib/ws-client";

/**
 * Export the session: transcript (txt / md / json), subtitles
 * (srt / vtt) and the translated audio (wav). Everything is generated
 * in the browser from what this session already received -- nothing is
 * uploaded anywhere.
 */

interface ExportMenuProps {
  segments: ExportSegment[];
  targetLanguage: string;
  /** Returns the concatenated translated PCM, or null if none. */
  getAudio: () => Uint8Array | null;
  audioFormat: AudioFormat | null;
  /** True if the retained audio hit the memory cap and was cut short. */
  audioTruncated: boolean;
}

const SUBTITLE_TEXT: { id: SubtitleText; label: string }[] = [
  { id: "translated", label: "Translated" },
  { id: "source", label: "Original" },
  { id: "both", label: "Both" },
];

// Called from click handlers only (never during render): the timestamp
// is impure.
const baseName = (language: string) =>
  `vernacular-${language}-${new Date()
    .toISOString()
    .slice(0, 19)
    .replace(/[:T]/g, "-")}`;

export default function ExportMenu({
  segments,
  targetLanguage,
  getAudio,
  audioFormat,
  audioTruncated,
}: ExportMenuProps) {
  const [subtitleText, setSubtitleText] = useState<SubtitleText>("translated");
  const [keyOnly, setKeyOnly] = useState(false);
  const empty = segments.length === 0;
  const opts = { targetLanguage, onlyKeyMoments: keyOnly };

  const button =
    "rounded bg-neutral-700 px-3 py-1 text-xs disabled:opacity-40";

  const exportAudio = () => {
    const pcm = getAudio();
    if (!pcm || pcm.length === 0 || !audioFormat) return;
    downloadFile(`${baseName(targetLanguage)}.wav`, pcmToWav(pcm, audioFormat), "audio/wav");
  };

  return (
    <section
      aria-label="Export"
      className="w-full max-w-2xl rounded border border-neutral-800 bg-neutral-900/60 p-3 text-xs"
    >
      <div className="mb-2 flex items-center gap-2">
        <span className="font-medium text-neutral-300">Export</span>
        <label className="ml-auto flex items-center gap-1.5 text-neutral-400">
          <input
            type="checkbox"
            checked={keyOnly}
            onChange={(e) => setKeyOnly(e.target.checked)}
          />
          ★ key moments only (transcript)
        </label>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <span className="w-20 text-neutral-500">Transcript</span>
        <button
          className={button}
          disabled={empty}
          onClick={() =>
            downloadFile(`${baseName(targetLanguage)}.txt`, toTranscriptText(segments, opts), "text/plain")
          }
        >
          .txt
        </button>
        <button
          className={button}
          disabled={empty}
          onClick={() =>
            downloadFile(`${baseName(targetLanguage)}.md`, toTranscriptMarkdown(segments, opts), "text/markdown")
          }
        >
          .md
        </button>
        <button
          className={button}
          disabled={empty}
          onClick={() =>
            downloadFile(`${baseName(targetLanguage)}.json`, toTranscriptJson(segments, opts), "application/json")
          }
        >
          .json
        </button>
      </div>

      <div className="mt-2 flex flex-wrap items-center gap-2">
        <span className="w-20 text-neutral-500">Subtitles</span>
        <select
          value={subtitleText}
          onChange={(e) => setSubtitleText(e.target.value as SubtitleText)}
          className="rounded bg-neutral-800 px-2 py-1"
          aria-label="Subtitle language"
        >
          {SUBTITLE_TEXT.map((o) => (
            <option key={o.id} value={o.id}>
              {o.label}
            </option>
          ))}
        </select>
        <button
          className={button}
          disabled={empty}
          onClick={() =>
            downloadFile(`${baseName(targetLanguage)}.srt`, toSrt(buildCues(segments, subtitleText)), "application/x-subrip")
          }
        >
          .srt
        </button>
        <button
          className={button}
          disabled={empty}
          onClick={() =>
            downloadFile(`${baseName(targetLanguage)}.vtt`, toVtt(buildCues(segments, subtitleText)), "text/vtt")
          }
        >
          .vtt
        </button>
      </div>

      <div className="mt-2 flex flex-wrap items-center gap-2">
        <span className="w-20 text-neutral-500">Audio</span>
        <button className={button} disabled={empty || !audioFormat} onClick={exportAudio}>
          Translated audio .wav
        </button>
        {audioTruncated && (
          <span className="text-amber-300/80">
            Audio was capped to save memory — only the start is included.
          </span>
        )}
      </div>
    </section>
  );
}
