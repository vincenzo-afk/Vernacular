"use client";

import { useState } from "react";
import type { SegmentMessage } from "@/lib/ws-client";
import ToneTags from "@/components/ToneTags";
import ExplainPanel from "@/components/ExplainPanel";

/**
 * Live captions + the original/translated transcript.
 *
 * - `live` is the caption for the utterance in progress (source
 *   language, straight from STT). It updates as words are recognized
 *   and stays on screen after the utterance ends until its translation
 *   arrives, so the speaker never sees a blank gap.
 * - `segments` is the running transcript. Key moments are starred; each
 *   turn can be expanded to see WHY its tone was read that way.
 */

export type TranscriptView = "both" | "original" | "translated";

export interface LiveCaption {
  text: string;
  isFinal: boolean;
  segmentId: number | null;
}

interface LiveTranscriptProps {
  live: LiveCaption | null;
  segments: SegmentMessage[];
  view: TranscriptView;
  onViewChange: (view: TranscriptView) => void;
  keyOnly: boolean;
  onKeyOnlyChange: (value: boolean) => void;
}

const VIEWS: { id: TranscriptView; label: string }[] = [
  { id: "both", label: "Both" },
  { id: "original", label: "Original" },
  { id: "translated", label: "Translated" },
];

export default function LiveTranscript({
  live,
  segments,
  view,
  onViewChange,
  keyOnly,
  onKeyOnlyChange,
}: LiveTranscriptProps) {
  const [open, setOpen] = useState<number | null>(null);
  const rows = keyOnly ? segments.filter((s) => s.key_moment.is_key) : segments;

  return (
    <div className="w-full max-w-2xl space-y-3">
      <div
        aria-live="polite"
        className="min-h-[3.5rem] rounded bg-neutral-900 px-4 py-3"
      >
        {live && live.text ? (
          <p className={live.isFinal ? "text-neutral-100" : "text-neutral-400"}>
            {live.text}
            {live.isFinal && (
              <span className="ml-2 text-xs text-neutral-500">
                translating…
              </span>
            )}
          </p>
        ) : (
          <p className="text-neutral-600">Live captions appear here…</p>
        )}
      </div>

      <div className="flex items-center gap-2 text-xs">
        <div className="flex overflow-hidden rounded border border-neutral-800">
          {VIEWS.map((v) => (
            <button
              key={v.id}
              onClick={() => onViewChange(v.id)}
              className={`px-3 py-1 ${
                view === v.id
                  ? "bg-neutral-700 text-neutral-50"
                  : "bg-neutral-900 text-neutral-400"
              }`}
            >
              {v.label}
            </button>
          ))}
        </div>
        <label className="ml-auto flex items-center gap-1.5 text-neutral-400">
          <input
            type="checkbox"
            checked={keyOnly}
            onChange={(e) => onKeyOnlyChange(e.target.checked)}
          />
          ★ key moments only
        </label>
      </div>

      <ol className="space-y-3">
        {rows.length === 0 && (
          <li className="text-sm text-neutral-600">
            {segments.length === 0
              ? "No turns yet."
              : "No key moments detected yet."}
          </li>
        )}
        {rows.map((s) => (
          <li
            key={s.segment_id}
            className={`rounded border px-4 py-3 ${
              s.key_moment.is_key
                ? "border-amber-700/60 bg-amber-950/20"
                : "border-neutral-800"
            }`}
          >
            <div className="mb-1 flex items-center gap-2 text-xs text-neutral-500">
              <span>#{s.segment_id}</span>
              {s.key_moment.is_key && (
                <span
                  className="rounded bg-amber-900 px-1.5 py-0.5 text-amber-200"
                  title={`Key moment (score ${Math.round(s.key_moment.score * 100)}%)`}
                >
                  ★ key moment · {s.key_moment.reasons.join(", ").replace(/_/g, " ")}
                </span>
              )}
              {s.degraded && (
                <span className="rounded bg-amber-900/60 px-1.5 py-0.5 text-amber-200">
                  fallback voice
                </span>
              )}
            </div>
            {view !== "translated" && (
              <p className="text-neutral-100">{s.source_text}</p>
            )}
            {view !== "original" && (
              <p className="text-emerald-400">{s.translated_text}</p>
            )}
            <div className="mt-2 flex items-center gap-3">
              <ToneTags tag={s.style} />
              <button
                onClick={() => setOpen(open === s.segment_id ? null : s.segment_id)}
                className="ml-auto text-xs text-neutral-400 underline underline-offset-2"
                aria-expanded={open === s.segment_id}
              >
                {open === s.segment_id ? "Hide why" : "Why this tone?"}
              </button>
            </div>
            {open === s.segment_id && <ExplainPanel segment={s} />}
          </li>
        ))}
      </ol>
    </div>
  );
}
