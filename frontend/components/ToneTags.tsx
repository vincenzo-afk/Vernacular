"use client";

import type { StyleTag } from "@/lib/ws-client";

/**
 * Renders the live StyleMetadata reading as tags — this is the visible
 * proof-of-concept for the register-detection stage during the demo.
 * Includes the multimodal signals: how activated the VOICE was
 * (measured from the audio) and whether it contradicted the words.
 */

const AROUSAL_STYLE: Record<StyleTag["arousal"], string> = {
  low: "bg-sky-900 text-sky-200",
  medium: "bg-neutral-800 text-neutral-300",
  high: "bg-rose-900 text-rose-200",
};

export default function ToneTags({ tag }: { tag: StyleTag | null }) {
  if (!tag) return null;

  return (
    <div className="flex flex-wrap gap-2 text-xs">
      <span className="rounded-full bg-neutral-800 px-3 py-1">{tag.tone}</span>
      <span className="rounded-full bg-neutral-800 px-3 py-1">{tag.pace}</span>
      <span className="rounded-full bg-neutral-800 px-3 py-1">
        {tag.formality}
      </span>
      <span className="rounded-full bg-neutral-800 px-3 py-1">
        {tag.emotion}
      </span>
      {tag.acoustic && (
        <span
          className={`rounded-full px-3 py-1 ${AROUSAL_STYLE[tag.arousal]}`}
          title="Voice activation measured from the audio, relative to this speaker's own baseline"
        >
          voice: {tag.arousal} energy
        </span>
      )}
      {tag.modality_conflict && (
        <span
          className="rounded-full bg-fuchsia-900 px-3 py-1 text-fuchsia-200"
          title="The delivery is quieter and flatter than the words suggest"
        >
          words ≠ voice
        </span>
      )}
      {tag.sarcasm_score > 0.5 && (
        <span className="rounded-full bg-amber-900 px-3 py-1">
          sarcasm {Math.round(tag.sarcasm_score * 100)}%
        </span>
      )}
    </div>
  );
}
