"use client";

import type { StyleTag } from "@/lib/ws-client";

/**
 * Renders the live StyleMetadata reading as tags — this is the visible
 * proof-of-concept for the register-detection stage during the demo.
 */

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
      {tag.sarcasm_score > 0.5 && (
        <span className="rounded-full bg-amber-900 px-3 py-1">
          sarcasm {Math.round(tag.sarcasm_score * 100)}%
        </span>
      )}
    </div>
  );
}
