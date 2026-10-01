"use client";

import type { SummaryMessage } from "@/lib/ws-client";

/** Automatic conversation summary, generated on request. */

interface SummaryPanelProps {
  summary: SummaryMessage | null;
  pending: boolean;
  disabled: boolean;
  onRequest: () => void;
}

export default function SummaryPanel({
  summary,
  pending,
  disabled,
  onRequest,
}: SummaryPanelProps) {
  return (
    <section
      aria-label="Conversation summary"
      className="w-full max-w-2xl rounded border border-neutral-800 bg-neutral-900/60 p-3 text-sm"
    >
      <div className="mb-2 flex items-center gap-2">
        <span className="text-xs font-medium text-neutral-300">Summary</span>
        <button
          onClick={onRequest}
          disabled={disabled || pending}
          className="ml-auto rounded bg-neutral-700 px-3 py-1 text-xs disabled:opacity-40"
        >
          {pending ? "Summarizing…" : summary ? "Refresh summary" : "Summarize conversation"}
        </button>
      </div>

      {!summary ? (
        <p className="text-xs text-neutral-600">
          Generates an overview, key points and action items from the
          conversation so far.
        </p>
      ) : (
        <div className="space-y-2">
          <p className="text-neutral-100">{summary.overview}</p>
          <p className="text-xs text-neutral-400">
            Overall tone: {summary.overall_tone} · {summary.turn_count} turn
            {summary.turn_count === 1 ? "" : "s"}
          </p>
          {summary.key_points.length > 0 && (
            <div>
              <h3 className="text-xs uppercase tracking-wide text-neutral-500">
                Key points
              </h3>
              <ul className="list-disc space-y-0.5 pl-5 text-neutral-200">
                {summary.key_points.map((p, i) => (
                  <li key={i}>{p}</li>
                ))}
              </ul>
            </div>
          )}
          {summary.action_items.length > 0 && (
            <div>
              <h3 className="text-xs uppercase tracking-wide text-neutral-500">
                Action items
              </h3>
              <ul className="list-disc space-y-0.5 pl-5 text-neutral-200">
                {summary.action_items.map((p, i) => (
                  <li key={i}>{p}</li>
                ))}
              </ul>
            </div>
          )}
          {summary.generated_by === "extractive" && (
            <p className="text-xs text-amber-300/80">
              The AI summary was unavailable, so this is a simple extract of
              key moments — not a full summary.
            </p>
          )}
        </div>
      )}
    </section>
  );
}
