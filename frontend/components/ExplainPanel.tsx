"use client";

import type { CueTag, SegmentMessage } from "@/lib/ws-client";

/**
 * Explainable tone / sarcasm: WHY the system read the register the way
 * it did. Two kinds of evidence are kept visually distinct on purpose:
 *
 *  - "measured" cues are computed from the audio and word timings and
 *    can be checked;
 *  - "model" cues are the language model's own account and can be
 *    wrong. Showing them as if they were facts would overstate how much
 *    the explanation proves.
 */

const KIND_LABEL: Record<CueTag["kind"], string> = {
  lexical: "words",
  prosodic: "timing",
  acoustic: "voice",
  contextual: "context",
  incongruence: "mismatch",
};

function CueRow({ cue }: { cue: CueTag }) {
  const measured = cue.source === "measured";
  return (
    <li className="flex items-start gap-2">
      <span
        className={`mt-0.5 shrink-0 rounded px-1.5 py-0.5 text-[10px] uppercase tracking-wide ${
          measured
            ? "bg-emerald-950 text-emerald-300"
            : "bg-neutral-800 text-neutral-400"
        }`}
        title={
          measured
            ? "Measured from the audio / word timings"
            : "Stated by the language model — not independently verified"
        }
      >
        {measured ? "measured" : "model"} · {KIND_LABEL[cue.kind]}
      </span>
      <span className="text-neutral-200">{cue.evidence}</span>
      <span
        className="ml-auto shrink-0 tabular-nums text-neutral-500"
        title="How much weight this cue carries"
      >
        {Math.round(cue.weight * 100)}%
      </span>
    </li>
  );
}

export default function ExplainPanel({ segment }: { segment: SegmentMessage }) {
  const { style } = segment;
  const explanation = style.explanation;
  const sarcasm = Math.round(style.sarcasm_score * 100);

  return (
    <div className="mt-2 space-y-2 rounded border border-neutral-800 bg-neutral-900/60 p-3 text-xs">
      <div className="flex items-center gap-2">
        <span className="text-neutral-400">Sarcasm</span>
        <div className="h-1.5 flex-1 rounded bg-neutral-800">
          <div
            className="h-1.5 rounded bg-amber-500"
            style={{ width: `${sarcasm}%` }}
          />
        </div>
        <span className="tabular-nums text-neutral-300">{sarcasm}%</span>
        <span className="text-neutral-500">
          · confidence {Math.round(style.confidence * 100)}%
        </span>
      </div>

      {explanation?.summary && (
        <p className="text-neutral-200">{explanation.summary}</p>
      )}

      {explanation && explanation.cues.length > 0 ? (
        <ul className="space-y-1.5">
          {explanation.cues.map((cue, i) => (
            <CueRow key={`${cue.kind}-${i}`} cue={cue} />
          ))}
        </ul>
      ) : (
        <p className="text-neutral-500">
          No specific evidence was recorded for this reading (it may have
          fallen back to neutral because the read was low-confidence).
        </p>
      )}

      <p className="text-neutral-500">
        {explanation?.context_used
          ? `Judged in context — the earlier conversation changed this reading (${segment.context_turns} earlier turn${segment.context_turns === 1 ? "" : "s"} available).`
          : segment.context_turns > 0
            ? `${segment.context_turns} earlier turn${segment.context_turns === 1 ? "" : "s"} were available; they did not change this reading.`
            : "No earlier conversation to draw on yet."}
      </p>
    </div>
  );
}
