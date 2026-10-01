"use client";

import {
  TOTAL_BUDGET_MS,
  criticalPathBars,
  type LatencySummary,
  type LatencyVerdict,
} from "@/lib/latency";
import type { NetworkTier } from "@/lib/ws-client";

/**
 * Real-time latency HUD. Shows where the last utterance's time went
 * against the ~2 s budget, plus the live network state.
 *
 * Honest about its limits: the server figures start when the FINAL
 * transcript reaches the orchestrator, so speech endpointing (~300 ms)
 * and browser playback start are not included -- the true delay the
 * listener feels is somewhat higher than the total shown.
 */

const VERDICT_BAR: Record<LatencyVerdict | "none", string> = {
  ok: "bg-emerald-600",
  warn: "bg-amber-500",
  over: "bg-red-600",
  none: "bg-neutral-500",
};

const VERDICT_TEXT: Record<LatencyVerdict, string> = {
  ok: "text-emerald-400",
  warn: "text-amber-400",
  over: "text-red-400",
};

const TIER_STYLE: Record<NetworkTier, string> = {
  good: "bg-emerald-900 text-emerald-200",
  fair: "bg-amber-900 text-amber-200",
  poor: "bg-red-900 text-red-200",
};

const TIER_HINT: Record<NetworkTier, string> = {
  good: "Full-rate captions and audio frames.",
  fair: "Captions throttled to ~4/s; audio frames merged (~100 ms).",
  poor: "Captions throttled to ~1.4/s; audio frames merged (~200 ms).",
};

const fmt = (ms: number) => (ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${Math.round(ms)} ms`);

interface LatencyHudProps {
  summary: LatencySummary | null;
  rttMs: number | null;
  tier: NetworkTier;
  serverSendMs: number | null;
}

export default function LatencyHud({
  summary,
  rttMs,
  tier,
  serverSendMs,
}: LatencyHudProps) {
  const bars = summary ? criticalPathBars(summary.last) : [];
  const total = summary?.last.server_total_ms ?? 0;
  const scale = Math.max(TOTAL_BUDGET_MS, total);

  return (
    <section
      aria-label="Latency"
      className="w-full max-w-2xl rounded border border-neutral-800 bg-neutral-900/60 p-3 text-xs"
    >
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <span className="font-medium text-neutral-300">Latency</span>
        <span
          className={`rounded-full px-2 py-0.5 ${TIER_STYLE[tier]}`}
          title={TIER_HINT[tier]}
        >
          network: {tier}
        </span>
        <span className="text-neutral-500">
          RTT {rttMs == null ? "—" : fmt(rttMs)}
          {serverSendMs != null && ` · server send ${fmt(serverSendMs)}`}
        </span>
        {summary && (
          <span className={`ml-auto tabular-nums ${VERDICT_TEXT[summary.lastVerdict]}`}>
            {fmt(total)} / {fmt(TOTAL_BUDGET_MS)}
          </span>
        )}
      </div>

      {!summary ? (
        <p className="text-neutral-600">Speak to see the pipeline breakdown.</p>
      ) : (
        <>
          <div
            className="relative flex h-3 overflow-hidden rounded bg-neutral-800"
            role="img"
            aria-label={`Last utterance took ${fmt(total)} on the server`}
          >
            {bars.map((b) => (
              <div
                key={b.key}
                className={VERDICT_BAR[b.verdict ?? "none"]}
                style={{ width: `${(b.ms / scale) * 100}%` }}
                title={`${b.label}: ${fmt(b.ms)}`}
              />
            ))}
            {/* the 2 s budget line */}
            <div
              className="absolute inset-y-0 w-px bg-neutral-200/70"
              style={{ left: `${(TOTAL_BUDGET_MS / scale) * 100}%` }}
              title="2 s budget"
            />
          </div>
          <dl className="mt-2 grid grid-cols-2 gap-x-4 gap-y-0.5 sm:grid-cols-4">
            {bars.map((b) => (
              <div key={b.key} className="flex justify-between gap-2">
                <dt className="text-neutral-500">{b.label}</dt>
                <dd
                  className={`tabular-nums ${
                    b.verdict ? VERDICT_TEXT[b.verdict] : "text-neutral-300"
                  }`}
                >
                  {fmt(b.ms)}
                </dd>
              </div>
            ))}
          </dl>
          <p className="mt-2 text-neutral-500">
            Tone detection ran {summary.last.register_detection_ms == null
              ? "(untimed)"
              : `${fmt(summary.last.register_detection_ms)} in the background`}
            . Over {summary.count} turn{summary.count === 1 ? "" : "s"}: median{" "}
            {fmt(summary.p50TotalMs)}, p90 {fmt(summary.p90TotalMs)}, worst{" "}
            {fmt(summary.maxTotalMs)}.
          </p>
          <p className="mt-1 text-neutral-600">
            Counted from when the finished sentence reached the server;
            excludes speech endpointing and browser playback, so the delay
            you hear is somewhat higher.
          </p>
        </>
      )}
    </section>
  );
}
