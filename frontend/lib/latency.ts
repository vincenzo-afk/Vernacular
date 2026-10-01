/**
 * Real-time latency HUD math. Pure (no DOM), unit-tested under Node.
 *
 * What is and is not measured -- shown honestly in the UI:
 *  - server figures come from the backend's monotonic clocks, per
 *    segment: queue wait + style wait + translation + TTS first byte
 *    ~= server total, counted from when the FINAL transcript reached
 *    the orchestrator.
 *  - STT endpointing (speaker stops -> final transcript) happens before
 *    that and is NOT included; nor is playback start in the browser.
 *  - `rtt` is a websocket ping/pong round trip measured by the client.
 * So "total" is a lower bound on what the listener actually experiences.
 */

import type { TimingsTag } from "./ws-client";

/** Mirrors backend/app/latency.py STAGE_BUDGET_MS. */
export const BUDGET_MS = {
  register_detection: 400,
  translation: 600,
  tts_first_byte: 500,
} as const;

/** End-to-end target from ARCHITECTURE.md §3. */
export const TOTAL_BUDGET_MS = 2000;

export type LatencyVerdict = "ok" | "warn" | "over";

export function verdictFor(valueMs: number, budgetMs: number): LatencyVerdict {
  if (valueMs > budgetMs) return "over";
  if (valueMs > budgetMs * 0.8) return "warn";
  return "ok";
}

export function percentile(values: number[], pct: number): number {
  if (values.length === 0) throw new Error("percentile of empty sequence");
  const sorted = [...values].sort((a, b) => a - b);
  if (sorted.length === 1) return sorted[0];
  const rank = (pct / 100) * (sorted.length - 1);
  const lo = Math.floor(rank);
  const hi = Math.min(lo + 1, sorted.length - 1);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (rank - lo);
}

export interface LatencySummary {
  count: number;
  last: TimingsTag;
  p50TotalMs: number;
  p90TotalMs: number;
  maxTotalMs: number;
  /** Verdict on the LAST segment's server total vs the 2 s budget
   * (only server time counts, so this is optimistic by design). */
  lastVerdict: LatencyVerdict;
}

export function summarizeLatency(timings: TimingsTag[]): LatencySummary | null {
  if (timings.length === 0) return null;
  const totals = timings.map((t) => t.server_total_ms);
  const last = timings[timings.length - 1];
  return {
    count: timings.length,
    last,
    p50TotalMs: percentile(totals, 50),
    p90TotalMs: percentile(totals, 90),
    maxTotalMs: Math.max(...totals),
    lastVerdict: verdictFor(last.server_total_ms, TOTAL_BUDGET_MS),
  };
}

export interface LatencyBar {
  key: "queue" | "style" | "translation" | "tts";
  label: string;
  ms: number;
  budgetMs: number | null;
  verdict: LatencyVerdict | null;
}

/** The critical-path stages of one segment, in order, for the HUD bar. */
export function criticalPathBars(t: TimingsTag): LatencyBar[] {
  const ttfb = t.tts_first_byte_ms ?? 0;
  return [
    { key: "queue", label: "Queue", ms: t.queue_wait_ms, budgetMs: null, verdict: null },
    { key: "style", label: "Tone wait", ms: t.style_wait_ms, budgetMs: null, verdict: null },
    {
      key: "translation",
      label: "Translate",
      ms: t.translation_ms,
      budgetMs: BUDGET_MS.translation,
      verdict: verdictFor(t.translation_ms, BUDGET_MS.translation),
    },
    {
      key: "tts",
      label: "Voice",
      ms: ttfb,
      budgetMs: BUDGET_MS.tts_first_byte,
      verdict: verdictFor(ttfb, BUDGET_MS.tts_first_byte),
    },
  ];
}

/** Matches `id`-tagged ping/pong pairs into round-trip times. */
export class RttProbe {
  private pending = new Map<string, number>();
  private seq = 0;
  last: number | null = null;

  /** Returns the id to send in the ping. */
  start(now: number): string {
    const id = `p${++this.seq}`;
    this.pending.set(id, now);
    // A lost pong must not leak: keep only the newest few.
    if (this.pending.size > 8) {
      const oldest = this.pending.keys().next().value;
      if (oldest !== undefined) this.pending.delete(oldest);
    }
    return id;
  }

  /** Age of the oldest ping still unanswered -- a lost pong is itself
   * evidence of a bad link, so callers treat a long wait as a high RTT. */
  oldestPendingMs(now: number): number | null {
    let oldest: number | null = null;
    for (const started of this.pending.values()) {
      if (oldest === null || started < oldest) oldest = started;
    }
    return oldest === null ? null : now - oldest;
  }

  /** Returns the RTT for a pong, or null if the id is unknown. */
  finish(id: string | null, now: number): number | null {
    if (id == null) return null;
    const started = this.pending.get(id);
    if (started === undefined) return null;
    this.pending.delete(id);
    this.last = now - started;
    return this.last;
  }
}
