/**
 * Client-side network quality: classifies the link and reports the tier
 * to the server, which adapts its framing (backend/app/adaptive.py).
 * The server also measures congestion itself from its own send times,
 * and the WORSE of the two signals wins -- this side contributes what
 * only the browser can see: round-trip time, the connection type and
 * how much the socket has queued but not yet sent.
 */

import type { NetworkTier } from "./ws-client";

export interface NetworkSample {
  rttMs: number | null;
  /** navigator.connection.effectiveType, when the browser exposes it. */
  effectiveType?: string | null;
  saveData?: boolean;
  /** WebSocket.bufferedAmount in bytes. */
  bufferedBytes?: number;
}

const RANK: Record<NetworkTier, number> = { good: 0, fair: 1, poor: 2 };
const worse = (a: NetworkTier, b: NetworkTier): NetworkTier =>
  RANK[a] >= RANK[b] ? a : b;

// Uplink audio is ~32 KB/s (16 kHz PCM16); a backlog of a second or
// more means the uplink is not keeping up.
const BUFFER_FAIR_BYTES = 32_000;
const BUFFER_POOR_BYTES = 128_000;

export function classifyNetwork(sample: NetworkSample): NetworkTier {
  let tier: NetworkTier = "good";
  if (sample.rttMs != null) {
    if (sample.rttMs >= 600) tier = worse(tier, "poor");
    else if (sample.rttMs >= 250) tier = worse(tier, "fair");
  }
  switch (sample.effectiveType) {
    case "slow-2g":
    case "2g":
      tier = worse(tier, "poor");
      break;
    case "3g":
      tier = worse(tier, "fair");
      break;
    default:
      break;
  }
  if (sample.saveData) tier = worse(tier, "fair");
  const buffered = sample.bufferedBytes ?? 0;
  if (buffered >= BUFFER_POOR_BYTES) tier = worse(tier, "poor");
  else if (buffered >= BUFFER_FAIR_BYTES) tier = worse(tier, "fair");
  return tier;
}

/**
 * Hysteresis on the client side too: report immediately when the link
 * gets worse, but only after `recoverAfter` consecutive better samples
 * when it gets better, so a noisy RTT does not flap the stream.
 */
export class TierTracker {
  private tier: NetworkTier = "good";
  private betterStreak = 0;

  constructor(private recoverAfter = 3) {}

  get current(): NetworkTier {
    return this.tier;
  }

  /** Returns the new tier if it changed, otherwise null. */
  update(observed: NetworkTier): NetworkTier | null {
    if (RANK[observed] > RANK[this.tier]) {
      this.tier = observed;
      this.betterStreak = 0;
      return this.tier;
    }
    if (RANK[observed] < RANK[this.tier]) {
      this.betterStreak += 1;
      if (this.betterStreak >= this.recoverAfter) {
        this.tier = observed;
        this.betterStreak = 0;
        return this.tier;
      }
      return null;
    }
    this.betterStreak = 0;
    return null;
  }
}

/** Reads the (non-standard, Chromium-only) Network Information API. */
export function readConnectionInfo(): Pick<NetworkSample, "effectiveType" | "saveData"> {
  const conn = (
    navigator as Navigator & {
      connection?: { effectiveType?: string; saveData?: boolean };
    }
  ).connection;
  return { effectiveType: conn?.effectiveType ?? null, saveData: conn?.saveData ?? false };
}
