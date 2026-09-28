/**
 * Pure PCM helpers (no Web Audio dependency, so they're testable in
 * plain Node). The wire carries raw signed 16-bit little-endian mono
 * PCM in arbitrarily-sized binary frames -- see ReadyMessage in
 * ws-client.ts and pipeline/tts.py on the backend.
 */

/**
 * Splits an incoming byte stream into whole 16-bit samples, carrying
 * an odd trailing byte forward to the next chunk.
 *
 * Network frames are NOT guaranteed to be sample-aligned. If one ends
 * mid-sample and we naively read it as Int16, every following sample
 * is byte-shifted and the audio becomes loud static. So we keep the
 * dangling byte and prepend it to the next chunk.
 */
export class PcmAligner {
  private carry: number | null = null;

  /** Returns the whole samples available so far as Float32 in [-1, 1). */
  push(chunk: ArrayBuffer): Float32Array<ArrayBuffer> {
    const incoming = new Uint8Array(chunk);
    let bytes: Uint8Array;
    if (this.carry !== null) {
      bytes = new Uint8Array(incoming.length + 1);
      bytes[0] = this.carry;
      bytes.set(incoming, 1);
      this.carry = null;
    } else {
      bytes = incoming;
    }

    const wholeBytes = bytes.length - (bytes.length % 2);
    if (wholeBytes !== bytes.length) {
      this.carry = bytes[bytes.length - 1];
    }

    const view = new DataView(bytes.buffer, bytes.byteOffset, wholeBytes);
    const out = new Float32Array(new ArrayBuffer(wholeBytes * 2));
    for (let i = 0; i < out.length; i++) {
      out[i] = view.getInt16(i * 2, true) / 0x8000; // little-endian
    }
    return out;
  }

  /** Drops any dangling byte (call at a segment/session boundary). */
  reset(): void {
    this.carry = null;
  }
}

/**
 * Computes when to start the next buffer for gapless playback.
 *
 * - Normally buffers are chained back-to-back at `nextStartTime`.
 * - If that time is already in the past (an underrun: the network
 *   delivered slower than real time, or playback just started),
 *   scheduling there would clip the audio or throw. Resync to "now"
 *   plus a small lead so the first buffer isn't cut off.
 */
export function nextStartTime(
  currentTime: number,
  scheduledUntil: number,
  leadSeconds = 0.05
): number {
  return Math.max(scheduledUntil, currentTime + leadSeconds);
}
