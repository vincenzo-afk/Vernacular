import { PcmAligner, nextStartTime } from "./pcm";
import type { AudioFormat } from "./ws-client";

/**
 * Plays the translated PCM audio the backend streams back.
 *
 * Segments arrive as a burst of arbitrarily-sized binary frames, not a
 * smooth real-time stream, so each frame is decoded (see PcmAligner
 * for the odd-byte handling) into an AudioBuffer and scheduled
 * back-to-back on the AudioContext clock for gapless playback.
 *
 * The sample rate comes from the server's `ready` message, never a
 * hardcoded constant. The browser resamples to the output device rate
 * automatically when the buffer's rate differs from the context's.
 */
export class PcmPlayer {
  private ctx: AudioContext;
  private aligner = new PcmAligner();
  private format: AudioFormat | null = null;
  private scheduledUntil = 0;
  private active = new Set<AudioBufferSourceNode>();

  /** Must be constructed from a user gesture (e.g. the Start click),
   * or browsers will leave the context suspended (autoplay policy). */
  constructor(ctx: AudioContext = new AudioContext()) {
    this.ctx = ctx;
  }

  setFormat(format: AudioFormat): void {
    this.format = format;
    this.aligner.reset();
  }

  async resume(): Promise<void> {
    if (this.ctx.state === "suspended") await this.ctx.resume();
  }

  push(chunk: ArrayBuffer): void {
    if (!this.format) {
      // Server contract: `ready` always precedes binary frames. If we
      // see audio first, drop it rather than guess a format and play
      // garbage at the wrong pitch.
      console.warn("vernacular: audio frame before `ready`; dropping");
      return;
    }

    const samples = this.aligner.push(chunk);
    if (samples.length === 0) return;

    const buffer = this.ctx.createBuffer(
      this.format.channels,
      samples.length,
      this.format.sample_rate
    );
    buffer.copyToChannel(samples, 0);

    const source = this.ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(this.ctx.destination);

    const startAt = nextStartTime(this.ctx.currentTime, this.scheduledUntil);
    source.start(startAt);
    this.scheduledUntil = startAt + buffer.duration;

    this.active.add(source);
    source.onended = () => this.active.delete(source);
  }

  /** Segment boundary: a half-sample left over must not bleed into the
   * next segment's first sample. */
  endSegment(): void {
    this.aligner.reset();
  }

  /** Cancels everything queued or playing (Stop button). */
  stop(): void {
    for (const source of this.active) {
      try {
        source.stop();
      } catch {
        // Already ended -- fine.
      }
    }
    this.active.clear();
    this.scheduledUntil = 0;
    this.aligner.reset();
  }

  close(): void {
    this.stop();
    void this.ctx.close();
  }
}
