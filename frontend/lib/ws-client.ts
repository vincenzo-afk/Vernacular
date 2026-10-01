/**
 * WebSocket client for the /ws/session endpoint. Streams mic audio to
 * the backend, receives translated audio + live StyleMetadata tags
 * back.
 *
 * Wire format (backend -> frontend), matching backend/app/ws/session.py
 * exactly. First message of every session:
 *   {"type": "ready", "audio": {"encoding": "pcm_s16le",
 *                               "sample_rate": 24000, "channels": 1}}
 * declaring the format of every binary frame that follows. — see that file's module docstring as the source of truth if
 * these ever drift (see CONTRIBUTING.md "Known gaps": this sync is
 * currently manual, there's no shared schema codegen):
 *
 *   One JSON text message per completed segment:
 *     {"type": "segment", "source_text": "...", "translated_text": "...",
 *      "style": {tone, pace, formality, emotion, sarcasm_score,
 *                emphasis_words, pause_pattern, confidence},
 *      "degraded": boolean}
 *   immediately followed by that segment's binary audio-chunk messages:
 *     <ArrayBuffer>
 *     <ArrayBuffer>
 *     ...
 *
 *   Or, on an unrecoverable session error:
 *     {"type": "error", "message": "session_failed"}
 *
 * Also: `caption` (live source-language transcript), `network` (the
 * server's adaptive-streaming tier changed), `pong` (reply to a
 * latency probe) and `summary` (reply to a summarize request). Client
 * -> server control messages: `network`, `ping`, `summarize` (see
 * backend/app/ws/session.py for the authoritative description).
 */

/** Declared by the server in the `ready` message; describes every
 * binary frame that follows. The client never hardcodes this -- the
 * backend (pipeline/tts.py) is the single source of truth. */
export interface AudioFormat {
  encoding: "pcm_s16le";
  sample_rate: number;
  channels: number;
}

export interface ReadyMessage {
  type: "ready";
  audio: AudioFormat;
}

export interface AcousticTag {
  arousal_score: number;
  energy_db: number;
  energy_delta_db: number;
  energy_variability_db: number;
  pitch_mean_hz: number | null;
  pitch_delta_pct: number | null;
  pitch_range_hz: number | null;
  voiced_ratio: number;
  duration_ms: number;
}

export type CueKind =
  | "lexical"
  | "prosodic"
  | "acoustic"
  | "contextual"
  | "incongruence";

export interface CueTag {
  kind: CueKind;
  evidence: string;
  weight: number;
  /** "llm" = the model's own (unverified) account; "measured" =
   * computed from the audio / word timings and checkable. The UI keeps
   * the two visually distinct. */
  source: "llm" | "measured";
}

export interface ExplanationTag {
  summary: string;
  cues: CueTag[];
  context_used: boolean;
}

export interface StyleTag {
  tone: string;
  pace: string;
  formality: string;
  emotion: string;
  sarcasm_score: number;
  emphasis_words: string[];
  pause_pattern: string;
  confidence: number;
}

export interface SegmentMessage {
  type: "segment";
  source_text: string;
  translated_text: string;
  style: StyleTag;
  /** True when the backend synthesized this segment with its fallback
   * TTS provider (stock voice, reduced style control). Must be
   * surfaced in the UI -- degraded fidelity is never presented as
   * full quality (ARCHITECTURE.md §5). */
  degraded: boolean;
}

export interface ErrorMessage {
  type: "error";
  message: string;
}

type ControlMessage = ReadyMessage | SegmentMessage | ErrorMessage;

function isControlMessage(value: unknown): value is ControlMessage {
  return (
    typeof value === "object" &&
    value !== null &&
    "type" in value &&
    ((value as { type: unknown }).type === "ready" ||
      (value as { type: unknown }).type === "segment" ||
      (value as { type: unknown }).type === "error")
  );
}

export interface VernacularSessionOptions {
  wsUrl: string;
  targetLanguage: string;
  onAudioChunk: (chunk: ArrayBuffer) => void;
  onReady?: (audio: AudioFormat) => void;
  onSegment?: (segment: SegmentMessage) => void;
  onError?: (message: string) => void;
  /** Fired if a message from the server doesn't match the expected
   * shape — indicates the frontend/backend wire format has drifted;
   * see the module docstring above. */
  onProtocolError?: (raw: string) => void;
}

export class VernacularSession {
  private ws: WebSocket | null = null;

  constructor(private options: VernacularSessionOptions) {}

  connect(): void {
    const url = `${this.options.wsUrl}/ws/session?target_language=${encodeURIComponent(
      this.options.targetLanguage
    )}`;
    this.ws = new WebSocket(url);
    this.ws.binaryType = "arraybuffer";

    this.ws.onmessage = (event) => {
      if (event.data instanceof ArrayBuffer) {
        this.options.onAudioChunk(event.data);
        return;
      }

      if (typeof event.data !== "string") {
        // Neither binary nor text -- shouldn't happen with
        // binaryType="arraybuffer", but don't silently drop it.
        this.options.onProtocolError?.(String(event.data));
        return;
      }

      let parsed: unknown;
      try {
        parsed = JSON.parse(event.data);
      } catch {
        this.options.onProtocolError?.(event.data);
        return;
      }

      if (!isControlMessage(parsed)) {
        this.options.onProtocolError?.(event.data);
        return;
      }

      if (parsed.type === "ready") {
        this.options.onReady?.(parsed.audio);
      } else if (parsed.type === "segment") {
        this.options.onSegment?.(parsed);
      } else {
        this.options.onError?.(parsed.message);
      }
    };
  }

  sendAudioChunk(chunk: ArrayBuffer): void {
    this.ws?.send(chunk);
  }

  close(): void {
    this.ws?.close();
  }
}
