/**
 * WebSocket client for the /ws/session endpoint. Streams mic audio to
 * the backend, receives translated audio + live StyleMetadata tags
 * back. See ARCHITECTURE.md for the wire protocol this wraps.
 */

export interface StyleTag {
  tone: string;
  pace: string;
  formality: string;
  emotion: string;
  sarcasm_score: number;
}

export interface VernacularSessionOptions {
  wsUrl: string;
  targetLanguage: string;
  onAudioChunk: (chunk: ArrayBuffer) => void;
  onStyleTag?: (tag: StyleTag) => void;
  onTranscript?: (text: string, isFinal: boolean) => void;
}

export class VernacularSession {
  private ws: WebSocket | null = null;

  constructor(private options: VernacularSessionOptions) {}

  connect(): void {
    const url = `${this.options.wsUrl}/ws/session?target_language=${this.options.targetLanguage}`;
    this.ws = new WebSocket(url);
    this.ws.binaryType = "arraybuffer";

    this.ws.onmessage = (event) => {
      if (event.data instanceof ArrayBuffer) {
        this.options.onAudioChunk(event.data);
      } else {
        // TODO: parse JSON control messages (style tags, transcripts)
        // once the backend wire protocol for those is finalized.
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
