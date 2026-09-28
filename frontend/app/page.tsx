"use client";

import { useCallback, useRef, useState } from "react";
import Waveform from "@/components/Waveform";
import LiveTranscript from "@/components/LiveTranscript";
import ToneTags from "@/components/ToneTags";
import { VernacularSession, type StyleTag } from "@/lib/ws-client";
import { PcmPlayer } from "@/lib/player";

/**
 * Main demo UI. Captures mic audio as raw PCM16 frames (matching what
 * backend/app/pipeline/stt.py's AssemblyAISTT expects -- see that
 * file's module docstring for the exact encoding/sample-rate
 * contract), streams it to the backend over VernacularSession, and
 * renders translated audio + live transcript + tone tags as segments
 * arrive.
 *
 * Audio capture uses the Web Audio API (AudioContext +
 * AudioWorkletNode) rather than MediaRecorder, since MediaRecorder
 * only produces compressed formats (webm/opus) and the backend needs
 * raw PCM16 frames at a known sample rate to match AssemblyAISTT's
 * `encoding=pcm_s16le` query param.
 */

const TARGET_SAMPLE_RATE = 16000; // must match AssemblyAISTT's default
const WS_URL = process.env.NEXT_PUBLIC_WS_URL ?? "ws://localhost:8000";

type ConnectionState = "idle" | "connecting" | "live" | "error";

// Inline AudioWorklet processor source. Downsamples from the
// AudioContext's native sample rate to TARGET_SAMPLE_RATE and posts
// PCM16 frames back to the main thread. Kept as a string and loaded
// via a Blob URL so this stays a single-file component without a
// separate worklet asset to manage.
const WORKLET_SOURCE = `
class PcmDownsampler extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.targetSampleRate = options.processorOptions.targetSampleRate;
    this.ratio = sampleRate / this.targetSampleRate;
    this.carry = [];
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;
    const channelData = input[0];

    for (let i = 0; i < channelData.length; i += this.ratio) {
      const index = Math.floor(i);
      if (index < channelData.length) {
        this.carry.push(channelData[index]);
      }
    }

    if (this.carry.length > 0) {
      const pcm16 = new Int16Array(this.carry.length);
      for (let i = 0; i < this.carry.length; i++) {
        const clamped = Math.max(-1, Math.min(1, this.carry[i]));
        pcm16[i] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
      }
      this.port.postMessage(pcm16.buffer, [pcm16.buffer]);
      this.carry = [];
    }

    return true;
  }
}
registerProcessor("pcm-downsampler", PcmDownsampler);
`;

export default function Home() {
  const [targetLanguage, setTargetLanguage] = useState("es");
  const [connectionState, setConnectionState] =
    useState<ConnectionState>("idle");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [sourceText, setSourceText] = useState("");
  const [translatedText, setTranslatedText] = useState("");
  const [styleTag, setStyleTag] = useState<StyleTag | null>(null);
  const [degraded, setDegraded] = useState(false);

  const sessionRef = useRef<VernacularSession | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const workletNodeRef = useRef<AudioWorkletNode | null>(null);
  const playerRef = useRef<PcmPlayer | null>(null);

  const stop = useCallback(() => {
    sessionRef.current?.close();
    sessionRef.current = null;

    workletNodeRef.current?.disconnect();
    workletNodeRef.current = null;

    mediaStreamRef.current?.getTracks().forEach((track) => track.stop());
    mediaStreamRef.current = null;

    audioContextRef.current?.close();
    audioContextRef.current = null;

    playerRef.current?.close();
    playerRef.current = null;

    setConnectionState("idle");
  }, []);

  const start = useCallback(async () => {
    setErrorMessage(null);
    setDegraded(false);
    setConnectionState("connecting");

    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch {
      setConnectionState("error");
      setErrorMessage(
        "Microphone access was denied or unavailable. Grant mic " +
          "permission and try again."
      );
      return;
    }
    mediaStreamRef.current = stream;

    const audioContext = new AudioContext();
    audioContextRef.current = audioContext;

    // Output side. Created here, inside the Start click's call stack,
    // so the browser's autoplay policy lets it run.
    const player = new PcmPlayer();
    void player.resume();
    playerRef.current = player;

    const workletBlob = new Blob([WORKLET_SOURCE], {
      type: "application/javascript",
    });
    const workletUrl = URL.createObjectURL(workletBlob);
    try {
      await audioContext.audioWorklet.addModule(workletUrl);
    } finally {
      URL.revokeObjectURL(workletUrl);
    }

    const session = new VernacularSession({
      wsUrl: WS_URL,
      targetLanguage,
      onReady: (audio) => player.setFormat(audio),
      onAudioChunk: (chunk) => player.push(chunk),
      onSegment: (segment) => {
        setSourceText(segment.source_text);
        setTranslatedText(segment.translated_text);
        setStyleTag(segment.style);
        // Sticky on the backend (it never flaps back to the primary
        // voice mid-session), so mirror that: once degraded, stay
        // flagged until the session is restarted.
        if (segment.degraded) setDegraded(true);
        // This segment's audio frames follow this message. Drop any
        // half-sample left over from the previous segment so it can't
        // shift this one's first sample.
        player.endSegment();
      },
      onError: (message) => {
        setConnectionState("error");
        setErrorMessage(`Session error: ${message}`);
      },
      onProtocolError: (raw) => {
        console.error(
          "vernacular: received a message that didn't match the " +
            "expected wire protocol (see lib/ws-client.ts):",
          raw
        );
      },
    });
    sessionRef.current = session;
    session.connect();

    const source = audioContext.createMediaStreamSource(stream);
    const workletNode = new AudioWorkletNode(audioContext, "pcm-downsampler", {
      processorOptions: { targetSampleRate: TARGET_SAMPLE_RATE },
    });
    workletNodeRef.current = workletNode;

    workletNode.port.onmessage = (event: MessageEvent<ArrayBuffer>) => {
      sessionRef.current?.sendAudioChunk(event.data);
    };

    source.connect(workletNode);
    setConnectionState("live");
  }, [targetLanguage]);

  const isLive = connectionState === "live";

  return (
    <main className="min-h-screen bg-neutral-950 text-neutral-50 flex flex-col items-center justify-center gap-6 p-8">
      <h1 className="text-3xl font-semibold">Vernacular</h1>
      <p className="text-neutral-400 max-w-md text-center">
        Real-time voice translation that preserves how you speak, not
        just what you say.
      </p>

      <div className="flex items-center gap-3">
        <label htmlFor="target-language" className="text-sm text-neutral-400">
          Target language
        </label>
        <select
          id="target-language"
          value={targetLanguage}
          onChange={(e) => setTargetLanguage(e.target.value)}
          disabled={isLive}
          className="rounded bg-neutral-900 px-2 py-1 text-sm"
        >
          <option value="es">Spanish</option>
          <option value="fr">French</option>
          <option value="ja">Japanese</option>
          <option value="de">German</option>
        </select>

        {isLive ? (
          <button
            onClick={stop}
            className="rounded bg-red-700 px-4 py-1.5 text-sm font-medium"
          >
            Stop
          </button>
        ) : (
          <button
            onClick={start}
            disabled={connectionState === "connecting"}
            className="rounded bg-emerald-700 px-4 py-1.5 text-sm font-medium disabled:opacity-50"
          >
            {connectionState === "connecting" ? "Connecting…" : "Start speaking"}
          </button>
        )}
      </div>

      {errorMessage && (
        <p className="text-sm text-red-400 max-w-md text-center">
          {errorMessage}
        </p>
      )}

      {degraded && (
        <p
          role="status"
          className="rounded bg-amber-900/60 px-3 py-1 text-xs text-amber-200 max-w-md text-center"
        >
          Fallback voice active — the primary voice provider failed, so
          speaker identity and delivery style are reduced for the rest of
          this session.
        </p>
      )}

      {isLive && <Waveform />}

      <ToneTags tag={styleTag} />

      <LiveTranscript
        sourceText={sourceText}
        translatedText={translatedText}
        isFinal={true}
      />
    </main>
  );
}
