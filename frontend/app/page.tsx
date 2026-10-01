"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Waveform from "@/components/Waveform";
import LiveTranscript, {
  type LiveCaption,
  type TranscriptView,
} from "@/components/LiveTranscript";
import LatencyHud from "@/components/LatencyHud";
import SummaryPanel from "@/components/SummaryPanel";
import ExportMenu from "@/components/ExportMenu";
import {
  VernacularSession,
  type AudioFormat,
  type NetworkTier,
  type SegmentMessage,
  type SummaryMessage,
} from "@/lib/ws-client";
import { PcmPlayer } from "@/lib/player";
import { MAX_AUDIO_BYTES, concatChunks, toExportSegment } from "@/lib/export";
import { RttProbe, summarizeLatency } from "@/lib/latency";
import { TierTracker, classifyNetwork, readConnectionInfo } from "@/lib/network";

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

// Network probe cadence (lib/network.ts, backend/app/adaptive.py).
const PROBE_INTERVAL_MS = 2000;
// A ping unanswered this long counts as a very high RTT: a lost pong
// is itself evidence of a bad link.
const LOST_PONG_MS = 3000;
const SUMMARY_TIMEOUT_MS = 15000;

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
  const [sessionLanguage, setSessionLanguage] = useState("es");
  const [connectionState, setConnectionState] =
    useState<ConnectionState>("idle");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [degraded, setDegraded] = useState(false);

  // Captions + transcript
  const [live, setLive] = useState<LiveCaption | null>(null);
  const [segments, setSegments] = useState<SegmentMessage[]>([]);
  const [view, setView] = useState<TranscriptView>("both");
  const [keyOnly, setKeyOnly] = useState(false);

  // Latency HUD / adaptive streaming
  const [rttMs, setRttMs] = useState<number | null>(null);
  const [tier, setTier] = useState<NetworkTier>("good");
  const [serverSendMs, setServerSendMs] = useState<number | null>(null);

  // Summary
  const [summary, setSummary] = useState<SummaryMessage | null>(null);
  const [summaryPending, setSummaryPending] = useState(false);

  // Export
  const [audioFormat, setAudioFormat] = useState<AudioFormat | null>(null);
  const [audioTruncated, setAudioTruncated] = useState(false);

  const sessionRef = useRef<VernacularSession | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const workletNodeRef = useRef<AudioWorkletNode | null>(null);
  const playerRef = useRef<PcmPlayer | null>(null);
  const probeRef = useRef(new RttProbe());
  const trackerRef = useRef(new TierTracker());
  const probeTimerRef = useRef<number | null>(null);
  const summaryTimerRef = useRef<number | null>(null);
  // Translated audio kept for export, capped (see MAX_AUDIO_BYTES).
  const audioChunksRef = useRef<ArrayBuffer[]>([]);
  const audioBytesRef = useRef(0);

  const stop = useCallback(() => {
    sessionRef.current?.close();
    sessionRef.current = null;

    if (probeTimerRef.current !== null) {
      window.clearInterval(probeTimerRef.current);
      probeTimerRef.current = null;
    }
    if (summaryTimerRef.current !== null) {
      window.clearTimeout(summaryTimerRef.current);
      summaryTimerRef.current = null;
    }
    setSummaryPending(false);

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

  // Release the mic, sockets and timers if the page is left mid-session.
  useEffect(() => stop, [stop]);

  const start = useCallback(async () => {
    setErrorMessage(null);
    setDegraded(false);
    setConnectionState("connecting");

    // A new session starts a new transcript (the previous one stays
    // exportable until now).
    setSegments([]);
    setLive(null);
    setSummary(null);
    setRttMs(null);
    setServerSendMs(null);
    setTier("good");
    setAudioTruncated(false);
    setAudioFormat(null);
    setSessionLanguage(targetLanguage);
    audioChunksRef.current = [];
    audioBytesRef.current = 0;
    probeRef.current = new RttProbe();
    trackerRef.current = new TierTracker();

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
      onReady: (audio) => {
        player.setFormat(audio);
        setAudioFormat(audio);
      },
      onAudioChunk: (chunk) => {
        player.push(chunk);
        // Keep a copy for export, up to the cap.
        if (audioBytesRef.current + chunk.byteLength <= MAX_AUDIO_BYTES) {
          audioChunksRef.current.push(chunk.slice(0));
          audioBytesRef.current += chunk.byteLength;
        } else {
          setAudioTruncated(true);
        }
      },
      onCaption: (caption) => {
        setLive({
          text: caption.text,
          isFinal: caption.is_final,
          segmentId: caption.segment_id,
        });
      },
      onSegment: (segment) => {
        setSegments((prev) => [...prev, segment]);
        // The final caption stays up until ITS translation lands; a
        // newer partial (segmentId null) is a different utterance.
        setLive((prev) =>
          prev && prev.segmentId === segment.segment_id ? null : prev
        );
        // Sticky on the backend (it never flaps back to the primary
        // voice mid-session), so mirror that: once degraded, stay
        // flagged until the session is restarted.
        if (segment.degraded) setDegraded(true);
        // This segment's audio frames follow this message. Drop any
        // half-sample left over from the previous segment so it can't
        // shift this one's first sample.
        player.endSegment();
      },
      onNetwork: (network) => {
        setServerSendMs(network.send_latency_ms);
        setTier(network.tier);
      },
      onPong: (pong) => {
        const rtt = probeRef.current.finish(pong.id, performance.now());
        if (rtt !== null) setRttMs(rtt);
      },
      onSummary: (message) => {
        setSummary(message);
        setSummaryPending(false);
        if (summaryTimerRef.current !== null) {
          window.clearTimeout(summaryTimerRef.current);
          summaryTimerRef.current = null;
        }
      },
      onError: (message) => {
        setConnectionState("error");
        setErrorMessage(`Session error: ${message}`);
      },
      onClose: () => {
        // The server (or network) closed the socket while we were still
        // live: release the mic, timers and audio instead of sitting on
        // a dead session. A close WE initiated has already cleared
        // sessionRef, so this is skipped for the Stop button.
        if (sessionRef.current === session) {
          stop();
          setErrorMessage("The connection to the server was closed.");
        }
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

    // Network monitor: measure RTT with ping/pong, classify the link
    // (RTT, connection type, our own send backlog) and tell the server,
    // which adapts caption rate and audio framing (see lib/network.ts
    // and backend/app/adaptive.py).
    probeTimerRef.current = window.setInterval(() => {
      const active = sessionRef.current;
      if (!active) return;
      const now = performance.now();
      const stalled = probeRef.current.oldestPendingMs(now);
      const rtt =
        stalled !== null && stalled > LOST_PONG_MS
          ? stalled
          : probeRef.current.last;
      const observed = classifyNetwork({
        rttMs: rtt,
        ...readConnectionInfo(),
        bufferedBytes: active.bufferedAmount,
      });
      const changed = trackerRef.current.update(observed);
      if (changed) active.sendNetworkReport(changed, rtt);
      active.sendPing(probeRef.current.start(now));
    }, PROBE_INTERVAL_MS);

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
  }, [targetLanguage, stop]);

  const requestSummary = useCallback(() => {
    if (!sessionRef.current) return;
    setSummaryPending(true);
    sessionRef.current.requestSummary("target");
    if (summaryTimerRef.current !== null) {
      window.clearTimeout(summaryTimerRef.current);
    }
    // Don't leave the button stuck if the reply never comes.
    summaryTimerRef.current = window.setTimeout(() => {
      setSummaryPending(false);
      summaryTimerRef.current = null;
    }, SUMMARY_TIMEOUT_MS);
  }, []);

  const getAudio = useCallback(
    () =>
      audioChunksRef.current.length > 0
        ? concatChunks(audioChunksRef.current)
        : null,
    []
  );

  const exportSegments = useMemo(() => segments.map(toExportSegment), [segments]);
  const latency = useMemo(
    () => summarizeLatency(segments.map((s) => s.timings)),
    [segments]
  );

  const isLive = connectionState === "live";

  return (
    <main className="min-h-screen bg-neutral-950 text-neutral-50 flex flex-col items-center gap-6 p-8">
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

      {tier !== "good" && isLive && (
        <p
          role="status"
          className="rounded bg-neutral-800 px-3 py-1 text-xs text-neutral-300 max-w-md text-center"
        >
          Slow connection detected — captions are updating less often and
          audio is sent in larger pieces. Nothing is being dropped.
        </p>
      )}

      {isLive && <Waveform />}

      <LatencyHud
        summary={latency}
        rttMs={rttMs}
        tier={tier}
        serverSendMs={serverSendMs}
      />

      <LiveTranscript
        live={live}
        segments={segments}
        view={view}
        onViewChange={setView}
        keyOnly={keyOnly}
        onKeyOnlyChange={setKeyOnly}
      />

      <SummaryPanel
        summary={summary}
        pending={summaryPending}
        disabled={!isLive || segments.length === 0}
        onRequest={requestSummary}
      />

      <ExportMenu
        segments={exportSegments}
        targetLanguage={sessionLanguage}
        getAudio={getAudio}
        audioFormat={audioFormat}
        audioTruncated={audioTruncated}
      />
    </main>
  );
}
