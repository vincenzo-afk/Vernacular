"use client";

/**
 * Main demo UI. See README.md for the demo UI extras this should
 * include: live transcript, waveform visualization, real-time tone
 * tags.
 */

export default function Home() {
  return (
    <main className="min-h-screen bg-neutral-950 text-neutral-50 flex flex-col items-center justify-center gap-6 p-8">
      <h1 className="text-3xl font-semibold">Vernacular</h1>
      <p className="text-neutral-400 max-w-md text-center">
        Real-time voice translation that preserves how you speak, not
        just what you say.
      </p>
      {/* TODO: mic capture, <Waveform />, <LiveTranscript />, <ToneTags /> */}
    </main>
  );
}
