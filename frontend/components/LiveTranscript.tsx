"use client";

/**
 * Displays the live source-language transcript (partial + final) as it
 * streams from the backend, alongside the translated text once
 * available.
 */

interface LiveTranscriptProps {
  sourceText: string;
  translatedText: string;
  isFinal: boolean;
}

export default function LiveTranscript({
  sourceText,
  translatedText,
  isFinal,
}: LiveTranscriptProps) {
  return (
    <div className="w-full max-w-xl space-y-2">
      <p className={isFinal ? "text-neutral-100" : "text-neutral-500"}>
        {sourceText}
      </p>
      <p className="text-emerald-400">{translatedText}</p>
    </div>
  );
}
