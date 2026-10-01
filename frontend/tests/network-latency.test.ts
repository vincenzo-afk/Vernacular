import assert from "node:assert/strict";
import { test } from "node:test";
import { TierTracker, classifyNetwork } from "../lib/network";
import {
  RttProbe,
  criticalPathBars,
  percentile,
  summarizeLatency,
  verdictFor,
} from "../lib/latency";
import type { TimingsTag } from "../lib/ws-client";

test("classifyNetwork: rtt, connection type and backlog each degrade the tier", () => {
  assert.equal(classifyNetwork({ rttMs: 40 }), "good");
  assert.equal(classifyNetwork({ rttMs: 300 }), "fair");
  assert.equal(classifyNetwork({ rttMs: 900 }), "poor");
  assert.equal(classifyNetwork({ rttMs: 20, effectiveType: "3g" }), "fair");
  assert.equal(classifyNetwork({ rttMs: 20, effectiveType: "2g" }), "poor");
  assert.equal(classifyNetwork({ rttMs: 20, saveData: true }), "fair");
  assert.equal(classifyNetwork({ rttMs: 20, bufferedBytes: 200_000 }), "poor");
  assert.equal(classifyNetwork({ rttMs: null }), "good");
});

test("the worst signal wins", () => {
  assert.equal(
    classifyNetwork({ rttMs: 300, effectiveType: "2g", bufferedBytes: 0 }),
    "poor"
  );
});

test("TierTracker degrades at once but recovers only after a streak", () => {
  const t = new TierTracker(3);
  assert.equal(t.update("poor"), "poor");
  assert.equal(t.update("good"), null);
  assert.equal(t.update("good"), null);
  assert.equal(t.update("good"), "good");
  // a bad sample in the middle resets the streak
  t.update("poor");
  t.update("good");
  t.update("poor");
  assert.equal(t.update("good"), null);
});

const timings = (total: number, over: Partial<TimingsTag> = {}): TimingsTag => ({
  queue_wait_ms: 0,
  style_wait_ms: 10,
  register_detection_ms: 300,
  translation_ms: 500,
  tts_first_byte_ms: 400,
  server_total_ms: total,
  ...over,
});

test("verdictFor: ok / warn above 80% / over", () => {
  assert.equal(verdictFor(300, 500), "ok");
  assert.equal(verdictFor(450, 500), "warn");
  assert.equal(verdictFor(501, 500), "over");
});

test("percentile interpolates and rejects empty input", () => {
  assert.equal(percentile([100, 200, 300, 400], 50), 250);
  assert.equal(percentile([5], 90), 5);
  assert.throws(() => percentile([], 50));
});

test("summarizeLatency reports p50/p90/max and last verdict", () => {
  assert.equal(summarizeLatency([]), null);
  const s = summarizeLatency([timings(900), timings(1200), timings(2500)])!;
  assert.equal(s.count, 3);
  assert.equal(s.maxTotalMs, 2500);
  assert.equal(s.p50TotalMs, 1200);
  assert.equal(s.lastVerdict, "over");
});

test("criticalPathBars orders queue, tone wait, translate, voice", () => {
  const bars = criticalPathBars(timings(1000, { tts_first_byte_ms: null }));
  assert.deepEqual(bars.map((b) => b.key), ["queue", "style", "translation", "tts"]);
  assert.equal(bars[3].ms, 0); // unknown TTFB is not invented
  assert.equal(bars[0].verdict, null); // unbudgeted stages carry no verdict
});

test("RttProbe pairs pongs with pings and ignores unknown ids", () => {
  const p = new RttProbe();
  const id = p.start(1000);
  assert.equal(p.finish("nope", 1100), null);
  assert.equal(p.finish(id, 1080), 80);
  assert.equal(p.finish(id, 1200), null); // already consumed
  assert.equal(p.last, 80);
});

test("RttProbe exposes the age of an unanswered ping (a lost pong is a signal)", () => {
  const p = new RttProbe();
  assert.equal(p.oldestPendingMs(1000), null);
  const a = p.start(1000);
  p.start(1500);
  assert.equal(p.oldestPendingMs(4000), 3000);
  p.finish(a, 1100);
  assert.equal(p.oldestPendingMs(4000), 2500);
});
