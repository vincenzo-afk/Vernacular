import assert from "node:assert/strict";
import { test } from "node:test";
import { PcmAligner, nextStartTime } from "../lib/pcm";

const i16le = (...samples: number[]): ArrayBuffer => {
  const buf = new ArrayBuffer(samples.length * 2);
  const v = new DataView(buf);
  samples.forEach((s, i) => v.setInt16(i * 2, s, true));
  return buf;
};

test("decodes little-endian int16 to float in [-1, 1)", () => {
  const out = new PcmAligner().push(i16le(0, 16384, -16384, -32768));
  assert.deepEqual(Array.from(out), [0, 0.5, -0.5, -1]);
});

test("odd-length chunk carries the dangling byte to the next chunk", () => {
  // Two samples (1000, -2000) = 4 bytes, split 3 + 1 across frames.
  const whole = new Uint8Array(i16le(1000, -2000));
  const a = new PcmAligner();
  const first = a.push(whole.slice(0, 3).buffer);
  const second = a.push(whole.slice(3).buffer);

  assert.equal(first.length, 1); // only one whole sample so far
  assert.equal(second.length, 1);
  assert.equal(Math.round(first[0] * 0x8000), 1000);
  assert.equal(Math.round(second[0] * 0x8000), -2000);
});

test("byte-shifted decode would corrupt: aligner output equals the aligned decode", () => {
  const samples = [123, -456, 789, -1011, 1213];
  const bytes = new Uint8Array(i16le(...samples));
  const aligned = Array.from(new PcmAligner().push(bytes.buffer));

  // Feed the same bytes one byte at a time (worst-case framing).
  const a = new PcmAligner();
  const trickled: number[] = [];
  for (const b of bytes) trickled.push(...a.push(new Uint8Array([b]).buffer));

  assert.deepEqual(trickled, aligned);
});

test("reset drops a dangling byte so it can't poison the next segment", () => {
  const a = new PcmAligner();
  a.push(new Uint8Array([0x7f]).buffer); // dangling half-sample
  a.reset();
  const out = a.push(i16le(500));
  assert.equal(Math.round(out[0] * 0x8000), 500);
});

test("nextStartTime chains gaplessly when ahead of the clock", () => {
  assert.equal(nextStartTime(1.0, 1.5), 1.5);
});

test("nextStartTime resyncs with a lead after an underrun", () => {
  // Scheduled-until (1.0) is already in the past relative to now (2.0).
  assert.equal(nextStartTime(2.0, 1.0, 0.05), 2.05);
});
