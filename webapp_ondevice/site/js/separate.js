/**
 * Chunked 2-speaker separation on ONNX Runtime Web.
 *
 * Contract of both shipping separators (`build/NOTES.md` §1):
 *   input   "mixture"    float32 [1, 32000]      4 s @ 8 kHz, STATIC shape
 *   output  "separated"  float32 [1, 2, 32000]   speaker-major, C-contiguous
 *
 * A routed region is normally exactly one chunk (`expand_to_chunk` pads to
 * 4.0 s), so the chunker below is the *long-region* path — real conversation
 * produces overlap regions longer than 4 s and the padder returns those
 * unpadded (`routing.js::expandToChunk`).
 *
 * Inherited from the January PoC (`mobile/webapp/app.js`, recon §2.3): the 4 s
 * window / 2 s hop and the linear crossfade, which is COLA-correct — with
 * hop = chunk/2 the fade-out and fade-in weights of two neighbouring chunks sum
 * to exactly 1. Three things the PoC got wrong are fixed here:
 *
 * 1. **Cross-chunk permutation flips** (recon §5a: measured 1/15 junctions on a
 *    30 s clip for two of three models). A PIT-trained separator emits its two
 *    streams in an arbitrary order *per input*, so taking ONNX channel 0 as
 *    "speaker 1" in every chunk blends A into B across a flipped junction.
 *    `alignToPrevious` correlates the new chunk's overlap head against the
 *    previous chunk's overlap tail in both orderings and swaps when the swap
 *    wins.
 * 2. **Normalisation.** These models are SI-SDR-trained and therefore
 *    scale-free — MF2 answers a peak-1.0 input with peaks of 3+ (and the recon
 *    measured ConvTasNet at ~111). The PoC peak-normalised the whole *blended*
 *    stream, per speaker independently, which destroys the relative level of
 *    the two speakers and lets one loud instant crush the clip. Here each chunk
 *    is peak-normalised on the way *in* and multiplied back by the same peak on
 *    the way *out*, so the two streams stay in the mixture's own scale, stay
 *    comparable to each other, and every chunk is blended in one consistent
 *    scale.
 * 3. **Runt final chunk.** The PoC's `i += hop while i < length` loop emitted a
 *    final chunk that could be 96 % zero padding — one wasted 4 s inference per
 *    clip. `chunkStarts` stops as soon as the region is covered.
 *
 * Nothing here clamps or normalises the returned audio: playback and WAV export
 * are the caller's business (`wav.js`), and a demo about separation quality
 * must not silently rescale what it is showing.
 */
import { ort } from './ortEnv.js';

/** Separator input window: 4 s @ 8 kHz. */
export const CHUNK_SAMPLES = 32000;
/** 50 % overlap — the hop the COLA crossfade assumes. */
export const HOP_SAMPLES = CHUNK_SAMPLES / 2;
/** Separator sample rate. */
export const SEPARATOR_SAMPLE_RATE = 8000;
/** Below this peak a chunk is treated as silent and fed through at unit gain. */
const SILENCE_PEAK = 1e-8;

/**
 * Chunk start offsets covering `length` samples with no runt tail.
 * @param {number} length
 * @returns {number[]}
 */
export function chunkStarts(length) {
  if (length <= CHUNK_SAMPLES) return [0];
  const n = Math.ceil((length - CHUNK_SAMPLES) / HOP_SAMPLES) + 1;
  return Array.from({ length: n }, (_, i) => i * HOP_SAMPLES);
}

/**
 * One separator call under the peak-gain convention.
 *
 * @param {object} session ORT `InferenceSession` on a separator model
 * @param {Float32Array} chunk8k exactly `CHUNK_SAMPLES` samples (zero-padded by
 *        the caller if the region ran out)
 * @returns {Promise<[Float32Array, Float32Array]>} both streams, in the input's
 *          own scale
 */
export async function separateChunk(session, chunk8k) {
  if (chunk8k.length !== CHUNK_SAMPLES) {
    throw new Error(`separator chunk must be ${CHUNK_SAMPLES} samples, got ${chunk8k.length}`);
  }
  let peak = 0;
  for (let i = 0; i < chunk8k.length; i++) {
    const a = Math.abs(chunk8k[i]);
    if (a > peak) peak = a;
  }
  const gain = peak > SILENCE_PEAK ? 1 / peak : 1;
  const scaled = new Float32Array(CHUNK_SAMPLES);
  for (let i = 0; i < CHUNK_SAMPLES; i++) scaled[i] = chunk8k[i] * gain;

  const results = await session.run({
    mixture: new ort.Tensor('float32', scaled, [1, CHUNK_SAMPLES]),
  });
  const out = results.separated;
  if (!out) throw new Error('separator returned no "separated" output');
  const data = out.data;
  const inverse = 1 / gain;
  const s1 = new Float32Array(CHUNK_SAMPLES);
  const s2 = new Float32Array(CHUNK_SAMPLES);
  for (let i = 0; i < CHUNK_SAMPLES; i++) {
    s1[i] = data[i] * inverse;
    s2[i] = data[CHUNK_SAMPLES + i] * inverse;
  }
  return [s1, s2];
}

/**
 * Swap a chunk's two streams if they are permuted relative to the previous
 * chunk, judged over the samples the two chunks share.
 *
 * `prev[HOP..CHUNK)` and `cur[0..HOP)` cover the same 2 s of audio, so the
 * correct pairing is simply the one with the larger summed inner product.
 *
 * @param {[Float32Array, Float32Array]} cur modified in place (order only)
 * @param {[Float32Array, Float32Array]} prev already-aligned previous chunk
 * @returns {boolean} true if the chunk was swapped
 */
export function alignToPrevious(cur, prev) {
  let same = 0;
  let swapped = 0;
  for (let i = 0; i < HOP_SAMPLES; i++) {
    const p1 = prev[0][HOP_SAMPLES + i];
    const p2 = prev[1][HOP_SAMPLES + i];
    const c1 = cur[0][i];
    const c2 = cur[1][i];
    same += p1 * c1 + p2 * c2;
    swapped += p1 * c2 + p2 * c1;
  }
  if (swapped > same) {
    const tmp = cur[0];
    cur[0] = cur[1];
    cur[1] = tmp;
    return true;
  }
  return false;
}

/**
 * COLA weight of sample `j` of chunk `i` (of `n`), the PoC's linear crossfade.
 * First chunk has no fade-in, last chunk no fade-out; everywhere else the two
 * overlapping weights sum to exactly 1.
 */
function blendWeight(j, i, n) {
  if (i > 0 && j < HOP_SAMPLES) return j / HOP_SAMPLES;
  if (i < n - 1 && j >= HOP_SAMPLES) return 1 - (j - HOP_SAMPLES) / HOP_SAMPLES;
  return 1;
}

/**
 * Separate one region of 8 kHz audio into two speaker streams.
 *
 * Regions shorter than one chunk are zero-padded for inference and trimmed
 * back afterwards; longer regions are chunked (4 s window, 2 s hop),
 * permutation-aligned at every junction, and crossfaded.
 *
 * @param {Float32Array} audio8k mono 8 kHz region (any length >= 1 sample)
 * @param {object} session ORT `InferenceSession` on a separator model
 * @param {{onChunk?: (done: number, total: number) => (void|Promise<void>)}} [opts]
 *        `onChunk` is awaited after each chunk, for progress UI. Awaiting it is
 *        what lets a caller on the main thread repaint between chunks (a plain
 *        callback only gets a microtask, which does not reach the renderer) and
 *        what lets it cancel a long region by throwing from the callback.
 * @returns {Promise<[Float32Array, Float32Array]>} two streams, each exactly
 *          `audio8k.length` samples, in the input's own scale
 */
export async function separateRegion(audio8k, session, opts = {}) {
  const length = audio8k.length;
  if (length === 0) return [new Float32Array(0), new Float32Array(0)];
  const starts = chunkStarts(length);
  const n = starts.length;
  const total = (n - 1) * HOP_SAMPLES + CHUNK_SAMPLES;
  const acc1 = new Float32Array(total);
  const acc2 = new Float32Array(total);
  const buffer = new Float32Array(CHUNK_SAMPLES);

  let prev = null;
  for (let i = 0; i < n; i++) {
    const start = starts[i];
    buffer.fill(0); // zero-pad a region that ends mid-chunk
    buffer.set(audio8k.subarray(start, Math.min(start + CHUNK_SAMPLES, length)));
    const cur = await separateChunk(session, buffer);
    if (prev) alignToPrevious(cur, prev);
    for (let j = 0; j < CHUNK_SAMPLES; j++) {
      const w = blendWeight(j, i, n);
      acc1[start + j] += cur[0][j] * w;
      acc2[start + j] += cur[1][j] * w;
    }
    prev = cur;
    if (opts.onChunk) await opts.onChunk(i + 1, n);
  }
  // Trim the zero-padded tail introduced by the last chunk.
  return [acc1.slice(0, length), acc2.slice(0, length)];
}
