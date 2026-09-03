/**
 * Overlapped-speech detection: pyannote-segmentation-3.0 on ONNX Runtime Web.
 *
 * Port of the inference half of `webapp_ondevice/reference/osd_reference.py`:
 *   - I/O contract              (reference lines 19-24, 182-199)
 *   - sliding window + stitch   (`infer_logits`, reference lines 214-249)
 *   - powerset decode table     (`decode_classes`, reference lines 162-166, 252-257)
 *
 * Contract:
 *   input   "input_values"  float32 [1, 1, 160000]   10 s mono @ 16 kHz
 *   output  "logits"        float32 [1, 589, 7]      LogSoftmax, so all <= 0
 * Both boundary tensors are float32 even in the fp16 file (the conversion wraps
 * the graph in Cast nodes), so there is no float16 packing to do here.
 *
 * The window strategy is 10 s windows with a hop of 79920 samples
 * (= 296 frames exactly) and centre-cropping: every output frame comes from
 * exactly one window, so stitching is index arithmetic with no aggregation
 * weights. The hop MUST be a whole number of frames — the "nice" 5 s hop of
 * 80000 samples is not a multiple of 270 and would smear the frame grid by a
 * fraction of a frame per window. See reference/README.md §3.
 *
 * We only ever use the speaker COUNT, never the identity, which is what removes
 * clustering, embeddings and cross-window permutation alignment from the demo.
 */
import { createSession, MODELS_URL, ort } from './ortEnv.js';
import {
  FRAME_STEP_SAMPLES,
  FRAME_STEP_S,
  SAMPLE_RATE,
  frameCount,
} from './routing.js';

/** The model's training window: 10 s. Fixed shape avoids ORT re-specialisation. */
export const WINDOW_SAMPLES = 160000;
/** Frames one window emits: `frameCount(160000)`. */
export const FRAMES_PER_WINDOW = 589;
/** Window hop in frames — half a window, rounded down to whole frames. */
export const HOP_FRAMES = 296;
/** Window hop in samples: 296 x 270 = 79920 (4.995 s). */
export const HOP_SAMPLES = HOP_FRAMES * FRAME_STEP_SAMPLES;
/** Central crop kept from each window: local frames [146, 442). */
export const KEEP_START = (FRAMES_PER_WINDOW - HOP_FRAMES) >> 1;
export const KEEP_END = KEEP_START + HOP_FRAMES;
/** Powerset class -> number of active speakers (model card `id2label` order). */
export const POWERSET_SPEAKER_COUNT = [0, 1, 1, 1, 2, 2, 2];
/** Number of powerset classes the head emits. */
export const NUM_CLASSES = 7;

export const OSD_MODEL_URL = new URL('pyannote_seg3_fp16.onnx', MODELS_URL).href;

/**
 * Create the OSD session. `graphOptimizationLevel: 'extended'` is not optional
 * for this graph — see `ortEnv.js`.
 * @param {string|Uint8Array|ArrayBuffer} [url] URL, or the model's bytes when
 *        the caller fetched them itself (see `models.js`).
 * @returns {Promise<import('../vendor/ort/types.d.ts').InferenceSession>}
 */
export function createOsdSession(url = OSD_MODEL_URL) {
  return createSession(url, { graphOptimizationLevel: 'extended' });
}

/**
 * Run OSD over arbitrary-length mono 16 kHz audio.
 *
 * The session is a required argument rather than a module-level singleton: the
 * app runs one model in memory at a time (12 MB of WASM heap for the pyannote
 * graph, 46 MB for MossFormer2), so whoever owns the lifecycle has to be able
 * to `release()` it. `createOsdSession()` above is the one-liner for callers
 * that just want the default model.
 *
 * @param {Float32Array} audio16k mono, 16 kHz, nominally in [-1, 1]
 * @param {object} session an ORT `InferenceSession` on the pyannote model
 * @param {{keepLogits?: boolean, onWindow?: (done: number, total: number) =>
 *          (void|Promise<void>)}} [opts]
 *        `keepLogits` also returns the stitched `[numFrames, 7]`
 *        log-probabilities (frames row-major) — ~28 KB/s of audio, so it is off
 *        by default. `onWindow` is awaited after each window, mirroring
 *        `separate.js::separateRegion`'s `onChunk`: inference runs on the main
 *        thread, so a UI that wants to repaint (or cancel) between windows has
 *        to be given a turn, and only an awaited callback can take one.
 * @returns {Promise<{frameClasses: Uint8Array, frameStepS: number,
 *                    numFrames: number, durationS: number,
 *                    powerset: Uint8Array, logits?: Float32Array}>}
 *          `frameClasses` is 0 nonspeech / 1 solo / 2 overlap;
 *          `powerset` is the raw 7-class argmax (kept because it is what the
 *          parity vectors pin, and it is one byte per frame).
 */
export async function runOsd(audio16k, session, opts = {}) {
  const total = frameCount(audio16k.length);
  const empty = {
    frameClasses: new Uint8Array(0),
    frameStepS: FRAME_STEP_S,
    numFrames: 0,
    durationS: audio16k.length / SAMPLE_RATE,
    powerset: new Uint8Array(0),
  };
  if (total === 0) return opts.keepLogits ? { ...empty, logits: new Float32Array(0) } : empty;

  const nWindows =
    audio16k.length <= WINDOW_SAMPLES
      ? 1
      : Math.ceil((audio16k.length - WINDOW_SAMPLES) / HOP_SAMPLES) + 1;

  const frameClasses = new Uint8Array(total);
  const powerset = new Uint8Array(total);
  const logits = opts.keepLogits ? new Float32Array(total * NUM_CLASSES) : null;
  const written = new Uint8Array(total);
  const buffer = new Float32Array(WINDOW_SAMPLES); // reused, zero-padded tail

  for (let w = 0; w < nWindows; w++) {
    const start = w * HOP_SAMPLES;
    buffer.fill(0);
    buffer.set(audio16k.subarray(start, Math.min(start + WINDOW_SAMPLES, audio16k.length)));
    const out = await runWindow(session, buffer);

    // Centre-crop: window 0 also supplies its leading frames, the last window
    // everything to the end (reference:240-247).
    const lo = w === 0 ? 0 : KEEP_START;
    const hi = w === nWindows - 1 ? FRAMES_PER_WINDOW : KEEP_END;
    for (let i = lo; i < hi; i++) {
      const g = w * HOP_FRAMES + i;
      if (g >= total) break; // frames that exist only because of zero padding
      let best = 0;
      let bestVal = out[i * NUM_CLASSES];
      for (let c = 1; c < NUM_CLASSES; c++) {
        const v = out[i * NUM_CLASSES + c];
        if (v > bestVal) {
          bestVal = v;
          best = c;
        }
      }
      powerset[g] = best;
      frameClasses[g] = POWERSET_SPEAKER_COUNT[best];
      written[g] = 1;
      if (logits) logits.set(out.subarray(i * NUM_CLASSES, (i + 1) * NUM_CLASSES), g * NUM_CLASSES);
    }
    if (opts.onWindow) await opts.onWindow(w + 1, nWindows);
  }
  for (let g = 0; g < total; g++) {
    if (!written[g]) throw new Error(`OSD sliding window left a hole at frame ${g}`);
  }

  const result = {
    frameClasses,
    frameStepS: FRAME_STEP_S,
    numFrames: total,
    durationS: audio16k.length / SAMPLE_RATE,
    powerset,
  };
  if (logits) result.logits = logits;
  return result;
}

/**
 * One model call on exactly one window.
 * @param {object} session
 * @param {Float32Array} window160k exactly `WINDOW_SAMPLES` samples
 * @returns {Promise<Float32Array>} `[589 * 7]` log-probabilities, row-major
 */
export async function runWindow(session, window160k) {
  if (window160k.length !== WINDOW_SAMPLES) {
    throw new Error(`OSD window must be ${WINDOW_SAMPLES} samples, got ${window160k.length}`);
  }
  const feeds = { input_values: new ort.Tensor('float32', window160k, [1, 1, WINDOW_SAMPLES]) };
  const results = await session.run(feeds);
  const logits = results.logits;
  if (!logits) throw new Error('OSD model returned no "logits" output');
  return logits.data;
}
