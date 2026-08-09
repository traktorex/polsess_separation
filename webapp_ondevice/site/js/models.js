/**
 * Model inventory + download-with-progress + session lifecycle.
 *
 * Three things are downloaded: the ONNX Runtime WASM binary, the OSD model, and
 * exactly one separator. They are fetched here rather than left to the runtime
 * so the page can show honest per-file progress bars (S1), which for a 30-46 MB
 * separator on a phone is the difference between "loading" and "broken".
 *
 * Session policy (12 GB-phone-friendly, plan §3):
 *   - the OSD session is tiny (~3 MB) and permanent — created once, kept;
 *   - exactly ONE separator session exists at a time — picking the other model
 *     releases the previous one first.
 * Both are cached for the lifetime of the page, so a second run on a new clip
 * downloads and initialises nothing.
 *
 * Sizes below are the shipping artifacts' real byte counts (`build/NOTES.md`);
 * they are only used for labels before `Content-Length` arrives.
 */
import { MODELS_URL, ORT_VENDOR_URL, createSession, configureOrt, ort } from './ortEnv.js';
import { OSD_MODEL_URL, createOsdSession } from './osd.js';

/** The runtime binary ORT would otherwise fetch by itself. */
export const RUNTIME = {
  key: 'ort',
  label: 'onnxruntime-web · WASM',
  url: new URL('ort-wasm-simd-threaded.wasm', ORT_VENDOR_URL).href,
  bytes: 11905541,
};

export const OSD = {
  key: 'osd',
  label: 'pyannote-segmentation-3.0 · fp16 ONNX',
  url: OSD_MODEL_URL,
  bytes: 3000918,
};

/**
 * Separator choices. SepFormer is preselected: the WASM gate (2026-08-05)
 * measured single-threaded RTF 1.46 against MossFormer2's 22.4, i.e. ~90 s per
 * 4 s window — usable only as a deliberate "best quality" choice.
 */
export const SEPARATORS = {
  sepformer: {
    key: 'sep',
    id: 'sepformer',
    label: 'separator · SepFormer-128k int8',
    option: 'SepFormer-128k int8 · 30 MB · szybszy',
    url: new URL('sepformer_128k_int8.onnx', MODELS_URL).href,
    bytes: 31684126,
    warn: '',
  },
  mf2: {
    id: 'mf2',
    key: 'sep',
    label: 'separator · MossFormer2-128k int8',
    option: 'MossFormer2-128k int8 · 46 MB · najlepsza jakość',
    url: new URL('mf2_128k_int8.onnx', MODELS_URL).href,
    bytes: 48016009,
    warn: 'najlepsza jakość, ale wolno: ~1,5 min na region 4 s na laptopie',
  },
};

export const DEFAULT_SEPARATOR = 'sepformer';

/** A download that failed — carries the human label for the error card. */
export class ModelFetchError extends Error {
  constructor(label, cause) {
    super(`${label}: ${cause}`);
    this.name = 'FetchError';
    this.label = label;
  }
}

/**
 * Fetch a binary with progress. Streams the body so `onProgress` can report
 * bytes as they arrive; falls back to a single 100 % report if the server sends
 * no `Content-Length` (then `total` is 0 and the caller shows an indeterminate
 * bar).
 *
 * @param {string} url
 * @param {(loaded: number, total: number) => void} onProgress
 * @returns {Promise<Uint8Array>}
 */
export async function fetchWithProgress(url, onProgress) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const total = Number(response.headers.get('content-length')) || 0;
  if (!response.body) {
    const buffer = new Uint8Array(await response.arrayBuffer());
    onProgress(buffer.length, buffer.length);
    return buffer;
  }
  const reader = response.body.getReader();
  const chunks = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    onProgress(loaded, total);
  }
  // Hand the stream back even though it is drained: a response left locked to a
  // reader is torn down as an *aborted* request when it is finally collected,
  // which is invisible to this code but shows up as a failed request in
  // `devcheck/check_page.py` (and in DevTools) long after the bytes arrived.
  reader.releaseLock();
  const out = new Uint8Array(loaded);
  let at = 0;
  for (const chunk of chunks) { out.set(chunk, at); at += chunk.length; }
  onProgress(loaded, loaded);
  // Hand the event loop one turn before returning. Callers go straight into
  // `InferenceSession.create`, which blocks the main thread for as long as it
  // takes to initialise the graph; a response the browser is still tearing
  // down when that happens gets reported as an ABORTED request even though
  // every byte arrived (measured on the pyannote model: 3/5 runs flagged
  // without this yield, 0/5 with it). It also lets the 100 % progress bar
  // paint before the freeze.
  await new Promise((resolve) => setTimeout(resolve, 0));
  return out;
}

let runtimeReady = false;
let osdSession = null;
let separatorSession = null;
let separatorId = null;

/**
 * Hand ORT its WASM binary instead of letting it fetch it unobserved.
 * `env.wasm.wasmBinary` is read at first session creation; setting it after
 * `configureOrt()` leaves every other documented setting alone.
 */
export async function ensureRuntime(onProgress) {
  if (runtimeReady) return;
  configureOrt();
  let bytes;
  try {
    bytes = await fetchWithProgress(RUNTIME.url, onProgress);
  } catch (err) {
    throw new ModelFetchError(RUNTIME.label, err.message || err);
  }
  ort.env.wasm.wasmBinary = bytes.buffer;
  runtimeReady = true;
}

/** The permanent OSD session. */
export async function ensureOsd(onProgress) {
  if (osdSession) return osdSession;
  let bytes;
  try {
    bytes = await fetchWithProgress(OSD.url, onProgress);
  } catch (err) {
    throw new ModelFetchError(OSD.label, err.message || err);
  }
  osdSession = await createOsdSession(bytes);
  return osdSession;
}

/**
 * The one separator session. Switching models releases the previous session
 * before the new one is created, so peak memory holds one separator, not two.
 * @param {'sepformer'|'mf2'} id
 */
export async function ensureSeparator(id, onProgress) {
  const spec = SEPARATORS[id];
  if (!spec) throw new Error(`unknown separator "${id}"`);
  if (separatorSession && separatorId === id) return separatorSession;
  await releaseSeparator();
  let bytes;
  try {
    bytes = await fetchWithProgress(spec.url, onProgress);
  } catch (err) {
    throw new ModelFetchError(spec.label, err.message || err);
  }
  separatorSession = await createSession(bytes);
  separatorId = id;
  return separatorSession;
}

export async function releaseSeparator() {
  if (separatorSession && separatorSession.release) await separatorSession.release();
  separatorSession = null;
  separatorId = null;
}

/** Which models are already in memory — the S1 panel skips those. */
export function loaded() {
  return { runtime: runtimeReady, osd: !!osdSession, separator: separatorId };
}
