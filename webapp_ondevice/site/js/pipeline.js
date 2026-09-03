/**
 * The run: decode → OSD → routing → per-region separation, plus the override
 * ("separate everything anyway") experiment.
 *
 * This module owns the *sequence* and the state transitions; every actual
 * computation belongs to the proven engine layer (`osd.js`, `routing.js`,
 * `separate.js`, `resample.js`). Two constraints shape the code:
 *
 * 1. **Everything runs on the main thread** (ORT-Web WASM, single-threaded —
 *    `ortEnv.js` explains why). Between model calls the run therefore hands the
 *    browser a real turn (`nextFrame`), otherwise the progressive timeline fill
 *    that makes the routing argument visible would never paint.
 * 2. **A run must be interruptible.** MossFormer2 costs ~90 s per 4 s window in
 *    WASM, so "Przerwij" cannot mean "wait for the queue to drain": the cancel
 *    token is checked in the awaited per-chunk callback, and throwing from
 *    there unwinds out of `separateRegion` at the next chunk boundary.
 */
import {
  CHUNK_S,
  NONSPEECH,
  OVERLAP,
  SAMPLE_RATE,
  SOLO,
  expandToChunk,
  framesToIntervals,
  selectOverlapRegions,
} from './routing.js';
import { runOsd } from './osd.js';
import { chunkStarts, separateRegion } from './separate.js';
import { decimateTo8k, decodeToMono16k } from './resample.js';
import {
  ModelFetchError,
  OSD,
  RUNTIME,
  SEPARATORS,
  ensureOsd,
  ensureRuntime,
  ensureSeparator,
  loaded,
} from './models.js';
import { PEAK_BINS, REGION_PEAK_BINS, S, resetRun, resetSeparation } from './state.js';
import { computePeaks } from './timeline.js';
import { render } from './ui.js';
import { stopAll } from './player.js';

/** Thrown by the cancel check; never surfaces to the user as an error card. */
class Cancelled extends Error {
  constructor() {
    super('anulowano');
    this.name = 'Cancelled';
  }
}

let token = { cancelled: false };

/** Abandon whatever is running at the next chunk/window boundary. */
export function cancelRun() {
  token.cancelled = true;
}

function freshToken() {
  token = { cancelled: false };
  return token;
}

function checkCancel(active) {
  if (active.cancelled) throw new Cancelled();
}

/**
 * Give the browser a turn: a microtask is not enough to repaint, and a hidden
 * tab throttles `requestAnimationFrame` to nothing — so race it with a timer.
 */
function nextFrame() {
  return new Promise((resolve) => {
    let settled = false;
    const finish = () => { if (!settled) { settled = true; resolve(); } };
    requestAnimationFrame(() => setTimeout(finish, 0));
    setTimeout(finish, 60);
  });
}

let lastPaint = 0;
/** Repaint at most ~11 fps for progress events; `force` for state changes. */
function paint(force) {
  const now = performance.now();
  if (force || now - lastPaint > 90) {
    lastPaint = now;
    render();
  }
}

// ---------------------------------------------------------------------------
// model loading, with the download panel
// ---------------------------------------------------------------------------
function downloadRow(spec) {
  return { key: spec.key, label: spec.label, loaded: 0, total: 0, bytes: spec.bytes, done: false };
}

function progressInto(row) {
  return (loadedBytes, total) => {
    row.loaded = loadedBytes;
    row.total = total || row.bytes;
    row.done = row.total > 0 && loadedBytes >= row.total;
    paint(row.done);
  };
}

/** Runtime + OSD. Shows S1 only when something actually has to be fetched. */
async function loadOsd() {
  const have = loaded();
  if (have.runtime && have.osd) return ensureOsd(() => {});
  S.state = 'S1';
  S.dlTitle = 'Pobieranie modeli — jednorazowo, potem z cache przeglądarki.';
  S.dl = [];
  const runtimeRow = downloadRow(RUNTIME);
  const osdRow = downloadRow(OSD);
  if (!have.runtime) S.dl.push(runtimeRow);
  if (!have.osd) S.dl.push(osdRow);
  render();
  await ensureRuntime(progressInto(runtimeRow));
  const session = await ensureOsd(progressInto(osdRow));
  S.dl = [];
  render();
  return session;
}

/**
 * The separator for the current dropdown choice. Deliberately lazy: a clip with
 * no overlap never pays for 30-46 MB it would not have used.
 */
async function loadSeparator() {
  const spec = SEPARATORS[S.modelId];
  if (loaded().separator === S.modelId && loaded().runtime) return ensureSeparator(S.modelId, () => {});
  S.dlTitle = `Pobieranie separatora — ${spec.option}.`;
  const runtimeRow = downloadRow(RUNTIME);
  const sepRow = downloadRow(spec);
  S.dl = loaded().runtime ? [sepRow] : [runtimeRow, sepRow];
  render();
  await ensureRuntime(progressInto(runtimeRow));
  const session = await ensureSeparator(S.modelId, progressInto(sepRow));
  S.dl = [];
  render();
  return session;
}

// ---------------------------------------------------------------------------
// OSD result -> bands, routed regions, stats
// ---------------------------------------------------------------------------
const KINDS = [[NONSPEECH, 'sil'], [SOLO, 'solo'], [OVERLAP, 'ovl']];

/** Per-frame labels -> the timeline's `[start, end, kind]` bands, in time order. */
function framesToSegs(frameClasses) {
  const segs = [];
  for (const [label, kind] of KINDS) {
    for (const [start, end] of framesToIntervals(frameClasses, label)) segs.push([start, end, kind]);
  }
  return segs.sort((a, b) => a[0] - b[0]);
}

/**
 * Apply the routing constants of the server pipeline: drop overlaps under
 * 0.2 s, merge those closer than 0.5 s, expand what survives to the
 * separator's 4 s window.
 */
function routeRegions(frameClasses, durationS, samples) {
  const raw = framesToIntervals(frameClasses, OVERLAP);
  const routed = selectOverlapRegions(raw);
  const padded = expandToChunk(routed, CHUNK_S, durationS);
  return routed.map((region, i) => {
    const [start, end] = padded[i];
    const from16 = Math.max(0, Math.round(start * SAMPLE_RATE));
    const to16 = Math.min(samples, Math.round(end * SAMPLE_RATE));
    const length8k = Math.round((to16 - from16) / 2);
    return {
      raw: region,
      pad: [start, end],
      from16,
      to16,
      chunks: chunkStarts(length8k).length,
      done: false,
      frac: 0,
      /** @type {Float32Array|null} */ mix: null,
      /** @type {Float32Array|null} */ s1: null,
      /** @type {Float32Array|null} */ s2: null,
      /** @type {{mix: Float32Array, s1: Float32Array, s2: Float32Array}|null} */ peaks: null,
      /** Assembled full-clip streams — only built for the single-region case. */
      full: null,
    };
  });
}

function applyOsd(osd) {
  S.segs = framesToSegs(osd.frameClasses);
  S.overlapS = S.segs.reduce((sum, seg) => sum + (seg[2] === 'ovl' ? seg[1] - seg[0] : 0), 0);
  S.regions = routeRegions(osd.frameClasses, S.durationS, S.audio16k.length);
  S.wholeCalls = chunkStarts(Math.round(S.audio16k.length / 2)).length;
}

/** The whole clip at the separator's rate — built once, reused by the override. */
function audio8k() {
  if (!S.audio8k) S.audio8k = decimateTo8k(S.audio16k);
  return S.audio8k;
}

// ---------------------------------------------------------------------------
// the run
// ---------------------------------------------------------------------------
/**
 * Decode + OSD + routing + separation for one input.
 * @param {{name: string, kind: 'file'|'mic'|'example', data: ArrayBuffer|Blob}} input
 */
export async function runPipeline(input) {
  const active = freshToken();
  stopAll();
  resetRun();
  S.input = { name: input.name, kind: input.kind };
  render();

  try {
    const osdSession = await loadOsd();
    checkCancel(active);

    S.state = 'S2';
    render();
    await nextFrame();
    let audio;
    try {
      audio = await decodeToMono16k(input.data);
    } catch (err) {
      throw decodeError(input.name, err);
    }
    if (!audio.length) throw decodeError(input.name, new Error('pusty plik'));
    S.audio16k = audio;
    S.durationS = audio.length / SAMPLE_RATE;
    S.peaks = computePeaks(audio, PEAK_BINS);
    checkCancel(active);

    S.state = 'S3';
    S.osdFrac = 0;
    render();
    await nextFrame();
    const osd = await runOsd(audio, osdSession, {
      onWindow: async (done, total) => {
        S.osdFrac = done / total;
        paint();
        await nextFrame();
        checkCancel(active);
      },
    });

    applyOsd(osd);
    S.state = 'S4';
    render();
    await nextFrame();

    if (!S.regions.length) {
      S.state = 'S6b';
      render();
      return;
    }
    await separateRegions(active);
  } catch (err) {
    handleRunFailure(err);
  }
}

/** S5: one routed region at a time, progressive fill, cancellable. */
async function separateRegions(active) {
  S.state = 'S5';
  render();
  const session = await loadSeparator();
  checkCancel(active);
  S.sepDone = 0;
  render();
  await nextFrame();

  for (let i = 0; i < S.regions.length; i++) {
    const region = S.regions[i];
    const mix8k = decimateTo8k(S.audio16k.subarray(region.from16, region.to16));
    region.frac = 0.02; // the region reads as "started" before the first chunk lands
    paint(true);
    await nextFrame();

    const [s1, s2] = await separateRegion(mix8k, session, {
      onChunk: async (done, total) => {
        checkCancel(active);
        region.frac = done / total;
        paint(true);
        await nextFrame();
      },
    });
    checkCancel(active);

    region.mix = mix8k;
    region.s1 = s1;
    region.s2 = s2;
    region.done = true;
    region.frac = 1;
    let scale = 0;
    for (let k = 0; k < mix8k.length; k++) scale = Math.max(scale, Math.abs(mix8k[k]));
    region.peaks = {
      mix: computePeaks(mix8k, REGION_PEAK_BINS, scale),
      s1: computePeaks(s1, REGION_PEAK_BINS, scale),
      s2: computePeaks(s2, REGION_PEAK_BINS, scale),
    };
    S.sepDone = i + 1;
    if (S.region === null) S.region = i; // open the inspector on the first result
    paint(true);
    await nextFrame();
  }

  if (S.regions.length === 1) assembleSingleRegion(S.regions[0]);
  S.state = 'S6';
  S.regHead = 0;
  render();
}

/**
 * The honest special case: with exactly ONE overlap region
 * there is no cross-region permutation to solve, so both speakers can be
 * offered as full-length streams — the clip at 8 kHz with the region's samples
 * replaced by that speaker's separated audio. Everything outside the region is
 * solo or silence and is passed through unchanged, which is precisely what the
 * server pipeline does with non-overlap audio.
 */
function assembleSingleRegion(region) {
  const base = audio8k();
  const at = Math.round(region.from16 / 2);
  const build = (stream) => {
    const out = Float32Array.from(base);
    out.set(stream.subarray(0, Math.min(stream.length, out.length - at)), at);
    return out;
  };
  region.full = { s1: build(region.s1), s2: build(region.s2) };
}

/** S7: the M2 demonstration — the same chunker over the entire clip. */
export async function runOverride() {
  const active = freshToken();
  const previous = S.state;
  S.override.running = true;
  S.override.frac = 0;
  render();
  try {
    const session = await loadSeparator();
    checkCancel(active);
    const [s1, s2] = await separateRegion(audio8k(), session, {
      onChunk: async (done, total) => {
        checkCancel(active);
        S.override.frac = done / total;
        paint(true);
        await nextFrame();
      },
    });
    S.override.s1 = s1;
    S.override.s2 = s2;
    S.override.markers = phantomMarkers();
    S.override.running = false;
    S.override.done = true;
    S.state = 'S7';
    render();
  } catch (err) {
    S.override.running = false;
    if (err instanceof Cancelled) {
      S.state = previous;
      render();
      return;
    }
    handleRunFailure(err);
  }
}

/**
 * Where the phantom stream is worth listening for: the middle of the longest
 * solo stretches, at most three of them, spread across the clip.
 */
function phantomMarkers() {
  const solos = S.segs.filter((seg) => seg[2] === 'solo' && seg[1] - seg[0] >= 1.2);
  const mids = solos.map((seg) => (seg[0] + seg[1]) / 2);
  if (mids.length <= 3) return mids;
  return [mids[0], mids[Math.floor(mids.length / 2)], mids[mids.length - 1]];
}

// ---------------------------------------------------------------------------
// input sources
// ---------------------------------------------------------------------------
/**
 * Fetch an audio file and run it. The one code path shared by the example
 * clips and (in dev) by the `#debug` E2E hook — which is why it takes a URL
 * instead of reaching for anything of its own.
 * @param {string} url
 * @param {string} [name] label for the input card
 */
export async function runFromUrl(url, name) {
  const label = name || decodeURIComponent(url.split('/').pop() || 'nagranie');
  let data;
  try {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    data = await response.arrayBuffer();
  } catch (err) {
    S.input = { name: label, kind: 'example' };
    handleRunFailure(new ModelFetchError(`nagranie ${label}`, err.message || err));
    return;
  }
  await runPipeline({ name: label, kind: 'example', data });
}

// ---------------------------------------------------------------------------
// errors (E1) — inline cards, never alert()
// ---------------------------------------------------------------------------
function decodeError(name, cause) {
  const err = new Error(cause && cause.message ? cause.message : String(cause));
  err.name = 'DecodeError';
  err.fileName = name;
  return err;
}

/** Turn an exception into the E1 card's payload. */
function errorCard(err) {
  if (err instanceof ModelFetchError) {
    return {
      title: 'Nie udało się pobrać modelu',
      type: 'FetchError',
      msg: `Pobieranie „${err.label}” nie powiodło się (${err.message.split(': ').slice(1).join(': ') || 'brak połączenia'}). ` +
        'Modele pobierają się raz i zostają w cache przeglądarki; ponowna próba zaczyna od nowa.',
      retry: 'Ponów pobieranie',
      action: 'retry',
      failStep: 1,
    };
  }
  if (err && err.name === 'DecodeError') {
    return {
      title: 'Nie udało się zdekodować pliku',
      type: 'DecodeError',
      msg: `Przeglądarka nie rozpoznała formatu nagrania${err.fileName ? ` (${err.fileName})` : ''}. ` +
        'Spróbuj WAV, MP3, FLAC, M4A, OGG lub WebM — dekodowanie odbywa się mechanizmem samej przeglądarki.',
      retry: 'Wybierz inny plik',
      action: 'file',
      failStep: 0,
    };
  }
  if (err && (err.name === 'NotAllowedError' || err.name === 'SecurityError')) {
    return {
      title: 'Brak dostępu do mikrofonu',
      type: err.name,
      msg: 'Przeglądarka odmówiła dostępu do mikrofonu. Zezwól na dostęp w ustawieniach strony ' +
        '(ikona kłódki obok adresu) i spróbuj ponownie. Nagranie i tak nie opuszcza urządzenia.',
      retry: 'Spróbuj ponownie',
      action: 'mic',
      failStep: 0,
    };
  }
  if (err && err.name === 'NotFoundError') {
    return {
      title: 'Nie znaleziono mikrofonu',
      type: err.name,
      msg: 'To urządzenie nie zgłasza żadnego wejścia audio. Podłącz mikrofon albo użyj zakładki „Plik”.',
      retry: 'Spróbuj ponownie',
      action: 'mic',
      failStep: 0,
    };
  }
  return {
    title: 'Przetwarzanie nie powiodło się',
    type: (err && err.name) || 'Error',
    msg: (err && err.message) || String(err),
    retry: 'Spróbuj ponownie',
    action: 'retry',
    failStep: 3,
  };
}

/** Show an error card for anything that is not a cancellation. */
export function showError(err) {
  S.err = errorCard(err);
  S.state = 'E1';
  render();
}

function handleRunFailure(err) {
  if (err instanceof Cancelled) {
    // Back to the routed state with partial results discarded (or to idle if
    // the run had not got that far).
    resetSeparation();
    S.state = S.regions.length || S.segs.length ? 'S4' : 'S0';
    if (S.state === 'S0') resetRun();
    render();
    return;
  }
  showError(err);
}

/** Re-run separation after a separator swap — everything before S4 is reused. */
export async function reseparate() {
  const active = freshToken();
  stopAll();
  resetSeparation();
  if (!S.regions.length) {
    render();
    return;
  }
  try {
    await separateRegions(active);
  } catch (err) {
    handleRunFailure(err);
  }
}
