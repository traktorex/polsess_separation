/**
 * Dev-only state switcher — loaded ONLY when the page URL carries `#debug`
 * (`app.js` imports it dynamically), so the shipped page pays nothing for it.
 *
 * Two jobs:
 *
 * 1. **Every state without a model.** It fills `state.js` with synthetic data
 *    in exactly the shape the real pipeline produces and re-renders, so all ten
 *    states listed in `state.js` can be screenshotted in a second
 *    instead of in minutes. The fake clips are the accepted mockup's, which is
 *    what makes the screenshot comparison against the mockup meaningful. This
 *    data exists nowhere else in the app.
 * 2. **An end-to-end handle.** `window.__debug.loadFromUrl(url)` runs the REAL
 *    pipeline on any audio URL (the example-clip code path, minus the manifest)
 *    and resolves `window.__e2e` with a summary of what the page ended up
 *    showing — the hook `devcheck/check_page.py --js` awaits. The URL is an
 *    argument, never a constant, so `index.html` itself stays free of dev paths.
 */
import { PEAK_BINS, REGION_PEAK_BINS, S, resetRun } from './state.js';
import { $, render } from './ui.js';
import { runFromUrl, runOverride } from './pipeline.js';
import { SEPARATORS } from './models.js';

// ---------------------------------------------------------------------------
// fake clips (the mockup's, verbatim in numbers)
// ---------------------------------------------------------------------------
const CLIP_A = {
  id: 'A',
  file: 'rozmowa_przeplot.wav',
  dur: 38.4,
  segs: [
    [0.0, 1.1, 'sil'], [1.1, 6.3, 'solo'], [6.3, 7.2, 'sil'], [7.2, 13.4, 'solo'],
    [13.4, 14.1, 'ovl'], [14.1, 20.4, 'solo'], [20.4, 21.2, 'sil'], [21.2, 22.8, 'solo'],
    [22.8, 23.6, 'ovl'], [23.6, 29.4, 'solo'], [29.4, 30.1, 'sil'], [30.1, 31.6, 'solo'],
    [31.6, 34.5, 'ovl'], [34.5, 37.5, 'solo'], [37.5, 38.4, 'sil'],
  ],
  regions: [
    { start: 12.4, end: 15.1, calls: 1 },
    { start: 21.8, end: 24.6, calls: 1 },
    { start: 30.6, end: 35.5, calls: 2 },
  ],
  ovlSec: 4.4,
  wholeCalls: 19,
  phantomAt: [4.0, 17.5, 27.0],
};
const CLIP_B = {
  id: 'B',
  file: 'pelne_nakladanie.wav',
  dur: 24.6,
  segs: [
    [0.0, 0.6, 'sil'], [0.6, 11.9, 'ovl'], [11.9, 12.4, 'solo'], [12.4, 23.8, 'ovl'],
    [23.8, 24.6, 'sil'],
  ],
  regions: [{ start: 0.0, end: 24.6, calls: 12 }],
  ovlSec: 22.7,
  wholeCalls: 12,
  phantomAt: [12.1],
};
const CLIP_NOOVL = {
  id: 'N',
  file: 'monolog_solo.wav',
  dur: 31.8,
  segs: [
    [0.0, 0.9, 'sil'], [0.9, 8.6, 'solo'], [8.6, 9.4, 'sil'], [9.4, 17.2, 'solo'],
    [17.2, 18.1, 'sil'], [18.1, 26.9, 'solo'], [26.9, 27.6, 'sil'], [27.6, 31.0, 'solo'],
    [31.0, 31.8, 'sil'],
  ],
  regions: [],
  ovlSec: 0.0,
  wholeCalls: 15,
  phantomAt: [4.5, 13.0, 22.0],
};

const ERRORS = [
  {
    title: 'Nie udało się pobrać modelu',
    type: 'FetchError',
    msg: 'Pobieranie „pyannote-segmentation-3.0 · fp16 ONNX” nie powiodło się (HTTP 503). ' +
      'Modele pobierają się raz i zostają w cache przeglądarki; ponowna próba zaczyna od nowa.',
    retry: 'Ponów pobieranie',
    action: 'retry',
    failStep: 1,
  },
  {
    title: 'Nie udało się zdekodować pliku',
    type: 'DecodeError',
    msg: 'Przeglądarka nie rozpoznała formatu nagrania (rozmowa.amr). Spróbuj WAV, MP3, FLAC, ' +
      'M4A, OGG lub WebM — dekodowanie odbywa się mechanizmem samej przeglądarki.',
    retry: 'Wybierz inny plik',
    action: 'file',
    failStep: 0,
  },
  {
    title: 'Brak dostępu do mikrofonu',
    type: 'NotAllowedError',
    msg: 'Przeglądarka odmówiła dostępu do mikrofonu. Zezwól na dostęp w ustawieniach strony ' +
      '(ikona kłódki obok adresu) i spróbuj ponownie. Nagranie i tak nie opuszcza urządzenia.',
    retry: 'Spróbuj ponownie',
    action: 'mic',
    failStep: 0,
  },
];

// ---------------------------------------------------------------------------
// synthetic waveform (deterministic, no assets)
// ---------------------------------------------------------------------------
function lcg(seed) {
  let s = (seed >>> 0) || 1;
  return () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 4294967296; };
}
function hashStr(text) {
  let h = 2166136261;
  for (let i = 0; i < text.length; i++) { h ^= text.charCodeAt(i); h = Math.imul(h, 16777619); }
  return h >>> 0;
}
function kindAt(clip, t) {
  for (const seg of clip.segs) if (t >= seg[0] && t < seg[1]) return seg[2];
  return 'sil';
}
function fakePeaks(clip, bins, from, to, source) {
  const a = from === undefined ? 0 : from;
  const b = to === undefined ? clip.dur : to;
  const rnd = lcg(hashStr(`${clip.id}|${source}|${a.toFixed(2)}|${bins}`));
  const out = new Float32Array(bins);
  for (let i = 0; i < bins; i++) {
    const t = a + ((i + 0.5) / bins) * (b - a);
    const kind = kindAt(clip, t);
    const syl1 = 0.28 + 0.72 * Math.abs(Math.sin(t * 6.1 + 1.3));
    const syl2 = 0.28 + 0.72 * Math.abs(Math.sin(t * 4.4 + 3.9));
    const jitter = 0.62 + 0.38 * rnd();
    let v;
    if (source === 's1') v = kind === 'sil' ? 0.02 + 0.03 * rnd() : 0.12 + 0.64 * syl1 * jitter;
    else if (source === 's2') v = kind === 'sil' ? 0.02 + 0.03 * rnd() : 0.12 + 0.64 * syl2 * jitter;
    else if (kind === 'sil') v = 0.02 + 0.05 * rnd();
    else if (kind === 'solo') v = 0.14 + 0.6 * syl1 * jitter;
    else v = 0.26 + 0.68 * Math.max(syl1, syl2) * jitter;
    out[i] = Math.max(0, Math.min(1, v));
  }
  return out;
}

// ---------------------------------------------------------------------------
// fake clip -> app state
// ---------------------------------------------------------------------------
function clipFor(state, clipId) {
  if (state === 'S6b') return CLIP_NOOVL;
  return clipId === 'B' ? CLIP_B : CLIP_A;
}

function applyClip(clip) {
  resetRun();
  S.input = { name: clip.file, kind: 'example' };
  S.durationS = clip.dur;
  S.peaks = fakePeaks(clip, PEAK_BINS, 0, clip.dur, 'mix');
  S.segs = clip.segs.map((seg) => [seg[0], seg[1], seg[2]]);
  S.overlapS = clip.ovlSec;
  S.wholeCalls = clip.wholeCalls;
  S.regions = clip.regions.map((region) => ({
    raw: [region.start + 0.6, region.end - 0.6],
    pad: [region.start, region.end],
    from16: Math.round(region.start * 16000),
    to16: Math.round(region.end * 16000),
    chunks: region.calls,
    done: false,
    frac: 0,
    mix: null,
    s1: null,
    s2: null,
    peaks: null,
    full: null,
  }));
}

function markSeparated(clip, count) {
  S.regions.forEach((region, i) => {
    if (i >= count) return;
    region.done = true;
    region.frac = 1;
    region.peaks = {
      mix: fakePeaks(clip, REGION_PEAK_BINS, region.pad[0], region.pad[1], 'mix'),
      s1: fakePeaks(clip, REGION_PEAK_BINS, region.pad[0], region.pad[1], 's1'),
      s2: fakePeaks(clip, REGION_PEAK_BINS, region.pad[0], region.pad[1], 's2'),
    };
  });
  S.sepDone = count;
  if (S.regions.length === 1 && count === 1) S.regions[0].full = { s1: null, s2: null };
}

let clipId = 'A';
let errIndex = 0;

/** Drive the UI into one of the ten states with synthetic data. */
function setFakeState(state) {
  const clip = clipFor(state, clipId);
  applyClip(clip);
  S.state = state;

  if (state === 'S0') { resetRun(); S.input = null; }
  if (state === 'S1') {
    S.dlTitle = 'Pobieranie modeli — jednorazowo, potem z cache przeglądarki.';
    S.dl = [
      { key: 'ort', label: 'onnxruntime-web · WASM', loaded: 11905541, total: 11905541, done: true },
      { key: 'osd', label: 'pyannote-segmentation-3.0 · fp16 ONNX', loaded: 2370000, total: 3000918, done: false },
      { key: 'sep', label: SEPARATORS.sepformer.label, loaded: 13300000, total: 31684126, done: false },
    ];
  }
  if (state === 'S3') S.osdFrac = 0.62;
  if (state === 'S5') {
    markSeparated(clip, 1);
    if (S.regions[1]) S.regions[1].frac = 0.55;
  }
  if (state === 'S6' || state === 'S7') {
    markSeparated(clip, S.regions.length);
    S.region = S.regions.length ? 0 : null;
  }
  if (state === 'S7') {
    S.override.done = true;
    S.override.markers = clip.phantomAt;
  }
  if (state === 'E1') S.err = ERRORS[errIndex];
  render();
}

// ---------------------------------------------------------------------------
// E2E probes
// ---------------------------------------------------------------------------
function rms(samples) {
  if (!samples || !samples.length) return 0;
  let sum = 0;
  for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i];
  return Math.sqrt(sum / samples.length);
}
function corr(a, b) {
  const n = Math.min(a.length, b.length);
  let ma = 0;
  let mb = 0;
  for (let i = 0; i < n; i++) { ma += a[i]; mb += b[i]; }
  ma /= n; mb /= n;
  let num = 0;
  let da = 0;
  let db = 0;
  for (let i = 0; i < n; i++) {
    const x = a[i] - ma;
    const y = b[i] - mb;
    num += x * y; da += x * x; db += y * y;
  }
  return num / (Math.sqrt(da * db) + 1e-12);
}
function peak(samples) {
  let top = 0;
  if (!samples) return 0;
  for (let i = 0; i < samples.length; i++) {
    const v = Math.abs(samples[i]);
    if (v > top) top = v;
  }
  return top;
}
const round = (x, d = 4) => Math.round(x * 10 ** d) / 10 ** d;

/** Per-region audio facts: are the two streams real, and are they different? */
function probeRegions() {
  return S.regions.map((region, i) => ({
    i,
    padS: [round(region.pad[0], 3), round(region.pad[1], 3)],
    rawS: [round(region.raw[0], 3), round(region.raw[1], 3)],
    chunks: region.chunks,
    done: region.done,
    samples: region.s1 ? region.s1.length : 0,
    rms1: region.s1 ? round(rms(region.s1), 5) : 0,
    rms2: region.s2 ? round(rms(region.s2), 5) : 0,
    rmsMix: region.mix ? round(rms(region.mix), 5) : 0,
    peak1: round(peak(region.s1), 3),
    peak2: round(peak(region.s2), 3),
    peakMix: round(peak(region.mix), 3),
    corr12: region.s1 && region.s2 ? round(corr(region.s1, region.s2)) : null,
  }));
}

/** What the page is showing right now — the E2E assertion surface. */
function summary() {
  const text = (id) => ($(id) ? $(id).textContent.trim() : null);
  return {
    state: S.state,
    model: S.modelId,
    input: S.input ? S.input.name : null,
    durationS: round(S.durationS, 3),
    regions: S.regions.length,
    chunks: S.regions.reduce((sum, r) => sum + r.chunks, 0),
    wholeCalls: S.wholeCalls,
    overlapS: round(S.overlapS, 3),
    tilesText: {
      t1: text('st1v'), t1sub: text('st1s'),
      t2: text('st2v'), t2sub: text('st2s'),
      t3: text('st3v'), t3sub: text('st3s'),
    },
    steps: [1, 2, 3].map((i) => {
      const node = document.querySelector(`.pstep[data-step="${i}"] .pstate`);
      return node ? node.textContent.trim() : null;
    }),
    durations: S.regions.map((r) => round(r.pad[1] - r.pad[0], 3)),
    audio: probeRegions(),
    error: S.err ? { type: S.err.type, title: S.err.title } : null,
  };
}

// ---------------------------------------------------------------------------
// the bar
// ---------------------------------------------------------------------------
const STATES = ['S0', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S6b', 'S7', 'E1'];
const BAR_CSS = `
#devbar { position: fixed; left: 0; right: 0; bottom: 0; z-index: 70;
  background: #000; color: #f0f0f0; border-top: 2px solid #ff3b3b;
  font-family: var(--mono); font-size: 11px; padding: 6px 8px 7px; }
#devbar .dvlabel { color: #ff8b8b; letter-spacing: .06em; font-weight: 700; margin-right: 8px; white-space: nowrap; }
#devbar .dvrow { display: flex; gap: 5px; align-items: center; flex-wrap: wrap; }
#devbar .dvrow + .dvrow { margin-top: 5px; padding-top: 5px; border-top: 1px dotted #444; }
#devbar button { background: #1c1c1c; color: #e8e8e8; border: 1px solid #444; border-radius: 4px;
  padding: 3px 8px; font-family: var(--mono); font-size: 11px; cursor: pointer; }
#devbar button:hover { border-color: #ff3b3b; }
#devbar button.on { background: #ff3b3b; border-color: #ff3b3b; color: #fff; font-weight: 700; }
#devbar .dvnote { color: #888; margin-left: 4px; }
body { padding-bottom: 76px; }
@media (max-width: 720px) { #toast { bottom: 132px; } }
`;

function buildBar() {
  const style = document.createElement('style');
  style.textContent = BAR_CSS;
  document.head.appendChild(style);

  const bar = document.createElement('div');
  bar.id = 'devbar';
  const rowStates = document.createElement('div');
  rowStates.className = 'dvrow';
  rowStates.innerHTML = '<span class="dvlabel">DEBUG — przełącznik stanów</span>';
  for (const state of STATES) {
    const button = document.createElement('button');
    button.type = 'button';
    button.dataset.state = state;
    button.textContent = state;
    button.addEventListener('click', () => {
      if (state === 'E1' && S.state === 'E1') errIndex = (errIndex + 1) % ERRORS.length;
      setFakeState(state);
      stampHash();
      paintBar();
    });
    rowStates.appendChild(button);
  }
  const rowClips = document.createElement('div');
  rowClips.className = 'dvrow';
  rowClips.innerHTML = '<span class="dvlabel">klip</span>';
  for (const [id, label] of [['A', 'A · przeplot (3 regiony)'], ['B', 'B · pełne nakładanie (1 region)']]) {
    const button = document.createElement('button');
    button.type = 'button';
    button.dataset.clip = id;
    button.textContent = label;
    button.addEventListener('click', () => {
      clipId = id;
      setFakeState(S.state);
      stampHash();
      paintBar();
    });
    rowClips.appendChild(button);
  }
  const note = document.createElement('span');
  note.className = 'dvnote';
  note.textContent = '▸ dane pozorowane, bez modeli · window.__debug.loadFromUrl(url) uruchamia prawdziwy przebieg';
  rowClips.appendChild(note);

  bar.append(rowStates, rowClips);
  document.body.appendChild(bar);
}

function paintBar() {
  document.querySelectorAll('#devbar [data-state]').forEach((button) => {
    button.classList.toggle('on', button.dataset.state === S.state);
  });
  document.querySelectorAll('#devbar [data-clip]').forEach((button) => {
    button.classList.toggle('on', button.dataset.clip === clipId);
  });
}

function stampHash() {
  try { location.hash = `debug&state=${S.state}&clip=${clipId}`; } catch (err) { /* file:// */ }
}

// ---------------------------------------------------------------------------
export function mount(api) {
  buildBar();

  /**
   * Run the REAL pipeline on an arbitrary audio URL — the example-clip path
   * with the manifest cut out. Also (re)arms `window.__e2e`.
   * @param {string} url
   * @param {{model?: 'sepformer'|'mf2'}} [opts]
   */
  function loadFromUrl(url, opts = {}) {
    if (opts.model && SEPARATORS[opts.model]) {
      S.modelId = opts.model;
      const select = $('model-sel');
      if (select) select.value = opts.model;
    }
    window.__e2e = runFromUrl(url).then(() => summary());
    return window.__e2e;
  }

  window.__debug = {
    setState: (state) => { setFakeState(state); paintBar(); },
    setClip: (id) => { clipId = id; setFakeState(S.state); paintBar(); },
    loadFromUrl,
    runOverride: () => {
      window.__e2eOverride = runOverride().then(() => ({
        state: S.state,
        markers: S.override.markers.map((t) => round(t, 3)),
        streams: ['s1', 's2'].map((key, i) => {
          const samples = i === 0 ? S.override.s1 : S.override.s2;
          return {
            key,
            samples: samples ? samples.length : 0,
            seconds: samples ? round(samples.length / 8000, 3) : 0,
            rms: samples ? round(rms(samples), 5) : 0,
          };
        }),
        corr12: S.override.s1 && S.override.s2 ? round(corr(S.override.s1, S.override.s2)) : null,
      }));
      return window.__e2eOverride;
    },
    probeRegions,
    summary,
    state: S,
    api,
  };

  const params = new URLSearchParams((location.hash || '').replace(/^#/, ''));
  const clip = params.get('clip');
  if (clip === 'A' || clip === 'B') clipId = clip;
  const state = params.get('state');
  if (STATES.includes(state)) setFakeState(state);
  paintBar();
}
