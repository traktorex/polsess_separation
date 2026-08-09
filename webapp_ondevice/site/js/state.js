/**
 * The single mutable app state. One object, read by `ui.js`, written by
 * `app.js` / `pipeline.js` (and, in dev, by `debug.js` — which is exactly why
 * the UI never reads audio buffers directly: everything the UI needs is either
 * a scalar or a pre-computed peak array, so the debug state-switcher can fill
 * the same fields with synthetic data and get a pixel-identical page).
 *
 * States (design_spec §3): S0 idle · S1 models · S2 decode · S3 OSD ·
 * S4 routed · S5 separating · S6 done · S6b done-no-overlap · S7 override ·
 * E1 error.
 */

/** Peak resolution of the full-clip waveform. Downsampled to the canvas width. */
export const PEAK_BINS = 1600;
/** Peak resolution of a region's mini waveform. */
export const REGION_PEAK_BINS = 400;

export const S = {
  /** @type {'S0'|'S1'|'S2'|'S3'|'S4'|'S5'|'S6'|'S6b'|'S7'|'E1'} */
  state: 'S0',
  /** Where the audio came from: `{name, kind: 'file'|'mic'|'example'}`. */
  input: null,
  /** Key into `models.js` SEPARATORS. */
  modelId: 'sepformer',

  // --- audio (never touched by ui.js) ---
  /** @type {Float32Array|null} mono 16 kHz — OSD input and mix playback. */
  audio16k: null,
  /** @type {Float32Array|null} mono 8 kHz — separator input, built lazily. */
  audio8k: null,
  durationS: 0,

  // --- what the timeline draws ---
  /** @type {Float32Array|null} full-clip peaks, 0..1 */
  peaks: null,
  /** @type {Array<[number, number, 'sil'|'solo'|'ovl']>} OSD frame runs */
  segs: [],
  /** Routed regions — see `pipeline.js::makeRegion`. */
  regions: [],
  /** Seconds of the clip labelled `overlap` by OSD (stats tile 1). */
  overlapS: 0,
  /** Separator calls a whole-clip run would have cost (stats tile 3). */
  wholeCalls: 0,

  // --- progress ---
  /** OSD windows done / total, 0..1 (S3 scan overlay). */
  osdFrac: 0,
  /** Regions finished (S5). */
  sepDone: 0,
  /** Model downloads in flight: `[{key, label, loaded, total, done}]`. */
  dl: [],
  dlTitle: '',

  // --- inspector / transport ---
  /** Selected region index, or null. */
  region: null,
  /** @type {'mix'|'s1'|'s2'} */
  src: 'mix',
  loop: false,
  /** Playhead in the full clip, seconds. */
  playhead: 0,
  /** Playhead inside the selected region, seconds. */
  regHead: 0,
  /** Which group is playing: null | 'clip' | 'region' | 'full'. */
  playing: null,
  /** Which full-length stream button is sounding: null | 's1' | 's2' | 'ov1' | 'ov2'. */
  playingFull: null,

  /** E1 payload: `{title, type, msg, retry, action, failStep}`. */
  err: null,

  /** Override panel ("Separuj mimo wszystko"). */
  override: {
    running: false,
    done: false,
    frac: 0,
    /** @type {Float32Array|null} */ s1: null,
    /** @type {Float32Array|null} */ s2: null,
    /** Phantom "posłuchaj tutaj" marker positions, seconds. */
    markers: [],
  },
};

/** True while a stage is running and the input/model must not change. */
export function isBusy() {
  return ['S1', 'S2', 'S3', 'S5'].includes(S.state) || S.override.running;
}

/** Drop everything derived from a clip. Called when a new input is chosen. */
export function resetRun() {
  S.audio16k = null;
  S.audio8k = null;
  S.durationS = 0;
  S.peaks = null;
  S.segs = [];
  S.regions = [];
  S.overlapS = 0;
  S.wholeCalls = 0;
  S.osdFrac = 0;
  S.sepDone = 0;
  S.dl = [];
  S.region = null;
  S.src = 'mix';
  S.playhead = 0;
  S.regHead = 0;
  S.playing = null;
  S.playingFull = null;
  S.err = null;
  resetSeparation();
}

/** Drop separator output but keep the clip, its OSD result and its routing. */
export function resetSeparation() {
  S.sepDone = 0;
  S.region = null;
  for (const region of S.regions) {
    region.done = false;
    region.frac = 0;
    region.s1 = null;
    region.s2 = null;
    region.peaks = null;
  }
  S.override.running = false;
  S.override.done = false;
  S.override.frac = 0;
  S.override.s1 = null;
  S.override.s2 = null;
  S.override.markers = [];
}
