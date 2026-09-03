/**
 * Frame grid + overlap routing — the pure half of the OSD stage.
 *
 * Port of `webapp_ondevice/reference/osd_reference.py`:
 *   - frame geometry            (reference lines 149-160, 202-211)
 *   - `frames_to_intervals`     (reference lines 263-283)
 *   - `route_overlaps`          (reference lines 301-313, via `_merge_close` 286-298)
 *   - `_boundary_aware_pad` / `context_pad` (reference lines 316-364)
 *
 * The routing constants themselves come from the server pipeline —
 * `asr_pipeline/config.py:244-245` (RoutingConfig) and `:360,372,382`
 * (SeparationConfig) — see `reference/README.md` §4 for the citation table.
 *
 * This module imports nothing. The frame geometry lives here rather than in
 * `osd.js` so that everything time-domain can be exercised without pulling in
 * the 12 MB ONNX Runtime WASM binary; `osd.js` imports the geometry from here.
 *
 * Time convention: all interval endpoints are seconds from the start of the
 * clip, `[start, end)`.
 */

/** Model sample rate. pyannote-segmentation-3.0 is 16 kHz only. */
export const SAMPLE_RATE = 16000;
/** `preprocessor_config.json` "step" — 16.875 ms. */
export const FRAME_STEP_SAMPLES = 270;
/** `preprocessor_config.json` "offset" — the 61.875 ms receptive field. */
export const FRAME_WIDTH_SAMPLES = 990;
export const FRAME_STEP_S = FRAME_STEP_SAMPLES / SAMPLE_RATE; // 0.016875

/** Per-frame labels. The demo only ever uses the speaker COUNT, never identity. */
export const NONSPEECH = 0;
export const SOLO = 1;
export const OVERLAP = 2;
export const LABEL_NAMES = ['nonspeech', 'solo', 'overlap'];

/** Routing constants, `asr_pipeline/config.py:244-245`. */
export const MIN_OVERLAP_DUR_S = 0.2;
export const MERGE_GAP_S = 0.5;
/** Separator context window, `asr_pipeline/config.py:360,382` (`expand_to_chunk`). */
export const CHUNK_S = 4.0;

/**
 * Number of frames the model emits for `nSamples` input.
 * `floor((T - 990) / 270) + 1`, 0 below one frame width (reference:202-206).
 * @param {number} nSamples
 * @returns {number}
 */
export function frameCount(nSamples) {
  if (nSamples < FRAME_WIDTH_SAMPLES) return 0;
  return Math.floor((nSamples - FRAME_WIDTH_SAMPLES) / FRAME_STEP_SAMPLES) + 1;
}

/**
 * Centre of frame k in seconds — pyannote's `SlidingWindow[k].middle`
 * (reference:209-211).
 * @param {number} k
 * @returns {number}
 */
export function frameMiddleS(k) {
  return (k * FRAME_STEP_SAMPLES + FRAME_WIDTH_SAMPLES / 2) / SAMPLE_RATE;
}

/**
 * Contiguous runs of `target` in a per-frame label array -> intervals.
 *
 * A run of frames k0..k1 becomes `[middle(k0) - step/2, middle(k1) + step/2]`,
 * so N frames span exactly N x 16.875 ms (reference:263-283). The half-frame
 * widening matters: `minDur` below is a *duration* test, and dropping 16.9 ms
 * of every region to a timestamping convention would be a silent behavioural
 * difference from the server.
 *
 * @param {Uint8Array|number[]} frameClasses per-frame labels
 * @param {number} target label to extract (usually `OVERLAP`)
 * @returns {Array<[number, number]>} intervals in seconds, in time order
 */
export function framesToIntervals(frameClasses, target) {
  const out = [];
  const half = FRAME_STEP_S / 2;
  let runStart = -1;
  for (let k = 0; k < frameClasses.length; k++) {
    const hit = frameClasses[k] === target;
    if (hit && runStart < 0) runStart = k;
    if (!hit && runStart >= 0) {
      out.push([frameMiddleS(runStart) - half, frameMiddleS(k - 1) + half]);
      runStart = -1;
    }
  }
  if (runStart >= 0) {
    out.push([
      frameMiddleS(runStart) - half,
      frameMiddleS(frameClasses.length - 1) + half,
    ]);
  }
  return out;
}

/**
 * Merge intervals whose gap is strictly below `mergeGap`
 * (reference:286-298, itself a port of `asr_pipeline/stages/routing.py::_merge_close`).
 * Strict `<` is the server's comparison — keep it strict.
 *
 * @param {Array<[number, number]>} segs
 * @param {number} mergeGap seconds
 * @returns {Array<[number, number]>}
 */
function mergeClose(segs, mergeGap) {
  if (segs.length === 0) return [];
  const sorted = segs.map((s) => [s[0], s[1]]).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const out = [sorted[0]];
  for (let i = 1; i < sorted.length; i++) {
    const [s, e] = sorted[i];
    const last = out[out.length - 1];
    if (s - last[1] < mergeGap) last[1] = Math.max(last[1], e);
    else out.push([s, e]);
  }
  return out;
}

/**
 * Drop short overlaps, then merge close ones — the server's order
 * (reference:301-313, `_select_overlap_regions`).
 *
 * Filtering FIRST is load-bearing: reversing it would resurrect a
 * sub-threshold overlap by gluing it to a neighbour.
 *
 * @param {Array<[number, number]>} intervals raw overlap intervals
 * @param {{minDur?: number, mergeGap?: number}} [opts]
 * @returns {Array<[number, number]>} routed regions
 */
export function selectOverlapRegions(intervals, opts = {}) {
  const minDur = opts.minDur ?? MIN_OVERLAP_DUR_S;
  const mergeGap = opts.mergeGap ?? MERGE_GAP_S;
  const kept = intervals.filter(([s, e]) => e - s >= minDur); // `>=`: 0.20 s survives
  return mergeClose(kept, mergeGap);
}

/**
 * Boundary-aware symmetric padding to a target total duration
 * (reference:316-337, verbatim from `asr_pipeline/stages/separation.py`).
 * Splits the deficit evenly, and pushes whatever one side cannot absorb
 * (because the region is near a clip boundary) onto the other side.
 */
function boundaryAwarePad(startS, endS, totalDurationS, targetTotalS) {
  const overlapDur = endS - startS;
  if (targetTotalS <= overlapDur) return [startS, endS]; // already long enough
  const extra = targetTotalS - overlapDur;
  const roomLeft = startS;
  const roomRight = Math.max(0, totalDurationS - endS);
  let leftTake = extra / 2;
  let rightTake = extra / 2;
  if (leftTake > roomLeft) {
    rightTake += leftTake - roomLeft;
    leftTake = roomLeft;
  }
  if (rightTake > roomRight) {
    leftTake += rightTake - roomRight;
    rightTake = roomRight;
  }
  leftTake = Math.min(leftTake, roomLeft);
  rightTake = Math.min(rightTake, roomRight);
  return [startS - leftTake, endS + rightTake];
}

/**
 * `context_window_mode: expand_to_chunk` — pad every routed region out to the
 * separator's training window (reference:340-364; the shipped server configs
 * `asr_pipeline/configs/default.yaml:43-45` and
 * `sweep_best_e31_refineplus.yaml:80-82` both select this mode).
 *
 * 4.0 s @ 8 kHz is exactly the separator ONNX's 32000-sample input, so one
 * padded region is normally one separator call. **A region already longer than
 * `chunkS` comes back unpadded** and must be chunked + overlap-added by the
 * caller (`separate.js` does this).
 *
 * NB the plan doc's "context_pad = 1.0 s" is the `fixed_pad` knob, which no
 * shipped config selects — see `reference/README.md` §4.
 *
 * @param {Array<[number, number]>} regions routed regions
 * @param {number} chunkS target window, seconds (4.0)
 * @param {number} clipDurS full clip duration, seconds (the padding boundary)
 * @returns {Array<[number, number]>} padded regions, same order and count
 */
export function expandToChunk(regions, chunkS, clipDurS) {
  return regions.map(([s, e]) => boundaryAwarePad(s, e, clipDurS, chunkS));
}
