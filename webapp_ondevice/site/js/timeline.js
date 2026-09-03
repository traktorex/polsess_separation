/**
 * The timeline: waveform canvas + percent-positioned band overlays.
 *
 * Ported from the accepted mockup's rendering code
 * with one change: the mockup synthesised its peaks, this reads them from a
 * pre-computed peak array. Everything is laid out in **percent of clip
 * duration**, so a resize needs no re-layout — only the canvas is repainted
 * (bitmap size follows CSS size × dpr) and the ruler is re-stepped, because its
 * spacing is chosen in pixels, not in seconds.
 *
 * Colours come from CSS custom properties, so a theme change means a repaint.
 */

/**
 * Peak envelope of a signal: max |x| per bin, normalised.
 *
 * @param {Float32Array} samples
 * @param {number} bins output length
 * @param {number} [scale] divisor (default: the signal's own peak). Pass the
 *        mixture's peak when drawing separated streams, so the two speakers
 *        keep their relative level instead of both filling the box.
 * @returns {Float32Array} `bins` values, nominally 0..1
 */
export function computePeaks(samples, bins, scale) {
  const out = new Float32Array(bins);
  if (!samples || samples.length === 0) return out;
  const per = samples.length / bins;
  let maxAll = 0;
  for (let b = 0; b < bins; b++) {
    const lo = Math.floor(b * per);
    const hi = Math.min(samples.length, Math.max(lo + 1, Math.floor((b + 1) * per)));
    let peak = 0;
    for (let i = lo; i < hi; i++) {
      const v = Math.abs(samples[i]);
      if (v > peak) peak = v;
    }
    out[b] = peak;
    if (peak > maxAll) maxAll = peak;
  }
  const div = scale || maxAll;
  if (div > 0) for (let b = 0; b < bins; b++) out[b] /= div;
  return out;
}

/** Current value of a CSS custom property, with a readable fallback. */
export function cssVar(name) {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name);
  return (v || '').trim() || '#59636e';
}

/**
 * Paint a peak array as a symmetric bar waveform.
 * @param {HTMLCanvasElement} canvas
 * @param {Float32Array|null} peaks values 0..1 (any length; downsampled by max)
 * @param {string} color
 */
export function drawWave(canvas, peaks, color) {
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.max(1, Math.round(rect.width * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const W = canvas.width;
  const H = canvas.height;
  const mid = H / 2;
  const half = H / 2 - 3 * dpr;
  ctx.clearRect(0, 0, W, H);
  if (!peaks || !peaks.length) return;

  const bars = Math.max(40, Math.min(peaks.length, Math.round(rect.width / 2.2)));
  const step = W / bars;
  const barW = Math.max(step * 0.8, 1);
  const per = peaks.length / bars;
  ctx.fillStyle = color;
  for (let i = 0; i < bars; i++) {
    const lo = Math.floor(i * per);
    const hi = Math.min(peaks.length, Math.max(lo + 1, Math.floor((i + 1) * per)));
    let v = 0;
    for (let k = lo; k < hi; k++) if (peaks[k] > v) v = peaks[k];
    const h = Math.max(Math.min(v, 1) * half, 0.5 * dpr);
    ctx.fillRect(i * step, mid - h, barW, h * 2);
  }
}

const pct = (v, total) => Math.max(0, Math.min(100, (v / (total || 1)) * 100));
const BAND_NAME = { sil: 'cisza', solo: 'solo', ovl: 'nakładanie' };

/**
 * Non-speech / solo / overlap bands.
 * @param {HTMLElement} host
 * @param {Array<[number, number, string]>} segs
 * @param {number} durationS
 */
export function renderBands(host, segs, durationS) {
  host.innerHTML = '';
  for (const [start, end, kind] of segs) {
    const el = document.createElement('div');
    el.className = `band ${kind}`;
    el.style.left = `${pct(start, durationS).toFixed(3)}%`;
    el.style.width = `${pct(end - start, durationS).toFixed(3)}%`;
    el.title = `${BAND_NAME[kind]} ${start.toFixed(1)}–${end.toFixed(1)} s`;
    host.appendChild(el);
  }
}

/**
 * Routed regions: the dashed expand-to-chunk windows the separator actually
 * runs on, with their per-region progress bar and, once finished, a play glyph.
 *
 * Clicks are handled by the caller through delegation on `host` — every region
 * carries `data-i` and `data-ready`.
 *
 * @param {HTMLElement} host
 * @param {Array<object>} regions
 * @param {number} durationS
 * @param {{selected: number|null, showBars: boolean}} opts
 */
export function renderRegions(host, regions, durationS, opts) {
  host.innerHTML = '';
  regions.forEach((region, i) => {
    const [start, end] = region.pad;
    const el = document.createElement('div');
    const fill = region.done ? 100 : Math.round((region.frac || 0) * 100);
    let cls = 'reg';
    if (region.done) cls += ' done';
    else if (region.frac > 0) cls += ' working';
    else cls += ' pending';
    if (opts.selected === i) cls += ' sel';
    el.className = cls;
    el.dataset.i = String(i);
    el.dataset.ready = region.done ? '1' : '0';
    el.style.left = `${pct(start, durationS).toFixed(3)}%`;
    el.style.width = `${pct(end - start, durationS).toFixed(3)}%`;
    el.title = `Region nakładania #${i + 1} · ${start.toFixed(1)}–${end.toFixed(1)} s`;

    const tag = document.createElement('span');
    tag.className = 'rtag';
    tag.textContent = `#${i + 1}`;
    el.appendChild(tag);

    if (region.done) {
      const play = document.createElement('span');
      play.className = 'rplay';
      play.textContent = '▶';
      el.appendChild(play);
    }
    if (opts.showBars) {
      const bar = document.createElement('span');
      bar.className = 'sepbar';
      const inner = document.createElement('i');
      inner.style.width = `${fill}%`;
      bar.appendChild(inner);
      el.appendChild(bar);
    }
    host.appendChild(el);
  });
}

/**
 * „posłuchaj tutaj” markers on solo bands (override result only).
 * Labels are wider than the band near the clip edges, so they stick to the edge
 * rather than get clipped by the track's overflow.
 */
export function renderMarkers(host, positionsS, durationS) {
  host.innerHTML = '';
  for (const t of positionsS) {
    const el = document.createElement('span');
    el.className = 'marker';
    const p = pct(t, durationS);
    if (p < 16) { el.style.left = '4px'; el.style.transform = 'none'; }
    else if (p > 84) { el.style.right = '4px'; el.style.transform = 'none'; }
    else el.style.left = `${p.toFixed(3)}%`;
    el.textContent = 'posłuchaj tutaj';
    host.appendChild(el);
  }
}

const RULER_STEPS = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600];
const RULER_MIN_PX = 54;

/**
 * Time ruler. The step is chosen in PIXELS, not seconds: the same 38 s clip
 * fits 8 ticks on a desktop and 4 on a phone.
 * @param {HTMLElement} host
 * @param {number} durationS
 * @param {number} widthPx measured width of the track
 * @param {(s: number) => string} fmt
 */
export function renderRuler(host, durationS, widthPx, fmt) {
  host.innerHTML = '';
  const w = widthPx || 640;
  let step = RULER_STEPS[RULER_STEPS.length - 1];
  for (const candidate of RULER_STEPS) {
    if ((candidate / durationS) * w >= RULER_MIN_PX) { step = candidate; break; }
  }
  const cutoff = 100 - (RULER_MIN_PX / w) * 100; // the last tick gets its own slot
  for (let t = 0; t <= durationS + 1e-6; t += step) {
    const at = pct(t, durationS);
    if (at > cutoff) break;
    const el = document.createElement('span');
    el.className = 'mk';
    el.style.left = `${at.toFixed(3)}%`;
    el.textContent = fmt(t);
    host.appendChild(el);
  }
  const last = document.createElement('span');
  last.className = 'mk end';
  last.style.left = '100%';
  last.textContent = fmt(durationS);
  host.appendChild(last);
}
