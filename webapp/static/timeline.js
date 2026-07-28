/* The timeline stack: per-speaker diarization lanes, translucent overlap
   bands, canvas peak-envelope waveform lanes, an adaptive time ruler, one
   shared playhead and click-anywhere-to-seek.

   Lane count comes from the data (design §5: "N lanes from metadata — no
   hardcoded 2"). Waveforms are the v1 simple peak envelope: server-computed
   0-100 buckets mirrored around the midline, zero client-side decode. */

import { el, clear } from "./dom.js";
import { clockShort } from "./format.js";
import { onThemeChange } from "./theme.js";

/** CSS custom properties used for speaker colours, in label order. */
const SPEAKER_VARS = ["--spkA", "--spkB", "--spkC", "--spkD"];

export function colorVarFor(index) {
  return SPEAKER_VARS[index % SPEAKER_VARS.length];
}

function resolve(varName) {
  const value = getComputedStyle(document.documentElement).getPropertyValue(varName);
  return (value || "").trim() || "#1f6feb";
}

function pct(value, total) {
  if (!total || !isFinite(total)) return 0;
  return Math.max(0, Math.min(100, (value / total) * 100));
}

function rulerStep(duration) {
  if (duration > 1200) return 300;
  if (duration > 600) return 120;
  if (duration > 240) return 60;
  if (duration > 120) return 30;
  if (duration > 40) return 10;
  return 5;
}

/** Draw one peak-envelope lane: mirrored bars around the midline. */
function drawWave(canvas, peaks, color) {
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.max(1, Math.round(rect.width * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  const values = peaks && peaks.length ? peaks : [];
  if (!values.length) return;
  const mid = H / 2;
  const half = H / 2 - 2 * dpr;
  const step = W / values.length;
  const barW = Math.max(step * 0.85, 1);
  ctx.fillStyle = color;
  for (let i = 0; i < values.length; i += 1) {
    const value = Number(values[i]) || 0;
    const h = Math.max((value / 100) * half, value > 0 ? 0.5 : 0);
    if (h <= 0) continue;
    ctx.fillRect(i * step, mid - h, barW, h * 2);
  }
}

/**
 * @param {object} spec
 * @param {number} spec.duration        total seconds (the shared clock)
 * @param {Array}  spec.lanes           [{label, sub, colorVar, turns:[{start,end}]}]
 * @param {Array}  spec.waves           [{label, colorVar, peaks:[int]}]
 * @param {Array}  spec.overlaps        [{start,end}] full-height bands
 * @param {Function} [spec.onSeek]      t => void; enables click-to-seek
 */
export function createTimeline({ duration, lanes = [], waves = [], overlaps = [], onSeek = null }) {
  const total = duration && isFinite(duration) && duration > 0 ? duration : 1;
  const node = el("div", { class: onSeek ? "tl seekable" : "tl" });
  const field = el("div", { class: "field" });
  const canvases = [];

  const laneRow = (label, sub, colorVar, kids, isWave) => {
    const dot = el("span", { class: "dot", style: `background: var(${colorVar})` });
    const labelParts = [el("span", { text: label })];
    if (sub) labelParts.push(el("span", { class: "id", text: sub }));
    labelParts.push(dot);
    return el("div", { class: "lane" }, [
      el("div", { class: "lab" }, labelParts),
      el("div", { class: isWave ? "track wf" : "track" }, kids),
    ]);
  };

  for (const lane of lanes) {
    const bars = (lane.turns || []).map((turn) => {
      const start = Number(turn.start) || 0;
      const end = Number(turn.end) || start;
      return el("div", {
        class: "bar",
        style: `left:${pct(start, total).toFixed(3)}%;width:${pct(end - start, total).toFixed(3)}%;background:var(${lane.colorVar})`,
        title: `${clockShort(start)} – ${clockShort(end)}`,
      });
    });
    field.appendChild(laneRow(lane.label, lane.sub, lane.colorVar, bars, false));
  }

  for (const wave of waves) {
    const canvas = el("canvas", { "aria-hidden": "true" });
    canvases.push({ canvas, wave });
    field.appendChild(laneRow(wave.label, wave.sub, wave.colorVar, [canvas], true));
  }

  const overlay = el("div", { class: "overlay" });
  for (const band of overlaps || []) {
    const start = Number(band.start) || 0;
    const end = Number(band.end) || start;
    overlay.appendChild(el("div", {
      class: "ovl-band",
      style: `left:${pct(start, total).toFixed(3)}%;width:${pct(end - start, total).toFixed(3)}%`,
    }));
  }
  const playhead = el("div", { class: "playhead", style: "left:0%" });
  overlay.appendChild(playhead);
  field.appendChild(overlay);
  node.appendChild(field);

  // ruler
  const marks = el("div", { class: "marks" });
  const ruler = el("div", { class: "ruler" }, [el("div"), marks]);
  node.appendChild(ruler);
  const drawRuler = () => {
    clear(marks);
    const step = rulerStep(total);
    for (let t = 0; t <= total + 1e-6; t += step) {
      marks.appendChild(el("span", {
        class: "mk",
        style: `left:${pct(t, total).toFixed(3)}%`,
        text: clockShort(t),
      }));
    }
  };
  drawRuler();

  if (onSeek) {
    field.addEventListener("click", (event) => {
      const rect = overlay.getBoundingClientRect();
      if (!rect.width) return;
      const ratio = (event.clientX - rect.left) / rect.width;
      if (ratio < 0 || ratio > 1) return;      // the label gutter: not a seek
      onSeek(ratio * total);
    });
  }

  const redraw = () => {
    for (const item of canvases) drawWave(item.canvas, item.wave.peaks, resolve(item.wave.colorVar));
  };

  let scheduled = false;
  const scheduleRedraw = () => {
    if (scheduled) return;
    scheduled = true;
    requestAnimationFrame(() => { scheduled = false; redraw(); });
  };

  let observer = null;
  if (window.ResizeObserver && canvases.length) {
    observer = new ResizeObserver(scheduleRedraw);
    observer.observe(node);
  } else if (canvases.length) {
    window.addEventListener("resize", scheduleRedraw);
  }
  const stopThemeWatch = canvases.length ? onThemeChange(scheduleRedraw) : null;
  scheduleRedraw();

  return {
    node,
    redraw: scheduleRedraw,
    setPlayhead(t) {
      playhead.style.left = `${pct(t, total).toFixed(3)}%`;
    },
    destroy() {
      if (observer) observer.disconnect();
      else if (canvases.length) window.removeEventListener("resize", scheduleRedraw);
      if (stopThemeWatch) stopThemeWatch();
    },
  };
}

/** Legend row matching the timeline (speakers + overlap band + playhead). */
export function timelineLegend(speakerLabels) {
  const items = speakerLabels.map((entry, index) =>
    el("span", {}, [
      el("span", { class: "sw", style: `background: var(${colorVarFor(index)})` }),
      `Mówca ${entry}`,
    ])
  );
  items.push(el("span", {}, [
    el("span", { class: "sw", style: "background: var(--ovl); border: 1px solid var(--ovl-line)" }),
    "nakładanie → separator",
  ]));
  items.push(el("span", { style: "color: var(--fail)", text: "▏kursor odtwarzania" }));
  return el("div", { class: "legend" }, items);
}
