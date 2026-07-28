/* The timeline stack: per-speaker diarization lanes, translucent overlap
   bands, canvas peak-envelope waveform lanes, an adaptive time ruler, one
   shared playhead and click-anywhere-to-seek.

   Lane count comes from the data (design §5: "N lanes from metadata — no
   hardcoded 2"). Waveforms are the v1 simple peak envelope: server-computed
   0-100 buckets mirrored around the midline, zero client-side decode.

   Zoom (1×/2×/4×/8×) is a pure layout trick: the lanes and the ruler live in a
   horizontal scrollport whose inner block is `zoom × 100%` wide. Everything
   inside is positioned in percent of that block, so bars, overlap bands and the
   playhead stay aligned for free, and the click-to-seek ratio keeps working
   against the overlay's own rect. The label gutter is a fixed-pixel grid column
   inside the zoomed block, so it is made sticky while zoomed — its box-shadow
   covers the grid gap that would otherwise show bars sliding past. At 1× the
   scrollport has nothing to scroll and the sticky rules are off. */

import { el, clear } from "./dom.js";
import { clockShort } from "./format.js";
import { onThemeChange } from "./theme.js";

/** CSS custom properties used for speaker colours, in label order. */
const SPEAKER_VARS = ["--spkA", "--spkB", "--spkC", "--spkD"];

const ZOOM_LEVELS = [1, 2, 4, 8];
/** Chromium refuses canvases past 32767 px on an axis; stay clear of it. */
const MAX_CANVAS_PX = 32000;
/** A manual pan wins for this long before the playhead may scroll again. */
const PAN_GRACE_MS = 4000;

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
  canvas.width = Math.max(1, Math.min(Math.round(rect.width * dpr), MAX_CANVAS_PX));
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

  // ruler
  const marks = el("div", { class: "marks" });
  const ruler = el("div", { class: "ruler" }, [el("div", { class: "rgut" }), marks]);

  // scrollport + zoomed inner block
  const inner = el("div", { class: "inner" }, [field, ruler]);
  const port = el("div", { class: "port" }, [inner]);

  // -- zoom control -------------------------------------------------------
  let zoom = 1;
  const zoomButtons = ZOOM_LEVELS.map((level) =>
    el("button", { type: "button", class: level === zoom ? "active" : "", text: `${level}×` })
  );
  const zoomCtl = el("div", { class: "zoomctl" }, [
    el("span", { class: "muted", text: "zoom" }),
    el("div", { class: "seg", role: "group", "aria-label": "Powiększenie osi czasu" }, zoomButtons),
  ]);
  node.appendChild(zoomCtl);
  node.appendChild(port);

  const drawRuler = () => {
    clear(marks);
    // Step from the VISIBLE span, so a zoomed view gets proportionally finer ticks.
    const step = rulerStep(total / zoom);
    for (let t = 0; t <= total + 1e-6; t += step) {
      const at = pct(t, total);
      marks.appendChild(el("span", {
        // A tick landing exactly on the end would poke past the inner block and
        // give the scrollport something to scroll at 1×; hug the edge instead.
        class: at > 99.99 ? "mk end" : "mk",
        style: `left:${at.toFixed(3)}%`,
        text: clockShort(t),
      }));
    }
  };
  drawRuler();

  // -- geometry ----------------------------------------------------------
  // `gutter` = the fixed label column (plus grid gap) inside the zoomed block;
  // `axis` = the seekable width the overlay spans. Both only change on zoom or
  // resize, so the per-frame playhead follow does not re-measure them.
  let gutter = 0;
  let axis = 0;
  let portW = 0;
  function measure() {
    const portRect = port.getBoundingClientRect();
    const axisRect = overlay.getBoundingClientRect();
    portW = portRect.width;
    axis = axisRect.width;
    gutter = axisRect.left - portRect.left + port.scrollLeft;
  }

  let lastProgrammaticPan = 0;
  let panSuppressUntil = 0;
  // Where OUR last pan left the port. A frame that finds scrollLeft anywhere
  // else knows the user moved it — detected synchronously, because the scroll
  // EVENT lands asynchronously and would lose a race against the per-frame
  // playhead follow (seen in the wild as the view snapping back mid-drag).
  let lastSetScroll = 0;
  function panTo(scrollLeft) {
    lastProgrammaticPan = performance.now();
    port.scrollLeft = Math.max(0, scrollLeft);
    lastSetScroll = port.scrollLeft;
  }
  function userMovedPort() {
    if (Math.abs(port.scrollLeft - lastSetScroll) <= 1) return false;
    lastSetScroll = port.scrollLeft;
    panSuppressUntil = performance.now() + PAN_GRACE_MS;
    return true;
  }
  port.addEventListener("scroll", () => {
    if (performance.now() - lastProgrammaticPan < 250) return;
    panSuppressUntil = performance.now() + PAN_GRACE_MS;
    lastSetScroll = port.scrollLeft;
  });

  /** Fraction of the time axis under a viewport x, clamped to [0,1]. */
  function ratioAt(clientX) {
    const rect = overlay.getBoundingClientRect();
    if (!rect.width) return 0;
    return Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
  }

  /**
   * @param {number} level    one of ZOOM_LEVELS
   * @param {object} [keep]   {ratio, clientX} — the time to pin under a pixel;
   *                          without it the current view centre stays centred.
   */
  function setZoom(level, keep) {
    const next = ZOOM_LEVELS.includes(level) ? level : 1;
    measure();
    const ratio = keep ? keep.ratio : ratioAt(port.getBoundingClientRect().left + portW / 2);
    const atPx = keep ? keep.clientX - port.getBoundingClientRect().left : portW / 2;

    zoom = next;
    inner.style.width = next > 1 ? `${next * 100}%` : "";
    node.classList.toggle("zoomed", next > 1);
    zoomButtons.forEach((button, i) => {
      button.classList.toggle("active", ZOOM_LEVELS[i] === next);
    });

    drawRuler();
    measure();
    panTo(gutter + ratio * axis - atPx);
    // The canvases are stretched by the new inner width; repaint at that width.
    scheduleRedraw();
  }

  zoomButtons.forEach((button, i) => {
    button.addEventListener("click", () => setZoom(ZOOM_LEVELS[i]));
  });
  port.addEventListener("wheel", (event) => {
    if (!event.ctrlKey && !event.metaKey) return;
    event.preventDefault();
    const at = ZOOM_LEVELS.indexOf(zoom);
    const next = event.deltaY < 0 ? ZOOM_LEVELS[Math.min(at + 1, ZOOM_LEVELS.length - 1)]
                                  : ZOOM_LEVELS[Math.max(at - 1, 0)];
    if (next === zoom) return;
    setZoom(next, { ratio: ratioAt(event.clientX), clientX: event.clientX });
  }, { passive: false });

  if (onSeek) {
    field.addEventListener("click", (event) => {
      // While zoomed the gutter is sticky and sits over the axis; a click on a
      // lane label is not a seek (at 1× the ratio test already rejected it).
      if (event.target.closest && event.target.closest(".lab")) return;
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
  const onResize = () => { measure(); scheduleRedraw(); };
  if (window.ResizeObserver) {
    // The zoomed inner block is what actually changes width — the outer node
    // keeps the container's width at every zoom level.
    observer = new ResizeObserver(onResize);
    observer.observe(inner);
  } else {
    window.addEventListener("resize", onResize);
  }
  const stopThemeWatch = canvases.length ? onThemeChange(scheduleRedraw) : null;
  scheduleRedraw();
  measure();

  let lastFollowT = null;
  return {
    node,
    redraw: scheduleRedraw,
    setPlayhead(t) {
      playhead.style.left = `${pct(t, total).toFixed(3)}%`;
      if (zoom <= 1) return;                                  // nothing to pan
      // Follow only while the clock advances. This runs every frame, paused
      // included — a paused playhead must never wrestle the view away from
      // wherever the user panned it.
      const moving = lastFollowT !== null && t !== lastFollowT;
      lastFollowT = t;
      if (!moving) return;
      if (userMovedPort()) return;                            // adopt the user's pan
      if (performance.now() < panSuppressUntil) return;       // the user is driving
      if (!axis || !portW) return;
      const x = gutter + (pct(t, total) / 100) * axis;
      const view = port.scrollLeft;
      // The sticky gutter hides the first `gutter` px of the view.
      if (x < view + gutter + 8 || x > view + portW - 8) panTo(x - portW / 2);
    },
    destroy() {
      if (observer) observer.disconnect();
      else window.removeEventListener("resize", onResize);
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
