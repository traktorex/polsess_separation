/* Transport: one custom player over N hidden <audio> elements, a segmented
   stream selector, a 0-300% volume control, and the shared
   requestAnimationFrame clock that drives the playhead and the karaoke.

   Two rows: play + clock + seek (the seek gets the whole width), then the
   stream buttons and the volume. The selector used to share the row with the
   seek and squeezed it to nothing on a wide layout.

   Stream switching is the review page's proven element swap: pause the old
   element, copy currentTime to the new one, resume if it was playing.

   Web Audio is opt-in and per element. `createMediaElementSource` may be called
   only ONCE per element for the lifetime of the page, so every routed element
   keeps its {source, panner, gain} chain in `routed` and is from then on
   controlled by its GainNode alone (element.volume pinned to 1). Two things
   need the graph: "Oba" (A→left / B→right panning) and any volume above 100%,
   which no HTMLMediaElement can give. Everything else keeps playing straight
   out of the element, exactly as before. */

import { el, clear } from "./dom.js";
import { clock } from "./format.js";

const NBSP_SLASH = " / ";
const VOL_MAX = 300;

function audioEl(src) {
  const node = el("audio", { preload: "metadata", src });
  node.style.display = "none";
  return node;
}

/**
 * @param {object} spec
 * @param {object} spec.files      result.files (mixture / stream_<label> URLs)
 * @param {string[]} spec.labels   speaker labels in display order
 * @param {number} spec.duration   total seconds
 */
export function createPlayer({ files = {}, labels = [], duration = 0, onModeChange = null }) {
  const streamLabels = labels.filter((label) => files[`stream_${label}`]);
  const elements = [];      // every element, for global play/pause wiring
  const modes = [];         // {key, node(button content), get element(s)}

  const single = (src) => {
    const node = audioEl(src);
    elements.push(node);
    return node;
  };

  if (files.mixture) {
    modes.push({ key: "mixture", text: ["Miks"], media: [single(files.mixture)] });
  }
  const soloFor = {};
  streamLabels.forEach((label, index) => {
    soloFor[label] = single(files[`stream_${label}`]);
    const span = el("span", { class: index === 0 ? "sA" : index === 1 ? "sB" : "", text: label });
    modes.push({ key: `solo:${label}`, text: [span], media: [soloFor[label]] });
  });

  let bothA = null;
  let bothB = null;
  let bothButton = null;
  if (streamLabels.length >= 2) {
    bothA = single(files[`stream_${streamLabels[0]}`]);
    bothB = single(files[`stream_${streamLabels[1]}`]);
    const bothText = [
      "Oba (",
      el("span", { class: "sA", text: streamLabels[0] }),
      "→L · ",
      el("span", { class: "sB", text: streamLabels[1] }),
      "→P)",
    ];
    // Insert straight after the first solo stream, matching the accepted layout
    // Miks | A | Oba | B.
    const at = modes.findIndex((m) => m.key === `solo:${streamLabels[1]}`);
    const entry = { key: "both", text: bothText, media: [bothA, bothB] };
    if (at >= 0) modes.splice(at, 0, entry);
    else modes.push(entry);
  }

  if (!modes.length) {
    return {
      node: el("div", { class: "muted", text: "Brak plików audio dla tego przebiegu." }),
      currentTime: () => 0,
      seek() {}, play() {}, pause() {}, seekAndPlay() {},
      onTick() {}, destroy() {},
    };
  }

  let mode = modes.find((m) => m.key === "both") || modes[0];
  const primary = () => mode.media[0];

  // -- Web Audio: one lazy graph, one chain per routed element -------------
  let audioCtx = null;
  let ctxFailed = false;
  const routed = new Map();          // media -> {source, panner|null, gain}
  let panState = "idle";             // idle -> ready | failed
  let volume = 100;

  function ensureCtx() {
    if (audioCtx || ctxFailed) return audioCtx;
    try {
      const Ctx = window.AudioContext || window.webkitAudioContext;
      if (!Ctx) throw new Error("AudioContext unavailable");
      audioCtx = new Ctx();
    } catch (err) {
      ctxFailed = true;
      audioCtx = null;
    }
    return audioCtx;
  }
  function resumeCtx() {
    if (audioCtx && audioCtx.state === "suspended") audioCtx.resume().catch(() => {});
  }

  /** The "Oba" pair always gets its panner, whichever need routes it first —
   *  a source cannot be re-created later to insert one. */
  const panOf = (media) => (media === bothA ? -1 : media === bothB ? 1 : null);

  /** Route one element through the graph. Returns its chain, or null if the
   *  browser refused (no AudioContext, or the element is already captured). */
  function routeMedia(media) {
    if (routed.has(media)) return routed.get(media);
    const ctx = ensureCtx();
    if (!ctx) return null;
    try {
      const source = ctx.createMediaElementSource(media);
      const pan = panOf(media);
      let tail = source;
      let panner = null;
      if (pan !== null && ctx.createStereoPanner) {
        panner = ctx.createStereoPanner();
        panner.pan.value = pan;
        tail = tail.connect(panner);
      }
      const gain = ctx.createGain();
      gain.gain.value = volume / 100;
      tail.connect(gain).connect(ctx.destination);
      media.volume = 1;
      const chain = { source, panner, gain };
      routed.set(media, chain);
      return chain;
    } catch (err) {
      return null;
    }
  }

  function setupPan() {
    if (panState !== "idle" || !bothA || !bothB) return;
    const a = routeMedia(bothA);
    const b = routeMedia(bothB);
    if (a && b && a.panner && b.panner) {
      panState = "ready";
      return;
    }
    panState = "failed";
    if (bothButton) {
      clear(bothButton).appendChild(document.createTextNode("Oba"));
      bothButton.title =
        "Przeglądarka nie pozwoliła rozdzielić kanałów — oba strumienie grają razem, bez panoramy.";
    }
  }

  // -- controls ----------------------------------------------------------
  const playBtn = el("button", { class: "playbtn", type: "button", "aria-label": "Odtwórz", text: "▶" });
  const timeLabel = el("span", { class: "ttime", text: `0:00,0${NBSP_SLASH}${clock(duration)}` });
  const seek = el("input", {
    type: "range", class: "seek", min: "0", step: "0.01",
    max: String(duration > 0 ? duration : 1), value: "0",
    "aria-label": "Pozycja odtwarzania",
  });
  const status = el("span", { class: "muted", style: "font-size:12.5px" });

  const segGroup = el("div", { class: "seg", role: "group", "aria-label": "Wybór strumienia" });
  for (const entry of modes) {
    const button = el("button", { type: "button", class: entry.key === mode.key ? "active" : "" }, entry.text);
    if (entry.key === "both") bothButton = button;
    button.addEventListener("click", () => {
      if (entry.key === "both") setupPan();
      switchTo(entry);
      for (const other of segGroup.children) other.classList.remove("active");
      button.classList.add("active");
    });
    segGroup.appendChild(button);
  }

  const volRange = el("input", {
    type: "range", class: "seek volrange", min: "0", max: String(VOL_MAX), step: "5", value: "100",
    "aria-label": "Głośność",
  });
  const volValue = el("span", { class: "volval", text: "100%" });
  const volBox = el("div", { class: "vol" }, [
    el("span", { text: "głośność" }), volRange, volValue,
  ]);

  /** Amplification is impossible here: fall back to the element's own volume
   *  and say why, instead of leaving a slider that silently does nothing. */
  function refuseBoost() {
    volume = 100;
    volRange.max = "100";
    volRange.value = "100";
    volRange.title = "Przeglądarka nie pozwoliła na wzmocnienie ponad 100%.";
    paintVolume();
  }
  function paintVolume() {
    for (const media of elements) {
      const chain = routed.get(media);
      if (chain) { media.volume = 1; chain.gain.gain.value = volume / 100; }
      else media.volume = Math.min(1, volume / 100);
    }
    volValue.textContent = `${volume}%`;
    volValue.classList.toggle("boost", volume > 100);
  }
  function applyVolume(next) {
    volume = Math.max(0, Math.min(VOL_MAX, Math.round(next)));
    if (volume > 100) {
      resumeCtx();
      for (const media of elements) {
        if (!routeMedia(media)) { refuseBoost(); return; }
      }
      resumeCtx();
    }
    paintVolume();
  }
  volRange.addEventListener("input", () => applyVolume(parseFloat(volRange.value) || 0));

  const row1 = el("div", { class: "transport" }, [playBtn, timeLabel, seek]);
  const row2 = el("div", { class: "transport2" }, [segGroup, volBox]);
  const wrapper = el("div", {}, [row1, row2, status]);
  for (const media of elements) wrapper.appendChild(media);

  // -- behaviour ---------------------------------------------------------
  let scrubbing = false;
  let lastTimeText = "";
  let total = duration && isFinite(duration) && duration > 0 ? duration : 0;
  const tickCallbacks = [];

  function isPlaying() {
    return mode.media.some((m) => !m.paused && !m.ended);
  }
  function paintPlayIcon() {
    const playing = isPlaying();
    playBtn.textContent = playing ? "❚❚" : "▶";
    playBtn.setAttribute("aria-label", playing ? "Pauza" : "Odtwórz");
  }
  function play() {
    if (mode.key === "both") setupPan();
    resumeCtx();                       // a routed element is silent while suspended
    const at = primary().currentTime;
    for (const media of mode.media) {
      // Only the paired mode needs a sync assignment; re-assigning an
      // element's own currentTime would make it re-seek for nothing.
      if (mode.media.length > 1 && media !== mode.media[0]) media.currentTime = at;
      const attempt = media.play();
      if (attempt && attempt.catch) attempt.catch(() => {});
    }
  }
  function pause() {
    for (const media of mode.media) media.pause();
  }
  function seekTo(t) {
    const target = Math.max(0, total > 0 ? Math.min(t, total) : t);
    for (const media of mode.media) media.currentTime = target;
  }
  function switchTo(entry) {
    if (entry.key === mode.key) return;
    const at = primary().currentTime;
    const wasPlaying = isPlaying();
    pause();
    mode = entry;
    for (const media of mode.media) media.currentTime = at;
    if (wasPlaying) play();
    paintPlayIcon();
    if (onModeChange) onModeChange(mode.key);
  }

  playBtn.addEventListener("click", () => (isPlaying() ? pause() : play()));
  seek.addEventListener("pointerdown", () => { scrubbing = true; });
  seek.addEventListener("pointerup", () => { scrubbing = false; });
  seek.addEventListener("input", () => seekTo(parseFloat(seek.value) || 0));
  for (const media of elements) {
    media.addEventListener("play", paintPlayIcon);
    media.addEventListener("pause", paintPlayIcon);
    media.addEventListener("ended", paintPlayIcon);
    media.addEventListener("error", () => {
      status.textContent = "Nie udało się wczytać pliku audio.";
    });
    media.addEventListener("loadedmetadata", () => {
      // The result carries the duration; this is the fallback when it does not.
      if (!total && isFinite(media.duration) && media.duration > 0) {
        total = media.duration;
        seek.max = String(total);
        lastTimeText = "";
      }
    });
  }

  let raf = null;
  function loop() {
    const at = primary().currentTime || 0;
    // Keep the "Oba" pair together; two independent elements can drift after a
    // buffering hiccup, and 0,15 s is where a listener starts to hear it.
    if (mode.key === "both" && mode.media.length > 1 && !mode.media[1].paused) {
      if (Math.abs(mode.media[1].currentTime - at) > 0.15) mode.media[1].currentTime = at;
    }
    if (!scrubbing) {
      seek.value = String(at);
      const text = `${clock(at)}${NBSP_SLASH}${clock(total)}`;
      if (text !== lastTimeText) {
        timeLabel.textContent = text;
        lastTimeText = text;
      }
    }
    for (const callback of tickCallbacks) callback(at);
    raf = requestAnimationFrame(loop);
  }
  raf = requestAnimationFrame(loop);

  return {
    node: wrapper,
    currentTime: () => primary().currentTime || 0,
    seek: seekTo,
    play,
    pause,
    seekAndPlay(t) { seekTo(t); play(); },
    onTick(callback) { tickCallbacks.push(callback); },
    destroy() {
      if (raf) cancelAnimationFrame(raf);
      pause();
      for (const media of elements) { media.removeAttribute("src"); media.load(); }
      if (audioCtx && audioCtx.close) audioCtx.close().catch(() => {});
    },
  };
}
