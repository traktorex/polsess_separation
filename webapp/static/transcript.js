/* Speaker transcript columns with word-level karaoke.

   One column per stream label (N columns, not two by assumption), each a tier
   of {id, start, end, text, words} — exactly the .eaf / WhisperX shape, so the
   read-only columns can later be swapped for an ELAN-style tier editor without
   touching the timeline or the transport (design §5.3 ELAN future-proofing).
   Segments keep their server-side stable ids in `data-segment-id`.

   Two behaviours are ported verbatim from the clarin_review page, because the
   author reads transcripts that way:
   - the scroll ANCHOR is the last segment that started at or before the clock,
     not the segment containing it. A speaker is silent most of the time; an
     anchor that vanishes in every gap leaves nothing to scroll to. The `.cur`
     highlight and the word karaoke still require real containment.
   - following re-engages at every anchor change. A manual scroll only
     suppresses it until the next turn starts, so the panel never strands the
     reader minutes away from the playhead. */

import { el, clear } from "./dom.js";
import { clock } from "./format.js";
import { colorVarFor } from "./timeline.js";

const LOW_SCORE = 0.45;
const SOFT_VARS = ["--spkA-soft", "--spkB-soft", "--spkC-soft", "--spkD-soft"];

/* Highlight cap (N13). When WhisperX alignment is weak it stretches the last
   word — and the segment — across the trailing silence (seen in the wild: a
   17 s "ciekawostka."). The DATA stays untouched; only the highlight treats a
   word as over MAX_WORD_HL_S after it starts, and a row as over once its last
   capped word is (falling back to the segment end when no word has times). */
const MAX_WORD_HL_S = 2.0;

/* "rozwiń całość" is global (N3): the columns are read side by side, so one
   expanded column next to a short one is never what the reader wanted. Every
   live column registers here and follows the shared flag. */
const liveColumns = new Set();
let globalExpanded = false;

function setGlobalExpanded(on) {
  globalExpanded = on;
  for (const column of liveColumns) column.setExpanded(on);
}

/** Last index with start <= t. -1 only before the first segment. */
function lastAtOrBefore(items, t) {
  let lo = 0;
  let hi = items.length - 1;
  let best = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (items[mid].start <= t) { best = mid; lo = mid + 1; } else hi = mid - 1;
  }
  return best;
}

/** Last index with start <= t, provided t is still inside that item. */
function activeIndex(items, t) {
  const best = lastAtOrBefore(items, t);
  return best >= 0 && t <= items[best].end ? best : -1;
}

function buildColumn({ label, title, subtitle, colorIndex, segments, onSeek }) {
  const body = el("div", { class: "tbody" });
  const index = [];

  for (const segment of segments) {
    const start = Number(segment.start);
    const end = Number(segment.end);
    const textBox = el("div", { class: "tx" });
    const words = [];

    const wordList = Array.isArray(segment.words) ? segment.words : [];
    if (wordList.length) {
      wordList.forEach((word, i) => {
        if (i > 0) textBox.appendChild(document.createTextNode(" "));
        const score = word.score === null || word.score === undefined ? null : Number(word.score);
        const span = el("span", {
          class: score !== null && score < LOW_SCORE ? "w w-low" : "w",
          text: String(word.word === null || word.word === undefined ? "" : word.word),
        });
        if (score !== null && score < LOW_SCORE) {
          span.title = `niska pewność dopasowania (score ${score.toFixed(2).replace(".", ",")})`;
        }
        const ws = Number(word.start);
        if (isFinite(ws)) {
          span.dataset.at = String(ws);
          const we = isFinite(Number(word.end)) ? Number(word.end) : ws + 0.2;
          words.push({ start: ws, end: Math.min(we, ws + MAX_WORD_HL_S), node: span });
        }
        textBox.appendChild(span);
      });
    } else {
      textBox.appendChild(document.createTextNode(String(segment.text || "").trim()));
    }

    const stamp = el("button", {
      class: "ts", type: "button", text: clock(start),
      "aria-label": `Przejdź do ${clock(start)}`,
    });
    const turn = el("div", { class: "turn" }, [stamp, textBox]);
    if (segment.id) turn.dataset.segmentId = String(segment.id);
    turn.addEventListener("click", (event) => {
      const word = event.target.closest ? event.target.closest(".w") : null;
      const at = word && word.dataset.at ? Number(word.dataset.at) : start;
      if (isFinite(at)) onSeek(at);
    });
    body.appendChild(turn);
    if (isFinite(start)) {
      const segEnd = isFinite(end) ? end : start + 1;
      const hlEnd = words.length
        ? Math.min(segEnd, Math.max(...words.map((w) => w.end)))
        : segEnd;
      index.push({ start, end: hlEnd, node: turn, words });
    }
  }

  if (!segments.length) {
    body.appendChild(el("div", { class: "empty", text: "Brak transkrypcji dla tego strumienia." }));
  }

  const backBtn = el("button", { type: "button", class: "hidden", text: "↑ wróć do kursora" });
  const expandBtn = el("button", { type: "button", text: "rozwiń całość ↓" });
  const footer = el("div", { class: "jump" }, [backBtn, expandBtn]);

  const colorVar = colorVarFor(colorIndex);
  const softVar = SOFT_VARS[colorIndex % SOFT_VARS.length];
  const column = el("div", {
    class: "tcol",
    style: `--col: var(${colorVar}); --col-soft: var(${softVar})`,
  }, [
    el("div", { class: "thead" }, [
      el("span", { text: title }),
      subtitle ? el("span", { class: "tid", text: subtitle }) : null,
    ]),
    body,
    footer,
  ]);

  // -- following ---------------------------------------------------------
  const state = { anchor: -1, turn: -1, word: null };
  let userScrolled = false;
  let expanded = false;
  let lastProgrammatic = 0;

  // Expanded, `.tbody` has no max-height: there is no inner scrollbox left to
  // return to, so the way-back button would scroll nothing (N4). Before the
  // first anchor exists (page loaded, nothing played) there is no cursor to
  // return to either (N14).
  const updateBack = () =>
    backBtn.classList.toggle("hidden", expanded || !userScrolled || state.anchor < 0);

  // offsetTop is relative to the nearest POSITIONED ancestor, which .tbody was
  // not — the old math measured against a far ancestor and landed minutes away.
  // Rects are always container-relative.
  const offsetIn = (node) =>
    node.getBoundingClientRect().top - body.getBoundingClientRect().top + body.scrollTop;

  const centerOn = (node) => {
    lastProgrammatic = performance.now();
    body.scrollTop = offsetIn(node) - body.clientHeight / 2 + node.offsetHeight / 2;
  };
  const centerIfOffscreen = (node) => {
    const top = offsetIn(node);
    const bottom = top + node.offsetHeight;
    if (top < body.scrollTop || bottom > body.scrollTop + body.clientHeight) centerOn(node);
  };

  body.addEventListener("scroll", () => {
    // Our own scrollTop writes fire this asynchronously; they are not the user.
    if (performance.now() - lastProgrammatic < 250) return;
    if (userScrolled) return;
    userScrolled = true;
    updateBack();
  });
  backBtn.addEventListener("click", () => {
    userScrolled = false;
    updateBack();
    if (state.anchor >= 0) centerOn(index[state.anchor].node);
  });

  const setExpanded = (on) => {
    expanded = on;
    column.classList.toggle("expanded", on);
    expandBtn.textContent = on ? "zwiń ↑" : "rozwiń całość ↓";
    updateBack();
    if (!on && state.anchor >= 0) centerOn(index[state.anchor].node);
  };
  expandBtn.addEventListener("click", () => setGlobalExpanded(!globalExpanded));

  function tick(t) {
    const anchorIdx = lastAtOrBefore(index, t);
    if (anchorIdx !== state.anchor) {
      state.anchor = anchorIdx;
      if (anchorIdx >= 0 && !expanded) centerIfOffscreen(index[anchorIdx].node);
      userScrolled = false;
      updateBack();
    }

    const turnIdx = anchorIdx >= 0 && t <= index[anchorIdx].end ? anchorIdx : -1;
    if (turnIdx !== state.turn) {
      if (state.turn >= 0) index[state.turn].node.classList.remove("cur");
      if (state.word) { state.word.classList.remove("w-cur"); state.word = null; }
      if (turnIdx >= 0) index[turnIdx].node.classList.add("cur");
      state.turn = turnIdx;
    }
    if (turnIdx < 0) return;
    const words = index[turnIdx].words;
    if (!words.length) return;
    const wordIdx = activeIndex(words, t);
    const node = wordIdx >= 0 ? words[wordIdx].node : null;
    if (node !== state.word) {
      if (state.word) state.word.classList.remove("w-cur");
      if (node) node.classList.add("w-cur");
      state.word = node;
    }
  }

  const entry = { label, node: column, tick, setExpanded };
  liveColumns.add(entry);
  setExpanded(globalExpanded);
  return entry;
}

/**
 * @param {object} spec
 * @param {object} spec.transcripts  {label: {segments: [...]}}
 * @param {string[]} spec.labels     column order
 * @param {object} [spec.speakerIds] {label: "SPEAKER_00"} for the column subtitle
 * @param {Function} spec.onSeek     t => void (seek + play)
 * @param {string} [spec.titlePrefix] default "Mówca"
 */
export function createTranscripts({ transcripts = {}, labels = [], speakerIds = {}, onSeek, titlePrefix = "Mówca" }) {
  const columns = [];
  const grid = el("div", { class: "cols" });
  labels.forEach((label, i) => {
    const segments = ((transcripts[label] || {}).segments) || [];
    const column = buildColumn({
      label,
      title: label === "mixture" ? "Miks — jeden strumień" : `${titlePrefix} ${label}`,
      subtitle: speakerIds[label] || "",
      colorIndex: label === "mixture" ? 2 : i,
      segments,
      onSeek: (t) => onSeek(t),
    });
    columns.push(column);
    grid.appendChild(column.node);
  });

  return {
    node: grid,
    tick(t) { for (const column of columns) column.tick(t); },
    destroy() {
      for (const column of columns) liveColumns.delete(column);
      if (!liveColumns.size) globalExpanded = false;
      clear(grid);
    },
  };
}

/** Adapt the examples' GT tiers ({label: [{start,end,text}]}) to tier shape. */
export function gtToTranscripts(gt) {
  const out = {};
  for (const [label, rows] of Object.entries(gt || {})) {
    out[label] = {
      segments: (rows || []).map((row, i) => ({
        id: `gt-${label}-${String(i).padStart(4, "0")}`,
        start: row.start,
        end: row.end,
        text: row.text,
        words: [],
      })),
    };
  }
  return out;
}
