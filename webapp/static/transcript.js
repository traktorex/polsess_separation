/* Speaker transcript columns with word-level karaoke.

   One column per stream label (N columns, not two by assumption), each a tier
   of {id, start, end, text, words} — exactly the .eaf / WhisperX shape, so the
   read-only columns can later be swapped for an ELAN-style tier editor without
   touching the timeline or the transport (design §5.3 ELAN future-proofing).
   Segments keep their server-side stable ids in `data-segment-id`.

   The highlighter is driven by the player's shared clock and writes to the DOM
   only when the current turn or word actually changes — the review page's
   lesson: change-only updates never fight the user's own scrolling. */

import { el, clear } from "./dom.js";
import { clock } from "./format.js";
import { colorVarFor } from "./timeline.js";

const LOW_SCORE = 0.45;
const SOFT_VARS = ["--spkA-soft", "--spkB-soft", "--spkC-soft", "--spkD-soft"];

/** Last index with start <= t, provided t is still inside that item. */
function activeIndex(items, t) {
  let lo = 0;
  let hi = items.length - 1;
  let best = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (items[mid].start <= t) { best = mid; lo = mid + 1; } else hi = mid - 1;
  }
  if (best >= 0 && t <= items[best].end) return best;
  return -1;
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
          words.push({ start: ws, end: isFinite(Number(word.end)) ? Number(word.end) : ws + 0.2, node: span });
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
      index.push({ start, end: isFinite(end) ? end : start + 1, node: turn, words });
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

  // Autoscroll that yields to the reader: any scroll we did not cause turns it
  // off and offers the way back.
  let follow = true;
  let lastProgrammatic = 0;
  body.addEventListener("scroll", () => {
    if (performance.now() - lastProgrammatic < 150) return;
    if (!follow) return;
    follow = false;
    backBtn.classList.remove("hidden");
  });
  const scrollTo = (node) => {
    lastProgrammatic = performance.now();
    body.scrollTop = node.offsetTop - body.clientHeight / 2 + node.offsetHeight / 2;
  };
  backBtn.addEventListener("click", () => {
    follow = true;
    backBtn.classList.add("hidden");
    if (state.turn >= 0) scrollTo(index[state.turn].node);
  });
  expandBtn.addEventListener("click", () => {
    const expanded = column.classList.toggle("expanded");
    expandBtn.textContent = expanded ? "zwiń ↑" : "rozwiń całość ↓";
  });

  const state = { turn: -1, word: null };

  function tick(t) {
    const turnIdx = activeIndex(index, t);
    if (turnIdx !== state.turn) {
      if (state.turn >= 0) index[state.turn].node.classList.remove("cur");
      if (state.word) { state.word.classList.remove("w-cur"); state.word = null; }
      if (turnIdx >= 0) {
        const node = index[turnIdx].node;
        node.classList.add("cur");
        if (follow) {
          const top = node.offsetTop;
          const bottom = top + node.offsetHeight;
          if (top < body.scrollTop || bottom > body.scrollTop + body.clientHeight) scrollTo(node);
        }
      }
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

  return { label, node: column, tick };
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
      title: label === "mixture" ? "Mieszanina — jeden strumień" : `${titlePrefix} ${label}`,
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
    destroy() { clear(grid); },
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
