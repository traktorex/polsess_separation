/* The progress view: the pipeline's architecture diagram executing live.

   One linear chain of stage rows built from `stages[]` (never a hardcoded list
   or count), a coarse duration-weighted ETA, warning banners as they appear,
   and the progressive-disclosure timeline that shows the diarization result
   while the GPU is still busy (design §5.2). Rows are updated in place — a
   full rebuild every second would restart the running row's shimmer. */

import { el, clear, card, banner } from "./dom.js";
import { clock, eta as etaText, elapsed as elapsedText, plural, secs, statusLabel } from "./format.js";
import { createTimeline, colorVarFor } from "./timeline.js";

const ICON = { pending: "○", running: "▶", done: "✓", failed: "✗" };

function stageRow(name) {
  const ic = el("div", { class: "ic pend", text: ICON.pending });
  const nm = el("div", { class: "nm", text: name });
  const st = el("div", { class: "st", text: "oczekuje" });
  const live = el("div", { class: "live" });
  const node = el("div", { class: "stage" }, [ic, el("div", { class: "body" }, [nm, st, live])]);
  return { name, node, ic, st, live, last: "" };
}

function timingText(row) {
  const load = row.load_s;
  const run = row.run_s;
  if (load === null && run === null) return "gotowe";
  if (load === null || Number(load) < 0.05) return secs(run);
  return `load ${secs(load)} · run ${secs(run)}`;
}

export function createProgressView() {
  const chain = el("div", { class: "chain" });
  const etaLeft = el("span", { text: "" });
  const etaRight = el("span", { class: "t", text: "" });
  const queueLine = el("div", { class: "lead hidden", style: "margin: 0 0 12px" });
  const chainCard = card("Etapy przetwarzania", [
    queueLine, chain, el("div", { class: "eta" }, [etaLeft, etaRight]),
  ]);

  const fileChips = el("div", { class: "chips" });
  const headerCard = el("div", { class: "card" }, [fileChips]);
  const noteBox = el("div");
  const warningBox = el("div");
  const errorBox = el("div");
  const partialBox = el("div");
  const node = el("div", {}, [headerCard, errorBox, chainCard, noteBox, warningBox, partialBox]);

  const rows = new Map();
  let shownWarnings = [];
  let lastHeader = "";
  let partialSignature = "";
  let partialTimeline = null;

  function syncChain(stages, status) {
    // Keep the DOM order equal to the payload order (pipeline order).
    stages.forEach((row, index) => {
      let entry = rows.get(row.stage);
      if (!entry) {
        entry = stageRow(row.stage);
        rows.set(row.stage, entry);
      }
      if (chain.children[index] !== entry.node) {
        chain.insertBefore(entry.node, chain.children[index] || null);
      }
      const isLastActive =
        status === "failed" && row.state === "running";
      const state = isLastActive ? "failed" : row.state;
      const progress = row.progress;
      const key = [
        state,
        row.load_s, row.run_s,
        progress ? `${progress.done}/${progress.total}` : "",
      ].join("|");
      if (key === entry.last) return;
      entry.last = key;

      entry.node.className = `stage${state === "running" ? " running shimmer" : ""}${state === "failed" ? " failed" : ""}`;
      entry.ic.textContent = ICON[state] || ICON.pending;
      entry.ic.className = `ic ${state === "done" ? "ok" : state === "failed" ? "bad" : state === "running" ? "" : "pend"}`;
      if (state === "done") entry.st.textContent = timingText(row);
      else if (state === "running") entry.st.textContent = "w toku…";
      else if (state === "failed") entry.st.textContent = "przerwane błędem";
      else entry.st.textContent = "oczekuje";
      entry.live.textContent =
        state === "running" && progress && progress.total
          ? `▶ region ${progress.done}/${progress.total}`
          : "";
    });
  }

  function syncWarnings(warnings) {
    const list = warnings || [];
    if (list.length === shownWarnings.length && list.every((w, i) => w === shownWarnings[i])) return;
    shownWarnings = list.slice();
    clear(warningBox);
    for (const text of shownWarnings) warningBox.appendChild(banner(`⚠ ${text}`));
  }

  function syncPartial(partial, duration) {
    if (!partial) return;
    const diarization = partial.diarization;
    const routing = partial.routing;
    const turns = (diarization && diarization.turns) || [];
    const overlaps =
      (routing && routing.overlap_regions) || (diarization && diarization.overlaps) || [];
    if (!turns.length && !overlaps.length) return;
    const signature = `${turns.length}:${overlaps.length}:${duration}`;
    if (signature === partialSignature) return;
    partialSignature = signature;

    const bySpeaker = new Map();
    for (const turn of turns) {
      const speaker = String(turn.speaker);
      if (!bySpeaker.has(speaker)) bySpeaker.set(speaker, []);
      bySpeaker.get(speaker).push(turn);
    }
    const lanes = Array.from(bySpeaker.entries()).map(([speaker, list], i) => ({
      label: speaker,
      sub: "",
      colorVar: colorVarFor(i),
      turns: list,
    }));
    if (partialTimeline) partialTimeline.destroy();
    partialTimeline = createTimeline({ duration, lanes, waves: [], overlaps });
    clear(partialBox);
    partialBox.appendChild(card("Oś czasu — diarization", [
      el("p", { class: "lead", style: "margin: 0 0 10px", text: "Wynik częściowy — etykiety mówców są wstępne do zakończenia etapów relabel i assembly." }),
      partialTimeline.node,
    ]));
  }

  function syncError(error, logUrl) {
    clear(errorBox);
    if (!error) return;
    errorBox.appendChild(el("div", { class: "card errcard" }, [
      el("h2", { text: "Przetwarzanie nie powiodło się" }),
      el("div", { class: "etype", text: String(error.type || "Error") }),
      el("div", { class: "emsg", text: String(error.message || "") }),
      el("p", { class: "lead", text: "Zadanie nie zostało wykonane — nie ma wyniku częściowego do pokazania. Log przebiegu zawiera pełny ślad." }),
      logUrl
        ? el("div", { class: "dl", style: "margin-top:10px" }, [
            el("a", { href: logUrl.replace(/\/log$/, "/files/debug.log"), download: "debug.log", text: "debug.log" }),
          ])
        : null,
    ]));
  }

  return {
    node,
    /** @param {object} state JobState payload; @param {string} logUrl */
    update(state, logUrl) {
      const status = state.status;
      const header = `${state.filename || ""}|${state.audio_duration_s || ""}|${status}`;
      if (header !== lastHeader) {
        lastHeader = header;
        clear(fileChips);
        if (state.filename) fileChips.appendChild(el("span", { class: "filename", text: state.filename }));
        if (state.audio_duration_s) {
          fileChips.appendChild(el("span", { class: "chip" }, [el("b", { text: clock(state.audio_duration_s) })]));
        }
        fileChips.appendChild(el("span", { class: `pill ${status}`, text: statusLabel(status) }));
      }
      syncChain(state.stages || [], status);
      syncWarnings(state.warnings);
      syncError(state.error, logUrl);
      syncPartial(state.partial, state.audio_duration_s);

      if (status === "queued") {
        const ahead = Number(state.queue_position) || 0;
        queueLine.classList.remove("hidden");
        queueLine.textContent = ahead > 0
          ? `W kolejce: ${ahead} ${plural(ahead, "zadanie", "zadania", "zadań")} przed Tobą.`
          : "W kolejce — start za chwilę.";
      } else {
        queueLine.classList.add("hidden");
      }

      etaLeft.textContent = status === "failed" ? "" : etaText(state.eta_s);
      etaRight.textContent = elapsedText(state.elapsed_s);
    },
    /** Connection notice ("reconnecting"), or "" to clear it. */
    setNote(text) {
      clear(noteBox);
      if (text) noteBox.appendChild(banner(text, "info"));
    },
    destroy() {
      if (partialTimeline) partialTimeline.destroy();
    },
  };
}
