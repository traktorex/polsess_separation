/* The diagnostics drawer: closed by default, dashed border, panels rendered
   lazily on first open (design §5.4 — this is a showcase, not a diagnostic
   tool). A panel whose data the run did not produce says so; it never shows an
   empty table pretending to be a result. */

import { el, clear } from "./dom.js";
import { clock, num, secs } from "./format.js";
import { getJSON, fileExists } from "./api.js";

const NO_DATA = "brak danych dla tego przebiegu";

function nodata(extra) {
  return el("div", { class: "nodata", text: extra ? `${NO_DATA} — ${extra}` : NO_DATA });
}

function table(headers, rows) {
  const head = el("tr", {}, headers.map((h) => el("th", { text: h })));
  const body = rows.map((row) => {
    const tr = el("tr", { class: row.klass || "" });
    for (const cell of row.cells) {
      tr.appendChild(el("td", { class: cell.tag ? "tag" : "", text: cell.text }));
    }
    return tr;
  });
  return el("table", { class: "rt" }, [head, ...body]);
}

/** A lazily-rendered <details> panel. `build()` runs once, on first open. */
function panel(title, hint, build) {
  const body = el("div", { class: "panel-body" });
  const node = el("details", { class: "panel" }, [
    el("summary", {}, [
      document.createTextNode(title),
      hint ? el("span", { class: "hint", text: hint }) : null,
    ]),
    body,
  ]);
  let built = false;
  node.addEventListener("toggle", () => {
    if (!node.open || built) return;
    built = true;
    try {
      const content = build();
      if (content) body.appendChild(content);
    } catch (err) {
      body.appendChild(el("div", { class: "nodata", text: `Nie udało się zbudować panelu: ${err}` }));
    }
  });
  return node;
}

// -- routing diff -----------------------------------------------------------

function routingPanel(result) {
  const accepted = (result.routing && result.routing.overlap_regions) || [];
  const raw = result.diarization ? result.diarization.overlaps : null;
  const rows = [];
  let dropped = 0;
  let merged = 0;

  const covers = (region, item) =>
    Number(item.start) >= Number(region.start) - 0.05 &&
    Number(item.end) <= Number(region.end) + 0.05;

  const counts = accepted.map(() => 0);
  const orphans = [];
  for (const item of raw || []) {
    const idx = accepted.findIndex((region) => covers(region, item));
    if (idx >= 0) counts[idx] += 1;
    else orphans.push(item);
  }
  dropped = orphans.length;
  merged = counts.filter((c) => c >= 2).length;

  const entries = accepted.map((region, i) => ({
    start: Number(region.start),
    end: Number(region.end),
    index: i + 1,
    tag: counts[i] >= 2 ? `scalone (${counts[i]} nakładania)` : "",
    klass: "",
  }));
  for (const item of orphans) {
    entries.push({
      start: Number(item.start), end: Number(item.end), index: null,
      tag: "odrzucony przez routing", klass: "dropped",
    });
  }
  entries.sort((a, b) => a.start - b.start);

  for (const entry of entries) {
    rows.push({
      klass: entry.klass,
      cells: [
        { text: entry.index === null ? "–" : String(entry.index) },
        { text: clock(entry.start) },
        { text: clock(entry.end) },
        { text: secs(entry.end - entry.start) },
        { text: entry.tag, tag: true },
      ],
    });
  }

  if (!rows.length) return nodata("routing nie zwrócił żadnych regionów");
  const parts = [table(["#", "start", "koniec", "czas", ""], rows)];
  if (raw === null || raw === undefined) {
    parts.push(el("div", {
      class: "nodata", style: "margin-top:8px",
      text: "Surowa lista nakładań diaryzacji nie jest dostępna w tym przebiegu — tabela pokazuje wyłącznie regiony przyjęte przez routing.",
    }));
  }
  return el("div", {}, parts);
}

function summariseRouting(result) {
  const accepted = ((result.routing && result.routing.overlap_regions) || []).length;
  const raw = result.diarization ? result.diarization.overlaps : null;
  if (!raw) return `${accepted} przyjętych`;
  return `${accepted} przyjętych · ${raw.length} surowych nakładań`;
}

// -- generic tables ---------------------------------------------------------

function cellText(value) {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : num(value, 3);
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function assemblyPanel(result) {
  const rows = result.assembly_diag;
  if (!Array.isArray(rows) || !rows.length) return nodata("hak diagnostyki przypisania nie zapisał danych");
  const keys = [];
  for (const row of rows) {
    for (const key of Object.keys(row || {})) if (!keys.includes(key)) keys.push(key);
  }
  return table(keys, rows.map((row) => ({
    cells: keys.map((key) => ({ text: cellText((row || {})[key]) })),
  })));
}

function diarizationPanel(result) {
  const diag = result.diarization_diag;
  if (!diag || typeof diag !== "object") return nodata("brak sekcji diarization_diag w metadata.json");
  const rows = Object.entries(diag).map(([key, value]) => ({
    cells: [{ text: key }, { text: cellText(value) }],
  }));
  return table(["klucz", "wartość"], rows);
}

function timingsPanel(result, stagesFallback) {
  let rows = (result.stage_timings || []).filter((row) => row && row.stage);
  let note = "";
  if (!rows.length && Array.isArray(stagesFallback)) {
    rows = stagesFallback
      .filter((row) => row && (row.load_s !== null || row.run_s !== null))
      .map((row) => ({ stage: row.stage, load_s: row.load_s, run_s: row.run_s }));
    if (rows.length) note = "czasy z przebiegu w tej sesji (metadata.json ich nie zawiera)";
  }
  if (!rows.length) return nodata("przebieg nie zapisał czasów etapów");
  let total = 0;
  const body = rows.map((row) => {
    const load = Number(row.load_s) || 0;
    const run = Number(row.run_s) || 0;
    total += load + run;
    return {
      cells: [
        { text: row.stage },
        { text: secs(row.load_s, 2) },
        { text: secs(row.run_s, 2) },
        { text: secs(load + run, 2) },
      ],
    };
  });
  body.push({ cells: [{ text: "razem" }, { text: "" }, { text: "" }, { text: secs(total, 2) }] });
  const parts = [table(["stage", "load", "run", "razem"], body)];
  if (note) parts.push(el("div", { class: "nodata", style: "margin-top:8px", text: note }));
  return el("div", {}, parts);
}

function logPanel(logUrl) {
  const box = el("pre", { class: "logbox", text: "wczytywanie…" });
  const refresh = el("button", { class: "btn small", type: "button", text: "odśwież" });
  const load = () => {
    box.textContent = "wczytywanie…";
    getJSON(`${logUrl}?offset=0`)
      .then((payload) => {
        const lines = (payload && payload.lines) || [];
        box.textContent = lines.length ? lines.join("\n") : "(log jest pusty)";
        box.scrollTop = box.scrollHeight;
      })
      .catch((err) => { box.textContent = `Nie udało się pobrać logu: ${err.message}`; });
  };
  refresh.addEventListener("click", load);
  load();
  return el("div", {}, [el("div", { style: "margin-bottom:8px" }, [refresh]), box]);
}

function enhancementPanel(result) {
  const mixture = (result.files || {}).mixture;
  if (!mixture) return nodata("brak pliku mieszaniny");
  const enhanced = mixture.replace(/mixture\.wav$/, "enhanced_full.wav");
  const holder = el("div", { class: "nodata", text: "sprawdzanie dostępności…" });
  fileExists(enhanced).then((exists) => {
    clear(holder);
    holder.className = "";
    if (!exists) {
      holder.appendChild(nodata("przebieg nie zachował enhanced_full.wav"));
      return;
    }
    holder.appendChild(el("div", { class: "abpair" }, [
      el("div", { class: "ab" }, [
        el("h4", { text: "przed — mieszanina" }),
        el("audio", { controls: "", preload: "none", src: mixture }),
      ]),
      el("div", { class: "ab" }, [
        el("h4", { text: "po — enhancement (enhanced_full.wav)" }),
        el("audio", { controls: "", preload: "none", src: enhanced }),
      ]),
    ]));
  });
  return holder;
}

/**
 * @param {object} spec
 * @param {object} spec.result           the result payload
 * @param {string} [spec.logUrl]         /api/jobs/<id>/log (jobs only)
 * @param {Array}  [spec.stagesFallback] job.stages, when metadata has no timings
 */
export function createDiagnostics({ result, logUrl = null, stagesFallback = null }) {
  const inner = el("div", { class: "diag-inner" });
  const drawer = el("details", { class: "diag" }, [
    el("summary", {}, [
      document.createTextNode("Panele diagnostyczne"),
      el("span", { style: "font-size:12px", text: " (domyślnie ukryte — narzędzie pokazowe, nie diagnostyczne)" }),
    ]),
    inner,
  ]);

  let built = false;
  drawer.addEventListener("toggle", () => {
    if (!drawer.open || built) return;
    built = true;
    inner.appendChild(panel("Routing — regiony nakładania", summariseRouting(result),
      () => routingPanel(result)));
    inner.appendChild(panel("Przypisanie mówców (ECAPA)",
      Array.isArray(result.assembly_diag) ? `${result.assembly_diag.length} regionów` : "",
      () => assemblyPanel(result)));
    inner.appendChild(panel("Diaryzacja — census",
      (result.diarization_diag && result.diarization_diag.backend) || "",
      () => diarizationPanel(result)));
    inner.appendChild(panel("Czasy etapów", "load / run", () => timingsPanel(result, stagesFallback)));
    inner.appendChild(panel("Enhancement — przed / po", "odsłuch", () => enhancementPanel(result)));
    if (logUrl) inner.appendChild(panel("Log przebiegu", "debug.log", () => logPanel(logUrl)));
  });

  return drawer;
}
