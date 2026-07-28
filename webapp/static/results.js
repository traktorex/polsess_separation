/* The results view — one renderer, two data sources.

   It takes a `result`-shaped object (API.md `JobState.result`, which the
   examples gallery serves verbatim as `job_like`) plus already-resolved file
   URLs, and builds: header · timeline stack · transport · transcripts ·
   downloads · diagnostics drawer. The job page and the examples page both call
   it; the only difference is the extra chips and the GT section the gallery
   passes in. */

import { el, clear, card, chip, banner } from "./dom.js";
import { clock, num, plural, secs } from "./format.js";
import { createTimeline, timelineLegend, colorVarFor } from "./timeline.js";
import { createPlayer } from "./player.js";
import { createTranscripts, gtToTranscripts } from "./transcript.js";
import { createDiagnostics } from "./diagnostics.js";

/** Stream labels in speaker order, with sensible fallbacks. */
function labelsOf(result) {
  const map = result.spk_to_label || {};
  const speakers = result.speakers || [];
  const labels = [];
  for (const speaker of speakers) {
    const label = map[speaker] || speaker;
    if (!labels.includes(label)) labels.push(label);
  }
  if (labels.length) return labels;
  for (const key of Object.keys(result.transcripts || {})) {
    if (key !== "mixture") labels.push(key);
  }
  if (labels.length) return labels;
  for (const key of Object.keys(result.files || {})) {
    if (key.startsWith("stream_")) labels.push(key.slice("stream_".length));
  }
  return labels;
}

function speakerIdsOf(result, labels) {
  const out = {};
  for (const [speaker, label] of Object.entries(result.spk_to_label || {})) {
    if (labels.includes(label)) out[label] = speaker;
  }
  return out;
}

function headerCard(result, { title, chips = [], banners = [], actions = [] }) {
  const duration = Number(result.total_duration_s);
  const nSpeakers = (result.speakers || []).length || labelsOf(result).length;
  const nRegions = Number(result.n_overlap_regions) || 0;
  const overlap = Number(result.overlap_total_s) || 0;
  const share = duration > 0 ? Math.round((overlap / duration) * 100) : null;

  const row = el("div", { class: "chips" }, [
    title ? el("span", { class: "filename", text: title }) : null,
    chip(clock(duration), ""),
    chip(String(nSpeakers), plural(nSpeakers, "mówca", "mówców", "mówców")),
    chip(String(nRegions), plural(nRegions, "region nakładania", "regiony nakładania", "regionów nakładania")),
    chip(secs(overlap), share === null ? "nakładania" : `nakładania (${share}%)`),
    ...chips,
  ]);

  const kids = [row];
  if (actions.length) {
    kids.push(el("div", { class: "exhead" }, actions));
  }
  if (result.weak_anchor) {
    kids.push(banner("⚠ Słaba kotwica mówcy (ECAPA) — przypisanie mówców w regionach nakładania może być mniej pewne."));
  }
  for (const text of banners) kids.push(banner(text));
  kids.push(el("div", { class: "prov", text: result.provenance || "(brak zapisanej konfiguracji)" }));
  return el("div", { class: "card" }, kids);
}

/**
 * Render the full results view into `mount` (which is cleared first).
 *
 * @param {HTMLElement} mount
 * @param {object} spec
 * @param {object} spec.result          API.md result payload
 * @param {string} [spec.title]         filename / example id for the header
 * @param {Array}  [spec.chips]         extra chip elements (scores, split, …)
 * @param {Array}  [spec.banners]       extra warning strings
 * @param {Array}  [spec.actions]       extra buttons for the header card
 * @param {string} [spec.logUrl]        job log endpoint (jobs only)
 * @param {Array}  [spec.stagesFallback] job.stages for the timings panel
 * @param {object} [spec.gt]            examples-only GT tiers {label: [...]}
 * @returns {{destroy: Function}}
 */
export function renderResults(mount, spec) {
  const result = spec.result || {};
  const labels = labelsOf(result);
  const speakerIds = speakerIdsOf(result, labels);
  const duration = Number(result.total_duration_s) || 0;
  const files = result.files || {};
  const peaks = result.peaks || {};
  const turns = (result.diarization && result.diarization.turns) || [];
  const overlaps = (result.routing && result.routing.overlap_regions) || [];

  clear(mount);
  mount.appendChild(headerCard(result, spec));

  // -- timeline ---------------------------------------------------------
  const bySpeaker = new Map();
  for (const turn of turns) {
    const speaker = String(turn.speaker);
    if (!bySpeaker.has(speaker)) bySpeaker.set(speaker, []);
    bySpeaker.get(speaker).push(turn);
  }
  const speakers = (result.speakers || []).length
    ? result.speakers
    : Array.from(bySpeaker.keys());

  const lanes = speakers.map((speaker, i) => {
    const label = (result.spk_to_label || {})[speaker] || speaker;
    return {
      label: `Mówca ${label}`,
      sub: `· ${speaker}`,
      colorVar: colorVarFor(labels.indexOf(label) >= 0 ? labels.indexOf(label) : i),
      turns: bySpeaker.get(speaker) || [],
    };
  });

  const waves = [];
  if (peaks.mixture) {
    waves.push({ label: "przebieg mieszaniny", colorVar: "--muted", peaks: peaks.mixture });
  }
  labels.forEach((label, i) => {
    if (peaks[label]) waves.push({ label: `przebieg ${label}`, colorVar: colorVarFor(i), peaks: peaks[label] });
  });

  const timeline = createTimeline({
    duration,
    lanes,
    waves,
    overlaps,
    onSeek: (t) => player.seek(t),
  });
  mount.appendChild(card("Oś czasu", [timeline.node, timelineLegend(labels)]));

  // -- transport --------------------------------------------------------
  const player = createPlayer({
    files,
    labels,
    duration,
    onModeChange: (key) => showMixtureColumn(key === "mixture"),
  });
  mount.appendChild(card("Odtwarzanie", [player.node]));

  // -- transcripts ------------------------------------------------------
  const seekAndPlay = (t) => player.seekAndPlay(t);
  const speakerCols = createTranscripts({
    transcripts: result.transcripts || {},
    labels: labels.filter((label) => (result.transcripts || {})[label]),
    speakerIds,
    onSeek: seekAndPlay,
  });
  const hasMixture = Boolean((result.transcripts || {}).mixture);
  const mixtureCols = hasMixture
    ? createTranscripts({
        transcripts: result.transcripts,
        labels: ["mixture"],
        onSeek: seekAndPlay,
      })
    : null;
  if (mixtureCols) mixtureCols.node.classList.add("hidden");

  const transcriptNote = el("span", { class: "h2sub muted", text: "" });
  const transcriptCard = card(
    "Transkrypcja",
    [speakerCols.node, mixtureCols ? mixtureCols.node : null].filter(Boolean)
  );
  transcriptCard.querySelector("h2").appendChild(transcriptNote);
  mount.appendChild(transcriptCard);

  function showMixtureColumn(show) {
    if (!mixtureCols) return;
    speakerCols.node.classList.toggle("hidden", show);
    mixtureCols.node.classList.toggle("hidden", !show);
    transcriptNote.textContent = show
      ? "— transkrypcja surowej mieszaniny (bez separacji)"
      : "";
  }
  showMixtureColumn(false);

  // -- ground truth (examples only) -------------------------------------
  let gtCols = null;
  if (spec.gt && Object.keys(spec.gt).length) {
    const gtTranscripts = gtToTranscripts(spec.gt);
    gtCols = createTranscripts({
      transcripts: gtTranscripts,
      labels: Object.keys(gtTranscripts),
      onSeek: seekAndPlay,
      titlePrefix: "GT · Mówca",
    });
    mount.appendChild(card("Referencyjna transkrypcja (GT)", [
      el("p", { class: "lead", style: "margin-top:0", text: "Ręcznie poprawiona referencja tego fragmentu — podstawa metryk cpWER / cpCER. Dostępna wyłącznie dla przykładów." }),
      gtCols.node,
    ]));
  }

  // -- downloads --------------------------------------------------------
  const links = [];
  const push = (url, name, klass = "") => {
    if (!url) return;
    links.push(el("a", { href: url, download: name, class: klass, text: name }));
  };
  for (const label of labels) push(files[`stream_${label}`], `stream_${label}.wav`);
  for (const label of labels) push(files[`transcript_${label}_txt`], `transcript_${label}.txt`);
  push(files.eaf, "annotation.eaf", "eaf");
  push(files.metadata, "metadata.json");
  push(files.mixture, "mixture.wav");
  if (links.length) mount.appendChild(card("Pobierz", [el("div", { class: "dl" }, links)]));

  // -- diagnostics ------------------------------------------------------
  mount.appendChild(createDiagnostics({
    result,
    logUrl: spec.logUrl || null,
    stagesFallback: spec.stagesFallback || null,
  }));

  // -- the shared clock -------------------------------------------------
  player.onTick((t) => {
    timeline.setPlayhead(t);
    speakerCols.tick(t);
    if (mixtureCols) mixtureCols.tick(t);
    if (gtCols) gtCols.tick(t);
  });

  return {
    destroy() {
      player.destroy();
      timeline.destroy();
      speakerCols.destroy();
      if (mixtureCols) mixtureCols.destroy();
      if (gtCols) gtCols.destroy();
    },
  };
}

/** Small chips for the examples gallery (scores are gallery-only). */
export function scoreChips(row) {
  const chips = [];
  if (row.split) {
    chips.push(el("span", { class: `badge ${row.split}`, text: String(row.split).toUpperCase() }));
  }
  if (row.cpwer !== null && row.cpwer !== undefined) chips.push(chip(num(row.cpwer, 1), "cpWER", "accent"));
  if (row.cpcer !== null && row.cpcer !== undefined) chips.push(chip(num(row.cpcer, 1), "cpCER", "accent"));
  return chips;
}
