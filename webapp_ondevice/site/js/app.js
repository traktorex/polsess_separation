/**
 * Wiring: every event the page can produce, and the playback groups.
 *
 * Layout of the app layer, flat on purpose:
 *   state.js     the one mutable state object
 *   ui.js        state -> DOM (no behaviour)
 *   pipeline.js  the run: decode -> OSD -> routing -> separation (no DOM events)
 *   player.js    AudioContext + position-preserving channel switching
 *   models.js    downloads with progress + session lifecycle
 *   timeline.js  peaks + band/ruler rendering
 *   debug.js     dev-only state switcher, loaded ONLY for `#debug`
 *
 * The engine layer (`osd.js`, `routing.js`, `separate.js`, `resample.js`,
 * `wav.js`, `ortEnv.js`) is used as-is and has its own parity suite in
 * `../../test.html`.
 */
import { S, isBusy, resetRun } from './state.js';
import { $, paintTransports, redrawCanvases, render, toast } from './ui.js';
import { onThemeChange, initTheme } from './theme.js';
import { Group, stopAll, toAudioBuffer } from './player.js';
import { SEPARATORS, DEFAULT_SEPARATOR, releaseSeparator } from './models.js';
import { SAMPLE_RATE } from './routing.js';
import { SEPARATOR_SAMPLE_RATE } from './separate.js';
import { encodeWav } from './wav.js';
import {
  cancelRun,
  reseparate,
  runFromUrl,
  runOverride,
  runPipeline,
  showError,
} from './pipeline.js';

// ---------------------------------------------------------------------------
// playback groups
// ---------------------------------------------------------------------------
let clipGroup = null;
let regionGroup = null;
let fullGroup = null;
/** Which region `regionGroup` was built for. */
let regionGroupIndex = null;
/** Which pair `fullGroup` holds: 'single' (assembled) or 'override'. */
let fullGroupKind = null;

/** Listening level: what a group's loudest channel is scaled to. */
const TARGET_PEAK = 0.95;
/** Ceiling on the boost, so a silent clip does not become an amplifier. */
const MAX_BOOST = 32;

/**
 * ONE gain for a whole group.
 *
 * The separators are SI-SDR-trained and therefore scale-free: on this machine
 * a mixture peaking at 0.06 comes back as streams peaking above 1.5, which
 * would clip on playback and in the exported WAV (`wav.js` clips rather than
 * rescale, deliberately). `separate.js` leaves the decision to the caller, so
 * the caller makes it here — one factor across mix / speaker 1 / speaker 2, so
 * the relative levels the separator produced survive intact and the A/B/C
 * switch stays a comparison of separation, not of volume.
 */
function groupGain(...streams) {
  let top = 0;
  for (const stream of streams) {
    if (!stream) continue;
    for (let i = 0; i < stream.length; i++) {
      const v = Math.abs(stream[i]);
      if (v > top) top = v;
    }
  }
  if (!(top > 1e-6)) return 1;
  return Math.min(TARGET_PEAK / top, MAX_BOOST);
}

function dropGroups() {
  stopAll();
  clipGroup = null;
  regionGroup = null;
  fullGroup = null;
  regionGroupIndex = null;
  fullGroupKind = null;
  S.playing = null;
  S.playingFull = null;
}

function onGroupEnd() {
  S.playing = null;
  S.playingFull = null;
  render();
}

function getClipGroup() {
  if (!clipGroup && S.audio16k) {
    clipGroup = new Group({ mix: toAudioBuffer(S.audio16k, SAMPLE_RATE) },
      { onEnd: onGroupEnd, gain: groupGain(S.audio16k) });
  }
  return clipGroup;
}

function getRegionGroup() {
  const region = S.region !== null ? S.regions[S.region] : null;
  if (!region || !region.done || !region.mix) return null; // `#debug` states have no audio
  if (!regionGroup || regionGroupIndex !== S.region) {
    if (regionGroup) regionGroup.stop(true);
    regionGroup = new Group({
      mix: toAudioBuffer(region.mix, SEPARATOR_SAMPLE_RATE),
      s1: toAudioBuffer(region.s1, SEPARATOR_SAMPLE_RATE),
      s2: toAudioBuffer(region.s2, SEPARATOR_SAMPLE_RATE),
    }, {
      loop: S.loop,
      onEnd: onGroupEnd,
      gain: groupGain(region.mix, region.s1, region.s2),
    });
    regionGroupIndex = S.region;
    regionGroup.setChannel(S.src);
  }
  return regionGroup;
}

/** The two full-length streams: assembled (single region) or override output. */
function getFullGroup(key) {
  const single = S.regions.length === 1 ? S.regions[0] : null;
  const wanted = key === 'ov1' || key === 'ov2' ? 'override' : 'single';
  if (!fullGroup || fullGroupKind !== wanted) {
    if (fullGroup) fullGroup.stop(true);
    const samples = wanted === 'override'
      ? { ov1: S.override.s1, ov2: S.override.s2 }
      : single && single.full ? { s1: single.full.s1, s2: single.full.s2 } : null;
    if (!samples || !samples[Object.keys(samples)[0]]) return null;
    fullGroup = new Group(
      Object.fromEntries(Object.entries(samples)
        .map(([name, data]) => [name, toAudioBuffer(data, SEPARATOR_SAMPLE_RATE)])),
      { onEnd: onGroupEnd, gain: groupGain(...Object.values(samples)) },
    );
    fullGroupKind = wanted;
  }
  return fullGroup;
}

let ticking = false;
/** Playhead follow: transports only, never a full re-render. */
function tick() {
  if (clipGroup && clipGroup.playing) S.playhead = clipGroup.time;
  if (regionGroup && regionGroup.playing) S.regHead = regionGroup.time;
  paintTransports();
  if ((clipGroup && clipGroup.playing) || (regionGroup && regionGroup.playing)
      || (fullGroup && fullGroup.playing)) {
    requestAnimationFrame(tick);
  } else {
    ticking = false;
  }
}
function startTicking() {
  if (ticking) return;
  ticking = true;
  requestAnimationFrame(tick);
}

// ---------------------------------------------------------------------------
// input sources
// ---------------------------------------------------------------------------
/** The last thing the user asked to process, for the error card's retry. */
let lastInput = null;

async function startFile(file) {
  if (!file) return;
  lastInput = { name: file.name, kind: 'file', data: file };
  dropGroups();
  await runPipeline(lastInput);
}

async function startExample(clip) {
  lastInput = { name: clip.name, kind: 'example', url: `./examples/${clip.file}` };
  dropGroups();
  await runFromUrl(lastInput.url, clip.name);
}

// --- microphone -------------------------------------------------------------
let recorder = null;
let recStart = 0;
let recTimer = null;

async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      // Honest input: the full pipeline never sees browser-side cleanup either.
      audio: { echoCancellation: false, autoGainControl: false, noiseSuppression: false },
    });
    const chunks = [];
    recorder = new MediaRecorder(stream);
    recorder.ondataavailable = (event) => { if (event.data.size) chunks.push(event.data); };
    recorder.onstop = async () => {
      stream.getTracks().forEach((track) => track.stop());
      clearInterval(recTimer);
      recorder = null;
      $('rec-btn').disabled = false;
      $('rec-stop').disabled = true;
      const blob = new Blob(chunks, { type: chunks[0] ? chunks[0].type : 'audio/webm' });
      lastInput = { name: 'nagranie-z-mikrofonu.webm', kind: 'mic', data: blob };
      dropGroups();
      await runPipeline(lastInput);
    };
    recorder.start();
    recStart = performance.now();
    $('rec-btn').disabled = true;
    $('rec-stop').disabled = false;
    $('rec-time').textContent = '0:00';
    recTimer = setInterval(() => {
      const seconds = Math.floor((performance.now() - recStart) / 1000);
      $('rec-time').textContent = `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`;
    }, 250);
  } catch (err) {
    showError(err);
  }
}

function stopRecording() {
  if (recorder && recorder.state !== 'inactive') recorder.stop();
}

// --- examples ---------------------------------------------------------------
let examples = [];
let exampleIndex = 0;

/**
 * Example clips are described by `site/examples/manifest.json`. Missing file,
 * empty list or malformed JSON all mean the same thing: show the empty state.
 * The manifest ships empty until the clips are chosen and licensed.
 */
async function loadExamples() {
  try {
    const response = await fetch('./examples/manifest.json');
    if (!response.ok) return;
    const data = await response.json();
    examples = Array.isArray(data.clips) ? data.clips : [];
  } catch (err) {
    examples = []; // absent manifest is a supported state, not an error
  }
  renderExamples();
}

function renderExamples() {
  const list = $('exlist');
  list.innerHTML = '';
  const any = examples.length > 0;
  list.classList.toggle('hidden', !any);
  $('ex-empty').classList.toggle('hidden', any);
  $('ex-runrow').classList.toggle('hidden', !any);
  examples.forEach((clip, i) => {
    const card = document.createElement('button');
    card.type = 'button';
    card.className = 'excard' + (i === exampleIndex ? ' selected' : '');
    card.dataset.i = String(i);
    const row = document.createElement('span');
    row.className = 'exrow';
    const name = document.createElement('span');
    name.className = 'exname';
    name.textContent = clip.name;
    row.appendChild(name);
    if (clip.durationS) {
      const chip = document.createElement('span');
      chip.className = 'chip mono';
      chip.textContent = `${Math.floor(clip.durationS / 60)}:${String(Math.round(clip.durationS % 60)).padStart(2, '0')}`;
      row.appendChild(chip);
    }
    const why = document.createElement('span');
    why.className = 'exwhy';
    why.textContent = clip.why || '';
    card.append(row, why);
    if (clip.source) {
      const source = document.createElement('span');
      source.className = 'exsrc';
      source.textContent = clip.source;
      card.appendChild(source);
    }
    list.appendChild(card);
  });
}

// ---------------------------------------------------------------------------
// actions
// ---------------------------------------------------------------------------
function selectRegion(index) {
  if (regionGroup) { regionGroup.stop(true); regionGroup = null; regionGroupIndex = null; }
  S.region = index;
  S.regHead = 0;
  S.src = 'mix';
  S.playing = null;
  render();
  if (window.innerWidth < 960) $('card-insp').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function toggleClip() {
  const group = getClipGroup();
  if (!group) return;
  if (group.playing) { group.pause(); S.playing = null; }
  else { group.play(S.playhead); S.playing = 'clip'; startTicking(); }
  render();
}

function toggleRegion() {
  const group = getRegionGroup();
  if (!group) return;
  if (group.playing) { group.pause(); S.playing = null; }
  else { group.setChannel(S.src); group.play(S.regHead); S.playing = 'region'; startTicking(); }
  render();
}

function setSource(key) {
  S.src = key;
  const group = getRegionGroup();
  if (group) group.setChannel(key);
  render();
}

function downloadWav(key) {
  const region = S.region !== null ? S.regions[S.region] : null;
  if (!region || !region.done) return;
  const samples = region[key];
  if (!samples) return;
  // Same single factor as playback, so the three files stay comparable with
  // each other and with what the page just played.
  const gain = groupGain(region.mix, region.s1, region.s2);
  const scaled = new Float32Array(samples.length);
  for (let i = 0; i < samples.length; i++) scaled[i] = samples[i] * gain;
  const url = URL.createObjectURL(encodeWav(scaled, SEPARATOR_SAMPLE_RATE));
  const link = document.createElement('a');
  link.href = url;
  link.download = `region${S.region + 1}_${key}.wav`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
}

function backToStart() {
  dropGroups();
  resetRun();
  S.input = null;
  S.state = 'S0';
  render();
}

async function changeModel(id) {
  if (!SEPARATORS[id] || id === S.modelId || isBusy()) return;
  S.modelId = id;
  await releaseSeparator();
  dropGroups();
  render();
  // Results from the previous separator are no longer what the page claims to
  // show, so they are dropped and the routed regions are re-run.
  if (['S5', 'S6', 'S7'].includes(S.state) || (S.state === 'S4' && S.regions.length)) {
    await reseparate();
  }
}

function retryFromError() {
  const action = S.err ? S.err.action : 'retry';
  if (action === 'file') { backToStart(); switchTab('file'); $('file-input').click(); return; }
  if (action === 'mic') { backToStart(); switchTab('mic'); startRecording(); return; }
  if (S.audio16k && S.regions.length) { S.err = null; reseparate(); return; }
  if (lastInput && lastInput.url) { runFromUrl(lastInput.url, lastInput.name); return; }
  if (lastInput && lastInput.data) { runPipeline(lastInput); return; }
  backToStart();
}

function switchTab(tab) {
  document.querySelectorAll('.tabs button').forEach((button) => {
    button.classList.toggle('active', button.dataset.tab === tab);
  });
  document.querySelectorAll('.tabpane').forEach((pane) => {
    pane.classList.toggle('hidden', pane.dataset.pane !== tab);
  });
}

// ---------------------------------------------------------------------------
// wiring
// ---------------------------------------------------------------------------
function bind() {
  initTheme($('themebtn'));
  onThemeChange(redrawCanvases);
  window.addEventListener('resize', () => { render(); });

  document.querySelectorAll('.tabs button').forEach((button) => {
    button.addEventListener('click', () => switchTab(button.dataset.tab));
  });

  // --- examples ---
  $('exlist').addEventListener('click', (event) => {
    const card = event.target.closest('.excard');
    if (!card) return;
    exampleIndex = Number(card.dataset.i);
    renderExamples();
  });
  $('run-btn').addEventListener('click', () => {
    const clip = examples[exampleIndex];
    if (clip) startExample(clip);
  });

  // --- file ---
  $('pick-btn').addEventListener('click', () => $('file-input').click());
  $('file-input').addEventListener('change', (event) => {
    startFile(event.target.files[0]);
    event.target.value = '';
  });
  const drop = $('drop');
  ['dragenter', 'dragover'].forEach((type) => drop.addEventListener(type, (event) => {
    event.preventDefault();
    drop.style.borderColor = 'var(--spkA)';
  }));
  ['dragleave', 'drop'].forEach((type) => drop.addEventListener(type, (event) => {
    event.preventDefault();
    drop.style.borderColor = '';
  }));
  drop.addEventListener('drop', (event) => {
    const file = event.dataTransfer && event.dataTransfer.files[0];
    if (file) startFile(file);
  });

  // --- microphone ---
  $('rec-btn').addEventListener('click', startRecording);
  $('rec-stop').addEventListener('click', stopRecording);

  // --- chosen bar ---
  $('ch-reset').addEventListener('click', backToStart);
  const select = $('model-sel');
  for (const spec of Object.values(SEPARATORS)) {
    const option = document.createElement('option');
    option.value = spec.id;
    option.textContent = spec.option;
    select.appendChild(option);
  }
  select.value = DEFAULT_SEPARATOR;
  S.modelId = DEFAULT_SEPARATOR;
  select.addEventListener('change', (event) => changeModel(event.target.value));

  // --- cancel ---
  $('sep-cancel').addEventListener('click', () => { cancelRun(); toast('Przerywanie…'); });
  $('ov-cancel').addEventListener('click', () => { cancelRun(); toast('Przerywanie…'); });

  // --- error card ---
  $('err-retry').addEventListener('click', retryFromError);
  $('err-back').addEventListener('click', backToStart);

  // --- timeline ---
  $('regs').addEventListener('click', (event) => {
    const node = event.target.closest('.reg');
    if (!node) return;
    event.stopPropagation();
    if (node.dataset.ready !== '1') { toast('Region będzie do odsłuchu po separacji'); return; }
    selectRegion(Number(node.dataset.i));
  });
  $('track').addEventListener('click', (event) => {
    if (!S.durationS || event.target.closest('.reg')) return;
    const rect = $('track').getBoundingClientRect();
    if (!rect.width) return;
    const ratio = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
    S.playhead = ratio * S.durationS;
    if (clipGroup && clipGroup.playing) clipGroup.seek(S.playhead);
    render();
  });
  $('tl-play').addEventListener('click', toggleClip);
  $('tl-seek').addEventListener('input', (event) => {
    S.playhead = (Number(event.target.value) / 1000) * S.durationS;
    if (clipGroup && clipGroup.playing) clipGroup.seek(S.playhead);
    render();
  });

  // --- inspector ---
  document.querySelectorAll('.seg [data-src]').forEach((button) => {
    button.addEventListener('click', () => setSource(button.dataset.src));
  });
  $('ins-play').addEventListener('click', toggleRegion);
  $('ins-loop').addEventListener('click', () => {
    S.loop = !S.loop;
    if (regionGroup) regionGroup.setLoop(S.loop);
    render();
  });
  $('ins-seek').addEventListener('input', (event) => {
    const region = S.region !== null ? S.regions[S.region] : null;
    if (!region) return;
    const length = region.pad[1] - region.pad[0];
    S.regHead = (Number(event.target.value) / 1000) * length;
    if (regionGroup && regionGroup.playing) regionGroup.seek(S.regHead);
    render();
  });
  $('insp-close').addEventListener('click', () => {
    if (regionGroup) { regionGroup.stop(true); regionGroup = null; regionGroupIndex = null; }
    S.region = null;
    S.playing = null;
    render();
  });
  $('ins-dl').addEventListener('click', (event) => {
    const link = event.target.closest('a[data-dl]');
    if (!link) return;
    event.preventDefault();
    downloadWav(link.dataset.dl);
  });

  // --- full streams (single-region card + override result) ---
  document.querySelectorAll('[data-full]').forEach((button) => {
    button.addEventListener('click', () => {
      const key = button.dataset.full;
      const group = getFullGroup(key);
      if (!group) return;
      if (group.playing && group.channel === key) { group.pause(); S.playing = null; S.playingFull = null; }
      else if (group.playing) { group.setChannel(key); S.playingFull = key; }
      else { group.setChannel(key); group.play(group.offset); S.playing = 'full'; S.playingFull = key; startTicking(); }
      render();
    });
  });

  // --- override ---
  $('ov-run').addEventListener('click', () => { dropGroups(); runOverride(); });

  // --- keyboard: space = play/pause, 1/2/3 = source ---
  document.addEventListener('keydown', (event) => {
    const tag = (event.target && event.target.tagName) || '';
    if (['INPUT', 'TEXTAREA', 'SELECT'].includes(tag)) return;
    if (event.code === 'Space') {
      if (['BUTTON', 'A', 'SUMMARY'].includes(tag)) return; // let the control act
      event.preventDefault();
      const region = S.region !== null ? S.regions[S.region] : null;
      if (region && region.done) toggleRegion();
      else toggleClip();
      return;
    }
    const region = S.region !== null ? S.regions[S.region] : null;
    if (!region || !region.done) return;
    const map = { Digit1: 'mix', Digit2: 's1', Digit3: 's2' };
    if (map[event.code]) setSource(map[event.code]);
  });
}

// ---------------------------------------------------------------------------
// boot
// ---------------------------------------------------------------------------
bind();
render();
loadExamples();

if (new URLSearchParams((location.hash || '').replace(/^#/, '')).has('debug')) {
  import('./debug.js').then((module) => module.mount({
    render,
    selectRegion,
    setSource,
    dropGroups,
    getRegionGroup,
    startExample: (clip) => startExample(clip),
  }));
}
