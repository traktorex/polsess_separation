/**
 * Rendering. One `render()` paints the whole page from `state.js`; nothing
 * here reads audio buffers and nothing here decides anything — the pipeline
 * writes state, this reflects it. Structure and copy are the accepted mockup's
 * (`design/road2_mockup.html`), so the two can be screenshot-compared.
 *
 * Events are NOT bound here: `app.js` owns behaviour and uses delegation, which
 * is why regions and download links carry `data-*` instead of listeners.
 */
import { S, isBusy } from './state.js';
import { clock, clockShort, mb, num, plural } from './format.js';
import { OSD, RUNTIME, SEPARATORS } from './models.js';
import {
  cssVar,
  drawWave,
  renderBands,
  renderMarkers,
  renderRegions,
  renderRuler,
} from './timeline.js';

export const $ = (id) => document.getElementById(id);

const has = (...states) => states.includes(S.state);
const pct = (v, total) => Math.max(0, Math.min(100, (v / (total || 1)) * 100));

/** Overlap-labelled runs — what step 1 reports having found. */
const overlapRuns = () => S.segs.filter((seg) => seg[2] === 'ovl').length;
/** Separator calls the routed regions cost in total. */
const totalCalls = () => S.regions.reduce((sum, r) => sum + r.chunks, 0);

// ---------------------------------------------------------------------------
// toast
// ---------------------------------------------------------------------------
let toastTimer = null;
export function toast(text) {
  const node = $('toast');
  node.textContent = text;
  node.classList.add('on');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => node.classList.remove('on'), 1700);
}

// ---------------------------------------------------------------------------
// canvases (repainted on resize and on theme change, hence the separate entry)
// ---------------------------------------------------------------------------
export function redrawCanvases() {
  requestAnimationFrame(() => {
    if (!$('split').classList.contains('hidden')) {
      const show = S.peaks && has('S3', 'S4', 'S5', 'S6', 'S6b', 'S7');
      drawWave($('wave'), show ? S.peaks : null, cssVar('--wave'));
    }
    const region = S.region !== null ? S.regions[S.region] : null;
    if (region && region.peaks && !$('insp-body').classList.contains('hidden')) {
      const color = S.src === 's1' ? '--spkA' : S.src === 's2' ? '--spkB' : '--wave';
      drawWave($('rwave'), region.peaks[S.src], cssVar(color));
    }
  });
}

// ---------------------------------------------------------------------------
// input card
// ---------------------------------------------------------------------------
const IN_PILL = {
  S0: 'wybierz źródło', S1: 'pobieranie modeli', S2: 'przygotowanie', S3: 'analiza',
  S4: 'po routingu', S5: 'separacja', S6: 'gotowe', S6b: 'gotowe', S7: 'gotowe', E1: 'błąd',
};

function renderInput() {
  const idle = S.state === 'S0';
  $('in-tabs').classList.toggle('hidden', !idle);
  $('in-chosen').classList.toggle('hidden', idle);
  $('ch-name').textContent = S.input ? S.input.name : '';
  $('ch-dur').textContent = S.durationS ? clock(S.durationS) : '—';

  const pill = $('in-pill');
  pill.textContent = IN_PILL[S.state];
  pill.className = 'pill' + (has('S1', 'S2', 'S3', 'S5') ? ' running'
    : has('S4', 'S6', 'S6b', 'S7') ? ' done'
      : S.state === 'E1' ? ' failed' : '');

  const select = $('model-sel');
  select.value = S.modelId;
  select.disabled = isBusy();
  const warn = SEPARATORS[S.modelId].warn;
  $('model-note').textContent = warn;
  $('model-note').classList.toggle('hidden', !warn);

  const total = RUNTIME.bytes + OSD.bytes + SEPARATORS[S.modelId].bytes;
  $('dl-note').textContent =
    `Modele (${mb(total)}) pobierają się raz i zostają w cache przeglądarki.`;
}

// ---------------------------------------------------------------------------
// pipeline strip + download panel
// ---------------------------------------------------------------------------
function setStep(index, stepState, text) {
  const node = document.querySelector(`.pstep[data-step="${index}"]`);
  node.className = 'pstep' + (stepState ? ` ${stepState}` : '');
  const label = node.querySelector('.pstate');
  label.innerHTML = '';
  if (stepState === 'running') {
    const spinner = document.createElement('span');
    spinner.className = 'spin';
    label.appendChild(spinner);
  }
  label.appendChild(document.createTextNode(text));
}

function renderPipe() {
  const nOvl = overlapRuns();
  const nReg = S.regions.length;
  const calls = totalCalls();
  const found = `${nOvl} ${plural(nOvl, 'nakładanie', 'nakładania', 'nakładań')} w klatkach`;
  const routed = `${nReg} ${plural(nReg, 'region', 'regiony', 'regionów')} do separacji`;

  if (S.state === 'S0') {
    setStep(1, '', 'oczekuje'); setStep(2, '', 'oczekuje'); setStep(3, '', 'oczekuje');
  } else if (S.state === 'S1') {
    setStep(1, '', 'czeka na model'); setStep(2, '', 'oczekuje'); setStep(3, '', 'czeka na model');
  } else if (S.state === 'S2') {
    setStep(1, '', 'modele gotowe'); setStep(2, '', 'oczekuje'); setStep(3, '', 'modele gotowe');
  } else if (S.state === 'S3') {
    setStep(1, 'running', 'klatki 16,875 ms · powerset argmax');
    setStep(2, '', 'oczekuje'); setStep(3, '', 'oczekuje');
  } else if (S.state === 'S4') {
    setStep(1, 'done', found);
    setStep(2, 'done', routed);
    setStep(3, '', nReg > 0 ? `w kolejce: ${routed}` : 'nic do zrobienia');
  } else if (S.state === 'S5') {
    setStep(1, 'done', found);
    setStep(2, 'done', routed);
    setStep(3, 'running', S.dl.length
      ? 'pobieranie modelu separatora…'
      : `region ${Math.min(S.sepDone + 1, nReg)} z ${nReg}`);
  } else if (S.state === 'S6' || S.state === 'S7') {
    setStep(1, 'done', found);
    setStep(2, 'done', routed);
    setStep(3, 'done', `${calls} ${plural(calls, 'wywołanie', 'wywołania', 'wywołań')} separatora`);
  } else if (S.state === 'S6b') {
    setStep(1, 'done', '0 nakładań w klatkach');
    setStep(2, 'done', '0 regionów — separacja niepotrzebna');
    setStep(3, '', 'pominięty');
  } else if (S.state === 'E1') {
    const failed = S.err && S.err.failStep;
    setStep(1, failed === 1 ? 'failed' : '', failed === 1 ? 'przerwane błędem' : 'oczekuje');
    setStep(2, '', 'oczekuje');
    setStep(3, failed === 3 ? 'failed' : '', failed === 3 ? 'przerwane błędem' : 'oczekuje');
  }

  renderDownloads();
  $('sep-cancelrow').classList.toggle('hidden', !(S.state === 'S5' && !S.dl.length));
}

function renderDownloads() {
  const panel = $('dlpanel');
  panel.classList.toggle('hidden', S.dl.length === 0);
  if (!S.dl.length) return;
  $('dltitle').textContent = S.dlTitle;
  const host = $('dlrows');
  host.innerHTML = '';
  for (const item of S.dl) {
    const row = document.createElement('div');
    row.className = 'dlrow';
    const head = document.createElement('div');
    head.className = 'dlhead';
    const name = document.createElement('span');
    name.className = 'dlname';
    name.textContent = item.label;
    const size = document.createElement('span');
    size.className = 'dlmb';
    size.textContent = item.done
      ? mb(item.loaded || item.total)
      : `${num((item.loaded || 0) / 1048576, 1)} / ${mb(item.total || item.bytes || 0)}`;
    head.append(name, size);
    const bar = document.createElement('div');
    bar.className = 'prog' + (item.done ? ' ok' : '');
    const fill = document.createElement('span');
    const known = item.total || item.bytes || 0;
    fill.style.width = `${item.done ? 100 : known ? pct(item.loaded || 0, known) : 5}%`;
    bar.appendChild(fill);
    row.append(head, bar);
    host.appendChild(row);
  }
}

// ---------------------------------------------------------------------------
// timeline + stats
// ---------------------------------------------------------------------------
const TL_PILL = {
  S2: 'dekodowanie', S3: 'OSD w toku', S4: 'po routingu', S5: 'separacja w toku',
  S6: 'gotowe', S6b: 'gotowe', S7: 'po nadpisaniu',
};

const TL_HINT = {
  S2: 'Przygotowanie nagrania do 16 kHz mono.',
  S3: 'Detektor nakładania przechodzi przez nagranie — pasma pojawią się po zakończeniu.',
  S4: 'Regiony wyznaczone. Mocniejsza czerwień to wykryte nakładanie, ramka wokół niej — dopełnienie do 4-sekundowego okna separatora. Separator wchodzi tylko w te ramki; reszta idzie dalej bez zmian.',
  S5: 'Separator pracuje region po regionie — pasek pod regionem pokazuje postęp.',
  S6: 'Kliknij czerwony region, żeby odsłuchać mix i rozdzielone ścieżki.',
  S6b: 'Nie wykryto nakładania — routing: separacja niepotrzebna. Cały klip to mowa pojedyncza lub cisza.',
  S7: 'Po nadpisaniu: separator przeszedł także przez fragmenty solo. Żółte znaczniki wskazują, gdzie słychać strumień fantomowy.',
};

function renderTimeline() {
  const dur = S.durationS;
  const bandsVisible = has('S4', 'S5', 'S6', 'S6b', 'S7');
  renderBands($('bands'), bandsVisible ? S.segs : [], dur);
  renderRegions($('regs'), bandsVisible ? S.regions : [], dur, {
    selected: S.region,
    showBars: has('S5', 'S6', 'S7'),
  });
  renderMarkers($('markers'), S.state === 'S7' ? S.override.markers : [], dur);
  renderRuler($('ruler'), dur, $('track').getBoundingClientRect().width, clockShort);

  $('skel').classList.toggle('hidden', S.state !== 'S2');
  const scan = $('scan');
  scan.classList.toggle('hidden', S.state !== 'S3');
  if (S.state === 'S3') scan.style.width = `${(S.osdFrac * 100).toFixed(1)}%`;

  $('tl-sub').textContent = `mix · 16 kHz mono · ${clock(dur)}`;

  const pill = $('tl-pill');
  pill.textContent = TL_PILL[S.state] || '';
  pill.className = 'pill' + (has('S2', 'S3', 'S5') ? ' running'
    : has('S4', 'S6', 'S6b', 'S7') ? ' done' : '');

  const hint = $('tl-hint');
  hint.textContent = TL_HINT[S.state] || '';
  hint.style.color = S.state === 'S6b' ? 'var(--ok)' : '';
}

function renderStats() {
  const nReg = S.regions.length;
  const calls = totalCalls();
  $('st1v').textContent = `${Math.round(pct(S.overlapS, S.durationS))} %`;
  $('st1s').textContent = `${num(S.overlapS, 1)} s z ${num(S.durationS, 1)} s`;
  $('st2v').textContent = `${nReg} → ${calls}`;
  $('st2s').textContent = nReg === 0 ? 'separator nie był uruchamiany' : 'okno 4 s, hop 2 s';
  if (calls === 0) {
    $('st3v').textContent = '—';
    $('st3s').textContent = 'separator nie był potrzebny';
  } else {
    $('st3v').textContent = `~${num(S.wholeCalls / calls, 1)}×`;
    $('st3s').textContent = `separacja całości: ${S.wholeCalls} ` +
      plural(S.wholeCalls, 'wywołanie', 'wywołania', 'wywołań');
  }
}

// ---------------------------------------------------------------------------
// region inspector
// ---------------------------------------------------------------------------
const INSP_EMPTY = {
  S2: 'Nagranie jest przygotowywane — regiony pojawią się po detekcji nakładania.',
  S3: 'Detektor nakładania jeszcze pracuje. Za chwilę zobaczysz, gdzie w ogóle warto uruchomić separator.',
  S4: 'Regiony nakładania są już wyznaczone. Inspektor otworzy się, gdy separator skończy pierwszy region.',
  S5: 'Separator pracuje. Region otworzy się, gdy będzie gotowy.',
  S6b: 'W tym nagraniu nie ma regionów nakładania — nie ma czego rozdzielać ani czego tu odsłuchiwać.',
};

function renderInspector() {
  const region = S.region !== null ? S.regions[S.region] : null;
  const body = $('insp-body');
  const empty = $('insp-empty');
  if (!region || !region.done) {
    body.classList.add('hidden');
    empty.classList.remove('hidden');
    $('insp-close').classList.add('hidden');
    empty.querySelector('.lead').textContent = INSP_EMPTY[S.state] ||
      'Kliknij region nakładania na osi czasu, żeby odsłuchać mix i dwie rozdzielone ścieżki tego fragmentu.';
    return;
  }
  empty.classList.add('hidden');
  body.classList.remove('hidden');
  $('insp-close').classList.remove('hidden');

  const [start, end] = region.pad;
  const length = end - start;
  $('ins-title').textContent = `Region nakładania #${S.region + 1}`;
  $('ins-time').textContent = `${clock(start)}–${clock(end)} · ${num(length, 1)} s`;
  document.querySelectorAll('.seg [data-src]').forEach((button) => {
    button.classList.toggle('active', button.dataset.src === S.src);
  });
  $('ins-loop').classList.toggle('active', S.loop);

  const host = $('ins-dl');
  host.innerHTML = '';
  for (const key of ['mix', 's1', 's2']) {
    const link = document.createElement('a');
    link.href = '#';
    link.dataset.dl = key;
    link.textContent = `region${S.region + 1}_${key}.wav`;
    host.appendChild(link);
  }
}

// ---------------------------------------------------------------------------
// single-overlap card + override panel
// ---------------------------------------------------------------------------
function renderSingle() {
  const region = S.regions.length === 1 ? S.regions[0] : null;
  const show = has('S6', 'S7') && !!region && region.done && !!region.full;
  $('card-single').classList.toggle('hidden', !show);
  if (!show) return;
  $('sg1len').textContent = clock(S.durationS);
  $('sg2len').textContent = clock(S.durationS);
}

function renderOverride() {
  const panel = $('card-override');
  panel.classList.toggle('hidden', !has('S4', 'S5', 'S6', 'S6b', 'S7'));
  const done = S.override.done;
  $('ov-result').classList.toggle('hidden', !done);
  if (done) {
    panel.open = true;
    $('ov1len').textContent = clock(S.durationS);
    $('ov2len').textContent = clock(S.durationS);
    const n = S.override.markers.length;
    $('ov-markernote').textContent = n
      ? `Na osi czasu zaznaczyliśmy ${n} ${plural(n, 'fragment', 'fragmenty', 'fragmentów')} solo, ` +
        'w których słychać strumień fantomowy: część głosu przecieka do drugiego wyjścia i wraca jako oddzielny „mówca”.'
      : 'W tym nagraniu nie ma fragmentów solo dość długich, żeby zaznaczyć je na osi czasu — posłuchaj obu strumieni w całości.';
  }
  const button = $('ov-run');
  button.disabled = done || isBusy();
  button.textContent = done ? 'Separacja całości wykonana'
    : S.override.running ? `Separacja całości… ${Math.round(S.override.frac * 100)} %`
      : 'Uruchom separację całości';
  $('ov-cancelrow').classList.toggle('hidden', !S.override.running);
}

// ---------------------------------------------------------------------------
// error card
// ---------------------------------------------------------------------------
function renderError() {
  const show = S.state === 'E1' && !!S.err;
  $('card-error').classList.toggle('hidden', !show);
  if (!show) return;
  $('err-title').textContent = S.err.title;
  $('err-type').textContent = S.err.type;
  $('err-msg').textContent = S.err.msg;
  $('err-retry').textContent = S.err.retry;
  $('err-retry').classList.toggle('hidden', !S.err.retry);
}

// ---------------------------------------------------------------------------
// transports — split out because playback updates them every animation frame,
// which must not cost a whole re-render
// ---------------------------------------------------------------------------
export function paintTransports() {
  document.querySelectorAll('[data-full]').forEach((button) => {
    button.textContent = S.playingFull === button.dataset.full ? '❚❚' : '▶';
  });

  const dur = S.durationS;
  $('ph').style.left = `${pct(S.playhead, dur).toFixed(3)}%`;
  $('tl-time').textContent = `${clock(S.playhead)} / ${clock(dur)}`;
  $('tl-seek').value = String(Math.round(pct(S.playhead, dur) * 10));
  $('tl-play').textContent = S.playing === 'clip' ? '❚❚' : '▶';

  const region = S.region !== null ? S.regions[S.region] : null;
  if (!region || !region.done) return;
  const length = region.pad[1] - region.pad[0];
  $('ins-clock').textContent = `${clock(S.regHead)} / ${clock(length)}`;
  $('ins-seek').value = String(Math.round(pct(S.regHead, length) * 10));
  $('rph').style.left = `${pct(S.regHead, length).toFixed(3)}%`;
  $('ins-play').textContent = S.playing === 'region' ? '❚❚' : '▶';
}

// ---------------------------------------------------------------------------
export function render() {
  renderInput();
  renderPipe();
  renderError();

  const showSplit = has('S2', 'S3', 'S4', 'S5', 'S6', 'S6b', 'S7');
  $('split').classList.toggle('hidden', !showSplit);
  $('card-stats').classList.toggle('hidden', !has('S4', 'S5', 'S6', 'S6b', 'S7'));
  if (showSplit) {
    renderTimeline();
    renderStats();
    renderInspector();
    paintTransports();
  }
  renderSingle();
  renderOverride();
  redrawCanvases();
}
