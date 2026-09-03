# Road 2 — OSD-gated on-device demo: UI/UX design spec (v1, for author acceptance)

Orchestrator-authored design decisions. Source plan: `docs/fable_plans/frontends_road2_ondevice.md`.
Recon findings baked in: `recon_poc.md` (same directory).

## 0. Framing

A **demonstrator, not an experiment** (B8 ruling): no quality numbers, no SI-SDR, no "demonstrated" claims anywhere in the UI. The page tells ONE story: *separation is valuable because it is routed*. Every screen state serves that story, including the failure demo ("separate anyway" = M2 made audible).

Fully client-side, GitHub Pages, QR-able from the thesis defense. Phone-first layout, desktop enhanced. Audio never leaves the device — say so prominently (it's a genuinely good demo line).

## 1. Visual language — lifted from Road 1, not the PoC

- **Tokens**: reuse `webapp/static/app.css` GitHub-Primer token system: light + dark + `auto/jasny/ciemny` three-way toggle (port `theme.js`), system font stack, no Google Fonts, no emoji headings, no gradients.
- **Speaker colors**: `--spkA #1f6feb` (blue), `--spkB #e36209` (orange) — but ONLY inside a region inspector (local identity); never across regions (no global speaker identity without an embedder — out of scope by plan).
- **Band colors on the timeline**:
  - *cisza* (non-speech): near-transparent / dimmed
  - *solo* (one speaker): neutral desaturated green-teal tint (NOT spkA blue — solo bands carry no identity)
  - *nakładanie* (overlap): Road 1's `--ovl` red tint + `--ovl-line`; after separation completes, the region gains a subtle "separated" affordance (filled underline bar in a success hue + play glyph)
- **Language**: Polish UI, English technical tokens untranslated (overlap, routing, OSD, ONNX, WASM) — same convention as Road 1.

## 2. Page structure (single page, mobile-first, single column; ≥ 960 px gets side-by-side timeline+inspector)

1. **Header** — title `Separacja mowy w przeglądarce` + subtitle one-liner (`Cały pipeline działa lokalnie — audio nie opuszcza urządzenia`), theme toggle right.
2. **Input card** — three tabs:
   - `Przykłady`: exactly two example clips, each with a one-line *why this clip* label: (a) `Rozmowa z przeplotem — mowa pojedyncza + krótkie nakładania` (the routing story), (b) `Pełne nakładanie — dwoje mówców przez cały klip` (the separator showcase). Duration shown, real values.
   - `Plik`: drag-drop / file picker. Accepted formats line. 
   - `Mikrofon`: record button + elapsed timer + stop; note that echo cancellation/AGC/noise suppression are disabled for honest input.
3. **Pipeline strip** — the routing philosophy made visible; three steps as connected stage chips:
   `1 · Detekcja nakładania (OSD)` → `2 · Routing` → `3 · Separacja — tylko regiony nakładania`.
   Each chip has idle / running (spinner + per-model download progress with MB) / done states. This strip IS the thesis argument in miniature — it must read as "the system decides WHERE to separate before it separates".
4. **Timeline (centerpiece)** — full-width card:
   - Canvas waveform peaks (mono mix), percent-positioned band overlays per §1 colors, click-to-seek, global play/pause + playhead.
   - Bands appear progressively: after OSD+routing all bands appear; overlap regions then fill with a per-region progress affordance as the separator processes them one by one (the user *watches* routing save work).
   - Region click → selects region → opens the inspector (scrolls to it on mobile).
   - Legend row under the timeline: cisza / solo / nakładanie / nakładanie (rozseparowane).
5. **Stats strip** — small row of three tiles after routing completes: `% nagrania z nakładaniem`, `liczba regionów → liczba wywołań separatora`, `~N× mniej obliczeń separatora niż separacja całości`. Routing facts, not quality claims. [FLAG FOR AUTHOR: is the ~N× tile acceptable under the no-numbers ruling? Recommendation: yes — it quantifies routing, not separation quality.]
6. **Region inspector** — appears when an overlap region is selected:
   - Region header: `Region nakładania #k · 0:12.4–0:15.1 · 2.7 s`
   - Big A/B/C switch: `Mix` / `Mówca 1` / `Mówca 2` (spkA/spkB colors; labeled as *local* to this region). Switching WHILE PLAYING keeps position — this instant-switch is the core "wow" interaction; add hint copy `Przełączaj podczas odtwarzania`.
   - Small region-detail waveform, loop toggle, download links (mix/s1/s2 wav).
7. **Special-case banner** (only when exactly ONE overlap region): `Dokładnie jeden region nakładania — brak problemu permutacji między regionami. Możesz odsłuchać złożone, pełne ścieżki obu mówców.` + two full-stream play buttons. (This is the plan's honest special case; hidden otherwise.)
8. **Override panel ("Separuj mimo wszystko")** — visually quarantined at the bottom: separate card, dashed border, warning tint, collapsed by default behind a disclosure:
   - Copy explains M2 in two sentences: the separator was trained on 100 % overlap; fed solo speech it is out-of-distribution and produces a phantom second stream. `Uruchom separację całego nagrania i posłuchaj, co się dzieje na fragmentach solo.`
   - Runs whole-clip chunked separation → two full-stream players + the timeline highlights solo regions where the phantom is audible (`posłuchaj tutaj` markers on solo bands).
   - This panel is the interactive M2 illustration — it must feel like a deliberate experiment, not a broken feature.
9. **Explainer (collapsible)** — `Dlaczego nie separujemy wszystkiego?` — 3 short paragraphs + a tiny static diagram of the routing decision; mentions the same constants run in the full server pipeline (min 0.2 s, merge 0.5 s, pad 1.0 s — fidelity claim).
10. **Footer** — model inventory with sizes (OSD: pyannote-segmentation-3.0 fp16 ONNX ~3 MB, MIT, cite Plaquet & Bredin 2023; separator: [PENDING B9 SPIKE — MossFormer2-matched-128k or SepFormer-128k], ORT-Web self-hosted), link to thesis context, license notices.

## 3. States (all must be shown in the mockup via a dev state-switcher)

S0 idle · S1 models downloading (per-model progress bars with MB) · S2 decoding/preparing audio · S3 OSD running · S4 routed (bands + stats visible, separation queued) · S5 separating region k/N (progressive fill) · S6 done · S6b done-no-overlap (**informative null**: `Nie wykryto nakładania — routing: separacja niepotrzebna. Cały klip to mowa pojedyncza lub cisza.` — positive framing, stats strip still shown) · S7 override running/done · E1 error (mic denied, decode failed, model fetch failed — inline card with retry, never `alert()`).

## 4. Interaction details

- Instant A/B/C source switch with position-preserving playback (single AudioContext, three gain nodes, mute-switch).
- Click-to-seek anywhere on timeline; region click selects; double-tap region = play region loop (mobile).
- Zoom: none in v1 (defense clips are 20–60 s; percent layout suffices). [Decision: cut scrollport-zoom from Road 1 — YAGNI for this scope.]
- Keyboard: space = play/pause, 1/2/3 = source switch when inspector open. Desktop nicety, not required on mobile.
- Reduced motion: respect `prefers-reduced-motion` (no pulsing fills).

## 5. Explicitly out of scope (per plan — restate in mockup footnotes)

Cross-region speaker assembly (except the single-overlap special case) · any speaker embedder · quality metrics in UI · server communication of any kind · zoom · more than 2 example clips.

## 6. Open items riding along with this design for the author

1. **CLARIN fragments on a public page?** If no: self-record the mostly-solo example (plan §5). [Only the author can answer.]
2. `~N× mniej obliczeń` stat tile — keep or drop (see §5 flag).
3. Override panel framing acceptable as designed (collapsed disclosure, experiment framing)?
4. Separator dropdown copy — resolves automatically when the B9 spike verdict lands.
