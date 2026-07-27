# Road 1 — Pipeline showcase webapp: design proposal

**Status: DESIGN PROPOSAL — awaiting author acceptance (author ruling 2026-07-27: no implementation until the design is accepted).**
Provenance: four-agent research session 2026-07-27 — `clarin_review` presentation inventory, `asr_pipeline` data-surface map, framework/UX research (web), plus author rulings from the conceptual session. Companion docs: `frontends_road2_ondevice.md` (browser demo), backlog B9 (quantization ladder).

## 1. Product definition

An **interactive showcase** for `asr_pipeline/`: upload (or pick) a recording → watch the pipeline stages execute live → listen to the two separated speaker streams and read speaker-attributed transcripts. "Real-time" means *interactive with live progress*, not streaming — the pipeline is phase-major batch. Ground truth from 6,854 historical `run_meta.json`: median **48.9 s** per fragment, p25 40.1 s, p75 58.0 s, **p95 168.8 s** (3.5× median — this kills any global percent bar; see §5.2).

Audiences: the author; supervisors (persistent authenticated link); defense audience (live or recorded). Explicitly **not** a diagnostic tool — diagnostic panels exist but ship **off by default** behind a drawer (author ruling).

The shipped config is `sweep_best_e31_refineplus.yaml`, which enables `relabel` → **8 stages**, not 7: diarization → routing → enhancement → separation → post_separation_processing → relabel → assembly → transcription.

## 2. Architecture decision

**Primary: FastAPI + hand-written, no-build-step JS. Progress via polling (`GET /jobs/{id}` ~1 Hz), not SSE.**

Decisive reasons:
1. **The components already exist, written by the author.** `scripts/build_review_page.py` contains a working 4-lane SVG diarization timeline with overlap bands + adaptive ruler + click-to-seek; server-side peak extraction (`_peaks`, 400 buckets); two-lane SVG waveforms; an A/B/Both stream switcher that preserves `currentTime`; karaoke word-highlighting with change-only DOM writes; one `requestAnimationFrame` clock driving everything. ~225 lines directly liftable, already debugged, already matching the data.
2. **Zero new dependencies.** `fastapi 0.136.0`, `uvicorn`, `starlette 1.0`, `python-multipart`, `jinja2` are all already in the fragile main venv. Starlette's `FileResponse` does HTTP Range natively → `<audio>` seeking works for free.
3. **SSE is a demo-day landmine**: cloudflared Quick Tunnels buffer SSE-over-GET until the connection closes (cloudflared#1449, open). Polling is immune to every proxy pathology, survives page reload and phone sleep, and a ~50 s job at 1 Hz is ~50 requests. WebSocket (supported by cloudflared) only if sub-second smoothness ever proves necessary.
4. **SCOPE §1 pushes the same way**: the queue must live outside `asr_pipeline/`; a framework that wants to own the queue fights the contract.
5. Design polish is the stated priority; FastAPI + own JS is the option where the layout is fully ours.

**Runner-up (genuine second): Gradio 6** — 6.12.0 already installed; `gr.HTML(html_template=…, js_on_load=…, server_functions=…)` + `head=` makes bespoke components possible without the custom-component toolchain now. Rejected for the showcase because the layout is the product and Gradio 6.x is churning (breaking changes across minor versions). **Escape hatch kept**: `gr.mount_gradio_app` can later mount a knob-twiddling "lab" Blocks app at `/lab` in the same process — a web `explore_pipeline` — without touching the showcase surface. Don't build it up front.

Rejected: Streamlit (rerun model fights a 50 s blocking GPU job), NiceGUI (build all audio UI ourselves *and* inherit Material look), Reflex/FastHTML (new dependency, no payoff).

## 3. Placement and SCOPE compliance

- New top-level **`webapp/`** directory, sibling of `asr_pipeline/`. Imports the package **read-only**; `asr_pipeline` never imports the webapp. No server code inside the package (SCOPE §1/§7 — Life-2-shaped work stays outside).
- **Queue**: one `threading.Thread` worker consuming a `queue.Queue`; `concurrency_limit = 1` by physics (12 GB GPU, phase-major). No platform job queue — SCOPE forbids it in Life 1.
- **Error semantics** (SCOPE §4.2 reading): the *webapp* owns per-job isolation exactly as `run_batch` does — a failed job renders an error card with the exception type; the pipeline keeps failing loudly underneath. No auto-retry, no partial-success framing, no fallback toggles (§4.1 no-silent-substitution — includes the Sortformer >240 s streaming-model swap warning, which the UI must surface).
- **Startup**: `check_preflight(cfg)` before serving — missing `$SORTFORMER_VENV_PY`/`$HF_TOKEN` fails in seconds, with the non-raising `preflight(cfg)` list rendered as a ✓/✗ config-health screen.

## 4. Backend design

```
webapp/
  app.py        # FastAPI: GET /, POST /jobs (upload), GET /jobs/{id} (poll), GET /jobs/{id}/files/*, GET /examples
  queue.py      # worker thread + queue.Queue; JOBS: dict[str, JobState]
  render.py     # _peaks() lifted from build_review_page.py; optional mp3/stereo encode
  static/       # app.js, app.css (ES modules, no build step)
  templates/    # index.html (jinja2)
```

- Worker executes `asr_pipeline.batch.run_batch(cfg, [(job_id, path)], out_root, "pipeline", on_event=job.record, skip_existing=False)` — already a complete single-job executor with GPU teardown and `write_run_outputs`.
- **Progress sources**: (a) `on_event` `stage_start`/`stage_end` (`load_s`/`run_s` split — trustworthy, measured around the actual calls) for the stage cards; (b) tail of `/tmp/asr_pipeline_debug.log` (`[  12.34s] [stage] message`, fsync'd, one run's worth) for intra-stage progress and **all warnings** — SCOPE §4.3 visibility carried into the UI. No `stage_end` on failure — exception propagation is the failure signal.
- **Job persistence**: jobs live on disk under an eval-tree-shaped `out_root` (the layout `run_batch` already writes). Free result caching for canned examples via `skip_existing`; job id in the URL → deep-linkable results (fixes the review page's no-URL-state gap).
- Data contract consumed (all existing): `stream_A/B.wav`, `transcript_*.{txt,json}` (word timestamps 100% populated + per-word alignment `score`), `annotation.eaf`, `diarization.json`, `routing.json`, `metadata.json` (provenance/config snapshot, `spk_to_label`, `weak_anchor`, sortformer `diarization_diag`), `run_meta.json`, input mixture copy. Verified: assembled streams are exactly mixture-length under the shipped config → everything shares one clock, **no timestamp_map needed** for the default surface. (Re-verify at implementation time; if `assembly.output_mode: shortened` is ever exposed, the timestamp_map hook becomes relevant.)

**Optional package hooks — each needs explicit author sign-off, none blocks v1** (in value order, all additive/default-inert, following the `on_event=None → byte-identical` precedent):
1. Assembly decision diagnostics onto `ctx` + `metadata.json` (per-overlap `pairing` + ECAPA cosine sums — currently function-local in `_assign_overlaps`; `diarization_diag` precedent; ~20 lines). Enables the attribution-confidence diagnostic panel.
2. `stage_progress` event `{stage, done, total}` (~5 lines at existing dlog points). Log-tailing is a viable alternative — only worth it if tailing proves brittle.
3. `overlaps` array into the eval-facing `diarization.json` (~3 lines) — enables the dropped/merged-region diff panel for file-based consumers.

## 5. UI design (the part to accept)

Visual language: carry the review page's palette (**A `#1f6feb` blue / B `#e36209` orange**, translucent red = overlap) and the notebook's color vocabulary (blue solo / red overlap pieces, magma spectrograms). One color = one speaker, everywhere — timeline, waveform, transcript headers, text tint. Render *N* speaker lanes from `metadata.speakers` (no hardcoded 2 in layout code — SCOPE §3; phantom-3rd-speaker runs display as-is, §10 q1).

### 5.1 Screen flow

**Home** (upload + canned examples) → **Job page** (progress morphing into results, same URL) → **Examples** (pre-run gallery). Uploads: drag-and-drop, client-side cap (§7), quality expectations stated (Polish speech, 2 speakers).

### 5.2 Progress: the pipeline diagram IS the progress UI

Eight stage cards in fixed order, grouped into three phases:

| Phase | Stages |
|---|---|
| **Understand** | diarization · routing |
| **Separate** | enhancement · separation · post_separation · relabel |
| **Transcribe** | assembly · transcription |

Card states: `pending / loading / running / done (X s) / skipped / failed`. The `load_s`/`run_s` split is shown as two sub-phases — **dead time is labeled, not hidden**: "⟳ loading MossFormer2 · 26.4 M params · matched-128k" → "▶ separating 12 overlap regions" (counts from the log tail). Completed cards show retrospective durations (GitHub-Actions pattern). **No global percentage** (p95/median = 3.5× guarantees a stalling bar); a coarse text ETA ("about a minute") from accumulated per-stage medians scaled by audio duration; past ~2.5× estimate, copy switches to "taking longer than usual — long recordings route through the streaming diarizer".

**Progressive disclosure**: the diarization timeline renders the moment routing completes (~second 8), audio players at assembly, transcripts at transcription. The viewer is reading real output while the GPU works. Queue: FIFO, visibly ("2 jobs ahead · ~2 min"), previous result stays interactive while queued.

### 5.3 Results layout (top to bottom)

1. **Header**: recording name, duration, a *small* chip row (speakers, № overlap regions, total overlap seconds) — not the review page's 10-chip wall. Warnings as banners (weak ECAPA anchor, streaming-diarizer swap, phantom 3rd speaker). Provenance line from `metadata.config` (`enh=… · sep=… · asr=…`).
2. **Timeline stack**, x-aligned, one shared playhead, click-anywhere-to-seek:
   - diarization lanes (one per speaker, overlap wash behind — `buildTimeline()` minus GT lanes);
   - waveform lanes A/B (server-rendered SVG from `_peaks()` — no client decode, no wavesurfer dependency);
   - adaptive time ruler.
3. **Transport**: one custom player with a four-way segmented control — **Mixture ↔ A ↔ Both (A→L·B→R) ↔ B** — preserving position and play state on switch (the review page's element-swap mechanism, kept deliberately over Web Audio crossfades: gapless enough, no iOS unlock gesture, debugged). *This control is the thesis claim made audible* — the audience hears the overlap collapse into two clean streams; the campaign's own result (separation the only Holm-significant stage) in ten seconds.
4. **Transcripts**: two side-by-side speaker columns (A left, B right) on a shared vertical axis — the honest rendering for an overlap-heavy corpus. Turn blocks (speaker + timestamp header, text below), karaoke word-pill highlight driven by the shared clock, click-any-word-to-seek, low-alignment-confidence words subtly shaded (per-word `score`, free), autoscroll that yields to manual scrolling with a "jump to playhead" affordance. Mixture mode collapses to a single column (`transcript_mixture`).
5. **Downloads**: `stream_A/B.wav`, transcripts (`.txt`/`.json`), `annotation.eaf` (first-class — SCOPE §9), `metadata.json`.

### 5.4 Diagnostics drawer (off by default — author ruling)

A single toggle reveals per-stage panels, lazy-rendered (the notebook's `MAX_DETAILED_PLOTS = 20` lesson — never render all N overlap panels eagerly):

| Panel | Source | Notebook precedent |
|---|---|---|
| Routing detail: region list, durations, dropped-by-`min_overlap_dur` / merged-by-`merge_gap` diff | `routing.json` (+hook 3 for the diff) | cell 9 |
| Enhancement before/after: waveform pair + two players | `enhanced_full` (spill or in-process) | cell 11 |
| Per-overlap separation drawer: mix/s1/s2 + VAD masks + silero prob step-plot + emit region | `ctx.overlap_separated` (in-process; never persisted) | cell 13 |
| BWE 2×2 spectrograms pre/post | recompute `s_raw × mask` (pre is not stored) | cell 15 |
| Assembly attribution: per-overlap straight/swapped + ECAPA cosines | **needs hook 1** | — (new) |
| Sortformer head census / fold accounting | `metadata.diarization_diag` | cell 7 |
| Stage timings table + debug-log stream | `run_meta.json` + log tail | cell 21 |
| Config health | `preflight(cfg)` list | cell 5 |

In-process panels require the server to keep the last job's `ctx` in memory (bounded: keep 1) or `spill_intermediate: true` per run — decide at implementation; no package change either way.

### 5.5 Examples page

Pre-run gallery (canned CLARIN fragments or self-recorded clips — see §8), cached via `skip_existing`. **Scoring (cpWER/cpCER vs frozen GT) appears here only, never on arbitrary uploads** (showcase/diagnostic line). Optionally per-mode rows (full / no_sep) as the academic comparison-table pattern — the only place config comparison exists in v1.

## 6. Deployment & demo-day runbook

- Serve on the training PC (WSL2). **Persistent supervisor link**: cloudflared *named* tunnel + Cloudflare Access (email OTP, free ≤50 users — enable the OTP IdP first). **Defense**: open quick tunnel, live. Cloudflare caps request bodies at 100 MB — irrelevant under our upload cap, but enforce client-side and say why.
- Upload cap **5 min** (recommendation; §8 q4): >240 s already auto-routes Sortformer to the streaming model with a loud warning the UI must display; the cap keeps VRAM headroom and demo latency sane.
- **Warm-up run at app start and ~5 min before the defense** (cold start ≫ 48.9 s: HF cache, per-stage loads). `HF_HUB_OFFLINE=1` (campaign-verified byte-identical, kills 504-aborts). Keep `deterministic: true` — a supervisor re-running the same file and getting the same transcript is worth more than the ~2× enhancement-stage speedup.
- Uploads deleted after 24 h (configurable); canned examples persistent.

## 7. Effort estimate

Backend (~250 lines Python): ~2 d. Front-end (lift + new layout + transcript columns): ~3–4 d. Examples page + polish: ~1–2 d. Deployment/auth/runbook: ~0.5–1 d. **Total ≈ 7–9 working days**, essentially zero GPU budget. Phasing: backend + progress + results core first (demoable at ~day 4); diagnostics drawer and examples page second.

## 8. Open questions for the author (accept/modify to unblock)

1. **Accept the FastAPI + no-build-JS + polling architecture?** (Gradio-6 `/lab` escape hatch reserved, not built.)
2. **Accept the §5 layout?** A visual mockup accompanies this doc — judge hierarchy and density there.
3. **Fixed config only?** Recommendation: showcase ships `sweep_best_e31_refineplus` with **zero knobs**; the diagnostics drawer may include a read-only config view; any knob-twiddling waits for the optional `/lab`. (Knobs on the main surface turn the showcase into the diagnostic tool it is explicitly not.)
4. Upload cap 5 min OK?
5. **UI language**: Polish (review-page precedent, supervisor/defense audience) vs English (thesis artifact, screenshots). Recommendation: Polish, English toggle only if cheap.
6. **Canned examples content**: may CLARIN fragments be exposed on an authenticated page (license/consent check needed) — or self-recorded clips?
7. **Package hooks** (§4): approve none/some — none blocks v1.
8. Is the webapp itself a thesis artifact (screenshots in ch7 / archived for the defense record)? If yes, pin versions and raise the polish bar accordingly.
9. Does `explore_pipeline.ipynb` stay the author's lab (recommendation: yes — the drawer showcases, the notebook investigates)?
