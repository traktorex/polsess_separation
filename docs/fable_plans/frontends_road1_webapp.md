# Road 1 — Pipeline showcase webapp: design (accepted)

**Status: ACCEPTED — IMPLEMENTATION GO given by author 2026-07-28** (examples-page mode §5.5 confirmed; branch `feature/webapp-showcase`; Opus 5 subagents implement, orchestrator reviews/decides/tests). Decision log §8.
Provenance: four-agent research session 2026-07-27 (clarin_review inventory, asr_pipeline surface map, framework/UX research) + author review 2026-07-28. Companions: `frontends_road2_ondevice.md` (separate track, discussed elsewhere), backlog B9.

## 1. Product definition

An **interactive showcase** for `asr_pipeline/`: upload (or pick) a recording → watch the pipeline stages execute live → listen to the two separated speaker streams and read speaker-attributed transcripts. "Real-time" means *interactive with live progress*, not streaming. Ground truth from 6,854 historical `run_meta.json` (predominantly ~90 s fragments — see the ETA weighting rule, §5.2): median **48.9 s**, p25 40.1, p75 58.0, **p95 168.8 s**.

Audiences: author; supervisors (persistent authenticated link); defense audience — **the webapp is a thesis artifact and will likely be shown live at the defense** (author, 2026-07-28) → pin dependency versions, archive a tagged build, and follow the §6 demo-day runbook strictly. Not a diagnostic tool — diagnostic panels ship **off by default** behind a drawer.

**Config identity (verified 2026-07-28):** the shipped config `configs/sweep_best_e31_refineplus.yaml` **is** the v41_merge arm — its header reads `SHIPPED BEST config — "v41_merge" (2026-07-04, V5_INSTRUMENT_PREREG.md §PHASE-2 VERDICT)`. The filename is the historical lineage name (dr_refineplus, 2026-06-19); content was updated through the v2→v5 campaigns (git: `7b3e9bd` finalists → `34d8fd3` v2_finalist → v41_merge fix-ups). It sets `relabel.enabled: true` → **8 stages**, and `assembly.output_mode: full_length` → assembled streams are mixture-length **by config**, so the whole results view shares one clock without timestamp remapping.

## 2. Architecture (accepted)

**FastAPI + hand-written, no-build-step JS. Progress via polling (`GET /jobs/{id}` ~1 Hz), not SSE** (cloudflared Quick Tunnels buffer SSE-over-GET — cloudflared#1449). ~225 lines of timeline/waveform/karaoke/stream-switcher JS liftable from `scripts/build_review_page.py`; zero new Python dependencies (fastapi/uvicorn/starlette/python-multipart/jinja2 all present); Starlette `FileResponse` gives HTTP-Range audio seeking free.

**Gradio rejected outright (author).** The previously mooted `/lab` escape hatch — mounting an auto-generated Gradio Blocks knob-panel at a sub-path as a web `explore_pipeline` — is dropped with it. Future config choice (author: "later a dropdown with choosable configs") will be a plain `<select>` over `asr_pipeline/configs/*.yaml` presets in our own UI, run through `apply_overrides`/`load_pipeline_config_from_yaml` — no framework needed. v1 ships **one fixed config** (v41_merge) and zero knobs.

## 3. Placement and SCOPE compliance

Unchanged from the accepted proposal: new top-level **`webapp/`**, pure read-only consumer of `asr_pipeline`; one worker thread + `queue.Queue` (concurrency 1 by physics, FIFO, visible queue position); webapp owns per-job failure isolation (error card with exception type; the pipeline keeps failing loudly underneath — SCOPE §4.2 reading); `check_preflight(cfg)` at startup with the non-raising `preflight()` list as a ✓/✗ config-health screen; no silent substitution anywhere — the >240 s Sortformer streaming-model swap warning is surfaced in the UI.

## 4. Backend design (accepted; hooks approved)

```
webapp/
  app.py        # FastAPI: GET /, POST /jobs, GET /jobs/{id}, GET /jobs/{id}/files/*, GET /examples
  queue.py      # worker thread + queue.Queue; JOBS: dict[str, JobState]
  render.py     # peaks (min/max + RMS, see §5.3); optional encode helpers
  static/       # app.js, app.css (ES modules, no build step)
  templates/    # index.html (jinja2)
```

Worker: `run_batch(cfg, [(job_id, path)], out_root, "pipeline", on_event=job.record, skip_existing=False)`. Progress = `on_event` (stage cards, load/run split) + debug-log tail (intra-stage counts, warnings — SCOPE §4.3 visibility). Jobs persist on disk under an eval-tree-shaped `out_root`; job id in URL (deep-linkable); `skip_existing` caches the examples gallery.

**Package hooks — ALL THREE APPROVED by the author 2026-07-28** (additive, default-inert, `on_event=None → byte-identical` precedent; implement during the build, each with a matching test):
1. **Assembly attribution diagnostics**: per-overlap `pairing` (straight/swapped/arbitrary-…) + ECAPA cosine sums onto `ctx` and into `metadata.json` (the `diarization_diag` precedent; currently function-local in `_assign_overlaps`). Feeds the attribution diagnostic panel.
2. **`stage_progress` event** `{"event","stage","done","total"}` emitted at the existing dlog loop points (separation/post-sep/assembly). Coarser fallback remains the log tail.
3. **`overlaps` array** added to the eval-facing `diarization.json` in `io.py`. Feeds the routing dropped/merged-diff panel.

(The formerly-listed timestamp_map export is **not needed**: `output_mode: full_length` is pinned in the shipped config — §1. Revisit only if `shortened` mode is ever exposed.)

## 5. UI design (accepted with modifications)

Visual language unchanged: review-page palette (A `#1f6feb` / B `#e36209`, translucent red overlap), one color = one speaker everywhere, N lanes from `metadata.speakers`. **Language (author ruling): Polish UI, but technical vocabulary stays English where Polish is unwieldy — stage names, `load`/`run` labels, file names.**

### 5.1 Screen flow

Home (upload + examples link) → Job page (progress morphing into results, same URL) → Examples gallery. No upload duration cap (author ruling — §6 has the practical ceiling).

### 5.2 Progress (modified)

**One linear chain of 8 stage rows — no phase grouping** (author ruling: enhancement/relabel sat awkwardly in a "Separate" phase; the pipeline is a strict sequence, so the UI shows a single chain): diarization → routing → enhancement → separation → post_separation → relabel → assembly → transcription. Row states `pending / loading / running / done (X s) / skipped / failed`; dead time labeled (`load MossFormer2 · 26.4 M params · matched-128k`), live counts from the log tail (`region 7/12`), retrospective durations on completed rows. No global percent bar.

**ETA (modified — duration-weighted):** the historical corpus is ~90 s fragments, so raw per-stage medians would mislead on other lengths. Estimator: per-stage `load_s` median (constant, duration-independent) + per-stage `run_s`-per-audio-second median × upload duration; separation/post-sep re-estimated **after routing completes** using actual overlap seconds (their cost tracks overlap, not duration). Presented as coarse text ("około minuty"), refined as stages land; accumulate every finished job's `stage_timings` (with audio duration) to improve the estimator over time.

**Progressive disclosure (modified):** diarization timeline renders when routing completes (~s 8) with **provisional** speaker labels; after relabel+assembly finalize the stream↔speaker mapping and `spk_to_label`, the timeline component silently re-renders with final labels (visually identical in the common case; the refresh guarantees consistency). Audio players appear at assembly, transcripts at transcription.

### 5.3 Results layout (modified in two places)

Order unchanged: header (name, duration, small chip row, warning banners, provenance line) → timeline stack → transport (Mixture ↔ A ↔ Both A→L·B→R ↔ B, position-preserving element swap) → transcripts → downloads.

**Waveforms (author ruling 2026-07-28: keep v1 simple, upgrade = deferred nice-to-have):** v1 ships the review page's proven form — server-computed peak envelope (`_peaks()`-style, ~800 buckets), canvas/SVG-rendered, click-to-seek, shared playhead, zero client decode. **Deferred nice-to-have (return if time allows):** min/max envelope + RMS body (DAW form, ~2,000 buckets) — design `render.py`'s peaks JSON with room for extra per-bucket fields so the upgrade is additive; zoom likewise deferred (wavesurfer.js v7 with precomputed peaks = drop-in path). Spectrograms stay in the diagnostics drawer.

**Transcripts (modified):** two speaker columns on the shared clock, turn blocks, karaoke word pill, click-word-to-seek, low-confidence shading, yielding autoscroll + "wróć do kursora". Additionally (author): columns scroll internally by default **and** each has a "rozwiń całość" affordance removing the inner max-height so the entire transcript reads in-page.

**ELAN future-proofing (author request — Life 2 / CLARIN, do not build now, do not block later):** keep the transcript component boundary clean so the read-only columns can later be swapped for a simple ELAN-style tier viewer/editor. Concretely: (a) segments get stable ids in the client data model; (b) the transcript component receives tier-shaped data (speaker → list of {id, start, end, text, words}) — which is exactly the `.eaf`/WhisperX shape already; (c) server routes keep an obvious slot for a future `PATCH /jobs/{id}/segments/{sid}`; (d) `annotation.eaf` remains a first-class download. No editor code in v1.

### 5.4 Diagnostics drawer (unchanged)

Off by default; lazy-rendered panels mapping to notebook cells: routing region table + dropped/merged diff (hook 3), enhancement before/after, per-overlap separation drawer, BWE spectrograms, **attribution panel (hook 1)**, sortformer head census, stage timings + log stream, config health. In-process `ctx` retention (last job) or per-run spill — decide at implementation.

### 5.5 Examples page (recommendation recorded — awaiting author confirmation)

Author framing: pre-processed gallery vs live-runnable examples. **Recommendation: hybrid that stays 95% option A.** The gallery shows **frozen, pre-processed results** (instant load, no GPU dependency, deterministic — the numbers match what the thesis reports) with GT transcripts + cpWER/cpCER shown **only here**. Each example additionally offers **"uruchom ponownie"** — a button that submits the example's mixture as a *normal new job* through the standard upload path (no GT, no scoring on that path). This gives the live-demo moment at the defense ("watch it process this exact recording") without integrating metric computation and GT display into the main job page. CLARIN fragments are cleared for the authenticated page (author, 2026-07-28).

## 6. Deployment & demo-day runbook

- **Local-first (author, 2026-07-28): the app is mainly used on the machine itself / home LAN.** Tunnels are the secondary path: cloudflared named tunnel + Cloudflare Access for a persistent supervisor link, open quick tunnel for the defense — the 100 MB body limit applies only there and the UI mentions it only when relevant.
- **No upload duration cap** (author ruling 2026-07-28); honest ceilings instead — the >240 s streaming-diarizer warning banner and a duration-scaled ETA. Pipeline verified end-to-end to 1:35:29 (pyannote path) — long jobs are allowed, just labeled.
- Warm-up run at app start and ~5 min before the defense; `HF_HUB_OFFLINE=1`; keep `deterministic: true`.
- Uploads deleted after 24 h (configurable); examples persistent.
- **License note:** ECAPA2 + Sortformer v1 are CC-BY-NC — fine for thesis/defense/supervisor use (non-commercial academic); flag before any public-facing or CLARIN-production deployment (the documented Life-2 path is the NVIDIA-Open streaming v2.1 swap).
- Thesis-artifact duties (author: shown live at defense): pin `requirements` versions for `webapp/`, tag the demo build, keep a recorded fallback video of one full run.

## 7. Effort estimate

Backend ~2 d · front-end ~3–4 d · package hooks + tests ~0.5–1 d · examples page + polish ~1–2 d · deployment/auth/runbook ~0.5–1 d → **≈ 8–10 working days**, no GPU budget beyond example pre-processing. Phasing: core (backend + progress + results) first — demoable ~day 4; hooks + diagnostics drawer + examples second.

## 8. Decision log (author, 2026-07-28)

| # | Decision |
|---|---|
| 1 | Architecture (FastAPI + no-build JS + polling) — **accepted**; Gradio rejected; `/lab` dropped |
| 2 | Layout — **accepted** with modifications below |
| 3 | v1 = one fixed config (v41_merge); config **dropdown later**, plain select over presets |
| 4 | Upload cap — **none** (technical ceilings only, stated honestly) |
| 5 | Language: **Polish UI, English technical terms** (load/run, stage names) |
| 6 | CLARIN fragments **may** appear on the authenticated examples page |
| 7 | Package hooks 1–3 — **all approved** |
| 8 | Webapp **is a thesis artifact**, likely live at defense → pin versions, tagged build |
| 9 | `explore_pipeline.ipynb` **stays** (author's lab; webapp = everyone else) |
| 5.a | Stage display: **single linear chain**, no phase grouping |
| 5.c | ETA: **duration-weighted** estimator (load constant + run per-audio-second; overlap-aware after routing) |
| 5.d | Early timeline **re-renders with final labels** after relabel/assembly |
| 5.e | Waveforms: v1 = simple peak envelope; **min/max+RMS upgrade = deferred nice-to-have** (author 2026-07-28), zoom likewise |
| 5.f | Transcripts: internal scroll + **"rozwiń całość"** full-read affordance |
| 5.g | **ELAN-style tier editor slot reserved** (Life 2): stable segment ids, tier-shaped data, future PATCH slot — no editor in v1 |
| 5.h | Examples page: **pre-processed gallery + "uruchom ponownie"-as-new-job** — **confirmed 2026-07-28** |
