# Road 2 — On-device browser demo: scope decision + design proposal

**Status: SCOPE DECISION PROPOSAL — awaiting author acceptance.** Provenance: research session 2026-07-27 (author-posed question: *"what is being showcased?"* — whole-clip separation presupposes overlap everywhere). Companions: `frontends_road1_webapp.md`, backlog **B9** (ONNX export + quantization ladder — separate, independent work item).

## 1. The problem with the January PoC's framing

`mobile/webapp/` applies chunked separation (4 s windows, 2 s hop, overlap-add, 8 kHz) to the **entire** uploaded recording. That presupposes 100 % overlapped speech — the exact assumption the thesis names as the training/deployment mismatch: ch6 **M2** ("100 % overlap in training vs. mostly-solo real conversations; a separator fed non-overlapped audio is out-of-distribution and empirically degenerates — phantom-stream behavior") and backlog B2's blocker note (separators "degenerate on 1-speaker input", author-tested). A whole-clip demo therefore argues *against* the thesis; its default failure mode (user records a solo clip → phantom second stream) is guaranteed and looks broken. Additionally, the current ONNX exports are **quality-stale** (January checkpoints, pre-`C_new_64`) — refresh is B9.

**Rejected fix — output VAD gating alone**: leaked phantom streams are *speech-shaped* and pass VAD (Kalda/Coria et al. 2024, arXiv:2402.00067 — leaked segments "are detected as speech by the following VAD module"). Energy-ratio/inter-stream-correlation gates have no canonical prior art and no eval set here — a second research project in disguise. Hiding the phantom stream is worse than avoiding it.

## 2. Decision: OSD-gated separation with the routing decision made visible

The browser demo becomes a **miniature of the system's routing philosophy**, run entirely client-side:

1. **Overlap detection**: pyannote **segmentation-3.0** ONNX — its powerset head yields per-frame non-speech / solo / overlap directly from an argmax (16.875 ms frames), **no clustering, no embeddings**. Raw-waveform input at 16 kHz, genuine ORT-Web drop-in.
2. **Routing, ported verbatim** from `asr_pipeline/stages/routing.py` (~12 lines): filter overlaps < `min_overlap_dur = 0.20 s`, merge within `merge_gap = 0.50 s`, `context_pad = 1.0 s` — same constants as the server (cheap, honest fidelity claim).
3. **Separation only on overlap regions** (existing chunked engine, applied per region). Solo regions pass through untouched.
4. **Presentation = interactive timeline** (wavesurfer.js v7 regions or hand-rolled SVG): waveform + colored bands (non-speech / solo / **overlap → separated**); clicking an overlap region plays mix vs s1 vs s2 for that region. **Per-region stream pairs — no cross-region speaker assembly** (that's what required ECAPA2 anchors + a whole campaign server-side; smallest credible ONNX embedder ≈ 25–38 MB; out of scope). Honest special case (~20 lines): exactly one overlap region → no permutation problem → offer assembled two-speaker playback.
5. **"Separate anyway" override**: user can force whole-clip separation and *hear* the phantom stream — an interactive illustration of M2. Stays within the B8 ruling (demonstrator; no numbers, no "demonstrated" claim).

Why this scope wins: it is the only option that *makes* the thesis claim rather than dodging it (separation is valuable **because it is routed**); routing is a compute **saving** (~5× fewer separator inferences on a realistic clip; a fully-overlapped clip degrades gracefully to today's behavior); it adds exactly **one 3 MB model**; and the null result ("no overlap found") becomes informative instead of embarrassing.

## 3. Model shopping list

| Item | Source | Size | License |
|---|---|---|---|
| pyannote-segmentation-3.0 **fp16** ONNX | `huggingface.co/onnx-community/pyannote-segmentation-3.0` → `onnx/model_fp16.onnx` | 2.86 MiB | MIT (CNRS) — ungated mirror; ship notice + cite Plaquet & Bredin, Interspeech 2023 |
| (alt., fp32 + LICENSE file) | `huggingface.co/csukuangfj/sherpa-onnx-pyannote-segmentation-3-0` | 5.99 MB | MIT |
| Timeline UI | wavesurfer.js v7 (+Regions/Timeline) or hand-rolled SVG | ~50 KB | BSD-3 |
| ORT-Web | pin (currently 1.20.1 via CDN); self-host `.wasm` for true offline | 11.2 MB | MIT |

**Do not buy**: silero-vad (redundant — powerset covers speech/non-speech), any speaker embedder, sherpa-onnx WASM diarization bundle (63.6 MiB), Sortformer (no clean export), **int8** pyannote-seg (+32 % relative overlap-frame shift vs fp32, zero WASM speedup — fp16 = 99.99 % argmax agreement with fp32).

## 4. Which separator ships — gated on B9

The narrative "we deploy the component our evaluation proved valuable" only holds with the **deployed** checkpoint: MossFormer2-matched-128k (8 kHz — drops into the existing chunker; ~27 MB INT8 ballpark). Its ONNX exportability is unknown (vendored rotary/token-shift; static 4 s shapes acceptable) → **the B9 export spike decides**; fallback SepFormer-128k (proven path). Model-dropdown copy depends on this — run the spike before UI work.

## 5. Plumbing and content

- **Sample rates**: OSD needs 16 kHz; separator needs 8 kHz. Move `AudioContext` to 16 kHz (also safer on iOS Safari than the current 8 kHz request) and decimate 2:1 per overlap region. ~0.5–1 d; touches every audio path in `app.js`.
- **Example clips, two classes required**: (i) fully-overlapped PolSESS-style clip (separator showcase; OSD will correctly mark ~100 % overlap); (ii) **mostly-solo conversation with overlap bursts** — without it the routing story is invisible. Licensing question: can CLARIN fragments go on a public page, or self-record? Pre-check (~30 min): pyannote-seg expects true 16 kHz — verify behavior on 8 kHz-bandwidth audio upsampled to 16 kHz before committing.
- Hosting: fully client-side → GitHub Pages, permanent QR-able artifact.

## 6. Thesis notes

- Prior-art searches found **no public browser-side speech-separation demo** (Transformers.js #788 unanswered; sherpa-onnx WASM separation is music-only) and nothing that visualizes "separator ran here / passed through here" — both defensible novelty remarks for the demonstrator section.
- Placement: deployment/demonstrator section (ch7 or appendix — TBD with Road 1). A demonstrator, not an experiment: stays out of results tables. B9's ladder is the *measured* artifact.

## 7. Effort

~5 d: 1 d sample-rate plumbing · 1 d OSD + powerset decode + routing port (~200 lines JS) · 1.5 d timeline UI · 0.5 d override + copy · 1 d example clips + phone testing. B9 spike (MF2 export) = separate ~1 d, can run first/parallel.

## 8. Open questions for the author

1. **Accept the OSD-gated scope** (vs. keeping whole-clip separation-only with honest framing)?
2. **"Separate anyway" override** — keep? (Most likely element to be second-guessed: a button whose purpose is to sound bad. Recommendation: keep; it makes M2 audible and interactive.)
3. Per-region presentation + single-overlap special case OK (no full-length per-speaker streams)?
4. Which separator ships — pending B9 spike result (MF2 vs SepFormer-128k fallback).
5. Example clips: CLARIN publishable, or self-recorded/PolSESS-derived?
6. Routing constants coupled verbatim to the server's (recommendation: yes, advertise it) — or tuned for demo aesthetics?
