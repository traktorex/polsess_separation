# `webapp_ondevice/build/` — asset provenance and how to regenerate

Everything under `webapp_ondevice/site/models/` and `webapp_ondevice/site/vendor/`
is **generated or downloaded** and is git-ignored. This file is the record of
where each byte came from. Built 2026-08-05 on the WSL2 training box
(torch 2.8.0+cu128, onnx 1.20.1, onnxruntime 1.23.2, CPU-only — the GPU was
never touched).

```
webapp_ondevice/
  build/                     <- tracked: scripts + this file
    export_separators.py       separator ONNX export + INT8 (run me)
    verify_models.py           loads every shipping artifact, checks parity
    NOTES.md                   you are here
    pyannote_seg3_config.json               } downloaded metadata, tracked
    pyannote_seg3_preprocessor_config.json  } (frame geometry lives here)
    _fp32/                   <- IGNORED: fp32 intermediates / reference models
  reference/                 <- tracked: OSD Python reference + parity vectors
  site/models/               <- IGNORED: the three shipping models
  site/vendor/ort/           <- IGNORED: self-hosted ONNX Runtime Web
```

---

## 1. Separators (generated)

| file | MB | recipe | source checkpoint | `val_sisdr` |
|---|---:|---|---|---|
| `site/models/mf2_128k_int8.onnx` | 45.79 | INT8 MatMul-only | `checkpoints/mossformer2/SB/mossformer2_matched_128k_final_42_e46/mossformer2_SB_best_e46.pt` | 17.044 |
| `site/models/sepformer_128k_int8.onnx` | 30.22 | INT8 MatMul-only | `checkpoints/sepformer/SB/128_run/sepformer_SB_best_128k_e41.pt` | 15.829 |

Regenerate (≈2 min, CPU):

```bash
CUDA_VISIBLE_DEVICES="" venv/bin/python webapp_ondevice/build/export_separators.py
CUDA_VISIBLE_DEVICES="" venv/bin/python webapp_ondevice/build/verify_models.py
```

Contract: `mixture` float32 `[1, 32000]` → `separated` float32 `[1, 2, 32000]`,
4 s @ 8 kHz, opset 17, **static shapes**. Recipes, the reason MossFormer2 needs
a hand-tuned INT8 recipe, and the `QuantType.QInt8` trap are documented in the
export script's docstring; the measurements behind them are the ones recorded in
this file.

Verified 2026-08-05 (`verify_models.py` ALL CHECKS PASSED): MF2 e46 MatMul-only
reproduces its eager checkpoint at corr 0.9995 / 30.28 dB worst-case; SepFormer
at 0.99983 / 34.68 dB.

### Checkpoint choice — decision 2026-08-05: ship e46

The demo ships **e46** — the exact checkpoint the ASR pipeline deploys
(`asr_pipeline/config.py:353-355` and both shipped configs), which is what "the
deployed separator" means everywhere else in the thesis (epoch 45, `val_sisdr`
17.044 vs e23's 16.366). The first export pass measured e23; re-running the same
parity harness on e46 exposed a real recipe interaction:

| candidate | recipe | MB | self-fidelity (corr / SI-SDR vs eager) |
|---|---|---:|---|
| e23 | "safe" | 33.64 | 0.99937 / 28.98 dB |
| e46 | "safe" | 33.64 | 0.99750 / **22.99 dB — rejected, audible risk** |
| **e46 (ships)** | **MatMul-only** | **45.79** | **0.99953 / 30.28 dB** |

The "safe" recipe that was transparent on e23 degrades on e46 (23 more epochs →
sharper activations, consistent with MF2's known fp16 fragility). MatMul-only
restores transparent fidelity at +12 MB — an acceptable one-time download for
the quality-default model; SepFormer INT8 (30.22 MB) remains the light/fast
option. The e23 "safe" row above is the historical evidence for that recipe.

## 2. pyannote-segmentation-3.0 ONNX (downloaded)

| file | bytes | sha256 |
|---|---:|---|
| `site/models/pyannote_seg3_fp16.onnx` | 3,000,918 (2.86 MiB) | `f3dba0c91270b923e9ce66e4cef123820cb73c9b07e47e03a3d7d4267e4fbec4` |
| `build/_fp32/pyannote_seg3_fp32.onnx` (reference only) | 5,986,908 (5.71 MiB) | `057ee564753071c0b09b5b611648b50ac188d50846bff5f01e9f7bbf1591ea25` |

* **Repo**: <https://huggingface.co/onnx-community/pyannote-segmentation-3.0>
* **Revision**: `733a93b6473d019a773298e08cefa686894b1854` (lastModified 2025-07-08)
* **Files**: `onnx/model_fp16.onnx`, `onnx/model.onnx`, `config.json`,
  `preprocessor_config.json`, `README.md` (model card, in `_fp32/`)
* **License**: **MIT** (declared on both the ONNX mirror and the upstream
  `pyannote/segmentation-3.0`). Attribution owed to **Plaquet & Bredin,
  "Powerset multi-class cross entropy loss for neural speaker diarization",
  Interspeech 2023** — the demo page must carry the notice + citation.
* Exact URL form used:
  `https://huggingface.co/onnx-community/pyannote-segmentation-3.0/resolve/733a93b6473d019a773298e08cefa686894b1854/onnx/model_fp16.onnx`

```bash
REV=733a93b6473d019a773298e08cefa686894b1854
B=https://huggingface.co/onnx-community/pyannote-segmentation-3.0/resolve/$REV
curl -sL "$B/onnx/model_fp16.onnx" -o webapp_ondevice/site/models/pyannote_seg3_fp16.onnx
curl -sL "$B/onnx/model.onnx"      -o webapp_ondevice/build/_fp32/pyannote_seg3_fp32.onnx
```

Measured here (`reference/osd_reference.py`): fp16 vs fp32 per-frame argmax
agreement **99.93 – 100 %**, routed-region IoU 0.967 – 1.000, max |logit| delta
0.53. The plan's "ship fp16, not int8" call holds.

### Warning: ORT's default optimization level segfaults on this file

`onnxruntime.InferenceSession(fp16_path)` — i.e. the **default**
`ORT_ENABLE_ALL` — **segfaults the process** at session creation on ORT 1.23.2
CPU EP. No exception, no message; the interpreter dies. `ORT_ENABLE_EXTENDED`
(one level down) loads and runs correctly, as does the fp32 file at any level.
The reference implementation therefore always sets EXTENDED, and the JS app
should pass `graphOptimizationLevel: 'extended'` to
`ort.InferenceSession.create`. The offending pass is almost certainly the
x86-only NCHWc layout transformer, which the WASM build should not contain — but
the cost of passing the flag is zero and the cost of being wrong is a dead tab,
so pass it, and have `devcheck` prove it either way.

## 3. ONNX Runtime Web (vendored, no CDN)

* **Version `1.23.2`** — chosen to match the `onnxruntime` **1.23.2** in
  `venv/`, i.e. the exact runtime that validated every model above. npm's
  `latest` is 1.27.0; matching the validation runtime beats being newest here.
  (Requirement was ≥ 1.17 for native `LayerNormalization` at opset 17 — the
  MossFormer2 graph has 45 of them.)
* **Source**: `https://registry.npmjs.org/onnxruntime-web/-/onnxruntime-web-1.23.2.tgz`
  (sha256 `0b8707d7efab9a2bea63564d66cad79e897cfd0bd627e5ae000488ffc1c45c7d`)
* **License**: MIT (Microsoft) — `site/vendor/ort/LICENSE`, fetched from
  `raw.githubusercontent.com/microsoft/onnxruntime/v1.23.2/LICENSE`.

Vendored set (12 MB total):

| file | KB | what |
|---|---:|---|
| `ort.wasm.bundle.min.mjs` | 67 | **recommended entry point** — ESM, WASM EP only, wasm *loader* inlined (fetches only the `.wasm`) |
| `ort.wasm.min.mjs` | 49 | same, non-bundled: also fetches `ort-wasm-simd-threaded.mjs` at runtime |
| `ort.wasm.min.js` | 49 | classic-script (non-module) fallback |
| `ort-wasm-simd-threaded.mjs` | 20 | wasm loader for the non-bundled build |
| `ort-wasm-simd-threaded.wasm` | 11,626 | **the runtime** |
| `types.d.ts`, `LICENSE` | – | provenance |

```bash
curl -sL https://registry.npmjs.org/onnxruntime-web/-/onnxruntime-web-1.23.2.tgz | tar xz
cp package/dist/{ort.wasm.bundle.min.mjs,ort.wasm.min.mjs,ort.wasm.min.js,ort-wasm-simd-threaded.mjs,ort-wasm-simd-threaded.wasm} \
   webapp_ondevice/site/vendor/ort/
```

Notes for the app:

* **There is only one CPU `.wasm` in this version line.** The separate
  single-threaded / non-SIMD builds were removed years ago;
  `ort-wasm-simd-threaded.wasm` serves both — set `ort.env.wasm.numThreads = 1`
  to run it single-threaded. SIMD is assumed available (every browser since
  ~2021).
* **GitHub Pages cannot serve COOP/COEP**, so `SharedArrayBuffer` is
  unavailable and multi-threading will not work there: pin
  `ort.env.wasm.numThreads = 1` (or feature-detect `crossOriginIsolated`). The
  4-thread RTFs measured in the 2026-08-05 export experiment (MF2 2.28,
  SepFormer 0.17) are therefore an *optimistic* bound for the deployed page.
* Point the runtime at the vendored binary:
  `ort.env.wasm.wasmPaths = new URL('./vendor/ort/', document.baseURI).href`.
* **WebGPU is deliberately not vendored.** It needs
  `ort.webgpu.bundle.min.mjs` + `ort-wasm-simd-threaded.jsep.wasm` (+24 MB of
  deploy), and the 2026-08-05 export experiment flagged MossFormer2's 133
  `Einsum` + 24 `InstanceNormalization` nodes as thin/partial on the JSEP backend. If someone
  wants to try it for SepFormer (a Gemm/Softmax/LayerNorm graph, far more
  WebGPU-friendly), the two files are one `tar x` away from the same tarball.
* **Do not ship a pre-optimized MossFormer2 `.onnx`**: ORT's own offline
  optimization materializes the rotary sin/cos tables and grows the fp32 file
  104 → 152 MB (measured 2026-08-05). Runtime folding at session init is fine and
  cheap.

## 4. What is deliberately absent

* No CDN URLs anywhere — `site/` must work from `file://`-ish static hosting
  with zero third-party requests.
* No `int8` pyannote (plan §3: +32 % relative overlap-frame shift, zero WASM
  speedup), no speaker embedder, no silero-VAD (the powerset head already
  gives non-speech).
