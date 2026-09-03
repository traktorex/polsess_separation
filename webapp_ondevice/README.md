# `webapp_ondevice/` — Road 2, the on-device demo

A single static page that runs the whole separation decision **in the browser**:
overlap detection (OSD) → routing → separation of the routed regions only. No
server, no upload, no CDN — after the models are cached the page works offline.

It exists to make one thesis argument visible: the separator is trained on 100 %
overlapped audio, real conversation is 3–7 % overlapped, so a deployment must
*decide where to run the separator* instead of feeding it everything. The page
shows that decision (which regions were routed, how many separator calls they
cost) and lets you listen to the result. It is a **demonstrator, not an
experiment**: it reports no quality metrics and must not be used to compare
models.

## Layout

```
webapp_ondevice/
  site/            what gets published — the entire deployable
    index.html app.css js/       app + engine modules (no build step, ES modules)
    examples/        the two example clips + manifest.json
    models/          GITIGNORED, must ship: 3 .onnx files (79 MB)
    vendor/ort/      GITIGNORED, must ship: onnxruntime-web 1.23.2 (12 MB)
  build/           how models/ and vendor/ were produced — NOTES.md is the provenance record
  reference/       osd_reference.py (Python spec of OSD+routing) + parity vectors
  devcheck/        headless-Chromium harness: serve.py + check_page.py (HARNESS.md)
  design/          accepted design spec + mockup the app is screenshot-compared against
  test.html/.js    engine parity proof page (dev-only, deliberately outside site/)
```

`site/js/` splits in two: the **engine** (`osd.js`, `routing.js`, `separate.js`,
`resample.js`, `wav.js`, `ortEnv.js`) — covered by `test.html` against the
Python reference vectors — and the **app** (`app.js`, `ui.js`, `pipeline.js`,
`state.js`, `player.js`, `models.js`, `timeline.js`, `theme.js`, `format.js`,
`debug.js`). Changing an engine module means re-running `test.html`.

## Regenerating the assets

Everything under `site/models/` and `site/vendor/` is generated or downloaded
and git-ignored. `build/NOTES.md` carries the exact commands, URLs, revisions,
sha256s and licences; the short version:

```bash
CUDA_VISIBLE_DEVICES="" venv/bin/python webapp_ondevice/build/export_separators.py   # separators -> INT8 ONNX
CUDA_VISIBLE_DEVICES="" venv/bin/python webapp_ondevice/build/verify_models.py       # parity vs the eager checkpoints
# pyannote fp16 ONNX + onnxruntime-web tarball: two curl recipes in build/NOTES.md §2, §3
```

## Dev workflow

Serve the **repo subdirectory root** (not `site/`) — `test.html` needs both
`site/` and `reference/vectors/`:

```bash
python3 webapp_ondevice/devcheck/serve.py webapp_ondevice --port 8123
#   app          http://127.0.0.1:8123/site/index.html
#   parity page  http://127.0.0.1:8123/test.html
```

Screenshot and probe with the harness (`devcheck/HARNESS.md` is the full
reference; exit 0 = no console errors, no page errors, no failed requests):

```bash
~/playwright_venv/bin/python webapp_ondevice/devcheck/check_page.py \
    http://127.0.0.1:8123/site/index.html --hash "debug&state=S6&clip=A" \
    --shot /tmp/s6.png --full-page
```

`#debug` loads `js/debug.js`: a state switcher that paints all ten UI states
from synthetic data (no models, no GPU, instant) plus
`window.__debug.loadFromUrl(url, {model})`, which runs the **real** pipeline on
any audio URL and resolves with what the page ended up showing. That hook is the
E2E smoke test — one per example clip, both must exit 0:

```bash
~/playwright_venv/bin/python webapp_ondevice/devcheck/check_page.py \
    http://127.0.0.1:8123/site/index.html --hash "debug" --wait-selector "#devbar" \
    --js "window.__debug.loadFromUrl('./examples/rozmowa_przeplot.wav', {model:'sepformer'})"

~/playwright_venv/bin/python webapp_ondevice/devcheck/check_page.py \
    http://127.0.0.1:8123/site/index.html --hash "debug" --wait-selector "#devbar" \
    --js "window.__debug.loadFromUrl('./examples/pelne_nakladanie.wav', {model:'sepformer'})"
```

Engine regression guard (~7 min: it runs MossFormer2 four times in WASM):

```bash
~/playwright_venv/bin/python webapp_ondevice/devcheck/check_page.py \
    http://127.0.0.1:8123/test.html --js "window.__done"
```

## Deploying

Publish the **`site/` subtree as-is** to GitHub Pages — it is self-contained and
references nothing above itself. The catch: `site/models/` and `site/vendor/`
are git-ignored here (they would bloat the thesis repo) but **must be uploaded**.

* total ≈ 93 MB, largest single file 48 MB — under the 100 MB per-file limit, so
  plain Pages serves it with no LFS and no chunking;
* suggested: a `gh-pages` branch or a separate pages repo where those two
  directories are tracked (author's choice — nothing in the app depends on it);
* do **not** enable COOP/COEP expectations: Pages cannot send those headers, so
  `SharedArrayBuffer` is unavailable and ORT stays single-threaded. The app
  already pins `numThreads = 1`.

## Example clips

Both live in `site/examples/` with `manifest.json` (an empty `clips` list is a
supported state — the page then shows „Przykłady w przygotowaniu"). Each ships
at its native sample rate, unnormalised: the app applies one playback gain per
listening group, so touching the audio would only cost honesty.

| clip | file | source | routing (measured in-browser) |
|---|---|---|---|
| Rozmowa z przeplotem | `rozmowa_przeplot.wav` · 45,5 s · 16 kHz | CLARIN-PL conversational recording, fragment `85cb678a__seg00`, 33,7–79,2 s | 3 routed regions, 4,4 s = 10 % of the clip, 3 separator calls |
| Pełne nakładanie | `pelne_nakladanie.wav` · 4,0 s · 8 kHz native | PolSESS `C_final_128_v2/test/mix`, id `uafytqfgcnn89s3j-…` (mixture built on CLARIN-EMU speech) | 1 routed region covering 97 % of the clip, 1 separator call |

Selected 2026-08-05 from a 147-recording / 260-mixture OSD sweep; every
candidate was validated by running `reference/osd_reference.py` **on the
extracted file**, because a PolSESS mixture does not automatically read as
overlap (`reference/README.md` §6).

The two clips are complements, and the second one matters precisely because it
does *not* flatter the gate: clip A shows the routing win, clip B is a
fully-overlapped 4 s item where the gate honestly saves nothing (`~1,0×`).

> Note on the „× fewer computations" tile: the page's whole-clip baseline counts
> windows of its **own** chunker (4 s window, 2 s hop — what the „Uruchom
> separację całości" override actually costs), so clip A reads `~7,3×`
> (3 calls vs 22). Counting disjoint 4 s windows instead — as
> `reference/osd_reference.py` does — the same clip is 3 vs 12 = 4,0×. Same
> routing, two accounting conventions; the page uses the one that matches the
> button next to it.

## Honest numbers

Measured 2026-08-05, headless Chromium, **single-threaded WASM** on the WSL2
box (`test.html` §5; RTF = seconds of compute per second of audio, so > 1 is
slower than real time):

| artifact | size | WASM RTF |
|---|---:|---:|
| `pyannote_seg3_fp16.onnx` (OSD) | 2,9 MB | ≈ 0,01 — 0,10 s per 10 s window, never the bottleneck |
| `sepformer_128k_int8.onnx` | 30,2 MB | **≈ 1,5** (gate 1,46 · re-run 1,62) |
| `mf2_128k_int8.onnx` (MossFormer2-matched, e46) | 45,8 MB | **≈ 22,5** (gate 22,4 · re-run 22,5) — ~90 s per 4 s region |
| `ort-wasm-simd-threaded.wasm` (runtime) | 11,4 MB | — |

SepFormer is therefore the default and MossFormer2 is an explicit „best quality,
slow" choice, flagged as such in the model picker. Multi-threaded WASM would be
~4× faster, but GitHub Pages cannot serve the headers it requires.

Fidelity of the shipped INT8 exports vs their eager checkpoints (`build/NOTES.md`):
MossFormer2 e46 corr 0,9995 / 30,3 dB, SepFormer 0,99983 / 34,7 dB. OSD fp16 vs
fp32: 99,93–100 % frame agreement. Bare argmax over the powerset head recovers
~87 % region IoU against the full pyannote-3.1 overlap timeline
(`reference/README.md` §5) — the fidelity claim the demo rests on.
