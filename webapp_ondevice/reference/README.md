# OSD reference — what the JS port must reproduce

`osd_reference.py` is the executable specification of the Road-2 demo's
**overlap detection + routing** stage. The browser reimplements it in
JavaScript on ONNX Runtime Web; `vectors/` is what that port is tested against.

Run everything with the repo venv, CPU-only:

```bash
V=venv/bin/python
$V webapp_ondevice/reference/osd_reference.py run <audio.wav> [--max-seconds N]
$V webapp_ondevice/reference/osd_reference.py validate    # 3 CLARIN recordings vs pyannote-3.1
$V webapp_ondevice/reference/osd_reference.py precheck     # the 8 kHz-bandwidth risk item
$V webapp_ondevice/reference/osd_reference.py vectors      # regenerate vectors/
```

`validate` on 950 s of audio takes ~4 s. The whole OSD stage is ~0.004 × RT on
20 CPU cores; it will not be the demo's bottleneck.

---

## 1. Model I/O contract

`site/models/pyannote_seg3_fp16.onnx` (onnx-community mirror, rev `733a93b`,
MIT — provenance in `../build/NOTES.md`).

```
input   "input_values"  float32  [batch, channels, samples]   16 kHz mono
output  "logits"        float32  [batch, frames, 7]
```

* **Both boundary tensors are float32 even in the fp16 file.** The conversion
  wrapped the graph in Cast nodes, so JS feeds and reads plain `Float32Array` —
  there is no float16 packing to write.
* `logits` are **log**-probabilities (the graph ends in `LogSoftmax`): all
  values ≤ 0, `exp()` gives a distribution. Irrelevant for argmax, relevant if
  the UI ever wants a confidence.
* The reference always calls the model with a **fixed** `[1, 1, 160000]`
  (10 s) input. The axes are dynamic in the file, but 10 s is the training
  window, and a fixed shape avoids per-shape re-specialisation in ORT-Web.
* **Create the session with `graphOptimizationLevel: 'extended'`.** ORT's
  default (`all`) *segfaults* on this file in Python — see `../build/NOTES.md`
  §2. Cheap insurance in the browser.

### Frame geometry

From the model's own `preprocessor_config.json`
(`{"offset": 990, "step": 270, "sampling_rate": 16000}`), confirmed empirically
(160000 samples → 589 frames, 300 s → 17775 frames):

```
frame step   270 samples = 16.875 ms      <- the plan's 16.875 ms, confirmed
frame width  990 samples = 61.875 ms
frames(T)    floor((T - 990) / 270) + 1   (0 if T < 990)
frame k      covers samples [k*270, k*270+990); middle = (k*270 + 495)/16000 s
```

## 2. Decode table — 7 powerset classes → 3 labels

Class order is fixed by the model card's `id2label` (mirrored in
`../build/pyannote_seg3_config.json`):

| class | id2label | speakers | our label |
|---:|---|---:|---|
| 0 | `NO_SPEAKER` | 0 | `nonspeech` (0) |
| 1 | `SPEAKER_1` | 1 | `solo` (1) |
| 2 | `SPEAKER_2` | 1 | `solo` (1) |
| 3 | `SPEAKER_3` | 1 | `solo` (1) |
| 4 | `SPEAKERS_1_AND_2` | 2 | `overlap` (2) |
| 5 | `SPEAKERS_1_AND_3` | 2 | `overlap` (2) |
| 6 | `SPEAKERS_2_AND_3` | 2 | `overlap` (2) |

Decode is `argmax` over the 7 classes, then this table. **We only ever use the
speaker COUNT, never the identity** — that is what removes clustering,
embeddings and per-window permutation alignment from the demo.

## 3. Sliding-window strategy: 10 s windows, 50 % overlap, centre-crop

```
WINDOW = 160000 samples (10 s)     window w starts at sample w * HOP
HOP    =  79920 samples (296 frames = 4.995 s)
each window yields 589 frames; window w's local frame i is GLOBAL frame w*296 + i
keep local frames [146, 442)  — the central 296 —
    except window 0, which also supplies [0, 146),
    and the last window, which supplies [146, 589).
tail is zero-padded to a full window; global frames >= frames(T) are discarded.
```

Why:

* **Deterministic and stateless.** Every output frame comes from exactly one
  window: a `for` loop plus index arithmetic, no aggregation weights.
* **No permutation problem.** Speaker indices are local to a window, but the
  speaker *count* is not, so frames from different windows are directly
  comparable. (pyannote's own `Inference` has to align permutations before it
  can overlap-add.)
* **The hop is a whole number of frames.** 79920 = 296 × 270, so every window's
  frame grid lands on the global grid. A "nice" 5 s hop (80000 samples) is not
  a multiple of 270 and would smear the grid by a fraction of a frame per
  window — a silent, cumulative off-by-N.
* **Centre-cropping drops the edge frames**, where the model has < 2.5 s of
  context on one side. Frames at the true start/end of the recording are kept,
  because there the window edge *is* the signal edge.
* Measured cost of getting this wrong: naive non-overlapping windows agree with
  centre-crop on only **96.8 %** of frames over 300 s of CLARIN audio and find
  **14 routed regions instead of 16** (region IoU 0.83). On the two 20 s
  vectors the two strategies agree to 1–2 frames, so a JS port that gets this
  wrong will still pass the vectors — that is why the strategy is spelled out
  here rather than left to inference.

## 4. Routing constants (ported from the server, with provenance)

| constant | value | source |
|---|---:|---|
| `min_overlap_dur` | 0.20 s | `asr_pipeline/config.py:244` (`RoutingConfig`) |
| `merge_gap` | 0.50 s | `asr_pipeline/config.py:245` (`RoutingConfig`) |
| `context_window_mode` | `expand_to_chunk` | `asr_pipeline/config.py:372` (`SeparationConfig`) |
| `training_chunk_length_s` | 4.0 s | `asr_pipeline/config.py:360` |
| `min_fragment_length_s` | 4.0 s | `asr_pipeline/config.py:382` |
| `context_pad_seconds` | 1.0 s | `asr_pipeline/config.py:373` — **`fixed_pad` only** |

Order of operations, exactly as in `asr_pipeline/stages/routing.py`:

1. contiguous `overlap` frame runs → raw intervals;
2. **drop** intervals shorter than `min_overlap_dur` (`>=`, so 0.20 s survives);
3. **merge** intervals whose gap is `< merge_gap` (strict `<`, as the server);
4. **context-pad** each surviving region (stage 3b, `separation.py`).

Filter before merge — the server's order. Reversing it would resurrect
sub-threshold overlaps by gluing them to a neighbour.

### The `context_pad = 1.0 s` in the plan doc is the wrong knob

`docs/fable_plans/frontends_road2_ondevice.md` §2 says "`context_pad = 1.0 s`".
That constant exists, but it belongs to `context_window_mode: fixed_pad`, which
neither shipped config selects. Both `asr_pipeline/configs/default.yaml:43-45`
and `sweep_best_e31_refineplus.yaml:80-82` use **`expand_to_chunk`**: every
overlap region is padded out to a **4.0 s** window, asymmetrically when it
bumps into a recording boundary (`_boundary_aware_pad`, ported verbatim).

Use `expand_to_chunk` in the demo. It is what the server does, and 4.0 s
@ 8 kHz is exactly the 32000-sample window the separator ONNX accepts — one
routed region becomes exactly one separator call. Both modes are implemented
and both are in the vectors; they coincide unless a routed region is longer
than 2.0 s.

**Consequence the app must handle:** a routed region *longer* than 4.0 s is
returned unpadded (`_boundary_aware_pad` no-ops when the target is already
reached) and must be chunked + overlap-added by the caller. The dense vector
has a 3.63 s region; real conversation produces longer ones.

## 5. Validation on real audio

`validate`, full recordings, against the pyannote-**3.1** overlap timeline the
server itself produced (`~/datasets/clarin_all_2speakers/diarization/<id>.json`,
routed through the same filter+merge for a like-for-like read):

| recording | dur | our routed regions | pyannote-3.1 | region IoU | ours ∩ theirs |
|---|---:|---|---|---:|---|
| `442dd69e` | 950 s | 84 (59.9 s, 6.3 %) | 90 (59.4 s) | 0.870 | 83/90 · 81/84 |
| `6d88daa2` | 950 s | 38 (29.7 s, 3.1 %) | 34 (29.2 s) | 0.878 | 34/34 · 35/38 |
| `649991bc` | 1622 s | 152 (116.1 s, 7.2 %) | 146 (113.0 s) | 0.886 | 143/146 · 144/152 |

Reading: a bare argmax over the segmentation model's powerset head, with no
clustering and no embeddings, recovers **~87 % IoU** and **92–100 % of the
regions** of the full pyannote-3.1 pipeline's overlap timeline. Disagreements
are boundary jitter and a handful of short regions near the 0.20 s threshold,
not structural — the two timelines mark the same bursts. That is the fidelity
claim the demo needs, and it is honest.

Also: **overlap is 3–7 % of a real Polish conversation.** That is the whole
argument for OSD-gated separation — ~15–30× fewer separator calls than
whole-clip processing, and the routing decision is visible instead of assumed.

## 6. ⚠ 8 kHz-bandwidth pre-check — VERDICT: **risk retired (with a caveat)**

Plan §5 flagged: "pyannote-seg expects true 16 kHz — verify behavior on 8 kHz
audio upsampled to 16 kHz before committing", because PolSESS-derived clips are
natively 8 kHz. Controlled experiment (`precheck`): take real 16 kHz audio,
decimate to 8 kHz and upsample back (soxr VHQ), changing *only* the bandwidth,
and re-run OSD.

| audio | frame agreement | overlap frames kept | routed-region IoU | region recall |
|---|---:|---:|---:|---:|
| `442dd69e` (950 s, Polish conv.) | 92.98 % | 80.3 % | 0.736 | 73/84 |
| `6d88daa2` (950 s, Polish conv.) | 93.76 % | 81.0 % | 0.715 | 33/38 |
| LibriCSS `OV40_session1_seg6` (17 s, dense overlap) | 87.41 % | 79.6 % | 0.728 | 1/1 |

**Verdict: usable. Ship it.** Losing the 4–8 kHz band costs ~20 % of overlap
*frames* and ~13 % of overlap *regions* — a real but graceful degradation, and
critically it is **conservative in the right direction**:

```
confusion, 442dd69e (rows = native 16 kHz, cols = 8 k→16 k)
              nonspeech    solo   overlap
  nonspeech      5553       259         1
       solo      2717     43863       255      <- 0.5 % false overlap
    overlap         7       710      2925      <- 19.5 % missed overlap
```

Almost nothing moves *into* `overlap` (1 frame from non-speech, 0.5 % of solo).
The failure mode on 8 kHz material is therefore **"a real overlap goes
unrouted"** — the user hears an unseparated overlap — and **not** "a solo region
is routed to the separator", which is the M2 phantom-stream embarrassment the
whole OSD-gated scope decision exists to avoid. The risk item can be closed.

Caveat for whoever builds the example clips: **do not assume a PolSESS mixture
reads as overlap.** Over 12 random `PolSESS_C_final_128_v2/test/mix` clips
(native 8 kHz, upsampled) the mean overlap fraction was 0.28, and 4 of 12 read
as ~100 % non-speech or ~100 % solo. That is *not* the bandwidth — level is
fine (RMS 0.09–0.15) and peak-normalising changes nothing. It is content: those
mixtures carry loud scene/event layers and the two speakers are not always
simultaneously active. **Validate every candidate showcase clip through
`osd_reference.py run` before putting it in the demo**, or the "fully
overlapped" example will render as a flat solo bar.

## 7. Parity vectors (`vectors/`)

1.3 MB total, tracked in git. Regenerate with `... osd_reference.py vectors`
(deterministic — re-running reproduces every field byte-for-byte).

| vector | audio | what it exercises |
|---|---|---|
| `clarin_442dd69e_sparse` | 20 s CLARIN (Polish, 442dd69e @ 865 s), 625 KB | the **filter**: 8 raw overlaps → one 0.084 s interval dropped → 7 routed; no merge fires |
| `libricss_ov40_dense` | 20 s LibriCSS 2-spk OV40 segment, 620 KB | the **merge**: 4 raw → 3 with gaps 0.34 s / 0.47 s collapse into one 3.63 s region → 2 routed; the only case where `expand_to_chunk` and `fixed_pad` differ |

Each `<name>.json` carries:

| field | for testing |
|---|---|
| `contract`, `num_samples`, `num_frames` | frame-count formula and window constants |
| `powerset_argmax` (int[frames]) | raw 7-class argmax — catches a wrong decode table |
| `labels` (int[frames], 0/1/2) | the 3-class labels |
| `first_window_logits` — frames 0 / 100 / 200, all 7 classes, 4 dp | **tensor plumbing**: check these before trusting anything downstream. Wrong channel layout, wrong normalisation or a missed `[1,1,T]` reshape shows up here, not in the argmax. |
| `raw_overlaps` | frames → intervals |
| `routed` | filter + merge |
| `padded_expand_to_chunk`, `padded_fixed_pad_1s` | context padding, both modes |
| `constants` | the routing constants in force |

**Suggested JS tolerances.** `labels` / `powerset_argmax`: expect ≥ 99.9 %
agreement, and treat any interval-level difference as a failure. Logits: fp16
plus a different WASM kernel set means bit-equality is not on offer — compare
with `atol ≈ 0.05` (measured fp16-vs-fp32 max |Δlogit| on the same graph was
0.53 at 99.93 % argmax agreement, so a JS-vs-Python delta above ~0.1 means a
real bug, not numerics). Interval endpoints are computed from integer frame
indices and must match to ~1e-6.

**Interval convention** (`frames_to_intervals`): a run of frames k0..k1 becomes
`[middle(k0) - step/2, middle(k1) + step/2]`, so N frames span exactly
N × 16.875 ms. pyannote timestamps runs at frame middles; this is that, widened
by half a frame each side so that durations are exact — `min_overlap_dur` is a
duration test and losing 16.9 ms of it to a convention would be a silent
behavioural difference from the server.

### Provenance / licensing of the vector audio

* `clarin_442dd69e_sparse.wav` — 20 s excerpt of
  `~/datasets/clarin_gotowy/gotowy/442dd69e.wav` (CLARIN-PL). Fine as a private
  test fixture in this repo; **the plan's open question about publishing CLARIN
  audio is unresolved**, so do not copy it into `site/` as an example clip
  without an answer.
* `libricss_ov40_dense.wav` — 20 s excerpt of LibriCSS
  (`~/datasets/LibriCSS_2spk/record/segments/OV40_session4_seg8.wav`),
  LibriSpeech-derived, CC BY 4.0 — publishable with attribution, and therefore
  the safe default if a vector ever has to ship publicly.
