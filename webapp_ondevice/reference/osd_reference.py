#!/usr/bin/env python3
"""Reference implementation of the Road-2 demo's overlap-detection + routing stage.

The browser app reimplements all of this in JavaScript on top of ONNX Runtime
Web. This file is the thing the JS port is tested against: it fixes the I/O
contract, the powerset decode table, the sliding-window strategy and the
routing constants, and it emits parity vectors (`vectors/`) that the JS port
must reproduce.

Everything here is CPU-only and deterministic.

===========================================================================
1. Model + I/O contract  (pyannote-segmentation-3.0, ONNX)
===========================================================================
File: `webapp_ondevice/site/models/pyannote_seg3_fp16.onnx`
(onnx-community/pyannote-segmentation-3.0, revision 733a93b, MIT —
provenance in `webapp_ondevice/build/NOTES.md`).

    input   "input_values"  float32  [batch, channels, samples]   16 kHz mono
    output  "logits"        float32  [batch, frames, 7]

Both boundary tensors are **float32 even in the fp16 file** — the fp16
conversion wraps the graph in Cast nodes, so JS feeds and reads plain
Float32Array. No manual float16 packing.

Frame geometry (from the model's own `preprocessor_config.json`:
`{"offset": 990, "step": 270, "sampling_rate": 16000}`, confirmed
empirically — see `frame_count()`):

    frame step      270 samples = 16.875 ms          <- the plan's 16.875 ms
    frame width     990 samples = 61.875 ms          (receptive field)
    frames(T)       floor((T - 990) / 270) + 1       (0 if T < 990)
    frame k covers  samples [k*270, k*270 + 990)
    frame k middle  (k*270 + 495) / 16000  seconds   (pyannote's convention)

`logits` are LOG-probabilities (the graph ends in LogSoftmax over the 7
classes), so they are all <= 0 and `exp()` gives a proper distribution.
For argmax decoding this is irrelevant — log is monotone — but it matters
if the app ever wants a confidence value.

===========================================================================
2. Powerset decode table  (7 classes -> 3 labels)
===========================================================================
seg-3.0 has a powerset head over max 3 simultaneous speakers. Class order is
fixed by the model card's `id2label`:

    0 NO_SPEAKER          -> 0 speakers -> NONSPEECH
    1 SPEAKER_1           -> 1 speaker  -> SOLO
    2 SPEAKER_2           -> 1 speaker  -> SOLO
    3 SPEAKER_3           -> 1 speaker  -> SOLO
    4 SPEAKERS_1_AND_2    -> 2 speakers -> OVERLAP
    5 SPEAKERS_1_AND_3    -> 2 speakers -> OVERLAP
    6 SPEAKERS_2_AND_3    -> 2 speakers -> OVERLAP

The demo only needs the speaker COUNT, never the identity. That is what
makes this cheap: mapping to a count is permutation-invariant, so the
per-window speaker-permutation alignment that pyannote's own pipeline needs
(and the clustering, and the embeddings) all disappear.

===========================================================================
3. Sliding window: 10 s windows, 50 % overlap, centre-crop stitching
===========================================================================
The model is trained on 10 s chunks. Longer input technically runs (the LSTM
takes any length) but is out of distribution, so we window.

Choice: fixed windows of 160000 samples (10 s) with a hop of **79920 samples
(296 frames = 4.995 s)**, keeping only the CENTRAL 296 frames of each
window's 589; the first window additionally supplies the leading 146 frames
and the last window supplies everything to the end.

Why this and not something else:

* **Deterministic and stateless** — every output frame comes from exactly one
  window. No aggregation weights, no hamming windows, no running state; the
  JS port is a `for` loop over windows plus index arithmetic.
* **No permutation problem** — speaker indices are local to a window, but we
  only ever use the speaker COUNT, so frames from different windows are
  directly comparable. (This is the trick that makes stitching trivial;
  pyannote's own `Inference` must align permutations before aggregating.)
* **Hop is a whole number of frames.** 79920 = 296 * 270, so every window's
  frame grid lands exactly on the global grid: window w's local frame i is
  global frame `w*296 + i`. A "nice" 5 s hop (80000 samples) is NOT a
  multiple of 270 and would smear the grid by a fraction of a frame per
  window — a silent, cumulative off-by-N.
* **Centre-cropping removes the edge frames**, where the model has < 2.5 s of
  context on one side and is measurably jumpier. The frames at the very start
  and end of the recording are kept from the first/last window because there
  the window edge IS the signal edge — no artificial truncation to hide.
* Cost is 2x inference on a 2.9 MB model: ~0.1 s of CPU per minute of audio.

Rejected: non-overlapping windows (simplest, but every 10 s there is a seam
where both sides of the boundary are edge frames) and pyannote's own
overlap-add with hamming weights (needs permutation alignment across windows
before the weights mean anything).

The tail is zero-padded to a full window and the frames past the true end of
the audio are dropped, so padding can never invent an overlap region.

===========================================================================
4. Routing port  (constants copied from the server, with citations)
===========================================================================
Stage 2 (`asr_pipeline/stages/routing.py`):
    min_overlap_dur = 0.20 s   asr_pipeline/config.py:244
    merge_gap       = 0.50 s   asr_pipeline/config.py:245
  -> drop overlap intervals shorter than min_overlap_dur, then merge
     intervals whose gap is strictly below merge_gap.

Stage 3b context padding — NOT in RoutingConfig; it lives in
`SeparationConfig` and is applied in `asr_pipeline/stages/separation.py`
(`_boundary_aware_pad`, `_window_expand_to_chunk`, `_window_fixed_pad`):
    context_window_mode   = "expand_to_chunk"  asr_pipeline/config.py:372
    training_chunk_length_s = 4.0              asr_pipeline/config.py:360
    min_fragment_length_s   = 4.0              asr_pipeline/config.py:382
    context_pad_seconds     = 1.0              asr_pipeline/config.py:373

NB the plan doc (`docs/fable_plans/frontends_road2_ondevice.md` §2) says
"context_pad = 1.0 s". That number is real but it is the `fixed_pad` knob,
which the shipped config does not use: `sweep_best_e31_refineplus.yaml:80-82`
and `default.yaml:43-45` both select `expand_to_chunk`, i.e. every overlap
region is padded out to a **4.0 s** window. Both modes are implemented below;
`expand_to_chunk` is the demo default because it is what the server does AND
because 4.0 s @ 8 kHz is exactly the 32000-sample window the separator ONNX
takes — one padded region becomes exactly one separator call.

===========================================================================
Usage
===========================================================================
    V=venv/bin/python
    $V webapp_ondevice/reference/osd_reference.py run   <audio.wav> [--json out.json]
    $V webapp_ondevice/reference/osd_reference.py validate            # 3 CLARIN recordings
    $V webapp_ondevice/reference/osd_reference.py precheck            # 8 kHz-bandwidth risk item
    $V webapp_ondevice/reference/osd_reference.py vectors             # write vectors/
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

APP_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = APP_DIR / "site" / "models" / "pyannote_seg3_fp16.onnx"
FP32_MODEL = APP_DIR / "build" / "_fp32" / "pyannote_seg3_fp32.onnx"
VECTORS_DIR = Path(__file__).resolve().parent / "vectors"

# --- model / frame geometry -------------------------------------------------
SAMPLE_RATE = 16_000
WINDOW_SAMPLES = 160_000          # 10 s — the model's training window
FRAME_STEP_SAMPLES = 270          # preprocessor_config.json "step"
FRAME_WIDTH_SAMPLES = 990         # preprocessor_config.json "offset"
FRAME_STEP_S = FRAME_STEP_SAMPLES / SAMPLE_RATE          # 0.016875 s
FRAME_WIDTH_S = FRAME_WIDTH_SAMPLES / SAMPLE_RATE        # 0.061875 s
FRAMES_PER_WINDOW = 589           # = frame_count(WINDOW_SAMPLES)
HOP_FRAMES = 296                  # half a window, rounded down to whole frames
HOP_SAMPLES = HOP_FRAMES * FRAME_STEP_SAMPLES            # 79920 = 4.995 s
KEEP_START = (FRAMES_PER_WINDOW - HOP_FRAMES) // 2       # 146
KEEP_END = KEEP_START + HOP_FRAMES                       # 442

# --- powerset decode --------------------------------------------------------
# index = powerset class, value = number of active speakers
POWERSET_SPEAKER_COUNT = (0, 1, 1, 1, 2, 2, 2)
NONSPEECH, SOLO, OVERLAP = 0, 1, 2
LABEL_NAMES = ("nonspeech", "solo", "overlap")

# --- routing constants (see module docstring §4 for file:line citations) ----
MIN_OVERLAP_DUR = 0.20
MERGE_GAP = 0.50
CONTEXT_WINDOW_MODE = "expand_to_chunk"   # "expand_to_chunk" | "fixed_pad" | "none"
TRAINING_CHUNK_LENGTH_S = 4.0
MIN_FRAGMENT_LENGTH_S = 4.0
CONTEXT_PAD_SECONDS = 1.0

Interval = Tuple[float, float]


# ===========================================================================
# model
# ===========================================================================
def load_session(model_path: Path = DEFAULT_MODEL):
    """Create the ORT CPU session.

    LANDMINE: ORT 1.23.2's default graph-optimization level (`ORT_ENABLE_ALL`)
    SEGFAULTS on this fp16 model at session creation — no exception, the
    process dies. `ORT_ENABLE_EXTENDED` (one level down, i.e. everything
    except the x86 NCHWc layout transformer) loads and runs fine, and so does
    the fp32 file at any level. The equivalent ORT-Web knob is
    `graphOptimizationLevel: 'extended'` in `InferenceSession.create`; the WASM
    build almost certainly does not contain the offending x86 transformer, but
    the app should pass it anyway — it costs nothing and the failure mode is a
    dead tab.
    """
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    return ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])


def frame_count(n_samples: int) -> int:
    """Number of output frames the model emits for `n_samples` input."""
    if n_samples < FRAME_WIDTH_SAMPLES:
        return 0
    return (n_samples - FRAME_WIDTH_SAMPLES) // FRAME_STEP_SAMPLES + 1


def frame_middle_s(k: int) -> float:
    """Centre of frame k in seconds (pyannote's `SlidingWindow[k].middle`)."""
    return (k * FRAME_STEP_SAMPLES + FRAME_WIDTH_SAMPLES / 2) / SAMPLE_RATE


def infer_logits(sess, wav: np.ndarray) -> np.ndarray:
    """Sliding-window inference over arbitrary-length mono 16 kHz audio.

    Returns `[num_frames, 7]` float32 log-probabilities on the global frame
    grid, `num_frames == frame_count(len(wav))`. See docstring §3.
    """
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    total_frames = frame_count(len(wav))
    if total_frames == 0:
        return np.zeros((0, 7), dtype=np.float32)

    n_windows = 1
    if len(wav) > WINDOW_SAMPLES:
        n_windows = int(np.ceil((len(wav) - WINDOW_SAMPLES) / HOP_SAMPLES)) + 1

    out = np.zeros((total_frames, 7), dtype=np.float32)
    written = np.zeros(total_frames, dtype=bool)
    for w in range(n_windows):
        start = w * HOP_SAMPLES
        chunk = wav[start : start + WINDOW_SAMPLES]
        if len(chunk) < WINDOW_SAMPLES:  # zero-pad the tail
            chunk = np.pad(chunk, (0, WINDOW_SAMPLES - len(chunk)))
        logits = sess.run(
            None, {"input_values": chunk.reshape(1, 1, WINDOW_SAMPLES)}
        )[0][0]  # [589, 7]

        lo = 0 if w == 0 else KEEP_START
        hi = FRAMES_PER_WINDOW if w == n_windows - 1 else KEEP_END
        for i in range(lo, hi):
            g = w * HOP_FRAMES + i
            if g >= total_frames:      # frames that only exist because of padding
                break
            out[g] = logits[i]
            written[g] = True
    assert written.all(), "sliding window left holes — check HOP/KEEP constants"
    return out


def decode_classes(logits: np.ndarray) -> np.ndarray:
    """Powerset argmax -> per-frame label in {NONSPEECH, SOLO, OVERLAP}."""
    powerset = logits.argmax(axis=-1)
    lut = np.array(POWERSET_SPEAKER_COUNT, dtype=np.int64)
    counts = lut[powerset]
    return np.clip(counts, 0, 2).astype(np.int64)


# ===========================================================================
# frames -> intervals -> routing
# ===========================================================================
def frames_to_intervals(labels: np.ndarray, target: int) -> List[Interval]:
    """Contiguous runs of `target` -> [start, end) seconds.

    A run of frames k0..k1 becomes `[middle(k0) - step/2, middle(k1) + step/2]`
    so that N frames span exactly N * 16.875 ms. (pyannote's own binarizer
    timestamps runs at frame middles; this is that, widened by half a frame on
    each side so durations are exact — the `min_overlap_dur` comparison below
    is a duration test, and losing 16.9 ms of it to a convention would be a
    silent behaviour difference from the server.)
    """
    out: List[Interval] = []
    hits = np.flatnonzero(labels == target)
    if hits.size == 0:
        return out
    breaks = np.flatnonzero(np.diff(hits) > 1)
    starts = np.concatenate(([hits[0]], hits[breaks + 1]))
    ends = np.concatenate((hits[breaks], [hits[-1]]))
    half = FRAME_STEP_S / 2
    for k0, k1 in zip(starts, ends):
        out.append((frame_middle_s(int(k0)) - half, frame_middle_s(int(k1)) + half))
    return out


def _merge_close(segs: Sequence[Interval], merge_gap: float) -> List[Interval]:
    """Verbatim port of `asr_pipeline/stages/routing.py::_merge_close`
    (strict `<` on the gap — keep it strict, the server does)."""
    if not segs:
        return []
    segs = sorted(segs)
    out = [list(segs[0])]
    for s, e in segs[1:]:
        if s - out[-1][1] < merge_gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(float(s), float(e)) for s, e in out]


def route_overlaps(
    raw: Sequence[Interval],
    min_overlap_dur: float = MIN_OVERLAP_DUR,
    merge_gap: float = MERGE_GAP,
) -> List[Interval]:
    """Port of `_select_overlap_regions`: drop short, then merge close.

    Order matters and is the server's: filter FIRST (so a sub-threshold
    overlap can never be resurrected by being merged into a neighbour), merge
    second.
    """
    kept = [(s, e) for s, e in raw if (e - s) >= min_overlap_dur]
    return _merge_close(kept, merge_gap)


def _boundary_aware_pad(
    start_s: float, end_s: float, total_duration_s: float, target_total_s: float
) -> Interval:
    """Verbatim port of `asr_pipeline/stages/separation.py::_boundary_aware_pad`
    (config.py citations in the module docstring §4)."""
    overlap_dur = end_s - start_s
    if target_total_s <= overlap_dur:
        return start_s, end_s
    extra = target_total_s - overlap_dur
    room_left = start_s
    room_right = max(0.0, total_duration_s - end_s)
    left_take = extra / 2.0
    right_take = extra / 2.0
    if left_take > room_left:
        right_take += left_take - room_left
        left_take = room_left
    if right_take > room_right:
        left_take += right_take - room_right
        right_take = room_right
    left_take = min(left_take, room_left)
    right_take = min(right_take, room_right)
    return start_s - left_take, end_s + right_take


def context_pad(
    regions: Sequence[Interval],
    total_duration_s: float,
    mode: str = CONTEXT_WINDOW_MODE,
) -> List[Interval]:
    """Stage-3b context window around each routed overlap region.

    `expand_to_chunk` (server default): pad to `max(training_chunk_length_s,
    min_fragment_length_s)` = 4.0 s total — exactly the separator ONNX's
    32000-sample window.
    `fixed_pad`: symmetric +-1.0 s with the same 4.0 s floor.
    `none`: the POC behaviour, no padding.
    """
    if mode == "none":
        return [(float(s), float(e)) for s, e in regions]
    out = []
    for s, e in regions:
        if mode == "expand_to_chunk":
            target = max(TRAINING_CHUNK_LENGTH_S, MIN_FRAGMENT_LENGTH_S)
        elif mode == "fixed_pad":
            target = max((e - s) + 2 * CONTEXT_PAD_SECONDS, MIN_FRAGMENT_LENGTH_S)
        else:
            raise ValueError(f"unknown context_window_mode {mode!r}")
        out.append(_boundary_aware_pad(s, e, total_duration_s, target))
    return [(float(s), float(e)) for s, e in out]


# ===========================================================================
# top level
# ===========================================================================
@dataclass
class OsdResult:
    duration_s: float
    num_frames: int
    labels: np.ndarray                      # [num_frames] in {0,1,2}
    raw_overlaps: List[Interval]            # contiguous overlap runs
    routed: List[Interval]                  # after min_overlap_dur + merge_gap
    padded: List[Interval]                  # after context padding
    logits: np.ndarray = field(repr=False, default=None)

    def fractions(self) -> dict:
        n = max(1, self.num_frames)
        return {
            LABEL_NAMES[c]: float((self.labels == c).sum()) / n for c in (0, 1, 2)
        }


def run_osd(
    wav: np.ndarray,
    sess=None,
    mode: str = CONTEXT_WINDOW_MODE,
    keep_logits: bool = False,
) -> OsdResult:
    sess = sess or load_session()
    logits = infer_logits(sess, wav)
    labels = decode_classes(logits)
    duration = len(wav) / SAMPLE_RATE
    raw = frames_to_intervals(labels, OVERLAP)
    routed = route_overlaps(raw)
    return OsdResult(
        duration_s=duration,
        num_frames=len(labels),
        labels=labels,
        raw_overlaps=raw,
        routed=routed,
        padded=context_pad(routed, duration, mode),
        logits=logits if keep_logits else None,
    )


# ===========================================================================
# audio helpers
# ===========================================================================
def load_audio(path: Path, sr: int = SAMPLE_RATE, offset_s: float = 0.0,
               duration_s: float | None = None) -> np.ndarray:
    """Mono float32 at `sr`. Stereo is averaged (the CLARIN roots are 2-ch)."""
    import soundfile as sf

    info = sf.info(str(path))
    start = int(offset_s * info.samplerate)
    frames = -1 if duration_s is None else int(duration_s * info.samplerate)
    x, native_sr = sf.read(str(path), start=start, frames=frames, dtype="float32",
                           always_2d=True)
    x = x.mean(axis=1)
    if native_sr != sr:
        x = resample(x, native_sr, sr)
    return np.ascontiguousarray(x, dtype=np.float32)


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    import soxr

    return soxr.resample(x, sr_in, sr_out, quality="VHQ").astype(np.float32)


def interval_iou(a: Sequence[Interval], b: Sequence[Interval]) -> float:
    """IoU of two interval SETS (union-of-intervals, not per-region matching)."""
    def total(iv):
        return sum(e - s for s, e in _merge_close(list(iv), 0.0))

    inter = 0.0
    for s1, e1 in a:
        for s2, e2 in b:
            inter += max(0.0, min(e1, e2) - max(s1, s2))
    union = total(a) + total(b) - inter
    return inter / union if union > 0 else 0.0


# ===========================================================================
# CLI subcommands
# ===========================================================================
CLARIN_ROOT = Path.home() / "datasets/clarin_gotowy/gotowy"
CLARIN_DIAR = Path.home() / "datasets/clarin_all_2speakers/diarization"


def _summary(res: OsdResult) -> str:
    f = res.fractions()
    tot = sum(e - s for s, e in res.routed)
    return (
        f"{res.duration_s:8.1f}s  frames={res.num_frames:6d}  "
        f"nonspeech={f['nonspeech']:.3f} solo={f['solo']:.3f} overlap={f['overlap']:.3f}  "
        f"raw={len(res.raw_overlaps):4d} routed={len(res.routed):4d} "
        f"({tot:6.1f}s = {100*tot/max(res.duration_s,1e-9):4.1f}% of audio)"
    )


def cmd_run(args) -> None:
    sess = load_session(Path(args.model))
    wav = load_audio(Path(args.audio), duration_s=args.max_seconds)
    res = run_osd(wav, sess)
    print(f"{Path(args.audio).name}: {_summary(res)}")
    for i, ((s, e), (ps, pe)) in enumerate(zip(res.routed, res.padded)):
        if i < args.show:
            print(f"  overlap {i:3d}: {s:8.3f}-{e:8.3f} ({e-s:5.2f}s) "
                  f"-> padded {ps:8.3f}-{pe:8.3f} ({pe-ps:5.2f}s)")
    if args.json:
        Path(args.json).write_text(json.dumps({
            "duration_s": res.duration_s,
            "num_frames": res.num_frames,
            "fractions": res.fractions(),
            "routed": res.routed,
            "padded": res.padded,
        }, indent=1))


def cmd_validate(args) -> None:
    """OSD on real CLARIN recordings, cross-read against pyannote-3.1's own
    overlap timeline (`clarin_all_2speakers/diarization/<id>.json`)."""
    sess = load_session(Path(args.model))
    ids = args.ids or ["442dd69e", "6d88daa2", "649991bc"]
    for rid in ids:
        wav_path = CLARIN_ROOT / f"{rid}.wav"
        if not wav_path.is_file():
            print(f"{rid}: MISSING {wav_path}")
            continue
        wav = load_audio(wav_path, duration_s=args.max_seconds)
        res = run_osd(wav, sess)
        print(f"\n{rid}: {_summary(res)}")

        diar = CLARIN_DIAR / f"{rid}.json"
        if not diar.is_file():
            print("  (no pyannote diarization json to compare against)")
            continue
        d = json.loads(diar.read_text())
        ref_raw = [(o["start"], o["end"]) for o in d["overlaps"]]
        ref = route_overlaps(ref_raw)          # same routing, server's overlaps
        if args.max_seconds:
            ref = [(s, min(e, res.duration_s)) for s, e in ref if s < res.duration_s]
        ref_tot = sum(e - s for s, e in ref)
        ours_tot = sum(e - s for s, e in res.routed)
        print(f"  pyannote-3.1 routed: {len(ref):4d} regions ({ref_tot:6.1f}s) | "
              f"ours: {len(res.routed):4d} ({ours_tot:6.1f}s) | "
              f"IoU={interval_iou(res.routed, ref):.3f}")
        # how many of the server's regions we cover at all (recall-ish)
        hit = sum(
            1 for s, e in ref
            if any(min(e, e2) - max(s, s2) > 0 for s2, e2 in res.routed)
        )
        hit2 = sum(
            1 for s, e in res.routed
            if any(min(e, e2) - max(s, s2) > 0 for s2, e2 in ref)
        )
        print(f"  region hit-rate: {hit}/{len(ref)} of pyannote's touched by ours; "
              f"{hit2}/{len(res.routed)} of ours touched by pyannote's")


def cmd_precheck(args) -> None:
    """Plan §5 risk item: does OSD survive 8 kHz-bandwidth audio?

    PolSESS-derived clips are natively 8 kHz; the demo has to upsample them to
    16 kHz to feed pyannote. This measures how much that costs by taking real
    16 kHz audio, decimating to 8 kHz and upsampling back (band-limiting it to
    0-4 kHz without changing anything else) and re-running OSD.
    """
    sess = load_session(Path(args.model))
    default = [
        "442dd69e",  # Polish conversation, sparse overlap
        "6d88daa2",  # Polish conversation, lots of silence
        # clean, densely overlapped speech — the analogue of the demo's
        # "fully overlapped PolSESS-style showcase clip"
        str(Path.home() / "datasets/LibriCSS_2spk/record/segments/OV40_session1_seg6.wav"),
    ]
    for rid in (args.ids or default):
        wav_path = Path(rid) if Path(rid).is_file() else CLARIN_ROOT / f"{rid}.wav"
        rid = wav_path.stem
        wav16 = load_audio(wav_path, duration_s=args.max_seconds)
        wav8 = resample(resample(wav16, SAMPLE_RATE, 8000), 8000, SAMPLE_RATE)
        wav8 = wav8[: len(wav16)]
        if len(wav8) < len(wav16):
            wav8 = np.pad(wav8, (0, len(wav16) - len(wav8)))

        a = run_osd(wav16, sess)
        b = run_osd(wav8, sess)
        n = min(a.num_frames, b.num_frames)
        agree = float((a.labels[:n] == b.labels[:n]).mean())
        conf = np.zeros((3, 3), dtype=int)
        for i in range(3):
            for j in range(3):
                conf[i, j] = int(((a.labels[:n] == i) & (b.labels[:n] == j)).sum())
        hit = sum(
            1 for s, e in a.routed
            if any(min(e, e2) - max(s, s2) > 0 for s2, e2 in b.routed)
        )
        print(f"\n8 kHz-bandwidth pre-check on {rid} ({a.duration_s:.0f}s)")
        print(f"  native 16k : {_summary(a)}")
        print(f"  8k->16k    : {_summary(b)}")
        print(f"  per-frame label agreement : {100*agree:.2f}%")
        print(f"  overlap-frame agreement   : "
              f"{100*float((b.labels[:n][a.labels[:n]==OVERLAP]==OVERLAP).mean()):.2f}% "
              f"of native-overlap frames still overlap")
        print(f"  region IoU (routed)       : {interval_iou(a.routed, b.routed):.3f}")
        print(f"  raw-overlap-interval IoU  : {interval_iou(a.raw_overlaps, b.raw_overlaps):.3f}")
        print(f"  region recall             : {hit}/{len(a.routed)} native regions "
              f"still touched after the 8 kHz round trip")
        print("  confusion (rows=16k, cols=8k->16k, order nonspeech/solo/overlap):")
        for i in range(3):
            print(f"    {LABEL_NAMES[i]:>9}: {conf[i].tolist()}")


# --- parity vectors ---------------------------------------------------------
@dataclass
class VectorSpec:
    name: str
    source: Path
    offset_s: float
    duration_s: float
    note: str


# Both excerpts were picked by scanning for windows that actually EXERCISE the
# routing logic: each contains at least one raw overlap below `min_overlap_dur`
# (must be dropped) and at least one pair closer than `merge_gap` (must be
# merged). A vector with one clean overlap would pass on a JS port that
# implements neither.
VECTOR_SPECS = (
    VectorSpec(
        "clarin_442dd69e_sparse",
        CLARIN_ROOT / "442dd69e.wav",
        offset_s=865.0, duration_s=20.0,
        note="Polish 2-speaker conversation (CLARIN), mostly solo with short "
             "overlap bursts — the routing story. Exercises the min_overlap_dur "
             "FILTER: one raw overlap of 0.084 s must be dropped (8 raw -> 7 "
             "routed), and no pair is close enough to merge.",
    ),
    VectorSpec(
        "libricss_ov40_dense",
        Path.home() / "datasets/LibriCSS_2spk/record/segments/OV40_session4_seg8.wav",
        offset_s=0.0, duration_s=20.0,
        note="LibriCSS 2-speaker segment, 40% overlap-ratio condition — dense "
             "overlap, English, LibriSpeech-derived (CC BY 4.0). Exercises the "
             "merge_gap MERGE (3 raw overlaps with 0.34 s / 0.47 s gaps collapse "
             "into one 3.63 s region) and is the one case where "
             "expand_to_chunk and fixed_pad disagree (region > 2 s).",
    ),
)


def cmd_vectors(args) -> None:
    import soundfile as sf

    sess = load_session(Path(args.model))
    VECTORS_DIR.mkdir(parents=True, exist_ok=True)
    index = []
    for spec in VECTOR_SPECS:
        if not spec.source.is_file():
            print(f"SKIP {spec.name}: missing {spec.source}")
            continue
        wav = load_audio(spec.source, offset_s=spec.offset_s, duration_s=spec.duration_s)
        wav_path = VECTORS_DIR / f"{spec.name}.wav"
        sf.write(str(wav_path), wav, SAMPLE_RATE, subtype="PCM_16")
        # re-read the 16-bit file so the vectors describe EXACTLY the audio
        # the JS port will decode, not the float32 it came from
        wav = load_audio(wav_path)

        res = run_osd(wav, sess, keep_logits=True)
        # raw logits from the FIRST window only, so the JS port can check its
        # tensor plumbing before it trusts its stitching
        first = sess.run(None, {
            "input_values": np.pad(
                wav[:WINDOW_SAMPLES], (0, max(0, WINDOW_SAMPLES - len(wav)))
            ).reshape(1, 1, WINDOW_SAMPLES).astype(np.float32)
        })[0][0]
        payload = {
            "name": spec.name,
            "note": spec.note,
            "audio": wav_path.name,
            "sample_rate": SAMPLE_RATE,
            "num_samples": int(len(wav)),
            "duration_s": round(res.duration_s, 6),
            "model": Path(args.model).name,
            "contract": {
                "input": "input_values float32 [1,1,160000]",
                "output": "logits float32 [1,589,7] (LogSoftmax)",
                "frame_step_samples": FRAME_STEP_SAMPLES,
                "frame_width_samples": FRAME_WIDTH_SAMPLES,
                "hop_samples": HOP_SAMPLES,
                "keep_frames": [KEEP_START, KEEP_END],
            },
            "num_frames": int(res.num_frames),
            "labels_legend": {str(i): LABEL_NAMES[i] for i in range(3)},
            "labels": res.labels.tolist(),
            "powerset_argmax": res.logits.argmax(-1).tolist(),
            "first_window_logits": {
                str(k): [round(float(v), 4) for v in first[k]]
                for k in (0, 100, 200)
            },
            "raw_overlaps": [[round(s, 6), round(e, 6)] for s, e in res.raw_overlaps],
            "routed": [[round(s, 6), round(e, 6)] for s, e in res.routed],
            "padded_expand_to_chunk": [
                [round(s, 6), round(e, 6)]
                for s, e in context_pad(res.routed, res.duration_s, "expand_to_chunk")
            ],
            "padded_fixed_pad_1s": [
                [round(s, 6), round(e, 6)]
                for s, e in context_pad(res.routed, res.duration_s, "fixed_pad")
            ],
            "constants": {
                "min_overlap_dur": MIN_OVERLAP_DUR,
                "merge_gap": MERGE_GAP,
                "training_chunk_length_s": TRAINING_CHUNK_LENGTH_S,
                "min_fragment_length_s": MIN_FRAGMENT_LENGTH_S,
                "context_pad_seconds": CONTEXT_PAD_SECONDS,
            },
        }
        out = VECTORS_DIR / f"{spec.name}.json"
        out.write_text(json.dumps(payload, indent=1))
        index.append({
            "name": spec.name, "json": out.name, "audio": wav_path.name,
            "note": spec.note,
            "source": str(spec.source),
            "offset_s": spec.offset_s, "duration_s": spec.duration_s,
        })
        print(f"{spec.name}: {_summary(res)}")
        print(f"  -> {out.name} ({out.stat().st_size/1024:.1f} KB) + "
              f"{wav_path.name} ({wav_path.stat().st_size/1024:.1f} KB)")
    (VECTORS_DIR / "index.json").write_text(json.dumps(index, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--ids", nargs="*", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="OSD on one audio file")
    p.add_argument("audio")
    p.add_argument("--json", default=None)
    p.add_argument("--show", type=int, default=10)
    p.set_defaults(func=cmd_run)

    sub.add_parser("validate", help="OSD on CLARIN recordings vs pyannote-3.1"
                   ).set_defaults(func=cmd_validate)
    sub.add_parser("precheck", help="8 kHz-bandwidth risk item").set_defaults(
        func=cmd_precheck)
    sub.add_parser("vectors", help="write reference/vectors/").set_defaults(
        func=cmd_vectors)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
