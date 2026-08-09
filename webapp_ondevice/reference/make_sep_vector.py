#!/usr/bin/env python3
"""Generate the separator parity vector consumed by `webapp_ondevice/test.html`.

The OSD vectors (`vectors/*.json`) pin the *detection* half of the browser
engine. This script pins the *separation* half: it takes one fixed 4 s excerpt,
runs it through both shipping separator ONNX files on the CPU execution
provider, and writes the input plus all four output streams as raw float32 so
the JS port can compare its own WASM run against them.

What is fixed here (and must stay in lockstep with `site/js/separate.js`):

* the excerpt — the *padded* routed region `[14.332187, 18.332187]` of
  `vectors/libricss_ov40_dense.wav`, i.e. exactly what the OSD stage hands the
  separator for that vector's second overlap region. LibriCSS is CC BY 4.0, so
  this fixture is publishable if it ever has to be;
* 16 kHz -> 8 kHz decimation with soxr VHQ (`osd_reference.resample`). The
  browser cannot reproduce soxr, which is exactly why the decimated audio is
  shipped in the vector instead of being derived in JS — the separator check
  must measure the separator, not the resampler;
* the per-chunk gain convention: `peak = max|x|`, model sees `x / peak`, output
  is multiplied back by `peak`. The separators are SI-SDR-trained and therefore
  scale-free, so this convention is a choice, not a law — but it is the choice
  `separate.js` implements, and the vector has to encode the same one or the
  comparison is not apples-to-apples.

The excerpt is exactly one 32000-sample chunk, so no chunking / crossfade /
permutation logic is exercised here. That is deliberate: this vector isolates
"does the browser run the graph correctly", and the chunker is covered by the
COLA/permutation assertions in `test.js`.

Usage (CPU only, ~5 s):

    CUDA_VISIBLE_DEVICES="" venv/bin/python webapp_ondevice/reference/make_sep_vector.py

Output (both tracked, 640 KB + 2 KB):

    vectors/sep_vector.bin    float32 LE, 5 x 32000: input, mf2 s1, mf2 s2,
                              sepformer s1, sepformer s2
    vectors/sep_vector.json   offsets into the .bin + provenance + the numbers
                              a human needs to sanity-check it
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from osd_reference import SAMPLE_RATE, load_audio, resample

VECTORS_DIR = Path(__file__).resolve().parent / "vectors"
MODELS_DIR = Path(__file__).resolve().parents[1] / "site" / "models"

SOURCE_WAV = "libricss_ov40_dense.wav"
# second padded (expand_to_chunk) region of that vector — see its .json
REGION_START_S = 14.332187
REGION_END_S = 18.332187
SEPARATOR_SR = 8000
CHUNK = 32000  # 4 s @ 8 kHz — the separators' static input shape

MODELS = (
    ("mf2", "mf2_128k_int8.onnx"),
    ("sepformer", "sepformer_128k_int8.onnx"),
)


def peak_normalized_separate(sess, chunk8k: np.ndarray) -> np.ndarray:
    """One separator call under the `separate.js` gain convention.

    Returns [2, CHUNK] float32 in the *input's* scale.
    """
    peak = float(np.max(np.abs(chunk8k)))
    gain = 1.0 / peak if peak > 1e-8 else 1.0
    x = (chunk8k * gain).astype(np.float32).reshape(1, CHUNK)
    y = sess.run(None, {"mixture": x})[0]  # [1, 2, CHUNK]
    return (y[0] / gain).astype(np.float32)


def main() -> None:
    import onnxruntime as ort

    wav16k = load_audio(VECTORS_DIR / SOURCE_WAV)
    start = round(REGION_START_S * SAMPLE_RATE)
    stop = start + round((REGION_END_S - REGION_START_S) * SAMPLE_RATE)
    assert stop <= len(wav16k), f"region runs past the vector ({stop} > {len(wav16k)})"
    region16k = wav16k[start:stop]

    chunk = resample(region16k, SAMPLE_RATE, SEPARATOR_SR)
    assert len(chunk) == CHUNK, f"expected {CHUNK} samples @ 8 kHz, got {len(chunk)}"
    chunk = chunk.astype(np.float32)

    blocks = [("input", chunk)]
    for tag, fname in MODELS:
        path = MODELS_DIR / fname
        # Default optimization level is fine for the INT8 separators; the
        # ORT_ENABLE_ALL segfault is specific to the fp16 pyannote graph
        # (build/NOTES.md §2).
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        out = peak_normalized_separate(sess, chunk)
        blocks.append((f"{tag}_s1", out[0]))
        blocks.append((f"{tag}_s2", out[1]))
        print(
            f"{fname}: peak={np.abs(out).max():.3f} "
            f"rms={np.sqrt((out[0]**2).mean()):.4f}/{np.sqrt((out[1]**2).mean()):.4f} "
            f"s1~s2 corr={np.corrcoef(out[0], out[1])[0, 1]:+.3f}"
        )

    raw = np.concatenate([b for _, b in blocks]).astype("<f4")
    bin_path = VECTORS_DIR / "sep_vector.bin"
    bin_path.write_bytes(raw.tobytes())

    payload = {
        "note": (
            "Separator parity vector: one 4 s @ 8 kHz chunk (the padded routed "
            "region [14.332187, 18.332187] of libricss_ov40_dense.wav, soxr-VHQ "
            "decimated 16k->8k) plus the CPU-EP output of both shipping "
            "separators under the separate.js peak-gain convention. Generated by "
            "reference/make_sep_vector.py."
        ),
        "binary": bin_path.name,
        "dtype": "float32-le",
        "sample_rate": SEPARATOR_SR,
        "chunk_samples": CHUNK,
        "source_audio": SOURCE_WAV,
        "region_s": [REGION_START_S, REGION_END_S],
        "gain_convention": "model input = x / max|x|; model output * max|x|",
        "models": {tag: fname for tag, fname in MODELS},
        "onnxruntime": ort.__version__,
        # offset = index into the float32 array, not a byte offset
        "layout": [
            {"name": name, "offset": i * CHUNK, "length": CHUNK}
            for i, (name, _) in enumerate(blocks)
        ],
        "stats": {
            name: {
                "peak": round(float(np.abs(a).max()), 6),
                "rms": round(float(np.sqrt((a ** 2).mean())), 6),
            }
            for name, a in blocks
        },
        "sha256": hashlib.sha256(bin_path.read_bytes()).hexdigest(),
        "tolerance": (
            "JS (ORT-Web WASM, single-threaded) vs this (ORT CPU EP): expect "
            "Pearson corr >= 0.999 per stream. Same graph, same weights, "
            "different kernels — bit-equality is not on offer."
        ),
    }
    json_path = VECTORS_DIR / "sep_vector.json"
    json_path.write_text(json.dumps(payload, indent=1))
    print(f"\n-> {bin_path.name} ({bin_path.stat().st_size / 1024:.1f} KB) + "
          f"{json_path.name} ({json_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
