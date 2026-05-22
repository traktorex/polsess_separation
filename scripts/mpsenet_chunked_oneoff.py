"""One-off A/B tool: run vendored MP-SENet on a whole audio file with Hann
overlap-add chunking, independent of the pipeline's routing / diarization.

Use this when you want to:

- Listen to MP-SENet on a full long recording without going through the
  partition / per-region splicing of `EnhancementStage`.
- A/B different chunk sizes and hop ratios (e.g. non-50% overlap), which
  the package's `EnhancementStage._enhance_chunked` doesn't expose
  (50% hop / canonical Hann COLA is hard-coded there).

The package's `EnhancementStage` now performs the same Hann overlap-add
chunking automatically for solo regions longer than
`cfg.enhancement.max_segment_length_s` — this script is no longer needed
to make long-file enhancement work *in the pipeline*. It survives as an
ad-hoc inspection tool.

Default: 8 s chunks with 50% overlap (Hann window, exact COLA in the
interior; edge weights divided out explicitly for the head/tail).

Usage:
    python scripts/mpsenet_chunked_oneoff.py INPUT.wav OUTPUT.wav \\
        [--chunk-s 8.0] [--hop-s 4.0]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

# Make `asr_pipeline` importable when running from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from asr_pipeline.stages.enhancement import (
    _AttrDict,
    _mpsenet_stft,
    _mpsenet_istft,
)
from asr_pipeline.vendor.mpsenet import MPNet


def enhance_full_file(
    audio: np.ndarray,
    model: MPNet,
    h: _AttrDict,
    device: torch.device,
    sample_rate: int,
    chunk_s: float,
    hop_s: float,
) -> np.ndarray:
    """Run MP-SENet over `audio` with Hann overlap-add chunking. Returns float32."""
    n = len(audio)
    chunk_n = int(chunk_s * sample_rate)
    hop_n = int(hop_s * sample_rate)
    if n <= chunk_n:
        # One-shot fits; no chunking needed.
        return _enhance_one_shot(audio, model, h, device).astype(np.float32)

    out = np.zeros(n, dtype=np.float32)
    weights = np.zeros(n, dtype=np.float32)
    window = np.hanning(chunk_n).astype(np.float32)

    start = 0
    n_chunks = 0
    while start < n:
        end = min(start + chunk_n, n)
        seg = audio[start:end]
        if len(seg) < h["win_size"]:
            # Too short for MP-SENet's STFT — pass through.
            chunk_out = seg.astype(np.float32)
        else:
            # Pad short final chunk so MP-SENet receives `chunk_n` samples.
            if len(seg) < chunk_n:
                seg_padded = np.pad(seg, (0, chunk_n - len(seg)))
            else:
                seg_padded = seg
            chunk_out = _enhance_one_shot(seg_padded, model, h, device).astype(np.float32)
            chunk_out = chunk_out[: len(seg)]

        w = window[: len(seg)]
        out[start:end] += chunk_out * w
        weights[start:end] += w
        n_chunks += 1
        if end == n:
            break
        start += hop_n

    weights = np.maximum(weights, 1e-6)
    return (out / weights).astype(np.float32), n_chunks


@torch.no_grad()
def _enhance_one_shot(
    audio_np: np.ndarray,
    model: MPNet,
    h: _AttrDict,
    device: torch.device,
) -> np.ndarray:
    audio = torch.from_numpy(audio_np).to(device)
    norm = torch.sqrt(audio.numel() / (audio.pow(2).sum() + 1e-12))
    audio_n = (audio * norm).unsqueeze(0)
    mag, pha = _mpsenet_stft(audio_n, h)
    out = model(mag, pha)
    if isinstance(out, (tuple, list)):
        amp_out, pha_out = out[0], out[1]
    else:
        amp_out, pha_out = out, pha
    audio_out = _mpsenet_istft(amp_out, pha_out, h).squeeze(0)
    audio_out = (audio_out / (norm + 1e-12)).cpu().numpy().astype(np.float32)
    if len(audio_out) < len(audio_np):
        audio_out = np.pad(audio_out, (0, len(audio_np) - len(audio_out)))
    else:
        audio_out = audio_out[: len(audio_np)]
    return audio_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input wav (mono, any sample rate; resampled to model SR if needed).")
    parser.add_argument("output", help="Output wav path.")
    parser.add_argument(
        "--mpsenet-config",
        default="/home/user/MP-SENet/config.json",
        help="MP-SENet config.json (provides STFT hparams).",
    )
    parser.add_argument(
        "--mpsenet-ckpt",
        default="/home/user/MP-SENet/best_ckpt/g_best_vb",
        help="MP-SENet generator checkpoint.",
    )
    parser.add_argument("--chunk-s", type=float, default=8.0, help="Chunk size in seconds.")
    parser.add_argument("--hop-s", type=float, default=4.0, help="Chunk hop in seconds (50% overlap is hop = chunk/2).")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)

    with open(args.mpsenet_config) as f:
        h = _AttrDict(json.load(f))
    model = MPNet(h).to(device)
    state = torch.load(args.mpsenet_ckpt, map_location=device)
    model.load_state_dict(state["generator"])
    model.eval()
    print(f"Loaded MP-SENet on {device}")

    audio, sr = sf.read(args.input)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    print(f"Input: {len(audio)/sr:.2f}s @ {sr}Hz  (chunk={args.chunk_s}s, hop={args.hop_s}s)")
    if sr != 16_000:
        print(f"WARNING: input SR is {sr}, MP-SENet expects 16000")

    t0 = time.time()
    out, n_chunks = enhance_full_file(
        audio, model, h, device, sr,
        chunk_s=args.chunk_s,
        hop_s=args.hop_s,
    )
    t = time.time() - t0

    sf.write(args.output, out, sr)
    print(f"Wrote {args.output}  ({len(out)/sr:.2f}s, {n_chunks} chunks, {t:.1f}s wall, "
          f"{(t / (len(out)/sr)):.2f}× real-time)")


if __name__ == "__main__":
    main()
