"""Batch NVIDIA Sortformer worker — many wav files, ONE model load.

Batch twin of ``sortformer_worker.py`` (same isolated NeMo venv, same model
dispatch, same (T, 4) @ 0.08 s output contract), for corpus-scale diarization
where re-loading the model per recording (~10 s) would dominate. Loads the model
once, then writes one ``<stem>.npz`` (``frame_probs`` float32 (T, 4),
``frame_rate_s``, ``model``) per input. Idempotent: existing outputs are
skipped. Turn-building stays in ``asr_pipeline.stages.diarization`` — this
script only emits raw per-frame activity, exactly like the single-file worker.

Run it under ``$SORTFORMER_VENV_PY`` from a cwd OUTSIDE the repo (the repo's
local ``datasets/`` package shadows HF ``datasets`` and breaks NeMo imports).
No repo imports here, for the same reason.

Usage:
    cd /tmp && $SORTFORMER_VENV_PY /path/to/scripts/sortformer_batch_worker.py \
        --in-dir ~/datasets/clarin_all_2speakers/clarin_download \
        --out-dir ~/datasets/clarin_all_2speakers/diarization_sortformer/probs \
        --model nvidia/diar_streaming_sortformer_4spk-v2.1
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

FRAME_SHIFT_S = 0.08   # one activity frame per 80 ms, v1 and v2.1 alike


def _to_np(x):
    import torch

    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def apply_offline_preset(model) -> None:
    """Very-high-latency OFFLINE preset for the streaming v2.1 model (model-card
    README values, in 80 ms frame units) — identical to sortformer_worker.py."""
    sm = model.sortformer_modules
    sm.chunk_len = 340
    sm.chunk_right_context = 40
    sm.fifo_len = 40
    sm.spkcache_update_period = 300
    sm.spkcache_len = 188
    sm._check_streaming_parameters()
    print("[sortformer] STREAMING model -> OFFLINE very-high-latency preset: "
          "chunk_len=340 chunk_right_context=40 fifo_len=40 "
          "spkcache_update_period=300 spkcache_len=188", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="nvidia/diar_streaming_sortformer_4spk-v2.1")
    ap.add_argument("--device", default=None)
    ap.add_argument("--only", nargs="*", default=None,
                    help="optional list of wav basenames to restrict to")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    import soundfile as sf
    import torch
    from nemo.collections.asr.models import SortformerEncLabelModel

    in_dir = Path(a.in_dir).expanduser()
    out_dir = Path(a.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = sorted(p for p in in_dir.iterdir() if p.suffix == ".wav")
    if a.only:
        keep = set(a.only)
        inputs = [p for p in inputs if p.name in keep]
    print(f"[sortformer] {len(inputs)} wav files in {in_dir}", flush=True)

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[sortformer] loading {a.model} on {device} ...", flush=True)
    t0 = time.perf_counter()
    model = SortformerEncLabelModel.from_pretrained(a.model)
    model.eval()
    if device == "cuda":
        model = model.to(torch.device("cuda"))
    if "streaming" in a.model.lower():
        apply_offline_preset(model)
    print(f"[sortformer] loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    done = skipped = 0
    failed = []
    for i, wav in enumerate(inputs, 1):
        out = out_dir / f"{wav.stem}.npz"
        if out.exists() and not a.force:
            skipped += 1
            continue
        info = sf.info(str(wav))
        dur = info.frames / info.samplerate
        t1 = time.perf_counter()
        try:
            _segs, probs = model.diarize(audio=str(wav), batch_size=1,
                                         include_tensor_outputs=True)
            arr = _to_np(probs[0])
            arr = arr.reshape(-1, arr.shape[-1]).astype(np.float32)
        except Exception as e:  # per-file isolation; batch continues
            failed.append((wav.name, repr(e)[:300]))
            print(f"[sortformer] ({i}/{len(inputs)}) {wav.name} FAILED: {e!r}",
                  flush=True)
            if device == "cuda":
                torch.cuda.empty_cache()
            continue
        el = time.perf_counter() - t1
        peak = (torch.cuda.max_memory_allocated() / 2**30) if device == "cuda" else 0.0
        tmp = out.with_suffix(".npz.tmp.npz")
        np.savez_compressed(tmp, frame_probs=arr, frame_rate_s=FRAME_SHIFT_S,
                            model=a.model)
        tmp.replace(out)
        done += 1
        print(f"[sortformer] ({i}/{len(inputs)}) {wav.name} {dur:.0f}s -> "
              f"{arr.shape[0]} frames in {el:.1f}s (RTF {el / dur:.4f}, "
              f"peak {peak:.2f} GiB)", flush=True)

    print(f"[sortformer] done: {done} written, {skipped} skipped, "
          f"{len(failed)} failed", flush=True)
    for name, msg in failed:
        print(f"  - {name}: {msg}", flush=True)
    return 2 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
