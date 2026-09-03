#!/usr/bin/env python3
"""Verify the shipping ONNX artifacts in `webapp_ondevice/site/models/`.

Three checks, all CPU-only:

1. Each separator loads in an `onnxruntime` CPU session, accepts the January
   contract tensor (`mixture` float32 [1, 32000]) and returns
   `separated` float32 [1, 2, 32000] that is finite and non-silent.
2. Separator output on a REAL PolSESS test mixture is compared against the
   eager PyTorch checkpoint it was exported from — correlation and
   "agreement SI-SDR" (how well the ONNX stream reconstructs the eager
   stream; this is graph fidelity, NOT separation quality).
3. The pyannote-segmentation-3.0 fp16 OSD model loads and reports its I/O
   contract. (Its real validation lives in `webapp_ondevice/reference/`.)

Usage:
    CUDA_VISIBLE_DEVICES="" venv/bin/python webapp_ondevice/build/verify_models.py
    ... --compare-dir <dir>   # also diff against a reference export (e.g. the
                              # 2026-08-05 export experiment's scratch dir) —
                              # expects corr == 1.0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from export_separators import CHUNK, SEPARATORS  # noqa: E402  (sibling module)

APP_DIR = PROJECT_ROOT / "webapp_ondevice"
MODELS_DIR = APP_DIR / "site" / "models"
POLSESS_TEST_MIX = Path.home() / "datasets/PolSESS_C_final_128_v2/test/mix"
# file names used by the 2026-08-05 export experiment, for --compare-dir
SPIKE_EQUIVALENT = {
    "mf2_128k_int8.onnx": "mf2_int8_mm_pw.onnx",
    "sepformer_128k_int8.onnx": "sepformer128k_int8_matmul.onnx",
}


def real_mixture(n: int = 1, offset_s: float = 1.0):
    """First n PolSESS test mixtures -> mono 8 kHz, 4 s window from offset_s."""
    import torchaudio

    files = sorted(POLSESS_TEST_MIX.glob("*.wav"))[:n]
    if not files:
        raise SystemExit(f"no test mixtures under {POLSESS_TEST_MIX}")
    out = []
    for f in files:
        wav, sr = torchaudio.load(str(f))
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        wav = torchaudio.transforms.Resample(sr, 8000)(wav)
        start = int(offset_s * 8000)
        seg = wav[:, start : start + CHUNK]
        if seg.shape[1] < CHUNK:
            seg = torch.nn.functional.pad(seg, (0, CHUNK - seg.shape[1]))
        out.append((f.name[:16], seg))
    return out


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))


def si_sdr(est: np.ndarray, ref: np.ndarray, eps: float = 1e-9) -> float:
    est, ref = est - est.mean(), ref - ref.mean()
    alpha = (est * ref).sum() / ((ref * ref).sum() + eps)
    tgt = alpha * ref
    return float(10 * np.log10(((tgt**2).sum() + eps) / (((est - tgt) ** 2).sum() + eps)))


def main() -> None:
    import onnxruntime as ort

    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    ap.add_argument("--compare-dir", type=Path, default=None)
    ap.add_argument("--n", type=int, default=1, help="how many test mixtures")
    args = ap.parse_args()

    clips = real_mixture(args.n)
    print(f"real PolSESS test clips: {[c[0] for c in clips]}")

    from utils.model_utils import load_model_for_inference

    ok = True
    for sep in SEPARATORS:
        path = args.models_dir / sep.out_name
        print(f"\n=== {path.name} ({path.stat().st_size / 1024**2:.2f} MB)")
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        i, o = sess.get_inputs()[0], sess.get_outputs()[0]
        print(f"  IO: {i.name}{i.shape} {i.type} -> {o.name}{o.shape} {o.type}")
        assert i.name == "mixture" and o.name == "separated", "tensor-name contract broken"
        assert list(i.shape) == [1, CHUNK], f"input shape not static [1,{CHUNK}]: {i.shape}"

        model, _ = load_model_for_inference(str(PROJECT_ROOT / sep.ckpt), device="cpu")
        model.eval()
        for name, w in clips:
            x = w.numpy().astype(np.float32)
            y = sess.run(None, {"mixture": x})[0]
            assert y.shape == (1, 2, CHUNK), y.shape
            assert np.isfinite(y).all(), "non-finite output"
            rms = [float(np.sqrt((y[0, c] ** 2).mean())) for c in range(2)]
            assert min(rms) > 1e-4, f"silent stream: rms={rms}"
            with torch.no_grad():
                ref = model(w).numpy()
            cs = [corr(ref[0, c], y[0, c]) for c in range(2)]
            ss = [si_sdr(y[0, c], ref[0, c]) for c in range(2)]
            xs = corr(y[0, 0], y[0, 1])  # inter-stream correlation
            print(
                f"  {name}: rms={rms[0]:.3f}/{rms[1]:.3f}  vs eager corr="
                f"{min(cs):.6f} SI-SDR={min(ss):.2f} dB  |  s1~s2 corr={xs:+.3f}"
            )
            if min(cs) < 0.999:
                ok = False
                print("  !! FAIL: ONNX diverges from eager PyTorch")

            if args.compare_dir:
                ref_path = args.compare_dir / SPIKE_EQUIVALENT.get(path.name, path.name)
                if ref_path.is_file():
                    rs = ort.InferenceSession(
                        str(ref_path), providers=["CPUExecutionProvider"]
                    )
                    yr = rs.run(None, {"mixture": x})[0]
                    c2 = min(corr(yr[0, c], y[0, c]) for c in range(2))
                    print(f"  vs {ref_path.name}: corr={c2:.8f}  (expect 1.0)")
                    if c2 < 0.99999:
                        ok = False
                        print("  !! FAIL: differs from reference export")

    osd = args.models_dir / "pyannote_seg3_fp16.onnx"
    print(f"\n=== {osd.name} ({osd.stat().st_size / 1024**2:.2f} MB)")
    # ORT 1.23.2's DEFAULT optimization level (ORT_ENABLE_ALL) SEGFAULTS on this
    # fp16 graph — see webapp_ondevice/reference/osd_reference.py::load_session.
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    sess = ort.InferenceSession(str(osd), so, providers=["CPUExecutionProvider"])
    for t in sess.get_inputs():
        print(f"  in : {t.name} {t.shape} {t.type}")
    for t in sess.get_outputs():
        print(f"  out: {t.name} {t.shape} {t.type}")

    print("\nALL CHECKS PASSED" if ok else "\nFAILURES ABOVE")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
