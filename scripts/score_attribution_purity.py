"""Reference-free attribution purity for assembled per-speaker streams.

The attribution gap (cpWER - ORC) measures mis-filed content in WER units, but it
needs GT transcripts and is buried under ASR/Polish-morphology noise. This is the
complementary, transcript-free instrument (idea #1): does each assembled stream
hold a *single* speaker? Cross-contamination — the per-overlap leakage that *is*
the cpWER-ORC gap — shows up as a stream carrying two voices.

Metric (centroid self-consistency, swap-invariant):
  - Slice each assembled stream (`<frag>/sweep/<cfg>/stream_{A,B}.wav`) into
    WINDOW_S windows, drop near-silent ones (full_length streams are mostly
    silence), ECAPA-embed the rest.
  - Build each stream's centroid (mean unit embedding). A window is "pure" if it
    is closer (cosine) to its OWN stream's centroid than to the other's.
  - purity = pure_windows / total_windows over both streams, in [~0.5, 1.0].
  A global A<->B swap swaps centroids AND labels together, so purity is
  unchanged (matching cpWER's permutation-invariance); only genuine
  cross-contamination lowers it. The speaker-distinctness confound is constant
  per fragment, so Δpurity between configs is valid (it cancels), exactly like
  the WER gap.

Writes per (frag, config): pure_window count + total_window count, so the
downstream rescorer can micro-average it the same way it does errors/length.

    python scripts/score_attribution_purity.py --split dev --configs f_oa03 ct_tau02
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.eval_harness import eval_root, load_split                    # noqa: E402

EVAL = eval_root()
SR = 16_000
WINDOW_S = 1.5            # ECAPA window; >= its 0.25 s floor with margin
RMS_GATE = 1e-3          # skip near-silent windows (assembled streams are sparse)
OUT_CSV = EVAL / "_attribution_purity.csv"


def _load_ecapa(device):
    from speechbrain.inference.speaker import EncoderClassifier
    cache = REPO / ".cache" / "ecapa"
    cache.mkdir(parents=True, exist_ok=True)
    ecapa = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": str(device)}, savedir=str(cache))
    ecapa.eval()
    return ecapa


@torch.no_grad()
def _embed_windows(audio: np.ndarray, ecapa, device) -> np.ndarray:
    """Unit ECAPA embeddings for each speech-active WINDOW_S window. (n, d)."""
    win = int(SR * WINDOW_S)
    embs = []
    for start in range(0, max(len(audio) - win + 1, 0), win):
        w = audio[start:start + win]
        if np.sqrt(np.mean(w.astype(np.float64) ** 2)) < RMS_GATE:
            continue
        t = torch.from_numpy(w.astype(np.float32)).unsqueeze(0).to(device)
        e = ecapa.encode_batch(t).squeeze(0).squeeze(0)
        e = e / (e.norm() + 1e-8)
        embs.append(e.cpu().numpy())
    return np.asarray(embs, dtype=np.float64) if embs else np.zeros((0, 192))


def _purity(emb_a: np.ndarray, emb_b: np.ndarray) -> tuple[int, int]:
    """(pure_windows, total_windows) by centroid self-consistency. (0, 0) when
    either stream has no active window (purity undefined — recorded as a skip)."""
    if len(emb_a) == 0 or len(emb_b) == 0:
        return 0, 0
    ca = emb_a.mean(0); ca /= (np.linalg.norm(ca) + 1e-8)
    cb = emb_b.mean(0); cb /= (np.linalg.norm(cb) + 1e-8)
    pure = int((emb_a @ ca >= emb_a @ cb).sum() + (emb_b @ cb >= emb_b @ ca).sum())
    return pure, len(emb_a) + len(emb_b)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--fragments-file", default=None)
    ap.add_argument("--out", default=str(OUT_CSV))
    args = ap.parse_args()

    frags = load_split(args.split, args.fragments_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ecapa = _load_ecapa(device)

    out_path = Path(args.out).expanduser()
    # Merge with any existing rows (keep other configs' results); rewrite ours.
    rows: dict[tuple[str, str], dict] = {}
    if out_path.exists():
        with open(out_path, newline="") as fh:
            for r in csv.DictReader(fh):
                rows[(r["frag_id"], r["config"])] = r

    for cfg in args.configs:
        done = 0
        for fid in frags:
            d = EVAL / fid / "sweep" / cfg
            wa, wb = d / "stream_A.wav", d / "stream_B.wav"
            if not (wa.exists() and wb.exists()):
                continue
            a, _ = sf.read(wa); b, _ = sf.read(wb)
            ea = _embed_windows(np.asarray(a), ecapa, device)
            eb = _embed_windows(np.asarray(b), ecapa, device)
            pure, total = _purity(ea, eb)
            rows[(fid, cfg)] = {"frag_id": fid, "config": cfg,
                                "pure": pure, "total": total}
            done += 1
        print(f"{cfg}: scored {done}/{len(frags)} fragments")

    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["frag_id", "config", "pure", "total"])
        w.writeheader()
        for key in sorted(rows):
            w.writerow(rows[key])
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
