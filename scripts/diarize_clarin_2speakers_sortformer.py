"""Sortformer diarization of every CLARIN 2-speaker recording + overlap census.

Sortformer twin of ``diarize_clarin_2speakers.py`` (pyannote). Two steps:

1. Raw per-frame speaker activity via ``scripts/sortformer_batch_worker.py`` in
   the isolated NeMo venv (``$SORTFORMER_VENV_PY``, cwd outside the repo — see
   that script), one model load for the whole corpus. All recordings but one
   exceed the pipeline's 240 s long-audio threshold, so the streaming v2.1 model
   with its offline preset is used throughout (the same routing the pipeline
   applies to long input). Probs land in ``<out>/probs/<id>.npz``.
2. Turn building with the pipeline's OWN adapter
   (``sortformer_turns_from_probs`` @ threshold 0.5, top-2 heads, no L1 merge)
   → ``<out>/<id>.json`` in the ``{total_duration_s, segments, overlaps}`` schema
   of ``diarization/`` (what ``clarin_fragment_finder`` reads), plus
   ``<out>/overlap_summary.csv`` — one row per recording with overlap / speech
   statistics, the Sortformer head-miscount diagnostics, the pyannote overlap
   for cross-check, and Korpus metadata (self-reported noise level / type,
   environment, device), and two cheap objective noisiness proxies:
   WADA-SNR on the speech-active audio and the speech-vs-nonspeech level gap.

Usage:
    source venv/bin/activate
    SORTFORMER_VENV_PY=~/sortformer_venv/bin/python \
        python scripts/diarize_clarin_2speakers_sortformer.py [--skip-worker]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from asr_pipeline.stages.diarization import (          # noqa: E402
    _overlaps_df_from_annotation,
    build_sortformer_annotation,
    diar_to_segments_df,
    sortformer_turns_from_probs,
)
from score_fragment_acoustics import wada_snr          # noqa: E402

INPUT_DIR = Path.home() / "datasets/clarin_all_2speakers/clarin_download"
PYANNOTE_DIR = Path.home() / "datasets/clarin_all_2speakers/diarization"
OUTPUT_DIR = Path.home() / "datasets/clarin_all_2speakers/diarization_sortformer"
MODEL_ID = "nvidia/diar_streaming_sortformer_4spk-v2.1"
THRESHOLD = 0.5          # DiarizationConfig.sortformer_threshold default
SR = 16_000
KORPUS_COLS = {
    "Nazwa": "name", "Długość": "length_meta", "Poziom szumów": "noise_level",
    "Typ szumów": "noise_type", "Środowisko": "environment",
    "Urządzenie nagrywające": "device", "Typ mikrofonu": "mic",
    "Domena": "domain",
}


def run_worker(venv_py: str, probs_dir: Path) -> None:
    worker = REPO_ROOT / "scripts" / "sortformer_batch_worker.py"
    cmd = [venv_py, str(worker), "--in-dir", str(INPUT_DIR),
           "--out-dir", str(probs_dir), "--model", MODEL_ID]
    with tempfile.TemporaryDirectory() as td:   # neutral cwd, outside the repo
        proc = subprocess.run(cmd, cwd=td)
    if proc.returncode != 0:
        raise RuntimeError(f"sortformer batch worker exited {proc.returncode}")


def frame_mask(intervals, n_frames: int, fs: float) -> np.ndarray:
    m = np.zeros(n_frames, dtype=bool)
    for a, b in intervals:
        m[int(a / fs): int(np.ceil(b / fs))] = True
    return m


def noise_proxies(wav: Path, speech_iv, nonspeech_iv):
    """(wada_snr_speech_db, speech_minus_nonspeech_level_db). Both read the whole
    file once; the level gap is RMS(speech frames) / RMS(non-speech frames) in dB —
    a small gap means a loud background relative to the talkers."""
    x, sr = sf.read(str(wav), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    assert sr == SR, (wav, sr)

    def _concat(ivs, cap_s=600.0):
        parts, tot = [], 0.0
        for a, b in ivs:
            parts.append(x[int(a * sr): int(b * sr)])
            tot += b - a
            if tot >= cap_s:
                break
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)

    sp = _concat(speech_iv)
    ns = _concat(nonspeech_iv)
    wada = wada_snr(sp) if sp.size else float("nan")
    if sp.size and ns.size:
        rms = lambda v: float(np.sqrt(np.mean(v.astype(np.float64) ** 2)) + 1e-9)
        gap = 20 * np.log10(rms(sp) / rms(ns))
    else:
        gap = float("nan")
    return wada, gap


def postprocess(probs_dir: Path, out_dir: Path, meta: dict) -> list[dict]:
    rows = []
    npzs = sorted(p for p in probs_dir.glob("*.npz") if ".tmp" not in p.name)
    for i, f in enumerate(npzs, 1):
        rid = f.stem
        z = np.load(f)
        probs, fs = z["frame_probs"], float(z["frame_rate_s"])
        turns, diag = sortformer_turns_from_probs(probs, fs, THRESHOLD)
        ann = build_sortformer_annotation(turns)
        seg_df = diar_to_segments_df(ann)
        ovl_df = _overlaps_df_from_annotation(ann)
        wav = INPUT_DIR / f"{rid}.wav"
        total = sf.info(str(wav)).frames / SR
        with open(out_dir / f"{rid}.json", "w") as fh:
            json.dump({"total_duration_s": total,
                       "segments": seg_df.to_dict(orient="records"),
                       "overlaps": ovl_df.to_dict(orient="records")},
                      fh, indent=2, ensure_ascii=False)

        top2 = diag["top2"]
        head_dur = np.asarray(diag["head_dur_s"])
        speech = float(diag["speech_s"])
        overlap = float(ovl_df["duration"].sum()) if len(ovl_df) else 0.0
        d0, d1 = head_dur[top2[0]], head_dur[top2[1]]
        balance = float(min(d0, d1) / max(d0, d1, 1e-9))

        # speech / non-speech intervals from the top-2 runs (frame grid)
        n = probs.shape[0]
        sp_mask = frame_mask([(a, b) for _, a, b in turns], n, fs)
        idx = np.flatnonzero(np.diff(np.r_[0, sp_mask.astype(int), 0]))
        speech_iv = [(s * fs, e * fs) for s, e in zip(idx[::2], idx[1::2])]
        idx = np.flatnonzero(np.diff(np.r_[0, (~sp_mask).astype(int), 0]))
        nonspeech_iv = [(s * fs, e * fs) for s, e in zip(idx[::2], idx[1::2])
                        if (e - s) * fs >= 1.0]   # ≥1 s gaps only
        wada, gap = noise_proxies(wav, speech_iv, nonspeech_iv)

        py_ovl = float("nan")
        pj = PYANNOTE_DIR / f"{rid}.json"
        if pj.exists():
            with open(pj) as fh:
                py_ovl = sum(o["duration"] for o in json.load(fh)["overlaps"])

        row = {
            "id": rid,
            "duration_s": round(total, 1),
            "speech_s": round(speech, 1),
            "overlap_s": round(overlap, 1),
            "overlap_frac_speech": round(overlap / speech, 4) if speech else 0.0,
            "overlap_frac_duration": round(overlap / total, 4),
            "overlap_min_per_10min": round(overlap / total * 600 / 60, 3),
            "n_overlap_regions": int(len(ovl_df)),
            "overlap_median_s": round(float(ovl_df["duration"].median()), 3)
                                if len(ovl_df) else 0.0,
            "speaker_balance": round(balance, 3),
            "n_spk_heads": int(diag["n_spk"]),
            "leak_frac": round(float(diag["leak"]), 4),
            "pyannote_overlap_s": round(py_ovl, 1),
            "wada_snr_db": round(wada, 1),
            "speech_nonspeech_gap_db": round(gap, 1),
        }
        row.update(meta.get(f"{rid}.wav", {v: "" for v in KORPUS_COLS.values()}))
        rows.append(row)
        print(f"({i}/{len(npzs)}) {rid}: {total:.0f}s, overlap {overlap:.0f}s "
              f"({row['overlap_frac_speech']:.1%} of speech), n_spk={diag['n_spk']}, "
              f"leak {diag['leak']:.1%}, WADA {wada:.1f} dB, gap {gap:.1f} dB",
              flush=True)
    return rows


def load_korpus() -> dict:
    meta = {}
    with open(INPUT_DIR / "Korpus_with_filename.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            meta[r["Nazwa pliku WAV"]] = {v: r.get(k, "") for k, v in KORPUS_COLS.items()}
    return meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-worker", action="store_true",
                    help="probs already computed; only post-process")
    a = ap.parse_args()
    probs_dir = OUTPUT_DIR / "probs"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not a.skip_worker:
        venv_py = os.environ.get("SORTFORMER_VENV_PY")
        if not venv_py or not Path(venv_py).exists():
            print("error: set $SORTFORMER_VENV_PY to the NeMo venv python",
                  file=sys.stderr)
            return 1
        run_worker(venv_py, probs_dir)
    rows = postprocess(probs_dir, OUTPUT_DIR, load_korpus())
    rows.sort(key=lambda r: -r["overlap_frac_speech"])
    out_csv = OUTPUT_DIR / "overlap_summary.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_csv} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
