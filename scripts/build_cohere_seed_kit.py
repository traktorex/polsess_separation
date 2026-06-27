"""Build a Cohere-seeded annotation kit for the GT-anchoring rebuttal (P1).

The existing CLARIN ground-truth was hand-corrected from a *WhisperX* seed, so a
fair worry is that the GT inherits Whisper's house style (digits, punctuation,
casing, disfluency handling) — which would flatter Whisper when it is later scored
against that GT. To measure that tilt, we hand-build a *second* GT from a **Cohere**
seed and compare.

This script mirrors `pre_annotate_fragments.py` exactly (same winning config, same
EAF writer, same aids) but swaps the transcription backend to ``coherex`` and writes
into a SEPARATE tree so the standard kit is never touched. The only difference
between the two seeds is the ASR model — every upstream stage (diarization,
separation, enhancement, assembly) is the identical deterministic config.

Per fragment, ``<out>/<fid>/`` gets:

  <fid>.wav                 a COPY of the fragment audio (eaf links to this copy)
  annotation.eaf            Cohere A/B transcript tiers, ELAN-linked to the local wav
  aids/enhanced_mixture.wav stage-3a enhanced full recording
  aids/stream_A.wav / _B    assembled per-speaker pipeline output
  aids/transcript_mixture.txt  single-pass Cohere over the raw mix (cross-reference)

Two groups (recorded in MANIFEST.csv):
  done    — already have a Whisper-seeded GT in clarin_fragments/. Correcting the
            Cohere seed here gives a PAIRED control (same audio, both GTs).
  notdone — no Whisper GT yet. Correct the Cohere seed FIRST (blind) for the
            cleanest, exposure-order-safe measurement.

Requires $COHEREX_VENV_PY (the isolated CohereX venv python). Usage:

    COHEREX_VENV_PY=~/asr_model_compare/coherex_venv/bin/python \
        python scripts/build_cohere_seed_kit.py
    # smoke one fragment:
    ... python scripts/build_cohere_seed_kit.py --limit 1
"""
from __future__ import annotations

import argparse
import csv
import gc
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from asr_pipeline import Pipeline                                          # noqa: E402
from asr_pipeline.config import load_pipeline_config_from_yaml            # noqa: E402
from asr_pipeline.transcript_format import (                              # noqa: E402
    format_transcript,
    write_eaf_from_whisper_results,
)

# Balanced spread (low/mid/high acoustic composite) of test-split fragments,
# chosen by scripts-side analysis on 2026-06-20. `done` = already Whisper-GT'd
# (paired control); `notdone` = not yet annotated (blind).
DONE = [
    "6214a217__seg00", "fadc48cc__seg00", "d851b532__cand01",
    "150d1ccc__seg00", "2bf3474d__seg01", "1f0c83da__cand01",
]
NOTDONE = [
    "3fed4030__seg00", "25a3049c__seg00", "6c6debac__seg00",
    "d1e63652__seg02", "68c67acf__seg00", "6d1d011a__seg00",
]
COHERE_MODEL = "CohereLabs/cohere-transcribe-03-2026"


def build_one(fid: str, group: str, root: Path, out: Path, config: str) -> int:
    """Run the Cohere-backed pipeline on one fragment and write its kit. Returns
    the number of seed annotations written (sum across the two speaker tiers)."""
    src_wav = root / fid / f"{fid}.wav"
    if not src_wav.exists():
        raise FileNotFoundError(src_wav)
    d = out / fid
    d.mkdir(parents=True, exist_ok=True)
    dst_wav = d / f"{fid}.wav"
    shutil.copy2(src_wav, dst_wav)  # eaf RELATIVE_MEDIA_URL points at this copy

    cfg = load_pipeline_config_from_yaml(config)
    cfg.transcription.backend = "coherex"
    cfg.transcription.model_name = COHERE_MODEL
    cfg.transcription.transcribe_mixture = True   # for the mixture-transcript aid
    cfg.__post_init__()

    p = Pipeline(cfg)
    try:
        ctx = p.run(str(dst_wav))
        psr = {ctx.spk_to_label[s]: ctx.transcripts[s]
               for s in ctx.speakers if s in ctx.transcripts}
        n = write_eaf_from_whisper_results(
            psr, media_path=dst_wav, eaf_path=d / "annotation.eaf",
            locale=cfg.transcription.language)

        aids = d / "aids"
        aids.mkdir(exist_ok=True)
        sr = ctx.sample_rate
        if getattr(ctx, "enhanced_full", None) is not None:
            sf.write(aids / "enhanced_mixture.wav",
                     np.asarray(ctx.enhanced_full, dtype=np.float32), sr)
        for s in ctx.speakers:
            lab = ctx.spk_to_label[s]
            sf.write(aids / f"stream_{lab}.wav",
                     np.asarray(ctx.assembled[s], dtype=np.float32), sr)
        if ctx.mixture_transcript:
            (aids / "transcript_mixture.txt").write_text(
                format_transcript(ctx.mixture_transcript) + "\n", encoding="utf-8")
        return n
    finally:
        try:
            p.unload()
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config",
                    default=str(REPO / "asr_pipeline/configs/sweep_best_e31_refineplus.yaml"))
    ap.add_argument("--frag-root", default="/mnt/f/clarin_fragments")
    ap.add_argument("--out", default="/mnt/f/clarin_fragments_cohere")
    ap.add_argument("--limit", type=int, default=0, help="process only the first N (smoke)")
    args = ap.parse_args()

    root = Path(args.frag_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    jobs = [(f, "done") for f in DONE] + [(f, "notdone") for f in NOTDONE]
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"Cohere-seed kit: {len(jobs)} fragments → {out}  "
          f"(config={Path(args.config).name}, model={COHERE_MODEL})\n")

    manifest_rows = []
    ok = fail = 0
    for i, (fid, group) in enumerate(jobs, 1):
        try:
            n = build_one(fid, group, root, out, args.config)
            print(f"[{i}/{len(jobs)}] {fid} ({group}): OK ({n} seed annotations)")
            manifest_rows.append({"frag_id": fid, "group": group,
                                  "seed_annotations": n, "status": "ok"})
            ok += 1
        except Exception as e:  # one bad fragment doesn't kill the batch
            print(f"[{i}/{len(jobs)}] {fid} ({group}): ERROR {type(e).__name__}: {e}")
            manifest_rows.append({"frag_id": fid, "group": group,
                                  "seed_annotations": 0, "status": f"error:{type(e).__name__}"})
            fail += 1

    with open(out / "MANIFEST.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["frag_id", "group", "seed_annotations", "status"])
        w.writeheader()
        w.writerows(manifest_rows)

    print(f"\ndone: {ok} ok, {fail} failed, of {len(jobs)}  (manifest: {out/'MANIFEST.csv'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
