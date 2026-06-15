"""Pre-annotate Not-Done fragments with the winning config + annotator aids.

For each Not-Done test fragment (manifest.csv: Subset=test, Done="Not Done") that
has a dir under --frag-root, run the winning pipeline config and write into the
fragment dir, to give the hand-annotator a head start:

  annotation.eaf            fresh A/B transcript tiers (winning config), ELAN-linked
                            to <id>.wav via RELATIVE_MEDIA_URL (existing eaf backed
                            up to annotation.eaf.bak first).
  aids/enhanced_mixture.wav stage-3a enhanced full recording (clean audio to hear)
  aids/stream_A.wav / _B    assembled per-speaker pipeline output (each speaker alone)
  aids/transcript_mixture.txt  single-pass WhisperX over the raw mix (cross-reference)

The annotator opens annotation.eaf in ELAN (audio auto-links) and corrects the two
tiers, consulting the aids. Dev/Done fragments are never touched.

Usage:
    python scripts/pre_annotate_fragments.py            # all 84 Not-Done test frags
    python scripts/pre_annotate_fragments.py --limit 1  # smoke one
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


def select_fragments(manifest: Path, root: Path) -> list[str]:
    rows = list(csv.DictReader(open(manifest, encoding="utf-8")))
    sel = [r["frag_id"] for r in rows
           if r.get("Subset") == "test" and r.get("Done") == "Not Done"]
    return [f for f in sel if (root / f / f"{f}.wav").exists()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(REPO / "asr_pipeline/configs/sweep_best_e31.yaml"))
    ap.add_argument("--manifest", default="/mnt/f/clarin_fragments/manifest.csv")
    ap.add_argument("--frag-root", default="/mnt/f/clarin_fragments")
    ap.add_argument("--limit", type=int, default=0, help="process only the first N (smoke test)")
    ap.add_argument("--force", action="store_true", help="re-process fragments already done")
    args = ap.parse_args()

    root = Path(args.frag_root)
    fids = select_fragments(Path(args.manifest), root)
    if args.limit:
        fids = fids[: args.limit]
    print(f"pre-annotating {len(fids)} Not-Done test fragments  (config={Path(args.config).name})\n")

    ok = fail = 0
    for i, fid in enumerate(fids, 1):
        d = root / fid
        wav = d / f"{fid}.wav"
        bak = d / "annotation.eaf.bak"
        if bak.exists() and not args.force:   # resumable: .bak marks "already done"
            print(f"[{i}/{len(fids)}] {fid}: skip (done)")
            ok += 1
            continue
        p = None
        try:
            # fresh cfg per fragment (matches the sweep harness; transcribe_mixture
            # on for the mixture-transcript aid).
            cfg = load_pipeline_config_from_yaml(args.config)
            cfg.transcription.transcribe_mixture = True
            cfg.__post_init__()

            p = Pipeline(cfg)
            ctx = p.run(str(wav))

            eaf = d / "annotation.eaf"
            if eaf.exists() and not bak.exists():   # preserve the ORIGINAL seed only
                shutil.copy2(eaf, bak)
            psr = {ctx.spk_to_label[s]: ctx.transcripts[s]
                   for s in ctx.speakers if s in ctx.transcripts}
            n = write_eaf_from_whisper_results(
                psr, media_path=wav, eaf_path=eaf, locale=cfg.transcription.language)

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

            print(f"[{i}/{len(fids)}] {fid}: OK ({n} annotations, {len(ctx.speakers)} tiers)")
            ok += 1
        except Exception as e:  # SCOPE §4.2 — one bad fragment doesn't kill the batch
            print(f"[{i}/{len(fids)}] {fid}: ERROR {type(e).__name__}: {e}")
            fail += 1
        finally:
            if p is not None:
                try:
                    p.unload()
                except Exception:
                    pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\ndone: {ok} ok, {fail} failed, of {len(fids)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
