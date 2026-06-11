"""Batch-run ``pipeline_noenh`` on every fragment in the CLARIN test set.

For each row in ``clarin_all_2speaker_fragments/manifest.csv``:

1. Ensure the fragment WAV exists (call ``build_clarin_fragment_set.py``'s
   extraction logic via subprocess if needed).
2. Ensure the eval-tree dir exists at ``~/datasets/eval/clarin_fragments/<frag_id>/``
   with ``mixture.wav`` symlinked to the fragment WAV.
3. Run the pipeline in ``pipeline_noenh`` mode (separator on, enhancement
   off, ``min_overlap_dur=0``, ``output_mode=full_length``,
   ``transcribe_mixture=True``).
4. Skip fragments whose ``pipeline_noenh/transcript_A.txt`` already exists
   (idempotent — safe to re-run after interruption).

Each fragment takes ~30-40 s on the GPU. Total batch ≈ 75 min for 132
fragments. Designed to be launched in the background with output redirected
to a log file.

Usage::

    python scripts/batch_pipeline_noenh.py
    python scripts/batch_pipeline_noenh.py --limit 10        # smoke test
    python scripts/batch_pipeline_noenh.py --force           # re-run all
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("HF_TOKEN", os.environ.get("HF_TOKEN", ""))

from asr_pipeline import Pipeline                            # noqa: E402
from asr_pipeline.io import write_pipeline_outputs           # noqa: E402
from scripts.run_pipeline_on_recording import _fresh_cfg     # noqa: E402


FRAGMENTS_DIR  = Path("~/datasets/clarin_all_2speaker_fragments/fragments").expanduser()
MANIFEST_PATH  = Path("~/datasets/clarin_all_2speaker_fragments/manifest.csv").expanduser()
EVAL_ROOT      = Path("~/datasets/eval/clarin_fragments").expanduser()
CFG_PATH       = REPO_ROOT / "asr_pipeline" / "configs" / "default.yaml"


def _ensure_eval_dir(frag_id: str) -> tuple[Path, Path]:
    """Create eval-tree dir + physically copy the fragment audio as
    ``<frag_id>.wav`` in it. Returns ``(rec_dir, audio_path)``.

    The audio is a physical copy (not a symlink) so ELAN's MEDIA_DESCRIPTOR
    resolves regardless of how the eval tree is shared / archived later.
    The audio's name matches the directory name so ELAN's UI shows a
    meaningful filename per fragment.
    """
    import shutil
    rec_dir = EVAL_ROOT / frag_id
    rec_dir.mkdir(parents=True, exist_ok=True)
    src = FRAGMENTS_DIR / f"{frag_id}.wav"
    dst = rec_dir / f"{frag_id}.wav"
    if not src.exists():
        raise FileNotFoundError(f"fragment WAV missing: {src}")
    # Re-copy only if the destination is missing or stale.
    needs_copy = (
        not dst.exists()
        or dst.is_symlink()                  # convert any old symlinks to real files
        or dst.stat().st_size != src.stat().st_size
    )
    if needs_copy:
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        shutil.copy2(src, dst)
    # Clean up old `mixture.wav` symlinks from the previous convention.
    old = rec_dir / "mixture.wav"
    if old.is_symlink() or old.exists():
        old.unlink()
    return rec_dir, dst


def _build_cfg():
    """pipeline_noenh: separator on, enhancement off. Eval overrides applied."""
    cfg = _fresh_cfg(CFG_PATH)
    cfg.enhancement.enabled = False
    cfg.__post_init__()
    return cfg


def _run_one(rec_dir: Path, audio_path: Path, force: bool) -> tuple[str, float]:
    """Run pipeline_noenh on one fragment. Returns (status, elapsed_s)."""
    target = rec_dir / "pipeline_noenh" / "transcript_A.txt"
    if target.exists() and not force:
        return ("skip", 0.0)

    cfg = _build_cfg()
    t0 = time.perf_counter()
    p = Pipeline(cfg)
    try:
        ctx = p.run(str(audio_path))
        write_pipeline_outputs(
            ctx, rec_dir,
            config_snapshot=asdict(cfg),
            subdir_name="pipeline_noenh",
        )
    finally:
        p.unload()
        del p
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return ("done", time.perf_counter() - t0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N manifest rows (smoke test).")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if pipeline_noenh/transcript_A.txt exists.")
    args = parser.parse_args()

    if not MANIFEST_PATH.exists():
        print(f"manifest not found: {MANIFEST_PATH}")
        print("  → run scripts/build_clarin_fragment_set.py first")
        return 1
    if not FRAGMENTS_DIR.exists() or not any(FRAGMENTS_DIR.glob("*.wav")):
        print(f"no fragment WAVs in {FRAGMENTS_DIR}")
        print("  → run scripts/build_clarin_fragment_set.py to extract them")
        return 1

    manifest = pd.read_csv(MANIFEST_PATH)
    rows = manifest if args.limit is None else manifest.head(args.limit)
    n = len(rows)

    # Check which fragment WAVs are present; bail early if any missing.
    missing = [r["frag_id"] for _, r in rows.iterrows()
               if not (FRAGMENTS_DIR / f"{r['frag_id']}.wav").exists()]
    if missing:
        print(f"{len(missing)} fragment WAV(s) missing — extract them first:")
        for m in missing[:5]:
            print(f"  - {m}")
        if len(missing) > 5:
            print(f"  - ... and {len(missing) - 5} more")
        print()
        print(f"  → python scripts/build_clarin_fragment_set.py")
        return 1

    print(f"batch starting: {n} fragment(s), force={args.force}")
    print(f"  eval root:    {EVAL_ROOT}")
    print(f"  source wavs:  {FRAGMENTS_DIR}")
    print()

    counts = {"done": 0, "skip": 0, "error": 0}
    elapsed_total = 0.0
    t_batch = time.perf_counter()

    for i, (_, row) in enumerate(rows.iterrows()):
        frag_id = row["frag_id"]
        print(f"[{i+1:>3d}/{n}] {frag_id}  ", end="", flush=True)
        try:
            rec_dir, audio_path = _ensure_eval_dir(frag_id)
            status, elapsed = _run_one(rec_dir, audio_path, args.force)
            counts[status] += 1
            elapsed_total += elapsed
            if status == "skip":
                print("skip (already done)")
            else:
                print(f"done in {elapsed:.1f}s  "
                      f"({frag_id} cell={row['overlap_bin']}/{row['noise']})")
        except Exception as e:
            counts["error"] += 1
            print(f"ERROR: {e}")
            # Continue with next fragment rather than aborting the whole batch.

    print()
    print(f"batch complete in {time.perf_counter()-t_batch:.0f}s "
          f"({(time.perf_counter()-t_batch)/60:.1f} min)")
    print(f"  done:  {counts['done']}")
    print(f"  skip:  {counts['skip']}")
    print(f"  error: {counts['error']}")
    if counts["done"]:
        print(f"  avg per fragment: {elapsed_total/counts['done']:.1f}s")
    return 0 if counts["error"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
