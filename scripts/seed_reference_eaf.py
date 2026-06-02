"""Seed a hand-correctable reference EAF at the root of each fragment dir.

For every fragment under ``~/datasets/eval/clarin_fragments/<id>/``, this
writes ``<id>/annotation.eaf`` — the ground-truth working copy you open and
correct in ELAN. It is seeded from the ``pipeline_noenh`` transcripts (the
no-enhancement run we chose for GT bootstrap: the separator cleans
cross-attribution while no enhancement avoids suppressing the quieter
speaker in overlap).

The EAF lives at the fragment *root*, beside ``<id>.wav`` — separate from
``pipeline_noenh/annotation.eaf``, which stays pristine as a scored
hypothesis. The MEDIA_DESCRIPTOR's relative URL therefore points at the
sibling ``<id>.wav`` (``./``), so ELAN finds the audio next to the EAF.

**Idempotent and correction-safe**: a fragment whose ``annotation.eaf``
already exists is skipped unless ``--force`` is given. Once you've started
hand-correcting, re-running this script will never overwrite your work.

Usage::

    python scripts/seed_reference_eaf.py                # seed all, skip existing
    python scripts/seed_reference_eaf.py --limit 5      # first 5 only (pilot)
    python scripts/seed_reference_eaf.py --force         # overwrite (DANGER:
                                                         #   wipes hand-corrections)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from asr_pipeline.eval.transcript_parser import parse_gt_txt  # noqa: E402
from asr_pipeline.transcript_format import write_eaf  # noqa: E402


EVAL_ROOT = Path("~/datasets/eval/clarin_fragments").expanduser()
SEED_SUBDIR = "pipeline_noenh"


def _seed_one(frag_dir: Path, force: bool) -> str:
    """Write ``<frag_dir>/annotation.eaf``. Returns a status string."""
    out = frag_dir / "annotation.eaf"
    if out.exists() and not force:
        return "skip"

    audio = frag_dir / f"{frag_dir.name}.wav"
    if not audio.exists():
        return "no-audio"

    src = frag_dir / SEED_SUBDIR
    utts_by_speaker: dict[str, list[tuple[float, float, str]]] = {}
    for label in ("A", "B"):
        txt = src / f"transcript_{label}.txt"
        if not txt.exists():
            return "no-transcript"
        utts_by_speaker[label] = [
            (float(u.start), float(u.end), u.text) for u in parse_gt_txt(txt)
        ]

    if not any(utts_by_speaker.values()):
        return "empty"

    write_eaf(utts_by_speaker, media_path=audio, eaf_path=out, locale="pl")
    return "wrote"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--eval-root", type=Path, default=EVAL_ROOT,
                        help=f"Root holding the fragment dirs (default: {EVAL_ROOT}).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only seed the first N fragments (pilot run).")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing annotation.eaf — WIPES any "
                             "hand-corrections. Use only on un-touched seeds.")
    args = parser.parse_args()

    root = args.eval_root.expanduser()
    if not root.is_dir():
        print(f"eval root not found: {root}", file=sys.stderr)
        return 1

    frag_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if args.limit is not None:
        frag_dirs = frag_dirs[:args.limit]

    counts: dict[str, int] = {}
    for frag_dir in frag_dirs:
        status = _seed_one(frag_dir, args.force)
        counts[status] = counts.get(status, 0) + 1
        if status not in ("skip", "wrote"):
            print(f"  {frag_dir.name}: {status}")

    print(f"\nseeded {root}")
    for k in ("wrote", "skip", "no-audio", "no-transcript", "empty"):
        if k in counts:
            print(f"  {k:14s} {counts[k]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
