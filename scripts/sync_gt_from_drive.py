"""Sync hand-corrected GT EAFs from the Windows F: working copy into the WSL
eval tree, so the eval/lint/sweep can actually see them.

The corrections are done in ELAN on ``/mnt/f/clarin_fragments/<id>/annotation.eaf``
(faster than the \\wsl.localhost network mount); completion is tracked in
``robione.txt`` there. This copies each *genuinely corrected* drive EAF onto
``~/datasets/eval/clarin_fragments/<id>/annotation.eaf``.

Safety: a drive EAF is copied only when its text differs from the noenh seed
(so an un-edited drive copy never clobbers a correction already in the eval
tree). Lines marked ``remove`` in robione.txt are skipped. The EAF's media
path isn't rewritten — scoring (``parse_eaf``) reads only the tiers, so it's
irrelevant; keep editing on F:.

Usage::

    python scripts/sync_gt_from_drive.py            # sync done fragments, then lint
    python scripts/sync_gt_from_drive.py --all      # sync every corrected drive EAF (ignore robione.txt)
    python scripts/sync_gt_from_drive.py --no-lint
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from asr_pipeline.eval.transcript_parser import parse_eaf  # noqa: E402

DRIVE = Path("/mnt/f/clarin_fragments")
EVAL = Path("~/datasets/eval/clarin_fragments").expanduser()


def _text(eaf: Path) -> str:
    t = parse_eaf(eaf)
    return " ".join(u.text.strip() for lab in sorted(t) for u in t[lab]).lower()


def _done_ids(drive: Path) -> list[str]:
    f = drive / "robione.txt"
    if not f.exists():
        return []
    ids = []
    for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
        if "done" in line and "remove" not in line.lower():
            tok = line.split()[0].strip()
            if "__seg" in tok:
                ids.append(tok)
    return ids


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--all", action="store_true",
                    help="Sync every corrected drive EAF, not just robione.txt 'done' ones.")
    ap.add_argument("--no-lint", action="store_true")
    args = ap.parse_args()

    if not DRIVE.exists():
        print(f"drive copy not found: {DRIVE}", file=sys.stderr)
        return 1

    if args.all:
        ids = sorted(p.name for p in DRIVE.iterdir() if p.is_dir() and (p / "annotation.eaf").exists())
    else:
        ids = _done_ids(DRIVE)
        print(f"robione.txt: {len(ids)} fragment(s) marked done")

    synced = kept = skipped = 0
    for fid in ids:
        f_eaf = DRIVE / fid / "annotation.eaf"
        ev_dir = EVAL / fid
        seed = ev_dir / "pipeline_noenh" / "annotation.eaf"
        dst = ev_dir / "annotation.eaf"
        if not f_eaf.exists():
            print(f"  {fid}: SKIP (no drive EAF)"); skipped += 1; continue
        if not ev_dir.is_dir():
            print(f"  {fid}: SKIP (no eval dir)"); skipped += 1; continue
        f_corr = not (seed.exists() and _text(f_eaf) == _text(seed))
        if f_corr:
            shutil.copy2(f_eaf, dst)
            print(f"  {fid}: synced"); synced += 1
        else:
            wsl_corr = dst.exists() and seed.exists() and _text(dst) != _text(seed)
            if wsl_corr:
                print(f"  {fid}: drive==seed; KEEP existing WSL correction"); kept += 1
            else:
                print(f"  {fid}: drive==seed and WSL uncorrected — nothing to sync"); skipped += 1

    print(f"\nsynced: {synced}  |  kept-WSL: {kept}  |  skipped: {skipped}")
    if not args.no_lint:
        print("\n--- lint ---")
        import subprocess
        subprocess.run([sys.executable, str(REPO / "scripts" / "check_gt_eaf.py"), "--errors-only"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
