"""Prepare per-recording eval references under ``~/datasets/eval/``.

Source datasets stay untouched; we materialise a uniform per-recording
layout that the eval module (``asr_pipeline.eval``) reads. Symlinks for
audio (cheap, keeps disk usage low), copies for small text files.

Per-recording layout::

    ~/datasets/eval/<dataset>/<recording_id>/
      mixture.wav                   symlink -> source mixture
      reference/
        speaker_A.wav               symlink -> source oracle channel (when available)
        speaker_B.wav               symlink -> source oracle channel
        speaker_A.txt               GT transcript, decimal-seconds format (parse_gt_txt)
        speaker_B.txt               GT transcript
        diarization.rttm            derived from the GT transcripts (true GT, not proxy)

The pipeline's own outputs land under ``pipeline/``, ``pipeline_nosep/``,
``pipeline_noenh/`` — produced separately by
``scripts/run_pipeline_on_recording.py``.

Datasets supported:

- ``clarin``: CLARIN debleed under ``~/datasets/clarin_gotowy/gotowy/``.
  Mixture = ``<id>.wav``, oracles = ``debleed/<id>_{L,R}.wav``,
  GT = ``true_transcripts/<id>_{L,R}.txt`` (decimal-seconds). Only
  recordings with both ``_L`` and ``_R`` transcripts are prepared.
  L → speaker A, R → speaker B.

- ``libricss``: LibriCSS 2-spk slim under ``~/datasets/LibriCSS_2spk/``.
  Driven by ``manifest.csv``. Mixture = ``record/segments/<key>.wav``,
  oracles = ``clean/segments/<key>_spk{A,B}.wav``, GT =
  ``transcriptions/<key>.txt`` (tab-separated, A/B-labelled). The
  transcript is split per-speaker into the same decimal-seconds format
  used for CLARIN so ``parse_gt_txt`` reads both.

Idempotent: existing files / symlinks are overwritten only if their
source has changed (mtime newer than dest).
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Iterable, NamedTuple


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from asr_pipeline.eval.transcript_parser import parse_gt_txt  # noqa: E402


DEFAULT_EVAL_ROOT = Path.home() / "datasets" / "eval"
CLARIN_ROOT = Path.home() / "datasets" / "clarin_gotowy" / "gotowy"
LIBRICSS_ROOT = Path.home() / "datasets" / "LibriCSS_2spk"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


class Turn(NamedTuple):
    """One speaker turn — start / end / label."""
    speaker: str    # "A" or "B"
    start: float
    end: float


def _symlink(src: Path, dst: Path) -> None:
    """Create or update a symlink from `dst` to `src`. Idempotent."""
    src = src.resolve()
    if dst.is_symlink() or dst.exists():
        if dst.is_symlink() and dst.resolve() == src:
            return
        dst.unlink()
    dst.symlink_to(src)


def _copy_text(src: Path, dst: Path) -> None:
    """Copy a small text file. Idempotent — skips if content is identical."""
    new = src.read_text(encoding="utf-8")
    if dst.exists() and dst.read_text(encoding="utf-8") == new:
        return
    dst.write_text(new, encoding="utf-8")


def _write_decimal_transcript(
    out_path: Path, utterances: list[tuple[float, float, str]]
) -> None:
    """Write utterances as ``[ s.cc → s.cc] text`` lines (parse_gt_txt format)."""
    lines = [
        f"[{start:6.2f} → {end:6.2f}]  {text}"
        for start, end, text in utterances
    ]
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _write_rttm(out_path: Path, recording_id: str, turns: Iterable[Turn]) -> None:
    """Write turns as RTTM (NIST diarization format).

    One ``SPEAKER`` line per turn::

        SPEAKER <file_id> 1 <start> <duration> <NA> <NA> <speaker> <NA> <NA>
    """
    lines = []
    for t in turns:
        dur = max(0.0, t.end - t.start)
        if dur <= 0:
            continue
        lines.append(
            f"SPEAKER {recording_id} 1 {t.start:.3f} {dur:.3f} "
            f"<NA> <NA> {t.speaker} <NA> <NA>"
        )
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLARIN
# ---------------------------------------------------------------------------


def prepare_clarin(source_root: Path, eval_root: Path) -> list[str]:
    """Prepare CLARIN recordings that have hand-corrected GT transcripts.

    Returns the list of prepared recording IDs.
    """
    true_transcripts_dir = source_root / "true_transcripts"
    if not true_transcripts_dir.exists():
        raise FileNotFoundError(
            f"CLARIN true_transcripts/ not found under {source_root}"
        )

    # Recording IDs that have both L and R transcripts.
    candidates: dict[str, dict[str, Path]] = {}
    for txt in true_transcripts_dir.glob("*.txt"):
        m = re.match(r"^(?P<id>.+)_(?P<channel>[LR])\.txt$", txt.name)
        if not m:
            continue
        candidates.setdefault(m["id"], {})[m["channel"]] = txt
    complete = {rid: paths for rid, paths in candidates.items() if "L" in paths and "R" in paths}

    if not complete:
        print(f"  no recordings with both L and R GT transcripts under {true_transcripts_dir}")
        return []

    dataset_dir = eval_root / "clarin"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    prepared: list[str] = []

    for rid, txt_paths in sorted(complete.items()):
        rec_dir = dataset_dir / rid
        (rec_dir / "reference").mkdir(parents=True, exist_ok=True)

        # Mixture + oracle channels (symlinks).
        mixture = source_root / f"{rid}.wav"
        l_oracle = source_root / "debleed" / f"{rid}_L.wav"
        r_oracle = source_root / "debleed" / f"{rid}_R.wav"
        if not mixture.exists():
            print(f"  skip {rid}: mixture {mixture} missing")
            continue
        if not l_oracle.exists() or not r_oracle.exists():
            print(f"  skip {rid}: debleed channels missing")
            continue
        _symlink(mixture, rec_dir / "mixture.wav")
        _symlink(l_oracle, rec_dir / "reference" / "speaker_A.wav")
        _symlink(r_oracle, rec_dir / "reference" / "speaker_B.wav")

        # Transcripts (copy verbatim — already in decimal-seconds format).
        _copy_text(txt_paths["L"], rec_dir / "reference" / "speaker_A.txt")
        _copy_text(txt_paths["R"], rec_dir / "reference" / "speaker_B.txt")

        # Diarization (derived from GT transcripts — true GT, not proxy).
        turns: list[Turn] = []
        for spk, label in (("L", "A"), ("R", "B")):
            for u in parse_gt_txt(txt_paths[spk]):
                turns.append(Turn(speaker=label, start=u.start, end=u.end))
        turns.sort(key=lambda t: t.start)
        _write_rttm(rec_dir / "reference" / "diarization.rttm", rid, turns)

        prepared.append(rid)
        print(f"  prepared {rid}: {len(turns)} reference turns")

    return prepared


# ---------------------------------------------------------------------------
# LibriCSS
# ---------------------------------------------------------------------------


_LIBRICSS_HEADER_RE = re.compile(r"^#")


def _parse_libricss_transcript(
    path: Path,
) -> tuple[list[tuple[float, float, str]], list[tuple[float, float, str]]]:
    """Parse a LibriCSS transcript file.

    Returns ``(speaker_A_utterances, speaker_B_utterances)`` where each
    list contains ``(start, end, text)`` tuples in source order.

    File format (one utterance per non-comment line, tab-separated)::

        start_s  end_s  speaker  utterance_id  text
    """
    a_utts: list[tuple[float, float, str]] = []
    b_utts: list[tuple[float, float, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or _LIBRICSS_HEADER_RE.match(line):
            continue
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        start = float(parts[0])
        end = float(parts[1])
        speaker = parts[2].strip()
        text = parts[4].strip()
        if speaker == "A":
            a_utts.append((start, end, text))
        elif speaker == "B":
            b_utts.append((start, end, text))
        else:
            # Should not happen for the 2-spk subset, but skip if it does.
            continue
    return a_utts, b_utts


def prepare_libricss(source_root: Path, eval_root: Path) -> list[str]:
    """Prepare every segment listed in LibriCSS_2spk/manifest.csv."""
    manifest_path = source_root / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"LibriCSS manifest not found at {manifest_path}"
        )

    dataset_dir = eval_root / "libricss"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    prepared: list[str] = []

    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = row["key"]
            rec_dir = dataset_dir / rid
            (rec_dir / "reference").mkdir(parents=True, exist_ok=True)

            mixture = source_root / row["record_path"]
            clean_a = source_root / row["clean_a_path"]
            clean_b = source_root / row["clean_b_path"]
            transcript = source_root / row["transcript_path"]

            if not mixture.exists() or not transcript.exists():
                print(f"  skip {rid}: missing files")
                continue

            _symlink(mixture, rec_dir / "mixture.wav")
            if clean_a.exists():
                _symlink(clean_a, rec_dir / "reference" / "speaker_A.wav")
            if clean_b.exists():
                _symlink(clean_b, rec_dir / "reference" / "speaker_B.wav")

            a_utts, b_utts = _parse_libricss_transcript(transcript)
            _write_decimal_transcript(rec_dir / "reference" / "speaker_A.txt", a_utts)
            _write_decimal_transcript(rec_dir / "reference" / "speaker_B.txt", b_utts)

            turns = [
                Turn(speaker="A", start=s, end=e) for s, e, _ in a_utts
            ] + [
                Turn(speaker="B", start=s, end=e) for s, e, _ in b_utts
            ]
            turns.sort(key=lambda t: t.start)
            _write_rttm(rec_dir / "reference" / "diarization.rttm", rid, turns)

            prepared.append(rid)
    print(f"  prepared {len(prepared)} LibriCSS segment(s)")
    return prepared


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dataset", required=True, choices=["clarin", "libricss"],
        help="Source dataset to prepare.",
    )
    parser.add_argument(
        "--source-root", type=Path, default=None,
        help="Override source dataset path (defaults to dataset-specific).",
    )
    parser.add_argument(
        "--eval-root", type=Path, default=DEFAULT_EVAL_ROOT,
        help=f"Output eval tree root (default: {DEFAULT_EVAL_ROOT}).",
    )
    args = parser.parse_args()

    if args.dataset == "clarin":
        source_root = args.source_root or CLARIN_ROOT
        print(f"clarin: source={source_root} eval_root={args.eval_root}")
        prepared = prepare_clarin(source_root, args.eval_root)
    elif args.dataset == "libricss":
        source_root = args.source_root or LIBRICSS_ROOT
        print(f"libricss: source={source_root} eval_root={args.eval_root}")
        prepared = prepare_libricss(source_root, args.eval_root)
    else:
        raise ValueError(args.dataset)

    if not prepared:
        print("nothing prepared.")
        return 1
    print(f"\nprepared {len(prepared)} recording(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
