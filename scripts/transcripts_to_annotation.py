"""Convert pipeline transcripts to ELAN (EAF) and/or Subtitle Edit (SRT) formats.

Reads ``transcript_A.txt`` + ``transcript_B.txt`` (decimal-seconds format,
the same one ``parse_gt_txt`` reads) from a pipeline output dir and writes:

- ``annotation.eaf`` (default) — single ELAN file with two parallel tiers
  (Speaker_A, Speaker_B) referencing the recording's audio file. User sees
  both speakers on the same timeline; can adjust boundaries by dragging on
  the waveform.

- ``transcript_A.srt`` / ``transcript_B.srt`` (with ``--also-srt``) — one
  SRT per speaker for Subtitle Edit / Aegisub. Each cue prefixed with
  ``[A]`` / ``[B]`` so a later cleanup pass can split them.

Audio-file lookup: the recording_dir is expected to contain a single
``<recording_dir.name>.wav`` (the new convention; physical copy, named
after the dir). Falls back to ``mixture.wav`` for backward compatibility
with the older symlink convention.

Use::

    python scripts/transcripts_to_annotation.py <recording_dir>
    python scripts/transcripts_to_annotation.py <recording_dir> --also-srt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from asr_pipeline.eval.transcript_parser import parse_gt_txt, Utterance  # noqa: E402
from asr_pipeline.transcript_format import write_eaf as _package_write_eaf  # noqa: E402


# ---------------------------------------------------------------------------
# SRT
# ---------------------------------------------------------------------------


def _seconds_to_srt_timestamp(s: float) -> str:
    """1234.567 → '00:20:34,567'."""
    ms = int(round(s * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    sec, ms = divmod(ms, 1_000)
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"


def write_srt(utts: list[Utterance], out_path: Path, speaker_label: str) -> None:
    """Write per-speaker SRT. Speaker label included in cue text as a prefix
    so a later step (cleanup) can mark which speaker said what."""
    lines = []
    for i, u in enumerate(utts, start=1):
        if not u.text.strip():
            continue
        lines.append(str(i))
        lines.append(
            f"{_seconds_to_srt_timestamp(u.start)} --> "
            f"{_seconds_to_srt_timestamp(u.end)}"
        )
        # Speaker label as a comment-style prefix — Subtitle Edit shows it,
        # and a future txt-converter can strip it if needed.
        lines.append(f"[{speaker_label}] {u.text.strip()}")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# ELAN EAF — thin wrapper around the package writer
# ---------------------------------------------------------------------------


def write_eaf(
    utts_by_speaker: dict[str, list[Utterance]],
    media_path: Path,
    out_path: Path,
) -> None:
    """Delegate to ``asr_pipeline.transcript_format.write_eaf``.

    The package function takes plain ``(start, end, text)`` tuples; we
    convert from ``Utterance`` named-tuples here so the script's old
    ``parse_gt_txt → write_eaf`` flow keeps working.
    """
    converted = {
        spk: [(float(u.start), float(u.end), u.text) for u in utts]
        for spk, utts in utts_by_speaker.items()
    }
    _package_write_eaf(converted, media_path, out_path)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _find_audio(rec_dir: Path) -> Path | None:
    """Locate the source audio in a recording dir under the new naming
    convention (``<dir>/<dir.name>.wav``), falling back to ``mixture.wav``."""
    candidates = [rec_dir / f"{rec_dir.name}.wav", rec_dir / "mixture.wav"]
    for p in candidates:
        if p.exists():
            return p
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("recording_dir", type=Path,
                        help="e.g. ~/datasets/eval/clarin_fragments/026eafb1__seg00/")
    parser.add_argument("--subdir", default="pipeline_noenh",
                        help="Subdir under recording_dir holding transcript_*.txt "
                             "(default: pipeline_noenh)")
    parser.add_argument("--also-srt", action="store_true",
                        help="Also write per-speaker SRT files (for Subtitle Edit). "
                             "Default: EAF only.")
    args = parser.parse_args()

    rec_dir = args.recording_dir.expanduser().resolve()
    src_dir = rec_dir / args.subdir
    if not src_dir.is_dir():
        print(f"missing {src_dir}", file=sys.stderr)
        return 1

    audio = _find_audio(rec_dir)
    if audio is None:
        print(f"no audio (looked for {rec_dir.name}.wav and mixture.wav) "
              f"under {rec_dir}", file=sys.stderr)
        return 1

    utts_by_speaker: dict[str, list[Utterance]] = {}
    for label in ("A", "B"):
        txt = src_dir / f"transcript_{label}.txt"
        if not txt.exists():
            print(f"missing {txt}", file=sys.stderr)
            return 1
        utts_by_speaker[label] = parse_gt_txt(txt)

    if args.also_srt:
        for label, utts in utts_by_speaker.items():
            srt_out = src_dir / f"transcript_{label}.srt"
            write_srt(utts, srt_out, speaker_label=label)
            print(f"wrote {srt_out}  ({len(utts)} cues)")

    eaf_out = src_dir / "annotation.eaf"
    write_eaf(utts_by_speaker, audio, eaf_out)
    n_anns = sum(len([u for u in v if u.text.strip()]) for v in utts_by_speaker.values())
    print(f"wrote {eaf_out}  ({n_anns} annotations across "
          f"{len(utts_by_speaker)} tiers; media={audio.name})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
