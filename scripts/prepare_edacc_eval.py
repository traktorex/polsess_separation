"""Prepare EdAcc recordings under ``~/datasets/eval/edacc/``.

EdAcc (Edinburgh International Accents of English Corpus) = dyadic accented-
English conversations. Each conversation has one audio (multi-part ones —
``EAEC-C05_P0`` / ``EAEC-C05_P1`` — are separate audios, hence separate
recordings) and one reference transcript built by
``scripts/build_edacc_transcripts.py``. EdAcc ships **no per-utterance
timestamps**, so the reference is written in the *untimed* GT format
(``# untimed`` header, one utterance per line) that ``parse_gt_txt`` reads;
L3 then skips tcpWER for these recordings (no times to gate on).

Per-recording layout (mirrors ``prepare_eval_references.py``)::

    ~/datasets/eval/edacc/<recording_id>/
      <recording_id>.wav            symlink -> source audio
      reference/
        speaker_A.txt               untimed GT, speaker A utterances
        speaker_B.txt               untimed GT, speaker B utterances

No oracle audio (EdAcc has none) → no ``speaker_{A,B}.wav``, no RTTM (untimed).

The reference's ``IGNORE_TIME_SEGMENT_IN_SCORING`` utterances mark each
speaker reading the Speech Accent Archive elicitation passage ("Please call
Stella…"). The official EdAcc eval excluded those via sclite time-gating; we
have no times, so they are **dropped here from the reference** (a per-recording
count is printed — no silent drops, SCOPE §4.1) and excised from the
*hypothesis* at scoring time by ``asr_pipeline.eval.edacc``.

Idempotent: symlinks/text files are rewritten only when the source changed
(same policy as the sibling prep script). Audio↔transcript mismatches (EdAcc
has more audios than transcripts) are printed loudly at the end; the
intersection is processed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from asr_pipeline.eval.transcript_parser import format_untimed_gt  # noqa: E402


DEFAULT_EVAL_ROOT = Path.home() / "datasets" / "eval"
EDACC_AUDIO_ROOT = Path.home() / "datasets" / "EdAcc" / "edacc" / "audios"
EDACC_TRANSCRIPT_ROOT = Path.home() / "datasets" / "EdAcc" / "transcripts"

_IGNORE_MARKER = "IGNORE_TIME_SEGMENT_IN_SCORING"


def _symlink(src: Path, dst: Path) -> None:
    """Create or update a symlink from ``dst`` to ``src``. Idempotent."""
    src = src.resolve()
    if dst.is_symlink() or dst.exists():
        if dst.is_symlink() and dst.resolve() == src:
            return
        dst.unlink()
    dst.symlink_to(src)


def _write_text(dst: Path, content: str) -> None:
    """Write a small text file. Idempotent — skips if content is identical."""
    if dst.exists() and dst.read_text(encoding="utf-8") == content:
        return
    dst.write_text(content, encoding="utf-8")


def parse_edacc_transcript(
    path: Path,
) -> tuple[dict[str, list[str]], int, int]:
    """Parse one EdAcc reference transcript.

    Returns ``(by_speaker, n_ignore_dropped, n_inline_ignore)`` where
    ``by_speaker`` maps ``"A"``/``"B"`` → list of utterance texts in source
    order, with whole-utterance ``IGNORE_TIME_SEGMENT_IN_SCORING`` rows
    dropped (counted in ``n_ignore_dropped``).

    The marker is expected only as a whole utterance text. If it ever appears
    *inline* inside a longer text, that is reported loudly (``n_inline_ignore``)
    and handled by stripping the marker token while keeping the surrounding
    words — never dropping the whole utterance silently (SCOPE §4.1).

    File format (tab-separated body, ``#``-comment header)::

        utterance_idx \t speaker \t text
    """
    by_speaker: dict[str, list[str]] = {"A": [], "B": []}
    n_ignore_dropped = 0
    n_inline_ignore = 0

    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        parts = raw.split("\t")
        if len(parts) < 3:
            continue
        speaker = parts[1].strip()
        text = parts[2].strip()
        if speaker not in by_speaker:
            # Unexpected speaker label — keep it visible rather than silently
            # dropping the row (the corpus is dyadic A/B).
            print(f"  WARN {path.name}: unexpected speaker label {speaker!r} "
                  f"(row kept under that label)")
            by_speaker.setdefault(speaker, [])

        if text == _IGNORE_MARKER:
            n_ignore_dropped += 1
            continue
        if _IGNORE_MARKER in text:
            # Inline marker — the case the deliverable asks us to detect loudly.
            n_inline_ignore += 1
            print(f"  WARN {path.name}: inline {_IGNORE_MARKER} in a longer "
                  f"utterance — stripping the marker, keeping surrounding text: "
                  f"{text!r}")
            text = " ".join(t for t in text.split() if t != _IGNORE_MARKER).strip()
            if not text:
                n_ignore_dropped += 1
                continue
        by_speaker[speaker].append(text)

    return by_speaker, n_ignore_dropped, n_inline_ignore


def _intersection(
    audio_root: Path, transcript_root: Path
) -> tuple[list[str], list[str], list[str]]:
    """Recording IDs present in both, audios-without-transcript, transcripts-
    without-audio. Recording ID = file stem (so ``EAEC-C05_P0`` ≠ ``_P1``)."""
    audio_ids = {p.stem for p in audio_root.glob("*.wav")}
    transcript_ids = {p.stem for p in transcript_root.glob("*.txt")}
    common = sorted(audio_ids & transcript_ids)
    audio_only = sorted(audio_ids - transcript_ids)
    transcript_only = sorted(transcript_ids - audio_ids)
    return common, audio_only, transcript_only


def prepare_edacc(
    audio_root: Path, transcript_root: Path, eval_root: Path
) -> list[str]:
    """Prepare every EdAcc recording that has both an audio and a transcript.

    Returns the list of prepared recording IDs. Prints per-recording IGNORE
    counts and an end-of-run mismatch summary.
    """
    if not audio_root.is_dir():
        raise FileNotFoundError(f"EdAcc audios not found at {audio_root}")
    if not transcript_root.is_dir():
        raise FileNotFoundError(f"EdAcc transcripts not found at {transcript_root}")

    common, audio_only, transcript_only = _intersection(audio_root, transcript_root)

    dataset_dir = eval_root / "edacc"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    prepared: list[str] = []
    total_ignore = 0
    total_inline = 0

    for rid in common:
        audio = audio_root / f"{rid}.wav"
        transcript = transcript_root / f"{rid}.txt"

        by_speaker, n_ignore, n_inline = parse_edacc_transcript(transcript)
        total_ignore += n_ignore
        total_inline += n_inline

        rec_dir = dataset_dir / rid
        (rec_dir / "reference").mkdir(parents=True, exist_ok=True)
        _symlink(audio, rec_dir / f"{rid}.wav")
        for label in ("A", "B"):
            _write_text(
                rec_dir / "reference" / f"speaker_{label}.txt",
                format_untimed_gt(by_speaker.get(label, [])),
            )

        prepared.append(rid)
        print(f"  prepared {rid}: A={len(by_speaker.get('A', []))} "
              f"B={len(by_speaker.get('B', []))} utts, "
              f"dropped {n_ignore} IGNORE utterance(s)")

    print(f"\nprepared {len(prepared)} recording(s); "
          f"dropped {total_ignore} IGNORE utterance(s) total.")
    if total_inline:
        print(f"WARN: {total_inline} INLINE {_IGNORE_MARKER} occurrence(s) "
              f"handled (marker stripped, text kept) — see warnings above.")

    if audio_only:
        print(f"\nWARN: {len(audio_only)} audio(s) with NO transcript "
              f"(skipped): {audio_only}")
    if transcript_only:
        print(f"WARN: {len(transcript_only)} transcript(s) with NO audio "
              f"(skipped): {transcript_only}")

    return prepared


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--audio-root", type=Path, default=EDACC_AUDIO_ROOT,
        help=f"EdAcc audios dir (default: {EDACC_AUDIO_ROOT}).",
    )
    parser.add_argument(
        "--transcript-root", type=Path, default=EDACC_TRANSCRIPT_ROOT,
        help=f"EdAcc transcripts dir (default: {EDACC_TRANSCRIPT_ROOT}).",
    )
    parser.add_argument(
        "--eval-root", type=Path, default=DEFAULT_EVAL_ROOT,
        help=f"Output eval tree root (default: {DEFAULT_EVAL_ROOT}).",
    )
    args = parser.parse_args()

    print(f"edacc: audios={args.audio_root} transcripts={args.transcript_root} "
          f"eval_root={args.eval_root}")
    prepared = prepare_edacc(args.audio_root, args.transcript_root, args.eval_root)
    if not prepared:
        print("nothing prepared.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
