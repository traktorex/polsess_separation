"""Discovery + dataclass for recordings in the eval tree.

The eval tree (built by ``scripts/prepare_eval_references.py``) puts every
recording under ``<eval_root>/<dataset>/<recording_id>/`` with a fixed
layout. This module walks the tree and packages each recording as a
``Recording`` dataclass the layers (L1/L2/L3) consume.

Missing optional files are tolerated — the eval layers gracefully skip
work they can't do (e.g. no oracle audio → only non-intrusive SQUIM in L2).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from asr_pipeline.eval.transcript_parser import Utterance, parse_eaf, parse_gt_txt


@dataclass
class Recording:
    """One recording in the eval tree.

    `pipeline_dir` / `pipeline_nosep_dir` / `pipeline_noenh_dir` correspond
    to the three ablation modes — only populated when the pipeline has
    actually been run for that mode (the writer creates the directory).

    `reference_eaf` is the hand-corrected GT (ELAN .eaf at the recording
    root) — the source of truth for both L1 turns and L3 transcripts when
    present; `reference_transcripts` / `reference_diarization` are the
    fallback for datasets prepared as .txt + .rttm.
    """

    id: str
    dataset: str
    mixture_path: Path
    reference_audio: Optional[dict[str, Path]]      # speaker_label -> wav
    reference_transcripts: dict[str, Path]          # speaker_label -> txt
    reference_diarization: Optional[Path]           # rttm
    reference_eaf: Optional[Path]                   # hand-corrected GT (.eaf)
    pipeline_dir: Optional[Path]                    # full pipeline
    pipeline_nosep_dir: Optional[Path]              # separation.enabled=false
    pipeline_noenh_dir: Optional[Path]              # enhancement.enabled=false


def _dir_if_present(p: Path) -> Optional[Path]:
    return p if p.is_dir() else None


def _file_if_present(p: Path) -> Optional[Path]:
    return p if p.exists() else None


def load_recording(recording_dir: Path) -> Optional[Recording]:
    """Build a Recording from one directory in the eval tree.

    Returns None if the directory doesn't have the minimum required
    contents (mixture.wav). Otherwise populates all available fields.
    """
    if not recording_dir.is_dir():
        return None
    # Audio: new convention is `<dir>/<dir.name>.wav` (physical copy named
    # after the dir); fall back to the older `mixture.wav` symlink layout.
    mixture = recording_dir / f"{recording_dir.name}.wav"
    if not mixture.exists():
        mixture = recording_dir / "mixture.wav"
    if not mixture.exists():
        return None

    ref_dir = recording_dir / "reference"
    ref_audio: dict[str, Path] = {}
    ref_transcripts: dict[str, Path] = {}
    if ref_dir.is_dir():
        for label in ("A", "B"):
            wav = ref_dir / f"speaker_{label}.wav"
            txt = ref_dir / f"speaker_{label}.txt"
            if wav.exists():
                ref_audio[label] = wav
            if txt.exists():
                ref_transcripts[label] = txt
    ref_rttm = _file_if_present(ref_dir / "diarization.rttm") if ref_dir.is_dir() else None

    # Hand-corrected GT EAF lives at the recording root, beside the audio.
    ref_eaf = _file_if_present(recording_dir / "annotation.eaf")

    # dataset is the parent directory's name (one level up from recording_id)
    dataset = recording_dir.parent.name

    return Recording(
        id=recording_dir.name,
        dataset=dataset,
        mixture_path=mixture,
        reference_audio=ref_audio if ref_audio else None,
        reference_transcripts=ref_transcripts,
        reference_diarization=ref_rttm,
        reference_eaf=ref_eaf,
        pipeline_dir=_dir_if_present(recording_dir / "pipeline"),
        pipeline_nosep_dir=_dir_if_present(recording_dir / "pipeline_nosep"),
        pipeline_noenh_dir=_dir_if_present(recording_dir / "pipeline_noenh"),
    )


def load_reference_utterances(rec: "Recording") -> dict[str, list[Utterance]]:
    """GT per-speaker utterances for one recording.

    Prefers the hand-corrected EAF (source of truth); falls back to the
    per-speaker ``reference/speaker_{A,B}.txt`` files. Empty tiers are kept
    here and dropped by the callers that need non-empty A/B.
    """
    if rec.reference_eaf is not None:
        return parse_eaf(rec.reference_eaf)
    return {label: parse_gt_txt(p) for label, p in rec.reference_transcripts.items()}


def walk_eval_tree(
    eval_root: Path, dataset: Optional[str] = None
) -> Iterator[Recording]:
    """Yield every Recording found under ``<eval_root>/<dataset>/*``.

    When ``dataset`` is given, restrict to that one. Without it, walk every
    dataset directory below ``eval_root``. Recordings without ``mixture.wav``
    are silently skipped.
    """
    eval_root = Path(eval_root).expanduser()
    if not eval_root.is_dir():
        return
    if dataset is not None:
        roots = [eval_root / dataset]
    else:
        roots = [p for p in eval_root.iterdir() if p.is_dir()]
    for ds_dir in sorted(roots):
        if not ds_dir.is_dir():
            continue
        for rec_dir in sorted(ds_dir.iterdir()):
            rec = load_recording(rec_dir)
            if rec is not None:
                yield rec


def parse_rttm(path: Path) -> dict[str, list[tuple[float, float]]]:
    """Parse an RTTM file into ``{speaker: [(start_s, end_s), ...]}``.

    RTTM rows::

        SPEAKER <file_id> 1 <start_s> <duration_s> <NA> <NA> <speaker_label> <NA> <NA>

    Other row types (e.g. ``SPKR-INFO``) are skipped.
    """
    out: dict[str, list[tuple[float, float]]] = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        parts = raw.split()
        if not parts or parts[0] != "SPEAKER" or len(parts) < 8:
            continue
        start = float(parts[3])
        dur = float(parts[4])
        speaker = parts[7]
        out.setdefault(speaker, []).append((start, start + dur))
    return out
