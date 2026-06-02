"""Layer 3 — ASR error rates + WER ablation table.

For each pipeline mode that's been run (full / no-sep / no-enh), compute
cpWER + tcpWER against the per-speaker GT transcripts. The differences
between modes give the ablation: how much does separation buy, how much
does enhancement buy?

Plus ORC-WER between the mixture-baseline transcript (single-stream
Whisper on the raw mixture, written by Stage 5 when
``transcription.transcribe_mixture: true``) and the GT — the *no-pipeline*
baseline. The full chain reads::

    mixture (no pipeline)  ─→  ORC-WER vs GT
    pipeline_noenh         ─→  cpWER vs GT
    pipeline_nosep         ─→  cpWER vs GT
    pipeline (full)        ─→  cpWER vs GT

Lower is better; the spread tells you which stages earn their compute.

Caveat: the WER comparison requires both sides to use the same
transcription backend with the same prompt / args, otherwise the diff
between numbers reflects model choice, not pipeline contribution. The
pipeline writer enforces this — the mixture transcript is produced by
the same Stage 5 backend as the per-speaker transcripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from asr_pipeline.eval.metrics import (
    cpwer_meeteval,
    mimo_wer_meeteval,
    orc_wer_meeteval,
)
from asr_pipeline.eval.recordings import Recording, load_reference_utterances
from asr_pipeline.eval.transcript_parser import parse_gt_txt


def _read_per_speaker(pipeline_dir: Path) -> Optional[dict]:
    """Read ``transcript_A.txt`` + ``transcript_B.txt`` if both exist."""
    a = pipeline_dir / "transcript_A.txt"
    b = pipeline_dir / "transcript_B.txt"
    if not a.exists() or not b.exists():
        return None
    return {"A": parse_gt_txt(a), "B": parse_gt_txt(b)}


def _read_mixture(pipeline_dir: Path) -> Optional[list]:
    """Read ``transcript_mixture.txt`` if present."""
    path = pipeline_dir / "transcript_mixture.txt"
    if not path.exists():
        return None
    return parse_gt_txt(path)


def compute_layer3(rec: Recording, tcp_collar_s: float = 5.0) -> Optional[dict]:
    """ASR error rates per ablation mode + ORC-WER baseline.

    Returns None when the GT transcripts are missing — without them L3 is
    meaningless. With at least one pipeline mode populated, returns::

        {
            "ref_lengths": {"A": int, "B": int},
            "modes": {
                "full":     {"cpwer": …, "tcpwer": …, ...},
                "no_sep":   {…} or None,
                "no_enh":   {…} or None,
            },
            "mixture_orc": {"orc_wer": …, …} or None,
            "tcp_collar_s": float,
        }
    """
    ref_utts = {k: v for k, v in load_reference_utterances(rec).items() if v}
    if "A" not in ref_utts or "B" not in ref_utts:
        return None
    ref_lengths = {label: len(utts) for label, utts in ref_utts.items()}

    modes_out: dict[str, Optional[dict]] = {}
    for mode, dir_ in (
        ("full", rec.pipeline_dir),
        ("no_sep", rec.pipeline_nosep_dir),
        ("no_enh", rec.pipeline_noenh_dir),
    ):
        if dir_ is None:
            modes_out[mode] = None
            continue
        hyp = _read_per_speaker(dir_)
        if hyp is None:
            modes_out[mode] = None
            continue
        modes_out[mode] = cpwer_meeteval(
            ref_utts, hyp, session_id=rec.id, tcp_collar_s=tcp_collar_s,
        )

    # Mixture baseline (single-stream) — try the full pipeline_dir first;
    # all three ablation runs use the same backend, so transcript_mixture
    # should be identical.
    mixture_utts = None
    for d in (rec.pipeline_dir, rec.pipeline_nosep_dir, rec.pipeline_noenh_dir):
        if d is None:
            continue
        mixture_utts = _read_mixture(d)
        if mixture_utts is not None:
            break
    # Mixture floor scored two ways: ORC (time-fixed reference merge) and
    # MIMO (optimised interleaving). MIMO <= ORC always; it doesn't penalise
    # the unpredictable order Whisper interleaves the speakers in overlaps,
    # and is more robust to faulty reference timestamps (MeetEval paper,
    # Fig 1c). MIMO is the more principled single-stream floor; we keep ORC
    # too for continuity.
    mixture_orc = (
        orc_wer_meeteval(ref_utts, mixture_utts, session_id=rec.id)
        if mixture_utts is not None
        else None
    )
    mixture_mimo = (
        mimo_wer_meeteval(ref_utts, mixture_utts, session_id=rec.id)
        if mixture_utts is not None
        else None
    )

    return {
        "ref_lengths": ref_lengths,
        "modes": modes_out,
        "mixture_orc": mixture_orc,
        "mixture_mimo": mixture_mimo,
        "tcp_collar_s": tcp_collar_s,
    }
