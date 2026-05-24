"""Metrics for the three-layer evaluation.

- **Layer 1 (DER)** — `compute_der`, thin wrapper around
  `pyannote.metrics.DiarizationErrorRate`. The metric does its own
  optimal speaker assignment via the Hungarian solver; we don't
  pre-permute.
- **Layer 2 (separation)** — no helper here; the notebook uses
  `torchmetrics.functional.audio.*` directly for SI-SDR / PESQ-WB /
  STOI (intrusive) and `torchaudio.pipelines.SQUIM_OBJECTIVE` for the
  non-intrusive estimates. Both libraries are already used elsewhere
  in the project (evaluate.py, asr/evaluate_asr.py), so we stay
  consistent with them rather than rolling our own.
- **Layer 3 (cpWER / tcpWER)** — `cpwer_meeteval`, wraps the MeetEval
  package's cpWER + tcpWER. Permutation, alignment, and time-constrained
  scoring all live inside MeetEval — the CHiME-7/8 evaluation toolkit.

We strip punctuation and lowercase before scoring (preserving Polish
diacritics), since both Whisper and the pipeline emit casing /
punctuation that doesn't reflect actual ASR errors. MeetEval's built-in
normalizers either over-strip (`lower,rm([^a-z0-9 ])` removes Polish
letters) or under-strip (`lower,rm(.?!,)` misses `;:—…`), so we
pre-normalize the SegLST `words` field ourselves.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

from asr_pipeline.eval.transcript_parser import Utterance


# ---------------------------------------------------------------------------
# Layer 1 — DER (diarization error rate)
# ---------------------------------------------------------------------------


def compute_der(
    ref_segments: Dict[str, List[Tuple[float, float]]],
    hyp_segments: Dict[str, List[Tuple[float, float]]],
    total_duration_s: float,
    collar: float = 0.0,
    skip_overlap: bool = False,
) -> Dict[str, float]:
    """DER + miss / false-alarm / confusion breakdown.

    `ref_segments`, `hyp_segments` map each speaker label to a list of
    (start, end) tuples (seconds). DER is reported as a fraction of the
    reference speech duration; multiply by 100 for a percent.

    `collar` (seconds) optionally forgives boundary mismatches within
    ±collar/2 of each reference segment edge.
    """
    from pyannote.core import Annotation, Segment
    from pyannote.metrics.diarization import DiarizationErrorRate

    def _to_annotation(per_spk):
        ann = Annotation()
        for spk, segs in per_spk.items():
            for s, e in segs:
                if e > s:
                    ann[Segment(s, e)] = spk
        return ann

    ref = _to_annotation(ref_segments)
    hyp = _to_annotation(hyp_segments)
    uem = Segment(0.0, total_duration_s)

    metric = DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap)
    detailed = metric.compute_components(ref, hyp, uem=uem)
    total = max(detailed.get("total", 1.0), 1e-9)
    miss = detailed.get("missed detection", 0.0)
    fa = detailed.get("false alarm", 0.0)
    conf = detailed.get("confusion", 0.0)
    return {
        "der": (miss + fa + conf) / total,
        "miss": miss / total,
        "false_alarm": fa / total,
        "confusion": conf / total,
        "total_ref_s": float(total),
        "collar": collar,
        "skip_overlap": skip_overlap,
    }


# ---------------------------------------------------------------------------
# Layer 3 — cpWER / tcpWER via MeetEval
# ---------------------------------------------------------------------------

# Strip ASCII + Unicode punctuation Whisper actually emits in Polish output.
# Keeps Polish letters (ą ć ę ł ń ó ś ź ż) because `\w` in Python's `re` is
# Unicode-aware by default and matches them.
_PUNCT_RE = re.compile(r"[^\w\s]+", flags=re.UNICODE)


def _normalize_text(s: str) -> str:
    """Lowercase + strip punctuation + collapse whitespace.

    Preserves Polish diacritics (phonemic — `ł` vs `l` is a real
    substitution and should count as a WER error).
    """
    return " ".join(_PUNCT_RE.sub(" ", s.lower()).split())


def cpwer_meeteval(
    ref_utts_by_spk: Dict[str, List[Utterance]],
    hyp_utts_by_spk: Dict[str, List[Utterance]],
    session_id: str,
    tcp_collar_s: float = 5.0,
) -> Dict[str, object]:
    """cpWER + tcpWER via MeetEval, with Polish-aware text normalization.

    Inputs are dicts mapping speaker label → list of
    `Utterance(start, end, text)` (as produced by
    `parse_transcript_file` and `parse_gt_txt`).

    Returns a dict with::

        {
            "cpwer": float,                  # 0..1 fraction
            "cp_assignment": tuple,          # (ref_spk, hyp_spk) pairs
            "tcpwer": float,
            "tcp_assignment": tuple,
            "cp_errors": int, "cp_length": int,
            "tcp_errors": int, "tcp_length": int,
        }

    `tcp_collar_s` is the per-word time tolerance for tcpWER (CHiME-7
    default is 5.0). MeetEval places each word at its segment midpoint
    by default — fine for our use since GT segments come from Whisper's
    own segmentation.
    """
    from meeteval.io.seglst import SegLST
    from meeteval.wer import cpwer, tcpwer

    def _to_seglst(utts_by_spk: Dict[str, List[Utterance]]) -> SegLST:
        return SegLST([
            {
                "session_id": session_id,
                "speaker": spk,
                "start_time": float(u.start),
                "end_time": float(u.end),
                "words": _normalize_text(u.text),
            }
            for spk, utts in utts_by_spk.items()
            for u in utts
            if u.text.strip()
        ])

    ref = _to_seglst(ref_utts_by_spk)
    hyp = _to_seglst(hyp_utts_by_spk)

    cp = cpwer(ref, hyp)[session_id]
    tcp = tcpwer(ref, hyp, collar=tcp_collar_s)[session_id]

    return {
        "cpwer": float(cp.error_rate),
        "cp_assignment": tuple(cp.assignment),
        "cp_errors": int(cp.errors),
        "cp_length": int(cp.length),
        "tcpwer": float(tcp.error_rate),
        "tcp_assignment": tuple(tcp.assignment),
        "tcp_errors": int(tcp.errors),
        "tcp_length": int(tcp.length),
        "tcp_collar_s": float(tcp_collar_s),
    }
