"""Layer 1 — Diarization Error Rate (DER).

Hypothesis: the pipeline's **stage-1** diarization (``pipeline/diarization.json``).
This is what the pipeline actually claimed — what drove routing, separation
windowing, and assembly downstream.

Reference: the per-recording diarization RTTM produced by
``scripts/prepare_eval_references.py``. For both CLARIN and LibriCSS the
RTTM is derived from the hand-corrected GT transcripts (true GT — one
turn per utterance, A/B labelled).
"""

from __future__ import annotations

import json
from typing import Optional

from asr_pipeline.eval.metrics import compute_der
from asr_pipeline.eval.recordings import (
    Recording,
    load_reference_utterances,
    parse_rttm,
)


def _load_pipeline_diarization(
    pipeline_dir,
) -> Optional[dict[str, list[tuple[float, float]]]]:
    """Read pipeline/diarization.json → ``{speaker: [(start, end), ...]}``."""
    path = pipeline_dir / "diarization.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    hyp: dict[str, list[tuple[float, float]]] = {}
    for turn in data.get("turns", []):
        hyp.setdefault(turn["speaker"], []).append(
            (float(turn["start"]), float(turn["end"]))
        )
    return hyp


def _total_duration(
    ref_segments: dict[str, list[tuple[float, float]]],
    pipeline_dir,
) -> float:
    """Total duration to use as the DER denominator's universe.

    Prefer the value the pipeline recorded in `diarization.json`; fall
    back to the latest end timestamp across all reference turns.
    """
    if pipeline_dir is not None:
        path = pipeline_dir / "diarization.json"
        if path.exists():
            data = json.loads(path.read_text())
            recorded_duration = data.get("total_duration_s")
            if recorded_duration is not None:
                return float(recorded_duration)
    latest = 0.0
    for turns in ref_segments.values():
        for _, end in turns:
            latest = max(latest, end)
    return latest


def _reference_turns(
    rec: Recording,
) -> Optional[dict[str, list[tuple[float, float]]]]:
    """Reference speaker turns for DER.

    Prefer the hand-corrected GT EAF (each annotation is a timed turn);
    fall back to the prepared RTTM. Returns None when neither exists.
    """
    if rec.reference_eaf is not None:
        utts = load_reference_utterances(rec)
        turns = {
            label: [(u.start, u.end) for u in u_list]
            for label, u_list in utts.items()
            if u_list
        }
        # A present-but-empty reference (all tiers empty) must read as absent,
        # not as a zero-turn reference: scoring against it would divide by
        # compute_der's 1e-9 floor and emit a ~5e9 DER that poisons aggregates.
        return turns or None
    if rec.reference_diarization is not None:
        return parse_rttm(rec.reference_diarization) or None
    return None


def compute_layer1(rec: Recording, collar: float = 0.0) -> Optional[dict]:
    """DER + breakdown for one recording.

    Returns None when L1 cannot be computed (no reference turns, or no
    pipeline diarization output). Otherwise::

        {
            "der_stage1": {"der": …, "miss": …, "false_alarm": …, "confusion": …,
                           "total_ref_s": …, "collar": …, "skip_overlap": …},
            "reference_source": "eaf" | "rttm",
        }

    ``skip_overlap`` is pinned ``False`` at the call site (overlap regions are
    scored) — it is reported for provenance, not exposed as a knob.
    """
    ref = _reference_turns(rec)
    if ref is None:
        return None
    # Stage-1 diarization is mode-independent — use whichever mode dir exists.
    pdir = rec.pipeline_dir or rec.pipeline_noenh_dir or rec.pipeline_nosep_dir
    if pdir is None:
        return None
    hyp = _load_pipeline_diarization(pdir)
    if hyp is None:
        return None
    total_dur = _total_duration(ref, pdir)
    der = compute_der(ref, hyp, total_dur, collar=collar, skip_overlap=False)
    return {
        "der_stage1": der,
        "reference_source": "eaf" if rec.reference_eaf is not None else "rttm",
    }
