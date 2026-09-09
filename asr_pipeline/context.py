"""Pipeline context — the dataclass that accumulates stage outputs.

A `PipelineContext` is created at the start of a run, populated stage by
stage, and returned as the result. All fields are optional (default
`None` or empty container) so callers can inspect a partial context if a
stage fails or is disabled.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, NotRequired, Optional, Tuple, TypedDict

import numpy as np
import pandas as pd


Interval = Tuple[float, float]


class OverlapSeparated(TypedDict):
    """One entry in `ctx.overlap_separated` — the central 3b → 3c → 4 contract.

    Stays a plain dict at runtime (TypedDict is a type-checker hint only).
    Notebook code that does ``ovl["s1_gated"]`` keeps working.

    Fields are populated incrementally:
      - Stage 3b (``SeparationStage``) constructs the dict with all fields
        EXCEPT ``s1_gated`` / ``s2_gated``.
      - Stage 3c (``PostSeparationProcessingStage``) writes ``s1_gated`` and
        ``s2_gated`` after applying the VAD mask + optional BWE.
    """

    idx: int
    start: float                        # raw overlap interval (stage 2)
    end: float
    pad_start: float                    # padded window picked by stage 3b
    pad_end: float
    emit_start: float                   # emit region (seam adjustments applied)
    emit_end: float
    chunked: bool
    volume_scale: float
    mix: np.ndarray                     # padded mixture fed to the separator
    s1_raw: np.ndarray                  # unmasked separator output (stream 1)
    s2_raw: np.ndarray                  # unmasked separator output (stream 2)
    mask1: np.ndarray                   # silero VAD mask on s1_raw
    mask2: np.ndarray
    probs1: np.ndarray                  # per-frame VAD probabilities for s1
    probs2: np.ndarray
    # Populated by Stage 3c — assembly reads these:
    s1_gated: NotRequired[np.ndarray]   # post-BWE * mask1
    s2_gated: NotRequired[np.ndarray]


@dataclass
class DiarizationResult:
    """Output of Stage 1 — pyannote diarization."""

    segments_df: pd.DataFrame   # columns: start, end, duration, speaker
    overlaps_df: pd.DataFrame   # columns: start, end, duration
    total_duration_s: float


@dataclass
class TimestampMapEntry:
    """One piece of one speaker's assembled stream.

    `concat_*` are timestamps inside the assembled per-speaker WAV; `orig_*`
    are the corresponding timestamps in the input recording. In `shortened`
    mode the two ranges have the same duration but offset/translated; in
    `full_length` mode they coincide exactly.
    """

    concat_start: float
    concat_end: float
    orig_start: float
    orig_end: float
    kind: str   # "solo" | "overlap"


@dataclass
class TimestampMap:
    """Per-speaker mapping from assembled-stream time to original-recording time.

    The weak-anchor diagnostic lives on `PipelineContext.weak_anchor` (the
    single source of truth read by `io.py` and the notebooks); it is not
    duplicated here.
    """

    per_speaker: Dict[str, List[TimestampMapEntry]] = field(default_factory=dict)


@dataclass
class PipelineContext:
    """Accumulator passed through the pipeline; final value = result."""

    # Input
    input_path: Optional[Path] = None
    audio: Optional[np.ndarray] = None   # mono, float32, at `config.sample_rate`
    sample_rate: int = 16_000

    # Stage 1 — diarization
    diarization: Optional[DiarizationResult] = None
    # Stage 1 diagnostics (currently populated only by the sortformer backend):
    # the head-miscount / L1-merge / L3-L4-fallback census fields, written verbatim
    # into metadata.json by io.write_pipeline_outputs so a later census can count
    # v4.1 lever firings. None on the pyannote path.
    diarization_diag: Optional[Dict[str, Any]] = None

    # Stage 2 — routing
    # List of (start_s, end_s) overlap intervals (unpadded). SepFormer's
    # context window is applied independently in Stage 3b.
    overlap_regions: Optional[List[Interval]] = None
    speakers: List[str] = field(default_factory=list)

    # Stage 3a — full-recording enhancement (single enhancer pass).
    # Same length as `ctx.audio`; sliced per-speaker at assembly time.
    enhanced_full: Optional[np.ndarray] = None

    # Overlap separation + post-processing. Each entry is an
    # `OverlapSeparated` (TypedDict above) — schema defined there. Stage 4
    # (assembly) reads `s_gated` only, so it doesn't matter to it which 3c
    # backend produced them — just that 3c ran.
    overlap_separated: List[OverlapSeparated] = field(default_factory=list)

    # RelabelStage (B+, `relabel.source='global'`) handoff — written between 3c
    # and assembly, consumed by assembly via the consensus-injection seam.
    # `{i_ovl -> "straight"|"swapped"}`, keyed by the DENSE list index of
    # `overlap_separated` (NOT the routing-region `idx`, which can have gaps),
    # matching the convention `_assign_overlaps`/`_consensus_pairings` already
    # use. For each covered overlap assembly takes the global-clustering pairing
    # in place of its own per-overlap anchor decision; overlaps B+ could not
    # decide (sub-min / leaked / degenerate) are simply absent → assembly falls
    # back to its own ladder (loudly logged). None (default) = no B+ handoff,
    # assembly behaves exactly as today.
    overlap_speaker_assignment: Optional[Dict[int, str]] = None

    # Stage 4 — assembly
    # key: speaker label from pyannote; value: 1-D float32 array
    assembled: Dict[str, np.ndarray] = field(default_factory=dict)
    # Mapping from short labels ("A", "B") back to pyannote speaker ids
    spk_to_label: Dict[str, str] = field(default_factory=dict)
    timestamp_map: Optional[TimestampMap] = None
    weak_anchor: bool = False
    # Stage 4 diagnostics — one compact, JSON-safe record per assigned overlap:
    # `{idx, pairing, cos_straight, cos_swapped}`, where `idx` is the
    # routing-region index and the cosine sums are None whenever the pairing did
    # not come from an ECAPA argmax (B+ handoff, too-short / weak-anchor
    # fallback, non-finite cosine). Written verbatim into metadata.json by
    # io.write_pipeline_outputs, like `diarization_diag`. None when assembly
    # didn't run or ran in no-separation mode (no per-overlap decision taken).
    assembly_diag: Optional[List[Dict[str, Any]]] = None

    # Stage 5 — transcription
    # key: speaker label from pyannote; value: Whisper result dict
    transcripts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Stage 5 — single-stream baseline on the whole mixture (populated only
    # when `transcription.transcribe_mixture: true`). Same shape as one
    # entry in `transcripts`.
    mixture_transcript: Optional[Dict[str, Any]] = None
