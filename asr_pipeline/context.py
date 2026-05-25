"""Pipeline context — the dataclass that accumulates stage outputs.

A `PipelineContext` is created at the start of a run, populated stage by
stage, and returned as the result. All fields are optional (default
`None` or empty container) so callers can inspect a partial context if a
stage fails or is disabled.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


Interval = Tuple[float, float]


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
    """Per-speaker mapping from assembled-stream time to original-recording time."""

    weak_anchor: bool
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

    # Stage 2 — routing
    # List of (start_s, end_s) overlap intervals (unpadded). SepFormer's
    # context window is applied independently in Stage 3b.
    overlap_regions: Optional[List[Interval]] = None
    speakers: List[str] = field(default_factory=list)

    # Stage 3a — full-recording enhancement (single MP-SENet pass).
    # Same length as `ctx.audio`; sliced per-speaker at assembly time.
    enhanced_full: Optional[np.ndarray] = None

    # Overlap separation + post-processing — one dict per overlap region.
    # Populated incrementally:
    #   - Stage 3b (separation) writes: start, end, pad_*, emit_*, mix,
    #     s1_raw, s2_raw, mask1, mask2, probs1, probs2, chunked, volume_scale
    #   - Stage 3c (post_separation_processing) writes: s1_gated, s2_gated
    # Stage 4 (assembly) reads `_gated` only, so it doesn't matter to it
    # which backend produced them — just that 3c ran.
    overlap_separated: List[Dict[str, Any]] = field(default_factory=list)

    # Stage 4 — assembly
    # key: speaker label from pyannote; value: 1-D float32 array
    assembled: Dict[str, np.ndarray] = field(default_factory=dict)
    # Mapping from short labels ("A", "B") back to pyannote speaker ids
    spk_to_label: Dict[str, str] = field(default_factory=dict)
    timestamp_map: Optional[TimestampMap] = None
    weak_anchor: bool = False

    # Stage 5 — transcription
    # key: speaker label from pyannote; value: Whisper result dict
    transcripts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
