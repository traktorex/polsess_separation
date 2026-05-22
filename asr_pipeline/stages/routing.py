"""Stage 2 — select overlap regions for SepFormer.

This stage no longer partitions the timeline into per-speaker solos. The
assembler in Stage 4 derives per-speaker solo intervals on the fly by
subtracting `ctx.overlap_regions` from `ctx.diarization.segments_df` per
speaker, so the only routing decision left is "which intervals get sent
to SepFormer".

The overlap source is `pyannote.core.Annotation.get_overlap()` (already
captured into `ctx.diarization.overlaps_df` in Stage 1). On a given
Annotation that is by construction the pairwise intersection of the
per-speaker tracks, so we trust it directly. We only:

1. Drop overlaps shorter than `min_overlap_dur` (too short to be worth
   separating — SepFormer needs at least a few hundred samples of
   meaningful overlap).
2. Merge overlap regions that sit within `merge_gap` of each other so
   SepFormer sees them as one contiguous region rather than back-to-back
   calls. Context padding (for SepFormer's training-window context) is a
   Stage 3b concern, not a Stage 2 one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pandas as pd

from asr_pipeline.config import RoutingConfig
from asr_pipeline.context import Interval, PipelineContext
from asr_pipeline.stages.base import Stage


def _merge_close(segs: List[Interval], merge_gap: float) -> List[Interval]:
    """Merge `(s, e)` intervals whose gap is below `merge_gap`."""
    if not segs:
        return []
    segs = sorted(segs)
    out = [list(segs[0])]
    for s, e in segs[1:]:
        if s - out[-1][1] < merge_gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(float(s), float(e)) for s, e in out]


def _select_overlap_regions(
    ovl_df: pd.DataFrame,
    min_overlap_dur: float,
    merge_gap: float,
) -> List[Interval]:
    """Filter + merge pyannote's overlap timeline into SepFormer-ready intervals."""
    raw = [
        (float(r.start), float(r.end))
        for r in ovl_df.itertuples()
        if r.duration >= min_overlap_dur
    ]
    return _merge_close(raw, merge_gap)


class RoutingStage(Stage):
    name = "routing"

    def __init__(self, config: RoutingConfig) -> None:
        super().__init__(enabled=config.enabled)
        self.config = config

    def run(self, ctx: PipelineContext) -> None:
        if ctx.diarization is None:
            raise RuntimeError(
                "RoutingStage.run requires ctx.diarization to be populated "
                "(DiarizationStage must run first)."
            )
        ctx.overlap_regions = _select_overlap_regions(
            ovl_df=ctx.diarization.overlaps_df,
            min_overlap_dur=self.config.min_overlap_dur,
            merge_gap=self.config.merge_gap,
        )
        ctx.speakers = sorted(ctx.diarization.segments_df["speaker"].unique())

    def spill(self, ctx: PipelineContext, artifact_dir: Path) -> None:
        if ctx.overlap_regions is None:
            return
        payload = {
            "speakers": ctx.speakers,
            "overlap_regions": [
                {"start": s, "end": e, "duration": e - s}
                for s, e in ctx.overlap_regions
            ],
        }
        with open(artifact_dir / "overlap_regions.json", "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
