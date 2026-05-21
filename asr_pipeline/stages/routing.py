"""Stage 2 — partition the timeline into {silence, solo-X, overlap}.

Ported from `asr/asr_pipeline.ipynb` cell 12. Pure timeline arithmetic on
the diarization output: filter short overlaps, merge same-speaker turns
across small gaps, pad, subtract overlap from per-speaker regions to get
solos, fill the rest with silence.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd

from asr_pipeline.config import RoutingConfig
from asr_pipeline.context import PipelineContext
from asr_pipeline.stages.base import Stage


Interval = Tuple[float, float]


def _merge_consecutive(segs: List[Interval], max_gap: float) -> List[Interval]:
    """Merge consecutive `(s, e)` intervals whose gap is below `max_gap`."""
    if not segs:
        return []
    segs = sorted(segs)
    out = [list(segs[0])]
    for s, e in segs[1:]:
        if s - out[-1][1] < max_gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [tuple(x) for x in out]


def _subtract(
    regions_a: List[Interval], regions_b: List[Interval]
) -> List[Interval]:
    """Set difference `regions_a − regions_b` on intervals."""
    result: List[Interval] = []
    regions_b = sorted(regions_b)
    for s, e in sorted(regions_a):
        current: List[Interval] = [(s, e)]
        for bs, be in regions_b:
            next_cur: List[Interval] = []
            for cs, ce in current:
                if be <= cs or bs >= ce:
                    next_cur.append((cs, ce))
                    continue
                if bs > cs:
                    next_cur.append((cs, bs))
                if be < ce:
                    next_cur.append((be, ce))
            current = next_cur
        result.extend(current)
    return [(s, e) for s, e in result if e - s > 1e-3]


def _build_partition(
    seg_df: pd.DataFrame,
    ovl_df: pd.DataFrame,
    total_dur: float,
    min_overlap: float,
    max_gap: float,
    pad: float,
) -> Tuple[pd.DataFrame, List[str]]:
    """Partition the timeline into {silence, solo-A, solo-B, overlap}."""
    if len(seg_df) == 0:
        return (
            pd.DataFrame(columns=["start", "end", "duration", "kind", "speaker"]),
            [],
        )

    ovl_raw = [
        (r.start, r.end)
        for r in ovl_df.itertuples()
        if r.duration >= min_overlap
    ]
    ovl_padded = _merge_consecutive(
        [(max(0.0, s - pad), min(total_dur, e + pad)) for s, e in ovl_raw],
        1e-6,
    )

    speakers = sorted(seg_df["speaker"].unique())
    spk_regions = {}
    for spk in speakers:
        segs = [
            (r.start, r.end)
            for r in seg_df[seg_df["speaker"] == spk].itertuples()
        ]
        segs = _merge_consecutive(segs, max_gap)
        segs = [(max(0.0, s - pad), min(total_dur, e + pad)) for s, e in segs]
        segs = _merge_consecutive(segs, 1e-6)
        spk_regions[spk] = segs

    partition = []
    for i, spk in enumerate(speakers):
        tag = f"solo-{chr(ord('A') + i)}"
        solos = _subtract(spk_regions[spk], ovl_padded)
        for s, e in solos:
            partition.append(
                {
                    "start": s,
                    "end": e,
                    "duration": e - s,
                    "kind": tag,
                    "speaker": spk,
                }
            )
    for s, e in ovl_padded:
        partition.append(
            {
                "start": s,
                "end": e,
                "duration": e - s,
                "kind": "overlap",
                "speaker": None,
            }
        )

    active = _merge_consecutive(
        [(r["start"], r["end"]) for r in partition], 1e-6
    )
    for s, e in _subtract([(0.0, total_dur)], active):
        partition.append(
            {
                "start": s,
                "end": e,
                "duration": e - s,
                "kind": "silence",
                "speaker": None,
            }
        )

    part_df = (
        pd.DataFrame(partition).sort_values("start").reset_index(drop=True)
    )
    return part_df, speakers


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
        part_df, speakers = _build_partition(
            seg_df=ctx.diarization.segments_df,
            ovl_df=ctx.diarization.overlaps_df,
            total_dur=ctx.diarization.total_duration_s,
            min_overlap=self.config.min_overlap_dur,
            max_gap=self.config.max_merge_gap,
            pad=self.config.segment_pad,
        )
        ctx.partition_df = part_df
        ctx.speakers = speakers

    def spill(self, ctx: PipelineContext, artifact_dir: Path) -> None:
        if ctx.partition_df is None:
            return
        ctx.partition_df.to_csv(artifact_dir / "partition.csv", index=False)
