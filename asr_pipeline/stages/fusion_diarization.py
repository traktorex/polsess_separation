"""Disagreement-aware diarization fusion (SECOND_PASS_PLAN.md option 3).

A SECOND pyannote diarization runs on the ENHANCED audio (`ctx.enhanced_full`)
and is FUSED with the raw pass-1 result by the asymmetry rule (plan §0):

    enhancement HELPS identity (denoise → cleaner embedding) but HURTS presence
    (a single-output SE model suppresses the quieter speaker in an overlap, so a
    re-diarization on enhanced audio under-detects overlaps and drops quiet
    turns).

So PRESENCE — every segment boundary, the overlap timeline, the speaker count —
stays with the raw pass-1 diarization (`ctx.diarization`, untouched). Pass 2 is
consulted ONLY for the IDENTITY label of regions pass 1 already calls
single-speaker, and only where pass 2 *confidently* disagrees. Pass 2's own
segmentation / overlaps are DISCARDED (presence rule). The stage overwrites only
`ctx.diarization.segments_df["speaker"]` in place; `overlaps_df`, boundaries, and
the label SET are preserved (the merge maps pass-2 ids back onto the pass-1
labels, so `ctx.speakers` / `spk_to_label` set by routing stay valid — same
invariant as RelabelStage, plan §2.4).

Why a separate post-enhancement stage (not a flag inside DiarizationStage):
pass 2 needs the enhanced audio that does not exist at stage 1 (plan §7.9). So
fusion runs AFTER enhancement, reloading pyannote (one model on the GPU at a
time — DiarizationStage's pass-1 pipeline is long unloaded by then).

Default `diarization.fusion.enabled=False` → the orchestrator skips the stage
entirely: pyannote is never reloaded, `ctx` is untouched — a byte-identical
no-op.

SCOPE §4 (no silent substitution): a configured fusion with no `enhanced_full`
(enhancement disabled) is a config-time crash (config.py cross-check), backed by
a runtime assert here. A single pass-1 speaker, too-short / low-confidence
regions are visible no-ops that say so via `dlog`, never quiet downgrades.
"""

from __future__ import annotations

import gc

import torch

from asr_pipeline.config import DiarizationConfig
from asr_pipeline.context import Interval, PipelineContext
from asr_pipeline.debug_log import dlog
from asr_pipeline.stages.assembly import _coalesce, _subtract
from asr_pipeline.stages.base import Stage
from asr_pipeline.stages.diarization import (
    build_pyannote_pipeline,
    diar_to_segments_df,
    run_pyannote,
)


def _log(msg: str) -> None:
    """Progress message — stdout + the durable debug log (pyannote's pass-2
    load + forward are multi-minute; the WSL stdout bridge can drop)."""
    dlog("fusion", msg)


# ---------------------------------------------------------------------------
# Pure helpers (interval overlap / label alignment / the merge rule)
# ---------------------------------------------------------------------------


def _overlap_len(spans_a: list[Interval], spans_b: list[Interval]) -> float:
    """Total overlapping duration (s) between two interval lists.

    Both lists are treated as unions of (possibly unsorted, possibly
    overlapping) intervals. Pure / deterministic.
    """
    total = 0.0
    for a0, a1 in spans_a:
        for b0, b1 in spans_b:
            lo = max(a0, b0)
            hi = min(a1, b1)
            if hi > lo:
                total += hi - lo
    return total


def _segments_by_speaker(seg_df) -> dict[str, list[Interval]]:
    """`{speaker -> [(start, end), ...]}` from a segments frame (any pass)."""
    by_spk: dict[str, list[Interval]] = {}
    for row in seg_df.itertuples():
        by_spk.setdefault(str(row.speaker), []).append(
            (float(row.start), float(row.end))
        )
    return by_spk


def _align_pass2_to_pass1(
    pass1_by_spk: dict[str, list[Interval]],
    pass2_by_spk: dict[str, list[Interval]],
    speakers: list[str],
) -> dict[str, str]:
    """Map the (≤2) pass-2 speaker ids onto the 2 pass-1 labels by max overlap.

    A 2×2 assignment: of the two permutations of {pass2_ids} → {pass1 labels},
    pick the one whose total time-overlap (pass-2 speaker vs the pass-1 label it
    maps to) is larger. Pass 2 has its own `SPEAKER_0x` ids that must be
    reconciled to pass 1 before any label comparison (plan §5.2 step 2). Returns
    `{pass2_id -> pass1_label}`; pass-2 ids not in the chosen permutation
    (a degenerate 1- or 3-cluster pass 2) map to None.

    `speakers` are the two pass-1 labels (the alignment targets). Pure /
    deterministic — ties resolve to the "straight" permutation.
    """
    a, b = speakers[0], speakers[1]
    p2_ids = sorted(pass2_by_spk.keys())
    # Degenerate pass 2 (not exactly 2 clusters): align each pass-2 id to the
    # pass-1 label it overlaps most (no permutation forced). The caller still
    # only overrides confident single-speaker regions, so a sloppy pass 2 just
    # produces low-confidence regions that are kept.
    if len(p2_ids) != 2:
        out: dict[str, str] = {}
        for p2 in p2_ids:
            ov_a = _overlap_len(pass2_by_spk[p2], pass1_by_spk.get(a, []))
            ov_b = _overlap_len(pass2_by_spk[p2], pass1_by_spk.get(b, []))
            out[p2] = a if ov_a >= ov_b else b
        return out

    p0, p1 = p2_ids
    straight = {p0: a, p1: b}
    swapped = {p0: b, p1: a}

    def agreement(perm: dict[str, str]) -> float:
        total = 0.0
        for p2, lbl in perm.items():
            total += _overlap_len(pass2_by_spk[p2], pass1_by_spk.get(lbl, []))
        return total

    return straight if agreement(straight) >= agreement(swapped) else swapped


def fuse_labels(
    pass1_seg_df,
    pass2_seg_df,
    overlap_regions: list[Interval],
    speakers: list[str],
    confidence_min: float,
    min_region_s: float,
) -> tuple[list[str], int]:
    """The disagreement-aware merge rule (plan §5.2). Pure / deterministic.

    For each pass-1 SOLO region (a segment minus `overlap_regions`):
      1. find pass-2's label over the region by max time-overlap, ``L2``, and its
         confidence ``conf`` = (max single-speaker overlap) / (region duration);
      2. pass-2 ids are reconciled to pass-1 labels globally first
         (`_align_pass2_to_pass1`), so ``L2`` is a pass-1 label;
      3. OVERRIDE iff ``L2 != L1 and conf >= confidence_min and dur >=
         min_region_s``; otherwise keep ``L1``.

    Overlap regions (multi-speaker presence) are NEVER relabeled — their identity
    is assembly's job. A segment fully inside the overlap timeline has an empty
    solo span and keeps its pass-1 label.

    Returns ``(new_labels, n_changed)`` — ``new_labels`` is one label per
    `pass1_seg_df` row (positional), ``n_changed`` how many differ from pass 1.
    """
    blocked = _coalesce(overlap_regions) if overlap_regions else []
    pass1_by_spk = _segments_by_speaker(pass1_seg_df)
    pass2_by_spk = _segments_by_speaker(pass2_seg_df)
    p2_to_p1 = _align_pass2_to_pass1(pass1_by_spk, pass2_by_spk, speakers)

    # Pre-map each pass-2 speaker's segments to its pass-1 label so per-region
    # overlap is computed against pass-1 labels directly.
    p1label_spans: dict[str, list[Interval]] = {speakers[0]: [], speakers[1]: []}
    for p2, spans in pass2_by_spk.items():
        lbl = p2_to_p1.get(p2)
        if lbl is not None:
            p1label_spans.setdefault(lbl, []).extend(spans)

    new_labels = pass1_seg_df["speaker"].tolist()
    n_changed = 0
    for pos, row in enumerate(pass1_seg_df.itertuples()):
        l1 = str(row.speaker)
        seg = [(float(row.start), float(row.end))]
        solo = _subtract(seg, blocked) if blocked else seg
        dur = sum(e - s for s, e in solo)
        if dur <= 0.0 or dur < min_region_s:
            continue  # fully-overlapped or too short → keep pass-1 label
        # pass-2 single-speaker overlap over this region, per pass-1 label.
        ov_a = _overlap_len(solo, p1label_spans.get(speakers[0], []))
        ov_b = _overlap_len(solo, p1label_spans.get(speakers[1], []))
        if ov_a >= ov_b:
            l2, best = speakers[0], ov_a
        else:
            l2, best = speakers[1], ov_b
        conf = best / dur if dur > 0 else 0.0
        if l2 != l1 and conf >= confidence_min:
            new_labels[pos] = l2
            n_changed += 1
    return new_labels, n_changed


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------


class FusionDiarizationStage(Stage):
    """Pass-2-on-enhanced re-diarization fused into pass 1 (identity only).

    Reads `config.diarization` (the whole DiarizationConfig — it reconstructs the
    SAME pyannote pipeline pass 1 used, only with `config.fusion.embedding` as the
    pass-2 embedder) and `config.diarization.fusion` (the merge knobs).
    """

    name = "fusion_diarization"

    def __init__(self, config: DiarizationConfig) -> None:
        # Gated by the nested fusion flag, not diarization.enabled.
        super().__init__(enabled=config.fusion.enabled)
        self.config = config
        self._pipeline = None  # pass-2 pyannote, populated by load()

    # ------------------------------------------------------------------
    # Lifecycle (model-bearing; mirror DiarizationStage)
    # ------------------------------------------------------------------
    def load(self, device: torch.device) -> None:
        # Build a pass-2 pipeline identical to pass 1 except for the embedder
        # (the identity model). Shallow-copy the DiarizationConfig and swap in
        # the fusion embedding so the front-end (segmentation / clustering) knobs
        # match pass 1 exactly.
        from dataclasses import replace

        pass2_cfg = replace(self.config, embedding=self.config.fusion.embedding)
        _log(
            f"load: building pass-2 pipeline (embedding="
            f"{self.config.fusion.embedding!r}) on {device}..."
        )
        self._pipeline = build_pyannote_pipeline(pass2_cfg, device)
        _log("load: pass-2 pipeline ready")

    def unload(self) -> None:
        self._pipeline = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load_signature(self) -> tuple:
        # Pass-2 model identity: the same front-end knobs as DiarizationStage but
        # with the FUSION embedder (a model identity), plus the enabled flag.
        return (
            self.config.model_id,
            self.config.fusion.enabled,
            self.config.fusion.embedding,
            self.config.segmentation_min_duration_off,
            self.config.clustering_threshold,
            self.config.clustering_min_cluster_size,
        )

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def run(self, ctx: PipelineContext) -> None:
        _log("run: entered")
        if self._pipeline is None:
            raise RuntimeError("FusionDiarizationStage.run called before load().")
        if ctx.diarization is None:
            raise RuntimeError(
                "FusionDiarizationStage.run requires ctx.diarization "
                "(DiarizationStage must run first — it is the presence source)."
            )
        if ctx.overlap_regions is None:
            raise RuntimeError(
                "FusionDiarizationStage.run requires ctx.overlap_regions "
                "(RoutingStage must run first)."
            )
        # SCOPE §4: the config cross-check already forbids fusion-without-
        # enhancement at config time; this assert is defence-in-depth (never a
        # silent fall-back to a single pass on the raw audio).
        assert ctx.enhanced_full is not None, (
            "diarization.fusion.enabled but ctx.enhanced_full is None "
            "(enhancement did not run)."
        )

        speakers = ctx.speakers
        if len(speakers) < 2:
            _log(
                f"no-op: {len(speakers)} pass-1 speaker(s) (< 2) — keeping "
                f"pass-1 labels (won't invent identity; SCOPE §3 'count comes "
                f"from diarization')."
            )
            return

        fcfg = self.config.fusion
        _log(
            f"run: re-diarizing {len(ctx.enhanced_full)/ctx.sample_rate:.1f}s of "
            f"ENHANCED audio for pass-2 identity..."
        )
        diar2 = run_pyannote(
            self._pipeline, ctx.enhanced_full, ctx.sample_rate,
            self.config.num_speakers,
        )
        pass2_seg_df = diar_to_segments_df(diar2)
        _log(
            f"run: pass-2 produced {len(pass2_seg_df)} segment(s), "
            f"{pass2_seg_df['speaker'].nunique()} speaker(s) (presence "
            f"DISCARDED; identity only)."
        )

        seg_df = ctx.diarization.segments_df
        new_labels, n_changed = fuse_labels(
            pass1_seg_df=seg_df,
            pass2_seg_df=pass2_seg_df,
            overlap_regions=ctx.overlap_regions,
            speakers=speakers,
            confidence_min=fcfg.confidence_min,
            min_region_s=fcfg.min_region_s,
        )
        seg_df["speaker"] = new_labels
        _log(
            f"fusion: overrode {n_changed} solo-region label(s) where pass 2 "
            f"confidently (>= {fcfg.confidence_min}) disagreed (min_region_s="
            f"{fcfg.min_region_s}). Presence/overlaps unchanged."
        )
        _log("done.")
