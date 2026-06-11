"""Run-level tests for Stage 3b (SeparationStage) with stub models.

Follows the `test_pipeline_assembly_run.py` pattern: stub separator + stub
VAD injected directly (no `load()`, which would pull the real checkpoint
and silero via torch.hub). Covers the wiring the pure-helper suite in
`test_pipeline_separation.py` does not: `run()` orchestration, the
3b→3c/4 entry contract, overlap-add chunking + PIT propagation, the
emit-seam round trip into assembly, and `spill()` provenance.
"""

import json
from dataclasses import fields

import numpy as np
import pandas as pd
import pytest
import torch

from asr_pipeline.config import SeparationConfig
from asr_pipeline.context import DiarizationResult, OverlapSeparated, PipelineContext
from asr_pipeline.stages.assembly import _slice_emit
from asr_pipeline.stages.separation import (
    SeparationStage,
    _separate_overlap_add,
    _Window,
)

SR = 16_000
DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _IdentitySeparator(torch.nn.Module):
    """Returns the mixture as both streams: [1, T] → [1, 2, T]."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([x, x], dim=1)


class _AlternatingSeparator(torch.nn.Module):
    """Returns (x, −x), swapping stream order on every call.

    Exercises the PIT chunk alignment: without the swap correction the
    reconstructed stream would flip sign mid-overlap.
    """

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        if self.calls % 2 == 1:
            return torch.stack([x, -x], dim=1)
        return torch.stack([-x, x], dim=1)


class _StubVad:
    """All-speech VAD: probability 1.0 for every 512-sample frame."""

    def reset_states(self) -> None:
        return None

    def __call__(self, chunk: torch.Tensor, sr: int) -> torch.Tensor:
        return torch.tensor(1.0)


def _make_stage(separator=None, **config_kwargs) -> SeparationStage:
    stage = SeparationStage(SeparationConfig(**config_kwargs))
    stage._separator = separator or _IdentitySeparator()
    stage._vad = _StubVad()
    stage._device = DEVICE
    return stage


def _make_ctx(duration_s: float, overlap_regions) -> PipelineContext:
    ctx = PipelineContext(sample_rate=SR)
    rng = np.random.default_rng(0)
    ctx.audio = rng.standard_normal(int(duration_s * SR)).astype(np.float32) * 0.1
    seg_df = pd.DataFrame([], columns=["start", "end", "duration", "speaker"])
    ovl_df = pd.DataFrame(columns=["start", "end", "duration"])
    ctx.diarization = DiarizationResult(
        segments_df=seg_df, overlaps_df=ovl_df, total_duration_s=duration_s
    )
    ctx.overlap_regions = list(overlap_regions)
    return ctx


# ---------------------------------------------------------------------------
# run() guards
# ---------------------------------------------------------------------------


def test_run_before_load_raises():
    stage = SeparationStage(SeparationConfig())
    with pytest.raises(RuntimeError, match="before load"):
        stage.run(_make_ctx(10.0, [(1.0, 2.0)]))


def test_run_requires_upstream_context():
    stage = _make_stage()
    ctx = PipelineContext(sample_rate=SR)  # no audio / regions / diarization
    with pytest.raises(RuntimeError, match="requires"):
        stage.run(ctx)


# ---------------------------------------------------------------------------
# Entry contract (3b → 3c → 4)
# ---------------------------------------------------------------------------


def test_entry_schema_matches_overlap_separated_contract():
    """3c and 4 consume `OverlapSeparated` dicts that 3c partially extends.
    Nothing else pins that 3b actually produces the required keys."""
    stage = _make_stage()
    ctx = _make_ctx(20.0, [(8.0, 9.0)])
    stage.run(ctx)
    assert len(ctx.overlap_separated) == 1
    entry = ctx.overlap_separated[0]
    # All required keys present; the NotRequired *_gated keys absent (3c's job).
    assert set(entry) == set(OverlapSeparated.__required_keys__)
    assert "s1_gated" not in entry
    # Padded window arrays all share the window's length.
    n = len(entry["mix"])
    assert len(entry["s1_raw"]) == n
    assert len(entry["s2_raw"]) == n
    assert len(entry["mask1"]) == n
    # Emit region sits inside the padded window.
    assert entry["pad_start"] <= entry["emit_start"] <= entry["emit_end"] <= entry["pad_end"] + 1e-9


def test_min_overlap_skip_leaves_nonconsecutive_idx():
    """Regions under _MIN_OVERLAP_SAMPLES are skipped; idx keeps indexing
    ctx.overlap_regions, so surviving entries carry their original index."""
    stage = _make_stage()
    ctx = _make_ctx(20.0, [(1.0, 1.005), (8.0, 9.0)])  # 0.005 s < 256 samples
    stage.run(ctx)
    assert [e["idx"] for e in ctx.overlap_separated] == [1]


def test_chunked_flag_reports_actual_chunking():
    """`chunked` must mean "overlap-add actually split the window", not
    "the overlap-add branch ran": with chunk length above the window size,
    _separate_overlap_add falls back to a single forward."""
    # Window ~13 s (> threshold 12) but chunk 20 s → no actual chunking.
    stage = _make_stage(
        context_window_mode="none",
        training_chunk_length_s=20.0,
        overlap_add_threshold_s=12.0,
    )
    ctx = _make_ctx(60.0, [(10.0, 23.0)])
    stage.run(ctx)
    assert ctx.overlap_separated[0]["chunked"] is False

    # Same window with chunk 4 s → genuine chunking.
    stage = _make_stage(
        context_window_mode="none",
        training_chunk_length_s=4.0,
        overlap_add_threshold_s=12.0,
    )
    ctx = _make_ctx(60.0, [(10.0, 23.0)])
    stage.run(ctx)
    assert ctx.overlap_separated[0]["chunked"] is True


# ---------------------------------------------------------------------------
# Emit-seam round trip into assembly (regression for the one-sample drift)
# ---------------------------------------------------------------------------


def test_emit_seam_round_trips_to_chosen_sample():
    """3b picks a seam index in the padded window, converts it to seconds;
    assembly's _slice_emit converts back to an index. The round trip must
    land on the same sample even when pad_start_s * sr is fractional
    (it used to come back one sample early)."""
    stage = _make_stage(seam_mode="zero_crossing", seam_search_radius_s=0.05)
    n = SR  # 1 s padded window
    for frac_samples in (16000.5, 31999.25, 7999.75):
        pad_start_s = frac_samples / SR
        window = _Window(pad_start_s=pad_start_s, pad_end_s=pad_start_s + 1.0)
        # Sign change between 7999 and 8000 → zero crossing near the middle.
        combined = np.ones(n, dtype=np.float32)
        combined[8000:] = -1.0
        overlap_start_s = pad_start_s + 0.5
        emit_start_s, emit_end_s = stage._pick_emit_region(
            window=window,
            overlap_start_s=overlap_start_s,
            overlap_end_s=pad_start_s + 0.9,
            combined_audio=combined,
            combined_vad_mask=np.ones(n, dtype=np.float32),
            sample_rate=SR,
        )
        # Reproduce assembly's slicing math on a gated stream of window length.
        gated = np.arange(n, dtype=np.float32)
        piece = _slice_emit(gated, pad_start_s, emit_start_s, emit_end_s, SR)
        start_idx = int(piece[0])
        # 3b chose the zero crossing at 8000 (sign change reported there);
        # assembly must slice from exactly that sample.
        assert start_idx == 8000, (
            f"pad_start={frac_samples}: assembly sliced from {start_idx}, "
            f"3b chose 8000"
        )


def test_adjacent_emit_regions_never_cross():
    """Under snap_to_silence, two adjacent overlaps' emit regions could
    cross (different VAD masks drive each decision). The run() guard must
    clamp the second region's start to the first's end."""
    stage = _make_stage(
        context_window_mode="none",
        seam_mode="snap_to_silence",
        snap_silence_max_extend_s=5.0,     # exaggerate the outward extension
        seam_search_radius_s=0.05,
        vad_threshold=0.99,                # stub VAD prob 1.0 > 0.99 → all speech
    )
    # Two overlaps 0.5 s apart (routing's merge_gap floor).
    ctx = _make_ctx(30.0, [(5.0, 9.5), (10.0, 14.0)])
    stage.run(ctx)
    assert len(ctx.overlap_separated) == 2
    first, second = ctx.overlap_separated
    assert second["emit_start"] >= first["emit_end"] - 1e-9
    assert second["emit_start"] <= second["emit_end"]


# ---------------------------------------------------------------------------
# Overlap-add: identity reconstruction + PIT propagation
# ---------------------------------------------------------------------------


def _edge_exclusion(chunk_samples: int) -> int:
    # Samples whose symmetric-Hann weight falls below the 1e-6 clamp at the
    # stream's outermost edges: w(k) ≈ (πk/(N−1))² < 1e-3 → k < (N−1)·1e-3/π...
    # measured bound used by the reviewers: int(1e-3 * (chunk_samples - 1) / np.pi) + 2.
    return int(1e-3 * (chunk_samples - 1) / np.pi) + 2


def test_overlap_add_identity_reconstruction():
    """With an identity separator and no resampling, overlap-add must
    reconstruct the mixture except the documented edge attenuation."""
    rng = np.random.default_rng(1)
    mix = rng.standard_normal(20 * SR).astype(np.float32) * 0.1
    s1, s2 = _separate_overlap_add(
        mix, _IdentitySeparator(), DEVICE,
        sr_pipeline=SR, sr_separator=SR, chunk_length_s=4.0,
    )
    edge = _edge_exclusion(4 * SR)
    np.testing.assert_allclose(s1[edge:-edge], mix[edge:-edge], atol=1e-4)
    np.testing.assert_allclose(s2[edge:-edge], mix[edge:-edge], atol=1e-4)


def test_overlap_add_pit_keeps_stream_order_across_chunks():
    """An alternating separator swaps its output order every call; PIT
    alignment must undo the swaps so each reconstructed stream stays
    sign-consistent across ≥3 chunks."""
    rng = np.random.default_rng(2)
    mix = rng.standard_normal(20 * SR).astype(np.float32) * 0.1  # 9 chunks @ 4 s/50%
    s1, s2 = _separate_overlap_add(
        mix, _AlternatingSeparator(), DEVICE,
        sr_pipeline=SR, sr_separator=SR, chunk_length_s=4.0,
    )
    edge = _edge_exclusion(4 * SR)
    np.testing.assert_allclose(s1[edge:-edge], mix[edge:-edge], atol=1e-4)
    np.testing.assert_allclose(s2[edge:-edge], -mix[edge:-edge], atol=1e-4)


# ---------------------------------------------------------------------------
# Spill
# ---------------------------------------------------------------------------


def test_spill_writes_full_config_as_knobs(tmp_path):
    """Provenance: the spilled knob set must track SeparationConfig exactly
    (the old hand-maintained list silently dropped newly added knobs)."""
    stage = _make_stage()
    ctx = _make_ctx(20.0, [(8.0, 9.0)])
    stage.run(ctx)
    stage.spill(ctx, tmp_path)

    meta = json.loads((tmp_path / "separation_metadata.json").read_text())
    expected_keys = {f.name for f in fields(SeparationConfig)}
    assert set(meta["knobs"]) == expected_keys
    assert len(meta["overlaps"]) == 1
    assert (tmp_path / "overlap_0_s1_raw.wav").exists()
    assert (tmp_path / "overlap_0_s2_raw.wav").exists()


def test_spill_empty_run_writes_nothing(tmp_path):
    """Policy: ran-but-found-nothing → no metadata file."""
    stage = _make_stage()
    ctx = _make_ctx(20.0, [])
    stage.run(ctx)
    stage.spill(ctx, tmp_path)
    assert not (tmp_path / "separation_metadata.json").exists()
