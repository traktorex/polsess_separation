"""Tests for Stage 3c (PostSeparationProcessingStage) with the naive backend.

The naive backend is identity, so the stage's own logic — VAD-mask
application, mask/output length alignment, silent-stream short-circuit —
is fully observable without any neural model.
"""

import numpy as np
import pytest
import torch

from asr_pipeline.config import PostSeparationProcessingConfig
from asr_pipeline.context import PipelineContext
from asr_pipeline.stages.post_separation_processing import (
    PostSeparationProcessingStage,
)

SR = 16_000


def _stage() -> PostSeparationProcessingStage:
    stage = PostSeparationProcessingStage(
        PostSeparationProcessingConfig(backend="naive")
    )
    stage.load(torch.device("cpu"))
    return stage


def _entry(s1_raw, mask1, s2_raw, mask2, idx=0) -> dict:
    return {
        "idx": idx,
        "s1_raw": np.asarray(s1_raw, dtype=np.float32),
        "s2_raw": np.asarray(s2_raw, dtype=np.float32),
        "mask1": np.asarray(mask1, dtype=np.float32),
        "mask2": np.asarray(mask2, dtype=np.float32),
    }


def _ctx_with(entries) -> PipelineContext:
    ctx = PipelineContext(sample_rate=SR)
    ctx.overlap_separated = entries
    return ctx


def test_run_before_load_raises():
    stage = PostSeparationProcessingStage(
        PostSeparationProcessingConfig(backend="naive")
    )
    with pytest.raises(RuntimeError, match="before load"):
        stage.run(_ctx_with([_entry([1.0], [1.0], [1.0], [1.0])]))


def test_no_overlaps_is_noop():
    ctx = _ctx_with([])
    _stage().run(ctx)
    assert ctx.overlap_separated == []


def test_naive_gated_is_raw_times_mask():
    raw1 = np.linspace(-1.0, 1.0, 1024)
    mask1 = np.concatenate([np.ones(512), np.zeros(512)])
    raw2 = np.full(1024, 0.5)
    mask2 = np.ones(1024)
    entry = _entry(raw1, mask1, raw2, mask2)
    _stage().run(_ctx_with([entry]))
    np.testing.assert_allclose(
        entry["s1_gated"], raw1.astype(np.float32) * mask1.astype(np.float32)
    )
    np.testing.assert_allclose(entry["s2_gated"], raw2.astype(np.float32))
    assert entry["s1_gated"].dtype == np.float32


def test_silent_mask_short_circuits_to_zeros():
    raw = np.ones(1024)
    entry = _entry(raw, np.zeros(1024), raw, np.ones(1024))
    _stage().run(_ctx_with([entry]))
    np.testing.assert_array_equal(entry["s1_gated"], np.zeros(1024, np.float32))
    np.testing.assert_allclose(entry["s2_gated"], np.ones(1024, np.float32))


def test_mask_shorter_than_output_pads_with_silence():
    # Backend output longer than the mask: the un-covered tail must be
    # gated out (padded mask = 0), not passed through.
    raw = np.ones(1000)
    mask = np.ones(900)
    entry = _entry(raw, mask, raw, np.ones(1000))
    _stage().run(_ctx_with([entry]))
    np.testing.assert_allclose(entry["s1_gated"][:900], 1.0)
    np.testing.assert_array_equal(entry["s1_gated"][900:], 0.0)


def test_mask_longer_than_output_is_truncated():
    raw = np.ones(900)
    mask = np.ones(1000)
    entry = _entry(raw, mask, raw, np.ones(900))
    _stage().run(_ctx_with([entry]))
    assert len(entry["s1_gated"]) == 900
    np.testing.assert_allclose(entry["s1_gated"], 1.0)


def test_raw_streams_left_untouched():
    raw1 = np.linspace(-1.0, 1.0, 1024).astype(np.float32)
    entry = _entry(raw1.copy(), np.zeros(1024), raw1.copy(), np.ones(1024))
    _stage().run(_ctx_with([entry]))
    np.testing.assert_array_equal(entry["s1_raw"], raw1)
    np.testing.assert_array_equal(entry["s2_raw"], raw1)
