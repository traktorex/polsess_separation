"""Tests for RE-SepFormer model."""

import pytest
import torch
from models import RESepFormer


def test_resepformer_initialization():
    """Test RE-SepFormer can be initialized with default params."""
    model = RESepFormer(N=128, C=2)
    assert model is not None
    assert model.masknet.num_spk == 2


def test_resepformer_forward_pass():
    """Test model forward pass works correctly with 2D input [batch, time]."""
    model = RESepFormer(N=128, C=2, num_layers=1, segment_size=50)
    batch_size, time_steps = 2, 16000

    x = torch.randn(batch_size, time_steps)
    output = model(x)

    assert output.shape == (
        batch_size,
        2,
        time_steps,
    ), f"Expected {(batch_size, 2, time_steps)}, got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains Inf values"


def test_resepformer_3d_input():
    """Test model handles 3D input [batch, 1, time]."""
    model = RESepFormer(N=128, C=2, num_layers=1, segment_size=50)
    batch_size, time_steps = 2, 16000

    x = torch.randn(batch_size, 1, time_steps)
    output = model(x)

    assert output.shape == (
        batch_size,
        2,
        time_steps,
    ), f"Expected {(batch_size, 2, time_steps)}, got {output.shape}"


def test_resepformer_single_source():
    """Test model with C=1 (enhancement task) returns [batch, time]."""
    model = RESepFormer(N=128, C=1, num_layers=1, segment_size=50)
    batch_size, time_steps = 2, 16000

    x = torch.randn(batch_size, time_steps)
    output = model(x)

    assert output.shape == (
        batch_size,
        time_steps,
    ), f"Expected {(batch_size, time_steps)}, got {output.shape}"


def test_resepformer_custom_config():
    """Test model accepts a custom (small) configuration and forwards."""
    config = {
        "N": 64,
        "kernel_size": 16,
        "stride": 8,
        "C": 2,
        "num_blocks": 2,
        "num_layers": 1,
        "nhead": 4,
        "d_ffn": 256,
        "dropout": 0.1,
        "segment_size": 50,
    }
    model = RESepFormer(**config)
    assert model is not None

    x = torch.randn(1, 8000)
    y = model(x)
    assert y.shape == (1, 2, 8000)
    assert not torch.isnan(y).any()


def test_resepformer_different_segment_sizes():
    """Test model with different non-overlapping segment sizes."""
    for segment_size in [50, 100, 150]:
        model = RESepFormer(N=64, C=2, num_layers=1, segment_size=segment_size)
        x = torch.randn(1, 16000)
        y = model(x)
        assert y.shape == (1, 2, 16000)
        assert not torch.isnan(y).any()


def test_resepformer_parameter_count():
    """Test model has a reasonable number of parameters (>1M for the paper config)."""
    model = RESepFormer(N=128, C=2)
    num_params = sum(p.numel() for p in model.parameters())
    assert num_params > 1_000_000, f"Expected >1M params, got {num_params/1e6:.2f}M"
    print(f"RE-SepFormer parameters: {num_params/1e6:.2f}M")


def test_resepformer_positional_encoding():
    """Test that positional encoding can be enabled/disabled."""
    x = torch.randn(1, 8000)

    model_with_pe = RESepFormer(N=64, C=2, num_layers=1, nhead=4, d_ffn=256,
                                segment_size=50, use_positional_encoding=True)
    model_no_pe = RESepFormer(N=64, C=2, num_layers=1, nhead=4, d_ffn=256,
                              segment_size=50, use_positional_encoding=False)

    y_with = model_with_pe(x)
    y_without = model_no_pe(x)

    assert y_with.shape == (1, 2, 8000)
    assert y_without.shape == (1, 2, 8000)
    assert not torch.isnan(y_with).any()
    assert not torch.isnan(y_without).any()
