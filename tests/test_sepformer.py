"""Tests for SepFormer model."""

import pytest
import torch
from models import SepFormer


def test_sepformer_initialization():
    """Test SepFormer can be initialized with default params."""
    model = SepFormer(N=256, C=2)
    assert model is not None
    assert model.masknet.num_spks == 2
    assert model.masknet.end_conv1x1.in_channels == 256


def test_sepformer_forward_pass():
    """Test model forward pass works correctly."""
    model = SepFormer(N=256, C=2)
    batch_size, time_steps = 2, 16000

    # Test with 2D input [batch, time]
    x = torch.randn(batch_size, time_steps)
    output = model(x)

    assert output.shape == (
        batch_size,
        2,
        time_steps,
    ), f"Expected {(batch_size, 2, time_steps)}, got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains Inf values"


def test_sepformer_3d_input():
    """Test model handles 3D input [batch, 1, time]."""
    model = SepFormer(N=256, C=2)
    batch_size, time_steps = 2, 16000

    x = torch.randn(batch_size, 1, time_steps)
    output = model(x)

    assert output.shape == (
        batch_size,
        2,
        time_steps,
    ), f"Expected {(batch_size, 2, time_steps)}, got {output.shape}"


def test_sepformer_single_source():
    """Test model with C=1 (enhancement task)."""
    model = SepFormer(N=256, C=1)
    batch_size, time_steps = 2, 16000

    x = torch.randn(batch_size, time_steps)
    output = model(x)

    assert output.shape == (
        batch_size,
        time_steps,
    ), f"Expected {(batch_size, time_steps)}, got {output.shape}"


def test_sepformer_parameter_count():
    """Test model has expected number of parameters."""
    model = SepFormer(N=256, C=2, num_layers=2, d_model=256)
    num_params = sum(p.numel() for p in model.parameters())

    # SepFormer should have significantly more params than ConvTasNet due to transformers
    assert num_params > 1_000_000, f"Expected >1M params, got {num_params/1e6:.2f}M"
    print(f"SepFormer parameters: {num_params/1e6:.2f}M")


def test_sepformer_custom_config():
    """Test model accepts custom configuration."""
    config = {
        "N": 128,
        "kernel_size": 16,
        "stride": 8,
        "C": 2,
        "num_layers": 1,
        "d_model": 128,
        "nhead": 4,
        "d_ffn": 512,
        "dropout": 0.1,
        "chunk_size": 100,
        "hop_size": 50,
    }
    model = SepFormer(**config)
    assert model is not None

    # Test forward pass with custom config
    x = torch.randn(1, 8000)
    y = model(x)
    assert y.shape == (1, 2, 8000)


def test_sepformer_different_chunk_sizes():
    """Test model with different dual-path chunk configurations."""
    model = SepFormer(N=256, C=2, chunk_size=200, hop_size=100)
    x = torch.randn(2, 16000)
    y = model(x)
    assert y.shape == (2, 2, 16000)
