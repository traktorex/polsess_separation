"""Tests for ConvTasNet model."""

import pytest
import torch
from models import ConvTasNet


def test_convtasnet_initialization(sample_config):
    """Test ConvTasNet can be initialized with default params."""
    model = ConvTasNet(**sample_config)
    assert model is not None
    assert model.C == sample_config["C"]
    assert model.N == sample_config["N"]


def test_convtasnet_forward_pass(device, sample_config):
    """Test model forward pass works correctly."""
    model = ConvTasNet(**sample_config).to(device)
    batch_size, time_steps = 2, 16000

    # Test with 2D input [batch, time]
    x = torch.randn(batch_size, time_steps).to(device)
    output = model(x)

    assert output.shape == (batch_size, time_steps), f"Expected {(batch_size, time_steps)}, got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains Inf values"


def test_convtasnet_3d_input(device, sample_config):
    """Test model handles 3D input [batch, 1, time]."""
    model = ConvTasNet(**sample_config).to(device)
    batch_size, time_steps = 2, 16000

    x = torch.randn(batch_size, 1, time_steps).to(device)
    output = model(x)

    assert output.shape == (batch_size, time_steps), f"Expected {(batch_size, time_steps)}, got {output.shape}"


def test_convtasnet_custom_kernel_size():
    """Test model accepts custom kernel_size and stride (Klec et al. config)."""
    config = {
        "N": 256,
        "B": 256,
        "H": 512,
        "P": 3,
        "X": 8,
        "R": 4,
        "C": 1,
        "kernel_size": 20,  # Klec et al. specification
        "stride": 10,
    }
    model = ConvTasNet(**config)
    assert model is not None


def test_convtasnet_multi_source_output():
    """Test model with C=2 (speech separation task)."""
    config = {
        "N": 256,
        "B": 256,
        "H": 512,
        "P": 3,
        "X": 8,
        "R": 4,
        "C": 2,  # Two sources
        "kernel_size": 16,
        "stride": 8,
    }
    model = ConvTasNet(**config)
    batch_size, time_steps = 2, 16000

    x = torch.randn(batch_size, time_steps)
    output = model(x)

    assert output.shape == (batch_size, 2, time_steps), f"Expected {(batch_size, 2, time_steps)}, got {output.shape}"


def test_convtasnet_parameter_count(sample_config):
    """Test model has expected number of parameters."""
    model = ConvTasNet(**sample_config)
    num_params = sum(p.numel() for p in model.parameters())

    # Default config should have ~8-9M parameters
    assert 7_000_000 < num_params < 10_000_000, f"Expected ~8M params, got {num_params/1e6:.2f}M"
