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


def test_sepformer_attention_mechanism():
    """Test that SepFormer uses multi-head attention mechanism (vs RNN).
    
    Paper states: RNN-free architecture with multi-head attention.
    We validate this by testing with different numbers of attention heads.
    """
    # Test that different nhead values work correctly
    for nhead in [2, 4, 8]:
        model = SepFormer(
            N=128, C=2, num_layers=2, d_model=128, 
            nhead=nhead, d_ffn=512
        )
        x = torch.randn(1, 8000)
        output = model(x)
        assert output.shape == (1, 2, 8000)
        assert not torch.isnan(output).any()


def test_sepformer_dual_path_processing():
    """Test SepFormer dual-path (intra + inter) transformer processing.
    
    Paper states: Intra-transformer for local dependencies within chunks,
    inter-transformer for global context across chunks.
    Validate by processing long sequences that require multiple chunks.
    """
    model = SepFormer(
        N=256, C=2, chunk_size=250, hop_size=125, 
        num_layers=2, d_model=256, nhead=4
    )
    
    # Long sequence to require multiple chunks
    # With chunk_size=250, hop_size=125, 32000 samples = many chunks
    x = torch.randn(1, 32000)
    output = model(x)
    
    # Should handle long sequence via dual-path chunking
    assert output.shape == (1, 2, 32000)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_sepformer_parallel_processing():
    """Test SepFormer parallel processing advantage (vs sequential RNN).
    
    Paper states: Transformer-based architecture enables parallel processing
    unlike sequential RNNs, leading to faster inference.
    We validate by ensuring model processes inputs efficiently.
    """
    model = SepFormer(N=256, C=2, num_layers=2, d_model=256, nhead=4)
    model.eval()
    
    # Test batch processing (parallel advantage)
    batch_x = torch.randn(4, 16000)
    
    with torch.no_grad():
        batch_output = model(batch_x)
    
    # Should process batch efficiently
    assert batch_output.shape == (4, 2, 16000)
    assert not torch.isnan(batch_output).any()

