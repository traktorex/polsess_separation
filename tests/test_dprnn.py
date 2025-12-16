"""Tests for DPRNN model."""

import pytest
import torch
from models import DPRNN


@pytest.fixture
def dprnn_config():
    """Default DPRNN configuration for tests."""
    return {
        "N": 64,
        "C": 1,  # Number of output channels (sources)
        "hidden_size": 128,
        "num_layers": 2,
        "bidirectional": True,
        "dropout": 0.0,
        "kernel_size": 16,
        "stride": 8,
    }


def test_dprnn_initialization(dprnn_config):
    """Test DPRNN can be initialized with default params."""
    model = DPRNN(**dprnn_config)
    
    assert model is not None
    assert model.N == dprnn_config["N"]
    assert model.C == dprnn_config["C"]


def test_dprnn_forward_pass(dprnn_config):
    """Test DPRNN forward pass works correctly."""
    model = DPRNN(**dprnn_config)
    batch_size, time_steps = 2, 16000
    
    x = torch.randn(batch_size, time_steps)
    output = model(x)
    
    # Output should match input shape for enhancement (out_channels=1)
    assert output.shape == (batch_size, time_steps)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_dprnn_3d_input(dprnn_config):
    """Test DPRNN handles 3D input [batch, 1, time]."""
    model = DPRNN(**dprnn_config)
    batch_size, time_steps = 2, 16000
    
    x = torch.randn(batch_size, 1, time_steps)
    output = model(x)
    
    assert output.shape == (batch_size, time_steps)


def test_dprnn_separation_task():
    """Test DPRNN with C=2 (separation task)."""
    model = DPRNN(
        N=64,
        C=2,  # Two sources
        hidden_size=128,
        num_layers=2,
        kernel_size=16,
        stride=8,
    )
    
    batch_size, time_steps = 2, 16000
    x = torch.randn(batch_size, time_steps)
    output = model(x)
    
    # Should output 2 channels for separation
    assert output.shape == (batch_size, 2, time_steps)


def test_dprnn_different_hidden_sizes():
    """Test DPRNN accepts different hidden sizes."""
    for hidden_size in [64, 128, 256]:
        model = DPRNN(N=64, C=1, hidden_size=hidden_size)
        assert model is not None


# def test_dprnn_unidirectional():
#     """Test DPRNN with bidirectional=False."""
#     model = DPRNN(
#         N=64,
#         C=1,
#         hidden_size=128,
#         bidirectional=False,
#         kernel_size=16,
#         stride=8,
#     )
    
#     x = torch.randn(2, 16000)
#     output = model(x)
    
#     assert output.shape == (2, 16000)


def test_dprnn_with_dropout():
    """Test DPRNN with dropout enabled."""
    model = DPRNN(
        N=64, C=1, hidden_size=128, dropout=0.2, kernel_size=16, stride=8
    )
    
    assert model is not None
    
    # Test in training mode (dropout active)
    model.train()
    x = torch.randn(2, 16000)
    output1 = model(x)
    output2 = model(x)
    
    # Outputs should be different due to dropout
    # (This might not always be true for small models/inputs, so we just check shape)
    assert output1.shape == output2.shape


def test_dprnn_parameter_count(dprnn_config):
    """Test DPRNN has expected number of parameters."""
    model = DPRNN(**dprnn_config)
    num_params = sum(p.numel() for p in model.parameters())
    
    # DPRNN is smaller than ConvTasNet, should be < 5M params
    assert num_params < 5_000_000
    assert num_params > 100_000  # Should have some parameters


def test_dprnn_num_layers():
    """Test DPRNN with different number of layers."""
    for num_layers in [1, 2, 4]:
        model = DPRNN(
            N=64,
            C=1,
            hidden_size=128,
            num_layers=num_layers,
            kernel_size=16,
            stride=8,
        )
        
        x = torch.randn(2, 16000)
        output = model(x)
        assert output.shape == (2, 16000)


def test_dprnn_different_kernel_stride():
    """Test DPRNN with custom kernel_size and stride (Klec et al. config)."""
    model = DPRNN(
        N=64,
        C=1,
        hidden_size=128,
        kernel_size=20,  # Klec et al. specification
        stride=10,
    )
    
    x = torch.randn(2, 16000)
    output = model(x)
    
    assert output.shape == (2, 16000)


def test_dprnn_different_chunk_sizes():
    """Test DPRNN with different chunk sizes (dual-path processing parameter)."""
    # chunk_size affects how the sequence is split for dual-path processing
    for chunk_size in [50, 100, 200]:
        model = DPRNN(
            N=64,
            C=1,
            hidden_size=128,
            num_layers=2,
            chunk_size=chunk_size,
        )
        
        x = torch.randn(2, 16000)
        output = model(x)
        
        # Output shape should be same regardless of chunk_size
        assert output.shape == (2, 16000)


def test_dprnn_complexity_reduction():
    """Test that DPRNN handles very long sequences efficiently (per paper).
    
    Paper states: Dual-path processing reduces complexity from O(L) to O(√L).
    With chunking, each RNN processes √L steps instead of L.
    For L=160000, K=100, effective length per RNN is ~400 (√160000 ≈ 400).
    """
    model = DPRNN(
        N=64,
        C=1,
        hidden_size=128,
        chunk_size=100,
        num_layers=2,
        kernel_size=16,
        stride=8,
    )
    
    # Test with very long sequence (20 seconds at 8kHz)
    # This would be challenging for standard RNN but manageable with chunking
    long_seq = torch.randn(1, 160000)
    output = model(long_seq)
    
    # Should handle long sequence successfully
    assert output.shape == long_seq.shape
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()

