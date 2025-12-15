"""Tests for SPMamba model.

SPMamba requires CUDA as it uses mamba-ssm which depends on Triton.
All tests in this file will be skipped if CUDA is not available.

Uses reduced config parameters for memory efficiency (spmamba_sb_reduced.yaml).
"""

import pytest
import torch
from models import SPMamba


# SPMamba requires CUDA (mamba-ssm uses Triton)
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="SPMamba requires CUDA (mamba-ssm dependency)",
)


@pytest.fixture
def device():
    """GPU device for SPMamba tests."""
    return "cuda"


@pytest.fixture
def spmamba_config():
    """Reduced SPMamba configuration for memory efficiency.
    
    Based on spmamba_sb_reduced.yaml to avoid memory overflow.
    """
    return {
        "n_fft": 256,  # Reduced from 512
        "stride": 64,  # Reduced from 128
        "input_dim": 64,
        "n_srcs": 1,
        "n_layers": 4,  # Reduced from 6
        "lstm_hidden_units": 192,  # Reduced from 256
        "attn_n_head": 2,  # Reduced from 4
        "attn_approx_qk_dim": 256,  # Reduced from 512
    }


def test_spmamba_initialization(spmamba_config):
    """Test SPMamba can be initialized with default params."""
    model = SPMamba(**spmamba_config)
    
    assert model is not None
    assert model.n_srcs == spmamba_config["n_srcs"]
    assert model.n_fft == spmamba_config["n_fft"]


def test_spmamba_forward_pass(spmamba_config, device):
    """Test SPMamba forward pass works correctly."""
    model = SPMamba(**spmamba_config).to(device)
    batch_size, time_steps = 1, 8000  # Reduced batch and length
    
    x = torch.randn(batch_size, time_steps).to(device)
    output = model(x)
    
    # Output should match input shape for enhancement
    assert output.shape == (batch_size, time_steps)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_spmamba_3d_input(spmamba_config, device):
    """Test SPMamba handles 3D input [batch, 1, time]."""
    model = SPMamba(**spmamba_config).to(device)
    batch_size, time_steps = 1, 8000
    
    x = torch.randn(batch_size, 1, time_steps).to(device)
    output = model(x)
    
    assert output.shape == (batch_size, time_steps)


def test_spmamba_separation_task(device):
    """Test SPMamba with n_srcs=2 (separation task)."""
    model = SPMamba(
        n_fft=256,
        stride=64,
        input_dim=64,
        n_srcs=2,  # Two sources
        n_layers=4,
        lstm_hidden_units=192,
        attn_n_head=2,
        attn_approx_qk_dim=256,
    ).to(device)
    
    batch_size, time_steps = 1, 8000  # Batch size 1
    x = torch.randn(batch_size, time_steps).to(device)
    output = model(x)
    
    # Should output 2 channels for separation
    assert output.shape == (batch_size, 2, time_steps)


def test_spmamba_different_layers(device):
    """Test SPMamba with different numbers of layers."""
    for n_layers in [2, 4]:  # Reduced range
        model = SPMamba(
            n_fft=256,
            stride=64,
            input_dim=64,
            n_srcs=1,
            n_layers=n_layers,
            lstm_hidden_units=128,  # Even more reduced
            attn_n_head=2,
            attn_approx_qk_dim=128,
        ).to(device)
        
        x = torch.randn(1, 8000).to(device)  # Batch size 1
        output = model(x)
        assert output.shape == (1, 8000)


def test_spmamba_different_nfft(device):
    """Test SPMamba with different STFT sizes."""
    for n_fft in [128, 256]:  # Reduced range
        model = SPMamba(
            n_fft=n_fft,
            stride=n_fft // 4,
            input_dim=64,
            n_srcs=1,
            n_layers=2,  # Minimal layers
            lstm_hidden_units=128,
            attn_n_head=2,
            attn_approx_qk_dim=128,
        ).to(device)
        
        x = torch.randn(1, 8000).to(device)  # Batch size 1
        output = model(x)
        assert output.shape == (1, 8000)


def test_spmamba_parameter_count(spmamba_config):
    """Test SPMamba has reasonable number of parameters."""
    model = SPMamba(**spmamba_config)
    num_params = sum(p.numel() for p in model.parameters())
    
    # SPMamba should have parameters in reasonable range
    assert num_params > 10_000  # At least some params
    assert num_params < 50_000_000  # Not too large


def test_spmamba_short_input(device):
    """Test SPMamba handles shorter inputs."""
    model = SPMamba(
        n_fft=256,
        stride=64,
        input_dim=64,
        n_srcs=1,
        n_layers=2,
        lstm_hidden_units=128,
        attn_n_head=2,
        attn_approx_qk_dim=128,
    ).to(device)
    
    # Shorter input
    x = torch.randn(1, 4000).to(device)  # Batch size 1
    output = model(x)
    
    # Output should match input length
    assert output.shape == (1, 4000)
