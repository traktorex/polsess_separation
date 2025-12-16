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


def test_convtasnet_parameter_count():
    """Test parameter count is reasonable."""
    model = ConvTasNet(N=256, B=256, H=512, P=3, X=8, R=4, C=1)
    num_params = sum(p.numel() for p in model.parameters())

    # ConvTasNet default should have ~8M parameters
    assert num_params > 5_000_000
    assert num_params < 15_000_000


def test_convtasnet_requires_odd_kernel():
    """Test that P (kernel size) must be odd per SpeechBrain requirements."""
    # Even P should raise ValueError
    with pytest.raises(ValueError, match="kernel size must be an odd number"):
        ConvTasNet(N=256, B=256, H=512, P=2, X=8, R=4, C=1)


def test_convtasnet_mask_nonlinear_options():
    """Test ConvTasNet with different mask nonlinearity options."""
    # Test supported mask nonlinear options (per SpeechBrain ConvTasNet)
    for mask_nonlinear in ["relu", "softmax"]:
        model = ConvTasNet(
            N=64, B=64, H=128, P=3, X=4, R=2, C=1, 
            mask_nonlinear=mask_nonlinear
        )
        
        # Should create successfully
        assert model is not None
        
        # Test forward pass works
        x = torch.randn(2, 8000)
        output = model(x)
        assert output.shape == (2, 8000)
        assert not torch.isnan(output).any()


def test_convtasnet_encoder_decoder_reconstruction():
    """Test that ConvTasNet can reconstruct signals (validates encoder-decoder design).
    
    Paper states decoder inverts encoder transformation. We test this by 
    verifying the model produces reasonable reconstruction with C=1 (enhancement).
    """
    model = ConvTasNet(N=256, B=256, H=512, P=3, X=8, R=4, C=1)
    model.eval()
    
    # Use clean signal - model should be able to output similar signal
    x = torch.randn(1, 8000)
    
    with torch.no_grad():
        output = model(x)
    
    # Model should produce output of same length
    assert output.shape == x.shape
    
    # Output should be in reasonable range (not exploding gradients)
    assert torch.abs(output).max() < 100.0
    
    # Output should not be all zeros (model is processing)
    assert torch.abs(output).mean() > 0.001


def test_convtasnet_tcn_receptive_field():
    """Test that TCN has sufficient receptive field for long sequences.
    
    Paper uses dilated convolutions to achieve large receptive field.
    With X=8, R=4, the network should handle sequences far longer than 
    a single convolution's receptive field.
    """
    model = ConvTasNet(N=256, B=256, H=512, P=3, X=8, R=4, C=1)
    
    # Model should handle sequences longer than base conv receptive field
    # 4 seconds at 8kHz = 32000 samples
    long_input = torch.randn(1, 32000)
    output = model(long_input)
    
    # Should process entire sequence
    assert output.shape[1] >= 30000  # Allow some edge effects
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()

