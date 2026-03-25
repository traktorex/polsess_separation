"""Tests for SPMamba3 model (SPMamba with Mamba-3 blocks).

SPMamba3 requires CUDA as it uses mamba-ssm which depends on Triton.
Additionally requires mamba-ssm installed from source for Mamba3 support.
All tests in this file will be skipped if CUDA is not available.

Uses reduced config parameters for memory efficiency.
"""

import pytest
import torch
from models import SPMamba3


# SPMamba3 requires CUDA (mamba-ssm uses Triton)
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="SPMamba3 requires CUDA (mamba-ssm dependency)",
)


@pytest.fixture(autouse=True)
def clear_cuda_cache():
    """Clear CUDA cache before each test to prevent memory overflow.

    SPMamba3 tests are memory-intensive. When running all tests, previous tests
    may leave GPU memory allocated, causing overflow. This fixture ensures
    a clean slate for each SPMamba3 test.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture
def device():
    """GPU device for SPMamba3 tests."""
    return "cuda"


@pytest.fixture
def spmamba3_config():
    """Minimal SPMamba3 configuration to fit alongside other GPU workloads."""
    return {
        "n_fft": 256,
        "stride": 64,
        "input_dim": 64,
        "n_srcs": 1,
        "n_layers": 2,
        "lstm_hidden_units": 64,
        "attn_n_head": 2,
        "attn_approx_qk_dim": 64,
        "d_state": 16,
        "headdim": 16,
    }


def test_spmamba3_initialization(spmamba3_config):
    """Test SPMamba3 can be initialized with default params."""
    model = SPMamba3(**spmamba3_config)

    assert model is not None
    assert model.n_srcs == spmamba3_config["n_srcs"]
    assert model.n_fft == spmamba3_config["n_fft"]


def test_spmamba3_forward_pass(spmamba3_config, device):
    """Test SPMamba3 forward pass works correctly."""
    model = SPMamba3(**spmamba3_config).to(device)
    batch_size, time_steps = 1, 4000

    x = torch.randn(batch_size, time_steps).to(device)
    output = model(x)

    # Output should match input shape for enhancement
    assert output.shape == (batch_size, time_steps)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_spmamba3_3d_input(spmamba3_config, device):
    """Test SPMamba3 handles 3D input [batch, 1, time]."""
    model = SPMamba3(**spmamba3_config).to(device)
    batch_size, time_steps = 1, 4000

    x = torch.randn(batch_size, 1, time_steps).to(device)
    output = model(x)

    assert output.shape == (batch_size, time_steps)


def test_spmamba3_separation_task(device):
    """Test SPMamba3 with n_srcs=2 (separation task)."""
    model = SPMamba3(
        n_fft=256,
        stride=64,
        input_dim=64,
        n_srcs=2,  # Two sources
        n_layers=2,
        lstm_hidden_units=64,
        attn_n_head=2,
        attn_approx_qk_dim=64,
        d_state=16,
        headdim=16,
    ).to(device)

    batch_size, time_steps = 1, 4000
    x = torch.randn(batch_size, time_steps).to(device)
    output = model(x)

    # Should output 2 channels for separation
    assert output.shape == (batch_size, 2, time_steps)


def test_spmamba3_different_layers(device):
    """Test SPMamba3 with different numbers of layers."""
    for n_layers in [2, 4]:
        model = SPMamba3(
            n_fft=256,
            stride=64,
            input_dim=64,
            n_srcs=1,
            n_layers=n_layers,
            lstm_hidden_units=64,
            attn_n_head=2,
            attn_approx_qk_dim=64,
            d_state=16,
            headdim=16,
        ).to(device)

        x = torch.randn(1, 4000).to(device)
        output = model(x)
        assert output.shape == (1, 4000)


def test_spmamba3_different_nfft(device):
    """Test SPMamba3 with different STFT sizes."""
    for n_fft in [128, 256]:
        model = SPMamba3(
            n_fft=n_fft,
            stride=n_fft // 4,
            input_dim=64,
            n_srcs=1,
            n_layers=2,
            lstm_hidden_units=64,
            attn_n_head=2,
            attn_approx_qk_dim=64,
            d_state=16,
            headdim=16,
        ).to(device)

        x = torch.randn(1, 4000).to(device)
        output = model(x)
        assert output.shape == (1, 4000)


def test_spmamba3_parameter_count(spmamba3_config):
    """Test SPMamba3 has reasonable number of parameters."""
    model = SPMamba3(**spmamba3_config)
    num_params = sum(p.numel() for p in model.parameters())

    # SPMamba3 should have parameters in reasonable range
    assert num_params > 10_000  # At least some params
    assert num_params < 50_000_000  # Not too large


def test_spmamba3_preserves_input_length(device):
    """Test that SPMamba3 preserves input length (STFT reconstruction)."""
    model = SPMamba3(
        n_fft=256,
        stride=64,
        input_dim=64,
        n_srcs=1,
        n_layers=2,
        lstm_hidden_units=64,
        attn_n_head=2,
        attn_approx_qk_dim=64,
        d_state=16,
        headdim=16,
    ).to(device)

    # Test two lengths to verify STFT round-trip without high memory cost
    for length in [4000, 8000]:
        x = torch.randn(1, length).to(device)
        output = model(x)

        # STFT-based models should preserve length exactly
        assert output.shape[1] == length, f"Length mismatch: {output.shape[1]} != {length}"


def test_spmamba3_headdim_constraint():
    """Test that SPMamba3 raises error when headdim doesn't divide in_channels.

    in_channels = emb_dim * emb_ks. With defaults emb_dim=16, emb_ks=4 -> 64.
    headdim=48 doesn't divide 64, so this should raise ValueError.
    """
    with pytest.raises(ValueError, match="must be divisible by headdim"):
        SPMamba3(
            n_fft=256,
            stride=64,
            input_dim=64,
            n_srcs=1,
            n_layers=2,
            lstm_hidden_units=64,
            attn_n_head=2,
            attn_approx_qk_dim=64,
            d_state=16,
            headdim=48,  # 64 % 48 != 0
        )


def test_spmamba3_output_dtype(spmamba3_config, device):
    """Test that SPMamba3 output dtype matches input dtype.

    Mamba-3 uses bfloat16 internally, but the output should be cast
    back to the input's dtype (float32) transparently.
    """
    model = SPMamba3(**spmamba3_config).to(device)

    x = torch.randn(1, 4000, dtype=torch.float32).to(device)
    output = model(x)

    assert output.dtype == torch.float32, f"Expected float32, got {output.dtype}"


def test_spmamba3_different_d_state(device):
    """Test SPMamba3 with different d_state values."""
    for d_state in [16, 32]:
        model = SPMamba3(
            n_fft=256,
            stride=64,
            input_dim=64,
            n_srcs=1,
            n_layers=2,
            lstm_hidden_units=64,
            attn_n_head=2,
            attn_approx_qk_dim=64,
            d_state=d_state,
            headdim=16,
        ).to(device)

        x = torch.randn(1, 4000).to(device)
        output = model(x)
        assert output.shape == (1, 4000)
        assert not torch.isnan(output).any()
