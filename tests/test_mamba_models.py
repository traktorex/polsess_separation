"""Tests for Mamba-based speech separation models.

MambaTasNet and DPMamba require CUDA (mamba-ssm uses Triton/CUDA kernels).
All tests in this file will be skipped if CUDA is not available.
"""

import pytest
import torch

from models.mamba import MAMBA_AVAILABLE

# Skip entire file if mamba-ssm not available or no CUDA
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not MAMBA_AVAILABLE,
    reason="Mamba models require CUDA + mamba-ssm",
)


@pytest.fixture(autouse=True)
def clear_cuda_cache():
    """Clear CUDA cache before/after each test to prevent OOM."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture
def device():
    return "cuda"


# ---------------------------------------------------------------------------
# MambaBlocksSequential
# ---------------------------------------------------------------------------

class TestMambaBlocksSequential:
    """Tests for the core MambaBlocksSequential building block."""

    def test_block_count(self):
        from models.mamba import MambaBlocksSequential
        for n in [1, 2, 4]:
            model = MambaBlocksSequential(
                n_mamba=n, bidirectional=True, d_model=64, d_state=8,
            )
            assert len(model.layers) == n

    def test_forward_shape(self, device):
        from models.mamba import MambaBlocksSequential
        model = MambaBlocksSequential(
            n_mamba=2, bidirectional=True, d_model=64, d_state=8,
        ).to(device)
        x = torch.randn(2, 100, 64, device=device)
        out = model(x)
        assert out.shape == (2, 100, 64)
        assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# MambaTasNet
# ---------------------------------------------------------------------------

class TestMambaTasNet:
    """Tests for Mamba-TasNet (single-path Mamba separator)."""

    @pytest.fixture
    def small_config(self):
        """Tiny config to avoid OOM during testing."""
        return dict(
            N=64, kernel_size=16, stride=8, C=1,
            bot_dim=64, n_mamba=2, d_state=8, d_conv=4, expand=2,
        )

    def test_init(self, small_config):
        from models import MambaTasNet
        model = MambaTasNet(**small_config)
        assert model.C == 1
        assert model.N == 64

    def test_forward_2d(self, small_config, device):
        from models import MambaTasNet
        model = MambaTasNet(**small_config).to(device)
        x = torch.randn(1, 8000, device=device)
        out = model(x)
        assert out.shape == (1, 8000)
        assert not torch.isnan(out).any()

    def test_forward_3d(self, small_config, device):
        from models import MambaTasNet
        model = MambaTasNet(**small_config).to(device)
        x = torch.randn(1, 1, 8000, device=device)
        out = model(x)
        assert out.shape == (1, 8000)

    def test_separation(self, device):
        from models import MambaTasNet
        model = MambaTasNet(
            N=64, kernel_size=16, stride=8, C=2,
            bot_dim=64, n_mamba=2, d_state=8, d_conv=4, expand=2,
        ).to(device)
        x = torch.randn(1, 8000, device=device)
        out = model(x)
        assert out.shape == (1, 2, 8000)

    def test_parameter_count(self, small_config):
        from models import MambaTasNet
        model = MambaTasNet(**small_config)
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 10_000
        assert n_params < 50_000_000

    def test_gradient_flow(self, small_config, device):
        from models import MambaTasNet
        model = MambaTasNet(**small_config).to(device)
        x = torch.randn(1, 8000, device=device)
        out = model(x)
        loss = out.mean()
        loss.backward()
        # Check gradients exist on encoder (first layer)
        for p in model.encoder.parameters():
            assert p.grad is not None


# ---------------------------------------------------------------------------
# DPMamba
# ---------------------------------------------------------------------------

class TestDPMamba:
    """Tests for DPMamba (dual-path Mamba separator)."""

    @pytest.fixture
    def small_config(self):
        return dict(
            N=64, kernel_size=16, stride=8, C=1,
            num_layers=2, chunk_size=100, n_mamba_dp=2,
            d_state=8, d_conv=4, expand=2,
        )

    def test_init(self, small_config):
        from models import DPMamba
        model = DPMamba(**small_config)
        assert model.C == 1
        assert model.N == 64

    def test_forward_2d(self, small_config, device):
        from models import DPMamba
        model = DPMamba(**small_config).to(device)
        x = torch.randn(1, 8000, device=device)
        out = model(x)
        assert out.shape == (1, 8000)
        assert not torch.isnan(out).any()

    def test_forward_3d(self, small_config, device):
        from models import DPMamba
        model = DPMamba(**small_config).to(device)
        x = torch.randn(1, 1, 8000, device=device)
        out = model(x)
        assert out.shape == (1, 8000)

    def test_separation(self, device):
        from models import DPMamba
        model = DPMamba(
            N=64, kernel_size=16, stride=8, C=2,
            num_layers=2, chunk_size=100, n_mamba_dp=2,
            d_state=8, d_conv=4, expand=2,
        ).to(device)
        x = torch.randn(1, 8000, device=device)
        out = model(x)
        assert out.shape == (1, 2, 8000)

    def test_different_chunk_sizes(self, device):
        from models import DPMamba
        for chunk_size in [50, 150]:
            model = DPMamba(
                N=64, kernel_size=16, stride=8, C=1,
                num_layers=2, chunk_size=chunk_size, n_mamba_dp=2,
                d_state=8, d_conv=4, expand=2,
            ).to(device)
            x = torch.randn(1, 8000, device=device)
            out = model(x)
            assert out.shape == (1, 8000)

    def test_parameter_count(self, small_config):
        from models import DPMamba
        model = DPMamba(**small_config)
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 10_000
        assert n_params < 50_000_000

    def test_gradient_flow(self, small_config, device):
        from models import DPMamba
        model = DPMamba(**small_config).to(device)
        x = torch.randn(1, 8000, device=device)
        out = model(x)
        loss = out.mean()
        loss.backward()
        for p in model.encoder.parameters():
            assert p.grad is not None
