"""Unit tests for the TIGER model implementation."""

import pytest
import torch
import numpy as np


class TestTIGERBandSplit:
    """Test band-split scheme correctness at different sample rates."""

    def test_band_split_8khz_covers_full_spectrum(self):
        """All enc_dim frequency bins must be covered by the band-split."""
        from models.tiger import TIGER

        model = TIGER(n_fft=320, sample_rate=8000)
        assert sum(model.band_width) == model.enc_dim, (
            f"Band widths sum to {sum(model.band_width)}, expected {model.enc_dim}"
        )

    def test_band_split_all_widths_positive(self):
        """Every sub-band must have at least 1 frequency bin."""
        from models.tiger import TIGER

        model = TIGER(n_fft=320, sample_rate=8000)
        for i, bw in enumerate(model.band_width):
            assert bw >= 1, f"Sub-band {i} has width {bw} (must be >= 1)"

    def test_band_split_16khz_covers_full_spectrum(self):
        """Sanity check: band-split also works at 16kHz (original paper setting)."""
        from models.tiger import TIGER

        model = TIGER(n_fft=640, sample_rate=16000)
        assert sum(model.band_width) == model.enc_dim
        assert all(bw >= 1 for bw in model.band_width)

    def test_band_split_nband_reasonable(self):
        """At 8kHz the number of sub-bands should be less than at 16kHz."""
        from models.tiger import TIGER

        model_8k = TIGER(n_fft=320, sample_rate=8000)
        model_16k = TIGER(n_fft=640, sample_rate=16000)
        # At 8kHz, upper bands (4-8kHz) are above Nyquist and get trimmed
        assert model_8k.nband <= model_16k.nband


class TestTIGERForwardShape:
    """Test TIGER forward pass output shapes."""

    @pytest.fixture
    def tiger_small_es(self):
        """Small TIGER for single-source enhancement (ES task)."""
        from models.tiger import TIGER
        return TIGER(
            out_channels=32,    # Reduced for test speed
            in_channels=64,
            num_blocks=2,
            upsampling_depth=2,
            att_n_head=2,
            att_hid_chan=2,
            n_fft=320,
            hop_length=80,
            n_srcs=1,
            sample_rate=8000,
        )

    @pytest.fixture
    def tiger_small_sb(self):
        """Small TIGER for two-source separation (SB task)."""
        from models.tiger import TIGER
        return TIGER(
            out_channels=32,
            in_channels=64,
            num_blocks=2,
            upsampling_depth=2,
            att_n_head=2,
            att_hid_chan=2,
            n_fft=320,
            hop_length=80,
            n_srcs=2,
            sample_rate=8000,
        )

    def test_forward_1d_input_single_source(self, tiger_small_es):
        """[B, T] input → [B, T] output for n_srcs=1."""
        B, T = 2, 8000
        x = torch.randn(B, T)
        with torch.no_grad():
            out = tiger_small_es(x)
        assert out.shape == (B, T), f"Expected ({B}, {T}), got {out.shape}"

    def test_forward_3d_input_single_source(self, tiger_small_es):
        """[B, 1, T] input → [B, T] output for n_srcs=1."""
        B, T = 2, 8000
        x = torch.randn(B, 1, T)
        with torch.no_grad():
            out = tiger_small_es(x)
        assert out.shape == (B, T), f"Expected ({B}, {T}), got {out.shape}"

    def test_forward_two_sources(self, tiger_small_sb):
        """[B, T] input → [B, 2, T] output for n_srcs=2."""
        B, T = 2, 8000
        x = torch.randn(B, T)
        with torch.no_grad():
            out = tiger_small_sb(x)
        assert out.shape == (B, 2, T), f"Expected ({B}, 2, {T}), got {out.shape}"

    def test_forward_batch_size_1(self, tiger_small_es):
        """Model should work with batch size 1."""
        x = torch.randn(1, 8000)
        with torch.no_grad():
            out = tiger_small_es(x)
        assert out.shape == (1, 8000)

    def test_forward_variable_length(self, tiger_small_es):
        """Output length should match input length."""
        for T in [4000, 8000, 16000]:
            x = torch.randn(1, T)
            with torch.no_grad():
                out = tiger_small_es(x)
            assert out.shape[-1] == T, f"Length mismatch for T={T}: got {out.shape[-1]}"

    def test_forward_no_nan(self, tiger_small_es):
        """Forward pass should not produce NaN or Inf."""
        x = torch.randn(2, 8000)
        with torch.no_grad():
            out = tiger_small_es(x)
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"


class TestTIGERParamCount:
    """Test TIGER parameter counts match paper expectations."""

    def test_small_variant_under_1m_params(self):
        """Small TIGER (B=4) at 8kHz should have < 1M parameters.

        With the correct paper hyperparameters (in_channels=256, upsampling_depth=5),
        TIGER has ~0.70M params at 8kHz and ~0.82M at 16kHz, matching the paper.
        """
        from models.tiger import TIGER

        model = TIGER(
            out_channels=128,
            in_channels=256,
            num_blocks=4,
            upsampling_depth=5,
            att_n_head=4,
            att_hid_chan=4,
            n_fft=320,
            hop_length=80,
            n_srcs=1,
            sample_rate=8000,
        )
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params < 1_000_000, (
            f"Small TIGER has {n_params/1e6:.2f}M params, expected < 1M"
        )

    def test_large_same_params_as_small(self):
        """Large TIGER (B=8) should have the same parameter count as small (B=4).

        The FFI blocks share parameters across iterations, so B only affects
        compute (number of forward passes through the shared block), not param count.
        """
        from models.tiger import TIGER

        small = TIGER(num_blocks=4, n_fft=320, sample_rate=8000)
        large = TIGER(num_blocks=8, n_fft=320, sample_rate=8000)

        n_small = sum(p.numel() for p in small.parameters())
        n_large = sum(p.numel() for p in large.parameters())

        assert n_small == n_large, (
            f"Small has {n_small} params, large has {n_large}. "
            "They should be equal since FFI blocks share parameters."
        )


class TestTIGERRegistry:
    """Test TIGER is correctly registered in the model factory."""

    def test_tiger_in_model_registry(self):
        """TIGER should be accessible via get_model('tiger')."""
        from models import get_model
        model_cls = get_model("tiger")
        from models.tiger import TIGER
        assert model_cls is TIGER

    def test_tiger_instantiable_from_config(self):
        """TIGER should be instantiable via create_model_from_config."""
        from config import ModelConfig
        from models.factory import create_model_from_config

        config = ModelConfig(model_type="tiger")
        model = create_model_from_config(config)
        from models.tiger import TIGER
        assert isinstance(model, TIGER)

    def test_tiger_config_defaults(self):
        """ModelConfig should auto-initialize TIGERParams when model_type='tiger'."""
        from config import ModelConfig, TIGERParams

        config = ModelConfig(model_type="tiger")
        assert config.tiger is not None
        assert isinstance(config.tiger, TIGERParams)
        assert config.tiger.sample_rate == 8000
        assert config.tiger.n_fft == 320
        assert config.tiger.hop_length == 80
