"""Tests for common utility functions."""

import pytest
import torch
import numpy as np
import random
from unittest.mock import patch, MagicMock
from utils.common import set_seed, apply_eps_patch, setup_warnings, setup_device_and_amp


class TestSetSeed:
    """Test seed setting for reproducibility."""

    def test_set_seed_makes_torch_reproducible(self):
        """Test that set_seed makes torch operations reproducible."""
        set_seed(42)
        x1 = torch.randn(10)
        
        set_seed(42)
        x2 = torch.randn(10)
        
        assert torch.allclose(x1, x2), "Same seed should produce same torch random numbers"

    def test_set_seed_makes_numpy_reproducible(self):
        """Test that set_seed makes numpy operations reproducible."""
        set_seed(42)
        x1 = np.random.randn(10)
        
        set_seed(42)
        x2 = np.random.randn(10)
        
        assert np.allclose(x1, x2), "Same seed should produce same numpy random numbers"

    def test_set_seed_makes_python_reproducible(self):
        """Test that set_seed makes Python random reproducible."""
        set_seed(42)
        x1 = [random.random() for _ in range(10)]
        
        set_seed(42)
        x2 = [random.random() for _ in range(10)]
        
        assert x1 == x2, "Same seed should produce same Python random numbers"

    def test_set_seed_enables_deterministic_cuda(self):
        """Test that set_seed enables deterministic CUDA operations."""
        set_seed(42)
        
        # Check that deterministic mode is enabled
        assert torch.backends.cudnn.deterministic == True
        assert torch.backends.cudnn.benchmark == False

    def test_set_seed_different_seeds_produce_different_values(self):
        """Test that different seeds produce different values."""
        set_seed(42)
        x1 = torch.randn(10)
        
        set_seed(123)
        x2 = torch.randn(10)
        
        assert not torch.allclose(x1, x2), "Different seeds should produce different values"


class TestEpsPatch:
    """Test EPS patching for AMP float16 safety."""

    def test_apply_eps_patch_modifies_speechbrain_eps(self):
        """Test that apply_eps_patch modifies SpeechBrain's EPS value."""
        import speechbrain.lobes.models.conv_tasnet as conv_tasnet_module
        
        original_eps = conv_tasnet_module.EPS
        
        test_eps = 1e-4
        apply_eps_patch(test_eps)
        
        assert conv_tasnet_module.EPS == test_eps, "EPS should be modified"
        
        # Restore original
        conv_tasnet_module.EPS = original_eps

    def test_apply_eps_patch_warns_on_underflow_risk(self, capsys):
        """Test that apply_eps_patch warns when EPS is too small for float16."""
        import speechbrain.lobes.models.conv_tasnet as conv_tasnet_module
        
        original_eps = conv_tasnet_module.EPS
        
        # Use dangerously small EPS (will underflow in float16)
        apply_eps_patch(1e-6)
        
        captured = capsys.readouterr()
        assert "WARNING" in captured.out or "may underflow" in captured.out.lower()
        
        # Restore original
        conv_tasnet_module.EPS = original_eps

    def test_apply_eps_patch_safe_value_no_warning(self, capsys):
        """Test that safe EPS values don't produce warnings."""
        import speechbrain.lobes.models.conv_tasnet as conv_tasnet_module
        
        original_eps = conv_tasnet_module.EPS
        
        # Use safe EPS for float16
        apply_eps_patch(1e-4)
        
        captured = capsys.readouterr()
        # Should not have warning
        assert "may underflow" not in captured.out.lower()
        
        # Restore original
        conv_tasnet_module.EPS = original_eps


class TestSetupFunctions:
    """Test setup utility functions."""

    def test_setup_warnings_suppresses_warnings(self):
        """Test that setup_warnings configures warning filters."""
        import warnings
        import os
        
        # Clear existing filters
        warnings.resetwarnings()
        
        setup_warnings()
        
        # Check environment variable is set
        assert "PYTHONWARNINGS" in os.environ
        
        # Verify some warnings are filtered
        # (We can't easily test all filters without triggering actual warnings)
        assert len(warnings.filters) > 0

    def test_setup_device_and_amp_cpu_no_amp(self):
        """Test setup_device_and_amp with CPU and no AMP."""
        from config import Config
        
        config = Config()
        config.training.use_amp = False
        summary_info = {}
        
        device = setup_device_and_amp(config, summary_info)
        
        assert "eps_patch" in summary_info
        assert summary_info["eps_patch"] == "disabled"
        assert "device" in summary_info
        assert device in ["cuda", "cpu"]

    @patch('torch.cuda.is_available')
    def test_setup_device_and_amp_forced_cpu(self, mock_cuda):
        """Test setup_device_and_amp returns CPU when CUDA unavailable."""
        from config import Config
        
        mock_cuda.return_value = False
        
        config = Config()
        config.training.use_amp = False
        summary_info = {}
        
        device = setup_device_and_amp(config, summary_info)
        
        assert device == "cpu"
        assert summary_info["device"] == "cpu"

    def test_setup_device_and_amp_with_amp_enabled(self):
        """Test setup_device_and_amp with AMP enabled."""
        from config import Config
        import speechbrain.lobes.models.conv_tasnet as conv_tasnet_module
        
        original_eps = conv_tasnet_module.EPS
        
        config = Config()
        config.training.use_amp = True
        config.training.amp_eps = 1e-4
        summary_info = {}
        
        device = setup_device_and_amp(config, summary_info)
        
        assert "eps_patch" in summary_info
        assert "1e-04" in summary_info["eps_patch"] or "0.0001" in summary_info["eps_patch"]
        assert "enabled" in summary_info["eps_patch"]
        
        # Restore original
        conv_tasnet_module.EPS = original_eps
