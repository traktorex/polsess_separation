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
"""Tests for file utility functions."""

import pytest
from pathlib import Path
from utils.common import ensure_dir


def test_ensure_dir_creates_directory(tmp_path):
    """Test ensure_dir creates directory."""
    new_dir = tmp_path / "test" / "nested" / "path"
    
    assert not new_dir.exists()
    
    result = ensure_dir(new_dir)
    
    assert new_dir.exists()
    assert new_dir.is_dir()
    assert result == new_dir


def test_ensure_dir_creates_parent_directories(tmp_path):
    """Test ensure_dir creates parent directories."""
    nested_dir = tmp_path / "level1" / "level2" / "level3"
    
    result = ensure_dir(nested_dir)
    
    assert (tmp_path / "level1").exists()
    assert (tmp_path / "level1" / "level2").exists()
    assert nested_dir.exists()
    assert result == nested_dir


def test_ensure_dir_existing_directory_no_error(tmp_path):
    """Test ensure_dir handles existing directory without error."""
    existing_dir = tmp_path / "existing"
    existing_dir.mkdir()
    
    # Should not raise error
    result = ensure_dir(existing_dir)
    
    assert result == existing_dir
    assert existing_dir.exists()


def test_ensure_dir_returns_path_object(tmp_path):
    """Test ensure_dir returns Path object."""
    new_dir = tmp_path / "test"
    
    result = ensure_dir(new_dir)
    
    assert isinstance(result, Path)


def test_ensure_dir_accepts_string_path(tmp_path):
    """Test ensure_dir accepts string paths."""
    new_dir = tmp_path / "string_test"
    new_dir_str = str(new_dir)
    
    result = ensure_dir(new_dir_str)
    
    assert new_dir.exists()
    assert isinstance(result, Path)
    assert result == new_dir


def test_ensure_dir_idempotent(tmp_path):
    """Test ensure_dir can be called multiple times safely."""
    new_dir = tmp_path / "idempotent"
    
    result1 = ensure_dir(new_dir)
    result2 = ensure_dir(new_dir)
    result3 = ensure_dir(new_dir)
    
    assert result1 == result2 == result3
    assert new_dir.exists()

"""Tests for config utility functions."""

import pytest
from dataclasses import dataclass
from types import SimpleNamespace
from utils.common import dataclass_to_dict


@dataclass
class SampleDataclass:
    """Sample dataclass for testing."""
    name: str
    value: int
    flag: bool = True


@dataclass
class NestedDataclass:
    """Nested dataclass for testing."""
    inner: SampleDataclass
    count: int


def test_dataclass_to_dict_simple():
    """Test converting simple dataclass to dict."""
    obj = SampleDataclass(name="test", value=42, flag=False)
    
    result = dataclass_to_dict(obj)
    
    assert isinstance(result, dict)
    assert result["name"] == "test"
    assert result["value"] == 42
    assert result["flag"] is False


def test_dataclass_to_dict_with_defaults():
    """Test dataclass with default values."""
    obj = SampleDataclass(name="default_test", value=100)
    
    result = dataclass_to_dict(obj)
    
    assert result["name"] == "default_test"
    assert result["value"] == 100
    assert result["flag"] is True  # Default value


def test_dataclass_to_dict_nested():
    """Test converting nested dataclass."""
    inner = SampleDataclass(name="inner", value=10)
    outer = NestedDataclass(inner=inner, count=5)
    
    result = dataclass_to_dict(outer)
    
    assert isinstance(result, dict)
    assert result["count"] == 5
    assert isinstance(result["inner"], dict)
    assert result["inner"]["name"] == "inner"
    assert result["inner"]["value"] == 10


def test_dataclass_to_dict_with_config_params():
    """Test with actual config dataclasses."""
    from config import ConvTasNetParams
    
    params = ConvTasNetParams(N=128, B=256, H=512)
    
    result = dataclass_to_dict(params)
    
    assert result["N"] == 128
    assert result["B"] == 256
    assert result["H"] == 512
    assert result["P"] == 3  # Default value


def test_dataclass_to_dict_simplenamespace():
    """Test converting SimpleNamespace to dict."""
    obj = SimpleNamespace(name="test", value=42, flag=True)
    
    result = dataclass_to_dict(obj)
    
    assert isinstance(result, dict)
    assert result["name"] == "test"
    assert result["value"] == 42
    assert result["flag"] is True


def test_dataclass_to_dict_non_dataclass_non_namespace():
    """Test that non-dataclass/non-namespace objects are returned as-is."""
    obj = "regular_string"
    
    result = dataclass_to_dict(obj)
    
    assert result == "regular_string"


def test_dataclass_to_dict_dict_passthrough():
    """Test that dicts are returned as-is."""
    obj = {"key": "value", "number": 123}
    
    result = dataclass_to_dict(obj)
    
    assert result == obj


def test_dataclass_to_dict_preserves_types():
    """Test that value types are preserved."""
    obj = SampleDataclass(name="type_test", value=999, flag=False)
    
    result = dataclass_to_dict(obj)
    
    assert isinstance(result["name"], str)
    assert isinstance(result["value"], int)
    assert isinstance(result["flag"], bool)



def test_ensure_dir_creates_directory(tmp_path):
    """Test ensure_dir creates directory."""
    new_dir = tmp_path / "test" / "nested" / "path"
    
    assert not new_dir.exists()
    
    result = ensure_dir(new_dir)
    
    assert new_dir.exists()
    assert new_dir.is_dir()
    assert result == new_dir


def test_ensure_dir_creates_parent_directories(tmp_path):
    """Test ensure_dir creates parent directories."""
    nested_dir = tmp_path / "level1" / "level2" / "level3"
    
    result = ensure_dir(nested_dir)
    
    assert (tmp_path / "level1").exists()
    assert (tmp_path / "level1" / "level2").exists()
    assert nested_dir.exists()
    assert result == nested_dir


def test_ensure_dir_existing_directory_no_error(tmp_path):
    """Test ensure_dir handles existing directory without error."""
    existing_dir = tmp_path / "existing"
    existing_dir.mkdir()
    
    # Should not raise error
    result = ensure_dir(existing_dir)
    
    assert result == existing_dir
    assert existing_dir.exists()


def test_ensure_dir_returns_path_object(tmp_path):
    """Test ensure_dir returns Path object."""
    new_dir = tmp_path / "test"
    
    result = ensure_dir(new_dir)
    
    assert isinstance(result, Path)


def test_ensure_dir_accepts_string_path(tmp_path):
    """Test ensure_dir accepts string paths."""
    new_dir = tmp_path / "string_test"
    new_dir_str = str(new_dir)
    
    result = ensure_dir(new_dir_str)
    
    assert new_dir.exists()
    assert isinstance(result, Path)
    assert result == new_dir


def test_ensure_dir_idempotent(tmp_path):
    """Test ensure_dir can be called multiple times safely."""
    new_dir = tmp_path / "idempotent"
    
    result1 = ensure_dir(new_dir)
    result2 = ensure_dir(new_dir)
    result3 = ensure_dir(new_dir)
    
    assert result1 == result2 == result3
    assert new_dir.exists()
"""Tests for config utility functions."""

import pytest
from dataclasses import dataclass
from types import SimpleNamespace


@dataclass
class SampleDataclass:
    """Sample dataclass for testing."""
    name: str
    value: int
    flag: bool = True


@dataclass
class NestedDataclass:
    """Nested dataclass for testing."""
    inner: SampleDataclass
    count: int


def test_dataclass_to_dict_simple():
    """Test converting simple dataclass to dict."""
    obj = SampleDataclass(name="test", value=42, flag=False)
    
    result = dataclass_to_dict(obj)
    
    assert isinstance(result, dict)
    assert result["name"] == "test"
    assert result["value"] == 42
    assert result["flag"] is False


def test_dataclass_to_dict_with_defaults():
    """Test dataclass with default values."""
    obj = SampleDataclass(name="default_test", value=100)
    
    result = dataclass_to_dict(obj)
    
    assert result["name"] == "default_test"
    assert result["value"] == 100
    assert result["flag"] is True  # Default value


def test_dataclass_to_dict_nested():
    """Test converting nested dataclass."""
    inner = SampleDataclass(name="inner", value=10)
    outer = NestedDataclass(inner=inner, count=5)
    
    result = dataclass_to_dict(outer)
    
    assert isinstance(result, dict)
    assert result["count"] == 5
    assert isinstance(result["inner"], dict)
    assert result["inner"]["name"] == "inner"
    assert result["inner"]["value"] == 10


def test_dataclass_to_dict_with_config_params():
    """Test with actual config dataclasses."""
    from config import ConvTasNetParams
    
    params = ConvTasNetParams(N=128, B=256, H=512)
    
    result = dataclass_to_dict(params)
    
    assert result["N"] == 128
    assert result["B"] == 256
    assert result["H"] == 512
    assert result["P"] == 3  # Default value


def test_dataclass_to_dict_simplenamespace():
    """Test converting SimpleNamespace to dict."""
    obj = SimpleNamespace(name="test", value=42, flag=True)
    
    result = dataclass_to_dict(obj)
    
    assert isinstance(result, dict)
    assert result["name"] == "test"
    assert result["value"] == 42
    assert result["flag"] is True


def test_dataclass_to_dict_non_dataclass_non_namespace():
    """Test that non-dataclass/non-namespace objects are returned as-is."""
    obj = "regular_string"
    
    result = dataclass_to_dict(obj)
    
    assert result == "regular_string"


def test_dataclass_to_dict_dict_passthrough():
    """Test that dicts are returned as-is."""
    obj = {"key": "value", "number": 123}
    
    result = dataclass_to_dict(obj)
    
    assert result == obj


def test_dataclass_to_dict_preserves_types():
    """Test that value types are preserved."""
    obj = SampleDataclass(name="type_test", value=999, flag=False)
    
    result = dataclass_to_dict(obj)
    
    assert isinstance(result["name"], str)
    assert isinstance(result["value"], int)
    assert isinstance(result["flag"], bool)
