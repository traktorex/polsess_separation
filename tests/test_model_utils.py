"""Tests for model utility functions."""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import patch, MagicMock
from utils.model_utils import (
    unwrap_compiled_model,
    count_parameters,
    format_parameter_count,
    load_checkpoint_file,
    load_model_from_checkpoint,
)


class TestUnwrapCompiledModel:
    """Test unwrapping torch.compile models."""

    def test_unwrap_compiled_model_with_compiled(self):
        """Test unwrapping a torch.compile compiled model."""
        # Create a simple model
        original_model = nn.Linear(10, 5)
        
        # Simulate compiled model by adding _orig_mod attribute
        compiled_model = MagicMock()
        compiled_model._orig_mod = original_model
        
        unwrapped = unwrap_compiled_model(compiled_model)
        
        assert unwrapped is original_model, "Should return original model"

    def test_unwrap_compiled_model_without_compiled(self):
        """Test unwrapping a regular (non-compiled) model."""
        model = nn.Linear(10, 5)
        
        unwrapped = unwrap_compiled_model(model)
        
        assert unwrapped is model, "Should return same model if not compiled"


class TestCountParameters:
    """Test parameter counting."""

    def test_count_parameters_total(self):
        """Test counting total parameters."""
        # Linear(10, 5) has 10*5 + 5 = 55 parameters
        model = nn.Linear(10, 5)
        
        count = count_parameters(model, trainable_only=False)
        
        assert count == 55, f"Expected 55 parameters, got {count}"

    def test_count_parameters_trainable_only(self):
        """Test counting only trainable parameters."""
        model = nn.Linear(10, 5)
        
        # Freeze half the parameters
        model.weight.requires_grad = False
        
        trainable_count = count_parameters(model, trainable_only=True)
        total_count = count_parameters(model, trainable_only=False)
        
        assert trainable_count < total_count, "Trainable count should be less than total"
        assert trainable_count == 5, "Only bias (5 params) should be trainable"

    def test_count_parameters_multi_layer(self):
        """Test counting parameters in multi-layer model."""
        model = nn.Sequential(
            nn.Linear(10, 20),  # 10*20 + 20 = 220
            nn.Linear(20, 5),   # 20*5 + 5 = 105
        )
        
        count = count_parameters(model, trainable_only=False)
        
        assert count == 325, f"Expected 325 parameters, got {count}"


class TestFormatParameterCount:
    """Test parameter count formatting."""

    def test_format_parameter_count_millions(self):
        """Test formatting millions of parameters."""
        result = format_parameter_count(1_234_567)
        
        assert "M" in result, "Should use M suffix for millions"
        assert "1.23" in result, "Should format to 2 decimal places"

    def test_format_parameter_count_thousands(self):
        """Test formatting thousands of parameters."""
        result = format_parameter_count(12_345)
        
        assert "K" in result, "Should use K suffix for thousands"
        assert "12.34" in result or "12.35" in result, "Should format to 2 decimal places"

    def test_format_parameter_count_small(self):
        """Test formatting small parameter counts."""
        result = format_parameter_count(999)
        
        assert "999" in result, "Should show exact count for small numbers"
        assert "K" not in result and "M" not in result, "Should not use suffix"

    def test_format_parameter_count_exact_million(self):
        """Test formatting exact million."""
        result = format_parameter_count(1_000_000)
        
        assert "1.00M" in result, "Should format exact million correctly"


class TestCheckpointLoading:
    """Test checkpoint loading utilities."""

    def test_load_checkpoint_file_exists(self, tmp_path):
        """Test loading existing checkpoint file."""
        checkpoint_path = tmp_path / "test_checkpoint.pth"
        
        # Create dummy checkpoint
        test_data = {"model_state_dict": {}, "epoch": 10}
        torch.save(test_data, checkpoint_path)
        
        loaded = load_checkpoint_file(str(checkpoint_path), device="cpu")
        
        assert loaded["epoch"] == 10, "Should load checkpoint data"

    def test_load_checkpoint_file_not_found(self):
        """Test loading non-existent checkpoint raises error."""
        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            load_checkpoint_file("/nonexistent/checkpoint.pth", device="cpu")

    def test_load_model_from_checkpoint(self, tmp_path):
        """Test loading model weights from checkpoint."""
        # Create a simple model
        model = nn.Linear(10, 5)
        
        # Save checkpoint
        checkpoint_path = tmp_path / "model.pth"
        checkpoint_data = {
            "model_state_dict": model.state_dict(),
            "epoch": 5,
            "loss": 0.123,
        }
        torch.save(checkpoint_data, checkpoint_path)
        
        # Create new model with different weights
        new_model = nn.Linear(10, 5)
        
        # Load checkpoint into new model
        loaded_checkpoint = load_model_from_checkpoint(
            str(checkpoint_path), new_model, device="cpu", strict=True
        )
        
        # Verify weights match
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2), "Weights should match after loading"
        
        # Verify checkpoint metadata returned
        assert loaded_checkpoint["epoch"] == 5
        assert loaded_checkpoint["loss"] == 0.123

    def test_load_model_from_checkpoint_with_compiled_model(self, tmp_path):
        """Test loading checkpoint into compiled model unwraps correctly."""
        # Create model and save checkpoint
        model = nn.Linear(10, 5)
        checkpoint_path = tmp_path / "model.pth"
        torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
        
        # Create "compiled" model with _orig_mod attribute
        unwrapped_model = nn.Linear(10, 5)
        
        # Simulate torch.compile wrapper
        class CompiledWrapper:
            def __init__(self, model):
                self._orig_mod = model
        
        compiled_model = CompiledWrapper(unwrapped_model)
        
        # Load checkpoint
        load_model_from_checkpoint(str(checkpoint_path), compiled_model, device="cpu")
        
        # Verify that weights were loaded into the unwrapped model
        # by checking they match original model
        for p1, p2 in zip(model.parameters(), unwrapped_model.parameters()):
            assert torch.allclose(p1, p2), "Weights should be loaded into unwrapped model"

    def test_load_model_from_checkpoint_strict_mode(self, tmp_path):
        """Test strict mode in checkpoint loading."""
        # Create model with different architecture
        checkpoint_model = nn.Linear(10, 5)
        load_model = nn.Linear(10, 10)  # Different size!
        
        checkpoint_path = tmp_path / "model.pth"
        torch.save({"model_state_dict": checkpoint_model.state_dict()}, checkpoint_path)
        
        # Should raise error in strict mode
        with pytest.raises(RuntimeError):
            load_model_from_checkpoint(
                str(checkpoint_path), load_model, device="cpu", strict=True
            )
