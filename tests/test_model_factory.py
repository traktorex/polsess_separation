"""Tests for model factory pattern."""

import pytest
import torch
from config import Config, ModelConfig
from models.factory import create_model_from_config


class TestModelFactory:
    """Test model factory creation."""

    def test_create_convtasnet_from_config(self):
        """Test creating ConvTasNet via factory."""
        config = Config()
        config.model.model_type = "convtasnet"
        model = create_model_from_config(config.model)
        
        assert model is not None
        assert hasattr(model, "forward")
        assert model.C == config.model.convtasnet.C

    def test_create_model_with_summary_info(self):
        """Test factory populates summary_info."""
        config = Config()
        summary = {}
        model = create_model_from_config(config.model, summary)
        
        assert "model_params_millions" in summary
        assert summary["model_params_millions"] > 0
        assert "model_type" in summary
        assert summary["model_type"] == config.model.model_type

    def test_create_model_invalid_type_raises_error(self):
        """Test invalid model type raises error."""
        config = ModelConfig(model_type="invalid_model_xyz")
        
        with pytest.raises(ValueError, match="Unknown model type"):
            create_model_from_config(config)

    def test_create_model_missing_params_raises_error(self):
        """Test missing model params raises error."""
        config = ModelConfig(model_type="convtasnet")
        # Don't set convtasnet params
        config.convtasnet = None
        
        with pytest.raises(ValueError, match="not configured"):
            create_model_from_config(config)

    def test_create_all_model_types(self):
        """Test all supported models can be created."""
        # Note: sepformer not included as it's not in default Config
        model_types = ["convtasnet", "dprnn", "spmamba"]
        
        for model_type in model_types:
            config = Config()
            config.model.model_type = model_type
            config.model.__post_init__()  # Initialize model params
            model = create_model_from_config(config.model)
            
            assert model is not None, f"Failed to create {model_type}"
            assert hasattr(model, "forward"), f"{model_type} missing forward method"

    def test_create_sepformer_model(self):
        """Test SepFormer model creation separately (not in default Config)."""
        config = Config()
        config.model.model_type = "sepformer"
        config.model.__post_init__()  # This will initialize sepformer params
        
        model = create_model_from_config(config.model)
        
        assert model is not None
        assert hasattr(model, "forward")
        # SepFormer specific attributes
        assert hasattr(model, "encoder")
        assert hasattr(model, "decoder")

    def test_create_model_with_custom_params(self):
        """Test creating model with custom parameters."""
        config = Config()
        config.model.model_type = "convtasnet"
        config.model.convtasnet.N = 128  # Custom value
        config.model.convtasnet.B = 128
        
        model = create_model_from_config(config.model)
        
        # ConvTasNet exposes N but not B (B is internal to mask_net)
        assert model.N == 128

    def test_model_forward_pass_after_factory(self):
        """Test that factory-created model can perform forward pass."""
        config = Config()
        config.model.model_type = "convtasnet"
        model = create_model_from_config(config.model)
        
        # Test forward pass
        x = torch.randn(2, 8000)
        output = model(x)
        
        assert output.shape[0] == 2
        assert not torch.isnan(output).any()

    def test_summary_info_parameter_count_accuracy(self):
        """Test that summary info parameter count is accurate."""
        config = Config()
        summary = {}
        model = create_model_from_config(config.model, summary)
        
        # Compute actual parameter count
        actual_params = sum(p.numel() for p in model.parameters()) / 1e6
        
        assert abs(summary["model_params_millions"] - actual_params) < 0.01
