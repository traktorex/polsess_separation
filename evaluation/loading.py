"""Model loading utilities for evaluation."""

from typing import Optional
import torch
from models import get_model
from models.factory import create_model_from_config
from config import Config, ModelConfig
from utils import load_checkpoint_file, count_parameters


def load_model_from_checkpoint_for_eval(
    checkpoint_path: str,
    config: Optional[Config] = None,
    device: str = "cuda",
) -> torch.nn.Module:
    """Load model from checkpoint for evaluation.
    
    This is specifically for evaluation - it only loads model weights,
    not training state (optimizer, scheduler, etc.).
    For resuming training, use Trainer.load_checkpoint() instead.
    """
    print(f"Loading model from {checkpoint_path}...")
    
    checkpoint = load_checkpoint_file(checkpoint_path, device)
    
    # Try to create model from checkpoint config first, fall back to provided config
    if "config" in checkpoint:
        print("Using model config from checkpoint")
        ckpt_config = checkpoint["config"]
        model_type = ckpt_config.get("model", {}).get("model_type", "convtasnet")
        model_class = get_model(model_type)
        
        # Get model-specific parameters
        model_params_dict = ckpt_config.get("model", {}).get(model_type, {})
        if not model_params_dict:
            model_params_dict = ckpt_config.get("model", {})  # Legacy flat config
        
        model = model_class(**model_params_dict)
    else:
        if config is None:
            raise ValueError("No config in checkpoint and no config provided")
        print("Using provided config")
        model = create_model_from_config(config.model)
    
    # Load weights and prepare for eval
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    # Log info
    model_type = checkpoint.get("config", {}).get("model", {}).get("model_type", "unknown")
    print(f"Model loaded: {model_type}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    if "val_sisdr" in checkpoint:
        print(f"  Validation SI-SDR: {checkpoint['val_sisdr']:.2f} dB")
    print(f"  Parameters: {count_parameters(model) / 1e6:.2f}M")
    
    return model
