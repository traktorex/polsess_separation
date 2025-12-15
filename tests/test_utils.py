"""Tests for utility functions."""

import pytest
import torch
import random
import numpy as np
from utils.common import set_seed
from utils import apply_torch_compile, unwrap_compiled_model
from utils.model_utils import count_parameters, load_checkpoint_file


def test_set_seed_deterministic():
    """Test that set_seed produces deterministic results."""
    set_seed(42)
    rand_py = random.random()
    rand_np = np.random.rand()
    rand_torch = torch.rand(1).item()

    set_seed(42)
    assert random.random() == rand_py
    assert np.random.rand() == rand_np
    assert torch.rand(1).item() == rand_torch


def test_set_seed_different_seeds():
    """Test that different seeds produce different results."""
    set_seed(42)
    rand1 = torch.rand(10)

    set_seed(123)
    rand2 = torch.rand(10)

    assert not torch.all(
        rand1 == rand2
    ), "Different seeds should produce different results"


def test_set_seed_cuda_deterministic():
    """Test that CUDA operations are deterministic after set_seed."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    set_seed(42)
    x = torch.randn(10, 10, device="cuda")
    y = torch.randn(10, 10, device="cuda")
    result1 = torch.mm(x, y)

    set_seed(42)
    x = torch.randn(10, 10, device="cuda")
    y = torch.randn(10, 10, device="cuda")
    result2 = torch.mm(x, y)

    assert torch.allclose(result1, result2), "CUDA operations should be deterministic"


def test_apply_torch_compile_returns_model():
    """Test apply_torch_compile returns a model."""
    from models import ConvTasNet
    
    model = ConvTasNet(N=64, B=64, H=128, P=3, X=4, R=2, C=1)
    result = apply_torch_compile(model)
    
    # Should return either compiled or original model
    assert result is not None
    assert hasattr(result, "forward")


def test_apply_torch_compile_no_crash_cpu():
    """Test apply_torch_compile doesn't crash on CPU."""
    from models import ConvTasNet
    
    model = ConvTasNet(N=64, B=64, H=128, P=3, X=4, R=2, C=1)
    result = apply_torch_compile(model)
    
    # Should work even if compilation not supported
    assert result is not None


def test_unwrap_compiled_model_returns_same_if_not_compiled():
    """Test unwrapping non-compiled model returns same model."""
    from models import ConvTasNet
    
    model = ConvTasNet(N=64, B=64, H=128, P=3, X=4, R=2, C=1)
    unwrapped = unwrap_compiled_model(model)
    
    assert unwrapped is model  # Should return same reference


def test_unwrap_compiled_model_extracts_original():
    """Test unwrapping compiled model extracts original."""
    from models import ConvTasNet
    
    model = ConvTasNet(N=64, B=64, H=128, P=3, X=4, R=2, C=1)
    
    # Simulate compiled model by adding _orig_mod attribute
    class MockCompiledModel:
        def __init__(self, original):
            self._orig_mod = original
    
    compiled = MockCompiledModel(model)
    unwrapped = unwrap_compiled_model(compiled)
    
    assert unwrapped is model


def test_count_parameters():
    """Test parameter counting utility."""
    from models import ConvTasNet
    
    model = ConvTasNet(N=64, B=64, H=128, P=3, X=4, R=2, C=1)
    
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    
    assert total_params > 0
    assert trainable_params > 0
    assert trainable_params <= total_params


def test_count_parameters_frozen_model():
    """Test parameter counting with frozen parameters."""
    from models import ConvTasNet
    
    model = ConvTasNet(N=64, B=64, H=128, P=3, X=4, R=2, C=1)
    
    # Freeze first layer
    for param in list(model.parameters())[:5]:
        param.requires_grad = False
    
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    
    assert trainable_params < total_params


def test_load_checkpoint_file(tmp_path):
    """Test loading checkpoint file."""
    checkpoint_path = tmp_path / "test.pt"
    
    data = {"epoch": 10, "val_sisdr": 15.5}
    torch.save(data, checkpoint_path)
    
    loaded = load_checkpoint_file(str(checkpoint_path), device="cpu")
    
    assert loaded["epoch"] == 10
    assert loaded["val_sisdr"] == 15.5


def test_load_checkpoint_file_nonexistent_raises_error():
    """Test loading nonexistent checkpoint raises error."""
    with pytest.raises(FileNotFoundError):
        load_checkpoint_file("/nonexistent/path.pt", device="cpu")
