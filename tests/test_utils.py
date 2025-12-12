"""Tests for utility functions."""

import pytest
import torch
import random
import numpy as np
from utils.common import set_seed


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
