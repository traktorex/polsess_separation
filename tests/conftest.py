"""Pytest configuration and shared fixtures."""

import pytest
import torch
from pathlib import Path


@pytest.fixture
def temp_data_root(tmp_path):
    """Create temporary data directory for testing."""
    return tmp_path / "data"


@pytest.fixture
def device():
    """Return available device (cuda or cpu)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def sample_config():
    """Provide sample configuration for testing."""
    return {
        "N": 256,
        "B": 256,
        "H": 512,
        "P": 3,
        "X": 8,
        "R": 4,
        "C": 1,
        "kernel_size": 16,
        "stride": 8,
    }
