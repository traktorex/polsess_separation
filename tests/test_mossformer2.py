"""Tests for MossFormer2 model.

Uses tiny configs (small N, num_blocks=2) so the suite stays fast and CPU-only.
The paper-faithful ~55.7M instantiation is exercised by the smoke-train step, not
here, to keep unit tests light.
"""

import pytest
import torch

from models import MossFormer2, get_model
from config import load_config_from_dict, MossFormer2Params
from models.factory import create_model_from_config


def test_mossformer2_initialization():
    """Test MossFormer2 can be initialized with default-style params."""
    model = MossFormer2(N=32, C=2, num_blocks=2)
    assert model is not None
    assert model.C == 2
    assert model.model.model.num_spks == 2  # wrapper -> MossFormer2_SS -> MossFormer


def test_mossformer2_forward_pass():
    """Test forward pass with 2D input [batch, time] -> [batch, C, time]."""
    model = MossFormer2(N=32, C=2, num_blocks=2)
    batch_size, time_steps = 2, 8000

    x = torch.randn(batch_size, time_steps)
    output = model(x)

    assert output.shape == (
        batch_size,
        2,
        time_steps,
    ), f"Expected {(batch_size, 2, time_steps)}, got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains Inf values"


def test_mossformer2_3d_input():
    """Test model handles 3D input [batch, 1, time]."""
    model = MossFormer2(N=32, C=2, num_blocks=2)
    batch_size, time_steps = 2, 8000

    x = torch.randn(batch_size, 1, time_steps)
    output = model(x)

    assert output.shape == (batch_size, 2, time_steps)


def test_mossformer2_single_source():
    """Test model with C=1 (enhancement task) returns [batch, time]."""
    model = MossFormer2(N=32, C=1, num_blocks=2)
    batch_size, time_steps = 2, 8000

    x = torch.randn(batch_size, time_steps)
    output = model(x)

    assert output.shape == (batch_size, time_steps)


def test_mossformer2_length_contract():
    """Output length must equal input length (required by the SI-SDR loss).

    The encoder downsamples by stride = kernel_size // 2, so non-multiple input
    lengths are the interesting case — MossFormer pads/trims back internally.
    """
    model = MossFormer2(N=32, C=2, num_blocks=2)
    for time_steps in [8000, 8001, 12345]:
        x = torch.randn(1, time_steps)
        y = model(x)
        assert y.shape == (1, 2, time_steps), f"length not preserved for T={time_steps}: {y.shape}"


def test_mossformer2_custom_config():
    """Test model accepts a custom (small) configuration and forwards."""
    config = {"N": 64, "kernel_size": 16, "C": 2, "num_blocks": 3}
    model = MossFormer2(**config)
    assert model is not None

    x = torch.randn(1, 8000)
    y = model(x)
    assert y.shape == (1, 2, 8000)
    assert not torch.isnan(y).any()


def test_mossformer2_parameter_count():
    """Test a moderate config has a reasonable (>1M) parameter count."""
    model = MossFormer2(N=128, C=2, num_blocks=4)
    num_params = sum(p.numel() for p in model.parameters())
    assert num_params > 1_000_000, f"Expected >1M params, got {num_params/1e6:.2f}M"
    print(f"MossFormer2 (N=128, blocks=4) parameters: {num_params/1e6:.2f}M")


def test_mossformer2_registry():
    """Test the model is registered and retrievable by name."""
    assert get_model("mossformer2") is MossFormer2


def test_mossformer2_config_roundtrip():
    """Test config dict -> factory build, with task-driven source-count override."""
    # SB task forces C=2
    cfg_sb = load_config_from_dict({
        "data": {"dataset_type": "polsess", "task": "SB"},
        "model": {"model_type": "mossformer2", "mossformer2": {"N": 32, "num_blocks": 2}},
        "training": {},
    })
    assert isinstance(cfg_sb.model.mossformer2, MossFormer2Params)
    assert cfg_sb.model.mossformer2.C == 2
    model = create_model_from_config(cfg_sb.model)
    y = model(torch.randn(1, 8000))
    assert y.shape == (1, 2, 8000)

    # ES task forces C=1
    cfg_es = load_config_from_dict({
        "data": {"dataset_type": "polsess", "task": "ES"},
        "model": {"model_type": "mossformer2", "mossformer2": {"N": 32, "num_blocks": 2}},
        "training": {},
    })
    assert cfg_es.model.mossformer2.C == 1


def test_mossformer2_default_params():
    """Test the dataclass defaults match the paper-faithful config (N=512, 24 blocks)."""
    p = MossFormer2Params()
    assert p.N == 512
    assert p.num_blocks == 24
    assert p.kernel_size == 16
    assert p.C == 2
