"""Tests for SPMamba3 model (SPMamba with Mamba-3 blocks).

SPMamba3 requires CUDA + Mamba-3 (mamba-ssm built from source, triton>=3.5).
The whole file is skipped unless both are present (i.e. run in venv_mamba3 on a
GPU box). Uses reduced config parameters for memory efficiency.
"""

import pytest
import torch

from models import SPMamba3, MAMBA3_AVAILABLE, MAMBA_MODELS, MODELS

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not MAMBA3_AVAILABLE,
    reason="SPMamba3 requires CUDA + Mamba-3 (build mamba-ssm from source in venv_mamba3)",
)


@pytest.fixture(autouse=True)
def clear_cuda_cache():
    """Clear CUDA cache before/after each test to prevent OOM."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture
def device():
    return "cuda"


@pytest.fixture
def spmamba3_config():
    """Minimal SPMamba3 configuration to fit alongside other GPU workloads.

    emb_dim=16, emb_ks=4 -> in_channels=64; expand=2 -> inner 128; headdim=32 -> 4 heads.
    """
    return {
        "n_fft": 256,
        "stride": 64,
        "input_dim": 64,
        "n_srcs": 1,
        "n_layers": 2,
        "emb_dim": 16,
        "emb_ks": 4,
        "lstm_hidden_units": 64,
        "attn_n_head": 2,
        "attn_approx_qk_dim": 64,
        "d_state": 16,
        "headdim": 32,
        "expand": 2,
    }


def test_spmamba3_initialization(spmamba3_config):
    model = SPMamba3(**spmamba3_config)
    assert model is not None
    assert model.n_srcs == spmamba3_config["n_srcs"]
    assert model.n_fft == spmamba3_config["n_fft"]


def test_spmamba3_forward_pass(spmamba3_config, device):
    model = SPMamba3(**spmamba3_config).to(device)
    batch_size, time_steps = 1, 4000
    x = torch.randn(batch_size, time_steps).to(device)
    output = model(x)
    assert output.shape == (batch_size, time_steps)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_spmamba3_3d_input(spmamba3_config, device):
    model = SPMamba3(**spmamba3_config).to(device)
    batch_size, time_steps = 1, 4000
    x = torch.randn(batch_size, 1, time_steps).to(device)
    output = model(x)
    assert output.shape == (batch_size, time_steps)


def test_spmamba3_separation_task(spmamba3_config, device):
    cfg = {**spmamba3_config, "n_srcs": 2}
    model = SPMamba3(**cfg).to(device)
    batch_size, time_steps = 1, 4000
    x = torch.randn(batch_size, time_steps).to(device)
    output = model(x)
    assert output.shape == (batch_size, 2, time_steps)


def test_spmamba3_backward_pass(spmamba3_config, device):
    """A loss.backward() must flow through the Mamba-3 SISO kernel cleanly."""
    model = SPMamba3(**spmamba3_config).to(device)
    x = torch.randn(1, 4000, device=device)
    loss = model(x).float().pow(2).mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)
    assert all(torch.isfinite(g).all() for g in grads if g is not None)


def test_spmamba3_different_layers(spmamba3_config, device):
    for n_layers in [2, 4]:
        model = SPMamba3(**{**spmamba3_config, "n_layers": n_layers}).to(device)
        x = torch.randn(1, 4000).to(device)
        assert model(x).shape == (1, 4000)


def test_spmamba3_different_nfft(spmamba3_config, device):
    for n_fft in [128, 256]:
        model = SPMamba3(**{**spmamba3_config, "n_fft": n_fft, "stride": n_fft // 4}).to(device)
        x = torch.randn(1, 4000).to(device)
        assert model(x).shape == (1, 4000)


def test_spmamba3_parameter_count(spmamba3_config):
    model = SPMamba3(**spmamba3_config)
    num_params = sum(p.numel() for p in model.parameters())
    assert num_params > 10_000
    assert num_params < 50_000_000


def test_spmamba3_preserves_input_length(spmamba3_config, device):
    model = SPMamba3(**spmamba3_config).to(device)
    for length in [4000, 8000]:
        x = torch.randn(1, length).to(device)
        output = model(x)
        assert output.shape[1] == length, f"Length mismatch: {output.shape[1]} != {length}"


def test_spmamba3_output_dtype(spmamba3_config, device):
    """Mamba-3 runs bf16 internally; the output must come back as the input dtype."""
    model = SPMamba3(**spmamba3_config).to(device)
    x = torch.randn(1, 4000, dtype=torch.float32).to(device)
    assert model(x).dtype == torch.float32


def test_spmamba3_different_d_state(spmamba3_config, device):
    for d_state in [16, 32]:
        model = SPMamba3(**{**spmamba3_config, "d_state": d_state}).to(device)
        x = torch.randn(1, 4000).to(device)
        output = model(x)
        assert output.shape == (1, 4000)
        assert not torch.isnan(output).any()


# ---------------------------------------------------------------------------
# Head-divisibility constraint (R1): the real rule is
# (expand * in_channels) % headdim == 0, NOT in_channels % headdim.
# ---------------------------------------------------------------------------

def test_spmamba3_headdim_constraint_raises(spmamba3_config):
    """headdim that does not divide expand*in_channels must raise.

    in_channels = emb_dim*emb_ks = 64; expand=2 -> 128; headdim=48 -> 128 % 48 != 0.
    """
    with pytest.raises(ValueError, match="divisible by headdim"):
        SPMamba3(**{**spmamba3_config, "headdim": 48})


def test_spmamba3_headdim_uses_expand():
    """Regression for R1: a headdim larger than in_channels but dividing
    expand*in_channels is VALID (nheads=1). The old `in_channels % headdim`
    check would have wrongly rejected this.
    """
    from models.spmamba3 import Mamba3Block
    # in_channels=64, expand=2 -> 128; headdim=128 -> 128 % 128 == 0 (nheads=1).
    blk = Mamba3Block(in_channels=64, n_layer=1, bidirectional=False, d_state=16,
                      headdim=128, expand=2)
    assert blk is not None
    # And the old (wrong) invariant would have failed here:
    assert 64 % 128 != 0 and (2 * 64) % 128 == 0


# ---------------------------------------------------------------------------
# Registry / config plumbing
# ---------------------------------------------------------------------------

def test_spmamba3_registered():
    assert "spmamba3" in MODELS
    assert MODELS["spmamba3"] is SPMamba3
    # Must inherit bf16/no-GradScaler + torch.compile-skip via MAMBA_MODELS.
    assert "spmamba3" in MAMBA_MODELS


def test_spmamba3_config_roundtrip_and_build(device):
    """load_config_from_dict -> create_model_from_config builds an SPMamba3, and
    the task post-init forces n_srcs to match (SB -> 2)."""
    from config import load_config_from_dict
    from models.factory import create_model_from_config

    cfg_dict = {
        "data": {"dataset_type": "polsess", "task": "SB"},
        "model": {
            "model_type": "spmamba3",
            "spmamba3": {
                "n_layers": 2, "emb_dim": 16, "emb_ks": 4, "lstm_hidden_units": 64,
                "attn_n_head": 2, "attn_approx_qk_dim": 64, "d_state": 16,
                "headdim": 32, "expand": 2, "n_srcs": 1,  # overridden to 2 by SB task
            },
        },
        "training": {"use_wandb": False, "device": "cuda"},
    }
    config = load_config_from_dict(cfg_dict)
    assert config.model.spmamba3.n_srcs == 2  # SB task post-init
    model = create_model_from_config(config.model).to(device)
    assert type(model).__name__ == "SPMamba3"
    x = torch.randn(1, 4000, device=device)
    assert model(x).shape == (1, 2, 4000)
