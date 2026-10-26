"""Tests for the TF-MossFormer model.

Pure PyTorch, so everything here runs on CPU with no skips (unlike the Mamba
family). Most cases use a tiny config (small D, one block, short STFT) to keep
the suite fast; the three paper sizes are built once, in the parameter-pin test,
because those counts are the regression guard on the paper's reconciliation.
"""

import pytest
import torch
import torch.nn.functional as F

from models import TFMossFormer, get_model
from models.tf_mossformer.local_global_attention import band_mask
from config import load_config_from_dict, TFMossFormerParams
from models.factory import create_model_from_config


def tiny(**overrides):
    """A small but structurally complete TF-MossFormer."""
    kwargs = dict(
        C=2,
        D=32,
        num_blocks=1,
        ffn_hidden_dim=32,
        n_heads=4,
        n_fft=64,
        hop_length=32,
        window_t=7,
        window_f=5,
    )
    kwargs.update(overrides)
    return TFMossFormer(**kwargs)


def test_tf_mossformer_initialization():
    """Model builds and exposes the source count the trainer contract needs."""
    model = tiny()
    assert model is not None
    assert model.C == 2
    assert len(model.separator.blocks) == 1
    # The architectural diff from TF-Locoformer: the block's attention is the
    # gated local+global module, on both TF paths.
    from models.tf_mossformer.local_global_attention import LocalGlobalMHSA

    block = model.separator.blocks[0]
    assert isinstance(block.freq_path.attn, LocalGlobalMHSA)
    assert isinstance(block.frame_path.attn, LocalGlobalMHSA)
    # ...with the per-path window sizes.
    assert block.freq_path.attn.local_attn.window == 5
    assert block.frame_path.attn.local_attn.window == 7


def test_tf_mossformer_forward_pass():
    """2D input [batch, time] -> [batch, C, time]."""
    model = tiny()
    batch_size, time_steps = 2, 4000

    output = model(torch.randn(batch_size, time_steps))

    assert output.shape == (batch_size, 2, time_steps)
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains Inf values"


def test_tf_mossformer_3d_input():
    """3D input [batch, 1, time] — what the trainer actually feeds."""
    model = tiny()
    output = model(torch.randn(2, 1, 4000))
    assert output.shape == (2, 2, 4000)


def test_tf_mossformer_single_source():
    """C=1 (enhancement task) returns [batch, time]."""
    model = tiny(C=1)
    output = model(torch.randn(2, 4000))
    assert output.shape == (2, 4000)


def test_tf_mossformer_length_contract():
    """Output length must equal input length (required by the SI-SDR loss).

    An STFT model only satisfies this because of ``length=`` on the inverse
    STFT, so the interesting lengths are the ones that are not a multiple of the
    hop.
    """
    model = tiny()
    for time_steps in [8000, 8001, 12345]:
        y = model(torch.randn(1, time_steps))
        assert y.shape == (1, 2, time_steps), (
            f"length not preserved for T={time_steps}: {y.shape}"
        )


def test_tf_mossformer_short_input():
    """Input shorter than one STFT window still round-trips.

    With n_fft=64 the signal is framed into 2 frames, so the temporal path's
    sequence is shorter than its own attention window — the banded mask
    degenerates to all-True, which must not break anything.
    """
    model = tiny()
    y = model(torch.randn(1, 48))
    assert y.shape == (1, 2, 48)
    assert torch.isfinite(y).all()


def test_band_mask_shape_and_band():
    """The mask keeps exactly the symmetric neighbourhood of Eq. (3)."""
    mask = band_mask(9, 5, torch.device("cpu"))
    assert mask.shape == (9, 9)
    assert mask.dtype == torch.bool
    # radius = (5 - 1) // 2 = 2
    assert mask[4].nonzero().flatten().tolist() == [2, 3, 4, 5, 6]
    # Truncated at the edges, and every query always sees at least itself, so no
    # row is fully masked (that is what makes the masked softmax NaN-free).
    assert mask[0].nonzero().flatten().tolist() == [0, 1, 2]
    assert mask.any(dim=-1).all()


def _unfold_windowed_attention(attn, x):
    """Reference oracle: true windowed attention built with ``unfold``.

    Gathers, for every query position, the ``w`` keys/values actually inside its
    window (edge positions padded and masked out), then does the softmax over
    that gathered axis. Deliberately the slow, obviously-correct implementation
    the banded-mask version in the model is checked against.
    """
    query, key, value = attn.get_qkv(x)
    if attn.rope is not None:
        query, key = attn.apply_rope(query, key)

    n_batch, n_heads, length, head_dim = query.shape
    radius = (attn.window - 1) // 2

    # Pad the key/value sequence so every window is complete, and track which of
    # the gathered positions are padding.
    key_p = F.pad(key, (0, 0, radius, radius))  # [B, h, L + 2r, d]
    value_p = F.pad(value, (0, 0, radius, radius))
    valid = F.pad(torch.ones(length, dtype=torch.bool), (radius, radius))

    window = 2 * radius + 1
    idx = torch.arange(length)[:, None] + torch.arange(window)[None, :]  # [L, w]
    key_w = key_p[:, :, idx, :]  # [B, h, L, w, d]
    value_w = value_p[:, :, idx, :]
    valid_w = valid[idx]  # [L, w]

    scores = (query[:, :, :, None, :] * key_w).sum(-1) / (head_dim**0.5)  # [B,h,L,w]
    scores = scores.masked_fill(~valid_w, float("-inf"))
    weights = scores.softmax(-1)
    out = (weights[..., None] * value_w).sum(-2)  # [B, h, L, d]

    out = out.transpose(1, 2).reshape(n_batch, length, -1)
    return attn.aggregate_heads(out)


def test_windowed_attention_matches_unfold_oracle():
    """The banded-mask SDPA must equal true windowed attention exactly."""
    from models.tf_mossformer.local_global_attention import WindowedMHSA

    torch.manual_seed(0)
    attn = WindowedMHSA(emb_dim=16, attention_dim=16, window=5, n_heads=4).eval()
    x = torch.randn(2, 13, 16)

    with torch.no_grad():
        fast = attn(x)
        slow = _unfold_windowed_attention(attn, x)

    assert torch.allclose(fast, slow, atol=1e-5), (
        f"max abs diff {(fast - slow).abs().max().item():.2e}"
    )


def test_windowed_attention_is_not_full_attention():
    """Guard against the mask silently becoming a no-op.

    If the window covered the whole sequence the previous test would pass
    against a broken (unmasked) implementation, so pin that a narrow window
    really does differ from global attention on the same weights.
    """
    from models.tf_mossformer.local_global_attention import WindowedMHSA
    from models.tf_mossformer.tf_locoformer_blocks import MultiHeadSelfAttention

    torch.manual_seed(0)
    local = WindowedMHSA(emb_dim=16, attention_dim=16, window=3, n_heads=4).eval()
    glob = MultiHeadSelfAttention(16, attention_dim=16, n_heads=4).eval()
    glob.load_state_dict(local.state_dict())

    x = torch.randn(1, 32, 16)
    with torch.no_grad():
        assert not torch.allclose(local(x), glob(x), atol=1e-4)


@pytest.mark.parametrize(
    "size, kwargs, expected_millions",
    [
        ("S", dict(D=96, num_blocks=4, ffn_hidden_dim=256), 5.92),
        ("M", dict(D=128, num_blocks=6, ffn_hidden_dim=384), 17.34),
        ("L", dict(D=128, num_blocks=9, ffn_hidden_dim=384), 26.00),
    ],
)
def test_tf_mossformer_paper_size_parameter_pins(size, kwargs, expected_millions):
    """Pin the reconciled parameter counts of the three paper sizes.

    These are the numbers the re-implementation is defended with (paper: 5.9/6.0,
    16.9, 25.4 M — the 2.5% excess on M/L is the paper's own inconsistency, see
    LocalGlobalMHSA's docstring). Any accidental change to the local branch, the
    gates or the attention dim moves them, so they are the regression guard on
    the whole reconciliation, not a cosmetic assertion.
    """
    model = TFMossFormer(C=2, **kwargs)
    millions = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    assert millions == pytest.approx(expected_millions, rel=0.01), (
        f"TF-MossFormer({size}) is {millions:.3f}M, pinned at {expected_millions}M"
    )


def test_tf_mossformer_registry():
    """The model is registered and retrievable by name."""
    assert get_model("tf_mossformer") is TFMossFormer


def test_tf_mossformer_default_params_are_paper_size_s():
    """Dataclass defaults must be the paper's size S at the 8 kHz geometry."""
    p = TFMossFormerParams()
    assert (p.D, p.num_blocks, p.ffn_hidden_dim) == (96, 4, 256)
    assert (p.n_heads, p.num_groups) == (4, 4)
    assert (p.window_t, p.window_f) == (31, 7)
    assert (p.conv_kernel_size, p.conv_stride, p.gate_kernel_size) == (4, 1, 4)
    assert (p.n_fft, p.hop_length) == (128, 64)
    assert p.attn_dropout == 0.0
    # SPMamba's removed `sample_rate` field is the precedent: STFT geometry is
    # in samples, never coupled to data.sample_rate.
    assert not hasattr(p, "sample_rate")


def test_tf_mossformer_config_roundtrip():
    """Config dict -> factory build, with the task-driven source-count override."""
    model_block = {
        "model_type": "tf_mossformer",
        "tf_mossformer": {"D": 32, "num_blocks": 1, "ffn_hidden_dim": 32,
                          "n_fft": 64, "hop_length": 32},
    }

    cfg_sb = load_config_from_dict({
        "data": {"dataset_type": "polsess", "task": "SB"},
        "model": model_block,
        "training": {},
    })
    assert isinstance(cfg_sb.model.tf_mossformer, TFMossFormerParams)
    assert cfg_sb.model.tf_mossformer.C == 2
    model = create_model_from_config(cfg_sb.model)
    y = model(torch.randn(1, 4000))
    assert y.shape == (1, 2, 4000)

    cfg_es = load_config_from_dict({
        "data": {"dataset_type": "polsess", "task": "ES"},
        "model": model_block,
        "training": {},
    })
    assert cfg_es.model.tf_mossformer.C == 1
    model_es = create_model_from_config(cfg_es.model)
    assert model_es(torch.randn(1, 4000)).shape == (1, 4000)


def test_tf_mossformer_bfloat16_autocast_forward_is_finite():
    """A bf16-autocast forward must stay finite.

    The model trains under bf16 autocast without a GradScaler (see
    Trainer._setup_amp), and the STFT / complex round-trip is the fragile part:
    torch.complex rejects bf16 outright, so the fp32 casts around it are what
    this exercises. CPU autocast is the CPU-runnable stand-in for the CUDA path.
    """
    model = tiny().eval()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        y = model(torch.randn(2, 4000))
    assert y.shape == (2, 2, 4000)
    assert torch.isfinite(y).all(), "bf16 autocast forward produced NaN/Inf"
