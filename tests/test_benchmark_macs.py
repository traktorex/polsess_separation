"""CPU unit tests for the MAC-counting corrections in scripts/benchmark_inference.py.

Currently covers the `F.scaled_dot_product_attention` correction (added 2026-09-07
for TF-MossFormer). SDPA is one fused aten op, so none of ptflops' Python-level
patches fire inside it and an attention core written with SDPA used to contribute
*zero* MACs — the failure mode `test_sdpa_is_invisible_to_ptflops_alone` pins.

The banded-mask case is the one that matters for TF-MossFormer specifically: its
local branch is dense SDPA with a sliding-window boolean mask, so the counter has
to charge the band, not the full score matrix.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ptflops import get_model_complexity_info

# Deliberately imported as a bare name as well: this binds the function into THIS
# module's namespace at import time, which is exactly the alias `_patch_sdpa` has
# to reach (the same trick `models/mossformer2/mossformer2_block.py` plays with
# `einsum`). `test_patch_reaches_module_level_aliases` uses it.
from torch.nn.functional import scaled_dot_product_attention

from scripts.benchmark_inference import (
    _patch_sdpa,
    _sdpa_macs,
    _unpatch_sdpa,
    count_macs,
)

# Geometry of the tiny model below. T = SEQ * DIM keeps the [B, 1, T] waveform
# shape `count_macs` insists on while giving a [B, SEQ, DIM] sequence to attend.
DIM, HEADS, SEQ = 8, 2, 16
HEAD_DIM = DIM // HEADS
NUM_SAMPLES = SEQ * DIM
LINEAR_MACS = SEQ * DIM * DIM  # the single nn.Linear, batch 1


def band_mask(n: int, w: int) -> torch.Tensor:
    """Symmetric banded boolean mask (True = attend), as the local branch uses."""
    i = torch.arange(n)
    return (i[None, :] - i[:, None]).abs() <= (w - 1) // 2


class _TinySDPA(nn.Module):
    """[B, 1, T] -> one Linear -> one SDPA call -> [B, T]. One of each, no more."""

    def __init__(self, window=None):
        super().__init__()
        self.proj = nn.Linear(DIM, DIM, bias=False)
        self.window = window

    def forward(self, x):
        b = x.shape[0]
        h = self.proj(x.reshape(b, SEQ, DIM))
        q = h.reshape(b, SEQ, HEADS, HEAD_DIM).transpose(1, 2)  # [B, h, SEQ, d]
        mask = band_mask(SEQ, self.window) if self.window else None
        out = F.scaled_dot_product_attention(q, q, q, attn_mask=mask)
        return out.transpose(1, 2).reshape(b, SEQ * DIM)


# --------------------------------------------------------------------------- #
# the closed form
# --------------------------------------------------------------------------- #

def test_sdpa_macs_dense_closed_form():
    """B*h*L_q*L_k*(d_qk + d_v), with the heads folded into the batch dims."""
    q = torch.zeros(2, 3, 5, 4)   # [B=2, h=3, L=5, d=4]
    v = torch.zeros(2, 3, 5, 6)   # a wider value head, to keep the two terms apart
    assert _sdpa_macs(q, q, v) == 2 * 3 * 5 * 5 * (4 + 6)


def test_sdpa_macs_banded_mask_counts_only_the_band():
    """A windowed branch is charged mask.sum() score entries, not L_q*L_k."""
    q = torch.zeros(1, 4, 32, 8)
    mask = band_mask(32, 7)
    kept = int(mask.sum())
    assert kept < 32 * 32  # the band really is sparse
    assert _sdpa_macs(q, q, q, attn_mask=mask) == 1 * 4 * kept * (8 + 8)


def test_sdpa_macs_mask_density_is_broadcast_invariant():
    """A [L, L] mask and its [B, h, L, L] expansion must cost the same."""
    q = torch.zeros(2, 3, 16, 4)
    mask = band_mask(16, 5)
    expanded = mask.expand(2, 3, 16, 16)
    assert _sdpa_macs(q, q, q, attn_mask=mask) == _sdpa_macs(q, q, q, attn_mask=expanded)


def test_sdpa_macs_float_mask_counts_finite_entries():
    """Additive float masks exclude with -inf; count what survives."""
    q = torch.zeros(1, 1, 8, 4)
    mask = torch.zeros(8, 8).masked_fill(~band_mask(8, 3), float("-inf"))
    kept = int(band_mask(8, 3).sum())
    assert _sdpa_macs(q, q, q, attn_mask=mask) == kept * (4 + 4)


def test_sdpa_macs_causal_square_counts_the_triangle():
    q = torch.zeros(1, 2, 10, 4)
    triangle = 10 * 11 // 2
    assert _sdpa_macs(q, q, q, is_causal=True) == 2 * triangle * (4 + 4)


# --------------------------------------------------------------------------- #
# the patch
# --------------------------------------------------------------------------- #

def test_sdpa_is_invisible_to_ptflops_alone():
    """The bug being fixed: without the patch the attention core counts as zero."""
    model = _TinySDPA().eval()
    macs, _ = get_model_complexity_info(
        model, (1, NUM_SAMPLES), as_strings=False,
        print_per_layer_stat=False, verbose=False,
        input_constructor=lambda shape: torch.randn(1, *shape),
    )
    assert int(macs) == LINEAR_MACS


def test_count_macs_includes_the_dense_sdpa_core():
    model = _TinySDPA().eval()
    core = HEADS * SEQ * SEQ * (HEAD_DIM + HEAD_DIM)
    assert count_macs(model, NUM_SAMPLES, "cpu") == LINEAR_MACS + core


def test_count_macs_charges_a_windowed_branch_its_banded_cost():
    window = 5
    model = _TinySDPA(window=window).eval()
    kept = int(band_mask(SEQ, window).sum())
    core = HEADS * kept * (HEAD_DIM + HEAD_DIM)
    total = count_macs(model, NUM_SAMPLES, "cpu")
    assert total == LINEAR_MACS + core
    # ...and that is genuinely cheaper than the dense count it replaces.
    assert total < count_macs(_TinySDPA().eval(), NUM_SAMPLES, "cpu")


def test_patch_reaches_module_level_aliases():
    """`from torch.nn.functional import scaled_dot_product_attention` must be patched too."""
    collected: list = []
    patched = _patch_sdpa(collected)
    try:
        assert scaled_dot_product_attention is not None
        q = torch.zeros(1, 2, 4, 3)
        # Call through THIS module's bare alias, not through torch.nn.functional.
        globals()["scaled_dot_product_attention"](q, q, q)
    finally:
        _unpatch_sdpa(patched)
    assert collected == [2 * 4 * 4 * (3 + 3)]


def test_unpatch_restores_the_original_function():
    original = F.scaled_dot_product_attention
    original_alias = globals()["scaled_dot_product_attention"]
    patched = _patch_sdpa([])
    assert F.scaled_dot_product_attention is not original
    _unpatch_sdpa(patched)
    assert F.scaled_dot_product_attention is original
    assert globals()["scaled_dot_product_attention"] is original_alias


def test_patch_forwards_keyword_calls_unchanged():
    """The vendored TF-Locoformer passes q/k/v by keyword; output must be identical."""
    torch.manual_seed(0)
    q = torch.randn(1, 2, 6, 4)
    expected = F.scaled_dot_product_attention(query=q, key=q, value=q, attn_mask=None)
    collected: list = []
    patched = _patch_sdpa(collected)
    try:
        got = F.scaled_dot_product_attention(query=q, key=q, value=q, attn_mask=None)
    finally:
        _unpatch_sdpa(patched)
    assert torch.allclose(got, expected)
    assert collected == [2 * 6 * 6 * (4 + 4)]


# --------------------------------------------------------------------------- #
# the real consumer: the vendored TF-Locoformer (and, by construction,
# TF-MossFormer, whose block is the same one with a different `attn`)
# --------------------------------------------------------------------------- #

def test_tf_locoformer_attention_cores_are_counted():
    """Every block's frequency and temporal attention must reach the collector.

    TF-Locoformer runs one SDPA per (block, path): the frequency path attends
    over F bins for each of B*T frames, the temporal path over T frames for each
    of B*F bins. WHAMR-medium constructor kwargs, with `n_layers` trimmed to 2 so
    the test builds in a second on CPU — the layer count is the one thing the
    expectation is linear in, so trimming it tests the multiplication rather than
    hiding it.
    """
    from asr_pipeline.vendor.tf_locoformer import WHAMR_MEDIUM_KWARGS
    from asr_pipeline.vendor.tf_locoformer.tflocoformer_separator import (
        TFLocoformerSeparator,
    )

    n_layers, batch, frames, freqs = 2, 1, 6, 5
    kwargs = dict(WHAMR_MEDIUM_KWARGS, n_layers=n_layers)
    head_dim = kwargs["attention_dim"] // kwargs["n_heads"]
    heads = kwargs["n_heads"]

    model = TFLocoformerSeparator(**kwargs).eval()
    spec = torch.zeros(batch, frames, freqs, dtype=torch.complex64)

    collected: list = []
    patched = _patch_sdpa(collected)
    try:
        with torch.no_grad():
            model(spec)
    finally:
        _unpatch_sdpa(patched)

    per_block = (
        batch * frames * heads * freqs * freqs * 2 * head_dim   # frequency path
        + batch * freqs * heads * frames * frames * 2 * head_dim  # temporal path
    )
    assert len(collected) == 2 * n_layers  # one attention per path per block
    assert sum(collected) == n_layers * per_block
