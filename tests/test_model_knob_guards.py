"""Guards for silently-misleading model constructor knobs (Work Package C,
items C2 and part of C3 / gap 9).

Five models (ConvTasNet, DPRNN, SepFormer, MambaTasNet, DPMamba) pass a
`stride` argument to a SpeechBrain `Decoder`, but their matching SpeechBrain
`Encoder` (speechbrain.lobes.models.dual_path.Encoder) hardcodes
`stride=kernel_size // 2` and has no `stride` parameter at all. A non-default
`stride` therefore desyncs the encoder/decoder frame rate silently. Each of
the five constructors now asserts `stride == kernel_size // 2` as its first
statement (before any layer — SpeechBrain or Mamba — is built), so the guard
fires on CPU with no GPU/CUDA involvement even for the Mamba-family models.

SPMamba's `window` constructor param is similarly cosmetic: `forward()`
hardcodes `torch.hann_window(...)` regardless of what's passed. That
constructor now asserts `window == "hann"` for the same reason.

MossFormer2 and SPMamba (for the stride check) are intentionally excluded:
MossFormer2 doesn't expose a `stride` parameter at all, and SPMamba's
`stride` is a genuine STFT hop length (not fed to a SpeechBrain Encoder), so
the kernel_size//2 lock does not apply to it.
"""

import pytest

from models import ConvTasNet, DPRNN, SepFormer
from models.mamba import MAMBA_AVAILABLE

if MAMBA_AVAILABLE:
    from models import MambaTasNet, DPMamba, SPMamba
else:  # CPU-only checkout without mamba-ssm: the Mamba cases below are skipped
    MambaTasNet = DPMamba = SPMamba = None

needs_mamba = pytest.mark.skipif(not MAMBA_AVAILABLE, reason="requires mamba-ssm")

# (model_class, kwargs with a *valid* stride == kernel_size // 2, sized small
# to keep construction fast). Only construction is exercised — never forward()
# — so this is safe on CPU even for the Mamba-family models (confirmed: their
# constructors build plain nn.Module layers; only the actual selective-scan
# forward pass needs a CUDA kernel).
STRIDE_GUARD_CASES = [
    pytest.param(
        ConvTasNet,
        dict(N=8, B=8, H=16, P=3, X=1, R=1, C=1, kernel_size=16),
        id="convtasnet",
    ),
    pytest.param(
        DPRNN,
        dict(N=8, C=1, num_layers=1, chunk_size=10, hidden_size=8, kernel_size=16),
        id="dprnn",
    ),
    pytest.param(
        SepFormer,
        dict(
            N=8, C=2, num_blocks=1, num_layers=1, d_model=8, nhead=2, d_ffn=16,
            chunk_size=10, kernel_size=16,
        ),
        id="sepformer",
    ),
    pytest.param(
        MambaTasNet,
        dict(N=8, C=1, bot_dim=8, n_mamba=1, d_state=4, kernel_size=16),
        id="mamba_tasnet",
        marks=needs_mamba,
    ),
    pytest.param(
        DPMamba,
        dict(N=8, C=1, num_layers=1, chunk_size=10, n_mamba_dp=1, d_state=4, kernel_size=16),
        id="dpmamba",
        marks=needs_mamba,
    ),
]


@pytest.mark.parametrize("model_class, kwargs", STRIDE_GUARD_CASES)
def test_non_default_stride_raises(model_class, kwargs):
    """stride != kernel_size // 2 must raise before any layer is constructed."""
    kernel_size = kwargs["kernel_size"]
    bad_stride = kernel_size // 2 + 1  # anything other than kernel_size // 2
    with pytest.raises(AssertionError, match="kernel_size // 2"):
        model_class(**kwargs, stride=bad_stride)


@pytest.mark.parametrize("model_class, kwargs", STRIDE_GUARD_CASES)
def test_default_stride_constructs_without_error(model_class, kwargs):
    """The guard must not reject the legitimate stride == kernel_size // 2 case."""
    kernel_size = kwargs["kernel_size"]
    model = model_class(**kwargs, stride=kernel_size // 2)
    assert model is not None


@needs_mamba
def test_spmamba_non_hann_window_raises():
    """SPMamba.forward() hardcodes torch.hann_window regardless of `window`."""
    with pytest.raises(AssertionError, match="hann"):
        SPMamba(
            n_fft=64, stride=16, input_dim=16, n_srcs=1, n_layers=1,
            lstm_hidden_units=16, attn_n_head=2, attn_approx_qk_dim=32,
            emb_dim=4, emb_ks=2, window="hamming",
        )
