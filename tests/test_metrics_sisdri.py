"""Parity tests for the shared SI-SDR/SI-SDRi helper (survey gap 12).

`compute_sisdr_and_sisdri` replaced copy-pasted baseline logic in
`evaluate.evaluate_model` and `Trainer._compute_sisdri`. This test uses synthetic
signals rather than a real checkpoint: on CPU/fp32 tensors it asserts that the
helper reproduces the *pre-refactor formulas of both call sites* exactly. The
reference formulas below are transcribed verbatim from the code that was
replaced and serve as the golden reference implementation.
"""

import torch
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr

from utils.metrics import compute_sisdr_and_sisdri


def _metrics():
    si_sdr_metric = ScaleInvariantSignalDistortionRatio()
    pit_loss = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
    return si_sdr_metric, pit_loss


# --- Reference implementations (pre-refactor formulas) ------------------------

def _reference_sb(estimates, clean, mix, si_sdr_metric, pit_loss):
    """evaluate.evaluate_model SB path == Trainer._compute_sisdri SB path."""
    min_len = min(estimates.shape[-1], clean.shape[-1])
    estimates = estimates[..., :min_len]
    clean = clean[..., :min_len]
    mix_trimmed = mix[..., :min_len]

    loss, reordered = pit_loss(estimates, clean, return_est=True)
    si_sdr = (-loss).item()
    mix_baseline = 0.0
    for spk in range(clean.shape[1]):
        mix_baseline += si_sdr_metric(mix_trimmed, clean[:, spk]).item()
    mix_baseline /= clean.shape[1]
    si_sdri = si_sdr - mix_baseline
    return si_sdr, si_sdri, reordered


def _reference_enh(estimates, clean, mix, si_sdr_metric):
    """evaluate.evaluate_model ES/EB path == Trainer._compute_sisdri ES/EB path."""
    min_len = min(estimates.shape[-1], clean.shape[-1])
    estimates = estimates[..., :min_len]
    clean = clean[..., :min_len]
    mix_trimmed = mix[..., :min_len]

    if clean.dim() == 3 and clean.shape[1] == 1:
        clean = clean.squeeze(1)
    if estimates.dim() == 3 and estimates.shape[1] == 1:
        estimates = estimates.squeeze(1)

    si_sdr = si_sdr_metric(estimates, clean).item()
    si_sdr_mix = si_sdr_metric(mix_trimmed, clean).item()
    si_sdri = si_sdr - si_sdr_mix
    return si_sdr, si_sdri


# --- Tests --------------------------------------------------------------------

def test_sb_matches_reference():
    torch.manual_seed(0)
    B, T = 3, 400
    # Unequal lengths so trimming is exercised on both sides.
    estimates = torch.randn(B, 2, T + 20)
    clean = torch.randn(B, 2, T)
    mix = torch.randn(B, T + 10)

    si_sdr_metric, pit_loss = _metrics()
    ref_sisdr, ref_sisdri, ref_reordered = _reference_sb(
        estimates, clean, mix, si_sdr_metric, pit_loss
    )
    got_sisdr, got_sisdri, got_reordered = compute_sisdr_and_sisdri(
        estimates, clean, mix, "SB", si_sdr_metric, pit_loss=pit_loss
    )

    assert abs(got_sisdr - ref_sisdr) < 1e-5
    assert abs(got_sisdri - ref_sisdri) < 1e-5
    assert torch.allclose(got_reordered, ref_reordered, atol=1e-6)


def test_enh_matches_reference_2d():
    torch.manual_seed(1)
    B, T = 4, 512
    estimates = torch.randn(B, T + 8)
    clean = torch.randn(B, T)
    mix = torch.randn(B, T + 4)

    si_sdr_metric, _ = _metrics()
    ref_sisdr, ref_sisdri = _reference_enh(estimates, clean, mix, si_sdr_metric)
    got_sisdr, got_sisdri, _ = compute_sisdr_and_sisdri(
        estimates, clean, mix, "ES", si_sdr_metric
    )

    assert abs(got_sisdr - ref_sisdr) < 1e-5
    assert abs(got_sisdri - ref_sisdri) < 1e-5


def test_enh_matches_reference_3d_singleton_channel():
    """Enhancement inputs shaped [B, 1, T] must be squeezed like the original."""
    torch.manual_seed(2)
    B, T = 2, 300
    estimates = torch.randn(B, 1, T)
    clean = torch.randn(B, 1, T)
    mix = torch.randn(B, T)

    si_sdr_metric, _ = _metrics()
    ref_sisdr, ref_sisdri = _reference_enh(estimates, clean, mix, si_sdr_metric)
    got_sisdr, got_sisdri, _ = compute_sisdr_and_sisdri(
        estimates, clean, mix, "EB", si_sdr_metric
    )

    assert abs(got_sisdr - ref_sisdr) < 1e-5
    assert abs(got_sisdri - ref_sisdri) < 1e-5


def test_sb_requires_pit_loss():
    si_sdr_metric, _ = _metrics()
    estimates = torch.randn(1, 2, 100)
    clean = torch.randn(1, 2, 100)
    mix = torch.randn(1, 100)
    try:
        compute_sisdr_and_sisdri(estimates, clean, mix, "SB", si_sdr_metric)
    except ValueError:
        return
    raise AssertionError("expected ValueError when pit_loss is missing for SB")
