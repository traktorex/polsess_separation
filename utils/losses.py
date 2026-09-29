"""Silence-aware PIT SI-SDR loss for SB training with speaker-count augmentation.

With ``data.single_speaker_prob > 0`` some SB training items carry only ONE
speaker: one of the two target slots is exactly all-zero. The plain PIT loss
(``PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")``) cannot handle that
slot: SI-SDR against a zero reference is the constant -10*log10(EPS) = +80 dB
loss whatever the estimate is, so the output assigned to silence receives
exactly zero gradient and may carry anything (typically a copy of the active
speaker). The PIT choice is then decided by the active slot alone and the
logged loss is shifted by ~40 dB.

This module replaces the undefined term with an energy penalty on the output
assigned to the silent slot, following the thresholded ("soft-capped") loss
that Wisdom et al. (2020) use for zero references in mixture invariant training:

    Wisdom, Tzinis, Erdogan, Weiss, Wilson, Hershey. "Unsupervised Sound
    Separation Using Mixture Invariant Training". NeurIPS 2020.

Per sample, with estimates e_1, e_2 and targets t_1, t_2:

- 2 active targets: the usual PIT over negative SI-SDR, mean over the two
  sources -- numerically identical to the asteroid wrapper above.
- 1 active target t_j: the output i* with the higher SI-SDR against t_j is
  assigned to it (for one reference this is exactly PIT), the other output o
  to the silent slot, and

      loss_active = -SI-SDR(e_i*, t_j)
      loss_silent = 10*log10(E_o / E_i* + tau),   tau = 10**(-tau_db / 10)
      loss        = (loss_active + loss_silent) / 2

  The division by two keeps the per-sample scale of the 2-speaker case (mean
  over two slots). tau caps the reward: once the silent output is tau_db below
  the active one the penalty saturates at -tau_db and stops pulling.
- 0 active targets: the sample is left out of the batch mean (``valid`` is
  False). If no sample of the batch is valid, ``loss`` is None and the Trainer
  skips the batch. This happens on corpus defects, not by design: 101 of the
  128,000 PolSESS_C_128_16kHz train rows carry a speaker file below the
  silence threshold (89 of them BOTH speakers, 12 files exact digital zero;
  scan of 2026-09-28), so dropping the other speaker, or the row itself,
  leaves nothing to separate. The plain PIT loss hid these rows as a constant
  +80 dB with zero gradient.

Design choices (see also the comments in the code):

- The silent energy is measured relative to the OTHER OUTPUT, not to the
  mixture as in Wisdom et al. The SI-SDR term is scale-invariant and our models'
  outputs already come out 15-20 dB below the input, so a mixture-relative
  penalty could be met by shrinking both outputs without removing leakage.
  Relative to the active output, only a silent output that is quiet *compared
  with the speech the model did find* scores well.
- Energies are taken after removing the mean over time, the same zero-mean
  convention as SI-SDR (Le Roux et al. 2019, asteroid's pairwise_neg_sisdr).
  A DC offset is invisible to the SI-SDR term, so with raw energies the model
  could lower the ratio by adding DC to the active output.
- A target slot counts as active when its RMS exceeds SILENT_RMS_DBFS
  (-60 dBFS). The dataset writes exact zeros for a removed speaker, but the
  corpus also holds near-silent speaker files (from 1-LSB dither around
  -96 dBFS upwards) that are not usable speech; SI-SDR against them would
  train the model to reproduce quantisation noise. The level distribution has
  no clean gap, but below -60 dBFS the files come in pairs (both speakers of a
  row), i.e. a row-level rendering defect rather than quiet talkers, and every
  real speaker in val/ is above -56 dBFS. A partially
  active slot (speaker onset/offset, ``partial_overlap_prob``) is non-zero over
  the clip and is therefore scored with ordinary SI-SDR.
- Everything is computed in float32 with autocast disabled: TF-MossFormer,
  MossFormer2 and the Mamba family train under bf16 autocast, and a log of an
  energy ratio is exactly where bf16's 8-bit mantissa hurts.
"""

from typing import NamedTuple, Optional

import torch
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr

# A target slot whose RMS is at or below this level (dB re full scale 1.0) is
# "silent": removed speakers are exact zeros, defective corpus files sit between
# digital zero and roughly -60 dBFS, real val/ speakers above -56 dBFS (see
# module docstring).
SILENT_RMS_DBFS = -60.0

# Suppression (dB, relative to the active output) at which the silent-output
# penalty saturates.
DEFAULT_TAU_DB = 30.0

# Keeps the energy ratio finite if the active output is exactly zero.
_ENERGY_EPS = 1e-8


class SilenceAwarePITResult(NamedTuple):
    """Outputs of ``silence_aware_pit_sisdr`` (all float32 tensors).

    loss:          scalar, batch mean of the per-sample loss (differentiable).
    si_sdr:        [B, 2] SI-SDR (dB) of the assigned output per target slot;
                   NaN in silent slots (SI-SDR is undefined there).
    active:        [B, 2] bool, which target slots carry speech.
    reordered:     [B, 2, T] estimates permuted to align with the targets
                   (for a 1-speaker sample the silent slot gets output o).
    silent_rel_db: [B] 10*log10(E_o / E_i*) of 1-speaker samples WITHOUT the
                   tau floor (the quantity we want low); NaN for 2-speaker ones.
    valid:         [B] bool, samples with at least one active slot (the others
                   are left out of ``loss``).
    loss is None when no sample of the batch is valid.
    """

    loss: Optional[torch.Tensor]
    si_sdr: torch.Tensor
    active: torch.Tensor
    reordered: torch.Tensor
    silent_rel_db: torch.Tensor
    valid: torch.Tensor


def _zero_mean_energy(x: torch.Tensor) -> torch.Tensor:
    """Energy over the last (time) axis after removing the mean over time."""
    x = x - x.mean(dim=-1, keepdim=True)
    return (x**2).sum(dim=-1)


def silence_aware_pit_sisdr(
    estimates: torch.Tensor,
    targets: torch.Tensor,
    tau_db: float = DEFAULT_TAU_DB,
) -> SilenceAwarePITResult:
    """Silence-aware PIT negative SI-SDR for two-output separation.

    Args:
        estimates: model output, [B, 2, T].
        targets: references, [B, 2, T]; a removed speaker is an all-zero slot.
        tau_db: suppression (dB) at which the silent-output penalty saturates.

    Returns:
        ``SilenceAwarePITResult`` (see its docstring). The formula is in the
        module docstring.
    """
    if estimates.ndim != 3 or estimates.shape[1] != 2 or targets.shape != estimates.shape:
        raise ValueError(
            "silence_aware_pit_sisdr expects estimates and targets of shape [B, 2, T], "
            f"got {tuple(estimates.shape)} and {tuple(targets.shape)}"
        )
    tau = 10.0 ** (-tau_db / 10.0)

    with torch.autocast(device_type=estimates.device.type, enabled=False):
        est = estimates.float()
        tgt = targets.float()

        # pw[b, i, j] = -SI-SDR(est_i, tgt_j), asteroid convention.
        pw = pairwise_neg_sisdr(est, tgt)
        active = (tgt**2).mean(dim=-1) > 10.0 ** (SILENT_RMS_DBFS / 10.0)  # [B, 2]
        out_energy = _zero_mean_energy(est)  # [B, 2]

        per_sample_loss = []
        si_sdr = torch.full(active.shape, float("nan"), device=est.device)
        silent_rel_db = torch.full(active.shape[:1], float("nan"), device=est.device)
        # perm[b, j] = index of the output aligned with target slot j.
        perm = torch.zeros(active.shape, dtype=torch.long, device=est.device)

        # Per-sample branch: batch sizes here are 1-4, and the three cases read
        # more clearly as code than as masked tensor algebra.
        for b in range(est.shape[0]):
            n_active = int(active[b].sum())
            if n_active == 2:
                # Exactly asteroid's PITLossWrapper(pairwise_neg_sisdr, "pw_mtx"):
                # best permutation of the mean pairwise loss.
                min_loss, best_perm = PITLossWrapper.find_best_perm(pw[b : b + 1])
                per_sample_loss.append(min_loss[0])
                perm[b] = best_perm[0]
                si_sdr[b] = -pw[b, best_perm[0], torch.arange(2, device=est.device)].detach()
            elif n_active == 1:
                j = int(active[b].nonzero()[0])
                # Higher SI-SDR = lower loss. With a single reference this IS PIT.
                i_star = int(torch.argmin(pw[b, :, j]))
                o = 1 - i_star
                loss_active = pw[b, i_star, j]
                # E_i* is NOT detached: the gradient may also raise the active
                # output's energy. That is harmless -- SI-SDR is scale-invariant,
                # so the active output's scale is otherwise free -- and it keeps
                # the penalty a plain function of both outputs.
                ratio = out_energy[b, o] / (out_energy[b, i_star] + _ENERGY_EPS)
                loss_silent = 10.0 * torch.log10(ratio + tau)
                per_sample_loss.append((loss_active + loss_silent) / 2.0)
                perm[b, j] = i_star
                perm[b, 1 - j] = o
                si_sdr[b, j] = -loss_active.detach()
                silent_rel_db[b] = 10.0 * torch.log10(ratio.detach() + _ENERGY_EPS)
            # else: no active slot (corpus defect, see module docstring) --
            # nothing to separate, so the sample contributes no loss.

        valid = active.any(dim=1)
        loss = torch.stack(per_sample_loss).mean() if per_sample_loss else None
        reordered = PITLossWrapper.reorder_source(est, perm)

    return SilenceAwarePITResult(loss, si_sdr, active, reordered, silent_rel_db, valid)
