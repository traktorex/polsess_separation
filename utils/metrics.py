"""Shared SI-SDR / SI-SDRi computation for one batch.

Single source of truth for the SI-SDRi mixture-baseline logic that was
previously copy-pasted between ``evaluate.evaluate_model`` and
``Trainer._compute_sisdri`` (survey gap 12). An edit to one used to silently
desync ``val_sisdr`` from the eval-table numbers; both now call this helper.

The metric objects are passed in by the caller so this stays framework-light
and reuses the caller's already-instantiated (and correctly-placed) metrics:
``evaluate.py`` and the trainer each keep one ``ScaleInvariantSignalDistortionRatio``
and, for the SB task, one ``PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")``.

Convention: SI-SDR as defined by Le Roux et al. (2019), i.e. both signals are
made zero-mean before the projection. asteroid's ``pairwise_neg_sisdr`` does
this by default; torchmetrics' ``ScaleInvariantSignalDistortionRatio`` does
not (``zero_mean=False``). The helper refuses a metric constructed with
``zero_mean=False`` so the two terms cannot drift apart: before 2026-09-22 they
did, and every SI-SDRi in the thesis tables sits about 0.07 dB above the fully
zero-mean value (a constant offset, comparisons unaffected).
"""


def _require_zero_mean(si_sdr_metric):
    if not getattr(si_sdr_metric, "zero_mean", False):
        raise ValueError(
            "si_sdr_metric must be ScaleInvariantSignalDistortionRatio(zero_mean=True): "
            "the PIT loss zero-means its inputs, so the mixture baseline must too "
            "(see utils/metrics.py)."
        )


def compute_sisdr_and_sisdri(estimates, clean, mix, task, si_sdr_metric, pit_loss=None):
    """Compute batch-mean SI-SDR and SI-SDRi for one batch.

    Args:
        estimates: model output. [B, C, T] for SB, [B, T] or [B, 1, T] otherwise.
        clean: target(s). [B, C, T] for SB, [B, T] or [B, 1, T] otherwise.
        mix: input mixture, [B, T].
        task: "SB" (speaker separation, PIT) or "ES"/"EB" (enhancement).
        si_sdr_metric: a ``ScaleInvariantSignalDistortionRatio`` instance on the
            same device as the tensors.
        pit_loss: a ``PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")``
            instance; required for ``task == "SB"``, ignored otherwise.

    Returns:
        (si_sdr, si_sdri, reordered):
          - si_sdr (float): batch-mean SI-SDR of the estimate vs clean (dB).
          - si_sdri (float): si_sdr minus the mixture baseline (dB). The baseline
            is the per-speaker-averaged SI-SDR(mix, clean_spk) for SB, or
            SI-SDR(mix, clean) for enhancement.
          - reordered: for SB, the PIT-aligned estimates (aligned to ``clean``)
            so the caller can score PESQ/STOI on the matched permutation; for
            enhancement, the (possibly channel-squeezed) estimates unchanged.

    Inputs are trimmed to a common time length first (matches evaluate.py).
    """
    _require_zero_mean(si_sdr_metric)
    min_len = min(estimates.shape[-1], clean.shape[-1])
    estimates = estimates[..., :min_len]
    clean = clean[..., :min_len]
    mix = mix[..., :min_len]

    if task == "SB":
        if pit_loss is None:
            raise ValueError("pit_loss is required for task='SB'")
        loss, reordered = pit_loss(estimates, clean, return_est=True)
        si_sdr = -loss.item()

        mix_baseline = 0.0
        for spk in range(clean.shape[1]):
            mix_baseline += si_sdr_metric(mix, clean[:, spk]).item()
        mix_baseline /= clean.shape[1]

        si_sdri = si_sdr - mix_baseline
        return si_sdr, si_sdri, reordered

    # Enhancement (ES / EB): standard SI-SDR against the single target.
    if clean.dim() == 3 and clean.shape[1] == 1:
        clean = clean.squeeze(1)
    if estimates.dim() == 3 and estimates.shape[1] == 1:
        estimates = estimates.squeeze(1)

    si_sdr = si_sdr_metric(estimates, clean).item()
    si_sdr_mix = si_sdr_metric(mix, clean).item()
    si_sdri = si_sdr - si_sdr_mix
    return si_sdr, si_sdri, estimates


def compute_silence_aware_sisdr_and_sisdri(estimates, clean, mix, si_sdr_metric):
    """SB metrics for batches that may contain 1-speaker items (all-zero slot).

    Counterpart of ``compute_sisdr_and_sisdri(task="SB")`` for training with
    speaker-count augmentation. The permutation comes from
    ``utils.losses.silence_aware_pit_sisdr``, so the metric and the loss always
    agree on which output is the speech and which is the "silent" one. On a
    batch where every item has two speakers the returned SI-SDR / SI-SDRi equal
    those of ``compute_sisdr_and_sisdri`` (up to float summation order).

    Args:
        estimates: model output, [B, 2, T].
        clean: targets, [B, 2, T]; a removed speaker is an all-zero slot.
        mix: input mixture, [B, T].
        si_sdr_metric: ``ScaleInvariantSignalDistortionRatio(zero_mean=True)``.

    Returns:
        dict with
          - si_sdr (float): mean SI-SDR over ACTIVE target slots (dB). A silent
            slot has no defined SI-SDR and is left out.
          - si_sdri (float): si_sdr minus the mixture baseline, the mean of
            SI-SDR(mix, clean_slot) over the same active slots.
          - n_active_slots (int): number of slots the two means are over, for
            weighting across batches.
          - silent_rel_db (float or None): mean over 1-speaker items of
            10*log10(E_other / E_best), the zero-mean energy of the output
            assigned to silence relative to the one assigned to the speaker;
            None when the batch has no 1-speaker item.
          - silent_rel_db_values (list[float]): the per-item values behind it.
          - reordered: estimates aligned to ``clean`` (see the loss module).
    """
    # Local import: utils.losses pulls in asteroid, which the enhancement-only
    # callers of this module do not need.
    from .losses import silence_aware_pit_sisdr

    _require_zero_mean(si_sdr_metric)
    min_len = min(estimates.shape[-1], clean.shape[-1])
    estimates = estimates[..., :min_len]
    clean = clean[..., :min_len]
    mix = mix[..., :min_len]

    result = silence_aware_pit_sisdr(estimates, clean)
    active = result.active
    n_active_slots = int(active.sum())
    if n_active_slots == 0:
        # Every item is a corpus defect with no speech (see utils/losses.py).
        return {"si_sdr": None, "si_sdri": None, "n_active_slots": 0,
                "silent_rel_db": None, "silent_rel_db_values": [],
                "reordered": result.reordered}

    si_sdr = result.si_sdr[active].mean().item()

    # Mixture baseline per active slot, one item at a time so a silent slot
    # never enters the torchmetrics batch mean.
    baseline_sum = 0.0
    for b, j in active.nonzero().tolist():
        baseline_sum += si_sdr_metric(mix[b : b + 1], clean[b : b + 1, j]).item()
    mix_baseline = baseline_sum / n_active_slots

    one_speaker = active.sum(dim=1) == 1
    silent_values = result.silent_rel_db[one_speaker].tolist()
    silent_rel_db = sum(silent_values) / len(silent_values) if silent_values else None

    return {
        "si_sdr": si_sdr,
        "si_sdri": si_sdr - mix_baseline,
        "n_active_slots": n_active_slots,
        "silent_rel_db": silent_rel_db,
        "silent_rel_db_values": silent_values,
        "reordered": result.reordered,
    }
