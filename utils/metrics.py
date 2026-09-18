"""Shared SI-SDR / SI-SDRi computation for one batch.

Single source of truth for the SI-SDRi mixture-baseline logic that was
previously copy-pasted between ``evaluate.evaluate_model`` and
``Trainer._compute_sisdri`` (survey gap 12). An edit to one used to silently
desync ``val_sisdr`` from the eval-table numbers; both now call this helper.

The metric objects are passed in by the caller so this stays framework-light
and reuses the caller's already-instantiated (and correctly-placed) metrics:
``evaluate.py`` and the trainer each keep one ``ScaleInvariantSignalDistortionRatio``
and, for the SB task, one ``PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")``.
"""


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
