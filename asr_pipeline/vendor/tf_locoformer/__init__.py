"""Vendored TF-Locoformer standalone separator (B1 external-separator arm).

Source: https://github.com/merlresearch/tf-locoformer
(`standalone/tflocoformer_separator.py`), Apache-2.0 (SPDX header in file),
vendored 2026-07-18 — upstream's ESPnet-compatible variant needs a full ESPnet
install; the standalone file depends only on torch + packaging +
rotary-embedding-torch (all already main-venv deps). File is byte-identical
upstream.

TF-Locoformer (Saijo et al., IWAENC 2024, arXiv:2408.03440). The WHAMR-medium
checkpoint (15.0M, 18.5 dB SI-SDRi on WHAMR — the paper's headline model)
lives at `checkpoints/external/tf_locoformer/whamr_medium_valid.loss.ave_5best.pth`
(fetched from the repo's `egs2/whamr/enh1/exp/enh_train_enh_tflocoformer_raw/`;
state-dict keys carry a `separator.` prefix that the loader strips, per
upstream README).

The model maps complex STFT [B, T, F] → per-speaker complex STFT
[B, num_spk, T, F]; the STFT/iSTFT wrapper (n_fft 256, hop 64, hann — the
values in the checkpoint's exp config) lives in the separation-stage adapter,
not here.
"""

from .tflocoformer_separator import TFLocoformerSeparator

__all__ = ["TFLocoformerSeparator"]

# Constructor kwargs of the WHAMR-medium checkpoint, verbatim from upstream
# `egs2/whamr/enh1/exp/enh_train_enh_tflocoformer_raw/config.yaml` (+ the
# attention_dim/pos_enc values from upstream's load-pretrained test).
WHAMR_MEDIUM_KWARGS = dict(
    num_spk=2,
    n_layers=6,
    emb_dim=128,
    norm_type="rmsgroupnorm",
    num_groups=4,
    tf_order="ft",
    n_heads=4,
    attention_dim=128,
    pos_enc="rope",
    flash_attention=False,
    ffn_type=["swiglu_conv1d", "swiglu_conv1d"],
    ffn_hidden_dim=[192, 192],
    conv1d_kernel=8,
    conv1d_shift=1,
    dropout=0.0,
    eps=1.0e-5,
)
WHAMR_MEDIUM_STFT = dict(n_fft=256, hop_length=64)
