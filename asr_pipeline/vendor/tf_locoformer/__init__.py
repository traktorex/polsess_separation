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
(state-dict keys carry a `separator.` prefix that the loader strips, per upstream
README).

WEIGHT PROVENANCE (pinned 2026-08-25, when the checkpoint first produced a number
that could reach the thesis). Upstream ships weights through git-lfs: the `.pth`
paths in the GitHub tree are 133-byte pointer files, and the real payload comes
from the LFS endpoint. Our local copy is bit-identical to the released artifact —
its sha256 equals the upstream LFS `oid` exactly:

    path   egs2/whamr/enh1/exp/enh_train_enh_tflocoformer_raw/valid.loss.ave_5best.pth
    sha256 3e14c736d19cd2158e09f0446fe214be4b47e28ed6d6c11d037cf2d4dae82d6a
    size   59942624 bytes        local file dated 2026-07-18

To re-fetch: `git lfs clone https://github.com/merlresearch/tf-locoformer` (or
`git clone` then `git lfs pull`) and copy that path, renaming with the
`whamr_medium_` prefix this repo uses. Verify with `sha256sum` against the oid
above; a 133-byte file means git-lfs did not run.

THREE SIBLING CHECKPOINTS exist upstream, same architecture family, same
`valid.loss.ave_5best.pth` filename under `egs2/<dataset>/enh1/exp/
enh_train_enh_tflocoformer_raw/`, all Apache-2.0 (author-flagged 2026-08-25 as
worth trying after the WHAMR arm lands):

    wsj0_2mix  c536fa8499b28e2cf812b1906d899ae6337665232ca523dbe628682dd7df983c  59979488
    librimix   9c691cee4bb0d9664a3fac024b031dfdf8ca3e66fc22a66f2f17119767b49129  59979488
    dns_ins20  6ffa518f2b73bed289e38349366eb777c94b9130acc32bf4c085cb049d9d1666  59972116

Two cautions before adding one as an arm. (1) Their sizes differ from WHAMR's, so
the constructor kwargs and STFT settings below are NOT automatically valid —
read each checkpoint's own `config.yaml` from the same upstream directory rather
than reusing `WHAMR_MEDIUM_KWARGS`. (2) They differ in training domain, which is
the whole point of trying them but also changes what a result means: WHAMR is
noisy + reverberant with dry targets (the convention PolSESS SB shares),
wsj0_2mix is anechoic and clean, librimix is read speech, and dns_ins20 is an
enhancement (denoising) task rather than 2-speaker separation — check the
separator actually emits two sources before wiring it as a separation arm.

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

# The librimix and wsj0_2mix siblings share one separator_conf, which differs from
# WHAMR-medium in two places (verbatim from their own upstream config.yaml, fetched
# 2026-08-25): ffn_hidden_dim [384, 384] not [192, 192], and conv1d_kernel 4 not 8;
# their STFT is n_fft 128, not 256. That is the whole 59 979 488 vs 59 942 624 byte
# difference. All three are 8 kHz (`sample_rate: 8000` in every config), so the
# pipeline's separator geometry and the ap_bwe band-extension stay correct for all.
_SIBLING_KWARGS = dict(WHAMR_MEDIUM_KWARGS, ffn_hidden_dim=[384, 384], conv1d_kernel=4)
_SIBLING_STFT = dict(n_fft=128, hop_length=64)

LIBRIMIX_MEDIUM_KWARGS = _SIBLING_KWARGS
LIBRIMIX_MEDIUM_STFT = _SIBLING_STFT
WSJ0_2MIX_MEDIUM_KWARGS = _SIBLING_KWARGS
WSJ0_2MIX_MEDIUM_STFT = _SIBLING_STFT

# Checkpoint-file basename → (constructor kwargs, STFT kwargs). Keyed by the names
# this repo stores under checkpoints/external/tf_locoformer/ (upstream ships all
# three as `valid.loss.ave_5best.pth`, so the corpus prefix is ours and load-bearing).
VARIANTS = {
    "whamr_medium_valid.loss.ave_5best.pth": (WHAMR_MEDIUM_KWARGS, WHAMR_MEDIUM_STFT),
    "librimix_medium_valid.loss.ave_5best.pth": (LIBRIMIX_MEDIUM_KWARGS, LIBRIMIX_MEDIUM_STFT),
    "wsj0_2mix_medium_valid.loss.ave_5best.pth": (WSJ0_2MIX_MEDIUM_KWARGS, WSJ0_2MIX_MEDIUM_STFT),
}


def variant_for(ckpt_path) -> tuple[dict, dict]:
    """Resolve (kwargs, stft) for a TF-Locoformer checkpoint by file basename.

    Raises on an unrecognised name rather than defaulting to WHAMR — the sibling
    architectures genuinely differ, and a silent wrong default is precisely the
    no-silent-substitution failure SCOPE §4 forbids. (A wrong choice would in fact
    be caught by `load_state_dict(strict=True)` here, but relying on a downstream
    crash to enforce a config rule is not the contract.)
    """
    import os

    name = os.path.basename(str(ckpt_path))
    if name not in VARIANTS:
        raise ValueError(
            f"Unknown TF-Locoformer checkpoint {name!r}. Known: {sorted(VARIANTS)}. "
            "Add its constructor kwargs here, read from that checkpoint's own "
            "egs2/<corpus>/enh1/exp/enh_train_enh_tflocoformer_raw/config.yaml "
            "upstream — the variants differ in ffn_hidden_dim, conv1d_kernel and n_fft."
        )
    return VARIANTS[name]
