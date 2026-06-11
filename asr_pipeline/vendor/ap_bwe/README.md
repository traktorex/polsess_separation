# Vendored AP-BWE

This directory contains a minimal copy of the **AP-BWE** generator code from:

> Ye-Xin Lu, Yang Ai, Hui-Peng Du, Zhen-Hua Ling.
> *Towards High-Quality and Efficient Speech Bandwidth Extension with Parallel
> Amplitude and Phase Prediction.*
> IEEE/ACM TASLP, 2024.
>
> Source: https://github.com/yxlu-0102/AP-BWE
> License: MIT (see `LICENSE` in this directory).

## Why vendored

The upstream repository is clone-only (no pip package) and uses top-level
`models/`, `datasets/`, `utils.py` namespaces that would clash with the host
project's own modules. Vendoring just the inference-relevant code into a local
namespace (`asr_pipeline.vendor.ap_bwe`) keeps the integration self-contained
and matches the pattern already used for MP-SENet.

## What was kept

- `model.py` — the `APNet_BWE_Model` generator and its `ConvNeXtBlock`
  dependency. All discriminators, GAN/feature-matching losses, training
  utilities, and metric helpers (`cal_snr`, `cal_lsd`, etc.) were removed
  since this package performs inference only.
- `stft.py` — the amplitude/phase STFT helpers (`amp_pha_stft`,
  `amp_pha_istft`) used to encode/decode the model's spectral input/output.
  Lifted verbatim from upstream `datasets/dataset.py`.
- `env.py` — `AttrDict` (dict with attribute access) used to load the
  upstream JSON hyperparameter config.
- `config_8kto16k.json` — the upstream 8→16 kHz hyperparameter config.
  Required so the user does not need to download anything alongside the
  generator weights themselves.

## Getting the pretrained weights

The model weights are NOT vendored. Download them from the upstream Google
Drive folder and place them at the path configured via
`SuperResolutionConfig.checkpoint_path` (or `$AP_BWE_CHECKPOINT`):

> https://drive.google.com/drive/folders/1IIYTf2zbJWzelu4IftKD6ooHloJ8mnZF

For the 8→16 kHz model, grab the file `g_8kto16k` from the `8kto16k/`
subfolder. Default install path is `/home/user/AP-BWE/checkpoints/8kto16k/g_8kto16k`.
The weights are also released under MIT — see `weights_LICENSE.txt` upstream.

## Modifications relative to upstream

- `from utils import init_weights, get_padding` → `get_padding` inlined into
  `model.py`; `init_weights` was unused by the generator (the imported
  symbol is referenced only by the upstream discriminators, which are not
  vendored here).
- Removed all classes and functions outside `ConvNeXtBlock` and
  `APNet_BWE_Model` from `model.py` (discriminators, losses, helpers).
- Removed dataset class and dataset list helper from `dataset.py`; kept only
  `amp_pha_stft` / `amp_pha_istft` and renamed the file to `stft.py`.
- `env.py` retains only `AttrDict`; the `build_env` helper (used by the
  upstream training script) was removed.
