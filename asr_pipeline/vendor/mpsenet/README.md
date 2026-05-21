# Vendored MP-SENet

This directory contains a minimal copy of the **MP-SENet** model code from:

> Yexin Lu, *MP-SENet: A Speech Enhancement Model with Parallel Denoising of Magnitude and Phase Spectra*, Interspeech 2023.
> Source: https://github.com/yxlu-0102/MP-SENet
> License: MIT (see `LICENSE` in this directory).

## Why vendored

The upstream `MP-SENet/` repository ships its model code under top-level
`models/` and `utils` namespaces, which clash with `polsess_separation`'s
own `models/` package. The POC notebook worked around this with a
`sys.modules` swap hack at load time. Vendoring eliminates the clash by
copying just the inference-relevant code into a local namespace
(`asr_pipeline.vendor.mpsenet`) with locally-resolving imports.

## What was kept

- `model.py` — the `MPNet` generator (DenseEncoder, MaskDecoder,
  PhaseDecoder, TSTransformerBlock, MPNet). Training-only helpers
  (`pesq_score`, `eval_pesq`) and the `pesq` / `joblib` dependencies they
  imported were removed; the loss functions `phase_losses` /
  `anti_wrapping_function` were also removed since this package only
  performs inference.
- `transformer.py` — `FFN`, `TransformerBlock`. The `__main__` smoke test
  in the upstream file was removed.
- `utils.py` — only `LearnableSigmoid2d` (and `LearnableSigmoid1d` for
  parity), which is the only utility imported by `model.py`. The rest of
  upstream `utils.py` (data loading, plotting, STFT helpers) is not
  vendored — the STFT helpers were ported inline into
  `asr_pipeline/stages/enhancement.py` since they're small and the
  enhancement stage already owns the surrounding I/O.

## Where the checkpoint and `config.json` live

These are **not** vendored — the user is expected to keep the
MP-SENet checkpoint (`g_best_vb` or `g_best_dns`) and the matching
`config.json` outside this repository (default path
`/home/user/MP-SENet/...`, configurable via
`EnhancementConfig.checkpoint_path` and `EnhancementConfig.config_path`).
The vendored code defines model architecture only.

## Modifications relative to upstream

- `from models.transformer import TransformerBlock` → `from .transformer import TransformerBlock`
- `from utils import LearnableSigmoid2d` → `from .utils import LearnableSigmoid2d`
- Removed: `from pesq import pesq`, `from joblib import Parallel, delayed`
- Removed: `phase_losses`, `anti_wrapping_function`, `pesq_score`, `eval_pesq` (training-only)
- Removed: `transformer.py`'s `__main__` smoke test
