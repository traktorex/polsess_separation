# Third-party code, model weights and data

Attribution for everything in this repository that was not written for the thesis,
and for the external models and corpora the code depends on at runtime. Licence
texts for vendored code are kept next to the code, as listed below.

## Vendored / adapted source code

| Path | Upstream | Licence | Local changes |
|---|---|---|---|
| `models/mossformer2/` | [ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio) (`train/speech_separation/models/mossformer2/`), fetched 2026-06-02 | Apache-2.0 — `models/mossformer2/LICENSE` | relative imports; rotary seq-len cache disabled for `torch.compile`; edits listed in each file header. `__init__.py` is the project wrapper. |
| `models/tf_mossformer/` | [merlresearch/tf-locoformer](https://github.com/merlresearch/tf-locoformer) (`standalone/tflocoformer_separator.py`), through the vendored copy fetched 2026-07-18 (then in this repository, now in DGSS-ASR) | Apache-2.0 (MERL) — `models/tf_mossformer/LICENSE` | `tf_locoformer_blocks.py` is the upstream file with four enumerated edits (header): pluggable attention module; `torch.backends.cuda.sdp_kernel` → `torch.nn.attention.sdpa_kernel`; `torch.cuda.amp.autocast` → `torch.amp.autocast("cuda")` on `RMSGroupNorm`; the fp32-forced encoder/decoder cast their activations to the weight dtype so the model runs under bf16 autocast; `local_global_attention.py` and `__init__.py` are original code implementing TF-MossFormer (arXiv:2607.21128), for which no upstream code exists. The byte-identical vendored copy it was taken from now lives in DGSS-ASR (`asr_pipeline/vendor/tf_locoformer/`), not here. |
| `models/mamba/` | [xi-j/Mamba-TasNet](https://github.com/xi-j/Mamba-TasNet) (`modules/mamba/`), itself adapted from [hustvl/Vim](https://github.com/hustvl/Vim); low-level kernels from [mamba-ssm 2.3.1](https://github.com/state-spaces/mamba) | GPL-3.0 (Mamba-TasNet) — `models/mamba/LICENSE-Mamba-TasNet`; Apache-2.0 (mamba-ssm, Vim) — `models/mamba/LICENSE-mamba-ssm` | ported to the mamba-ssm 2.3.1 / causal-conv1d 1.6.1 API; `MambaInnerFnNoOutProj` = upstream `MambaInnerFn` without the output projection |
| `models/mamba_tasnet.py`, `models/dpmamba.py` | architectures from Mamba-TasNet / DPMamba (Jiang et al. 2024), built on `models/mamba/` | own implementation following the papers; uses the GPL-3.0-derived blocks above | — |
| `models/spmamba.py` | re-implementation following [JusperLee/SPMamba](https://github.com/JusperLee/SPMamba); GridNet block after TF-GridNet (Wang et al. 2023) | Apache-2.0 (upstream) | own code; see file header |
| `models/{conv_tasnet,dprnn,sepformer}.py` | thin wrappers over [SpeechBrain](https://github.com/speechbrain/speechbrain) lobes (pip dependency, not vendored) | Apache-2.0 | `utils/common.py` patches SpeechBrain's EPS for float16 AMP |

## Corpora and audio excerpts in the repository

- PolSESS (training corpus) is **not** distributed with the repository.
- `scripts/thesis_figures/data/pipeline_signals_026eafb1__seg00_0-22s.npz` — processed
  signal arrays from a CLARIN-PL fragment. Redistribution terms of the underlying
  CLARIN-PL resource apply.

## Licensing note for this repository

`models/mamba/` derives from GPL-3.0 code (Mamba-TasNet), so the Mamba-family models
(`models/mamba/`, `models/mamba_tasnet.py`, `models/dpmamba.py`) are distributed under the
GPL-3.0 terms of that upstream. The licence of the author's own code is stated in the
top-level `LICENSE` file when present.
