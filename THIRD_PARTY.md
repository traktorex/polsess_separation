# Third-party code, model weights and data

Attribution for everything in this repository that was not written for the thesis,
and for the external models and corpora the code depends on at runtime. Licence
texts for vendored code are kept next to the code, as listed below.

## Vendored / adapted source code

| Path | Upstream | Licence | Local changes |
|---|---|---|---|
| `models/mossformer2/` | [ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio) (`train/speech_separation/models/mossformer2/`), fetched 2026-06-02 | Apache-2.0 — `models/mossformer2/LICENSE` | relative imports; rotary seq-len cache disabled for `torch.compile`; edits listed in each file header. `__init__.py` is the project wrapper. |
| `models/mamba/` | [xi-j/Mamba-TasNet](https://github.com/xi-j/Mamba-TasNet) (`modules/mamba/`), itself adapted from [hustvl/Vim](https://github.com/hustvl/Vim); low-level kernels from [mamba-ssm 2.3.1](https://github.com/state-spaces/mamba) | GPL-3.0 (Mamba-TasNet) — `models/mamba/LICENSE-Mamba-TasNet`; Apache-2.0 (mamba-ssm, Vim) — `models/mamba/LICENSE-mamba-ssm` | ported to the mamba-ssm 2.3.1 / causal-conv1d 1.6.1 API; `MambaInnerFnNoOutProj` = upstream `MambaInnerFn` without the output projection |
| `models/mamba_tasnet.py`, `models/dpmamba.py` | architectures from Mamba-TasNet / DPMamba (Jiang et al. 2024), built on `models/mamba/` | own implementation following the papers; uses the GPL-3.0-derived blocks above | — |
| `models/spmamba.py` | re-implementation following [JusperLee/SPMamba](https://github.com/JusperLee/SPMamba); GridNet block after TF-GridNet (Wang et al. 2023) | Apache-2.0 (upstream) | own code; see file header |
| `models/{conv_tasnet,dprnn,sepformer}.py` | thin wrappers over [SpeechBrain](https://github.com/speechbrain/speechbrain) lobes (pip dependency, not vendored) | Apache-2.0 | `utils/common.py` patches SpeechBrain's EPS for float16 AMP |
| `asr_pipeline/vendor/ap_bwe/` | [yxlu-0102/AP-BWE](https://github.com/yxlu-0102/AP-BWE) | MIT — `asr_pipeline/vendor/ap_bwe/LICENSE` | inference-only subset; see its `README.md` |
| `asr_pipeline/vendor/mossformer2_dp/` | [alibabasglab/MossFormer2](https://github.com/alibabasglab/MossFormer2) (`MossFormer2_standalone/model/`) | MIT — `asr_pipeline/vendor/mossformer2_dp/LICENSE` | `__init__`-time `.to(cuda)` removed |
| `asr_pipeline/vendor/tf_locoformer/` | [merlresearch/tf-locoformer](https://github.com/merlresearch/tf-locoformer) (`standalone/`) | Apache-2.0 (MERL) — `asr_pipeline/vendor/tf_locoformer/LICENSE` | byte-identical |
| `asr_pipeline/vendor/tiger/` | [JusperLee/TIGER](https://github.com/JusperLee/TIGER) (`look2hear/models`, `look2hear/layers`) | MIT — `asr_pipeline/vendor/tiger/LICENSE` | import path `..layers` → `.layers` |
| `webapp_ondevice/site/vendor/ort/` (not tracked, see `webapp_ondevice/build/NOTES.md`) | onnxruntime-web 1.23.2 | MIT (Microsoft) | — |
| `webapp_ondevice/build/pyannote_seg3_*.json` | [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) config files | MIT | — |

## Model weights used at runtime (not in the repository)

| Model | Used by | Licence |
|---|---|---|
| NVIDIA Sortformer `diar_sortformer_4spk-v1` / streaming `v2.1` | `asr_pipeline` diarization (shipped best config) | CC BY-NC 4.0 (v1); NVIDIA Open Model License (v2.1) |
| Jenthe/ECAPA2 | `asr_pipeline` relabel / assembly | CC BY-NC 4.0 |
| speechbrain/spkrec-ecapa-voxceleb | `asr_pipeline` embeddings (alternative) | Apache-2.0 |
| pyannote speaker-diarization-3.1 / segmentation-3.0 / wespeaker | `asr_pipeline` diarization (`pyannote` backend), on-device OSD | MIT (gated download) |
| Whisper large-v2 via WhisperX; wav2vec2-large-xlsr-53-polish | `asr_pipeline` transcription | MIT (Whisper), BSD-4-Clause (WhisperX), Apache-2.0 (wav2vec2) |
| FRCRN_SE_16K, MossFormerGAN_SE_16K, MossFormer2_SS_16K (ClearerVoice) | `asr_pipeline` enhancement / B1 arm | Apache-2.0 |
| AP-BWE `g_8kto16k` | `asr_pipeline` bandwidth extension | MIT |
| TIGER-speech, SR-CorrNet, TF-Locoformer WHAMR, SpeechBrain sepformer-* checkpoints | B1 external-separator comparison (chapter 6) | TIGER-speech and SR-CorrNet declare no licence upstream; TF-Locoformer Apache-2.0; SpeechBrain Apache-2.0 |

## Corpora and audio excerpts in the repository

- PolSESS (training corpus) and the CLARIN-PL conversational recordings (evaluation) are
  **not** distributed with the repository.
- `webapp_ondevice/site/examples/rozmowa_przeplot.wav` — 45 s excerpt of a CLARIN-PL
  conversational recording (fragment `85cb678a`, see `site/examples/manifest.json`);
  `webapp_ondevice/reference/vectors/clarin_442dd69e_sparse.wav` — 20 s CLARIN-PL excerpt used
  as a parity test vector; `scripts/thesis_figures/data/pipeline_signals_026eafb1__seg00_0-22s.npz`
  — processed signal arrays from a CLARIN-PL fragment. Redistribution terms of the underlying
  CLARIN-PL resource apply.
- `webapp_ondevice/site/examples/pelne_nakladanie.wav` — 4 s PolSESS test mixture.
- `webapp_ondevice/reference/vectors/libricss_ov40_dense.wav` — 20 s LibriCSS excerpt
  (CC BY 4.0, derived from LibriSpeech).

## Licensing note for this repository

`models/mamba/` derives from GPL-3.0 code (Mamba-TasNet), so the Mamba-family models
(`models/mamba/`, `models/mamba_tasnet.py`, `models/dpmamba.py`) are distributed under the
GPL-3.0 terms of that upstream. The licence of the author's own code is stated in the
top-level `LICENSE` file when present.
