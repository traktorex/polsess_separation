"""Vendored dual-path MossFormer2 standalone model (B1 external-separator arm).

Source: https://github.com/alibabasglab/MossFormer2 (`MossFormer2_standalone/
model/`), MIT, by the MossFormer2 paper authors (Zhao, Yip 2024); vendored
2026-07-18 — the repo is not pip-installable. Files byte-identical upstream except ONE
documented patch (the `__init__`-time auto-`.to(cuda)` removed — the
separation stage owns device placement); package/import layout preserved
(`utils/` subpackage, relative imports).

This is the *dual-path chunked* MossFormer2 variant (SpeechBrain-style
`Dual_Path_Model` skeleton, chunk K=250, with FLASH+gated-FSMN intra blocks) —
a DIFFERENT architecture from the full-sequence MossFormer2_SS vendored at the
repo root (`models/mossformer2/`, ClearerVoice lineage). It exists here because
it is the only published code that loads the WHAMR-trained checkpoint
`alibabasglab/mossformer2-whamr-2spk` (8 kHz, ~17 dB WHAMR SI-SDRi per the
MossFormer2 paper) — verified key-for-key against the checkpoint state dict.

Load: ``Mossformer2Wrapper.from_pretrained("alibabasglab/mossformer2-whamr-2spk")``
(PyTorchModelHubMixin; config.json carries the architecture, pytorch_model.bin
the weights). Forward: [B, T] waveform → [B, T, num_spks].
"""

from .mossformer2 import Mossformer2Wrapper

__all__ = ["Mossformer2Wrapper"]
