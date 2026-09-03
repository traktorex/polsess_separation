"""Vendored TIGER separation model (B1 external-separator arm).

Source: https://github.com/JusperLee/TIGER (`look2hear/models/{tiger,base_model}.py`
+ `look2hear/layers/{activations,normalizations}.py`), MIT (upstream LICENSE,
copied here as `LICENSE`), vendored
2026-07-18 because `look2hear` is not on PyPI. Only the import path was patched
(`..layers` → `.layers`); model code is otherwise byte-identical upstream.

TIGER (Xu et al., ICLR 2025, arXiv:2410.01469): time-frequency interleaved
gain extraction, 822K params, trained on EchoSet (realistic noise+reverb).
Checkpoint: HF `JusperLee/TIGER-speech` (16 kHz, 2 sources), loaded through
`TIGER.from_pretrained` (PyTorchModelHubMixin — config.json carries the
architecture kwargs). Forward: [B, C, T] waveform → [B, num_sources, T].
"""

from .tiger import TIGER

__all__ = ["TIGER"]
