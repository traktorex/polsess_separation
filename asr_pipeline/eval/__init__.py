"""Three-layer evaluation utilities for the ASR pipeline.

- **Layer 1 — diarization**: `compute_der` (pyannote.metrics).
- **Layer 2 — separation**: not exported here. Use
  `torchmetrics.functional.audio` (SI-SDR / PESQ-WB / STOI) and
  `torchaudio.pipelines.SQUIM_OBJECTIVE` (non-intrusive estimates)
  directly in the notebook — matches what `evaluate.py` and
  `asr/evaluate_asr.py` already do.
- **Layer 3 — ASR**: `cpwer_meeteval` (cpWER + tcpWER via MeetEval, the
  CHiME-7/8 evaluation toolkit, with Polish-aware text normalization).

Plus transcript parsers for the two file formats this project uses:
`parse_transcript_file` (pipeline output) and `parse_gt_txt`
(human-corrected GT from `scripts/transcribe_clarin_debleed.py`).
"""

from asr_pipeline.eval.metrics import compute_der, cpwer_meeteval
from asr_pipeline.eval.transcript_parser import (
    Utterance,
    concat_utterances,
    parse_gt_txt,
    parse_transcript_file,
)

__all__ = [
    "Utterance",
    "compute_der",
    "concat_utterances",
    "cpwer_meeteval",
    "parse_gt_txt",
    "parse_transcript_file",
]
