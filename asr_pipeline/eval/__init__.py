"""Three-layer evaluation for the ASR pipeline.

- **Layer 1 — diarization** (`layer1.py`): DER between pipeline stage-1
  diarization (``pipeline/diarization.json``) and the reference RTTM.
- **Layer 2 — audio quality** (`layer2.py`): intrusive SI-SDR / PESQ-WB /
  STOI (chunked, median-aggregated, speech-presence filtered) when oracle
  audio is available; non-intrusive SQUIM (chunked, mean-aggregated)
  always.
- **Layer 3 — ASR** (`layer3.py`): cpWER + tcpWER per ablation mode
  (full / no-sep / no-enh) and ORC-WER on the mixture baseline transcript.

Orchestrator: :func:`evaluate_recording` (one call → ScoreCard) and
:func:`evaluate_many` (a sweep, with SQUIM loaded once).

Discovery: :func:`walk_eval_tree` yields one ``Recording`` per directory
under the eval root.

Low-level helpers (kept exported for direct use in notebooks):

- ``compute_der``, ``cpwer_meeteval``, ``orc_wer_meeteval`` from `metrics.py`.
- ``parse_gt_txt``, ``parse_transcript_file`` from `transcript_parser.py`.
- ``parse_rttm`` from `recordings.py`.
"""

from asr_pipeline.eval.layer1 import compute_layer1
from asr_pipeline.eval.layer2 import (
    compute_intrusive,
    compute_layer2,
    load_squim_model,
    pesq_wb_chunked,
    squim_chunked,
    stoi_chunked,
    unload_squim_model,
)
from asr_pipeline.eval.layer3 import compute_layer3
from asr_pipeline.eval.metrics import (
    compute_der,
    cpwer_meeteval,
    orc_wer_meeteval,
)
from asr_pipeline.eval.recordings import (
    Recording,
    load_recording,
    parse_rttm,
    walk_eval_tree,
)
from asr_pipeline.eval.run import ScoreCard, evaluate_many, evaluate_recording
from asr_pipeline.eval.summary import (
    inventory,
    summarize_layer1,
    summarize_layer2_intrusive,
    summarize_layer2_squim,
    summarize_layer3,
)
from asr_pipeline.eval.transcript_parser import (
    Utterance,
    concat_utterances,
    parse_gt_txt,
    parse_transcript_file,
)

__all__ = [
    # Discovery
    "Recording", "ScoreCard", "load_recording", "walk_eval_tree", "parse_rttm",
    # Orchestration
    "evaluate_recording", "evaluate_many",
    # Layers
    "compute_layer1", "compute_layer2", "compute_layer3",
    # Low-level metrics
    "compute_der", "cpwer_meeteval", "orc_wer_meeteval",
    "compute_intrusive", "pesq_wb_chunked", "stoi_chunked", "squim_chunked",
    "load_squim_model", "unload_squim_model",
    # Transcript IO
    "Utterance", "concat_utterances", "parse_gt_txt", "parse_transcript_file",
    # Summaries
    "inventory",
    "summarize_layer1", "summarize_layer2_intrusive",
    "summarize_layer2_squim", "summarize_layer3",
]
