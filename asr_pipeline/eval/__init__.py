"""Two-layer evaluation for the ASR pipeline.

L1/DER is retired (SCOPE §10 q8): no valid reference diarization exists for
any dataset, so diarization error rate is not computed anywhere.

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

- the cpWER / ORC / MIMO WER & CER family (``cpwer_meeteval``,
  ``orc_wer_meeteval``, ``mimo_wer_meeteval``, ``orc_wer_multistream``,
  ``cp_cer_meeteval``, ``mimo_cer_meeteval``) from `metrics.py`.
- ``parse_gt_txt``, ``parse_transcript_file`` from `transcript_parser.py`.
"""

from asr_pipeline.eval.config_presets import fresh_eval_cfg
from asr_pipeline.eval.layer2 import (
    compute_intrusive,
    compute_layer2,
    load_squim_model,
    pesq_wb_chunked,
    squim_chunked,
    stoi_chunked,
    unload_squim_model,
)
from asr_pipeline.eval.edacc import ExcisionReport, excise_stella_passage
from asr_pipeline.eval.layer3 import compute_layer3
from asr_pipeline.eval.metrics import (
    cp_cer_meeteval,
    cpwer_meeteval,
    mimo_cer_meeteval,
    mimo_wer_meeteval,
    orc_wer_meeteval,
    orc_wer_multistream,
)
from asr_pipeline.eval.recordings import (
    Recording,
    load_recording,
    walk_eval_tree,
)
from asr_pipeline.eval.run import ScoreCard, evaluate_many, evaluate_recording
from asr_pipeline.eval.summary import (
    inventory,
    summarize_layer2_intrusive,
    summarize_layer2_squim,
    summarize_layer3,
)
from asr_pipeline.eval.transcript_parser import (
    Utterance,
    concat_utterances,
    format_untimed_gt,
    is_untimed,
    parse_gt_txt,
    parse_transcript_file,
)

__all__ = [
    # Config presets
    "fresh_eval_cfg",
    # Discovery
    "Recording", "ScoreCard", "load_recording", "walk_eval_tree",
    # Orchestration
    "evaluate_recording", "evaluate_many",
    # Layers
    "compute_layer2", "compute_layer3",
    # Low-level metrics
    "cpwer_meeteval", "orc_wer_meeteval", "mimo_wer_meeteval",
    "orc_wer_multistream", "cp_cer_meeteval", "mimo_cer_meeteval",
    "compute_intrusive", "pesq_wb_chunked", "stoi_chunked", "squim_chunked",
    "load_squim_model", "unload_squim_model",
    # Transcript IO
    "Utterance", "concat_utterances", "parse_gt_txt", "parse_transcript_file",
    "format_untimed_gt", "is_untimed",
    # EdAcc hypothesis filtering (dataset-specific, opt-in hyp_filter)
    "excise_stella_passage", "ExcisionReport",
    # Summaries
    "inventory",
    "summarize_layer2_intrusive",
    "summarize_layer2_squim", "summarize_layer3",
]
