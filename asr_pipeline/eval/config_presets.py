"""Eval-time pipeline config presets shared across the eval drivers.

The L3 / sweep drivers (`scripts/run_pipeline_on_recording.py`,
`scripts/sweep_pipeline.py`) all need the *same* set of eval-friendly
overrides on top of ``default.yaml``. Keeping that policy in one place stops
the two drivers from drifting — a silent mismatch here would make their
numbers incomparable.
"""

from __future__ import annotations

from pathlib import Path

from asr_pipeline.config import PipelineConfig, load_pipeline_config_from_yaml


def fresh_eval_cfg(yaml_path: Path) -> PipelineConfig:
    """Load ``default.yaml`` + force the eval-friendly overrides L3 depends on.

    - ``transcribe_mixture=True`` so L3 ORC-WER has the single-stream baseline.
    - ``output_mode='full_length'`` so per-speaker streams stay on the
      mixture timeline. Transcripts produced from these streams have
      timestamps that line up with the original recording — required for
      tcpWER (time-constrained WER) scoring and for hand-correcting GT
      against the source audio.
    - ``routing.min_overlap_dur=0`` so every pyannote-detected overlap
      goes through the separator (no quiet backchannels get dropped at
      routing time). Eval should be apples-to-apples across modes; we
      let the separator decide what to do with short overlaps.
    """
    cfg = load_pipeline_config_from_yaml(str(yaml_path))
    cfg.transcription.transcribe_mixture = True
    cfg.assembly.output_mode = "full_length"
    cfg.routing.min_overlap_dur = 0.0
    return cfg
