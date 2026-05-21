"""Audio I/O and artefact directory handling for the pipeline.

Phase 1 keeps this minimal: audio loading (ported from POC cell 10) and a
single artefact-directory helper. Per-stage spill formats are added by
each stage starting in Phase 2.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio
import torchaudio.functional as AF


def load_audio_as_mono_16k(audio_path: str, target_sr: int = 16_000) -> np.ndarray:
    """Load any audio file, downmix to mono, resample to `target_sr`.

    Returns a 1-D float32 numpy array. Ported from POC cell 10.
    """
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = AF.resample(waveform, sr, target_sr)
    return waveform.squeeze(0).numpy().astype(np.float32)


def ensure_artifact_dir(artifact_dir: Optional[str]) -> Optional[Path]:
    """Create the artefact directory if a path is provided. Return as Path."""
    if artifact_dir is None:
        return None
    path = Path(artifact_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path
