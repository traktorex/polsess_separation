"""Audio I/O and artefact directory handling for the pipeline.

Phase 1 keeps this minimal: audio loading (ported from POC cell 10) and a
single artefact-directory helper. Per-stage spill formats are added by
each stage starting in Phase 2.
"""

import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio
import torchaudio.functional as AF

from asr_pipeline.debug_log import dlog


def _log(msg: str) -> None:
    """I/O progress log. Stays on stdout so long-file loads show progress."""
    dlog("io", msg)


def load_audio_as_mono_16k(audio_path: str, target_sr: int = 16_000) -> np.ndarray:
    """Load any audio file, downmix to mono, resample to `target_sr`.

    Returns a 1-D float32 numpy array. Ported from POC cell 10.

    Progress is logged step-by-step so that a hang on long recordings
    (decode vs downmix vs resample) is immediately distinguishable. Reading
    from `/mnt/c/...` paths inside WSL is much slower than from the WSL
    filesystem — if the decode step takes minutes, copying the file into
    `/tmp` first is worthwhile.
    """
    path = Path(audio_path)
    try:
        size_mb = path.stat().st_size / (1024 * 1024)
        _log(f"loading {audio_path} ({size_mb:.1f} MB)")
    except OSError as e:
        _log(f"loading {audio_path} (stat failed: {e})")

    t0 = time.perf_counter()
    waveform, sr = torchaudio.load(audio_path)
    _log(
        f"  torchaudio.load: {waveform.shape[0]} ch @ {sr} Hz, "
        f"{waveform.shape[1]} samples ({waveform.shape[1]/sr:.2f}s) "
        f"in {time.perf_counter()-t0:.2f}s"
    )

    if waveform.shape[0] > 1:
        t0 = time.perf_counter()
        waveform = waveform.mean(dim=0, keepdim=True)
        _log(f"  downmixed to mono in {time.perf_counter()-t0:.2f}s")

    if sr != target_sr:
        t0 = time.perf_counter()
        waveform = AF.resample(waveform, sr, target_sr)
        _log(
            f"  resampled {sr} -> {target_sr} Hz "
            f"({waveform.shape[1]} samples) in {time.perf_counter()-t0:.2f}s"
        )

    out = waveform.squeeze(0).numpy().astype(np.float32)
    _log(f"  done: {len(out)/target_sr:.2f}s mono @ {target_sr} Hz")
    return out


def ensure_artifact_dir(artifact_dir: Optional[str]) -> Optional[Path]:
    """Create the artefact directory if a path is provided. Return as Path."""
    if artifact_dir is None:
        return None
    path = Path(artifact_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path
