"""Stage 5 — Whisper ASR per assembled per-speaker stream.

One Whisper call per speaker on the full per-speaker assembled stream.
``language`` is forced, ``initial_prompt`` discourages language drift on
noisy tails, and ``word_timestamps`` is enabled so per-segment / per-word
analysis remains tractable for evaluation (cpWER / tcpWER in later
phases).
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from asr_pipeline.config import TranscriptionConfig
from asr_pipeline.context import PipelineContext
from asr_pipeline.stages.base import Stage


def _format_transcript(whisper_result: dict) -> str:
    """Pretty per-segment transcript dump (POC cell 20 format)."""
    if not isinstance(whisper_result, dict):
        return ""
    segments = whisper_result.get("segments") or []
    if not segments:
        return (whisper_result.get("text") or "").strip()
    lines = []
    for seg in segments:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        text = (seg.get("text") or "").strip()
        lines.append(f"[{start:6.2f} → {end:6.2f}]  {text}")
    return "\n".join(lines)


def _jsonable(obj: Any) -> Any:
    """Recursively convert numpy/torch scalars to plain Python for json.dump."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


class TranscriptionStage(Stage):
    name = "transcription"

    def __init__(self, config: TranscriptionConfig) -> None:
        super().__init__(enabled=config.enabled)
        self.config = config
        self._model = None
        self._device: Optional[torch.device] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def load(self, device: torch.device) -> None:
        import whisper

        self._model = whisper.load_model(self.config.model_name, device=str(device))
        self._device = device

    def unload(self) -> None:
        self._model = None
        self._device = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def run(self, ctx: PipelineContext) -> None:
        if self._model is None:
            raise RuntimeError("TranscriptionStage.run called before load().")

        if not ctx.assembled:
            ctx.transcripts = {}
            return

        min_samples = ctx.sample_rate // 2  # 0.5 s — POC's lower bound
        results: dict[str, dict] = {}
        for spk, audio in ctx.assembled.items():
            audio32 = audio.astype(np.float32)
            if len(audio32) < min_samples:
                results[spk] = {"text": "", "segments": [], "language": self.config.language}
                continue
            result = self._model.transcribe(
                audio32,
                language=self.config.language,
                initial_prompt=self.config.initial_prompt,
                word_timestamps=self.config.word_timestamps,
                verbose=False,
            )
            results[spk] = result
        ctx.transcripts = results

    # ------------------------------------------------------------------
    # Spill
    # ------------------------------------------------------------------
    def spill(self, ctx: PipelineContext, artifact_dir: Path) -> None:
        if not ctx.transcripts:
            return
        for spk, result in ctx.transcripts.items():
            label = ctx.spk_to_label.get(spk, spk)
            txt_path = artifact_dir / f"transcript_{label}.txt"
            json_path = artifact_dir / f"transcript_{label}.json"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(_format_transcript(result))
                f.write("\n")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(_jsonable(result), f, indent=2, ensure_ascii=False)
