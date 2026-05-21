"""Pipeline orchestrator.

Phase-major execution: each stage is loaded onto the device, runs, then is
unloaded before the next stage begins. Optional disk-spill happens after a
stage's `run` returns.

Two usage modes:

- One-shot: ``Pipeline(cfg).run(audio_path)`` — runs every enabled stage in
  order on a fresh context and returns the populated `PipelineContext`.
- Interactive: ``ctx = pipeline.load_audio(path); pipeline.run_stage("diarization", ctx); ...``
  — run stages one at a time, with the current stage's model kept loaded
  between successive calls. Switching stages (or calling ``pipeline.unload()``)
  frees the previous stage's model. This preserves the one-model-at-a-time
  invariant required by the 12 GB GPU budget.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import List, Optional

import torch

from asr_pipeline.config import PipelineConfig
from asr_pipeline.context import PipelineContext
from asr_pipeline.io import ensure_artifact_dir, load_audio_as_mono_16k
from asr_pipeline.stages import (
    AssemblyStage,
    DiarizationStage,
    EnhancementStage,
    RoutingStage,
    SeparationStage,
    Stage,
    TranscriptionStage,
)


class Pipeline:
    """Top-level orchestrator. Construct once per recording (or reuse)."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.stages: List[Stage] = [
            DiarizationStage(config.diarization),
            RoutingStage(config.routing),
            EnhancementStage(config.enhancement),
            SeparationStage(config.separation),
            AssemblyStage(config.assembly),
            TranscriptionStage(config.transcription),
        ]
        self.artifact_dir: Optional[Path] = (
            ensure_artifact_dir(config.artifact_dir)
            if config.spill_intermediate
            else None
        )
        self._current_stage_name: Optional[str] = None

    def __repr__(self) -> str:
        active = [s.name for s in self.stages if s.enabled]
        loaded = self._current_stage_name or "none"
        return (
            f"Pipeline(device={self.device}, "
            f"sample_rate={self.config.sample_rate}, "
            f"stages={active}, loaded={loaded})"
        )

    # ------------------------------------------------------------------
    # One-shot API
    # ------------------------------------------------------------------
    def run(self, audio_path: str) -> PipelineContext:
        """Run every enabled stage on a fresh context. Returns the context."""
        ctx = self.load_audio(audio_path)
        for stage in self.stages:
            if not stage.enabled:
                continue
            self.run_stage(stage.name, ctx)
        # Tidy up: free the final stage's model on exit.
        self.unload()
        return ctx

    # ------------------------------------------------------------------
    # Interactive API
    # ------------------------------------------------------------------
    def load_audio(self, audio_path: str) -> PipelineContext:
        """Load audio into a fresh `PipelineContext`. No model touched."""
        ctx = PipelineContext(
            input_path=Path(audio_path),
            sample_rate=self.config.sample_rate,
        )
        ctx.audio = load_audio_as_mono_16k(
            audio_path, target_sr=self.config.sample_rate
        )
        return ctx

    def run_stage(self, stage_name: str, ctx: PipelineContext) -> None:
        """Run one stage by name on `ctx`. Loads the stage's model only if it
        isn't the currently-loaded one; unloads the previously-loaded stage
        first when switching."""
        stage = self.get_stage(stage_name)
        if not stage.enabled:
            raise RuntimeError(
                f"Stage {stage_name!r} is disabled (config.{stage_name}.enabled = False)."
            )
        self._ensure_loaded(stage_name)
        stage.run(ctx)
        if self.artifact_dir is not None:
            stage.spill(ctx, self.artifact_dir)

    def unload(self) -> None:
        """Free the currently-loaded stage's model, if any."""
        if self._current_stage_name is None:
            return
        self.get_stage(self._current_stage_name).unload()
        self._current_stage_name = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    def get_stage(self, name: str) -> Stage:
        for s in self.stages:
            if s.name == name:
                return s
        valid = [s.name for s in self.stages]
        raise ValueError(f"Unknown stage {name!r}; valid: {valid}")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _ensure_loaded(self, stage_name: str) -> None:
        """Make `stage_name` the currently-loaded stage.

        - If it's already loaded: no-op (this is what makes within-stage
          iteration fast — the user can re-run the same stage without
          paying the load cost again).
        - Otherwise: unload whatever is currently loaded, then load this
          stage's model.
        """
        if self._current_stage_name == stage_name:
            return
        if self._current_stage_name is not None:
            self.get_stage(self._current_stage_name).unload()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
        self.get_stage(stage_name).load(self.device)
        self._current_stage_name = stage_name
