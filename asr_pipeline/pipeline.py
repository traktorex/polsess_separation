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
from asr_pipeline.debug_log import LOG_PATH, dlog, reset_log
from asr_pipeline.io import ensure_artifact_dir, load_audio_as_mono
from asr_pipeline.stages import (
    AssemblyStage,
    DiarizationStage,
    EnhancementStage,
    RoutingStage,
    SeparationStage,
    Stage,
    PostSeparationProcessingStage,
    TranscriptionStage,
)


def _log(msg: str) -> None:
    """Pipeline-orchestrator debug log — file only.

    Orchestrator events (stage transitions, load/unload, empty_cache) are
    valuable when diagnosing a hang via `tail -f /tmp/asr_pipeline_debug.log`,
    but they clutter the notebook cell output during normal use. Assembly's
    own `_log` keeps `to_stdout=True` so user-facing progress stays visible.
    """
    dlog("pipeline", msg, to_stdout=False)


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
            PostSeparationProcessingStage(config.post_separation_processing),
            AssemblyStage(config.assembly),
            TranscriptionStage(config.transcription),
        ]
        self.artifact_dir: Optional[Path] = (
            ensure_artifact_dir(config.artifact_dir)
            if config.spill_intermediate
            else None
        )
        self._current_stage_name: Optional[str] = None
        # Signature recorded the last time the current stage was loaded.
        # If the stage's `load_signature()` differs from this on the next
        # call, the stage is reloaded — this is what lets the user edit a
        # checkpoint-determining knob (e.g. `cfg.enhancement.backend`) and
        # re-run the same stage's cell without manually unloading first.
        self._loaded_signature: tuple = ()

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
        """Load audio into a fresh `PipelineContext`. No model touched.

        Resets the debug log file so a `tail -f` from another terminal sees
        only the current run's events (instead of accumulating forever).
        """
        reset_log()
        _log(f"load_audio: {audio_path} (debug log at {LOG_PATH})")
        ctx = PipelineContext(
            input_path=Path(audio_path),
            sample_rate=self.config.sample_rate,
        )
        ctx.audio = load_audio_as_mono(
            audio_path, target_sr=self.config.sample_rate
        )
        _log(f"load_audio: loaded {len(ctx.audio)/ctx.sample_rate:.2f}s audio")
        return ctx

    def run_stage(self, stage_name: str, ctx: PipelineContext) -> None:
        """Run one stage by name on `ctx`. Loads the stage's model only if it
        isn't the currently-loaded one; unloads the previously-loaded stage
        first when switching."""
        _log(f"run_stage({stage_name!r}) called")
        stage = self.get_stage(stage_name)
        if not stage.enabled:
            raise RuntimeError(
                f"Stage {stage_name!r} is disabled (config.{stage_name}.enabled = False)."
            )
        self._ensure_loaded(stage_name)
        try:
            _log(f"run_stage({stage_name!r}): calling stage.run()")
            stage.run(ctx)
            _log(f"run_stage({stage_name!r}): stage.run() returned")
            if self.artifact_dir is not None:
                stage.spill(ctx, self.artifact_dir)
        except BaseException:
            # A stage failure (OOM, bad checkpoint, KeyboardInterrupt) must not
            # leave its heavyweight model resident — that breaks the
            # one-model-at-a-time GPU budget for whatever runs next. Free it,
            # then re-raise (fail loud; don't march the loop on a corrupt ctx).
            _log(f"run_stage({stage_name!r}): FAILED — releasing model")
            self._release_current()
            raise
        _log(f"run_stage({stage_name!r}): complete")

    def unload(self) -> None:
        """Free the currently-loaded stage's model, if any."""
        self._release_current()

    def get_stage(self, name: str) -> Stage:
        for s in self.stages:
            if s.name == name:
                return s
        valid = [s.name for s in self.stages]
        raise ValueError(f"Unknown stage {name!r}; valid: {valid}")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _release_current(self) -> None:
        """Unload the currently-loaded stage's model and reset bookkeeping.

        No-op when nothing is loaded. The single home for the
        unload -> empty_cache -> gc.collect teardown (previously duplicated
        across ``unload()`` and both ``_ensure_loaded`` branches).
        """
        if self._current_stage_name is None:
            return
        _log(f"_release_current: unloading {self._current_stage_name!r}...")
        self.get_stage(self._current_stage_name).unload()
        self._current_stage_name = None
        self._loaded_signature = ()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        _log("_release_current: released")

    def _ensure_loaded(self, stage_name: str) -> None:
        """Make `stage_name` the currently-loaded stage.

        - Same stage, same load signature → no-op (this is what makes
          within-stage iteration fast — the user can re-run the same
          stage with unchanged model-defining config without paying the
          load cost again).
        - Same stage, different signature → unload + reload (the user
          changed a checkpoint-determining knob between runs).
        - Different stage → unload current + load new.
        """
        stage = self.get_stage(stage_name)
        new_sig = stage.load_signature()

        if self._current_stage_name == stage_name:
            if new_sig == self._loaded_signature:
                _log(f"_ensure_loaded({stage_name!r}): already loaded, no-op")
                return
            _log(
                f"_ensure_loaded({stage_name!r}): signature changed "
                f"{self._loaded_signature!r} -> {new_sig!r}, reloading"
            )
            self._release_current()
        elif self._current_stage_name is not None:
            _log(
                f"_ensure_loaded({stage_name!r}): switching from "
                f"{self._current_stage_name!r}"
            )
            self._release_current()

        _log(f"_ensure_loaded({stage_name!r}): calling stage.load()...")
        stage.load(self.device)
        self._current_stage_name = stage_name
        self._loaded_signature = new_sig
        _log(f"_ensure_loaded({stage_name!r}): stage.load() returned (sig={new_sig!r})")
