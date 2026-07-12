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
import time
from pathlib import Path
from typing import Callable, List, Optional

import torch

from asr_pipeline.config import PipelineConfig
from asr_pipeline.context import PipelineContext
from asr_pipeline.debug_log import LOG_PATH, dlog, reset_log
from asr_pipeline.io import ensure_artifact_dir, load_audio_as_mono
from asr_pipeline.stages import (
    AssemblyStage,
    DiarizationStage,
    EnhancementStage,
    RelabelStage,
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
    """Top-level orchestrator. Construct once per recording (or reuse).

    ``on_event`` is an optional callback that receives a small dict per stage
    boundary so a CLI can print progress and the outputs can record trustworthy
    per-stage timings. Two event kinds fire per stage:

      - ``{"event": "stage_start", "stage": <name>}`` — just before the stage's
        model is (re)loaded.
      - ``{"event": "stage_end", "stage": <name>, "load_s": float,
        "run_s": float, "wall_s": float}`` — after ``run()`` returns. ``load_s``
        is the wall time spent in ``stage.load()`` alone (``0.0`` when the stage
        was already resident — the reload-skip no-op); ``run_s`` is the wall
        time in ``stage.run()`` (inference). Measuring around the *actual* load
        and run calls — not the whole stage wrapper — is what makes the split
        trustworthy (it retires the stale-``run_meta`` timing-mirage class).

    Default ``on_event=None`` → nothing is emitted and behaviour is byte-identical
    to before the instrumentation existed.
    """

    def __init__(
        self,
        config: PipelineConfig,
        on_event: Optional[Callable[[dict], None]] = None,
    ) -> None:
        self.config = config
        self._on_event = on_event
        self.device = torch.device(config.device)
        if config.deterministic:
            # The enhancement conv stack is the pipeline's only nondeterministic
            # stage (nondeterministic cuDNN algorithms → ~1e-7 noise in
            # enhanced_full that WhisperX can amplify into a token flip). Forcing
            # deterministic algorithms makes the whole pipeline reproducible.
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        self.stages: List[Stage] = [
            DiarizationStage(config.diarization),
            RoutingStage(config.routing),
            EnhancementStage(config.enhancement),
            SeparationStage(config.separation),
            PostSeparationProcessingStage(config.post_separation_processing),
            # 2nd-pass identity re-clustering (B / B+). Default OFF → the loop
            # skips it (byte-identical no-op). Must sit AFTER 3c (B+ reads the
            # `_gated` overlap streams 3c writes) and BEFORE assembly (the first
            # and only consumer of segments_df["speaker"] + the overlap handoff).
            RelabelStage(config.relabel),
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

    def _emit(self, event: str, stage: str, **fields) -> None:
        """Fire an ``on_event`` callback, if one was wired. No-op otherwise."""
        if self._on_event is None:
            return
        payload = {"event": event, "stage": stage}
        payload.update(fields)
        self._on_event(payload)

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
        self._emit("stage_start", stage_name)
        # `_ensure_loaded` returns the wall time spent in `stage.load()` alone
        # (0.0 on a reload-skip no-op) — the load half of the load/run split.
        load_seconds = self._ensure_loaded(stage_name)
        try:
            _log(f"run_stage({stage_name!r}): calling stage.run()")
            t_run = time.perf_counter()
            stage.run(ctx)
            run_seconds = time.perf_counter() - t_run
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
        self._emit(
            "stage_end", stage_name,
            load_s=load_seconds, run_s=run_seconds,
            wall_s=load_seconds + run_seconds,
        )
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

    def _ensure_loaded(self, stage_name: str) -> float:
        """Make `stage_name` the currently-loaded stage; return load wall seconds.

        - Same stage, same load signature → no-op (this is what makes
          within-stage iteration fast — the user can re-run the same
          stage with unchanged model-defining config without paying the
          load cost again). Returns ``0.0`` (nothing loaded).
        - Same stage, different signature → unload + reload (the user
          changed a checkpoint-determining knob between runs).
        - Different stage → unload current + load new.

        The returned float times ``stage.load()`` *alone* — not the preceding
        ``_release_current()`` of the outgoing stage (that is the previous
        stage's teardown, not this stage's load). This keeps the load/run split
        in the emitted timing events honest.
        """
        stage = self.get_stage(stage_name)
        new_sig = stage.load_signature()

        if self._current_stage_name == stage_name:
            if new_sig == self._loaded_signature:
                _log(f"_ensure_loaded({stage_name!r}): already loaded, no-op")
                return 0.0
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
        t_load = time.perf_counter()
        stage.load(self.device)
        load_seconds = time.perf_counter() - t_load
        self._current_stage_name = stage_name
        self._loaded_signature = new_sig
        _log(f"_ensure_loaded({stage_name!r}): stage.load() returned (sig={new_sig!r})")
        return load_seconds
