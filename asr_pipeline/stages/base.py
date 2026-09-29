"""Stage ABC.

A stage owns one phase of the pipeline:

    load(device)   # bring the stage's model onto `device` (no-op if stageless)
    run(ctx)       # read inputs from ctx, write outputs back onto ctx
    unload()       # free the model (no-op if stageless)

The orchestrator (`pipeline.py`) calls these three in order for each
enabled stage. The default `load`/`unload` are no-ops so stages without a
model (e.g. routing) need only override `run`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

from asr_pipeline.context import PipelineContext


def match_length(x: np.ndarray, n: int) -> np.ndarray:
    """Right-trim or zero-pad ``x`` to exactly ``n`` samples (tail-aligned).

    Shared primitive for the several stages that reconcile a neural
    backend's output length against a target: the separator's resample
    round-trip, ClearerVoice enhancement's resample round-trip, and the
    BWE backends' STFT/iSTFT framing can each drift by a handful of
    samples. Trims from the tail when too long, zero-pads the tail when
    too short, and preserves the input dtype.
    """
    return x[:n] if len(x) >= n else np.pad(x, (0, n - len(x)))


class Stage(ABC):
    """Base class for all pipeline stages."""

    name: str = "stage"

    # Optional inner-loop progress sink, wired by the orchestrator around
    # `run()` and cleared afterwards. `None` (the class default, and what every
    # stage sees when `Pipeline` has no `on_event` callback) makes `_progress`
    # a no-op — stages behave exactly as they did before the sink existed. A
    # class attribute rather than an `__init__` field so a stage that doesn't
    # chain to `super().__init__` still has it.
    on_progress: Optional[Callable[[int, int], None]] = None

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def load(self, device: torch.device) -> None:  # noqa: D401 — see class docstring
        """Bring the stage's model onto `device`. Default: no-op."""
        return None

    @abstractmethod
    def run(self, ctx: PipelineContext) -> None:
        """Read inputs from `ctx`, write outputs back onto `ctx`."""

    def unload(self) -> None:
        """Free the stage's model. Default: no-op."""
        return None

    def load_signature(self) -> tuple:
        """Identity of the model that `load()` would currently load.

        The orchestrator compares this against the signature recorded
        at last load — if they differ, the stage is unloaded and
        reloaded before the next `run`. Include in the tuple only the
        config values that determine *which model* gets instantiated
        (checkpoint paths, backend selectors, model names). Runtime
        knobs that affect behaviour but not the loaded model (VAD
        thresholds, chunking lengths, language hints, etc.) should
        NOT be in the signature — those can be re-read on every call
        without paying the reload cost.

        Default: empty tuple = stage's loaded model never depends on
        config, so it's never reloaded due to a config change.
        """
        return ()

    def spill(self, ctx: PipelineContext, artifact_dir: Path) -> None:
        """Write this stage's outputs in `ctx` to `artifact_dir`.

        Default: no-op. Stages with disk-serialisable outputs override this.
        Called by the orchestrator after `run()` when `spill_intermediate`
        is enabled.
        """
        return None

    def _progress(self, done: int, total: int) -> None:
        """Report inner-loop progress (`done` of `total` items), if wired.

        Stages with long per-item loops (separation,
        post_separation_processing, assembly) call this at their existing
        per-item log points; the orchestrator turns each call into a
        `stage_progress` event.
        """
        if self.on_progress is not None:
            self.on_progress(done, total)
