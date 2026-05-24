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

import torch

from asr_pipeline.context import PipelineContext


class Stage(ABC):
    """Base class for all pipeline stages."""

    name: str = "stage"

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
