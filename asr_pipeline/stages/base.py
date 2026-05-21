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

    def spill(self, ctx: PipelineContext, artifact_dir: Path) -> None:
        """Write this stage's outputs in `ctx` to `artifact_dir`.

        Default: no-op. Stages with disk-serialisable outputs override this.
        Called by the orchestrator after `run()` when `spill_intermediate`
        is enabled.
        """
        return None
