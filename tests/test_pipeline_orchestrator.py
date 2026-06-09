"""Tests for the `Pipeline` orchestrator lifecycle (asr_pipeline/pipeline.py).

The orchestrator owns the one-model-at-a-time invariant: load on first
use, no-op on re-run with unchanged signature, unload + reload on
signature change, unload previous on stage switch. These tests exercise
that bookkeeping with dummy stages — no real models involved.
"""

import pytest

from asr_pipeline.config import PipelineConfig
from asr_pipeline.context import PipelineContext
from asr_pipeline.pipeline import Pipeline
from asr_pipeline.stages.base import Stage


class _DummyStage(Stage):
    """Records (event, stage_name) tuples into a shared logbook."""

    def __init__(self, name: str, logbook: list, signature: tuple = ()):
        super().__init__(enabled=True)
        self.name = name
        self._logbook = logbook
        self.signature = tuple(signature)

    def load(self, device) -> None:
        self._logbook.append(("load", self.name))

    def run(self, ctx) -> None:
        self._logbook.append(("run", self.name))

    def unload(self) -> None:
        self._logbook.append(("unload", self.name))

    def load_signature(self) -> tuple:
        return self.signature


@pytest.fixture
def pipeline(monkeypatch):
    """CPU pipeline with two dummy stages 'a' and 'b' + their shared logbook."""
    monkeypatch.setenv("HF_TOKEN", "test-hf-token")
    cfg = PipelineConfig()
    cfg.device = "cpu"
    p = Pipeline(cfg)
    logbook: list = []
    p.stages = [_DummyStage("a", logbook), _DummyStage("b", logbook)]
    return p, logbook


def _ctx() -> PipelineContext:
    return PipelineContext()


def test_first_run_loads_then_runs(pipeline):
    p, log = pipeline
    p.run_stage("a", _ctx())
    assert log == [("load", "a"), ("run", "a")]


def test_rerun_same_stage_does_not_reload(pipeline):
    p, log = pipeline
    p.run_stage("a", _ctx())
    p.run_stage("a", _ctx())
    assert log == [("load", "a"), ("run", "a"), ("run", "a")]


def test_switching_stage_unloads_previous(pipeline):
    p, log = pipeline
    p.run_stage("a", _ctx())
    p.run_stage("b", _ctx())
    assert log == [
        ("load", "a"), ("run", "a"),
        ("unload", "a"), ("load", "b"), ("run", "b"),
    ]


def test_signature_change_triggers_reload(pipeline):
    p, log = pipeline
    stage_a = p.get_stage("a")
    stage_a.signature = ("ckpt_v1",)
    p.run_stage("a", _ctx())
    stage_a.signature = ("ckpt_v2",)
    p.run_stage("a", _ctx())
    assert log == [
        ("load", "a"), ("run", "a"),
        ("unload", "a"), ("load", "a"), ("run", "a"),
    ]


def test_unload_releases_current_stage(pipeline):
    p, log = pipeline
    p.run_stage("a", _ctx())
    p.unload()
    assert log[-1] == ("unload", "a")
    # Unload again is a no-op (nothing loaded).
    p.unload()
    assert log[-1] == ("unload", "a")
    # Next run loads fresh.
    p.run_stage("a", _ctx())
    assert log[-2:] == [("load", "a"), ("run", "a")]


def test_disabled_stage_raises(pipeline):
    p, _ = pipeline
    p.get_stage("a").enabled = False
    with pytest.raises(RuntimeError, match="disabled"):
        p.run_stage("a", _ctx())


def test_unknown_stage_raises_with_valid_names(pipeline):
    p, _ = pipeline
    with pytest.raises(ValueError, match="nonexistent"):
        p.get_stage("nonexistent")
