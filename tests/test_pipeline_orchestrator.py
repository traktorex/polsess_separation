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


class _FailingStage(_DummyStage):
    """A dummy stage whose run() logs then raises — exercises failure cleanup."""

    def run(self, ctx) -> None:
        self._logbook.append(("run", self.name))
        raise RuntimeError("boom")


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


def test_deterministic_config_sets_cudnn_flags(monkeypatch):
    # deterministic=True (default) forces deterministic cuDNN algorithms — the
    # enhancement stage is otherwise the pipeline's only nondeterminism source.
    import torch
    prev = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    try:
        monkeypatch.setenv("HF_TOKEN", "test-hf-token")
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        cfg = PipelineConfig()
        cfg.device = "cpu"
        cfg.deterministic = True
        Pipeline(cfg)
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
    finally:
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = prev


def test_deterministic_false_leaves_cudnn_untouched(monkeypatch):
    import torch
    prev = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    try:
        monkeypatch.setenv("HF_TOKEN", "test-hf-token")
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        cfg = PipelineConfig()
        cfg.device = "cpu"
        cfg.deterministic = False
        Pipeline(cfg)
        assert torch.backends.cudnn.deterministic is False
        assert torch.backends.cudnn.benchmark is True
    finally:
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = prev


def test_stage_failure_unloads_model(pipeline):
    # A stage raising in run() must release its model (one-model-at-a-time GPU
    # budget) and reset bookkeeping, then re-raise — never strand the model.
    p, log = pipeline
    p.stages = [_FailingStage("a", log)]
    with pytest.raises(RuntimeError, match="boom"):
        p.run_stage("a", _ctx())
    assert log == [("load", "a"), ("run", "a"), ("unload", "a")]
    assert p._current_stage_name is None


def test_run_oneshot_happy_path_runs_every_enabled_stage(monkeypatch):
    # The one-shot run() entry point on a clean path: load_audio → every
    # enabled stage in order → final unload → returns the same ctx it built.
    # A disabled stage is skipped (run never logged).
    monkeypatch.setenv("HF_TOKEN", "test-hf-token")
    cfg = PipelineConfig()
    cfg.device = "cpu"
    p = Pipeline(cfg)
    log: list = []
    a, b, c = _DummyStage("a", log), _DummyStage("b", log), _DummyStage("c", log)
    b.enabled = False
    p.stages = [a, b, c]
    sentinel = _ctx()
    monkeypatch.setattr(p, "load_audio", lambda path: sentinel)

    out = p.run("dummy.wav")
    assert out is sentinel                       # returns the ctx it built
    assert ("run", "b") not in log               # disabled stage skipped
    # a then c ran in order; switching a→c unloads a; final unload frees c.
    assert [e for e in log if e[0] == "run"] == [("run", "a"), ("run", "c")]
    assert log[-1] == ("unload", "c")
    assert p._current_stage_name is None


def test_spill_intermediate_calls_stage_spill(monkeypatch, tmp_path):
    # spill_intermediate=True wires an artifact_dir; each stage's spill() is
    # invoked after its run() (the legacy per-stage artefact path).
    monkeypatch.setenv("HF_TOKEN", "test-hf-token")
    cfg = PipelineConfig()
    cfg.device = "cpu"
    cfg.spill_intermediate = True
    cfg.artifact_dir = str(tmp_path / "artefacts")
    p = Pipeline(cfg)
    assert p.artifact_dir is not None
    assert p.artifact_dir.exists()               # ensure_artifact_dir created it

    spilled: list = []

    class _SpillStage(_DummyStage):
        def spill(self, ctx, artifact_dir) -> None:
            spilled.append((self.name, artifact_dir))

    log: list = []
    p.stages = [_SpillStage("a", log)]
    p.run_stage("a", _ctx())
    assert spilled == [("a", p.artifact_dir)]


def test_no_spill_when_intermediate_disabled(monkeypatch):
    # Default (spill_intermediate=False): artifact_dir is None, so spill() is
    # never reached even if a stage defines it.
    monkeypatch.setenv("HF_TOKEN", "test-hf-token")
    cfg = PipelineConfig()
    cfg.device = "cpu"
    p = Pipeline(cfg)
    assert p.artifact_dir is None

    spilled: list = []

    class _SpillStage(_DummyStage):
        def spill(self, ctx, artifact_dir) -> None:
            spilled.append(self.name)

    p.stages = [_SpillStage("a", [])]
    p.run_stage("a", _ctx())
    assert spilled == []


def test_run_oneshot_failure_halts_loop_and_unloads(monkeypatch):
    # The one-shot run() entry point: a mid-pipeline failure must HALT the
    # loop (the later stage 'c' never runs) and leave no model resident —
    # run() adds no failure handling of its own beyond run_stage's cleanup.
    monkeypatch.setenv("HF_TOKEN", "test-hf-token")
    cfg = PipelineConfig()
    cfg.device = "cpu"
    p = Pipeline(cfg)
    log: list = []
    p.stages = [_DummyStage("a", log), _FailingStage("b", log), _DummyStage("c", log)]
    monkeypatch.setattr(p, "load_audio", lambda path: _ctx())
    with pytest.raises(RuntimeError, match="boom"):
        p.run("dummy.wav")
    assert ("run", "c") not in log          # loop halted at the failure
    assert ("unload", "b") in log           # failed stage's model freed
    assert p._current_stage_name is None


# ---------------------------------------------------------------------------
# on_event stage progress / timing instrumentation (A6)
# ---------------------------------------------------------------------------


def _events_pipeline(monkeypatch):
    """CPU pipeline with an on_event collector + two dummy stages 'a' and 'b'.

    Returns (pipeline, events, logbook): `events` accumulates the emitted event
    dicts, `logbook` the (load/run/unload, stage) tuples from the dummy stages.
    """
    monkeypatch.setenv("HF_TOKEN", "test-hf-token")
    cfg = PipelineConfig()
    cfg.device = "cpu"
    events: list = []
    p = Pipeline(cfg, on_event=lambda e: events.append(e))
    log: list = []
    p.stages = [_DummyStage("a", log), _DummyStage("b", log)]
    return p, events, log


def test_on_event_fires_start_end_in_order(monkeypatch):
    p, events, _ = _events_pipeline(monkeypatch)
    monkeypatch.setattr(p, "load_audio", lambda path: _ctx())
    p.run("dummy.wav")
    assert [(e["event"], e["stage"]) for e in events] == [
        ("stage_start", "a"), ("stage_end", "a"),
        ("stage_start", "b"), ("stage_end", "b"),
    ]


def test_on_event_end_carries_load_run_split(monkeypatch):
    p, events, _ = _events_pipeline(monkeypatch)
    p.run_stage("a", _ctx())
    end = next(e for e in events if e["event"] == "stage_end")
    assert {"load_s", "run_s", "wall_s"} <= set(end)
    assert isinstance(end["load_s"], float) and end["load_s"] >= 0.0
    assert isinstance(end["run_s"], float) and end["run_s"] >= 0.0
    # wall_s is exactly the two halves summed (the split is not a re-measure).
    assert end["wall_s"] == pytest.approx(end["load_s"] + end["run_s"])


def test_on_event_rerun_same_stage_reports_zero_load(monkeypatch):
    # The reload-skip no-op: re-running the same stage with an unchanged
    # signature loads nothing, so its stage_end reports load_s == 0.0 — the
    # boundary that makes the load/run split trustworthy.
    p, events, _ = _events_pipeline(monkeypatch)
    p.run_stage("a", _ctx())
    p.run_stage("a", _ctx())
    ends = [e for e in events if e["event"] == "stage_end"]
    assert len(ends) == 2
    assert ends[1]["load_s"] == 0.0


def test_on_event_switching_stage_counts_load(monkeypatch):
    # Switching a → b actually loads b, so b's load_s is measured (>= 0.0 and a
    # real float, not the 0.0 no-op sentinel path).
    p, events, _ = _events_pipeline(monkeypatch)
    p.run_stage("a", _ctx())
    p.run_stage("b", _ctx())
    b_end = next(e for e in events if e["event"] == "stage_end" and e["stage"] == "b")
    assert isinstance(b_end["load_s"], float)


def test_on_event_failure_emits_start_not_end(monkeypatch):
    # A stage that raises fires stage_start but never stage_end (the emit sits
    # after a successful run()).
    p, events, log = _events_pipeline(monkeypatch)
    p.stages = [_FailingStage("a", log)]
    with pytest.raises(RuntimeError, match="boom"):
        p.run_stage("a", _ctx())
    kinds = [e["event"] for e in events]
    assert "stage_start" in kinds
    assert "stage_end" not in kinds


def test_on_event_none_is_a_noop(monkeypatch):
    # Default on_event=None: nothing emitted, behaviour unchanged (no crash,
    # same stage lifecycle as the un-instrumented pipeline).
    monkeypatch.setenv("HF_TOKEN", "test-hf-token")
    cfg = PipelineConfig()
    cfg.device = "cpu"
    p = Pipeline(cfg)                 # no on_event
    assert p._on_event is None
    log: list = []
    p.stages = [_DummyStage("a", log)]
    p.run_stage("a", _ctx())         # must not raise
    assert log == [("load", "a"), ("run", "a")]
