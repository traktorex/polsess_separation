"""Tests for the batch runner (asr_pipeline/batch.py, A4/A7).

Covers the pieces the three old scripts encoded inconsistently, now unified in
`run_batch`: the skip sentinel + ``--force`` (skip_existing), SCOPE §4.2 failure
isolation (one recording fails → recorded in failures.csv, batch continues), the
GPU teardown running between recordings, the ablation-mode presets matching the
old MODES lambdas, and the legacy sweep sentinel (``transcript_A.txt``) still
being honored so existing sweep trees are not recomputed.

All CPU-only: `asr_pipeline.batch.Pipeline` is monkeypatched with a fake that
returns a populated `PipelineContext` — no models, no GPU (SCOPE §7 stubbing
pattern, same as tests/test_pipeline_orchestrator.py / test_pipeline_cli.py).
"""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from asr_pipeline import batch as batch_mod
from asr_pipeline.batch import BatchReport, MODE_PRESETS, run_batch
from asr_pipeline.config import PipelineConfig, apply_overrides
from asr_pipeline.context import PipelineContext

SR = 16_000


@pytest.fixture(autouse=True)
def _fake_hf_token(monkeypatch):
    # PipelineConfig().__post_init__ requires a token on the pyannote path.
    monkeypatch.setenv("HF_TOKEN", "test-hf-token")


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _make_ctx(audio_path) -> PipelineContext:
    """Minimal populated context so `write_pipeline_outputs` writes both the
    metadata.json (default sentinel) and transcript_A.txt (legacy sentinel)."""
    ctx = PipelineContext(input_path=Path(audio_path), sample_rate=SR)
    ctx.audio = np.zeros(SR, dtype=np.float32)
    ctx.assembled = {"SPEAKER_00": np.zeros(SR, dtype=np.float32)}
    ctx.spk_to_label = {"SPEAKER_00": "A"}
    ctx.speakers = ["SPEAKER_00"]
    ctx.transcripts = {
        "SPEAKER_00": {"text": "t", "segments": [
            {"start": 0.0, "end": 1.0, "text": "t"}], "language": "pl"}
    }
    return ctx


def _install_fake_pipeline(monkeypatch, runlog, *, fail_on=frozenset()):
    """Patch batch.Pipeline with a fake recording construct/run/unload into
    ``runlog``. A recording id in ``fail_on`` raises from run()."""

    class _FakePipeline:
        def __init__(self, cfg, on_event=None):
            self.on_event = on_event
            self.rid = None
            runlog.append(("construct", None))

        def run(self, audio_path):
            self.rid = Path(audio_path).stem
            runlog.append(("run", self.rid))
            if self.rid in fail_on:
                raise RuntimeError(f"boom {self.rid}")
            if self.on_event is not None:
                self.on_event({"event": "stage_start", "stage": "diarization"})
                self.on_event({"event": "stage_end", "stage": "diarization",
                               "load_s": 0.1, "run_s": 0.2, "wall_s": 0.3})
            return _make_ctx(audio_path)

        def unload(self):
            runlog.append(("unload", self.rid))

    monkeypatch.setattr(batch_mod, "Pipeline", _FakePipeline)


def _make_inputs(tmp_path, ids):
    """Create empty <id>.wav inputs (only existence is checked by run_batch)."""
    paths = []
    for rid in ids:
        p = tmp_path / f"{rid}.wav"
        p.touch()
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# Skip sentinel + --force
# ---------------------------------------------------------------------------


def test_skip_sentinel_and_force(tmp_path, monkeypatch):
    runlog = []
    _install_fake_pipeline(monkeypatch, runlog)
    cfg = PipelineConfig()
    audio = _make_inputs(tmp_path, ["a"])
    out_root = tmp_path / "out"

    # First run: produces the outputs incl. metadata.json (the default sentinel).
    r1 = run_batch(cfg, audio, out_root, "pipeline")
    assert r1.succeeded == ["a"] and r1.skipped == [] and r1.failed == []
    assert (out_root / "a" / "pipeline" / "metadata.json").exists()
    runs_1 = [e for e in runlog if e[0] == "run"]
    assert runs_1 == [("run", "a")]

    # Second run, skip_existing=True (default): recognised complete → skipped.
    r2 = run_batch(cfg, audio, out_root, "pipeline")
    assert r2.skipped == ["a"] and r2.succeeded == []
    assert [e for e in runlog if e[0] == "run"] == runs_1   # no new run

    # Third run, skip_existing=False (the CLI --force): re-runs.
    r3 = run_batch(cfg, audio, out_root, "pipeline", skip_existing=False)
    assert r3.succeeded == ["a"]
    assert [e for e in runlog if e[0] == "run"] == [("run", "a"), ("run", "a")]


# ---------------------------------------------------------------------------
# Failure isolation + failures.csv (SCOPE §4.2)
# ---------------------------------------------------------------------------


def test_failure_isolation_and_failures_csv(tmp_path, monkeypatch):
    runlog = []
    _install_fake_pipeline(monkeypatch, runlog, fail_on={"b"})
    cfg = PipelineConfig()
    audio = _make_inputs(tmp_path, ["a", "b", "c"])
    out_root = tmp_path / "out"

    report = run_batch(cfg, audio, out_root, "pipeline")

    # b failed but a and c still ran to completion (batch continued).
    assert report.succeeded == ["a", "c"]
    assert report.failed == [("b", "RuntimeError")]
    assert (out_root / "a" / "pipeline" / "metadata.json").exists()
    assert (out_root / "c" / "pipeline" / "metadata.json").exists()
    assert not (out_root / "b" / "pipeline" / "metadata.json").exists()

    # failures.csv: recording id, exception type, one-line traceback summary.
    fcsv = report.failures_csv
    assert fcsv == out_root / "failures.csv" and fcsv.exists()
    rows = list(csv.DictReader(fcsv.open()))
    assert len(rows) == 1
    assert rows[0]["recording_id"] == "b"
    assert rows[0]["exception_type"] == "RuntimeError"
    assert "boom b" in rows[0]["traceback_summary"]


def test_clean_batch_writes_no_failures_csv(tmp_path, monkeypatch):
    _install_fake_pipeline(monkeypatch, [])
    report = run_batch(PipelineConfig(), _make_inputs(tmp_path, ["a"]),
                       tmp_path / "out", "pipeline")
    assert report.failed == []
    assert report.failures_csv is None
    assert not (tmp_path / "out" / "failures.csv").exists()


def test_missing_input_is_skipped_not_failed(tmp_path, monkeypatch):
    runlog = []
    _install_fake_pipeline(monkeypatch, runlog)
    # 'ghost' never created on disk → treated as a skip (batch continues), not a
    # failure (mirrors the old sweep's "MISSING audio" behaviour).
    _make_inputs(tmp_path, ["a"])
    recs = [tmp_path / "a.wav", tmp_path / "ghost.wav"]
    report = run_batch(PipelineConfig(), recs, tmp_path / "out", "pipeline")
    assert report.succeeded == ["a"]
    assert report.skipped == ["ghost"]
    assert report.failed == []


# ---------------------------------------------------------------------------
# GPU teardown runs between recordings
# ---------------------------------------------------------------------------


def test_teardown_called_per_recording(tmp_path, monkeypatch):
    runlog = []
    _install_fake_pipeline(monkeypatch, runlog)
    audio = _make_inputs(tmp_path, ["a", "b"])

    # Spy on the centralized teardown to confirm it is the path used.
    torn = []
    real_teardown = batch_mod._teardown_pipeline
    monkeypatch.setattr(batch_mod, "_teardown_pipeline",
                        lambda p: (torn.append(p.rid), real_teardown(p)))

    run_batch(PipelineConfig(), audio, tmp_path / "out", "pipeline")

    # A fresh pipeline per recording, each unloaded via the one teardown home.
    assert [e[1] for e in runlog if e[0] == "construct"] == [None, None]
    assert [e[1] for e in runlog if e[0] == "unload"] == ["a", "b"]
    assert torn == ["a", "b"]


def test_teardown_runs_even_on_failure(tmp_path, monkeypatch):
    runlog = []
    _install_fake_pipeline(monkeypatch, runlog, fail_on={"a"})
    run_batch(PipelineConfig(), _make_inputs(tmp_path, ["a"]),
              tmp_path / "out", "pipeline")
    # The failed recording's model is still released (finally block).
    assert ("unload", "a") in runlog


# ---------------------------------------------------------------------------
# run_meta.json gains per-stage seconds (A6)
# ---------------------------------------------------------------------------


def test_run_meta_has_total_and_per_stage_seconds(tmp_path, monkeypatch):
    _install_fake_pipeline(monkeypatch, [])
    run_batch(PipelineConfig(), _make_inputs(tmp_path, ["a"]),
              tmp_path / "out", "pipeline")
    meta = json.loads(
        (tmp_path / "out" / "a" / "pipeline" / "run_meta.json").read_text())
    assert "seconds" in meta and isinstance(meta["seconds"], float)
    # The fake emits one diarization stage_end → one timing row.
    assert meta["stages"] == [
        {"stage": "diarization", "load_s": 0.1, "run_s": 0.2}
    ]
    # And metadata.json embeds the same per-stage timings.
    md = json.loads(
        (tmp_path / "out" / "a" / "pipeline" / "metadata.json").read_text())
    assert md["stage_timings"] == meta["stages"]


def test_on_event_is_forwarded_to_caller(tmp_path, monkeypatch):
    _install_fake_pipeline(monkeypatch, [])
    seen = []
    run_batch(PipelineConfig(), _make_inputs(tmp_path, ["a"]),
              tmp_path / "out", "pipeline", on_event=seen.append)
    assert [e["event"] for e in seen] == ["stage_start", "stage_end"]


# ---------------------------------------------------------------------------
# Mode presets replicate the old MODES lambdas exactly
# ---------------------------------------------------------------------------


def test_mode_presets_match_run_pipeline_modes():
    # Cross-check the batch dotted-dict presets against the run_pipeline_on_recording
    # setattr-lambda encoding they replace (subdir name + resulting sep/enh state).
    from scripts.run_pipeline_on_recording import MODES

    modes_by_subdir = dict(MODES)
    mode_to_subdir = {
        "full": "pipeline",
        "no_sep": "pipeline_nosep",
        "no_enh": "pipeline_noenh",
        "minimal": "pipeline_minimal",
    }
    for mode, subdir in mode_to_subdir.items():
        preset_subdir, preset_overrides = MODE_PRESETS[mode]
        assert preset_subdir == subdir, mode

        cfg_batch = PipelineConfig()
        apply_overrides(cfg_batch, preset_overrides)

        cfg_lambda = PipelineConfig()
        modes_by_subdir[subdir](cfg_lambda)

        assert (cfg_batch.separation.enabled ==
                cfg_lambda.separation.enabled), mode
        assert (cfg_batch.enhancement.enabled ==
                cfg_lambda.enhancement.enabled), mode


def test_mode_preset_keys_are_the_four_ablation_modes():
    assert set(MODE_PRESETS) == {"full", "no_sep", "no_enh", "minimal"}


# ---------------------------------------------------------------------------
# Legacy sweep sentinel (transcript_A.txt) is honored
# ---------------------------------------------------------------------------


def test_legacy_sweep_sentinel_recognises_existing_tree(tmp_path, monkeypatch):
    runlog = []
    _install_fake_pipeline(monkeypatch, runlog)
    cfg = PipelineConfig()
    _make_inputs(tmp_path, ["rec1"])
    audio = [(tmp_path / "rec1.wav")]  # id = rec1
    out_root = tmp_path

    # Simulate an existing completed sweep tree: transcript_A.txt present but NO
    # metadata.json (so the DEFAULT sentinel would not see it).
    legacy_dir = out_root / "rec1" / "sweep" / "cfgA"
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "transcript_A.txt").write_text("done\n", encoding="utf-8")

    legacy_sentinel = lambda d: (d / "transcript_A.txt").exists()  # noqa: E731

    # With the legacy sentinel the existing tree is recognised → NOT recomputed.
    r_legacy = run_batch(cfg, audio, out_root, "sweep/cfgA",
                         is_complete=legacy_sentinel, copy_mixture=False)
    assert r_legacy.skipped == ["rec1"]
    assert [e for e in runlog if e[0] == "run"] == []

    # With the DEFAULT sentinel (metadata.json, absent here) it is NOT complete →
    # it runs. Proves the sentinel choice is what makes the migration safe.
    r_default = run_batch(cfg, audio, out_root, "sweep/cfgA", copy_mixture=False)
    assert r_default.succeeded == ["rec1"]
    assert [e for e in runlog if e[0] == "run"] == [("run", "rec1")]


# ---------------------------------------------------------------------------
# (recording_id, path) items pin the id independently of the filename
# ---------------------------------------------------------------------------


def test_tuple_recording_pins_id(tmp_path, monkeypatch):
    _install_fake_pipeline(monkeypatch, [])
    # A legacy-style mixture.wav under an <id>/ dir: the id must come from the
    # tuple, not the filename stem ("mixture").
    rec_dir = tmp_path / "myrec"
    rec_dir.mkdir()
    mixture = rec_dir / "mixture.wav"
    mixture.touch()
    report = run_batch(PipelineConfig(), [("myrec", mixture)], tmp_path,
                       "pipeline", copy_mixture=False)
    assert report.succeeded == ["myrec"]
    assert (tmp_path / "myrec" / "pipeline" / "metadata.json").exists()


# ---------------------------------------------------------------------------
# BatchReport shape
# ---------------------------------------------------------------------------


def test_batch_report_counts(tmp_path, monkeypatch):
    _install_fake_pipeline(monkeypatch, [], fail_on={"b"})
    audio = _make_inputs(tmp_path, ["a", "b"])
    report = run_batch(PipelineConfig(), audio, tmp_path / "out", "pipeline")
    assert isinstance(report, BatchReport)
    assert report.n_total == 2
    assert report.out_root == (tmp_path / "out")
    assert report.subdir_name == "pipeline"
