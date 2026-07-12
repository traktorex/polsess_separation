"""Tests for the `python -m asr_pipeline run` CLI surface (asr_pipeline/__main__.py, A3).

Covers the argument wiring (`--set`, `--write-outputs`), the `_StageProgress`
on_event sink (A6 → CLI), and the `write_run_outputs` helper that materialises
the per-recording eval layout + copies the mixture (the eval-discovery
requirement). All CPU-only: no `Pipeline.run`, no models — the pipeline body is
exercised by the orchestrator tests, here we test the CLI plumbing around it
with a fake context.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from asr_pipeline import __main__ as cli
from asr_pipeline.config import PipelineConfig
from asr_pipeline.context import PipelineContext

SR = 16_000
REC_ID = "rec_cli"


@pytest.fixture(autouse=True)
def _fake_hf_token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-hf-token")


# ---------------------------------------------------------------------------
# Argument parser wiring
# ---------------------------------------------------------------------------


def test_parser_accepts_set_and_write_outputs():
    args = cli._build_parser().parse_args([
        "run", "--config", "c.yaml", "--input", "i.wav",
        "--set", "enhancement.observation_mix_ratio=0.3",
        "--set", "transcription.model_name=large-v3",
        "--write-outputs", "/tmp/evalroot",
    ])
    assert args.command == "run"
    assert args.overrides == [
        "enhancement.observation_mix_ratio=0.3",
        "transcription.model_name=large-v3",
    ]
    assert args.write_outputs == "/tmp/evalroot"


def test_parser_defaults_empty_overrides_and_no_write():
    args = cli._build_parser().parse_args(
        ["run", "--config", "c.yaml", "--input", "i.wav"]
    )
    assert args.overrides == []
    assert args.write_outputs is None
    assert args.output is None


def test_parser_requires_subcommand():
    with pytest.raises(SystemExit):
        cli._build_parser().parse_args([])


# ---------------------------------------------------------------------------
# _StageProgress — the on_event sink (prints + records timings)
# ---------------------------------------------------------------------------


def test_stage_progress_records_timings_and_prints(capsys):
    progress = cli._StageProgress()
    progress({"event": "stage_start", "stage": "diarization"})
    progress({"event": "stage_end", "stage": "diarization",
              "load_s": 1.5, "run_s": 3.0, "wall_s": 4.5})
    assert progress.timings == [
        {"stage": "diarization", "load_s": 1.5, "run_s": 3.0}
    ]
    out = capsys.readouterr().out
    assert "diarization" in out
    assert "running" in out and "done" in out


def test_stage_progress_start_event_records_nothing():
    progress = cli._StageProgress()
    progress({"event": "stage_start", "stage": "x"})
    assert progress.timings == []


# ---------------------------------------------------------------------------
# write_run_outputs — per-recording layout + mixture copy + run_meta
# ---------------------------------------------------------------------------


def _fake_ctx(mixture_path: Path) -> PipelineContext:
    ctx = PipelineContext(input_path=mixture_path, sample_rate=SR)
    ctx.audio = np.zeros(SR, dtype=np.float32)
    # Minimal populated output so metadata.json + a stream/transcript are written.
    ctx.assembled = {"SPEAKER_00": np.zeros(SR, dtype=np.float32)}
    ctx.spk_to_label = {"SPEAKER_00": "A"}
    ctx.speakers = ["SPEAKER_00"]
    ctx.transcripts = {
        "SPEAKER_00": {"text": "test", "segments": [
            {"start": 0.0, "end": 1.0, "text": "test"}], "language": "pl"}
    }
    return ctx


def test_write_run_outputs_full_layout(tmp_path):
    mixture = tmp_path / f"{REC_ID}.wav"
    sf.write(mixture, np.zeros(SR, np.float32), SR)
    eval_root = tmp_path / "eval"
    ctx = _fake_ctx(mixture)
    cfg = PipelineConfig()
    timings = [{"stage": "diarization", "load_s": 1.0, "run_s": 2.0}]

    pipeline_dir = cli.write_run_outputs(
        ctx, cfg, str(eval_root), timings, total_seconds=12.3
    )

    rec_dir = eval_root / REC_ID
    # 1. outputs under <eval_root>/<id>/pipeline/
    assert pipeline_dir == rec_dir / "pipeline"
    assert (pipeline_dir / "metadata.json").exists()
    # 2. mixture copied to <eval_root>/<id>/<id>.wav (eval discovery)
    assert (rec_dir / f"{REC_ID}.wav").exists()
    # 3. run_meta.json holds total + per-stage seconds
    run_meta = json.loads((pipeline_dir / "run_meta.json").read_text())
    assert run_meta["seconds"] == 12.3
    assert run_meta["stages"] == timings
    # 4. metadata.json embeds the per-stage timings (A6)
    meta = json.loads((pipeline_dir / "metadata.json").read_text())
    assert meta["stage_timings"] == timings


def test_write_run_outputs_metadata_redacts_token(tmp_path):
    mixture = tmp_path / f"{REC_ID}.wav"
    sf.write(mixture, np.zeros(SR, np.float32), SR)
    ctx = _fake_ctx(mixture)
    cfg = PipelineConfig()
    cfg.diarization.hf_token = "hf_live_secret"
    pipeline_dir = cli.write_run_outputs(
        ctx, cfg, str(tmp_path / "eval"), [], total_seconds=1.0
    )
    raw = (pipeline_dir / "metadata.json").read_text()
    assert "hf_live_secret" not in raw
    assert json.loads(raw)["config"]["diarization"]["hf_token"] == "REDACTED"
