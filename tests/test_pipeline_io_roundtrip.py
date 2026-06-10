"""Round-trip test: `write_pipeline_outputs` → eval-module readers.

This is the central disk contract between the pipeline and the eval
module: the writer's per-recording layout must be readable by
`load_recording`, `compute_layer1` (DER) and `compute_layer3` (WER).
A fake but complete `PipelineContext` whose outputs exactly match the
reference must score DER = 0 and cpWER = 0.

Also asserts the config snapshot embedded in `metadata.json` never
carries a live HF token.
"""

import json

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from asr_pipeline.context import DiarizationResult, PipelineContext
from asr_pipeline.eval.layer1 import compute_layer1
from asr_pipeline.eval.layer3 import compute_layer3
from asr_pipeline.eval.recordings import load_recording
from asr_pipeline.eval.transcript_parser import parse_eaf
from asr_pipeline.io import write_pipeline_outputs

SR = 16_000
REC_ID = "rec1"

# (speaker_id, label, start_s, end_s, text)
_UTTS = [
    ("SPEAKER_00", "A", 1.0, 2.0, "ala ma kota"),
    ("SPEAKER_01", "B", 5.0, 6.0, "pies je obiad"),
]


def _whisper_result(segments) -> dict:
    return {
        "text": " ".join(t for _, _, t in segments),
        "segments": [
            {"start": s, "end": e, "text": t} for s, e, t in segments
        ],
        "language": "pl",
    }


def _fake_ctx(rec_dir) -> PipelineContext:
    seg_df = pd.DataFrame(
        [
            {"start": s, "end": e, "duration": e - s, "speaker": spk}
            for spk, _, s, e, _ in _UTTS
        ],
        columns=["start", "end", "duration", "speaker"],
    )
    ovl_df = pd.DataFrame(columns=["start", "end", "duration"])

    ctx = PipelineContext(
        input_path=rec_dir / f"{REC_ID}.wav", sample_rate=SR
    )
    ctx.audio = np.zeros(10 * SR, dtype=np.float32)
    ctx.diarization = DiarizationResult(
        segments_df=seg_df, overlaps_df=ovl_df, total_duration_s=10.0
    )
    ctx.overlap_regions = []
    ctx.speakers = [spk for spk, *_ in _UTTS]
    ctx.assembled = {spk: np.zeros(SR, dtype=np.float32) for spk, *_ in _UTTS}
    ctx.spk_to_label = {spk: label for spk, label, *_ in _UTTS}
    ctx.transcripts = {
        spk: _whisper_result([(s, e, t)]) for spk, _, s, e, t in _UTTS
    }
    ctx.mixture_transcript = _whisper_result(
        [(s, e, t) for _, _, s, e, t in _UTTS]
    )
    return ctx


@pytest.fixture
def eval_recording(tmp_path):
    """Write a full fake recording (mixture + reference + pipeline outputs)
    and return the loaded `Recording`."""
    rec_dir = tmp_path / "eval" / "clarin" / REC_ID
    rec_dir.mkdir(parents=True)
    sf.write(rec_dir / f"{REC_ID}.wav", np.zeros(10 * SR, np.float32), SR)

    # Reference: per-speaker GT txt + RTTM, matching the pipeline output.
    ref_dir = rec_dir / "reference"
    ref_dir.mkdir()
    rttm_lines = []
    for _, label, s, e, text in _UTTS:
        (ref_dir / f"speaker_{label}.txt").write_text(
            f"[{s:6.2f} → {e:6.2f}]  {text}\n", encoding="utf-8"
        )
        rttm_lines.append(
            f"SPEAKER {REC_ID} 1 {s:.2f} {e - s:.2f} <NA> <NA> {label} <NA> <NA>"
        )
    (ref_dir / "diarization.rttm").write_text("\n".join(rttm_lines) + "\n")

    ctx = _fake_ctx(rec_dir)
    config_snapshot = {
        "sample_rate": SR,
        "diarization": {"model_id": "test", "hf_token": "hf_live_secret"},
    }
    write_pipeline_outputs(
        ctx, rec_dir, config_snapshot=config_snapshot, subdir_name="pipeline"
    )

    rec = load_recording(rec_dir)
    assert rec is not None
    return rec


def test_writer_layout_is_discoverable(eval_recording):
    rec = eval_recording
    assert rec.id == REC_ID
    assert rec.dataset == "clarin"
    assert rec.pipeline_dir is not None
    assert set(rec.reference_transcripts) == {"A", "B"}
    assert rec.reference_diarization is not None
    for label in ("A", "B"):
        assert (rec.pipeline_dir / f"stream_{label}.wav").exists()
        assert (rec.pipeline_dir / f"transcript_{label}.txt").exists()
        assert (rec.pipeline_dir / f"transcript_{label}.json").exists()
    assert (rec.pipeline_dir / "transcript_mixture.txt").exists()


def test_metadata_config_token_is_redacted(eval_recording):
    meta_path = eval_recording.pipeline_dir / "metadata.json"
    raw = meta_path.read_text(encoding="utf-8")
    assert "hf_live_secret" not in raw
    meta = json.loads(raw)
    assert meta["config"]["diarization"]["hf_token"] == "REDACTED"
    # The rest of the snapshot survives untouched.
    assert meta["config"]["diarization"]["model_id"] == "test"


def test_writer_omits_config_when_snapshot_none(tmp_path):
    """With config_snapshot=None, metadata.json omits the 'config' key — the
    branch that runs when a run dies before a config is attached."""
    rec_dir = tmp_path / "eval" / "clarin" / REC_ID
    rec_dir.mkdir(parents=True)
    ctx = _fake_ctx(rec_dir)
    write_pipeline_outputs(ctx, rec_dir, config_snapshot=None, subdir_name="pipeline")
    meta = json.loads(
        (rec_dir / "pipeline" / "metadata.json").read_text(encoding="utf-8")
    )
    assert "config" not in meta


def test_layer1_der_is_zero_on_matching_diarization(eval_recording):
    l1 = compute_layer1(eval_recording)
    assert l1 is not None
    assert l1["reference_source"] == "rttm"
    assert l1["der_stage1"]["der"] == pytest.approx(0.0, abs=1e-6)


def test_layer3_wer_is_zero_on_matching_transcripts(eval_recording):
    l3 = compute_layer3(eval_recording)
    assert l3 is not None
    assert l3["modes"]["full"]["cpwer"] == pytest.approx(0.0)
    assert l3["modes"]["no_sep"] is None
    assert l3["mixture_orc"]["orc_wer"] == pytest.approx(0.0)
    assert l3["mixture_mimo"]["mimo_wer"] == pytest.approx(0.0)


def test_writer_eaf_round_trips(eval_recording):
    eaf_path = eval_recording.pipeline_dir / "annotation.eaf"
    assert eaf_path.exists()
    tiers = parse_eaf(eaf_path)
    assert set(tiers) == {"A", "B"}
    assert tiers["A"][0].text == "ala ma kota"
    assert tiers["A"][0].start == pytest.approx(1.0)
