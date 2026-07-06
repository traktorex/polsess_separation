"""Unit tests for Stage 1 diarization (asr_pipeline/stages/diarization.py).

Exercises the stage-owned logic — the run-before-load / audio-None guards, the
segment/overlap DataFrame construction (incl. the empty-input column guard and
3-dp rounding), the pyannote 3.x/4.x result duck-typing, load_signature, the
empty-token guard, and the spill schema — all with a fake pipeline object. No
real pyannote model is loaded; the model forward is a third-party boundary.
"""

import json

import numpy as np
import pytest
import torch

from asr_pipeline.config import DiarizationConfig
from asr_pipeline.context import PipelineContext
from asr_pipeline.stages.diarization import DiarizationStage

MODEL_ID = "pyannote/speaker-diarization-3.1"


class _Seg:
    """A pyannote-Segment stand-in: .start / .end / .duration."""

    def __init__(self, start, end, duration=None):
        self.start = start
        self.end = end
        self.duration = duration if duration is not None else end - start


class _FakeDiar:
    """Bare-Annotation stand-in (pyannote 3.x shape)."""

    def __init__(self, tracks, overlaps):
        self._tracks = tracks       # list of (_Seg, speaker_label)
        self._overlaps = overlaps   # list of _Seg

    def itertracks(self, yield_label=False):
        for seg, label in self._tracks:
            yield (seg, "_", label) if yield_label else (seg, "_")

    def get_overlap(self):
        return self._overlaps


class _FakeWrapper:
    """DiarizeOutput stand-in (pyannote 4.x shape) — wraps the Annotation."""

    def __init__(self, diar):
        self.speaker_diarization = diar


class _FakePipeline:
    """Callable stand-in for a loaded pyannote Pipeline."""

    def __init__(self, diar):
        self.diar = diar
        self.calls = []

    def __call__(self, inputs, num_speakers=None):
        self.calls.append((inputs, num_speakers))
        return self.diar


def _stage(diar, model_id=MODEL_ID, num_speakers=2) -> DiarizationStage:
    stage = DiarizationStage(DiarizationConfig(model_id=model_id, num_speakers=num_speakers))
    stage._pipeline = _FakePipeline(diar)   # inject; bypass load()
    return stage


def _ctx(audio=None, sr=16_000) -> PipelineContext:
    ctx = PipelineContext(sample_rate=sr)
    ctx.audio = np.zeros(sr, dtype=np.float32) if audio is None else audio
    return ctx


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def test_run_before_load_raises():
    stage = DiarizationStage(DiarizationConfig())   # no _pipeline
    with pytest.raises(RuntimeError, match="before load"):
        stage.run(_ctx())


def test_run_audio_none_raises():
    stage = _stage(_FakeDiar([], []))
    ctx = PipelineContext()
    ctx.audio = None
    with pytest.raises(RuntimeError, match="audio is None"):
        stage.run(ctx)


# ---------------------------------------------------------------------------
# run() — segment/overlap construction
# ---------------------------------------------------------------------------


def test_run_builds_segments_and_overlaps():
    diar = _FakeDiar(
        tracks=[(_Seg(1.0, 2.0), "SPEAKER_00"), (_Seg(3.5, 4.25), "SPEAKER_01")],
        overlaps=[_Seg(1.5, 1.8)],
    )
    stage = _stage(diar)
    ctx = _ctx()
    stage.run(ctx)

    seg = ctx.diarization.segments_df
    assert list(seg.columns) == ["start", "end", "duration", "speaker"]
    assert len(seg) == 2
    assert seg.iloc[0]["speaker"] == "SPEAKER_00"
    assert seg.iloc[1]["end"] == 4.25
    ovl = ctx.diarization.overlaps_df
    assert len(ovl) == 1
    assert ovl.iloc[0]["start"] == 1.5
    assert ctx.diarization.total_duration_s == pytest.approx(1.0)   # 16000 / 16000
    # num_speakers threaded into the pipeline call.
    assert stage._pipeline.calls[0][1] == 2


def test_run_rounds_to_3dp_and_reads_duration_attribute():
    # duration set independent of end-start to prove the stage reads the
    # attribute rather than recomputing it.
    diar = _FakeDiar(tracks=[(_Seg(1.23456, 2.34567, duration=0.98765), "A")],
                     overlaps=[_Seg(1.111111, 1.222222)])
    stage = _stage(diar)
    ctx = _ctx()
    stage.run(ctx)
    row = ctx.diarization.segments_df.iloc[0]
    assert row["start"] == 1.235
    assert row["end"] == 2.346
    assert row["duration"] == 0.988
    assert ctx.diarization.overlaps_df.iloc[0]["start"] == 1.111


def test_run_empty_diarization_keeps_columns():
    stage = _stage(_FakeDiar([], []))
    ctx = _ctx()
    stage.run(ctx)
    seg = ctx.diarization.segments_df
    ovl = ctx.diarization.overlaps_df
    assert len(seg) == 0
    assert list(seg.columns) == ["start", "end", "duration", "speaker"]
    assert len(ovl) == 0
    assert list(ovl.columns) == ["start", "end", "duration"]


def test_run_handles_pyannote4_wrapper_result():
    diar = _FakeDiar([(_Seg(0.0, 1.0), "A")], [])
    stage = DiarizationStage(DiarizationConfig())
    stage._pipeline = _FakePipeline(_FakeWrapper(diar))   # .speaker_diarization arm
    ctx = _ctx()
    stage.run(ctx)
    assert len(ctx.diarization.segments_df) == 1
    assert ctx.diarization.segments_df.iloc[0]["speaker"] == "A"


# ---------------------------------------------------------------------------
# load_signature + load guard
# ---------------------------------------------------------------------------


def test_load_signature_includes_frontend_knobs_not_num_speakers():
    # num_speakers is a call-time knob (excluded); the instantiate-time front-end
    # hyperparameters ARE included so changing one triggers a reload.
    stage = DiarizationStage(DiarizationConfig(model_id=MODEL_ID, num_speakers=3))
    assert stage.load_signature() == (MODEL_ID, None, 0.0, 0.7045654963945799, 12)
    # a changed front-end knob changes the signature
    s2 = DiarizationStage(
        DiarizationConfig(model_id=MODEL_ID, segmentation_min_duration_off=0.5)
    )
    assert s2.load_signature() != stage.load_signature()
    # an embedding swap also changes the signature (triggers reload)
    s3 = DiarizationStage(
        DiarizationConfig(model_id=MODEL_ID, embedding="eek/wespeaker-voxceleb-resnet293-LM")
    )
    assert s3.load_signature() != stage.load_signature()


def test_load_empty_token_raises(monkeypatch):
    # The guard sits behind `from pyannote.audio import Pipeline`, so the import
    # must succeed first — skip on a pyannote-absent machine rather than fake it.
    pytest.importorskip("pyannote.audio")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    stage = DiarizationStage(DiarizationConfig(hf_token=""))
    with pytest.raises(RuntimeError, match="hf_token"):
        stage.load(torch.device("cpu"))


# ---------------------------------------------------------------------------
# spill — the {segments, overlaps} schema (NOT the eval-facing {turns})
# ---------------------------------------------------------------------------


def test_spill_writes_segments_overlaps_schema(tmp_path):
    diar = _FakeDiar([(_Seg(1.0, 2.0), "A")], [_Seg(1.5, 1.8)])
    stage = _stage(diar)
    ctx = _ctx()
    stage.run(ctx)
    stage.spill(ctx, tmp_path)

    payload = json.loads((tmp_path / "diarization.json").read_text())
    assert set(payload) == {"segments", "overlaps", "total_duration_s"}
    assert "turns" not in payload          # deliberately NOT io.py's eval schema
    assert payload["segments"][0]["speaker"] == "A"
    assert len(payload["overlaps"]) == 1


def test_spill_noop_when_no_diarization(tmp_path):
    stage = DiarizationStage(DiarizationConfig())
    stage.spill(PipelineContext(), tmp_path)        # ctx.diarization is None
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# Sortformer (EEND) backend — turn-building adapter + stage dispatch
# ---------------------------------------------------------------------------
#
# Ground truth for the turn-building is docs/sweep_plan/eend_probe.py (the probe
# that justified the arm). These exercise the port + the stage's subprocess seam
# with a canned-JSON stub worker — no GPU, no NeMo.

from asr_pipeline.stages.diarization import (
    _merge_surplus_heads,
    _runs_from_mask,
    build_sortformer_annotation,
    sortformer_turns_from_probs,
)

_FS = 0.08   # Sortformer frame shift (seconds)


def _probs(active_by_head, T, S=4):
    """(T, S) prob array: `active_by_head` maps head index -> list of active
    frame ranges [(lo, hi), ...] (hi exclusive); active frames get 0.9, else 0."""
    probs = np.zeros((T, S), dtype=np.float32)
    for head, ranges in active_by_head.items():
        for lo, hi in ranges:
            probs[lo:hi, head] = 0.9
    return probs


def test_runs_from_mask_min_dur_drops_blip_and_splits_on_gap():
    # frames 0-3 contiguous (0.32s run, kept); the 1-frame silence at index 4 is a
    # 0.16s gap (> 0.10s max_gap in the probe's (i-p)*fs metric) so runs never
    # merge; the lone blip at index 5 is 0.08s < 0.10s min-dur -> dropped.
    mask = np.zeros(12, dtype=bool)
    mask[[0, 1, 2, 3, 5]] = True
    runs = _runs_from_mask(mask, _FS, 0.10, 0.10)
    assert runs == [(0.0, 4 * _FS)]


def test_sortformer_turns_top2_and_labels():
    # heads 0 and 1 carry all speech (disjoint in time); heads 2, 3 silent.
    probs = _probs({0: [(0, 50)], 1: [(50, 100)]}, T=100)
    turns, diag = sortformer_turns_from_probs(probs, _FS, threshold=0.5)
    assert diag["top2"] == [0, 1]
    assert diag["n_spk"] == 2
    assert diag["leak"] == 0.0
    # labels: most-active head first -> SPEAKER_00 / SPEAKER_01
    assert {t[0] for t in turns} == {"SPEAKER_00", "SPEAKER_01"}


def test_sortformer_turns_threshold_gates_activity():
    # a head just under threshold contributes no turns.
    probs = np.zeros((50, 4), dtype=np.float32)
    probs[0:50, 0] = 0.9
    probs[0:50, 1] = 0.45          # below default 0.5 -> silent
    turns, diag = sortformer_turns_from_probs(probs, _FS, threshold=0.5)
    # only head 0 clears the bar; head 1 gone -> a single active speaker
    assert diag["n_spk"] == 1
    active_labels = {t[0] for t in turns}
    assert "SPEAKER_00" in active_labels


def test_sortformer_overlap_encoded_via_annotation():
    # heads 0 (0-4.8s) and 1 (3.2-8.0s) overlap over 3.2-4.8s.
    probs = _probs({0: [(0, 60)], 1: [(40, 100)]}, T=100)
    turns, _ = sortformer_turns_from_probs(probs, _FS, threshold=0.5)
    ann = build_sortformer_annotation(turns)
    overlap = ann.get_overlap()
    segs = list(overlap)
    assert len(segs) == 1
    assert segs[0].start == pytest.approx(3.2, abs=1e-6)
    assert segs[0].end == pytest.approx(4.8, abs=1e-6)


def test_sortformer_leak_flags_third_head():
    # head 2 carries ~20% of speech -> n_spk==3 and leak>5% (the 82627719 signal).
    probs = _probs({0: [(0, 100)], 1: [(0, 100)], 2: [(0, 25)]}, T=100)
    _turns, diag = sortformer_turns_from_probs(probs, _FS, threshold=0.5)
    assert diag["n_spk"] == 3
    assert diag["leak"] > 0.05
    assert diag["top2"] == [0, 1]      # the third head is still not selected


# --- Stage dispatch: env-var guard + canned-JSON stub worker ----------------


def test_sortformer_load_fail_loud_without_venv_env(monkeypatch):
    # SCOPE §4: a missing isolated venv is a loud crash at load, never a silent
    # fall-back to pyannote.
    monkeypatch.delenv("SORTFORMER_VENV_PY", raising=False)
    stage = DiarizationStage(DiarizationConfig(backend="sortformer"))
    with pytest.raises(RuntimeError, match="SORTFORMER_VENV_PY"):
        stage.load(torch.device("cpu"))


def test_sortformer_load_signature_distinct_from_pyannote():
    sf = DiarizationStage(DiarizationConfig(backend="sortformer"))
    py = DiarizationStage(DiarizationConfig(backend="pyannote"))
    assert sf.load_signature() == ("sortformer", "nvidia/diar_sortformer_4spk-v1")
    assert sf.load_signature() != py.load_signature()   # a backend swap reloads


def _stub_venv_py(tmp_path, probs_body):
    """Write an executable stub interpreter that ignores the worker path and
    emits canned frame_probs JSON to --out. Mirrors how the real
    $SORTFORMER_VENV_PY is invoked: argv = [worker, --in, IN, --out, OUT,
    --model, MODEL]. Built line-by-line (no textwrap.dedent) so the shebang stays
    at column 0 — `probs_body`'s statements are unindented, which would defeat
    dedent's common-prefix and push the shebang off column 0 (exec-format error)."""
    stub = tmp_path / "stub_venv_python"
    lines = [
        "#!/usr/bin/env python3",
        "import sys, json",
        "a = sys.argv[1:]",
        'out = a[a.index("--out") + 1]',
        *probs_body.splitlines(),
        'json.dump({"frame_probs": probs, "frame_rate_s": 0.08,',
        '           "num_frames": len(probs), "num_heads": 4,',
        '           "model": "stub"}, open(out, "w"))',
    ]
    stub.write_text("\n".join(lines) + "\n")
    stub.chmod(0o755)
    return stub


def test_sortformer_stage_run_with_stub_worker(tmp_path, monkeypatch):
    # heads 0 (0-4.8s) + 1 (3.2-8.0s) active, overlap 3.2-4.8s; heads 2,3 silent.
    body = (
        "T = 100\n"
        "probs = [[0.0, 0.0, 0.0, 0.0] for _ in range(T)]\n"
        "for t in range(0, 60):   probs[t][0] = 0.9\n"
        "for t in range(40, 100): probs[t][1] = 0.9\n"
    )
    stub = _stub_venv_py(tmp_path, body)
    monkeypatch.setenv("SORTFORMER_VENV_PY", str(stub))

    stage = DiarizationStage(DiarizationConfig(backend="sortformer"))
    stage.load(torch.device("cpu"))       # validates the (stub) venv, no model
    ctx = _ctx(np.zeros(16_000, dtype=np.float32))
    stage.run(ctx)

    seg = ctx.diarization.segments_df
    assert list(seg.columns) == ["start", "end", "duration", "speaker"]
    assert set(seg["speaker"]) == {"SPEAKER_00", "SPEAKER_01"}
    ovl = ctx.diarization.overlaps_df
    assert list(ovl.columns) == ["start", "end", "duration"]
    assert len(ovl) == 1
    assert ovl.iloc[0]["start"] == pytest.approx(3.2, abs=1e-3)
    assert ovl.iloc[0]["end"] == pytest.approx(4.8, abs=1e-3)
    assert ctx.diarization.total_duration_s == pytest.approx(1.0)   # 16000/16000


def test_sortformer_stage_leak_warning_logged(tmp_path, monkeypatch):
    # heads 0,1 full + head 2 at ~20% -> the miscount alarm must reach _log.
    body = (
        "T = 100\n"
        "probs = [[0.0, 0.0, 0.0, 0.0] for _ in range(T)]\n"
        "for t in range(0, 100): probs[t][0] = 0.9; probs[t][1] = 0.9\n"
        "for t in range(0, 25):  probs[t][2] = 0.9\n"
    )
    stub = _stub_venv_py(tmp_path, body)
    monkeypatch.setenv("SORTFORMER_VENV_PY", str(stub))

    logged: list[str] = []
    monkeypatch.setattr(
        "asr_pipeline.stages.diarization._log", lambda m: logged.append(m)
    )
    stage = DiarizationStage(DiarizationConfig(backend="sortformer"))
    stage.load(torch.device("cpu"))
    stage.run(_ctx(np.zeros(16_000, dtype=np.float32)))
    assert any("head-miscount signal" in m for m in logged)


def test_sortformer_stage_run_before_load_raises():
    stage = DiarizationStage(DiarizationConfig(backend="sortformer"))
    with pytest.raises(RuntimeError, match="before load"):
        stage.run(_ctx(np.zeros(16_000, dtype=np.float32)))


def test_sortformer_worker_nonzero_exit_fails_loud(tmp_path, monkeypatch):
    # A worker crash is a loud RuntimeError (SCOPE §4), never a silent empty result.
    stub = tmp_path / "boom_python"
    stub.write_text("#!/usr/bin/env python3\nimport sys\nsys.exit(3)\n")
    stub.chmod(0o755)
    monkeypatch.setenv("SORTFORMER_VENV_PY", str(stub))
    stage = DiarizationStage(DiarizationConfig(backend="sortformer"))
    stage.load(torch.device("cpu"))
    with pytest.raises(RuntimeError, match="Sortformer worker failed"):
        stage.run(_ctx(np.zeros(16_000, dtype=np.float32)))


# ===========================================================================
# v4.1 rehabilitation levers (L1 merge / L2 hysteresis / L3 coverage / L4 gate)
# V41_PREREG.md. Pure helpers unit-tested directly; stage wiring with the same
# canned-JSON stub-worker + stub embedder / coverage reference pattern (no GPU,
# no NeMo, no pyannote).
# ===========================================================================


# --- flat binarization -----------------------------------------------------


def test_flat_binarization_byte_identical_on_fixture():
    # Regression guard: head 0 active [0, 4.0)s, head 1 [4.0, 8.0)s → the exact
    # flat turns.
    probs = _probs({0: [(0, 50)], 1: [(50, 100)]}, T=100)
    turns, diag = sortformer_turns_from_probs(probs, _FS, threshold=0.5)
    assert [(l, round(float(a), 4), round(float(b), 4)) for l, a, b in turns] == [
        ("SPEAKER_00", 0.0, 4.0),
        ("SPEAKER_01", 4.0, 8.0),
    ]
    assert diag["top2"] == [0, 1]
    assert diag["n_spk"] == 2
    assert diag["leak"] == 0.0


# --- L1 merge-not-discard --------------------------------------------------


class _StubEmbedder:
    """Deterministic stub: maps a waveform to a point on the unit circle by its
    mean amplitude, so audio value 0.5 vs -0.5 give distinct, comparable
    embeddings. dimension 2, min_num_samples 1."""

    dimension = 2
    min_num_samples = 1

    def __call__(self, wav):
        import math
        v = float(np.asarray(wav).mean())
        v = max(-1.0, min(1.0, v))
        return np.array([[v, math.sqrt(max(0.0, 1.0 - v * v))]], dtype=np.float32)


def _merge_fixture(sr=16_000):
    """Audio + head_runs where head 0 solo = +0.5, head 1 solo = -0.5, surplus
    head 2 has a +0.5 run (→ merge to SPEAKER_00, wide margin) and a 0.0 run
    (ambiguous → unresolved), and head 3 a 0.2 s run (→ too short)."""
    audio = np.zeros(int(9.0 * sr), dtype=np.float32)
    audio[int(0.0 * sr):int(3.0 * sr)] = 0.5     # head 0 solo
    audio[int(3.0 * sr):int(6.0 * sr)] = -0.5    # head 1 solo
    audio[int(6.0 * sr):int(6.6 * sr)] = 0.5     # head 2 run A (→ spk0)
    audio[int(7.0 * sr):int(7.6 * sr)] = 0.0     # head 2 run B (ambiguous)
    audio[int(8.0 * sr):int(8.2 * sr)] = 0.5     # head 3 short run
    head_runs = [
        [(0.0, 3.0)],
        [(3.0, 6.0)],
        [(6.0, 6.6), (7.0, 7.6)],
        [(8.0, 8.2)],
    ]
    return audio, head_runs


def test_merge_above_margin_below_margin_and_short():
    audio, head_runs = _merge_fixture()
    labels = {0: "SPEAKER_00", 1: "SPEAKER_01"}
    merged_turns, stats = _merge_surplus_heads(
        head_runs, [0, 1], labels, audio, 16_000, _StubEmbedder(),
        merge_margin=0.10, speech_tot=12.0,
    )
    # the +0.5 run merges to SPEAKER_00; the 0.0 run is a below-margin discard;
    # the 0.2 s run is skipped as too short.
    assert len(merged_turns) == 1
    assert merged_turns[0][0] == "SPEAKER_00"
    assert merged_turns[0][1] == pytest.approx(6.0)
    assert merged_turns[0][2] == pytest.approx(6.6)
    assert stats["n_merged"] == 1
    assert stats["merged_s"] == pytest.approx(0.6, abs=1e-3)
    assert stats["n_unresolved"] == 1
    assert stats["unresolved_s"] == pytest.approx(0.6, abs=1e-3)
    assert stats["n_short"] == 1
    assert stats["short_s"] == pytest.approx(0.2, abs=1e-3)
    # unresolved leak = unresolved_s / speech_tot (feeds L4).
    assert stats["unresolved_leak"] == pytest.approx(0.6 / 12.0, abs=1e-4)


def test_merge_higher_margin_rejects_the_clear_case():
    # bump the required margin above the 0.5 gap → nothing merges, all unresolved.
    audio, head_runs = _merge_fixture()
    labels = {0: "SPEAKER_00", 1: "SPEAKER_01"}
    merged_turns, stats = _merge_surplus_heads(
        head_runs, [0, 1], labels, audio, 16_000, _StubEmbedder(),
        merge_margin=0.9, speech_tot=12.0,
    )
    assert merged_turns == []
    assert stats["n_merged"] == 0
    assert stats["n_unresolved"] == 2          # both eligible runs now unresolved
    assert stats["n_short"] == 1


def test_merge_no_reference_when_a_top2_speaker_lacks_solo():
    # head 1 fully coincides with head 0 → head 1 has no SOLO frames → no
    # reference → every eligible surplus run falls to unresolved (never a guess).
    audio, head_runs = _merge_fixture()
    head_runs[1] = [(0.0, 3.0)]     # head 1 now overlaps head 0 entirely
    labels = {0: "SPEAKER_00", 1: "SPEAKER_01"}
    merged_turns, stats = _merge_surplus_heads(
        head_runs, [0, 1], labels, audio, 16_000, _StubEmbedder(),
        merge_margin=0.10, speech_tot=12.0,
    )
    assert merged_turns == []
    assert stats["n_merged"] == 0
    assert stats["n_unresolved"] == 2


def _sf_probs_body(active_by_head, T):
    """Build a stub-worker body emitting (T,4) probs with the given active ranges
    (0.9 in range, else 0.0) — mirrors _probs but as source for the stub venv."""
    lines = [f"T = {T}",
             "probs = [[0.0, 0.0, 0.0, 0.0] for _ in range(T)]"]
    for head, ranges in active_by_head.items():
        for lo, hi in ranges:
            lines.append(f"for t in range({lo}, {hi}): probs[t][{head}] = 0.9")
    return "\n".join(lines) + "\n"


def test_sortformer_stage_l1_merge_reassigns_surplus(tmp_path, monkeypatch):
    # Full L1 wiring through the stage: stub worker + stub embedder (injected by
    # monkeypatching build_custom_embedding). head0 [0,10)s, head1 [10,20)s,
    # surplus head2 [20,22)s; audio aligns so head2 embeds to SPEAKER_00.
    body = _sf_probs_body({0: [(0, 125)], 1: [(125, 250)], 2: [(250, 275)]}, T=275)
    stub = _stub_venv_py(tmp_path, body)
    monkeypatch.setenv("SORTFORMER_VENV_PY", str(stub))
    # The L1 merge builds its ECAPA2 embedder via `with_ecapa2`, which calls
    # `build_custom_embedding` in the custom_embeddings module — patch it there.
    monkeypatch.setattr(
        "asr_pipeline.stages.custom_embeddings.build_custom_embedding",
        lambda name, device: _StubEmbedder(),
    )
    sr = 16_000
    audio = np.zeros(int(22.0 * sr), dtype=np.float32)
    audio[int(0.0 * sr):int(10.0 * sr)] = 0.5      # head 0 solo (spk0 ref)
    audio[int(10.0 * sr):int(20.0 * sr)] = -0.5    # head 1 solo (spk1 ref)
    audio[int(20.0 * sr):int(22.0 * sr)] = 0.5     # surplus run → spk0

    cfg = DiarizationConfig(backend="sortformer", sortformer_head_policy="merge")
    stage = DiarizationStage(cfg)
    stage.load(torch.device("cpu"))
    ctx = _ctx(audio)
    stage.run(ctx)

    seg = ctx.diarization.segments_df
    # the surplus run was reassigned to SPEAKER_00 (not discarded).
    merged_rows = seg[(seg["speaker"] == "SPEAKER_00") & (seg["start"] >= 19.9)]
    assert len(merged_rows) == 1
    assert merged_rows.iloc[0]["end"] == pytest.approx(22.0, abs=1e-2)
    # and the merge is recorded in the census diag → metadata.
    assert ctx.diarization_diag["merge"]["n_merged"] == 1
    assert ctx.diarization_diag["head_policy"] == "merge"


