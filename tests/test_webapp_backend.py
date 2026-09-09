"""Backend tests for the showcase webapp (`webapp/`), contract = `webapp/API.md`.

No GPU, no `asr_pipeline` import, no real pipeline: every test injects a
`FakeRunner` through the `Runner` seam. The fake replays a scripted event
sequence and materialises a schema-correct output directory (tiny synthetic
wavs + the JSON files `write_pipeline_outputs` writes), so the assertions are
about the webapp's own behaviour, not about the pipeline's.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from webapp.app import create_app
from webapp.eta import EtaEstimator
from webapp.examples_build import (
    METRIC_KEYS,
    build_manifest,
    compute_metric_set,
    load_scores,
)
from webapp.queue import PipelineRunner, RunnerFailure, job_config
from webapp.render import DEFAULT_BUCKETS, peaks

STAGES = [
    "diarization", "routing", "enhancement", "separation",
    "post_separation_processing", "relabel", "assembly", "transcription",
]


# ---------------------------------------------------------------------------
# Fixtures / fakes
# ---------------------------------------------------------------------------


def write_wav(path: Path, seconds: float = 1.0, sr: int = 16_000, freq: float = 440.0):
    """A tiny deterministic sine wav."""
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(seconds * sr), dtype=np.float32) / sr
    sf.write(str(path), 0.5 * np.sin(2 * np.pi * freq * t), sr)
    return path


def write_fake_outputs(pipeline_dir: Path, *, weak_anchor: bool = False,
                       with_hooks: bool = False) -> None:
    """Materialise a minimal but schema-correct `write_pipeline_outputs` tree."""
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    write_wav(pipeline_dir / "stream_A.wav", 1.0, freq=300)
    write_wav(pipeline_dir / "stream_B.wav", 1.0, freq=600)

    diarization = {
        "turns": [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 0.4},
            {"speaker": "SPEAKER_01", "start": 0.3, "end": 0.9},
        ],
        "total_duration_s": 1.0,
    }
    if with_hooks:                       # hook 3: extended diarization.json
        diarization["overlaps"] = [{"start": 0.3, "end": 0.4, "merged": False}]
    (pipeline_dir / "diarization.json").write_text(json.dumps(diarization))
    (pipeline_dir / "routing.json").write_text(
        json.dumps({"overlap_regions": [{"start": 0.3, "end": 0.4}]})
    )

    for label, text in (("A", "pierwszy mówca"), ("B", "drugi mówca")):
        result = {
            "language": "pl",
            "segments": [{
                "start": 0.1, "end": 0.8, "text": text,
                "words": [{"word": text.split()[0], "start": 0.1,
                           "end": 0.4, "score": 0.9}],
            }],
        }
        (pipeline_dir / f"transcript_{label}.json").write_text(
            json.dumps(result, ensure_ascii=False), encoding="utf-8"
        )
        (pipeline_dir / f"transcript_{label}.txt").write_text(
            f"[  0.10 →   0.80]  {text}\n", encoding="utf-8"
        )
    (pipeline_dir / "annotation.eaf").write_text("<ANNOTATION_DOCUMENT/>")

    metadata = {
        "input_path": "fake.wav",
        "sample_rate": 16_000,
        "speakers": ["SPEAKER_00", "SPEAKER_01"],
        "spk_to_label": {"SPEAKER_00": "A", "SPEAKER_01": "B"},
        "weak_anchor": weak_anchor,
        "total_duration_s": 1.0,
        "n_overlap_regions": 1,
        "n_overlap_separated": 1,
        "stage_timings": [
            {"stage": s, "load_s": 0.1, "run_s": 0.2} for s in STAGES
        ],
        "config": {
            "enhancement": {"enabled": True, "backend": "frcrn_se_16k"},
            "separation": {
                "enabled": True, "separator_backend": "repo",
                "checkpoint_path": "checkpoints/mossformer2/SB/mf2_128k/best.pt",
            },
            "transcription": {
                "enabled": True, "backend": "whisperx", "model_name": "large-v2",
            },
        },
    }
    if with_hooks:                       # hook 1: assembly attribution diagnostics
        metadata["assembly_diag"] = [
            {"region": 0, "pairing": "straight", "cos_sum": 1.23}
        ]
    (pipeline_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
    )


def write_gt_eaf(path: Path, tiers: dict[str, list[str]]) -> Path:
    """A minimal ELAN GT file: one tier per speaker, one annotation per text.

    Only the elements `parse_gt_eaf` (and `asr_pipeline.eval.parse_eaf`) read —
    TIME_SLOTs, TIERs named ``Speaker_<label>``, ALIGNABLE_ANNOTATIONs — with
    one non-overlapping second per utterance.
    """
    slots, tier_xml = [], []
    n = 0
    for label, texts in tiers.items():
        rows = []
        for text in texts:
            n += 1
            start, end = f"ts{2 * n - 1}", f"ts{2 * n}"
            slots.append(f'<TIME_SLOT TIME_SLOT_ID="{start}" TIME_VALUE="{1000 * n}"/>')
            slots.append(f'<TIME_SLOT TIME_SLOT_ID="{end}" TIME_VALUE="{1000 * n + 900}"/>')
            rows.append(
                f'<ANNOTATION><ALIGNABLE_ANNOTATION ANNOTATION_ID="a{n}" '
                f'TIME_SLOT_REF1="{start}" TIME_SLOT_REF2="{end}">'
                f'<ANNOTATION_VALUE>{text}</ANNOTATION_VALUE>'
                f'</ALIGNABLE_ANNOTATION></ANNOTATION>'
            )
        tier_xml.append(
            f'<TIER TIER_ID="Speaker_{label}">' + "".join(rows) + "</TIER>"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?><ANNOTATION_DOCUMENT>'
        "<TIME_ORDER>" + "".join(slots) + "</TIME_ORDER>"
        + "".join(tier_xml) + "</ANNOTATION_DOCUMENT>",
        encoding="utf-8",
    )
    return path


def write_fake_spill(spill_dir: Path, *, truncated: bool = False,
                     enhanced: bool = True) -> None:
    """Per-stage spill as `Pipeline` writes it when `spill_intermediate` is on.

    Note the schema is the spill one — ``segments`` (not ``turns``), with a
    ``duration`` column — which is exactly what the backend has to map.
    `truncated` simulates catching a file mid-write.
    """
    spill_dir.mkdir(parents=True, exist_ok=True)
    diarization = json.dumps({
        "total_duration_s": 1.0,
        "segments": [
            {"start": 0.0, "end": 0.4, "duration": 0.4, "speaker": "SPEAKER_00"},
            {"start": 0.3, "end": 0.9, "duration": 0.6, "speaker": "SPEAKER_01"},
        ],
        "overlaps": [{"start": 0.3, "end": 0.4, "duration": 0.1}],
    })
    if truncated:
        diarization = diarization[: len(diarization) // 2]
    (spill_dir / "diarization.json").write_text(diarization)
    (spill_dir / "overlap_regions.json").write_text(json.dumps({
        "speakers": ["SPEAKER_00", "SPEAKER_01"],
        "overlap_regions": [{"start": 0.3, "end": 0.4, "duration": 0.1}],
    }))
    if enhanced:
        write_wav(spill_dir / "enhanced_full.wav", 1.0, freq=200)


class FakeRunner:
    """Scripted stand-in for `PipelineRunner` — the whole point of the seam."""

    def __init__(self, *, fail: bool = False, emit_progress: bool = False,
                 with_hooks: bool = False, weak_anchor: bool = False,
                 gate: threading.Event | None = None, spill: bool = False,
                 truncated_spill: bool = False, enhanced: bool = True):
        self.fail = fail
        self.emit_progress = emit_progress
        self.with_hooks = with_hooks
        self.weak_anchor = weak_anchor
        self.gate = gate                 # optional block, for queue-order tests
        self.spill = spill or truncated_spill
        self.truncated_spill = truncated_spill
        self.enhanced = enhanced
        self.calls: list[str] = []
        self.events_seen: list[dict] = []

    def stage_names(self):
        return list(STAGES)

    def run(self, job_id, wav_path, out_root, on_event):
        self.calls.append(job_id)
        if self.spill:
            write_fake_spill(
                Path(out_root) / job_id / "spill",
                truncated=self.truncated_spill, enhanced=self.enhanced,
            )
        if self.gate is not None:
            self.gate.wait(timeout=10)
        for stage in STAGES:
            on_event({"event": "stage_start", "stage": stage})
            if self.emit_progress and stage == "separation":
                on_event({"event": "stage_progress", "stage": stage,
                          "done": 1, "total": 3})
            on_event({"event": "stage_end", "stage": stage,
                      "load_s": 0.1, "run_s": 0.2, "wall_s": 0.3})
        if self.fail:
            raise RunnerFailure("RuntimeError", "separator checkpoint missing")
        write_fake_outputs(
            Path(out_root) / job_id / "pipeline",
            weak_anchor=self.weak_anchor, with_hooks=self.with_hooks,
        )


@pytest.fixture
def jobs_root(tmp_path):
    root = tmp_path / "jobs"
    root.mkdir()
    return root


@pytest.fixture
def upload_wav(tmp_path):
    return write_wav(tmp_path / "rozmowa.wav", seconds=1.5)


def make_client(jobs_root, runner, **kwargs):
    app = create_app(
        runner=runner, jobs_root=jobs_root, skip_preflight=True,
        examples_manifest=kwargs.pop("examples_manifest", jobs_root / "none.json"),
        **kwargs,
    )
    return TestClient(app)


def submit(client, wav: Path):
    with open(wav, "rb") as fh:
        response = client.post(
            "/api/jobs", files={"file": (wav.name, fh, "audio/wav")}
        )
    assert response.status_code == 200, response.text
    return response.json()["job_id"]


def wait_for(client, job_id, statuses=("done", "failed"), timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        payload = client.get(f"/api/jobs/{job_id}").json()
        if payload["status"] in statuses:
            return payload
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} stuck in {payload['status']}")


# ---------------------------------------------------------------------------
# Happy path: shape of the JobState JSON
# ---------------------------------------------------------------------------


def test_upload_returns_job_id_and_queued_state(jobs_root, upload_wav):
    """A fresh submission is queued, with the full pending stage chain."""
    runner = FakeRunner(gate=threading.Event())
    client = make_client(jobs_root, runner)
    job_id = submit(client, upload_wav)

    payload = client.get(f"/api/jobs/{job_id}").json()
    assert payload["status"] in ("queued", "running")
    assert payload["filename"] == "rozmowa.wav"
    assert [s["stage"] for s in payload["stages"]] == STAGES
    assert payload["queue_position"] == 0
    assert payload["audio_duration_s"] == pytest.approx(1.5, abs=0.05)
    assert payload["eta_s"] is not None and payload["eta_s"] > 0
    assert payload["result"] is None and payload["error"] is None
    runner.gate.set()


def test_done_job_matches_api_contract(jobs_root, upload_wav):
    """Every field of API.md's JobState is present with the documented shape."""
    client = make_client(jobs_root, FakeRunner(with_hooks=True))
    job_id = submit(client, upload_wav)
    payload = wait_for(client, job_id)

    assert set(payload) == {
        "id", "filename", "status", "queue_position", "submitted_at",
        "audio_duration_s", "stages", "eta_s", "elapsed_s", "warnings",
        "error", "partial", "result",
    }
    assert payload["status"] == "done"
    assert payload["error"] is None
    assert payload["partial"] is None            # terminal -> no partial
    assert payload["elapsed_s"] > 0
    assert isinstance(payload["warnings"], list)

    result = payload["result"]
    assert set(result) == {
        "speakers", "spk_to_label", "weak_anchor", "total_duration_s",
        "n_overlap_regions", "overlap_total_s", "provenance", "diarization",
        "routing", "transcripts", "files", "peaks", "stage_timings",
        "assembly_diag", "diarization_diag",
    }
    assert result["spk_to_label"] == {"SPEAKER_00": "A", "SPEAKER_01": "B"}
    assert result["n_overlap_regions"] == 1
    assert result["overlap_total_s"] == pytest.approx(0.1)
    assert "enh=frcrn_se_16k" in result["provenance"]
    assert "sep=mf2_128k" in result["provenance"]
    assert "asr=whisperx large-v2" in result["provenance"]
    assert result["files"]["mixture"] == f"/api/jobs/{job_id}/files/mixture.wav"
    assert result["files"]["stream_A"].endswith("/files/stream_A.wav")
    assert result["files"]["eaf"].endswith("/files/annotation.eaf")
    assert sorted(result["peaks"]) == ["A", "B", "mixture"]
    assert len(result["peaks"]["mixture"]) == DEFAULT_BUCKETS
    assert len(result["stage_timings"]) == len(STAGES)

    segments = result["transcripts"]["A"]["segments"]
    assert segments[0]["id"] == "A-0000"           # stable ids (design §5.3)
    assert segments[0]["words"][0]["score"] == 0.9
    # hooks landed -> surfaced verbatim
    assert result["assembly_diag"][0]["pairing"] == "straight"
    assert result["diarization"]["overlaps"][0]["start"] == 0.3


def test_hooks_absent_yield_nulls(jobs_root, upload_wav):
    """Without the parallel work stream's hooks the fields are null, not missing."""
    client = make_client(jobs_root, FakeRunner(with_hooks=False))
    payload = wait_for(client, submit(client, upload_wav))
    result = payload["result"]
    assert result["assembly_diag"] is None
    assert result["diarization"]["overlaps"] is None
    assert all(s["progress"] is None for s in payload["stages"])


def test_stage_events_drive_stage_rows(jobs_root, upload_wav):
    """stage_end fills load/run; the stage_progress hook fills progress."""
    client = make_client(jobs_root, FakeRunner(emit_progress=True))
    payload = wait_for(client, submit(client, upload_wav))
    by_stage = {s["stage"]: s for s in payload["stages"]}
    assert all(s["state"] == "done" for s in payload["stages"])
    assert by_stage["diarization"]["load_s"] == pytest.approx(0.1)
    assert by_stage["diarization"]["run_s"] == pytest.approx(0.2)
    assert by_stage["separation"]["progress"] == {"done": 1, "total": 3}


def test_running_state_visible_midflight(jobs_root, upload_wav):
    """A gated runner lets us observe status=running before the outputs exist."""
    gate = threading.Event()
    client = make_client(jobs_root, FakeRunner(gate=gate))
    job_id = submit(client, upload_wav)
    payload = wait_for(client, job_id, statuses=("running",))
    assert payload["result"] is None
    assert payload["elapsed_s"] is not None
    assert payload["eta_s"] is not None
    gate.set()
    assert wait_for(client, job_id)["status"] == "done"


# ---------------------------------------------------------------------------
# Progressive disclosure: `partial` from the per-job spill (API.md v1.1)
# ---------------------------------------------------------------------------


def test_partial_appears_midrun_from_spill(jobs_root, upload_wav):
    """Spill files land -> partial fills in, with segments mapped to `turns`."""
    gate = threading.Event()
    client = make_client(jobs_root, FakeRunner(gate=gate, spill=True))
    job_id = submit(client, upload_wav)
    payload = wait_for(client, job_id, statuses=("running",))
    # The spill is written before the gate, but the poll may still land first.
    deadline = time.monotonic() + 5
    while payload["partial"] is None and time.monotonic() < deadline:
        time.sleep(0.02)
        payload = client.get(f"/api/jobs/{job_id}").json()

    partial = payload["partial"]
    assert set(partial) == {"diarization", "routing"}
    assert partial["diarization"]["turns"] == [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 0.4},
        {"speaker": "SPEAKER_01", "start": 0.3, "end": 0.9},
    ]                                            # `duration` dropped, one shape
    assert partial["diarization"]["overlaps"] == [
        {"start": 0.3, "end": 0.4, "duration": 0.1}
    ]
    assert partial["routing"]["overlap_regions"] == [{"start": 0.3, "end": 0.4}]
    assert payload["result"] is None

    gate.set()
    done = wait_for(client, job_id)
    assert done["partial"] is None                # terminal -> partial cleared
    assert done["result"] is not None


def test_partial_is_null_before_any_spill(jobs_root, upload_wav):
    gate = threading.Event()
    client = make_client(jobs_root, FakeRunner(gate=gate, spill=False))
    job_id = submit(client, upload_wav)
    assert wait_for(client, job_id, statuses=("running",))["partial"] is None
    gate.set()
    wait_for(client, job_id)


def test_partial_tolerates_a_half_written_spill_file(jobs_root, upload_wav):
    """A file caught mid-write reads as absent — a poll must never 500."""
    gate = threading.Event()
    client = make_client(jobs_root, FakeRunner(gate=gate, truncated_spill=True))
    job_id = submit(client, upload_wav)
    payload = wait_for(client, job_id, statuses=("running",))
    deadline = time.monotonic() + 5
    while payload["partial"] is None and time.monotonic() < deadline:
        time.sleep(0.02)
        payload = client.get(f"/api/jobs/{job_id}").json()
    assert payload["partial"]["diarization"] is None      # truncated -> absent
    assert payload["partial"]["routing"]["overlap_regions"]  # the intact one shows
    gate.set()
    wait_for(client, job_id)


def test_queued_job_has_no_partial(jobs_root, tmp_path):
    gate = threading.Event()
    client = make_client(jobs_root, FakeRunner(gate=gate, spill=True))
    first = submit(client, write_wav(tmp_path / "a.wav", 1.0))
    second = submit(client, write_wav(tmp_path / "b.wav", 1.0))
    wait_for(client, first, statuses=("running",))
    assert client.get(f"/api/jobs/{second}").json()["partial"] is None
    gate.set()
    for job_id in (first, second):
        wait_for(client, job_id)


def test_enhanced_full_is_served_from_the_spill_dir(jobs_root, upload_wav):
    client = make_client(jobs_root, FakeRunner(spill=True))
    job_id = submit(client, upload_wav)
    wait_for(client, job_id)
    response = client.get(f"/api/jobs/{job_id}/files/enhanced_full.wav")
    assert response.status_code == 200
    assert response.content


def test_enhanced_full_absent_is_404(jobs_root, upload_wav):
    """No spill (or the enhancement stage never ran) -> 404, not a fake file."""
    client = make_client(jobs_root, FakeRunner(spill=True, enhanced=False))
    job_id = submit(client, upload_wav)
    wait_for(client, job_id)
    assert client.get(f"/api/jobs/{job_id}/files/enhanced_full.wav").status_code == 404


def test_examples_have_no_enhanced_full(jobs_root, tmp_path):
    """Frozen examples have no spill dir — the same whitelist entry just 404s."""
    manifest = _example_manifest(tmp_path, jobs_root)
    client = make_client(jobs_root, FakeRunner(), examples_manifest=manifest)
    assert client.get(
        "/api/examples/demo__seg00/files/enhanced_full.wav"
    ).status_code == 404


def test_result_files_carry_enhanced_full_when_spilled(jobs_root, upload_wav):
    """`files.enhanced_full` appears exactly when the spill file exists."""
    client = make_client(jobs_root, FakeRunner(spill=True))
    job_id = submit(client, upload_wav)
    payload = wait_for(client, job_id)
    assert payload["result"]["files"]["enhanced_full"].endswith(
        f"/api/jobs/{job_id}/files/enhanced_full.wav"
    )

    client2 = make_client(jobs_root, FakeRunner(spill=True, enhanced=False))
    job2 = submit(client2, upload_wav)
    payload2 = wait_for(client2, job2)
    assert "enhanced_full" not in payload2["result"]["files"]


def test_job_page_unknown_is_html_404(jobs_root):
    """A stale /j/ link in a browser gets an HTML page, not raw JSON."""
    client = make_client(jobs_root, FakeRunner())
    response = client.get("/j/nie-ma-takiego")
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("text/html")
    assert "404" in response.text and "Nie ma takiego zadania" in response.text


# ---------------------------------------------------------------------------
# Per-job config copy (shared startup config must stay untouched)
# ---------------------------------------------------------------------------


class FakeConfig:
    """Stand-in for `PipelineConfig`: the two knobs `job_config` touches."""

    def __init__(self):
        self.spill_intermediate = False
        self.artifact_dir = None
        self.nested = {"separation": {"enabled": True}}
        self.validated = 0

    def __post_init__(self):
        self.validated += 1


def test_job_config_enables_spill_on_a_copy(tmp_path):
    shared = FakeConfig()
    cfg = job_config(shared, tmp_path / "job1" / "spill")

    assert cfg is not shared
    assert cfg.spill_intermediate is True
    assert cfg.artifact_dir == str(tmp_path / "job1" / "spill")
    assert cfg.validated == 1                     # re-validated after the change

    # The shared startup config is never mutated — not the scalars, and not the
    # nested structures (deep copy, so a later edit cannot leak either way).
    assert shared.spill_intermediate is False
    assert shared.artifact_dir is None
    assert shared.validated == 0
    cfg.nested["separation"]["enabled"] = False
    assert shared.nested["separation"]["enabled"] is True


def test_job_config_is_per_job(tmp_path):
    shared = FakeConfig()
    first = job_config(shared, tmp_path / "a" / "spill")
    second = job_config(shared, tmp_path / "b" / "spill")
    assert first.artifact_dir != second.artifact_dir
    assert shared.artifact_dir is None


def test_pipeline_runner_holds_the_shared_config_unmodified(tmp_path):
    """The runner keeps the startup config as-is; spill lives on the per-job copy."""
    shared = FakeConfig()
    runner = PipelineRunner(shared)
    assert runner.config is shared
    job_config(runner.config, tmp_path / "spill")
    assert runner.config.spill_intermediate is False
    assert runner.config.artifact_dir is None


def test_weak_anchor_becomes_a_warning(jobs_root, upload_wav):
    client = make_client(jobs_root, FakeRunner(weak_anchor=True))
    payload = wait_for(client, submit(client, upload_wav))
    assert any("weak_anchor" in w for w in payload["warnings"])


# ---------------------------------------------------------------------------
# Failure path
# ---------------------------------------------------------------------------


def test_failed_job_reports_error_and_no_result(jobs_root, upload_wav):
    """Runner raises -> failed + {type, message}, no partial-success framing."""
    client = make_client(jobs_root, FakeRunner(fail=True))
    payload = wait_for(client, submit(client, upload_wav))
    assert payload["status"] == "failed"
    assert payload["result"] is None
    assert payload["error"]["type"] == "RuntimeError"
    assert "checkpoint missing" in payload["error"]["message"]


def test_failure_is_never_retried(jobs_root, upload_wav):
    runner = FakeRunner(fail=True)
    client = make_client(jobs_root, runner)
    job_id = submit(client, upload_wav)
    wait_for(client, job_id)
    time.sleep(0.2)
    assert runner.calls == [job_id]


def test_unknown_job_is_404(jobs_root):
    client = make_client(jobs_root, FakeRunner())
    assert client.get("/api/jobs/nope").status_code == 404
    assert client.get("/api/jobs/nope/log").status_code == 404
    assert client.get("/api/jobs/nope/files/metadata.json").status_code == 404
    assert client.get("/j/nope").status_code == 404


def test_upload_without_file_is_400(jobs_root):
    client = make_client(jobs_root, FakeRunner())
    assert client.post("/api/jobs", data={"note": "x"}).status_code == 400
    assert client.post("/api/jobs").status_code == 400


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_non_wav_upload_is_converted(jobs_root, tmp_path):
    """An mp3 is transcoded to 16 kHz mono WAV before it is ever enqueued."""
    mp3 = tmp_path / "nagranie.mp3"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
         "-ar", "44100", "-ac", "2", str(mp3)],
        check=True,
    )
    client = make_client(jobs_root, FakeRunner())
    with open(mp3, "rb") as fh:
        response = client.post(
            "/api/jobs", files={"file": ("nagranie.mp3", fh, "audio/mpeg")}
        )
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    payload = client.get(f"/api/jobs/{job_id}").json()
    assert payload["filename"] == "nagranie.mp3"
    assert payload["audio_duration_s"] == pytest.approx(2.0, abs=0.05)
    info = sf.info(str(jobs_root / job_id / f"{job_id}.wav"))
    assert (info.samplerate, info.channels) == (16_000, 1)
    wait_for(client, job_id)


def test_unreadable_upload_is_400_and_leaves_no_job(jobs_root):
    """Garbage bytes: ffmpeg cannot convert them, so the submission is refused."""
    client = make_client(jobs_root, FakeRunner())
    response = client.post(
        "/api/jobs", files={"file": ("broken.wav", b"not audio at all", "audio/wav")}
    )
    assert response.status_code == 400
    assert client.get("/api/jobs").json() == []
    assert list(jobs_root.glob("*/")) == []


# ---------------------------------------------------------------------------
# Files endpoint
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", [
    "mixture.wav", "stream_A.wav", "stream_B.wav",
    "transcript_A.txt", "transcript_B.json", "annotation.eaf", "metadata.json",
])
def test_whitelisted_files_are_served(jobs_root, upload_wav, name):
    client = make_client(jobs_root, FakeRunner())
    job_id = submit(client, upload_wav)
    wait_for(client, job_id)
    response = client.get(f"/api/jobs/{job_id}/files/{name}")
    assert response.status_code == 200, name
    assert response.content


@pytest.mark.parametrize("name", [
    "run_meta.json", "routing.json", "diarization.json", "job.json",
    "config.yaml", "stream_A.wav.bak", "transcript_A.md",
])
def test_non_whitelisted_files_are_404(jobs_root, upload_wav, name):
    client = make_client(jobs_root, FakeRunner())
    job_id = submit(client, upload_wav)
    wait_for(client, job_id)
    assert client.get(f"/api/jobs/{job_id}/files/{name}").status_code == 404


@pytest.mark.parametrize("name", [
    "../job.json", "../../etc/passwd", "..%2F..%2Fetc%2Fpasswd",
    "%2e%2e%2fjob.json", "/etc/passwd", "pipeline/metadata.json",
])
def test_path_traversal_is_404(jobs_root, upload_wav, name):
    client = make_client(jobs_root, FakeRunner())
    job_id = submit(client, upload_wav)
    wait_for(client, job_id)
    response = client.get(f"/api/jobs/{job_id}/files/{name}")
    assert response.status_code == 404, name


def test_debug_log_is_copied_and_servable(jobs_root, upload_wav, monkeypatch, tmp_path):
    """The live log is copied next to the outputs and served from the copy."""
    live = tmp_path / "live_debug.log"
    live.write_text("[   0.10s] [io] loading rec.wav\n"
                    "[   1.00s] [diarization] run: WARNING long-recording model swap\n")
    monkeypatch.setenv("ASR_PIPELINE_DEBUG_LOG", str(live))
    client = make_client(jobs_root, FakeRunner())
    job_id = submit(client, upload_wav)
    payload = wait_for(client, job_id)
    assert (jobs_root / job_id / "debug.log").exists()
    assert any("long-recording model swap" in w for w in payload["warnings"])
    assert client.get(f"/api/jobs/{job_id}/files/debug.log").status_code == 200


# ---------------------------------------------------------------------------
# Log tail endpoint
# ---------------------------------------------------------------------------


def test_log_offset_is_incremental(jobs_root, upload_wav, monkeypatch, tmp_path):
    live = tmp_path / "live_debug.log"
    live.write_text("line one\nline two\nline three\n")
    monkeypatch.setenv("ASR_PIPELINE_DEBUG_LOG", str(live))
    client = make_client(jobs_root, FakeRunner())
    job_id = submit(client, upload_wav)
    wait_for(client, job_id)

    first = client.get(f"/api/jobs/{job_id}/log").json()
    assert first["offset"] == 3
    assert first["lines"] == ["line one", "line two", "line three"]

    second = client.get(f"/api/jobs/{job_id}/log?offset=3").json()
    assert second == {"offset": 3, "lines": []}

    partial = client.get(f"/api/jobs/{job_id}/log?offset=2").json()
    assert partial["lines"] == ["line three"]


def test_log_offset_beyond_end_is_clamped(jobs_root, upload_wav):
    client = make_client(jobs_root, FakeRunner())
    job_id = submit(client, upload_wav)
    wait_for(client, job_id)
    payload = client.get(f"/api/jobs/{job_id}/log?offset=9999").json()
    assert payload["lines"] == []


# ---------------------------------------------------------------------------
# Queue behaviour
# ---------------------------------------------------------------------------


def test_queue_is_fifo_with_concurrency_one(jobs_root, tmp_path):
    """Two jobs run one at a time, in submission order."""
    gate = threading.Event()
    runner = FakeRunner(gate=gate)
    client = make_client(jobs_root, runner)
    first = submit(client, write_wav(tmp_path / "a.wav", 1.0))
    second = submit(client, write_wav(tmp_path / "b.wav", 1.0))

    wait_for(client, first, statuses=("running",))
    waiting = client.get(f"/api/jobs/{second}").json()
    assert waiting["status"] == "queued"
    assert waiting["queue_position"] == 0        # nothing queued ahead of it

    gate.set()
    assert wait_for(client, first)["status"] == "done"
    assert wait_for(client, second)["status"] == "done"
    assert runner.calls == [first, second]


def test_queue_position_counts_jobs_ahead(jobs_root, tmp_path):
    gate = threading.Event()
    client = make_client(jobs_root, FakeRunner(gate=gate))
    ids = [submit(client, write_wav(tmp_path / f"{i}.wav", 1.0)) for i in range(3)]
    wait_for(client, ids[0], statuses=("running",))
    assert client.get(f"/api/jobs/{ids[1]}").json()["queue_position"] == 0
    assert client.get(f"/api/jobs/{ids[2]}").json()["queue_position"] == 1
    gate.set()
    for job_id in ids:
        wait_for(client, job_id)


def test_clear_jobs_removes_terminal_jobs_from_registry_and_disk(jobs_root, tmp_path):
    """DELETE /api/jobs drops finished jobs entirely — rows, files, everything."""
    client = make_client(jobs_root, FakeRunner())
    done = submit(client, write_wav(tmp_path / "a.wav", 1.0))
    wait_for(client, done)

    # A second client on the same root rebuilds the registry from disk (so the
    # done job is there) and adds a failed one.
    client = make_client(jobs_root, FakeRunner(fail=True))
    failed = submit(client, write_wav(tmp_path / "b.wav", 1.0))
    wait_for(client, failed)
    assert {r["id"] for r in client.get("/api/jobs").json()} == {done, failed}

    assert client.delete("/api/jobs").json() == {"removed": 2, "skipped": 0}
    assert client.get("/api/jobs").json() == []
    assert client.get(f"/api/jobs/{done}").status_code == 404
    assert not (jobs_root / done).exists()
    assert not (jobs_root / failed).exists()


def test_clear_jobs_is_idempotent(jobs_root, tmp_path):
    client = make_client(jobs_root, FakeRunner())
    wait_for(client, submit(client, write_wav(tmp_path / "a.wav", 1.0)))
    assert client.delete("/api/jobs").json()["removed"] == 1
    assert client.delete("/api/jobs").json() == {"removed": 0, "skipped": 0}


def test_clear_jobs_never_touches_jobs_in_flight(jobs_root, tmp_path):
    """A running job and a queued one survive, and both count as skipped."""
    gate = threading.Event()
    client = make_client(jobs_root, FakeRunner(gate=gate))
    running = submit(client, write_wav(tmp_path / "a.wav", 1.0))
    queued = submit(client, write_wav(tmp_path / "b.wav", 1.0))
    wait_for(client, running, statuses=("running",))

    assert client.delete("/api/jobs").json() == {"removed": 0, "skipped": 2}
    assert (jobs_root / running).exists() and (jobs_root / queued).exists()

    gate.set()
    for job_id in (running, queued):
        assert wait_for(client, job_id)["status"] == "done"
    assert client.delete("/api/jobs").json() == {"removed": 2, "skipped": 0}


def test_recent_jobs_listing(jobs_root, tmp_path):
    client = make_client(jobs_root, FakeRunner())
    first = submit(client, write_wav(tmp_path / "a.wav", 1.0))
    wait_for(client, first)
    second = submit(client, write_wav(tmp_path / "b.wav", 1.0))
    wait_for(client, second)
    rows = client.get("/api/jobs").json()
    assert {r["id"] for r in rows} == {first, second}
    assert all(set(r) == {"id", "filename", "status", "submitted_at",
                          "audio_duration_s"} for r in rows)


# ---------------------------------------------------------------------------
# Registry rebuild from disk
# ---------------------------------------------------------------------------


def test_registry_rebuilt_from_disk(jobs_root, upload_wav):
    """A restart recovers completed jobs from the metadata.json sentinel."""
    client = make_client(jobs_root, FakeRunner())
    job_id = submit(client, upload_wav)
    wait_for(client, job_id)

    fresh = make_client(jobs_root, FakeRunner())      # "restart"
    payload = fresh.get(f"/api/jobs/{job_id}").json()
    assert payload["status"] == "done"
    assert payload["filename"] == "rozmowa.wav"
    assert payload["result"]["files"]["mixture"].startswith(f"/api/jobs/{job_id}/")
    assert [s["stage"] for s in payload["stages"]] == STAGES
    assert fresh.get(f"/api/jobs/{job_id}/files/stream_A.wav").status_code == 200


def test_registry_rebuild_recovers_failures(jobs_root, upload_wav):
    client = make_client(jobs_root, FakeRunner(fail=True))
    job_id = submit(client, upload_wav)
    wait_for(client, job_id)

    fresh = make_client(jobs_root, FakeRunner())
    payload = fresh.get(f"/api/jobs/{job_id}").json()
    assert payload["status"] == "failed"
    assert payload["error"]["type"] == "RuntimeError"


def test_registry_rebuild_marks_interrupted_jobs_failed(jobs_root):
    """A job dir with a sidecar but no outputs was killed mid-run — say so."""
    job_dir = jobs_root / "abc123"
    job_dir.mkdir(parents=True)
    (job_dir / "job.json").write_text(json.dumps({
        "id": "abc123", "filename": "x.wav",
        "submitted_at": "2026-07-28T10:00:00+00:00", "audio_duration_s": 12.0,
    }))
    client = make_client(jobs_root, FakeRunner())
    payload = client.get("/api/jobs/abc123").json()
    assert payload["status"] == "failed"
    assert payload["error"]["type"] == "InterruptedRun"


# ---------------------------------------------------------------------------
# Examples gallery
# ---------------------------------------------------------------------------


EXAMPLE_METRICS = {key: float(i) for i, key in enumerate(METRIC_KEYS)}


def _example_manifest(tmp_path, jobs_root) -> Path:
    frag = tmp_path / "frag" / "demo__seg00"
    write_wav(frag / "demo__seg00.wav", 1.0)
    write_fake_outputs(frag / "sweep" / "v41_merge")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "generated_at": "2026-07-28T00:00:00+00:00",
        "arm": "v41_merge",
        "examples": [{
            "id": "demo__seg00", "title": "demo__seg00", "duration_s": 1.0,
            "split": "dev", "n_overlap_regions": 1, "cpwer": None, "cpcer": None,
            "stratum": "MID",
            "metrics": dict(EXAMPLE_METRICS),
            "gt": {"A": [{"start": 0.1, "end": 0.8, "text": "referencja"}]},
            "gt_swapped": True,
            "pipeline_dir": str(frag / "sweep" / "v41_merge"),
            "mixture_path": str(frag / "demo__seg00.wav"),
        }],
    }))
    return manifest


def test_examples_listing_and_files(jobs_root, tmp_path):
    manifest = _example_manifest(tmp_path, jobs_root)
    client = make_client(jobs_root, FakeRunner(), examples_manifest=manifest)
    rows = client.get("/api/examples").json()
    assert len(rows) == 1
    row = rows[0]
    assert row["split"] == "dev" and row["gt_available"] is True
    assert row["cpwer"] is None
    assert row["job_like"]["files"]["mixture"] == \
        "/api/examples/demo__seg00/files/mixture.wav"
    assert len(row["job_like"]["peaks"]["A"]) == DEFAULT_BUCKETS
    assert client.get("/api/examples/demo__seg00/files/stream_A.wav").status_code == 200
    assert client.get("/api/examples/demo__seg00/files/run_meta.json").status_code == 404
    assert client.get("/api/examples/nope/files/mixture.wav").status_code == 404


def test_examples_light_and_ids_filters(jobs_root, tmp_path):
    """The optional knobs; the bare call stays exactly the contract."""
    manifest = _example_manifest(tmp_path, jobs_root)
    client = make_client(jobs_root, FakeRunner(), examples_manifest=manifest)
    light = client.get("/api/examples?light=1").json()
    assert light[0]["job_like"] is None and light[0]["gt_available"] is True
    # light also nulls the GT payload itself — the gallery needs only the flag.
    assert light[0]["gt"] is None
    full = client.get("/api/examples?ids=demo__seg00").json()
    assert full[0]["job_like"] and full[0]["gt"]
    assert client.get("/api/examples?ids=other").json() == []


def test_examples_carry_stratum_metrics_and_swap_flag_in_both_modes(jobs_root, tmp_path):
    """API.md v1.2 draft: the three small additions ride along even in light mode."""
    manifest = _example_manifest(tmp_path, jobs_root)
    client = make_client(jobs_root, FakeRunner(), examples_manifest=manifest)
    for url in ("/api/examples?light=1", "/api/examples?ids=demo__seg00"):
        row = client.get(url).json()[0]
        assert row["stratum"] == "MID", url
        assert row["gt_swapped"] is True, url
        assert row["metrics"] == EXAMPLE_METRICS, url
        assert set(row["metrics"]) == set(METRIC_KEYS), url


def test_missing_examples_manifest_is_an_empty_gallery(jobs_root):
    client = make_client(jobs_root, FakeRunner())
    assert client.get("/api/examples").json() == []


def test_rerun_example_creates_a_plain_job(jobs_root, tmp_path):
    """'uruchom ponownie' submits the mixture as a normal job — no GT, no scores."""
    manifest = _example_manifest(tmp_path, jobs_root)
    client = make_client(jobs_root, FakeRunner(), examples_manifest=manifest)
    response = client.post("/api/jobs", data={"source_example": "demo__seg00"})
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    payload = wait_for(client, job_id)
    assert payload["status"] == "done"
    assert payload["filename"] == "demo__seg00.wav"
    assert "gt" not in payload["result"]

    missing = client.post("/api/jobs", data={"source_example": "nieznany"})
    assert missing.status_code == 400


# ---------------------------------------------------------------------------
# examples_build: score join
# ---------------------------------------------------------------------------


# The frozen rescore sheets' header verbatim (`dump_per_fragment` in
# scripts/rescore_stratified.py), so the fixtures exercise the real layout.
SCORES_COLUMNS = [
    "frag_id", "recid", "config", "stratum", "composite",
    "cp_wer", "cp_cer", "cp_err", "cp_len", "cer_err", "cer_len",
    "orc_wer", "mimo_wer", "orc_cer", "mix_mimo_wer", "mix_mimo_cer",
    "purity_pct",
]


def _scores_row(frag: str, config: str = "v41_merge", **cells) -> str:
    """One sheet row; every column not named comes out blank, as the real ones do."""
    values = {"frag_id": frag, "recid": frag.split("__")[0], "config": config}
    values.update(cells)
    return ",".join(str(values.get(c, "")) for c in SCORES_COLUMNS) + "\n"


def _scores_csv(path: Path, rows: str) -> Path:
    path.write_text(",".join(SCORES_COLUMNS) + "\n" + rows, encoding="utf-8")
    return path


# One fully populated v41_merge row, mirroring the real sheet's 005cba37__seg00.
_FULL_ROW_CELLS = dict(
    stratum="HIGH", composite="1.643", cp_wer="54.0", cp_cer="34.44",
    cp_err="81", cp_len="150", cer_err="280", cer_len="813",
    orc_wer="52.67", mimo_wer="48.67", orc_cer="36.04",
    mix_mimo_wer="61.33", mix_mimo_cer="51.23",
)


def test_load_scores_keeps_only_the_frozen_arm(tmp_path):
    """Rows for other sweep arms must never leak into the gallery's numbers."""
    csv_path = _scores_csv(tmp_path / "rescore.csv", (
        _scores_row("aaa__seg00", stratum="HIGH", cp_wer="54.0", cp_cer="34.44")
        + _scores_row("aaa__seg00", "v3_phraseloop", stratum="HIGH",
                      cp_wer="99.9", cp_cer="88.8")
        + _scores_row("bbb__seg00", stratum="LOW", cp_wer="11.95", cp_cer="5.87")
    ))
    scores = load_scores([csv_path], "v41_merge")
    assert sorted(scores) == ["aaa__seg00", "bbb__seg00"]
    assert scores["aaa__seg00"]["cpwer"] == 54.0        # raw, not rounded
    assert scores["aaa__seg00"]["cpcer"] == 34.44
    assert scores["aaa__seg00"]["stratum"] == "HIGH"
    assert scores["bbb__seg00"]["cpwer"] == 11.95


def test_load_scores_reads_every_metric_column(tmp_path):
    """The sheet columns land in `metrics` unrounded — rounding happens on merge."""
    csv_path = _scores_csv(tmp_path / "rescore.csv",
                           _scores_row("aaa__seg00", **_FULL_ROW_CELLS))
    metrics = load_scores([csv_path], "v41_merge")["aaa__seg00"]["metrics"]
    assert metrics == {
        "cpwer": 54.0, "cpcer": 34.44, "orcwer": 52.67, "mimower": 48.67,
        "orccer": 36.04, "floor_mimower": 61.33, "floor_mimocer": 51.23,
    }


def test_load_scores_merges_sheets_and_survives_a_missing_one(tmp_path):
    dev = _scores_csv(tmp_path / "dev.csv",
                      _scores_row("aaa__seg00", stratum="MID", cp_wer="14.2",
                                  cp_cer="9.53"))
    test = _scores_csv(tmp_path / "test.csv",
                       _scores_row("bbb__seg00", stratum="LOW", cp_wer="11.9",
                                   cp_cer="5.8"))
    scores = load_scores([dev, test, tmp_path / "absent.csv"], "v41_merge")
    assert sorted(scores) == ["aaa__seg00", "bbb__seg00"]


def test_load_scores_handles_blank_cells(tmp_path):
    csv_path = _scores_csv(tmp_path / "r.csv", _scores_row("aaa__seg00", stratum="MID"))
    row = load_scores([csv_path], "v41_merge")["aaa__seg00"]
    assert row["cpwer"] is None and row["cpcer"] is None
    assert row["stratum"] == "MID"
    assert set(row["metrics"]) == {
        "cpwer", "cpcer", "orcwer", "mimower", "orccer",
        "floor_mimower", "floor_mimocer",
    }
    assert all(v is None for v in row["metrics"].values())


def test_build_manifest_joins_scores_onto_examples(tmp_path):
    root = tmp_path / "frags"
    for frag in ("aaa__seg00", "bbb__seg00"):
        write_wav(root / frag / f"{frag}.wav", 1.0)
        write_fake_outputs(root / frag / "sweep" / "v41_merge")
    csv_path = _scores_csv(tmp_path / "rescore.csv",
                           _scores_row("aaa__seg00", **_FULL_ROW_CELLS))

    manifest = build_manifest(root, "v41_merge", scores_csvs=[csv_path])
    rows = {r["id"]: r for r in manifest["examples"]}
    assert rows["aaa__seg00"]["cpwer"] == 54.0
    assert rows["aaa__seg00"]["cpcer"] == 34.44
    # A fragment absent from the sheets keeps null scores rather than a guess.
    assert rows["bbb__seg00"]["cpwer"] is None
    assert rows["bbb__seg00"]["cpcer"] is None


def test_build_manifest_carries_stratum_and_the_metric_schema(tmp_path):
    """The row shape the frontend is built against (API.md v1.2 draft)."""
    root = tmp_path / "frags"
    for frag in ("aaa__seg00", "bbb__seg00"):
        write_wav(root / frag / f"{frag}.wav", 1.0)
        write_fake_outputs(root / frag / "sweep" / "v41_merge")
    csv_path = _scores_csv(tmp_path / "rescore.csv",
                           _scores_row("aaa__seg00", **_FULL_ROW_CELLS))

    rows = {r["id"]: r
            for r in build_manifest(root, "v41_merge",
                                    scores_csvs=[csv_path])["examples"]}
    scored = rows["aaa__seg00"]
    assert scored["stratum"] == "HIGH"
    assert set(scored["metrics"]) == set(METRIC_KEYS)
    # Sheet columns win and are rounded to one decimal.
    assert scored["metrics"]["cpwer"] == 54.0
    assert scored["metrics"]["cpcer"] == 34.4
    assert scored["metrics"]["orcwer"] == 52.7
    assert scored["metrics"]["floor_mimocer"] == 51.2
    # attr_gap = cpWER − MIMO-WER, derived from the two shown values.
    assert scored["metrics"]["attr_gap"] == pytest.approx(54.0 - 48.7)
    # Nothing the sheet lacks is invented: these fixtures have no GT EAF, so the
    # recomputation cannot run and those entries stay null.
    assert scored["metrics"]["tcpwer"] is None
    assert scored["metrics"]["floor_orcwer"] is None
    # A fragment with neither sheet row nor computable metrics carries no block.
    assert rows["bbb__seg00"]["stratum"] is None
    assert rows["bbb__seg00"]["metrics"] is None


# ---------------------------------------------------------------------------
# examples_build: GT speaker-swap detection (display alignment only)
# ---------------------------------------------------------------------------


def _swap_fixture(tmp_path, tiers: dict) -> Path:
    """One fragment whose pipeline says A="pierwszy mówca", B="drugi mówca"."""
    root = tmp_path / "frags"
    frag = root / "ccc__seg00"
    write_wav(frag / "ccc__seg00.wav", 1.0)
    write_fake_outputs(frag / "sweep" / "v41_merge")
    write_gt_eaf(frag / "annotation.eaf", tiers)
    return root


def test_gt_tiers_are_swapped_when_they_match_crossed(tmp_path):
    """GT tier B matching pipeline A -> tiers relabelled, flag set."""
    root = _swap_fixture(tmp_path, {"A": ["drugi mówca"], "B": ["pierwszy mówca"]})
    row = build_manifest(root, "v41_merge", scores_csvs=[])["examples"][0]
    assert row["gt_swapped"] is True
    assert row["gt"]["A"][0]["text"] == "pierwszy mówca"    # now beside stream A
    assert row["gt"]["B"][0]["text"] == "drugi mówca"


def test_gt_tiers_are_left_alone_when_they_match_straight(tmp_path):
    root = _swap_fixture(tmp_path, {"A": ["pierwszy mówca"], "B": ["drugi mówca"]})
    row = build_manifest(root, "v41_merge", scores_csvs=[])["examples"][0]
    assert row["gt_swapped"] is False
    assert row["gt"]["A"][0]["text"] == "pierwszy mówca"


def test_gt_swap_flag_is_false_without_gt(tmp_path):
    root = tmp_path / "frags"
    write_wav(root / "ddd__seg00" / "ddd__seg00.wav", 1.0)
    write_fake_outputs(root / "ddd__seg00" / "sweep" / "v41_merge")
    row = build_manifest(root, "v41_merge", scores_csvs=[])["examples"][0]
    assert row["gt"] is None and row["gt_swapped"] is False


# ---------------------------------------------------------------------------
# examples_build: recomputing what the frozen sheets do not carry
# ---------------------------------------------------------------------------


def test_compute_metric_set_uses_the_frozen_eval_code(tmp_path):
    """The recomputed entries come from asr_pipeline.eval, not a local re-write.

    Deliberately the only test that touches the eval chain: everything else in
    this file must stay import-free of meeteval/torch, which is why the
    computation is lazy and skipped whenever a fragment has no GT EAF.
    """
    pytest.importorskip("meeteval")
    frag = tmp_path / "eee__seg00"
    pipeline_dir = frag / "sweep" / "v41_merge"
    write_fake_outputs(pipeline_dir)
    write_gt_eaf(frag / "annotation.eaf",
                 {"A": ["pierwszy mówca"], "B": ["drugi mówca"]})
    (pipeline_dir / "transcript_mixture.txt").write_text(
        "[  0.10 →   1.60]  pierwszy mówca drugi mówca\n", encoding="utf-8"
    )

    metrics = compute_metric_set(frag, "v41_merge")
    assert set(metrics) == set(METRIC_KEYS)
    # Hypothesis == reference, so every rate is 0 and every entry is populated.
    for key in METRIC_KEYS:
        if key == "attr_gap":
            continue                       # derived in merge_metrics, not here
        assert metrics[key] == 0.0, key
    assert metrics["attr_gap"] is None


def test_compute_metric_set_is_none_without_gt_or_transcripts(tmp_path):
    """No EAF (or no transcripts) -> nothing to score, and no eval import."""
    frag = tmp_path / "fff__seg00"
    write_fake_outputs(frag / "sweep" / "v41_merge")
    assert compute_metric_set(frag, "v41_merge") is None
    write_gt_eaf(frag / "annotation.eaf", {"A": ["cokolwiek"]})
    assert compute_metric_set(frag, "v41_nonexistent") is None


# ---------------------------------------------------------------------------
# HTML shells
# ---------------------------------------------------------------------------


def test_html_routes_render(jobs_root, upload_wav):
    client = make_client(jobs_root, FakeRunner())
    job_id = submit(client, upload_wav)
    assert client.get("/").status_code == 200
    assert client.get("/examples").status_code == 200
    page = client.get(f"/j/{job_id}")
    assert page.status_code == 200
    assert f'data-job-id="{job_id}"' in page.text
    assert f'data-poll-url="/api/jobs/{job_id}"' in page.text


# ---------------------------------------------------------------------------
# render.peaks
# ---------------------------------------------------------------------------


def test_peaks_shape_and_range(tmp_path):
    wav = write_wav(tmp_path / "tone.wav", seconds=2.0)
    values = peaks(wav)
    assert DEFAULT_BUCKETS == 2400                # timeline zooms to 8x (API.md v1.2)
    assert len(values) == DEFAULT_BUCKETS
    assert all(isinstance(v, int) and 0 <= v <= 100 for v in values)
    assert max(values) == 100                     # normalised to the stream peak


def test_peaks_tracks_amplitude(tmp_path):
    """A ramp's envelope must rise: bucket values follow the signal."""
    sr = 16_000
    x = (np.linspace(0.0, 1.0, sr, dtype=np.float32)
         * np.sin(2 * np.pi * 440 * np.arange(sr, dtype=np.float32) / sr))
    path = tmp_path / "ramp.wav"
    sf.write(str(path), x, sr)
    values = peaks(path, buckets=100)
    assert values[0] < values[50] < values[-1]


def test_peaks_on_missing_file_is_zeros(tmp_path):
    assert peaks(tmp_path / "nope.wav", buckets=16) == [0] * 16


def test_peaks_rejects_bad_bucket_count(tmp_path):
    with pytest.raises(ValueError):
        peaks(write_wav(tmp_path / "t.wav", 0.2), buckets=0)


# ---------------------------------------------------------------------------
# ETA estimator
# ---------------------------------------------------------------------------


def test_eta_is_monotonic_as_stages_complete(tmp_path):
    est = EtaEstimator(tmp_path / "timings.jsonl")
    previous = est.estimate_remaining(STAGES, 90.0, completed=[])
    assert previous is not None and previous > 0
    for i in range(1, len(STAGES) + 1):
        current = est.estimate_remaining(STAGES, 90.0, completed=STAGES[:i])
        assert current < previous
        previous = current
    assert previous == 0.0


def test_eta_scales_with_duration(tmp_path):
    est = EtaEstimator(tmp_path / "timings.jsonl")
    short = est.estimate_total(STAGES, 30.0)
    long = est.estimate_total(STAGES, 600.0)
    assert long > short > 0


def test_eta_is_none_without_duration(tmp_path):
    est = EtaEstimator(tmp_path / "timings.jsonl")
    assert est.estimate_total(STAGES, None) is None


def test_eta_learns_and_persists(tmp_path):
    """One recorded job replaces the seed and survives a restart."""
    path = tmp_path / "timings.jsonl"
    est = EtaEstimator(path)
    seeded = est.estimate_total(["transcription"], 90.0)
    est.record(
        duration_s=90.0,
        stage_timings=[{"stage": "transcription", "load_s": 60.0, "run_s": 90.0}],
        overlap_s=9.0, n_overlap_regions=10,
    )
    learned = est.estimate_total(["transcription"], 90.0)
    assert learned == pytest.approx(150.0)
    assert learned != pytest.approx(seeded)
    assert path.exists()

    reloaded = EtaEstimator(path)
    assert reloaded.estimate_total(["transcription"], 90.0) == pytest.approx(150.0)


def test_eta_uses_overlap_for_overlap_scaled_stages(tmp_path):
    """Separation cost tracks overlap seconds, not recording length."""
    est = EtaEstimator(tmp_path / "timings.jsonl")
    little = est.estimate_total(["separation"], 90.0, overlap_s=1.0)
    lots = est.estimate_total(["separation"], 90.0, overlap_s=45.0)
    assert lots > little
    # The region count (all the stage_progress hook exposes) is a usable proxy.
    by_count = est.estimate_total(["separation"], 90.0, n_overlap_regions=50)
    assert by_count > little
