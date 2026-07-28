"""Job registry, the single worker thread, and the `Runner` seam.

Concurrency is **1 by physics** (one model on the GPU at a time — the pipeline's
phase-major invariant), so the whole scheduler is one `queue.Queue` drained by
one daemon thread, FIFO, with a visible queue position. Everything the HTTP
layer needs sits on `JobService`.

The pipeline-facing call is isolated behind the `Runner` protocol:

    runner.stage_names()                                  -> list[str]
    runner.run(job_id, wav_path, out_root, on_event)      -> None (raises on failure)

`PipelineRunner` is the real implementation (`asr_pipeline.batch.run_batch`);
tests inject a fake, which is what keeps the test suite GPU-free and free of any
pipeline import.

Failure policy (SCOPE §4.2, design §3): the pipeline keeps failing loudly
underneath; the webapp catches *per job*, records ``{type, message}``, shows an
error card, and **never retries**. A failed job has no ``result`` — no
partial-success framing.
"""

from __future__ import annotations

import copy
import json
import queue as _queue
import shutil
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Protocol

from webapp.eta import EtaEstimator
from webapp.render import build_result

# Terminal states a rebuilt-from-disk job can be in.
STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"

# Output subdirectory `run_batch` writes under `<jobs_root>/<job_id>/`.
PIPELINE_SUBDIR = "pipeline"

# Per-stage intermediate spill directory, also under `<jobs_root>/<job_id>/`.
# It is what `partial` (progressive disclosure) and the `enhanced_full.wav`
# download are read from — API.md v1.1.
SPILL_SUBDIR = "spill"

# Debug-log lines carrying these markers are surfaced as user-visible warnings
# (SCOPE §4.3). "WARN" catches both `[warn]` and `WARNING`; the long-recording
# Sortformer model swap is listed explicitly so a rephrasing of its (currently
# WARNING-prefixed) message cannot silently drop it.
_WARNING_MARKERS = ("warn", "long-recording model swap")


def debug_log_path() -> Path:
    """The live pipeline debug log (one run's worth; truncated at load_audio).

    Read from ``$ASR_PIPELINE_DEBUG_LOG`` on every call rather than cached at
    import, so a test (or an operator) can repoint it without reimporting.
    """
    import os

    return Path(os.environ.get("ASR_PIPELINE_DEBUG_LOG", "/tmp/asr_pipeline_debug.log"))


# ---------------------------------------------------------------------------
# Runner seam
# ---------------------------------------------------------------------------


class RunnerFailure(Exception):
    """A recording failed inside the runner. Carries the pipeline's own type name."""

    def __init__(self, exc_type: str, message: str) -> None:
        super().__init__(f"{exc_type}: {message}")
        self.exc_type = exc_type
        self.message = message


class Runner(Protocol):
    """What the worker needs from the pipeline. The one seam tests replace."""

    def stage_names(self) -> List[str]:
        """Enabled stage names in pipeline execution order."""

    def run(
        self,
        job_id: str,
        wav_path: Path,
        out_root: Path,
        on_event: Callable[[dict], None],
    ) -> None:
        """Process one recording, writing ``<out_root>/<job_id>/pipeline/``.

        Raises on failure (`RunnerFailure` for a pipeline-reported failure).
        """


def job_config(config, spill_dir: Path):
    """A per-job **copy** of the shared config, with spilling switched on.

    Every job runs with ``spill_intermediate: true`` and
    ``artifact_dir = <job_dir>/spill`` (API.md v1.1): the spill is where the
    progressive-disclosure `partial` payload and the ``enhanced_full.wav``
    download come from.

    The startup config object is **never mutated** — it is shared by every job
    and by whatever else holds a reference to it, so each job gets a
    `copy.deepcopy` (the config is a tree of plain nested dataclasses) and the
    copy is re-validated through ``__post_init__``, exactly as the CLI does when
    ``--output`` turns spilling on (`asr_pipeline/__main__.py` `_run_command`).
    """
    cfg = copy.deepcopy(config)
    cfg.spill_intermediate = True
    cfg.artifact_dir = str(spill_dir)
    cfg.__post_init__()      # re-validate now that the spill settings changed
    return cfg


class PipelineRunner:
    """The real runner: one `run_batch` call per job.

    `run_batch` gives us per-item failure isolation, the ``failures.csv`` record,
    and the single GPU-teardown block — so the webapp adds no pipeline-lifecycle
    logic of its own. Imports are deferred to call time so that constructing the
    app object (and importing this module) costs nothing.

    ``self.config`` is the shared, immutable startup config; each `run` builds
    its own spill-enabled copy via `job_config`.
    """

    def __init__(self, config) -> None:
        self.config = config

    def stage_names(self) -> List[str]:
        """Enabled stages read from the pipeline itself, not hardcoded.

        Constructing a `Pipeline` builds the stage objects only (no model load,
        no GPU allocation), so this is cheap and always agrees with what will
        actually run.
        """
        from asr_pipeline.pipeline import Pipeline

        return [s.name for s in Pipeline(self.config).stages if s.enabled]

    def run(
        self,
        job_id: str,
        wav_path: Path,
        out_root: Path,
        on_event: Callable[[dict], None],
    ) -> None:
        from asr_pipeline.batch import run_batch

        cfg = job_config(self.config, Path(out_root) / job_id / SPILL_SUBDIR)
        report = run_batch(
            cfg,
            [(job_id, str(wav_path))],
            out_root=out_root,
            subdir_name=PIPELINE_SUBDIR,
            on_event=on_event,
            skip_existing=False,
        )
        if report.failed:
            exc_type = report.failed[0][1]
            raise RunnerFailure(exc_type, _failure_message(out_root, job_id, exc_type))
        if job_id in report.skipped:
            # The only skip reasons are "input missing" and "already complete";
            # skip_existing=False rules out the second, so this is a missing file.
            raise RunnerFailure("FileNotFoundError", f"input audio missing: {wav_path}")


def _failure_message(out_root: Path, job_id: str, fallback: str) -> str:
    """Pull this job's one-line traceback summary out of ``failures.csv``.

    `run_batch` rewrites the file per call and concurrency is 1, so the row for
    `job_id` is this run's. Falls back to the exception type name if the file is
    unreadable — the type is still honest, just less specific.
    """
    import csv

    path = Path(out_root) / "failures.csv"
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row.get("recording_id") == job_id:
                    return row.get("traceback_summary") or fallback
    except (OSError, ValueError):
        pass
    return fallback


# ---------------------------------------------------------------------------
# Job state
# ---------------------------------------------------------------------------


@dataclass
class StageState:
    """One row of the linear stage chain (design §5.2)."""

    stage: str
    state: str = "pending"          # "pending" | "running" | "done"
    load_s: Optional[float] = None
    run_s: Optional[float] = None
    progress: Optional[dict] = None  # {"done": int, "total": int} — hook 2

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "state": self.state,
            "load_s": self.load_s,
            "run_s": self.run_s,
            "progress": self.progress,
        }


@dataclass
class JobState:
    """Everything known about one submitted recording."""

    id: str
    filename: str
    submitted_at: str
    audio_duration_s: Optional[float] = None
    status: str = STATUS_QUEUED
    stages: List[StageState] = field(default_factory=list)
    error: Optional[dict] = None
    result: Optional[dict] = None
    started_at: Optional[float] = None      # monotonic clock
    finished_at: Optional[float] = None
    # Overlap-region count learned mid-run from the stage_progress hook; feeds
    # the overlap-aware ETA refinement.
    n_overlap_regions: Optional[int] = None
    stage_timings: List[dict] = field(default_factory=list)
    _warnings_cache: Optional[List[str]] = None

    def stage(self, name: str) -> StageState:
        """The row for `name`, appended if the config produced an unexpected stage."""
        for s in self.stages:
            if s.stage == name:
                return s
        row = StageState(stage=name)
        self.stages.append(row)
        return row

    @property
    def completed_stages(self) -> List[str]:
        return [s.stage for s in self.stages if s.state == "done"]

    @property
    def elapsed_s(self) -> Optional[float]:
        if self.started_at is None:
            return None
        end = self.finished_at if self.finished_at is not None else time.monotonic()
        return end - self.started_at


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# The service
# ---------------------------------------------------------------------------


class JobService:
    """Registry + FIFO queue + the one worker thread.

    Owns the on-disk job tree (``<jobs_root>/<job_id>/``), which is
    eval-tree-shaped so `run_batch` writes straight into it:

        <jobs_root>/<job_id>/<job_id>.wav      the (converted) input
        <jobs_root>/<job_id>/job.json          webapp sidecar: name, time, duration
        <jobs_root>/<job_id>/debug.log         the run's debug log, copied after the run
        <jobs_root>/<job_id>/error.json        failure record (failed jobs only)
        <jobs_root>/<job_id>/pipeline/...      what write_pipeline_outputs wrote
    """

    def __init__(self, jobs_root: Path, runner: Runner) -> None:
        self.jobs_root = Path(jobs_root).expanduser()
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self.runner = runner
        self.estimator = EtaEstimator(self.jobs_root / "timings.jsonl")
        self.stage_names: List[str] = list(runner.stage_names())
        self._jobs: Dict[str, JobState] = {}
        self._order: List[str] = []          # submission order (FIFO bookkeeping)
        self._lock = threading.RLock()
        self._queue: "_queue.Queue[Optional[str]]" = _queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._current: Optional[str] = None

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        """Start the worker thread (idempotent)."""
        if self._worker is not None and self._worker.is_alive():
            return
        self._worker = threading.Thread(
            target=self._work, name="webapp-pipeline-worker", daemon=True
        )
        self._worker.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Ask the worker to exit after the current job (best effort)."""
        self._queue.put(None)
        if self._worker is not None:
            self._worker.join(timeout=timeout)

    def wait_idle(self, timeout: float = 60.0) -> bool:
        """Block until the queue is drained. Test/CLI helper; unused by routes."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                pending = any(
                    j.status in (STATUS_QUEUED, STATUS_RUNNING) for j in self._jobs.values()
                )
            if not pending:
                return True
            time.sleep(0.02)
        return False

    # -- paths -------------------------------------------------------------
    def job_dir(self, job_id: str) -> Path:
        return self.jobs_root / job_id

    def mixture_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / f"{job_id}.wav"

    def pipeline_dir(self, job_id: str) -> Path:
        return self.job_dir(job_id) / PIPELINE_SUBDIR

    def spill_dir(self, job_id: str) -> Path:
        return self.job_dir(job_id) / SPILL_SUBDIR

    def log_copy_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "debug.log"

    # -- registry ----------------------------------------------------------
    def get(self, job_id: str) -> Optional[JobState]:
        with self._lock:
            return self._jobs.get(job_id)

    def register(self, job: JobState) -> None:
        """Add a job to the registry, seeding its stage chain if it has none.

        A queued job already shows the full pending chain — the stage list is the
        config's enabled stages in pipeline order, discovered once at startup
        through the Runner seam.
        """
        with self._lock:
            if not job.stages and job.status in (STATUS_QUEUED, STATUS_RUNNING):
                job.stages = [StageState(stage=n) for n in self.stage_names]
            self._jobs[job.id] = job
            if job.id not in self._order:
                self._order.append(job.id)

    def enqueue(self, job_id: str) -> None:
        self._queue.put(job_id)

    def recent(self, limit: int = 20) -> List[dict]:
        """Compact rows for the home page's recent-jobs list, newest first."""
        with self._lock:
            jobs = [self._jobs[j] for j in self._order if j in self._jobs]
        jobs.sort(key=lambda j: j.submitted_at, reverse=True)
        return [
            {
                "id": j.id,
                "filename": j.filename,
                "status": j.status,
                "submitted_at": j.submitted_at,
                "audio_duration_s": j.audio_duration_s,
            }
            for j in jobs[:limit]
        ]

    def queue_position(self, job_id: str) -> int:
        """Queued jobs ahead of `job_id` (0 = running, or next in line).

        Only meaningful while ``status == "queued"``; 0 for every other state.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status != STATUS_QUEUED:
                return 0
            ahead = 0
            for jid in self._order:
                if jid == job_id:
                    break
                other = self._jobs.get(jid)
                if other is not None and other.status == STATUS_QUEUED:
                    ahead += 1
            return ahead

    # -- JSON snapshot -----------------------------------------------------
    def snapshot(self, job_id: str) -> Optional[dict]:
        """The `JobState` JSON object of `webapp/API.md`, or None if unknown."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            stages = [s.to_dict() for s in job.stages]
            completed = job.completed_stages
            status = job.status
            payload = {
                "id": job.id,
                "filename": job.filename,
                "status": status,
                "queue_position": self.queue_position(job_id),
                "submitted_at": job.submitted_at,
                "audio_duration_s": job.audio_duration_s,
                "stages": stages,
                "eta_s": None,
                "elapsed_s": job.elapsed_s,
                "warnings": self._warnings(job),
                "error": job.error,
                "partial": self._partial(job) if status == STATUS_RUNNING else None,
                "result": job.result,
            }
            if status in (STATUS_QUEUED, STATUS_RUNNING):
                payload["eta_s"] = self.estimator.estimate_remaining(
                    [s.stage for s in job.stages] or self.stage_names,
                    job.audio_duration_s,
                    completed=completed,
                    n_overlap_regions=job.n_overlap_regions,
                )
            return payload

    def _partial(self, job: JobState) -> Optional[dict]:
        """Progressive-disclosure payload read from the running job's spill dir.

        API.md v1.1: present only while the job is running and at least one spill
        file exists; ``null`` otherwise, so the frontend's check is a plain
        truthiness test. Sub-fields are independently ``null`` until their file
        lands, and everything is read defensively — a spill file can be caught
        mid-write, and a half-written JSON must degrade to "not there yet"
        rather than 500 a 1 Hz poll.

        The spill's ``segments`` are mapped to the eval-facing ``turns`` shape so
        the frontend consumes ONE diarization shape before and after completion
        (the two on-disk schemas are deliberately different — see the note in
        `asr_pipeline/stages/diarization.py` `spill`).
        """
        spill = self.spill_dir(job.id)
        diarization = _partial_diarization(_read_json_file(spill / "diarization.json"))
        routing = _partial_routing(_read_json_file(spill / "overlap_regions.json"))
        if diarization is None and routing is None:
            return None
        return {"diarization": diarization, "routing": routing}

    def _warnings(self, job: JobState) -> List[str]:
        """Warning lines for a job: debug-log WARNs + weak-anchor + swap notice."""
        if job._warnings_cache is not None:
            return job._warnings_cache
        lines = self._log_lines(job)
        out = [
            ln for ln in lines
            if any(marker in ln.lower() for marker in _WARNING_MARKERS)
        ]
        if job.result and job.result.get("weak_anchor"):
            out.append(
                "weak_anchor: the speaker anchor is shorter than the configured "
                "minimum — speaker attribution may be unreliable."
            )
        if job.status in (STATUS_DONE, STATUS_FAILED):
            job._warnings_cache = out
        return out

    # -- log tail ----------------------------------------------------------
    def _log_source(self, job: JobState) -> Optional[Path]:
        """Which file this job's log lines come from.

        Live during the run (the pipeline's own log file, which holds exactly one
        run's worth), the job's copy afterwards. A queued job has neither — the
        live file still belongs to whatever ran before it.
        """
        if job.status == STATUS_RUNNING:
            return debug_log_path()
        copy = self.log_copy_path(job.id)
        return copy if copy.exists() else None

    def _log_lines(self, job: JobState) -> List[str]:
        path = self._log_source(job)
        if path is None or not path.exists():
            return []
        try:
            return path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return []

    def log_tail(self, job_id: str, offset: int) -> Optional[dict]:
        """``{"offset", "lines"}`` from line `offset` on. None if job is unknown.

        The offset is a **line index**, not a byte offset: the log is small
        (one run) and line indices can never split a line mid-write.
        """
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            return None
        lines = self._log_lines(job)
        offset = max(0, min(int(offset), len(lines)))
        return {"offset": len(lines), "lines": lines[offset:]}

    # -- worker ------------------------------------------------------------
    def _on_event(self, job: JobState, event: dict) -> None:
        """Translate one pipeline event into stage-row state.

        Handles ``stage_start`` / ``stage_end`` (always emitted) and
        ``stage_progress`` (hook 2 — consumed defensively: if the hook is not
        present, rows simply keep ``progress: null``).
        """
        kind = event.get("event")
        name = event.get("stage")
        if not name:
            return
        with self._lock:
            row = job.stage(name)
            if kind == "stage_start":
                row.state = "running"
            elif kind == "stage_end":
                row.state = "done"
                row.load_s = _as_float(event.get("load_s"))
                row.run_s = _as_float(event.get("run_s"))
                job.stage_timings.append({
                    "stage": name,
                    "load_s": row.load_s or 0.0,
                    "run_s": row.run_s or 0.0,
                })
            elif kind == "stage_progress":
                total = event.get("total")
                row.progress = {"done": event.get("done"), "total": total}
                # The separation stage iterates over overlap fragments, so its
                # `total` is the overlap-region count — the earliest honest input
                # to the overlap-aware ETA refinement (design §5.2).
                if name == "separation" and isinstance(total, int) and total >= 0:
                    job.n_overlap_regions = total

    def _work(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                if job_id is None:
                    return
                job = self.get(job_id)
                if job is None:
                    continue
                self._run_one(job)
            finally:
                self._queue.task_done()

    def _run_one(self, job: JobState) -> None:
        with self._lock:
            self._current = job.id
            job.status = STATUS_RUNNING
            job.started_at = time.monotonic()
            job.stages = [StageState(stage=n) for n in self.stage_names]
            job.stage_timings = []
            job._warnings_cache = None
        try:
            self.runner.run(
                job.id,
                self.mixture_path(job.id),
                self.jobs_root,
                lambda ev: self._on_event(job, ev),
            )
        except BaseException as exc:   # noqa: BLE001 — per-job isolation, never retry
            exc_type = getattr(exc, "exc_type", type(exc).__name__)
            message = getattr(exc, "message", None) or _one_line(exc)
            with self._lock:
                job.status = STATUS_FAILED
                job.error = {"type": exc_type, "message": message}
                job.result = None
                job.finished_at = time.monotonic()
            self._write_error_sidecar(job)
        else:
            result = build_result(
                self.pipeline_dir(job.id),
                self.mixture_path(job.id),
                f"/api/jobs/{job.id}/files",
            )
            with self._lock:
                job.result = result
                job.status = STATUS_DONE if result is not None else STATUS_FAILED
                job.finished_at = time.monotonic()
                if result is None:
                    # run_batch reported success but wrote no metadata.json —
                    # report it rather than showing an empty results page.
                    job.error = {
                        "type": "MissingOutputs",
                        "message": "the run reported success but wrote no "
                                   f"metadata.json under {self.pipeline_dir(job.id)}",
                    }
                    self._write_error_sidecar(job)
            self._record_timings(job)
        finally:
            self._copy_debug_log(job)
            with self._lock:
                job._warnings_cache = None   # recompute once from the copied log
                self._current = None

    def _record_timings(self, job: JobState) -> None:
        """Feed this job's real stage timings back into the ETA table."""
        overlap_s = None
        n_regions = job.n_overlap_regions
        if job.result:
            overlap_s = job.result.get("overlap_total_s")
            n_regions = job.result.get("n_overlap_regions", n_regions)
        self.estimator.record(
            duration_s=job.audio_duration_s,
            stage_timings=job.stage_timings,
            overlap_s=overlap_s,
            n_overlap_regions=n_regions,
        )

    def _copy_debug_log(self, job: JobState) -> None:
        """Copy the run's debug log next to its outputs — success AND failure.

        The live file holds exactly one run and is truncated by the next one, so
        this copy is the only lasting record of a failed run's log.
        """
        src = debug_log_path()
        if not src.exists():
            return
        try:
            self.job_dir(job.id).mkdir(parents=True, exist_ok=True)
            shutil.copy(src, self.log_copy_path(job.id))
        except OSError as exc:
            print(f"[webapp] WARNING: could not copy debug log for {job.id}: {exc}")

    def _write_error_sidecar(self, job: JobState) -> None:
        """Persist the failure so a restart still shows the error card."""
        try:
            self.job_dir(job.id).mkdir(parents=True, exist_ok=True)
            (self.job_dir(job.id) / "error.json").write_text(
                json.dumps(job.error, indent=2), encoding="utf-8"
            )
        except OSError as exc:
            print(f"[webapp] WARNING: could not write error.json for {job.id}: {exc}")

    # -- disk -> registry --------------------------------------------------
    def write_job_sidecar(self, job: JobState) -> None:
        """Persist the upload-time facts the output tree does not carry."""
        self.job_dir(job.id).mkdir(parents=True, exist_ok=True)
        (self.job_dir(job.id) / "job.json").write_text(
            json.dumps(
                {
                    "id": job.id,
                    "filename": job.filename,
                    "submitted_at": job.submitted_at,
                    "audio_duration_s": job.audio_duration_s,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def rebuild_from_disk(self) -> int:
        """Repopulate the registry from `jobs_root`. Returns the job count.

        Completion is the pipeline's own sentinel: ``pipeline/metadata.json``
        (written last). A directory with an ``error.json`` is a recorded failure;
        one with neither was interrupted by a restart — reported as a failure,
        because the webapp cannot honestly claim it is still running.
        """
        found = 0
        for entry in sorted(self.jobs_root.iterdir() if self.jobs_root.exists() else []):
            if not entry.is_dir():
                continue
            job_id = entry.name
            sidecar = _read_json_file(entry / "job.json") or {}
            metadata = entry / PIPELINE_SUBDIR / "metadata.json"
            error = _read_json_file(entry / "error.json")
            if not sidecar and not metadata.exists():
                continue          # not a job directory
            job = JobState(
                id=job_id,
                filename=sidecar.get("filename", f"{job_id}.wav"),
                submitted_at=sidecar.get("submitted_at", _now_iso()),
                audio_duration_s=sidecar.get("audio_duration_s"),
            )
            if metadata.exists():
                job.status = STATUS_DONE
                job.result = build_result(
                    entry / PIPELINE_SUBDIR,
                    self.mixture_path(job_id),
                    f"/api/jobs/{job_id}/files",
                )
                job.stages = [
                    StageState(
                        stage=row.get("stage", "?"),
                        state="done",
                        load_s=row.get("load_s"),
                        run_s=row.get("run_s"),
                    )
                    for row in (job.result or {}).get("stage_timings", [])
                ]
            elif error is not None:
                job.status = STATUS_FAILED
                job.error = error
            else:
                job.status = STATUS_FAILED
                job.error = {
                    "type": "InterruptedRun",
                    "message": "the server restarted while this job was in "
                               "flight; nothing was written. Submit it again.",
                }
            self.register(job)
            found += 1
        return found


def _rows(payload: Optional[dict], key: str) -> List[dict]:
    """``payload[key]`` as a list of dicts — anything else reads as empty.

    The spill files are written by a live run and may be read mid-write, so no
    shape is assumed beyond what is actually there.
    """
    if not isinstance(payload, dict):
        return []
    value = payload.get(key)
    return [r for r in value if isinstance(r, dict)] if isinstance(value, list) else []


def _partial_diarization(payload: Optional[dict]) -> Optional[dict]:
    """Spill ``{segments, overlaps}`` -> the eval-facing ``{turns, overlaps}``."""
    if not isinstance(payload, dict):
        return None
    turns = [
        {"speaker": str(r.get("speaker", "")),
         "start": _as_float(r.get("start")),
         "end": _as_float(r.get("end"))}
        for r in _rows(payload, "segments")
    ]
    overlaps = [
        {"start": _as_float(r.get("start")),
         "end": _as_float(r.get("end")),
         "duration": _as_float(r.get("duration"))}
        for r in _rows(payload, "overlaps")
    ]
    return {"turns": turns, "overlaps": overlaps}


def _partial_routing(payload: Optional[dict]) -> Optional[dict]:
    """Spill ``overlap_regions`` (a ``duration``/``speakers`` superset) -> ``{start, end}``."""
    if not isinstance(payload, dict):
        return None
    return {
        "overlap_regions": [
            {"start": _as_float(r.get("start")), "end": _as_float(r.get("end"))}
            for r in _rows(payload, "overlap_regions")
        ]
    }


def _as_float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _one_line(exc: BaseException) -> str:
    return traceback.format_exception_only(type(exc), exc)[-1].strip()


def _read_json_file(path: Path) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def new_job(
    job_id: str, filename: str, audio_duration_s: Optional[float]
) -> JobState:
    """Freshly submitted job in the ``queued`` state."""
    return JobState(
        id=job_id,
        filename=filename,
        submitted_at=_now_iso(),
        audio_duration_s=audio_duration_s,
    )
