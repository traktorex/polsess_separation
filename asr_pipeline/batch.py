"""Batch pipeline runner + the shared per-recording output writer.

One home for the multi-recording loop that used to be copy-pasted across three
scripts (``sweep_pipeline.run_config``, ``run_pipeline_on_recording.run_one_mode``,
``batch_pipeline_noenh._run_one``) with diverging semantics. `run_batch` fixes
the semantics to SCOPE §4.2: a single recording's failure is caught, recorded in
``failures.csv``, and the batch continues; the GPU teardown (unload → gc →
empty_cache) lives in exactly one place (`_teardown_pipeline`).

Skip sentinel: completion defaults to ``metadata.json`` present in the target
subdir (it is written LAST by `write_pipeline_outputs`). Callers that must honor
a legacy sentinel — the sweep's existing on-disk trees are marked complete by
``transcript_A.txt`` — pass their own `is_complete` callable, so migrating the
sweep to this runner never recomputes an already-finished tree.

`write_run_outputs` is the per-recording writer shared with the CLI ``run``
subcommand (`asr_pipeline/__main__.py`): eval layout + ``run_meta.json`` (total +
per-stage seconds) + an optional mixture copy for eval discovery.
"""

from __future__ import annotations

import csv
import gc
import json
import shutil
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import torch

from asr_pipeline.config import PipelineConfig
from asr_pipeline.io import write_pipeline_outputs
from asr_pipeline.pipeline import Pipeline

# A recording source: an audio path (id = its stem) or an explicit
# ``(recording_id, audio_path)`` pair (lets a caller pin the id independently of
# the filename — e.g. a legacy ``mixture.wav`` under an ``<id>/`` directory).
RecordingItem = Union[str, Path, Tuple[str, Union[str, Path]]]


# ---------------------------------------------------------------------------
# Ablation-mode presets (replaces the MODES setattr-lambda encoding)
# ---------------------------------------------------------------------------
#
# ``mode -> (output subdir, dotted-override dict applied via apply_overrides)``.
# The override sets replicate `scripts/run_pipeline_on_recording.py`'s MODES
# lambdas exactly, so the eval tree the L3 table reads is unchanged:
#   - full     = the full pipeline (no override).
#   - no_sep   = pipeline_nosep   : separation.enabled = False.
#   - no_enh   = pipeline_noenh   : enhancement.enabled = False.
#   - minimal  = pipeline_minimal : both off (diarize + slice + transcribe only) —
#     falls out trivially as the union of no_sep + no_enh, so it is kept.
MODE_PRESETS: dict[str, Tuple[str, dict]] = {
    "full":    ("pipeline",         {}),
    "no_sep":  ("pipeline_nosep",   {"separation.enabled": False}),
    "no_enh":  ("pipeline_noenh",   {"enhancement.enabled": False}),
    "minimal": ("pipeline_minimal", {"separation.enabled": False,
                                     "enhancement.enabled": False}),
}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass
class BatchReport:
    """Outcome of one `run_batch` call.

    ``failed`` holds ``(recording_id, exception_type)`` pairs; the full one-line
    traceback summary lands in ``failures.csv`` under ``out_root`` (only written
    when there is at least one failure).
    """

    out_root: Path
    subdir_name: str
    succeeded: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    failed: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def n_total(self) -> int:
        return len(self.succeeded) + len(self.skipped) + len(self.failed)

    @property
    def failures_csv(self) -> Optional[Path]:
        p = Path(self.out_root) / "failures.csv"
        return p if p.exists() else None


# ---------------------------------------------------------------------------
# Per-recording output writer (shared with the CLI `run` subcommand)
# ---------------------------------------------------------------------------


def write_run_outputs(
    ctx,
    config,
    out_root,
    stage_timings,
    total_seconds,
    *,
    recording_id: Optional[str] = None,
    subdir_name: str = "pipeline",
    copy_mixture: bool = True,
) -> Path:
    """Write the per-recording eval layout for one run + drop ``run_meta.json``.

    Outputs land under ``<out_root>/<id>/<subdir_name>/`` (``write_pipeline_outputs``
    schema). ``recording_id`` pins ``<id>`` explicitly; ``None`` (the CLI ``run``
    default) derives it from ``ctx.input_path``'s filename stem. Pinning it lets a
    batch caller keep a stable id when the input filename differs from the id
    (e.g. a legacy ``mixture.wav`` under an ``<id>/`` directory).

    ``run_meta.json`` records the total wall seconds and the per-stage
    ``{stage, load_s, run_s}`` rows from the ``on_event`` timing (A6) — the
    load-vs-inference split that retires the stale-``run_meta`` timing-mirage
    class of bugs.

    ``copy_mixture`` (default True) copies the input to ``<out_root>/<id>/<id>.wav``
    so the eval module's discovery (``walk_eval_tree`` / ``load_recording`` skip a
    recording directory whose root lacks the mixture) finds it. The copy is
    skipped when source and destination resolve to the same file — the case when
    writing back into an existing eval tree whose input already sits at
    ``<id>/<id>.wav`` (avoids ``shutil.SameFileError``).

    Returns the ``<subdir_name>/`` subdirectory path.
    """
    if recording_id is None:
        recording_id = ctx.input_path.stem if ctx.input_path else "unknown"
    out_dir = Path(out_root).expanduser() / recording_id
    pipeline_dir = write_pipeline_outputs(
        ctx, out_dir,
        config_snapshot=asdict(config),
        subdir_name=subdir_name,
        stage_timings=stage_timings,
    )
    (pipeline_dir / "run_meta.json").write_text(
        json.dumps({"seconds": total_seconds, "stages": stage_timings}, indent=2),
        encoding="utf-8",
    )
    if copy_mixture and ctx.input_path is not None:
        mixture_dst = out_dir / f"{recording_id}.wav"
        if Path(ctx.input_path).resolve() != mixture_dst.resolve():
            shutil.copy(ctx.input_path, mixture_dst)
    return pipeline_dir


# ---------------------------------------------------------------------------
# Internal: timing collector + GPU teardown
# ---------------------------------------------------------------------------


class _StageTimings:
    """``on_event`` sink that records per-stage timings and forwards events.

    Wired as ``Pipeline(cfg, on_event=self)``. Each ``stage_end`` contributes one
    ``{"stage", "load_s", "run_s"}`` row to ``self.timings`` (what ``run_meta`` /
    ``metadata`` record); every event is also forwarded to the caller's
    ``on_event`` if one was passed to `run_batch`.
    """

    def __init__(self, forward: Optional[Callable[[dict], None]] = None) -> None:
        self.timings: list = []
        self._forward = forward

    def __call__(self, event: dict) -> None:
        if event.get("event") == "stage_end":
            self.timings.append({
                "stage": event.get("stage", "?"),
                "load_s": float(event.get("load_s", 0.0)),
                "run_s": float(event.get("run_s", 0.0)),
            })
        if self._forward is not None:
            self._forward(event)


def _teardown_pipeline(pipeline) -> None:
    """Free a pipeline's GPU references between recordings — the ONE home for the
    ``unload → gc.collect → empty_cache`` block copy-pasted across the old scripts.

    ``Pipeline.run`` frees only the final stage's model; this additionally drops
    any straggler references and reclaims the CUDA cache so the next recording
    starts from a clean allocator (the one-model-at-a-time GPU budget). The caller
    still drops its own loop-local reference to the pipeline object.
    """
    pipeline.unload()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _normalize_recording(item: RecordingItem) -> Tuple[str, Path]:
    """``item`` → ``(recording_id, audio_path)``. A 2-tuple pins the id explicitly;
    a bare path derives the id from the filename stem."""
    if isinstance(item, tuple):
        rid, audio = item
        return str(rid), Path(audio)
    audio = Path(item)
    return audio.stem, audio


# ---------------------------------------------------------------------------
# The batch runner
# ---------------------------------------------------------------------------


def run_batch(
    cfg: PipelineConfig,
    recordings,
    out_root,
    subdir_name: str,
    *,
    skip_existing: bool = True,
    on_event: Optional[Callable[[dict], None]] = None,
    is_complete: Optional[Callable[[Path], bool]] = None,
    copy_mixture: bool = True,
) -> BatchReport:
    """Run ``cfg`` over ``recordings``, writing each to ``<out_root>/<id>/<subdir_name>/``.

    ``recordings`` is an iterable of audio paths (id = the file stem) or explicit
    ``(id, audio_path)`` pairs. Semantics (SCOPE §4.2 batch): one recording's
    failure is caught, recorded, and the batch continues; the fleet-level result
    is the returned `BatchReport`.

    Skip logic: when ``skip_existing`` is True, a recording whose target subdir is
    already complete is skipped. Completion defaults to ``metadata.json`` present
    (written last by `write_pipeline_outputs`); pass ``is_complete`` to override —
    the sweep passes its legacy ``transcript_A.txt`` sentinel so existing sweep
    trees are recognised without recompute. ``skip_existing=False`` re-runs
    everything (the CLI ``--force``).

    ``on_event`` receives the pipeline's per-stage events for progress display;
    per-stage timings are always captured into each ``run_meta.json`` regardless.
    """
    out_root = Path(out_root).expanduser()
    if is_complete is None:
        def is_complete(target_dir: Path) -> bool:  # noqa: E306 (default sentinel)
            return (target_dir / "metadata.json").exists()

    report = BatchReport(out_root=out_root, subdir_name=subdir_name)
    failure_rows: List[Tuple[str, str, str]] = []
    items = [_normalize_recording(r) for r in recordings]
    n = len(items)

    for i, (rid, audio) in enumerate(items, start=1):
        target_dir = out_root / rid / subdir_name
        prefix = f"  [{i}/{n}] {rid}"

        if not Path(audio).exists():
            print(f"{prefix}: MISSING input {audio}")
            report.skipped.append(rid)
            continue
        if skip_existing and is_complete(target_dir):
            print(f"{prefix}: skip (done)")
            report.skipped.append(rid)
            continue

        collector = _StageTimings(forward=on_event)
        t0 = time.perf_counter()
        pipeline = Pipeline(cfg, on_event=collector)
        try:
            ctx = pipeline.run(str(audio))
            elapsed = time.perf_counter() - t0
            write_run_outputs(
                ctx, cfg, out_root, collector.timings,
                total_seconds=elapsed, recording_id=rid,
                subdir_name=subdir_name, copy_mixture=copy_mixture,
            )
            print(f"{prefix}: done in {elapsed:.1f}s")
            report.succeeded.append(rid)
        except Exception as exc:
            summary = traceback.format_exception_only(type(exc), exc)[-1].strip()
            print(f"{prefix}: ERROR {type(exc).__name__}: {exc}")
            report.failed.append((rid, type(exc).__name__))
            failure_rows.append((rid, type(exc).__name__, summary))
        finally:
            _teardown_pipeline(pipeline)
            del pipeline

    _write_failures_csv(out_root, failure_rows)
    return report


def _write_failures_csv(
    out_root: Path, rows: List[Tuple[str, str, str]]
) -> None:
    """Write ``<out_root>/failures.csv`` when any recording failed (SCOPE §4.2).

    Columns: recording id, exception type, one-line traceback summary. Nothing is
    written on a clean batch (no stale zero-row file to mistake for a failure)."""
    if not rows:
        return
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "failures.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["recording_id", "exception_type", "traceback_summary"])
        writer.writerows(rows)
