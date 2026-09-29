"""Duration-weighted ETA estimator for a queued/running pipeline job (design §5.2).

The historical run corpus is ~90 s CLARIN fragments, so a flat per-stage median
would mislead on a 10-minute upload. The model therefore splits every stage into

    stage_cost = load_s  +  run_rate * scale_seconds

where ``load_s`` is a duration-independent constant (model load) and
``scale_seconds`` is what the stage's inference time actually tracks:

- ``duration``  — the whole recording (diarization, enhancement, assembly,
  relabel, transcription; routing is duration-scaled but ~free).
- ``overlap``   — only the overlapping seconds (separation and
  post_separation_processing run on overlap fragments only).

Overlap seconds are unknown until routing has run, so before that the estimate
uses `DEFAULT_OVERLAP_FRACTION` of the duration (the median over the 141 frozen
CLARIN fragments). Once the run reveals the real overlap-region count — via the
``stage_progress`` hook's ``total`` on the separation stage — the estimate is
refined with `MEDIAN_OVERLAP_REGION_S`, and after the job finishes the true
overlap seconds from ``routing.json`` are recorded.

Self-correction: `EtaEstimator.record` appends one row per finished job to
``<jobs_root>/timings.jsonl`` and folds it into the in-memory sample lists. A
stage with at least one observed sample uses the median of its samples; a stage
with none falls back to the seed constant below. Nothing here ever blocks a job
— a missing/corrupt timings file is skipped with a warning.

**Seed constants are priors, not measurements of record.** They were derived
from the pipeline's documented ~90 s-fragment behaviour (median total run
48.9 s, design §1) split across the eight stages by their known cost profile;
they exist only so the very first job shows a plausible number, and the first
real run starts replacing them.
"""

from __future__ import annotations

import json
import statistics
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

# Median overlap share of a recording, measured over the 141 frozen CLARIN
# fragments (sum of routing.json region durations / metadata total_duration_s).
DEFAULT_OVERLAP_FRACTION = 0.10

# Median duration of a single overlap region in the same corpus. Used to turn an
# overlap-region *count* (all the stage_progress hook exposes) into seconds.
MEDIAN_OVERLAP_REGION_S = 0.9


@dataclass(frozen=True)
class StageSeed:
    """Prior cost model for one stage: constant load + rate x scale seconds."""

    load_s: float
    run_rate: float          # inference seconds per second of `scales_with`
    scales_with: str         # "duration" | "overlap"


# Pipeline order (mirrors `asr_pipeline.pipeline.Pipeline.__init__`); the actual
# stage list a job displays comes from the Runner seam, never from this dict.
STAGE_SEEDS: Dict[str, StageSeed] = {
    "diarization":                 StageSeed(load_s=6.0, run_rate=0.045, scales_with="duration"),
    "routing":                     StageSeed(load_s=0.0, run_rate=0.001, scales_with="duration"),
    "enhancement":                 StageSeed(load_s=3.0, run_rate=0.067, scales_with="duration"),
    "separation":                  StageSeed(load_s=4.0, run_rate=0.375, scales_with="overlap"),
    "post_separation_processing":  StageSeed(load_s=2.0, run_rate=0.190, scales_with="overlap"),
    "relabel":                     StageSeed(load_s=2.5, run_rate=0.011, scales_with="duration"),
    "assembly":                    StageSeed(load_s=0.5, run_rate=0.017, scales_with="duration"),
    "transcription":               StageSeed(load_s=5.0, run_rate=0.089, scales_with="duration"),
}

# Fallback for a stage name the seed table doesn't know (a config that enables
# something new): a small duration-scaled cost, so the ETA stays finite.
_UNKNOWN_STAGE_SEED = StageSeed(load_s=1.0, run_rate=0.05, scales_with="duration")


class EtaEstimator:
    """Per-stage {load_s, run-rate} table that learns from finished jobs.

    Thread-safe: the worker thread calls `record`, request threads call
    `estimate_remaining`.
    """

    def __init__(self, timings_path: Optional[Path] = None) -> None:
        self.timings_path = Path(timings_path) if timings_path else None
        self._lock = threading.Lock()
        # stage -> list of (load_s, run_rate) observations
        self._samples: Dict[str, List[tuple]] = {}
        # observed seconds per overlap region, for the count -> seconds refinement
        self._region_seconds: List[float] = []
        if self.timings_path is not None and self.timings_path.exists():
            self._load_history()

    # -- reading -----------------------------------------------------------
    def _seed(self, stage: str) -> StageSeed:
        return STAGE_SEEDS.get(stage, _UNKNOWN_STAGE_SEED)

    def _cost(self, stage: str, duration_s: float, overlap_s: float) -> float:
        """Estimated total seconds (load + run) for one stage."""
        seed = self._seed(stage)
        scale = overlap_s if seed.scales_with == "overlap" else duration_s
        samples = self._samples.get(stage)
        if samples:
            load = statistics.median(s[0] for s in samples)
            rate = statistics.median(s[1] for s in samples)
        else:
            load, rate = seed.load_s, seed.run_rate
        return max(0.0, load + rate * max(0.0, scale))

    def overlap_seconds(
        self,
        duration_s: float,
        *,
        overlap_s: Optional[float] = None,
        n_overlap_regions: Optional[int] = None,
    ) -> float:
        """Best available estimate of the overlapping seconds in a recording.

        Preference order: measured seconds > region count x median region length
        > `DEFAULT_OVERLAP_FRACTION` of the duration.
        """
        if overlap_s is not None:
            return max(0.0, float(overlap_s))
        if n_overlap_regions is not None:
            with self._lock:
                region_s = (
                    statistics.median(self._region_seconds)
                    if self._region_seconds
                    else MEDIAN_OVERLAP_REGION_S
                )
            return max(0.0, float(n_overlap_regions) * region_s)
        return max(0.0, float(duration_s) * DEFAULT_OVERLAP_FRACTION)

    def estimate_total(
        self,
        stages: Sequence[str],
        duration_s: Optional[float],
        *,
        overlap_s: Optional[float] = None,
        n_overlap_regions: Optional[int] = None,
    ) -> Optional[float]:
        """Estimated wall seconds for a full run of `stages` over `duration_s`."""
        return self.estimate_remaining(
            stages, duration_s, completed=(),
            overlap_s=overlap_s, n_overlap_regions=n_overlap_regions,
        )

    def estimate_remaining(
        self,
        stages: Sequence[str],
        duration_s: Optional[float],
        *,
        completed: Iterable[str] = (),
        overlap_s: Optional[float] = None,
        n_overlap_regions: Optional[int] = None,
    ) -> Optional[float]:
        """Estimated seconds still to go, summing the not-yet-finished stages.

        `eta_s` in the JSON API is this value — **time remaining**, not the total
        run time. A queued job's ETA is the full run (nothing completed yet);
        completed stages drop out one by one, so the number decreases monotonically
        as the run progresses. Returns ``None`` when the duration is unknown.
        """
        if duration_s is None:
            return None
        done = set(completed)
        ov = self.overlap_seconds(
            duration_s, overlap_s=overlap_s, n_overlap_regions=n_overlap_regions
        )
        with self._lock:
            return float(
                sum(self._cost(s, float(duration_s), ov) for s in stages if s not in done)
            )

    # -- writing -----------------------------------------------------------
    def record(
        self,
        *,
        duration_s: Optional[float],
        stage_timings: Sequence[dict],
        overlap_s: Optional[float] = None,
        n_overlap_regions: Optional[int] = None,
    ) -> None:
        """Fold one finished job's stage timings into the table (and the log file).

        `stage_timings` are the ``{"stage", "load_s", "run_s"}`` rows collected
        from the pipeline's ``stage_end`` events. Rows without a usable duration
        scale are ignored rather than poisoning the medians.
        """
        if not duration_s or duration_s <= 0 or not stage_timings:
            return
        ov = self.overlap_seconds(
            float(duration_s), overlap_s=overlap_s, n_overlap_regions=n_overlap_regions
        )
        with self._lock:
            self._absorb(float(duration_s), ov, stage_timings, n_overlap_regions)
        self._append_history(
            {
                "duration_s": float(duration_s),
                "overlap_s": ov,
                "n_overlap_regions": n_overlap_regions,
                "stages": [dict(r) for r in stage_timings],
            }
        )

    def _absorb(
        self,
        duration_s: float,
        overlap_s: float,
        stage_timings: Sequence[dict],
        n_overlap_regions: Optional[int],
    ) -> None:
        """Turn one job's rows into (load_s, run_rate) samples. Caller holds the lock."""
        for row in stage_timings:
            stage = row.get("stage")
            if not stage:
                continue
            seed = self._seed(stage)
            scale = overlap_s if seed.scales_with == "overlap" else duration_s
            if scale <= 0:
                # No scale to divide by (e.g. a recording with zero overlap):
                # the run rate is unidentifiable, so only the load is usable.
                continue
            try:
                load_s = float(row.get("load_s", 0.0))
                run_s = float(row.get("run_s", 0.0))
            except (TypeError, ValueError):
                continue
            self._samples.setdefault(stage, []).append((load_s, run_s / scale))
        if n_overlap_regions:
            self._region_seconds.append(overlap_s / float(n_overlap_regions))

    def _append_history(self, row: dict) -> None:
        if self.timings_path is None:
            return
        try:
            self.timings_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.timings_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
        except OSError as exc:  # persistence is best-effort, never fatal
            print(f"[webapp] WARNING: could not append {self.timings_path}: {exc}")

    def _load_history(self) -> None:
        """Replay ``timings.jsonl`` into the sample lists. Bad lines are skipped."""
        try:
            lines = self.timings_path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            print(f"[webapp] WARNING: could not read {self.timings_path}: {exc}")
            return
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                duration_s = float(row["duration_s"])
                overlap_s = float(row.get("overlap_s") or 0.0)
                stages = row.get("stages") or []
            except (ValueError, KeyError, TypeError):
                continue
            self._absorb(duration_s, overlap_s, stages, row.get("n_overlap_regions"))
