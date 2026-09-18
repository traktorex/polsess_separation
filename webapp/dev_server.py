"""Front-end dev harness: the real webapp, a scripted fake pipeline runner.

Runs `webapp.app.create_app` with a `FakeRunner` injected through the same
`Runner` seam the tests use — so there is no GPU, no model load, no preflight,
and no `asr_pipeline` import at all, but every route, poll payload and file
download behaves exactly as in production.

The fake replays a realistic, *timed* event script (stage_start →
stage_progress → stage_end, with per-stage sleeps in the proportions the real
v41_merge arm shows on a ~90 s fragment), spills `diarization.json` and
`overlap_regions.json` mid-run so the progressive-disclosure `partial` payload
appears, writes a debug log the log panel can stream, and finally copies a
frozen example's output directory into the job's `pipeline/` dir — giving the
results view REAL transcripts, peaks and diarization to render.

    venv/bin/python -m webapp.dev_server --port 8899
    # then: open /, upload anything (or use an example's "uruchom ponownie")

Nothing here is used by the production server; `webapp/run.sh` is unaffected.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Callable, List, Optional

from webapp.app import create_app, DEFAULT_EXAMPLES_MANIFEST
from webapp.queue import RunnerFailure, SPILL_SUBDIR, debug_log_path

# The v41_merge chain (relabel enabled -> 8 stages), in pipeline order.
STAGES = [
    "diarization",
    "routing",
    "enhancement",
    "separation",
    "post_separation_processing",
    "relabel",
    "assembly",
    "transcription",
]

# Rough shape of a real ~90 s run: (load_s, run_s). Scaled by --speed.
STAGE_COST = {
    "diarization": (3.1, 4.8),
    "routing": (0.0, 0.1),
    "enhancement": (2.4, 6.1),
    "separation": (4.9, 7.4),
    "post_separation_processing": (1.2, 2.6),
    "relabel": (0.8, 1.1),
    "assembly": (0.4, 2.0),
    "transcription": (5.6, 9.8),
}


def _default_example_dir() -> Optional[Path]:
    """The frozen example the fake replays, taken from the real manifest."""
    try:
        data = json.loads(DEFAULT_EXAMPLES_MANIFEST.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    for row in data.get("examples") or []:
        pdir = Path(row.get("pipeline_dir", ""))
        if pdir.is_dir() and (pdir / "metadata.json").exists():
            return pdir
    return None


class FakeRunner:
    """Replays a timed event script, then copies a frozen result directory."""

    def __init__(
        self,
        source_dir: Optional[Path],
        speed: float = 4.0,
        fail_stage: Optional[str] = None,
    ) -> None:
        self.source_dir = source_dir
        self.speed = max(speed, 0.01)
        self.fail_stage = fail_stage

    def stage_names(self) -> List[str]:
        return list(STAGES)

    # -- helpers ---------------------------------------------------------
    def _log(self, lines: List[str]) -> None:
        try:
            with open(debug_log_path(), "a", encoding="utf-8") as fh:
                for line in lines:
                    fh.write(line + "\n")
        except OSError:
            pass

    def _spill_diarization(self, spill: Path) -> None:
        """Write the spill-shaped `{segments, overlaps}` file (partial payload)."""
        src = (
            json.loads((self.source_dir / "diarization.json").read_text("utf-8"))
            if self.source_dir and (self.source_dir / "diarization.json").exists()
            else {}
        )
        segments = [
            {"speaker": t.get("speaker"), "start": t.get("start"), "end": t.get("end")}
            for t in src.get("turns") or []
        ]
        overlaps = src.get("overlaps") or []
        spill.mkdir(parents=True, exist_ok=True)
        (spill / "diarization.json").write_text(
            json.dumps({"segments": segments, "overlaps": overlaps}, ensure_ascii=False),
            encoding="utf-8",
        )

    def _spill_routing(self, spill: Path) -> List[dict]:
        src = (
            json.loads((self.source_dir / "routing.json").read_text("utf-8"))
            if self.source_dir and (self.source_dir / "routing.json").exists()
            else {}
        )
        regions = src.get("overlap_regions") or []
        spill.mkdir(parents=True, exist_ok=True)
        (spill / "overlap_regions.json").write_text(
            json.dumps({"overlap_regions": regions}, ensure_ascii=False),
            encoding="utf-8",
        )
        return regions

    # -- the seam --------------------------------------------------------
    def run(
        self,
        job_id: str,
        wav_path: Path,
        out_root: Path,
        on_event: Callable[[dict], None],
    ) -> None:
        job_dir = Path(out_root) / job_id
        spill = job_dir / SPILL_SUBDIR
        try:                                   # one run's worth, like the real log
            debug_log_path().write_text("", encoding="utf-8")
        except OSError:
            pass

        n_regions = 0
        for stage in STAGES:
            load_s, run_s = STAGE_COST[stage]
            on_event({"event": "stage_start", "stage": stage})
            self._log([f"[{stage}] start"])
            time.sleep(load_s / self.speed)

            if stage == "separation" and n_regions:
                for i in range(1, n_regions + 1):
                    on_event({"event": "stage_progress", "stage": stage,
                              "done": i, "total": n_regions})
                    self._log([f"[separation] region {i}/{n_regions}"])
                    time.sleep(run_s / n_regions / self.speed)
            else:
                time.sleep(run_s / self.speed)

            if stage == self.fail_stage:
                raise RunnerFailure(
                    "RuntimeError", f"fake failure injected in stage {stage!r}"
                )
            if stage == "diarization":
                self._spill_diarization(spill)
                self._log(["WARNING: fake run — diarization spill written"])
            elif stage == "routing":
                n_regions = len(self._spill_routing(spill))
                self._log([f"[routing] {n_regions} overlap regions accepted"])

            on_event({"event": "stage_end", "stage": stage,
                      "load_s": load_s, "run_s": run_s, "wall_s": load_s + run_s})
            self._log([f"[{stage}] done in {load_s + run_s:.2f} s"])

        if self.source_dir is None:
            raise RunnerFailure(
                "FileNotFoundError",
                "dev harness: no frozen example directory to copy results from",
            )
        target = job_dir / "pipeline"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(self.source_dir, target)
        # The spill's enhanced_full.wav stand-in, so the A/B panel has something.
        mixture = Path(wav_path)
        if mixture.exists():
            spill.mkdir(parents=True, exist_ok=True)
            shutil.copy(mixture, spill / "enhanced_full.wav")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8899)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--speed", type=float, default=4.0,
                        help="time compression of the scripted run (1 = realistic)")
    parser.add_argument("--fail-stage", default=None,
                        help="raise inside this stage, to exercise the error card")
    parser.add_argument("--jobs-root", default=None,
                        help="default: a fresh temp directory")
    parser.add_argument("--source-dir", default=None,
                        help="frozen pipeline output dir the fake replays")
    args = parser.parse_args()

    import uvicorn

    jobs_root = Path(args.jobs_root) if args.jobs_root else Path(
        tempfile.mkdtemp(prefix="webapp_dev_jobs_")
    )
    source = Path(args.source_dir) if args.source_dir else _default_example_dir()
    os.environ.setdefault(
        "ASR_PIPELINE_DEBUG_LOG", str(jobs_root / "dev_debug.log")
    )
    print(f"[dev] jobs root : {jobs_root}")
    print(f"[dev] replaying : {source}")
    app = create_app(
        runner=FakeRunner(source, speed=args.speed, fail_stage=args.fail_stage),
        jobs_root=jobs_root,
        skip_preflight=True,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
