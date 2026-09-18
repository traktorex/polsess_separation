"""File + stdout debug logger for pipeline diagnostics.

VSCode's WSL bridge sometimes loses the kernel during long-running cells,
which means stdout messages never reach the notebook view. Tailing a file
from a separate WSL terminal (``tail -f /tmp/asr_pipeline_debug.log``)
survives that — the Python process keeps appending even when VSCode is
disconnected, so the last log line before any hang stays recoverable.

Usage::

    from asr_pipeline.debug_log import dlog
    dlog("assembly", "starting anchor computation")

The log file is overwritten at the start of every `Pipeline.run()` /
`Pipeline.load_audio()` (via `reset_log()`); it accumulates across
subsequent `run_stage()` calls within the same pipeline run.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional


# Override via env var if a different path is desired (e.g. inside CLARIN).
_DEFAULT_PATH = "/tmp/asr_pipeline_debug.log"
LOG_PATH: Path = Path(os.environ.get("ASR_PIPELINE_DEBUG_LOG", _DEFAULT_PATH))

_T0: Optional[float] = None


def reset_log() -> None:
    """Truncate the log file and reset the relative-time clock.

    Called at the start of a new pipeline run so each run gets a clean log
    rather than appending forever.
    """
    global _T0
    _T0 = time.perf_counter()
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, "w") as f:
            f.write(
                f"# asr_pipeline debug log — started {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
    except OSError as e:
        # File logging is best-effort; never break the pipeline because of it.
        print(f"[dlog] WARNING: could not reset {LOG_PATH}: {e}", flush=True)


def dlog(stage: str, msg: str, *, to_stdout: bool = True) -> None:
    """Write a message to the debug log file (always) and stdout (optional).

    Each line is prefixed with elapsed seconds since the last `reset_log()`
    so the relative timing of events is visible at a glance.

    Set ``to_stdout=False`` for low-level diagnostic logs (e.g. orchestrator
    transitions, GPU unload sync points) that are useful for `tail -f` in a
    separate WSL terminal but would just clutter the notebook cell output.
    User-facing progress messages (e.g. assembly-stage per-speaker progress)
    keep ``to_stdout=True`` so they remain visible in the notebook.
    """
    global _T0
    if _T0 is None:
        _T0 = time.perf_counter()
    dt = time.perf_counter() - _T0
    line = f"[{dt:7.2f}s] [{stage}] {msg}"
    if to_stdout:
        print(line, flush=True)
        sys.stdout.flush()
    try:
        with open(LOG_PATH, "a") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
    except OSError:
        # Don't propagate file-IO errors — stdout already received the message
        # (if it was requested), and the file log is best-effort anyway.
        pass
