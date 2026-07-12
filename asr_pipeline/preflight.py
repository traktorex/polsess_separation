"""Preflight environment checks for a loaded config (SCOPE §4: fail loud, early).

A CLI ``run`` (or a notebook) should discover a missing ``$SORTFORMER_VENV_PY``,
gated HF token, or absent checkpoint in *seconds*, not after minutes of model
loading. `preflight(cfg)` inspects only the config + environment (it loads no
models), returns the list of blocking-failure strings for the configured
backends, and prints the warn-only conditions to stdout so they stay visible
(SCOPE §4.3). `check_preflight(cfg)` is the raising wrapper the CLI calls before
constructing the pipeline.

Ported from the ``explore_pipeline.ipynb`` preflight cell so the notebook and
the CLI share one check — the package function is the single source of truth,
the notebook cell becomes a one-line call.

Note on the two warn-only conditions (visible, but never blocking):
  - ``$HF_TOKEN`` on the *sortformer* path — needed only for the FIRST (public)
    NeMo model download; harmless once cached.
  - ``num2words`` importability — its absence silently changes cpWER/cpCER
    scoring (digits stay digits), but whether to make it a hard dependency is
    SCOPE §10 q2, reserved for the author. Preflight only warns.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from asr_pipeline.config import PipelineConfig


def _num2words_importable() -> bool:
    """True iff ``num2words`` can be imported. Factored out so tests can stub the
    absent-dependency case without uninstalling the package."""
    return importlib.util.find_spec("num2words") is not None


def preflight(cfg: PipelineConfig) -> list:
    """Return the blocking-failure strings for ``cfg``'s configured backends.

    Empty list = every requirement the loaded config needs is satisfied. Loads
    no models (safe + fast). Warn-only conditions ($HF_TOKEN on the sortformer
    path, missing ``num2words``) print to stdout and never enter the returned
    list. Callers that want a hard stop use `check_preflight`.
    """
    problems: list = []

    # --- Stage 1: diarizer backend ---
    if cfg.diarization.enabled:
        if cfg.diarization.backend == "sortformer":
            sf = os.environ.get("SORTFORMER_VENV_PY")
            if not sf:
                problems.append(
                    "diarization.backend='sortformer' needs $SORTFORMER_VENV_PY "
                    "(isolated NeMo venv python, e.g. ~/sortformer_venv/bin/python; "
                    "recipe in scripts/sortformer_worker.py). Set it, or switch to "
                    "cfg.diarization.backend='pyannote'."
                )
            elif not Path(sf).exists():
                problems.append(
                    f"$SORTFORMER_VENV_PY points at a missing file: {sf}"
                )
            if not os.environ.get("HF_TOKEN"):
                print(
                    "[warn] $HF_TOKEN unset — needed only for the FIRST sortformer "
                    "model download (fine if already cached)."
                )
        elif cfg.diarization.backend == "pyannote":
            if not (cfg.diarization.hf_token or os.environ.get("HF_TOKEN")):
                problems.append(
                    "diarization.backend='pyannote' needs $HF_TOKEN "
                    "(gated pyannote model)."
                )

    # --- Stage 3c: post-separation BWE backend ---
    psp = cfg.post_separation_processing
    if psp.backend == "ap_bwe":
        ck = Path(psp.checkpoint_path)
        if not ck.exists():
            problems.append(
                f"post_separation_processing.backend='ap_bwe' checkpoint missing: "
                f"{ck} (set $AP_BWE_CHECKPOINT or the YAML checkpoint_path)."
            )

    # --- Stage 5: transcription backend (coherex = isolated venv subprocess) ---
    if cfg.transcription.backend == "coherex":
        cx = os.environ.get("COHEREX_VENV_PY")
        if not cx or not Path(cx).exists():
            problems.append(
                "transcription.backend='coherex' needs $COHEREX_VENV_PY "
                "(isolated CohereX venv python)."
            )

    # --- Stage 3b: separator checkpoint (repo-relative path) ---
    if cfg.separation.enabled:
        sep_ck = Path(cfg.separation.checkpoint_path)
        if not sep_ck.exists():
            problems.append(
                f"separator checkpoint missing: {sep_ck} (the path is repo-relative "
                "— run from the repo root or set an absolute separation.checkpoint_path)."
            )

    # --- Warn-only: num2words (SCOPE §10 q2 is the author's; preflight never blocks) ---
    if not _num2words_importable():
        print(
            "[warn] num2words not importable — digit tokens stay as digits in "
            "cpWER/cpCER scoring. Hard-dependency ruling is SCOPE §10 q2 (author's)."
        )

    return problems


def check_preflight(cfg: PipelineConfig) -> None:
    """Raise ``RuntimeError`` if `preflight` finds any blocking problem (SCOPE §4).

    No-op (returns None) when the config's requirements are all satisfied. The
    CLI ``run`` calls this before building the pipeline so a misconfiguration
    dies immediately, never mid-load.
    """
    problems = preflight(cfg)
    if problems:
        detail = "\n".join(f"  - {p}" for p in problems)
        raise RuntimeError(
            "Preflight failed — fix before running stages "
            "(SCOPE §4: no silent downgrade):\n" + detail
        )
