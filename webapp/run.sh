#!/usr/bin/env bash
# Launch the showcase webapp from the repo root.
#
#   ./webapp/run.sh            # normal (demo) start
#   ./webapp/run.sh --dev      # add uvicorn --reload (see note below)
#
# Startup order is enforced by the app itself: the shipped config is loaded and
# `check_preflight` runs BEFORE the server serves anything, so a missing
# $SORTFORMER_VENV_PY / AP-BWE checkpoint fails in seconds with a readable
# message instead of mid-run.
#
# HF_HUB_OFFLINE=1 is the demo-day default (design §6): every model this config
# needs is already cached, and going offline removes the HuggingFace 504-abort
# class of failures. Unset it for the one run that first downloads a new model.
#
# --dev note: `--reload` restarts the process on any file change, which kills an
# in-flight pipeline run and empties the in-process job registry (completed jobs
# are rebuilt from disk, running ones are marked failed). Fine while editing the
# front-end, never for a demo.
set -euo pipefail

cd "$(dirname "$0")/.."

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export WEBAPP_JOBS_ROOT="${WEBAPP_JOBS_ROOT:-$HOME/webapp_jobs}"
PORT="${WEBAPP_PORT:-8871}"

EXTRA=()
if [[ "${1:-}" == "--dev" ]]; then
    EXTRA+=(--reload --reload-dir webapp)
fi

exec venv/bin/python -m uvicorn webapp.app:app \
    --host 0.0.0.0 --port "$PORT" "${EXTRA[@]}"
