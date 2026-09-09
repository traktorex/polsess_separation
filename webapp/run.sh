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

# Friendly URLs: uvicorn prints its BIND address (0.0.0.0 = "all interfaces"),
# which is not something a browser can open. Under WSL2, localhost is forwarded
# to Windows out of the box; the WSL IP is the fallback for phones on the LAN
# (requires a Windows portproxy / mirrored networking) and for the rare setup
# where localhost forwarding is off.
WSL_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
echo "─────────────────────────────────────────────────"
echo "  Otwórz w przeglądarce:  http://localhost:${PORT}"
[[ -n "$WSL_IP" ]] && \
echo "  (fallback / LAN:        http://${WSL_IP}:${PORT})"
echo "─────────────────────────────────────────────────"

exec venv/bin/python -m uvicorn webapp.app:app \
    --host 0.0.0.0 --port "$PORT" "${EXTRA[@]}"
