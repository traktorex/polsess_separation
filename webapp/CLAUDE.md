# webapp/CLAUDE.md

Thesis-defense showcase UI over `asr_pipeline/`. Stacks on top of the repository root `CLAUDE.md`.

```bash
./webapp/run.sh                                  # real server, port ${WEBAPP_PORT:-8871}, HF_HUB_OFFLINE=1; preflight at startup
venv/bin/python -m webapp.dev_server --port 8899 # GPU-free dev harness (FakeRunner replays a timed run; --fail-stage for error UI)
venv/bin/python -m webapp.examples_build         # regenerate examples_manifest.json (gitignored) from the frozen v41_merge eval tree
pytest tests/test_webapp_backend.py              # backend suite (no GPU, no asr_pipeline import)
```

- FastAPI + vanilla ES modules: no build step, no CDN. The HTTP contract is `API.md` (binding).
- A pure read-only consumer of `asr_pipeline` — the job queue lives here, never in the package (`asr_pipeline/SCOPE.md` §1).
- One worker thread, one job at a time (12 GB GPU, phase-major). Every job runs with per-job `spill_intermediate` → `<job>/spill`, the source of mid-run `partial` results and the enhancement A/B panel.
- Jobs land in `$WEBAPP_JOBS_ROOT` (default `~/webapp_jobs`).
- Polish UI with English technical terms. Ground truth and scores appear only on the examples page.
- The in-browser on-device demo (`webapp_ondevice/`) lives on branch `experiment/webapp-ondevice`, is not in the thesis, and must not be re-added to `main`.
