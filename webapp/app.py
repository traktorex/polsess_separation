"""FastAPI application: every route in `webapp/API.md`.

Composition root. `create_app` wires the three pieces — the loaded pipeline
config, the `Runner` seam, and the on-disk jobs root — and is the only place
that knows about HTTP.

Startup order matters and is deliberate (design §3):

1. load `asr_pipeline/configs/sweep_best_e31_refineplus.yaml` **once**;
2. `check_preflight(cfg)` — fail loud, before serving, before any model load;
3. rebuild the job registry from disk;
4. start the single worker thread.

Steps 1-2 are skipped when a `runner` is injected, which is what lets the tests
exercise every route with no GPU, no env vars, and no pipeline import at all.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from webapp.queue import SPILL_SUBDIR, JobService, PipelineRunner, new_job
from webapp.render import build_result

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent

# The one shipped config (design §1: v1 ships a single fixed arm, no knobs).
DEFAULT_CONFIG_PATH = (
    _REPO_ROOT / "asr_pipeline" / "configs" / "sweep_best_e31_refineplus.yaml"
)
DEFAULT_JOBS_ROOT = Path(os.environ.get("WEBAPP_JOBS_ROOT", str(Path.home() / "webapp_jobs")))
DEFAULT_EXAMPLES_MANIFEST = _HERE / "examples_manifest.json"

# Uploads are normalised to the pipeline's working sample rate. The pipeline
# resamples anything it is given (`io.load_audio_as_mono`), so this is a
# convenience — smaller files, one probe path — not a correctness requirement.
UPLOAD_SAMPLE_RATE = 16_000

# Whitelisted downloadable names (API.md). Exact names plus the two per-speaker
# families, so an N-speaker run exposes stream_C.wav / transcript_C.txt without
# a code change. Nothing else is servable, and no name may contain a separator.
_EXACT_FILES = {
    "mixture.wav", "metadata.json", "annotation.eaf", "debug.log",
    "enhanced_full.wav",       # from <job>/spill/ — enhancement A/B panel
}
_FILE_PATTERNS = (
    re.compile(r"^stream_[A-Za-z0-9_]{1,32}\.wav$"),
    re.compile(r"^transcript_[A-Za-z0-9_]{1,32}\.(txt|json)$"),
)


# ---------------------------------------------------------------------------
# Upload helpers
# ---------------------------------------------------------------------------


def _probe_duration(path: Path) -> Optional[float]:
    """Audio duration in seconds via soundfile, or None if it cannot be read."""
    try:
        info = sf.info(str(path))
    except (RuntimeError, OSError):
        return None
    if not info.samplerate:
        return None
    return float(info.frames) / float(info.samplerate)


def _ffmpeg_to_wav(src: Path, dst: Path) -> None:
    """Convert `src` to 16 kHz mono 16-bit WAV at `dst`.

    Loud on failure (SCOPE §4.1 — never pretend a conversion happened): raises
    `HTTPException(400)` carrying ffmpeg's own last words.
    """
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(src),
        "-ac", "1", "-ar", str(UPLOAD_SAMPLE_RATE), "-c:a", "pcm_s16le",
        str(dst),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    except FileNotFoundError:
        raise HTTPException(
            400, "ffmpeg is not installed — cannot convert a non-WAV upload."
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(400, "ffmpeg timed out converting the upload.")
    if proc.returncode != 0 or not dst.exists():
        tail = (proc.stderr or "").strip().splitlines()[-3:]
        raise HTTPException(
            400, "ffmpeg could not convert the upload: " + " / ".join(tail)
        )


def _resolve_whitelisted(
    name: str, pipeline_dir: Path, mixture_path: Optional[Path], job_dir: Path
) -> Optional[Path]:
    """Map a requested download name to a real path, or None if not allowed.

    The whitelist is a name test, not a path test: `name` never becomes part of
    a directory walk, and the resolved path is confirmed to sit inside the job
    directory before it is served — traversal has nothing to grip.
    """
    if name not in _EXACT_FILES and not any(p.match(name) for p in _FILE_PATTERNS):
        return None
    if name == "mixture.wav":
        path = mixture_path
    elif name == "debug.log":
        path = job_dir / "debug.log"
    elif name == "enhanced_full.wav":
        # The enhanced mixture is a per-stage spill artefact, not a pipeline
        # output. Absent (spill off, or the stage has not run yet) -> 404, which
        # is also what an example directory gives, since examples have no spill.
        path = job_dir / SPILL_SUBDIR / name
    else:
        path = pipeline_dir / name
    if path is None or not path.is_file():
        return None
    resolved = path.resolve()
    root = job_dir.resolve()
    if root not in resolved.parents and resolved.parent != root:
        return None
    return resolved


# ---------------------------------------------------------------------------
# Examples gallery
# ---------------------------------------------------------------------------


class ExamplesLibrary:
    """The frozen v41_merge gallery, backed by `examples_manifest.json`.

    The manifest holds the light per-example row (built offline by
    `webapp.examples_build`); the heavy ``job_like`` payload — transcripts and
    the peak envelopes (`render.DEFAULT_BUCKETS` buckets) for three wavs — is
    assembled on demand and cached in memory, optionally warmed in a background
    thread at startup so the first visitor never waits.
    """

    def __init__(self, manifest_path: Optional[Path]) -> None:
        self.manifest_path = manifest_path
        self.rows: List[dict] = []
        self._cache: Dict[str, Optional[dict]] = {}
        self._lock = threading.Lock()
        if manifest_path and Path(manifest_path).exists():
            try:
                data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
                self.rows = list(data.get("examples") or [])
            except (OSError, ValueError) as exc:
                print(f"[webapp] WARNING: unreadable examples manifest: {exc}")

    def row(self, example_id: str) -> Optional[dict]:
        for row in self.rows:
            if row.get("id") == example_id:
                return row
        return None

    def pipeline_dir(self, example_id: str) -> Optional[Path]:
        row = self.row(example_id)
        return Path(row["pipeline_dir"]) if row and row.get("pipeline_dir") else None

    def mixture_path(self, example_id: str) -> Optional[Path]:
        row = self.row(example_id)
        return Path(row["mixture_path"]) if row and row.get("mixture_path") else None

    def job_like(self, example_id: str) -> Optional[dict]:
        with self._lock:
            if example_id in self._cache:
                return self._cache[example_id]
        pdir = self.pipeline_dir(example_id)
        payload = (
            build_result(
                pdir, self.mixture_path(example_id), f"/api/examples/{example_id}/files"
            )
            if pdir is not None
            else None
        )
        with self._lock:
            self._cache[example_id] = payload
        return payload

    def listing(
        self, ids: Optional[List[str]] = None, light: bool = False
    ) -> List[dict]:
        """The `GET /api/examples` body.

        `ids` restricts the listing; `light` returns the rows with
        ``job_like: null``. Both default to off, so the plain call returns
        exactly what API.md specifies — they exist because the full gallery is
        ~6 MB of transcripts and peak envelopes, which is fine on the LAN and
        slow over a quick tunnel.
        """
        out = []
        for row in self.rows:
            example_id = row.get("id")
            if ids is not None and example_id not in ids:
                continue
            out.append({
                "id": example_id,
                "title": row.get("title", example_id),
                "duration_s": row.get("duration_s"),
                "split": row.get("split"),
                "cpwer": row.get("cpwer"),
                "cpcer": row.get("cpcer"),
                "n_overlap_regions": row.get("n_overlap_regions"),
                "gt_available": bool(row.get("gt")),
                # Additions beyond API.md's row keys (v1.2 draft), all passed
                # through verbatim from the manifest: the acoustic-complexity
                # stratum, the full per-fragment metric set, and whether the GT
                # tiers were relabelled to match the pipeline's A/B. All three
                # are a few bytes, so unlike `gt` they ride along in light mode
                # too — the gallery filters and sorts on them.
                "stratum": row.get("stratum"),
                "metrics": row.get("metrics"),
                "gt_swapped": bool(row.get("gt_swapped")),
                # The GT transcript itself: the design puts GT "only here", and
                # the page has no other source for it. Omitted (null) in light
                # mode — the gallery needs only gt_available, and shipping 141
                # GTs cost ~423 KB.
                "gt": None if light else row.get("gt"),
                "job_like": None if light else self.job_like(example_id),
            })
        return out

    def warm(self) -> None:
        for row in self.rows:
            self.job_like(row.get("id"))


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app(
    config_path: str | Path | None = None,
    runner=None,
    jobs_root: str | Path | None = None,
    skip_preflight: bool = False,
    examples_manifest: str | Path | None = None,
    warm_examples: bool = False,
) -> FastAPI:
    """Build the app.

    - `config_path`  pipeline YAML (default: the shipped v41_merge arm). Loaded
      once, and only when `runner` is not injected.
    - `runner`       the `Runner` seam. ``None`` builds the real
      `PipelineRunner` after preflight; tests pass a fake.
    - `jobs_root`    where job directories live (default ``$WEBAPP_JOBS_ROOT``).
    - `skip_preflight`  skip `check_preflight` — for a dev/offline start where a
      backend's env var is deliberately absent. Never the default.
    - `warm_examples`  precompute the gallery payloads in a background thread.
    """
    jobs_root = Path(jobs_root) if jobs_root else DEFAULT_JOBS_ROOT
    config = None
    if runner is None:
        # Imported here, not at module import: with an injected runner the
        # webapp never touches asr_pipeline (nor torch) at all.
        from asr_pipeline.config import load_pipeline_config_from_yaml
        from asr_pipeline.preflight import check_preflight

        config = load_pipeline_config_from_yaml(
            str(config_path or DEFAULT_CONFIG_PATH)
        )
        if not skip_preflight:
            check_preflight(config)
        runner = PipelineRunner(config)

    service = JobService(jobs_root, runner)
    service.rebuild_from_disk()
    service.start()

    examples = ExamplesLibrary(
        Path(examples_manifest) if examples_manifest else DEFAULT_EXAMPLES_MANIFEST
    )
    if warm_examples and examples.rows:
        threading.Thread(
            target=examples.warm, name="webapp-examples-warm", daemon=True
        ).start()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # The worker is already running (started above, so a TestClient used
        # without its context manager still processes jobs); the lifespan exists
        # to shut it down cleanly on Ctrl-C.
        yield
        service.stop(timeout=2.0)

    app = FastAPI(
        title="asr_pipeline showcase", docs_url=None, redoc_url=None,
        lifespan=lifespan,
    )
    app.state.service = service
    app.state.config = config
    app.state.examples = examples

    static_dir = _HERE / "static"
    static_dir.mkdir(exist_ok=True)

    class _NoCacheStatic(StaticFiles):
        """Static files with ``Cache-Control: no-cache``.

        Without it, browsers apply heuristic freshness to the ES modules and a
        plain F5 can run a stale module against a new server (Chrome only
        revalidates the document itself). ``no-cache`` means "revalidate every
        time", not "don't cache": unchanged files still come back as 304s via
        ETag/Last-Modified, so the cost is one conditional request per file.
        """

        def file_response(self, *args, **kwargs):
            response = super().file_response(*args, **kwargs)
            response.headers["Cache-Control"] = "no-cache"
            return response

    app.mount("/static", _NoCacheStatic(directory=str(static_dir)), name="static")
    templates = Jinja2Templates(directory=str(_HERE / "templates"))

    @app.exception_handler(RequestValidationError)
    def _validation_error(request: Request, exc: RequestValidationError):
        """A malformed submission is a 400 with a reason, per API.md.

        FastAPI's default for a missing/!multipart body is 422; the upload route
        is contractually a 400, so it is translated here (and only here).
        """
        if request.url.path == "/api/jobs" and request.method == "POST":
            return JSONResponse(
                {"detail": "Brak pliku audio (pole `file`) lub `source_example`."},
                status_code=400,
            )
        return JSONResponse({"detail": exc.errors()}, status_code=422)

    # -- HTML ------------------------------------------------------------
    @app.get("/")
    def index(request: Request):
        return templates.TemplateResponse(
            request, "index.html", {"jobs_api": "/api/jobs"}
        )

    @app.get("/j/{job_id}")
    def job_page(request: Request, job_id: str):
        if service.get(job_id) is None:
            # HTML route -> HTML 404 (a browser hitting a stale /j/ link should
            # not see raw JSON); the JSON API routes keep their JSON 404s.
            return templates.TemplateResponse(
                request,
                "404.html",
                {"message": "Nie ma takiego zadania."},
                status_code=404,
            )
        return templates.TemplateResponse(
            request,
            "job.html",
            {
                "job_id": job_id,
                "poll_url": f"/api/jobs/{job_id}",
                "log_url": f"/api/jobs/{job_id}/log",
            },
        )

    @app.get("/examples")
    def examples_page(request: Request):
        return templates.TemplateResponse(
            request, "examples.html", {"examples_api": "/api/examples"}
        )

    # -- JSON: jobs ------------------------------------------------------
    @app.post("/api/jobs")
    async def create_job(
        file: Optional[UploadFile] = File(None),
        source_example: Optional[str] = Form(None),
    ):
        job_id = uuid.uuid4().hex[:12]
        job_dir = service.job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        target = service.mixture_path(job_id)
        try:
            if source_example:
                filename = _seed_from_example(examples, source_example, target)
            elif file is not None and file.filename:
                filename = await _seed_from_upload(file, job_dir, target)
            else:
                raise HTTPException(400, "Brak pliku audio (pole `file`).")
            duration = _probe_duration(target)
            if duration is None:
                raise HTTPException(
                    400, "Nie udało się odczytać pliku audio po konwersji."
                )
        except HTTPException:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise
        except Exception as exc:                       # noqa: BLE001
            shutil.rmtree(job_dir, ignore_errors=True)
            raise HTTPException(400, f"Nie udało się przyjąć pliku: {exc}")

        job = new_job(job_id, filename, duration)
        service.write_job_sidecar(job)
        service.register(job)
        service.enqueue(job_id)
        return {"job_id": job_id}

    @app.get("/api/jobs")
    def list_jobs(limit: int = Query(20, ge=1, le=200)):
        """Recent-jobs list for the home page.

        Addition beyond API.md's JSON section: `GET /` is specified to show a
        recent-jobs list and the contract defines no endpoint that serves one.
        Compact rows only — the full state stays at `/api/jobs/{id}`.
        """
        return service.recent(limit)

    @app.delete("/api/jobs")
    def clear_jobs():
        """Delete every finished job — registry entry and files on disk.

        Addition beyond API.md's JSON section (v1.2 draft): the demo machine
        accumulates job directories with no way to tidy them from the UI. Only
        terminal (done/failed) jobs go; anything queued or running is left
        untouched and counted as skipped, as is a directory that refuses to be
        removed.
        """
        return service.clear_terminal()

    @app.get("/api/jobs/{job_id}")
    def job_state(job_id: str):
        payload = service.snapshot(job_id)
        if payload is None:
            raise HTTPException(404, "Nie ma takiego zadania.")
        return JSONResponse(payload)

    @app.get("/api/jobs/{job_id}/log")
    def job_log(job_id: str, offset: int = Query(0, ge=0)):
        payload = service.log_tail(job_id, offset)
        if payload is None:
            raise HTTPException(404, "Nie ma takiego zadania.")
        return payload

    @app.get("/api/jobs/{job_id}/files/{name}")
    def job_file(job_id: str, name: str):
        if service.get(job_id) is None:
            raise HTTPException(404, "Nie ma takiego zadania.")
        path = _resolve_whitelisted(
            name,
            service.pipeline_dir(job_id),
            service.mixture_path(job_id),
            service.job_dir(job_id),
        )
        if path is None:
            raise HTTPException(404, "Nie ma takiego pliku.")
        return FileResponse(path)

    # -- JSON: examples --------------------------------------------------
    @app.get("/api/examples")
    def list_examples(
        ids: Optional[str] = Query(None, description="Comma-separated example ids."),
        light: bool = Query(False, description="Omit job_like (index-only rows)."),
    ):
        """Additions beyond API.md: the optional `ids` / `light` query knobs.

        Bare `GET /api/examples` is exactly the contract; the knobs let the
        gallery render an index instantly and pull each example's payload on
        demand instead of downloading the whole ~6 MB frozen set up front.
        """
        selected = [i.strip() for i in ids.split(",") if i.strip()] if ids else None
        return examples.listing(selected, light)

    @app.get("/api/examples/{example_id}/files/{name}")
    def example_file(example_id: str, name: str):
        """Same whitelist as job files.

        Addition beyond API.md: the gallery needs playable audio and downloadable
        transcripts, and the contract defines file serving only under
        `/api/jobs/...`, which frozen examples are not.
        """
        pdir = examples.pipeline_dir(example_id)
        if pdir is None:
            raise HTTPException(404, "Nie ma takiego przykładu.")
        mixture = examples.mixture_path(example_id)
        root = mixture.parent if mixture is not None else pdir
        path = _resolve_whitelisted(name, pdir, mixture, root)
        if path is None:
            raise HTTPException(404, "Nie ma takiego pliku.")
        return FileResponse(path)

    return app


async def _seed_from_upload(file: UploadFile, job_dir: Path, target: Path) -> str:
    """Persist an uploaded file as `target` (16 kHz mono WAV). Returns its name."""
    suffix = Path(file.filename or "").suffix.lower()
    raw = job_dir / f"_upload{suffix or '.bin'}"
    data = await file.read()
    if not data:
        raise HTTPException(400, "Przesłany plik jest pusty.")
    raw.write_bytes(data)
    try:
        if suffix == ".wav" and _probe_duration(raw) is not None:
            os.replace(raw, target)
        else:
            # Non-WAV (or a .wav soundfile refuses) -> ffmpeg, loud on failure.
            _ffmpeg_to_wav(raw, target)
    finally:
        raw.unlink(missing_ok=True)
    return file.filename or target.name


def _seed_from_example(
    examples: "ExamplesLibrary", example_id: str, target: Path
) -> str:
    """Copy an example's mixture as a normal new job ("uruchom ponownie").

    No GT and no scores travel with it — the copy is an ordinary upload from the
    pipeline's point of view (design §5.5).
    """
    mixture = examples.mixture_path(example_id)
    if mixture is None or not mixture.exists():
        raise HTTPException(400, f"Nie znaleziono przykładu {example_id!r}.")
    shutil.copy(mixture, target)
    return mixture.name


# ---------------------------------------------------------------------------
# `uvicorn webapp.app:app` entry point (see webapp/run.sh)
# ---------------------------------------------------------------------------
#
# Built lazily via PEP 562: uvicorn's `getattr(module, "app")` constructs it
# (config load + preflight + worker thread), while `from webapp.app import
# create_app` stays a pure import — no config, no preflight, no GPU. That is
# what lets the tests import this module on a machine with none of the
# pipeline's env vars set.
_APP: Optional[FastAPI] = None


def __getattr__(name: str):
    global _APP
    if name == "app":
        if _APP is None:
            # Warm the gallery in the background at server start so the first
            # visitor to /examples never waits on 141 payload builds.
            _APP = create_app(warm_examples=True)
        return _APP
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
