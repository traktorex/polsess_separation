"""Read-side rendering helpers: waveform peaks + result-payload assembly.

Everything here reads an already-written pipeline output directory (the
`asr_pipeline.io.write_pipeline_outputs` layout) and turns it into the
``JobState.result`` object defined in `webapp/API.md`. It never runs the
pipeline and never writes into an output directory.

The same assembly powers the examples gallery, whose frozen results live under
``<fragment>/sweep/v41_merge/`` instead of ``<job>/pipeline/`` — hence
`build_result` takes the pipeline directory and the mixture path explicitly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf

# Waveform resolution. Design §5.3 pins v1 to the review page's proven simple
# peak envelope; the JSON shape ({"mixture": [...], "A": [...]}) leaves room for
# additive per-bucket min/max/RMS fields later without breaking clients.
DEFAULT_BUCKETS = 800


def peaks(wav_path: str | Path, buckets: int = DEFAULT_BUCKETS) -> List[int]:
    """Downsampled peak envelope of a mono wav -> `buckets` ints in 0..100.

    One value per time bucket = max |sample| in that bucket, scaled to the
    stream's own peak (so a quiet stream still draws). Mirrors the review page's
    ``_peaks`` (`scripts/build_review_page.py`) at a higher bucket count.

    An unreadable/absent file yields all zeros rather than raising — a missing
    waveform must never take the results page down.
    """
    if buckets <= 0:
        raise ValueError(f"buckets must be positive, got {buckets}")
    try:
        x, _ = sf.read(str(wav_path), dtype="float32", always_2d=False)
    except (RuntimeError, OSError):
        return [0] * buckets
    if x.ndim > 1:
        x = x.mean(axis=1)
    n = len(x)
    if n == 0:
        return [0] * buckets
    ax = np.abs(x)
    peak = float(ax.max()) or 1.0
    edges = (np.arange(buckets + 1) * n / buckets).astype(int)
    return [
        int(round(float(ax[edges[i]:edges[i + 1]].max()) / peak * 100))
        if edges[i + 1] > edges[i] else 0
        for i in range(buckets)
    ]


def _read_json(path: Path) -> Optional[dict]:
    """Parse a JSON file, or ``None`` if it is missing/corrupt."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def provenance_line(config: Optional[dict]) -> str:
    """Compact one-line provenance from a ``metadata.json`` config snapshot.

    Shape per API.md: ``enh=... - sep=... - asr=...``. A disabled stage reads
    ``off``; an absent config snapshot yields ``"(no config recorded)"``.
    """
    if not config:
        return "(no config recorded)"

    enh = config.get("enhancement") or {}
    sep = config.get("separation") or {}
    asr = config.get("transcription") or {}

    enh_txt = enh.get("backend", "?") if enh.get("enabled", True) else "off"

    if not sep.get("enabled", True):
        sep_txt = "off"
    else:
        backend = sep.get("separator_backend", "repo")
        ckpt = str(sep.get("checkpoint_path", "") or "")
        # For the repo backend the checkpoint directory name is the model
        # identity (e.g. mossformer2_matched_128k_final_42_e46); external
        # backends put an HF id / model name in the same field.
        name = Path(ckpt).parent.name if backend == "repo" and ckpt else ckpt
        sep_txt = name or backend
        if backend != "repo":
            sep_txt = f"{backend}:{sep_txt}" if sep_txt else backend

    if not asr.get("enabled", True):
        asr_txt = "off"
    else:
        asr_txt = " ".join(
            p for p in (asr.get("backend"), asr.get("model_name")) if p
        ) or "?"

    return f"enh={enh_txt} · sep={sep_txt} · asr={asr_txt}"


def _segments_from_result(label: str, result: Optional[dict]) -> dict:
    """One WhisperX result dict -> the API's tier-shaped transcript object.

    Segments get **stable ids** (``A-0000``, ``A-0001``, ...) so a future
    ELAN-style editor can address them (design §5.3 (a)); WhisperX itself emits
    no ids, and positional indices are stable for a frozen output directory.
    """
    segments = []
    for i, seg in enumerate((result or {}).get("segments") or []):
        words = []
        for w in seg.get("words") or []:
            words.append({
                "word": w.get("word", ""),
                "start": w.get("start"),
                "end": w.get("end"),
                "score": w.get("score"),
            })
        segments.append({
            "id": f"{label}-{i:04d}",
            "start": seg.get("start"),
            "end": seg.get("end"),
            "text": seg.get("text", ""),
            "words": words,
        })
    return {"segments": segments}


def _stage_timings(metadata: dict, pipeline_dir: Path) -> List[dict]:
    """Per-stage ``{stage, load_s, run_s}`` rows, from metadata or run_meta.

    `metadata.json` carries them only when the run was instrumented; older trees
    (and runs whose caller passed no ``on_event``) have neither — an empty list
    then, never a fabricated one.
    """
    rows = metadata.get("stage_timings")
    if not rows:
        run_meta = _read_json(pipeline_dir / "run_meta.json") or {}
        rows = run_meta.get("stages")
    out = []
    for row in rows or []:
        out.append({
            "stage": row.get("stage", "?"),
            "load_s": row.get("load_s"),
            "run_s": row.get("run_s"),
        })
    return out


def build_result(
    pipeline_dir: str | Path,
    mixture_path: Optional[str | Path],
    files_url_prefix: str,
    *,
    buckets: int = DEFAULT_BUCKETS,
) -> Optional[dict]:
    """Assemble the ``JobState.result`` payload from one pipeline output dir.

    `pipeline_dir` is the directory `write_pipeline_outputs` wrote
    (``<job>/pipeline/`` for a webapp job, ``<frag>/sweep/v41_merge/`` for a
    frozen example). `mixture_path` is the input recording, which lives one level
    up and is served under the logical name ``mixture.wav``. `files_url_prefix`
    is the URL the whitelisted file route is mounted at, e.g.
    ``/api/jobs/<id>/files``.

    Returns ``None`` when ``metadata.json`` is absent — i.e. the run did not
    finish, so there is no result to show (no partial-success framing, SCOPE §4).
    """
    pipeline_dir = Path(pipeline_dir)
    metadata = _read_json(pipeline_dir / "metadata.json")
    if metadata is None:
        return None

    prefix = files_url_prefix.rstrip("/")
    spk_to_label: Dict[str, str] = dict(metadata.get("spk_to_label") or {})
    speakers = list(metadata.get("speakers") or [])
    # Stream labels in a stable, speaker-ordered sequence; no hardcoded 2.
    labels = [spk_to_label.get(spk, spk) for spk in speakers] or sorted(
        p.stem[len("stream_"):] for p in pipeline_dir.glob("stream_*.wav")
    )

    diarization = _read_json(pipeline_dir / "diarization.json") or {}
    routing = _read_json(pipeline_dir / "routing.json") or {}
    overlap_regions = routing.get("overlap_regions") or []
    overlap_total_s = sum(
        float(r.get("end", 0.0)) - float(r.get("start", 0.0)) for r in overlap_regions
    )

    transcripts: Dict[str, dict] = {}
    for label in labels:
        result = _read_json(pipeline_dir / f"transcript_{label}.json")
        if result is not None:
            transcripts[label] = _segments_from_result(label, result)
    mixture_result = _read_json(pipeline_dir / "transcript_mixture.json")
    if mixture_result is not None:
        transcripts["mixture"] = _segments_from_result("mixture", mixture_result)

    files: Dict[str, str] = {}
    if mixture_path is not None and Path(mixture_path).exists():
        files["mixture"] = f"{prefix}/mixture.wav"
    for label in labels:
        if (pipeline_dir / f"stream_{label}.wav").exists():
            files[f"stream_{label}"] = f"{prefix}/stream_{label}.wav"
        if (pipeline_dir / f"transcript_{label}.txt").exists():
            files[f"transcript_{label}_txt"] = f"{prefix}/transcript_{label}.txt"
    if (pipeline_dir / "annotation.eaf").exists():
        files["eaf"] = f"{prefix}/annotation.eaf"
    # Per-job spill sits beside the pipeline dir (API.md v1.1); frozen example
    # trees have no spill, so the key is simply absent there.
    if (pipeline_dir.parent / "spill" / "enhanced_full.wav").exists():
        files["enhanced_full"] = f"{prefix}/enhanced_full.wav"
    files["metadata"] = f"{prefix}/metadata.json"

    peaks_map: Dict[str, List[int]] = {}
    if mixture_path is not None and Path(mixture_path).exists():
        peaks_map["mixture"] = peaks(mixture_path, buckets)
    for label in labels:
        stream = pipeline_dir / f"stream_{label}.wav"
        if stream.exists():
            peaks_map[label] = peaks(stream, buckets)

    return {
        "speakers": speakers,
        "spk_to_label": spk_to_label,
        "weak_anchor": bool(metadata.get("weak_anchor", False)),
        "total_duration_s": metadata.get("total_duration_s"),
        "n_overlap_regions": metadata.get("n_overlap_regions", len(overlap_regions)),
        "overlap_total_s": overlap_total_s,
        "provenance": provenance_line(metadata.get("config")),
        "diarization": {
            "turns": diarization.get("turns") or [],
            # Hook 3 (extended diarization.json): present -> surface, absent -> null.
            "overlaps": diarization.get("overlaps"),
        },
        "routing": {"overlap_regions": overlap_regions},
        "transcripts": transcripts,
        "files": files,
        "peaks": peaks_map,
        "stage_timings": _stage_timings(metadata, pipeline_dir),
        # Hook 1 payload, verbatim when present.
        "assembly_diag": metadata.get("assembly_diag"),
        "diarization_diag": metadata.get("diarization_diag"),
    }
