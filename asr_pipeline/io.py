"""Audio I/O, artefact directory handling, and the per-recording output writer.

- ``load_audio_as_mono``: input-side, used by `Pipeline.load_audio`.
- ``ensure_artifact_dir``: per-stage spill directory (the legacy intermediate
  spill mechanism — one file per output, all artefacts in one flat dir).
- ``write_pipeline_outputs``: per-recording layout (NEW, used by the eval
  module and the batch script). Stable schema, dataset-agnostic.
"""

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.functional as AF

from asr_pipeline.context import PipelineContext
from asr_pipeline.debug_log import dlog
from asr_pipeline.transcript_format import (
    format_transcript,
    to_jsonable,
    write_eaf_from_whisper_results,
)


def _log(msg: str) -> None:
    """I/O progress log. Stays on stdout so long-file loads show progress."""
    dlog("io", msg)


def load_audio_as_mono(audio_path: str, target_sr: int = 16_000) -> np.ndarray:
    """Load any audio file, downmix to mono, resample to `target_sr`.

    Returns a 1-D float32 numpy array. Ported from POC cell 10.

    Progress is logged step-by-step so that a hang on long recordings
    (decode vs downmix vs resample) is immediately distinguishable. Reading
    from `/mnt/c/...` paths inside WSL is much slower than from the WSL
    filesystem — if the decode step takes minutes, copying the file into
    `/tmp` first is worthwhile.
    """
    path = Path(audio_path)
    try:
        size_mb = path.stat().st_size / (1024 * 1024)
        _log(f"loading {audio_path} ({size_mb:.1f} MB)")
    except OSError as e:
        _log(f"loading {audio_path} (stat failed: {e})")

    t0 = time.perf_counter()
    waveform, sr = torchaudio.load(audio_path)
    _log(
        f"  torchaudio.load: {waveform.shape[0]} ch @ {sr} Hz, "
        f"{waveform.shape[1]} samples ({waveform.shape[1]/sr:.2f}s) "
        f"in {time.perf_counter()-t0:.2f}s"
    )

    if waveform.shape[0] > 1:
        t0 = time.perf_counter()
        waveform = waveform.mean(dim=0, keepdim=True)
        _log(f"  downmixed to mono in {time.perf_counter()-t0:.2f}s")

    if sr != target_sr:
        t0 = time.perf_counter()
        waveform = AF.resample(waveform, sr, target_sr)
        _log(
            f"  resampled {sr} -> {target_sr} Hz "
            f"({waveform.shape[1]} samples) in {time.perf_counter()-t0:.2f}s"
        )

    out = waveform.squeeze(0).numpy().astype(np.float32)
    _log(f"  done: {len(out)/target_sr:.2f}s mono @ {target_sr} Hz")
    return out


def ensure_artifact_dir(artifact_dir: Optional[str]) -> Optional[Path]:
    """Create the artefact directory if a path is provided. Return as Path."""
    if artifact_dir is None:
        return None
    path = Path(artifact_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Per-recording output writer
# ---------------------------------------------------------------------------


def write_pipeline_outputs(
    ctx: PipelineContext,
    out_dir: str | Path,
    config_snapshot: Optional[dict] = None,
    subdir_name: str = "pipeline",
) -> Path:
    """Materialise the pipeline outputs as a stable per-recording layout.

    ``subdir_name`` lets the caller distinguish ablation runs (``"pipeline"`` for
    the full pipeline, ``"pipeline_nosep"`` / ``"pipeline_noenh"`` for the
    --no-separation / --no-enhancement variants).

    Writes to ``<out_dir>/<subdir_name>/``::

        diarization.json         stage-1 turns on the mixture timeline
                                 (the DER hypothesis for the eval module)
        routing.json             stage-2 overlap regions (debugging aid)
        stream_<label>.wav       one per assembled speaker stream (Stage 4)
        transcript_<label>.txt   decimal-seconds format, no speaker header
                                 (parse_gt_txt-compatible)
        transcript_<label>.json  full Whisper result for the speaker stream
        transcript_mixture.txt   single-stream baseline (only if produced)
        transcript_mixture.json
        metadata.json            speaker map + counts + (optional) config

    ``<label>`` is the A / B letter assigned by ECAPA anchoring in Stage 4
    (``ctx.spk_to_label``). Stages that didn't run are silently skipped (we
    write what's populated, not what we expected to be populated).

    ``config_snapshot`` is an optional dict (typically ``dataclasses.asdict(cfg)``)
    embedded in ``metadata.json`` for reproducibility.

    Returns the path of the ``pipeline/`` subdirectory.
    """
    out_dir = Path(out_dir).expanduser()
    pipeline_dir = out_dir / subdir_name
    pipeline_dir.mkdir(parents=True, exist_ok=True)

    if ctx.diarization is not None:
        turns = [
            {
                "speaker": str(row.speaker),
                "start": float(row.start),
                "end": float(row.end),
            }
            for row in ctx.diarization.segments_df.itertuples()
        ]
        with open(pipeline_dir / "diarization.json", "w") as f:
            json.dump(
                {
                    "turns": turns,
                    "total_duration_s": float(ctx.diarization.total_duration_s),
                },
                f,
                indent=2,
            )

    if ctx.overlap_regions is not None:
        with open(pipeline_dir / "routing.json", "w") as f:
            json.dump(
                {
                    "overlap_regions": [
                        {"start": float(s), "end": float(e)}
                        for s, e in ctx.overlap_regions
                    ],
                },
                f,
                indent=2,
            )

    for spk, audio in ctx.assembled.items():
        label = ctx.spk_to_label.get(spk, spk)
        sf.write(
            pipeline_dir / f"stream_{label}.wav",
            audio.astype(np.float32),
            ctx.sample_rate,
        )

    # Per-speaker text + JSON.
    per_speaker_results: dict[str, dict] = {}
    for spk, result in (ctx.transcripts or {}).items():
        label = ctx.spk_to_label.get(spk, spk)
        per_speaker_results[label] = result
        with open(
            pipeline_dir / f"transcript_{label}.txt", "w", encoding="utf-8"
        ) as f:
            f.write(format_transcript(result))
            f.write("\n")
        with open(
            pipeline_dir / f"transcript_{label}.json", "w", encoding="utf-8"
        ) as f:
            json.dump(to_jsonable(result), f, indent=2, ensure_ascii=False)

    # EAF (ELAN annotation format) — one multi-tier file referencing the
    # source audio. Skipped when we don't have a per-speaker transcript or
    # when input_path is unknown (no audio to reference).
    if per_speaker_results and ctx.input_path is not None:
        # Pull the per-speaker tier locale from one of the result dicts —
        # whisperx writes ``language`` at the top of every result; fall back
        # to ``"pl"`` to match the project default.
        eaf_locale = "pl"
        for r in per_speaker_results.values():
            lang = (r or {}).get("language")
            if isinstance(lang, str) and lang:
                eaf_locale = lang
                break
        n_annotations = write_eaf_from_whisper_results(
            per_speaker_results,
            media_path=ctx.input_path,
            eaf_path=pipeline_dir / "annotation.eaf",
            locale=eaf_locale,
        )
        if n_annotations:
            _log(
                f"wrote annotation.eaf ({n_annotations} annotations across "
                f"{len(per_speaker_results)} tiers, locale={eaf_locale})"
            )

    if ctx.mixture_transcript:
        with open(pipeline_dir / "transcript_mixture.txt", "w", encoding="utf-8") as f:
            f.write(format_transcript(ctx.mixture_transcript))
            f.write("\n")
        with open(pipeline_dir / "transcript_mixture.json", "w", encoding="utf-8") as f:
            json.dump(
                to_jsonable(ctx.mixture_transcript), f, indent=2, ensure_ascii=False
            )

    meta: dict = {
        "input_path": str(ctx.input_path) if ctx.input_path else None,
        "sample_rate": ctx.sample_rate,
        "speakers": list(ctx.speakers),
        "spk_to_label": dict(ctx.spk_to_label),
        "weak_anchor": ctx.weak_anchor,
        "total_duration_s": (
            float(ctx.diarization.total_duration_s)
            if ctx.diarization is not None
            else None
        ),
        "n_overlap_regions": len(ctx.overlap_regions or []),
        "n_overlap_separated": len(ctx.overlap_separated or []),
    }
    if config_snapshot is not None:
        meta["config"] = config_snapshot
    with open(pipeline_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    _log(f"wrote pipeline outputs to {pipeline_dir}")
    return pipeline_dir
