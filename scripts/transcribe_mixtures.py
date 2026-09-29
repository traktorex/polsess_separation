"""Single-stream Whisper baseline on every raw mixture in the eval tree.

For each recording produces ``<recording>/pipeline/transcript_mixture.{txt,json}``
— the *no-pipeline* baseline the eval module reads for the ORC-WER row in the
Layer 3 ablation table.

Uses the same backend (WhisperX large-v2) as the per-speaker transcription so
the comparison is fair, but with ``word_timestamps=False``: segment-level
timestamps are enough for ORC-WER, and skipping forced alignment lets us
transcribe LibriCSS English without needing an English wav2vec2 model.

Per-dataset config:
- ``clarin``  → Polish (Whisper ``language=pl``, prompt ``"Rozmowa po polsku."``)
- ``libricss`` → English (``language=en``, no prompt)
- anything else → falls back to Polish (override with ``--language`` if needed)

Idempotent: skips recordings whose ``pipeline/transcript_mixture.txt`` already
exists, unless ``--force`` is passed. Loads the backend ONCE per language and
reuses it across recordings — the per-recording cost is just decode + Whisper.

Usage::

    python scripts/transcribe_mixtures.py                   # everything in the tree
    python scripts/transcribe_mixtures.py --dataset clarin  # one dataset
    python scripts/transcribe_mixtures.py --force           # re-transcribe
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Pre-empt the asr_pipeline.config HF_TOKEN validation — we don't use
# diarization here, but the dataclass default factory still tries to read
# the env var when other config classes are constructed by neighbouring
# imports. Set a placeholder if the user hasn't.
os.environ.setdefault("HF_TOKEN", "unused-for-mixture-transcription")

from asr_pipeline.config import TranscriptionConfig                     # noqa: E402
from asr_pipeline.eval.recordings import Recording, walk_eval_tree      # noqa: E402
from asr_pipeline.io import load_audio_as_mono                          # noqa: E402
from asr_pipeline.stages.transcription import _WhisperXBackend          # noqa: E402
from asr_pipeline.transcript_format import format_transcript, to_jsonable  # noqa: E402


def _config_for_dataset(dataset: str, override_language: str | None = None) -> TranscriptionConfig:
    """Pick language + prompt by dataset; override via ``--language``."""
    if override_language is not None:
        language = override_language
        prompt = ""
    elif dataset == "libricss":
        language = "en"
        prompt = ""
    else:
        language = "pl"
        prompt = "Rozmowa po polsku."
    return TranscriptionConfig(
        enabled=True,
        backend="whisperx",
        model_name="large-v2",
        language=language,
        initial_prompt=prompt,
        word_timestamps=False,        # segment-level only — skip wav2vec2 alignment
        align_model_name="",          # unused when word_timestamps=False
        transcribe_mixture=False,     # irrelevant; we drive the backend directly
    )


def _save(out_dir: Path, result: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "transcript_mixture.txt").write_text(
        format_transcript(result) + "\n", encoding="utf-8"
    )
    with open(out_dir / "transcript_mixture.json", "w", encoding="utf-8") as f:
        json.dump(to_jsonable(result), f, indent=2, ensure_ascii=False)


def _group_by_dataset(recordings: list[Recording]) -> dict[str, list[Recording]]:
    grouped: dict[str, list[Recording]] = {}
    for r in recordings:
        grouped.setdefault(r.dataset, []).append(r)
    return grouped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--eval-root", type=Path,
        default=Path.home() / "datasets" / "eval",
        help="Root of the eval tree.",
    )
    parser.add_argument(
        "--dataset", default=None,
        help="Restrict to one dataset (e.g. clarin | libricss). Default: all.",
    )
    parser.add_argument(
        "--language", default=None,
        help="Override the per-dataset language pick. e.g. --language pl",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-transcribe even if transcript_mixture.txt already exists.",
    )
    args = parser.parse_args()

    recordings = list(walk_eval_tree(args.eval_root, dataset=args.dataset))
    if not recordings:
        print(f"no recordings found under {args.eval_root}"
              + (f" (dataset={args.dataset})" if args.dataset else ""))
        return 1

    print(f"discovered {len(recordings)} recording(s) across "
          f"{len(set(r.dataset for r in recordings))} dataset(s)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    grouped = _group_by_dataset(recordings)

    total_done = 0
    total_skipped = 0
    total_audio_s = 0.0
    total_transcribe_s = 0.0

    for dataset, recs in grouped.items():
        cfg = _config_for_dataset(dataset, override_language=args.language)
        print(f"\n=== {dataset}: {len(recs)} recording(s) ===")
        print(f"  backend=whisperx model={cfg.model_name} language={cfg.language} "
              f"word_timestamps={cfg.word_timestamps} prompt={cfg.initial_prompt!r}")

        # Filter: skip those that already have the output, unless --force.
        if not args.force:
            pending = []
            for rec in recs:
                out_dir = rec.mixture_path.parent / "pipeline"
                if (out_dir / "transcript_mixture.txt").exists():
                    total_skipped += 1
                    print(f"  [skip] {rec.id}")
                    continue
                pending.append(rec)
        else:
            pending = recs

        if not pending:
            print("  (all up-to-date)")
            continue

        print(f"  loading backend...")
        t_load = time.perf_counter()
        backend = _WhisperXBackend(cfg)
        backend.load(device)
        print(f"  backend loaded in {time.perf_counter()-t_load:.1f}s")

        try:
            for rec in pending:
                t0 = time.perf_counter()
                audio = load_audio_as_mono(str(rec.mixture_path), target_sr=16_000)
                dur = len(audio) / 16_000
                result = backend.transcribe(audio)
                out_dir = rec.mixture_path.parent / "pipeline"
                _save(out_dir, result)
                elapsed = time.perf_counter() - t0
                rtf = dur / elapsed if elapsed > 0 else float("inf")
                n_segs = len(result.get("segments", []))
                print(
                    f"  [done] {rec.id}  "
                    f"audio={dur:6.1f}s  whisper={elapsed:5.1f}s  "
                    f"rtf={rtf:5.1f}×  segments={n_segs}"
                )
                total_done += 1
                total_audio_s += dur
                total_transcribe_s += elapsed
        finally:
            backend.unload()
            del backend
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print()
    print(f"transcribed:    {total_done}")
    print(f"skipped (had):  {total_skipped}")
    print(f"total audio:    {total_audio_s:.1f}s ({total_audio_s/60:.1f} min)")
    print(f"total whisper:  {total_transcribe_s:.1f}s ({total_transcribe_s/60:.1f} min)")
    if total_transcribe_s > 0:
        print(f"aggregate RTF:  {total_audio_s/total_transcribe_s:.1f}×")
    return 0


if __name__ == "__main__":
    sys.exit(main())
