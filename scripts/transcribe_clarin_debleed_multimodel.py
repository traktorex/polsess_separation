"""Multi-backend transcription on enhanced CLARIN debleed channels.

For each WAV in `debleed_enhanced/`:

  1. Run pyannote speaker diarization once.
  2. Identify the loudest speaker on this channel (the others are residue).
  3. Build a binary mask keeping only the main speaker's turns.
  4. For each of the 5 stage-5 backend configs, load the model, run
     `backend.transcribe(gated_audio)`, unload, dump the transcript.

Output layout (mirrors `whisper_test/` from the earlier 90-s comparison)::

    <gotowy>/whisper_test_debleed/<variant>/<stem>.txt

`<variant>` ∈ {whisper-large-v2, whisper-large-v3, whisperx-large-v2,
whisperx-large-v3, whisperx-bardsai-large-v2}.

Uses the same `_WhisperBackend` / `_WhisperXBackend` classes as the live
pipeline (asr_pipeline.stages.transcription), so the comparison reflects
what stage 5 would actually emit in production. No special anti-hallucination
hyperparameter tuning is applied — that would change the comparison from
"which backend works best" to "which backend works best with hand-tuning".

Idempotent: per-variant outputs that already exist are skipped.

Usage:
    source venv/bin/activate
    python scripts/transcribe_clarin_debleed_multimodel.py
    python scripts/transcribe_clarin_debleed_multimodel.py --only 442dd69e
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from asr_pipeline.config import TranscriptionConfig, load_pipeline_config_from_yaml
from asr_pipeline.stages.transcription import (
    _WhisperBackend,
    _WhisperXBackend,
    _format_transcript,
)


GOTOWY = Path("/home/user/datasets/clarin_gotowy/gotowy")
ENH = GOTOWY / "debleed_enhanced"
OUT_ROOT = GOTOWY / "whisper_test_debleed"
PIPELINE_CONFIG = (
    Path(__file__).resolve().parent.parent / "asr_pipeline" / "configs" / "default.yaml"
)


# (variant_dir_name, backend, model_name) — align_model only matters for whisperx.
VARIANTS: list[tuple[str, str, str]] = [
    ("whisper-large-v2",          "whisper",  "large-v2"),
    ("whisper-large-v3",          "whisper",  "large-v3"),
    ("whisperx-large-v2",         "whisperx", "large-v2"),
    ("whisperx-large-v3",         "whisperx", "large-v3"),
    ("whisperx-bardsai-large-v2", "whisperx", "bardsai/whisper-large-v2-pl-v2"),
]
ALIGN_MODEL = "jonatasgrosman/wav2vec2-large-xlsr-53-polish"

DIAR_MIN_SPEAKERS = 1
DIAR_MAX_SPEAKERS = 2


def _load_diarizer(model_id: str, hf_token: str, device: torch.device):
    from pyannote.audio import Pipeline as PyannotePipeline
    return PyannotePipeline.from_pretrained(model_id, token=hf_token).to(device)


def _main_speaker_mask(audio: np.ndarray, sr: int, diarizer):
    """Diarize the channel; build a mask keeping only the highest-energy speaker.

    Returns ``(mask, rms_by_speaker, main_speaker_label)``.
    """
    waveform = torch.from_numpy(audio).unsqueeze(0)
    diar = diarizer(
        {"waveform": waveform, "sample_rate": sr},
        min_speakers=DIAR_MIN_SPEAKERS,
        max_speakers=DIAR_MAX_SPEAKERS,
    )
    diar = diar.speaker_diarization if hasattr(diar, "speaker_diarization") else diar

    spk_segments: dict[str, list[tuple[float, float]]] = {}
    for turn, _, spk in diar.itertracks(yield_label=True):
        spk_segments.setdefault(spk, []).append((turn.start, turn.end))
    if not spk_segments:
        return np.zeros(len(audio), dtype=np.float32), {}, None

    rms_by_spk: dict[str, float] = {}
    for spk, segs in spk_segments.items():
        sq_sum, n = 0.0, 0
        for s, e in segs:
            i = max(0, int(s * sr))
            j = min(len(audio), int(e * sr))
            if j > i:
                seg = audio[i:j]
                sq_sum += float((seg ** 2).sum())
                n += (j - i)
        rms_by_spk[spk] = math.sqrt(sq_sum / max(n, 1))

    main_spk = max(rms_by_spk, key=rms_by_spk.get)
    mask = np.zeros(len(audio), dtype=np.float32)
    for s, e in spk_segments[main_spk]:
        i = max(0, int(s * sr))
        j = min(len(audio), int(e * sr))
        mask[i:j] = 1.0
    return mask, rms_by_spk, main_spk


def _make_backend(backend_name: str, model_name: str, base_cfg: TranscriptionConfig):
    cfg = replace(
        base_cfg,
        backend=backend_name,
        model_name=model_name,
        align_model_name=ALIGN_MODEL,
    )
    if backend_name == "whisper":
        return _WhisperBackend(cfg)
    if backend_name == "whisperx":
        return _WhisperXBackend(cfg)
    raise ValueError(backend_name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="442dd69e",
                    help="Substring match against the stem. Default: 442dd69e "
                         "(only the recording the user has GT for).")
    ap.add_argument("--out", default=str(OUT_ROOT),
                    help="Output root. Per-variant subdirs are created under here.")
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    if not ENH.is_dir():
        raise SystemExit(f"debleed_enhanced not found: {ENH}")

    inputs = sorted(p for p in ENH.iterdir() if p.suffix == ".wav")
    if args.only:
        inputs = [p for p in inputs if args.only in p.stem]
    print(f"[multimodel] {len(inputs)} wav file(s) match --only {args.only!r}")
    if not inputs:
        return

    # Pipeline config gives us diarization knobs + the base TranscriptionConfig
    # (language, initial_prompt, word_timestamps) we want to keep constant
    # across variants.
    cfg_full = load_pipeline_config_from_yaml(str(PIPELINE_CONFIG))
    base_tc = cfg_full.transcription
    diar_model_id = cfg_full.diarization.model_id
    hf_token = cfg_full.diarization.hf_token
    if not hf_token:
        raise SystemExit("HF token empty in pipeline config; set $HF_TOKEN or edit default.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # 1) Diarize each input once. Cache masked audio per (stem, audio).
    # ------------------------------------------------------------------
    print(f"[multimodel] loading pyannote diarization ({diar_model_id}) on {device}")
    t0 = time.perf_counter()
    diarizer = _load_diarizer(diar_model_id, hf_token, device)
    print(f"[multimodel] diarizer ready in {time.perf_counter()-t0:.1f}s")

    masked: dict[str, np.ndarray] = {}
    for wav in inputs:
        audio, sr = sf.read(str(wav), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1).astype(np.float32)
        if sr != 16_000:
            raise RuntimeError(f"expected 16 kHz, got {sr} Hz: {wav}")

        t_d = time.perf_counter()
        mask, rms_by_spk, main_spk = _main_speaker_mask(audio, sr, diarizer)
        elapsed_d = time.perf_counter() - t_d
        if main_spk is None:
            print(f"[multimodel] {wav.name}: empty diarization, will dump empty transcripts")
            masked[wav.stem] = np.zeros_like(audio)
            continue
        rms_summary = ", ".join(
            f"{s}={rms_by_spk[s]:.4f}{'*' if s == main_spk else ''}"
            for s in sorted(rms_by_spk, key=lambda x: -rms_by_spk[x])
        )
        masked[wav.stem] = (audio * mask).astype(np.float32)
        print(
            f"[multimodel] {wav.name}: diar {elapsed_d:.1f}s, main={main_spk} "
            f"(RMS {rms_summary}), kept {float(mask.mean())*100:.1f}% of audio"
        )

    # Free pyannote before loading Whisper variants.
    del diarizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 2) For each variant: load, transcribe each masked audio, unload.
    # ------------------------------------------------------------------
    for variant_dir, backend_name, model_name in VARIANTS:
        variant_out = out_root / variant_dir
        variant_out.mkdir(parents=True, exist_ok=True)

        outputs_needed = []
        for wav in inputs:
            txt_out = variant_out / f"{wav.stem}.txt"
            if txt_out.exists():
                print(f"[multimodel] {variant_dir}/{wav.stem}.txt — skip (exists)")
                continue
            outputs_needed.append((wav.stem, txt_out))
        if not outputs_needed:
            continue

        print(f"\n[multimodel] === {variant_dir} ({backend_name} / {model_name}) ===")
        t_l = time.perf_counter()
        backend = _make_backend(backend_name, model_name, base_tc)
        backend.load(device)
        print(f"[multimodel]   loaded in {time.perf_counter()-t_l:.1f}s")

        for stem, txt_out in outputs_needed:
            audio = masked[stem]
            t_t = time.perf_counter()
            result = backend.transcribe(audio)
            elapsed_t = time.perf_counter() - t_t
            txt_out.write_text(_format_transcript(result) + "\n", encoding="utf-8")
            n_segs = len(result.get("segments") or [])
            print(f"[multimodel]   {stem}: {n_segs} segs ({elapsed_t:.1f}s) → {txt_out}")

        backend.unload()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n[multimodel] done.")


if __name__ == "__main__":
    main()
