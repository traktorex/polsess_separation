"""Fixed-stream WhisperX decode-knob arms (ASR-only swap).

Sibling of ``sweep_asr_prompt.py``: holds the best config's assembled per-speaker
streams CONSTANT and varies only WhisperX *decode* knobs — re-transcribing the
existing ``stream_{A,B}.wav`` under each knob setting and writing the result as a
new sweep config dir ``<frag>/sweep/<arm>/`` that ``rescore_stratified.py`` can
score directly. No pipeline re-run.

Scope: this fills the two decode knobs the big definitive sweep never touched —
WhisperX's own internal **`vad_onset`** (the decoder-side VAD that gates what
Whisper sees; distinct from the upstream `separation.vad_threshold` swept by the
`dr_vad*` arms) and **`compression_ratio_threshold`** — plus it completes the
barely-run `hallucination_silence_threshold` arm (`ah_hall2`, 3/23 on dev). The
other anti-hallucination / VAD-mask / no-speech / condition-on-previous knobs are
already settled in the definitive sweep; don't re-run them here.

``dk_base`` (the default-knob arm) reproduces the anchor's transcription on the
SAME PCM_16-quantised streams, so anchoring deltas to ``dk_base`` (not to the
full-precision pipeline run) cancels the stream-quantisation offset — the same
correction used for the prompt sweep.

Faithfulness: the base transcription config is loaded from the anchor's own saved
``metadata.json`` snapshot, so every other decode knob matches the best config
exactly; ``dataclasses.replace`` overrides only the swept field(s). The stage's
silence/duration gate (``_skip_transcription`` -> ``_empty_result``) is replicated.

Usage::

    source venv/bin/activate
    python scripts/sweep_asr_decode.py                       # dr_refineplus, dev, all arms
    python scripts/rescore_stratified.py --configs dk_base dk_vad_on06 dk_vad_on07 \
        dk_vad_on06_off45 dk_cr20 dk_cr18 dk_hst20 --anchor dk_base --split dev
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

import librosa
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from asr_pipeline.config import TranscriptionConfig  # noqa: E402
from asr_pipeline.stages.transcription import (  # noqa: E402
    _MIN_TRANSCRIBE_DURATION_S,
    _WhisperXBackend,
    _empty_result,
)
from asr_pipeline.transcript_format import format_transcript, to_jsonable  # noqa: E402

EVAL = Path.home() / "datasets/eval/clarin_fragments"
SPLIT_CSV = Path(__file__).resolve().parent.parent / "asr_pipeline/eval/clarin_split.csv"
SAMPLE_RATE = 16_000

# arm name -> TranscriptionConfig field overrides (applied via replace()).
# dk_base = no override = the default-knob anchor (PCM_16-matched).
ARMS = {
    "dk_base":            {},                                       # anchor
    # --- mitigation 1: WhisperX internal VAD (genuinely un-swept) -----------
    "dk_vad_on06":        {"vad_onset": 0.60},                      # stricter onset
    "dk_vad_on07":        {"vad_onset": 0.70},                      # stricter still
    "dk_vad_on06_off45":  {"vad_onset": 0.60, "vad_offset": 0.45},  # tighten both
    # --- mitigation 2: compression-ratio gate (un-swept) -------------------
    "dk_cr20":            {"compression_ratio_threshold": 2.0},     # trip fallback sooner
    "dk_cr18":            {"compression_ratio_threshold": 1.8},     # tighter
    # --- complete the barely-run hallucination-silence arm -----------------
    "dk_hst20":           {"hallucination_silence_threshold": 2.0},
}


def split_fragments(split: str) -> list[str]:
    with open(SPLIT_CSV) as f:
        return sorted(r["frag_id"] for r in csv.DictReader(f) if r["role"] == split)


def load_base_cfg(base: str, frags: list[str]) -> TranscriptionConfig:
    """Build the anchor's exact TranscriptionConfig from its saved snapshot."""
    for frag in frags:
        meta = EVAL / frag / "sweep" / base / "metadata.json"
        if not meta.exists():
            continue
        m = json.load(open(meta))

        def find(d, k="transcription"):
            if isinstance(d, dict):
                if k in d:
                    return d[k]
                for v in d.values():
                    r = find(v, k)
                    if r is not None:
                        return r
            return None
        t = find(m)
        if t:
            return TranscriptionConfig(**t)
    raise SystemExit(f"no metadata.json with a transcription block for base={base}")


def skip_stream(audio: np.ndarray, min_samples: int, silence_floor: float) -> bool:
    """Mirror TranscriptionStage._skip_transcription (len/peak gate)."""
    if len(audio) < min_samples or len(audio) == 0:
        return True
    return float(np.max(np.abs(audio))) < silence_floor


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="dr_refineplus",
                    help="config whose streams to reuse (default: the best config).")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--arms", nargs="*", default=list(ARMS),
                    help="subset of arm names to run (default: all).")
    ap.add_argument("--force", action="store_true",
                    help="re-transcribe even if the arm transcript already exists.")
    ap.add_argument("--fragments", nargs="*", default=None,
                    help="explicit fragment ids (default: all of --split).")
    args = ap.parse_args(argv)

    frags = args.fragments or split_fragments(args.split)
    base_cfg = load_base_cfg(args.base, frags)
    print(f"[cfg] base={args.base}  model={base_cfg.model_name}  "
          f"silence_floor={base_cfg.silence_floor}")
    print(f"[scope] {len(frags)} {args.split} fragment(s); arms={args.arms}")

    # Match the pipeline's determinism (Pipeline.__init__) so re-transcribing
    # identical audio is reproducible — same rationale as the prompt sweep.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    min_samples = int(SAMPLE_RATE * _MIN_TRANSCRIBE_DURATION_S)

    # Phase-major BY ARM: decode knobs are baked into asr_options/vad_options at
    # load(), so each arm needs its own model load. One load, all fragments.
    for arm in args.arms:
        overrides = ARMS[arm]
        cfg = replace(base_cfg, **overrides)
        backend = _WhisperXBackend(cfg)
        print(f"\n[{arm}] loading WhisperX {cfg.model_name}  overrides={overrides}")
        t0 = time.perf_counter()
        backend.load(device)
        print(f"[{arm}] loaded in {time.perf_counter()-t0:.1f}s")
        n_done = n_skip = n_empty = 0
        try:
            for frag in frags:
                base_dir = EVAL / frag / "sweep" / args.base
                out_dir = EVAL / frag / "sweep" / arm
                if not base_dir.is_dir():
                    print(f"  [skip] {frag}: no base dir {base_dir}")
                    continue
                if (out_dir / "transcript_A.txt").exists() and not args.force:
                    n_skip += 1
                    continue
                # Copy everything EXCEPT the stream wavs — rescore needs the
                # mixture transcript + json metadata, not the audio. Streams are
                # read from base_dir below.
                out_dir.mkdir(parents=True, exist_ok=True)
                for p in base_dir.iterdir():
                    if p.suffix == ".wav":
                        continue
                    shutil.copy2(p, out_dir / p.name)
                for spk in ("A", "B"):
                    wav = base_dir / f"stream_{spk}.wav"
                    if not wav.exists():
                        for ext in (".txt", ".json"):
                            (out_dir / f"transcript_{spk}{ext}").unlink(missing_ok=True)
                        continue
                    audio, _ = librosa.load(wav, sr=SAMPLE_RATE, mono=True)
                    audio = audio.astype(np.float32)
                    if skip_stream(audio, min_samples, cfg.silence_floor):
                        result = _empty_result(cfg.language)
                        n_empty += 1
                    else:
                        result = backend.transcribe(audio)
                    (out_dir / f"transcript_{spk}.txt").write_text(
                        format_transcript(result) + "\n", encoding="utf-8")
                    with open(out_dir / f"transcript_{spk}.json", "w",
                              encoding="utf-8") as f:
                        json.dump(to_jsonable(result), f, indent=2, ensure_ascii=False)
                n_done += 1
        finally:
            backend.unload()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print(f"[{arm}] done: {n_done} transcribed, {n_skip} kept, "
              f"{n_empty} silent-stream empties")

    print("\n=== done. score with: ===")
    print(f"python scripts/rescore_stratified.py --configs {' '.join(args.arms)} "
          f"--anchor dk_base --split {args.split}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
