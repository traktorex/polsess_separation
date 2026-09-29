"""Fixed-stream WhisperX prompt-variation arms (ASR-only swap).

Holds the OA-winner's assembled per-speaker streams CONSTANT and varies ONLY the
WhisperX ``initial_prompt`` — re-transcribing the existing ``stream_{A,B}.wav``
under each prompt and writing the result as a new sweep config dir
``<frag>/sweep/<arm>/`` that ``rescore_stratified.py`` can score directly. No
pipeline re-run; the separation/enhancement/assembly outputs are reused verbatim.

Why: the shipped prompt ``"Rozmowa po polsku."`` ("a conversation") describes the
two-speaker *mixture*, but WhisperX actually transcribes a *single separated
speaker stream*. This sweep disentangles two hypotheses the author raised:
  (H1) speaker-count framing — does "one person" match the input better than
       "a conversation / two people"?
  (H2) channel/artifact cue — does "telephone recording" prime Whisper to expect
       degraded / artifact-laden audio and help?

Design: a 2×2 over {one|two speakers} × {no channel cue | telephone cue}, plus
``pp_base`` (the current default prompt — should REPRODUCE the anchor, a harness
sanity check) and ``pp_one_artifacts`` (an explicit "poor audio quality" probe).

Faithfulness: the transcription config is loaded from the anchor's own saved
snapshot (``metadata.json``) so every decode knob (beam, temperature schedule,
retry-collapse, VAD, silence floor) matches the winner exactly — only the prompt
changes. The stage's silence/duration gate (``_skip_transcription`` →
``_empty_result``) is replicated so a silent stream yields an empty transcript
exactly as in the real run.

Usage::

    source venv/bin/activate
    python scripts/sweep_asr_prompt.py                 # dr_oa070, dev, all arms
    python scripts/sweep_asr_prompt.py --base dr_oa050
    # then score:
    python scripts/rescore_stratified.py --configs pp_base pp_two_plain \
        pp_one_plain pp_two_phone pp_one_phone pp_one_artifacts \
        --anchor dr_oa070 --split dev
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

# arm name -> initial_prompt.  2×2 (speaker × channel) + base + artifact probe.
ARMS = {
    "pp_base":          "Rozmowa po polsku.",                                    # current default (≈ anchor)
    "pp_two_plain":     "Rozmowa dwóch osób po polsku.",                         # two, no channel
    "pp_one_plain":     "Wypowiedź jednej osoby po polsku.",                     # one, no channel
    "pp_two_phone":     "Nagranie rozmowy telefonicznej po polsku, dwie osoby.", # two, telephone (prior rich)
    "pp_one_phone":     "Nagranie rozmowy telefonicznej po polsku, jedna osoba.",# one, telephone (isolates speaker count)
    "pp_one_artifacts": "Nagranie jednej osoby po polsku, słaba jakość dźwięku.",# one + explicit "poor audio quality"
}


def dev_fragments() -> list[str]:
    with open(SPLIT_CSV) as f:
        return sorted(r["frag_id"] for r in csv.DictReader(f) if r["role"] == "dev")


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
    ap.add_argument("--base", default="dr_oa070",
                    help="config whose streams to reuse (default: the OA-winner).")
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
          f"silence_floor={base_cfg.silence_floor}  prompt(orig)={base_cfg.initial_prompt!r}")
    print(f"[scope] {len(frags)} {args.split} fragment(s); arms={args.arms}")

    # Match the pipeline's determinism (Pipeline.__init__): without these,
    # WhisperX's wav2vec2 alignment uses nondeterministic cuDNN algorithms and
    # re-transcribing identical audio flips ~37% of streams — noise as large as
    # the prompt effect. With them, pp_base reproduces the anchor exactly.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    min_samples = int(SAMPLE_RATE * _MIN_TRANSCRIBE_DURATION_S)

    # Phase-major BY ARM: the prompt is baked into asr_options at load(), so each
    # arm needs its own model load. One load, all fragments, unload.
    for arm in args.arms:
        prompt = ARMS[arm]
        cfg = replace(base_cfg, initial_prompt=prompt)
        backend = _WhisperXBackend(cfg)
        print(f"\n[{arm}] loading WhisperX {cfg.model_name}  prompt={prompt!r}")
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
                # Copy everything EXCEPT the (large) stream wavs — rescore needs
                # the mixture transcript + json metadata, not the audio. Streams
                # are read from base_dir below.
                out_dir.mkdir(parents=True, exist_ok=True)
                for p in base_dir.iterdir():
                    if p.suffix == ".wav":
                        continue
                    shutil.copy2(p, out_dir / p.name)
                # Re-transcribe each per-speaker stream under this arm's prompt.
                for spk in ("A", "B"):
                    wav = base_dir / f"stream_{spk}.wav"
                    if not wav.exists():
                        # No such stream in the winner → leave absent (a 1-speaker
                        # collapse). Remove any copied stale transcript for safety.
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
          f"--anchor {args.base} --split {args.split}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
