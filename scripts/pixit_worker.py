"""Standalone PixIT (pyannote speech-separation) worker — B3 architecture arm.

Joint diarization + separation in ONE model call: `pyannote/speech-separation-
ami-1.0` (PixIT recipe, Kalda et al. 2024; ToTaToNet trained on AMI-SDM) emits
per-speaker SOURCES + the diarization in a single pass — the joint-architecture
rival to our modular region-routed cascade (backlog B3; ch7 §7.8 external-
reference slot).

Runs in the ISOLATED PixIT venv (`~/pixit_venv`, `$PIXIT_VENV_PY`) — the model
card pins `pyannote.audio[separation]==3.3.2` while the main venv carries
pyannote.audio 4.x for the diarization stage; same isolation rationale as the
Sortformer / CohereX / Brouhaha workers. Invoked as a subprocess by the B3 glue
(`scripts/pixit_baseline.py`, to be written when runs start), NOT imported.

VENV RECIPE (persistent, ``~/pixit_venv`` — built + import-verified 2026-07-18;
every pin below was NEEDED — unpinned pip resolves 2026-era versions that break
the 2024-era pyannote 3.3.2 stack in sequence: setuptools≥81 drops
pkg_resources (torchmetrics), torchaudio≥2.9 drops AudioMetaData (pyannote),
transformers 5.x needs torch≥2.5 DTensor):

    python3 -m venv ~/pixit_venv
    ~/pixit_venv/bin/pip install --upgrade pip
    ~/pixit_venv/bin/pip install "pyannote.audio[separation]==3.3.2" soundfile \
        "setuptools<81" "torch==2.4.1" "torchaudio==2.4.1" \
        "transformers==4.44.2" "lightning==2.4.0" "pytorch-lightning==2.4.0" \
        matplotlib  # pyannote 3.3.2 imports it unconditionally
    export PIXIT_VENV_PY=~/pixit_venv/bin/python
    # verified: pyannote.audio 3.3.2 | torch 2.4.1+cu121 | cuda True

GATED ACCESS (author action, one-time): the model is HF-gated and the current
$HF_TOKEN gets 403 (checked 2026-07-18) — accept the user conditions on
https://huggingface.co/pyannote/speech-separation-ami-1.0 (and any dependent
model repo its config names; this worker fails loud with the blocked repo id).
Keep $HF_TOKEN in the env for the first download; do NOT set $HF_HUB_OFFLINE.

STATUS: UNTESTED — written against the 3.3.2 API + the model card's usage
snippet while the gate is closed. Self-test the moment access works:

    $PIXIT_VENV_PY scripts/pixit_worker.py \
        --in ~/datasets/clarin_all_2speaker_fragments/fragments/005cba37__seg00.wav \
        --out-dir /tmp/pixit_selftest

CWD WARNING (same as sortformer_worker.py): the repo root's local ``datasets/``
package shadows HF ``datasets``. Callers invoke this worker with cwd OUTSIDE
the repo; the script imports nothing from the repo.

I/O PROTOCOL:
    input : --in IN.wav (mono 16 kHz) --out-dir DIR
            [--model pyannote/speech-separation-ami-1.0] [--device cuda]
            [--num-speakers 2]
    output: DIR/source_<LABEL>.wav       one 16 kHz mono wav per speaker
            DIR/diarization.json         {"segments": [{"start","end","speaker"}...]}
            DIR/meta.json                model id, labels in column order, sr,
                                         input path, durations

The scoring-side speaker mapping (which source is speaker A/B) is the GLUE's
job — this worker only emits what the model returns, in the model's label
order.
"""
import argparse
import json
import os
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--in", dest="in_wav", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="pyannote/speech-separation-ami-1.0")
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--num-speakers", type=int, default=2,
        help="speaker-count hint passed to the pipeline; 0 = let it decide",
    )
    ap.add_argument(
        "--params-json", default=None,
        help="JSON dict deep-merged into the pipeline's instantiated params "
             "(e.g. '{\"clustering\": {\"min_cluster_size\": 2}}') — for "
             "dev-only sensitivity probes; recorded in meta.json",
    )
    args = ap.parse_args()

    import numpy as np
    import soundfile as sf
    import torch
    from pyannote.audio import Pipeline

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("pixit_worker: $HF_TOKEN unset — the model is HF-gated.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline = Pipeline.from_pretrained(args.model, use_auth_token=token)
    if pipeline is None:
        sys.exit(
            f"pixit_worker: Pipeline.from_pretrained({args.model!r}) returned "
            "None — usually gated access not granted for this repo or a "
            "dependent model repo. Accept the conditions on its HF page."
        )
    pipeline.to(torch.device(args.device))

    param_override = json.loads(args.params_json) if args.params_json else None
    if param_override:
        params = pipeline.parameters(instantiated=True)
        for section, overrides in param_override.items():
            params.setdefault(section, {}).update(overrides)
        pipeline.instantiate(params)
        print(f"pixit_worker: params override applied: {param_override}")

    kwargs = {}
    if args.num_speakers > 0:
        kwargs["num_speakers"] = args.num_speakers
    diarization, sources = pipeline(args.in_wav, **kwargs)

    # sources: SlidingWindowFeature, .data shape (n_samples, n_sources) @ 16 kHz,
    # columns ordered like diarization.labels() (model-card contract).
    labels = list(diarization.labels())
    data = sources.data
    if data.ndim != 2 or data.shape[1] < len(labels):
        sys.exit(
            f"pixit_worker: sources shape {data.shape} does not cover "
            f"{len(labels)} diarization labels — API drift, inspect manually."
        )
    sr = 16_000
    # ToTaToNet sources are unnormalised and HOT (observed peak ≈ 8 on the
    # first CLARIN self-test); soundfile's WAV default subtype is PCM_16,
    # which clipped 40% of samples flat and destroyed the ASR (cpWER 92.7 on
    # the smoke fragment). One COMMON scale across sources (preserves the
    # inter-speaker balance) + FLOAT subtype (no quantisation clipping).
    peak = float(np.abs(data[:, : len(labels)]).max()) if len(labels) else 0.0
    scale = 0.95 / peak if peak > 0.95 else 1.0
    for col, label in enumerate(labels):
        sf.write(
            out_dir / f"source_{label}.wav",
            (data[:, col] * scale).astype(np.float32),
            sr,
            subtype="FLOAT",
        )

    segments = [
        {"start": float(turn.start), "end": float(turn.end), "speaker": label}
        for turn, _, label in diarization.itertracks(yield_label=True)
    ]
    (out_dir / "diarization.json").write_text(
        json.dumps({"segments": segments}, indent=2)
    )
    info = sf.info(args.in_wav)
    (out_dir / "meta.json").write_text(json.dumps({
        "model": args.model,
        "labels": labels,
        "sample_rate": sr,
        "input": str(Path(args.in_wav).resolve()),
        "input_duration_s": float(info.frames) / float(info.samplerate),
        "source_samples": int(data.shape[0]),
        "source_peak_raw": peak,
        "source_scale_applied": scale,
        "num_speakers_hint": args.num_speakers,
        "params_override": param_override,
    }, indent=2))
    print(
        f"pixit_worker: OK — {len(labels)} source(s) "
        f"({', '.join(labels)}) → {out_dir}"
    )


if __name__ == "__main__":
    main()
