"""Standalone NVIDIA Sortformer (offline EEND) diarization worker — v1 + streaming-v2.1.

Run as a SUBPROCESS by ``asr_pipeline.stages.diarization`` when
``diarization.backend == "sortformer"``. It runs in the ISOLATED NeMo venv —
``nemo_toolkit[asr]``'s pins (torch 2.12, numpy 2.4, transformers 4.57, a full
Lightning/Hydra stack) conflict with the project's main torch-2.8 venv, so they
cannot share it. The stage invokes this script with the ``$SORTFORMER_VENV_PY``
interpreter, NOT the main-venv python. (Same isolation rationale as the CohereX
transcription worker and the Brouhaha scorer.)

MODELS. Two model ids run through this one worker, dispatched by whether
``"streaming"`` appears in the ``--model`` id:
  * ``nvidia/diar_sortformer_4spk-v1`` (default) — a single full-attention
    offline pass; no configuration needed (CC-BY-NC-4.0, non-commercial).
  * ``nvidia/diar_streaming_sortformer_4spk-v2.1`` — a *streaming* model with no
    single-window offline mode. Run "offline" by applying the model-card
    very-high-latency preset (largest chunk + speaker-cache/context params) to
    ``model.sortformer_modules`` before ``diarize()``; the file is then decoded
    in a few chunks with the Arrival-Order Speaker Cache carried across them.
    Same ``SortformerEncLabelModel`` class, same ``diarize()`` call, same
    (T, 4) @ 0.08 s output contract — probe-verified
    (``thesis-log/sweep_plan/SORTFORMER_V21_PROBE.md``). NVIDIA Open Model License
    (commercial use OK). Loads in the SAME NeMo 2.7.3 venv — no new environment.

VENV RECIPE (persistent, ``~/sortformer_venv`` — rebuilt fresh, NOT copied from
the probe's ``/tmp/sortformer_venv``; venvs don't relocate):

    python3 -m venv ~/sortformer_venv
    ~/sortformer_venv/bin/pip install --upgrade pip wheel setuptools
    ~/sortformer_venv/bin/pip install "nemo_toolkit[asr]==2.7.3"
    # $HF_TOKEN in env for the first model download; do NOT set HF_HUB_OFFLINE.

Verify it loads:
    ~/sortformer_venv/bin/python -c \
        "from nemo.collections.asr.models import SortformerEncLabelModel; \
         SortformerEncLabelModel.from_pretrained('nvidia/diar_sortformer_4spk-v1')"

The same venv also loads the streaming v2.1 model —
``nemo_toolkit[asr]==2.7.3`` supports it with zero changes (probe-verified
2026-07-03); no separate v2.1 venv is needed.

Set ``$SORTFORMER_VENV_PY=~/sortformer_venv/bin/python`` for the stage to find it.

CWD WARNING. The repo root carries a local ``datasets/`` package that SHADOWS the
HuggingFace ``datasets`` library and breaks NeMo's imports. The calling stage
therefore invokes this worker with cwd set OUTSIDE the repo (a tempfile dir).
Keep this script free of any ``import`` from the repo (no ``asr_pipeline`` /
``datasets`` etc.) — it must import cleanly under the NeMo venv, which does not
have the repo on its path, and runs from a neutral cwd so it never resolves the
repo's local ``datasets`` package.

I/O PROTOCOL (mirrors ``coherex_worker.py``):
    input  : --in IN.wav (mono 16 kHz) --out OUT.json --model <id> [--device cuda]
    output : OUT.json with
        frame_probs  [[f0,f1,f2,f3], ...]  # (T, 4) sigmoid speaker-activity
        frame_rate_s 0.08                  # seconds per frame
        num_frames   T
        num_heads    4
        model        the model id

PARITY. Reproduces ``thesis-log/sweep_plan/sortformer_probe.py``'s frame cache exactly
(same ``model.diarize`` call, same batch collapse to (T,4)) — the probe's cached
``_eend_cache/sortformer/<frag>.npz`` is the ground-truth check for this worker.
The turn-building / thresholding lives in the STAGE, not here: this worker only
emits the raw per-frame activity.

Usage:
    python sortformer_worker.py --in IN.wav --out OUT.json \
        --model nvidia/diar_sortformer_4spk-v1 [--device cuda]
    # streaming v2.1 (offline preset auto-applied when "streaming" is in the id):
    python sortformer_worker.py --in IN.wav --out OUT.json \
        --model nvidia/diar_streaming_sortformer_4spk-v2.1
"""
import argparse
import json
import sys

import numpy as np


# Sortformer emits one activity frame every 0.08 s — v1 and streaming-v2.1 alike
# (model card / probe constant, thesis-log/sweep_plan/SORTFORMER_V21_PROBE.md).
FRAME_SHIFT_S = 0.08


def _to_np(x):
    """Tensor / array-like → float32 numpy (detached, on CPU)."""
    import torch

    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--model", default="nvidia/diar_sortformer_4spk-v1")
    ap.add_argument("--device", default=None,
                    help="cuda | cpu; default = cuda if available")
    a = ap.parse_args()

    import torch
    from nemo.collections.asr.models import SortformerEncLabelModel

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[sortformer] loading {a.model} on {device} ...", flush=True)
    model = SortformerEncLabelModel.from_pretrained(a.model)
    model.eval()
    if device == "cuda":
        model = model.to(torch.device("cuda"))

    # Model-mode dispatch. v1 runs a single full-attention offline pass and needs
    # no configuration. The streaming v2.1 model has no single-window offline mode,
    # so it is run "offline" by applying the model-card very-high-latency preset to
    # its `sortformer_modules` (largest chunk + speaker-cache/context params); the
    # file is then decoded in a few chunks with the Arrival-Order Speaker Cache
    # carried across them. Detect by "streaming" in the model id, and log loudly.
    # (The v1 branch is untouched — nothing new runs — so v1 stays byte-identical.)
    if "streaming" in a.model.lower():
        sm = model.sortformer_modules
        # very-high-latency OFFLINE preset (v2.1 model-card README, verbatim; the
        # values are in 80 ms frame units). spkcache_update_period=300 < chunk_len
        # =340 is NVIDIA's own recommended combo — _check_streaming_parameters()
        # emits a benign "effective update period will be chunk_len" warning but
        # does NOT raise; it DOES raise (fail-loud) on any genuinely illegal combo.
        sm.chunk_len = 340
        sm.chunk_right_context = 40
        sm.fifo_len = 40
        sm.spkcache_update_period = 300
        sm.spkcache_len = 188
        sm._check_streaming_parameters()
        print("[sortformer] STREAMING model -> OFFLINE very-high-latency preset: "
              "chunk_len=340 chunk_right_context=40 fifo_len=40 "
              "spkcache_update_period=300 spkcache_len=188", flush=True)

    # API (model card): diarize(audio, batch_size=1, include_tensor_outputs=True)
    # -> (segments, probs); probs is a length-1 list of (1, T, 4) tensors — one
    # per input file. Collapse the batch dim so the cache is (T, 4), matching the
    # probe. (segments are the model's own thresholded turns; the stage builds its
    # own from the raw probs, so we ignore them here.)
    _segs, probs = model.diarize(audio=a.inp, batch_size=1,
                                 include_tensor_outputs=True)
    arr = _to_np(probs[0])
    arr = arr.reshape(-1, arr.shape[-1]).astype(np.float32)   # (T, 4)

    payload = {
        "frame_probs": arr.tolist(),
        "frame_rate_s": FRAME_SHIFT_S,
        "num_frames": int(arr.shape[0]),
        "num_heads": int(arr.shape[1]),
        "model": a.model,
    }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    print(f"[sortformer] wrote {arr.shape[0]} frames x {arr.shape[1]} heads "
          f"-> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
