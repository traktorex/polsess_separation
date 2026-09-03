#!/usr/bin/env python3
"""Export the two shipping separators of the Road-2 on-device demo to ONNX (INT8).

Promoted from the 2026-08-05 export experiment, which answered "can
MossFormer2 ship in a browser?" (yes) and "which INT8 recipe survives?" (a
hand-tuned one); this script is the reproducible, documented form of that
answer, with both shipping artifacts built by one command.

--------------------------------------------------------------------------
Tensor contract  (the "January contract" — unchanged since mobile/webapp/)
--------------------------------------------------------------------------
    input   "mixture"    float32  [1, 32000]      4 s @ 8 kHz
    output  "separated"  float32  [1, 2, 32000]   [batch, speaker, sample]

8 kHz / 4 s is what every separator in this repo is trained and benchmarked
at (`scripts/benchmark_inference.py: SAMPLE_RATE = 8000`,
`asr_pipeline/config.py: separator_sample_rate = 8_000`,
`training_chunk_length_s = 4.0`), and what the browser chunker feeds.

Shapes are STATIC — no `dynamic_axes`. Two reasons:
  1. The chunker always feeds exactly 32000 samples (zero-padding the tail),
     so dynamic axes buy nothing.
  2. MossFormer2 *requires* it. Its vendored rotary block, token-shift and
     group-rearrange logic trace as Python-level branches on T; with T fixed
     the traced branch is the correct one for every call the browser will
     ever make. Under `dynamic_axes` those branches would be baked in and
     then silently wrong for other lengths.

--------------------------------------------------------------------------
INT8 recipes  (measured in the 2026-08-05 export experiment, not guessed)
--------------------------------------------------------------------------
`quantize_dynamic` with its defaults (all Conv + MatMul) DESTROYS
MossFormer2: 0.9964 corr / 21.4 dB self-SI-SDR, and it is also 3x SLOWER
than fp32 because ORT's `ConvInteger` kernels lose to fp32 `Conv` here.
Bisecting showed the culprit is not the depthwise convs (the usual suspect)
but the group-1 pointwise convs — above all the waveform encoder
(`/model/model/enc/conv1d/Conv`, 1->512, k=16, fed raw audio) and the mask
decoder. Per-tensor uint8 activation quantization of a raw waveform is
hopeless; consistent with the known fp16 fragility of this architecture —
its squared-ReLU attention overflows fp16 once activations sharpen, which is
why it is trained in bf16.

  "safe"    uint8 weights on MatMul + Conv, EXCLUDING the encoder/decoder
            convs and every grouped (depthwise) conv.  MF2: 33.6 MB,
            0.99937 corr / 28.99 dB vs its own fp32.  <- MF2 ships this
  "matmul"  per-channel uint8 weights, MatMul nodes only. Leaves every Conv
            in fp32: bigger for MF2 (45.8 MB) but for SepFormer — whose
            compute is Gemm/MatMul-dominated — it is both the smallest and
            the most faithful option (30.2 MB, 0.99983 / 34.69 dB).
            <- SepFormer ships this

Always `QuantType.QUInt8`. `QuantType.QInt8` quantizes and saves without
complaint and then throws at session creation:
`NOT_IMPLEMENTED : Could not find an implementation for ConvInteger(10)` —
signed-weight `ConvInteger` has no kernel in ORT, CPU or WASM.

--------------------------------------------------------------------------
Why `load_model_for_inference` and not a hand-written constructor
--------------------------------------------------------------------------
`mobile/webapp/export_sepformer.py` (January) re-typed SepFormer's
hyperparameters by hand and omitted `use_positional_encoding`. The current
SepFormer-128k checkpoint was trained with `use_positional_encoding: true`
(and `dropout: 0.0455`), so that script raises on `load_state_dict` — and
`checkpoints/sepformer/SB/128_run/` has no sibling `config.yaml` to copy the
values from either. Reading the architecture out of the checkpoint's own
embedded config removes the whole class of problem.

--------------------------------------------------------------------------
Usage (CPU-only; the GPU is never touched)
--------------------------------------------------------------------------
    CUDA_VISIBLE_DEVICES="" venv/bin/python webapp_ondevice/build/export_separators.py
    ... --only mf2            # just one of them
    ... --force               # re-export even if the fp32 intermediate exists
    ... --ckpt <path>         # override the checkpoint (with --only)

fp32 intermediates (~104 MB each) land in `build/_fp32/` (git-ignored); the
shipping INT8 files land in `site/models/` (also git-ignored — regenerable).
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch

# webapp_ondevice/build/export_separators.py -> repo root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

APP_DIR = PROJECT_ROOT / "webapp_ondevice"
DEFAULT_FP32_DIR = APP_DIR / "build" / "_fp32"
DEFAULT_OUT_DIR = APP_DIR / "site" / "models"

CHUNK = 32000  # 4 s @ 8 kHz — the January contract
OPSET = 17     # >=17 so LayerNormalization stays one native op (ORT-Web >= 1.17)


@dataclass(frozen=True)
class Separator:
    key: str
    ckpt: str          # repo-relative
    out_name: str      # file written to site/models/
    recipe: str        # "safe" | "matmul"
    note: str


SEPARATORS = (
    Separator(
        key="mf2",
        # The SAME checkpoint the ASR pipeline deploys (asr_pipeline/config.py
        # + configs/{default,sweep_best_e31_refineplus}.yaml) — the demo's
        # narrative is "we ship the deployed separator", so this must track it.
        # The first export pass measured export + INT8 parity on the e23 sibling
        # (.../mossformer2_matched_128k_final_42/mossformer2_SB_best_e23.pt,
        # val_sisdr 16.37); e46 re-verified 2026-08-05 with the same harness.
        ckpt="checkpoints/mossformer2/SB/mossformer2_matched_128k_final_42_e46/mossformer2_SB_best_e46.pt",
        out_name="mf2_128k_int8.onnx",
        # "matmul", not "safe": e46's extra training sharpens activations and the
        # "safe" recipe that gave 29.0 dB self-fidelity on e23 drops to 23.0 dB
        # on e46 (audible risk). MatMul-only on e46 = 30.3 dB @ 45.8 MB
        # (measured 2026-08-05); the +12 MB buys the deployed checkpoint at
        # transparent fidelity. SepFormer remains the light/fast option.
        recipe="matmul",
        note="default / best quality (val_sisdr 17.04)",
    ),
    Separator(
        key="sepformer",
        ckpt="checkpoints/sepformer/SB/128_run/sepformer_SB_best_128k_e41.pt",
        out_name="sepformer_128k_int8.onnx",
        recipe="matmul",
        note="fast option (val_sisdr 15.83; ~10x faster than MF2 at INT8)",
    ),
)


def export_fp32(ckpt_path: Path, fp32_path: Path, opset: int = OPSET) -> None:
    """Trace the checkpoint to a static-shape fp32 ONNX graph and check it."""
    from utils.model_utils import load_model_for_inference

    model, ckpt = load_model_for_inference(str(ckpt_path), device="cpu")
    model.eval()
    model_type = ckpt.get("config", {}).get("model", {}).get("model_type")
    print(
        f"  {model_type}: {sum(p.numel() for p in model.parameters()):,} params "
        f"(epoch {ckpt.get('epoch')}, val_sisdr {ckpt.get('val_sisdr'):.3f})"
    )

    dummy = torch.randn(1, CHUNK)
    with torch.no_grad():
        ref = model(dummy)
    assert ref.shape == (1, 2, CHUNK), f"unexpected forward shape {tuple(ref.shape)}"

    fp32_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    torch.onnx.export(
        model,
        (dummy,),
        str(fp32_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,  # legacy TorchScript exporter; the dynamo one is not needed
        input_names=["mixture"],
        output_names=["separated"],
        # deliberately no dynamic_axes -> [1, 32000] / [1, 2, 32000]
    )
    print(
        f"  fp32 export OK in {time.time() - t0:.1f}s -> {fp32_path.name} "
        f"({fp32_path.stat().st_size / 1024**2:.2f} MB)"
    )


def quantize(fp32_path: Path, int8_path: Path, recipe: str) -> None:
    """Apply one of the two measured INT8 recipes (see module docstring)."""
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    model = onnx.load(str(fp32_path))
    onnx.checker.check_model(model, full_check=True)
    ops = Counter(n.op_type for n in model.graph.node)
    print(f"  onnx.checker PASS | nodes={sum(ops.values())} distinct_ops={len(ops)}")

    int8_path.parent.mkdir(parents=True, exist_ok=True)
    if recipe == "matmul":
        quantize_dynamic(
            str(fp32_path),
            str(int8_path),
            weight_type=QuantType.QUInt8,
            per_channel=True,
            op_types_to_quantize=["MatMul"],
        )
    elif recipe == "safe":
        # Keep in fp32: the waveform encoder/decoder convs (raw-audio range is
        # untenable for per-tensor uint8) and every grouped/depthwise conv.
        exclude = [
            n.name
            for n in model.graph.node
            if n.op_type == "ConvTranspose"
            or (
                n.op_type == "Conv"
                and (
                    any(a.name == "group" and a.i > 1 for a in n.attribute)
                    or "/enc/" in n.name
                    or "/dec" in n.name
                    or "decoder" in n.name
                )
            )
        ]
        print(f"  'safe' recipe: {len(exclude)} conv nodes kept in fp32")
        quantize_dynamic(
            str(fp32_path),
            str(int8_path),
            weight_type=QuantType.QUInt8,
            nodes_to_exclude=exclude,
        )
    else:  # pragma: no cover - argparse constrains this
        raise ValueError(f"unknown recipe {recipe!r}")
    print(
        f"  int8({recipe}) -> {int8_path.name} "
        f"({int8_path.stat().st_size / 1024**2:.2f} MB)"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", choices=[s.key for s in SEPARATORS], default=None)
    ap.add_argument("--ckpt", default=None, help="override checkpoint (use with --only)")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--fp32-dir", type=Path, default=DEFAULT_FP32_DIR)
    ap.add_argument("--opset", type=int, default=OPSET)
    ap.add_argument("--force", action="store_true", help="re-export existing fp32 files")
    args = ap.parse_args()

    if args.ckpt and not args.only:
        ap.error("--ckpt requires --only (which artifact is being overridden?)")

    targets = [s for s in SEPARATORS if args.only in (None, s.key)]
    for sep in targets:
        ckpt = Path(args.ckpt) if args.ckpt else PROJECT_ROOT / sep.ckpt
        if not ckpt.is_file():
            raise SystemExit(f"missing checkpoint: {ckpt}")
        print(f"\n=== {sep.key} — {sep.note}\n  ckpt: {ckpt}")
        fp32 = args.fp32_dir / f"{sep.key}_128k_fp32.onnx"
        if fp32.is_file() and not args.force:
            print(f"  fp32 exists, reusing {fp32} (--force to re-export)")
        else:
            export_fp32(ckpt, fp32, opset=args.opset)
        quantize(fp32, args.out_dir / sep.out_name, sep.recipe)

    print("\nDone. Shipping artifacts:")
    for sep in targets:
        p = args.out_dir / sep.out_name
        print(f"  {p}  ({p.stat().st_size / 1024**2:.2f} MB)")


if __name__ == "__main__":
    main()
