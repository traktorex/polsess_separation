#!/usr/bin/env python
"""Inference-side benchmark: MACs, latency, real-time factor, peak inference VRAM.

WHAT THIS PROVES / PRODUCES
---------------------------
The *objective, machine-measured* half of the thesis's multi-axis architecture
comparison. Every number here is a property of the network and this GPU — none of
it depends on how a particular training run went. Run-result facts (SI-SDR,
epochs to convergence, wall-clock to convergence) belong in a separate table
sourced from W&B / the experiment logs; see
`thesis/thesis-log/sweep_plan/ch5_test_evals/BENCHMARK_AUDIT.md`.

Per model:
  * parameter count
  * MACs per clip and **per second of audio** (the portable unit)
  * inference latency at batch_size=1 (median + IQR over N passes)
  * real-time factor RTF = latency / audio duration (the deployment-relevant
    number: the downstream use is ASR preprocessing)
  * peak inference VRAM

HOW THE MACs ARE COUNTED (read this before trusting the number)
--------------------------------------------------------------
ptflops' PyTorch backend counts `nn.Module`s via hooks *and* patches a set of
functional / tensor ops (`torch.matmul`, `bmm`, `addmm`, `F.softmax`, ...). That
combination has two failure modes this repo's architectures walk straight into,
both found and fixed on 2026-07-30:

1. **`nn.MultiheadAttention` was double-counted.** ptflops' own module hook
   accounts for the whole attention (projections *and* the QK^T / AV products),
   but `need_weights=True` — which SpeechBrain's `MultiheadAttention` passes by
   default, so every SepFormer block — makes torch execute the attention core as
   `torch.bmm` + `F.softmax`, which ptflops' patch *also* counts. SepFormer's
   stored 258.48 GMACs overstated the true figure by 20.4 GMACs (7.9%).
   Fix: `_mha_hook` counts only the projections, and the core arrives from the
   patched ops — each op counted exactly once. `calibrate_mha_counting()` proves
   that decomposition against a closed form on the installed torch/ptflops before
   any model is measured, and flips the hook if a future torch stops routing the
   core through `bmm`.

2. **`einsum` was invisible.** ptflops patches `matmul`/`bmm` but not `einsum`,
   and MossFormer2's attention (both the quadratic and the linear branch) is
   written entirely in `einsum` — 48.7 of its 157.6 GMACs, a 31% undercount.
   Fix: `_patch_einsum` (see its docstring for why patching `torch.einsum` alone
   is not enough).

Ops that no hook can see, and what is done about them:
  * **Mamba selective scan** (`selective_scan_cuda`, `causal_conv1d`) — counted
    analytically by `_mamba_hooks`. The reference figure 9·B·L·D·N from
    state-spaces/mamba#110 is **FLOPs**, so it enters the MAC accumulator as
    4.5·B·L·D·N; the previous code put 9·B·L·D·N in and then doubled it for the
    GFLOPs column, overstating the scan by 2×.
  * **`torch.stft` / `torch.istft`** (SPMamba's encoder/decoder) — not counted.
    At n_fft=256 / hop=64 an FFT costs O(n·log n) per frame, ~0.5 MMAC for a 4 s
    clip against SPMamba's ~10^11: five orders of magnitude below the noise floor.
  * Elementwise ops reached through operators (`a * b`, `a + b`) rather than
    `torch.mul` / `torch.add`. Negligible and, being adds rather than
    multiply-accumulates, arguably shouldn't be in a MAC count anyway.

`--cross-check` runs the same forward pass through `torch.utils.flop_counter`
(aten-dispatch level, an independent implementation) and prints both. The two
agree to <1% on ConvTasNet, SepFormer and MossFormer2. It is **not** a valid
check for DPRNN or the Mamba models: the aten counter does not see fused RNN
kernels (it reports 1.8 GMACs for DPRNN's true ~21.6) or custom CUDA kernels.

Counting is device-independent — verified identical on CPU and CUDA — so the
figures can be regenerated without a GPU (`--device cpu`, non-Mamba models only).

KNOWN LIMITS
------------
  * `peak_infer_mem_mb` is `torch.cuda.max_memory_allocated()`: tensor
    allocations only (weights + activations + cached workspaces re-used during
    the timed region). It excludes the CUDA context (~300-500 MB) and driver
    overhead, so read it as a *relative* comparison, not as "GPU needed".
  * MACs are only exactly linear in clip length for the chunked/grouped-attention
    models (SepFormer's K=250 chunks, MossFormer2's 256-frame groups). SPMamba's
    GridNet attention spans the whole time axis and is O(T²). Pass several
    `--duration` values to see it.

HOW TO CITE
-----------
    scripts/benchmark_inference.py -> docs/generated/benchmark_inference.csv
    MACs: ptflops 0.7.5, PyTorch backend, with the MultiheadAttention,
    einsum and Mamba corrections documented in the script docstring.
    Latency: median of N=50 forward passes, batch_size=1, after 10 warmup passes.

USAGE
-----
    python scripts/benchmark_inference.py
    python scripts/benchmark_inference.py --duration 2 4 8 --cross-check
    python scripts/benchmark_inference.py --only sepformer mossformer2
    CUDA_VISIBLE_DEVICES="" python scripts/benchmark_inference.py --device cpu \
        --only convtasnet dprnn sepformer mossformer2      # MACs only, no GPU
"""

import argparse
import csv
import gc
import sys
import time
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from ptflops import get_model_complexity_info

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling benchmark_models

from benchmark_models import MAMBA_PREFIXES, MODELS, select  # noqa: E402
from config import load_config_from_yaml  # noqa: E402
from models.factory import create_model_from_config  # noqa: E402
from utils.model_utils import count_parameters, format_parameter_count  # noqa: E402

SAMPLE_RATE = 8000  # PolSESS sample rate; corpus clips are a fixed 4.000 s


# --------------------------------------------------------------------------- #
# einsum counting
# --------------------------------------------------------------------------- #

def _einsum_macs(equation: str, operands) -> int:
    """MACs of one einsum = the product of the extent of every distinct index.

    Each output element accumulates over the contracted indices, and every
    (output element, contraction step) pair is one multiply-accumulate, so the
    total is prod(extent of each index label appearing anywhere in the equation).
    For 'b i d, b j d -> b i j' that is B·I·J·D, the usual attention count.

    A '...' is treated as one broadcast batch volume: the largest product of the
    dimensions it covers across the operands.
    """
    lhs = equation.replace(" ", "").split("->")[0]
    extents: dict = {}
    ellipsis_volume = 1
    for term, operand in zip(lhs.split(","), operands):
        shape = list(operand.shape)
        if "..." in term:
            head, tail = term.split("...")
            n_hidden = len(shape) - len(head) - len(tail)
            covered = shape[len(head):len(head) + n_hidden]
            ellipsis_volume = max(ellipsis_volume, reduce(lambda a, b: a * b, covered, 1))
            labels = list(head) + list(tail)
            sizes = shape[:len(head)] + shape[len(head) + n_hidden:]
        else:
            labels, sizes = list(term), shape
        for label, size in zip(labels, sizes):
            extents[label] = max(extents.get(label, 1), size)
    return reduce(lambda a, b: a * b, extents.values(), 1) * ellipsis_volume


def _patch_einsum(collector: list) -> list:
    """Route every reachable `einsum` alias through a counting wrapper.

    ptflops patches `torch.matmul`/`bmm`/`addmm` but not `einsum`. Patching
    `torch.einsum` alone is still not enough: `models/mossformer2/
    mossformer2_block.py` does `from torch import nn, einsum`, which binds the
    function into that module's own namespace at import time, so the alias has to
    be replaced as well.

    Uses `module.__dict__.get` rather than `getattr` on purpose — SpeechBrain
    installs lazy module proxies whose `__getattr__` raises ImportError for
    optional dependencies (k2), and `getattr` would trip them.

    Returns the list of module objects that were patched, for `_unpatch_einsum`.
    """
    original = torch.einsum

    def wrapper(equation, *operands, **kwargs):
        # torch.einsum accepts both einsum(eq, a, b) and einsum(eq, [a, b]).
        args = operands
        if len(operands) == 1 and isinstance(operands[0], (list, tuple)):
            args = operands[0]
        if isinstance(equation, str):
            try:
                collector.append(_einsum_macs(equation, args))
            except Exception:  # never let accounting break the forward pass
                pass
        return original(equation, *operands, **kwargs)

    wrapper.original = original
    patched = []
    for module in list(sys.modules.values()):
        if module is None:
            continue
        if getattr(module, "__dict__", {}).get("einsum") is original:
            module.einsum = wrapper
            patched.append(module)
    torch.einsum = wrapper
    return patched


def _unpatch_einsum(patched: list) -> None:
    original = torch.einsum.original
    for module in patched:
        module.einsum = original
    torch.einsum = original


# --------------------------------------------------------------------------- #
# nn.MultiheadAttention: count the projections here, the core from the patch
# --------------------------------------------------------------------------- #

# Set by calibrate_mha_counting(). True  => torch routes the attention core
# through torch.bmm, which ptflops' tensor-op patch already counts, so the hook
# must NOT count it again.  False => the core is invisible (e.g. a torch build
# that always takes the SDPA path) and the hook has to supply it.
_MHA_CORE_FROM_PATCH: Optional[bool] = None


def _mha_terms(module, q, k, v) -> tuple:
    """(projection MACs, attention-core MACs) for one nn.MultiheadAttention call.

    Mirrors ptflops' own `multihead_attention_counter_hook` term for term, split
    at the boundary between what `F.linear` does (invisible: ptflops deliberately
    leaves F.linear unpatched, so as not to double-count nn.Linear modules) and
    what `torch.bmm` + `F.softmax` do (visible to the tensor-op patch).
    """
    batch_first = getattr(module, "batch_first", False)
    batch = q.shape[0] if batch_first else q.shape[1]
    len_axis = 1 if batch_first else 0
    q_len, k_len, v_len = q.shape[len_axis], k.shape[len_axis], v.shape[len_axis]
    q_dim, k_dim, v_dim = q.shape[2], k.shape[2], v.shape[2]
    heads = module.num_heads

    projections = q_len * q_dim  # scaling of Q
    projections += q_len * q_dim * q_dim + k_len * k_dim * k_dim + v_len * v_dim * v_dim
    if module.in_proj_bias is not None:
        projections += (q_len + k_len + v_len) * q_dim
    projections += q_len * v_dim * (v_dim + 1)  # out_proj, bias always present

    core = heads * (
        q_len * k_len * (q_dim // heads)   # QK^T
        + q_len * k_len                    # softmax
        + q_len * k_len * (v_dim // heads)  # AV
    )
    return batch * projections, batch * core


def _mha_hook(module, input, output) -> None:
    projections, core = _mha_terms(module, input[0], input[1], input[2])
    module.__flops__ += projections
    if not _MHA_CORE_FROM_PATCH:
        module.__flops__ += core


# --------------------------------------------------------------------------- #
# Mamba: the selective scan runs in a custom CUDA kernel, count it analytically
# --------------------------------------------------------------------------- #

# The scan's cost is quoted in the literature as 9·B·L·D·N **FLOPs**
# (state-spaces/mamba#110). Everything else in this script is accumulated in
# MACs and doubled for the GFLOPs column, so the scan enters as half that.
_SCAN_FLOPS_PER_ELEMENT = 9.0
_SCAN_MACS_PER_ELEMENT = _SCAN_FLOPS_PER_ELEMENT / 2


def _mamba_hooks() -> dict:
    """Custom ptflops hooks for the two Mamba implementations in this repo.

    Registering a class in `custom_modules_hooks` makes ptflops treat it as a
    leaf — the hook replaces, not supplements, the accounting of the module *and
    all its children* — so each hook counts every op in its block. That is also
    what keeps these blocks safe from the functional patches: `mamba_inner_fn`
    and `causal_conv1d_fn` are opaque CUDA kernels, the projections go through
    `F.linear` (unpatched) or the `@` operator (`Tensor.__matmul__`, which
    ptflops does not patch), so nothing here is seen twice.
    """
    hooks = {}

    try:
        from mamba_ssm.modules.mamba_simple import Mamba as MambaUni

        def unidirectional(module, input, output):
            batch, length, _ = input[0].shape
            d_model, d_inner = module.d_model, module.d_inner
            d_state, dt_rank = module.d_state, module.dt_rank
            macs = length * d_model * 2 * d_inner                     # in_proj
            macs += length * d_inner * module.d_conv                  # depthwise conv1d
            macs += length * d_inner * (dt_rank + 2 * d_state)        # x_proj
            macs += length * dt_rank * d_inner                        # dt_proj
            macs += _SCAN_MACS_PER_ELEMENT * length * d_inner * d_state  # selective scan
            macs += length * d_inner * d_model                        # out_proj
            module.__flops__ += int(batch * macs)

        hooks[MambaUni] = unidirectional
    except ImportError:
        pass

    try:
        from models.mamba.bimamba import Mamba as MambaBi

        def bidirectional(module, input, output):
            # in_proj / out_proj are shared between the directions; conv1d,
            # x_proj, dt_proj and the scan are duplicated (see bimamba.forward).
            batch, length, _ = input[0].shape
            d_model, d_inner = module.d_model, module.d_inner
            d_state, dt_rank = module.d_state, module.dt_rank
            macs = length * d_model * 2 * d_inner                     # in_proj  (shared)
            macs += 2 * length * d_inner * module.d_conv              # conv1d   x2
            macs += 2 * length * d_inner * (dt_rank + 2 * d_state)    # x_proj   x2
            macs += 2 * length * dt_rank * d_inner                    # dt_proj  x2
            macs += 2 * _SCAN_MACS_PER_ELEMENT * length * d_inner * d_state  # scan x2
            macs += length * d_inner * d_model                        # out_proj (shared)
            module.__flops__ += int(batch * macs)

        hooks[MambaBi] = bidirectional
    except ImportError:
        pass

    return hooks


# --------------------------------------------------------------------------- #
# the counter
# --------------------------------------------------------------------------- #

def count_macs(model: nn.Module, num_samples: int, device: str) -> Optional[int]:
    """Total MACs of one forward pass on a (1, 1, num_samples) input."""
    einsum_macs: list = []
    patched = _patch_einsum(einsum_macs)
    try:
        hooks = _mamba_hooks()
        hooks[nn.MultiheadAttention] = _mha_hook
        macs, _ = get_model_complexity_info(
            model, (1, num_samples), as_strings=False,
            print_per_layer_stat=False, verbose=False,
            input_constructor=lambda shape: torch.randn(1, *shape, device=device),
            custom_modules_hooks=hooks,
        )
    except Exception as exc:
        print(f"    MAC counting failed: {exc}")
        return None
    finally:
        _unpatch_einsum(patched)
    if macs is None or macs <= 0:
        return None
    return int(macs) + int(sum(einsum_macs))


def calibrate_mha_counting(device: str) -> bool:
    """Prove the MultiheadAttention decomposition on the installed torch/ptflops.

    `_mha_hook` counts the projections and assumes the attention core reaches the
    total through ptflops' `torch.bmm` / `F.softmax` patches. That assumption is
    a property of the torch version, not of this repo, so it is measured rather
    than trusted: count a reference attention block both ways and keep the
    setting whose total equals the closed form. Raises if neither matches, which
    is the honest outcome — a silently wrong MAC count is worse than a crash.
    """
    global _MHA_CORE_FROM_PATCH
    seq_len, dim, heads = 32, 64, 4

    class _Reference(nn.Module):
        def __init__(self):
            super().__init__()
            self.att = nn.MultiheadAttention(dim, heads, batch_first=True)

        def forward(self, x):
            out, _ = self.att(x, x, x, need_weights=True)
            return out

    reference = _Reference().to(device).eval()
    dummy = torch.zeros(1, seq_len, dim, device=device)
    projections, core = _mha_terms(reference.att, dummy, dummy, dummy)
    expected = projections + core

    for core_from_patch in (True, False):
        _MHA_CORE_FROM_PATCH = core_from_patch
        got, _ = get_model_complexity_info(
            reference, (seq_len, dim), as_strings=False,
            print_per_layer_stat=False, verbose=False,
            input_constructor=lambda shape: torch.randn(1, *shape, device=device),
            custom_modules_hooks={nn.MultiheadAttention: _mha_hook},
        )
        if got == expected:
            return core_from_patch

    _MHA_CORE_FROM_PATCH = None
    raise RuntimeError(
        "MultiheadAttention MAC accounting could not be calibrated: neither "
        f"setting reproduced the closed form ({expected:,} MACs for "
        f"L={seq_len}, d={dim}, h={heads}). ptflops or torch changed which ops "
        "are visible to the tensor-op patch — re-derive _mha_hook before "
        "quoting any transformer row."
    )


def crosscheck_macs(model: nn.Module, x: torch.Tensor) -> Optional[float]:
    """Independent MAC count at the aten-dispatch level, for validation only.

    Blind to fused RNN and custom CUDA kernels, so meaningless for DPRNN and the
    Mamba family — see the module docstring.
    """
    try:
        from torch.utils.flop_counter import FlopCounterMode
        counter = FlopCounterMode(display=False)
        with counter, torch.no_grad():
            model(x)
        return counter.get_total_flops() / 2  # the counter reports FLOPs
    except Exception as exc:
        print(f"    cross-check failed: {exc}")
        return None


# --------------------------------------------------------------------------- #
# benchmark
# --------------------------------------------------------------------------- #

@dataclass
class Result:
    name: str
    size: str
    config: str
    params: int
    duration_s: float
    macs: Optional[int]
    macs_crosscheck: Optional[float]
    latency_ms_median: float
    latency_ms_iqr: float
    peak_mem_mb: float

    @property
    def gmacs(self) -> Optional[float]:
        return self.macs / 1e9 if self.macs else None

    @property
    def gmacs_per_s(self) -> Optional[float]:
        return self.macs / 1e9 / self.duration_s if self.macs else None

    @property
    def rtf(self) -> float:
        return self.latency_ms_median / 1000 / self.duration_s


def _percentile(sorted_values: list, fraction: float) -> float:
    return sorted_values[min(len(sorted_values) - 1, int(fraction * len(sorted_values)))]


def benchmark_model(name, size, config_path, device, durations, n_warmup, n_runs,
                    cross_check) -> list:
    """Measure one config at every requested clip duration. Returns Result rows."""
    path = REPO_ROOT / config_path
    if not path.exists():
        print(f"  SKIP {name}: config not found ({config_path})")
        return []
    try:
        config = load_config_from_yaml(str(path))
        model = create_model_from_config(config.model).to(device).eval()
    except Exception as exc:
        print(f"  SKIP {name}: could not build model ({exc})")
        return []

    params = count_parameters(model)
    on_cuda = device.startswith("cuda")
    results = []

    for duration in durations:
        num_samples = int(duration * SAMPLE_RATE)
        x = torch.randn(1, 1, num_samples, device=device)

        # --- MACs. ptflops builds internal copies; drop them before measuring
        # memory so the peak below reflects the forward pass only.
        macs = count_macs(model, num_samples, device)
        crosscheck = crosscheck_macs(model, x) if cross_check else None
        gc.collect()
        if on_cuda:
            torch.cuda.empty_cache()

        # --- warm up first (allocates cuDNN/inductor workspaces), then reset the
        # peak counter, so peak memory describes the steady state rather than a
        # cold pass. The pre-2026-07-30 script measured a single cold forward.
        with torch.no_grad():
            for _ in range(n_warmup):
                model(x)
        if on_cuda:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

        latencies = []
        with torch.no_grad():
            for _ in range(n_runs):
                if on_cuda:
                    torch.cuda.synchronize(device)
                start = time.perf_counter()
                model(x)
                if on_cuda:
                    torch.cuda.synchronize(device)
                latencies.append((time.perf_counter() - start) * 1000)

        peak_mem = torch.cuda.max_memory_allocated(device) / 1024**2 if on_cuda else 0.0
        latencies.sort()
        median = _percentile(latencies, 0.50)
        iqr = _percentile(latencies, 0.75) - _percentile(latencies, 0.25)

        results.append(Result(
            name=name, size=size, config=config_path, params=params,
            duration_s=duration, macs=macs, macs_crosscheck=crosscheck,
            latency_ms_median=median, latency_ms_iqr=iqr, peak_mem_mb=peak_mem,
        ))
        del x

    del model
    gc.collect()
    if on_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--device", default="cuda", help="Device (default: cuda)")
    parser.add_argument("--duration", type=float, nargs="+", default=[4.0],
                        help="Clip duration(s) in seconds. PolSESS clips are 4.0 s; "
                             "pass several to check how MACs scale with length "
                             "(default: 4.0)")
    parser.add_argument("--n-warmup", type=int, default=10, help="Warmup passes (default: 10)")
    parser.add_argument("--n-runs", type=int, default=50, help="Timed passes (default: 50)")
    parser.add_argument("--only", nargs="+", default=None,
                        help="Only models whose name contains any of these substrings")
    parser.add_argument("--cross-check", action="store_true",
                        help="Also count MACs with torch.utils.flop_counter "
                             "(valid for conv/attention models only — see docstring)")
    parser.add_argument("--output", default="docs/generated/benchmark_inference.csv",
                        help="Output CSV, relative to the repo root")
    args = parser.parse_args()

    device = args.device
    on_cuda = device.startswith("cuda")
    if on_cuda and not torch.cuda.is_available():
        print("ERROR: CUDA not available. Use --device cpu for a MACs-only run "
              "(latency and memory columns will be meaningless, Mamba models "
              "cannot run at all).")
        sys.exit(1)

    gpu = torch.cuda.get_device_name(device) if on_cuda else "cpu"
    from importlib.metadata import version as _pkg_version
    provenance = {
        "gpu": gpu,
        "torch": torch.__version__,
        "cuda": torch.version.cuda or "",
        "ptflops": _pkg_version("ptflops"),
    }

    print(f"device: {device} ({gpu})")
    print(f"torch {provenance['torch']} / cuda {provenance['cuda']} / "
          f"ptflops {provenance['ptflops']}")
    core_from_patch = calibrate_mha_counting(device)
    print(f"MAC counter calibrated: attention core "
          f"{'from patched torch.bmm' if core_from_patch else 'from the module hook'}")
    print(f"durations: {', '.join(f'{d}s' for d in args.duration)} @ {SAMPLE_RATE} Hz")
    print(f"latency: {args.n_warmup} warmup + {args.n_runs} timed passes, batch_size=1")
    print()

    models = select(MODELS, args.only)
    if not models:
        print(f"No models match --only {args.only}. Available:",
              ", ".join(m[0] for m in MODELS))
        sys.exit(1)

    results = []
    for name, size, config_path in models:
        if not on_cuda and name.startswith(MAMBA_PREFIXES):
            print(f"SKIP {name}: the Mamba kernels require CUDA")
            continue
        print(f"Benchmarking {name}...")
        rows = benchmark_model(name, size, config_path, device, args.duration,
                              args.n_warmup, args.n_runs, args.cross_check)
        for row in rows:
            gmacs = f"{row.gmacs:.2f}" if row.gmacs else "n/a"
            print(f"  {row.duration_s:>4.1f}s  {format_parameter_count(row.params):>8} params  "
                  f"{gmacs:>8} GMACs  {row.latency_ms_median:6.1f} ms  "
                  f"RTF {row.rtf:.4f}  {row.peak_mem_mb:6.0f} MB")
        results.extend(rows)
        print()

    header = f"{'Model':<22}{'Params':>9}{'GMACs':>9}{'GMAC/s':>9}{'ms (IQR)':>16}{'RTF':>9}{'Mem MB':>9}"
    for duration in args.duration:
        rows = [r for r in results if r.duration_s == duration]
        if not rows:
            continue
        print("=" * len(header))
        print(f"{duration} s clip @ {SAMPLE_RATE} Hz, batch_size=1, {gpu}")
        print(header)
        print("-" * len(header))
        for r in rows:
            gmacs = f"{r.gmacs:.2f}" if r.gmacs else "n/a"
            per_s = f"{r.gmacs_per_s:.2f}" if r.gmacs else "n/a"
            latency = f"{r.latency_ms_median:.1f} ({r.latency_ms_iqr:.1f})"
            print(f"{r.name:<22}{format_parameter_count(r.params):>9}{gmacs:>9}"
                  f"{per_s:>9}{latency:>16}{r.rtf:>9.4f}{r.peak_mem_mb:>9.0f}")
        print("=" * len(header))
    print("Latency is a median with the inter-quartile range in brackets — not a "
          "standard deviation.")
    print("Peak memory = tensor allocations only (excludes the ~300-500 MB CUDA context).")

    output = REPO_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "model", "size", "config", "params", "duration_s", "gmacs",
            "gmacs_per_s_audio", "gflops", "gflops_per_s_audio",
            "latency_ms_median", "latency_ms_iqr",
            "rtf", "peak_infer_mem_mb", "gmacs_aten_crosscheck",
            "gpu", "torch", "cuda", "ptflops",
        ])
        for r in results:
            writer.writerow([
                r.name, r.size, r.config, r.params, r.duration_s,
                f"{r.gmacs:.4f}" if r.gmacs else "",
                f"{r.gmacs_per_s:.4f}" if r.gmacs else "",
                f"{2 * r.gmacs:.4f}" if r.gmacs else "",
                f"{2 * r.gmacs_per_s:.4f}" if r.gmacs else "",
                f"{r.latency_ms_median:.2f}", f"{r.latency_ms_iqr:.2f}",
                f"{r.rtf:.5f}", f"{r.peak_mem_mb:.1f}",
                f"{r.macs_crosscheck / 1e9:.4f}" if r.macs_crosscheck else "",
                provenance["gpu"], provenance["torch"], provenance["cuda"],
                provenance["ptflops"],
            ])
    print(f"\nResults saved to {output}")


if __name__ == "__main__":
    main()
