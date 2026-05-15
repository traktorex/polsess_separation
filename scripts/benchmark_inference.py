"""Inference benchmark: GMACs/GFLOPs, inference time, peak GPU memory for all architectures.

Measures per-model:
- GMACs / GFLOPs (via ptflops — runs on CUDA so Mamba custom kernels work)
- Inference time (median of N forward passes on a 4s sample)
- Peak GPU memory allocated during inference (batch_size=1)
- Parameter count

Memory note: peak_memory_mb is torch.cuda.max_memory_allocated(), which tracks
tensor allocations only (weights + activations). It excludes CUDA context (~300-500MB)
and driver overhead. Use it as a *relative* comparison between models, not as total
GPU memory required.

Usage:
    python benchmark_inference.py
    python benchmark_inference.py --device cuda:1
    python benchmark_inference.py --duration 4.0 --n-runs 50
"""

import argparse
import gc
import time
import torch
import csv
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from ptflops import get_model_complexity_info

from config import load_config_from_yaml
from models.factory import create_model_from_config
from utils.model_utils import count_parameters, format_parameter_count


SAMPLE_RATE = 8000  # PolSESS sample rate

# All 14 architecture configs to benchmark
# (name, config_path)
CONFIGS = [
    ("ConvTasNet", "experiments/convtasnet/baseline.yaml"),
    ("DPRNN (k=16)", "experiments/dprnn/dprnn_baseline.yaml"),
    ("DPRNN (k=2)", "experiments/dprnn/variants/dprnn_baseline_kernel2.yaml"),
    ("SepFormer", "experiments/sepformer/sepformer_baseline.yaml"),
    ("SepFormer (pos-enc)", "experiments/sepformer/sepformer_baseline_positionalenc.yaml"),
    ("SPMamba", "experiments/spmamba/spmamba_sb_reduced.yaml"),
    ("MambaTasNet-XS", "experiments/mamba_tasnet/mamba_tasnet_xs.yaml"),
    ("MambaTasNet-S", "experiments/mamba_tasnet/mamba_tasnet_s.yaml"),
    ("MambaTasNet-M", "experiments/mamba_tasnet/mamba_tasnet_m.yaml"),
    ("MambaTasNet-L", "experiments/mamba_tasnet/mamba_tasnet_l.yaml"),
    ("DPMamba-XS", "experiments/dpmamba/dpmamba_xs.yaml"),
    ("DPMamba-S", "experiments/dpmamba/dpmamba_s.yaml"),
    ("DPMamba-M", "experiments/dpmamba/dpmamba_m.yaml"),
    ("DPMamba-L", "experiments/dpmamba/dpmamba_l.yaml"),
]


@dataclass
class BenchmarkResult:
    name: str
    params: int
    params_str: str
    gmacs: Optional[float]
    gflops: Optional[float]
    inference_time_ms: float
    inference_std_ms: float
    peak_memory_mb: float


def _build_mamba_hooks() -> dict:
    """Build custom_modules_hooks dict for ptflops to count Mamba selective scan MACs.

    ptflops can't see into custom CUDA kernels (selective_scan_cuda). When we pass
    a class via custom_modules_hooks, ptflops replaces its default counting for that
    module entirely, so we must count ALL ops (not just the scan).

    Handles two Mamba implementations:
    - mamba_ssm.modules.mamba_simple.Mamba (unidirectional, used by SPMamba)
    - models.mamba.bimamba.Mamba (bidirectional, 2 scans, used by MambaTasNet/DPMamba)

    Selective scan formula: ~9 * B * L * D * N per direction (state-spaces/mamba#110).
    """
    hooks = {}

    try:
        from mamba_ssm.modules.mamba_simple import Mamba as MambaUni

        def _uni_hook(module, input, output):
            # input[0]: (B, L, d_model)
            B, L, _ = input[0].shape
            D = module.d_inner
            N = module.d_state
            dt_rank = module.dt_rank
            d_model = module.d_model
            # in_proj: Linear(d_model, 2*D)
            macs = L * d_model * 2 * D
            # conv1d: depthwise, kernel=d_conv
            macs += L * D * module.d_conv
            # x_proj: Linear(D, dt_rank + 2*N)
            macs += L * D * (dt_rank + 2 * N)
            # dt_proj: Linear(dt_rank, D)
            macs += L * dt_rank * D
            # selective_scan: 9*L*D*N
            macs += 9 * L * D * N
            # out_proj: Linear(D, d_model)
            macs += L * D * d_model
            module.__flops__ += B * macs

        hooks[MambaUni] = _uni_hook
    except ImportError:
        pass

    try:
        from models.mamba.bimamba import Mamba as MambaBi

        def _bi_hook(module, input, output):
            # input[0]: (B, L, d_model)
            B, L, _ = input[0].shape
            D = module.d_inner
            N = module.d_state
            dt_rank = module.dt_rank
            d_model = module.d_model
            # in_proj: Linear(d_model, 2*D) — shared, runs once
            macs = L * d_model * 2 * D
            # Per direction (x2): conv1d + x_proj + dt_proj + scan
            macs += 2 * L * D * module.d_conv
            macs += 2 * L * D * (dt_rank + 2 * N)
            macs += 2 * L * dt_rank * D
            macs += 2 * 9 * L * D * N
            # out_proj: Linear(D, d_model) — shared, runs once
            macs += L * D * d_model
            module.__flops__ += B * macs

        hooks[MambaBi] = _bi_hook
    except ImportError:
        pass

    return hooks


def count_macs(model: torch.nn.Module, input_shape: tuple, device: str) -> Optional[float]:
    """Count MACs using ptflops + custom Mamba hooks.

    Standard ops (Linear, Conv1d, etc.) are counted by ptflops. Mamba modules
    are counted analytically via custom_modules_hooks since ptflops can't see
    inside the selective scan CUDA kernel.
    """
    mamba_hooks = _build_mamba_hooks()
    try:
        macs, _ = get_model_complexity_info(
            model, input_shape, as_strings=False,
            print_per_layer_stat=False, verbose=False,
            input_constructor=lambda s: torch.randn(1, *s, device=device),
            custom_modules_hooks=mamba_hooks,
        )
        if macs is not None and macs > 0:
            return macs / 1e9  # GMACs
        return None
    except Exception as e:
        print(f"    ptflops failed: {e}")
        return None


def benchmark_model(
    name: str,
    config_path: str,
    device: str,
    duration: float,
    n_warmup: int,
    n_runs: int,
) -> Optional[BenchmarkResult]:
    """Benchmark a single model configuration."""
    config_path = Path(config_path)
    if not config_path.exists():
        print(f"  SKIP {name}: config not found ({config_path})")
        return None

    try:
        config = load_config_from_yaml(str(config_path))
    except Exception as e:
        print(f"  SKIP {name}: config load failed ({e})")
        return None

    # Create model
    try:
        model = create_model_from_config(config.model)
    except Exception as e:
        print(f"  SKIP {name}: model creation failed ({e})")
        return None

    model = model.to(device)
    model.eval()

    num_params = count_parameters(model)
    params_str = format_parameter_count(num_params)

    # Create input: batch_size=1, mono, 4 seconds at 8kHz
    num_samples = int(duration * SAMPLE_RATE)
    input_shape = (1, num_samples)  # (channels, samples) — model expects (B, 1, T)
    x = torch.randn(1, 1, num_samples, device=device)

    # --- GMACs (on CUDA so Mamba kernels work) ---
    # ptflops creates internal model copies — clean up before memory measurement
    gmacs = count_macs(model, input_shape, device)
    gflops = gmacs * 2 if gmacs is not None else None
    gc.collect()
    torch.cuda.empty_cache()

    # --- Peak GPU memory ---
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    with torch.no_grad():
        _ = model(x)

    torch.cuda.synchronize(device)
    peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024

    # --- Inference time ---
    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x)
    torch.cuda.synchronize(device)

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            _ = model(x)
            torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)  # ms

    times.sort()
    median_time = times[len(times) // 2]
    # IQR-based std (robust)
    q1 = times[len(times) // 4]
    q3 = times[3 * len(times) // 4]
    std_time = (q3 - q1) / 1.35  # approximate std from IQR

    # Cleanup — ensure GPU memory is fully freed before next model
    del model, x
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    return BenchmarkResult(
        name=name,
        params=num_params,
        params_str=params_str,
        gmacs=gmacs,
        gflops=gflops,
        inference_time_ms=median_time,
        inference_std_ms=std_time,
        peak_memory_mb=peak_memory_mb,
    )


def main():
    parser = argparse.ArgumentParser(description="Inference benchmark for all architectures")
    parser.add_argument("--device", default="cuda", help="Device (default: cuda)")
    parser.add_argument("--duration", type=float, default=4.0, help="Input duration in seconds (default: 4.0)")
    parser.add_argument("--n-warmup", type=int, default=10, help="Warmup forward passes (default: 10)")
    parser.add_argument("--n-runs", type=int, default=50, help="Timed forward passes (default: 50)")
    parser.add_argument("--output", default="benchmark_inference.csv", help="Output CSV path")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Benchmark requires GPU.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(args.device)
    print(f"GPU: {gpu_name}")
    print(f"Input: {args.duration}s @ {SAMPLE_RATE}Hz = {int(args.duration * SAMPLE_RATE)} samples")
    print(f"Runs: {args.n_warmup} warmup + {args.n_runs} timed")
    print(f"Memory: torch.cuda.max_memory_allocated (tensor allocations only, excludes CUDA context)")
    print()

    results = []
    for name, config_path in CONFIGS:
        print(f"Benchmarking {name}...")
        result = benchmark_model(
            name, config_path, args.device,
            args.duration, args.n_warmup, args.n_runs,
        )
        if result:
            gmacs_str = f"{result.gmacs:.2f}" if result.gmacs else "N/A"
            gflops_str = f"{result.gflops:.2f}" if result.gflops else "N/A"
            print(f"  {result.params_str} params | {gmacs_str} GMACs | {gflops_str} GFLOPs | "
                  f"{result.inference_time_ms:.1f}ms | {result.peak_memory_mb:.0f} MB")
            results.append(result)
        print()

    # Print summary table
    print("=" * 115)
    print(f"{'Model':<25} {'Params':>10} {'GMACs':>10} {'GFLOPs':>10} {'Inference (ms)':>16} {'Peak Mem (MB)':>15}")
    print("-" * 115)
    for r in results:
        gmacs_str = f"{r.gmacs:.2f}" if r.gmacs else "N/A"
        gflops_str = f"{r.gflops:.2f}" if r.gflops else "N/A"
        time_str = f"{r.inference_time_ms:.1f} +/- {r.inference_std_ms:.1f}"
        print(f"{r.name:<25} {r.params_str:>10} {gmacs_str:>10} {gflops_str:>10} {time_str:>16} {r.peak_memory_mb:>14.0f}")
    print("=" * 115)
    print(f"GPU: {gpu_name}")
    print(f"Note: Peak memory = tensor allocations only (excludes ~300-500MB CUDA context overhead)")

    # Save CSV
    output_path = Path(args.output)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "params", "gmacs", "gflops", "inference_ms", "inference_std_ms", "peak_memory_mb", "gpu"])
        for r in results:
            writer.writerow([
                r.name, r.params,
                f"{r.gmacs:.4f}" if r.gmacs else "",
                f"{r.gflops:.4f}" if r.gflops else "",
                f"{r.inference_time_ms:.2f}",
                f"{r.inference_std_ms:.2f}",
                f"{r.peak_memory_mb:.1f}",
                gpu_name,
            ])
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
