# Handoff: torch 2.14 venv upgrade on the 3080 box

Written 2026-09-08 on the 4070 box after doing the same upgrade there. Follow it in
order; every step that touches the GPU is marked. Do **not** modify `venv/` — it is
the thesis reproducibility environment. Build a sibling `venv_t214/` instead.

## 0. Preconditions (read-only, do these first)

1. `nvidia-smi` and `tmux ls`. If anything is training (queue in `~/queue_3080.sh`),
   do not run any GPU step below until it is finished. CPU-only steps (venv creation,
   pip installs, the Mamba source build) are fine while training runs, but cap the
   build with `MAX_JOBS` so the DataLoader workers keep their cores.
2. Driver: the header of `nvidia-smi` must show `CUDA Version: 13.x`. WSL inherits
   the Windows driver, so if it shows 12.x, update the Windows driver first and
   reboot. cu130 wheels will not load on a 12.x driver.
3. Toolkit: `nvcc --version` must be 13.x and `gcc --version` ≥ 13 (C++20 needed).
   If `/usr/local/cuda` is 12.x, install a 13.x toolkit alongside it from NVIDIA's
   WSL-Ubuntu repo (`cuda-toolkit-13-0`, **not** the `cuda` meta-package, which would
   pull a Linux driver into WSL) and point `/usr/local/cuda` or `CUDA_HOME` at it.
4. GPU arch for the build: RTX 3080 = `sm_86`, so `TORCH_CUDA_ARCH_LIST="8.6"`.
5. This WSL has no systemd; nothing below needs it.

## 1. Build the venv (CPU only, ~10 min plus download)

```bash
cd ~/polsess_separation && git pull        # picks up requirements.txt with python-multipart
python3 -m venv venv_t214
venv_t214/bin/pip install --upgrade pip wheel "setuptools<81" ninja packaging
venv_t214/bin/pip install "torch==2.14.0" torchaudio torchcodec     # PyPI default = cu130
venv_t214/bin/pip install -r requirements.txt
# pip's resolver downgrades two packages the old venv has at newer versions; re-pin:
venv_t214/bin/pip install "pyannote.audio==4.0.4" "torchmetrics==1.9.0"
# ASR-pipeline extras that were only transitive in the old venv:
venv_t214/bin/pip install librosa meeteval ptflops
```

`pip check` will report exactly one line — `asteroid 0.7.0 has requirement
torchmetrics<=0.11.4` — which the old venv has too. Anything else is a problem.

## 2. Mamba kernels from source (CPU only, ~25 min)

Official mamba-ssm / causal-conv1d wheels stop at torch 2.10. The sdists hard-code
`-std=c++17` and torch 2.14's ATen headers `#error` on anything below C++20, so the
setup.py must be patched before building.

```bash
mkdir -p /tmp/mamba_build && cd /tmp/mamba_build
~/polsess_separation/venv_t214/bin/pip download --no-deps --no-binary :all: --no-build-isolation \
    "causal-conv1d==1.7.0" "mamba-ssm==2.3.2.post1"
tar xzf causal_conv1d-1.7.0.tar.gz && tar xzf mamba_ssm-2.3.2.post1.tar.gz
sed -i 's/-std=c++17/-std=c++20/g' causal_conv1d-1.7.0/setup.py mamba_ssm-2.3.2.post1/setup.py
cd ~/polsess_separation
export TORCH_CUDA_ARCH_LIST="8.6" MAX_JOBS=8 CAUSAL_CONV1D_FORCE_BUILD=TRUE MAMBA_FORCE_BUILD=TRUE
venv_t214/bin/pip install --no-build-isolation /tmp/mamba_build/causal_conv1d-1.7.0
venv_t214/bin/pip install --no-build-isolation /tmp/mamba_build/mamba_ssm-2.3.2.post1
venv_t214/bin/python -c "import mamba_ssm, causal_conv1d; print(mamba_ssm.__version__, causal_conv1d.__version__)"
```

`--no-build-isolation` is mandatory; without it pip builds against a second,
freshly downloaded torch. mamba-ssm also drags in transformers/tilelang/cutlass-dsl;
that is expected.

## 3. Verify (GPU — only when the GPU is idle)

Keep this short; the 4070 pass used 256 mixtures per model and that was enough.

```bash
S=/tmp/t214_check; mkdir -p $S
COMMON="--train-samples 256 --warmup-samples 64 --batch-modes trained --scratch-dir $S/scratch"
# A/B throughput, old then new, same models, back to back, nothing else on the box
for v in venv venv_t214; do
  $v/bin/python scripts/benchmark_training.py --only ConvTasNet "DPRNN (k=16)" SepFormer-reduced \
      MossFormer2-matched "SPMamba (reduced)" TF-MossFormer-S $COMMON --output $S/bench_$v.csv
done
# numerical parity on one real checkpoint (any *_best.pt on this box; use the same one in both venvs)
CK=<path/to/a/best.pt>; DR=<dataset root with train/ val/ test/>
for v in venv venv_t214; do
  $v/bin/python evaluate.py --checkpoint $CK --data-root $DR --variant SER --max-samples 200 \
      --batch-size 1 --no-pesq --no-stoi --output $S/parity_$v.csv
done
# full test suite on the new venv (CPU-heavy, ~10 min; Mamba/CUDA tests now run instead of skipping)
venv_t214/bin/python -m pytest -q -p no:cacheprovider
```

Expected: SI-SDR identical to ~1e-5 dB between venvs; pytest exit 0; throughput on
the 4070 was +18 % ConvTasNet, +8 % DPRNN, +5 % SepFormer-reduced, +9 % TF-MossFormer-S,
flat on MossFormer2 and SPMamba, with a noise floor of about ±5 % at this length.
A number outside that pattern deserves one clean rerun before being believed —
the 4070 pass produced a spurious −27 % on MossFormer2 when nvcc was still
compiling in the background.

## 4. Do not

- `pip install -U` inside `venv/` (the 4070's old venv had cu12 and cu13 NVIDIA wheels
  mixed and was silently loading a cu13 cuDNN under a cu12 torch — rebuild, never patch).
- Test `training.deterministic: false` (cuDNN autotune) or fused Adam — both measured
  null-to-negative on the 4070 under torch 2.14.
- Run pytest or benchmarks while a training run is on the GPU.
- Add `venv_t214/` to git — it is already in `.gitignore`.

## 5. Report back

A six-row table old vs new samples/s, the parity numbers, the pytest verdict, and
anything that deviated from this document. Adoption as the default venv (alias,
`webapp/run.sh`) is the author's call, not the session's.
