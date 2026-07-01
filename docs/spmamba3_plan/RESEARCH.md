# SPMamba3 — Research: Mamba-1 → Mamba-3 and the concrete API

Scope: background needed to add a `spmamba3` model (SPMamba with the Mamba-1
selective-scan blocks swapped for Mamba-3). Companion file: `PLAN.md`.

## 1. Sources

- Together.ai blog, "Mamba-3": https://www.together.ai/blog/mamba-3
- Paper: "Mamba-3: Improved Sequence Modeling using State Space Principles
  Through Structured State Space Duality" (Lahoti, K. Li, B. Chen, C. Wang,
  Bick, Kolter, Dao, Gu), arXiv:2603.15569 — https://arxiv.org/abs/2603.15569
- Repo: https://github.com/state-spaces/mamba (README, `mamba_ssm/modules/mamba3.py`)
- Installed/available wheels confirmed locally via `venv/bin/pip index versions mamba-ssm`.

## 2. Mamba-1 → Mamba-3 architectural delta

Per the blog and paper, Mamba-3 keeps the Mamba-2 SSD (structured state-space
duality) framework and adds three changes plus several refinements:

1. **More expressive recurrence (exponential-trapezoidal discretization).**
   Mamba-1/2 use a first-order (effectively Euler/ZOH) discretization of the
   continuous SSM. Mamba-3 uses a second-order trapezoidal rule, giving a more
   general state-transition than the simplified per-step decay of prior versions.

2. **Complex-valued state tracking via rotations (RoPE).** Mamba-1/2 states are
   real-valued. Mamba-3 models a complex-valued SSM; the complex transition is
   realized as a rotation, implemented with RoPE-style modules
   (`rope_fraction=0.5` by default — half the head channels get rotary phase).
   This is reported to let the SSM track periodic / position-relative structure
   that a real diagonal state cannot.

3. **MIMO SSMs.** Mamba-1 is single-input single-output (SISO) per channel;
   Mamba-3 can run multiple SSMs in parallel as a multi-input multi-output block
   (`is_mimo=True`, `mimo_rank`), raising accuracy without slowing decode. MIMO
   is **optional** and off by default (`is_mimo=False`).

Refinements:
- **Short causal conv removed.** Mamba-1/2 prepend a `d_conv=4` causal depthwise
  conv1d to the SSM input. Mamba-3 drops it; the new discretization + bias terms
  absorb the local-mixing role. (Consequence for us: the Mamba-1 SPMamba block
  passes `d_conv=4`; Mamba-3 has **no `d_conv` argument**.)
- **QK-Norm** added for training stability (Transformer-style), exposed indirectly
  via `fuse_pregate_headwise_norm=True` / `is_outproj_norm`.
- **Interleaved MLPs** in the full LM stack (not relevant to our GridNet reuse —
  SPMamba uses `mlp_cls=nn.Identity`).

## 3. Concrete Mamba-3 module API (vs Mamba-1)

**Mamba-1 (what `models/spmamba.py:27` uses today):**
```python
from mamba_ssm.modules.mamba_simple import Mamba
Mamba(d_model, layer_idx=i, d_state=16, d_conv=4, expand=4)   # via mamba_ssm Block wrapper
```

**Mamba-3 (verified from `mamba_ssm/modules/mamba3.py` on `main`):**
```python
from mamba_ssm import Mamba3        # top-level re-export confirmed in README
Mamba3(
    d_model,                 # required
    d_state=128,
    expand=2,
    headdim=64,              # d_model*expand must be divisible by headdim (see risk)
    ngroups=1,
    rope_fraction=0.5,
    dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, A_floor=1e-4,
    is_outproj_norm=False,
    is_mimo=False,           # SISO by default
    mimo_rank=4,
    fuse_pregate_headwise_norm=True,
    chunk_size=64,
    dropout=0.0,
    layer_idx=None, n_layer=None,
    device=None, dtype=None,
)
```
- `forward(u, seq_idx=None, cu_seqlens=None, inference_params=None)`:
  input `u` shape `(batch, seqlen, d_model)`, output **same shape**. (Same I/O
  contract as Mamba-1's mixer, so it drops into the existing
  `[B, T, C] -> [B, T, C]` bidirectional wrapper without reshaping changes.)
- **No `d_conv`** argument (conv removed). `expand` default dropped from 4→2.
- The Mamba-1 path in SPMamba wraps the mixer in `mamba_ssm.modules.block.Block`
  (fused add+norm + `_init_weights`). Mamba-3 does its own internal init/norms;
  the natural pattern is a plain `RMSNorm -> Mamba3 -> residual` (no `Block`).

`RMSNorm` import path is unchanged and still valid:
`from mamba_ssm.ops.triton.layer_norm import RMSNorm`.

## 4. Package / version availability — findings

- Local main venv: `torch 2.8.0`, `triton 3.4.0`, `mamba-ssm 2.3.1` (installed),
  `2.3.2.post1` is the latest PyPI wheel.
- **Mamba-3 is NOT in the installed 2.3.1.** The Mamba-3 modules (`mamba3.py`,
  the `Mamba3` re-export, and the SISO/MIMO kernels) live on the repo `main`
  branch. The README instructs a **source build**, not a wheel:
  ```
  MAMBA_FORCE_BUILD=TRUE pip install --no-cache-dir --force-reinstall \
      git+https://github.com/state-spaces/mamba.git --no-build-isolation
  ```
  **UNCERTAIN:** whether the published `2.3.2.post1` wheel already ships
  `mamba3.py` + compiled kernels. The README pointing at a source build implies
  the wheel may not carry the new Triton/TileLang/CuTe kernels. Treat "source
  build from `main`" as the supported path; verify a pinned commit (see PLAN).

## 5. Kernel / hardware requirements — findings (important nuance)

The blog advertises a three-kernel stack — **Triton** (prefill), **TileLang**
(MIMO projections), **CuTe DSL** (decode step). But `mamba3.py` imports them
**softly** and only invokes each under specific conditions:

| Scenario                         | Kernel actually used                       |
|----------------------------------|--------------------------------------------|
| Training, `is_mimo=False` (SISO) | **Pure Triton** (`mamba3_siso_combined`)   |
| Training, `is_mimo=True`         | TileLang (`mamba3_mimo_combined`)          |
| Autoregressive decode step       | CuTe (`mamba3_step_fn`), if available      |

For speech separation we run **full-sequence forward only** (no token-by-token
decode) and the bidirectional GridNet blocks are small — so **`is_mimo=False`
(SISO) is the natural choice**, which needs **only Triton**. That means:
- **TileLang, CuTe/CUTLASS, and `quack-kernels` are NOT required** for the SISO
  path we will use. The CLAUDE.md note on the *prior* `venv_mamba3`
  ("tilelang, quack-kernels, cuda-bindings, nvidia-cutlass-dsl") was almost
  certainly provisioning for MIMO/decode and is **over-spec for our use**.
- `causal-conv1d` is **not imported** by `mamba3.py` (conv removed) — though it
  is still wanted by the existing Mamba-1 models, so a shared env should keep it.
- README baseline reqs: **PyTorch 1.12+, CUDA 11.6+, Linux + NVIDIA GPU**.
  Hopper (H100) was used for benchmarks but is not stated as required.
- **UNCERTAIN:** the Mamba-3 Triton SISO kernel may use Triton features newer
  than 3.4.0. The prior attempt used `triton 3.6.0` + `torch 2.11`. Plan keeps
  this as the documented fallback if the SISO kernel fails to import/run on the
  current torch 2.8 / triton 3.4 toolchain.

## 6. dtype note

Mamba-3's combined Triton kernels run in low precision; the prior attempt's
wrapper instantiated `Mamba3(..., dtype=torch.bfloat16)` and cast activations to
bf16 around each call. This is consistent with how the project already treats
Mamba models (bf16 autocast, no GradScaler — `training/trainer.py:_setup_amp`).
**UNCERTAIN but low-risk:** confirm the SISO kernel accepts bf16 input on the
chosen toolchain during the venv smoke test.
