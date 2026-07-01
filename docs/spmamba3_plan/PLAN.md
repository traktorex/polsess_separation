# SPMamba3 — Implementation Plan

Add a `spmamba3` model: SPMamba with the Mamba-1 selective-scan blocks replaced
by Mamba-3 (SISO). Separate file `models/spmamba3.py` mirroring `models/spmamba.py`
(no fork-by-flag). Read `RESEARCH.md` first. Branch: `feature/spmamba3-rebuild`.

---

## AS-BUILT (2026-06-27) — implemented & verified

This plan was executed. Outcome notes (where reality diverged from the pre-build
uncertainties):

- **Env (resolves U1/U2/U3 and Q1).** The heavy torch 2.11 fallback was **not**
  needed. `venv_mamba3` = clone of main `venv` (torch 2.8.0+cu128) + `mamba-ssm`
  source build (`@0048fbf2`, CUDA 12.9 toolkit) + **triton bumped 3.4.0→3.5.1**.
  Mamba-3 requires `triton>=3.5`; the SISO Triton kernel raises a `tl.dot`
  `CompilationError` on triton 3.4 (failure was triton-version, not headdim).
  After the bump: forward + backward pass at all SPMamba shapes incl. seqlen 1/8
  (R2 fine — `chunk_size=64` pads internally), bf16 accepted (U3). SISO needs
  only Triton — no tilelang/cutlass/quack (those were over-spec for MIMO).
- **R1 confirmed & fixed.** Ground truth in `mamba3.py:92-94`:
  `assert (expand * d_model) % headdim == 0`. The model + tests use this rule
  (not the old `in_channels % headdim`). `expand` is now a knob (default 2).
- **Integration was fully additive.** `factory.py`/`trainer.py`/`model_utils.py`
  needed **zero** edits (all key off `MAMBA_MODELS`, which now includes
  `spmamba3`). `spmamba3` is registered only when `MAMBA3_AVAILABLE` (its own flag
  separate from `MAMBA_AVAILABLE`); the module import is silent in the main venv.
- **Verified:** main-venv imports clean (no Mamba-3 warning, not registered);
  `tests/test_spmamba3.py` 15/15 pass in venv_mamba3; existing config+mamba tests
  76/76 pass in main venv (no regression); 1-epoch train smoke ran end-to-end
  (trainer shows `AMP: True (bf16, no GradScaler)`, best ckpt saved); checkpoint
  reload round-trip = 0 missing / 0 unexpected keys.
- **Not done (author-gated, Q4):** the {16k,32k,64k} scaling curve YAMLs — only
  `spmamba3_baseline.yaml` (full, emb_ks=8) + `spmamba3_sb_reduced.yaml`
  (4/192/emb_ks=4, matches the thesis reduced-SPMamba arch) were created. Generate
  scaling configs by copying `experiments/spmamba/5-scaling/*` once the arm is
  picked. **Not committed** — work sits on the branch pending review.

Constraints honored: do not modify existing models, additive registry/config
integration only (the old `feature/spmamba3` branch's destructive diffs are
**reference only** — see §6), thesis-code principles (clarity > cleverness, no
new abstraction).

---

## 1. Environment (recreate `venv_mamba3/`)

The broken `venv_mamba3/` (no `bin/python`) must be removed and rebuilt.
Recommendation: **two venvs** — keep the main `venv/` on `mamba-ssm 2.3.1`
(reproducibility for the existing `spmamba`/`mamba_tasnet`/`dpmamba` runs) and
put the Mamba-3 source build in `venv_mamba3/` so the upgrade can't perturb
trained Mamba-1 models. This matches the existing CLAUDE.md two-venv pattern.

Recipe (clone-from-main strategy, minimal):

```bash
rm -rf venv_mamba3
python -m venv venv_mamba3 --system-site-packages   # OR clone main venv libs
# Bring in the main toolchain first (torch 2.8.0 + triton 3.4.0 as installed),
# then build Mamba-3 from source WITHOUT pulling a different torch:
venv_mamba3/bin/pip install torch==2.8.0   # only if not inherited
venv_mamba3/bin/pip install causal-conv1d  # kept for the Mamba-1 models; harmless
MAMBA_FORCE_BUILD=TRUE venv_mamba3/bin/pip install --no-cache-dir \
    --force-reinstall "git+https://github.com/state-spaces/mamba.git" \
    --no-build-isolation
```

Pin a specific commit (not bare `main`) once the build works, for thesis
reproducibility: `git+https://github.com/state-spaces/mamba.git@<COMMIT_SHA>`.

**Do NOT** preinstall tilelang / cutlass-dsl / quack — not needed for SISO (§5
of RESEARCH). Add them only if you later switch `is_mimo=True`.

**Verification snippet** (must pass before writing any model code):
```python
import torch
from mamba_ssm import Mamba3
m = Mamba3(d_model=64, d_state=64, headdim=32, is_mimo=False,
           dtype=torch.bfloat16).cuda()
x = torch.randn(2, 100, 64, device="cuda", dtype=torch.bfloat16)
y = m(x)
print(y.shape, y.dtype)           # expect (2, 100, 64)
assert torch.isfinite(y).all()
```

**Flagged uncertainties (resolve during this step):**
- (U1) Whether the build succeeds on torch 2.8 / triton 3.4. If the SISO Triton
  kernel import/compile fails → fall back to `torch==2.11.*` + `triton 3.6.*` in
  `venv_mamba3` (the prior attempt's pins), then rebuild mamba from source.
- (U2) Whether the `2.3.2.post1` PyPI wheel already contains `Mamba3` (would let
  us skip the source build). Quick check: `pip download mamba-ssm==2.3.2.post1`
  and inspect for `mamba3.py` + kernels. Source build is the safe default.
- (U3) bf16 acceptance of the SISO kernel (confirmed by the snippet above).

---

## 2. Model design — `models/spmamba3.py`

Structure mirrors `models/spmamba.py`. **Reuse verbatim** (copy into the new
file; do not import across model files): `LayerNormalization4D`,
`LayerNormalization4DCF`, the attention half of `GridNetBlock`, and the entire
`SPMamba.forward` STFT/iSTFT body + RMS normalize/denormalize. **Swap only the
recurrent block.**

### 2.1 Imports / availability guard
```python
try:
    from mamba_ssm import Mamba3
    from mamba_ssm.ops.triton.layer_norm import RMSNorm
    MAMBA3_AVAILABLE = True
except ImportError as e:
    MAMBA3_AVAILABLE = False
    warnings.warn(..., UserWarning, stacklevel=2)
```
(The file imports cleanly even without Mamba-3 — the guard defers the hard error
to construction, so registry import in `models/__init__.py` is always safe.)

### 2.2 The swap point — `Mamba3Block` replaces `MambaBlock`
The Mamba-1 `MambaBlock` (`models/spmamba.py:41-118`) wraps `Mamba` in
`mamba_ssm.modules.block.Block` (fused add-norm + `_init_weights`). Mamba-3 needs
neither. Replace with a plain bidirectional `RMSNorm -> Mamba3 -> residual`:

- Forward dir: `n_layer` × (`RMSNorm(in_channels)` then
  `Mamba3(d_model=in_channels, d_state=..., headdim=..., expand=2, is_mimo=False,
  is_outproj_norm=False, dtype=torch.bfloat16)`), residual add.
- If `bidirectional`: same on `torch.flip(x, [1])`, flip back, `cat([fwd, bwd], -1)`.
- Output dim: `in_channels` (uni) or `in_channels*2` (bi) — **identical to the
  Mamba-1 block**, so the downstream `intra_linear`/`inter_linear`
  `ConvTranspose1d(in_channels*2, emb_dim, ...)` are unchanged.
- Cast around the kernel: `out = mixer(normed.to(torch.bfloat16)).to(orig_dtype)`.

This is exactly what the prior attempt did (§6) and is a sound design — salvage it.

### 2.3 `GridNetBlock` / `SPMamba3`
- `GridNetBlock`: identical to spmamba except `intra_mamba`/`inter_mamba` are
  `Mamba3Block(in_channels, 1, bidirectional=True, d_state=d_state, headdim=headdim)`.
  `hidden_channels`/`lstm_hidden_units` becomes a no-op kept only for config
  symmetry (document it; Mamba-3 has no equivalent single hidden knob).
- `SPMamba3.__init__`: same signature as `SPMamba.__init__` **plus** `d_state`
  and `headdim`; thread them into every `GridNetBlock`.
- `SPMamba3.forward`: copy `SPMamba.forward` unchanged (STFT → conv → blocks →
  deconv → iSTFT, RMS norm/denorm, the bf16→float cast before `torch.complex`).

### 2.4 Risks specific to the Mamba-3 API (call out, verify in tests)
- **(R1) Head-divisibility constraint.** Mamba-3 inner dim = `expand * d_model`,
  `nheads = expand*d_model / headdim`. The real constraint is
  `(expand * in_channels) % headdim == 0`, **not** just `in_channels % headdim`
  as the prior attempt asserted. With `in_channels = emb_dim * emb_ks` and
  `expand=2`: e.g. `emb_dim=16, emb_ks=4 → in_channels=64 → expand*64=128`,
  `headdim=32 → 4 heads` ✓. Validate against the actual constructor and write the
  assertion to match Mamba-3's own check, or just let Mamba-3 raise and surface
  its message. Also confirm `d_state % ngroups == 0`.
- **(R2) Short sequences vs `chunk_size`.** GridNet intra-sequences can be short
  (≈ n_freqs unfold chunks). Default `chunk_size=64` should pad internally;
  verify the forward smoke test at the real STFT shapes (n_fft=256 → 129 freqs).
- **(R3) `lstm_hidden_units` is now dead.** Keep for config parity but document;
  capacity is governed by `emb_dim`, `n_layers`, `d_state`, `headdim`, `expand`.
- **(R4) Determinism / dtype.** bf16 kernel output feeds an fp32 iSTFT; the
  existing `if batch.dtype == torch.bfloat16: batch = batch.float()` guard covers
  it. Keep it.

---

## 3. Codebase integration checklist (ADDITIVE — preserve existing gating)

> The old branch *replaced* the registry and deleted `MAMBA_AVAILABLE`,
> `MAMBA_MODELS`, mossformer2, etc. Do **none** of that. All edits below are
> insertions alongside the current code.

1. **`models/__init__.py`**
   - Inside the existing `if MAMBA_AVAILABLE:` block, add
     `from .spmamba3 import SPMamba3` and `'spmamba3': SPMamba3` to the
     `MODELS.update({...})`.
   - Add `'spmamba3'` to the `MAMBA_MODELS` tuple (line ~36) → gives bf16/no-GradScaler
     in the trainer **and** torch.compile-skip in model_utils for free.
   - Add `'SPMamba3'` to the `if MAMBA_AVAILABLE:` `__all__ +=` line.
   - Update the `get_model` hint tuple `('spmamba','mamba_tasnet','dpmamba')` to
     include `'spmamba3'`.
   - **Gating note:** `MAMBA_AVAILABLE` tests the Mamba-1 path only. Registering
     `spmamba3` under it is fine because the `spmamba3.py` import never hard-fails
     and construction raises a clear Mamba-3-specific error if absent. (Optionally
     expose a separate `MAMBA3_AVAILABLE` for messaging — not required.)

2. **`config.py`**
   - Add `@dataclass SPMamba3Params` after `SPMambaParams` (~line 116): all
     `SPMambaParams` fields **plus** `d_state: int = 64` and `headdim: int = 32`.
     Default `emb_ks=4` (so `in_channels=64`, satisfies R1 with headdim 32). Do
     **not** add a `sample_rate` field (the project removed it from SPMamba;
     keeping it out avoids re-introducing the backward-compat pop).
   - `ModelConfig`: add field `spmamba3: Optional[SPMamba3Params] = None`; add the
     `elif self.model_type == "spmamba3" and self.spmamba3 is None:` branch in
     `__post_init__`.
   - `Config.__post_init__` task→n_srcs (lines 268-298): add `spmamba3` branches
     setting `self.model.spmamba3.n_srcs = 1`/`2` (mirrors the `spmamba` branch).
   - `print summary` (~line 371): add an `elif mt == "spmamba3":` block (clone the
     spmamba one; optionally append `d_state`/`headdim`).
   - AMP-summary tuple (~line 414) and any other `("spmamba","mamba_tasnet",
     "dpmamba","mossformer2")` literals: add `"spmamba3"`.
   - `load_config_from_dict` loader (~line 475): add
     `spmamba3_dict = model_dict.pop("spmamba3", None)` →
     `SPMamba3Params(**spmamba3_dict)`, pass `spmamba3=spmamba3_params` into
     `ModelConfig(...)`.
   - `save_config_to_yaml` (~line 536): add the `elif key == "spmamba3"` branch
     and append `"spmamba3"` to the final exclusion list.

3. **`models/factory.py`** — no change needed. `create_model_from_config` is
   generic (`getattr(config, config.model_type)` + `**vars(params)`); it works
   once the dataclass and registry exist.

4. **`training/trainer.py`** — no change needed. `_setup_amp` keys off
   `MAMBA_MODELS` membership (step 1) → spmamba3 automatically gets
   bf16 + no GradScaler. The `ConsecutiveNaNError` abort applies unchanged.

5. **`utils/model_utils.py`** — no change needed. `compile_for_model_type` skips
   anything in `MAMBA_MODELS` → spmamba3 torch.compile-skip is automatic.

6. **`experiments/spmamba3/`** — add `spmamba3_baseline.yaml` (and a `_reduced`
   variant). Mirror `experiments/spmamba/spmamba_sb.yaml` exactly, change
   `model_type: spmamba3`, key `spmamba3:`, add `d_state`/`headdim`, set
   `emb_ks: 4`. Keep `use_amp: true` (bf16 path) — but see Open Question Q3.
   For the capacity curve, additionally mirror
   `experiments/spmamba/5-scaling/{16k,32k,64k}_full.yaml`.

7. **`tests/test_spmamba3.py`** — port the prior attempt's tests (§6); they are
   well-shaped (CUDA-skip guard, reduced config, forward/length/n_srcs/headdim
   checks). Fix the headdim-constraint test to match the real R1 rule. Add a
   param-count sanity test sized to whatever the baseline lands at.

8. **`CLAUDE.md`** — after it works, update: Repository Overview model list,
   Model Registry section (add `spmamba3` next to `spmamba`, note "Mamba-3 SISO,
   source-built mamba-ssm, `venv_mamba3`"), Virtual Environments section (rewrite
   the stale `venv_mamba3` description to the real recipe), and the MAMBA_MODELS
   note. Targeted edits, not a rewrite (per CLAUDE.md upkeep rule).

---

## 4. Reuse / salvage vs redo — the old `feature/spmamba3` branch

The stale branch forked from a much older repo state and its non-model diffs are
**destructive** (delete mossformer2, MAMBA_AVAILABLE gating, MAMBA_MODELS,
per-variant val, NaN-abort, etc.). **Do not merge or cherry-pick its config/
trainer/registry diffs.** Re-author the integration the current additive way (§3).

**Salvage (port, lightly fixed):**
- `models/spmamba3.py` — its `Mamba3Block` (manual `RMSNorm→Mamba3→residual`,
  bidirectional flip/cat, bf16 cast) is a correct, clean swap. Its `GridNetBlock`
  and `SPMamba3.forward` are faithful copies of spmamba with only the block
  swapped. **Reuse as the basis.** Fix: the headdim assertion (R1 — should be on
  `expand*in_channels`, not `in_channels`); drop the `sample_rate` field to match
  current SPMamba; re-check the `Mamba3(...)` kwargs against the verified
  signature in RESEARCH §3 (the prior code only passed
  `d_model,d_state,headdim,is_mimo,is_outproj_norm,dtype` — fine, all valid).
- `tests/test_spmamba3.py` — reuse wholesale; only the headdim test needs the R1
  fix, and `sample_rate` must not be passed.
- The two YAMLs — reuse shape; but drop `sample_rate`, and note their
  `data_root` points at the old `PolSESS_C_both` (fine for a baseline; use
  `C_new_64` + `train_max_samples` for the scaling configs).

**Redo from scratch:** `models/__init__.py`, `config.py`, `training/trainer.py`,
`utils/model_utils.py`, `models/factory.py` integration — additive against today's
`main`.

---

## 5. Open questions / decisions for the author

- **Q1 (env pins).** Confirm whether torch 2.8 / triton 3.4 builds and runs the
  Mamba-3 SISO kernel, or whether to pin torch 2.11 / triton 3.6 in `venv_mamba3`
  (U1). Decide the exact mamba commit SHA to pin for the thesis.
- **Q2 (one venv or two).** Plan recommends two (protect Mamba-1 reproducibility).
  Author may prefer a single venv if willing to re-validate spmamba/mamba_tasnet/
  dpmamba on the upgraded mamba-ssm.
- **Q3 (AMP).** `use_amp: true` routes through bf16 autocast; the block also
  hard-casts to bf16. Confirm this double-bf16 is fine, or set `use_amp: false`
  and rely solely on the in-block cast (the old baseline YAML used
  `use_amp: false`). Pick one and be consistent.
- **Q4 (capacity / fairness).** To compare against SPMamba honestly, size
  spmamba3 to a comparable param budget. Note from MEMORY: the thesis SPMamba
  runs used a *reduced* 4-layer/192 config and a separate "paper recipe" scaling
  set exists. Decide whether spmamba3 mirrors the reduced config, the paper
  config, or both — and whether to run the {16k,32k,64k} scaling curve.
- **Q5 (SISO vs MIMO).** Plan fixes `is_mimo=False` (no TileLang). Revisit only
  if a MIMO ablation is wanted — that pulls in TileLang and a heavier env.

---

## 6. Step-ordered task list

1. **Env:** remove broken `venv_mamba3`, rebuild (§1), run the verification
   snippet. Resolve U1–U3. Pin the mamba commit. **Gate: snippet passes.**
2. **Model:** write `models/spmamba3.py` from the salvaged prior version with the
   R1 fix + `sample_rate` drop (§2). Smoke-test a forward pass standalone.
3. **Config:** add `SPMamba3Params` + all `config.py` wiring (§3.2).
4. **Registry:** additive edits to `models/__init__.py` (§3.1).
5. **Verify plumbing:** `create_model_from_config` builds spmamba3; trainer picks
   bf16/no-GradScaler; compile is skipped. (No edits expected in factory/trainer/
   model_utils — confirm.)
6. **Configs:** add `experiments/spmamba3/*.yaml` (baseline + reduced; optionally
   scaling) (§3.6).
7. **Tests:** add `tests/test_spmamba3.py` (§3.7); `pytest tests/test_spmamba3.py`
   on a CUDA box (skips elsewhere).
8. **Smoke train:** 1–2 epochs on a tiny subset to confirm the loop, AMP path,
   and checkpoint save/load round-trip.
9. **CLAUDE.md:** targeted update (§3.8).
10. **(Optional) Scaling/eval** per Q4.
