# ASR pipeline — Life 2 kickoff: extraction + separator import

Drafted 2026-09-18 (post-thesis). Status 2026-09-22: **done and committed on
both sides** (parent `aa112d8` + tag `polsess-models-v0.1.0` pushed; DGSS-ASR
`25d7760` + `53aab7c`, no remote yet) — see the end of §7 for what is left.
§1–§6 are the proposal as written before the rulings.

Frame (author, 2026-09-18): the paper comes first and is built mostly on thesis
work, so `polsess_separation` — including its copy of `asr_pipeline/` — stays
intact as the paper's instrument. Life 2 opens with two tasks only:

- **A.** stand up `asr_pipeline` as an independent codebase, *without* removing
  it from this repo (removal happens after the paper);
- **B.** decide how that codebase consumes separator checkpoints trained here.

Both are the "extraction" half of SCOPE §8's "extraction + deployment". Nothing
platform-grade (queues, APIs, hardening) is in scope yet.

## 1. Coupling map (measured 2026-09-18, `main` @ 19be152)

**Outbound — `asr_pipeline/` reaching into the parent: five seams, all shallow.**

| # | Site | What crosses | Note |
|---|------|--------------|------|
| 1 | `stages/separation.py` `SeparationStage.load` (`backend == "repo"`) | `from utils.model_utils import load_model_for_inference` | The real seam. Importing it executes `utils/__init__.py`, which drags in `wandb_logger`, `logger`, `metrics`, `common` — the pipeline loads the training stack to build one `nn.Module`. |
| 2 | `stages/separation.py` `_load_spmamba_external_separator` | `from models.spmamba import SPMamba` | B1 arm only (peer weights in the repo's own class). |
| 3 | `__main__.py` `_resolve_batch_sources` | `from scripts.eval_harness import eval_root, load_split` | Lazy, `--split` only. `eval_harness` is 100 % ASR code that happens to live in `scripts/`. |
| 4 | `stages/diarization.py`, `stages/transcription.py` | `parents[2] / "scripts" / {sortformer,coherex}_worker.py` | Isolated-venv workers located by repo layout. |
| 5 | `config.py`, `configs/*.yaml`, `preflight.py` | repo-relative `checkpoints/...` paths; `checkpoints/external/speechbrain` savedir | Forces CWD = repo root. |

**Inbound — parent code that imports `asr_pipeline`:** `webapp/` (3 files),
27 test files (`tests/test_pipeline_*.py`, `test_webapp_backend.py`; plus one
mention in `test_benchmark_macs.py`), 20 scripts, 2 notebooks in `asr/`,
`scripts/thesis_figures/fig_ch6_corpus_map.py`. Checked: `webapp/`, the pipeline
tests and the ASR scripts import **nothing else** from the parent (`models`,
`utils`, `config`, `training`, `datasets` — zero hits); the tests use only the
generic `device` fixture from `conftest.py` and two sibling script modules
(`run_pipeline_on_recording`, `sweep_pipeline`). `models/` never imports
`asr_pipeline` (one comment reference in `tf_mossformer/tf_locoformer_blocks.py`).

The SCOPE §7 coupling rule held: the boundary is already a package boundary.

## 2. Workstream A — extraction without removal

**Mechanics.** `git filter-repo` on a *fresh clone* of this repo, keeping the
paths below, pushed to a new remote. This repo is not touched; the new repo
keeps the full history of every moved file (blame through the whole sweep
campaign survives).

**Moves (becomes the new repo):**

- `asr_pipeline/` — whole package incl. `vendor/`, `configs/`, `eval/`,
  `SCOPE.md`, `CLAUDE.md`, `SWEEP_KNOBS.md`.
- Workers → into the package: `scripts/sortformer_worker.py`,
  `scripts/coherex_worker.py` (+ `sortformer_batch_worker.py`) →
  `asr_pipeline/workers/`. Closes seam 4.
- `scripts/eval_harness.py` → `asr_pipeline/eval/harness.py`. Closes seam 3.
- Runtime/eval drivers: `run_pipeline_on_recording`, `sweep_pipeline`,
  `rescore_stratified`, `prepare_eval_references`, `dump_sweep_results`,
  `score_attribution_purity`, `transcribe_mixtures`, `compare_asr`.
- `tests/test_pipeline_*.py`, `test_webapp_backend.py`, a trimmed `conftest.py`.
- `webapp/` (front-end to the pipeline; has no place in a separation-training
  repo) — **author's call, §6 q2**.
- `asr/explore_pipeline.ipynb`, `asr/evaluate_pipeline.ipynb`.

**Stays behind (campaign residue, lifetime rule SCOPE §6):** `build_nzr_aids`,
`build_review_page`, `mixref_*`, `squim_validation`, `sweep_asr_decode`,
`sweep_asr_prompt`, `score_fragment_acoustics`, the three
`*_clarin_2speakers*` batch scripts, `enhance_clarin_debleed`,
`thesis_figures/`, `asr/clarin_*.ipynb`, `asr/archive/`. They served the thesis
tables; the paper may still need them *here*, Life 2 does not. Anything that
turns out to be needed is one `git filter-repo` path away.

**Seam cuts, done in the new repo only:**

1. `pyproject.toml`, installable package, console script for the CLI; the ASR
   block of `requirements.txt` becomes its dependency list.
2. Seams 3 + 4 closed by the moves above.
3. Seam 5: one explicit root for model assets (env var + YAML key, absolute
   paths allowed), replacing CWD-relative `checkpoints/...`. Preflight keeps
   failing loud on a missing file (SCOPE §4).
4. Seams 1 + 2: see Workstream B.

**Dual-copy rule (until the paper is done).** The copy in this repo is the
frozen paper instrument — SCOPE §8 already limits it to bugfixes that block
tables. Development happens only in the new repo. A bugfix found here is
cherry-picked across (paths are identical under `asr_pipeline/`, so
`git format-patch | git am` applies cleanly for untouched files).

**Parity gate (needs a free GPU).** Before any behaviour change in the new
repo: shipped-best config, `deterministic: true`, `HF_HUB_OFFLINE=1`, N dev
fragments → transcripts byte-identical to the existing eval tree. That pins
"new repo at t0 == thesis instrument" and makes later divergence intentional.

## 3. Workstream B — how the new codebase gets separators

**What the pipeline actually needs:** an `nn.Module` with
`forward([B, T]) -> [B, n_src, T]` at `separator_sample_rate`. That is the whole
contract. Behind it sit two different things with different lifecycles —
**architecture code** and **weights** — and they should travel separately.

**Facts established 2026-09-18:**

- The deployed checkpoint (`mossformer2_SB_best_e46.pt`) is a *training*
  checkpoint: 318 MB, of which the model is ~106 MB (26.4 M params fp32); the
  rest is optimizer + scheduler state.
- It loads with `torch.load(weights_only=True)` — `config` is a plain dict, no
  pickled project classes. Nothing in the file needs this repo to unpickle.
- `models/` uses relative imports throughout (sole exception: `factory.py`,
  which imports `config`). Third-party deps: `torch`, `speechbrain`, `einops`,
  `rotary_embedding_torch`; `mamba_ssm`/`causal_conv1d` optional.
- **Feasibility test (CPU, scratchpad copy, this repo untouched):** a
  `pyproject.toml` with `package-dir = {polsess_models = "models"}` builds a
  57 KB pure-Python wheel from `models/` *as it stands*; with the loader placed
  inside it, the e46 checkpoint loads from a neutral CWD with no parent on
  `sys.path`, all 8 architectures register, **output is bit-identical** to
  `utils.model_utils.load_model_for_inference` (max abs diff 0.0), and none of
  `utils` / `config` / `training` / `wandb` gets imported.

**Options considered:**

| Option | Verdict |
|--------|---------|
| Keep importing the parent (`sys.path` / CWD) | **No.** Top-level names `models`, `utils`, `config`, `datasets` are collision bait (`datasets` shadows HF's); drags the training stack; unusable on the PJATK server. |
| Vendor architecture code into the ASR repo | **No as the main route.** Fine for frozen third-party code (that is what `vendor/` is for), wrong for the author's own live code — TF-MossFormer is being trained now and will likely replace the 8 kHz separator; a second copy drifts. |
| TorchScript / `torch.export` / ONNX artifacts | **Not now.** Mamba kernels don't export, dynamic-length tracing is its own validation project, and bit-parity with eager is not guaranteed. Revisit if the server needs a Python-free runtime. |
| **`models/` as an installable package + slim weight bundles** | **Recommended.** Below. |

**Recommendation — two channels.**

*Code channel: `polsess-models`, built from this repo.* Add a root
`pyproject.toml` mapping the existing `models/` directory to the import name
`polsess_models`. Training code here keeps `from models import ...` unchanged
(CWD import, as today); consumers import `polsess_models`. Move
`load_model_for_inference` (+ its back-compat shims) into `models/inference.py`
with relative imports; `utils/model_utils.py` re-exports it, so nothing here
changes behaviour. The ASR repo depends on it **pinned to a commit**
(`polsess-models @ git+https://github.com/traktorex/polsess_separation@<sha>`)
or on a wheel copied to the server — so a given ASR release is tied to an exact
architecture revision. Seams 1 + 2 become `from polsess_models...`.

*Weights channel: inference bundles.* `scripts/export_separator.py` here turns a
training checkpoint into a directory:

```
mossformer2_matched_128k_e46/
  weights.safetensors   # model_state_dict only (~106 MB vs 318 MB), no pickle
  separator.json        # model_type, constructor kwargs (shims already applied),
                        # sample_rate, n_src, epoch, val_sisdr,
                        # source-checkpoint sha256, git commit, training-set tag
```

Back-compat shims (SepFormer key rename, SPMamba `sample_rate` pop, `_orig_mod`
strip) run once at export, where the knowledge lives, instead of on every load
downstream. `sample_rate` in the bundle lets the pipeline *check*
`separator_sample_rate` against the model instead of trusting two YAML keys to
agree (the 8 k/16 k + `naive`-BWE foot-gun in `asr_pipeline/CLAUDE.md`). Bundles
are plain files: a directory under the assets root, a GDrive folder, a W&B
artifact or an HF repo all work; no decision needed yet.

Order matters: the code channel alone already cuts the seam (the new repo can
load today's `.pt` training checkpoints through `polsess_models`). Bundles are
step two and only become necessary when something ships to the server.

**Licence flag.** `models/mamba/` carries GPL-3.0 code (Mamba-TasNet). SCOPE §1
says "possibly commercial". The deployed separator (MossFormer2, from
ClearerVoice) is unaffected, but the wheel as built above *contains* the Mamba
files. If that matters, the package list simply omits `polsess_models.mamba`
and the three Mamba model modules — `MAMBA_AVAILABLE` already gates them.

## 4. Order of work

1. Rulings (§6).
2. Parent, small + behaviour-neutral: `models/inference.py`, `utils` re-export,
   root `pyproject.toml`, a test that the two loaders agree. (Touches this repo
   during the paper period — hence a ruling.)
3. New repo: `filter-repo` extraction → packaging → seam cuts 3/4/5 → swap
   seams 1/2 to `polsess_models` → CPU test suite green.
4. Parity gate on GPU when the 4070 is free.
5. `export_separator.py` + bundle loading, when deployment work starts.

## 5. Out of scope here

Removing `asr_pipeline/` from this repo; `.eaf` exporter; N-speaker support;
adaptive stage application; any platform integration (SCOPE §9). The venv
question (the `venv/` ASR stack vs the torch 2.14 build) gets resolved
naturally by the new repo owning its own environment.

## 6. Rulings needed (author)

1. **Code channel** — accept `polsess-models` built from this repo (small
   change here, during the paper period), or prefer vendoring after all?
2. **`webapp/`** — moves with the pipeline, or stays here until needed?
3. **New repo** — name, location, GitHub visibility (decides git-dep vs wheel
   for the server).
4. **Mamba in the package** — include (research convenience) or exclude
   (licence hygiene)?
5. **History** — `filter-repo` with history (recommended) or a clean first
   commit?

## 7. Rulings and execution state (2026-09-18)

**Rulings (author):** (1) code channel accepted — polsess_separation is
responsible for making its checkpoints usable elsewhere, plug-and-play, no
vendoring downstream; keep the dependency easy to reason about. (2) `webapp/`
moves with the pipeline. (3) Name **DGSS-ASR** (Diarization-Guided Speech
Separation for ASR), folder beside this repository, GitHub repo may be public.
(4) Mamba family excluded from the package. (5) `git filter-repo` with history.

**Here — worktree `~/polsess_separation-models`, branch `feature/polsess-models`,
uncommitted** (the live tree was not touched: `lr_branching` spawns fresh
`train.py` processes from it):

- `models/inference.py` — `load_model_for_inference` + `load_checkpoint_file`
  moved here verbatim (split into `resolve_architecture` /
  `normalize_state_dict`), plus `load_separator(bundle_dir)`;
  `utils/model_utils.py` re-exports, so no caller changed.
- `models/__init__.py` — `__version__ = "0.1.0"`, guarded `.mamba` import,
  exports the two loaders.
- `pyproject.toml` — hatchling; `models/` -> `polsess_models`; Mamba family and
  `factory.py` excluded. Changed from the §3 sketch (setuptools `package-dir`):
  setuptools cannot leave single modules out of a package without a custom
  `build_py`, and two of the Mamba model files are GPL-derived.
- `scripts/export_separator.py` — checkpoint -> bundle, with a bit-identical
  round-trip check; `models/README.md` — usage + the maintenance rule;
  `tests/test_polsess_models_package.py`; `requirements.txt` (+safetensors);
  `CLAUDE.md` paragraph.
- Verified on CPU: wheel 53 KB with exactly the intended files; e46 bundle loaded
  through the wheel from a neutral directory is bit-identical to the ORIGINAL
  loader on `main`; 176 passed / 27 skipped over the model, loader and
  separation-stage tests.

**Maintenance rule** (the author's "what if this repository changes?"): only
`models/` is in the package, so training-side changes never reach a consumer. A
consumer pins a tag `polsess-models-vX.Y.Z` and moves it only when a bundle
needs an architecture its installed version lacks — `load_separator` fails loud
and names both versions. Trained architectures must stay loadable anyway
(`evaluate.py` needs the same). Bump `models.__version__` + tag on a
consumer-visible change. Full text: `models/README.md`.

**New repository — `~/DGSS-ASR`, local only, working-tree changes uncommitted on
top of the filtered history** (106 commits, .git 1.2 MB, no remote; history
scanned: no secrets, no audio, no large blobs):

- moved in with history: `asr_pipeline/`, `webapp/`, 11 scripts, 29 test files,
  two notebooks, `pytest.ini`; `scripts/polish_scoring_normalizer.py` copied
  without history (missed in the path list; `rescore_stratified` imports it).
- seams: workers -> `asr_pipeline/workers/`, harness -> `asr_pipeline/eval/harness.py`;
  backend `repo` -> `polsess` (bundle directory through `polsess_models`, sample
  rate checked against the bundle in preflight and at load); local paths resolve
  under `$DGSS_ASR_ASSETS` (default `~/dgss_asr_assets`), not the CWD;
  `spmamba_external` kept, fails loud without a Mamba-enabled polsess-models.
- bundles exported to `~/dgss_asr_assets/separators/`:
  `mossformer2_matched_128k_e46`, `..._e31`, `mossformer2_full_128k`; the seven
  YAML configs that named those checkpoints now name the bundles.
- new: `pyproject.toml` (dependency list derived from imports, **not yet
  install-tested**), `README.md`, `CLAUDE.md`, `THIRD_PARTY.md`, `.gitignore`;
  `SCOPE.md` v1.3 marked *pending author approval*.
- Verified on CPU with the parent off `sys.path`: 1009 passed; no module loads
  from this checkout.

**Verification, 2026-09-19 (4070 free):**

- **Parity gate PASSED.** Shipped config, three dev fragments
  (`065a9896`, `0ab10929`, `33a47eae`), this repository (`repo` backend, `.pt`
  checkpoint) vs DGSS-ASR (`polsess` backend, bundle through the built wheel, run
  from a neutral working directory, this checkout off `sys.path`): 33 of 36
  output files byte-identical — both stream WAVs, all transcripts (`.txt` and
  `.json`), `diarization.json`, `routing.json`. The other three are
  `annotation.eaf`, which differ only in the `DATE` stamp; a parent-vs-parent
  control run differs in exactly the same way and is otherwise byte-identical.
- **The gate's first run FAILED and found a sixth seam.** `clearvoice` reads its
  weights from the *relative* directory `checkpoints/<model>` (hard-coded in the
  package's YAML). From another working directory, with `HF_HUB_OFFLINE=1`, the
  download fails, the package prints a warning and returns a **randomly
  initialised FRCRN**; the pipeline "enhanced" with it and reported success
  (streams ~1–3 dB SNR vs the parent, 144/246 words changed) — SCOPE §4.1's
  silent substitution. Fixed in DGSS-ASR: `stages/enhancement.load_clearvoice`
  (assets root as working directory, raises when the checkpoint is absent;
  also used by the `clearvoice` separator backend), a preflight check, the
  ECAPA cache moved off `Path.cwd()`, tests (incl. one against the real
  package). **The hazard is latent in this repository's copy too**: a fresh
  machine running `webapp/run.sh` or an eval with the recommended
  `HF_HUB_OFFLINE=1` gets an untrained enhancer without an error. Not ported —
  author's call (frozen instrument). Thesis numbers are unaffected: the FRCRN
  checkpoint has been on the 4070 since 2026-05-23, before the first eval output
  (05-28), and no surviving log carries the warning.
- **Install test.** Both wheels installed (non-editable) into a clean venv, run
  from a neutral directory: `dgss-asr` console script, packaged configs / split
  lists / workers / webapp assets, preflight clean, separation and enhancement
  stages load the real e46 bundle and FRCRN; 5,576 modules loaded, none from any
  source checkout. Dependency *resolution* of DGSS-ASR's `pyproject.toml` is
  still untested (its `polsess-models` pin names a tag that does not exist yet).
- **Test suites, GPU visible.** DGSS-ASR 1015 passed. This repository's worktree
  1507 passed with `POLSESS_DATA_ROOT` set; without it 15 tests in
  `test_config_yaml.py` / `test_sample_rate.py` fail on this box because `main`'s
  default `data_root` is the 3080's 16 kHz corpus (the live tree carries an
  uncommitted local override) — independent of this work.
- Model assets now under `~/dgss_asr_assets/`: three separator bundles,
  `checkpoints/FRCRN_SE_16K`, `checkpoints/MossFormerGAN_SE_16K` (copies).

**Independent review, 2026-09-19/22.** Two Fable reviewers audited the two sides.
Parent side (no blockers), applied 2026-09-22 per author rulings: `load_separator`
names both versions on every build/load failure and checks required keys; the
export refuses Mamba-family checkpoints, cleans up after any failure, and records
`val_metric`, `git_dirty`, `exported_with`; `rotary-embedding-torch>=0.5.3`;
README release procedure (tag = version, push the tag, never move a tag) and the
GPL-3.0 consequence of shipping `mamba/`; packaging tests now cover relative
imports of excluded modules, Mamba-only libraries, and a real wheel build where
`hatchling` exists. The three bundles' `separator.json` were regenerated (weights
byte-identical). Declined by the author: 1-based `epoch` (documented as 0-based
instead), ConvTasNet EPS handling. Open from that review: licence metadata in
`pyproject.toml` (needs the top-level licence decision). Parent side committed
2026-09-22 as `aa112d8` on `main`, tagged `polsess-models-v0.1.0` (local; not
pushed).

DGSS-ASR side, applied 2026-09-22 ("fix all"; commit names, pseudonyms and
licences deferred by the author): package-data covers every tracked non-.py file
(`tests/test_packaging.py` checks the globs and builds a wheel from a copy);
`load_clearvoice` downloads the weights itself and checks every file the marker
names before construction (`config.clearvoice_missing_files`, shared with
preflight; tests reproduce the interrupted-download case on disk);
`$HF_HUB_OFFLINE` read like huggingface_hub does; `torchmetrics` imported lazily
so `batch` runs without the `eval` extra; `constraints.txt` = the verified
environment, `clearvoice` installed `--no-deps` (its metadata pins conflict) and
no longer a declared dependency; extras completed (`jinja2`, `dev` includes
eval+webapp+httpx); preflight tests stub `polsess_models` importability and cover
corrupt `separator.json` and the `spmamba_external` import; notebook uses
`check_preflight`, no chdir; README install section, `run.sh` interpreter
override (`$DGSS_ASR_PYTHON`); `webapp/render.py` shows the run-directory name
for legacy `.pt` metadata; stale parent-path text fixed; `metadata.json` records
the bundle + source checkpoint sha256 (`separator` key); `sr_corrnet` accepts a
local path under the assets root; porting recipe (`format-patch | git am -3`) in
root `CLAUDE.md`; the suite isolates the debug log (it used to append to
/tmp/asr_pipeline_debug.log). **Verified 2026-09-22 on the 4070:** suite 1035
passed with `polsess_models`, 1031 + 1 skipped without; the 3-fragment parity
set from a NON-EDITABLE install of both wheels in a neutral directory is
byte-identical to the parent outputs (EAF DATE line excepted). Everything is
staged (`git add -A`) except `asr_pipeline/SCOPE.md`, left for its own commit.

**Closed 2026-09-22:** parent `main` + tag `polsess-models-v0.1.0` pushed;
DGSS-ASR committed (`25d7760` tree, `53aab7c` SCOPE.md v1.3); worktree
`~/polsess_separation-models` and branch `feature/polsess-models` removed.
Author rulings: the ClearVoice check is NOT ported to the frozen copy; the
parent suite keeps appending to `/tmp/asr_pipeline_debug.log`.

**Left:**

1. Create the GitHub repository for DGSS-ASR and push. Deferred until
   publication matters: top-level licence (none chosen in either repository;
   the shipped config depends on non-commercial weights, Sortformer v1 and
   ECAPA2), commit author addresses, the "asdf" commit message, the local
   `backup/main-pre-split` branch (excluded from a plain `git push`), the
   pseudonymous CLARIN IDs in `eval/clarin_split.csv`.
2. DGSS-ASR's own venv from `constraints.txt` (README, Installing), then one
   pipeline run from it: the first test of the repository standing alone.
~~3. Decide whether to port the ClearVoice fail-loud check to this repository's
   frozen `asr_pipeline/` (a ~10-line version: after constructing `ClearVoice`,
   raise unless `checkpoints/<model>/last_best_checkpoint` exists).
4. DGSS-ASR's own venv; then the first dependency-resolution test of its
   `pyproject.toml`.
5. `scripts/sweep_pipeline.py` in DGSS-ASR still carries the campaign's
   checkpoint-swap arms (`checkpoints/...pt`); they fail preflight there by design
   and stay runnable here. Export a bundle for any arm that should live on.
