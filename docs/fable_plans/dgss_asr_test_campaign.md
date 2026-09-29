# DGSS-ASR + polsess-models: overnight test campaign (2026-09-22)

Goal: establish that DGSS-ASR stands on its own (own venv from the README, no
polsess_separation checkout), that every documented entry point works from that
environment, and that the polsess_separation side (training, evaluation, export)
did not regress from the `models/` packaging. Orchestrated by Claude; the work
runs in Opus subagents, one per lane. Reports land in the scratchpad
(`campaign/reports/<lane>.md`) and are synthesised into §5 of this file.

## 0. What is already verified (not repeated)

- Both test suites on CPU (parent 1500+, DGSS-ASR 1035 / 1031+1 skipped).
- Parity: shipped config, 3 dev fragments, parent `.pt` vs DGSS-ASR bundle,
  run from a non-editable wheel install of both packages in a neutral directory,
  byte-identical streams and transcripts (EAF `DATE` line excepted). But this
  used the **parent's venv** for every dependency other than the two wheels.
- Bundle export round trip for MossFormer2 (three bundles) and ConvTasNet (tests).

## 1. Ground rules for every lane

- Read-only on `/home/user/polsess_separation`, `/home/user/DGSS-ASR` (git
  trees) and `~/dgss_asr_assets`. No commits, no pushes, no edits. Scratch work
  under the scratchpad `campaign/<lane>/`; pytest with `-p no:cacheprovider`.
- Never install into `/home/user/polsess_separation/venv`.
- GPU: the 4070 (12 GB) is free all night, but **one GPU lane at a time**
  (lanes A and D are sequential; B and C are CPU-only, `CUDA_VISIBLE_DEVICES=""`).
- Never `pkill -f` (self-match); kill by PID.
- Every finding: severity, what was run (exact command), what was observed,
  what was expected, and where the evidence file is. "Not checked" list at the end.

## 2. Phase 0 (orchestrator): the fresh venv

`~/DGSS-ASR/venv` built exactly as README "Installing" says:
`pip install -c constraints.txt -e ".[eval,webapp,dev]"`, then
`pip install --no-deps clearvoice==0.1.2`. This is the first time the install
recipe, the GitHub tag pin and `constraints.txt` are exercised. Log:
`campaign/build_venv.log`. Everything below uses `~/DGSS-ASR/venv/bin/python`
for DGSS-ASR.

## 3. Lanes

### Lane A (GPU, first): DGSS-ASR end to end from its own venv

1. `dgss-asr batch --split clarin_dev --mode full` (23 fragments, shipped
   config) into a scratch eval root; then `--mode no_sep` and `--mode no_enh`
   (`MODE_PRESETS`). `HF_HUB_OFFLINE=1`, `deterministic` as shipped.
2. Compare against the thesis eval tree: streams/transcripts of
   `~/datasets/eval/clarin_fragments/<id>/sweep/v41_merge/` (the shipped config
   is `v41_merge`) — byte comparison, then `dgss-asr score` on both trees and
   the frozen `_rescore_perfrag_dev.csv`. Expected: identical transcripts where
   the environment matches; where they differ, the score on the same fragments
   must agree to the thesis dev numbers (report the delta per fragment).
3. Other diarizer + config surfaces: `configs/default.yaml` (pyannote,
   `$HF_TOKEN` is set), `english.yaml` on a LibriCSS or EdAcc clip
   (`~/datasets/eval/libricss`, `edacc`), `run` with `--set` overrides,
   `--write-outputs`, `spill_intermediate`.
4. External separator arms whose weights are cached: `b1_tiger_echoset.yaml`,
   `b1_mossformer2_librimix.yaml`, `b1_mossformer2_whamr.yaml`,
   `b1_cv_mossformer2_ss16k.yaml` (clearvoice separator), `b1_sb_*` (speechbrain,
   downloads into the assets root), `b1_tf_locoformer_whamr.yaml` (copy the
   `.pth` from `~/polsess_separation/checkpoints/external/tf_locoformer/` into
   `$DGSS_ASR_ASSETS/checkpoints/external/tf_locoformer/` first).
   `b1_spmamba_*` must fail preflight with the polsess-models message;
   `b1_sr_corrnet_*` needs an uninstalled package: preflight must say so.
5. Webapp live: `DGSS_ASR_PYTHON=venv/bin/python ./webapp/run.sh` on a free
   port, submit a fragment through the API (`webapp/API.md`), poll to completion,
   fetch the result and the log tail; `webapp/examples_build.py`; `dev_server`.
6. ClearVoice for real: a temporary empty `DGSS_ASR_ASSETS` root with only the
   separator bundle copied in; online run must download FRCRN into it and
   produce output identical to the cached-weights run; the same with
   `HF_HUB_OFFLINE=1` must fail in preflight, not mid-run.
7. Robustness of `run`: stereo input, 48 kHz input, an mp3, a 3 s clip, a clip
   with one speaker, a 10-minute recording (`~/datasets/clarin_gotowy/gotowy/`).
   Expected: loud errors or sane output, never a silent downgrade; record wall
   time and peak VRAM per case.

### Lane B (CPU): environment, packaging, docs, porting

1. Audit the built venv: `pip check`, versions vs `constraints.txt`, what pip
   resolved for unpinned packages, anything the suite or scripts import that is
   not a declared dependency (`ipykernel`, `ipywidgets`, `matplotlib` for the
   notebooks are suspects).
2. Full test suite in the fresh venv (CPU). Then the suite with
   `polsess_models` uninstalled from a copy of the venv.
3. Fresh resolution without `constraints.txt` in a scratch venv
   (`campaign/B/venv_free`): does `pip install -e ".[eval,webapp,dev]"`
   resolve at all, to what, does the suite pass there, does
   `preflight(shipped config)` pass. This is the "untested environment" the
   review flagged.
4. Every script in `scripts/`: `--help` and imports in the fresh venv; each
   scoring/aggregation script (`rescore_stratified`, `dump_sweep_results`,
   `score_attribution_purity`, `compare_asr`) run on the real eval tree and its
   output compared to the same script run from the parent (`main`, parent venv).
   Expected bit-identical CSVs.
5. Notebooks executed headless (`jupyter nbconvert --execute`, with whatever
   packages that needs installed into a *copy* of the venv): `explore_pipeline`
   up to and including the preflight cell (stage cells need the GPU: skip via a
   cell tag or stop there), `evaluate_pipeline` fully.
6. Docs against reality: every command in README.md, CLAUDE.md,
   `asr_pipeline/CLAUDE.md`, `webapp/CLAUDE.md`, `webapp/API.md` executed or
   dry-run; every path and file name mentioned exists.
7. Porting recipe (root CLAUDE.md): in scratch clones of both repos, make a
   synthetic fix in the parent's `asr_pipeline/stages/routing.py` and one in
   `scripts/sortformer_worker.py`, `format-patch | git am -3` into DGSS-ASR;
   and one fix made in DGSS-ASR ported back. Record what applies cleanly and
   what conflicts.
8. Wheel from the committed tree (`pip wheel` from a `git archive` copy):
   install into a scratch target, import from a neutral directory, run
   `preflight` on every YAML in `configs/`.

### Lane C (CPU): polsess-models package and the export path

1. Export a bundle for every architecture that has a checkpoint on this
   machine (`~/polsess_separation/checkpoints/{convtasnet,dprnn,sepformer,
   mossformer2,tf_mossformer}` — one per architecture, including the 16 kHz
   `tf_mossformer/SB/16k_tf_mossformer_s_42`); Mamba checkpoints must be
   refused. Check `separator.json` contents (sample_rate 16000 for the 16 kHz
   run, `val_metric`, provenance fields) and the printed summary.
2. For each bundle: load through the **installed** `polsess_models` in
   `~/DGSS-ASR/venv` (no parent on `sys.path`, neutral cwd) and through the
   parent's `load_model_for_inference` in the parent venv; run 20 PolSESS test
   mixtures (`$HOME/datasets/PolSESS_C_final_128_v2/test`, and the 16 kHz
   corpus if present on this machine) through both on CPU; SI-SDR per sample
   must be identical to the last bit. Also compare one sample against
   `evaluate.py --device cpu` output for the same checkpoint.
3. Negative paths of `load_separator` on real bundles: missing file, edited
   `format_version`, edited `model_type`, an extra constructor kwarg, a
   truncated `weights.safetensors`, a bundle from "polsess-models 9.0.0".
   Expected: the documented errors, naming both versions.
4. The release procedure in `models/README.md`, dry: `pip install
   "polsess-models @ git+https://github.com/traktorex/polsess_separation@polsess-models-v0.1.0"`
   into a scratch venv (this is what DGSS-ASR's pin resolves to); check that
   the installed files equal the wheel built from the tag; `pip download` of
   the sdist and its contents (no checkpoints, no thesis material).
5. Parent suite in the live tree (`POLSESS_DATA_ROOT=$HOME/datasets/PolSESS_C_final_128_v2`,
   CPU), plus `python scripts/model_manifest.py` and `scripts/audit_mmipc.py`
   as CPU sanity of untouched tooling.

### Lane D (GPU, after A): parent training/eval regression + 16 kHz end to end

1. Smoke training in the parent venv from the live tree: `mossformer2`
   matched config and `tf_mossformer/s_8k.yaml`, `--no-wandb`,
   `train_max_samples=256`, 1 epoch, val on the 1000-item val; and one Mamba
   config if `MAMBA_AVAILABLE` there. Expected: runs to completion, checkpoint
   written with a config the exporter accepts, `evaluate.py --no-pesq
   --no-stoi` on that checkpoint runs. This is the regression check that the
   `models/__init__` / `utils.model_utils` changes did not touch training.
2. `evaluate.py` on the e46 checkpoint, 200 test samples, vs the same 200
   through the bundle in DGSS-ASR's venv on GPU (fp32, `deterministic`):
   identical SI-SDR.
3. 16 kHz end to end: the 16 kHz TF-MossFormer S bundle from lane C, DGSS-ASR
   config = shipped config with `separation.checkpoint_path` → that bundle,
   `separator_sample_rate: 16000`, `post_separation_processing.backend: naive`;
   run the 3 parity fragments in DGSS-ASR's venv AND the same config in the
   parent (`repo` backend, the `.pt`, parent venv). Expected byte-identical
   streams and transcripts between the two. Report separation quality
   qualitatively (it is a 16k-subset model, not a deployable one).
4. Bundles of the other architectures from lane C through the pipeline on one
   fragment each (SepFormer, DPRNN, ConvTasNet at 8 kHz): loads, runs, output
   sane; and the SepFormer length-trim caveat checked at the seam (`match_length`).

## 4. Order and hand-offs

Phase 0 → A ∥ B ∥ C → D (needs A's GPU and C's bundles) → synthesis. A
finding that blocks a later step is reported immediately in the report file;
the orchestrator decides on a fix and a rerun.

## 5. Results (2026-09-22, four Opus lanes, ~12 h wall clock incl. an API outage)

Reports: scratchpad `campaign/reports/{A,B,C,D}.md`. Fixes applied the same day
are marked ✔.

### Verdict

DGSS-ASR stands on its own. From the README venv, in a neutral directory,
offline: `batch`/`run`/`score`, both diarizers, seven external separator arms,
the webapp's full HTTP contract, the real ClearVoice download path, and the
robustness cases (stereo, 48 kHz, mp3, 3 s, single speaker, 10 min) all work
with nothing written to the working directory. No blocker on either side.

- **Seam**: parent code vs DGSS-ASR code in the same venv is byte-identical
  (streams, transcripts, diarization, routing) at 8 kHz (re-confirmed against
  the committed tree) and at **16 kHz** with the e23 TF-MossFormer S bundle.
- **Package**: installed `polsess_models` reproduces the parent's inference bit
  for bit for all five architectures (CPU, 20 samples each) and on GPU for the
  e46 bundle (200/200); `pip install` from the tag installs files byte-equal to
  the tag; the sdist is `models/` only; all nine corruption cases fail loud.
- **Parent regression**: smoke training (MossFormer2, TF-MossFormer S, SPMamba),
  export of the fresh checkpoints, `evaluate.py`: unaffected. Full suites green.
- **Scoring**: every scoring script is byte-identical between the two repos;
  rescoring the frozen `v41_merge` outputs reproduces the thesis dev sheet on
  23/23 fragments to 2 dp.

### The one surprise: the parent venv's cuDNN

Re-running the shipped config from the fresh venv reproduces diarization,
routing and the mixture ASR byte for byte but not the audio path: streams differ
from the parent venv's at up to −47 dB relative RMS end to end (median one
16-bit LSB), 1.2 % of stream-transcript words move, macro cpWER −0.19 pt with no
direction. Cause, verified: the parent venv on the 4070 loads **cuDNN 9.19**
(`nvidia-cudnn-cu13`, left behind by the torch 2.14 attempt, overwrote torch
2.8's `nvidia-cudnn-cu12 9.10.2` in `nvidia/cudnn/lib/`); the fresh venv loads
9.10.2. Different convolution kernels in FRCRN, amplified by VAD gating and beam
search. Not a DGSS-ASR defect; a byte-identity claim must name the venv.
**Author's ruling (09-22): leave it.** `venv/` is frozen — it exists only for
the ASR pipeline and the paper and goes away once the paper is done; the
4070's main venv is `venv_t214/`, whose cuDNN was checked the same day and
matches torch 2.14's own pin (`nvidia-cudnn-cu13==9.24.0.43`, runtime 9.24).
"Right" cuDNN = whatever the installed torch wheel pins, not the newest.

### Fixed the same day ✔ (DGSS-ASR, uncommitted)

- A1 `batch --split` read its inputs from `--out-root`; a missing input counted
  as "skipped" and the batch exited 0 → inputs always from the eval root, a
  missing input is a failure (`failures.csv`, exit 1). Tests.
- A3 `--mode no_sep/no_enh` aborted on the shipped config (relabel guards) →
  presets relax relabel the way the campaign arms did. Test on the shipped config.
- B2 a free `pip install` (no constraints) succeeded and could not import
  (pyannote 3.3.2 + torchaudio 2.11) → `pyannote.audio>=4,<5`.
- Phase 0: `clearvoice --no-deps` lacked `yamlargparse`, `pydub`, `torchinfo` →
  declared dependencies; README, constraints. Venv rebuilt from scratch with
  the final recipe (`[eval,webapp,notebooks,dev]`).
- S1 `rescore_stratified.py` bootstrap CIs/p-values varied run to run (set
  iteration order) → `sorted()`; verified identical under two hash seeds.
  Identical code in the parent (not ported; frozen copy).
- B1 webapp: `status: done` visible before `debug.log` was copied → copy first.
- A4/E1 `examples_build` cross-check fired on ORC-CER for 133/141 fragments:
  the sheets hold the pre-2026-08-10 char-re-optimised definition (commit
  b6d3c03 changed it) → `orccer` now comes from today's code, API.md corrected.
  **Changes the displayed ORC-CER on the examples page by ~0.3 pt** (author may
  prefer regenerating the sheets instead).
- D6 nothing refused a 16 kHz separator with `ap_bwe` → config validator does;
  `configs/tf_mossformer_s_16khz_e23.yaml` added (shipped config + the two
  knobs; 3-fragment sanity: 9.15/11.36/27.69 vs 13.88/11.36/39.09 cpWER — n=3).
- D1 porting recipe dropped root-`CLAUDE.md` hunks silently → path list + note.
- Docs: notebooks extra (N1), video-only list (N2), three stale paths (N3),
  `layer1.py` in SCOPE, provenance wording for external backends (A6),
  16 kHz rule in README.
- Parent (uncommitted, `polsess-models` 0.1.1): `load_separator` names the file
  on corrupt `separator.json` / truncated `weights.safetensors` (C NIT-1/2).

### Open, author's rulings

- ~~Parent `utils/metrics.py`: SI-SDRi mixes conventions~~ **FIXED 09-22**
  (author: "fix it"): `evaluate.py` and the trainer construct
  `ScaleInvariantSignalDistortionRatio(zero_mean=True)` so the mixture
  baseline uses the same Le Roux 2019 convention as asteroid's PIT loss; the
  helper refuses a non-zero-mean metric; the per-variant validation block in
  the trainer (a third copy of the formula) now calls the helper. Effect: every
  SI-SDRi reported from now on is ~0.07 dB below what the same checkpoint
  scored before (200 SER samples; ~0.17 dB on 20); thesis tables and saved
  `val_sisdr` in existing checkpoints carry the old convention. Comparisons
  within one convention are unaffected.
- Parent `repo` backend accepts a sample-rate mismatch silently (a 16 kHz
  checkpoint fed 8 kHz audio: 137/270 words change, no warning). DGSS-ASR
  refuses it; the frozen copy is the author's (D5). **Ruling 09-22: leave for now.**
- `match_length` pads any short separator output unbounded (D2, no observed
  trigger; ≤7 samples today from the 16/8 stride arithmetic, handled).
- `run` has no `--id` (A5); `compare_asr.py --score-only` on an empty dir raises
  a bare KeyError (S2); `evaluate_pipeline.ipynb` has no scope knob (N4).
- `english.yaml` on LibriCSS reproduces a pre-existing quality problem
  (per-speaker transcripts nearly empty while the mixture transcript is full).
- Peak VRAM 11.98/12.28 GB on a 10-minute recording; a 20-minute one is untested.
- DGSS-ASR pin stays at `polsess-models-v0.1.0` (0.1.1 changes error messages
  only; the maintenance rule says move the pin only when a bundle needs it).
