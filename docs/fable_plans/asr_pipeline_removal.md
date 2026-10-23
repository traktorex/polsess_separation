# Removing `asr_pipeline/` (and its consumers) from polsess_separation

Drafted and executed 2026-10-23. Rulings (author): R1 yes, R2 yes, R3 push,
R4 back up and also keep the local copies, R5 backup location as done. Suite
after removal: 29 files, 468 tests collected, exit 0 on CPU (`venv/`).

Frame (author, 2026-10-23): the pipeline moved to `~/DGSS-ASR` on 2026-09-18
(`asr_life2_extraction.md`). That plan deferred removal to "after the paper"
because this copy was the paper's frozen instrument. The author now rules that
the paper's experiments will be redone from scratch in DGSS-ASR, so the frozen
copy has no remaining job here. polsess_separation goes back to being a
separation-training repository that also publishes `polsess-models`.

## 0. Safety net (step 0, running)

- **Full-tree backup** (repo + both venvs + ignored checkpoints/logs/wandb, 34 GB)
  → `/mnt/wsl/hdd/backups/polsess_separation_2026-10-23_pre-asr-removal/`.
  `/mnt/wsl/hdd` is the 1.5 TB ext4 volume inside `I:\wsl\hdd-data.vhdx`, i.e.
  it *is* the I: drive's WSL space; ext4→ext4 keeps the venv symlinks and
  permissions that a copy through `/mnt/i` (9p/NTFS) would lose. The `thesis`
  symlink is copied as a symlink (the vault is F:'s own git repo). Launched with
  `rsync -aH`, log at `<dir>.rsync.log`, markers `<dir>.DONE` / `.FAILED`.
  **Done and verified 2026-10-23 13:36**: 33 GB, 132,351 entries, 3 min 15 s;
  the checksum dry run (`rsync -naic`) listed only this plan file, written
  after launch and copied in afterwards.
  Caveat: `/mnt/wsl/hdd` has no fstab entry; after a WSL restart it is gone
  until re-mounted from Windows (`wsl --mount --vhd I:\wsl\hdd-data.vhdx`).
  The data is safe in the vhdx either way.
- **Git**: push `main` (4 commits ahead of origin). Two tags, pushed:
  `handin-2026-09-05` → `c089782` (last commit before the hand-in; same name as
  the vault's tag), and `pre-asr-removal-2026-10-23` → the last commit before
  the removal commit. Everything removed below stays one
  `git show pre-asr-removal-2026-10-23:<path>` away. Note the cost of
  "delete": DGSS-ASR kept only `b1_tiger_echoset.yaml` + `life2_default.yaml`
  of the configs — `default.yaml`, `sweep_best*`, `v41_*` and the other twelve
  B1 arms exist afterwards only under these tags.
- **Loose end to settle first**: the working tree has an uncommitted hunk in
  `CLAUDE.md` that deletes the "SI-SDRi is zero-mean since 2026-09-22" bullet
  (author's edit?). Commit or revert it before the removal branch, otherwise it
  tangles into the removal diff.

## 1. What leaves (git rm, one commit on `chore/remove-asr-pipeline`)

| Group | Paths | Tracked files |
|---|---|---|
| Package | `asr_pipeline/` (stages, eval, configs incl. 14 B1 arms, vendor/{ap_bwe,mossformer2_dp,tf_locoformer,tiger}, SCOPE.md, CLAUDE.md, SWEEP_KNOBS.md, `eval/clarin_split.csv`) | 90 |
| Webapp (ruling R1) | `webapp/` — cannot run without the package; DGSS-ASR has its own copy | 29 |
| Notebooks | `asr/` (5 tracked; ignored `archive/`, `Korpus.csv` removed with `rm -rf`) | 5 |
| Tests | `tests/test_pipeline_*.py` (26), `test_webapp_backend.py`, `test_rescore_stratified.py`, `test_fragment_acoustics.py` (imports `score_fragment_acoustics` → `asr_pipeline.eval.layer2`), `test_nzr_aids.py` (imports `build_nzr_aids`; found by the first pytest collection) | 30 |
| Scripts (ruling R2) | 28 ASR-only scripts: `run_pipeline_on_recording`, `sweep_pipeline`, `rescore_stratified`, `polish_scoring_normalizer`, `prepare_eval_references`, `dump_sweep_results`, `analyze_sweep`, `score_attribution_purity`, `transcribe_mixtures`, `compare_asr`, `eval_harness`, `sortformer_worker`, `sortformer_batch_worker`, `coherex_worker`, `build_nzr_aids`, `build_review_page`, `mixref_enhanced`, `mixref_squim`, `sweep_asr_decode`, `sweep_asr_prompt`, `score_fragment_acoustics`, `squim_validation`, `diarize_clarin_2speakers`, `diarize_clarin_2speakers_sortformer`, `transcribe_clarin_2speakers`, `enhance_clarin_debleed`, `build_clarin_fragment_set`, `clarin_fragment_finder` | 28 |
| Figure scripts (ruling R2) | `scripts/thesis_figures/fig_ch6_corpus_map.py` (imports `assign_strata`, reads `clarin_split.csv`). Executed with a tighter criterion than first drafted: every other ch6 figure script only *mentions* pipeline paths in its docstring or reads CSVs from `~/datasets/eval`, imports nothing removed, and stays (`fig_ch6_pipeline.py` and `fig_ch6_routing_timeline.py` were briefly removed and restored). | 1 |
| Docs | `docs/SWEEP_RESULTS_SCHEMA.md` (column dictionary for CSVs under `~/datasets/eval`) and `docs/fable_plans/dgss_asr_test_campaign.md` → **move** to `~/DGSS-ASR/docs/` (they document that repo's data and birth). `asr_life2_extraction.md` stays here (it also records the `models/` packaging) with a closing status line. | 2 |

Tally: 185 of 495 tracked files. `git rm` leaves ignored files behind
(`asr/archive`, `asr/Korpus.csv`, `webapp/examples_manifest.json`, `__pycache__`),
so each removed directory gets an `rm -rf` after the `git rm`.

Checked and **staying**: `models/`, `utils/`, `training/`, `datasets/`,
`config.py`, `train*.py`, `evaluate*.py`, `scripts/{audit_mmipc,
audit_split_leakage,benchmark_*,model_manifest,export_separator,
build_prezentacja}.py`, the ch2/ch4/ch5 figure
scripts, `pyproject.toml` (packages only `models/`, so `polsess-models` and
DGSS-ASR's pin are untouched), `requirements-freeze.txt` (historical exact
freeze; keep as is, it is a record).

## 2. The one code change: `tests/test_benchmark_macs.py`

`test_tf_locoformer_attention_cores_are_counted` imports the vendored
TF-Locoformer from `asr_pipeline/vendor/` as "the real consumer" of the MAC
counter. Retarget it at `models.tf_mossformer.TFMossFormer` (small `D`,
`num_blocks=2`, short `window_t`/`window_f`). Each TF block runs two paths
(frequency, temporal); each path is a `LocalGlobalMHSA` = one windowed SDPA
(`WindowedMHSA`, banded cost) followed by one dense SDPA
(`MultiHeadSelfAttention`), so the expectation becomes **4 SDPA calls per
block** and per-block MACs = banded local + dense global on both paths; the
banded term reuses the formula already tested in
`test_count_macs_charges_a_windowed_branch_its_banded_cost`. Same trimmed-depth
trick as today so the test builds on CPU in a second. This is the only edit
that changes behaviour and gets the T2 `/code-review` (correctness only).

## 3. Contract and doc edits (T0, inline)

- `README.md`: title line and subsystem list, layout tree (drop `asr_pipeline/`,
  `asr/`, `webapp/`), venv paragraph, "Sweeps and tests" (counts after pytest;
  drop the "half covers asr_pipeline" sentence), delete "ASR pipeline" and
  "Front-end" sections, env-var table (drop `HF_TOKEN`, `HF_HUB_OFFLINE`,
  `AP_BWE_CHECKPOINT`, `SORTFORMER_VENV_PY`, `COHEREX_VENV_PY`,
  `BROUHAHA_VENV_PY`, `ASR_PIPELINE_DEBUG_LOG`, `WEBAPP_*`), licence list
  (drop the four `asr_pipeline/vendor` rows). Add one sentence: the ASR pipeline
  and the showcase webapp described in thesis ch6/ch7 lived here until tag
  `pre-asr-removal-2026-10-23`; live code is github.com/traktorex/DGSS-ASR.
- `CLAUDE.md`: overview paragraph and the subsystem-guidance line; the
  "Review gates" paragraph (asr_pipeline frozen → gone, replace with the tag
  pointer); Key Commands blocks "ASR pipeline" and "Showcase webapp";
  "Thesis audit artifacts" loses `squim_validation.py`; the 16 kHz bullet loses
  the `separator_sample_rate` / `naive` sentence (it is DGSS-ASR's foot-gun now,
  already documented there); the EchoSet parenthetical redirects to DGSS-ASR's
  `asr_pipeline/configs/b1_tiger_echoset.yaml`; Virtual Environments: `venv/`
  no longer backs anything ASR here (it still backs the deck build and the
  notebooks; the 09-22 ruling to keep both venvs stands, not reopened); Environment
  Variables: drop `HF_TOKEN`, `AP_BWE_CHECKPOINT`, `SORTFORMER_VENV_PY`,
  `COHEREX_VENV_PY`.
- `requirements.txt`: drop `num2words`, `rapidfuzz`, `pyannote.audio`,
  `onnxruntime`, `pyloudnorm`, `huggingface_hub` (no staying code imports it;
  speechbrain pulls it transitively), `fastapi`, `uvicorn`, `python-multipart`,
  and the "ASR pipeline extras" + `sr-corrnet` comment blocks. `soundfile`
  stays (tests, scripts). Header comment loses "webapp".
- `THIRD_PARTY.md`: drop rows 17–20 (vendor dirs) and 26–32 (ASR models);
  reword the `models/tf_mossformer/` note ("second copy on purpose — must not
  import from the liftable asr_pipeline/" → "copy of the upstream file; the
  byte-identical vendored copy now lives in DGSS-ASR").
- `models/tf_mossformer/tf_locoformer_blocks.py` header lines 7–10: same reword
  (comment only).
- `experiments/tf_mossformer/s_16k.yaml` header: the "ASR handoff" open
  question references `asr_pipeline/configs/default.yaml`; rewrite as a pointer
  to DGSS-ASR (the question itself belongs there now).
- `.gitignore`: drop the "asr_pipeline outputs" block, `/asr/archive`,
  `!asr_pipeline/eval/clarin_split.csv`, `webapp/examples_manifest.json`; keep
  the `webapp_ondevice/` lines (branch note still true).
- `.claude/agents/adversarial-fact-checker.md` (local, ignored dir): the
  source list's `asr_pipeline/` entry becomes "ch6 claims: DGSS-ASR or tag
  `pre-asr-removal-2026-10-23`". `.claude/reviews/code/*` and `FOLLOWUPS.md`
  are archived review reports — leave.
- `docs/fable_plans/asr_life2_extraction.md`: append "Removed from this
  repository 2026-10-xx (tag …), see asr_pipeline_removal.md".

## 4. Cross-repo edits in `~/DGSS-ASR` (separate commit there)

- `README.md` "Status" (line ~218) and `CLAUDE.md` "Boundary" bullet: the
  parent copy is gone; the thesis instrument is reachable at the parent's tag.
- `CLAUDE.md` "Porting fixes between the two copies": obsolete; replace with
  one line (history before 2026-09-18 is shared; nothing to port any more).
- Receive the two moved docs (`docs/`).

## 5. Verification and order

1. Backup DONE → `rsync -aic` dry run empty → record size and path in this file.
2. Settle the `CLAUDE.md` hunk; push `main`; create and push the two tags
   (and `experiment/webapp-ondevice` if R3 says so).
3. Branch `chore/remove-asr-pipeline`: §1 removals + `rm -rf` leftovers, §2, §3.
4. `nvidia-smi` / `tmux ls` first (no training in flight: checked 2026-10-23,
   no `train.py`, no tmux server, no compute app on the GPU). Then `pytest`
   (CPU-safe suite, ~31 files). Expected: green; Mamba/GPU/dataset tests skip
   as before. Fill the README test counts from the run.
5. T4 Composer stray-reference sweep over the post-removal tree for
   `asr_pipeline|webapp|clarin|cpWER|SORTFORMER|COHEREX|AP_BWE|eval_harness`
   (data, not verdicts; the orchestrator re-reads every hit). T2 Cursor Sonnet
   `/code-review` of `main..HEAD` (broken imports, the retargeted test, doc
   drift). Fix, merge to `main`, push.
6. §4 in DGSS-ASR, commit, push.
7. One line in the thesis log (`thesis/thesis-log/`) recording the removal date,
   the two tags and the backup path; memory notes updated; `.claude/README.md`
   unchanged (no agent or skill added or removed).
8. Optional local housekeeping after 1 (ruling R4).

## 6. Rulings needed (author)

**R1 — `webapp/` goes with the pipeline.** It imports `asr_pipeline` in three
files and DGSS-ASR carries its own, newer copy. Options: (a) remove, (b) keep
as dead code. Recommendation (a). Default if silent: (a).

**R2 — the 28 campaign-residue scripts and the 3 eval-tree ch6 figure scripts
are deleted, not moved.** Every one imports `asr_pipeline` or `eval_harness`,
or exists only for the CLARIN campaign; `squim_validation.py` (listed in
CLAUDE.md as a thesis audit artifact) is among them, and its result already
lives in the vault (`thesis-log/sweep_plan/b7_squim/`). DGSS-ASR already took
the four it needed (`rescore_stratified`, `prepare_eval_references`,
`polish_scoring_normalizer`, the workers). Options: (a) delete, recoverable via
the tag, (b) move the rest into DGSS-ASR as a `scripts/thesis_residue/` folder
(they would not run there: old config names, CWD-relative paths). Recommendation
(a). Default: (a).

**R3 — push `experiment/webapp-ondevice` to origin.** It exists only in this
local clone (and now in the backup); once `webapp/` leaves `main` it is the only
copy of the on-device demo. Pushing changes nothing on `main`. Recommendation:
push. Default: push.

**R4 — local, untracked ASR assets after the backup verifies.**
`checkpoints/{FRCRN_SE_16K,MossFormerGAN_SE_16K,MossFormer2_SS_16K,external}`
(≈1.2 GB; FRCRN and MossFormerGAN already duplicated in `~/dgss_asr_assets`),
`logs/asr_tfmf16k` (1.9 MB, the 09-23 adoption evidence), `~/webapp_jobs`.
Options: (a) leave on disk, (b) move under `~/dgss_asr_assets/`, (c) delete.
Root disk is at 94 % (64 GB free), so (b) or (c) buys little. Recommendation
(a) now, revisit when DGSS-ASR's B1 rerun needs them. Default: (a).

**R5 — backup location.** Done to `/mnt/wsl/hdd/...` (inside the vhdx on I:),
not to `/mnt/i/...` directly, for the symlink/permission reason in §0. If the
author wants a copy visible from Windows Explorer as well, a `tar` of the
tracked tree + venvs onto `/mnt/i/Backups/` is a follow-up. Default: as done.
