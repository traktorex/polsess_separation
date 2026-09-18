# Sweep run ledger — column dictionary (`sweeps/all_runs.csv`)

Style model: `docs/SWEEP_RESULTS_SCHEMA.md` — the equivalent doc for the
ASR pipeline's fragment-level sweep CSVs. This one covers the *training-side*
run ledger instead: one row per training run (any model architecture), not
per (config, fragment).

**Provenance (important):** the original W&B project (`polsess-separation`
sweeps) was accidentally deleted 2026-02-10. `all_runs.csv` (955 rows,
`;`-delimited) is a **manual, partial recovery** from local `wandb/` run logs
on both training PCs — see `sweeps/EXPERIMENT_LOG_monolithic.md` ("Run data
has been partially recovered..."). It is not the output of a checked-in
script; no generator exists in this repo.

## Columns

| Column | Meaning | Notes |
|---|---|---|
| `run_id` | W&B run id (8-char slug) | never blank |
| `source_pc` | which training machine ran it (`pc1`/`pc2`) | per CLAUDE.md's "two training PCs" |
| `sweep_id` | W&B sweep id the run belongs to | blank for 41/955 rows — standalone (non-sweep) runs, unknown, verify |
| `run_date` | ISO 8601 run-start timestamp | local recovery — unknown, verify against actual W&B server time (may be local mtime) |
| `config` | YAML config path used (relative to repo root) | never blank; passed to `train_sweep.py`/`train.py` |
| `model_type` | architecture key (`convtasnet`/`dprnn`/`sepformer`/`spmamba`) | matches `models/__init__.py` registry; only 4 of the current architectures appear — no MossFormer2/Mamba-TasNet/DPMamba rows (predates those models) |
| `sweep_group` | human-assigned label for a run series (21 distinct values, e.g. `dprnn-expB-lr-sweep1`) | free text set at sweep-launch time, not a fixed enum — cross-reference `EXPERIMENT_LOG_monolithic.md` by name for what a group means |
| `task` | separation task | column is blank or `SB` in the data seen so far — unknown, verify whether blank means "not recorded" or an implicit default |
| `train_max_samples` | dataset-size scaling cap (`data.polsess.train_max_samples`) | blank/1000/16000 in current data; blank presumably means "full dataset", unknown, verify |
| `lr` | learning rate | sweep-search or fixed value depending on `sweep_group` |
| `weight_decay` | optimizer weight decay | — |
| `grad_clip_norm` | gradient clipping norm | — |
| `lr_factor` | LR scheduler decay factor | blank for runs whose config didn't use the plateau scheduler — unknown, verify per-config |
| `lr_patience` | LR scheduler patience (epochs) | same caveat as `lr_factor` |
| `num_epochs` | configured epoch budget | not the same as `last_epoch` (actual epochs run) |
| `seed` | RNG seed | — |
| `early_stopping_patience` | early-stopping patience (epochs) | blank for runs predating this knob or with it disabled — unknown, verify |
| `best_val_sisdr` | best validation SI-SDR achieved (dB) | blank for 8/955 rows (likely runs that crashed before any validation epoch) |
| `last_epoch` | last epoch reached before stop/finish/crash | — |
| `completed` | `True`/`False` — did the run reach its full `num_epochs` budget | not the same as W&B `wandb_state`; a run can be `wandb_state=finished` early-stopped with `completed=False`, unknown, verify the exact predicate used during recovery |
| `checkpoint_dir` | path to the run's checkpoint directory | blank for 8/955 rows (checkpoint not recovered / run never checkpointed) |
| `wandb_name` | W&B display name (e.g. `radiant-sweep-113`) | blank for 406/955 rows — recovered from local logs that didn't carry a display name; **display names collide across runs/projects**, don't treat as a unique key (see W&B storage cleanup note) |
| `wandb_state` | W&B terminal run state | one of `finished`/`crashed`/`killed`/`running` (`running` rows are stale — recovered from logs, the run is long over), or blank if unrecovered |
| `runtime_seconds` | wall-clock training duration | blank where unrecovered; not cross-checked against `run_date`/`last_epoch` |

## Honesty notes

- This ledger predates `utils.collect_run_manifest()` (git SHA / torch version /
  GPU / provenance dict) — none of that exists for these historical rows.
- Several columns above are marked "unknown, verify": the recovery was done
  from raw local logs after the W&B project was gone, so the exact extraction
  logic that produced each column is not preserved anywhere in this repo.
  Treat this file as the best-effort ground truth for what *should* be true,
  not a verified contract — spot-check before citing an exact number from it
  in the thesis.

## How to regenerate

**Not regenerable:** the source W&B project was deleted on 2026-02-10, so this
CSV is the recovered ledger, not a reproducible export. No W&B export script
produced it — see "Provenance" above. Only manual append of new rows (matching
this header) is possible.
