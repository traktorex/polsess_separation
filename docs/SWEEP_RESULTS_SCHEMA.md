# Sweep results — column dictionary & regeneration

Source-of-truth CSVs for the pipeline-config sweep, produced by
`scripts/dump_sweep_results.py`. All files live under
`~/datasets/eval/clarin_fragments/`. That eval tree is the author's local
working data — the fragment-level CSVs documented here are **not** part of this
repository, and this file is the column dictionary for them, not a pointer to
checked-in data.

The config set scored = `GROUPS["definitive"] ∪ GROUPS["phase2"] ∪ {"baseline"}`
from `scripts/sweep_pipeline.py` (87 configs on the dev split as of writing),
scored over the fragments in `asr_pipeline/eval/clarin_<split>.txt`
(dev = 23 fragments → 20 recordings; `88741282` has 3 segments, `9a651086` has 2).

Scoring **reuses the eval module** (`asr_pipeline.eval.metrics` / `layer3` /
`recordings`) and mirrors `scripts/rescore_stratified.py:per_fragment` exactly, so
the per-config micro-averages reproduce the rescorer's ALL-stratum numbers.

---

## Files

| File | Shape | Purpose |
|---|---|---|
| `sweep_results_tidy.csv` | one row per (config, fragment) | the tidy source of truth — every metric + raw counts |
| `sweep_cpwer_wide.csv` | rows = config, cols = fragment | cpWER pivot (+ `MICRO_ALL`) |
| `sweep_cpcer_wide.csv` | rows = config, cols = fragment | cpCER pivot (+ `MICRO_ALL`) |

---

## Micro-averaging note (READ THIS BEFORE AGGREGATING)

Every error rate here is **micro-averaged**: an aggregate over fragments is
`100 · Σ errors / Σ reference_units`, **not** the mean of the per-fragment rates.
Always aggregate from the count columns:

```python
import pandas as pd
df = pd.read_csv("sweep_results_tidy.csv")
g = df.groupby("config")
cpwer = 100 * g["cpwer_errors"].sum() / g["cpwer_ref_words"].sum()   # correct
# df.groupby("config")["cpwer"].mean()   # WRONG — mean of rates, do not use
```

A fragment with **no reference units** (empty GT after normalization) records
`0 errors / 0 ref` and a **blank** rate cell — never a fabricated `0.0`. Because
the rate is blank but the counts are `0/0`, it contributes nothing to a count-based
micro-average (correct) and is excluded from a mean-of-rates (also why mean-of-rates
is wrong/fragile here). A recording's aggregate is just the sum over its fragments,
so micro-averaging the tidy CSV over all of a config's fragments equals the
rescorer's recording-level ALL-stratum micro-average.

The two wide pivots already carry the correct aggregate in their trailing
`MICRO_ALL` column (computed from the count columns, not the cells).

---

## `sweep_results_tidy.csv` columns

### Identifiers / metadata

| Column | Meaning | Units |
|---|---|---|
| `config` | sweep config name (key in `CONFIGS`) | — |
| `fragment` | fragment id, e.g. `88741282__seg01` | — |
| `recording` | recording id (recid) = `fragment.split("__")[0]` | — |
| `stratum` | acoustic-complexity tertile (`LOW`/`MID`/`HIGH`), assigned at the **recording** level by `rescore_stratified._strata` (mean of the recording's fragments' composite scores) | — |
| `composite` | per-fragment acoustic-complexity score (from `composite_scores.csv`) | z-ish score |
| `brouhaha_snr` | Brouhaha SNR estimate (from `acoustic_scores.csv`) | dB |
| `dnsmos_ovr` | DNSMOS overall MOS (from `acoustic_scores.csv`) | 1–5 MOS |
| `squim_pesq` | TorchAudio-SQUIM PESQ estimate (from `acoustic_scores.csv`) | PESQ |

### Per-config knob values (override-then-default)

Extracted from the config's `CONFIGS` override dict; an un-overridden knob takes
the committed `default.yaml` / dataclass default. So you can `groupby('oa_ratio')`,
`groupby('asr_model')`, etc.

| Column | Knob (dotted path) | Default |
|---|---|---|
| `oa_ratio` | `enhancement.observation_mix_ratio` (OA dry/wet mix) | `0.0` |
| `asr_model` | `transcription.model_name` | `large-v2` |
| `bwe_backend` | `post_separation_processing.backend` | `ap_bwe` |
| `enh_backend` | `enhancement.backend` | `frcrn_se_16k` |
| `enh_enabled` | `enhancement.enabled` | `True` |
| `diar_embedding` | `diarization.embedding` (`stock_3.1` = the `None` default = stock pyannote 3.1) | `stock_3.1` |
| `relabel_enabled` | `relabel.enabled` (2nd-pass identity re-clustering) | `False` |
| `relabel_source` | `relabel.source` (`solos`/`global`); **blank when relabel is off** (inert) | `solos` |
| `vad_threshold` | `separation.vad_threshold` | `0.25` |

### Metrics + raw counts

Every rate has a `*_errors` numerator and a `*_ref_words` / `*_ref_chars`
denominator so `Σerrors / Σref` reproduces any aggregate exactly. Rates are
percentages (0–100); a blank rate = no reference units on that fragment.

| Column | Meaning | Counts |
|---|---|---|
| `cpwer` | cpWER — speaker-attributed WER (charges attribution errors); the headline | `cpwer_errors`, `cpwer_ref_words` |
| `cpcer` | cpCER — character ER under the cpWER speaker assignment; co-headline (Polish morphology inflates WER) | `cpcer_errors`, `cpcer_ref_chars` |
| `tcpwer` | tcpWER — time-constrained cpWER (collar 5 s) | `tcpwer_errors`, `tcpwer_ref_words` |
| `orcwer` | ORC-WER on the multi-stream hyp — attribution-blind WER content floor | `orcwer_errors`, `orcwer_ref_words` |
| `mimower` | MIMO-WER on the multi-stream hyp — granularity-robust WER content floor | `mimower_errors`, `mimower_ref_words` |
| `orccer` | ORC-CER on the multi-stream hyp — CER content floor | `orccer_errors`, `orccer_ref_chars` |
| `wer_attr_gap_orc` | `cpwer − orcwer` (attribution penalty, ORC floor) | derived from counts above |
| `wer_attr_gap_mimo` | `cpwer − mimower` (attribution penalty, MIMO floor) | derived |
| `cer_attr_gap_orc` | `cpcer − orccer` (CER attribution penalty) | derived |
| `purity_pct` | stream attribution purity % (reference-free; higher = cleaner), if `_attribution_purity.csv` present | `purity_pure`, `purity_total` |
| `mix_orcwer` | mixture-baseline ORC-WER (single-stream Whisper on the raw mix vs GT) | `mix_orcwer_errors`, `mix_orcwer_ref_words` |
| `mix_mimower` | mixture-baseline MIMO-WER | `mix_mimower_errors`, `mix_mimower_ref_words` |
| `mix_mimocer` | mixture-baseline MIMO-CER | `mix_mimocer_errors`, `mix_mimocer_ref_chars` |

Notes:
- The attribution-gap columns are derived from the micro counts of their two
  inputs **on that single fragment**; for an aggregated gap, re-derive it from the
  summed counts (`100·Σcpwer_err/Σcpwer_ref − 100·Σorcwer_err/Σorcwer_ref`), not by
  averaging the per-fragment gap.
- Mixture-floor columns are blank if the run did not write `transcript_mixture.txt`
  (the sweep writes it under the eval preset, so they are populated on this set).
- `purity_*` columns are blank for (config, fragment) pairs absent from
  `_attribution_purity.csv`.

---

## `sweep_cpwer_wide.csv` / `sweep_cpcer_wide.csv`

- Row = `config`; one column per fragment id; cell = that fragment's `cpwer`
  (resp. `cpcer`) percentage (blank when no reference units).
- Trailing **`MICRO_ALL`** column = the exact micro-average over all fragments
  (`100·Σerrors/Σref` from the count columns) — the correct config-level aggregate.

---

## How to pivot / group

```python
import pandas as pd
df = pd.read_csv("sweep_results_tidy.csv")

# config-level micro cpWER (matches rescore_stratified ALL stratum)
g = df.groupby("config")
micro_cpwer = 100 * g["cpwer_errors"].sum() / g["cpwer_ref_words"].sum()

# marginal effect of the OA knob, micro-averaged
oa = df.groupby("oa_ratio")
micro_by_oa = 100 * oa["cpwer_errors"].sum() / oa["cpwer_ref_words"].sum()

# per-stratum micro cpWER for one config
sub = df[df.config == "dr_refineplus"].groupby("stratum")
100 * sub["cpwer_errors"].sum() / sub["cpwer_ref_words"].sum()
```

---

## How to regenerate

```bash
# dev split (default); --validate prints the micro-avg spot-check vs the rescorer
python scripts/dump_sweep_results.py --split dev --validate

# held-out test later (same generator, just the split flag)
python scripts/dump_sweep_results.py --split test
```

The generator is deterministic (no sampling), parses `CONFIGS`/`GROUPS` statically
from `scripts/sweep_pipeline.py` (so adding a config there flows through on re-run),
and reuses the eval-module scoring. It writes a blank-metrics row (never a silent
drop) and a `!!`-flagged list for any (config, fragment) pair missing pipeline
output. `--validate` checks that the tidy micro-averages reproduce
`scripts/rescore_stratified.py`'s ALL-stratum cpWER/cpCER for `baseline`,
`dr_refineplus`, `dr_oa050`, `dr_oa070` (within ≈0.1).
