# ASR Pipeline — Clean Sweep (MossFormer2 **e31** separator)

One coherent, documentable sweep on the MossFormer2 matched-128k **e31** separator
(`checkpoints/mossformer2/SB/mossformer2_matched_128k_final_42_e31/mossformer2_SB_best_e31.pt`),
with detect-and-retry **on**. This supersedes the exploratory, iteratively-re-anchored
rounds in `SWEEP_RUNLOG.md` (kept for history) — those were on the e23 separator and
shifted as the pipeline changed. Date: 2026-06-15. Eval = **23-fragment dev split**.
All numbers are **dev-only**; the held-out **test** split is the final validator (scored once).

## Setup
- **Separator**: MossFormer2 matched-128k **e31**.
- **Baseline** = `default.yaml`, the full pipeline: diarization → routing →
  enhancement (FRCRN) → separation (e31) → BWE (ap_bwe) → assembly → WhisperX
  large-v2, **detect-and-retry on** (`retry_collapsed_chunk_size=8`). cpWER **27.02**.
- **Metrics** (micro-averaged): cpWER (primary) + CER + MIMO-WER + ORC-WER + tcpWER;
  bootstrap 95% CI over fragments (2000 resamples, seed 1234); paired diff vs
  baseline (`sig` = CI excludes 0); secs/frag. Harness `scripts/sweep_pipeline.py`,
  groups `final` (round 1) and `final2` (round 2). Full CSV:
  `~/datasets/eval/clarin_fragments/_sweep_results.csv`.
- **Design**: Round 1 = OFAT (one knob off baseline) + ablation corners + the two
  ship endpoints (41 configs + baseline). Round 2 = **pre-specified** combos of the
  round-1 winners (no further re-anchoring).

## Round 1 — OFAT (42 configs, 0 errors)

### Top of the ranking (full 42-row table in the CSV)
| # | config | cpWER | CER | MIMO | ORC | db15fc57 | Δ vs base | sig |
|---|---|---|---|---|---|---|---|---|
| 1 | **f_oa03** — OA r=0.3 | **20.65** | **13.58** | 18.09 | 18.58 | 37.7 | **−6.37** | ✅ |
| 2 | f_oa07 — OA 0.7 | 21.19 | 14.11 | 17.68 | 18.29 | 39.3 | −5.83 | |
| 3 | oa_frcrn_05 — OA0.5+naive+nrng3 | 21.62 | 14.57 | 18.27 | 18.88 | 39.3 | −5.40 | |
| 4 | r1_asr_largev3 — large-v3 | 21.66 | 14.63 | 18.41 | 19.29 | 39.8 | −5.36 | |
| 5 | f_oa05 — OA 0.5 | 21.82 | 15.08 | 18.88 | 19.43 | 37.7 | −5.20 | ✅ |
| 6 | ah_nrng3 — nrng=3 | 22.15 | 14.24 | 19.49 | 20.12 | 40.8 | −4.87 | |
| 7 | r1_seam_boundary | 22.94 | 15.72 | 20.40 | 20.83 | 41.4 | −4.08 | |
| 8 | r1_asr_beam10 | 22.96 | 15.61 | 20.18 | 20.83 | 41.4 | −4.06 | |
| … | r3_best — **enh OFF** + naive | 23.63 | 16.87 | 20.16 | 20.65 | 63.9 | −3.39 | |
| | **baseline** — FRCRN + ap_bwe | 27.02 | 17.87 | 24.40 | 25.05 | 141.9 | — | |
| | r1_bwe_naive — naive BWE | 27.57 | 17.97 | 24.50 | 25.09 | 139.8 | +0.55 | |

### Findings
1. **OA-0.3 is the best config on BOTH cpWER (20.65) and CER (13.58)**; its paired CI
   vs baseline excludes 0 (as does OA-0.5's) — **but this does NOT survive multiple-comparison
   correction** (see "Statistical rigor"); treat it as a direction, not a per-config p-value.
2. **Detect-and-retry unlocked Observation-Adding.** OA *tames* db15fc57 (37.7 vs
   baseline 141.9) instead of backfiring. On e23 (no retry) OA backfired there (its
   over-merge collapse); with retry catching the collapse, OA's broad benefit dominates.
   This **vindicates enhancement**: OA wins WER *and* CER (audio/SQUIM pending).
3. **Enhancement-OFF is NOT best on e31** (r3_best 23.63) — **reverses the e23
   conclusion**. With OA + retry, enhancement-on beats enhancement-off here.
4. **e31 flips two earlier calls:** `large-v3` is now strong (−5.36; was worse on e23),
   and `naive` BWE is now **worse** than `ap_bwe` (27.57 vs 27.02 — reverses e23).
5. **Dead knobs reconfirmed** (Δ = 0.00 exactly vs baseline): `temperature`,
   `condition_on_previous_text`, `no_speech_threshold` (lo & hi).
6. **retry's payoff is failure-mode-specific:** on the baseline (nrng off), db15fc57
   is a *repetition* catastrophe (141.9) — which retry can't fix (it fixes empty
   *collapses*, not repetition blowups), so `f_noretry` ≈ baseline. retry pays off
   under OA, where the residual failure is a collapse. (nrng=3 is the lever for the
   repetition mode.)
7. **Caveat — wide CIs (n=23):** variance is dominated by db15fc57 / fe65d170 /
   33a47eae. Only OA-0.3 / OA-0.5 clear 0 in the paired test. The OA *direction* is
   robust + mechanistically grounded; the ordering among the rest is within noise.

## Round 2 — OA-0.3-anchored combos (9 configs, 0 errors)
Pre-specified test of whether stacking the round-1 winners beats plain OA-0.3
(motivated by the OA endpoint `oa_frcrn_05` being *worse* than f_oa03 → combos aren't additive).

| config | cpWER | CER | MIMO | ORC |
|---|---|---|---|---|
| **g_oa03_v3** (OA0.3 + large-v3) | **20.30** | 13.89 | 17.32 | 18.05 |
| g_oa03_seamb (+ seam-boundary) | 20.38 | **13.49** | 17.78 | 18.25 |
| g_oa03_beam10 (+ beam10) | 20.52 | 13.67 | 17.91 | 18.39 |
| **f_oa03** (OA-0.3 alone) | 20.65 | 13.58 | 18.09 | 18.58 |
| g_oa03_v3_seamb | 20.73 | 14.37 | 17.80 | 18.53 |
| f_oa02 (OA 0.2) | 20.97 | 13.85 | 18.35 | 18.76 |
| f_oa04 (OA 0.4) | 21.03 | 14.15 | 18.11 | 18.62 |
| g_oa03_nrng3 (+ nrng3) | 21.30 | 13.84 | 18.74 | 19.24 |
| g_oa03_v3_nrng3 | 22.88 | 14.83 | 20.04 | 20.79 |

- **OA ratio 0.3 is the optimum** (0.2→20.97, 0.4→21.03; round 1 had 0.5→21.82, 0.7→21.19) — confirmed across both rounds.
- **large-v3 / seam-boundary give marginal, within-CI cpWER gains** on OA-0.3 (20.30 / 20.38 vs 20.65); beam10 ≈ neutral. The **top five (20.30–20.73) are a statistical tie.**
- **nrng3 HURTS once retry is on** (g_oa03_nrng3 21.30, v3+nrng3 22.88): OA+retry has no catastrophe for nrng to repair, so it only adds collateral. Do NOT pair nrng3 with the OA finalist.

## Audio quality (SQUIM) on the finalists — the dual-objective axis
Non-intrusive SQUIM_OBJECTIVE on the assembled `stream_{A,B}.wav` (2×23 streams):

| config | STOI | PESQ | SI-SDR |
|---|---|---|---|
| baseline (full enh + ap_bwe) | **0.899** | **2.33** | **11.1** |
| **f_oa03 / g_oa03_v3 (OA-0.3)** | 0.890 | 2.03 | 8.9 |
| r3_best (enh OFF) | 0.853 | 1.77 | 5.6 |

- **OA-0.3 is the Pareto sweet spot**: PESQ 2.03 recovers most of enhancement's audio gain over OFF (1.77) *while being the WER-best config*; full enhancement (baseline) has the best audio (2.33) but the worst WER (27.0, the catastrophes). (v3 = identical audio to f_oa03 — it changes only transcription.)
- **Dual-objective vindication: OA-0.3 beats OFF on transcripts AND audio.**

## Statistical rigor & caveats (read before citing any "winner")
- **Multiplicity**: round 1 made 41 paired comparisons at uncorrected 95% CIs → ~2 false positives expected under the null; exactly 2 cleared zero (f_oa03 by 0.52 in a ~16-wide CI). **No single config survives Holm/FDR correction** — so "OA-0.3 is *significantly* better" is NOT supported as a per-config claim. What IS supported is the OA **direction**: a pre-registered mechanism (Iwamoto 2022 / Wang 2024 dry-wet) that replicates across **ratios** (0.3 optimum) and **backends** (FRCRN + MossFormerGAN) and improves WER *and* audio. Lead with the mechanism, not the p-value.
- **Report with AND without db15fc57**: the "enh-on beats OFF" reversal is amplified by db15fc57 (baseline 141.9, a repetition blowup). Excl-db15 OA-0.3 still leads, by less. Its GT was edited mid-investigation — disclosed.
- **Cross-separator comparisons are confounded**: e23→e31 also changed the GT (16/23 dev EAFs edited during the campaign) and added detect-and-retry — do NOT attribute the naive↔ap_bwe / OFF↔OA "flips" to e31 alone.
- **Dev is upper-biased** (knob set chosen via 100+ dev-scored configs). The **frozen held-out test split** (`asr_pipeline/eval/CLARIN_SPLIT.md`: dev 23 / test 124, unit-disjointness verified) is the only unbiased estimate — score ONCE on the locked finalist.
- cpWER is primary (charges attribution; right for per-speaker downstream ASR); CER + ORC reported alongside and the OA lead holds on both.

## Recommendation (dev-final; test-set is the real number)
**Finalist = OA r=0.3**: `enhancement.backend=frcrn_se_16k` + `observation_mix_ratio=0.3`, `ap_bwe` BWE, detect-and-retry on, large-v2, else baseline (config `f_oa03`). Best/tied cpWER (20.65) + CER (13.58) **and** recovers most of enhancement's audio gain (PESQ 2.03 vs OFF 1.77) — the dual-objective pick. `g_oa03_v3` (+large-v3) is a within-noise cpWER alternative at a heavier model.
- **Honest framing for the thesis**: on n=23 no single config is statistically distinguishable; the robust, citable finding is the **OA mechanism** (enhancement made WER-positive by diluting its artifacts + catching its collapses with detect-and-retry).
- **Config note**: `sweep_best_excl_db15fc57.yaml` currently encodes OA **0.5** + naive + nrng3 (the e23-era pick); to adopt the e31 finalist, set it to OA **0.3** + ap_bwe + nrng off. Lock it, then run the **held-out test split once** and report whatever it gives.
