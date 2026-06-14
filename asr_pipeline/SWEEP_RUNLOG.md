# ASR pipeline comprehensive sweep — run log

Durable state for the multi-day unattended sweep (survives context compaction).
Started 2026-06-13. Author away; Claude babysitting. Eval = **23-fragment dev split**.

> **Complete record:** experiment narrative + results below; a chronological **RUN INDEX** and a full **CHANGELOG** (every code/config/dependency change, all uncommitted) are at the **bottom** of this file.

## Setup (done, verified)
- Separation default = MossFormer2 matched-128k (all 5 configs), pin test green.
- Whisper decode knobs added to `TranscriptionConfig` (beam_size, temperature, condition_on_previous_text, no_speech_threshold, compression_ratio_threshold, patience); defaults = WhisperX `default_asr_options` → **baseline byte-identical** (regression: cpWER 49e09a03 = 13.745704 before & after).
- Sweep harness (`scripts/sweep_pipeline.py`): per-config `secs_per_frag`, bootstrap 95% CI over fragments (`cpwer_ci_lo/hi`, 2000 resamples, seed 1234), paired diff vs `baseline` (`vs_base_delta/ci_lo/ci_hi/sig`). Config-level try/except added (one bad config can't abort the sweep).
- Knob inventory: `asr_pipeline/SWEEP_KNOBS.md`.

## Protocol
- Primary metric: micro-averaged cpWER (lower better). Report tcpWER, CER, ORC/MIMO, + bootstrap CIs + paired-vs-baseline significance + secs_per_frag.
- Dropped knobs: separation-checkpoint A/B (MossFormer2 only, per author), diarization internals (num_speakers=2 fixed), minor assembly/separation knobs (target_rms, crossfade, min_solo_for_anchor, etc.), **align_model_name** (only affects tcpWER timestamps, not the primary cpWER — near-zero value for the ranking).
- ~53 s/fragment → ~20 min/config on 23 dev. Each (config,recording) is a fresh pipeline.
- Launch in rounds (background process per round, contained crash blast radius).
- Re-anchor: pick Round-1 winners → Round 2 interactions → Round 3 finalists + ablation corners.

## Dev recordings (23)
`/tmp/dev_recordings.txt` — also: 065a9896 0ab10929 33a47eae 48fbaab6 49e09a03 4f9251fe 543bf543 595aa511 62e43b45 72ca135e 82627719 88741282(seg00/01/02) 94a0d89a 9a651086(seg00/01) da1b5a78 db15fc57 eaabbc3b eb33c0f0 f59efd6d fe65d170 (all `__seg00` unless noted).

## Launch commands
```
DEV=$(cat /tmp/dev_recordings.txt)
# Round 1 (OFAT, 33 configs + baseline):
venv/bin/python scripts/sweep_pipeline.py --groups r1 --recordings $DEV --force > /tmp/sweep_r1.log 2>&1
# Re-score only (no rerun):
venv/bin/python scripts/sweep_pipeline.py --groups r1 --recordings $DEV --score-only
```

## Status
- [done] Smoke test: `r1_enh_mossformer2_48k` OK (kept). `r1_asr_distil_pl` FAILED (CT2-format repo, WhisperX wants a transformers repo) → **dropped** from group r1. No transformers-format Polish conversational finetune worth the risk found → model axis = large-v2/v3/v3-turbo only. (Confirmed the harness skips a failed arm gracefully.)
- [RUNNING] **Round 1** launched 2026-06-13 ~23:10, `--groups r1 --recordings $DEV` (no --force, resume-friendly), log `/tmp/sweep_r1.log`. 32 OFAT configs + baseline × 23 dev frags, ETA ~10-12 h. Results CSV → `~/datasets/eval/clarin_fragments/_sweep_results.csv`. Re-score partial anytime: `--groups r1 --recordings $DEV --score-only`.
- [pending] Round 2 (interactions among Round-1 winners), Round 3 (finalists + ablation corners).

### Resume / crash recovery
If the Round-1 process dies, relaunch the SAME command WITHOUT --force — completed (config,fragment) outputs are skipped (write_pipeline_outputs only writes on a successful run, so no half-written transcript falsely skips). Then continue.

## Round 1 config groups (OFAT off baseline)
enhancement: r1_enh_mossformer2_48k (+ baseline=frcrn; enh_none corner=r1_noenh) ·
transcription model: r1_asr_largev3/_turbo/_distil_pl ·
decode: r1_asr_beam1/beam10/temp0/cond_prev/nospeech_lo/nospeech_hi/prompt_empty/prompt_rich ·
sep VAD: r1_vad_040/050/060/ar0/ar2 · context: r1_ctx_fixedpad/fixedpad15/none ·
seam: r1_seam_zc/boundary · vol: r1_vol_none · BWE: r1_bwe_naive/flowhigh8/flowhigh16 ·
routing: r1_merge_03/08 · assembly: r1_no_rmsmatch/perpiece_rms ·
ablation corners: r1_nosep/noenh/nosep_noenh.

## Results

### Round 1 (done 2026-06-14, 759 runs, 0 errors) — baseline cpWER 26.00
Data-quality note: baseline/`065a9896` had a stale 21.1 cell from the killed first launch; re-ran fresh → 14.8 (matches 4 no-op configs). Cross-config identical-to-6-decimals values confirm the pipeline is deterministic; that was a lone artifact. CIs below are post-fix.

**Headline — enhancement is a liability.** `r1_noenh` (enh off) = **23.13**, best point estimate (Δ−2.88 vs baseline). The new `mossformer2_se_48k` enhancer = **32.09**, worst (Δ+6.09). On `db15fc57`, FRCRN enhancement caused a catastrophic hallucination (107.8% cpWER) that enh-off cuts to 39.9 (−68 pts) — so the enh-off win concentrates in one fragment (better on 9/20 recs, worse on 5), which is why its paired CI [−9.05,+0.63] crosses 0 (correctly flagged fragile). Mirrors SCOPE §9 (always-on enhancement risks catastrophic failure on clean audio).

**Winning directions (point estimate; most individually within noise on n=23):**
- enhancement OFF (−2.88), large-v3-turbo (−2.84) ≈ large-v3 (−2.68) [turbo faster: 29 vs 37 s/frag]
- BWE: flowhigh16 (−2.72) and naive (−1.50) both beat baseline ap_bwe; flowhigh8 worse (+1.79)
- assembly `overlap_rms_match_solo=False` (−2.29, **sig**, the only significant single-knob win)
- beam_size 10 (−1.91); beam 1/greedy hurts a lot (+3.59 **sig**)
- separation helps: nosep +2.56, nosep_noenh +0.71

**Dead knobs (Δ=0.00 exactly, drop):** temperature, no_speech_threshold, condition_on_previous_text.
**Within-noise (keep baseline value):** vad_threshold/soft (040/050/060 all ≈ baseline), seam_mode, context_window_mode, volume_normalization, routing.merge_gap, vad attack/release (ar0 slightly hurts → keep 1/1).

Full table: `~/datasets/eval/clarin_fragments/_sweep_results.csv` (re-score: `--groups r1 --recordings $DEV --score-only`).

### Round 2 (done 2026-06-14, 207 runs, 0 errors) — combo + leave-one-out
Combo `r2_best_naive` (enh off + large-v3-turbo + naive + rms-off + beam10) = 22.85. LOO marginals from it (revert one knob to baseline):
- model turbo→**v2: 21.47 (−1.38, BETTER)** — v2 beats turbo once enh is off (OFAT had turbo>v2; interaction).
- beam 10→**5: 21.61 (−1.24, BETTER)** — and beam5 is faster (19 vs 21 s/frag).
- rms off→**on: 22.46 (−0.39, better)** — rms-match ON slightly better in-combo (OFAT had off>on; interaction).
- bwe naive→ap_bwe: **25.22 (+2.37, WORSE)** — naive is important; ap_bwe (current default) is bad.
- enh off→on: 23.09 (+0.24, off better but small here).
- `r2_enh_mossformer_gan` 25.22 (better than frcrn baseline 26.0, still worse than off).
**Implication:** the improving reverts stack to {enh off, v2, naive, rms on, beam5} = **baseline + only {enh off, bwe naive}**. Everything else stays at baseline. Predicted ~20-21. CIs still wide (all configs' paired CI vs baseline cross 0 — n=23 variance dominated by db15fc57 et al.); the ~5pt direction is consistent but dev-significance is fragile → TEST set is the real validator.

### Round 3 (done 2026-06-14, 0 errors) — CONVERGED
Top four are a statistical tie (CIs ±~10): r2_loo_v2 21.47, r3_best_beam10 21.55, r3_best_turbo 21.61 (fastest, 18.4 s/frag), **r3_best 21.91**. They differ only in knobs that don't robustly matter (beam 5/10, rms on/off, v2/turbo). At this operating point: large-v3 worse (24.0), flowhigh16 worse (26.3, unreliable across operating points), rms-off hurts (24.9), nosep hurts (26.71). 

**FINALIST = `r3_best` = enhancement OFF + BWE naive (else baseline: large-v2, beam 5, rms-match on, separation on).** cpWER **21.91 vs 26.00 baseline (−4.1, ~16% relative)**. Chose the simplest robust config (2 changes from baseline, each justified); turbo (`r3_best_turbo`, 21.61, 1.5× faster) is the speed-optimized near-equal alternative. Written to `asr_pipeline/configs/sweep_best.yaml` (default.yaml NOT changed; not yet test-validated).

**Honest caveat:** all configs' paired CIs vs baseline cross 0 on n=23 (variance dominated by a few hard fragments, esp. db15fc57). The ~4pt direction is large and consistent across 3 rounds, but dev-significance is fragile → the held-out **test set is the real validator** (scored once, by the author).

### Round 4 (done) — enhancement × separation 2×2 (micro-avg cpWER, BWE naive, large-v2/beam5/rms-on)

|              | separation ON | separation OFF |
|---|---|---|
| **enhancement OFF** | **21.91** (finalist) | 26.71 |
| **enhancement ON**  | 24.51 | 28.57 |

Reads cleanly: separation helps in both rows (~−4 to −5 cpWER); enhancement hurts in both columns (~−2 to −2.6). Best corner = enh-off + sep-on.
- Shipped baseline (enh on + sep on + **ap_bwe**) = 26.00 → finalist beats it by 4.1 (the ap_bwe→naive swap is the gap between the 24.51 cell and 26.00).
- Mixture floor (no pipeline): ORC-WER 22.68, MIMO-WER 21.81. Finalist cpWER 21.91 ≈ mixture MIMO; finalist **orcWER 19.07 beats mixture ORC 22.68 by 3.6** → the pipeline's recognition gain is real; attribution eats some of it (consistent with prior "benefit is recognition, not attribution").

**Re-scoring the full union:** `--groups r1 r2 r3 r4 --recordings $DEV --score-only` (the CSV holds only the last run's configs; this rebuilds all 50 from the on-disk outputs).

## ⚠️ Per-fragment re-analysis (2026-06-14) — refines the enhancement claim
Isolating enhancement cleanly (`r4_enhon` vs `r3_best`, BWE=naive both, only enh differs):
- **All 23:** enh-off 21.91 vs enh-on 24.51 (Δ+2.60). **Excluding db15fc57:** 21.18 vs 21.24 (**Δ+0.06**).
- So the *entire* "enhancement hurts" aggregate is ONE fragment (db15fc57, +66.8 cpWER from a WhisperX hallucination). On the other 22, enhancement is **neutral**.
- Per-fragment: enh-off better on 6, worse on 9, tie on 8 — and the magnitudes are tiny (<3) except db15fc57.
- **No correlation** of the enhancement effect with any measured aspect: Spearman vs composite +0.13 (p.57), brouhaha_snr −0.09, dnsmos_sig −0.01, overlap −0.07 — all ≈0. The Niski-mean +7.19 / Wysoki −3.34 split is purely the db15fc57 artifact (drop it and Niski mean → −1.3).
- **Corrected framing:** enhancement is *on-average neutral but carries catastrophic tail risk* (rare WhisperX-hallucination blowup). Turning it off buys tail-risk insurance at ~zero average cost — NOT a systematic accuracy gain, and NOT a clean-audio effect.
- **CORRECTION (independent analysis, 2026-06-14): the "naive BWE broad gain" is NOT real — it too is single-fragment tail-risk.** naive vs ap_bwe at enh-off: all-23 Δ+1.22, but **excl 94a0d89a flips to −0.60** (ap_bwe marginally better elsewhere). On 94a0d89a, ap_bwe=55.3 vs naive=13.7 (ap_bwe hallucination "nie"×95). So BOTH finalist changes are tail-risk insurance against deterministic neural-stage hallucinations on single fragments (enh→db15fc57; ap_bwe→94a0d89a; flowhigh→94a0d89a/eaabbc3b), each ~neutral on the other 22. naive still chosen = the only BWE with ZERO neural tail risk + free + fastest, but it is NOT a broad accuracy win.
- **The real broad, robust gain is SEPARATION** (−5 to −7 cpWER, *stronger* under outlier removal). Foreground this. Striking: the shipped baseline's recognition (orcWER 23.03) is *worse* than the raw-mixture floor (mixORC 22.68) — the pipeline only beats the mixture *because of* the finalist changes (finalist orcWER 19.07, −3.6 vs mixture), all of it separation.
- Finalist `sweep_best.yaml` recommendation stands (both changes net-positive as tail-risk insurance), BUT independent analysis recommends shipping **`r3_best_turbo`** instead of `r3_best`: tied-best point estimate (21.61 vs 21.91), ~1.7× faster (18 vs 28 s/frag), more LOO-robust. Tradeoff = uses large-v3-turbo vs baseline large-v2. Author's call (thesis-simplicity → r3_best; deployment → turbo).
- Validity: all 23 dev fragments confirmed hand-corrected GT (≠ seed). Hardest strata (high-noise/hard-composite, e.g. 33a47eae/fe65d170 ~50% for every config) are floor-limited; the finalist's win concentrates on easier fragments. Full independent report: `SWEEP_ANALYSIS.md`.
- **db15fc57 catastrophe is DETERMINISTIC, not a seed/noise artifact** (tested 2026-06-14): enh-on + naive on db15fc57 gave cpWER 107.3 / 193 hyp words identically across 4 runs — 1 with deterministic=True + 3 with deterministic=False (cuDNN autotuning on). So the FRCRN→WhisperX hallucination is a stable, reproducible failure for that input (it's the temp-0 beam path, not stochastic fallback; the ~1e-7 enh conv noise is too small to change this decode). Re-running won't dodge it → "enh off" removes a reliable landmine, not re-rollable bad luck. flowhigh BWE has the same hallucination tail risk on *other* fragments (94a0d89a +43.8, eaabbc3b +33.3) — rejected (worse avg, slower, operating-point-unstable).

## Anti-hallucination knobs (2026-06-14) — GATE PASSED ✅
The catastrophes are repetition hallucinations; `no_repeat_ngram_size` (faster-whisper, was OFF) kills them. Gate on catastrophe fragments (on baseline = enh-on + ap_bwe):
| knob | 49e09a03 (regr) | db15fc57 (enh cat.) | 94a0d89a (ap_bwe cat.) |
|---|---|---|---|
| baseline | 13.7 | 107.8 | 56.6 |
| no_repeat_ngram_size=2 | 14.8 | 65.8 | 13.7 |
| **no_repeat_ngram_size=3** | 14.1 | **62.7** | **14.6** |
| repetition_penalty=1.15 | 16.5 | 66.3 | 14.6 |
| hallucination_silence_threshold=2.0 | 13.7 | 107.8 (no effect) | 56.6 |
- **no_repeat_ngram_size=3** FULLY fixes the ap_bwe catastrophe (94a0d89a 56.6→14.6) and HALVES the enhancement catastrophe (db15fc57 107.8→62.7), +0.4 collateral. repetition_penalty works but 7× the collateral; hallucination_silence_threshold is inert. Knob is free (decode-time).
- Implication: the catastrophic tail risk that drove "enhancement/ap_bwe bad" is largely a fixable WhisperX repetition bug. → full 23-frag sweep next (collateral + does it rescue the baseline pipeline + does BWE choice still matter once hallucinations are guarded).

### Full anti-hallucination sweep (23 dev, 2026-06-14) — nrng3 is a TARGETED fix, not a universal default
| config | cpWER | Δ vs baseline | note |
|---|---|---|---|
| baseline (enh on + ap_bwe) | 26.00 | — | 2 catastrophes |
| **ah_nrng3** (baseline + nrng3) | **23.35** | −2.66 | nrng3 rescues the enh+ap_bwe baseline (catastrophes were the main WER cost) |
| ah_nrng2 (baseline + nrng2) | 24.55 | −1.46 | nrng2 < nrng3 at scale |
| r4_enhon (enh on + naive) | 24.51 | −1.50 | |
| r1_noenh (enh off + ap_bwe) | 23.13 | −2.88 | |
| ah_apbwe_nrng3 (enh off + ap_bwe + nrng3) | 22.16 | −3.84 | once guarded, ap_bwe(22.16) > naive+nrng(23.11) |
| **r3_best (enh off + naive, NO nrng)** | **21.91** | −4.10 | **still WER-best overall** |
| ah_finalist_nrng3 (enh off + naive + nrng3) | 23.11 | −2.90 | nrng3 HURTS the clean finalist (+1.2 collateral, no catastrophe to fix) |
| r3_best_flowhigh16 (enh off + flowhigh) | 26.28 | +0.28 | |
| ah_flowhigh_nrng3 (+ nrng3) | 24.15 | −1.85 | nrng3 fixes flowhigh's catastrophes too |

**Interpretation:** nrng3 trades catastrophe-repair against collateral on legitimate speech repeats (blocks real "tak tak tak"/number repeats). Net win ONLY where catastrophes exist (baseline −2.66, flowhigh −2.13, ap_bwe −1.0); net LOSS on the clean finalist (+1.2). So:
- **WER-optimal stays `r3_best` (enh off + naive, NO nrng) = 21.91** — it has no catastrophe to fix, so nrng only adds collateral. Don't make nrng a global default.
- **The catastrophes WERE the main WER cost of enhancement+ap_bwe** — guarding them rescues baseline 26→23.35, confirming the mechanism.
- **Dual-objective sweet spot: `enh on + ap_bwe + nrng3` (23.35) = +1.4 WER vs the WER-optimal, in exchange for the big SQUIM audio gain.** That's the "keep enhancement" config if the audio deliverable matters.
- Once guarded, ap_bwe ≥ naive (22.16 vs 23.11) — but naive-without-guard (21.91) is still simplest-best, since naive has no catastrophe to guard against.
- All CIs cross 0 (n=23); directions consistent. Test set validates.

## SQUIM on output streams (2026-06-14) — enhancement WINS on audio quality
Non-intrusive SQUIM_OBJECTIVE on the assembled `stream_{A,B}.wav`, enh-OFF (r3_best) vs enh-ON (r4_enhon), identical except enhancement, 23 dev frags:
| metric | enh-off | enh-on | Δ | enh-on better on |
|---|---|---|---|---|
| STOI | 0.852 | 0.898 | +0.046 | 21/23 |
| PESQ | 1.765 | 2.277 | +0.512 | **23/23** |
| SI-SDR | 5.64 | 10.83 | +5.19 | 21/23 |
→ Enhancement clearly improves perceptual output quality (the supervisor's SQUIM-on-output use case). Combined with the nrng3 hallucination fix, "enhancement is bad" is wrong: **enhancement is good for the audio deliverable, and its WER cost is a largely-fixable WhisperX repetition bug.** The pipeline is dual-objective (transcripts + listenable streams); enhancement trades a small, now-mitigable WER cost for a real audio-quality gain.

## Enhancement-model frontier (point 3, 2026-06-14) — all + naive BWE + nrng3
| enhancement | cpWER | CER | STOI | PESQ | SI-SDR |
|---|---|---|---|---|---|
| OFF (r3_best) | 21.91 | 15.90 | 0.852 | 1.76 | 5.6 |
| FRCRN | 23.36 | **15.82** | 0.898 | 2.28 | 10.8 |
| MossFormerGAN | 24.17 | 17.02 | **0.903** | **2.41** | **11.5** |
| MossFormer2-48k | 34.48 | 26.93 | 0.859 | 2.08 | 7.8 |

- **FRCRN+nrng3 is near-Pareto**: ~FREE in CER (15.82 vs 15.90 off — character-level, fairer for Polish), small cpWER cost (+1.45, shown character-light/morphological), BIG audio gain (PESQ +0.52, SI-SDR +5.2). The dual-objective "keep enhancement" pick.
- **MossFormerGAN = max-audio** (PESQ 2.41, STOI 0.903, SI-SDR 11.5) at a bit more WER cost — choose if audio is weighted heavily.
- **MossFormer2-48k is DOMINATED** — worst WER even guarded (34.48; the guard can't save it) AND mediocre audio (PESQ 2.08 < FRCRN). Drop it.
- **Recommendations:** WER-only product → enh OFF + naive (21.91). Transcript+audio product → **FRCRN + naive + nrng3** (balanced) or MossFormerGAN for max audio. ClearerVoice backends now exhausted; a genuinely new SE model would need a new backend (code).
- dry/wet mix (point 3) now LOW priority: FRCRN+nrng3 is already ~free in CER, so blending would mostly just shrink the audio gain. Worth it only to push the cpWER axis specifically. NB the dry/wet idea = Iwamoto et al. 2022's "observation adding" (arXiv:2201.06685) — the literature's recommended SE-artifact fix, so it's principled if revisited.

### BWE — kept OPEN (author decision 2026-06-14): keep all three (naive, ap_bwe, flowhigh) as live options. Findings below are informative, NOT a closure; future sweeps keep all three × the WER/CER/MIMO + SQUIM metrics.
- **WER:** naive ≈ ap_bwe (wash; flips by metric/guard), flowhigh rejected (own catastrophes, slower, unstable).
- **Audio (SQUIM on output streams, enh off, BWE-only difference):** naive 0.852/1.76/5.6 vs ap_bwe & flowhigh both 0.856/1.79/5.9 (STOI/PESQ/SI-SDR) — neural BWE +0.03 PESQ, negligible (BWE touches overlap regions only → diluted in the full stream).
- → **Use `naive` BWE**: free, fastest, no catastrophe, ~identical audio AND WER. Neural BWE buys nothing meaningful on either axis here.
- Lit note (SE-artifact→ASR, supports our enhancement finding): Iwamoto et al. Interspeech 2022 (2201.06685) — artifact component is the main ASR-degradation cause; IEEE/ACM TASLP 2024 extension; Wang et al. 2024 (2406.12699) — "NN-based SE often introduces artifacts ... harms ASR ... when SE and ASR are independently trained."

## Toolchain updates (2026-06-14)
- **MIMO-WER now reported as standard** alongside cpWER + CER in every sweep (`scripts/sweep_pipeline.py` score_configs; `mimo_wer_meeteval` extended to accept the per-speaker dict hyp). Verified sane: mimoWER < cpWER (MIMO is attribution-lenient; r3_best 18.44 vs 21.91, baseline 22.46 vs 26.00).
- **mossformer2_se_48k REMOVED** from active code (config enum, _CLEARVOICE_BACKENDS, docstrings, default/sweep_best yaml, r1_enh_mossformer2_48k + c_moss2_nrng3 configs, tests). 48 kHz model on 16 kHz sources = wrong tool (proven). 0 active-code refs remain.
- **ZipEnhancer wired** as `zipenhancer_16k` backend (ModelScope `iic/speech_zipenhancer_ans_multiloss_16k_base`, native 16k, DNS-2020 PESQ leader). Runs via a SUBPROCESS worker (`scripts/zipenhancer_worker.py`) because the repo's local `datasets/` package shadows the HF `datasets` modelscope needs. modelscope + addict/simplejson/yapf/datasets installed into main venv (all conflict-free, pins intact). Frontier run (23 dev) in progress.
- **Dry/wet "observation adding" (OA) — DECIDED to try** (both Iwamoto 2022 & Wang 2024 propose the same lever): blend observed signal back into enhanced output to dilute artifacts ASR mishandles. Iwamoto fixed ω∈[0.3,0.8] (~20% rel WER on frozen-SE→frozen-ASR), Wang learned ratio (no ASR access, no Polish data). Plan: add `mix_ratio` knob to enhancement, sweep ω∈{0,0.3,0.5,0.8} on solo regions, WER/CER/MIMO vs SQUIM frontier; learned bridge deferred. Complementary to nrng3 (tail vs steady-state artifacts). Trades audio↓ for WER↓ (moves the frontier).

## Enhancement-model frontier v2 (2026-06-14, all + naive BWE + nrng3, WER+MIMO+CER+SQUIM)
| enhancement | cpWER | MIMO | CER | STOI | PESQ | SI-SDR |
|---|---|---|---|---|---|---|
| OFF | 21.91 | 18.44 | 15.90 | 0.852 | 1.76 | 5.6 |
| FRCRN | 23.36 | 20.00 | 15.82 | 0.898 | 2.28 | 10.8 |
| MossFormerGAN | 24.17 | 21.63 | 17.02 | 0.903 | 2.41 | 11.5 |
| ZipEnhancer | 25.24 | 22.44 | 18.12 | **0.906** | **2.46** | **12.0** |
- **Textbook Pareto curve**: each step up in audio costs more WER. OFF=WER-best/audio-worst → FRCRN (≈free CER, good audio) → MossFormerGAN → ZipEnhancer (audio-best, WER-worst).
- **ZipEnhancer verdict: works, marginally best audio, but NOT worth it.** Its DNS-benchmark PESQ lead (3.69 vs FRCRN 3.23) collapses on our 16k phone-band CLARIN to +0.05 PESQ over MossFormerGAN, while costing +1.07 cpWER more — i.e. MossFormerGAN ≈ same audio at less WER cost → ZipEnhancer is effectively dominated. Plus it needs the modelscope + subprocess-worker machinery. Keep it as an option but don't adopt.
- **Practical picks unchanged:** FRCRN+naive+nrng3 = best balance (≈free CER + real audio gain); MossFormerGAN if maximizing audio; OFF if WER-only.

## OA / dry-wet sweep (2026-06-14) — CORRECTED: OA DOES help (outlier-masked)
Two backends (FRCRN + MossFormerGAN), ratios 0/0.3/0.5/0.7, 23 dev. Initial micro-avg looked null/negative — but db15fc57 (the recurring pathological fragment) **backfires hard under OA** (FRCRN 63.2→88.1 at r=0.3/0.5, +24.9) and dominates the micro-average. Per-fragment, OA-0.5 beat FRCRN-base on 13/22. Excluding db15fc57:
| config | all23 cpWER | excl-db15fc57 |
|---|---|---|
| OFF | 21.91 | 21.18 |
| FRCRN base | 23.36 | 21.79 |
| OA-frcrn 0.3 / 0.5 / 0.7 | 23.64 / 23.21 / 24.27 | 21.09 / **20.64** / 22.59 |
| MossGAN base | 24.17 | 21.71 |
| OA-mgan 0.3 / 0.5 / 0.7 | 22.73 / 24.23 / 24.72 | **20.93** / 21.71 / 23.10 |
- **Excl the outlier, OA at a MODERATE ratio improves WER on BOTH backends** (FRCRN best at r=0.5 → 20.64, even below OFF 21.18; MossGAN best at r=0.3 → 20.93). r=0.7 over-does it (noise dominates) on both. Iwamoto replicates, and the two-backend agreement (user's noise-guard) shows it's not a FRCRN fluke.
- **Caveat 1:** effect is small (~0.7–1.2 cpWER) and within the n=23 bootstrap CI — but directionally consistent across 2 backends + 2 ratios.
- **Caveat 2:** OA catastrophically backfires on db15fc57 (re-injecting its pathological observed signal worsens the WhisperX hallucination) — so with all 23, OA looks neutral. OA helps the typical fragment, hurts the one pathological one.
- **SQUIM frontier (OA re-adds noise → audio drops monotonically with r):** FRCRN PESQ 2.28→1.99→1.89→1.83 (r=0→.3→.5→.7); MossGAN 2.41→2.03→1.91→1.84. SI-SDR similarly drops toward the OFF floor (5.6).
- **KEY (excl db15fc57): OA-frcrn-0.5 (cpWER 20.64, PESQ 1.89, STOI 0.875, SISDR 7.6) Pareto-DOMINATES OFF (21.18, 1.76, 0.852, 5.6)** — better on BOTH WER and audio than no-enhancement. So a moderate OA recovers most of enhancement's audio gain AND beats no-enhancement on WER. OA-mgan-0.3 (20.93, PESQ 2.03) is a nice middle. This VINDICATES enhancement: with OA you get both better transcripts and better audio than OFF (on 22/23 frags).
- **Caveat (db15fc57):** with all 23, OA-0.5 = 23.21 cpWER (worse than OFF 21.91) — the db15fc57 backfire negates the broad benefit. → motivates the **Wang learned-bridge** (per-fragment adaptive ratio; deferred) which could set a low ratio on db15fc57 to avoid the backfire. db15fc57 = recurring pathological fragment (enh catastrophe + OA backfire) — worth a dedicated look.

## SWEEP COMPLETE ✅
Finalist `asr_pipeline/configs/sweep_best.yaml` (enh off + naive BWE). Top 4 a statistical tie (21.5–21.9); turbo variant equal & 1.5× faster. **Next: author validates on the held-out test split** (run sweep_best.yaml on test fragments, score once).

## FINALIST (recommended)
`asr_pipeline/configs/sweep_best.yaml` — enhancement off, BWE naive, else baseline. Validate on the held-out **test split** before adopting (run the same config on test fragments, score once, report).

---

# Follow-on arc — WhisperX over-merge collapse + detect-and-retry (2026-06-14 → 06-15)

Post-sweep, while inspecting `db15fc57` in `explore_pipeline.ipynb`, a deeper failure surfaced and was chased down. NB the author also re-edited GT (tighter speaker blocks) on several dev fragments in this window, so numbers here are on an **updated GT** and differ slightly from the rounds above.

## Root cause: WhisperX VAD over-merge → Whisper 30 s collapse (NOT diarization/separation)
db15fc57's headline error is a **transcription dropout**, not attribution. WhisperX (`merge_chunks`, v3.8.6) merges VAD speech into windows up to `chunk_size` (=30 s; `max_duration == chunk_size` — no independent max-segment knob; `min_duration_off` hardcoded 0.1 s). A ~18–30 s merged window makes Whisper **collapse** — emit ~nothing for the whole window. On db15fc57 the 39–68 s solo block became ONE 29.9 s window → "Nawiązań?" (1 word), dropping ~20 s of clear speech; re-transcribing that span at chunk_size=8 recovers ~50 correct words.
- Diarization was a red herring: pyannote is **88 % correct** on db15fc57 (perm-max); the apparent "swap" is benign positional A/B naming. Enhancement(3a)/separation(3b/c) clean (confirmed by ear; data agrees). ECAPA anchors richly fed (29.7 s / 39.1 s, weak_anchor=False).

## Correction: the collapse is GENERAL, not OA-specific (the earlier "OA-only" read was wrong)
Collapse detector (`/tmp/collapse_detector.py`; energy-VAD a stream, flag ≥8 s speech regions with <20 % transcript coverage), across configs:
| config | collapses | dropped speech |
|---|---|---|
| oa_frcrn_05 (OA) | 5 | 69 s |
| r3_best (OFF, best) | **2** | **23 s** |
| baseline / c_frcrn_nrng3 | 2 / 2 | ~22 s |
| ah_flowhigh_nrng3 | 3 | 52 s |
- db15fc57 AND **fe65d170** collapse in **every** config incl. OFF; some instances are config-independent (fe65d170 A[1.2–21.6], db15fc57 B[0–26]). OA *aggravates* (refilled inter-utterance gaps → more long windows), it does not *cause*. Even the best config silently drops ~23 s — part of why fe65d170/db15fc57 sit ~50/47.

## chunk_size knob — added, swept, REJECTED as the fix
Added `transcription.chunk_size` (default 30 = WhisperX default = no-op; whisper-backend rejects non-default). Sweep on the OA base (re-transcription, 23 dev):
| chunk_size | cpWER | excl-db15 |
|---|---|---|
| 30 | 24.7 | 22.1 |
| 24 | **22.0** | 21.2 |
| 20 | 22.2 | 21.4 |
| 15 | 24.4 | 23.7 |
- cs15 (blunt) fixes catastrophes but broad +3 collateral → wash; on the OFF config it's **pure** collateral (+2.9, no catastrophe to fix). cs24 is a sweet spot but **overfits** these 23 (a Life-2 adaptive-chunking problem). → do NOT ship a global chunk_size cut. Default stays 30.

## Detect-and-retry — the fix (SHIPPED, on by default)
Run the normal cs30 pass; from the raw pre-alignment segments, re-transcribe ONLY collapsed windows (dur ≥ `collapse_min_duration_s`=18 s AND words/sec < `collapse_max_wps`=0.7) at `retry_collapsed_chunk_size`=8, offset timestamps, splice, align. Guarded ("more-words-wins"); self-gating → fires only on near-empty long windows, clean fragments untouched.
- **Validated (re-transcription, 23 dev, micro-avg cpWER):** OA `oa_frcrn_05` 25.3→**22.3**; OFF `r3_best` 22.0→**21.3** — beats both cs30 and cs15 on each. ~18/23 fragments byte-identical; only fe65d170/94a0d89a tiny regressions.
- End-to-end on the SHIPPED code (`_WhisperXBackend`): db15fc57 retry-off → "Nawiązań?"; retry-on → logs `retry: collapsed window [39.0-68.9] (1w, 0.03 w/s) → 49w` and recovers "Nie wiem, czy pozarządowe organizacje…".
- whisper-backend: retry is whisperX-only + default-on, so it **logs-and-ignores** (visible, not silent) rather than erroring (which would break the default).
- Two independent reads agree (an investigation subagent on OA + the main session on OFF). Implementation by a second subagent; 25 new tests, `-k pipeline` = 511 pass.

## Best config on the UPDATED GT (faithful on-disk transcripts) + CER
| config | cpWER | CER | ORC |
|---|---|---|---|
| OFF (`sweep_best.yaml`) | **22.2** | 16.2 | 19.2 |
| OA (`sweep_best_excl_db15fc57.yaml`) | 23.3 | 16.5 | 20.9 |
| baseline (enh on + ap_bwe) | 26.0 | 17.4 | 23.0 |
- On the updated GT, **OFF is best on cpWER** and avoids the catastrophe entirely (db15fc57 47 vs OA's 91) — this **flips** the older-GT OA "vindication".
- **CER reframes it (Polish is inflected → CER fairer):** all-23, FRCRN+naive+nrng3 has the **best CER (15.9 < OFF 16.2)**; excl-db15, the enhancement configs (OA 13.7, FRCRN 14.1, MossGAN 14.2) all beat OFF (15.1) on CER. Enhancement wins character-level; the cpWER cost is largely word-boundary/attribution.
- **Ship-candidate × retry** (current GT, 23 dev, re-transcription through the shipped backend; * = excl db15fc57):

  | config | retry | cpWER | CER | ORC | MIMO | cpWER* | CER* |
  |---|---|---|---|---|---|---|---|
  | OFF (`sweep_best`) | off | 22.0 | 16.1 | 19.1 | 18.4 | 21.1 | 15.0 |
  | OFF (`sweep_best`) | **on** | **21.3** | 15.1 | 18.3 | 17.7 | 20.7 | 14.5 |
  | OA (`excl_db15`) | off | 24.7 | 18.1 | 22.2 | 21.4 | 22.1 | 15.3 |
  | OA (`excl_db15`) | **on** | **21.6** | **14.3** | 19.1 | 18.3 | **20.5** | **13.4** |

  Retry helps OA far more (cpWER −3.1, CER −3.8) than OFF (−0.7/−1.0) — OA had more collapses. **With retry on: cpWER is a tie** (OFF 21.3 all-23 / OA 20.5 excl-db15), but **OA wins CER** (14.3 vs 15.1 all-23; 13.4 vs 14.5 excl-db15) and audio. → with the collapse fixed, **OA (`sweep_best_excl_db15fc57.yaml`) is the stronger dual-objective pick** if CER/audio are weighted; OFF stays marginally ahead on all-23 cpWER (purely db15fc57). Author's call; no longer "OFF clearly wins." All dev-only, n=23; test set validates.

## Other changes in this arc
- `assembly.anchor_max_duration_s` 30 → **240** (all yamls + dataclass): fully covers ≤90 s eval fragments (was clipping, e.g. db15fc57 SPEAKER_01 39.1 s → 30 s); OOM-safe.
- NEW `configs/sweep_best_excl_db15fc57.yaml` = OA frcrn 0.5 + naive + nrng3 + full_length (dual-objective candidate; the "best excl the db15fc57 outlier" on older GT).
- `explore_pipeline.ipynb` — eval cells added (cpWER/ORC/attr_gap/MIMO/CER scoring; SQUIM on streams; recommended-configs notes; Stage 3a OA + zipenhancer knobs; Stage 5 decode/anti-hall knobs incl. no_repeat_ngram_size).

**Pending:** a newer MossFormer2 separator is incoming (author) → swap = edit checkpoint path (dataclass default + yamls; pin test enforces consistency) + a **full** dev re-run (separation change invalidates the assembled streams, so the re-transcription shortcut won't apply). Held-out TEST validation remains the real validator.

---

# RUN INDEX (chronological, 2026-06-13 → 06-14)
All on the 23-fragment dev split unless noted. Per-(config,recording) outputs at `<id>/sweep/<config>/`; aggregate CSV `~/datasets/eval/clarin_fragments/_sweep_results.csv` (overwritten per run — re-score any group with `--groups X --recordings $(cat /tmp/dev_recordings.txt) --score-only`). /tmp logs are ephemeral; key numbers are in the sections above.

| run / group | configs | purpose | headline |
|---|---|---|---|
| Round 1 (`r1`) | 33 OFAT + baseline | one-factor screen | baseline 26.0; enh-off best; decode knobs inert |
| Round 2 (`r2`) | combo + leave-one-out | stack winners | enh-off + naive + nrng best direction; v2≈turbo>v3, beam interaction |
| Round 3 (`r3`) | finalist + axis re-validate | converge | **finalist = enh off + naive = 21.91** (top-4 a tie) |
| Round 4 (`r4_enhon`) | enh-on at best op | enh×sep 2×2 | sep helps ~−5; enh hurts ~−2.6 (mostly db15fc57) |
| `ah_gate` | 5 knobs × 2 catastrophe frags | does nrng fix blowups? | nrng3 fixes 94a0d89a, halves db15fc57 |
| `ah_full` | nrng on 4 op points | collateral check | nrng3 = targeted fix (helps catastrophe configs, hurts clean finalist) |
| `c_enh` (+rerun) | frcrn/mossgan/zip/off | enhancement-model frontier | Pareto curve; ZipEnhancer best audio but dominated; FRCRN best balance |
| `oa` | frcrn + ratio 0.3/0.5/0.7 | observation-adding | excl db15fc57, OA-0.5 beats OFF on WER+audio |
| `oa_mossgan` | mossgan + ratio 0.3/0.5/0.7 | OA noise-guard (2nd backend) | confirms OA-0.3 helps; not a FRCRN fluke |
| SQUIM runs | outputs / bwe / oa | L2 audio-quality | enh improves audio; BWE ~no-op; OA trades audio↓ for WER↓ |
| `det_test` | enh-on × db15fc57 ×4 | determinism probe | db15fc57 catastrophe reproducible (not a seed artifact) |
| `chunk_size` (re-transcribe) | OA base, cs 30/24/20/15/8 | fix the over-merge collapse? | cs24 sweet-spot but overfit; global cut rejected |
| `collapse_detector` | 6 configs, CPU | how widespread is the collapse? | general, not OA-only (db15fc57+fe65d170 every config) |
| `retry` (re-transcribe) | OA + OFF, cs30/cs15/detect-and-retry | validate the fix | retry beats both: OA 25.3→22.3, OFF 22.0→21.3, ~0 collateral |
| `ship_compare` (re-transcribe) | OFF + OA × retry{off,on} | concrete cpWER+CER for the two ship candidates | [running `bwr4ll1en`] |

# CHANGELOG — code / config / dependency changes (2026-06-13 → 06-14, ALL UNCOMMITTED)
Nothing committed; review before committing. `default.yaml` unchanged in spirit except the separator swap — finalist lives in `sweep_best.yaml`, not `default.yaml`.

**Code (`asr_pipeline/`, `scripts/`)**
- `config.py` — separation default → MossFormer2 matched-128k; +6 Whisper decode knobs (beam_size, temperature, condition_on_previous_text, no_speech_threshold, compression_ratio_threshold, patience); +3 anti-hallucination knobs (no_repeat_ngram_size, repetition_penalty, hallucination_silence_threshold); +`zipenhancer_16k` enum; +`observation_mix_ratio` (OA); **removed `mossformer2_se_48k`**; validation for all.
- `stages/transcription.py` — wire the 9 decode/anti-hall knobs into the WhisperX `asr_options`; whisper-backend guard (rejects unsupported knobs).
- `stages/enhancement.py` — +`_ZipEnhancerBackend` (subprocess worker); **removed `mossformer2_se_48k`** from `_CLEARVOICE_BACKENDS`; +OA blend `(1-r)*enhanced + r*observed` at end of `run()` (solo-scoped via enhanced_full).
- `eval/metrics.py` — `mimo_wer_meeteval` now accepts a per-speaker dict (pipeline MIMO-WER), mixture (list) path byte-identical.
- `scripts/sweep_pipeline.py` — separation default; config registry (r1–r4, ah_*, c_*, oa_*, oa_mossgan_*) + groups; **MIMO-WER + secs_per_frag columns**; bootstrap CIs; paired-vs-baseline significance; config-level try/except (one bad config can't abort a sweep); removed mossformer2 configs.
- `scripts/zipenhancer_worker.py` — NEW; standalone ModelScope ANS worker (isolates `datasets` namespace clash).

**Configs / docs**
- `configs/default.yaml`, `english.yaml`, `frcrn_vadstrict.yaml`, `p4_fixed_pad.yaml`, `p5_full_length.yaml` — separator → MossFormer2; new knobs added to default.yaml; mossformer2_se_48k dropped from comments.
- `configs/sweep_best.yaml` — NEW finalist (enh off + naive BWE, else baseline).
- `SWEEP_KNOBS.md` — NEW knob inventory. `SWEEP_RUNLOG.md` — NEW (this file).
- `CLAUDE.md` — separation-default line updated. `SCOPE.md` — §9 adaptive-stage note (this arc).

**Tests** — `test_pipeline_config.py`, `test_pipeline_enhancement.py`, `test_pipeline_sweep_helpers.py`, `test_pipeline_transcription.py` extended for every knob/metric/removal above (full `-k pipeline` suite green at each step).

**Dependencies (main venv, all conflict-free, pins intact)** — `modelscope 1.37.1` + `addict`, `simplejson`, `yapf`, `datasets` (+ `dill`, `multiprocess`, `xxhash`), for the ZipEnhancer backend. Reversible via pip uninstall.

**Data** — eval set 142→143 earlier (9a651086 reslice+add); per-config sweep outputs written under each dev fragment's `sweep/`. No source data modified.

**NOT part of this arc (pre-existing uncommitted):** `asr/explore_pipeline.ipynb` (+ earlier SCOPE.md edits).

## CHANGELOG addendum — follow-on arc (2026-06-14 → 06-15, ALL UNCOMMITTED)
- `config.py` — `TranscriptionConfig`: +`chunk_size` (int, default 30 = WhisperX default = no-op); +detect-and-retry knobs `retry_collapsed_chunk_size` (int, **default 8 = ON**), `collapse_min_duration_s` (18.0), `collapse_max_wps` (0.7); validation for all. `AssemblyConfig.anchor_max_duration_s` 30 → 240.
- `stages/transcription.py` — `_WhisperXBackend.transcribe`: pass `chunk_size`; guarded detect-and-retry (`_retry_collapsed`) between raw transcribe and align, logged via `_log` (SCOPE §4.1-clean). `_WhisperBackend`: rejects non-default `chunk_size` (no-op default), but logs-and-ignores the retry knobs (default-on, whisperX-only).
- `configs/` — `anchor_max_duration_s: 240` in all yamls; `default.yaml` documents `chunk_size` + the 3 retry knobs (retry on at 8). NEW `configs/sweep_best_excl_db15fc57.yaml` (OA frcrn 0.5 + naive + nrng3 + full_length).
- `scripts/sweep_pipeline.py` — `chunk_size` group (oa_cs15/cs08, r3_cs15/cs08) for the chunk_size sweep.
- `asr/explore_pipeline.ipynb` — eval cells (WER/CER/attr_gap/SQUIM), Stage 3a OA+zipenhancer knobs, Stage 5 decode/anti-hall knobs.
- Tests — `test_pipeline_config.py` + `test_pipeline_transcription.py` extended for chunk_size + the 3 retry knobs (25 new; `-k pipeline` = 511 pass).
- Investigation artifacts in `/tmp` (not in repo): `collapse_detector.py`, `probe_raw.py`, `retry_harness.py`, `retry_harness_guarded.py`, `chunk_sweep.py`, `ship_compare.py`, `subagent_report.md`.
