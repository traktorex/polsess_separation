# ASR pipeline — sweepable knob inventory

Reference for planning a comprehensive config sweep. Generated 2026-06-13 from
`asr_pipeline/config.py` + `configs/default.yaml` + `scripts/sweep_pipeline.py`.

## How the sweep applies knobs

- The sweep harness (`scripts/sweep_pipeline.py`) is **data-driven**: each entry
  in the `CONFIGS` registry is a dict of dotted `stage.field → value` overrides
  applied on top of the baseline. Any config field is reachable — no new code to
  sweep a new knob; a typo'd path fails loud (`AttributeError`).
- **Baseline** = `configs/default.yaml` **+ eval overrides** from
  `asr_pipeline.eval.config_presets.fresh_eval_cfg`. Overrides stack on that.
- **Phase-major, one model on GPU at a time.** Each (config, recording) is a
  fresh `Pipeline` that loads+unloads every model, so per-run model-load cost
  dominates on short fragments — keep the active config set focused, prefer OFAT
  (one-factor-at-a-time) rows + targeted interaction rows over a full grid.
- Scoring: cpWER / tcpWER (+ CER, ORC/MIMO baselines) against the hand-corrected
  GT, ranked. Dev set only (the 23-fragment dev split); test is scored once at
  the end with the frozen winner.

## Legend

- ✅ **swept** — already has row(s) in `CONFIGS`
- 🆕 **gap** — reachable + meaningful, but never tried (candidates for the comprehensive sweep)
- 📌 **eval-pinned** — forced by `fresh_eval_cfg`; do **not** sweep (breaks L3 scoring)
- 🔒 **fixed** — tied to the model / corpus / infrastructure; not a quality lever

> **2026-06-13 change:** the separation checkpoint default is now the MossFormer2
> matched-128k model (`checkpoints/mossformer2/SB/mossformer2_matched_128k_final_42_e31/mossformer2_SB_best_e31.pt`,
> ~0.6 dB SI-SDRi over the prior SepFormer 128k on val). `separator_sample_rate`
> stays 8000 (both trained on 8 kHz PolSESS).

## Eval-pinned — set by `fresh_eval_cfg`, never sweep these

| dotted path | forced value | why |
|---|---|---|
| `transcription.transcribe_mixture` | `True` | ORC-WER mixture baseline needs it |
| `assembly.output_mode` | `full_length` | tcpWER needs streams on the mixture timeline |
| `routing.min_overlap_dur` | `0.0` | apples-to-apples — every overlap reaches the separator |

(`min_overlap_dur` is the dataclass/yaml `0.20`, but eval overrides it to `0.0`;
sweeping it would re-introduce routing-time drops and is off the table for eval.)

---

## Stage 1 — Diarization (`diarization.*`)

| knob | type | baseline | options/range | status | notes |
|---|---|---|---|---|---|
| `backend` | enum | pyannote | pyannote \| sortformer | ✅ | **`sortformer` ADOPTED in the shipped best config 2026-07-04 (V5 instrument reframe, `V5_INSTRUMENT_PREREG.md` §PHASE-2 VERDICT)** — dataclass default stays pyannote. NVIDIA Sortformer v1 offline, pure EEND (no embedding clustering) via isolated-venv worker (`scripts/sortformer_worker.py`, `$SORTFORMER_VENV_PY`, no default → loud crash). History: `v4_eend` test NOT ADOPTED under the deployment standard (13-rec veto tail, `EEND_ARM_PREREG.md` §VERDICT); re-adopted WITH the fold under the instrument criterion — test 24.0/16.5 vs pyannote 26.9/19.4, attribution gap collapsed, 12-rec tail documented (all inherited, none fold-caused) |
| `num_speakers` | int | 2 | — | 🔒 | corpus is 2-speaker; sortformer backend hard-errors on ≠2 |
| `model_id` | str | pyannote/speaker-diarization-3.1 | — | 🔒 | one model in use (pyannote path) |
| `sortformer_model_id` | str | nvidia/diar_sortformer_4spk-v1 | — | ✅ | fixed-4-head model, top-2 by activity; head-miscount detector logs loudly at leak >5%. `nvidia/diar_streaming_sortformer_4spk-v2.1` (NVIDIA Open license = Life-2 track) swept as `v2p1_*` dev arms 2026-07-04: near-parity but lost to v1+fold on every V5 instrument criterion (`V5_INSTRUMENT_PREREG.md` §PHASE-1); worker auto-applies the offline very-high-latency preset for streaming ids |
| `sortformer_threshold` | float | 0.5 | 0.0–1.0 | ⬜ | frame-activity binarization; probe showed purity robust 0.4–0.6, but the v4 under-detection tail (quiet-speaker deletions) makes <0.5 the named candidate lever for any v4.1 |
| `sortformer_head_policy` | enum | top2 | top2 \| merge | ✅ | **`merge` (the fold) ADOPTED in the shipped best config 2026-07-04.** Reassigns discarded surplus-head runs (>= 0.4 s) to the nearest top-2 speaker by local ECAPA2 cosine margin (no global clustering); default `top2` = v4 discard. Test: repairs the miscount class (15a55a8a 26.5→7.6) with ZERO collateral (strictly dominates discard on all 118 frags); dataclass default stays top2 |
| `sortformer_merge_margin` | float | 0.10 | >= 0 | ⬜ | **v4.1 L1** cosine margin a surplus run must clear to merge (anatomy true-margins 0.15–0.57). Only used when `head_policy=merge` |
| `sortformer_long_audio_threshold_s` | float s | 240.0 | >= 0 | 🔒 | **deployment/robustness, not a quality lever.** Recordings longer than this route to the streaming model below (offline v1 is O(T²) memory, OOMs past ~5-6 min on 12 GB — observed e14aa22f.wav 32:47). 0 = disable routing. 240 s keeps eval fragments (≤~95 s) on v1 (byte-identical). Loud warning on swap (not silent, SCOPE §4) |
| `sortformer_long_audio_model_id` | str | nvidia/diar_streaming_sortformer_4spk-v2.1 | — | 🔒 | **deployment/robustness, not a quality lever.** Long-audio replacement model (streaming, run offline via the same worker; memory flat in duration, identical (T,4)@0.08 s contract). Used only when routing engages |
| `embedding` | str | resnet34-LM | resnet34-LM \| resnet293-LM \| ecapa2 | ✅ | first-pass speaker embedder (triggers a 3.1-equivalent rebuild); ecapa2 adopted (`dr_emb_*`). (`eres2netv2` removed 2026-07-06 — swept-and-lost, zh-cn domain mismatch.) |
| `clustering_method` | enum | centroid | centroid \| average \| ... | ✅ | **promoted 2026-06-22** (was hard-coded linkage); agglomerative linkage in the embedding-swap path the best config uses; `dr_linkage_avg` |
| `enabled` | bool | true | — | 🔒 | diarization is mandatory upstream |

## Stage 2 — Routing (`routing.*`)

| knob | type | baseline | options/range | status | notes |
|---|---|---|---|---|---|
| `min_overlap_dur` | float s | 0.20 | ~0.0–0.5 | 📌 | eval pins to 0.0 |
| `merge_gap` | float s | 0.50 | ~0.2–1.0 | 🆕 | merges nearby overlaps into one separator call; untried |
| `enabled` | bool | true | — | 🔒 | |

## Stage 3a — Enhancement (`enhancement.*`) — solo regions

| knob | type | baseline | options/range | status | notes |
|---|---|---|---|---|---|
| `backend` | enum | frcrn_se_16k | frcrn_se_16k \| mossformer_gan_se_16k | ✅ | frcrn & mossformer_gan swept. ⚠️ `mossformer2_se_48k` and ModelScope `zipenhancer_16k` are **NOT valid** — `config.py.__post_init__` allows only frcrn/mossformer_gan; they would raise. (zipenhancer removed 2026-07-06, swept-and-lost.) |
| `observation_mix_ratio` | float | 0.0 (yaml 0.3) | 0.0–0.5 | ✅ | dry/wet artifact-dilution (Observation Adding); heavily swept (`f_oa0*`, `dr_oa0*`) |
| `resample_quality` | enum | soxr_hq | soxr_hq \| soxr_vhq \| kaiser_best | ✅ | **promoted 2026-06-22** (was hard-coded `res_type`); anti-alias filter into the enhancer + OA blend, upstream of OA; `dr_resample_*` |
| `enabled` | bool | true | true/false | ✅ | `enh_none` ablation corner |
| `max_segment_length_s` | float s | 8.0 | ~4–16 | 🆕 | Hann overlap-add chunk for long solos; **HAND-SET, untested** (not "swept-inert"), minor |

## Stage 3b — Separation (`separation.*`) — overlap regions + VAD

| knob | type | baseline | options/range | status | notes |
|---|---|---|---|---|---|
| `checkpoint_path` | str | mossformer2 matched-128k | (swappable) | 🔒/🆕 | now MossFormer2; old SepFormer 128k is an optional A/B ablation row |
| `separator_sample_rate` | int | 8000 | — | 🔒 | = model training SR |
| `training_chunk_length_s` | float s | 4.0 | — | 🔒 | property of the trained model; don't sweep blind |
| `context_window_mode` | enum | expand_to_chunk | expand_to_chunk \| fixed_pad \| none | 🆕 | **how much context the separator sees — real lever, never swept** (p4/p5 ablation configs touch related behaviour) |
| `context_pad_seconds` | float s | 1.0 | ~0.5–2.0 | 🆕 | only used by `fixed_pad` |
| `min_fragment_length_s` | float s | 4.0 | ~2–6 | 🆕 | floor for padded-window length; untried |
| `seam_mode` | enum | snap_to_silence | zero_crossing \| overlap_boundary \| snap_to_silence | ✅ | `sep_seam_zc`, `sep_seam_boundary` |
| `seam_search_radius_s` | float s | 0.05 | ~0.02–0.1 | 🆕 | zero-crossing nudge radius; minor |
| `snap_silence_max_extend_s` | float s | 0.3 | ~0.1–0.5 | 🆕 | snap_to_silence outward reach; untried |
| `seam_silence_threshold` | float | 0.5 | 0.3–0.7 | ✅ | **promoted 2026-06-22** (was hard-coded); the silence cutoff `snap_to_silence` uses to grow emit regions — companion to the swept VAD pair; `dr_seamsil*` |
| `overlap_add_threshold_s` | float s | 12.0 | ~6–20 | 🆕 | when long overlaps get chunked; minor |
| `volume_normalization` | enum | sum_equals_mix | sum_equals_mix \| none | ✅ | `sep_vol_none` |
| `vad_threshold` | float | 0.25 | 0.2–0.6 | ✅ | heavily swept (round 3 ≈ 0.5 best) |
| `vad_soft_threshold` | float | 0.10 | 0.0–0.25 | ✅ | Schmitt lower; swept with `vad_threshold` |
| `vad_attack_frames` | int (32 ms) | 1 | 0–2 | ✅ | `ar0`, `ar2` |
| `vad_release_frames` | int (32 ms) | 1 | 0–2 | ✅ | `ar0`, `ar2` |
| `enabled` | bool | true | true/false | ✅ | `nosep` ablation corner |

## Stage 3c — Post-separation / BWE (`post_separation_processing.*`) — overlap regions

| knob | type | baseline | options/range | status | notes |
|---|---|---|---|---|---|
| `backend` | enum | ap_bwe | naive \| ap_bwe | ✅ | `bwe_naive` vs `ap_bwe`. (FlowHigh backend + `flowhigh_input_sr` removed 2026-07-06 — swept-and-lost, external git dep.) |
| `checkpoint_path` | str | AP-BWE g_8kto16k | — | 🔒 | AP-BWE weights path |

## Stage 4 — Assembly (`assembly.*`)

| knob | type | baseline | options/range | status | notes |
|---|---|---|---|---|---|
| `output_mode` | enum | shortened | shortened \| full_length | 📌 | eval pins to full_length |
| `overlap_rms_match_solo` | bool | true | true/false | 🆕 | toggle off untried (only on-by-default) |
| `per_piece_rms_norm` | bool | false | true/false | ✅ | `asm_perpiece_rms` |
| `target_rms` | float?\| None | null | — | 🆕 | only with `per_piece_rms_norm`; untried |
| `weak_anchor_warn_below_s` | float s | 3.0 | ~2–5 | 🆕 | sets the *diagnostic* `weak_anchor` flag only — NOT the actual anchor fallback (that's `anchor_min_duration_s`); minor |
| `anchor_min_duration_s` | float s | 0.25 | 0.25–1.0 | ✅ | **promoted 2026-06-22** (was hard-coded `_ECAPA_MIN_DURATION_S`); the REAL fallback trigger — speaker with < this solo gets no anchor → positional attribution; `dr_anchmin*` |
| `overlap_min_duration_s` | float s | 0.1 | 0.1–0.35 | ✅ | **promoted 2026-06-22** (was hard-coded `_ECAPA_OVERLAP_MIN_DURATION_S`); gates whether the per-overlap ECAPA decision runs at all (~⅓ of overlaps are sub-0.5 s); `dr_ovmin*` |
| `crossfade_ms` | float ms | 5.0 | ~2–10 | 🆕 | internal seam fade; minor |
| `edge_fade_ms` | float ms | 2.0 | ~1–5 | 🆕 | stream-edge fade; minor |
| `silence_separator_s` | float s | 0.3 | — | 🔒 | only matters in `shortened` (eval uses full_length) |
| `anchor_max_duration_s` | float?\| None | 240.0 | — | 🔒 | OOM guard, not a quality lever (dataclass default is 240.0, not 30.0) |
| `enabled` | bool | true | — | 🔒 | |

## Stage 5 — Transcription (`transcription.*`)

| knob | type | baseline | options/range | status | notes |
|---|---|---|---|---|---|
| `model_name` | str | large-v2 | large-v2 \| large-v3 \| HF finetune id | ✅ | **biggest single WER lever**; `asr_largev3`. HF Polish Whisper finetunes untried 🆕 |
| `align_model_name` | str\| None | None (→ pl XLSR-53) | None \| HF wav2vec2 id | 🆕 | alternative Polish aligners affect tcpWER; untried |
| `silence_floor` | float | 1e-4 | 0.0–1e-3 | ✅ | **promoted 2026-06-22** (was hard-coded `_SILENCE_FLOOR`); peak-amp gate below which a stream → empty transcript; direct deletion↔insertion trade on the WER decomposition; `dr_silfloor*` |
| `retry_collapsed_chunk_size` | int | 8 | 0 (off) \| 8 | ✅ | detect-and-retry on collapsed streams; `f_noretry`, `dr_retry0` |
| `collapse_min_duration_s` / `collapse_max_wps` | float | 18.0 / 0.7 | — | 📌 | collapse-detector gate feeding the chunked retry above |
| `loop_retry` | bool | false (finalist YAML: **true**) | on/off | ✅ | **adopted 2026-07-02** (`V2_TEST_PREREG.md`): token-repetition-loop detect-and-retry; gate ×12; ngram constraint on the retry decode only (always-on ngram rejected — 61 % collateral); killed 6/6 test loops, zero off-target |
| `loop_retry_phrase` | bool | false (finalist YAML: **true**) | on/off | ✅ | **adopted 2026-07-03** (`V3_TEST_PREREG.md` §v3.1): multi-word phrase-run detector (run ≥4; max genuine GT run = 2) the token gate can't see; same retry decode + length bound (retry may never out-grow the window — rejects counting-evasion hallucinations) |
| `loop_retry_ngram` / `loop_score_threshold` | int/float | 3 / 0.4 | — | 📌 | shared mechanics for both loop levers; calibrated, not re-swept |
| `initial_prompt` | str | "Rozmowa po polsku." | (text) | 🆕 | prompt wording can shift WER; untried |
| `backend` | enum | whisperx | whisperx \| coherex | 📌 | eval needs whisperx (word alignment for tcpWER); coherex kept for re-test. (base openai-`whisper` removed 2026-07-06.) |
| `language` | str | pl | — | 🔒 | corpus is Polish |
| `word_timestamps` | bool | true | — | 🔒 | needed for alignment |
| `transcribe_mixture` | bool | (eval: true) | — | 📌 | eval pins true (ORC-WER) |

## Top-level (`*`)

| knob | type | baseline | status | notes |
|---|---|---|---|---|
| `sample_rate` | int | 16000 | 🔒 | pipeline working rate |
| `device` | str | cuda | 🔒 | |
| `deterministic` | bool | true | 🔒 | reproducibility; set false only for dev-speed, not a quality knob |
| `spill_intermediate` / `artifact_dir` | bool/str | false/null | 🔒 | I/O plumbing |

---

## Comprehensive-sweep gaps (🆕 worth adding)

Ranked rough priority for a thorough sweep, beyond what `CONFIGS` already covers:

1. ~~**`enhancement.backend = mossformer2_se_48k`**~~ — **INVALID / removed 2026-06-22.** `config.py.__post_init__` rejects it (allows only frcrn/mossformer_gan); it was also the worst enhancer (34.5, e23-era). Do not add it.
2. **`separation.context_window_mode`** (expand_to_chunk / fixed_pad / none) — a real, untouched lever on what the separator sees; with `context_pad_seconds` for the `fixed_pad` arm.
3. **`transcription.model_name`** — extend beyond large-v2/v3 to a Polish Whisper finetune (HF id via whisperx).
4. **`transcription.align_model_name`** — alternative Polish wav2vec2 aligners (tcpWER quality).
5. **`transcription.initial_prompt`** — prompt-wording A/B.
6. **`routing.merge_gap`** — overlap-merge distance.
7. **`assembly.overlap_rms_match_solo`** off; **`separation.snap_silence_max_extend_s`**, **`min_fragment_length_s`**, **`overlap_add_threshold_s`** — second-tier separation/assembly knobs.
8. **Separation checkpoint A/B** — new MossFormer2 vs old SepFormer 128k, to confirm the val-set SI-SDRi gain carries to downstream WER.

Re-anchor each new OFAT row on the current best base config (per the existing
round-N pattern: round-2/3 re-anchored on the FRCRN winner), not on the bare
baseline, so marginal effects are read against the right operating point.
