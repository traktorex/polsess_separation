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
| `num_speakers` | int | 2 | — | 🔒 | corpus is 2-speaker |
| `model_id` | str | pyannote/speaker-diarization-3.1 | — | 🔒 | one model in use |
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
| `backend` | enum | frcrn_se_16k | frcrn_se_16k \| mossformer_gan_se_16k \| **mossformer2_se_48k** | ✅/🆕 | frcrn & mossformer_gan swept; **mossformer2_se_48k never tried** |
| `enabled` | bool | true | true/false | ✅ | `enh_none` ablation corner |
| `max_segment_length_s` | float s | 8.0 | ~4–16 | 🆕 | Hann overlap-add chunk for long solos; untried, minor |

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
| `backend` | enum | ap_bwe | naive \| ap_bwe \| flowhigh | ✅ | `bwe_naive`, `bwe_flowhigh`, flowhigh variants |
| `flowhigh_input_sr` | int | 16000 | 8000 \| 16000 | ✅ | only used by flowhigh; `flowhigh16` arm |
| `checkpoint_path` | str | AP-BWE g_8kto16k | — | 🔒 | AP-BWE weights path |

## Stage 4 — Assembly (`assembly.*`)

| knob | type | baseline | options/range | status | notes |
|---|---|---|---|---|---|
| `output_mode` | enum | shortened | shortened \| full_length | 📌 | eval pins to full_length |
| `overlap_rms_match_solo` | bool | true | true/false | 🆕 | toggle off untried (only on-by-default) |
| `per_piece_rms_norm` | bool | false | true/false | ✅ | `asm_perpiece_rms` |
| `target_rms` | float?\| None | null | — | 🆕 | only with `per_piece_rms_norm`; untried |
| `min_solo_for_anchor_s` | float s | 3.0 | ~2–5 | 🆕 | ECAPA anchor min; minor |
| `crossfade_ms` | float ms | 5.0 | ~2–10 | 🆕 | internal seam fade; minor |
| `edge_fade_ms` | float ms | 2.0 | ~1–5 | 🆕 | stream-edge fade; minor |
| `silence_separator_s` | float s | 0.3 | — | 🔒 | only matters in `shortened` (eval uses full_length) |
| `anchor_max_duration_s` | float?\| None | 30.0 | — | 🔒 | OOM guard, not a quality lever |
| `enabled` | bool | true | — | 🔒 | |

## Stage 5 — Transcription (`transcription.*`)

| knob | type | baseline | options/range | status | notes |
|---|---|---|---|---|---|
| `model_name` | str | large-v2 | large-v2 \| large-v3 \| HF finetune id | ✅ | **biggest single WER lever**; `asr_largev3`. HF Polish Whisper finetunes untried 🆕 |
| `align_model_name` | str\| None | None (→ pl XLSR-53) | None \| HF wav2vec2 id | 🆕 | alternative Polish aligners affect tcpWER; untried |
| `initial_prompt` | str | "Rozmowa po polsku." | (text) | 🆕 | prompt wording can shift WER; untried |
| `backend` | enum | whisperx | whisper \| whisperx | 📌 | eval needs whisperx (word alignment for tcpWER) |
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

1. **`enhancement.backend = mossformer2_se_48k`** — the one enhancement backend never tried.
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
