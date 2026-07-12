"""Sweep ASR-pipeline configurations over a fixed eval set and rank by cpWER.

Runs each named config (a set of overrides on ``default.yaml`` + the eval
overrides from ``asr_pipeline.eval.fresh_eval_cfg``) across the pilot
recordings, writes outputs to ``<id>/sweep/<config_name>/``, then scores
every config's per-speaker transcripts against the hand-corrected GT EAF
(``<id>/annotation.eaf``) with cpWER / tcpWER and ranks them.

The config registry is **data-driven**: each entry is a dict of dotted
override paths → values, applied on top of the baseline. Add a row to
``CONFIGS`` (and optionally to ``GROUPS``) to explore a new knob — no new
code. Single-knob (OFAT) entries isolate each knob's marginal effect from
the baseline; combine paths in one dict for interactions.

Phase-major: each (config, recording) is a fresh ``Pipeline`` that loads +
unloads its models. That re-runs every stage per config — simple and
correct, but the per-run model-load overhead dominates on short fragments,
so keep the active config set focused.

Usage::

    python scripts/sweep_pipeline.py --groups asr enhance      # run two groups
    python scripts/sweep_pipeline.py --configs baseline enh_mossformer
    python scripts/sweep_pipeline.py --score-only              # re-rank, no runs
    python scripts/sweep_pipeline.py --groups asr --force      # re-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# The per-recording run loop (Pipeline construction, GPU teardown, per-run
# outputs + run_meta) now lives in `asr_pipeline.batch.run_batch`; run_config
# only wires the sweep's registry + legacy sentinel into it.
from asr_pipeline.batch import run_batch                            # noqa: E402
from asr_pipeline.config import apply_overrides                     # noqa: E402
from asr_pipeline.eval.metrics import per_fragment_metrics          # noqa: E402
from asr_pipeline.eval.config_presets import fresh_eval_cfg         # noqa: E402
from asr_pipeline.eval.layer3 import read_mixture, read_per_speaker  # noqa: E402
from asr_pipeline.eval.recordings import (                          # noqa: E402
    load_recording,
    load_reference_utterances,
)
from scripts.eval_harness import eval_root, load_split             # noqa: E402


EVAL_ROOT = eval_root()
CFG_PATH = REPO_ROOT / "asr_pipeline" / "configs" / "default.yaml"

# The 5 pilot recordings (hand-corrected GT present).
PILOT = [
    "065a9896__seg00", "150d1ccc__seg00", "ccfbb9db__seg00",
    "2bf3474d__seg00", "d1e63652__seg00",
]

# --- Config registry ------------------------------------------------------
# Each value is a dict of "stage.field" -> value, applied on the baseline.
# baseline = default.yaml (full pipeline: enh=frcrn_se_16k, sep=128k SepFormer,
# bwe=ap_bwe, whisperx large-v2) + eval overrides (full_length,
# transcribe_mixture, min_overlap_dur=0).
CONFIGS: dict[str, dict] = {
    "baseline":          {},   # full pipeline; transcription = large-v2
    # --- transcription model (the biggest WER lever); large-v2 = baseline ---
    "asr_largev3":       {"transcription.model_name": "large-v3"},
    # --- enhancement backend (solo regions) ---
    "enh_mossformer":    {"enhancement.backend": "mossformer_gan_se_16k"},
    "enh_frcrn":         {"enhancement.backend": "frcrn_se_16k"},
    "enh_none":          {"enhancement.enabled": False},
    # --- bandwidth extension (overlap regions) ---
    "bwe_naive":         {"post_separation_processing.backend": "naive"},
    # --- separation knobs ---
    "sep_seam_zc":       {"separation.seam_mode": "zero_crossing"},
    "sep_seam_boundary": {"separation.seam_mode": "overlap_boundary"},
    "sep_vad_strict":    {"separation.vad_threshold": 0.5,
                          "separation.vad_soft_threshold": 0.2},
    "sep_vol_none":      {"separation.volume_normalization": "none"},
    # --- assembly knobs ---
    "asm_perpiece_rms":  {"assembly.per_piece_rms_norm": True},
    # --- ablation corners (2x2 separation x enhancement) ---
    #   baseline   = sep + enh   (full)
    #   enh_none   = sep, no enh (above, in the enhance group)
    #   nosep      = enh, no sep
    #   nosep_noenh = neither
    "nosep":             {"separation.enabled": False},
    "nosep_noenh":       {"separation.enabled": False, "enhancement.enabled": False},
    # --- round 2: re-anchor the promising knobs on FRCRN (round-1 winner) ---
    "frcrn_vad_strict":  {"enhancement.backend": "frcrn_se_16k",
                          "separation.vad_threshold": 0.5,
                          "separation.vad_soft_threshold": 0.2},
    "frcrn_seam_zc":     {"enhancement.backend": "frcrn_se_16k",
                          "separation.seam_mode": "zero_crossing"},
    "frcrn_vad_seam":    {"enhancement.backend": "frcrn_se_16k",
                          "separation.vad_threshold": 0.5,
                          "separation.vad_soft_threshold": 0.2,
                          "separation.seam_mode": "zero_crossing"},
    "frcrn_largev3":     {"enhancement.backend": "frcrn_se_16k",
                          "transcription.model_name": "large-v3"},
    # --- round 3: micro-variants around frcrn_vad_strict (round-2 winner) ---
    # frcrn_vad_strict = frcrn + VAD 0.5/0.2 + attack/release 1/1 (defaults).
    "frcrn_vad_040":     {"enhancement.backend": "frcrn_se_16k",
                          "separation.vad_threshold": 0.4,
                          "separation.vad_soft_threshold": 0.15},
    "frcrn_vad_060":     {"enhancement.backend": "frcrn_se_16k",
                          "separation.vad_threshold": 0.6,
                          "separation.vad_soft_threshold": 0.25},
    "frcrn_vad_strict_ar0": {"enhancement.backend": "frcrn_se_16k",
                             "separation.vad_threshold": 0.5,
                             "separation.vad_soft_threshold": 0.2,
                             "separation.vad_attack_frames": 0,
                             "separation.vad_release_frames": 0},
    "frcrn_vad_strict_ar2": {"enhancement.backend": "frcrn_se_16k",
                             "separation.vad_threshold": 0.5,
                             "separation.vad_soft_threshold": 0.2,
                             "separation.vad_attack_frames": 2,
                             "separation.vad_release_frames": 2},

    # ======================================================================
    # MossFormer2-separator comprehensive sweep (2026-06-13). Baseline =
    # default.yaml (enh=frcrn, sep=MossFormer2 matched-128k, bwe=ap_bwe,
    # whisperx large-v2, vad 0.25/0.10, snap_to_silence, expand_to_chunk).
    # All Round-1 entries are OFAT (one knob off baseline). See GROUPS["r1"].
    # ======================================================================
    # --- transcription model ---
    "r1_asr_largev3":       {"transcription.model_name": "large-v3"},
    "r1_asr_largev3_turbo": {"transcription.model_name": "large-v3-turbo"},
    "r1_asr_distil_pl":     {"transcription.model_name": "mmalyska/distil-whisper-large-v3-pl-ct2"},
    # --- transcription decode knobs (NEW) ---
    "r1_asr_beam1":         {"transcription.beam_size": 1},
    "r1_asr_beam10":        {"transcription.beam_size": 10},
    "r1_asr_temp0":         {"transcription.temperature": 0.0},
    "r1_asr_cond_prev":     {"transcription.condition_on_previous_text": True},
    "r1_asr_nospeech_lo":   {"transcription.no_speech_threshold": 0.3},
    "r1_asr_nospeech_hi":   {"transcription.no_speech_threshold": 0.8},
    "r1_asr_prompt_empty":  {"transcription.initial_prompt": ""},
    "r1_asr_prompt_rich":   {"transcription.initial_prompt":
                             "Nagranie rozmowy telefonicznej po polsku, dwie osoby."},
    # --- separation VAD ---
    "r1_vad_040":           {"separation.vad_threshold": 0.4, "separation.vad_soft_threshold": 0.15},
    "r1_vad_050":           {"separation.vad_threshold": 0.5, "separation.vad_soft_threshold": 0.2},
    "r1_vad_060":           {"separation.vad_threshold": 0.6, "separation.vad_soft_threshold": 0.25},
    "r1_vad_ar0":           {"separation.vad_attack_frames": 0, "separation.vad_release_frames": 0},
    "r1_vad_ar2":           {"separation.vad_attack_frames": 2, "separation.vad_release_frames": 2},
    # --- separation context window ---
    "r1_ctx_fixedpad":      {"separation.context_window_mode": "fixed_pad"},
    "r1_ctx_fixedpad15":    {"separation.context_window_mode": "fixed_pad",
                             "separation.context_pad_seconds": 1.5},
    "r1_ctx_none":          {"separation.context_window_mode": "none"},
    # --- separation seam ---
    "r1_seam_zc":           {"separation.seam_mode": "zero_crossing"},
    "r1_seam_boundary":     {"separation.seam_mode": "overlap_boundary"},
    # --- separation volume norm ---
    "r1_vol_none":          {"separation.volume_normalization": "none"},
    # --- BWE (overlap regions) ---
    "r1_bwe_naive":         {"post_separation_processing.backend": "naive"},
    # --- routing ---
    "r1_merge_03":          {"routing.merge_gap": 0.3},
    "r1_merge_08":          {"routing.merge_gap": 0.8},
    # --- assembly ---
    "r1_no_rmsmatch":       {"assembly.overlap_rms_match_solo": False},
    "r1_perpiece_rms":      {"assembly.per_piece_rms_norm": True},
    # --- ablation corners (thesis table; not tuning) ---
    "r1_nosep":             {"separation.enabled": False},
    "r1_noenh":             {"enhancement.enabled": False},
    "r1_nosep_noenh":       {"separation.enabled": False, "enhancement.enabled": False},

    # ======================================================================
    # Round 2 (2026-06-14): combine Round-1 winning directions + leave-one-out.
    # Winning combo = enh OFF + large-v3-turbo + bwe naive + rms-match OFF + beam 10.
    # ======================================================================
    "r2_best_naive": {"enhancement.enabled": False, "transcription.model_name": "large-v3-turbo",
                      "post_separation_processing.backend": "naive",
                      "assembly.overlap_rms_match_solo": False, "transcription.beam_size": 10},
    "r2_best_v3":    {"enhancement.enabled": False, "transcription.model_name": "large-v3",
                      "post_separation_processing.backend": "naive",
                      "assembly.overlap_rms_match_solo": False, "transcription.beam_size": 10},
    # leave-one-out from r2_best_naive (each restores ONE knob to baseline)
    "r2_loo_enhon":  {"transcription.model_name": "large-v3-turbo",
                      "post_separation_processing.backend": "naive",
                      "assembly.overlap_rms_match_solo": False, "transcription.beam_size": 10},
    "r2_loo_v2":     {"enhancement.enabled": False,
                      "post_separation_processing.backend": "naive",
                      "assembly.overlap_rms_match_solo": False, "transcription.beam_size": 10},
    "r2_loo_apbwe":  {"enhancement.enabled": False, "transcription.model_name": "large-v3-turbo",
                      "assembly.overlap_rms_match_solo": False, "transcription.beam_size": 10},
    "r2_loo_rmson":  {"enhancement.enabled": False, "transcription.model_name": "large-v3-turbo",
                      "post_separation_processing.backend": "naive", "transcription.beam_size": 10},
    "r2_loo_beam5":  {"enhancement.enabled": False, "transcription.model_name": "large-v3-turbo",
                      "post_separation_processing.backend": "naive",
                      "assembly.overlap_rms_match_solo": False},
    # the one enhancement backend not yet tried OFAT (confirm it's also >= none)
    "r2_enh_mossformer_gan": {"enhancement.backend": "mossformer_gan_se_16k"},

    # ======================================================================
    # Round 3 (2026-06-14): Round-2 LOO said the per-knob optima STACK to
    # {enh off, model large-v2, bwe naive, rms-match ON, beam 5} = baseline + just
    # {enh off, bwe naive}. r3_best tests that prediction; the rest re-confirm
    # each axis AT this operating point (model & rms & beam interacted with enh-off
    # in R2, so OFAT winners must be re-validated here), + the separation ablation.
    # ======================================================================
    "r3_best":            {"enhancement.enabled": False,
                           "post_separation_processing.backend": "naive"},
    "r3_best_turbo":      {"enhancement.enabled": False,
                           "post_separation_processing.backend": "naive",
                           "transcription.model_name": "large-v3-turbo"},
    "r3_best_v3":         {"enhancement.enabled": False,
                           "post_separation_processing.backend": "naive",
                           "transcription.model_name": "large-v3"},
    "r3_best_rmsoff":     {"enhancement.enabled": False,
                           "post_separation_processing.backend": "naive",
                           "assembly.overlap_rms_match_solo": False},
    "r3_best_beam10":     {"enhancement.enabled": False,
                           "post_separation_processing.backend": "naive",
                           "transcription.beam_size": 10},
    # thesis ablation: separation's value at the best config
    "r3_best_nosep":      {"enhancement.enabled": False,
                           "post_separation_processing.backend": "naive",
                           "separation.enabled": False},

    # ======================================================================
    # Round 4 (2026-06-14): the one missing ablation cell — enhancement ON at
    # the best operating point (naive BWE, v2/beam5/rms-on/sep). Completes the
    # enhancement × separation 2x2 for the thesis table. Sweep has converged;
    # this is the only remaining run.
    # ======================================================================
    "r4_enhon":           {"post_separation_processing.backend": "naive"},

    # ======================================================================
    # Anti-hallucination decode knobs (2026-06-14): on the BASELINE config
    # (enh on + ap_bwe), which triggers BOTH catastrophes (db15fc57 via FRCRN
    # enhancement 107.8; 94a0d89a via ap_bwe 56.6). Q: can faster-whisper's
    # repetition/hallucination guards fix the blowups without collateral, so
    # enhancement+ap_bwe become WER-safe? Gate on catastrophe frags first.
    # ======================================================================
    "ah_nrng2":      {"transcription.no_repeat_ngram_size": 2},
    "ah_nrng3":      {"transcription.no_repeat_ngram_size": 3},
    "ah_reppen115":  {"transcription.repetition_penalty": 1.15},
    "ah_hall2":      {"transcription.hallucination_silence_threshold": 2.0},
    "ah_combo":      {"transcription.no_repeat_ngram_size": 3,
                      "transcription.repetition_penalty": 1.15,
                      "transcription.hallucination_silence_threshold": 2.0},
    # Full sweep with the gate winner (no_repeat_ngram_size). ah_nrng3/ah_nrng2
    # above = baseline + nrng (the "rescue enh+ap_bwe" test). These add nrng3 to
    # the other operating points: the finalist, ap_bwe-only, and flowhigh.
    "ah_finalist_nrng3": {"enhancement.enabled": False,
                          "post_separation_processing.backend": "naive",
                          "transcription.no_repeat_ngram_size": 3},
    "ah_apbwe_nrng3":    {"enhancement.enabled": False,
                          "transcription.no_repeat_ngram_size": 3},

    # ======================================================================
    # Enhancement-model comparison (point 3, 2026-06-14): which SE backend gives
    # the best OUTPUT AUDIO (SQUIM) at least WER cost? All: enh ON + naive BWE +
    # nrng3 (guard removes the repetition-catastrophe confound). Score WER + SQUIM.
    # ======================================================================
    "c_frcrn_nrng3":   {"enhancement.backend": "frcrn_se_16k",
                        "post_separation_processing.backend": "naive",
                        "transcription.no_repeat_ngram_size": 3},
    "c_mossgan_nrng3": {"enhancement.backend": "mossformer_gan_se_16k",
                        "post_separation_processing.backend": "naive",
                        "transcription.no_repeat_ngram_size": 3},

    # ======================================================================
    # Observation Adding (OA) / dry-wet mix (2026-06-14, Iwamoto et al. 2022 /
    # Wang et al. 2024): re-introduce the observed signal into the enhanced
    # solo regions to dilute SE artifacts that hurt ASR. Base = c_frcrn_nrng3
    # (enh frcrn + naive BWE + nrng3 = the enhancement-on dual-objective ref);
    # c_frcrn_nrng3 itself is the ratio-0 reference. Score WER + SQUIM to trace
    # the WER↔quality frontier as r climbs.
    # ======================================================================
    "oa_frcrn_03":     {"enhancement.backend": "frcrn_se_16k",
                        "post_separation_processing.backend": "naive",
                        "transcription.no_repeat_ngram_size": 3,
                        "enhancement.observation_mix_ratio": 0.3},
    "oa_frcrn_05":     {"enhancement.backend": "frcrn_se_16k",
                        "post_separation_processing.backend": "naive",
                        "transcription.no_repeat_ngram_size": 3,
                        "enhancement.observation_mix_ratio": 0.5},
    "oa_frcrn_07":     {"enhancement.backend": "frcrn_se_16k",
                        "post_separation_processing.backend": "naive",
                        "transcription.no_repeat_ngram_size": 3,
                        "enhancement.observation_mix_ratio": 0.7},
    # OA on MossFormerGAN too (2026-06-14) — confirm the OA effect isn't a
    # single-backend fluke. Base = c_mossgan_nrng3 (ratio 0 reference).
    "oa_mossgan_03":   {"enhancement.backend": "mossformer_gan_se_16k",
                        "post_separation_processing.backend": "naive",
                        "transcription.no_repeat_ngram_size": 3,
                        "enhancement.observation_mix_ratio": 0.3},
    "oa_mossgan_05":   {"enhancement.backend": "mossformer_gan_se_16k",
                        "post_separation_processing.backend": "naive",
                        "transcription.no_repeat_ngram_size": 3,
                        "enhancement.observation_mix_ratio": 0.5},
    "oa_mossgan_07":   {"enhancement.backend": "mossformer_gan_se_16k",
                        "post_separation_processing.backend": "naive",
                        "transcription.no_repeat_ngram_size": 3,
                        "enhancement.observation_mix_ratio": 0.7},

    # ======================================================================
    # WhisperX chunk_size sweep (2026-06-14): db15fc57's headline failure is a
    # transcription dropout — WhisperX merged 39-68s into one ~30s VAD segment
    # and Whisper collapsed (emitted 1 word for ~20s of clear speech). Lowering
    # chunk_size forces a split and recovers it (verified ad-hoc: cs15/cs8 fix it,
    # cs30 doesn't). Q: does bounding chunk_size fix db15fc57 WITHOUT collateral?
    #   OA base (oa_frcrn_05) = where the dropout occurs (OA fills inter-utterance
    #     gaps → over-merge). cs30 = oa_frcrn_05 (already on disk).
    #   r3_best base = clean finalist (enh off + naive); collateral / default-safety
    #     check on a config with no OA. cs30 = r3_best (already on disk).
    # ======================================================================
    "oa_cs15": {"enhancement.backend": "frcrn_se_16k",
                "post_separation_processing.backend": "naive",
                "transcription.no_repeat_ngram_size": 3,
                "enhancement.observation_mix_ratio": 0.5,
                "transcription.chunk_size": 15},
    "oa_cs08": {"enhancement.backend": "frcrn_se_16k",
                "post_separation_processing.backend": "naive",
                "transcription.no_repeat_ngram_size": 3,
                "enhancement.observation_mix_ratio": 0.5,
                "transcription.chunk_size": 8},
    # r3_best base (clean finalist, enh off + naive, no OA): global-default
    # safety check — does lowering chunk_size hurt the WER-optimal config, and
    # does any non-OA fragment also have the 30s over-merge dropout?
    "r3_cs15": {"enhancement.enabled": False,
                "post_separation_processing.backend": "naive",
                "transcription.chunk_size": 15},
    "r3_cs08": {"enhancement.enabled": False,
                "post_separation_processing.backend": "naive",
                "transcription.chunk_size": 8},

    # ======================================================================
    # CLEAN DOCUMENTABLE SWEEP (2026-06-15, e31 separator). One coherent OFAT
    # off the default.yaml baseline (enh frcrn + ap_bwe + retry8 + large-v2) +
    # ablation corners + the two ship endpoints. Pure single-knob configs for
    # the knobs the older r1_* set predates (OA, retry-off, chunk_size, zip).
    # See GROUPS["final"]. (Replaces the erratic r1–r4/ah/c/oa rounds for the
    # record — those stay defined for reproducibility.)
    # ======================================================================
    "f_oa03":     {"enhancement.observation_mix_ratio": 0.3},
    "f_oa05":     {"enhancement.observation_mix_ratio": 0.5},
    "f_oa07":     {"enhancement.observation_mix_ratio": 0.7},
    "f_noretry":  {"transcription.retry_collapsed_chunk_size": 0},
    "f_cs15":     {"transcription.chunk_size": 15},

    # ======================================================================
    # CLEAN SWEEP — ROUND 2 (2026-06-15, e31): combos of the round-1 OFAT
    # winners, anchored on the significant OA-0.3 win. Pre-specified (no further
    # re-anchoring): does stacking the other positive directions (large-v3,
    # nrng3, seam-boundary) beat plain OA-0.3? + refine the OA ratio (0.2/0.4).
    # BWE stays ap_bwe (naive was WORSE than ap_bwe on e31).
    # ======================================================================
    "g_oa03_v3":       {"enhancement.observation_mix_ratio": 0.3,
                        "transcription.model_name": "large-v3"},
    "g_oa03_nrng3":    {"enhancement.observation_mix_ratio": 0.3,
                        "transcription.no_repeat_ngram_size": 3},
    "g_oa03_seamb":    {"enhancement.observation_mix_ratio": 0.3,
                        "separation.seam_mode": "overlap_boundary"},
    "g_oa03_beam10":   {"enhancement.observation_mix_ratio": 0.3,
                        "transcription.beam_size": 10},
    "g_oa03_v3_nrng3": {"enhancement.observation_mix_ratio": 0.3,
                        "transcription.model_name": "large-v3",
                        "transcription.no_repeat_ngram_size": 3},
    "g_oa03_v3_seamb": {"enhancement.observation_mix_ratio": 0.3,
                        "transcription.model_name": "large-v3",
                        "separation.seam_mode": "overlap_boundary"},
    "f_oa02":          {"enhancement.observation_mix_ratio": 0.2},
    "f_oa04":          {"enhancement.observation_mix_ratio": 0.4},

    # ======================================================================
    # Tier-2 "easy-win" levers (2026-06-15). Verified-easy-wins audit. Each is
    # a NEW sweep-ready knob whose committed default is a NO-OP, anchored on the
    # OA-0.3 finalist (enhancement.observation_mix_ratio: 0.3 = f_oa03) so they
    # are testable against it. OFAT off that anchor (one new lever each), except
    # the model swap (a finetune, no anchor knob needed). See GROUPS["t2"].
    # ======================================================================
    # WhisperX suppress_numerals: spell numbers out — GT is word-form Polish, so
    # digit hypotheses score as substitutions.
    "t2_suppress_numerals": {"enhancement.observation_mix_ratio": 0.3,
                             "transcription.suppress_numerals": True},
    # Length penalty 1.1 (mild long-hypothesis preference).
    "t2_length_penalty11": {"enhancement.observation_mix_ratio": 0.3,
                            "transcription.length_penalty": 1.1},
    # Lower WhisperX VAD offset: keep trailing speech the VAD would clip.
    "t2_vad_offset_lo":    {"enhancement.observation_mix_ratio": 0.3,
                            "transcription.vad_offset": 0.20},
    # bardsai Polish Whisper-large-v2 finetune (transformers format; the
    # pipeline's _ensure_ct2_model auto-converts it to CT2 on first use).
    "t2_bardsai_pl":       {"enhancement.observation_mix_ratio": 0.3,
                            "transcription.model_name":
                            "bardsai/whisper-large-v2-pl-v2"},

    # ======================================================================
    # Phase 3: DIARIZATION front-end knobs (pyannote instantiate), OFAT off
    # f_oa03. These change diarization → full pipeline re-run per config.
    # Effective under num_speakers=2: segmentation.min_duration_off (fill intra-
    # turn pauses → longer, cleaner embedding spans) and clustering.min_cluster_size.
    # clustering.threshold is INERT (fixed cluster count), so not swept. Diagnosis
    # (DIAGNOSIS_db15fc57) says these likely can't fix embedding-clustering
    # mislabels — run to confirm; the one shot is fe65d170's un-routed overlap.
    # See GROUPS["diar"].
    # ======================================================================
    "dr_mdoff03": {"enhancement.observation_mix_ratio": 0.3,
                   "diarization.segmentation_min_duration_off": 0.3},
    "dr_mdoff05": {"enhancement.observation_mix_ratio": 0.3,
                   "diarization.segmentation_min_duration_off": 0.5},
    "dr_mcs06":   {"enhancement.observation_mix_ratio": 0.3,
                   "diarization.clustering_min_cluster_size": 6},

    # ======================================================================
    # Phase 4: speaker-EMBEDDING swap (EMBEDDING_RESEARCH.md), OFAT off f_oa03.
    # Reconstructs a 3.1-equivalent pipeline with a swapped embedder → full
    # re-run per config. dr_emb_r34 is the reconstruction BASELINE (same
    # construction as the swaps, stock resnet34-LM embedder): compare swaps to
    # THIS, and confirm dr_emb_r34 ≈ f_oa03 (reconstruction reproduces 3.1).
    # 293-LM is tuned for >3s utts and may NOT help the ~2.5s db15fc57 mislabel
    # — hence also the non-LM variant + ERes2NetV2 (added once ids/wrapper ready).
    # See GROUPS["emb"]. Score on ALL dev (watch for regressions elsewhere).
    # ======================================================================
    "dr_emb_r34":    {"enhancement.observation_mix_ratio": 0.3,
                      "diarization.embedding": "pyannote/wespeaker-voxceleb-resnet34-LM"},
    "dr_emb_r293lm": {"enhancement.observation_mix_ratio": 0.3,
                      "diarization.embedding": "eek/wespeaker-voxceleb-resnet293-LM"},
    # Custom embedder (stages/custom_embeddings.py wrapper). ECAPA2 = best
    # short-utterance EER (the on-target lever for db15fc57's ~2.5s mislabel;
    # CC-BY-NC, fine for the thesis).
    "dr_emb_ecapa2": {"enhancement.observation_mix_ratio": 0.3,
                      "diarization.embedding": "ecapa2"},

    # ======================================================================
    # 2nd-pass identity re-clustering + assembly embedder swap
    # (SECOND_PASS_PLAN.md options 4, B, B+). All OFAT off the ADOPTED ECAPA2
    # diarization (`dr_emb_ecapa2`) — i.e. each row carries the same OA-0.3 +
    # ECAPA2-on-raw diarization base (`_E2`) plus its one new lever, so each
    # isolates its marginal effect over the embedder swap the diagnosis says does
    # NOT by itself fix db15fc57. Score on ALL dev; dr_emb_ecapa2 is the anchor
    # in the rescore (GROUPS["refine2"]).
    # ======================================================================
    # Base shared by all rows below (the adopted ECAPA2 diarization).
    # Inlined per row (CONFIGS values are flat dicts), matching dr_emb_ecapa2.
    # Option 4 — ECAPA2 assembly anchor only, no 2nd pass (the cheapest probe:
    # does a stronger embedder in the identity-deciding role help at all?).
    "as_ecapa2anchor": {"enhancement.observation_mix_ratio": 0.3,
                        "diarization.embedding": "ecapa2",
                        "assembly.anchor_embedding": "ecapa2"},
    # B — solos relabel on ENHANCED audio (the hypothesis: cleaner identity
    # audio re-clusters the db15fc57 solo mislabel into the right speaker).
    "dr_refine":      {"enhancement.observation_mix_ratio": 0.3,
                       "diarization.embedding": "ecapa2",
                       "relabel.enabled": True, "relabel.source": "solos",
                       "relabel.embedding": "ecapa2",
                       "relabel.audio_source": "enhanced"},
    # B control — solos relabel on RAW audio. Isolates the enhancement effect:
    # if dr_refine beats this, enhanced audio is the active ingredient, not the
    # re-clustering alone.
    "dr_refine_raw":  {"enhancement.observation_mix_ratio": 0.3,
                       "diarization.embedding": "ecapa2",
                       "relabel.enabled": True, "relabel.source": "solos",
                       "relabel.embedding": "ecapa2",
                       "relabel.audio_source": "raw"},
    # B+ — global identity clustering (enhanced solos + separated overlap
    # streams); also feeds assembly the per-overlap pairing, so it can move the
    # overlap half of db15fc57's gap that plain B cannot reach.
    "dr_refineplus":  {"enhancement.observation_mix_ratio": 0.3,
                       "diarization.embedding": "ecapa2",
                       "relabel.enabled": True, "relabel.source": "global",
                       "relabel.embedding": "ecapa2",
                       "relabel.audio_source": "enhanced"},
    # Stacked best-candidate: the three small winners combined — ECAPA2 diar +
    # ECAPA2 assembly anchor (opt 4) + B+ global relabel (dr_refineplus). Tests
    # whether the wins stack or B+'s overlap handoff already subsumes the anchor.
    "dr_best":        {"enhancement.observation_mix_ratio": 0.3,
                       "diarization.embedding": "ecapa2",
                       "assembly.anchor_embedding": "ecapa2",
                       "relabel.enabled": True, "relabel.source": "global",
                       "relabel.embedding": "ecapa2",
                       "relabel.audio_source": "enhanced"},
    # ASR-model re-check on the best base (dr_refineplus = ecapa2-diar + B+). The
    # recognition floor (~18.6 of 20.7 cpWER) dominates the error and the ASR
    # model is the biggest lever — both untested on the new diarization base.
    # large-v3 = bigger general model; bardsai = Polish-finetuned large-v2.
    "dr_refineplus_v3": {"enhancement.observation_mix_ratio": 0.3,
                         "diarization.embedding": "ecapa2",
                         "relabel.enabled": True, "relabel.source": "global",
                         "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced",
                         "transcription.model_name": "large-v3"},
    "dr_refineplus_pl": {"enhancement.observation_mix_ratio": 0.3,
                         "diarization.embedding": "ecapa2",
                         "relabel.enabled": True, "relabel.source": "global",
                         "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced",
                         "transcription.model_name": "bardsai/whisper-large-v2-pl-v2"},

    # ======================================================================
    # DEFINITIVE SWEEP arms (docs/sweep_plan/SWEEP_DESIGN.md §3.2, pre-flight
    # §6.4). All anchored on the `dr_refineplus` knob set (= the fixed anchor):
    # OA-0.3 + ECAPA2 diar + B+ global relabel (enhanced). Each row is a flat
    # dotted dict that takes the dr_refineplus base and changes ONE thing (or one
    # interaction corner), so a CONFIGS diff vs `dr_refineplus` reads as the arm's
    # delta. See GROUPS["definitive"]. The Tier-A grids (OA refine, BWE toggle)
    # are re-anchored HERE on the post-B+ base, NOT the stale pre-B+ f_oa0*/r1_*
    # rows (04b §0 wedge: the old "within-noise" verdicts were measured 3
    # operating-point hops upstream).
    #
    # Shared dr_refineplus base, inlined per row (CONFIGS values are flat dicts):
    #   enhancement.observation_mix_ratio: 0.3
    #   diarization.embedding: ecapa2
    #   relabel.enabled: True, source: global, embedding: ecapa2,
    #   audio_source: enhanced
    # ----------------------------------------------------------------------
    # A1 — OA-ratio refine grid OFF the anchor (anchor itself = OA 0.3 = dr_refineplus).
    "dr_oa020": {"enhancement.observation_mix_ratio": 0.20,
                 "diarization.embedding": "ecapa2",
                 "relabel.enabled": True, "relabel.source": "global",
                 "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced"},
    "dr_oa025": {"enhancement.observation_mix_ratio": 0.25,
                 "diarization.embedding": "ecapa2",
                 "relabel.enabled": True, "relabel.source": "global",
                 "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced"},
    "dr_oa035": {"enhancement.observation_mix_ratio": 0.35,
                 "diarization.embedding": "ecapa2",
                 "relabel.enabled": True, "relabel.source": "global",
                 "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced"},
    "dr_oa040": {"enhancement.observation_mix_ratio": 0.40,
                 "diarization.embedding": "ecapa2",
                 "relabel.enabled": True, "relabel.source": "global",
                 "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced"},
    # OA=0.5 confirmatory bracket at the REAL (post-B+) base (04b §2.4 insurance):
    # the design grid stops at 0.4, but 0.5 was unbracketed at the dr_refineplus
    # base. One arm; cheap; not a candidate, just bracket-checking the optimum.
    "dr_oa050": {"enhancement.observation_mix_ratio": 0.50,
                 "diarization.embedding": "ecapa2",
                 "relabel.enabled": True, "relabel.source": "global",
                 "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced"},
    # A3 / Tier-B — BWE toggle (ap_bwe=anchor) re-confirmed at the post-B+ base.
    # Also the (OA0.3, naive) corner of the A4 OA×BWE cell.
    "dr_bwe_naive": {"enhancement.observation_mix_ratio": 0.3,
                     "diarization.embedding": "ecapa2",
                     "relabel.enabled": True, "relabel.source": "global",
                     "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced",
                     "post_separation_processing.backend": "naive"},
    # A4 OA×BWE 2×2: corners are (OA0.3,ap_bwe)=anchor, (OA0.3,naive)=dr_bwe_naive,
    # (OA0.2,ap_bwe)=dr_oa020, (OA0.2,naive)=this row.
    "dr_oa02_naive": {"enhancement.observation_mix_ratio": 0.20,
                      "diarization.embedding": "ecapa2",
                      "relabel.enabled": True, "relabel.source": "global",
                      "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced",
                      "post_separation_processing.backend": "naive"},
    # A5 retry×OA 2×2: corners are (retry8,OA0.3)=anchor, (retry0,OA0.3),
    # (retry8,OA0.0), (retry0,OA0.0). retry8 = the shipped default; retry0 = OFF.
    "dr_retry0": {"enhancement.observation_mix_ratio": 0.3,
                  "diarization.embedding": "ecapa2",
                  "relabel.enabled": True, "relabel.source": "global",
                  "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced",
                  "transcription.retry_collapsed_chunk_size": 0},
    "dr_oa00": {"enhancement.observation_mix_ratio": 0.0,
                "diarization.embedding": "ecapa2",
                "relabel.enabled": True, "relabel.source": "global",
                "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced"},
    "dr_retry0_oa00": {"enhancement.observation_mix_ratio": 0.0,
                       "diarization.embedding": "ecapa2",
                       "relabel.enabled": True, "relabel.source": "global",
                       "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced",
                       "transcription.retry_collapsed_chunk_size": 0},
    # A6 enh×relabel 2×2: corners are (enhON,B+on)=anchor, (enhON,B+off)=
    # dr_emb_ecapa2 (existing), (enhOFF,B+on)=dr_enhoff, (enhOFF,B+off)=
    # dr_enhoff_norelabel. NOTE: with enhancement OFF there is no enhanced_full,
    # so config.py:765 forbids relabel.audio_source='enhanced'; the enh-OFF B+
    # corner therefore uses audio_source='raw' (the only valid B+ form). This is
    # the A6 enh-OFF-ceiling probe (SWEEP_DESIGN §3 Phase-0 / 04b enh-OFF gate).
    "dr_enhoff": {"enhancement.observation_mix_ratio": 0.3,
                  "enhancement.enabled": False,
                  "diarization.embedding": "ecapa2",
                  "relabel.enabled": True, "relabel.source": "global",
                  "relabel.embedding": "ecapa2", "relabel.audio_source": "raw"},
    "dr_enhoff_norelabel": {"enhancement.observation_mix_ratio": 0.3,
                            "enhancement.enabled": False,
                            "diarization.embedding": "ecapa2",
                            "relabel.enabled": False},
    # Tier-B near-Pareto re-confirm (mossformer_gan enhancer) at the anchor base.
    "dr_enh_mossgan": {"enhancement.observation_mix_ratio": 0.3,
                       "enhancement.backend": "mossformer_gan_se_16k",
                       "diarization.embedding": "ecapa2",
                       "relabel.enabled": True, "relabel.source": "global",
                       "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced"},
    # large-v3 writeup-reference arm (closed door; for the table, not a candidate).
    # = dr_refineplus_v3 (already defined above); aliased here only via the GROUP.

    # ======================================================================
    # HOLES-BUNDLE arms (docs/sweep_plan/04_HOLES.md DECISION: EVERYTHING).
    # All anchored on the same `dr_refineplus` base as the definitive block
    # (OA-0.3 + ECAPA2 diar + B+ global relabel, enhanced), inlined per row.
    # Two classes:
    #   Class-1 (C1/C4/C5/C6 + A2 VAD grid): arms on existing/just-promoted
    #     knobs that the definitive block didn't yet cover.
    #   Class-2 (D1-D6): the six uninventoried hard-coded levers promoted to
    #     config fields in this change (04a_holes_uninventoried.md). Each grid
    #     point is one OFAT arm off the anchor.
    # ----------------------------------------------------------------------
    # --- Class-1 ---
    # A2 VAD threshold grid at the post-B+ base (the swept VAD pair; doc 03 §2).
    "dr_vad040": {"enhancement.observation_mix_ratio": 0.3,
                  "diarization.embedding": "ecapa2",
                  "relabel.enabled": True, "relabel.source": "global",
                  "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced",
                  "separation.vad_threshold": 0.4,
                  "separation.vad_soft_threshold": 0.15},
    "dr_vad050": {"enhancement.observation_mix_ratio": 0.3,
                  "diarization.embedding": "ecapa2",
                  "relabel.enabled": True, "relabel.source": "global",
                  "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced",
                  "separation.vad_threshold": 0.5,
                  "separation.vad_soft_threshold": 0.20},
    # C1 — VAD × BWE interaction corner (VAD050 crossed with naive BWE).
    "dr_vad050_naive": {"enhancement.observation_mix_ratio": 0.3,
                        "diarization.embedding": "ecapa2",
                        "relabel.enabled": True, "relabel.source": "global",
                        "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced",
                        "separation.vad_threshold": 0.5,
                        "separation.vad_soft_threshold": 0.20,
                        "post_separation_processing.backend": "naive"},
    # C5 — VAD attack/release toggle at the post-B+ base (a/r = same mask mech).
    "dr_vad_ar2": {"enhancement.observation_mix_ratio": 0.3,
                   "diarization.embedding": "ecapa2",
                   "relabel.enabled": True, "relabel.source": "global",
                   "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced",
                   "separation.vad_attack_frames": 2,
                   "separation.vad_release_frames": 2},
    # C4 — mossformer_gan enhancer re-confirmed at ITS OWN OA neighbourhood (0.4),
    # not frcrn's 0.3 (the dr_enh_mossgan row sat at the wrong dilution). This row
    # OVERRIDES the base OA 0.3 -> 0.4.
    "dr_enh_mossgan_oa040": {"enhancement.observation_mix_ratio": 0.40,
                             "enhancement.backend": "mossformer_gan_se_16k",
                             "diarization.embedding": "ecapa2",
                             "relabel.enabled": True, "relabel.source": "global",
                             "relabel.embedding": "ecapa2",
                             "relabel.audio_source": "enhanced"},
    # C6 — B+ over a resnet34-diar first pass (settles stack-vs-subsume vs the
    # ECAPA2-diar base). OVERRIDES the base diarization.embedding only; the
    # relabel embedder stays ecapa2.
    "dr_refineplus_r34": {"enhancement.observation_mix_ratio": 0.3,
                          "diarization.embedding": "pyannote/wespeaker-voxceleb-resnet34-LM",
                          "relabel.enabled": True, "relabel.source": "global",
                          "relabel.embedding": "ecapa2",
                          "relabel.audio_source": "enhanced"},
    # --- Class-2: promoted uninventoried levers D1-D6 (one OFAT arm per grid pt) ---
    # D1 transcription.silence_floor grid {0.0, 5e-4, 1e-3} (anchor = 1e-4).
    "dr_silfloor0": {"enhancement.observation_mix_ratio": 0.3,
                     "diarization.embedding": "ecapa2",
                     "relabel.enabled": True, "relabel.source": "global",
                     "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced",
                     "transcription.silence_floor": 0.0},
    "dr_silfloor5e4": {"enhancement.observation_mix_ratio": 0.3,
                       "diarization.embedding": "ecapa2",
                       "relabel.enabled": True, "relabel.source": "global",
                       "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced",
                       "transcription.silence_floor": 5e-4},
    "dr_silfloor1e3": {"enhancement.observation_mix_ratio": 0.3,
                       "diarization.embedding": "ecapa2",
                       "relabel.enabled": True, "relabel.source": "global",
                       "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced",
                       "transcription.silence_floor": 1e-3},
    # D2 enhancement.resample_quality grid {soxr_vhq, kaiser_best} (anchor = soxr_hq).
    "dr_resample_vhq": {"enhancement.observation_mix_ratio": 0.3,
                        "diarization.embedding": "ecapa2",
                        "relabel.enabled": True, "relabel.source": "global",
                        "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced",
                        "enhancement.resample_quality": "soxr_vhq"},
    "dr_resample_kaiser": {"enhancement.observation_mix_ratio": 0.3,
                           "diarization.embedding": "ecapa2",
                           "relabel.enabled": True, "relabel.source": "global",
                           "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced",
                           "enhancement.resample_quality": "kaiser_best"},
    # D3 separation.seam_silence_threshold grid {0.3, 0.7} (anchor = 0.5).
    "dr_seamsil03": {"enhancement.observation_mix_ratio": 0.3,
                     "diarization.embedding": "ecapa2",
                     "relabel.enabled": True, "relabel.source": "global",
                     "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced",
                     "separation.seam_silence_threshold": 0.3},
    "dr_seamsil07": {"enhancement.observation_mix_ratio": 0.3,
                     "diarization.embedding": "ecapa2",
                     "relabel.enabled": True, "relabel.source": "global",
                     "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced",
                     "separation.seam_silence_threshold": 0.7},
    # D4 assembly.overlap_min_duration_s grid {0.2, 0.35} (anchor = 0.1).
    "dr_ovmin02": {"enhancement.observation_mix_ratio": 0.3,
                   "diarization.embedding": "ecapa2",
                   "relabel.enabled": True, "relabel.source": "global",
                   "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced",
                   "assembly.overlap_min_duration_s": 0.2},
    "dr_ovmin035": {"enhancement.observation_mix_ratio": 0.3,
                    "diarization.embedding": "ecapa2",
                    "relabel.enabled": True, "relabel.source": "global",
                    "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced",
                    "assembly.overlap_min_duration_s": 0.35},
    # D5 assembly.anchor_min_duration_s grid {0.5, 1.0} (anchor = 0.25).
    "dr_anchmin05": {"enhancement.observation_mix_ratio": 0.3,
                     "diarization.embedding": "ecapa2",
                     "relabel.enabled": True, "relabel.source": "global",
                     "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced",
                     "assembly.anchor_min_duration_s": 0.5},
    "dr_anchmin10": {"enhancement.observation_mix_ratio": 0.3,
                     "diarization.embedding": "ecapa2",
                     "relabel.enabled": True, "relabel.source": "global",
                     "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced",
                     "assembly.anchor_min_duration_s": 1.0},
    # D6 diarization.clustering_method robustness ref {average} (anchor = centroid).
    "dr_linkage_avg": {"enhancement.observation_mix_ratio": 0.3,
                       "diarization.embedding": "ecapa2",
                       "relabel.enabled": True, "relabel.source": "global",
                       "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced",
                       "diarization.clustering_method": "average"},

    # ======================================================================
    # COMPREHENSIVE expansion (full-grid, 2026-06-22)
    # Every row built on the dr_refineplus base (OA-0.3 + ECAPA2 diar + B+
    # global relabel, enhanced); each adds/replaces only its named override(s).
    # Inlined per row in the same flat-dotted-dict style as the dr_* arms above.
    # See GROUPS["definitive"] (COMPREHENSIVE expansion sub-section).
    # ----------------------------------------------------------------------
    # OA-ratio finer grid (replaces observation_mix_ratio off the 0.3 anchor).
    "dr_oa010": {"enhancement.observation_mix_ratio": 0.10,
                 "diarization.embedding": "ecapa2",
                 "relabel.enabled": True, "relabel.source": "global",
                 "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced"},
    "dr_oa015": {"enhancement.observation_mix_ratio": 0.15,
                 "diarization.embedding": "ecapa2",
                 "relabel.enabled": True, "relabel.source": "global",
                 "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced"},
    "dr_oa045": {"enhancement.observation_mix_ratio": 0.45,
                 "diarization.embedding": "ecapa2",
                 "relabel.enabled": True, "relabel.source": "global",
                 "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced"},
    "dr_oa060": {"enhancement.observation_mix_ratio": 0.60,
                 "diarization.embedding": "ecapa2",
                 "relabel.enabled": True, "relabel.source": "global",
                 "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced"},
    "dr_oa070": {"enhancement.observation_mix_ratio": 0.70,
                 "diarization.embedding": "ecapa2",
                 "relabel.enabled": True, "relabel.source": "global",
                 "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced"},
    # VAD threshold finer grid (vad_threshold / vad_soft_threshold pair).
    "dr_vad020": {"enhancement.observation_mix_ratio": 0.3,
                  "diarization.embedding": "ecapa2",
                  "relabel.enabled": True, "relabel.source": "global",
                  "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced",
                  "separation.vad_threshold": 0.2,
                  "separation.vad_soft_threshold": 0.08},
    "dr_vad030": {"enhancement.observation_mix_ratio": 0.3,
                  "diarization.embedding": "ecapa2",
                  "relabel.enabled": True, "relabel.source": "global",
                  "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced",
                  "separation.vad_threshold": 0.3,
                  "separation.vad_soft_threshold": 0.12},
    "dr_vad060": {"enhancement.observation_mix_ratio": 0.3,
                  "diarization.embedding": "ecapa2",
                  "relabel.enabled": True, "relabel.source": "global",
                  "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced",
                  "separation.vad_threshold": 0.6,
                  "separation.vad_soft_threshold": 0.25},
    # VAD attack/release grid (vad_attack_frames / vad_release_frames pair).
    "dr_vad_ar0": {"enhancement.observation_mix_ratio": 0.3,
                   "diarization.embedding": "ecapa2",
                   "relabel.enabled": True, "relabel.source": "global",
                   "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced",
                   "separation.vad_attack_frames": 0,
                   "separation.vad_release_frames": 0},
    "dr_vad_ar3": {"enhancement.observation_mix_ratio": 0.3,
                   "diarization.embedding": "ecapa2",
                   "relabel.enabled": True, "relabel.source": "global",
                   "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced",
                   "separation.vad_attack_frames": 3,
                   "separation.vad_release_frames": 3},
    # seam_silence_threshold finer grid (anchor = 0.5).
    "dr_seamsil02": {"enhancement.observation_mix_ratio": 0.3,
                     "diarization.embedding": "ecapa2",
                     "relabel.enabled": True, "relabel.source": "global",
                     "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced",
                     "separation.seam_silence_threshold": 0.2},
    "dr_seamsil04": {"enhancement.observation_mix_ratio": 0.3,
                     "diarization.embedding": "ecapa2",
                     "relabel.enabled": True, "relabel.source": "global",
                     "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced",
                     "separation.seam_silence_threshold": 0.4},
    "dr_seamsil06": {"enhancement.observation_mix_ratio": 0.3,
                     "diarization.embedding": "ecapa2",
                     "relabel.enabled": True, "relabel.source": "global",
                     "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced",
                     "separation.seam_silence_threshold": 0.6},
    "dr_seamsil08": {"enhancement.observation_mix_ratio": 0.3,
                     "diarization.embedding": "ecapa2",
                     "relabel.enabled": True, "relabel.source": "global",
                     "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced",
                     "separation.seam_silence_threshold": 0.8},
    # transcription.silence_floor finer grid (anchor = 1e-4).
    "dr_silfloor2e4": {"enhancement.observation_mix_ratio": 0.3,
                       "diarization.embedding": "ecapa2",
                       "relabel.enabled": True, "relabel.source": "global",
                       "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced",
                       "transcription.silence_floor": 0.0002},
    "dr_silfloor2e3": {"enhancement.observation_mix_ratio": 0.3,
                       "diarization.embedding": "ecapa2",
                       "relabel.enabled": True, "relabel.source": "global",
                       "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced",
                       "transcription.silence_floor": 0.002},
    # assembly.overlap_min_duration_s finer grid (anchor = 0.1).
    "dr_ovmin015": {"enhancement.observation_mix_ratio": 0.3,
                    "diarization.embedding": "ecapa2",
                    "relabel.enabled": True, "relabel.source": "global",
                    "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced",
                    "assembly.overlap_min_duration_s": 0.15},
    "dr_ovmin03": {"enhancement.observation_mix_ratio": 0.3,
                   "diarization.embedding": "ecapa2",
                   "relabel.enabled": True, "relabel.source": "global",
                   "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced",
                   "assembly.overlap_min_duration_s": 0.3},
    "dr_ovmin05": {"enhancement.observation_mix_ratio": 0.3,
                   "diarization.embedding": "ecapa2",
                   "relabel.enabled": True, "relabel.source": "global",
                   "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced",
                   "assembly.overlap_min_duration_s": 0.5},
    # assembly.anchor_min_duration_s finer grid (anchor = 0.25).
    "dr_anchmin04": {"enhancement.observation_mix_ratio": 0.3,
                     "diarization.embedding": "ecapa2",
                     "relabel.enabled": True, "relabel.source": "global",
                     "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced",
                     "assembly.anchor_min_duration_s": 0.4},
    "dr_anchmin075": {"enhancement.observation_mix_ratio": 0.3,
                      "diarization.embedding": "ecapa2",
                      "relabel.enabled": True, "relabel.source": "global",
                      "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced",
                      "assembly.anchor_min_duration_s": 0.75},
    "dr_anchmin15": {"enhancement.observation_mix_ratio": 0.3,
                     "diarization.embedding": "ecapa2",
                     "relabel.enabled": True, "relabel.source": "global",
                     "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced",
                     "assembly.anchor_min_duration_s": 1.5},
    # diarization.clustering_method linkage refs (anchor = centroid).
    "dr_linkage_complete": {"enhancement.observation_mix_ratio": 0.3,
                            "diarization.embedding": "ecapa2",
                            "relabel.enabled": True, "relabel.source": "global",
                            "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced",
                            "diarization.clustering_method": "complete"},
    "dr_linkage_ward": {"enhancement.observation_mix_ratio": 0.3,
                        "diarization.embedding": "ecapa2",
                        "relabel.enabled": True, "relabel.source": "global",
                        "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced",
                        "diarization.clustering_method": "ward"},
    # ASR — Cohere Transcribe (coherex backend; isolated venv at run time).
    "dr_cohere": {"enhancement.observation_mix_ratio": 0.3,
                  "diarization.embedding": "ecapa2",
                  "relabel.enabled": True, "relabel.source": "global",
                  "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced",
                  "transcription.backend": "coherex",
                  "transcription.model_name": "CohereLabs/cohere-transcribe-03-2026"},
    # re-confirm @ e46/B+ base — context / routing / ASR-decode knobs OFAT.
    "dr_ctx_fixedpad": {"enhancement.observation_mix_ratio": 0.3,
                        "diarization.embedding": "ecapa2",
                        "relabel.enabled": True, "relabel.source": "global",
                        "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced",
                        "separation.context_window_mode": "fixed_pad"},
    "dr_ctx_fixedpad15": {"enhancement.observation_mix_ratio": 0.3,
                          "diarization.embedding": "ecapa2",
                          "relabel.enabled": True, "relabel.source": "global",
                          "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced",
                          "separation.context_window_mode": "fixed_pad",
                          "separation.context_pad_seconds": 1.5},
    "dr_ctx_none": {"enhancement.observation_mix_ratio": 0.3,
                    "diarization.embedding": "ecapa2",
                    "relabel.enabled": True, "relabel.source": "global",
                    "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced",
                    "separation.context_window_mode": "none"},
    "dr_merge03": {"enhancement.observation_mix_ratio": 0.3,
                   "diarization.embedding": "ecapa2",
                   "relabel.enabled": True, "relabel.source": "global",
                   "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced",
                   "routing.merge_gap": 0.3},
    "dr_merge08": {"enhancement.observation_mix_ratio": 0.3,
                   "diarization.embedding": "ecapa2",
                   "relabel.enabled": True, "relabel.source": "global",
                   "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced",
                   "routing.merge_gap": 0.8},
    "dr_align_xlsr1b": {"enhancement.observation_mix_ratio": 0.3,
                        "diarization.embedding": "ecapa2",
                        "relabel.enabled": True, "relabel.source": "global",
                        "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced",
                        "transcription.align_model_name": "jonatasgrosman/wav2vec2-xls-r-1b-polish"},
    "dr_cs15": {"enhancement.observation_mix_ratio": 0.3,
                "diarization.embedding": "ecapa2",
                "relabel.enabled": True, "relabel.source": "global",
                "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced",
                "transcription.chunk_size": 15},
    "dr_beam10": {"enhancement.observation_mix_ratio": 0.3,
                  "diarization.embedding": "ecapa2",
                  "relabel.enabled": True, "relabel.source": "global",
                  "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced",
                  "transcription.beam_size": 10},
    "dr_nospeech_hi": {"enhancement.observation_mix_ratio": 0.3,
                       "diarization.embedding": "ecapa2",
                       "relabel.enabled": True, "relabel.source": "global",
                       "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced",
                       "transcription.no_speech_threshold": 0.8},
    "dr_condprev": {"enhancement.observation_mix_ratio": 0.3,
                    "diarization.embedding": "ecapa2",
                    "relabel.enabled": True, "relabel.source": "global",
                    "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced",
                    "transcription.condition_on_previous_text": True},
    # Interaction corners — OA × BWE × enhancer × VAD crosses off the anchor.
    "dr_oa04_naive": {"enhancement.observation_mix_ratio": 0.40,
                      "diarization.embedding": "ecapa2",
                      "relabel.enabled": True, "relabel.source": "global",
                      "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced",
                      "post_separation_processing.backend": "naive"},

    # ======================================================================
    # PHASE 2 (2026-06-24): OA × large-v3 stack + OA-peak bracket. Phase-1
    # found the OA-blend (HIGH-stratum, Holm-significant) and large-v3 (best
    # ALL point estimate, FAILED Holm at OA-0.3) as the two strongest
    # INDEPENDENT signals — never crossed. Tests whether they stack robustly.
    # All off the dr_refineplus base. See GROUPS["phase2"].
    # ======================================================================
    "dr_oa045_v3": {"enhancement.observation_mix_ratio": 0.45,
                    "diarization.embedding": "ecapa2",
                    "relabel.enabled": True, "relabel.source": "global",
                    "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced",
                    "transcription.model_name": "large-v3"},
    "dr_oa050_v3": {"enhancement.observation_mix_ratio": 0.5,
                    "diarization.embedding": "ecapa2",
                    "relabel.enabled": True, "relabel.source": "global",
                    "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced",
                    "transcription.model_name": "large-v3"},
    "dr_oa070_v3": {"enhancement.observation_mix_ratio": 0.7,
                    "diarization.embedding": "ecapa2",
                    "relabel.enabled": True, "relabel.source": "global",
                    "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced",
                    "transcription.model_name": "large-v3"},
    "dr_oa075": {"enhancement.observation_mix_ratio": 0.75,
                 "diarization.embedding": "ecapa2",
                 "relabel.enabled": True, "relabel.source": "global",
                 "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced"},
    "dr_oa080": {"enhancement.observation_mix_ratio": 0.8,
                 "diarization.embedding": "ecapa2",
                 "relabel.enabled": True, "relabel.source": "global",
                 "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced"},
    "dr_oa050_v3_naive": {"enhancement.observation_mix_ratio": 0.5,
                          "diarization.embedding": "ecapa2",
                          "relabel.enabled": True, "relabel.source": "global",
                          "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced",
                          "transcription.model_name": "large-v3",
                          "post_separation_processing.backend": "naive"},

    # ======================================================================
    # CAMPAIGN v2 — attribution-repair levers (2026-07-02). All anchored on the
    # `dr_oa050` finalist knob set (OA-0.5 + ECAPA2 diar + B+ global relabel,
    # enhanced), inlined per row like the dr_* arms above; each adds only its new
    # attribution lever(s). Diagnosis: the concentrated cpWER tax is 89% class-A
    # chunk swaps (contiguous same-speaker solo runs filed to the wrong stream)
    # that the GLOBAL B+ relabel provably cannot repair — the repair is LOCAL.
    # See GROUPS["v2dev"]. Levers:
    #   v2_ngram3    - anti-hallucination guard (recognition-floor lever, not
    #                  attribution; the block's non-attribution control).
    #   v2_pad       - solo onset boundary pad (0.15 s; ear-pass repair for
    #                  shaved/split first phonemes at solo piece starts —
    #                  boundary recovery, not stream re-routing, so it is OFAT
    #                  here and deliberately NOT folded into v2_attr/v2_full,
    #                  which were already sweeping when it landed).
    # ----------------------------------------------------------------------
    "v2_ngram3": {"enhancement.observation_mix_ratio": 0.50,
                  "diarization.embedding": "ecapa2",
                  "relabel.enabled": True, "relabel.source": "global",
                  "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced",
                  "transcription.no_repeat_ngram_size": 3},
    "v2_pad": {"enhancement.observation_mix_ratio": 0.50,
               "diarization.embedding": "ecapa2",
               "relabel.enabled": True, "relabel.source": "global",
               "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced",
               "assembly.solo_onset_pad_s": 0.15},
    # ----------------------------------------------------------------------
    # Campaign v2, round 2 — the two levers that survived diagnosis
    # (see GROUPS["v2fix"]; anchor stays dr_oa050):
    #   v2_reseed    - relabel degeneracy rescue: when the pass-1-seeded solo
    #                  2-means lands on a duration-degenerate outlier-peel
    #                  (min-cluster share < 0.10), search all pair-seeded
    #                  Lloyd's fixed points and adopt the best BALANCED one.
    #                  Validated analytically (+342 tax words, 0 harmed of 42);
    #                  no dev fragment trips the trigger, so dev = do-no-harm.
    #   v2_loopretry - repetition-loop detect-and-retry: score windows with the
    #                  scanner metric, re-transcribe flagged windows with
    #                  no_repeat_ngram_size on the retry only. Replaces the
    #                  always-on v2_ngram3, whose dev collateral (+58 err across
    #                  16 frags) ate 61% of its loop win (94a0d89a 124->29).
    # ----------------------------------------------------------------------
    "v2_reseed": {"enhancement.observation_mix_ratio": 0.50,
                  "diarization.embedding": "ecapa2",
                  "relabel.enabled": True, "relabel.source": "global",
                  "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced",
                  "relabel.solo_clustering_init": "rescue"},
    "v2_loopretry": {"enhancement.observation_mix_ratio": 0.50,
                     "diarization.embedding": "ecapa2",
                     "relabel.enabled": True, "relabel.source": "global",
                     "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced",
                     "transcription.loop_retry": True},
    # Campaign v2 FINALIST — dr_oa050 + the two levers that survived round 2
    # (rescue: fires only on duration-degenerate solo partitions; loop_retry:
    # touches only detector-flagged windows, min-repeat gate 12). Pre-registered
    # for the ONE held-out test re-eval: docs/sweep_plan/V2_TEST_PREREG.md.
    "v2_finalist": {"enhancement.observation_mix_ratio": 0.50,
                    "diarization.embedding": "ecapa2",
                    "relabel.enabled": True, "relabel.source": "global",
                    "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced",
                    "relabel.solo_clustering_init": "rescue",
                    "transcription.loop_retry": True},
    # Campaign v3 — phrase-loop mirror of loop_retry: = v2_finalist + the
    # multi-token phrase-loop detect-and-retry (2 known test sites; see
    # docs/sweep_plan/V3_TEST_PREREG.md).
    "v3_phraseloop": {"enhancement.observation_mix_ratio": 0.50,
                      "diarization.embedding": "ecapa2",
                      "relabel.enabled": True, "relabel.source": "global",
                      "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced",
                      "relabel.solo_clustering_init": "rescue",
                      "transcription.loop_retry": True,
                      "transcription.loop_retry_phrase": True},
    # Campaign v3 — LINKAGE OFAT off the v3_phraseloop finalist. Each row copies
    # v3_phraseloop and swaps only the AgglomerativeClustering linkage
    # (diarization.clustering_method, default "centroid" = stock pyannote 3.1).
    # Applied via Pipeline.instantiate in the ECAPA2 embedding-swap path, where
    # linkage decides exactly where short-segment mislabels are decided. This is
    # the last exposed-but-never-swept diarizer knob. Expected NULL — the fused-
    # cluster failures are embedding-geometry-bound, not linkage-bound (see
    # _forensics/EMBEDDER_BAKEOFF.md); run to CLOSE the knob inventory with
    # numbers, not to hunt a win. See GROUPS["link"].
    "v3_link_ward": {"enhancement.observation_mix_ratio": 0.50,
                     "diarization.embedding": "ecapa2",
                     "relabel.enabled": True, "relabel.source": "global",
                     "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced",
                     "relabel.solo_clustering_init": "rescue",
                     "transcription.loop_retry": True,
                     "transcription.loop_retry_phrase": True,
                     "diarization.clustering_method": "ward"},
    "v3_link_avg": {"enhancement.observation_mix_ratio": 0.50,
                    "diarization.embedding": "ecapa2",
                    "relabel.enabled": True, "relabel.source": "global",
                    "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced",
                    "relabel.solo_clustering_init": "rescue",
                    "transcription.loop_retry": True,
                    "transcription.loop_retry_phrase": True,
                    "diarization.clustering_method": "average"},
    "v3_link_comp": {"enhancement.observation_mix_ratio": 0.50,
                     "diarization.embedding": "ecapa2",
                     "relabel.enabled": True, "relabel.source": "global",
                     "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced",
                     "relabel.solo_clustering_init": "rescue",
                     "transcription.loop_retry": True,
                     "transcription.loop_retry_phrase": True,
                     "diarization.clustering_method": "complete"},
    # Campaign v4 — EEND diarizer arm off the v3_phraseloop finalist. Swaps the
    # pyannote+ECAPA2 clustering diarizer for NVIDIA Sortformer-v1 offline EEND
    # (clustering-free), which the probe showed un-fuses the two speakers pyannote
    # physically merged on the fused fragments (docs/sweep_plan/eend_probe.py +
    # _forensics/EEND_PROBE.md: trio purity 0.790 -> 0.907). = v3_phraseloop MINUS
    # diarization.embedding (pyannote-only; ignored on the EEND path) PLUS
    # diarization.backend=sortformer. Needs $SORTFORMER_VENV_PY (isolated NeMo
    # venv). PREREG PENDING — the open question is whether the diarization-purity
    # gain survives the re-route + re-separate + relabel cascade into a cpWER win.
    "v4_eend": {"enhancement.observation_mix_ratio": 0.50,
                "diarization.backend": "sortformer",
                "relabel.enabled": True, "relabel.source": "global",
                "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced",
                "relabel.solo_clustering_init": "rescue",
                "transcription.loop_retry": True,
                "transcription.loop_retry_phrase": True},
    # Campaign v4.1 — Sortformer rehabilitation levers (docs/sweep_plan/V41_PREREG.md).
    # Base = v4_eend (its exact override set, copied per row); each arm turns on one
    # or more of the four config-gated, default-OFF levers:
    #   L1 merge-not-discard   (diarization.sortformer_head_policy=merge)
    # Needs $SORTFORMER_VENV_PY. See GROUPS["v41"].
    "v41_merge": {"enhancement.observation_mix_ratio": 0.50,
                  "diarization.backend": "sortformer",
                  "diarization.sortformer_head_policy": "merge",
                  "relabel.enabled": True, "relabel.source": "global",
                  "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced",
                  "relabel.solo_clustering_init": "rescue",
                  "transcription.loop_retry": True,
                  "transcription.loop_retry_phrase": True},
    # V5 Phase-2 ablation (V5_INSTRUMENT_PREREG.md §PHASE-2): the adopted
    # instrument minus its separation subsystem — the "does separation help?"
    # contrast under the v41_merge instrument (the historical `nosep` arm
    # ablates off default.yaml, a different base). The relabel block is
    # dropped too, of necessity: B+ relabel clusters the separated overlap
    # streams, so config validation (correctly) rejects it with separation
    # off. The ablation therefore removes separation AND its dependent
    # attribution refinements — the full separation-dependent chain.
    "v41_merge_nosep": {"enhancement.observation_mix_ratio": 0.50,
                        "diarization.backend": "sortformer",
                        "diarization.sortformer_head_policy": "merge",
                        "separation.enabled": False,
                        "transcription.loop_retry": True,
                        "transcription.loop_retry_phrase": True},
    # Campaign v5 (instrument reframe) — Sortformer v2.1 dev arms
    # (docs/sweep_plan/V5_INSTRUMENT_PREREG.md §"Phase 1"). Base = v4_eend's exact
    # override set (OA 0.5 + sortformer + relabel B+/global/ecapa2/enhanced/rescue
    # + both loop retries), copied per row, with the diarizer model swapped to the
    # streaming v2.1 checkpoint (NVIDIA Open Model License — the Life-2 commercial
    # unlock) run offline via the worker's very-high-latency preset. NO
    # diarization.embedding, NO fallback, NO hysteresis (author: one diarizer only;
    # hysteresis dev-falsified). Needs $SORTFORMER_VENV_PY. See GROUPS["v2p1"].
    #   v2p1_eend        — plain (top-2, flat 0.5): the raw v2.1 instrument.
    #   v2p1_merge       — + head_policy=merge (the fold): reads the fold on v2.1.
    #   v2p1_merge_th040 — + threshold=0.40: prices the overlap-starvation risk
    #                      (probe: v2.1 detected 6.9 s overlap vs v1's 26.3 s).
    "v2p1_eend": {"enhancement.observation_mix_ratio": 0.50,
                  "diarization.backend": "sortformer",
                  "diarization.sortformer_model_id":
                      "nvidia/diar_streaming_sortformer_4spk-v2.1",
                  "relabel.enabled": True, "relabel.source": "global",
                  "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced",
                  "relabel.solo_clustering_init": "rescue",
                  "transcription.loop_retry": True,
                  "transcription.loop_retry_phrase": True},
    "v2p1_merge": {"enhancement.observation_mix_ratio": 0.50,
                   "diarization.backend": "sortformer",
                   "diarization.sortformer_model_id":
                       "nvidia/diar_streaming_sortformer_4spk-v2.1",
                   "diarization.sortformer_head_policy": "merge",
                   "relabel.enabled": True, "relabel.source": "global",
                   "relabel.embedding": "ecapa2", "relabel.audio_source": "enhanced",
                   "relabel.solo_clustering_init": "rescue",
                   "transcription.loop_retry": True,
                   "transcription.loop_retry_phrase": True},
    "v2p1_merge_th040": {"enhancement.observation_mix_ratio": 0.50,
                         "diarization.backend": "sortformer",
                         "diarization.sortformer_model_id":
                             "nvidia/diar_streaming_sortformer_4spk-v2.1",
                         "diarization.sortformer_head_policy": "merge",
                         "diarization.sortformer_threshold": 0.40,
                         "relabel.enabled": True, "relabel.source": "global",
                         "relabel.embedding": "ecapa2",
                         "relabel.audio_source": "enhanced",
                         "relabel.solo_clustering_init": "rescue",
                         "transcription.loop_retry": True,
                         "transcription.loop_retry_phrase": True},
}

# Named groups for --groups selection. "baseline" is always included.
GROUPS: dict[str, list[str]] = {
    "asr":        ["asr_largev3"],
    "enhance":    ["enh_mossformer", "enh_frcrn", "enh_none"],
    "bwe":        ["bwe_naive"],
    "separation": ["sep_seam_zc", "sep_seam_boundary", "sep_vad_strict", "sep_vol_none"],
    "assembly":   ["asm_perpiece_rms"],
    "ablation":   ["nosep", "nosep_noenh"],
    # enh_frcrn (round-1 winner) included as the round-2 anchor — already run,
    # so it's skipped on run and just rescored alongside the new variants.
    "round2":     ["enh_frcrn", "frcrn_vad_strict", "frcrn_seam_zc",
                   "frcrn_vad_seam", "frcrn_largev3"],
    # frcrn_vad_strict (round-2 winner) included as the round-3 anchor.
    "round3":     ["frcrn_vad_strict", "frcrn_vad_040", "frcrn_vad_060",
                   "frcrn_vad_strict_ar0", "frcrn_vad_strict_ar2"],
    # MossFormer2-separator comprehensive sweep, Round 1 (OFAT off baseline).
    # "baseline" is auto-included by the runner.
    "r1": ["r1_asr_largev3", "r1_asr_largev3_turbo",
           "r1_asr_beam1", "r1_asr_beam10", "r1_asr_temp0", "r1_asr_cond_prev",
           "r1_asr_nospeech_lo", "r1_asr_nospeech_hi",
           "r1_asr_prompt_empty", "r1_asr_prompt_rich",
           "r1_vad_040", "r1_vad_050", "r1_vad_060", "r1_vad_ar0", "r1_vad_ar2",
           "r1_ctx_fixedpad", "r1_ctx_fixedpad15", "r1_ctx_none",
           "r1_seam_zc", "r1_seam_boundary", "r1_vol_none",
           "r1_bwe_naive",
           "r1_merge_03", "r1_merge_08",
           "r1_no_rmsmatch", "r1_perpiece_rms",
           "r1_nosep", "r1_noenh", "r1_nosep_noenh"],
    # Round 2: combined winners + leave-one-out + the last enhancement backend.
    # r1_noenh kept in for a same-table reference point.
    "r2": ["r2_best_naive", "r2_best_v3",
           "r2_loo_enhon", "r2_loo_v2", "r2_loo_apbwe", "r2_loo_rmson", "r2_loo_beam5",
           "r2_enh_mossformer_gan", "r1_noenh"],
    # Round 3: confirm the LOO-stacked prediction + re-validate each axis at that
    # operating point + separation ablation. Carry r2_loo_v2 (R2 best) as a ref.
    "r3": ["r3_best", "r3_best_turbo", "r3_best_v3", "r3_best_rmsoff",
           "r3_best_beam10", "r3_best_nosep",
           "r2_loo_v2", "r1_noenh"],
    # Round 4: final ablation cell (enh ON at the best operating point).
    "r4": ["r4_enhon"],
    # Anti-hallucination full sweep (no_repeat_ngram_size=3 at every operating
    # point) across all 23 dev. Refs (baseline/r3_best/r1_noenh/r3_best_flowhigh16)
    # rescored alongside. ah_nrng3/ah_nrng2 = baseline+nrng (already partial from gate).
    "ah_full": ["ah_nrng3", "ah_nrng2", "ah_finalist_nrng3", "ah_apbwe_nrng3",
                "r3_best", "r1_noenh", "r4_enhon"],
    # Enhancement-model comparison (WER + SQUIM). r3_best = enh-off ref.
    "c_enh": ["c_frcrn_nrng3", "c_mossgan_nrng3", "r3_best"],
    # Observation Adding sweep (WER + SQUIM). c_frcrn_nrng3 = ratio-0 reference.
    "oa": ["c_frcrn_nrng3", "oa_frcrn_03", "oa_frcrn_05", "oa_frcrn_07"],
    "oa_mossgan": ["c_mossgan_nrng3", "oa_mossgan_03", "oa_mossgan_05", "oa_mossgan_07"],
    # chunk_size sweep at cs15/cs8 on two bases: OA best config (oa_frcrn_05=cs30
    # ref) — does it fix the db15fc57 dropout w/o collateral; and r3_best (cs30
    # ref, clean finalist) — global-default safety + is the over-merge general?
    "chunk_size": ["oa_frcrn_05", "oa_cs15", "oa_cs08", "r3_best", "r3_cs15", "r3_cs08"],
    # The clean documentable sweep (e31). baseline auto-included. OFAT off the
    # default.yaml baseline + ablation corners + the two ship endpoints.
    "final": [
        # enhancement backend
        "r1_noenh", "enh_mossformer",
        # observation-adding (dry/wet on the frcrn baseline)
        "f_oa03", "f_oa05", "f_oa07",
        # bandwidth extension
        "r1_bwe_naive",
        # transcription: collapse-retry, chunk_size, anti-hall, model, decode
        "f_noretry", "f_cs15", "ah_nrng3",
        "r1_asr_largev3", "r1_asr_largev3_turbo",
        "r1_asr_beam1", "r1_asr_beam10", "r1_asr_temp0", "r1_asr_cond_prev",
        "r1_asr_nospeech_lo", "r1_asr_nospeech_hi",
        "r1_asr_prompt_empty", "r1_asr_prompt_rich",
        # separation: VAD, attack/release, context, seam, volume-norm
        "r1_vad_040", "r1_vad_050", "r1_vad_060", "r1_vad_ar0", "r1_vad_ar2",
        "r1_ctx_fixedpad", "r1_ctx_fixedpad15", "r1_ctx_none",
        "r1_seam_zc", "r1_seam_boundary", "r1_vol_none",
        # routing / assembly
        "r1_merge_03", "r1_merge_08", "r1_no_rmsmatch", "r1_perpiece_rms",
        # ablation corners
        "r1_nosep", "r1_nosep_noenh",
        # named endpoints (the two ship candidates)
        "r3_best", "oa_frcrn_05",
    ],
    # Round 2: OA-0.3-anchored combos + ratio refine (baseline + f_oa03 as refs).
    "final2": [
        "f_oa03", "f_oa02", "f_oa04",
        "g_oa03_v3", "g_oa03_nrng3", "g_oa03_seamb", "g_oa03_beam10",
        "g_oa03_v3_nrng3", "g_oa03_v3_seamb",
    ],
    # Tier-2 easy-win levers, OFAT off the OA-0.3 finalist (f_oa03 = the anchor
    # reference; baseline auto-included).
    "t2": [
        "f_oa03",
        "t2_suppress_numerals", "t2_length_penalty11", "t2_vad_offset_lo",
        "t2_bardsai_pl",
    ],
    # Attribution: continuity tie-break tau grid, OFAT off f_oa03 (the no-op
    # anchor ref; baseline auto-included).
    "attr": [
        "f_oa03",
    ],
    # Phase 3 diarization front-end grid, OFAT off f_oa03 (the anchor ref).
    "diar": [
        "f_oa03",
        "dr_mdoff03", "dr_mdoff05", "dr_mcs06",
    ],
    # Phase 4 embedding swap. dr_emb_r34 = reconstruction baseline (compare swaps
    # to it). 293-non-LM + ERes2NetV2 appended once ids/wrapper are confirmed.
    "emb": [
        "f_oa03",
        "dr_emb_r34", "dr_emb_r293lm", "dr_emb_ecapa2",
    ],
    # 2nd-pass identity re-clustering (SECOND_PASS_PLAN.md opts 4 / B / B+),
    # OFAT off the adopted ECAPA2 diarization. f_oa03 = current no-op finalist;
    # dr_emb_ecapa2 = ECAPA2-on-raw diarization, the immediate anchor each option
    # must beat in the rescore.
    "refine2": [
        "f_oa03", "dr_emb_ecapa2",
        "as_ecapa2anchor", "dr_refine", "dr_refine_raw", "dr_refineplus",
    ],
    # ======================================================================
    # The DEFINITIVE SWEEP (SWEEP_DESIGN.md §3.2). Run with
    #   --groups definitive --split dev --anchor dr_refineplus
    # then route headline scoring through
    #   scripts/rescore_stratified.py --anchor dr_refineplus --configs ...
    # for recording-clustered CIs + Holm/FDR (the harness CI is diagnostic).
    # The anchor `dr_refineplus` is auto-included (and `baseline` is always
    # prepended by the runner; it is harmless as a fixed reference column).
    # ======================================================================
    "definitive": [
        # Phase 0 — anchor + Tier-B diagnostics (existing rows).
        "dr_refineplus",          # 0.1 the fixed anchor (reproduce current best)
        "dr_emb_ecapa2",          # 0.2 B+ OFF (= enhON,B+off = A6 corner)
        "nosep",                  # 0.3 separation ablation
        "dr_refine_raw",          # 0.4 relabel audio_source control
        "f_oa03",                 # 0.5 ECAPA2-diar OFF confirm
        # Phase 1 — Tier-A coordinate-ascent (OFAT off the anchor).
        "dr_oa020", "dr_oa025", "dr_oa035", "dr_oa040",  # A1 OA refine
        "dr_oa050",                                       # A1 0.5 bracket insurance
        "dr_bwe_naive",                                   # A3 BWE toggle
        "dr_enh_mossgan",                                 # Tier-B Pareto re-confirm
        "dr_refineplus_v3",                               # large-v3 writeup ref
        # Phase 2 — interaction-confirmation 2×2 cells.
        "dr_oa02_naive",                                  # A4 OA×BWE 4th corner
        "dr_retry0", "dr_oa00", "dr_retry0_oa00",         # A5 retry×OA
        "dr_enhoff", "dr_enhoff_norelabel",               # A6 enh×relabel
        # Phase 3 — HOLES-BUNDLE Class-1 (04_HOLES.md; existing/just-promoted knobs).
        "dr_vad040", "dr_vad050",                         # A2 VAD threshold grid
        "dr_vad050_naive",                                # C1 VAD×BWE corner
        "dr_vad_ar2",                                     # C5 VAD attack/release
        "dr_enh_mossgan_oa040",                           # C4 mossgan @ own OA 0.4
        "dr_refineplus_r34",                              # C6 B+ over resnet34 diar
        # Phase 4 — HOLES-BUNDLE Class-2: promoted uninventoried levers D1–D6.
        "dr_silfloor0", "dr_silfloor5e4", "dr_silfloor1e3",      # D1 silence_floor
        "dr_resample_vhq", "dr_resample_kaiser",                 # D2 resample_quality
        "dr_seamsil03", "dr_seamsil07",                          # D3 seam_silence_threshold
        "dr_ovmin02", "dr_ovmin035",                             # D4 overlap_min_duration_s
        "dr_anchmin05", "dr_anchmin10",                          # D5 anchor_min_duration_s
        "dr_linkage_avg",                                        # D6 clustering_method
        # Phase 5 — COMPREHENSIVE expansion (full-grid, 2026-06-22). All off the
        # same dr_refineplus anchor; finer grids + new levers + interaction corners.
        "dr_oa010", "dr_oa015", "dr_oa045", "dr_oa060", "dr_oa070",   # OA-finer
        "dr_vad020", "dr_vad030", "dr_vad060",                        # VAD-finer
        "dr_vad_ar0", "dr_vad_ar3",                                   # VAD-a/r
        "dr_seamsil02", "dr_seamsil04", "dr_seamsil06", "dr_seamsil08",  # seam-finer
        "dr_silfloor2e4", "dr_silfloor2e3",                          # silence_floor-finer
        "dr_ovmin015", "dr_ovmin03", "dr_ovmin05",                   # overlap_min-finer
        "dr_anchmin04", "dr_anchmin075", "dr_anchmin15",             # anchor_min-finer
        "dr_linkage_complete", "dr_linkage_ward",                    # linkage
        "dr_cohere",                                                 # Cohere ASR
        # re-confirm @ e46/B+ base
        "dr_ctx_fixedpad", "dr_ctx_fixedpad15", "dr_ctx_none",
        "dr_merge03", "dr_merge08",
        "dr_align_xlsr1b", "dr_cs15", "dr_beam10",
        "dr_nospeech_hi", "dr_condprev",
        # interactions
        "dr_oa04_naive",
    ],
    # Phase 2 — OA × large-v3 stack + OA-peak bracket (2026-06-24). Rescore with
    # LOO refs: dr_refineplus / dr_refineplus_v3 / dr_oa045 / dr_oa050 / dr_oa070.
    "phase2": [
        "dr_oa045_v3", "dr_oa050_v3", "dr_oa070_v3",
        "dr_oa075", "dr_oa080", "dr_oa050_v3_naive",
    ],
    # Campaign v2 attribution-repair levers, OFAT + combined off the dr_oa050
    # finalist (the anchor reference; baseline auto-included by the runner).
    "v2dev": [
        "dr_oa050",
        "v2_ngram3", "v2_pad",
    ],
    # Campaign v2 round 2 — post-diagnosis levers (dr_oa050 already run, so it
    # is skipped on run and just anchors the rescore).
    "v2fix": ["dr_oa050", "v2_reseed", "v2_loopretry"],
    # Campaign v3 — phrase-loop mirror of loop_retry (off the v2_finalist base).
    "v3": ["v3_phraseloop"],
    # Campaign v3 — linkage OFAT off v3_phraseloop (closes the diarizer-knob
    # inventory; expected null). Rescore with --anchor v3_phraseloop.
    "link": ["v3_link_ward", "v3_link_avg", "v3_link_comp"],
    # Campaign v4 — EEND (Sortformer) diarizer arm off v3_phraseloop. Needs
    # $SORTFORMER_VENV_PY. Rescore with --anchor v3_phraseloop. Prereg pending.
    "v4": ["v4_eend"],
    # Campaign v4.1 — Sortformer rehabilitation levers (docs/sweep_plan/V41_PREREG.md).
    # v41_full = the test candidate (L1+L2+L3+L4); v41_merge/v41_hyst/v41_fallback
    # decompose the levers. All off the v4_eend base. Needs $SORTFORMER_VENV_PY;
    # rescore with --anchor v3_phraseloop (report vs v4_eend too).
    "v41": ["v41_merge"],
    # Campaign v5 (instrument reframe) — Sortformer v2.1 dev arms off the v4_eend
    # base (docs/sweep_plan/V5_INSTRUMENT_PREREG.md §"Phase 1"). Needs
    # $SORTFORMER_VENV_PY; rescore with --anchor v3_phraseloop (report vs v4_eend).
    "v2p1": ["v2p1_eend", "v2p1_merge", "v2p1_merge_th040"],
}


# Dotted-path overrides now live in the package (asr_pipeline.config.apply_overrides,
# A1) so the CLI `--set` flag and this registry share ONE fail-loud policy: a
# typo'd path raises AttributeError instead of silently running the baseline under
# the typo'd name. Kept under the local name `_apply` for the sweep's callers +
# tests; behaviour is byte-identical to the former in-script version.
_apply = apply_overrides


def _build_cfg(overrides: dict):
    return _apply(fresh_eval_cfg(CFG_PATH), overrides)


# --- Bootstrap uncertainty (pure, unit-tested) ----------------------------
# The pipeline runs deterministic=True, so re-running a config is ~zero-variance
# (SCOPE: no fabricated signal — re-running would invent precision that isn't
# there). The uncertainty that matters on the small dev split is DEV-SET
# SAMPLING uncertainty: had we drawn a different set of fragments, would the
# ranking hold? We quantify it by resampling the FRAGMENTS with replacement and
# recomputing the micro-averaged cpWER from each fragment's (errors, ref_words).
# This is the correct unit because cpWER is micro-averaged (sum errors / sum ref
# words), NOT an average of per-fragment rates.
#
# FRAGMENT-level, non-clustered — descriptive only. Multi-segment recordings
# contribute more than one draw each, so this understates the true unit of
# independence and overstates significance relative to the authoritative
# recording-clustered bootstrap in scripts/rescore_stratified.py. For
# inference (CIs/Holm/FDR you'd actually quote), use that script instead.

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 1234
CI_LEVEL = 0.95


def _micro_avg(errors: np.ndarray, ref_words: np.ndarray) -> float:
    """Micro-averaged rate (%) over selected fragments: 100 * Σerr / Σref.

    Returns NaN when the resampled reference length is zero (degenerate; the
    caller's CI of an all-empty set is itself NaN, never a crash)."""
    total_ref = float(ref_words.sum())
    if total_ref <= 0:
        return float("nan")
    return 100.0 * float(errors.sum()) / total_ref


def bootstrap_microavg_ci(
    frag_counts: list[tuple[float, float]],
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    ci_level: float = CI_LEVEL,
) -> tuple[float, float]:
    """Bootstrap CI for a config's micro-averaged cpWER over its fragments.

    ``frag_counts`` is one ``(errors, ref_words)`` pair per fragment. We draw
    ``n_resamples`` bootstrap samples — each a draw of ``len(frag_counts)``
    fragments *with replacement* — and recompute the micro-average (Σerr / Σref
    over the drawn fragments) for each. The CI is the empirical percentile
    interval at ``ci_level``.

    Deterministic for a fixed ``seed``. Returns ``(lo, hi)`` in percent.
    Edge cases: empty input or all-zero reference length → ``(nan, nan)``; a
    single fragment → a degenerate ``(point, point)`` interval (every resample
    is that fragment), which is the honest answer — one fragment carries no
    sampling spread.
    """
    if not frag_counts:
        return (float("nan"), float("nan"))
    errors = np.array([e for e, _ in frag_counts], dtype=float)
    ref_words = np.array([w for _, w in frag_counts], dtype=float)
    n = len(frag_counts)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    stats = np.array([_micro_avg(errors[i], ref_words[i]) for i in idx])
    stats = stats[~np.isnan(stats)]
    if stats.size == 0:
        return (float("nan"), float("nan"))
    lo_q = (1.0 - ci_level) / 2.0
    hi_q = 1.0 - lo_q
    lo, hi = np.quantile(stats, [lo_q, hi_q])
    return (float(lo), float(hi))


def bootstrap_paired_diff_ci(
    cfg_counts: list[tuple[float, float]],
    base_counts: list[tuple[float, float]],
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    ci_level: float = CI_LEVEL,
) -> tuple[float, float, float, bool]:
    """Paired bootstrap of the micro-averaged cpWER difference (config − base).

    ``cfg_counts`` and ``base_counts`` are aligned fragment-by-fragment over the
    SHARED fragment set (same fragment at the same index on both sides) — the
    caller must align them. Pairing matters: both configs are scored on the same
    fragments, so resampling the same fragment indices jointly cancels the
    shared fragment-difficulty variance and tightens the interval that a
    head-to-head comparison actually needs.

    Each resample draws fragment indices once and applies them to BOTH sides,
    then takes ``micro(config) − micro(base)`` on the drawn fragments. The point
    estimate is the difference of the two full-set micro-averages.

    Returns ``(delta, lo, hi, sig)`` in percent, where ``sig`` is True iff the
    CI excludes 0 (the difference is distinguishable from noise at ``ci_level``).
    Deterministic for a fixed ``seed``. Empty input → ``(nan, nan, nan, False)``.
    """
    if not cfg_counts or not base_counts:
        return (float("nan"), float("nan"), float("nan"), False)
    if len(cfg_counts) != len(base_counts):
        raise ValueError(
            "paired bootstrap needs aligned fragment lists "
            f"(got {len(cfg_counts)} vs {len(base_counts)})"
        )
    cfg_err = np.array([e for e, _ in cfg_counts], dtype=float)
    cfg_ref = np.array([w for _, w in cfg_counts], dtype=float)
    base_err = np.array([e for e, _ in base_counts], dtype=float)
    base_ref = np.array([w for _, w in base_counts], dtype=float)
    n = len(cfg_counts)

    delta = _micro_avg(cfg_err, cfg_ref) - _micro_avg(base_err, base_ref)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    diffs = np.array([
        _micro_avg(cfg_err[i], cfg_ref[i]) - _micro_avg(base_err[i], base_ref[i])
        for i in idx
    ])
    diffs = diffs[~np.isnan(diffs)]
    if diffs.size == 0:
        return (float(delta), float("nan"), float("nan"), False)
    lo_q = (1.0 - ci_level) / 2.0
    hi_q = 1.0 - lo_q
    lo, hi = np.quantile(diffs, [lo_q, hi_q])
    sig = bool(lo > 0 or hi < 0)
    return (float(delta), float(lo), float(hi), sig)


# --- Run ------------------------------------------------------------------


def run_config(name, overrides, force, eval_root, recordings) -> None:
    """Run one config over the given recordings → ``<id>/sweep/<name>/``.

    Delegates to the shared ``asr_pipeline.batch.run_batch`` — the one home for
    the per-recording loop, SCOPE §4.2 failure isolation, and the GPU teardown
    (unload → gc → empty_cache) this loop used to inline. The sweep pins its
    LEGACY skip sentinel (``transcript_A.txt`` present, not the new default
    ``metadata.json``) so every completed sweep tree already on disk is still
    recognised as done and NEVER recomputed — the campaign ledger + provenance
    hashes depend on those exact outputs staying put. ``run_meta.json`` still
    carries ``seconds`` (now alongside per-stage ``stages``, a superset
    ``score_configs._read_run_seconds`` reads transparently). The mixture already
    lives at ``<id>/<id>.wav`` (the input), so it is not copied back onto itself.
    """
    subdir = f"sweep/{name}"
    cfg = _build_cfg(overrides)
    audio_paths = [Path(eval_root) / fid / f"{fid}.wav" for fid in recordings]
    run_batch(
        cfg, audio_paths, out_root=Path(eval_root), subdir_name=subdir,
        skip_existing=not force,
        is_complete=lambda d: (d / "transcript_A.txt").exists(),
        copy_mixture=False,
    )


# --- Score ----------------------------------------------------------------


def _read_run_seconds(run_dir: Path) -> tuple[float, float]:
    """(wall-clock seconds, run_meta mtime epoch) for one (config, recording) run.

    The mtime dates WHEN the outputs were produced — run_config skips fragments
    that already have outputs, so a config row can mix runs from different days
    (different GPU-contention regimes). ``secs_per_frag`` alone made a June run
    look comparable to a July one; the ``runs_span`` column built from these
    mtimes makes that staleness visible. Older runs predate ``run_meta.json``;
    a missing/garbled file degrades to (NaN, NaN) (→ blank columns), never a
    crash."""
    meta = run_dir / "run_meta.json"
    try:
        secs = float(json.loads(meta.read_text(encoding="utf-8"))["seconds"])
        return secs, meta.stat().st_mtime
    except (OSError, ValueError, KeyError, TypeError):
        return float("nan"), float("nan")


def score_configs(config_names, eval_root, recordings, anchor="baseline") -> pd.DataFrame:
    """cpWER / MIMO-WER / tcpWER / ORC per (config, recording), micro-averaged.

    Micro-average (sum errors / sum reference length) is the standard WER
    aggregation; we also keep the per-recording rates for inspection. GT is
    read via the eval loader — the EAF at ``<id>/annotation.eaf`` if present,
    else ``<id>/reference/speaker_{A,B}.txt`` — so this works for both the
    ELAN-annotated fragments and the older .txt-GT recordings.

    Appended (additive) columns:
      - ``secs_per_frag`` — mean wall-clock per fragment from each run's
        ``run_meta.json`` (blank if no run_meta on disk). CAVEAT: run_config
        skips already-done fragments, so this is the timing of whenever the
        outputs were PRODUCED, not of the current sweep — check ``runs_span``.
      - ``runs_span`` — mm-dd (or mm-dd..mm-dd) range of the run_meta mtimes
        behind this row. Rows with different spans ran under different GPU-load
        regimes; their secs_per_frag are not comparable.
      - ``cpwer_ci_lo`` / ``cpwer_ci_hi`` — bootstrap 95% CI for the config's
        micro-averaged cpWER, resampling the FRAGMENTS with replacement
        (``BOOTSTRAP_RESAMPLES`` draws, seed ``BOOTSTRAP_SEED``). This captures
        dev-set sampling uncertainty — the only uncertainty that matters under
        ``deterministic=True`` (re-running is ~zero-variance).
      - ``vs_base_delta`` / ``vs_base_ci_lo`` / ``vs_base_ci_hi`` / ``sig`` —
        paired bootstrap of the micro-averaged cpWER difference vs the
        ``anchor`` config (default ``baseline``), on the fragments both scored;
        ``sig`` True iff the CI excludes 0. Blank for the anchor itself and
        skipped (blank, with a printed note) if the anchor is not among the
        scored configs.

    ``anchor`` defaults to ``baseline`` so the historical behaviour is byte-
    identical; the definitive sweep passes ``anchor='dr_refineplus'``. This is a
    FRAGMENT-level diagnostic CI — the HEADLINE recording-clustered + Holm/FDR
    scoring is ``scripts/rescore_stratified.py --anchor <anchor>`` (SWEEP_DESIGN
    §3.4). The column name stays ``vs_base_*`` for CSV continuity.
    """
    gt = {}
    for fid in recordings:
        rec = load_recording(eval_root / fid)
        gt[fid] = load_reference_utterances(rec) if rec is not None else {}
    rows = []
    # Per-config per-fragment cpWER (errors, ref_words) for the bootstrap, keyed
    # by fid so the paired-vs-baseline comparison can align on shared fragments.
    cp_frag_counts: dict[str, dict[str, tuple[float, float]]] = {}
    for name in config_names:
        # Per-metric [error_sum, ref_length_sum] accumulators. cpWER carries an
        # extra `tcp` pair (cpwer_meeteval returns both in one call), so its
        # entry holds two (err, len) pairs.
        acc: dict[str, list[float]] = {
            "cp": [0, 0], "tcp": [0, 0],   # 2-stream output, cpWER permutation
            "mimo": [0, 0],                # 2-stream output, MIMO-WER (pipeline hyp)
            "orc": [0, 0],                 # 2-stream output, ORC-WER
            "cer": [0, 0],                 # 2-stream output, cp-CER
            "mixORC": [0, 0],              # mixture floor, ORC-WER (time-fixed merge)
            "mixMIMO": [0, 0],             # mixture floor, MIMO-WER (optimised interleaving)
            "mixCER": [0, 0],              # mixture floor, MIMO-CER
        }
        per_rec = {}
        frag_counts: dict[str, tuple[float, float]] = {}
        secs_list: list[float] = []
        mtime_list: list[float] = []
        n_done = 0
        for fid in recordings:
            d = eval_root / fid / "sweep" / name
            hyp = read_per_speaker(d)
            if hyp is None:
                continue
            ref = {k: v for k, v in gt[fid].items() if v}
            if not ref:
                continue
            mix_utts = read_mixture(d)
            # Shared per-fragment scoring core — the SAME meeteval calls
            # rescore_stratified / dump_sweep_results make. It also computes an
            # ORC-CER content floor this harness does not report; that sub-result
            # is simply not read here, so no reported number changes.
            scored = per_fragment_metrics(ref, hyp, session_id=fid, mix=mix_utts)
            r = scored["cp"]
            acc["cp"][0] += r["cp_errors"];  acc["cp"][1] += r["cp_length"]
            acc["tcp"][0] += r["tcp_errors"]; acc["tcp"][1] += r["tcp_length"]
            # MIMO-WER on the SAME per-speaker pipeline hyp (speaker-agnostic).
            mw = scored["mimo"]
            acc["mimo"][0] += mw["errors"]; acc["mimo"][1] += mw["length"]
            # Per-fragment cpWER counts for the bootstrap (errors, ref_words).
            frag_counts[fid] = (float(r["cp_errors"]), float(r["cp_length"]))
            run_secs, run_mtime = _read_run_seconds(d)
            secs_list.append(run_secs)
            mtime_list.append(run_mtime)
            o = scored["orc"]
            acc["orc"][0] += o["errors"]; acc["orc"][1] += o["length"]
            cc = scored["cpcer"]
            acc["cer"][0] += cc["errors"]; acc["cer"][1] += cc["length"]
            if mix_utts is not None:
                m = scored["mix_orc"]
                acc["mixORC"][0] += m["errors"]; acc["mixORC"][1] += m["length"]
                mm = scored["mix_mimo"]
                acc["mixMIMO"][0] += mm["errors"]; acc["mixMIMO"][1] += mm["length"]
                mxc = scored["mix_cer"]
                acc["mixCER"][0] += mxc["errors"]; acc["mixCER"][1] += mxc["length"]
            per_rec[fid] = r["cpwer"]
            n_done += 1
        if n_done == 0:
            continue

        def pct(metric: str) -> float:
            err, length = acc[metric]
            return 100 * err / length if length else float("nan")

        cpwer = pct("cp")
        orcwer = pct("orc")
        cp_frag_counts[name] = frag_counts
        ci_lo, ci_hi = bootstrap_microavg_ci(list(frag_counts.values()))
        # Mean wall-clock per scored fragment; NaN (→ blank) if no run had meta.
        secs = np.array(secs_list, dtype=float)
        secs_per_frag = float(np.nanmean(secs)) if np.any(~np.isnan(secs)) else float("nan")
        # When those outputs were produced (run_meta mtimes). A span crossing
        # days = mixed-age row: its secs_per_frag averages different GPU-load
        # regimes and must not be compared across configs as "current speed".
        mtimes = np.array(mtime_list, dtype=float)
        if np.any(~np.isnan(mtimes)):
            lo = datetime.fromtimestamp(float(np.nanmin(mtimes))).strftime("%m-%d")
            hi = datetime.fromtimestamp(float(np.nanmax(mtimes))).strftime("%m-%d")
            runs_span = lo if lo == hi else f"{lo}..{hi}"
        else:
            runs_span = ""
        row = {
            "config": name,
            "n": n_done,
            "cpWER": cpwer,
            "mimoWER": pct("mimo"),          # speaker-agnostic WER of the pipeline hyp
            "cpwer_ci_lo": ci_lo,
            "cpwer_ci_hi": ci_hi,
            "secs_per_frag": secs_per_frag,
            "runs_span": runs_span,
            "orcWER": orcwer,
            "attr_gap": cpwer - orcwer,      # speaker-attribution penalty
            "tcpWER": pct("tcp"),
            "CER": pct("cer"),
            "mixMIMO": pct("mixMIMO"),
            "mixORC": pct("mixORC"),
            "mixCER": pct("mixCER"),
        }
        for fid in recordings:
            row[fid[:8]] = round(100 * per_rec[fid], 1) if fid in per_rec else None
        rows.append(row)

    # --- Paired comparison vs the anchor config (section C) -----------------
    # For each non-anchor config, bootstrap the micro-averaged cpWER difference
    # on the fragments BOTH it and the anchor scored (paired = same fragment
    # indices resampled jointly). `sig` flags a CI that excludes 0. The anchor
    # defaults to `baseline` (historical behaviour); the definitive sweep uses
    # `dr_refineplus`. FRAGMENT-level diagnostic only — see docstring.
    base_counts = cp_frag_counts.get(anchor)
    if base_counts is None:
        print(f"note: anchor {anchor!r} not among scored configs — "
              "skipping paired vs_base columns (left blank).")
    for row in rows:
        row["vs_base_delta"] = float("nan")
        row["vs_base_ci_lo"] = float("nan")
        row["vs_base_ci_hi"] = float("nan")
        row["sig"] = None
        if base_counts is None or row["config"] == anchor:
            continue
        cfg_counts = cp_frag_counts[row["config"]]
        shared = [f for f in cfg_counts if f in base_counts]
        if not shared:
            continue
        delta, lo, hi, sig = bootstrap_paired_diff_ci(
            [cfg_counts[f] for f in shared],
            [base_counts[f] for f in shared],
        )
        row["vs_base_delta"] = delta
        row["vs_base_ci_lo"] = lo
        row["vs_base_ci_hi"] = hi
        row["sig"] = sig

    df = pd.DataFrame(rows).sort_values("cpWER").reset_index(drop=True)
    return df


# --- Provenance + durable append-only ledger ------------------------------
# SWEEP_DESIGN §3.3 / §6.5: every reported number must reproduce from the
# committed CONFIGS rows + a durable ledger, and the GT must be pinned by a
# snapshot hash recorded with each run. The per-run `_sweep_results.csv` is kept
# for eyeballing (overwritten); the ledger is APPEND-only and never rewritten.


def _git_head() -> str:
    """Short git HEAD + a `-dirty` suffix if the tree has uncommitted changes.

    Degrades to ``"unknown"`` (never raises) so provenance capture can't abort a
    sweep on a box without git."""
    try:
        head = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        return f"{head}{'-dirty' if dirty else ''}" if head else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _configs_hash(config_names) -> str:
    """Stable hash of the RESOLVED override dicts for the scored arms.

    Pins the exact knob values behind each arm name, so a later CONFIGS edit that
    silently changed a row is detectable from the ledger. Sorted keys → order-
    independent; only the scored arms are hashed."""
    payload = {n: CONFIGS.get(n, {}) for n in sorted(set(config_names))}
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _gt_snapshot_hash(eval_root: Path, recordings) -> str:
    """Hash of the GT actually read by the scorer for these recordings.

    Tracks the REAL reference (the loader resolves EAF → reference/*.txt), so the
    ledger pins the GT snapshot the design's §6.1 requires and any mid-campaign
    GT edit shows up as a changed hash. Missing GT for a recording contributes a
    sentinel rather than crashing."""
    h = hashlib.sha256()
    for fid in sorted(recordings):
        rec = load_recording(eval_root / fid)
        gt = load_reference_utterances(rec) if rec is not None else {}
        h.update(fid.encode("utf-8"))
        for spk in sorted(gt):
            h.update(spk.encode("utf-8"))
            for u in gt[spk]:
                h.update((u.text or "").encode("utf-8"))
        if not gt:
            h.update(b"<no-gt>")
    return h.hexdigest()[:16]


def build_provenance(config_names, eval_root, recordings, anchor) -> dict:
    """The provenance/GT-snapshot header for one scoring pass (SWEEP_DESIGN §3.3)."""
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_head": _git_head(),
        "configs_hash": _configs_hash(config_names),
        "gt_snapshot_hash": _gt_snapshot_hash(eval_root, recordings),
        "anchor": anchor,
        "eval_root": str(eval_root),
        "n_recordings": len(recordings),
        "recordings": list(recordings),
        "configs": list(config_names),
    }


def append_ledger(ledger_path: Path, df: pd.DataFrame, provenance: dict) -> None:
    """Append every scored row to the durable ledger, tagged with provenance.

    The ledger is APPEND-only (SWEEP_DESIGN §3.3 forbids the overwrite-per-run
    pattern of doc 02 §4.7). Each appended row carries the run timestamp, git
    head, configs/GT hashes, and anchor, so the union table across the whole
    campaign reconstructs from this one file. A sibling ``<stem>_provenance.json``
    accumulates one JSON object per run for the full header."""
    tagged = df.copy()
    tagged.insert(0, "run_ts", provenance["timestamp_utc"])
    tagged.insert(1, "git_head", provenance["git_head"])
    tagged.insert(2, "configs_hash", provenance["configs_hash"])
    tagged.insert(3, "gt_snapshot_hash", provenance["gt_snapshot_hash"])
    tagged.insert(4, "anchor", provenance["anchor"])
    header = not ledger_path.exists()
    if not header:
        # Append-only + never rewritten ⇒ the on-disk header is the schema.
        # Align to it so a column added to score_configs later (e.g. runs_span)
        # can't silently shift values under the old header; brand-new columns
        # simply don't enter the ledger until it is re-created.
        on_disk = pd.read_csv(ledger_path, nrows=0).columns
        tagged = tagged.reindex(columns=on_disk)
    tagged.to_csv(ledger_path, mode="a", header=header, index=False)
    prov_path = ledger_path.with_name(ledger_path.stem + "_provenance.json")
    with prov_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(provenance) + "\n")


def report_coverage(config_names, eval_root, recordings) -> None:
    """Warn (don't exit) when scored configs disagree on recording coverage.

    The harness is the diagnostic tool; the strict paired-coverage GATE is
    ``rescore_stratified.py`` (which exits if no recording is common to all
    configs). This mirrors that report so a partial cache is visible here too —
    SWEEP_DESIGN §3.4: drops reported, never silently intersected away."""
    cov = {}
    for name in config_names:
        present = [fid for fid in recordings
                   if read_per_speaker(eval_root / fid / "sweep" / name) is not None]
        cov[name] = set(present)
    if not cov:
        return
    common = set.intersection(*cov.values())
    seen = set().union(*cov.values())
    if any(cov[c] != common for c in cov):
        for c in config_names:
            miss = seen - cov[c]
            if miss:
                print(f"!! coverage WARNING {c}: missing {len(miss)} of "
                      f"{len(seen)} recording(s): "
                      f"{', '.join(sorted(r[:8] for r in miss))}")
        print(f"!! configs disagree on coverage; harness scores each on its own "
              f"{len(common)}–{len(seen)} present recordings (paired headline "
              f"scoring is rescore_stratified.py).")


# --- Driver ---------------------------------------------------------------


def _die(msg: str) -> int:
    print(f"ERROR: {msg}", file=sys.stderr)
    return 2


def _resolve_recordings(args) -> list[str]:
    """Precedence: explicit --recordings > --split > PILOT (ad-hoc smoke set).

    The frozen split is read via the shared ``load_split`` (the SAME file the
    rescorer reads, so the run set and the scoring set cannot drift on which
    fragments are the dev/test split — SWEEP_DESIGN §3.3 frozen-set control)."""
    if args.recordings:
        return list(args.recordings)
    if args.split:
        return load_split(args.split)
    return list(PILOT)


def _selected_configs(args) -> list[str]:
    if args.configs:
        names = list(args.configs)
    elif args.groups:
        names = []
        for g in args.groups:
            names += GROUPS[g]
    else:
        names = [n for n in CONFIGS if n != "baseline"]  # all but baseline
    # baseline + the paired anchor always present for comparison; dedupe, order.
    # The anchor (default `baseline`) must be in the scored set so the paired
    # vs-anchor column has its reference; `baseline` stays a fixed reference too.
    anchor = getattr(args, "anchor", "baseline")
    lead = ["baseline"] if anchor == "baseline" else ["baseline", anchor]
    seen, ordered = set(), []
    for n in lead + names:
        if n in CONFIGS and n not in seen:
            seen.add(n); ordered.append(n)
    return ordered


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--groups", nargs="+", choices=list(GROUPS),
                    help="Config groups to run (baseline always included).")
    ap.add_argument("--configs", nargs="+", choices=list(CONFIGS),
                    help="Explicit config names (overrides --groups).")
    ap.add_argument("--score-only", action="store_true",
                    help="Skip running; just score + rank existing outputs.")
    ap.add_argument("--force", action="store_true", help="Re-run even if done.")
    ap.add_argument("--eval-root", type=Path, default=EVAL_ROOT,
                    help="Eval-tree root holding <id>/ recording dirs "
                         f"(default: {EVAL_ROOT}).")
    ap.add_argument("--recordings", nargs="+", default=None,
                    help="Recording ids under --eval-root "
                         "(default: --split if given, else the 5 pilot fragments).")
    ap.add_argument("--split", choices=["dev", "test"], default=None,
                    help="Load the frozen fragment list "
                         "asr_pipeline/eval/clarin_<split>.txt (same file the "
                         "rescorer reads). Ignored if --recordings is given.")
    ap.add_argument("--anchor", default="baseline",
                    help="Config the paired vs-anchor diagnostic column compares "
                         "to (default: baseline). The definitive sweep uses "
                         "dr_refineplus; headline scoring is rescore_stratified.py.")
    ap.add_argument("--csv", type=Path, default=None,
                    help="Per-run results CSV, overwritten "
                         "(default: <eval-root>/_sweep_results.csv).")
    ap.add_argument("--ledger", type=Path, default=None,
                    help="Durable APPEND-only ledger CSV + provenance header "
                         "(default: <eval-root>/_sweep_ledger.csv). Reproduces "
                         "every reported number across the campaign.")
    args = ap.parse_args()

    eval_root = args.eval_root.expanduser()
    recordings = _resolve_recordings(args)
    csv = args.csv or (eval_root / "_sweep_results.csv")
    ledger = args.ledger or (eval_root / "_sweep_ledger.csv")

    if args.anchor not in CONFIGS:
        return _die(f"--anchor {args.anchor!r} is not a known config name")

    names = _selected_configs(args)
    print(f"eval_root: {eval_root}")
    print(f"recordings ({len(recordings)}): {recordings}")
    print(f"anchor: {args.anchor}")
    print(f"configs: {names}\n")

    if not args.score_only:
        for name in names:
            print(f"[{name}]  overrides={CONFIGS[name] or '(baseline)'}")
            # Belt-and-suspenders for unattended runs: run_config already catches
            # per-recording failures, but a config-build / Pipeline-construction
            # error would otherwise abort the whole sweep. Contain it to the one
            # config so the remaining configs still run.
            try:
                run_config(name, CONFIGS[name], args.force, eval_root, recordings)
            except Exception as e:
                print(f"  [{name}] CONFIG-LEVEL ERROR {type(e).__name__}: {e} — skipping")
            print()

    report_coverage(names, eval_root, recordings)
    df = score_configs(names, eval_root, recordings, anchor=args.anchor)
    pd.set_option("display.width", 200)
    print("\n=== ranked by micro-averaged cpWER (lower is better) ===")
    print(df.to_string(index=False))
    print("(cpwer_ci_* / vs_base_* CIs above are a FRAGMENT-level, non-clustered "
          "bootstrap — descriptive only, not for inference. For a citable CI/"
          "significance call, use scripts/rescore_stratified.py.)")
    df.to_csv(csv, index=False)
    print(f"\nwrote {csv}")

    # Durable provenance + append-only ledger (SWEEP_DESIGN §3.3 / §6.5).
    provenance = build_provenance(names, eval_root, recordings, args.anchor)
    print("\n=== provenance ===")
    for k in ("timestamp_utc", "git_head", "configs_hash",
              "gt_snapshot_hash", "anchor"):
        print(f"  {k}: {provenance[k]}")
    append_ledger(ledger, df, provenance)
    print(f"appended {len(df)} rows to {ledger}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
