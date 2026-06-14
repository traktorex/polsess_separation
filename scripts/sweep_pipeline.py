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
import gc
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from asr_pipeline import Pipeline                                   # noqa: E402
from asr_pipeline.io import write_pipeline_outputs                 # noqa: E402
from asr_pipeline.eval.metrics import (                             # noqa: E402
    cp_cer_meeteval,
    cpwer_meeteval,
    mimo_cer_meeteval,
    mimo_wer_meeteval,
    orc_wer_meeteval,
    orc_wer_multistream,
)
from asr_pipeline.eval.config_presets import fresh_eval_cfg         # noqa: E402
from asr_pipeline.eval.layer3 import read_mixture, read_per_speaker  # noqa: E402
from asr_pipeline.eval.recordings import (                          # noqa: E402
    load_recording,
    load_reference_utterances,
)


EVAL_ROOT = Path("~/datasets/eval/clarin_fragments").expanduser()
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
    "bwe_flowhigh":      {"post_separation_processing.backend": "flowhigh"},
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
    # --- BWE decision (SCOPE open question 4): flowhigh vs ap_bwe on the
    # round-2/3 winner, not the mpsenet baseline (where round-1's bwe_*
    # configs were swamped by enhancement errors) ---
    "frcrn_vad_strict_flowhigh": {"enhancement.backend": "frcrn_se_16k",
                                  "separation.vad_threshold": 0.5,
                                  "separation.vad_soft_threshold": 0.2,
                                  "post_separation_processing.backend": "flowhigh"},
    # input_sr A/B: default.yaml ships flowhigh_input_sr=8000 (matches the
    # separator's spectral content); this arm feeds 16 kHz instead.
    "frcrn_vad_strict_flowhigh16": {"enhancement.backend": "frcrn_se_16k",
                                    "separation.vad_threshold": 0.5,
                                    "separation.vad_soft_threshold": 0.2,
                                    "post_separation_processing.backend": "flowhigh",
                                    "post_separation_processing.flowhigh_input_sr": 16_000},

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
    "r1_bwe_flowhigh8":     {"post_separation_processing.backend": "flowhigh",
                             "post_separation_processing.flowhigh_input_sr": 8_000},
    "r1_bwe_flowhigh16":    {"post_separation_processing.backend": "flowhigh",
                             "post_separation_processing.flowhigh_input_sr": 16_000},
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
    "r2_best_fh16":  {"enhancement.enabled": False, "transcription.model_name": "large-v3-turbo",
                      "post_separation_processing.backend": "flowhigh",
                      "post_separation_processing.flowhigh_input_sr": 16_000,
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
    "r3_best_flowhigh16": {"enhancement.enabled": False,
                           "post_separation_processing.backend": "flowhigh",
                           "post_separation_processing.flowhigh_input_sr": 16_000},
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
    "ah_flowhigh_nrng3": {"enhancement.enabled": False,
                          "post_separation_processing.backend": "flowhigh",
                          "post_separation_processing.flowhigh_input_sr": 16_000,
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
    "c_zipenhancer_nrng3": {"enhancement.backend": "zipenhancer_16k",
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
    "f_enh_zip":  {"enhancement.backend": "zipenhancer_16k"},
    "f_oa03":     {"enhancement.observation_mix_ratio": 0.3},
    "f_oa05":     {"enhancement.observation_mix_ratio": 0.5},
    "f_oa07":     {"enhancement.observation_mix_ratio": 0.7},
    "f_noretry":  {"transcription.retry_collapsed_chunk_size": 0},
    "f_cs15":     {"transcription.chunk_size": 15},
}

# Named groups for --groups selection. "baseline" is always included.
GROUPS: dict[str, list[str]] = {
    "asr":        ["asr_largev3"],
    "enhance":    ["enh_mossformer", "enh_frcrn", "enh_none"],
    "bwe":        ["bwe_naive", "bwe_flowhigh"],
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
           "r1_bwe_naive", "r1_bwe_flowhigh8", "r1_bwe_flowhigh16",
           "r1_merge_03", "r1_merge_08",
           "r1_no_rmsmatch", "r1_perpiece_rms",
           "r1_nosep", "r1_noenh", "r1_nosep_noenh"],
    # Round 2: combined winners + leave-one-out + the last enhancement backend.
    # r1_noenh kept in for a same-table reference point.
    "r2": ["r2_best_naive", "r2_best_fh16", "r2_best_v3",
           "r2_loo_enhon", "r2_loo_v2", "r2_loo_apbwe", "r2_loo_rmson", "r2_loo_beam5",
           "r2_enh_mossformer_gan", "r1_noenh"],
    # Round 3: confirm the LOO-stacked prediction + re-validate each axis at that
    # operating point + separation ablation. Carry r2_loo_v2 (R2 best) as a ref.
    "r3": ["r3_best", "r3_best_turbo", "r3_best_v3", "r3_best_rmsoff",
           "r3_best_beam10", "r3_best_flowhigh16", "r3_best_nosep",
           "r2_loo_v2", "r1_noenh"],
    # Round 4: final ablation cell (enh ON at the best operating point).
    "r4": ["r4_enhon"],
    # Anti-hallucination full sweep (no_repeat_ngram_size=3 at every operating
    # point) across all 23 dev. Refs (baseline/r3_best/r1_noenh/r3_best_flowhigh16)
    # rescored alongside. ah_nrng3/ah_nrng2 = baseline+nrng (already partial from gate).
    "ah_full": ["ah_nrng3", "ah_nrng2", "ah_finalist_nrng3", "ah_apbwe_nrng3",
                "ah_flowhigh_nrng3", "r3_best", "r1_noenh", "r3_best_flowhigh16", "r4_enhon"],
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
        "r1_noenh", "enh_mossformer", "f_enh_zip",
        # observation-adding (dry/wet on the frcrn baseline)
        "f_oa03", "f_oa05", "f_oa07",
        # bandwidth extension
        "r1_bwe_naive", "r1_bwe_flowhigh8", "r1_bwe_flowhigh16",
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
}


def _apply(cfg, overrides: dict):
    """Apply dotted-path overrides onto a config, then re-validate.

    A typo'd path must fail loud (SCOPE §4.2): bare ``setattr`` would create a
    junk attribute, leave the intended knob at its default, and silently run
    the baseline under the typo'd name — a fabricated sweep row with no signal.
    """
    for path, val in overrides.items():
        obj = cfg
        *parents, leaf = path.split(".")
        for p in parents:
            obj = getattr(obj, p)
        if not hasattr(obj, leaf):
            raise AttributeError(f"unknown override path: {path!r}")
        setattr(obj, leaf, val)
    cfg.__post_init__()
    return cfg


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
    """Run one config over the given recordings → ``<id>/sweep/<name>/``."""
    subdir = f"sweep/{name}"
    for fid in recordings:
        rec_dir = eval_root / fid
        audio = rec_dir / f"{fid}.wav"
        target = rec_dir / subdir / "transcript_A.txt"
        if not audio.exists():
            print(f"    {fid}: MISSING audio {audio}")
            continue
        if target.exists() and not force:
            print(f"    {fid}: skip (done)")
            continue
        t0 = time.perf_counter()
        cfg = _build_cfg(overrides)
        p = Pipeline(cfg)
        try:
            ctx = p.run(str(audio))
            out = write_pipeline_outputs(ctx, rec_dir, config_snapshot=asdict(cfg),
                                         subdir_name=subdir)
            secs = time.perf_counter() - t0
            # Per-run wall-clock for the secs_per_frag column. Written into the
            # same run dir as the outputs; absent run_meta degrades to blank in
            # score_configs (older runs predate this), never a crash.
            (out / "run_meta.json").write_text(
                json.dumps({"seconds": secs}), encoding="utf-8")
            print(f"    {fid}: done in {secs:.1f}s")
        except Exception as e:
            print(f"    {fid}: ERROR {type(e).__name__}: {e}")
        finally:
            p.unload(); del p; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# --- Score ----------------------------------------------------------------


def _read_run_seconds(run_dir: Path) -> float:
    """Wall-clock seconds for one (config, recording) run, or NaN if absent.

    Older runs predate ``run_meta.json``; a missing/garbled file degrades to NaN
    (→ blank ``secs_per_frag``), never a crash."""
    meta = run_dir / "run_meta.json"
    try:
        return float(json.loads(meta.read_text(encoding="utf-8"))["seconds"])
    except (OSError, ValueError, KeyError, TypeError):
        return float("nan")


def score_configs(config_names, eval_root, recordings) -> pd.DataFrame:
    """cpWER / MIMO-WER / tcpWER / ORC per (config, recording), micro-averaged.

    Micro-average (sum errors / sum reference length) is the standard WER
    aggregation; we also keep the per-recording rates for inspection. GT is
    read via the eval loader — the EAF at ``<id>/annotation.eaf`` if present,
    else ``<id>/reference/speaker_{A,B}.txt`` — so this works for both the
    ELAN-annotated fragments and the older .txt-GT recordings.

    Appended (additive) columns:
      - ``secs_per_frag`` — mean wall-clock per fragment from each run's
        ``run_meta.json`` (blank if no run_meta on disk).
      - ``cpwer_ci_lo`` / ``cpwer_ci_hi`` — bootstrap 95% CI for the config's
        micro-averaged cpWER, resampling the FRAGMENTS with replacement
        (``BOOTSTRAP_RESAMPLES`` draws, seed ``BOOTSTRAP_SEED``). This captures
        dev-set sampling uncertainty — the only uncertainty that matters under
        ``deterministic=True`` (re-running is ~zero-variance).
      - ``vs_base_delta`` / ``vs_base_ci_lo`` / ``vs_base_ci_hi`` / ``sig`` —
        paired bootstrap of the micro-averaged cpWER difference vs the
        ``baseline`` config, on the fragments both scored; ``sig`` True iff the
        CI excludes 0. Blank for ``baseline`` itself and skipped (blank, with a
        printed note) if ``baseline`` is not among the scored configs.
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
        n_done = 0
        for fid in recordings:
            d = eval_root / fid / "sweep" / name
            hyp = read_per_speaker(d)
            if hyp is None:
                continue
            ref = {k: v for k, v in gt[fid].items() if v}
            if not ref:
                continue
            r = cpwer_meeteval(ref, hyp, session_id=fid)
            acc["cp"][0] += r["cp_errors"];  acc["cp"][1] += r["cp_length"]
            acc["tcp"][0] += r["tcp_errors"]; acc["tcp"][1] += r["tcp_length"]
            # MIMO-WER on the SAME per-speaker pipeline hyp (speaker-agnostic).
            mw = mimo_wer_meeteval(ref, hyp, session_id=fid)
            acc["mimo"][0] += mw["errors"]; acc["mimo"][1] += mw["length"]
            # Per-fragment cpWER counts for the bootstrap (errors, ref_words).
            frag_counts[fid] = (float(r["cp_errors"]), float(r["cp_length"]))
            secs_list.append(_read_run_seconds(d))
            o = orc_wer_multistream(ref, hyp, session_id=fid)
            acc["orc"][0] += o["errors"]; acc["orc"][1] += o["length"]
            cc = cp_cer_meeteval(ref, hyp, session_id=fid)
            acc["cer"][0] += cc["errors"]; acc["cer"][1] += cc["length"]
            mix_utts = read_mixture(d)
            if mix_utts is not None:
                m = orc_wer_meeteval(ref, mix_utts, session_id=fid)
                acc["mixORC"][0] += m["errors"]; acc["mixORC"][1] += m["length"]
                mm = mimo_wer_meeteval(ref, mix_utts, session_id=fid)
                acc["mixMIMO"][0] += mm["errors"]; acc["mixMIMO"][1] += mm["length"]
                mxc = mimo_cer_meeteval(ref, mix_utts, session_id=fid)
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
        row = {
            "config": name,
            "n": n_done,
            "cpWER": cpwer,
            "mimoWER": pct("mimo"),          # speaker-agnostic WER of the pipeline hyp
            "cpwer_ci_lo": ci_lo,
            "cpwer_ci_hi": ci_hi,
            "secs_per_frag": secs_per_frag,
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

    # --- Paired comparison vs the `baseline` config (section C) -------------
    # For each non-baseline config, bootstrap the micro-averaged cpWER
    # difference on the fragments BOTH it and baseline scored (paired = same
    # fragment indices resampled jointly). `sig` flags a CI that excludes 0.
    base_counts = cp_frag_counts.get("baseline")
    if base_counts is None:
        print("note: `baseline` not among scored configs — "
              "skipping paired vs_base columns (left blank).")
    for row in rows:
        row["vs_base_delta"] = float("nan")
        row["vs_base_ci_lo"] = float("nan")
        row["vs_base_ci_hi"] = float("nan")
        row["sig"] = None
        if base_counts is None or row["config"] == "baseline":
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


# --- Driver ---------------------------------------------------------------


def _selected_configs(args) -> list[str]:
    if args.configs:
        names = list(args.configs)
    elif args.groups:
        names = []
        for g in args.groups:
            names += GROUPS[g]
    else:
        names = [n for n in CONFIGS if n != "baseline"]  # all but baseline
    # baseline always present for comparison; dedupe, preserve order
    seen, ordered = set(), []
    for n in ["baseline"] + names:
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
    ap.add_argument("--recordings", nargs="+", default=PILOT,
                    help="Recording ids under --eval-root (default: the 5 pilot fragments).")
    ap.add_argument("--csv", type=Path, default=None,
                    help="Results CSV (default: <eval-root>/_sweep_results.csv).")
    args = ap.parse_args()

    eval_root = args.eval_root.expanduser()
    recordings = args.recordings
    csv = args.csv or (eval_root / "_sweep_results.csv")

    names = _selected_configs(args)
    print(f"eval_root: {eval_root}")
    print(f"recordings: {recordings}")
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

    df = score_configs(names, eval_root, recordings)
    pd.set_option("display.width", 200)
    print("\n=== ranked by micro-averaged cpWER (lower is better) ===")
    print(df.to_string(index=False))
    df.to_csv(csv, index=False)
    print(f"\nwrote {csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
