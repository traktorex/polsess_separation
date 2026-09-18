# asr_pipeline/CLAUDE.md

Agent guidance for the productionised CLARIN ASR pipeline. Stacks on top of the repository root `CLAUDE.md`.

**Before changing code here, read `SCOPE.md`** — the scope contract (purpose, error philosophy, fallback ledger, rules for agents). It overrides reviewer instincts; its `UNDECIDED` items are reserved for the author. The package is designed to be liftable: nothing in `models/` may import from it, and the webapp's job queue lives in `webapp/`, never here (SCOPE §1). Sweepable-knob inventory: `SWEEP_KNOBS.md`. Config rationale: `configs/README.md`.

## Stages

`Pipeline` runs seven stages in fixed order, phase-major (one model on GPU at a time):

1. **diarization** — `pyannote` (`speaker-diarization-3.1`, token via `$HF_TOKEN`) or `sortformer` (NVIDIA Sortformer v1 offline EEND in an isolated NeMo venv subprocess, `scripts/sortformer_worker.py`, reached via `$SORTFORMER_VENV_PY`; no default, missing → loud crash). `num_speakers=2`, mono 16 kHz. The shipped best config (`configs/sweep_best_e31_refineplus.yaml`) uses `backend: sortformer` + `sortformer_head_policy: merge` (surplus-head runs re-assigned to the top-2 speakers by ECAPA2 match instead of discarded). Recordings longer than `sortformer_long_audio_threshold_s` (default 240 s) are auto-routed to the streaming `sortformer_long_audio_model_id` with a loud warning — v1 is O(T²) in memory and OOMs past ~5-6 min on 12 GB; threshold 0 disables. Streaming v2.1 model ids get the offline very-high-latency preset automatically in the worker.
2. **routing** — split overlap vs solo regions.
3. **enhancement** — ClearerVoice backends `frcrn_se_16k` (interim default, SCOPE §10 q7) and `mossformer_gan_se_16k`.
4. **separation** — runs on overlap fragments only. Default = MossFormer2 matched-128k, `checkpoints/mossformer2/SB/mossformer2_matched_128k_final_42_e46/mossformer2_SB_best_e46.pt` (epoch 45, val_sisdr 17.04). **Gotcha:** the sibling `final_42/` dir holds the e23 checkpoint and is NOT what ships. Dataclass defaults and `configs/default.yaml` must agree (a pin test enforces the consistency, not the literal path); `load_model_for_inference` reads `model_type` from the checkpoint config, so any trained architecture loads. Runs at `separator_sample_rate=8000` (a 16 kHz separator needs `16000` **and** `post_separation_processing.backend: naive`). `separator_backend` (default `repo`) also accepts, for the B1 external-separator experiment (`configs/b1_*.yaml`): `speechbrain` (HF id, cached under `checkpoints/external/speechbrain/`), `clearvoice` (2 s one-pass window, adapter refuses longer input), `sr_corrnet` (its WHAMR checkpoint degenerates on clean input — smoke with noisy audio), `tf_locoformer` (local .pth; vendored at `vendor/tf_locoformer/`), `tiger` (HF id, 16 kHz; `vendor/tiger/`), `mossformer2_dp` (`vendor/mossformer2_dp/`, a different arch from `models/mossformer2`).
5. **post_separation_processing** — VAD mask + optional BWE (`naive` / `ap_bwe`, checkpoint via `$AP_BWE_CHECKPOINT`). Always on (downstream depends on its `_gated` arrays); `backend: naive` applies only the mask. `configs/default.yaml` ships `ap_bwe`, the dataclass default is `naive`.
6. **assembly** — stitch per-speaker streams; ECAPA anchor for speaker identity across pieces.
7. **transcription** — `whisperx` (default, `large-v2`) or `coherex`. Alignment is per-language: `align_model_name=None` lets WhisperX pick its wav2vec2 default (`jonatasgrosman/wav2vec2-large-xlsr-53-polish` for `pl`); English preset `configs/english.yaml`. `coherex` runs Cohere Transcribe in an isolated venv subprocess (`scripts/coherex_worker.py`, `$COHEREX_VENV_PY`, `model_name` = Cohere model id) and loads the model per call — for interactive use, not batch eval.

`PipelineConfig.input_bandlimit_hz` (default 0 = off) band-limits the input at load (decimate to `2*hz`, interpolate back) for the "what if recordings were 8 kHz" ablation (`4000` + `post_separation_processing.backend: naive`); model stages stay at 16 kHz.

`PipelineConfig.deterministic` (default `true`) forces deterministic cuDNN algorithms — the enhancement conv stage is otherwise the only run-to-run nondeterminism source (≈1e-7 noise that WhisperX can amplify into a flipped token). Costs ~2× on that stage; set `false` for faster non-reproducible dev runs.

Configs in `configs/`: `default.yaml` (POC-equivalent), `sweep_best_e31_refineplus.yaml` (shipped best), `p4_fixed_pad.yaml` / `p5_full_length.yaml` (ablation knobs), `b1_*.yaml` (external separators), `english.yaml`. Debug log: `/tmp/asr_pipeline_debug.log` (override `ASR_PIPELINE_DEBUG_LOG`) — survives the WSL stdout bridge dropping. Config serializers mask `diarization.hf_token` as `REDACTED`.

## CLI and batch

```bash
python -m asr_pipeline run --config configs/sweep_best_e31_refineplus.yaml \
    --input rec.wav --write-outputs <eval_root> --set diarization.backend=pyannote
python -m asr_pipeline batch --split clarin_dev --mode no_enh      # or --manifest/--glob/--inputs + --out-root
python -m asr_pipeline score --eval-root <eval_root> --out-dir <csv_dir>
```

- `run`/`batch` call `preflight.py` first — fail-loud env/checkpoint checks before any model load (`num2words` warn-only, SCOPE §10 q2).
- `--set stage.knob=value` (repeatable) goes through `config.apply_overrides`, the one override mechanism; the sweep script imports it.
- `batch.py:run_batch` owns the batch loop: per-recording failure isolation (`failures.csv`, batch continues — SCOPE §4.2), completion sentinel = `metadata.json` in the target subdir (`--force` re-runs), `--mode full|no_sep|no_enh|minimal` presets, and the single GPU-teardown block. `scripts/sweep_pipeline.py` delegates to it while pinning its legacy `transcript_A.txt` sentinel, so completed sweep trees never recompute.
- `Pipeline(config, on_event=...)` emits per-stage timing events (load vs run seconds) that land in `metadata.json` / `run_meta.json`.
- Use `HF_HUB_OFFLINE=1` for eval runs (byte-identical output, avoids hub 504 aborts).

## Evaluation (`eval/`)

Two layers; `evaluate_recording(rec) → ScoreCard`, `evaluate_many` (SQUIM loaded once), `walk_eval_tree` yields a `Recording` per directory under the eval root. There is no L1/DER layer — no valid reference diarization exists (SCOPE §10 q8).

- **L2 audio quality** — intrusive SI-SDR / PESQ-WB / STOI (chunked, median-aggregated, speech-presence filtered) when oracle audio exists; non-intrusive TorchAudio-SQUIM (chunked, mean-aggregated) always.
- **L3 ASR** — cpWER + tcpWER + cpCER per ablation mode (full / no-sep / no-enh), ORC-WER on the mixture baseline; backed by `meeteval`. `metrics.per_fragment_metrics` owns the ORC/MIMO combinatorial guard (long recordings skip those metrics with a printed note instead of hanging).
- **Campaign statistics** — `stats.py`: recording-clustered paired bootstrap, Holm-Bonferroni, Benjamini-Hochberg, micro-averages, strata assignment; `scripts/rescore_stratified.py` is a thin driver over it (fixed-seed golden tests keep them bit-identical).

Notebook helpers: `parse_gt_txt`, `parse_transcript_file`, `cpwer_meeteval`, `orc_wer_meeteval`. Notebooks in `asr/`: `explore_pipeline.ipynb` (per-stage frontend), `evaluate_pipeline.ipynb` (L2 + L3 against the CLARIN oracles), `clarin_fragments.ipynb` / `clarin_subset_review.ipynb` (fragment selection, via `scripts/clarin_fragment_finder.py`).

## CLARIN datasets

- `~/datasets/clarin_gotowy/gotowy/` — debleed eval set with oracle per-speaker channels. Root `<id>.wav` stereo inputs; `debleed/<id>_{L,R}.wav` oracles; `debleed_enhanced/` MossFormerGAN-enhanced oracles; `after_pipeline/<id>_{s1,s2}.wav` pipeline outputs; `transcripts/<id>.txt`; `eval_cache/` cached references.
- `~/datasets/clarin_all_2speakers/` — full 2-speaker download, no oracles. `clarin_download/<id>.wav` (+ `Korpus.csv`, `Korpus_with_filename.csv`); `diarization/<id>.json` pyannote; `enhanced_mossformer/<id>.wav`; `auto_transcription_raw/` and `auto_transcription_enhanced_mossformer/` WhisperX transcripts.

The 141-fragment eval split (23 dev / 118 test) is frozen in `eval/clarin_{dev,test}.txt`; `--split` resolves them through `scripts/eval_harness.load_split`.

## Helper scripts (`scripts/`)

- `run_pipeline_on_recording.py` — one recording in three ablation modes; thin wrapper over `batch.run_batch`.
- `sweep_pipeline.py` — config-arm sweep runner (delegates to `run_batch`).
- `rescore_stratified.py` — campaign statistics driver over `eval/stats.py`.
- `prepare_eval_references.py` — cache enhanced oracles + GT-style transcripts for the eval module.
- `enhance_clarin_debleed.py`, `diarize_clarin_2speakers.py`, `transcribe_clarin_2speakers.py` — batch MossFormerGAN / pyannote / WhisperX over the CLARIN sets.
- `score_fragment_acoustics.py` — acoustic-complexity scorer for the eval fragments (SQUIM, DNSMOS, Brouhaha SNR/C50, WADA-SNR, LUFS, clipping) → `acoustic_scores.csv` + report beside the fragments. Brouhaha runs in an isolated venv (`/tmp/brouhaha_venv`, override `BROUHAHA_VENV_PY`/`BROUHAHA_CKPT`); if absent, falls back to WADA-SNR and says so.
- `compare_asr.py` — fixed-audio ASR-only WhisperX-vs-Cohere swap on the dr_refineplus streams (prereq: `sweep_pipeline.py --configs dr_refineplus`; needs `$COHEREX_VENV_PY`); writes per-fragment bundles + scores under `<eval>/_forensics/asr_compare/`. Cross-architecture WER is reference-seed biased (~±4 pp) — read numbers with `--gt2-root` as well.
- `sortformer_worker.py`, `coherex_worker.py` — the isolated-venv subprocess workers.
