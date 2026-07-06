# Life-1 Code Audit — ASR Subproject

**Date:** 2026-07-06 · **Scope:** `asr_pipeline/` (11.7k lines), ASR-related `scripts/` (~8k lines), `asr/` notebooks · **Method:** 8 parallel Opus reviewers (5× maintainability over package partitions, 1× scripts, 1× archaeologist cruft-sweep + coupling map, 1× notebook hygiene), each briefed on SCOPE.md and the campaign history, findings cross-checked and merged. Correctness was NOT re-audited (covered by the 2026-06-09 and 2026-06-11 audits and the 2026-07-01 campaign verification).

## Verdict

**The code is not an embarrassment risk — the opposite: reviewers independently called it "unusually disciplined for machine-generated code."** No secrets, no tracked `__pycache__`, zero committed notebook outputs/tracebacks, exactly one TODO in the whole package (and it's a documented ledger entry), exemplary vendoring hygiene in `vendor/ap_bwe/`, consistent SCOPE §4 fail-loud discipline, and single-source-of-truth normalization in the thesis-critical metric path. The eval subpackage's L1/DER retirement left literally zero dead code.

The real findings cluster into two themes, both consequences of the campaign's (deliberate) config-gated-lever process:

1. **Campaign sediment (~850–900 lines of swept-and-rejected levers still wired in).** SCOPE §6 explicitly authorizes deleting these now that the thesis decision is written down ("losing branches may be deleted — git remembers them").
2. **Documentation drift.** Docstrings and one supervisor-facing notebook still describe the pre-June pipeline (pyannote + SepFormer) instead of the shipped one (Sortformer + MossFormer2), and two backends (`coherex`, `zipenhancer_16k`) are missing from the docs that enumerate backends.

Neither theme is a correctness problem. Both are exactly what a supervisor would trip on when reading, so both are worth a cleanup pass before closing Life 1.

---

## A. Fix before closing Life 1

### A1. `presentation_figures.ipynb` describes a pipeline that no longer exists — HIGH, effort S
The Slajd-3 diagram (cell 7) hardcodes "Diaryzacja (pyannote)" and "Separacja (SepFormer)"; cell 3 reads a retired sweep-arm dir (`sweep/frcrn_vad_strict/`); cell 11 ships superseded "WSTĘPNE" WER bars (24.7→17.2 vs the final test result). This is the single highest embarrassment-risk artifact because it's a deck *made for supervisors*. Fix the two box labels, repoint or drop cell 3, update or drop cell 11.

### A2. Delete the swept-and-rejected levers (campaign sediment) — HIGH, effort M–L, author's call per SCOPE §6
All of these lost their campaigns, are absent from the shipped config (`sweep_best_e31_refineplus.yaml`), and have written-down verdicts. Deleting them shrinks the package by ~850 lines, roughly halves `_run_sortformer` and `_assign_overlaps`, and removes an entire stage from the orchestrator. Each deletion also takes its config fields, `__post_init__` cross-checks, and tests:

| Lever | Where | Lines | Verdict on record |
|---|---|---|---|
| `FusionDiarizationStage` | `stages/fusion_diarization.py` (whole file) + `FusionConfig` (`config.py:68`) + `pipeline.py:80` wiring + cross-checks (`config.py:1055–1073`) + `test_pipeline_fusion.py` | ~319 + config | `dr_fuse` net-negative; "B+ relabel captured the win" (`docs/sweep_plan/02_sweep_history.md:218`) |
| 3 rejected overlap-assignment strategies + margin prior | `stages/assembly.py:231–502` (`_continuity_decision`, `_consensus_pairings`, `_cluster2_pairings`, `_local_anchor`) + their branches in `_assign_overlaps:896–1020` + `overlap_assign_min_margin`, `continuity_*`, `assignment_mode`/cluster2 config (`config.py:505–569`) | ~300 | "4 attr arms no-ops"; `_cluster2_pairings` docstring itself says "expected to be a near-no-op". Keep `ecapa_argmax` + `external_pairings` (the shipped B+ path) |
| ERes2NetV2 embedder | `stages/custom_embeddings.py:242–324` (incl. the modelscope `sys.path` surgery) | ~85 | Actively harmful (22.6 cpWER, zh-cn domain mismatch, `02_sweep_history.md:216`); shrink `CUSTOM_EMBEDDING_NAMES` to `("ecapa2",)` |
| Sortformer L2 hysteresis + L3/L4 fallback gates | `stages/diarization.py`: `_hysteresis_mask` (92–119), `pyannote_segmentation_speech` (369–403), `_uncovered_seconds` (406–411), L3/L4 gate block (844–886), `_pyannote_fallback` (911–945) + config `191–211` | ~150 | V41_PREREG drops hysteresis as harmful; shipped config uses `binarization: flat`, `fallback: none` |
| Relabel `run_level`/`run_margin` | `stages/relabel.py:742–760` + `_run_level_flips` (462–517) + config (`config.py:889–908`) | ~75 | Default-off, absent from finalist |

Sub-decisions bundled here: after the assembly deletion, extract what remains of `_assign_overlaps` per-overlap choice into a small helper (mostly dissolves on its own); if the L4 fallback path survives, split the double-duty `_SF_LEAK_WARN_FRAC` constant (`diarization.py:60`) into separate warn/gate thresholds.

**Do NOT prune `scripts/sweep_pipeline.py`'s 235-arm `CONFIGS` registry the same way.** The ledger and `dump_sweep_results.py` hash the literal dict (`_configs_hash`), so deleting arms breaks reproduction of committed result tables. Treat it as a frozen campaign record: stop adding to it, optionally move it to its own clearly-labelled module (`sweep_configs.py`, "append-only campaign registry"), but leave the arms in place.

### A3. Documentation drift — HIGH (cheap, do all of it), effort S
- **Stale "SepFormer" naming** (the separator is MossFormer2 since 2026-06-13, and the loader is architecture-agnostic — say "the separator"): `stages/routing.py` (6×, incl. module docstring), `config.py:221,225,496`, `context.py:110`, `stages/assembly.py` (1×), and `explore_pipeline.ipynb` cells 8/9/14/15 (incl. a plot band literally labelled "pyannote" — same fix, say "diarizer").
- **`coherex` backend missing** from `transcription.py:1–12` module docstring and `TranscriptionConfig` docstring/inline enum comment (`config.py:588–611`) — the validator accepts it, the docs enumerate two backends.
- **`zipenhancer_16k` backend missing** from `enhancement.py:13–24` docstring (which also can't decide with `CLAUDE.md:106` — that line lists removed `mossformer2_se_48k` and omits ZipEnhancer). Also drop the stale MossFormer2-SE clause in the decode-window comment (`enhancement.py:126`).
- **`SWEEP_KNOBS.md:30` says the separator checkpoint is e31; the shipped config + dataclass load e46** (verified 2026-07-06: `sweep_best_e31_refineplus.yaml:77` and `config.py:301` both point at `..._final_42_e46/mossformer2_SB_best_e46.pt`). Update the doc line; the `e31` in the YAML's filename is historical naming, fine to leave.
- **Stale module docstring on `diarization.py:1–12`** ("Ported from asr/archive/… cell 10", pyannote-only) — refresh to name both backends.

---

## B. Worthwhile improvements (MED)

**Package:**
1. **`config.py:991–1398`** — `__post_init__` has 29 near-identical hand-rolled numeric range checks (~400 lines). Add a `_require_range(...)` helper sibling to the existing `_one_of`; roughly halves the method. Effort M. (Deleting A2's config fields first shrinks it further.)
2. **`transcription.py:486–741`** — the three winning retry methods (`_retry_collapsed`, `_retry_loops`, `_retry_phrase_loops`) each copy the slice→re-transcribe→offset-splice skeleton; a guard fix currently needs 2–3 edits. Extract `_retranscribe_span(s0, e0, ngram_override=None)`; keep each detector's accept-guard inline. Effort M.
3. **Embedding plumbing triplication** — the "slice intervals → concat → embed → drop-if-unreliable" pattern exists in `diarization.py:240–270`, `relabel.py:98–127`, `relabel.py:130–174` (assembly has a cousin); the ECAPA2 load→use→`del`/`gc`/`empty_cache` dance is copied in `diarization.py:808–823`, `relabel.py:542–567`, `assembly.py:1229–1235`. One `embed_intervals(...)` helper + a `with_ecapa2(device)` context manager in `custom_embeddings.py` collapses six copies. Effort M.
4. **CER naming overload** — in `eval/metrics.py` the `_multistream` suffix means "multi-stream hypothesis" for WER but "re-optimized at char level" for CER, while `_meeteval`-suffixed CER variants also take multi-stream input. `rescore_stratified.py` uses the more permissive `orc_cer_multistream` for thesis numbers, and the names hide that axis. Rename the char-reoptimized pair to `orc_cer_charopt`/`mimo_cer_charopt`. Effort M (rename + call sites).
5. **Dead eval exports** — `transcript_parser.py:236` (`format_untimed_gt`), `:248` (`concat_utterances`) have zero callers repo-wide; `metrics.py:547` (`mimo_cer_multistream`) self-describes as "kept for one-off checks" (speculative-keep, against SCOPE §7). Delete all three and sync `eval/__init__.py.__all__`, which currently exports unused CER variants while the actually-used one is imported directly from `metrics` by scripts. Effort S.
6. **`layer2.py:256–261` `TODO(E5)`** — a live TODO in L2 metric code admitting non-finite samples flow into SI-SDR/PESQ/STOI and get silently nan-dropped. Either resolve (detect + fail loud, per SCOPE §4) or move the decision into SCOPE's open-questions ledger so it reads as a recorded decision rather than a loose end. Effort M.
7. **Near-homonym anchor knobs** — `config.py:457` `min_solo_for_anchor_s` (diagnostic-only) vs `:465` `anchor_min_duration_s` (the real fallback trigger); SWEEP_KNOBS already has to warn about the confusion. Rename the diagnostic one (e.g. `weak_anchor_warn_below_s`). Effort S.
8. **`separation.py:577–738`** — `run()` is ~160 lines / eight concerns; extract the self-contained emit-boundary reconciliation block (677–706) into `_reconcile_emit_boundary(...)`. Effort M. Same medicine available for `_run_sortformer` (`diarization.py:729–909`) if A2 doesn't already dissolve it.
9. **Enhancement backend duplication** — `_ClearVoiceBackend.enhance` vs `_ZipEnhancerBackend.enhance` are ~30 near-identical lines; hoist a shared resample-roundtrip wrapper. Also promote the 4× reimplemented trim-or-pad primitive (`separation.py:307`, `enhancement.py:193,287`, `post_sep._match_length`) to one shared util. Effort M+S.

**Scripts:**
10. **Per-fragment metric loop re-implemented 3×** — `rescore_stratified.py:122–165`, `dump_sweep_results.py:134–193`, `sweep_pipeline.py:1842–1874` (dump's docstring admits it "mirrors rescore exactly"). Hoist one `per_fragment_metrics(...)` into `asr_pipeline.eval`. Effort M.
11. **Split-loader + eval-root quintuplication** — the `clarin_<split>.txt` loader and the `~/datasets/eval/clarin_fragments` root are pasted into 5 scripts (3 with no CLI override). One shared `load_split()` + env-overridable root helper. Effort M.
12. **Bootstrap kernel ×4** — `rescore_stratified.py:209–284,344–371` copy the resample/nan-guard/percentile block four times; extract one resampler taking a `stat(sample)->float` callback. Effort M. `load_purity()` is byte-identical in two scripts — same fix. Effort S.
13. **`compare_asr.py` defaults** still hardwire `dr_refineplus` as the base config; add a docstring note that it's the *fixed-audio base for the ASR swap*, not the current best. Effort S.

---

## C. Disk hygiene & cosmetics (LOW)

- `rm -rf asr_pipeline/vendor/mpsenet/` (untracked `__pycache__` only; source removed 2026-06-11; nothing references it). Also delete the three stale orphan `.pyc` files: `eval/__pycache__/{layer1,edacc}*.pyc`, `stages/__pycache__/super_resolution*.pyc`.
- `asr_pipeline/configs/frcrn_vadstrict.yaml` — zero references anywhere; sweep-era comparison config. Delete or move to an archive dir.
- `asr/Korpus.csv` (119 KB) — untracked orphan, no notebook reads it (they read the copy in `~/datasets/clarin_all_2speakers/`). Remove.
- `asr/__init__.py` — zero refs, `asr/` is notebooks-only now. Remove (or keep deliberately and say why).
- `config.py:427` — `AP_BWE_CHECKPOINT` fallback hardcodes `/home/user/AP-BWE/...`; env-guarded and documented, but it's the one leak of SCOPE §2's "paths belong in configs". Move the literal into `default.yaml`.
- `clarin_subset_review.ipynb` — "132 selected fragments" prose vs the frozen 141-frag eval set; add one line clarifying it browses the *candidate* pool. Consider an "archival — eval set frozen 2026-06-29" banner on both fragment-selection notebooks.
- `evaluate_pipeline.ipynb` cell 0 — layout doc leads with `mixture.wav`; package now prefers `<id>.wav`. One line.
- `assembly.py:1275–1278` — leftover `_log("run: entered")` stall-diagnosis instrumentation from a resolved investigation; drop or fold into the start log line.
- `run_pipeline_on_recording.py:76–86` — `MODES` mixes two lambdas with two named helpers; name all four, and confirm/drop the unscored `pipeline_nosep_mossformer` bootstrap mode.
- Import-style inconsistency between scripts (`from polish_scoring_normalizer import` vs `from scripts.rescore_stratified import`) — standardize on the `scripts.`-prefixed form.
- Minor singles: inline `_target_essentially_silent` (`layer2.py:198`), move `_SILENCE_FLOOR_DB` (`layer2.py:219`) up to the constants block, dedupe the mode-dir tuple in `layer3.py:59,176`, drop the redundant `soundfile` re-import in `enhancement.py:300`, drop "(default)" from the `whisper` backend docstring line (`config.py:611`) or note the shipped default is whisperx.

### Script archive pass
Of ~34 ASR scripts, **21 are ALIVE, 13 are campaign artifacts, 0 are orphans.** The 13 done-their-job artifacts could move to `scripts/archive/` with a one-line README (GT freeze / candidate mining / concluded sweep arms): `mine_clarin_candidates`, `build_candidate_report`, `build_cohere_seed_kit`, `pre_annotate_fragments`, `build_nzr_aids`, `build_gt_review_inputs`, `build_gt_review_index`, `sync_gt_from_drive`, `sweep_asr_prompt`, `sweep_asr_decode`, `batch_pipeline_noenh`*, `build_libricss_2spk_slim`, `transcripts_to_annotation`**.
· \*`batch_pipeline_noenh` still writes the `pipeline_noenh/` dir the L3 no-enh table reads — retire only after confirming that consumer moved.
· \*\*`transcripts_to_annotation` writes `.eaf`/`.srt` — the natural seed for SCOPE §9's Life-2 exporter; archive, don't delete.
Also: `transcribe_mixtures.py` is ALIVE (ORC-WER mixture-baseline producer) but undocumented — add it to CLAUDE.md's script list.

---

## D. Protected list — things reviewers explicitly said NOT to "simplify"

- `vendor/ap_bwe/` vendoring hygiene (license + README + modification list) — the standard, don't touch.
- `Stage` ABC (`base.py`) and `pipeline.py`'s `_release_current`/`_ensure_loaded`/`load_signature` one-model-on-GPU bookkeeping.
- SCOPE §4 boundary rejection in transcription (`_reject_unsupported_knobs`, loop-retry load guard) and the min-sample passthrough gates — deliberate fail-loud, not boilerplate.
- `_normalize_text` + fold constants (`eval/metrics.py:82–217`), `_iter_speech_chunks` (`layer2.py:78`), `_cer_under_routing` (`metrics.py:619`) — single sources of truth for thesis numbers.
- `text_metrics.py` (stdlib-only by design, shared by scanner + pipeline), `redact_config_snapshot`, the `OverlapSeparated` TypedDict contract doc, `transcript_format`'s timestamp sanitisation seam.
- Pure permutation logic: `fuse_labels`/`_align_*` (keep even if fusion stage is cut — move next to relabel if needed), `_lloyd`.
- `sortformer_worker.py`'s isolation contract comments and `coherex_worker.py`'s `_unstub_tf` workaround (documented, load-bearing).
- Sweep provenance discipline (`_configs_hash`, `_gt_snapshot_hash`, append-only ledger, `_check_frag_parity`) and the "diagnostic, non-clustered" honesty labels.
- `unload()` CUDA-deadlock instrumentation in `separation.py:45–56` — documented deliberate diagnostic.
- `debug_log.py`'s per-line `fsync` (WSL stdout-bridge survival, documented).

---

## E. Repo split: recommendation — **stay put until the thesis is submitted; split at Life 2**

The coupling map makes this an easy call, and in an unexpected direction: **the split is already cheap, which is precisely why there's no reason to do it now.**

Facts (verified by the archaeologist pass):
- **Exactly one import tendril** ties the package to the parent: `separation.py:531` → `utils.model_utils.load_model_for_inference` → the `models/` registry (plus the same import in `scripts/build_nzr_aids.py`). Nothing else in `asr_pipeline/**` imports parent code.
- **Zero reverse coupling in production code** — no `train.py`/`evaluate.py`/`models/`/`config.py` imports `asr_pipeline`. Only the 23 `tests/test_pipeline_*.py` files couple back, and they'd move with the package.
- The rest is co-location, not coupling: the separator checkpoint lives in the parent's `checkpoints/` tree, and one `tests/` dir mixes both sides.

Why stay for Life 1:
1. **The move cost doesn't grow.** The boundary is already SCOPE §7-clean; waiting doesn't tangle it. Nothing is being bought by moving early.
2. **The campaign state is frozen and verified** (GT sha256 snapshot, ledger hashes, byte-identical reruns under `HF_HUB_OFFLINE=1`). A move churns paths, imports, the test split, and the checkpoint pointer — invalidating exactly the reproducibility guarantees you just paid to verify, weeks before hand-in.
3. **Thesis reproducibility favors one repo:** one commit hash covers separator training *and* the pipeline that consumes its checkpoint; supervisors review one place.
4. Your original stated advantage (using separation models without vendoring) remains real: the checkpoint's `model_type` is resolved through the live registry, so today the pipeline can load any architecture you trained without copying code.

At Life 2, the split becomes the right move, and the recipe is already written: vendor `load_model_for_inference` + the `mossformer2` model class (pure PyTorch, cross-platform — the only architecture the shipped checkpoint needs), carry/repoint the checkpoint (3 config path references), move the 23 pipeline tests, and take the ALIVE ASR scripts + `docs/sweep_plan/` along. The parent repo loses nothing.

One cheap thing worth doing *now* to make the future split trivial: keep the single-tendril discipline — any new pipeline code that wants something from the parent repo should go through `load_model_for_inference` or not at all.

---

## Suggested execution order

1. **A1 + A3** (an afternoon): presentation notebook + all doc-drift renames. Highest embarrassment-reduction per minute.
2. **C disk hygiene** (30 min): dead vendor dir, orphan pycs/CSV/config, script archive pass.
3. **A2 lever deletion** (1–2 days incl. test pruning + a full `pytest` + one pipeline rerun on a dev fragment to confirm byte-identical output of the shipped config): decide per-lever; every candidate has a written verdict on record.
4. **B items opportunistically** — none block Life 1; B1/B2/B3 (range-check helper, retry dedup, embedding helpers) give the most future-maintenance value if Life 2 happens; B4–B6 (CER naming, dead exports, TODO(E5)) are the ones a supervisor reading eval code would most plausibly notice.
