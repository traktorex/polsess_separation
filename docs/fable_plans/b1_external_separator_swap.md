# B1 — Off-the-shelf separator swap: adapter built, runs pending author decisions

*2026-07-18 (overnight session, author delegated B1/B3 prep). Status: the code
side of B1 is DONE and tested; no experiment has been run. Backlog entry:
`thesis/x_notes/backlog.md` §B1; chapter slot: `[PENDING B1]` in
`thesis/claude-writing/ch07/07_arc.md` §7.8.*

## The question (unchanged from backlog)

Does PolSESS's *acoustic variety* (MM-IPC condition diversity) pay on the
target domain? Pipeline stays byte-identical (v41_merge); only the separator
checkpoint is swapped for an off-the-shelf model trained on standard corpora.
Pre-committed interpretation branches (07_arc §7.8): ours ≫ off-the-shelf ⇒
the training data pays (RQ4 extends to deployment); ours ≈ off-the-shelf ⇒
the pipeline architecture carries the effect (also publishable).

## What was built (all uncommitted, full suite green: 1327 passed)

- **`separation.separator_backend`** config knob (`repo` | `speechbrain` |
  `clearvoice`), default `repo` = shipped behavior, byte-identical; validated
  in `PipelineConfig.__post_init__`; pinned `repo` in the default.yaml pin
  test. `checkpoint_path` semantics now depend on the backend (repo ckpt file
  / HF id / ClearVoice model name).
- **Adapters** in `asr_pipeline/stages/separation.py` presenting the repo
  separator contract `separator([1,T]) → [1,2,T]`:
  - `_SpeechBrainSeparator` — wraps `speechbrain.inference.separation.
    SepformerSeparation` (`separate_batch` returns `[B,T,n_src]` → permute);
    HF weights cache under `checkpoints/external/speechbrain/` (gitignored;
    repo-relative so `HF_HUB_OFFLINE=1` eval runs work once cached).
  - `_ClearVoiceSeparator` — tensor-to-tensor ClearVoice call (enhancement-
    stage idiom); **refuses input beyond the model's one-pass decode window**
    (2 s for MossFormer2_SS_16K) instead of letting ClearVoice's internal
    segmented decode silently substitute for our configured Hann overlap-add
    (SCOPE §4.1); cross-checks declared `separator_sample_rate` against the
    model's own `sampling_rate`.
- **Preflight** knows the backends (file check only for `repo`; import checks
  + a first-download warning for the external ones).
- **Tests**: adapter units with fakes (permute correctness, source-count
  rejection, shape tolerance, over-window refusal), enum-guard row, pin-test
  line. GPU sanity through the real `SeparationStage.load → _separate_single`
  path passed for both backends on a CLARIN fragment (2026-07-18).
- **Staged arm configs** in `asr_pipeline/configs/` — each is the shipped
  best config with ONLY the separator loader block changed:
  `b1_sb_wsj02mix.yaml`, `b1_sb_wham.yaml`, `b1_sb_whamr.yaml`,
  `b1_sb_libri2mix.yaml`, `b1_cv_mossformer2_ss16k.yaml` (caveated, see
  below). All five load through the config loader.

## Verified candidate ladder (research pass 2026-07-18, HF cards + recipes)

| arm | HF source | corpus (degradation) | SI-SNRi (home test) | license | fairness |
|---|---|---|---|---|---|
| b1_sb_wsj02mix | speechbrain/sepformer-wsj02mix | WSJ0-2mix (clean, anechoic) | 22.4 dB | Apache 2.0 | byte-identical pipeline |
| b1_sb_wham | speechbrain/sepformer-wham | WHAM! (+noise) | 16.3 dB | Apache 2.0 | byte-identical pipeline |
| b1_sb_whamr | speechbrain/sepformer-whamr | WHAMR! (+noise+reverb) | 13.7 dB | Apache 2.0 | byte-identical pipeline |
| b1_sb_libri2mix | speechbrain/sepformer-libri2mix | Libri2Mix clean train-360 (bigger, more speakers; recipe cross-checked: no WHAM noise) | 20.6 dB | Apache 2.0 | byte-identical pipeline |
| b1_cv_mossformer2_ss16k | ClearVoice `MossFormer2_SS_16K` | VCTK + **undocumented internal TTS** + LibriTTS, their own noise+reverb | 16.7 dB (Libri2Mix cross-eval) | code Apache 2.0, ckpt unstated | **NOT byte-identical**: 16 kHz + 2 s window geometry |

All four SpeechBrain arms: 8 kHz (same as deployed separator path), one API,
zero new dependencies. The four form a degradation ladder
(clean → noise → noise+reverb → clean-but-diverse) — B1 can read *which kind*
of training-condition mismatch costs the most, not just "ours vs theirs".

**Same-architecture contrast caveat**: MossFormer2_SS_16K is the only
same-arch candidate, but its training data is partly non-public (breaks the
"standard corpora, documented" requirement) and its arm changes pipeline
geometry. Run it only as a disclosed-caveat secondary, or not at all.
**Unverified lead worth a follow-up**: `alibabasglab/mossformer2-librimix-2spk`
(HF, MIT, loads via `transformers.AutoModel`) may be the vendored MossFormer2
architecture trained on standard LibriMix — if verifiable, it would be the
cleanest same-arch arm; corpus variant/SR/metrics undocumented on the card.
**Dead ends**: no official SpeechBrain ConvTasNet/DPRNN separation
checkpoints exist on HF; Asteroid's `JorisCos/ConvTasNet_Libri2Mix_*` work
but need a new `asteroid` dep — only worth it if an *architecture* contrast
is wanted (the SepFormer ladder already covers the corpora).

## Tier 2 — SOTA stress-test candidates (research pass 2, 2026-07-18)

Author directive: beyond the 2021-era SpeechBrain ladder, stress-test the
deployed separator against the strongest currently-downloadable models. Key
proxy metric: **WHAMR! monaural SI-SDRi** (noise+reverb = closest public
analogue to conversational deployment audio; clean WSJ0-2mix leaderboard
numbers are near-useless here).

| candidate | WHAMR SI-SDRi | weights | license | integration |
|---|---|---|---|---|
| **SR-CorrNet-SS** (2026, SepReformer lineage successor) | **19.7 dB** (+ real-recorded LibriCSS wins) | HF `shinuh/sr-corrnet-ss-1ch-whamr` — `model.pt` confirmed present | **NOT STATED** (blocker) | own pip pkg (`SSInference.from_pretrained`, in-memory tensors, 13.6M, 8 kHz) |
| **TF-Locoformer-M** (MERL 2024) | 18.5 dB (beats TF-GridNet 17.1) | GitHub `merlresearch/tf-locoformer` — exact `.pth` acquisition UNCONFIRMED | Apache-2.0 (code) | standalone loader, strip `separator.` prefix, 15M, 8 kHz |
| **mossformer2-whamr-2spk** (alibabasglab) | ~17.0 dB (paper proxy) | HF — ClearerVoice-format files confirmed (masknet/encoder/decoder ckpt) | **MIT** | same family as deployed; likely wire-in via clearvoice checkpoint-dir or the vendored arch + key mapping — **needs a loader probe** |
| TIGER-speech (ICLR 2025, 822K params) | n/a (EchoSet-trained — most realistic data here) | HF `JusperLee/TIGER-speech` | Apache-2.0 | `look2hear` loader; the realistic-data-vs-capacity point |
| SepReformer-B (released -B only, 23.8 dB clean WSJ0) | clean-only | GitHub LFS | Apache-2.0 | the deliberate clean-specialist foil |

Dead ends (verified): no released MONO 2-spk TF-GridNet weights anywhere
(ESPnet's is multichannel/spatialized) — TF-Locoformer is the TF-domain
representative; USES/USES2 = universal enhancement, tangential; external
SPMamba weights exist (JusperLee) but add nothing over the in-repo one.

**Verification probes (2026-07-18, same day):**
- SR-CorrNet HF repo: `model.pt` + `config.yaml` present; HF license unstated
  but the authors' code repo (`dmlguq456/SR_CorrNet_SS`) is **MIT** — treat
  as MIT-intended, confirm via a GitHub issue before the thesis table.
- TF-Locoformer: the WHAMR `.pth` weights ARE in the GitHub tree
  (`egs2/whamr/enh1/exp/enh_train_enh_tflocoformer_raw/valid.loss.ave_5best.pth`,
  medium + small variants, git-LFS), repo Apache-2.0. Weight-acquisition
  question resolved.
- mossformer2-whamr-2spk: **DEMOTED** — key-set probe shows it is a
  *dual-path chunked* MossFormer2 variant (`dual_mdl.intra_mdl…`, chunksize
  250; masknet 55.7M) that does NOT match our vendored full-sequence
  MossFormer2_SS (same sibling front layers, different trunk), and its HF
  config has no `auto_map`, so the advertised `AutoModel` load has no model
  code behind it. Loading requires hunting its actual architecture definition
  — medium effort, uncertain; optional.

**Tier-2 IMPLEMENTED (2026-07-18, author green-lit "get those ready to use"):**
four new `separator_backend` values, all verified loading their real
checkpoints end-to-end:
- `sr_corrnet` — `sr-corrnet-ss` pip pkg installed into the main venv
  (`--no-deps` + `loguru`; every other dep already present). NB their
  `from_pretrained` takes the HF id as `checkpoint_path=`, not positionally
  (upstream docstring is stale). Arms: `b1_sr_corrnet_whamr.yaml` (B, 13.6M —
  the only published WHAMR variant) + `b1_sr_corrnet_wsj_l.yaml` (L-DM 38.1M,
  clean-WSJ leaderboard-ceiling arm; the author asked for L where published).
- `tf_locoformer` — standalone model vendored (`asr_pipeline/vendor/
  tf_locoformer/`, Apache-2.0); WHAMR-medium .pth (15.0M — the paper's
  headline WHAMR model; no L was published for WHAMR) downloaded to
  `checkpoints/external/tf_locoformer/`; adapter owns the STFT/iSTFT
  round-trip (n_fft 256 / hop 64 / hann, per the checkpoint's exp config;
  identity-model round-trip unit-tested). Arm: `b1_tf_locoformer_whamr.yaml`.
- `mossformer2_dp` — the earlier "no model code" demotion is REVERSED: a
  detective pass found the paper authors' own standalone release
  (github.com/alibabasglab/MossFormer2, MIT) whose code matches the
  `mossformer2-whamr-2spk` checkpoint key-for-key (incl. the live
  `norm`/`conv1d_encoder` params that the ClearerVoice AV-TSE cousin comments
  out). Vendored at `asr_pipeline/vendor/mossformer2_dp/` — ONE documented
  patch (removed upstream's `__init__`-time auto-`.to(cuda)`; the stage owns
  device placement). Arm: `b1_mossformer2_whamr.yaml` — the same-family
  standard-corpus contrast the author wanted.
- `tiger` — look2hear's TIGER vendored (`asr_pipeline/vendor/tiger/`,
  Apache-2.0, not on PyPI; one import-path patch), loads
  `JusperLee/TIGER-speech` via its HF mixin. 16 kHz native → the arm config
  sets `separator_sample_rate: 16000` (resample round-trip becomes a no-op).
  Arm: `b1_tiger_echoset.yaml` — the EchoSet different-corpus point.

**Tier-2 GPU smoke (2026-07-18, synthetic PolSESS clean-pair mix @ 0 dB,
best-permutation SI-SDRi through the real adapters):**

| backend arm | SI-SDRi | note |
|---|---|---|
| sb_whamr (tier-1 ref) | +8.2 dB | |
| tf_locoformer_whamr | +17.1 dB | STFT wrapper faithful (also identity-round-trip unit test) |
| mossformer2_dp_whamr | +14.9 dB | |
| tiger_echoset | +18.6 dB | 16 kHz probe |
| sr_corrnet_wsj_l (clean-trained) | +23.5 dB | (B variant +23.0) — adapter/harness control |
| sr_corrnet_whamr | **−1.2 dB clean / +19.1 dB noisy** | see below |

**SR-CorrNet-WHAMR domain-prior finding:** the checkpoint degenerates on
perfectly CLEAN mixtures (≈0 dB SI-SDRi) but scores +19.1 dB the moment the
mix contains noise (10 dB PolSESS scene noise) — WHAMR training inputs always
contain noise, so clean input is out-of-manifold. Not a defect for B1
(CLARIN audio is always noisy/reverberant = in-domain) and a thesis-quotable
mirror of our own separator's 1-speaker degeneracy: models degenerate outside
their training manifold. Any future smoke of this arm must probe with noisy
input.

## Proposed run plan (NOT started — author decides arms first)

1. **Dev (23 frags), no prereg needed**: `scripts/sweep_pipeline.py --configs
   b1_sb_whamr b1_sb_wsj02mix b1_sb_libri2mix --recordings <dev ids>` (b1
   configs live beside the sweep's; first run per arm needs network for the
   HF download — do one warm-up run before batching with `HF_HUB_OFFLINE=1`).
   Score with the standard rescore/stats machinery vs the v41_merge dev
   numbers, like-for-like.
2. **Recommended arm set** (if trimming): `whamr` (degradation-matched
   primary) + `wsj02mix` (clean pole) + `libri2mix` (corpus-diversity pole);
   `wham` adds little over whamr; the CV arm is the author's call.
3. **Test**: only after a fresh prereg (budget spent 2026-07-04; guardrail 12
   in 07_arc §5). Skeleton: primary = cpCER, confirm = cpWER, vs v41_merge
   like-for-like; recording-clustered bootstrap + Holm over the arm family;
   strata LOW/MID/HIGH read-only. One-shot.
4. **Disclosures for the chapter**: English/read-speech training vs Polish
   conversational eval (separation is largely language-agnostic — same
   disclosure pattern as B3's AMI asymmetry); SpeechBrain separators emit
   unnormalised streams (handled by the pipeline's `sum_equals_mix`, which
   these arms inherit unchanged).

## Author decision points

- [ ] Which tier-1 arms run on dev (recommendation above).
- [ ] CV same-arch arm: run with caveat, or drop.
- [ ] Tier-2: green-light SR-CorrNet + TF-Locoformer integration (new deps /
      isolated venvs; ~a session of adapter work)?
- [ ] SR-CorrNet weights license: accept MIT-by-authorship, or ask upstream?
- [ ] mossformer2-whamr-2spk: chase its dual-path model code, or drop?
- [ ] After dev numbers: which arms (if any) go to test → fresh prereg.
