# CLARIN evaluation split — FROZEN 2026-06-29

Canonical dev/test split for the CLARIN 2-speaker ASR evaluation. **Frozen** — do
not change membership without re-versioning. The machine-readable split is
`asr_pipeline/eval/clarin_split.csv` (columns: `frag_id, role, unit, Autor, Sesja,
pool`); this file records *how* it was built.

## Counts
- **141 fragments** = **23 dev** + **118 test**.
- **2 DROPPED** from the prior 143 and removed from the CSV at freeze (dirs parked under `_dropped/`):
  - `4d93b275__seg00` — 3 distinct speakers (the eval assumes 2).
  - `d64c055b__seg00` — audio largely incomprehensible; no reliable GT obtainable.

## Disjointness rule (v2, relaxed)
Dev/test units are disjoint at the **(Autor, Sesja)** level (falling back to the
recording id when `Sesja` is empty), **not** at the author level. Rationale: the
corpus has only ~10 author accounts, and the author groups assigned to dev have no
acoustically-hard material elsewhere in the corpus, so author-level disjointness
would needlessly strand usable test material. Unit- and recording-level disjointness
between dev and test is verified.

## Dev composition (methodology; authoritative membership = `role==dev` in the CSV)
- The 4 smallest author groups (keeps dev small and fully removes the held-out
  authors from test): `64a6a2ec…` (×10), `68179aa6…` (×3), `68493cf9…` (×2),
  `69386555…` (×1) → 16.
- + 6 "S1-mover" fragments promoted to lift dev's acoustic difficulty toward test's
  (all pre-annotated — zero added annotation work).
- + 1 added hard car-noise window `9a651086__seg01`.
- → **23 dev**.

## Acoustic anchoring
Composite difficulty scores are z-scaled against the **original 128-fragment**
statistics (`dnsmos_sig` 2.783 / 0.597, `brouhaha_snr` 16.754 / 10.979). Do **not**
re-anchor — every reported composite assumes this baseline.

## Provenance / see also
- `SELECTION.md` + `acoustic_scores.csv` / `ACOUSTIC_SCORES_REPORT.md` — fragment
  selection rationale and the acoustic audit behind the composite scores.
  (The 2 dropped rows are listed under Counts above.)

This split governs all reported cpWER / cpCER. Per-arm tuning is permitted on dev;
test is the held-out one-shot (task #35).
