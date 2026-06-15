# CLARIN dev/test split — FROZEN

Frozen 2026-06-15 (author-authorized). Supersedes the PROPOSED
`~/datasets/eval/clarin_fragments/SPLIT_PROPOSAL.md`. Machine-readable companions:
`clarin_split.csv` (frag_id, role, unit, Autor, Sesja, pool), `clarin_dev.txt`,
`clarin_test.txt`.

## The split
- **dev = 23** — the exact fragments every config sweep ran on (the `f_oa03` /
  `sweep_best` results in `SWEEP_FINAL.md`). All tuning happens here.
- **test = 120** — every other fragment in the eval manifest. Scored **once**,
  with frozen configs, after the single finalist is locked. Do NOT tune,
  eyeball-rank, or condition any pipeline decision on test.
- Source of truth: `~/datasets/eval/clarin_fragments/manifest.csv` (143 rows);
  role = dev for the 23, test for the remaining 120.

## Disjointness (verified 2026-06-15)
Unit = (Autor, Sesja) when Sesja is set, else the recording id. **No unit spans
dev and test** (20 dev units, 66 test units, 0 overlap) — no recording/session
leaks across the split. Residual risk (per the proposal): an author *account* may
appear on both sides via different sessions (its anchor speaker recurs) — accepted,
because author-level disjointness starved dev of hard material; conversation
*partners* across accounts are unverifiable from metadata.

## Reconciliation note (for the author)
- Matches the SPLIT_PROPOSAL exactly: eval = **143** (23 dev / 120 test). The 4
  extra manifest rows from the initial freeze (`442dd69e`, `e14aa22f`, `649991bc`,
  `f7ed6ed8`) were **full-length GT recordings** (counterparts in
  `~/datasets/clarin_gotowy/gotowy/`), not fragments — removed 2026-06-15.
- `pool=backchannel` is NOT used as an eval-exclusion here: two dev fragments
  (`94a0d89a__seg00`, `48fbaab6__seg00`) are pool=backchannel yet are in dev and
  score normally — so the pool column does not gate eval membership.

## Protocol
Sweep/tune on **dev only** (`SWEEP_FINAL.md`). Lock the single finalist, THEN run
it once on **test** and report the number whatever it is. The test number is the
only unbiased estimate — dev numbers are upper-biased (100+ configs scored on dev).
