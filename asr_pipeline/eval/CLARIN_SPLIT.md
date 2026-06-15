# CLARIN dev/test split — FROZEN

Frozen 2026-06-15 (author-authorized). Supersedes the PROPOSED
`~/datasets/eval/clarin_fragments/SPLIT_PROPOSAL.md`. Machine-readable companions:
`clarin_split.csv` (frag_id, role, unit, Autor, Sesja, pool), `clarin_dev.txt`,
`clarin_test.txt`.

## The split
- **dev = 23** — the exact fragments every config sweep ran on (the `f_oa03` /
  `sweep_best` results in `SWEEP_FINAL.md`). All tuning happens here.
- **test = 124** — every other fragment in the eval manifest. Scored **once**,
  with frozen configs, after the single finalist is locked. Do NOT tune,
  eyeball-rank, or condition any pipeline decision on test.
- Source of truth: `~/datasets/eval/clarin_fragments/manifest.csv` (147 rows);
  role = dev for the 23, test for the remaining 124.

## Disjointness (verified 2026-06-15)
Unit = (Autor, Sesja) when Sesja is set, else the recording id. **No unit spans
dev and test** (20 dev units, 70 test units, 0 overlap) — no recording/session
leaks across the split. Residual risk (per the proposal): an author *account* may
appear on both sides via different sessions (its anchor speaker recurs) — accepted,
because author-level disjointness starved dev of hard material; conversation
*partners* across accounts are unverifiable from metadata.

## Reconciliation note (for the author)
- The earlier SPLIT_PROPOSAL counted eval = 143 (test = 120); this freeze uses the
  full manifest (147 → **test = 124**). The +4 delta is unreconciled manifest rows
  beyond the proposal's count — including them in test is safe (they stay held-out);
  trim them if you want test to match the proposal's 143 exactly.
- `pool=backchannel` is NOT used as an eval-exclusion here: two dev fragments
  (`94a0d89a__seg00`, `48fbaab6__seg00`) are pool=backchannel yet are in dev and
  score normally — so the pool column does not gate eval membership.

## Protocol
Sweep/tune on **dev only** (`SWEEP_FINAL.md`). Lock the single finalist, THEN run
it once on **test** and report the number whatever it is. The test number is the
only unbiased estimate — dev numbers are upper-biased (100+ configs scored on dev).
