"""Rysunek (ch6) — SQUIM: SI-SDR, PESQ i STOI a wynik ASR potoku (zbiór TEST).

Jeden z trójki rysunków „jakość separacji a wynik ASR" w układzie per-źródło
(decyzja autora 2026-08-28; wcześniejszy `fig_ch6_ladder.py` zastąpiony):
trzy panele = trzy metryki JEDNEGO źródła — estymaty TorchAudio-SQUIM na
obszarach nakładania ocenianych fragmentów. Panel (a) SI-SDR to dawna oś (b)
rysunku drabinowego; (b) PESQ i (c) STOI dokładają jakość percepcyjną i
zrozumiałość — `squim_probe.py` od początku zapisywał je obok `squim_si_sdr`,
tylko `ladder_table.py` ich nie eksponował.

Wszystkie trzy metryki są NIEINTRUZYJNE (estymowane bez sygnału
referencyjnego), liczone na obszarach nakładania transkrybowanych fragmentów i
uśrednione po czasie trwania tych obszarów — konwencja agregacji
`ladder_table.squim_axis` (SI-SDR bierzemy wprost z tabeli drabinowej).

Oś Y, rodziny separatorów i linia „bez separacji" pochodzą z `_ch6_pq_common`,
wspólnego dla całej trójki rysunków PESQ/STOI.

Dane: `<eval>/_forensics/swap_family/squim_probe_test.csv` (osie X) +
`ladder_table_test_r3.csv` (oś Y, rodziny) + `family_test_perfrag.csv`
(linia „bez separacji"). Rysunek można przegenerować, gdy dojdą nowe ramiona.

Usage: python scripts/thesis_figures/fig_ch6_squim_pesq_stoi.py [--split test|dev]
                                                                [--metric cpwer|cpcer]
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _ch6_pq_common import DEFAULT_OUT, SWAP, draw, load_ladder, nosep_value  # noqa: E402

TITLE = "CLARIN_fragments"
PANELS = [
    ("x_squim_sisdr", "(a) SQUIM SI-SDR [dB]"),
    ("squim_pesq", "(b) SQUIM PESQ"),
    ("squim_stoi", "(c) SQUIM STOI"),
]


def squim_axes(split):
    """arm -> {kolumna: średnia ważona czasem trwania nakładań}.

    Ta sama agregacja co `ladder_table.squim_axis`, tylko dla PESQ i STOI.
    """
    p = SWAP / f"squim_probe_{split}.csv"
    acc = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
    with open(p, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            dur = float(r["overlap_s"])
            for col in ("squim_pesq", "squim_stoi"):
                v = r.get(col)
                if v in (None, "", "None"):
                    continue
                cell = acc[r["arm"]][col]
                cell[0] += float(v) * dur
                cell[1] += dur
    return {arm: {c: v[0] / v[1] for c, v in cols.items() if v[1]}
            for arm, cols in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=("test", "dev"))
    ap.add_argument("--metric", default="cpwer", choices=("cpwer", "cpcer"))
    ap.add_argument("--csv", type=Path, default=None,
                    help="tabela drabinowa z osią Y i rodzinami "
                         "(domyślnie ladder_table_<split>_r3.csv)")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    seps = load_ladder(args.split, args.csv)
    ax_vals = squim_axes(args.split)
    for col in ("squim_pesq", "squim_stoi"):
        seps[col] = [ax_vals.get(a, {}).get(col) for a in seps["arm"]]

    nosep = nosep_value(args.split, args.metric)
    stem = f"rys_ch6_squim_{args.split}" + ("" if args.metric == "cpwer" else "_cpcer")
    draw(seps, PANELS, args.metric, nosep, args.out_dir, stem, title=TITLE)

    n_p = seps["squim_pesq"].notna().sum()
    n_s = seps["squim_stoi"].notna().sum()
    print(f"written: {args.out_dir}/{stem}.{{png,pdf}}  "
          f"({len(seps)} separatorów; PESQ {n_p}, STOI {n_s}; "
          f"bez separacji = {nosep:.2f})")


if __name__ == "__main__":
    main()
