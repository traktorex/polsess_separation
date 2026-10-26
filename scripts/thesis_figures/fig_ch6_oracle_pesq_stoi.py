"""Rysunek (ch6) — sonda oracle (kanały debleed): SI-SDRi, PESQ i STOI a wynik ASR.

Jeden z trójki rysunków „jakość separacji a wynik ASR" w układzie per-źródło
(decyzja autora 2026-08-28; wcześniejszy `fig_ch6_ladder.py` zastąpiony):
trzy panele = trzy metryki JEDNEGO źródła — intruzyjny pomiar wobec referencji,
czyli kanałów debleed zbioru CLARIN „gotowy". Panel (a) SI-SDRi to dawna oś (a)
rysunku drabinowego; (b) PESQ i (c) STOI dokładają jakość percepcyjną i
zrozumiałość.

Osie X pochodzą z `real_sep_probe.py` (SI-SDRi) i
`real_sep_probe_intrusive.py` (PESQ/STOI) — ta sama sonda: 90 czterosekundowych
okien z 6 nagrań, 8 kHz, przypisanie estymat do referencji permutacją optymalną
w sensie SI-SDR.

CO TE LICZBY ZNACZĄ. Referencją są kanały *debleed* — nagrania krawatowe po
tłumieniu przesłuchu, nie czysta mowa studyjna. Bezwzględne PESQ jest przez to
zaniżone dla KAŻDEGO ramienia, także dla hipotetycznego separatora idealnego.
Wartości są porównywalne MIĘDZY ramionami i nie należy ich zestawiać z PESQ
publikowanym w literaturze.

Dane: `<eval>/_forensics/swap_family/real_sep_probe_intrusive_summary.json` +
`ladder_table_<split>_r3.csv` (oś Y, rodziny, mapowanie ramię→checkpoint) +
`family_<split>_perfrag.csv`.

Usage: python scripts/thesis_figures/fig_ch6_oracle_pesq_stoi.py [--split test|dev]
                                                                 [--metric cpwer|cpcer]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _ch6_pq_common import DEFAULT_OUT, SWAP, draw, load_ladder, nosep_value  # noqa: E402

TITLE = "CLARIN_oracle"
PANELS = [
    ("x_real_sisdri", "(a) SI-SDRi [dB]"),
    ("pesq", "(b) PESQ"),
    ("stoi", "(c) STOI"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=("test", "dev"))
    ap.add_argument("--metric", default="cpwer", choices=("cpwer", "cpcer"))
    ap.add_argument("--csv", type=Path, default=None,
                    help="tabela drabinowa z osią Y i rodzinami "
                         "(domyślnie ladder_table_<split>_r3.csv)")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    summary_path = SWAP / "real_sep_probe_intrusive_summary.json"
    if not summary_path.exists():
        raise SystemExit(f"brak {summary_path} — uruchom najpierw real_sep_probe_intrusive.py")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    seps = load_ladder(args.split, args.csv)
    for col in ("pesq", "stoi"):
        seps[col] = [summary.get(lab, {}).get(col) for lab in seps["label"]]

    nosep = nosep_value(args.split, args.metric)
    stem = f"rys_ch6_oracle_{args.split}" + ("" if args.metric == "cpwer" else "_cpcer")
    draw(seps, PANELS, args.metric, nosep, args.out_dir, stem, title=TITLE)

    n_p, n_s = seps["pesq"].notna().sum(), seps["stoi"].notna().sum()
    print(f"written: {args.out_dir}/{stem}.{{png,pdf}}  "
          f"({len(seps)} separatorów; PESQ {n_p}, STOI {n_s}; "
          f"bez separacji = {nosep:.2f})")


if __name__ == "__main__":
    main()
