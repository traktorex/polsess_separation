"""Rysunek (ch6) — w domenie (PolSESS_128k): SI-SDRi, PESQ i STOI a wynik ASR.

Jeden z trójki rysunków „jakość separacji a wynik ASR" w układzie per-źródło
(decyzja autora 2026-08-28; wcześniejszy `fig_ch6_ladder.py` zastąpiony):
  * `fig_ch6_oracle_pesq_stoi.py` — intruzyjnie, na realnych nakładaniach CLARIN;
  * `fig_ch6_squim_pesq_stoi.py`  — nieintruzyjnie, na audio wdrożeniowym;
  * ten                           — intruzyjnie, na syntetycznym korpusie IN-DOMAIN.

Panel (a) SI-SDRi (dawna oś (c) rysunku drabinowego) to oś, która zawodzi jako
predyktor: checkpointy trenowane na C_new_64 osiągają tu dobre wartości i
katastrofalne cpWER. Panele (b) PESQ i (c) STOI pokazują, że ślepota nie jest
własnością metryki: żadna z nich jej nie naprawia.

PROTOKÓŁ. `PolSESS_C_final_128_v2/test`, 8 kHz natywnie (bez konwersji częstotliwości),
zadanie SB, `batch_size=1`, permutacja PIT, PESQ wąskopasmowe — czyli protokół
`evaluate.py` bez zmian, wywołany z `--max-samples 400` na każdy z ośmiu wariantów
MM-IPC. Podzbiór 400 jest zagnieżdżony w zamrożonym podzbiorze 2000 osi SI-SDRi
(`PolSESSDataset._filter_metadata` bierze `.head(max_samples)` bez losowania), więc
ramiona są dokładnie sparowane, a oś SI-SDRi obok pozostaje nietknięta.

Oś X to ŚREDNIA NIEWAŻONA po ośmiu wariantach MM-IPC — ta sama konwencja co
`ladder_table.unweighted_all8`, wymuszona tym, że manifest testowy podwaja wariant C.

OSTRZEŻENIE O PORÓWNYWALNOŚCI dziedziczone po osi SI-SDRi: PolSESS jest w domenie
dla checkpointów tego repozytorium i poza domeną dla ośmiu separatorów zewnętrznych,
które nie widziały polskiej mowy ani miksrur MM-IPC. Ta oś mówi „jak daleko
transferuje gotowy separator", nie „która architektura jest lepsza".

Dane: `<eval>/_forensics/swap_family/ladder_pq/<label>__polsess128_test_sub400_pq.csv`
+ `ladder_table_<split>_r3.csv` + `family_<split>_perfrag.csv`.

Usage: python scripts/thesis_figures/fig_ch6_indomain_pesq_stoi.py [--split test|dev]
                                                                   [--metric cpwer|cpcer]
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _ch6_pq_common import DEFAULT_OUT, SWAP, draw, load_ladder, nosep_value  # noqa: E402

PQ_DIR = SWAP / "ladder_pq"
TITLE = "PolSESS_128k test"
PANELS = [
    ("x_indomain_all8", "(a) SI-SDRi [dB]"),
    ("pesq", "(b) PESQ"),
    ("stoi", "(c) STOI"),
]


def unweighted(label):
    """{pesq, stoi, pesqi, stoii}: średnia nieważona po ośmiu wariantach MM-IPC."""
    p = PQ_DIR / f"{label}__polsess128_test_sub400_pq.csv"
    if not p.exists():
        return {}
    cols = {k: [] for k in ("pesq", "stoi", "pesqi", "stoii")}
    with open(p, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if not r.get("variant"):
                continue
            for k in cols:
                if r.get(k) not in (None, "", "None"):
                    cols[k].append(float(r[k]))
    return {k: float(np.mean(v)) for k, v in cols.items() if v}


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
    vals = {lab: unweighted(lab) for lab in seps["label"]}
    for col in ("pesq", "stoi"):
        seps[col] = [vals.get(lab, {}).get(col) for lab in seps["label"]]

    nosep = nosep_value(args.split, args.metric)
    stem = f"rys_ch6_indomain_{args.split}" + ("" if args.metric == "cpwer" else "_cpcer")
    draw(seps, PANELS, args.metric, nosep, args.out_dir, stem, title=TITLE)

    n_p, n_s = seps["pesq"].notna().sum(), seps["stoi"].notna().sum()
    print(f"written: {args.out_dir}/{stem}.{{png,pdf}}  "
          f"({len(seps)} separatorów; PESQ {n_p}, STOI {n_s}; "
          f"bez separacji = {nosep:.2f})")


if __name__ == "__main__":
    main()
