#!/usr/bin/env python
"""Single-speaker probe — what does a 2-output SB separator do with ONE speaker?

WHY THIS EXISTS
---------------
Continuous speech separation (CSS) slides a separator over long recordings, so
most windows hold one speaker, not two. A PolSESS SB model has only ever seen
two fully overlapped speakers. This probe measures how such a model treats a
lone speaker: does it put the speaker into one output and leave the other
silent, or split the speaker across both outputs? It is the before/after check
for training with ``data.single_speaker_prob`` / ``data.partial_overlap_prob``.

CONDITIONS (per test item; speaker k alternates sp1/sp2 by row to avoid an SSR bias)
    2spk_full   original test mix — reference two-speaker behaviour
    1spk_clean  dry speaker k alone
    1spk_rev    speaker k dry + its reverb tail (indoor rows only)
    1spk_full   full mix minus the other speaker's layers (speaker k + scene +
                event [+ reverbs]), built by the same layer subtraction as MM-IPC

METRICS (single-speaker conditions; "best" = output with the higher SI-SDR vs speaker k)
    best_sisdr          SI-SDR of the best output vs the dry speaker
    sum_sisdr           SI-SDR of (output 1 + output 2) — high while best_sisdr is
                        low means the speaker was SPLIT across the outputs, not lost
    other_rel_best_db   energy of the other output relative to the best one
    split_frac_10db     fraction of items with other_rel_best_db > -10 dB
    switch_frac         fraction of speech-active 250 ms frames where the other
                        output is louder than the best one (temporal switching)

The first run (2026-09-28, tf_mossformer_s_128k_16kHz_final_42, 400 items) gave
split_frac_10db of 0.23 / 0.28 / 0.27 for clean / reverb / full.

USAGE
-----
    python scripts/single_speaker_probe.py \\
        --checkpoint checkpoints/tf_mossformer/SB/<run>/tf_mossformer_SB_best.pt \\
        --data-root ~/datasets/PolSESS_C_128_16kHz --n 400 --out-dir /tmp/probe
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.inference import load_model_for_inference  # noqa: E402

EPS = 1e-8
FRAME_SECONDS = 0.25


def sisdr(est, ref):
    """Zero-mean SI-SDR in dB (Le Roux et al. 2019), numpy float64."""
    est = est - est.mean()
    ref = ref - ref.mean()
    target = (est @ ref) / (ref @ ref + EPS) * ref
    noise = est - target
    return 10 * np.log10((target @ target + EPS) / (noise @ noise + EPS))


def energy_db(x):
    return 10 * np.log10(np.mean(x ** 2) + 1e-12)


def frame_energy(x, hop):
    n = len(x) // hop
    return np.array([np.mean(x[i * hop:(i + 1) * hop] ** 2) for i in range(n)]) + 1e-12


def single_speaker_metrics(y, target, x, hop):
    """Metrics for a single-speaker input x with dry target, outputs y [2, T]."""
    scores = [sisdr(y[j], target) for j in range(2)]
    best = int(np.argmax(scores))
    other = 1 - best
    rec = {
        "best_sisdr": scores[best],
        "other_sisdr": scores[other],
        "sum_sisdr": sisdr(y[0] + y[1], target),
        "input_sisdr": sisdr(x, target),
        "best_rel_in_db": energy_db(y[best]) - energy_db(x),
        "other_rel_best_db": energy_db(y[other]) - energy_db(y[best]),
    }
    et = frame_energy(target, hop)
    active = et > et.max() * 1e-3
    eb, eo = frame_energy(y[best], hop), frame_energy(y[other], hop)
    rec["switch_frac"] = float(np.mean(eo[active] > eb[active])) if active.any() else np.nan
    return rec, best, other


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True, help="directory containing test/")
    parser.add_argument("--n", type=int, default=400, help="number of test rows (from the top)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-examples", type=int, default=3, help="items to save as wav")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    root = Path(args.data_root).expanduser() / "test"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model, ckpt = load_model_for_inference(args.checkpoint, device=args.device)
    sample_rate = ckpt["config"]["data"].get("sample_rate", 8000)
    hop = int(FRAME_SECONDS * sample_rate)
    print(f"checkpoint epoch={ckpt.get('epoch')} val={ckpt.get('val_sisdr')} sr={sample_rate}")

    df = pd.read_csv(next(root.glob("corpus_*.csv")))

    def load(sub, name):
        wav, sr = torchaudio.load(str(root / sub / name))
        assert sr == sample_rate, f"{name}: {sr} Hz, checkpoint expects {sample_rate} Hz"
        return wav[0].numpy().astype(np.float64)

    @torch.no_grad()
    def separate(x):
        y = model(torch.from_numpy(x).float().to(args.device)[None])[0]
        return y.cpu().double().numpy()[:, : len(x)]

    rows = []
    for i in range(min(args.n, len(df))):
        r = df.iloc[i]
        indoor = pd.notna(r["reverbForSpeaker1"])
        k, o = (1, 2) if i % 2 == 0 else (2, 1)
        spk = {1: load("clean", r["speaker1File"]), 2: load("clean", r["speaker2File"])}
        rev = {}
        if indoor:
            rev = {1: load("sp1_reverb", r["reverbForSpeaker1"]),
                   2: load("sp2_reverb", r["reverbForSpeaker2"])}
        mix = load("mix", r["mixFile"])

        conditions = {
            "2spk_full": mix,
            "1spk_clean": spk[k],
            "1spk_full": mix - spk[o] - (rev[o] if indoor else 0.0),
        }
        if indoor:
            conditions["1spk_rev"] = spk[k] + rev[k]

        for cond, x in conditions.items():
            y = separate(x)
            rec = {"idx": i, "cond": cond, "indoor": indoor}
            if cond == "2spk_full":
                perm_a = (sisdr(y[0], spk[1]) + sisdr(y[1], spk[2])) / 2
                perm_b = (sisdr(y[0], spk[2]) + sisdr(y[1], spk[1])) / 2
                baseline = (sisdr(x, spk[1]) + sisdr(x, spk[2])) / 2
                rec["best_sisdr"] = max(perm_a, perm_b)
                rec["sisdri"] = rec["best_sisdr"] - baseline
            else:
                metrics, best, other = single_speaker_metrics(y, spk[k], x, hop)
                rec.update(metrics)
                if cond != "1spk_clean":  # dry input has no mixture baseline
                    rec["sisdri"] = rec["best_sisdr"] - rec["input_sisdr"]
                if i < args.n_examples:
                    for j, label in ((best, "best"), (other, "other")):
                        torchaudio.save(str(out / f"ex{i:03d}_{cond}_{label}.wav"),
                                        torch.from_numpy(y[j]).float()[None], sample_rate)
                    torchaudio.save(str(out / f"ex{i:03d}_{cond}_input.wav"),
                                    torch.from_numpy(x).float()[None], sample_rate)
            rows.append(rec)
        if i % 50 == 0:
            print(f"{i}/{args.n}", flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(out / "per_item.csv", index=False)

    pd.set_option("display.width", 200)
    cols = ["best_sisdr", "sisdri", "sum_sisdr", "other_rel_best_db", "switch_frac"]
    print("\nmedians:")
    print(d.groupby("cond")[cols].median().round(2).T)
    summary = []
    for cond in ("1spk_clean", "1spk_rev", "1spk_full"):
        g = d[d.cond == cond]
        if g.empty:
            continue
        summary.append({
            "cond": cond,
            "n": len(g),
            "split_frac_10db": (g.other_rel_best_db > -10).mean(),
            "split_frac_3db": (g.other_rel_best_db > -3).mean(),
            "switch_gt_0.1": (g.switch_frac > 0.1).mean(),
        })
    summary = pd.DataFrame(summary)
    summary.to_csv(out / "summary.csv", index=False)
    print("\nsplitting:")
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
