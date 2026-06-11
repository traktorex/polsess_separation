"""Download W&B model artifacts (best checkpoints) for the scaling-phase eval set."""

from pathlib import Path
import wandb

ENTITY = "s17060-polsko-japo-ska-akademia-technik-komputerowych"

# (project, run_id, model_type, display_name)
DOWNLOADS = [
    ("polsess-separation-64k", "om6c4kyx", "sepformer", "64k_test_valconfig3"),
    ("polsess-separation-64k", "z0omra18", "sepformer", "64k_baseline_posenc"),
    ("polsess-separation-64k", "swipmxd4", "sepformer", "64k_test_valconfig3"),
    ("polsess-separation-64k", "ovcblvj0", "sepformer", "64k_test_valconfig3"),
    ("polsess-separation-real16k", "1tvoj7dk", "sepformer", "16k_baseline_posenc"),
    ("polsess-separation-32k", "gckob0q1", "sepformer", "32k_baseline_posenc"),
    ("polsess-separation-real16k", "knkhn29o", "spmamba", "16k_test_valconfig1"),
    ("polsess-separation-real16k", "r634a0ua", "sepformer", "16k_test_valconfig3"),
    ("polsess-separation-32k",     "qc0w70hd", "sepformer", "32k_test_valconfig3"),
    ("polsess-separation-64k",     "87lepmzg", "spmamba",   "64k_baseline"),
]


def main():
    api = wandb.Api()
    repo_root = Path(__file__).resolve().parent.parent

    for project, run_id, model_type, display_name in DOWNLOADS:
        target_dir = repo_root / "checkpoints" / model_type / "SB" / f"{display_name}_{run_id}"
        expected_file = target_dir / f"{model_type}_SB_best.pt"
        if expected_file.exists():
            print(f"[skip] already exists: {expected_file}")
            continue

        target_dir.mkdir(parents=True, exist_ok=True)
        run = api.run(f"{ENTITY}/{project}/{run_id}")
        arts = [a for a in run.logged_artifacts() if a.type == "model"]
        if not arts:
            print(f"[warn] no model artifacts in {run_id}")
            continue

        latest = sorted(arts, key=lambda a: a.created_at)[-1]
        print(f"[get ] {run_id} ({latest.name}) -> {target_dir}")
        latest.download(root=str(target_dir))

        # Sanity-check the expected file appeared
        if expected_file.exists():
            size_mb = expected_file.stat().st_size / 1e6
            print(f"       ok ({size_mb:.1f} MB)")
        else:
            inside = list(target_dir.glob("*"))
            print(f"[warn] expected {expected_file.name} not found; got: {[p.name for p in inside]}")


if __name__ == "__main__":
    main()
