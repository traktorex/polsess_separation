"""Golden-schema + append-guard tests for evaluate_all.

These are the regression guards for survey gaps 1 and 2: the exact eval-CSV
column tuple is pinned here (so a shape change to flatten_results can never
again silently corrupt evaluation_results.csv), and the schema guard is checked
to divert on mismatch instead of appending a second header.
"""

import csv
from pathlib import Path

from evaluate_all import FLATTEN_COLUMNS, flatten_results, resolve_output_target


# The pinned schema. If flatten_results legitimately changes shape, update BOTH
# this tuple and FLATTEN_COLUMNS in the same commit — that is the point.
EXPECTED_COLUMNS = (
    "model_type", "task", "run_name",
    "epoch", "val_sisdr", "num_params", "checkpoint_path",
    "avg_sisdr", "avg_sisdri",
    "si_sdr_SER", "si_sdri_SER", "si_sdr_SR", "si_sdri_SR",
    "si_sdr_ER", "si_sdri_ER", "si_sdr_R", "si_sdri_R",
    "si_sdr_SE", "si_sdri_SE", "si_sdr_S", "si_sdri_S",
    "si_sdr_E", "si_sdri_E", "si_sdr_C", "si_sdri_C",
    "lr", "batch_size", "epochs", "optimizer", "scheduler", "grad_clip",
    "train_segment_length", "train_sample_rate",
    "model_config", "evaluated_at",
    "git_sha", "git_dirty", "torch_version",
    "eval_dataset_name", "eval_data_root", "eval_subset", "eval_batch_size",
)


def _fake_inputs():
    info = {
        "model_type": "mossformer2",
        "task": "SB",
        "run_name": "some_run",
        "epoch": 42,
        "val_sisdr": 12.3,
        "checkpoint_path": "checkpoints/mossformer2/SB/some_run/mossformer2_SB_best.pt",
        "model_config": {"N": 256},
        "training_config": {"lr": 3e-4, "batch_size": 2},
        "data_config": {"segment_length": 32000, "sample_rate": 8000},
    }
    variant_results = {
        v: {"si_sdr": 10.0, "si_sdri": 5.0}
        for v in ("SER", "SR", "ER", "R", "SE", "S", "E", "C")
    }
    eval_context = {
        "git_sha": "abc1234",
        "git_dirty": True,
        "eval_data_root": "/data/PolSESS_C_new_64",
        "eval_subset": "test",
        "eval_batch_size": 1,
    }
    return info, variant_results, eval_context


def test_flatten_columns_constant_is_pinned():
    assert FLATTEN_COLUMNS == EXPECTED_COLUMNS


def test_flatten_results_keys_match_schema_exactly():
    info, variant_results, eval_context = _fake_inputs()
    row = flatten_results(info, num_params=26_400_000,
                          variant_results=variant_results, eval_context=eval_context)
    assert tuple(row.keys()) == FLATTEN_COLUMNS


def test_flatten_results_renames_train_columns_and_adds_provenance():
    info, variant_results, eval_context = _fake_inputs()
    row = flatten_results(info, num_params=1,
                          variant_results=variant_results, eval_context=eval_context)
    # Renamed (gap 2): train-derived, never confused with eval dataset.
    assert "train_segment_length" in row and "train_sample_rate" in row
    assert "segment_length" not in row and "sample_rate" not in row
    assert row["train_segment_length"] == 32000
    assert row["train_sample_rate"] == 8000
    # Provenance values wired through.
    assert row["git_sha"] == "abc1234"
    assert row["git_dirty"] is True
    assert row["eval_dataset_name"] == "PolSESS_C_new_64"  # = Path(data_root).name
    assert row["eval_data_root"] == "/data/PolSESS_C_new_64"
    assert row["eval_subset"] == "test"
    assert row["eval_batch_size"] == 1
    assert row["torch_version"]  # non-empty


def _write_csv(path, header):
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(header)


def test_guard_returns_same_path_when_absent(tmp_path):
    out = tmp_path / "results.csv"
    assert resolve_output_target(out, FLATTEN_COLUMNS) == out


def test_guard_returns_same_path_when_schema_matches(tmp_path):
    out = tmp_path / "results.csv"
    _write_csv(out, FLATTEN_COLUMNS)
    assert resolve_output_target(out, FLATTEN_COLUMNS) == out


def test_guard_diverts_on_schema_mismatch(tmp_path):
    out = tmp_path / "results.csv"
    # A legacy 26-col-era header (missing avg_sisdri and per-variant si_sdri cols).
    _write_csv(out, ["model_type", "task", "run_name", "avg_sisdr"])
    target = resolve_output_target(out, FLATTEN_COLUMNS)
    assert target != out
    assert target.parent == out.parent
    assert target.name.startswith("results_")
    assert target.suffix == ".csv"
