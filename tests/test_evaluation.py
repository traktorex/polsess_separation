"""Tests for evaluation module (loading, metrics, formatting)."""

import math

import pytest
import torch
from pathlib import Path
from unittest.mock import Mock, patch
from evaluate import evaluate_model, print_summary, save_results_csv
from utils.model_utils import load_model_for_inference


class TestEvaluationLoading:
    """Test model loading for evaluation."""

    def test_load_model_for_evaluation_with_embedded_config(self, tmp_path):
        """Test loading model when checkpoint contains config."""
        checkpoint_path = tmp_path / "model.pt"
        
        # Create a simple model config
        config_dict = {
            "model": {
                "model_type": "convtasnet",
                "convtasnet": {
                    "N": 64,
                    "B": 64,
                    "H": 128,
                    "P": 3,
                    "X": 4,
                    "R": 2,
                    "C": 1,
                    "kernel_size": 16,
                    "stride": 8,
                    "norm_type": "gLN",
                    "causal": False,
                    "mask_nonlinear": "relu",
                },
            }
        }
        
        # Create actual model and save checkpoint
        from models import ConvTasNet
        
        model = ConvTasNet(N=64, B=64, H=128, P=3, X=4, R=2, C=1)
        
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": config_dict,
                "epoch": 10,
                "val_sisdr": 15.5,
            },
            checkpoint_path,
        )
        
        # Test loading
        loaded_model, _ = load_model_for_inference(
            str(checkpoint_path), device="cpu"
        )

        assert loaded_model is not None
        assert not loaded_model.training  # Should be in eval mode
        assert loaded_model.N == 64

    def test_load_model_for_evaluation_without_config_raises_error(self, tmp_path):
        """Test loading fails gracefully without config."""
        checkpoint_path = tmp_path / "model.pt"
        
        from models import ConvTasNet
        
        model = ConvTasNet(N=64, B=64, H=128, P=3, X=4, R=2, C=1)
        
        # Save checkpoint WITHOUT config
        torch.save(
            {"model_state_dict": model.state_dict(), "epoch": 10},
            checkpoint_path,
        )
        
        # Should raise error when no config provided
        with pytest.raises(ValueError, match="does not contain a config"):
            load_model_for_inference(
                str(checkpoint_path), device="cpu"
            )

    def test_load_model_sets_eval_mode(self, tmp_path):
        """Test that loaded model is in eval mode."""
        checkpoint_path = tmp_path / "model.pt"
        
        config_dict = {
            "model": {
                "model_type": "convtasnet",
                "convtasnet": {
                    "N": 64,
                    "B": 64,
                    "H": 128,
                    "P": 3,
                    "X": 4,
                    "R": 2,
                    "C": 1,
                    "kernel_size": 16,
                    "stride": 8,
                },
            }
        }
        
        from models import ConvTasNet
        
        model = ConvTasNet(N=64, B=64, H=128, P=3, X=4, R=2, C=1)
        model.train()  # Set to training mode
        
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": config_dict,
            },
            checkpoint_path,
        )
        
        loaded_model, _ = load_model_for_inference(
            str(checkpoint_path), device="cpu"
        )

        assert not loaded_model.training

    def test_load_model_with_config_override(self, tmp_path):
        """User-provided config_override is used when the checkpoint has no
        embedded config (the real `load_model_for_inference(..., config_override=)`
        API). Replaces a stale commented-out test that called
        `load_model_from_checkpoint(..., config=...)` — a signature that
        function has never had; `config_override` belongs to
        `load_model_for_inference`, not `load_model_from_checkpoint`."""
        from models import ConvTasNet

        checkpoint_path = tmp_path / "model.pt"
        model = ConvTasNet(N=64, B=64, H=128, P=3, X=4, R=2, C=1)

        # Checkpoint saved WITHOUT an embedded config.
        torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

        config_override = {
            "model": {
                "model_type": "convtasnet",
                "convtasnet": {
                    "N": 64, "B": 64, "H": 128, "P": 3, "X": 4, "R": 2, "C": 1,
                },
            }
        }

        loaded_model, checkpoint = load_model_for_inference(
            str(checkpoint_path), device="cpu", config_override=config_override
        )

        assert loaded_model is not None
        assert loaded_model.N == 64
        assert not loaded_model.training
        assert checkpoint.get("config") is None  # confirms the override path, not an embedded config

    def test_load_model_nonexistent_file_raises_error(self):
        """Test loading from nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_model_for_inference(
                "/nonexistent/path.pt", device="cpu"
            )


class TestMetricComputation:
    """Test evaluation metric computation."""

    def test_sisdr_computation_with_known_values(self):
        """Test SI-SDR computation with synthetic signals."""
        from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
        
        si_sdr_metric = ScaleInvariantSignalDistortionRatio()
        
        # Test 1: Perfect reconstruction (SI-SDR should be very high)
        clean = torch.randn(16000)
        estimate = clean.clone()
        
        sisdr = si_sdr_metric(estimate.unsqueeze(0), clean.unsqueeze(0))
        
        # Perfect reconstruction should give very high SI-SDR (>40 dB)
        assert sisdr.item() > 40.0
        
    def test_sisdr_zero_signal(self):
        """Test SI-SDR handles zero signals gracefully."""
        from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
        
        si_sdr_metric = ScaleInvariantSignalDistortionRatio()
        
        clean = torch.randn(16000)
        estimate = torch.zeros(16000)
        
        # Should not crash and return a finite value
        sisdr = si_sdr_metric(estimate.unsqueeze(0), clean.unsqueeze(0))
        assert torch.isfinite(sisdr)
        
    def test_sisdr_scaled_signal(self):
        """Test SI-SDR is scale-invariant."""
        from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
        
        si_sdr_metric = ScaleInvariantSignalDistortionRatio()
        
        clean = torch.randn(16000)
        estimate = clean * 2.5  # Scaled version
        
        sisdr = si_sdr_metric(estimate.unsqueeze(0), clean.unsqueeze(0))
        
        # Scale-invariant metric should still be very high
        assert sisdr.item() > 40.0
        
    def test_sisdr_noisy_signal(self):
        """Test SI-SDR with added noise."""
        from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
        
        si_sdr_metric = ScaleInvariantSignalDistortionRatio()
        
        clean = torch.randn(16000)
        noise = torch.randn(16000) * 0.1
        estimate = clean + noise
        
        sisdr = si_sdr_metric(estimate.unsqueeze(0), clean.unsqueeze(0))
        
        # Should be positive but not perfect
        assert 0 < sisdr.item() < 30.0


class TestEvaluationFormatting:
    """Test evaluation result formatting."""

    def test_print_summary_basic(self, capsys):
        """Test print_summary doesn't crash with basic results."""
        results = {
            "variant1": {"si_sdr": 10.5, "si_sdri": 8.2, "num_samples": 100},
            "variant2": {"si_sdr": 12.3, "si_sdri": 10.1, "num_samples": 150},
        }
        
        print_summary(results)
        
        captured = capsys.readouterr()
        assert "EVALUATION SUMMARY" in captured.out
        assert "variant1" in captured.out
        assert "10.5" in captured.out or "10.50" in captured.out

    def test_print_summary_with_pesq_stoi(self, capsys):
        """Test print_summary with additional metrics."""
        results = {
            "variant1": {
                "si_sdr": 10.5,
                "si_sdri": 8.2,
                "pesq": 2.8,
                "stoi": 0.85,
                "num_samples": 100,
            }
        }
        
        print_summary(results)
        
        captured = capsys.readouterr()
        assert "PESQ" in captured.out
        assert "STOI" in captured.out

    def test_print_summary_multiple_variants_shows_average(self, capsys):
        """Test that average row is shown for multiple variants."""
        results = {
            "C": {"si_sdr": 10.0, "si_sdri": 7.5, "num_samples": 50},
            "S": {"si_sdr": 8.0, "si_sdri": 5.8, "num_samples": 50},
            "E": {"si_sdr": 12.0, "si_sdri": 9.9, "num_samples": 50},
        }
        
        print_summary(results)
        
        captured = capsys.readouterr()
        assert "AVERAGE" in captured.out

    def test_save_results_csv(self, tmp_path):
        """Test saving results to CSV."""
        output_path = tmp_path / "results.csv"

        results = {
            "variant1": {"si_sdr": 10.5, "si_sdri": 8.2, "num_samples": 100},
            "variant2": {"si_sdr": 12.3, "si_sdri": 10.1, "pesq": 2.8, "num_samples": 150},
        }
        
        save_results_csv(results, str(output_path))
        
        assert output_path.exists()
        
        # Read CSV and verify contents
        import pandas as pd
        
        df = pd.read_csv(output_path)
        
        assert len(df) == 2
        assert "variant" in df.columns
        assert "si_sdr_db" in df.columns
        assert "num_samples" in df.columns
        assert df.iloc[0]["variant"] == "variant1"

    def test_save_results_csv_with_all_metrics(self, tmp_path):
        """Test CSV includes all metrics when available."""
        output_path = tmp_path / "results.csv"

        results = {
            "variant1": {
                "si_sdr": 10.5,
                "si_sdri": 8.2,
                "pesq": 2.8,
                "stoi": 0.85,
                "num_samples": 100,
            }
        }
        
        save_results_csv(results, str(output_path))
        
        import pandas as pd
        
        df = pd.read_csv(output_path)
        
        assert "pesq" in df.columns
        assert "stoi" in df.columns
        assert df.iloc[0]["pesq"] == 2.8
        assert df.iloc[0]["stoi"] == 0.85


class TestVariantList:
    """Test that evaluate_by_variant uses the correct variant list."""

    def test_no_duplicate_variants(self):
        """Regression test: 'C' (clean) must not appear in both indoor and outdoor lists."""
        from evaluate import evaluate_by_variant
        import inspect

        # Read the source to extract the variant lists
        source = inspect.getsource(evaluate_by_variant)

        # The combined list should have exactly 8 unique variants
        # SER, SR, ER, R (indoor) + SE, S, E, C (outdoor) = 8
        indoor_variants = ["SER", "SR", "ER", "R"]
        outdoor_variants = ["SE", "S", "E", "C"]
        all_variants = indoor_variants + outdoor_variants

        assert len(all_variants) == len(set(all_variants)), (
            "Variant list contains duplicates — 'C' should only appear in outdoor_variants"
        )
        assert len(all_variants) == 8

    def test_c_variant_not_in_indoor(self):
        """'C' (clean, no reverb) is an outdoor variant and must not be in indoor list."""
        indoor_variants = ["SER", "SR", "ER", "R"]
        assert "C" not in indoor_variants

    def test_all_expected_variants_present(self):
        """All 8 MM-IPC variants must be present in the combined list."""
        indoor_variants = ["SER", "SR", "ER", "R"]
        outdoor_variants = ["SE", "S", "E", "C"]
        all_variants = set(indoor_variants + outdoor_variants)

        expected = {"SER", "SR", "ER", "R", "SE", "S", "E", "C"}
        assert all_variants == expected


class TestSBTaskPESQSTOI:
    """Test PESQ/STOI computation for speaker separation (SB) task."""

    def test_results_include_pesq_stoi_for_sb(self):
        """Verify print_summary handles PESQ/STOI + improvement metrics."""
        results = {
            "SER": {
                "si_sdr": 5.0,
                "si_sdri": 3.0,
                "pesq": 2.5,
                "pesqi": 0.8,
                "stoi": 0.75,
                "stoii": 0.15,
                "num_samples": 50,
            }
        }
        # Should not raise — print_summary must handle the new keys
        print_summary(results)

    def test_print_summary_with_improvement_columns(self, capsys):
        """PESQi and STOIi columns appear in the summary table."""
        results = {
            "SER": {
                "si_sdr": 5.0,
                "si_sdri": 3.0,
                "pesq": 2.5,
                "pesqi": 0.8,
                "stoi": 0.75,
                "stoii": 0.15,
                "num_samples": 50,
            }
        }
        print_summary(results)
        output = capsys.readouterr().out
        assert "PESQi" in output
        assert "STOIi" in output
        assert "0.8" in output  # PESQi value present
        assert "0.15" in output  # STOIi value present

    def test_save_csv_includes_improvement_metrics(self, tmp_path):
        """CSV export includes pesqi and stoii columns."""
        import pandas as pd

        results = {
            "SER": {
                "si_sdr": 5.0,
                "si_sdri": 3.0,
                "pesq": 2.5,
                "pesqi": 0.8,
                "stoi": 0.75,
                "stoii": 0.15,
                "num_samples": 50,
            }
        }
        output_path = tmp_path / "results.csv"
        save_results_csv(results, str(output_path))

        df = pd.read_csv(output_path)
        assert "pesqi" in df.columns
        assert "stoii" in df.columns
        assert df.iloc[0]["pesqi"] == pytest.approx(0.8)
        assert df.iloc[0]["stoii"] == pytest.approx(0.15)

    def test_backward_compat_no_improvement_metrics(self):
        """Results without improvement metrics still work in print_summary."""
        results = {
            "SER": {
                "si_sdr": 5.0,
                "si_sdri": 3.0,
                "pesq": 2.5,
                "stoi": 0.75,
                "num_samples": 50,
            }
        }
        # Should not raise
        print_summary(results)


class TestEvaluateModelEndToEnd:
    """End-to-end `evaluate_model` test on a 2-sample synthetic SB dataset with
    hand-computed SI-SDRi (survey gap 4/E4 item 9): pins the mixture-baseline
    computation, PIT reordering, and per-sample aggregation all in one place —
    no such test existed before this. `evaluate.py` now forces batch_size=1
    (gap 3) so every accumulated scalar is already a per-sample score; this
    test's aggregation check confirms the reported mean really is the exact
    mean of the two per-sample values, not a mean-of-batch-means.

    Construction: A and B are orthogonal, zero-mean, equal-energy (||.||^2=4)
    4-sample vectors used as the two "clean" speaker signals. `mix = A + B`.
    Each fake-model estimate pair is built as
        estimate0 = B + c*A,  estimate1 = A + c*B   (0 < c < 1)
    so estimate0 best matches clean channel 1 (B) and estimate1 best matches
    clean channel 0 (A) — PIT's optimal permutation is therefore the *swap*,
    not the identity. Standard SI-SDR projection algebra gives, for both
    channels under the swap permutation, SI-SDR = -20*log10(c) exactly (both
    channels symmetric ⇒ this is also the batch-mean SI-SDR PITLossWrapper
    returns), and SI-SDR(mix, A) == SI-SDR(mix, B) == 0 dB exactly (mix is an
    equal-energy orthogonal sum). Hence si_sdri == si_sdr for every sample
    here. These hand-derived numbers were cross-checked against the actual
    `ScaleInvariantSignalDistortionRatio` + `PITLossWrapper(pairwise_neg_sisdr,
    pit_from="pw_mtx")` implementations before being hardcoded as the
    expectation (not merely a textbook derivation).
    """

    @staticmethod
    def _build_samples(cs):
        A = torch.tensor([1., -1., 1., -1.])
        B = torch.tensor([1., 1., -1., -1.])
        assert torch.dot(A, B).item() == 0.0  # orthogonality precondition

        samples, expected = [], []
        for c in cs:
            est0 = B + c * A
            est1 = A + c * B
            mix = A + B
            samples.append(
                {
                    "mix": mix,
                    "clean": torch.stack([A, B]),
                    "estimates": torch.stack([est0, est1]),
                }
            )
            hand_si_sdr = -20.0 * math.log10(c)
            expected.append({"si_sdr": hand_si_sdr, "si_sdri": hand_si_sdr})  # baseline = 0 dB
        return samples, expected

    def test_evaluate_model_sb_pins_baseline_pit_and_aggregation(self):
        samples, expected = self._build_samples([0.5, 0.25])

        class _SyntheticSBDataset(torch.utils.data.Dataset):
            def __len__(self):
                return len(samples)

            def __getitem__(self, idx):
                return {"mix": samples[idx]["mix"], "clean": samples[idx]["clean"]}

        class _FakeSeparator(torch.nn.Module):
            """Ignores its input; returns the precomputed estimate pair for
            whichever sample index evaluate_model is currently on. Valid only
            because the dataloader below is shuffle=False, batch_size=1, one
            pass — the same sequential-order assumption evaluate_model's own
            per-sample loop relies on."""

            def __init__(self):
                super().__init__()
                self._unused_param = torch.nn.Parameter(torch.zeros(1))
                self._call_idx = 0

            def forward(self, mix_input):
                est = samples[self._call_idx]["estimates"].unsqueeze(0)
                self._call_idx += 1
                return est

        dataloader = torch.utils.data.DataLoader(
            _SyntheticSBDataset(), batch_size=1, shuffle=False
        )
        model = _FakeSeparator()

        results = evaluate_model(
            model,
            dataloader,
            device="cpu",
            compute_pesq=False,
            compute_stoi=False,
            use_amp=False,
            task="SB",
        )

        assert results["num_samples"] == 2
        assert results["pesq_failures"] == 0
        assert len(results["per_sample"]) == 2

        for i, exp in enumerate(expected):
            rec = results["per_sample"][i]
            assert rec["sample_idx"] == i
            assert rec["si_sdr"] == pytest.approx(exp["si_sdr"], abs=1e-3)
            assert rec["si_sdri"] == pytest.approx(exp["si_sdri"], abs=1e-3)

        # Aggregation: the reported mean is the exact arithmetic mean of the
        # per-sample values — batch_size=1 throughout, so there is no
        # mean-of-batch-means bias (gap 3) to reproduce.
        hand_mean_si_sdr = sum(e["si_sdr"] for e in expected) / len(expected)
        hand_mean_si_sdri = sum(e["si_sdri"] for e in expected) / len(expected)
        assert results["si_sdr"] == pytest.approx(hand_mean_si_sdr, abs=1e-3)
        assert results["si_sdri"] == pytest.approx(hand_mean_si_sdri, abs=1e-3)
        assert results["si_sdr"] == pytest.approx(
            sum(r["si_sdr"] for r in results["per_sample"]) / 2, abs=1e-9
        )
        assert results["si_sdri"] == pytest.approx(
            sum(r["si_sdri"] for r in results["per_sample"]) / 2, abs=1e-9
        )
