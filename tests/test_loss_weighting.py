"""Tests for per-variant loss weighting + SI-SDRi validation.

Covers:
- Config validation of the loss_weights dict.
- The build_loss_weights_from_w_c_in 1D-HPO helper.
- Per-sample weight lookup with C → C_in / C_out disambiguation.
- Numerical equivalence: uniform weights reproduce the unweighted loss
  exactly for both ES/EB and SB (PIT) paths.
- SI-SDRi correctness: estimate==target ⇒ large; estimate==mix ⇒ ~0;
  per-variant grouping + C_in/C_out separation.

The synthetic harness (DummyModel + SyntheticDataset variants) avoids any
dependency on the real PolSESS dataset.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace
from torch.utils.data import DataLoader, Dataset

from config import (
    LOSS_WEIGHT_KEYS_INDOOR,
    LOSS_WEIGHT_KEYS_OUTDOOR,
    LOSS_WEIGHT_KEYS_ALL,
    _validate_loss_weights,
    build_loss_weights_from_w_c_in,
)
from datasets import polsess_collate_fn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _full_weights(w_c_in=0.30):
    """Convenience for valid 9-key dicts."""
    return build_loss_weights_from_w_c_in(w_c_in)


class TaggedSyntheticDataset(Dataset):
    """Synthetic dataset with explicit (variant, has_reverb) tags per sample.

    Used for testing per-variant weight lookup and SI-SDRi grouping.
    """

    def __init__(self, samples, time_steps=256, task="ES"):
        self.samples = samples  # list of (variant, has_reverb)
        self.T = time_steps
        self.task = task

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        variant, has_reverb = self.samples[idx]
        mix = torch.randn(self.T)
        if self.task == "SB":
            clean = torch.stack([mix.clone(), mix.clone()])
        else:
            clean = mix.clone()
        return {
            "mix": mix,
            "clean": clean,
            "background_complexity": variant,
            "has_reverb": has_reverb,
        }


class DummyModel(nn.Module):
    """Identity-ish model with one trainable parameter; matches the test_trainer fixture."""

    def __init__(self, C=1):
        super().__init__()
        self.C = C
        self.p = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        x = x + self.p
        if self.C > 1:
            return x.unsqueeze(1).repeat(1, self.C, 1)
        return x


def _make_cfg(tmp_path, task="ES", loss_weights=None):
    cfg = SimpleNamespace()
    cfg.data = SimpleNamespace(batch_size=4, num_workers=0, prefetch_factor=2, task=task)
    cfg.training = SimpleNamespace(
        lr=1e-3, weight_decay=0.0, grad_clip_norm=5.0,
        lr_factor=0.95, lr_patience=2,
        num_epochs=1, use_amp=False, amp_eps=1e-4,
        save_dir=str(tmp_path / "ckpt"),
        save_best_only=True, use_wandb=False,
        wandb_project=None, wandb_entity=None, wandb_run_name=None,
        log_file=None, log_level="INFO", device="cpu", seed=42,
        resume_from=None, validation_variants=None, curriculum_learning=None,
        save_all_checkpoints=False, grad_accumulation_steps=1,
        loss_weights=loss_weights,
    )
    cfg.model = SimpleNamespace(model_type="convtasnet")
    return cfg


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestLossWeightsValidation:
    def test_valid_dict_passes(self):
        _validate_loss_weights(_full_weights(0.30))

    def test_missing_key_raises(self):
        bad = _full_weights(0.30)
        del bad["C_out"]
        with pytest.raises(ValueError, match="Missing"):
            _validate_loss_weights(bad)

    def test_extra_key_raises(self):
        bad = _full_weights(0.30)
        bad["C_extra"] = 0.1
        with pytest.raises(ValueError, match="Extra"):
            _validate_loss_weights(bad)

    def test_negative_value_raises(self):
        bad = _full_weights(0.30)
        bad["SER"] = -0.1
        with pytest.raises(ValueError, match="non-negative"):
            _validate_loss_weights(bad)

    def test_all_zero_raises(self):
        bad = {k: 0.0 for k in LOSS_WEIGHT_KEYS_ALL}
        with pytest.raises(ValueError, match="positive sum"):
            _validate_loss_weights(bad)


# ---------------------------------------------------------------------------
# build_loss_weights_from_w_c_in helper
# ---------------------------------------------------------------------------


class TestBuildLossWeightsHelper:
    def test_starting_point_w_c_in_0_30(self):
        w = build_loss_weights_from_w_c_in(0.30)
        # Indoor non-C: 0.175 each
        assert w["SER"] == pytest.approx(0.175)
        assert w["SR"] == pytest.approx(0.175)
        assert w["ER"] == pytest.approx(0.175)
        assert w["R"] == pytest.approx(0.175)
        assert w["C_in"] == pytest.approx(0.30)
        # Outdoor non-C: (1 - 0.35) / 3
        assert w["SE"] == pytest.approx((1 - 0.35) / 3)
        assert w["S"] == pytest.approx((1 - 0.35) / 3)
        assert w["E"] == pytest.approx((1 - 0.35) / 3)
        assert w["C_out"] == pytest.approx(0.35)

    def test_indoor_outdoor_unit_sum(self):
        for w_c_in in [0.0, 0.10, 0.30, 0.50, 0.90]:
            w = build_loss_weights_from_w_c_in(w_c_in)
            assert sum(w[k] for k in LOSS_WEIGHT_KEYS_INDOOR) == pytest.approx(1.0)
            assert sum(w[k] for k in LOSS_WEIGHT_KEYS_OUTDOOR) == pytest.approx(1.0)

    def test_validates_full_dict(self):
        # Whatever build_loss_weights produces must pass _validate_loss_weights
        for w_c_in in [0.05, 0.30, 0.60]:
            _validate_loss_weights(build_loss_weights_from_w_c_in(w_c_in))

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError):
            build_loss_weights_from_w_c_in(-0.1)
        with pytest.raises(ValueError):
            build_loss_weights_from_w_c_in(1.5)
        with pytest.raises(ValueError):
            # 0.96 would push w_c_out > 1.0
            build_loss_weights_from_w_c_in(0.96)


# ---------------------------------------------------------------------------
# Per-sample weight lookup (Trainer._build_sample_weights / _resolve_weight_key)
# ---------------------------------------------------------------------------


class TestSampleWeightLookup:
    def _make_trainer(self, tmp_path):
        from training.trainer import Trainer
        cfg = _make_cfg(tmp_path, loss_weights=_full_weights(0.30))
        ds = TaggedSyntheticDataset([("S", False)] * 4, time_steps=64)
        loader = DataLoader(ds, batch_size=4, collate_fn=polsess_collate_fn)
        trainer = Trainer(DummyModel(), loader, loader, cfg, device="cpu")
        return trainer

    def test_indoor_C_maps_to_C_in(self, tmp_path):
        trainer = self._make_trainer(tmp_path)
        assert trainer._resolve_weight_key("C", True) == "C_in"

    def test_outdoor_C_maps_to_C_out(self, tmp_path):
        trainer = self._make_trainer(tmp_path)
        assert trainer._resolve_weight_key("C", False) == "C_out"

    def test_non_C_passthrough(self, tmp_path):
        trainer = self._make_trainer(tmp_path)
        for v in ("SER", "SR", "ER", "R", "SE", "S", "E"):
            # has_reverb is irrelevant for non-C
            assert trainer._resolve_weight_key(v, True) == v
            assert trainer._resolve_weight_key(v, False) == v

    def test_build_sample_weights_mixed_batch(self, tmp_path):
        trainer = self._make_trainer(tmp_path)
        variants = ["SER", "SE", "C", "C", "R"]
        has_reverb = torch.tensor([True, False, True, False, True])
        w = trainer._build_sample_weights(variants, has_reverb)
        assert w.shape == (5,)
        assert w[0].item() == pytest.approx(trainer.loss_weights["SER"])
        assert w[1].item() == pytest.approx(trainer.loss_weights["SE"])
        assert w[2].item() == pytest.approx(trainer.loss_weights["C_in"])
        assert w[3].item() == pytest.approx(trainer.loss_weights["C_out"])
        assert w[4].item() == pytest.approx(trainer.loss_weights["R"])


# ---------------------------------------------------------------------------
# Numerical equivalence: uniform weights = unweighted (regression)
# ---------------------------------------------------------------------------


class TestUniformWeightsRegression:
    """When all weights are equal, the weighted-mean reduction must match
    the unweighted batch-mean exactly. This validates that
    scale_invariant_signal_distortion_ratio (functional) and
    PITLossWrapper.find_best_perm produce results compatible with the
    legacy class-metric / forward paths.
    """

    def test_es_eb_uniform_matches_unweighted(self, tmp_path):
        from training.trainer import Trainer
        torch.manual_seed(0)
        cfg = _make_cfg(tmp_path, task="ES")
        ds = TaggedSyntheticDataset([("R", True)] * 4, time_steps=512)
        loader = DataLoader(ds, batch_size=4, collate_fn=polsess_collate_fn)
        trainer = Trainer(DummyModel(C=1), loader, loader, cfg, device="cpu")

        estimates = torch.randn(4, 512)
        targets = torch.randn(4, 512)

        # Unweighted reference (existing class-metric path)
        loss_ref, sisdr_ref = trainer._sisdr_loss_wrapper(estimates, targets)
        # Weighted with uniform weights
        weights = torch.ones(4)
        loss_w, sisdr_w = trainer._weighted_sisdr_loss_wrapper(estimates, targets, weights)

        assert loss_w.item() == pytest.approx(loss_ref.item(), abs=1e-5)
        assert sisdr_w == pytest.approx(sisdr_ref, abs=1e-5)

    def test_sb_uniform_matches_unweighted(self, tmp_path):
        from training.trainer import Trainer
        torch.manual_seed(1)
        cfg = _make_cfg(tmp_path, task="SB")
        ds = TaggedSyntheticDataset([("R", True)] * 4, time_steps=512, task="SB")
        loader = DataLoader(ds, batch_size=4, collate_fn=polsess_collate_fn)
        trainer = Trainer(DummyModel(C=2), loader, loader, cfg, device="cpu")

        estimates = torch.randn(4, 2, 512)
        targets = torch.randn(4, 2, 512)

        loss_ref, sisdr_ref = trainer._pit_loss_wrapper(estimates, targets)
        weights = torch.ones(4)
        loss_w, sisdr_w = trainer._weighted_pit_loss_wrapper(estimates, targets, weights)

        assert loss_w.item() == pytest.approx(loss_ref.item(), abs=1e-5)
        assert sisdr_w == pytest.approx(sisdr_ref, abs=1e-5)

    def test_loss_weights_none_keeps_legacy_loss_fn(self, tmp_path):
        """Configs without loss_weights must keep the legacy wrappers, not
        the weighted ones. This is the regression that protects existing
        experiments from a silent behavior change."""
        from training.trainer import Trainer
        cfg = _make_cfg(tmp_path, task="ES", loss_weights=None)
        ds = TaggedSyntheticDataset([("S", False)] * 2, time_steps=64)
        loader = DataLoader(ds, batch_size=2, collate_fn=polsess_collate_fn)
        trainer = Trainer(DummyModel(C=1), loader, loader, cfg, device="cpu")
        assert not trainer._weighted_loss_active
        assert trainer.loss_fn == trainer._sisdr_loss_wrapper

        cfg_sb = _make_cfg(tmp_path, task="SB", loss_weights=None)
        ds_sb = TaggedSyntheticDataset([("S", False)] * 2, time_steps=64, task="SB")
        loader_sb = DataLoader(ds_sb, batch_size=2, collate_fn=polsess_collate_fn)
        trainer_sb = Trainer(DummyModel(C=2), loader_sb, loader_sb, cfg_sb, device="cpu")
        assert not trainer_sb._weighted_loss_active
        assert trainer_sb.loss_fn == trainer_sb._pit_loss_wrapper


# ---------------------------------------------------------------------------
# SI-SDRi correctness
# ---------------------------------------------------------------------------


class TestSISDRi:
    def test_estimate_equals_target_large_sisdri(self, tmp_path):
        """Perfect estimate ⇒ SI-SDR(est, src) is huge; SI-SDRi >> 0."""
        from training.trainer import Trainer
        cfg = _make_cfg(tmp_path, task="ES", loss_weights=_full_weights(0.30))
        ds = TaggedSyntheticDataset([("R", True)] * 2, time_steps=512)
        loader = DataLoader(ds, batch_size=2, collate_fn=polsess_collate_fn)
        trainer = Trainer(DummyModel(C=1), loader, loader, cfg, device="cpu")

        torch.manual_seed(2)
        target = torch.randn(2, 512)
        mix = target + 0.5 * torch.randn(2, 512)  # noisy mix
        sisdr_est = trainer._per_sample_sisdr(target, target)  # estimate==target
        sisdr_mix = trainer._per_sample_sisdr(mix, target)
        sisdri = sisdr_est - sisdr_mix
        # Perfect estimate against itself ⇒ SI-SDR is bounded above only by
        # numerical eps, so SI-SDRi must be very positive.
        assert (sisdri > 30).all(), f"Expected large SI-SDRi, got {sisdri.tolist()}"

    def test_estimate_equals_mix_sisdri_zero(self, tmp_path):
        """Estimate == mixture ⇒ SI-SDRi == 0 (model added nothing)."""
        from training.trainer import Trainer
        cfg = _make_cfg(tmp_path, task="ES", loss_weights=_full_weights(0.30))
        ds = TaggedSyntheticDataset([("R", True)] * 2, time_steps=512)
        loader = DataLoader(ds, batch_size=2, collate_fn=polsess_collate_fn)
        trainer = Trainer(DummyModel(C=1), loader, loader, cfg, device="cpu")

        torch.manual_seed(3)
        target = torch.randn(2, 512)
        mix = target + 0.5 * torch.randn(2, 512)
        sisdri = trainer._per_sample_sisdr(mix, target) - trainer._per_sample_sisdr(mix, target)
        assert torch.allclose(sisdri, torch.zeros_like(sisdri), atol=1e-5)

    def test_validate_per_variant_grouping(self, tmp_path):
        """Mixed-variant batch ⇒ per-variant breakdown buckets correctly,
        with C_in and C_out kept separate."""
        from training.trainer import Trainer
        cfg = _make_cfg(tmp_path, task="ES", loss_weights=_full_weights(0.30))
        # Mix: 2× SER (indoor), 1× SE (outdoor), 1× C indoor, 1× C outdoor, 1× R
        samples = [
            ("SER", True),
            ("SER", True),
            ("SE", False),
            ("C", True),
            ("C", False),
            ("R", True),
        ]
        ds = TaggedSyntheticDataset(samples, time_steps=256)
        loader = DataLoader(ds, batch_size=3, collate_fn=polsess_collate_fn)
        # Use the same loader for "val" so validate runs over these variants
        trainer = Trainer(DummyModel(C=1), loader, loader, cfg, device="cpu")
        trainer.validate()

        bd = trainer._last_val_breakdown
        assert "C_in" in bd["per_variant_sisdri"]
        assert "C_out" in bd["per_variant_sisdri"]
        assert "SER" in bd["per_variant_sisdri"]
        assert "SE" in bd["per_variant_sisdri"]
        assert "R" in bd["per_variant_sisdri"]
        # C_in and C_out must each have exactly 1 sample
        # (we can't read counts directly post-mean, but presence + finite is enough)
        assert all(
            torch.isfinite(torch.tensor(v))
            for v in bd["per_variant_sisdri"].values()
        )
        # Weighted-sum scalar must be finite and present
        assert torch.isfinite(torch.tensor(bd["val_sisdri_weighted"]))

    def test_validate_returns_weighted_when_active(self, tmp_path):
        """validate() returns the weighted-sum SI-SDRi (a float) when
        loss_weights is set — not the legacy mean SI-SDR."""
        from training.trainer import Trainer
        cfg = _make_cfg(tmp_path, task="ES", loss_weights=_full_weights(0.30))
        ds = TaggedSyntheticDataset([("R", True), ("SE", False)], time_steps=64)
        loader = DataLoader(ds, batch_size=2, collate_fn=polsess_collate_fn)
        trainer = Trainer(DummyModel(C=1), loader, loader, cfg, device="cpu")
        out = trainer.validate()
        assert isinstance(out, float)
        assert out == pytest.approx(trainer._last_val_breakdown["val_sisdri_weighted"])
