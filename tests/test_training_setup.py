"""Tests for training/setup.py builders, run-manifest provenance, and the
determinism policy (Work Packages E1 + B1 + B2).

CPU-only: a fake dataset and a stub model factory stand in for the real PolSESS
data and GPU models, so the shared build path both entry points funnel through
is exercised without touching disk datasets or the GPU.
"""

import os
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import Dataset, DataLoader

from datasets import polsess_collate_fn

# Reuse the trainer test's stub model/config (tests is a package under pytest).
try:
    from tests.test_trainer import make_config, DummyModel, SyntheticDataset
except ImportError:  # pragma: no cover - depends on pytest import mode
    from test_trainer import make_config, DummyModel, SyntheticDataset


class _FakeVariantDataset(Dataset):
    """Dataset stub that ignores paths and just yields tiny random tensors."""

    def __init__(self, data_root, subset="train", task="ES", max_samples=None, allowed_variants=None,
                 sample_rate=None):
        self.subset = subset
        self.task = task
        self.allowed_variants = allowed_variants
        self.n = max_samples if max_samples is not None else 6

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if self.task == "SB":
            clean = torch.randn(2, 64)
        else:
            clean = torch.randn(64)
        return {"mix": torch.randn(64), "clean": clean, "background_complexity": "S"}


def _stub_config(tmp_path, *, per_variant=False, validation_variants=None,
                 deterministic=None, task="ES", train_max=6, val_max=4):
    """Build a SimpleNamespace config sufficient for the setup builders."""
    cfg = make_config(tmp_path, task=task)
    cfg.data.dataset_type = "polsess"
    cfg.data.polsess = SimpleNamespace(data_root="/ignored/by/fake/dataset")
    cfg.data.train_max_samples = train_max
    cfg.data.val_max_samples = val_max
    cfg.data.num_workers = 0
    cfg.data.sample_rate = 8000
    cfg.training.deterministic = deterministic
    cfg.training.per_variant_validation = per_variant
    cfg.training.validation_variants = validation_variants
    cfg.training.curriculum_learning = None
    return cfg


# ---------------------------------------------------------------------------
# build_dataloaders
# ---------------------------------------------------------------------------

def test_build_dataloaders_single_val(tmp_path, monkeypatch):
    from training import setup
    monkeypatch.setattr(setup, "get_dataset", lambda name: _FakeVariantDataset)

    cfg = _stub_config(tmp_path, per_variant=False, train_max=6, val_max=4)
    summary_info = {"seed": cfg.training.seed}

    train_loader, val_loader, per_variant = setup.build_dataloaders(cfg, summary_info)

    assert per_variant is None
    assert len(train_loader.dataset) == 6
    assert len(val_loader.dataset) == 4
    assert summary_info["train_samples"] == 6
    assert summary_info["val_samples"] == 4
    # Gap 6: train loader carries a seeded generator + worker_init_fn.
    assert train_loader.generator is not None
    assert train_loader.worker_init_fn is setup.seed_worker


def test_build_dataloaders_per_variant_all(tmp_path, monkeypatch):
    from training import setup
    monkeypatch.setattr(setup, "get_dataset", lambda name: _FakeVariantDataset)

    cfg = _stub_config(tmp_path, per_variant=True, validation_variants=None)
    summary_info = {"seed": cfg.training.seed}

    train_loader, val_loader, per_variant = setup.build_dataloaders(cfg, summary_info)

    assert val_loader is None
    assert list(per_variant.keys()) == setup.ALL_VARIANTS
    assert summary_info["val_variants"] == setup.ALL_VARIANTS


def test_build_dataloaders_per_variant_filtered(tmp_path, monkeypatch):
    from training import setup
    monkeypatch.setattr(setup, "get_dataset", lambda name: _FakeVariantDataset)

    cfg = _stub_config(tmp_path, per_variant=True, validation_variants=["SER", "C"])
    summary_info = {"seed": cfg.training.seed}

    _, val_loader, per_variant = setup.build_dataloaders(cfg, summary_info)

    assert val_loader is None
    assert set(per_variant.keys()) == {"SER", "C"}


def test_build_dataloaders_seed_generator_reproducible(tmp_path, monkeypatch):
    """The seeded generator makes shuffle order identical across two builds —
    the reproducibility-by-contract the plan asks for."""
    from training import setup
    monkeypatch.setattr(setup, "get_dataset", lambda name: _FakeVariantDataset)

    cfg = _stub_config(tmp_path)
    loader_a, _, _ = setup.build_dataloaders(cfg, {"seed": cfg.training.seed})
    loader_b, _, _ = setup.build_dataloaders(cfg, {"seed": cfg.training.seed})

    # Compare the sampler orders produced by each loader's generator directly.
    order_a = list(torch.utils.data.RandomSampler(loader_a.dataset, generator=loader_a.generator))
    order_b = list(torch.utils.data.RandomSampler(loader_b.dataset, generator=loader_b.generator))
    assert order_a == order_b


def test_build_dataloaders_rejects_non_polsess(tmp_path, monkeypatch):
    from training import setup
    monkeypatch.setattr(setup, "get_dataset", lambda name: _FakeVariantDataset)
    cfg = _stub_config(tmp_path)
    cfg.data.dataset_type = "libri2mix"
    with pytest.raises(ValueError, match="Only PolSESS"):
        setup.build_dataloaders(cfg, {"seed": cfg.training.seed})


# ---------------------------------------------------------------------------
# build_trainer  (shared model+compile+Trainer path used by both mains)
# ---------------------------------------------------------------------------

def test_build_trainer_wraps_model_and_provenance(tmp_path, monkeypatch):
    from training import setup

    def fake_factory(model_config, summary_info):
        summary_info["model_params_millions"] = 0.01
        return DummyModel()

    monkeypatch.setattr(setup, "create_model_from_config", fake_factory)
    monkeypatch.setattr(setup, "compile_for_model_type", lambda m, mt, logger=None: m)

    cfg = _stub_config(tmp_path)
    train_loader = DataLoader(SyntheticDataset(4), batch_size=2, collate_fn=polsess_collate_fn)
    val_loader = DataLoader(SyntheticDataset(2), batch_size=2, collate_fn=polsess_collate_fn)
    summary_info = {"seed": cfg.training.seed}
    manifest = {"git_sha": "deadbee"}

    trainer = setup.build_trainer(
        cfg, train_loader, val_loader, None,
        device="cpu", logger=None, wandb_logger=None,
        summary_info=summary_info, provenance=manifest,
    )

    assert isinstance(trainer.model, DummyModel)
    assert trainer.provenance == manifest
    # Factory populated the param count into the shared summary_info.
    assert summary_info["model_params_millions"] == 0.01
    # Both mains hand build_trainer the same objects; this is that single path.
    assert trainer.train_loader is train_loader
    assert trainer.val_loader is val_loader


# ---------------------------------------------------------------------------
# collect_run_manifest
# ---------------------------------------------------------------------------

def test_collect_run_manifest_keys_and_seed():
    from utils import collect_run_manifest

    m = collect_run_manifest(seed=1234)
    for key in (
        "git_sha", "git_dirty", "torch_version", "cuda_version", "cudnn_version",
        "gpu_name", "mamba_ssm_version", "triton_version", "hostname",
        "python_version", "seed", "argv",
    ):
        assert key in m, f"missing manifest key: {key}"
    assert m["seed"] == 1234
    assert isinstance(m["argv"], list)
    assert m["torch_version"] == torch.__version__


def test_collect_run_manifest_degrades_without_cuda(monkeypatch):
    from utils import collect_run_manifest

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    m = collect_run_manifest(seed=None)
    assert m["gpu_name"] is None
    assert m["seed"] is None


def test_optional_pkg_version_missing_is_none():
    from utils.common import _optional_pkg_version

    assert _optional_pkg_version("definitely_not_a_real_module_xyz") is None
    # A guaranteed-present module returns a version-ish string (or "unknown").
    assert _optional_pkg_version("torch") == torch.__version__


# ---------------------------------------------------------------------------
# configure_determinism
# ---------------------------------------------------------------------------

@pytest.fixture
def _determinism_state():
    """Snapshot/restore the global torch determinism state the tests mutate."""
    prev_bench = torch.backends.cudnn.benchmark
    prev_det = torch.are_deterministic_algorithms_enabled()
    prev_env = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    try:
        yield
    finally:
        torch.backends.cudnn.benchmark = prev_bench
        torch.use_deterministic_algorithms(prev_det, warn_only=True)
        if prev_env is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = prev_env


def test_configure_determinism_none_is_noop(_determinism_state):
    from utils import configure_determinism

    torch.backends.cudnn.benchmark = False
    configure_determinism(None)
    # Default (None) must not touch benchmark or global determinism.
    assert torch.backends.cudnn.benchmark is False
    assert torch.are_deterministic_algorithms_enabled() is False


def test_configure_determinism_false_enables_benchmark(_determinism_state):
    from utils import configure_determinism

    torch.backends.cudnn.benchmark = False
    configure_determinism(False)
    assert torch.backends.cudnn.benchmark is True
    assert torch.are_deterministic_algorithms_enabled() is False


def test_configure_determinism_true_is_strict(_determinism_state):
    from utils import configure_determinism

    configure_determinism(True)
    assert torch.are_deterministic_algorithms_enabled() is True
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
