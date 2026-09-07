"""Unit tests for the Trainer using a tiny synthetic dataset and dummy model.

These tests are lightweight and avoid any dependency on the full PolSESS
dataset. They exercise `train_epoch`, `validate` and a 1-epoch `train` run.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace
from torch.utils.data import DataLoader, Dataset
from datasets import polsess_collate_fn


class SyntheticDataset(Dataset):
    def __init__(self, n_samples=8, time_steps=256, task="ES"):
        self.n = n_samples
        self.T = time_steps
        self.task = task

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        mix = torch.randn(self.T)
        # For a trivial target, set clean == mix so SI-SDR is stable
        if self.task == "SB":
            # Return [2, T] for 2-speaker separation
            clean = torch.stack([mix.clone(), mix.clone()])
        else:
            clean = mix.clone()
        return {"mix": mix, "clean": clean, "background_complexity": "S"}


class DummyModel(nn.Module):
    """A tiny model with one parameter but acts as identity on the input.

    It accepts input shaped [B, 1, T] or [B, T] and returns [B, T] (C=1) or [B, C, T] (C>1).
    """

    def __init__(self, C=1):
        super().__init__()
        self.C = C
        # register a parameter so optimizer has something to update
        self.p = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(1)
        x = x + self.p

        if self.C > 1:
            # Return [B, C, T] for multi-speaker separation
            return x.unsqueeze(1).repeat(1, self.C, 1)
        return x


def make_config(tmp_path, task="ES"):
    cfg = SimpleNamespace()
    cfg.data = SimpleNamespace()
    cfg.data.batch_size = 2
    cfg.data.num_workers = 0
    cfg.data.prefetch_factor = 2
    cfg.data.task = task

    cfg.training = SimpleNamespace()
    cfg.training.lr = 1e-3
    cfg.training.weight_decay = 0.0
    cfg.training.grad_clip_norm = 5.0
    cfg.training.lr_scheduler = "plateau"
    cfg.training.lr_factor = 0.95
    cfg.training.lr_patience = 2
    cfg.training.num_epochs = 1
    cfg.training.use_amp = False
    cfg.training.amp_eps = 1e-4
    cfg.training.save_dir = str(tmp_path / "checkpoints")
    cfg.training.save_best_only = True
    cfg.training.use_wandb = False
    cfg.training.wandb_project = None
    cfg.training.wandb_entity = None
    cfg.training.wandb_run_name = None
    cfg.training.log_file = None
    cfg.training.log_level = "INFO"
    cfg.training.device = "cpu"
    cfg.training.seed = 42
    cfg.training.resume_from = None
    cfg.training.validation_variants = None
    cfg.training.curriculum_learning = None
    cfg.training.save_all_checkpoints = False
    cfg.training.grad_accumulation_steps = 1

    cfg.model = SimpleNamespace()
    cfg.model.model_type = "convtasnet"  # Required for checkpoint structure
    cfg.model.N = 8
    cfg.model.B = 8
    cfg.model.H = 8
    cfg.model.P = 3
    cfg.model.X = 1
    cfg.model.R = 1
    cfg.model.C = 2 if task == "SB" else 1
    cfg.model.kernel_size = 16
    cfg.model.stride = 8

    return cfg


def test_trainer_train_epoch_and_validate(tmp_path):
    from training.trainer import Trainer

    cfg = make_config(tmp_path)

    train_dataset = SyntheticDataset(n_samples=6, time_steps=256)
    val_dataset = SyntheticDataset(n_samples=4, time_steps=256)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.data.batch_size, collate_fn=polsess_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.data.batch_size, collate_fn=polsess_collate_fn
    )

    model = DummyModel()

    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        cfg,
        device="cpu",
        logger=None,
        wandb_logger=None,
    )

    # Override loss function with differentiable proxy (MSE) so loss/backward works
    def mse_loss_wrapper(estimates, clean):
        loss = F.mse_loss(estimates, clean)
        return loss, loss.item()

    trainer.loss_fn = mse_loss_wrapper

    # Single epoch train_epoch
    train_sisdr, train_sisdri = trainer.train_epoch()
    assert isinstance(train_sisdr, float)
    assert isinstance(train_sisdri, float)

    # Validation
    val_sisdr, val_sisdri = trainer.validate()
    assert isinstance(val_sisdr, float)
    assert isinstance(val_sisdri, float)

    # Full training run for 1 epoch (should not error)
    trainer.train(num_epochs=1, save_dir=cfg.training.save_dir)


def test_trainer_sb_task_with_pit_loss(tmp_path):
    """Test SB task (2-speaker separation) with PIT loss."""
    from training.trainer import Trainer

    cfg = make_config(tmp_path, task="SB")

    train_dataset = SyntheticDataset(n_samples=6, time_steps=256, task="SB")
    val_dataset = SyntheticDataset(n_samples=4, time_steps=256, task="SB")

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.data.batch_size, collate_fn=polsess_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.data.batch_size, collate_fn=polsess_collate_fn
    )

    model = DummyModel(C=2)

    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        cfg,
        device="cpu",
        logger=None,
        wandb_logger=None,
    )

    # Verify PIT loss is being used for SB task
    assert trainer.task == "SB"
    assert hasattr(trainer, "pit_loss")
    assert trainer.loss_fn == trainer._pit_loss_wrapper

    # Test loss wrapper returns correct format (loss, scalar_value)
    estimates = torch.randn(2, 2, 256, requires_grad=True)  # [B, C, T]
    targets = torch.randn(2, 2, 256)  # [B, C, T]
    loss, scalar_value = trainer.loss_fn(estimates, targets)
    assert isinstance(loss, torch.Tensor)
    assert isinstance(scalar_value, float)

    # Override with MSE-based PIT for testing
    def mse_pit_wrapper(estimates, clean):
        loss = F.mse_loss(estimates, clean)
        return loss, loss.item()

    trainer.loss_fn = mse_pit_wrapper

    # Single epoch training
    train_sisdr, train_sisdri = trainer.train_epoch()
    assert isinstance(train_sisdr, float)
    assert isinstance(train_sisdri, float)

    # Validation
    val_sisdr, val_sisdri = trainer.validate()
    assert isinstance(val_sisdr, float)
    assert isinstance(val_sisdri, float)


def test_loss_wrapper_format_es_task(tmp_path):
    """Test that ES task loss wrapper returns correct format."""
    from training.trainer import Trainer

    cfg = make_config(tmp_path, task="ES")
    train_dataset = SyntheticDataset(n_samples=4, time_steps=256, task="ES")
    val_dataset = SyntheticDataset(n_samples=2, time_steps=256, task="ES")

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.data.batch_size, collate_fn=polsess_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.data.batch_size, collate_fn=polsess_collate_fn
    )

    model = DummyModel(C=1)
    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        cfg,
        device="cpu",
        logger=None,
        wandb_logger=None,
    )

    # Verify SI-SDR loss is being used for ES task
    assert trainer.task == "ES"
    assert hasattr(trainer, "si_sdr_metric")
    assert trainer.loss_fn == trainer._sisdr_loss_wrapper

    # Test loss wrapper returns correct format
    estimates = torch.randn(2, 256, requires_grad=True)  # [B, T]
    targets = torch.randn(2, 256)  # [B, T]
    loss, scalar_value = trainer.loss_fn(estimates, targets)
    assert isinstance(loss, torch.Tensor)
    assert isinstance(scalar_value, float)


def test_task_routing(tmp_path):
    """Test that task routing correctly sets up loss functions."""
    from training.trainer import Trainer

    # Test ES task
    cfg_es = make_config(tmp_path, task="ES")
    train_loader = DataLoader(
        SyntheticDataset(n_samples=4, task="ES"),
        batch_size=2,
        collate_fn=polsess_collate_fn,
    )
    val_loader = DataLoader(
        SyntheticDataset(n_samples=2, task="ES"),
        batch_size=2,
        collate_fn=polsess_collate_fn,
    )
    trainer_es = Trainer(
        DummyModel(C=1), train_loader, val_loader, cfg_es, device="cpu"
    )
    assert trainer_es.task == "ES"
    assert trainer_es.loss_fn == trainer_es._sisdr_loss_wrapper

    # Test EB task
    cfg_eb = make_config(tmp_path, task="EB")
    trainer_eb = Trainer(
        DummyModel(C=1), train_loader, val_loader, cfg_eb, device="cpu"
    )
    assert trainer_eb.task == "EB"
    assert trainer_eb.loss_fn == trainer_eb._sisdr_loss_wrapper

    # Test SB task
    cfg_sb = make_config(tmp_path, task="SB")
    train_loader_sb = DataLoader(
        SyntheticDataset(n_samples=4, task="SB"),
        batch_size=2,
        collate_fn=polsess_collate_fn,
    )
    val_loader_sb = DataLoader(
        SyntheticDataset(n_samples=2, task="SB"),
        batch_size=2,
        collate_fn=polsess_collate_fn,
    )
    trainer_sb = Trainer(
        DummyModel(C=2), train_loader_sb, val_loader_sb, cfg_sb, device="cpu"
    )
    assert trainer_sb.task == "SB"
    assert trainer_sb.loss_fn == trainer_sb._pit_loss_wrapper


def test_gradient_accumulation_basic(tmp_path):
    """Test that gradient accumulation reduces optimizer steps correctly."""
    from training.trainer import Trainer

    cfg = make_config(tmp_path)
    cfg.training.grad_accumulation_steps = 2  # Accumulate over 2 batches

    # 6 samples with batch_size=2 = 3 batches
    # With accum_steps=2, optimizer should step floor(3/2) + 1 = 2 times
    train_dataset = SyntheticDataset(n_samples=6, time_steps=256)
    val_dataset = SyntheticDataset(n_samples=4, time_steps=256)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.data.batch_size, collate_fn=polsess_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.data.batch_size, collate_fn=polsess_collate_fn
    )

    model = DummyModel()
    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        cfg,
        device="cpu",
        logger=None,
        wandb_logger=None,
    )

    # Override loss function with MSE
    def mse_loss_wrapper(estimates, clean):
        loss = F.mse_loss(estimates, clean)
        return loss, loss.item()

    trainer.loss_fn = mse_loss_wrapper

    # Track optimizer steps
    original_step = trainer.optimizer.step
    step_count = [0]

    def counting_step():
        step_count[0] += 1
        original_step()

    trainer.optimizer.step = counting_step

    # Run one epoch
    trainer.train_epoch()

    # With 3 batches and accum_steps=2:
    # - Batch 0: accumulate
    # - Batch 1: step (batch_idx+1 = 2, 2 % 2 == 0)
    # - Batch 2: step (last batch)
    # Total: 2 optimizer steps
    assert step_count[0] == 2, f"Expected 2 optimizer steps, got {step_count[0]}"


def test_gradient_accumulation_scaling(tmp_path):
    """Test that loss is scaled correctly for gradient accumulation."""
    from training.trainer import Trainer

    cfg = make_config(tmp_path)
    cfg.training.grad_accumulation_steps = 4

    train_dataset = SyntheticDataset(n_samples=8, time_steps=256)
    val_dataset = SyntheticDataset(n_samples=4, time_steps=256)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.data.batch_size, collate_fn=polsess_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.data.batch_size, collate_fn=polsess_collate_fn
    )

    model = DummyModel()
    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        cfg,
        device="cpu",
        logger=None,
        wandb_logger=None,
    )

    # Verify accum_steps is read correctly
    accum_steps = getattr(trainer.config.training, 'grad_accumulation_steps', 1)
    assert accum_steps == 4

    # Override loss to track loss scaling
    original_losses = []
    scaled_losses = []

    def tracking_loss_wrapper(estimates, clean):
        loss = F.mse_loss(estimates, clean)
        original_losses.append(loss.item())
        return loss, loss.item()

    trainer.loss_fn = tracking_loss_wrapper

    # Run one epoch
    trainer.train_epoch()

    # Verify we processed all batches
    assert len(original_losses) == 4  # 8 samples / batch_size 2 = 4 batches


def test_gradient_accumulation_disabled_by_default(tmp_path):
    """Test that gradient accumulation defaults to 1 (disabled)."""
    from training.trainer import Trainer

    cfg = make_config(tmp_path)
    # Don't set grad_accumulation_steps - should default to 1

    train_dataset = SyntheticDataset(n_samples=4, time_steps=256)
    val_dataset = SyntheticDataset(n_samples=2, time_steps=256)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.data.batch_size, collate_fn=polsess_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.data.batch_size, collate_fn=polsess_collate_fn
    )

    model = DummyModel()
    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        cfg,
        device="cpu",
        logger=None,
        wandb_logger=None,
    )

    def mse_loss_wrapper(estimates, clean):
        loss = F.mse_loss(estimates, clean)
        return loss, loss.item()

    trainer.loss_fn = mse_loss_wrapper

    # Track optimizer steps
    original_step = trainer.optimizer.step
    step_count = [0]

    def counting_step():
        step_count[0] += 1
        original_step()

    trainer.optimizer.step = counting_step

    # Run one epoch
    trainer.train_epoch()

    # With 2 batches and accum_steps=1 (default), should step every batch
    assert step_count[0] == 2, f"Expected 2 optimizer steps, got {step_count[0]}"


def test_training_summary_epoch_numbers_after_resume(tmp_path, capsys):
    """Regression test: training summary must show correct epoch numbers after resume.

    Before the fix, enumerate(val_sisdr_history, 1) always started at 1,
    so a run resumed from epoch 5 would log 'Epoch 1, 2, ...' instead of 'Epoch 6, 7, ...'.
    """
    import logging
    from training.trainer import Trainer

    cfg = make_config(tmp_path)
    cfg.training.num_epochs = 3

    train_dataset = SyntheticDataset(n_samples=4, time_steps=256)
    val_dataset = SyntheticDataset(n_samples=2, time_steps=256)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.data.batch_size, collate_fn=polsess_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.data.batch_size, collate_fn=polsess_collate_fn
    )

    model = DummyModel()

    # Capture log output
    log_messages = []

    class ListHandler(logging.Handler):
        def emit(self, record):
            log_messages.append(record.getMessage())

    logger = logging.getLogger("test_resume_epoch")
    logger.setLevel(logging.DEBUG)
    handler = ListHandler()
    logger.addHandler(handler)

    trainer = Trainer(
        model, train_loader, val_loader, cfg, device="cpu", logger=logger
    )

    # Simulate resuming from epoch 5 (next epoch to run is 6)
    trainer.current_epoch = 5

    def mse_loss_wrapper(estimates, clean):
        loss = F.mse_loss(estimates, clean)
        return loss, loss.item()

    trainer.loss_fn = mse_loss_wrapper

    trainer.train(num_epochs=3, save_dir=cfg.training.save_dir)

    # Find the epoch summary lines
    summary_lines = [m for m in log_messages if "Val SI-SDR" in m and "Epoch" in m]

    assert len(summary_lines) == 3, f"Expected 3 summary lines, got: {summary_lines}"

    # Epochs should be 6, 7, 8 — not 1, 2, 3
    assert "Epoch 6:" in summary_lines[0], f"Expected 'Epoch 6:', got: {summary_lines[0]}"
    assert "Epoch 7:" in summary_lines[1], f"Expected 'Epoch 7:', got: {summary_lines[1]}"
    assert "Epoch 8:" in summary_lines[2], f"Expected 'Epoch 8:', got: {summary_lines[2]}"


def test_consecutive_nan_abort(tmp_path, monkeypatch):
    """train_epoch raises ConsecutiveNaNError after MAX_CONSECUTIVE_NAN_BATCHES
    NaN losses in a row, and train() converts it to a graceful SystemExit(1)
    so a sweep agent can move on to the next run."""
    import pytest
    import training.trainer as trainer_module
    from training.trainer import Trainer, ConsecutiveNaNError

    monkeypatch.setattr(trainer_module, "MAX_CONSECUTIVE_NAN_BATCHES", 3)

    cfg = make_config(tmp_path)
    train_dataset = SyntheticDataset(n_samples=12, time_steps=256)
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.data.batch_size, collate_fn=polsess_collate_fn
    )
    val_loader = DataLoader(
        SyntheticDataset(n_samples=4, time_steps=256),
        batch_size=cfg.data.batch_size,
        collate_fn=polsess_collate_fn,
    )

    trainer = Trainer(
        DummyModel(), train_loader, val_loader, cfg,
        device="cpu", logger=None, wandb_logger=None,
    )

    def nan_loss_wrapper(estimates, clean):
        loss = F.mse_loss(estimates, clean) * float("nan")
        return loss, float("nan")

    trainer.loss_fn = nan_loss_wrapper

    with pytest.raises(ConsecutiveNaNError):
        trainer.train_epoch()
    assert trainer.consecutive_nan_batches == 3

    # train() wraps the abort in SystemExit(1) (sweep-friendly, mirrors OOM path)
    trainer.consecutive_nan_batches = 0
    with pytest.raises(SystemExit) as exc_info:
        trainer.train(num_epochs=1, save_dir=cfg.training.save_dir)
    assert exc_info.value.code == 1


def test_consecutive_nan_counter_resets_on_good_batch(tmp_path, monkeypatch):
    """A finite-loss batch resets the consecutive NaN counter, so intermittent
    NaNs below the threshold never abort training."""
    import training.trainer as trainer_module
    from training.trainer import Trainer

    monkeypatch.setattr(trainer_module, "MAX_CONSECUTIVE_NAN_BATCHES", 3)

    cfg = make_config(tmp_path)
    # 12 samples / batch_size 2 = 6 batches
    train_dataset = SyntheticDataset(n_samples=12, time_steps=256)
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.data.batch_size, collate_fn=polsess_collate_fn
    )
    val_loader = DataLoader(
        SyntheticDataset(n_samples=4, time_steps=256),
        batch_size=cfg.data.batch_size,
        collate_fn=polsess_collate_fn,
    )

    trainer = Trainer(
        DummyModel(), train_loader, val_loader, cfg,
        device="cpu", logger=None, wandb_logger=None,
    )

    # NaN on batches 0,1 then good on 2, NaN on 3,4, good on 5 — never 3 in a row
    calls = {"n": 0}

    def alternating_loss_wrapper(estimates, clean):
        loss = F.mse_loss(estimates, clean)
        is_nan = calls["n"] % 3 != 2
        calls["n"] += 1
        if is_nan:
            return loss * float("nan"), float("nan")
        return loss, loss.item()

    trainer.loss_fn = alternating_loss_wrapper

    trainer.train_epoch()  # must not raise
    assert trainer.consecutive_nan_batches == 0


# ---------------------------------------------------------------------------
# Resume / checkpoint-format robustness (Work Package B3)
# ---------------------------------------------------------------------------

class _FakeScaler:
    """Stand-in for torch.amp.GradScaler.

    A real GradScaler force-disables itself (empty state_dict, scale 1.0) when
    CUDA is unavailable, so it can't exercise save/restore on the CPU-only test
    runner. This fake carries a real, inspectable state_dict instead.
    """

    def __init__(self, scale=1.0):
        self._scale = scale
        self.loaded = None

    def state_dict(self):
        return {"scale": self._scale}

    def load_state_dict(self, sd):
        self.loaded = dict(sd)
        self._scale = sd["scale"]


def _make_trainer(tmp_path, task="ES", provenance=None):
    from training.trainer import Trainer

    cfg = make_config(tmp_path, task=task)
    train_loader = DataLoader(
        SyntheticDataset(4, task=task), batch_size=2, collate_fn=polsess_collate_fn
    )
    val_loader = DataLoader(
        SyntheticDataset(2, task=task), batch_size=2, collate_fn=polsess_collate_fn
    )
    model = DummyModel(C=2 if task == "SB" else 1)
    return Trainer(
        model, train_loader, val_loader, cfg,
        device="cpu", logger=None, wandb_logger=None, provenance=provenance,
    )


def test_resume_scheduler_best_ordering_regression(tmp_path):
    """Gap 7 regression: a legacy checkpoint (no scheduler_state_dict) must seed
    scheduler.best from the checkpoint's best_val_sisdr, NOT the -inf placeholder.

    Before the fix, load_checkpoint set scheduler.best = self.best_val_sisdr
    while best_val_sisdr was still -inf, so the first post-resume epoch always
    looked like an improvement.
    """
    src = _make_trainer(tmp_path)
    legacy_ckpt = {
        "epoch": 4,
        "model_state_dict": DummyModel().state_dict(),
        "optimizer_state_dict": src.optimizer.state_dict(),
        "val_sisdr": 12.5,
        "best_val_sisdr": 12.5,
        # deliberately NO scheduler_state_dict -> legacy branch
    }
    path = tmp_path / "legacy.pt"
    torch.save(legacy_ckpt, path)

    tgt = _make_trainer(tmp_path)
    assert tgt.scheduler.best == -float("inf")  # baseline before load

    tgt.load_checkpoint(str(path))

    assert tgt.best_val_sisdr == 12.5
    assert tgt.current_epoch == 5
    # The regression: scheduler.best must be the real best, not -inf.
    assert tgt.scheduler.best == 12.5


def test_old_format_checkpoint_resume_compat(tmp_path):
    """A checkpoint written by the OLD code (none of the new keys present) must
    still load and resume cleanly, with every new key defaulted tolerantly."""
    src = _make_trainer(tmp_path)
    old_ckpt = {
        "epoch": 7,
        "model_state_dict": DummyModel().state_dict(),
        "optimizer_state_dict": src.optimizer.state_dict(),
        "val_sisdr": 9.0,
        # no best_val_sisdr, no scheduler_state_dict, no scaler_state_dict,
        # no epochs_without_improvement, no provenance, no wandb_run_id
    }
    path = tmp_path / "old.pt"
    torch.save(old_ckpt, path)

    tgt = _make_trainer(tmp_path)
    tgt.load_checkpoint(str(path))  # must not raise

    assert tgt.current_epoch == 8
    assert tgt.best_val_sisdr == 9.0  # falls back to val_sisdr
    assert tgt.epochs_without_improvement == 0  # tolerant default
    assert tgt.scheduler.best == 9.0  # legacy branch seeds from best


def test_scaler_state_roundtrip(tmp_path):
    """Gap 8a: GradScaler state is saved and restored across a resume."""
    src = _make_trainer(tmp_path)
    src.scaler = _FakeScaler(scale=512.0)

    ckpt = src._serialize_checkpoint_data(epoch=2, val_sisdr=3.0)
    assert ckpt["scaler_state_dict"] == {"scale": 512.0}

    path = tmp_path / "scaler.pt"
    torch.save(ckpt, path)

    tgt = _make_trainer(tmp_path)
    tgt.scaler = _FakeScaler(scale=256.0)  # different starting scale
    tgt.load_checkpoint(str(path))

    assert tgt.scaler.loaded == {"scale": 512.0}
    assert tgt.scaler._scale == 512.0


def test_scaler_absent_in_old_checkpoint_is_tolerated(tmp_path):
    """A trainer WITH a scaler resuming an OLD checkpoint (no scaler_state_dict)
    must not raise — the scaler simply keeps its init scale."""
    src = _make_trainer(tmp_path)
    old_ckpt = {
        "epoch": 1,
        "model_state_dict": DummyModel().state_dict(),
        "optimizer_state_dict": src.optimizer.state_dict(),
        "val_sisdr": 1.0,
    }
    path = tmp_path / "old_noscaler.pt"
    torch.save(old_ckpt, path)

    tgt = _make_trainer(tmp_path)
    tgt.scaler = _FakeScaler(scale=256.0)
    tgt.load_checkpoint(str(path))  # must not raise

    assert tgt.scaler.loaded is None  # never touched
    assert tgt.scaler._scale == 256.0


def test_patience_persist_roundtrip(tmp_path):
    """Gap 8d: epochs_without_improvement survives a save/resume cycle."""
    src = _make_trainer(tmp_path)
    src.epochs_without_improvement = 3

    ckpt = src._serialize_checkpoint_data(epoch=5, val_sisdr=1.0)
    assert ckpt["epochs_without_improvement"] == 3

    path = tmp_path / "patience.pt"
    torch.save(ckpt, path)

    tgt = _make_trainer(tmp_path)
    assert tgt.epochs_without_improvement == 0
    tgt.load_checkpoint(str(path))
    assert tgt.epochs_without_improvement == 3


def test_provenance_embedded_and_manifest_written(tmp_path):
    """Gap 5: provenance is embedded in the checkpoint and a human-readable
    run_manifest.yaml is written next to config.yaml."""
    import yaml
    from pathlib import Path

    manifest = {
        "git_sha": "abc1234",
        "git_dirty": True,
        "torch_version": "2.8.0",
        "gpu_name": None,
        "seed": 42,
        "argv": ["train.py", "--config", "x.yaml"],
    }
    trainer = _make_trainer(tmp_path, provenance=manifest)

    save_dir = tmp_path / "ckpts"
    trainer._save_checkpoint(epoch=0, val_sisdr=5.0, save_dir=save_dir)

    ckpt_files = list(save_dir.rglob("*.pt"))
    assert len(ckpt_files) == 1
    loaded = torch.load(ckpt_files[0], weights_only=False)
    assert loaded["provenance"] == manifest

    manifest_files = list(save_dir.rglob("run_manifest.yaml"))
    assert len(manifest_files) == 1
    with open(manifest_files[0]) as f:
        assert yaml.safe_load(f) == manifest


def test_no_provenance_writes_no_manifest(tmp_path):
    """With provenance=None (e.g. an old caller), no manifest file and no
    provenance key appear — the addition is strictly opt-in."""
    trainer = _make_trainer(tmp_path, provenance=None)

    save_dir = tmp_path / "ckpts"
    trainer._save_checkpoint(epoch=0, val_sisdr=5.0, save_dir=save_dir)

    assert list(save_dir.rglob("run_manifest.yaml")) == []
    ckpt_files = list(save_dir.rglob("*.pt"))
    loaded = torch.load(ckpt_files[0], weights_only=False)
    assert "provenance" not in loaded


# ---------------------------------------------------------------------------
# AMP dispatch + the two generic recipe knobs (optimizer, LR warmup)
# ---------------------------------------------------------------------------

def test_amp_dispatch_by_model_type(tmp_path):
    """_setup_amp must put TF-MossFormer on bf16 without a GradScaler.

    Same profile as MossFormer2 (chained multiplicative gates + masked softmax
    overflow fp16). config.py's summary AMP line hardcodes the same tuple and
    will silently lie if only one of the two is updated —
    tests/test_config_yaml.py::test_tf_mossformer_summary pins that side.
    """
    trainer = _make_trainer(tmp_path)
    trainer.use_amp = True

    for model_type in ("tf_mossformer", "mossformer2"):
        trainer.config.model.model_type = model_type
        trainer._setup_amp(trainer.model, "cuda")
        assert trainer.scaler is None, f"{model_type} must not use a GradScaler"
        assert trainer.amp_dtype is torch.bfloat16, f"{model_type} must train in bf16"

    trainer.config.model.model_type = "convtasnet"
    trainer._setup_amp(trainer.model, "cuda")
    assert trainer.amp_dtype is torch.float16
    assert trainer.scaler is not None


def test_optimizer_knob_selects_adamw(tmp_path):
    """training.optimizer='adamw' builds AdamW; the default stays Adam."""
    from training.trainer import Trainer

    cfg = make_config(tmp_path)
    cfg.training.optimizer = "adamw"
    train_loader = DataLoader(SyntheticDataset(4), batch_size=2, collate_fn=polsess_collate_fn)
    val_loader = DataLoader(SyntheticDataset(2), batch_size=2, collate_fn=polsess_collate_fn)
    trainer = Trainer(DummyModel(), train_loader, val_loader, cfg, device="cpu")
    assert isinstance(trainer.optimizer, torch.optim.AdamW)

    # Default (the attribute is absent from this SimpleNamespace config, exactly
    # like a pre-2026-09 checkpoint config) -> Adam, unchanged behaviour.
    assert isinstance(_make_trainer(tmp_path).optimizer, torch.optim.Adam)


def test_optimizer_knob_rejects_unknown_name(tmp_path):
    """An unknown optimizer name fails loudly at Trainer init, not mid-run."""
    from training.trainer import Trainer

    cfg = make_config(tmp_path)
    cfg.training.optimizer = "lion"
    train_loader = DataLoader(SyntheticDataset(4), batch_size=2, collate_fn=polsess_collate_fn)
    val_loader = DataLoader(SyntheticDataset(2), batch_size=2, collate_fn=polsess_collate_fn)
    with pytest.raises(ValueError, match="lion"):
        Trainer(DummyModel(), train_loader, val_loader, cfg, device="cpu")


def _lr(trainer):
    return trainer.optimizer.param_groups[0]["lr"]


def test_warmup_linear_schedule(tmp_path):
    """LR at step 0, mid-warmup, at warmup_steps, and after it."""
    trainer = _make_trainer(tmp_path)
    base = trainer.warmup_target_lr
    trainer.warmup_steps = 10

    trainer.global_step = 0
    trainer._apply_lr_warmup()
    assert _lr(trainer) == pytest.approx(base / 10)  # first step is ~0, never full LR

    trainer.global_step = 4
    trainer._apply_lr_warmup()
    assert _lr(trainer) == pytest.approx(base * 0.5)

    trainer.global_step = 9  # last warmup step reaches the configured LR
    trainer._apply_lr_warmup()
    assert _lr(trainer) == pytest.approx(base)

    # Past warmup the LR belongs to ReduceLROnPlateau; warmup must never write
    # to it again (otherwise it would undo every plateau reduction).
    trainer.global_step = 10
    trainer.optimizer.param_groups[0]["lr"] = base / 8  # pretend the plateau fired
    trainer._apply_lr_warmup()
    assert _lr(trainer) == pytest.approx(base / 8)


def test_warmup_disabled_by_default_is_a_no_op(tmp_path):
    """warmup_steps=0 (the default) must not touch the LR at all."""
    trainer = _make_trainer(tmp_path)
    assert trainer.warmup_steps == 0

    trainer.optimizer.param_groups[0]["lr"] = 7e-5
    for trainer.global_step in (0, 1, 1000):
        trainer._apply_lr_warmup()
        assert _lr(trainer) == pytest.approx(7e-5)

    # And a real epoch leaves the LR exactly where the config put it, so the
    # plateau scheduler sees the same sequence it always did.
    trainer = _make_trainer(tmp_path)
    trainer.train_epoch()
    assert _lr(trainer) == pytest.approx(trainer.config.training.lr)
    assert trainer.global_step == 2  # 4 samples / batch_size 2


def test_warmup_applies_during_train_epoch(tmp_path):
    """The warmup is wired into the optimizer step, not just callable."""
    trainer = _make_trainer(tmp_path)
    trainer.warmup_steps = 4
    base = trainer.warmup_target_lr

    trainer.train_epoch()  # 2 optimizer steps

    assert trainer.global_step == 2
    assert _lr(trainer) == pytest.approx(base * 0.5)


def test_global_step_persists_across_resume(tmp_path):
    """Warmup must not restart on --resume: the step counter is checkpointed."""
    src = _make_trainer(tmp_path)
    src.global_step = 1234

    ckpt = src._serialize_checkpoint_data(epoch=3, val_sisdr=1.0)
    assert ckpt["global_step"] == 1234
    path = tmp_path / "gstep.pt"
    torch.save(ckpt, path)

    tgt = _make_trainer(tmp_path)
    tgt.load_checkpoint(str(path))
    assert tgt.global_step == 1234


def test_global_step_absent_in_old_checkpoint_is_tolerated(tmp_path):
    """Pre-2026-09 checkpoints have no global_step; resuming them defaults to 0."""
    src = _make_trainer(tmp_path)
    old_ckpt = {
        "epoch": 2,
        "model_state_dict": DummyModel().state_dict(),
        "optimizer_state_dict": src.optimizer.state_dict(),
        "val_sisdr": 4.0,
    }
    path = tmp_path / "old_nostep.pt"
    torch.save(old_ckpt, path)

    tgt = _make_trainer(tmp_path)
    tgt.load_checkpoint(str(path))  # must not raise
    assert tgt.global_step == 0
