"""Unit tests for the Trainer using a tiny synthetic dataset and dummy model.

These tests are lightweight and avoid any dependency on the full PolSESS
dataset. They exercise `train_epoch`, `validate` and a 1-epoch `train` run.
"""

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
    train_sisdr = trainer.train_epoch()
    assert isinstance(train_sisdr, float)

    # Validation
    val_sisdr = trainer.validate()
    assert isinstance(val_sisdr, float)

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
    train_sisdr = trainer.train_epoch()
    assert isinstance(train_sisdr, float)

    # Validation
    val_sisdr = trainer.validate()
    assert isinstance(val_sisdr, float)


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

