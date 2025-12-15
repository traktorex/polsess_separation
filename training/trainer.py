"""Trainer class for speech enhancement/separation models."""

import time
import torch
import logging
import os
import yaml
from pathlib import Path
from tqdm import tqdm
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
from typing import Optional, Any
from dataclasses import is_dataclass, asdict
from config import Config
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr

DEFAULT_SISDR_FALLBACK = -999.0  # Default SI-SDR when not found in checkpoint


class Trainer:
    """Trainer for speech separation models with AMP."""

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        config: Config,
        device: str = "cuda",
        logger: Optional[logging.Logger] = None,
        wandb_logger: Optional[Any] = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.use_amp = config.training.use_amp
        self.logger = logger or logging.getLogger("polsess")
        self.wandb_logger = wandb_logger

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.training.lr,
            weight_decay=config.training.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=config.training.lr_factor,
            patience=config.training.lr_patience,
        )

        # Task-specific loss function
        self.task = config.data.task
        if self.task == "SB":
            self.pit_loss = PITLossWrapper(
                pairwise_neg_sisdr, pit_from="pw_mtx"  # Pairwise matrix mode
            ).to(device)
            self.loss_fn = self._pit_loss_wrapper

        else:
            self.si_sdr_metric = ScaleInvariantSignalDistortionRatio().to(device)
            self.loss_fn = self._sisdr_loss_wrapper

        self.scaler = (
            torch.amp.GradScaler("cuda")
            if (self.use_amp and device == "cuda")
            else None
        )

        self.best_val_sisdr = -float("inf")
        self.current_epoch = 0

        # Curriculum learning setup
        self.curriculum_schedule = config.training.curriculum_learning
        self.lr_scheduler_enabled = (
            self.curriculum_schedule is None
        )  # Enabled by default if no curriculum

    def _get_curriculum_variants(self, epoch):
        """Get allowed variants for current epoch from curriculum schedule.

        If no schedule for this epoch, returns most recent schedule's variants.
        """
        if not self.curriculum_schedule:
            return None

        # Find the most recent curriculum entry for this epoch
        current_variants = None
        for entry in self.curriculum_schedule:
            if entry["epoch"] <= epoch:
                current_variants = entry.get("variants")
            else:
                break

        return current_variants

    def _should_enable_lr_scheduler(self, epoch):
        """Check if LR scheduler should be enabled at this epoch."""
        if not self.curriculum_schedule:
            return True  # Always enabled if no curriculum

        for entry in self.curriculum_schedule:
            if entry["epoch"] == epoch and entry.get("lr_scheduler") == "start":
                return True

        return False

    def _update_training_variants(self, epoch):
        """Update training dataset's allowed_variants based on curriculum schedule."""
        if not self.curriculum_schedule:
            return  # No curriculum learning

        variants = self._get_curriculum_variants(epoch)
        if variants is not None and (
            self.current_epoch == 1
            or self.train_loader.dataset.allowed_variants != variants
        ):
            self.train_loader.dataset.allowed_variants = variants
            self.logger.info(f"Training variants set to: {variants}")

        # Check if we should enable LR scheduler at this epoch
        if not self.lr_scheduler_enabled and self._should_enable_lr_scheduler(epoch):
            self.lr_scheduler_enabled = True
            self.logger.info("Learning rate scheduler enabled")

    def _sisdr_loss_wrapper(self, estimates, targets):
        """Compute SI-SDR loss for ES/EB tasks."""
        sisdr = self.si_sdr_metric(estimates, targets)
        return -sisdr, sisdr.item()

    def _pit_loss_wrapper(self, estimates, targets):
        """Compute PIT-SI-SDR loss for SB task."""
        loss = self.pit_loss(estimates, targets)
        # Loss is negative SI-SDR averaged over batch
        sisdr = -loss.item()
        return loss, sisdr

    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint to resume training.

        Handles both compiled and non-compiled models by unwrapping
        torch.compile() wrapper if needed.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Get underlying model if compiled with torch.compile()
        model_to_load = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model

        model_to_load.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"] + 1  # Resume from next epoch
        self.best_val_sisdr = checkpoint.get(
            "best_val_sisdr", checkpoint.get("val_sisdr", DEFAULT_SISDR_FALLBACK)
        )
        self.logger.info(
            f"Resumed from epoch {checkpoint['epoch']}, best SI-SDR: {self.best_val_sisdr:.2f} dB"
        )

    def _save_checkpoint(self, epoch: int, val_sisdr: float, save_dir):
        """Save model checkpoint with best validation SI-SDR.

        Structure: checkpoints/{model}/{task}/{experiment_name}/{run_id}/best_model.pt
        Also creates/updates symlink: checkpoints/{model}/{task}/{experiment_name}/latest/
        """
        base_dir = Path(save_dir)

        # Determine experiment name and run_id
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

        # Get experiment name from wandb_run_name or config
        experiment_name = getattr(self.config.training, 'experiment_name', None)
        if not experiment_name and self.config.training.wandb_run_name:
            # Use wandb_run_name as experiment name (e.g., "spmamba_sb_baseline")
            experiment_name = self.config.training.wandb_run_name
        if not experiment_name:
            experiment_name = "default"

        # Create run_id from timestamp or wandb run name
        if (
            self.wandb_logger
            and self.wandb_logger.enabled
            and self.wandb_logger.run
            and self.wandb_logger.run.name
        ):
            run_id = f"{self.wandb_logger.run.name}_{timestamp}"
        else:
            run_id = f"run_{timestamp}"

        # New structure: checkpoints/{model}/{task}/{experiment_name}/{run_id}/
        model_type = self.config.model.model_type
        task = self.config.data.task
        checkpoint_dir = base_dir / model_type / task / experiment_name / run_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / "best_model.pt"
        config_path = checkpoint_dir / "config.yaml"

        def to_dict(obj):
            """Convert dataclass or SimpleNamespace to dict."""
            if is_dataclass(obj):
                return asdict(obj)
            elif hasattr(obj, "__dict__"):
                return vars(obj)
            else:
                return obj

        config_dict = {
            "model": to_dict(self.config.model),
            "data": to_dict(self.config.data),
            "training": to_dict(self.config.training),
        }

        # Get state dict from unwrapped model if using torch.compile
        model_to_save = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model

        # Save checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model_to_save.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_sisdr": val_sisdr,
                "config": config_dict,
            },
            checkpoint_path,
        )

        # Save config as YAML for easy viewing
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        # Create/update 'latest' symlink within experiment folder
        latest_link = base_dir / model_type / task / experiment_name / "latest"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()

        # Create relative symlink (more portable)
        try:
            if os.name == 'nt':  # Windows
                # Windows requires admin for symlinks, use junction instead
                import subprocess
                subprocess.run(['mklink', '/J', str(latest_link), str(checkpoint_dir.name)],
                             shell=True, check=False, capture_output=True)
            else:  # Unix/Linux
                latest_link.symlink_to(run_id, target_is_directory=True)
        except Exception as e:
            self.logger.warning(f"Could not create 'latest' symlink: {e}")

        self.logger.info(
            f"Saved best model (SI-SDR: {val_sisdr:.2f} dB) to: {checkpoint_path}"
        )
        self.logger.info(f"  Run ID: {run_id}")
        self.logger.info(f"  Latest link: {latest_link}")

        if self.wandb_logger:
            self.wandb_logger.log_metrics(
                {"best_val_sisdr": val_sisdr}, step=self.current_epoch
            )
            self.wandb_logger.log_model(str(checkpoint_path))
        else:
            self.logger.info(f"Logged model artifact: {checkpoint_path}")

    def train_epoch(self) -> float:
        """Train for one epoch and return average SI-SDR."""
        self.model.train()
        total_sisdr = 0
        total_samples = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch}",
            unit="batch",
            leave=True,
            ncols=100,
        )

        for batch_idx, batch in enumerate(pbar):
            mix = batch["mix"].to(self.device)
            clean = batch["clean"].to(self.device)

            if self.scaler:
                with torch.amp.autocast("cuda"):
                    mix_input = mix.unsqueeze(1)
                    estimates = self.model(mix_input)
            else:
                mix_input = mix.unsqueeze(1)
                estimates = self.model(mix_input)

            loss, sisdr_value = self.loss_fn(estimates, clean)

            if torch.isnan(loss) or torch.isinf(loss):
                self.logger.warning(
                    f"NaN/Inf detected at batch {batch_idx}, skipping batch"
                )
                self.logger.warning(f"  Loss: {loss.item()}, SI-SDR: {sisdr_value}")
                self.optimizer.zero_grad()
                continue

            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient clipping and optimizer step
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config.training.grad_clip_norm,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config.training.grad_clip_norm,
                )
                self.optimizer.step()

            self.optimizer.zero_grad()

            # Weight by actual batch size for correct averaging
            batch_size = len(mix)
            total_sisdr += sisdr_value * batch_size
            total_samples += batch_size

            pbar.set_postfix(
                {
                    "SI-SDR": f"{sisdr_value:.2f}dB",
                    "LR": f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                }
            )

        avg_sisdr = total_sisdr / total_samples
        return avg_sisdr

    def validate(self) -> float:
        """Run validation and return average SI-SDR."""
        self.model.eval()
        total_sisdr = 0
        total_samples = 0

        pbar = tqdm(
            self.val_loader,
            desc="Validation",
            unit="batch",
            leave=False,
            ncols=100,
        )

        with torch.no_grad():
            for batch in pbar:
                mix = batch["mix"].to(self.device)
                clean = batch["clean"].to(self.device)

                if self.scaler:
                    with torch.amp.autocast("cuda"):
                        mix_input = mix.unsqueeze(1)
                        estimates = self.model(mix_input)
                else:
                    mix_input = mix.unsqueeze(1)
                    estimates = self.model(mix_input)

                _, sisdr_value = self.loss_fn(estimates, clean)
                # Weight by actual batch size for correct averaging
                batch_size = len(mix)
                total_sisdr += sisdr_value * batch_size
                total_samples += batch_size
                pbar.set_postfix({"SI-SDR": f"{sisdr_value:.2f}dB"})

        avg_sisdr = total_sisdr / total_samples
        return avg_sisdr

    def train(self, num_epochs: int, save_dir: str = "checkpoints"):
        """Main training loop."""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        val_sisdr_history = []

        final_epoch = self.current_epoch + num_epochs
        for epoch in range(self.current_epoch, self.current_epoch + num_epochs):
            self.current_epoch = epoch + 1
            self.logger.info(f"Epoch {self.current_epoch}/{final_epoch}")

            # Update training variants based on curriculum schedule
            self._update_training_variants(self.current_epoch)

            train_sisdr = self.train_epoch()
            self.logger.info(f"Train - SI-SDR: {train_sisdr:.2f} dB")

            val_sisdr = self.validate()
            self.logger.info(f"Val - SI-SDR: {val_sisdr:.2f} dB")

            val_sisdr_history.append(val_sisdr)

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Update learning rate based on validation performance (only if enabled)
            old_lr = current_lr
            if self.lr_scheduler_enabled:
                self.scheduler.step(val_sisdr)
                current_lr = self.optimizer.param_groups[0]["lr"]

            if current_lr != old_lr:
                self.logger.info(
                    f"Learning rate decreased: {old_lr:.2e} -> {current_lr:.2e}"
                )
            else:
                self.logger.info(f"Learning rate: {current_lr:.2e}")

            if self.wandb_logger:
                self.wandb_logger.log_metrics(
                    {
                        "epoch": self.current_epoch,
                        "train_si_sdr": train_sisdr,
                        "val_si_sdr": val_sisdr,
                        "train_lr": current_lr,
                    },
                    step=self.current_epoch,
                )

            if val_sisdr > self.best_val_sisdr:
                self.best_val_sisdr = val_sisdr
                self._save_checkpoint(epoch, val_sisdr, save_dir)

        self.logger.info("\nTraining Summary:")
        self.logger.info(f"Batch size: {self.train_loader.batch_size}")
        self.logger.info(f"Number of epochs: {num_epochs}")
        self.logger.info("\nValidation SI-SDR history:")
        for idx, val_sisdr in enumerate(val_sisdr_history, 1):
            self.logger.info(f"  Epoch {idx}: Val SI-SDR = {val_sisdr:.2f} dB")
