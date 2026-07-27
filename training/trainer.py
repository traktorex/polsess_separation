"""Trainer class for speech enhancement/separation models."""

import time
import torch
import logging
import yaml
from pathlib import Path
from tqdm import tqdm
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
from typing import Optional, Any
from config import Config
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from models import MAMBA_MODELS
from utils import (
    unwrap_compiled_model,
    dataclass_to_dict,
    ensure_dir,
    load_model_from_checkpoint,
    compute_sisdr_and_sisdri,
)

DEFAULT_SISDR_FALLBACK = -999.0  # Default SI-SDR when not found in checkpoint

# Abort the run after this many NaN/Inf training batches in a row. Skipping the
# odd bad batch is fine, but a long unbroken stretch means the model state is
# unrecoverable and every further batch is wasted compute (seen in the
# MossFormer2 128k sweep before the bf16 fix, 2026-06-05).
MAX_CONSECUTIVE_NAN_BATCHES = 1000


class ConsecutiveNaNError(RuntimeError):
    """Training produced MAX_CONSECUTIVE_NAN_BATCHES NaN/Inf losses in a row."""


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
        per_variant_val_loaders: Optional[dict] = None,
        provenance: Optional[dict] = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.per_variant_val_loaders = per_variant_val_loaders
        self.per_variant_mode = per_variant_val_loaders is not None
        # Run provenance manifest (survey gap 5): embedded in every checkpoint and
        # written to run_manifest.yaml beside config.yaml. None → nothing added.
        self.provenance = provenance
        assert (val_loader is None) != (per_variant_val_loaders is None), (
            "Trainer requires exactly one of val_loader or per_variant_val_loaders"
        )
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

        # SI-SDR metric is needed by every task: for ES/EB it's the loss; for SB
        # it's used as the mixture baseline in per-variant SI-SDRi.
        self.si_sdr_metric = ScaleInvariantSignalDistortionRatio().to(device)

        # Task-specific loss function
        self.task = config.data.task
        if self.task == "SB":
            self.pit_loss = PITLossWrapper(
                pairwise_neg_sisdr, pit_from="pw_mtx"  # Pairwise matrix mode
            ).to(device)
            self.loss_fn = self._pit_loss_wrapper

        else:
            self.loss_fn = self._sisdr_loss_wrapper

        self._setup_amp(model, device)

        # In per_variant_mode this tracks the best avg SI-SDRi; otherwise best SI-SDR.
        self.best_val_sisdr = -float("inf")
        # Per-variant SI-SDRi from the previous epoch (for the delta row in the table).
        self.prev_variant_sisdri = None
        self.current_epoch = 0
        # Consecutive NaN/Inf batch counter — persists across epoch boundaries,
        # reset by any finite-loss batch. See MAX_CONSECUTIVE_NAN_BATCHES.
        self.consecutive_nan_batches = 0
        # Early-stopping patience counter. Persisted in checkpoints and restored
        # on resume so patience survives a resume (survey gap 8d).
        self.epochs_without_improvement = 0
        self.run_start_timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

        # Curriculum learning setup
        self.curriculum_schedule = config.training.curriculum_learning
        self.lr_scheduler_enabled = (
            self.curriculum_schedule is None
        )  # Enabled by default if no curriculum

    def _setup_amp(self, model, device):
        """Configure AMP strategy based on model type.

        Mamba models: bf16 autocast, no GradScaler — Mamba CUDA kernels accept
        bf16 activations with float32 weights. GradScaler is unnecessary (bf16
        has the same exponent range as float32) and harmful (grows unbounded
        since the model internally runs float32).

        MossFormer2: bf16 autocast, no GradScaler — its squared-ReLU attention
        overflows fp16's range (max 65504) once activations sharpen. Validation
        already runs fp32 for this reason (see _validate); training hit the same
        overflow at low attn_dropout / higher LR (128k sweep, 2026-06-05). bf16
        has fp32's exponent range, so the squared activations cannot overflow.

        Other models: fp16 autocast + GradScaler — standard mixed precision.
        """
        if not self.use_amp or device != "cuda":
            self.scaler = None
            self.amp_dtype = None
            return

        # Dispatch on model_type rather than class name: torch.compile wraps the
        # model in OptimizedModule, which would silently defeat a class-name check.
        if self.config.model.model_type in MAMBA_MODELS or (
            self.config.model.model_type == "mossformer2"
        ):
            self.scaler = None
            self.amp_dtype = torch.bfloat16
        else:
            self.scaler = torch.amp.GradScaler("cuda", init_scale=256)
            self.amp_dtype = torch.float16

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
            if entry["epoch"] <= epoch and entry.get("lr_scheduler") == "start":
                return True

        return False

    def _update_training_variants(self, epoch):
        """Update training dataset's allowed_variants based on curriculum schedule.

        LANDMINE (survey gap 18): this mutates ``train_loader.dataset`` *in place*.
        With the default ``persistent_workers=False`` the workers are re-forked
        every epoch and pick up the new ``allowed_variants``; if anyone ever sets
        ``persistent_workers=True`` (see the note at the DataLoader site in
        training/setup.py) the workers keep their stage-0 fork and every epoch
        silently trains on the stage-0 variant set. Keep persistent_workers off
        whenever a curriculum is configured.
        """
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

    def _compute_sisdri(self, mix, clean, estimates) -> float:
        """SI-SDRi for the batch, via the shared ``compute_sisdr_and_sisdri``.

        Keeps this in lockstep with evaluate.py (survey gap 12): the SI-SDRi
        mixture-baseline logic lives in one place. Only SI-SDRi is used here —
        the batch SI-SDR the caller reports comes from the loss, which equals
        the helper's SI-SDR in the fp32 validation path (validation runs without
        autocast); under AMP training the two agree to within float noise.
        """
        _, sisdri, _ = compute_sisdr_and_sisdri(
            estimates,
            clean,
            mix,
            self.task,
            self.si_sdr_metric,
            pit_loss=self.pit_loss if self.task == "SB" else None,
        )
        return sisdri

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
        # map_location must be "cpu", not self.device: torch.load(map_location="cuda")
        # puts Adam's per-param `step` counters on the GPU, and load_state_dict
        # leaves them there (it only re-homes `step` for capturable/fused
        # optimizers). CUDA step tensors force a GPU->CPU sync per parameter on
        # every optimizer.step(), which made every resumed run 7-25% slower than
        # a fresh one (measured 2026-07-12). Loading on CPU keeps `step` on CPU;
        # load_state_dict still moves exp_avg/exp_avg_sq to the param device.
        checkpoint = load_model_from_checkpoint(checkpoint_path, self.model, "cpu")

        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Allow overriding LR via config even when resuming, but only if the
        # user explicitly provided a config (not when it was loaded from the
        # checkpoint itself — in that case config.training.lr is the original
        # starting LR and would clobber any scheduler-reduced LR).
        current_lr = self.optimizer.param_groups[0]['lr']
        config_from_ckpt = getattr(self.config, '_loaded_from_checkpoint', False)
        if current_lr != self.config.training.lr and not config_from_ckpt:
             self.logger.info(f"Overriding checkpoint LR {current_lr:.2e} with config LR {self.config.training.lr:.2e}")
             for param_group in self.optimizer.param_groups:
                 param_group['lr'] = self.config.training.lr
        else:
             self.logger.info(f"Resuming from checkpoint LR {current_lr:.2e}")

        # Load best_val_sisdr BEFORE the scheduler block (survey gap 7): the
        # legacy-checkpoint branch below seeds scheduler.best from it, so it must
        # already hold the checkpoint's real best — not the -inf placeholder set
        # in __init__ — or the first post-resume epoch always looks like an
        # improvement (the exact bug the legacy branch's comment claims to fix).
        self.current_epoch = checkpoint["epoch"] + 1  # Resume from next epoch
        self.best_val_sisdr = checkpoint.get(
            "best_val_sisdr", checkpoint.get("val_sisdr", DEFAULT_SISDR_FALLBACK)
        )

        # Load scheduler state if available
        if self.scheduler is not None:
             if "scheduler_state_dict" in checkpoint:
                self.logger.info("Loading scheduler state from checkpoint")
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
             else:
                # Legacy checkpoint: manually set 'best' so it doesn't start from scratch
                # This prevents it from thinking the first epoch after resume is the new best
                self.logger.info("Legacy checkpoint detected: Manually initializing scheduler 'best'")
                self.scheduler.best = self.best_val_sisdr

        # Restore GradScaler state for fp16 models (survey gap 8a). Tolerate its
        # absence in old checkpoints — those resume at the default init_scale.
        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.logger.info("Loading GradScaler state from checkpoint")
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        # Restore early-stopping patience (survey gap 8d); missing in old
        # checkpoints → default 0 (as if resuming right after an improvement).
        self.epochs_without_improvement = checkpoint.get("epochs_without_improvement", 0)

        self.logger.info(
            f"Loaded checkpoint from epoch {checkpoint['epoch']} (best SI-SDR: {self.best_val_sisdr:.2f} dB), "
            f"resuming at epoch {self.current_epoch}"
        )

    def _get_run_name(self) -> str:
        """Determine run name from wandb or config."""
        # Use wandb run name if available
        if (
            self.wandb_logger
            and self.wandb_logger.enabled
            and self.wandb_logger.run
            and self.wandb_logger.run.name
        ):
            return self.wandb_logger.run.name

        # Fall back to experiment_name from config
        experiment_name = getattr(self.config.training, 'experiment_name', None)
        if not experiment_name and self.config.training.wandb_run_name:
            experiment_name = self.config.training.wandb_run_name

        if experiment_name:
            return experiment_name

        # Default: use timestamp from training start
        return f"run_{self.run_start_timestamp}"

    def _create_checkpoint_paths(self, save_dir: str, run_name: str, epoch: int) -> tuple[Path, Path]:
        """Create checkpoint directory structure and return file paths.

        Args:
            save_dir: Base directory for checkpoints.
            run_name: Name for this training run.
            epoch: Current epoch number.

        Returns:
            Tuple of (checkpoint_path, config_path).
        """
        base_dir = Path(save_dir)
        model_name = self.config.model.model_type
        task = self.config.data.task

        # Structure: checkpoints/{model_name}/{task}/{run_name}/
        # run_name already includes timestamp from _get_run_name() 
        checkpoint_dir = base_dir / model_name / task / run_name
        ensure_dir(checkpoint_dir)

        # Checkpoint file logic
        if self.config.training.save_all_checkpoints:
             # Keep distinct filenames for every improvement
             checkpoint_filename = f"{model_name}_{task}_epoch{epoch+1}.pt"
        else:
             # Overwrite single best file
             checkpoint_filename = f"{model_name}_{task}_best.pt"

        checkpoint_path = checkpoint_dir / checkpoint_filename
        config_path = checkpoint_dir / "config.yaml"

        return checkpoint_path, config_path

    def _serialize_checkpoint_data(self, epoch: int, val_sisdr: float) -> dict:
        """Prepare checkpoint data for saving.

        Args:
            epoch: Current training epoch.
            val_sisdr: Validation SI-SDR score.

        Returns:
            Dictionary containing all checkpoint data.
        """
        config_dict = {
            "model": dataclass_to_dict(self.config.model),
            "data": dataclass_to_dict(self.config.data),
            "training": dataclass_to_dict(self.config.training),
        }

        # Get state dict from unwrapped model if using torch.compile
        model_to_save = unwrap_compiled_model(self.model)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_sisdr": val_sisdr,
            # Persist best + patience explicitly (survey gaps 7/8d). load_checkpoint
            # prefers this key; old checkpoints (lacking it) fall back cleanly.
            # max() keeps the invariant local: the train loop saves only on
            # improvement (best == val_sisdr there), but a direct _save_checkpoint
            # call must still record this checkpoint's own score as the best.
            "best_val_sisdr": max(self.best_val_sisdr, val_sisdr),
            "epochs_without_improvement": self.epochs_without_improvement,
            "config": config_dict,
        }

        # Save scheduler state if available
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        # Save GradScaler state so fp16 models resume at the right loss scale
        # rather than the default init_scale (survey gap 8a).
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        # Run provenance (survey gap 5) — git/env/GPU that produced this run.
        if self.provenance is not None:
            checkpoint["provenance"] = self.provenance

        # W&B run id so a later --resume can reconnect to the same run instead of
        # orphaning a fresh one (survey gap 14).
        wandb_run_id = self._wandb_run_id()
        if wandb_run_id is not None:
            checkpoint["wandb_run_id"] = wandb_run_id

        return checkpoint

    def _wandb_run_id(self) -> Optional[str]:
        """Return the active W&B run id, or None when W&B is disabled/unavailable."""
        wl = self.wandb_logger
        if wl and getattr(wl, "enabled", False) and getattr(wl, "run", None) is not None:
            return getattr(wl.run, "id", None)
        return None

    def _save_checkpoint(self, epoch: int, val_sisdr: float, save_dir: str):
        """Save model checkpoint with best validation SI-SDR."""
        # Determine run name and create directory structure
        run_name = self._get_run_name()
        checkpoint_path, config_path = self._create_checkpoint_paths(save_dir, run_name, epoch)

        # Prepare checkpoint data
        checkpoint_data = self._serialize_checkpoint_data(epoch, val_sisdr)

        # Save checkpoint file
        torch.save(checkpoint_data, checkpoint_path)

        # Save config as YAML for easy viewing
        with open(config_path, 'w') as f:
            yaml.dump(checkpoint_data["config"], f, default_flow_style=False, sort_keys=False)

        # Write a human-readable run manifest next to config.yaml (survey gap 5).
        if self.provenance is not None:
            manifest_path = config_path.parent / "run_manifest.yaml"
            with open(manifest_path, "w") as f:
                yaml.dump(self.provenance, f, default_flow_style=False, sort_keys=False)

        # Log checkpoint save
        metric_name = "avg SI-SDRi" if self.per_variant_mode else "SI-SDR"
        self.logger.info(
            f"Saved best model ({metric_name}: {val_sisdr:.2f} dB) to: {checkpoint_path}"
        )
        self.logger.info(f"  Run name: {run_name}")

        # Log to wandb if enabled
        if self.wandb_logger:
            self.wandb_logger.log_metrics(
                {"best_val_sisdr": val_sisdr}, step=self.current_epoch
            )
            self.wandb_logger.log_model(str(checkpoint_path))

    def train_epoch(self) -> tuple:
        """Train for one epoch; return (avg SI-SDR, avg SI-SDRi)."""
        self.model.train()
        total_sisdr = 0
        total_sisdri = 0
        total_samples = 0
        
        # Gradient accumulation setup
        accum_steps = getattr(self.config.training, 'grad_accumulation_steps', 1)
        num_batches = len(self.train_loader)

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

            if self.use_amp:
                with torch.amp.autocast("cuda", dtype=self.amp_dtype):
                    mix_input = mix.unsqueeze(1)
                    estimates = self.model(mix_input)
                    loss, sisdr_value = self.loss_fn(estimates, clean)
            else:
                mix_input = mix.unsqueeze(1)
                estimates = self.model(mix_input)
                loss, sisdr_value = self.loss_fn(estimates, clean)

            # Scale loss for gradient accumulation. Divide by the actual number
            # of micro-batches in THIS window, not the nominal accum_steps, so the
            # final short "tail" window isn't under-weighted (survey gap 8c). For
            # full windows window_size == accum_steps; for accum_steps == 1 it is
            # always 1, so default runs are unaffected. (Audit 2026-07-12: no
            # shipped experiment/sweep config used accum_steps>1 — latent only.)
            if accum_steps > 1:
                window_start = (batch_idx // accum_steps) * accum_steps
                window_size = min(accum_steps, num_batches - window_start)
                loss = loss / window_size

            if torch.isnan(loss) or torch.isinf(loss):
                self.consecutive_nan_batches += 1
                self.logger.warning(
                    f"NaN/Inf detected at batch {batch_idx}, skipping batch"
                )
                self.logger.warning(f"  Loss: {loss.item()}, SI-SDR: {sisdr_value}")
                # Do NOT zero_grad here (survey gap 8b): the NaN is caught BEFORE
                # backward(), so this batch contributed no gradient, while the
                # current accumulation window may already hold valid gradients
                # from earlier micro-batches. Zeroing would silently discard them.
                # The window still steps normally at its boundary; any grads left
                # un-stepped by a NaN on the final window are cleared at epoch end
                # (see below) so nothing leaks into the next epoch.
                # Free the computation graph and intermediate tensors before
                # calling empty_cache — otherwise they keep GPU memory pinned
                del loss, estimates, mix_input, mix, clean
                torch.cuda.empty_cache()
                if self.consecutive_nan_batches >= MAX_CONSECUTIVE_NAN_BATCHES:
                    raise ConsecutiveNaNError(
                        f"{self.consecutive_nan_batches} consecutive NaN/Inf batches "
                        f"(epoch {self.current_epoch}, last batch {batch_idx}) — "
                        "model state is unrecoverable, aborting run"
                    )
                continue
            self.consecutive_nan_batches = 0

            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Only step optimizer every accum_steps batches (or at end of epoch)
            is_accum_step = (batch_idx + 1) % accum_steps == 0
            is_last_batch = (batch_idx + 1) == len(self.train_loader)

            if is_accum_step or is_last_batch:
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
            sisdri_value = self._compute_sisdri(mix, clean, estimates)
            total_sisdr += sisdr_value * batch_size
            total_sisdri += sisdri_value * batch_size
            total_samples += batch_size

            pbar.set_postfix(
                {
                    "SI-SDR": f"{sisdr_value:.2f}",
                    "SI-SDRi": f"{sisdri_value:.2f}",
                    "LR": f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                }
            )

        # Clear any gradients left un-stepped by a NaN-skipped final window so
        # they cannot leak into the next epoch (only reachable with
        # grad_accumulation_steps>1 and a NaN on the window's boundary batch; see
        # the NaN-skip branch above). No-op in the common case — the last good
        # batch already stepped and zeroed.
        self.optimizer.zero_grad()

        # A fully-NaN epoch skips every batch, leaving total_samples == 0 — the
        # pre-2026-06 Mamba collapse runs died here with a bare ZeroDivisionError.
        # Reachable when an epoch has fewer batches than MAX_CONSECUTIVE_NAN_BATCHES.
        if total_samples == 0:
            raise ConsecutiveNaNError(
                f"epoch {self.current_epoch}: every training batch was NaN/Inf "
                "and skipped — model state is unrecoverable, aborting run"
            )

        avg_sisdr = total_sisdr / total_samples
        avg_sisdri = total_sisdri / total_samples
        return avg_sisdr, avg_sisdri

    def validate(self) -> tuple:
        """Run validation; return (avg SI-SDR, avg SI-SDRi)."""
        self.model.eval()
        total_sisdr = 0
        total_sisdri = 0
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

                # Validation always runs in fp32 (no autocast), matching evaluate.py.
                # Under fp16 autocast, MossFormer2's squared-ReLU attention can
                # overflow in eval mode once the model sharpens, producing NaN val
                # SI-SDR; the fp32 forward avoids it. Training precision is unchanged.
                mix_input = mix.unsqueeze(1)
                estimates = self.model(mix_input)
                _, sisdr_value = self.loss_fn(estimates, clean)
                # Weight by actual batch size for correct averaging
                batch_size = len(mix)
                sisdri_value = self._compute_sisdri(mix, clean, estimates)
                total_sisdr += sisdr_value * batch_size
                total_sisdri += sisdri_value * batch_size
                total_samples += batch_size
                pbar.set_postfix({
                    "SI-SDR": f"{sisdr_value:.2f}",
                    "SI-SDRi": f"{sisdri_value:.2f}",
                })

        avg_sisdr = total_sisdr / total_samples
        avg_sisdri = total_sisdri / total_samples
        return avg_sisdr, avg_sisdri

    def validate_per_variant(self) -> dict:
        """Run validation once per MM-IPC variant.

        Returns {variant: {"si_sdr": float, "si_sdri": float}}. The per-variant
        loaders each force a specific MM-IPC variant via allowed_variants=[v],
        so every val sample is re-rendered once per variant.
        """
        self.model.eval()
        results = {}

        with torch.no_grad():
            for variant, loader in self.per_variant_val_loaders.items():
                total_sisdr = 0.0
                total_sisdri = 0.0
                total_samples = 0

                pbar = tqdm(
                    loader,
                    desc=f"Val [{variant}]",
                    unit="batch",
                    leave=False,
                    ncols=100,
                )

                for batch in pbar:
                    mix = batch["mix"].to(self.device)
                    clean = batch["clean"].to(self.device)
                    batch_size = len(mix)

                    # fp32 forward (see validate()): avoids fp16 overflow in eval.
                    estimates = self.model(mix.unsqueeze(1))

                    # Trim everyone to common length (matches evaluate.py).
                    min_len = min(estimates.shape[-1], clean.shape[-1])
                    estimates = estimates[..., :min_len]
                    clean_t = clean[..., :min_len]
                    mix_trimmed = mix[..., :min_len]

                    if self.task == "SB":
                        loss = self.pit_loss(estimates, clean_t)
                        si_sdr = -loss.item()
                        mix_baseline = 0.0
                        for spk in range(clean_t.shape[1]):
                            mix_baseline += self.si_sdr_metric(
                                mix_trimmed, clean_t[:, spk]
                            ).item()
                        mix_baseline /= clean_t.shape[1]
                        si_sdri = si_sdr - mix_baseline
                    else:
                        if clean_t.dim() == 3 and clean_t.shape[1] == 1:
                            clean_t = clean_t.squeeze(1)
                        if estimates.dim() == 3 and estimates.shape[1] == 1:
                            estimates = estimates.squeeze(1)
                        si_sdr = self.si_sdr_metric(estimates, clean_t).item()
                        si_sdr_mix = self.si_sdr_metric(mix_trimmed, clean_t).item()
                        si_sdri = si_sdr - si_sdr_mix

                    total_sisdr += si_sdr * batch_size
                    total_sisdri += si_sdri * batch_size
                    total_samples += batch_size
                    pbar.set_postfix({
                        "SI-SDR": f"{si_sdr:.2f}",
                        "SI-SDRi": f"{si_sdri:.2f}",
                    })

                results[variant] = {
                    "si_sdr": total_sisdr / total_samples,
                    "si_sdri": total_sisdri / total_samples,
                }

        return results

    def _log_variant_table(self, variant_results: dict, avg_sisdri: float):
        """Pretty-print the per-variant validation table to the logger.

        Third row shows the SI-SDRi delta vs. the previous epoch (— on epoch 1
        or after resume).
        """
        variants = list(variant_results.keys())
        col_w = 7
        header = "           " + "".join(f"{v:>{col_w}}" for v in variants)
        sisdr_row = "SI-SDR    " + "".join(
            f"{variant_results[v]['si_sdr']:>{col_w}.2f}" for v in variants
        )
        sisdri_row = "SI-SDRi   " + "".join(
            f"{variant_results[v]['si_sdri']:>{col_w}.2f}" for v in variants
        )
        if self.prev_variant_sisdri is None:
            delta_cells = [f"{'—':>{col_w}}" for _ in variants]
        else:
            delta_cells = []
            for v in variants:
                prev = self.prev_variant_sisdri.get(v)
                if prev is None:
                    delta_cells.append(f"{'—':>{col_w}}")
                else:
                    d = variant_results[v]["si_sdri"] - prev
                    # Explicit sign so + and - line up visually.
                    delta_cells.append(f"{d:>+{col_w}.2f}")
        delta_row = "Δ SI-SDRi " + "".join(delta_cells)
        self.logger.info("Validation:")
        self.logger.info(header)
        self.logger.info(sisdr_row)
        self.logger.info(sisdri_row)
        self.logger.info(delta_row)
        self.logger.info(f"Avg Val SI-SDRi: {avg_sisdri:.2f} dB")

    def train(self, num_epochs: int, save_dir: str = "checkpoints", early_stopping_patience: int = None):
        """Main training loop with OOM error handling."""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        val_sisdr_history = []

        # Early-stopping patience lives on self.epochs_without_improvement so it
        # survives a resume (survey gap 8d) — load_checkpoint restores it, __init__
        # seeds it to 0. All uses below are gated on early_stopping_patience.

        try:
            final_epoch = self.current_epoch + num_epochs
            for epoch in range(self.current_epoch, self.current_epoch + num_epochs):
                self.current_epoch = epoch + 1
                self.logger.info(f"Epoch {self.current_epoch}/{final_epoch}")

                # Update training variants based on curriculum schedule
                self._update_training_variants(self.current_epoch)

                train_sisdr, train_sisdri = self.train_epoch()
                self.logger.info(f"Train - SI-SDR:  {train_sisdr:.2f} dB")
                self.logger.info(f"Train - SI-SDRi: {train_sisdri:.2f} dB")

                if self.per_variant_mode:
                    variant_results = self.validate_per_variant()
                    avg_si_sdr = sum(v["si_sdr"] for v in variant_results.values()) / len(variant_results)
                    avg_si_sdri = sum(v["si_sdri"] for v in variant_results.values()) / len(variant_results)
                    self._log_variant_table(variant_results, avg_si_sdri)
                    self.prev_variant_sisdri = {v: vr["si_sdri"] for v, vr in variant_results.items()}
                    monitor_value = avg_si_sdri
                else:
                    val_sisdr, val_sisdri = self.validate()
                    self.logger.info(f"Val - SI-SDR:  {val_sisdr:.2f} dB")
                    self.logger.info(f"Val - SI-SDRi: {val_sisdri:.2f} dB")
                    monitor_value = val_sisdr
                    variant_results = None

                val_sisdr_history.append(monitor_value)

                # Get current learning rate
                current_lr = self.optimizer.param_groups[0]["lr"]

                # Update learning rate based on validation performance (only if enabled)
                old_lr = current_lr
                if self.lr_scheduler_enabled:
                    self.scheduler.step(monitor_value)
                    current_lr = self.optimizer.param_groups[0]["lr"]

                if current_lr != old_lr:
                    self.logger.info(
                        f"Learning rate decreased: {old_lr:.2e} -> {current_lr:.2e}"
                    )
                else:
                    self.logger.info(f"Learning rate: {current_lr:.2e}")

                # Prepare metrics
                metrics = {
                    "epoch": self.current_epoch,
                    "train_si_sdr": train_sisdr,
                    "train_si_sdri": train_sisdri,
                    "train_lr": current_lr,
                }
                if self.per_variant_mode:
                    metrics["val_si_sdr"] = avg_si_sdr
                    metrics["val_si_sdri"] = avg_si_sdri
                    for v, vr in variant_results.items():
                        metrics[f"val_si_sdr_{v}"] = vr["si_sdr"]
                        metrics[f"val_si_sdri_{v}"] = vr["si_sdri"]
                else:
                    metrics["val_si_sdr"] = val_sisdr
                    metrics["val_si_sdri"] = val_sisdri

                # Add early stopping metric if enabled
                if early_stopping_patience:
                    metrics["epochs_no_improvement"] = self.epochs_without_improvement

                if self.wandb_logger:
                    self.wandb_logger.log_metrics(metrics, step=self.current_epoch)

                # Check for improvement
                if monitor_value > self.best_val_sisdr:
                    self.best_val_sisdr = monitor_value
                    if early_stopping_patience:
                        self.epochs_without_improvement = 0  # Reset counter
                    self._save_checkpoint(epoch, monitor_value, save_dir)
                else:
                    if early_stopping_patience:
                        self.epochs_without_improvement += 1
                        self.logger.info(f"No improvement for {self.epochs_without_improvement} epoch(s)")

                # Early stopping check (only if enabled)
                if early_stopping_patience and self.epochs_without_improvement >= early_stopping_patience:
                    self.logger.warning("=" * 80)
                    self.logger.warning("EARLY STOPPING")
                    self.logger.warning("=" * 80)
                    self.logger.warning(f"No improvement for {early_stopping_patience} epochs")
                    metric_name = "avg SI-SDRi" if self.per_variant_mode else "SI-SDR"
                    self.logger.warning(f"Best Val {metric_name}: {self.best_val_sisdr:.2f} dB")
                    self.logger.warning(f"Stopping at epoch {self.current_epoch}/{final_epoch}")
                    
                    if self.wandb_logger:
                        self.wandb_logger.log_metrics({
                            "early_stopped": 1,
                            "early_stop_epoch": self.current_epoch,
                        })
                    
                    break  # Exit training loop

            self.logger.info("\nTraining Summary:")
            self.logger.info(f"Batch size: {self.train_loader.batch_size}")
            self.logger.info(f"Number of epochs: {num_epochs}")
            metric_name = "avg SI-SDRi" if self.per_variant_mode else "SI-SDR"
            self.logger.info(f"\nValidation {metric_name} history:")
            start_epoch = final_epoch - num_epochs + 1
            for idx, val_sisdr in enumerate(val_sisdr_history, start_epoch):
                self.logger.info(f"  Epoch {idx}: Val {metric_name} = {val_sisdr:.2f} dB")

        except ConsecutiveNaNError as e:
            self.logger.error("=" * 80)
            self.logger.error("TRAINING ABORTED — PERSISTENT NaN/Inf LOSS")
            self.logger.error("=" * 80)
            self.logger.error(str(e))
            self.logger.error(f"Model: {self.config.model.model_type}")
            self.logger.error(f"LR: {self.config.training.lr}")

            # Log to W&B if available
            if self.wandb_logger and self.wandb_logger.enabled:
                import wandb
                self.wandb_logger.log_metrics({
                    "nan_abort": 1,
                    "nan_abort_epoch": self.current_epoch,
                })
                self.logger.info("Logged NaN abort to W&B")
                wandb.finish(exit_code=1)

            self.logger.error("Terminating run gracefully to continue sweep...")
            self.logger.error("=" * 80)

            # Exit gracefully (sweep will continue with next run)
            raise SystemExit(1)

        except torch.cuda.OutOfMemoryError as e:
            self.logger.error("=" * 80)
            self.logger.error("CUDA OUT OF MEMORY ERROR")
            self.logger.error("=" * 80)
            self.logger.error(f"Error details: {str(e)}")
            self.logger.error(f"Failed at epoch: {self.current_epoch}")
            self.logger.error(f"Model: {self.config.model.model_type}")
            self.logger.error(f"Batch size: {self.config.data.batch_size}")
            
            # Log to W&B if available
            if self.wandb_logger and self.wandb_logger.enabled:
                import wandb
                self.wandb_logger.log_metrics({
                    "oom_error": 1,
                    "oom_epoch": self.current_epoch,
                })
                self.logger.info("Logged OOM error to W&B")
                wandb.finish(exit_code=1)
            
            self.logger.error("Terminating run gracefully to continue sweep...")
            self.logger.error("=" * 80)
            
            # Clean up GPU memory
            import gc
            torch.cuda.empty_cache()
            gc.collect()
            
            # Exit gracefully (sweep will continue with next run)
            raise SystemExit(1)

