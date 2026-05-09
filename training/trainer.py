"""Trainer class for speech enhancement/separation models."""

import time
import torch
import logging
import yaml
from pathlib import Path
from tqdm import tqdm
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio
from typing import Optional, Any, List, Dict
from config import Config
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from models import MAMBA_MODELS
from utils import (
    unwrap_compiled_model,
    dataclass_to_dict,
    ensure_dir,
    load_model_from_checkpoint,
)

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
        # Per-variant loss weighting + per-variant SI-SDRi validation. When
        # loss_weights is None we keep the original code paths bit-identical.
        # getattr default keeps tests with SimpleNamespace configs working.
        self.loss_weights: Optional[Dict[str, float]] = getattr(
            config.training, "loss_weights", None
        )
        self._weighted_loss_active = self.loss_weights is not None

        if self.task == "SB":
            self.pit_loss = PITLossWrapper(
                pairwise_neg_sisdr, pit_from="pw_mtx"  # Pairwise matrix mode
            ).to(device)
            if self._weighted_loss_active:
                self.loss_fn = self._weighted_pit_loss_wrapper
            else:
                self.loss_fn = self._pit_loss_wrapper
        else:
            self.si_sdr_metric = ScaleInvariantSignalDistortionRatio().to(device)
            if self._weighted_loss_active:
                self.loss_fn = self._weighted_sisdr_loss_wrapper
            else:
                self.loss_fn = self._sisdr_loss_wrapper

        self._setup_amp(model, device)

        self.best_val_sisdr = -float("inf")
        self.current_epoch = 0
        self.run_start_timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        # Side-channel for the train loop to read after validate(): per-variant
        # SI-SDR / SI-SDRi breakdowns and the weighted-sum scalars.
        self._last_val_breakdown: Dict[str, Any] = {}

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

        Other models: fp16 autocast + GradScaler — standard mixed precision.
        """
        if not self.use_amp or device != "cuda":
            self.scaler = None
            self.amp_dtype = None
            return

        # Dispatch on model_type rather than class name: torch.compile wraps the
        # model in OptimizedModule, which would silently defeat a class-name check.
        if self.config.model.model_type in MAMBA_MODELS:
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

    def _sisdr_loss_wrapper(self, estimates, targets, weights=None):
        """Compute SI-SDR loss for ES/EB tasks (unweighted, batch-mean).

        ``weights`` is accepted but ignored — exists so the call site in
        train_epoch can pass weights uniformly without branching.
        """
        sisdr = self.si_sdr_metric(estimates, targets)
        return -sisdr, sisdr.item()

    def _pit_loss_wrapper(self, estimates, targets, weights=None):
        """Compute PIT-SI-SDR loss for SB task (unweighted, batch-mean).

        ``weights`` is accepted but ignored.
        """
        loss = self.pit_loss(estimates, targets)
        # Loss is negative SI-SDR averaged over batch
        sisdr = -loss.item()
        return loss, sisdr

    def _weighted_sisdr_loss_wrapper(self, estimates, targets, weights):
        """Per-sample weighted SI-SDR loss for ES/EB tasks.

        Reduces as ``(weights * neg_sisdr).sum() / weights.sum()`` so the
        scale of weights doesn't change loss magnitude — they only shift
        relative contribution between samples. With uniform weights this
        equals the unweighted batch-mean loss exactly.
        """
        sisdr = scale_invariant_signal_distortion_ratio(estimates, targets)  # [B]
        neg_sisdr = -sisdr
        w_sum = weights.sum()
        loss = (weights * neg_sisdr).sum() / w_sum
        sisdr_value = ((weights * sisdr).sum() / w_sum).item()
        return loss, sisdr_value

    def _weighted_pit_loss_wrapper(self, estimates, targets, weights):
        """Per-sample weighted PIT-SI-SDR loss for SB task.

        Uses ``PITLossWrapper.find_best_perm`` to get the matched per-sample
        loss (avg over matched sources) before reduction. Weighted-mean
        normalization: with uniform weights this matches the unweighted
        ``PITLossWrapper.forward`` mean exactly.
        """
        pw_losses = pairwise_neg_sisdr(estimates, targets)  # [B, n_src, n_src]
        min_loss, _ = PITLossWrapper.find_best_perm(pw_losses)  # [B] (per-sample neg SI-SDR avg)
        w_sum = weights.sum()
        loss = (weights * min_loss).sum() / w_sum
        # Weighted mean SI-SDR for logging (note: -min_loss is per-sample SI-SDR)
        sisdr_value = (-(weights * min_loss).sum() / w_sum).item()
        return loss, sisdr_value

    def _resolve_weight_key(self, variant: str, has_reverb: bool) -> str:
        """Map (variant, has_reverb) to the loss_weights dict key.

        C is the only variant with separate indoor/outdoor weights —
        all other variant strings map to themselves.
        """
        if variant == "C":
            return "C_in" if has_reverb else "C_out"
        return variant

    def _build_sample_weights(
        self, variants: List[str], has_reverb: torch.Tensor
    ) -> torch.Tensor:
        """Build per-sample weight tensor [B] from variant tags and has_reverb.

        Raises KeyError if a variant maps to a key absent from
        ``self.loss_weights`` — config validation ensures all 9 keys are
        present, so this is defense in depth.
        """
        has_reverb_list = has_reverb.tolist()
        weights = [
            self.loss_weights[self._resolve_weight_key(v, hr)]
            for v, hr in zip(variants, has_reverb_list)
        ]
        return torch.tensor(weights, device=self.device, dtype=torch.float32)

    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint to resume training.

        Handles both compiled and non-compiled models by unwrapping
        torch.compile() wrapper if needed.
        """
        checkpoint = load_model_from_checkpoint(checkpoint_path, self.model, self.device)

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
            
        self.current_epoch = checkpoint["epoch"] + 1  # Resume from next epoch
        self.best_val_sisdr = checkpoint.get(
            "best_val_sisdr", checkpoint.get("val_sisdr", DEFAULT_SISDR_FALLBACK)
        )

        # Guard against silently comparing incomparable metrics when the
        # checkpoint was saved under a different val regime (raw SI-SDR vs
        # weighted SI-SDRi). When the regime changes, reset best so the
        # first epoch under the new regime is correctly accepted as best.
        ckpt_loss_weights = checkpoint.get("config", {}).get("training", {}).get("loss_weights")
        ckpt_was_weighted = ckpt_loss_weights is not None
        if ckpt_was_weighted != self._weighted_loss_active:
            self.logger.warning(
                f"Checkpoint val regime mismatch: checkpoint weighted={ckpt_was_weighted}, "
                f"current weighted={self._weighted_loss_active}. "
                f"Resetting best_val_sisdr (-inf) so the first epoch under the new "
                f"regime is accepted as best."
            )
            self.best_val_sisdr = -float("inf")

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
            "config": config_dict,
        }
        
        # Save scheduler state if available
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
            
        return checkpoint

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

        # Log checkpoint save
        self.logger.info(
            f"Saved best model (SI-SDR: {val_sisdr:.2f} dB) to: {checkpoint_path}"
        )
        self.logger.info(f"  Run name: {run_name}")

        # Log to wandb if enabled
        if self.wandb_logger:
            best_metrics = {"best_val_sisdr": val_sisdr}
            # When the weighted-loss path is active, val_sisdr is the
            # weighted-sum SI-SDRi — log under that name too so sweeps can
            # target a clearly-named metric.
            if self._weighted_loss_active:
                best_metrics["best_val_sisdri_weighted"] = val_sisdr
            self.wandb_logger.log_metrics(best_metrics, step=self.current_epoch)
            self.wandb_logger.log_model(str(checkpoint_path))

    def train_epoch(self) -> float:
        """Train for one epoch and return average SI-SDR."""
        self.model.train()
        total_sisdr = 0
        total_samples = 0
        
        # Gradient accumulation setup
        accum_steps = getattr(self.config.training, 'grad_accumulation_steps', 1)

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

            # Per-sample weights (only when active — preserves SyntheticDataset
            # compat in tests, which doesn't return has_reverb).
            if self._weighted_loss_active:
                weights = self._build_sample_weights(
                    batch["background_complexity"], batch["has_reverb"]
                )
            else:
                weights = None

            if self.use_amp:
                with torch.amp.autocast("cuda", dtype=self.amp_dtype):
                    mix_input = mix.unsqueeze(1)
                    estimates = self.model(mix_input)
                    loss, sisdr_value = self.loss_fn(estimates, clean, weights)
            else:
                mix_input = mix.unsqueeze(1)
                estimates = self.model(mix_input)
                loss, sisdr_value = self.loss_fn(estimates, clean, weights)

            # Scale loss for gradient accumulation
            if accum_steps > 1:
                loss = loss / accum_steps

            if torch.isnan(loss) or torch.isinf(loss):
                self.logger.warning(
                    f"NaN/Inf detected at batch {batch_idx}, skipping batch"
                )
                self.logger.warning(f"  Loss: {loss.item()}, SI-SDR: {sisdr_value}")
                self.optimizer.zero_grad()
                # Free the computation graph and intermediate tensors before
                # calling empty_cache — otherwise they keep GPU memory pinned
                del loss, estimates, mix_input, mix, clean
                torch.cuda.empty_cache()
                continue

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
        """Run validation and return the scheduler signal (single float).

        Two regimes:

        - Legacy (``loss_weights is None``): unchanged batch-mean SI-SDR.
        - Weighted (``loss_weights is not None``): computes per-sample
          SI-SDRi (= SI-SDR(estimate, target) − SI-SDR(mix, target)),
          groups by variant key (with C disambiguated to C_in/C_out via
          ``has_reverb``), and returns the weighted-sum SI-SDRi using the
          training-time per-variant weights. Per-variant breakdowns
          (raw SI-SDR and SI-SDRi) are stashed on
          ``self._last_val_breakdown`` for the train loop to log.
        """
        if self._weighted_loss_active:
            return self._validate_weighted()
        return self._validate_legacy()

    def _validate_legacy(self) -> float:
        """Original validation path — kept bit-identical for back-compat."""
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

                if self.use_amp:
                    with torch.amp.autocast("cuda", dtype=self.amp_dtype):
                        mix_input = mix.unsqueeze(1)
                        estimates = self.model(mix_input)
                        _, sisdr_value = self.loss_fn(estimates, clean)
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

    def _per_sample_sisdr(self, estimates: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Per-sample SI-SDR. For SB task, applies PIT then averages over sources.

        Returns shape [B]. Used by validation for both estimate-vs-target and
        mix-vs-target SI-SDR (for SI-SDRi).
        """
        if self.task == "SB":
            # estimates and targets are [B, n_src, T]. Note: when this is called
            # for mix-vs-target, ``estimates`` is the broadcast mix expanded to
            # [B, n_src, T] — PIT is a no-op (every permutation gives the same
            # value because all source slots hold the same waveform), so it's
            # safe to run unconditionally.
            pw = pairwise_neg_sisdr(estimates, targets)  # [B, n_src, n_src]
            min_loss, batch_indices = PITLossWrapper.find_best_perm(pw)
            # min_loss is per-sample mean neg-SI-SDR over matched sources.
            return -min_loss
        # ES/EB: estimates and targets are [B, T]
        return scale_invariant_signal_distortion_ratio(estimates, targets)

    def _validate_weighted(self) -> float:
        """Per-variant SI-SDRi validation. See ``validate`` for semantics."""
        self.model.eval()
        # Per-variant accumulators (key → sum, count)
        sisdri_sum: Dict[str, float] = {}
        sisdri_cnt: Dict[str, int] = {}
        sisdr_sum: Dict[str, float] = {}

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
                variants = batch["background_complexity"]
                has_reverb = batch["has_reverb"].tolist()

                if self.use_amp:
                    with torch.amp.autocast("cuda", dtype=self.amp_dtype):
                        mix_input = mix.unsqueeze(1)
                        estimates = self.model(mix_input)
                        sisdr_est = self._per_sample_sisdr(estimates, clean)  # [B]
                        if self.task == "SB":
                            mix_broadcast = mix.unsqueeze(1).expand_as(clean)
                            sisdr_mix = self._per_sample_sisdr(mix_broadcast, clean)
                        else:
                            sisdr_mix = self._per_sample_sisdr(mix, clean)
                else:
                    mix_input = mix.unsqueeze(1)
                    estimates = self.model(mix_input)
                    sisdr_est = self._per_sample_sisdr(estimates, clean)
                    if self.task == "SB":
                        mix_broadcast = mix.unsqueeze(1).expand_as(clean)
                        sisdr_mix = self._per_sample_sisdr(mix_broadcast, clean)
                    else:
                        sisdr_mix = self._per_sample_sisdr(mix, clean)

                sisdri = sisdr_est - sisdr_mix  # [B]
                # Promote to fp32 for stable per-variant accumulation under AMP.
                sisdri_list = sisdri.float().cpu().tolist()
                sisdr_list = sisdr_est.float().cpu().tolist()

                for v, hr, si, sd in zip(variants, has_reverb, sisdri_list, sisdr_list):
                    key = self._resolve_weight_key(v, hr)
                    sisdri_sum[key] = sisdri_sum.get(key, 0.0) + si
                    sisdri_cnt[key] = sisdri_cnt.get(key, 0) + 1
                    sisdr_sum[key] = sisdr_sum.get(key, 0.0) + sd

                pbar.set_postfix({"SI-SDRi": f"{sum(sisdri_list)/len(sisdri_list):.2f}dB"})

        # Per-variant means
        per_variant_sisdri = {k: sisdri_sum[k] / sisdri_cnt[k] for k in sisdri_cnt}
        per_variant_sisdr = {k: sisdr_sum[k] / sisdri_cnt[k] for k in sisdri_cnt}

        # Weighted-sum scalars (normalized by sum of weights of *present* keys
        # so that curriculum-gated runs don't compare against absent variants).
        present_keys = list(per_variant_sisdri.keys())
        w_total = sum(self.loss_weights[k] for k in present_keys)
        val_sisdri_weighted = sum(
            self.loss_weights[k] * per_variant_sisdri[k] for k in present_keys
        ) / w_total
        val_sisdr_weighted = sum(
            self.loss_weights[k] * per_variant_sisdr[k] for k in present_keys
        ) / w_total

        self._last_val_breakdown = {
            "per_variant_sisdri": per_variant_sisdri,
            "per_variant_sisdr": per_variant_sisdr,
            "val_sisdri_weighted": val_sisdri_weighted,
            "val_sisdr_weighted": val_sisdr_weighted,
        }
        return val_sisdri_weighted

    def train(self, num_epochs: int, save_dir: str = "checkpoints", early_stopping_patience: int = None):
        """Main training loop with OOM error handling."""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        val_sisdr_history = []
        
        # Early stopping tracking (only if enabled)
        epochs_without_improvement = 0 if early_stopping_patience else None

        try:
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

                # Prepare metrics
                metrics = {
                    "epoch": self.current_epoch,
                    "train_si_sdr": train_sisdr,
                    "val_si_sdr": val_sisdr,
                    "train_lr": current_lr,
                }

                # Per-variant breakdown when weighted-loss path is active.
                # val_sisdri_weighted is the same as val_si_sdr (the scheduler
                # signal); we log it under both names — val_si_sdr keeps
                # dashboard-panel continuity, val_sisdri_weighted makes the
                # semantics explicit and is what sweeps target.
                if self._weighted_loss_active and self._last_val_breakdown:
                    bd = self._last_val_breakdown
                    metrics["val_sisdri_weighted"] = bd["val_sisdri_weighted"]
                    metrics["val_sisdr_weighted"] = bd["val_sisdr_weighted"]
                    for k, v in bd["per_variant_sisdri"].items():
                        metrics[f"val_sisdri_{k}"] = v
                    for k, v in bd["per_variant_sisdr"].items():
                        metrics[f"val_sisdr_{k}"] = v

                # Add early stopping metric if enabled
                if early_stopping_patience:
                    metrics["epochs_no_improvement"] = epochs_without_improvement

                if self.wandb_logger:
                    self.wandb_logger.log_metrics(metrics, step=self.current_epoch)

                # Check for improvement
                if val_sisdr > self.best_val_sisdr:
                    self.best_val_sisdr = val_sisdr
                    if early_stopping_patience:
                        epochs_without_improvement = 0  # Reset counter
                    self._save_checkpoint(epoch, val_sisdr, save_dir)
                else:
                    if early_stopping_patience:
                        epochs_without_improvement += 1
                        self.logger.info(f"No improvement for {epochs_without_improvement} epoch(s)")
                    
                # Early stopping check (only if enabled)
                if early_stopping_patience and epochs_without_improvement >= early_stopping_patience:
                    self.logger.warning("=" * 80)
                    self.logger.warning("EARLY STOPPING")
                    self.logger.warning("=" * 80)
                    self.logger.warning(f"No improvement for {early_stopping_patience} epochs")
                    self.logger.warning(f"Best Val SI-SDR: {self.best_val_sisdr:.2f} dB")
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
            self.logger.info("\nValidation SI-SDR history:")
            start_epoch = final_epoch - num_epochs + 1
            for idx, val_sisdr in enumerate(val_sisdr_history, start_epoch):
                self.logger.info(f"  Epoch {idx}: Val SI-SDR = {val_sisdr:.2f} dB")

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

