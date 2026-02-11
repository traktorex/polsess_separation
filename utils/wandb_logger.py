"""Weights & Biases logging wrapper for experiment tracking."""

import logging
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import asdict

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class WandbLogger:
    """Wrapper for Weights & Biases experiment tracking."""

    def __init__(
        self,
        project: str = "polsess-separation",
        entity: Optional[str] = None,
        run_name: Optional[str] = None,
        config: Optional[Any] = None,
        enabled: bool = True,
        logger: Optional[logging.Logger] = None,
        run: Optional[Any] = None,
        upload_checkpoints: bool = True,
    ):
        """Initialize W&B logger. If run is provided, uses it instead of creating new one.
        
        Args:
            upload_checkpoints: If False, skip uploading model artifacts to W&B.
                Set to False during sweeps to avoid storage bloat.
        """
        self.logger = logger or logging.getLogger("polsess")
        self.enabled = enabled and WANDB_AVAILABLE
        self.upload_checkpoints = upload_checkpoints
        self.run = None

        if not self.enabled:
            if not WANDB_AVAILABLE:
                self.logger.warning(
                    "W&B not available"
                )
            else:
                self.logger.info("W&B logging disabled")
            return

        try:
            # Use existing run if provided, otherwise create new one
            if run is not None:
                self.run = run
                self.logger.info("Using existing W&B run")
            else:
                config_dict = {}
                if config:
                    if hasattr(config, "data"):
                        config_dict = {
                            "data": asdict(config.data),
                            "model": asdict(config.model),
                            "training": asdict(config.training),
                        }
                    else:
                        config_dict = config

                self.run = wandb.init(
                    project=project,
                    entity=entity,
                    name=run_name,
                    config=config_dict,
                    resume="allow",
                )

        except Exception as e:
            self.logger.error(f"Failed to initialize W&B: {e}")
            self.logger.warning("Continuing without W&B logging")
            self.enabled = False

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to W&B."""
        if not self.enabled or not self.run:
            return

        try:
            if step is not None:
                wandb.log(metrics, step=step)
            else:
                wandb.log(metrics)
        except Exception as e:
            self.logger.warning(f"Failed to log metrics to W&B: {e}")

    def log_model(self, model_path: str, name: Optional[str] = None):
        """Save model checkpoint as W&B artifact."""
        if not self.enabled or not self.run or not self.upload_checkpoints:
            return

        try:
            model_path = Path(model_path)
            if not model_path.exists():
                self.logger.warning(f"Model file not found: {model_path}")
                return

            artifact_name = name or model_path.stem
            # Include run name to avoid all runs versioning the same artifact
            if self.run.name and not name:
                artifact_name = f"{artifact_name}-{self.run.name}"
            artifact = wandb.Artifact(artifact_name, type="model")
            artifact.add_file(str(model_path))
            self.run.log_artifact(artifact)

            self.logger.info(f"Logged model artifact: {artifact_name}")

        except Exception as e:
            self.logger.warning(f"Failed to log model to W&B: {e}")

    def finish(self):
        """Finish W&B run gracefully."""
        if not self.enabled or not self.run:
            return

        try:
            wandb.finish()
            self.logger.info("W&B run finished")
        except Exception as e:
            self.logger.warning(f"Error finishing W&B run: {e}")
